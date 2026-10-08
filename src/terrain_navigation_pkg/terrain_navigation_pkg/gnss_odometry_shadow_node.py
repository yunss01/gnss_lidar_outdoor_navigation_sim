"""Publish TF-independent local GNSS odometry for sensor-only evaluation."""

import json
import math

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import String

from .localization_shadow_core import base_position_from_antenna_delta
from .localization_shadow_core import circular_mean_angle
from .localization_shadow_core import quaternion_yaw
from .navigation_core import GeodeticPoint
from .navigation_core import geodetic_to_enu
from .navigation_core import median_geodetic_point


def _stamp_ns(message):
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


class GnssOdometryShadowNode(Node):
    """Convert GNSS to startup-relative ENU without using the CARLA TF tree.

    ``navsat_transform_node`` needs an odom-to-base transform. The shadow EKF
    cannot publish that transform while CARLA owns the real odom-to-base
    transform, so using it here either creates a TF conflict or produces a
    zero-valued GPS odometry message. This adapter performs only the WGS84 to
    ENU conversion and measured antenna lever-arm correction. It never reads
    ground-truth odometry or publishes TF.
    """

    def __init__(self):
        super().__init__('gnss_odometry_shadow_node')
        self.declare_parameter('gnss_topic', '/gnss/fix')
        self.declare_parameter('imu_topic', '/vectornav/imu')
        self.declare_parameter(
            'output_topic', '/localization/odometry_gps_shadow')
        self.declare_parameter(
            'status_topic', '/localization/gnss_shadow_status')
        self.declare_parameter('frame_id', 'odom_sensor_shadow')
        self.declare_parameter('child_frame_id', 'base_link')
        self.declare_parameter('projection_mode', 'wgs84')
        self.declare_parameter('origin_initialization_sample_count', 20)
        self.declare_parameter('maximum_imu_age_s', 0.25)
        self.declare_parameter('antenna_forward_m', 1.0)
        self.declare_parameter('antenna_left_m', 0.0)
        self.declare_parameter('default_horizontal_covariance_m2', 2.25)
        self.declare_parameter('default_vertical_covariance_m2', 9.0)
        self.declare_parameter('zero_altitude', True)

        self._frame_id = str(self.get_parameter('frame_id').value)
        self._child_frame_id = str(
            self.get_parameter('child_frame_id').value)
        self._projection_mode = str(
            self.get_parameter('projection_mode').value)
        # Fail at startup rather than silently interpreting real GNSS with a
        # simulator-only projection (or vice versa).
        geodetic_to_enu(
            GeodeticPoint(0.0, 0.0, 0.0),
            GeodeticPoint(0.0, 0.0, 0.0),
            self._projection_mode,
        )
        self._sample_count_required = int(
            self.get_parameter(
                'origin_initialization_sample_count').value)
        self._maximum_imu_age_ns = int(
            float(self.get_parameter('maximum_imu_age_s').value) * 1e9)
        self._antenna_forward_m = float(
            self.get_parameter('antenna_forward_m').value)
        self._antenna_left_m = float(
            self.get_parameter('antenna_left_m').value)
        self._default_horizontal_covariance_m2 = float(
            self.get_parameter('default_horizontal_covariance_m2').value)
        self._default_vertical_covariance_m2 = float(
            self.get_parameter('default_vertical_covariance_m2').value)
        self._zero_altitude = bool(
            self.get_parameter('zero_altitude').value)
        if self._sample_count_required <= 0:
            raise ValueError(
                'origin_initialization_sample_count must be positive')
        if self._maximum_imu_age_ns <= 0:
            raise ValueError('maximum_imu_age_s must be positive')
        if self._default_horizontal_covariance_m2 <= 0.0:
            raise ValueError(
                'default_horizontal_covariance_m2 must be positive')
        if self._default_vertical_covariance_m2 <= 0.0:
            raise ValueError(
                'default_vertical_covariance_m2 must be positive')

        self._latest_imu = None
        self._origin_samples = []
        self._origin_yaws = []
        self._origin = None
        self._reference_yaw = None
        self._published_count = 0
        self._invalid_fix_count = 0
        self._missing_imu_count = 0
        self._stale_imu_count = 0
        self._latest_output_stamp_ns = None

        self._odometry_publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter('output_topic').value),
            10,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            10,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter('imu_topic').value),
            self._on_imu,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            NavSatFix,
            str(self.get_parameter('gnss_topic').value),
            self._on_gnss,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            'GNSS shadow adapter waiting for {} stationary samples; '
            'antenna offset x={:.3f} m, y={:.3f} m; projection={}'.format(
                self._sample_count_required,
                self._antenna_forward_m,
                self._antenna_left_m,
                self._projection_mode,
            )
        )

    def _on_imu(self, message):
        yaw = quaternion_yaw(message.orientation)
        if math.isfinite(yaw):
            self._latest_imu = (message, yaw)

    @staticmethod
    def _valid_fix(message):
        return (
            message.status.status != NavSatStatus.STATUS_NO_FIX
            and math.isfinite(message.latitude)
            and math.isfinite(message.longitude)
            and math.isfinite(message.altitude)
            and -90.0 <= message.latitude <= 90.0
            and -180.0 <= message.longitude <= 180.0
        )

    def _on_gnss(self, message):
        if not self._valid_fix(message):
            self._invalid_fix_count += 1
            return
        if self._latest_imu is None:
            self._missing_imu_count += 1
            return
        imu, yaw = self._latest_imu
        if abs(_stamp_ns(message) - _stamp_ns(imu)) > self._maximum_imu_age_ns:
            self._stale_imu_count += 1
            return

        point = GeodeticPoint(
            float(message.latitude),
            float(message.longitude),
            float(message.altitude),
        )
        if self._origin is None:
            self._origin_samples.append(point)
            self._origin_yaws.append(yaw)
            if len(self._origin_samples) < self._sample_count_required:
                return
            self._origin = median_geodetic_point(self._origin_samples)
            self._reference_yaw = circular_mean_angle(self._origin_yaws)
            self.get_logger().info(
                'GNSS shadow origin initialized from {} samples'.format(
                    len(self._origin_samples)
                )
            )

        antenna_position = geodetic_to_enu(
            point, self._origin, self._projection_mode
        )
        base_east_m, base_north_m = base_position_from_antenna_delta(
            antenna_position.east_m,
            antenna_position.north_m,
            yaw,
            self._reference_yaw,
            self._antenna_forward_m,
            self._antenna_left_m,
        )
        odometry = Odometry()
        odometry.header = message.header
        odometry.header.frame_id = self._frame_id
        odometry.child_frame_id = self._child_frame_id
        odometry.pose.pose.position.x = base_east_m
        odometry.pose.pose.position.y = base_north_m
        odometry.pose.pose.position.z = (
            0.0 if self._zero_altitude else antenna_position.up_m)
        odometry.pose.pose.orientation.w = 1.0
        odometry.pose.covariance = self._pose_covariance(message)
        odometry.twist.covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            odometry.twist.covariance[index] = 1e6
        self._odometry_publisher.publish(odometry)
        self._published_count += 1
        self._latest_output_stamp_ns = _stamp_ns(odometry)

    def _pose_covariance(self, message):
        source = list(message.position_covariance)
        horizontal = self._default_horizontal_covariance_m2
        vertical = self._default_vertical_covariance_m2
        if (
            message.position_covariance_type
            != NavSatFix.COVARIANCE_TYPE_UNKNOWN
        ):
            if math.isfinite(source[0]) and source[0] > 0.0:
                horizontal = source[0]
            if math.isfinite(source[4]) and source[4] > 0.0:
                north_covariance = source[4]
            else:
                north_covariance = horizontal
            if math.isfinite(source[8]) and source[8] > 0.0:
                vertical = source[8]
        else:
            north_covariance = horizontal

        covariance = [0.0] * 36
        covariance[0] = horizontal
        covariance[1] = source[1] if math.isfinite(source[1]) else 0.0
        covariance[6] = source[3] if math.isfinite(source[3]) else 0.0
        covariance[7] = north_covariance
        covariance[14] = vertical
        covariance[21] = 1e6
        covariance[28] = 1e6
        covariance[35] = 1e6
        return covariance

    def _publish_status(self):
        output_age_s = None
        if self._latest_output_stamp_ns is not None:
            output_age_s = max(
                0.0,
                (self.get_clock().now().nanoseconds
                 - self._latest_output_stamp_ns) / 1e9,
            )
        status = {
            'mode': 'gnss_enu_shadow',
            'control_effect': 'none',
            'ground_truth_input_used': False,
            'initialized': self._origin is not None,
            'origin_sample_count': len(self._origin_samples),
            'origin_sample_count_required': self._sample_count_required,
            'published_count': self._published_count,
            'invalid_fix_count': self._invalid_fix_count,
            'missing_imu_count': self._missing_imu_count,
            'stale_imu_count': self._stale_imu_count,
            'output_age_s': output_age_s,
            'antenna_forward_m': self._antenna_forward_m,
            'antenna_left_m': self._antenna_left_m,
            'projection_mode': self._projection_mode,
        }
        if self._origin is not None:
            status['origin'] = {
                'latitude': self._origin.latitude_deg,
                'longitude': self._origin.longitude_deg,
                'altitude': self._origin.altitude_m,
            }
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self._status_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = GnssOdometryShadowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
