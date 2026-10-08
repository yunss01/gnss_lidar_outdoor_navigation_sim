"""Manage a GNSS destination and publish local ENU goal guidance."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path as FilesystemPath
import time
from typing import Optional

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool, Float32, String

from .navigation_core import ArrivalDebouncer
from .navigation_core import EnuPoint
from .navigation_core import GeodeticPoint
from .navigation_core import bearing_from_north_deg
from .navigation_core import geodetic_to_enu
from .navigation_core import horizontal_distance
from .navigation_core import interpolate_line
from .navigation_core import median_geodetic_point
from .navigation_core import validate_geodetic
from .navigation_core import yaw_from_east_rad


def _quaternion_z_w(yaw_rad: float):
    return math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def anchor_fix_is_new_enough(
        latest_gnss_stamp_ns, minimum_anchor_gnss_stamp_ns):
    """Return whether a GNSS fix is safe for the pending odometry anchor."""
    if latest_gnss_stamp_ns is None:
        return False
    return (
        minimum_anchor_gnss_stamp_ns is None
        or latest_gnss_stamp_ns >= minimum_anchor_gnss_stamp_ns
    )


NAVIGATION_LOG_FIELDS = [
    'sequence',
    'wall_time_iso',
    'ros_time_sec',
    'ros_time_nanosec',
    'status',
    'position_source',
    'gnss_age_s',
    'odometry_age_s',
    'gnss_stamp_sec',
    'gnss_stamp_nanosec',
    'odometry_stamp_sec',
    'odometry_stamp_nanosec',
    'current_latitude',
    'current_longitude',
    'current_altitude',
    'current_east_m',
    'current_north_m',
    'current_up_m',
    'odometry_x_m',
    'odometry_y_m',
    'speed_mps',
    'goal_latitude',
    'goal_longitude',
    'goal_altitude',
    'goal_east_m',
    'goal_north_m',
    'goal_up_m',
    'distance_to_goal_m',
    'bearing_to_goal_deg',
    'goal_reached',
    'arrival_inside_count',
]


class NavigationCsvLogger:
    """Write navigation state immediately to a timestamped run directory."""

    def __init__(self, root_directory: FilesystemPath):
        root = FilesystemPath(root_directory).expanduser().resolve()
        run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        self.run_directory = root / run_name
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.path = self.run_directory / 'navigation.csv'
        self.metadata_path = self.run_directory / 'run_metadata.json'
        self._stream = self.path.open(
            'w', encoding='utf-8', newline='', buffering=1
        )
        self._writer = csv.DictWriter(
            self._stream, fieldnames=NAVIGATION_LOG_FIELDS
        )
        self._writer.writeheader()
        self._stream.flush()

    def write(self, values):
        """Append and flush one state row."""
        self._writer.writerow(values)
        self._stream.flush()

    def write_metadata(self, values):
        """Replace the small run metadata snapshot."""
        with self.metadata_path.open('w', encoding='utf-8') as stream:
            json.dump(values, stream, indent=2, ensure_ascii=False)

    def close(self):
        """Close the CSV stream."""
        if not self._stream.closed:
            self._stream.close()


class GnssGoalManagerNode(Node):
    """Turn a WGS84 goal into stable local navigation guidance.

    ``/navigation/direct_path`` deliberately ignores obstacles. It is a goal
    direction reference, not a collision-free global plan.
    """

    def __init__(self):
        super().__init__('gnss_goal_manager_node')

        self.declare_parameter('gnss_topic', '/gnss/fix')
        self.declare_parameter('odometry_topic', '/vehicle/odometry')
        self.declare_parameter('goal_topic', '/navigation/goal_gnss')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('projection_mode', 'wgs84')
        self.declare_parameter('goal_enabled', False)
        self.declare_parameter('goal_latitude', 0.0)
        self.declare_parameter('goal_longitude', 0.0)
        self.declare_parameter('goal_altitude', 0.0)
        self.declare_parameter('use_odometry_position', True)
        self.declare_parameter('odometry_is_enu_aligned', True)
        self.declare_parameter('maximum_odometry_step_m', 10.0)
        self.declare_parameter('arrival_radius_m', 3.0)
        self.declare_parameter('leave_radius_m', 4.5)
        self.declare_parameter('arrival_consecutive_samples', 5)
        self.declare_parameter('direct_path_spacing_m', 1.0)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('sensor_timeout_s', 1.0)
        self.declare_parameter('origin_initialization_sample_count', 20)
        self.declare_parameter('terminal_log_period_s', 2.0)
        self.declare_parameter('log_navigation', True)
        self.declare_parameter(
            'log_directory', '~/terrain_nav_data/logs/navigation'
        )

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.projection_mode = str(
            self.get_parameter('projection_mode').value
        )
        # Validate the configured mode before accepting any live GNSS data.
        geodetic_to_enu(
            GeodeticPoint(0.0, 0.0, 0.0),
            GeodeticPoint(0.0, 0.0, 0.0),
            self.projection_mode,
        )
        self.use_odometry = bool(
            self.get_parameter('use_odometry_position').value
        )
        self.odometry_is_enu_aligned = bool(
            self.get_parameter('odometry_is_enu_aligned').value
        )
        if self.use_odometry and not self.odometry_is_enu_aligned:
            raise ValueError(
                'use_odometry_position requires an ENU-aligned odometry frame'
            )
        self.maximum_odometry_step_m = float(
            self.get_parameter('maximum_odometry_step_m').value
        )
        self.path_spacing_m = float(
            self.get_parameter('direct_path_spacing_m').value
        )
        publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.sensor_timeout_s = float(
            self.get_parameter('sensor_timeout_s').value
        )
        self.origin_sample_count = int(
            self.get_parameter(
                'origin_initialization_sample_count'
            ).value
        )
        self.terminal_log_period_s = float(
            self.get_parameter('terminal_log_period_s').value
        )
        if self.maximum_odometry_step_m <= 0.0:
            raise ValueError('maximum_odometry_step_m must be positive')
        if self.path_spacing_m <= 0.0:
            raise ValueError('direct_path_spacing_m must be positive')
        if publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self.sensor_timeout_s <= 0.0:
            raise ValueError('sensor_timeout_s must be positive')
        if self.origin_sample_count <= 0:
            raise ValueError(
                'origin_initialization_sample_count must be positive'
            )

        self.arrival = ArrivalDebouncer(
            float(self.get_parameter('arrival_radius_m').value),
            float(self.get_parameter('leave_radius_m').value),
            int(self.get_parameter(
                'arrival_consecutive_samples'
            ).value),
        )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        goal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.origin_publisher = self.create_publisher(
            NavSatFix, '/navigation/origin_gnss', latched_qos
        )
        self.current_local_publisher = self.create_publisher(
            PointStamped, '/navigation/current_local', output_qos
        )
        self.goal_local_publisher = self.create_publisher(
            PointStamped, '/navigation/goal_local', latched_qos
        )
        self.goal_pose_publisher = self.create_publisher(
            PoseStamped, '/navigation/goal_pose', latched_qos
        )
        self.goal_vector_publisher = self.create_publisher(
            Vector3Stamped, '/navigation/goal_vector', output_qos
        )
        self.distance_publisher = self.create_publisher(
            Float32, '/navigation/distance_to_goal', output_qos
        )
        self.bearing_publisher = self.create_publisher(
            Float32, '/navigation/bearing_to_goal_deg', output_qos
        )
        self.reached_publisher = self.create_publisher(
            Bool, '/navigation/goal_reached', latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, '/navigation/status', latched_qos
        )
        self.direct_path_publisher = self.create_publisher(
            Path, '/navigation/direct_path', output_qos
        )

        self.create_subscription(
            NavSatFix,
            str(self.get_parameter('gnss_topic').value),
            self._gnss_callback,
            sensor_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self._odometry_callback,
            sensor_qos,
        )
        self.create_subscription(
            NavSatFix,
            str(self.get_parameter('goal_topic').value),
            self._goal_callback,
            goal_qos,
        )

        self.origin: Optional[GeodeticPoint] = None
        self.origin_samples = []
        self.origin_source_message: Optional[NavSatFix] = None
        self.goal: Optional[GeodeticPoint] = None
        self.goal_local: Optional[EnuPoint] = None
        self.latest_gnss_local: Optional[EnuPoint] = None
        self.current_local: Optional[EnuPoint] = None
        self.latest_gnss_message: Optional[NavSatFix] = None
        self.latest_gnss_wall_time: Optional[float] = None
        self.latest_gnss_stamp_ns = None
        self.latest_odometry_xy = None
        self.latest_odometry_stamp = None
        self.previous_odometry_xy = None
        self.odometry_anchor_xy = None
        self.odometry_anchor_local: Optional[EnuPoint] = None
        self.pending_odometry_anchor_local: Optional[EnuPoint] = None
        self.minimum_anchor_gnss_stamp_ns = None
        self.latest_odometry_wall_time: Optional[float] = None
        self.position_source = 'none'
        self.last_status = None
        self.last_terminal_log_time = None
        self.latest_gnss_point: Optional[GeodeticPoint] = None
        self.latest_speed_mps = float('nan')
        self.log_sequence = 0
        self.navigation_logger: Optional[NavigationCsvLogger] = None
        if bool(self.get_parameter('log_navigation').value):
            self.navigation_logger = NavigationCsvLogger(FilesystemPath(
                str(self.get_parameter('log_directory').value)
            ))
            self.get_logger().info(
                'Navigation CSV: {}'.format(
                    self.navigation_logger.path
                )
            )

        if bool(self.get_parameter('goal_enabled').value):
            self._set_goal(GeodeticPoint(
                float(self.get_parameter('goal_latitude').value),
                float(self.get_parameter('goal_longitude').value),
                float(self.get_parameter('goal_altitude').value),
            ))

        self._write_metadata()
        self.timer = self.create_timer(
            1.0 / publish_rate_hz, self._publish_guidance
        )
        self.get_logger().info(
            'GNSS goal manager ready; goal topic={} frame={} '
            'odometry_position={} origin_samples={} projection={}'.format(
                self.get_parameter('goal_topic').value,
                self.frame_id,
                self.use_odometry,
                self.origin_sample_count,
                self.projection_mode,
            )
        )

    def _set_goal(self, goal: GeodeticPoint):
        validate_geodetic(goal)
        self.goal = goal
        self.arrival.reset()
        self.goal_local = (
            geodetic_to_enu(
                goal, self.origin, self.projection_mode
            )
            if self.origin is not None else None
        )
        if hasattr(self, 'navigation_logger'):
            self._write_metadata()
        self.get_logger().info(
            'GNSS goal set: latitude={:.9f}, longitude={:.9f}, '
            'altitude={:.3f} m'.format(
                goal.latitude_deg,
                goal.longitude_deg,
                goal.altitude_m,
            )
        )

    def _goal_callback(self, message: NavSatFix):
        try:
            self._set_goal(GeodeticPoint(
                float(message.latitude),
                float(message.longitude),
                float(message.altitude),
            ))
        except ValueError as error:
            self.get_logger().error('Rejected GNSS goal: {}'.format(error))

    @staticmethod
    def _valid_fix(message: NavSatFix) -> bool:
        if message.status.status == NavSatStatus.STATUS_NO_FIX:
            return False
        try:
            validate_geodetic(GeodeticPoint(
                float(message.latitude),
                float(message.longitude),
                float(message.altitude),
            ))
        except ValueError:
            return False
        return True

    def _publish_origin(self, source: NavSatFix):
        message = NavSatFix()
        message.header = source.header
        message.header.frame_id = 'wgs84'
        message.status = source.status
        message.latitude = self.origin.latitude_deg
        message.longitude = self.origin.longitude_deg
        message.altitude = self.origin.altitude_m
        message.position_covariance = list(source.position_covariance)
        message.position_covariance_type = (
            source.position_covariance_type
        )
        self.origin_publisher.publish(message)

    def _gnss_callback(self, message: NavSatFix):
        if not self._valid_fix(message):
            self._publish_status('invalid_gnss_fix')
            return
        current_fix = GeodeticPoint(
            float(message.latitude),
            float(message.longitude),
            float(message.altitude),
        )
        self.latest_gnss_point = current_fix
        self.latest_gnss_message = message
        self.latest_gnss_wall_time = time.monotonic()
        self.latest_gnss_stamp_ns = _stamp_ns(message.header.stamp)
        if self.origin is None:
            self.origin_samples.append(current_fix)
            self.origin_source_message = message
            collected = len(self.origin_samples)
            if collected < self.origin_sample_count:
                if collected == 1 or collected % 5 == 0:
                    self.get_logger().info(
                        'Stabilizing GNSS origin: {}/{} samples'.format(
                            collected, self.origin_sample_count
                        )
                    )
                return
            self._finalize_origin()

        self.latest_gnss_local = geodetic_to_enu(
            current_fix, self.origin, self.projection_mode
        )
        if not self.use_odometry:
            self._update_position(self.latest_gnss_local, 'gnss')
        elif (
            self.odometry_anchor_xy is None
            and self.latest_odometry_xy is not None
        ):
            self._anchor_odometry()

    def _finalize_origin(self):
        """Fix the ENU origin from stationary startup GNSS samples."""
        self.origin = median_geodetic_point(self.origin_samples)
        source = self.origin_source_message
        self._publish_origin(source)
        if self.goal is not None:
            self.goal_local = geodetic_to_enu(
                self.goal, self.origin, self.projection_mode
            )

        offsets = [
            geodetic_to_enu(
                point, self.origin, self.projection_mode
            )
            for point in self.origin_samples
        ]
        radial_offsets = [
            math.hypot(point.east_m, point.north_m)
            for point in offsets
        ]
        median_offset = sorted(radial_offsets)[
            len(radial_offsets) // 2
        ]
        maximum_offset = max(radial_offsets)

        # The vehicle is stationary while the origin samples are collected.
        # Anchor odometry at the robust origin itself, not at the final noisy
        # GNSS sample, otherwise the same local-frame translation is re-added.
        self.pending_odometry_anchor_local = EnuPoint(0.0, 0.0, 0.0)
        self.get_logger().info(
            'Local ENU origin fixed from {} samples at '
            '{:.9f}, {:.9f}, {:.3f}; median/max horizontal offset '
            '{:.2f}/{:.2f} m'.format(
                len(self.origin_samples),
                self.origin.latitude_deg,
                self.origin.longitude_deg,
                self.origin.altitude_m,
                median_offset,
                maximum_offset,
            )
        )
        self.origin_samples.clear()
        self._write_metadata()

    def _anchor_odometry(self):
        if (
            self.latest_odometry_xy is None
            or self.latest_gnss_local is None
            or not anchor_fix_is_new_enough(
                self.latest_gnss_stamp_ns,
                self.minimum_anchor_gnss_stamp_ns,
            )
        ):
            return
        anchor_local = (
            self.pending_odometry_anchor_local
            if self.pending_odometry_anchor_local is not None
            else self.latest_gnss_local
        )
        self.odometry_anchor_xy = self.latest_odometry_xy
        self.odometry_anchor_local = anchor_local
        self.pending_odometry_anchor_local = None
        self.minimum_anchor_gnss_stamp_ns = None
        self.previous_odometry_xy = self.latest_odometry_xy
        self._update_position(anchor_local, 'odometry')
        self.get_logger().info(
            'ENU-aligned odometry anchored at local '
            '({:.3f}, {:.3f}) m'.format(
                anchor_local.east_m, anchor_local.north_m
            )
        )

    def _odometry_callback(self, message: Odometry):
        x_value = float(message.pose.pose.position.x)
        y_value = float(message.pose.pose.position.y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            return
        current_xy = (x_value, y_value)
        self.latest_odometry_xy = current_xy
        self.latest_odometry_stamp = message.header.stamp
        velocity = message.twist.twist.linear
        self.latest_speed_mps = math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )
        self.latest_odometry_wall_time = time.monotonic()
        if not self.use_odometry:
            return
        if self.odometry_anchor_xy is None:
            self._anchor_odometry()
            return

        if self.previous_odometry_xy is not None:
            step = math.hypot(
                current_xy[0] - self.previous_odometry_xy[0],
                current_xy[1] - self.previous_odometry_xy[1],
            )
            if step > self.maximum_odometry_step_m:
                self.get_logger().warning(
                    'Odometry jumped {:.2f} m; waiting for a post-jump '
                    'GNSS fix before re-anchoring'.format(
                        step
                    )
                )
                self.odometry_anchor_xy = None
                self.odometry_anchor_local = None
                self.current_local = None
                self.position_source = 'waiting_for_post_jump_gnss'
                self.previous_odometry_xy = current_xy
                self.minimum_anchor_gnss_stamp_ns = _stamp_ns(
                    message.header.stamp)
                return
        self.previous_odometry_xy = current_xy
        local = EnuPoint(
            self.odometry_anchor_local.east_m
            + current_xy[0] - self.odometry_anchor_xy[0],
            self.odometry_anchor_local.north_m
            + current_xy[1] - self.odometry_anchor_xy[1],
            self.odometry_anchor_local.up_m,
        )
        self._update_position(local, 'odometry')

    def _update_position(self, local: EnuPoint, source: str):
        self.current_local = local
        self.position_source = source
        if self.goal_local is not None:
            self.arrival.update(
                horizontal_distance(local, self.goal_local)
            )

    def _publish_status(self, value: str):
        if value == self.last_status:
            return
        message = String()
        message.data = value
        self.status_publisher.publish(message)
        self.last_status = value

    @staticmethod
    def _optional_value(point, attribute):
        if point is None:
            return ''
        return '{:.9f}'.format(float(getattr(point, attribute)))

    @staticmethod
    def _age(now, timestamp):
        if timestamp is None:
            return ''
        return '{:.6f}'.format(now - timestamp)

    def _write_metadata(self):
        if self.navigation_logger is None:
            return

        def geodetic_dict(point):
            if point is None:
                return None
            return {
                'latitude': point.latitude_deg,
                'longitude': point.longitude_deg,
                'altitude': point.altitude_m,
            }

        self.navigation_logger.write_metadata({
            'created_at': datetime.now().astimezone().isoformat(),
            'frame_id': self.frame_id,
            'projection_mode': self.projection_mode,
            'gnss_topic': str(self.get_parameter('gnss_topic').value),
            'odometry_topic': str(
                self.get_parameter('odometry_topic').value
            ),
            'goal_topic': str(self.get_parameter('goal_topic').value),
            'use_odometry_position': self.use_odometry,
            'odometry_is_enu_aligned': self.odometry_is_enu_aligned,
            'arrival_radius_m': self.arrival.arrival_radius_m,
            'leave_radius_m': self.arrival.leave_radius_m,
            'arrival_consecutive_samples': (
                self.arrival.required_consecutive_samples
            ),
            'origin_initialization_sample_count': self.origin_sample_count,
            'origin': geodetic_dict(self.origin),
            'goal': geodetic_dict(self.goal),
        })

    def _write_navigation_log(
        self,
        status,
        distance=None,
        bearing=None,
    ):
        if self.navigation_logger is None:
            return
        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        self.log_sequence += 1
        current = self.current_local
        goal = self.goal_local
        gnss = self.latest_gnss_point
        odometry_x = (
            self.latest_odometry_xy[0]
            if self.latest_odometry_xy is not None else None
        )
        odometry_y = (
            self.latest_odometry_xy[1]
            if self.latest_odometry_xy is not None else None
        )

        def number(value, digits=6):
            if value is None or not math.isfinite(float(value)):
                return ''
            return ('{:.%df}' % digits).format(float(value))

        self.navigation_logger.write({
            'sequence': self.log_sequence,
            'wall_time_iso': (
                datetime.now().astimezone().isoformat(
                    timespec='milliseconds'
                )
            ),
            'ros_time_sec': int(stamp.sec),
            'ros_time_nanosec': int(stamp.nanosec),
            'status': status,
            'position_source': self.position_source,
            'gnss_age_s': self._age(
                now, self.latest_gnss_wall_time
            ),
            'odometry_age_s': self._age(
                now, self.latest_odometry_wall_time
            ),
            'gnss_stamp_sec': (
                int(self.latest_gnss_message.header.stamp.sec)
                if self.latest_gnss_message is not None else ''
            ),
            'gnss_stamp_nanosec': (
                int(self.latest_gnss_message.header.stamp.nanosec)
                if self.latest_gnss_message is not None else ''
            ),
            'odometry_stamp_sec': (
                int(self.latest_odometry_stamp.sec)
                if self.latest_odometry_stamp is not None else ''
            ),
            'odometry_stamp_nanosec': (
                int(self.latest_odometry_stamp.nanosec)
                if self.latest_odometry_stamp is not None else ''
            ),
            'current_latitude': (
                number(gnss.latitude_deg, 9) if gnss else ''
            ),
            'current_longitude': (
                number(gnss.longitude_deg, 9) if gnss else ''
            ),
            'current_altitude': (
                number(gnss.altitude_m, 6) if gnss else ''
            ),
            'current_east_m': (
                number(current.east_m) if current else ''
            ),
            'current_north_m': (
                number(current.north_m) if current else ''
            ),
            'current_up_m': (
                number(current.up_m) if current else ''
            ),
            'odometry_x_m': number(odometry_x),
            'odometry_y_m': number(odometry_y),
            'speed_mps': number(self.latest_speed_mps),
            'goal_latitude': (
                number(self.goal.latitude_deg, 9)
                if self.goal else ''
            ),
            'goal_longitude': (
                number(self.goal.longitude_deg, 9)
                if self.goal else ''
            ),
            'goal_altitude': (
                number(self.goal.altitude_m, 6)
                if self.goal else ''
            ),
            'goal_east_m': (
                number(goal.east_m) if goal else ''
            ),
            'goal_north_m': (
                number(goal.north_m) if goal else ''
            ),
            'goal_up_m': number(goal.up_m) if goal else '',
            'distance_to_goal_m': number(distance),
            'bearing_to_goal_deg': number(bearing),
            'goal_reached': int(self.arrival.reached),
            'arrival_inside_count': self.arrival.inside_count,
        })

    def _sensor_is_stale(self) -> bool:
        now = time.monotonic()
        if self.latest_gnss_wall_time is None:
            return True
        if now - self.latest_gnss_wall_time > self.sensor_timeout_s:
            return True
        if self.use_odometry:
            if self.latest_odometry_wall_time is None:
                return True
            if now - self.latest_odometry_wall_time > self.sensor_timeout_s:
                return True
        return False

    def _publish_guidance(self):
        if self.origin is None or self.current_local is None:
            self._publish_status('waiting_for_position')
            self._write_navigation_log('waiting_for_position')
            return
        if self.goal is None or self.goal_local is None:
            self._publish_status('waiting_for_goal')
            self._write_navigation_log('waiting_for_goal')
            return
        if self._sensor_is_stale():
            self._publish_status('sensor_stale')
            self._write_navigation_log('sensor_stale')
            return

        current = self.current_local
        goal = self.goal_local
        distance = horizontal_distance(current, goal)
        bearing = bearing_from_north_deg(current, goal)
        yaw = yaw_from_east_rad(current, goal)
        stamp = self.get_clock().now().to_msg()

        current_message = PointStamped()
        current_message.header.stamp = stamp
        current_message.header.frame_id = self.frame_id
        current_message.point.x = current.east_m
        current_message.point.y = current.north_m
        current_message.point.z = current.up_m
        self.current_local_publisher.publish(current_message)

        goal_message = PointStamped()
        goal_message.header.stamp = stamp
        goal_message.header.frame_id = self.frame_id
        goal_message.point.x = goal.east_m
        goal_message.point.y = goal.north_m
        goal_message.point.z = goal.up_m
        self.goal_local_publisher.publish(goal_message)

        goal_pose = PoseStamped()
        goal_pose.header = goal_message.header
        goal_pose.pose.position = goal_message.point
        goal_pose.pose.orientation.z, goal_pose.pose.orientation.w = (
            _quaternion_z_w(yaw)
        )
        self.goal_pose_publisher.publish(goal_pose)

        vector_message = Vector3Stamped()
        vector_message.header = goal_message.header
        vector_message.vector.x = goal.east_m - current.east_m
        vector_message.vector.y = goal.north_m - current.north_m
        vector_message.vector.z = goal.up_m - current.up_m
        self.goal_vector_publisher.publish(vector_message)

        distance_message = Float32()
        distance_message.data = distance
        self.distance_publisher.publish(distance_message)
        bearing_message = Float32()
        bearing_message.data = bearing
        self.bearing_publisher.publish(bearing_message)
        reached_message = Bool()
        reached_message.data = self.arrival.reached
        self.reached_publisher.publish(reached_message)

        path = Path()
        path.header = goal_message.header
        path_points = interpolate_line(
            current, goal, self.path_spacing_m
        )
        for point in path_points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point.east_m
            pose.pose.position.y = point.north_m
            pose.pose.position.z = point.up_m
            pose.pose.orientation.z, pose.pose.orientation.w = (
                _quaternion_z_w(yaw)
            )
            path.poses.append(pose)
        self.direct_path_publisher.publish(path)

        status = 'goal_reached' if self.arrival.reached else 'navigating'
        self._publish_status(status)
        self._write_navigation_log(status, distance, bearing)
        now = time.monotonic()
        if (
            self.last_terminal_log_time is None
            or self.terminal_log_period_s == 0.0
            or now - self.last_terminal_log_time
            >= self.terminal_log_period_s
        ):
            self.get_logger().info(
                'status={} distance={:.2f} m bearing={:.1f} deg '
                'position_source={}'.format(
                    status, distance, bearing, self.position_source
                )
            )
            self.last_terminal_log_time = now

    def destroy_node(self):
        if self.navigation_logger is not None:
            self.navigation_logger.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GnssGoalManagerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
