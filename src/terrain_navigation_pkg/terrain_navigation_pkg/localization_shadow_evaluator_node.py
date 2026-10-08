"""Evaluate sensor-only shadow odometry against isolated CARLA truth."""

from collections import deque
import csv
import datetime
import json
import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .localization_shadow_core import align_enu_pose
from .localization_shadow_core import assess_localization_transition
from .localization_shadow_core import quaternion_yaw
from .localization_shadow_core import planar_error_components
from .localization_shadow_core import relative_translation_error
from .localization_shadow_core import summarize_errors
from .localization_shadow_core import summarize_signed_errors
from .localization_shadow_core import wrap_angle


def _stamp_ns(message):
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


class LocalizationShadowEvaluator(Node):
    """Measure drift without giving ground truth to either EKF."""

    def __init__(self):
        super().__init__('localization_shadow_evaluator_node')
        self.declare_parameter('ground_truth_topic', '/vehicle/odometry')
        self.declare_parameter(
            'estimate_topic', '/localization/odometry_shadow')
        self.declare_parameter(
            'gnss_odometry_topic',
            '/localization/odometry_gps_shadow')
        self.declare_parameter(
            'status_topic', '/localization/shadow_status')
        self.declare_parameter(
            'output_directory',
            '~/terrain_nav_data/logs/localization_shadow')
        self.declare_parameter('maximum_pair_delay_s', 0.15)
        self.declare_parameter('history_size', 100000)
        self.declare_parameter('pose_jump_threshold_m', 0.30)
        self.declare_parameter('transition_minimum_moving_sample_count', 1000)
        self.declare_parameter('transition_minimum_reference_speed_mps', 0.20)
        self.declare_parameter('transition_maximum_estimate_age_s', 0.15)
        self.declare_parameter('transition_maximum_gnss_age_s', 0.25)
        self.declare_parameter('transition_maximum_position_p95_m', 0.30)
        self.declare_parameter(
            'transition_maximum_longitudinal_p95_m', 0.25)
        self.declare_parameter('transition_maximum_lateral_p95_m', 0.20)
        self.declare_parameter(
            'transition_maximum_absolute_longitudinal_bias_m', 0.10)
        self.declare_parameter(
            'transition_maximum_relative_translation_p95_m', 0.08)
        self.declare_parameter('transition_maximum_pose_jump_count', 0)
        self.declare_parameter('active_navigation_tf_ready', False)
        self.declare_parameter('independent_velocity_ready', False)

        ground_truth_topic = str(
            self.get_parameter('ground_truth_topic').value)
        estimate_topic = str(self.get_parameter('estimate_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)
        history_size = max(100, int(
            self.get_parameter('history_size').value))
        self._maximum_pair_delay_ns = int(
            float(self.get_parameter('maximum_pair_delay_s').value) * 1e9)
        self._ground_truth = deque(maxlen=200)
        self._estimate_origin = None
        self._reference_origin = None
        self._position_errors = deque(maxlen=history_size)
        self._longitudinal_errors = deque(maxlen=history_size)
        self._lateral_errors = deque(maxlen=history_size)
        self._signed_longitudinal_errors = deque(maxlen=history_size)
        self._signed_lateral_errors = deque(maxlen=history_size)
        self._relative_translation_errors = deque(maxlen=history_size)
        self._yaw_errors = deque(maxlen=history_size)
        self._speed_errors = deque(maxlen=history_size)
        self._moving_position_errors = deque(maxlen=history_size)
        self._moving_longitudinal_errors = deque(maxlen=history_size)
        self._moving_lateral_errors = deque(maxlen=history_size)
        self._moving_signed_longitudinal_errors = deque(maxlen=history_size)
        self._moving_signed_lateral_errors = deque(maxlen=history_size)
        self._moving_relative_translation_errors = deque(maxlen=history_size)
        self._gnss_position_errors = deque(maxlen=history_size)
        self._gnss_longitudinal_errors = deque(maxlen=history_size)
        self._gnss_lateral_errors = deque(maxlen=history_size)
        self._signed_gnss_longitudinal_errors = deque(maxlen=history_size)
        self._signed_gnss_lateral_errors = deque(maxlen=history_size)
        self._pending_gnss_odometry = deque(maxlen=200)
        self._gnss_estimate_origin = None
        self._gnss_reference_origin = None
        self._sample_count = 0
        self._gnss_sample_count = 0
        self._gnss_unmatched_count = 0
        self._unmatched_count = 0
        self._invalid_count = 0
        self._latest_pair_delay_s = None
        self._latest_estimate_stamp_ns = None
        self._latest_gnss_odometry_stamp_ns = None
        self._previous_aligned_pair = None
        self._pose_jump_threshold_m = float(
            self.get_parameter('pose_jump_threshold_m').value)
        self._pose_jump_count = 0
        if self._pose_jump_threshold_m <= 0.0:
            raise ValueError('pose_jump_threshold_m must be positive')

        self._transition_thresholds = {
            'minimum_sample_count': int(self.get_parameter(
                'transition_minimum_moving_sample_count').value),
            'maximum_estimate_age_s': float(self.get_parameter(
                'transition_maximum_estimate_age_s').value),
            'maximum_gnss_age_s': float(self.get_parameter(
                'transition_maximum_gnss_age_s').value),
            'maximum_position_p95_m': float(self.get_parameter(
                'transition_maximum_position_p95_m').value),
            'maximum_longitudinal_p95_m': float(self.get_parameter(
                'transition_maximum_longitudinal_p95_m').value),
            'maximum_lateral_p95_m': float(self.get_parameter(
                'transition_maximum_lateral_p95_m').value),
            'maximum_absolute_longitudinal_bias_m': float(
                self.get_parameter(
                    'transition_maximum_absolute_longitudinal_bias_m'
                ).value
            ),
            'maximum_relative_translation_p95_m': float(self.get_parameter(
                'transition_maximum_relative_translation_p95_m').value),
            'maximum_pose_jump_count': int(self.get_parameter(
                'transition_maximum_pose_jump_count').value),
        }
        if self._transition_thresholds['minimum_sample_count'] <= 0:
            raise ValueError(
                'transition_minimum_sample_count must be positive')
        for key in (
            'maximum_estimate_age_s',
            'maximum_gnss_age_s',
            'maximum_position_p95_m',
            'maximum_longitudinal_p95_m',
            'maximum_lateral_p95_m',
            'maximum_absolute_longitudinal_bias_m',
            'maximum_relative_translation_p95_m',
        ):
            if self._transition_thresholds[key] <= 0.0:
                raise ValueError('{} must be positive'.format(key))
        if self._transition_thresholds['maximum_pose_jump_count'] < 0:
            raise ValueError(
                'transition_maximum_pose_jump_count must be non-negative')
        self._active_navigation_tf_ready = bool(self.get_parameter(
            'active_navigation_tf_ready').value)
        self._independent_velocity_ready = bool(self.get_parameter(
            'independent_velocity_ready').value)
        self._transition_minimum_reference_speed_mps = float(
            self.get_parameter(
                'transition_minimum_reference_speed_mps').value)
        if self._transition_minimum_reference_speed_mps <= 0.0:
            raise ValueError(
                'transition_minimum_reference_speed_mps must be positive')

        output_root = os.path.expanduser(str(
            self.get_parameter('output_directory').value))
        run_name = 'run_{}_{}'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
            os.getpid(),
        )
        self._run_directory = os.path.join(output_root, run_name)
        os.makedirs(self._run_directory, exist_ok=True)
        csv_path = os.path.join(
            self._run_directory, 'localization_shadow.csv')
        self._csv_file = open(csv_path, 'w', newline='', buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'stamp_ns', 'pair_delay_s',
            'ground_truth_x_m', 'ground_truth_y_m',
            'aligned_estimate_x_m', 'aligned_estimate_y_m',
            'position_error_m',
            'longitudinal_error_m', 'lateral_error_m',
            'relative_translation_error_m', 'pose_jump',
            'yaw_error_deg', 'speed_error_mps',
            'estimate_covariance_x', 'estimate_covariance_y',
            'estimate_covariance_yaw',
        ])
        gnss_csv_path = os.path.join(
            self._run_directory, 'gnss_adapter.csv')
        self._gnss_csv_file = open(
            gnss_csv_path, 'w', newline='', buffering=1)
        self._gnss_csv_writer = csv.writer(self._gnss_csv_file)
        self._gnss_csv_writer.writerow([
            'stamp_ns', 'pair_delay_s',
            'ground_truth_x_m', 'ground_truth_y_m',
            'aligned_gnss_x_m', 'aligned_gnss_y_m',
            'position_error_m',
            'longitudinal_error_m', 'lateral_error_m',
            'gnss_covariance_x', 'gnss_covariance_y',
        ])

        self._status_publisher = self.create_publisher(
            String, status_topic, 10)
        self.create_subscription(
            Odometry, ground_truth_topic, self._on_ground_truth, 50)
        self.create_subscription(
            Odometry, estimate_topic, self._on_estimate, 50)
        self.create_subscription(
            Odometry,
            str(self.get_parameter('gnss_odometry_topic').value),
            self._on_gnss_odometry,
            50,
        )
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            'Localization shadow evaluator: {} against {}; output={}'.format(
                estimate_topic,
                ground_truth_topic,
                self._run_directory,
            )
        )

    def _on_ground_truth(self, message):
        self._ground_truth.append((_stamp_ns(message), message))
        self._process_pending_gnss_odometry()

    def _on_gnss_odometry(self, message):
        self._latest_gnss_odometry_stamp_ns = _stamp_ns(message)
        self._pending_gnss_odometry.append(
            (self._latest_gnss_odometry_stamp_ns, message))
        self._process_pending_gnss_odometry()

    def _process_pending_gnss_odometry(self):
        if not self._ground_truth:
            return
        latest_truth_stamp_ns = self._ground_truth[-1][0]
        while self._pending_gnss_odometry:
            stamp_ns, message = self._pending_gnss_odometry[0]
            if latest_truth_stamp_ns < stamp_ns:
                return
            self._pending_gnss_odometry.popleft()
            truth_stamp_ns, truth = min(
                self._ground_truth,
                key=lambda item: abs(item[0] - stamp_ns),
            )
            pair_delay_ns = abs(truth_stamp_ns - stamp_ns)
            if pair_delay_ns > self._maximum_pair_delay_ns:
                self._gnss_unmatched_count += 1
                continue
            gnss_xy = (
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y),
            )
            truth_pose = (
                float(truth.pose.pose.position.x),
                float(truth.pose.pose.position.y),
                quaternion_yaw(truth.pose.pose.orientation),
            )
            if not all(math.isfinite(value) for value in gnss_xy + truth_pose):
                self._invalid_count += 1
                continue
            if self._gnss_estimate_origin is None:
                self._gnss_estimate_origin = gnss_xy
                self._gnss_reference_origin = truth_pose[:2]
            aligned_x = (
                self._gnss_reference_origin[0]
                + gnss_xy[0] - self._gnss_estimate_origin[0]
            )
            aligned_y = (
                self._gnss_reference_origin[1]
                + gnss_xy[1] - self._gnss_estimate_origin[1]
            )
            position_error = math.hypot(
                aligned_x - truth_pose[0],
                aligned_y - truth_pose[1],
            )
            longitudinal_error, lateral_error = planar_error_components(
                (aligned_x, aligned_y), truth_pose)
            self._gnss_position_errors.append(position_error)
            self._gnss_longitudinal_errors.append(abs(longitudinal_error))
            self._gnss_lateral_errors.append(abs(lateral_error))
            self._signed_gnss_longitudinal_errors.append(
                longitudinal_error)
            self._signed_gnss_lateral_errors.append(lateral_error)
            self._gnss_sample_count += 1
            covariance = message.pose.covariance
            self._gnss_csv_writer.writerow([
                stamp_ns,
                pair_delay_ns / 1e9,
                truth_pose[0], truth_pose[1],
                aligned_x, aligned_y,
                position_error,
                longitudinal_error, lateral_error,
                covariance[0], covariance[7],
            ])

    def _on_estimate(self, message):
        if not self._ground_truth:
            self._unmatched_count += 1
            return
        estimate_stamp_ns = _stamp_ns(message)
        truth_stamp_ns, truth = min(
            self._ground_truth,
            key=lambda item: abs(item[0] - estimate_stamp_ns),
        )
        pair_delay_ns = abs(truth_stamp_ns - estimate_stamp_ns)
        if pair_delay_ns > self._maximum_pair_delay_ns:
            self._unmatched_count += 1
            return

        estimate_pose = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            quaternion_yaw(message.pose.pose.orientation),
        )
        truth_pose = (
            truth.pose.pose.position.x,
            truth.pose.pose.position.y,
            quaternion_yaw(truth.pose.pose.orientation),
        )
        if not all(
                math.isfinite(value)
                for value in estimate_pose + truth_pose):
            self._invalid_count += 1
            return
        if self._estimate_origin is None:
            self._estimate_origin = estimate_pose
            self._reference_origin = truth_pose
        aligned = align_enu_pose(
            estimate_pose, self._estimate_origin, self._reference_origin)
        position_error = math.hypot(
            aligned[0] - truth_pose[0],
            aligned[1] - truth_pose[1],
        )
        longitudinal_error, lateral_error = planar_error_components(
            aligned[:2], truth_pose)
        aligned_pair = (
            aligned[0], aligned[1], truth_pose[0], truth_pose[1])
        relative_error = None
        pose_jump = False
        if self._previous_aligned_pair is not None:
            relative_error = relative_translation_error(
                aligned_pair, self._previous_aligned_pair)
            pose_jump = relative_error > self._pose_jump_threshold_m
            if pose_jump:
                self._pose_jump_count += 1
            self._relative_translation_errors.append(relative_error)
        self._previous_aligned_pair = aligned_pair
        yaw_error = abs(wrap_angle(aligned[2] - truth_pose[2]))
        speed_error = math.hypot(
            message.twist.twist.linear.x - truth.twist.twist.linear.x,
            message.twist.twist.linear.y - truth.twist.twist.linear.y,
        )
        reference_speed = math.hypot(
            truth.twist.twist.linear.x,
            truth.twist.twist.linear.y,
        )
        if not all(math.isfinite(value) for value in (
                position_error, yaw_error, speed_error)):
            self._invalid_count += 1
            return

        covariance = message.pose.covariance
        self._position_errors.append(position_error)
        self._longitudinal_errors.append(abs(longitudinal_error))
        self._lateral_errors.append(abs(lateral_error))
        self._signed_longitudinal_errors.append(longitudinal_error)
        self._signed_lateral_errors.append(lateral_error)
        self._yaw_errors.append(math.degrees(yaw_error))
        self._speed_errors.append(speed_error)
        if reference_speed >= self._transition_minimum_reference_speed_mps:
            self._moving_position_errors.append(position_error)
            self._moving_longitudinal_errors.append(abs(longitudinal_error))
            self._moving_lateral_errors.append(abs(lateral_error))
            self._moving_signed_longitudinal_errors.append(
                longitudinal_error)
            self._moving_signed_lateral_errors.append(lateral_error)
            if relative_error is not None:
                self._moving_relative_translation_errors.append(
                    relative_error)
        self._sample_count += 1
        self._latest_pair_delay_s = pair_delay_ns / 1e9
        self._latest_estimate_stamp_ns = estimate_stamp_ns
        self._csv_writer.writerow([
            estimate_stamp_ns,
            self._latest_pair_delay_s,
            truth_pose[0], truth_pose[1], aligned[0], aligned[1],
            position_error,
            longitudinal_error, lateral_error,
            '' if relative_error is None else relative_error,
            int(pose_jump),
            math.degrees(yaw_error), speed_error,
            covariance[0], covariance[7], covariance[35],
        ])

    def _publish_status(self):
        now_ns = self.get_clock().now().nanoseconds
        estimate_age_s = None
        if self._latest_estimate_stamp_ns is not None:
            estimate_age_s = max(
                0.0, (now_ns - self._latest_estimate_stamp_ns) / 1e9)
        gnss_odometry_age_s = None
        if self._latest_gnss_odometry_stamp_ns is not None:
            gnss_odometry_age_s = max(
                0.0,
                (now_ns - self._latest_gnss_odometry_stamp_ns) / 1e9,
            )
        position_summary = summarize_errors(self._position_errors)
        lateral_summary = summarize_errors(self._lateral_errors)
        relative_summary = summarize_errors(
            self._relative_translation_errors)
        moving_position_summary = summarize_errors(
            self._moving_position_errors)
        moving_longitudinal_summary = summarize_errors(
            self._moving_longitudinal_errors)
        moving_lateral_summary = summarize_errors(
            self._moving_lateral_errors)
        moving_signed_longitudinal_summary = summarize_signed_errors(
            self._moving_signed_longitudinal_errors)
        moving_signed_lateral_summary = summarize_signed_errors(
            self._moving_signed_lateral_errors)
        moving_relative_summary = summarize_errors(
            self._moving_relative_translation_errors)
        moving_longitudinal_mean = moving_signed_longitudinal_summary['mean']
        transition = assess_localization_transition(
            sample_count=len(self._moving_position_errors),
            estimate_age_s=estimate_age_s,
            gnss_age_s=gnss_odometry_age_s,
            position_p95_m=moving_position_summary['p95'],
            longitudinal_p95_m=moving_longitudinal_summary['p95'],
            lateral_p95_m=moving_lateral_summary['p95'],
            absolute_longitudinal_bias_m=(
                None if moving_longitudinal_mean is None
                else abs(moving_longitudinal_mean)
            ),
            relative_translation_p95_m=moving_relative_summary['p95'],
            pose_jump_count=self._pose_jump_count,
            active_navigation_tf_ready=self._active_navigation_tf_ready,
            independent_velocity_ready=self._independent_velocity_ready,
            **self._transition_thresholds,
        )
        transition['thresholds'] = dict(self._transition_thresholds)
        transition['active_navigation_tf_ready'] = (
            self._active_navigation_tf_ready)
        transition['independent_velocity_ready'] = (
            self._independent_velocity_ready)
        transition['evaluation_scope'] = 'moving_samples_only'
        transition['minimum_reference_speed_mps'] = (
            self._transition_minimum_reference_speed_mps)
        status = {
            'mode': 'sensor_shadow',
            'control_effect': 'none_ground_truth_odometry_remains_active',
            'filter_inputs': [
                '/vectornav/imu',
                '/localization/odometry_gps_shadow',
            ],
            'ground_truth_input_used_by_filter': False,
            'ground_truth_input_used_by_evaluator': True,
            'wheel_encoder_input_available': (
                self._independent_velocity_ready),
            'alignment': 'enu_translation_plus_independent_yaw',
            'initialized': self._estimate_origin is not None,
            'sample_count': self._sample_count,
            'unmatched_count': self._unmatched_count,
            'invalid_count': self._invalid_count,
            'latest_pair_delay_s': self._latest_pair_delay_s,
            'estimate_age_s': estimate_age_s,
            'gnss_odometry_received': (
                self._latest_gnss_odometry_stamp_ns is not None
            ),
            'gnss_odometry_age_s': gnss_odometry_age_s,
            'position_error_m': position_summary,
            'longitudinal_error_m': summarize_errors(
                self._longitudinal_errors),
            'lateral_error_m': lateral_summary,
            'signed_longitudinal_bias_m': summarize_signed_errors(
                self._signed_longitudinal_errors),
            'signed_lateral_bias_m': summarize_signed_errors(
                self._signed_lateral_errors),
            'relative_translation_error_m': relative_summary,
            'pose_jump_threshold_m': self._pose_jump_threshold_m,
            'pose_jump_count': self._pose_jump_count,
            'yaw_error_deg': summarize_errors(self._yaw_errors),
            'speed_error_mps': summarize_errors(self._speed_errors),
            'moving_evaluation': {
                'minimum_reference_speed_mps': (
                    self._transition_minimum_reference_speed_mps),
                'sample_count': len(self._moving_position_errors),
                'position_error_m': moving_position_summary,
                'longitudinal_error_m': moving_longitudinal_summary,
                'lateral_error_m': moving_lateral_summary,
                'signed_longitudinal_bias_m': (
                    moving_signed_longitudinal_summary),
                'signed_lateral_bias_m': moving_signed_lateral_summary,
                'relative_translation_error_m': moving_relative_summary,
            },
            'gnss_adapter': {
                'sample_count': self._gnss_sample_count,
                'unmatched_count': self._gnss_unmatched_count,
                'position_error_m': summarize_errors(
                    self._gnss_position_errors),
                'longitudinal_error_m': summarize_errors(
                    self._gnss_longitudinal_errors),
                'lateral_error_m': summarize_errors(
                    self._gnss_lateral_errors),
                'signed_longitudinal_bias_m': summarize_signed_errors(
                    self._signed_gnss_longitudinal_errors),
                'signed_lateral_bias_m': summarize_signed_errors(
                    self._signed_gnss_lateral_errors),
            },
            'transition_readiness': transition,
            'run_directory': self._run_directory,
        }
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self._status_publisher.publish(message)

    def destroy_node(self):
        if getattr(self, '_csv_file', None) is not None:
            self._csv_file.close()
            self._csv_file = None
        if getattr(self, '_gnss_csv_file', None) is not None:
            self._gnss_csv_file.close()
            self._gnss_csv_file = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationShadowEvaluator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
