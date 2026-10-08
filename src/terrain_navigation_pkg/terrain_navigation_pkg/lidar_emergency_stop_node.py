"""Gate navigation velocity commands with a 3D LiDAR emergency stop."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as PathMessage
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String, UInt32

from .safety_core import CollisionStopLatch
from .safety_core import EmergencyStopHysteresis
from .safety_core import ForwardObstacleObservation
from .safety_core import RecoveryCurvatureHold
from .safety_core import observe_forward_corridor
from .local_avoidance_core import observe_commanded_trajectory
from .local_avoidance_core import sample_constant_curvature_path
from .waypoint_route_core import parse_waypoint_route_json
from .ground_obstacle_core import extract_ground_relative_obstacles


SAFETY_LOG_FIELDS = [
    'wall_time_iso',
    'state',
    'emergency_stop',
    'cloud_age_s',
    'command_age_s',
    'cloud_stamp_ns',
    'command_sequence',
    'stop_point_count',
    'clear_point_count',
    'nearest_distance_m',
    'raw_command_sequence',
    'raw_input_speed_mps',
    'raw_input_yaw_rate_rps',
    'raw_input_curvature_per_m',
    'input_speed_mps',
    'input_yaw_rate_rps',
    'output_speed_mps',
    'output_yaw_rate_rps',
    'output_curvature_per_m',
    'corridor_mode',
    'commanded_curvature_per_m',
    'tested_curvature_count',
    'nominal_stop_point_count',
    'nominal_clear_point_count',
    'trigger_x_min_m',
    'trigger_x_max_m',
    'trigger_y_min_m',
    'trigger_y_max_m',
    'trigger_z_min_m',
    'trigger_z_max_m',
    'trigger_x_mean_m',
    'trigger_y_mean_m',
    'trigger_z_mean_m',
    'caution_active',
    'collision_latched',
    'collision_event_count',
    'last_collision_intensity',
    'last_collision_age_s',
    'collision_reset_count',
    'trajectory_rejection_count',
    'recovery_active',
    'recovery_curvature_per_m',
]


class SafetyCsvLogger:
    """Write a line-buffered safety timeline for post-run inspection."""

    def __init__(self, root_directory):
        root = Path(root_directory).expanduser().resolve()
        run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        self.run_directory = root / run_name
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.path = self.run_directory / 'safety.csv'
        self.stream = self.path.open(
            'w', encoding='utf-8', newline='', buffering=1
        )
        self.writer = csv.DictWriter(
            self.stream, fieldnames=SAFETY_LOG_FIELDS
        )
        self.writer.writeheader()

    def write(self, values):
        self.writer.writerow(values)
        self.stream.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.close()


class LidarEmergencyStopNode(Node):
    """Publish a final command only when the forward corridor is clear."""

    def __init__(self):
        super().__init__('lidar_emergency_stop_node')
        self.declare_parameter('input_cloud_topic', '/lidar/points')
        self.declare_parameter(
            'input_command_topic', '/cmd_vel_avoidance'
        )
        self.declare_parameter(
            'raw_command_topic', '/nav2/cmd_vel_raw'
        )
        self.declare_parameter('output_command_topic', '/cmd_vel')
        self.declare_parameter(
            'emergency_stop_topic', '/safety/emergency_stop'
        )
        self.declare_parameter(
            'nearest_obstacle_topic', '/safety/nearest_obstacle_distance'
        )
        self.declare_parameter('state_topic', '/safety/state')
        self.declare_parameter('point_count_topic', '/safety/obstacle_points')
        self.declare_parameter(
            'trajectory_rejection_topic', '/safety/trajectory_rejection'
        )
        self.declare_parameter(
            'trigger_points_topic', '/safety/trigger_points'
        )
        self.declare_parameter(
            'checked_trajectory_topic', '/safety/checked_trajectory'
        )
        self.declare_parameter('collision_topic', '/vehicle/collision')
        self.declare_parameter(
            'collision_reset_topic', '/safety/reset_collision'
        )
        self.declare_parameter('route_topic', '/navigation/waypoint_route')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('lidar_timeout_s', 1.0)
        self.declare_parameter('command_timeout_s', 0.6)
        # The roof LiDAR sees the Lincoln hood up to about x=2.0 m. Start the
        # safety corridor beyond the ego footprint to reject those returns.
        self.declare_parameter('minimum_x_m', 2.5)
        self.declare_parameter('stop_distance_m', 4.0)
        self.declare_parameter('clear_distance_m', 6.0)
        self.declare_parameter('corridor_half_width_m', 1.35)
        self.declare_parameter('vehicle_front_m', 2.4)
        self.declare_parameter('vehicle_rear_m', 2.5)
        self.declare_parameter('vehicle_width_m', 2.0)
        self.declare_parameter('minimum_z_m', -1.4)
        self.declare_parameter('maximum_z_m', 1.0)
        self.declare_parameter('ground_relative_low_obstacles_enabled', True)
        self.declare_parameter('ground_fit_minimum_range_m', 2.0)
        self.declare_parameter('ground_fit_maximum_range_m', 20.0)
        self.declare_parameter('ground_fit_minimum_z_m', -2.5)
        self.declare_parameter('ground_fit_maximum_z_m', -0.8)
        self.declare_parameter('ground_seed_quantile', 0.35)
        self.declare_parameter('ground_inlier_tolerance_m', 0.06)
        self.declare_parameter('minimum_ground_points', 80)
        self.declare_parameter('low_obstacle_minimum_height_m', 0.07)
        self.declare_parameter('low_obstacle_maximum_height_m', 0.45)
        self.declare_parameter('local_ground_enabled', True)
        self.declare_parameter('local_ground_resolution_m', 0.75)
        self.declare_parameter('local_ground_radius_m', 1.50)
        self.declare_parameter('local_ground_quantile', 0.25)
        self.declare_parameter('local_ground_plane_enabled', True)
        self.declare_parameter('minimum_obstacle_points', 20)
        self.declare_parameter('clear_required_scans', 3)
        self.declare_parameter('use_commanded_trajectory', True)
        self.declare_parameter('trajectory_sample_spacing_m', 0.25)
        self.declare_parameter('maximum_curvature_per_m', 0.25)
        self.declare_parameter(
            'trajectory_curvature_uncertainty_per_m', 0.04
        )
        self.declare_parameter('trajectory_safety_extra_width_m', 0.25)
        self.declare_parameter('caution_speed_mps', 0.35)
        self.declare_parameter('hold_last_safe_trajectory', True)
        self.declare_parameter('minimum_recovery_curvature_per_m', 0.06)
        self.declare_parameter('recovery_candidate_max_age_s', 1.0)
        self.declare_parameter('recovery_max_duration_s', 8.0)
        self.declare_parameter('recovery_curvature_smoothing_alpha', 0.35)
        self.declare_parameter('collision_latch_enabled', True)
        self.declare_parameter('minimum_collision_intensity', 1.0)
        self.declare_parameter('reset_collision_on_new_route', True)
        self.declare_parameter('terminal_log_period_s', 2.0)
        self.declare_parameter('log_safety', True)
        self.declare_parameter(
            'log_directory', '~/terrain_nav_data/logs/safety'
        )

        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.lidar_timeout_s = float(
            self.get_parameter('lidar_timeout_s').value
        )
        self.command_timeout_s = float(
            self.get_parameter('command_timeout_s').value
        )
        self.minimum_x_m = float(
            self.get_parameter('minimum_x_m').value
        )
        self.stop_distance_m = float(
            self.get_parameter('stop_distance_m').value
        )
        self.clear_distance_m = float(
            self.get_parameter('clear_distance_m').value
        )
        self.corridor_half_width_m = float(
            self.get_parameter('corridor_half_width_m').value
        )
        self.vehicle_front_m = float(
            self.get_parameter('vehicle_front_m').value
        )
        self.vehicle_rear_m = float(
            self.get_parameter('vehicle_rear_m').value
        )
        self.vehicle_width_m = float(
            self.get_parameter('vehicle_width_m').value
        )
        self.minimum_z_m = float(
            self.get_parameter('minimum_z_m').value
        )
        self.maximum_z_m = float(
            self.get_parameter('maximum_z_m').value
        )
        self.ground_filter_parameters = {
            'enabled': bool(self.get_parameter(
                'ground_relative_low_obstacles_enabled'
            ).value),
            'ground_fit_minimum_range_m': float(self.get_parameter(
                'ground_fit_minimum_range_m'
            ).value),
            'ground_fit_maximum_range_m': float(self.get_parameter(
                'ground_fit_maximum_range_m'
            ).value),
            'ground_fit_minimum_z_m': float(self.get_parameter(
                'ground_fit_minimum_z_m'
            ).value),
            'ground_fit_maximum_z_m': float(self.get_parameter(
                'ground_fit_maximum_z_m'
            ).value),
            'ground_seed_quantile': float(self.get_parameter(
                'ground_seed_quantile'
            ).value),
            'ground_inlier_tolerance_m': float(self.get_parameter(
                'ground_inlier_tolerance_m'
            ).value),
            'minimum_ground_points': int(self.get_parameter(
                'minimum_ground_points'
            ).value),
            'low_obstacle_minimum_height_m': float(self.get_parameter(
                'low_obstacle_minimum_height_m'
            ).value),
            'low_obstacle_maximum_height_m': float(self.get_parameter(
                'low_obstacle_maximum_height_m'
            ).value),
            'local_ground_enabled': bool(self.get_parameter(
                'local_ground_enabled'
            ).value),
            'local_ground_resolution_m': float(self.get_parameter(
                'local_ground_resolution_m'
            ).value),
            'local_ground_radius_m': float(self.get_parameter(
                'local_ground_radius_m'
            ).value),
            'local_ground_quantile': float(self.get_parameter(
                'local_ground_quantile'
            ).value),
            'local_ground_plane_enabled': bool(self.get_parameter(
                'local_ground_plane_enabled'
            ).value),
        }
        # ``latest_points`` is already height-filtered by
        # ``extract_ground_relative_obstacles``.  The swept-footprint helper
        # still requires finite z bounds, however, so retain a finite lower
        # bound that also includes the restored curb / low-obstacle returns.
        # Passing +/-inf here makes the helper reject every cloud as invalid.
        self.filtered_minimum_z_m = min(
            self.minimum_z_m,
            self.ground_filter_parameters['ground_fit_minimum_z_m'],
        )
        self.filtered_maximum_z_m = max(
            self.maximum_z_m,
            self.ground_filter_parameters['ground_fit_maximum_z_m'],
        )
        extract_ground_relative_obstacles(
            [],
            fixed_minimum_z_m=self.minimum_z_m,
            fixed_maximum_z_m=self.maximum_z_m,
            **self.ground_filter_parameters,
        )
        self.terminal_log_period_s = float(
            self.get_parameter('terminal_log_period_s').value
        )
        minimum_points = int(
            self.get_parameter('minimum_obstacle_points').value
        )
        self.minimum_obstacle_points = minimum_points
        clear_required_scans = int(
            self.get_parameter('clear_required_scans').value
        )
        self.use_commanded_trajectory = bool(
            self.get_parameter('use_commanded_trajectory').value
        )
        self.trajectory_sample_spacing_m = float(
            self.get_parameter('trajectory_sample_spacing_m').value
        )
        self.maximum_curvature_per_m = float(
            self.get_parameter('maximum_curvature_per_m').value
        )
        self.trajectory_curvature_uncertainty_per_m = float(
            self.get_parameter(
                'trajectory_curvature_uncertainty_per_m'
            ).value
        )
        self.trajectory_safety_extra_width_m = float(
            self.get_parameter('trajectory_safety_extra_width_m').value
        )
        self.caution_speed_mps = float(
            self.get_parameter('caution_speed_mps').value
        )
        self.hold_last_safe_trajectory = bool(
            self.get_parameter('hold_last_safe_trajectory').value
        )
        minimum_recovery_curvature = float(
            self.get_parameter('minimum_recovery_curvature_per_m').value
        )
        recovery_candidate_max_age = float(
            self.get_parameter('recovery_candidate_max_age_s').value
        )
        recovery_max_duration = float(
            self.get_parameter('recovery_max_duration_s').value
        )
        recovery_smoothing_alpha = float(
            self.get_parameter(
                'recovery_curvature_smoothing_alpha'
            ).value
        )
        self.collision_latch_enabled = bool(
            self.get_parameter('collision_latch_enabled').value
        )
        self.reset_collision_on_new_route = bool(
            self.get_parameter('reset_collision_on_new_route').value
        )
        minimum_collision_intensity = float(
            self.get_parameter('minimum_collision_intensity').value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self.lidar_timeout_s <= 0.0 or self.command_timeout_s <= 0.0:
            raise ValueError('input timeouts must be positive')
        if (
            self.trajectory_sample_spacing_m <= 0.0
            or self.maximum_curvature_per_m <= 0.0
        ):
            raise ValueError('trajectory safety parameters must be positive')
        if min(
            self.vehicle_front_m,
            self.vehicle_rear_m,
            self.vehicle_width_m,
        ) <= 0.0:
            raise ValueError('vehicle footprint lengths must be positive')
        if self.trajectory_curvature_uncertainty_per_m < 0.0:
            raise ValueError('trajectory curvature uncertainty is invalid')
        if self.trajectory_safety_extra_width_m < 0.0:
            raise ValueError('trajectory safety width is invalid')
        if self.caution_speed_mps <= 0.0:
            raise ValueError('caution_speed_mps must be positive')

        # Validate all geometric parameters before the first cloud arrives.
        observe_forward_corridor(
            [],
            self.minimum_x_m,
            self.stop_distance_m,
            self.clear_distance_m,
            self.corridor_half_width_m,
            self.minimum_z_m,
            self.maximum_z_m,
        )
        self.monitor = EmergencyStopHysteresis(
            minimum_points,
            clear_required_scans,
        )
        self.recovery_hold = RecoveryCurvatureHold(
            minimum_recovery_curvature,
            recovery_candidate_max_age,
            recovery_max_duration,
            recovery_smoothing_alpha,
        )
        self.collision_latch = CollisionStopLatch(
            minimum_collision_intensity
        )
        self.observation = ForwardObstacleObservation(0, 0, math.inf)
        self.nominal_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.caution_active = False
        self.latest_command = Twist()
        self.latest_raw_command = Twist()
        self.cloud_wall_time = None
        self.command_wall_time = None
        self.cloud_valid = False
        self.cloud_error = None
        self.cloud_stamp_ns = 0
        self.cloud_sequence = 0
        self.command_sequence = 0
        self.raw_command_sequence = 0
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.nominal_stop_points = np.empty((0, 3), dtype=np.float32)
        self.latest_cloud_header = None
        self.commanded_curvature_per_m = 0.0
        self.previous_commanded_curvature_per_m = 0.0
        self.tested_curvature_count = 1
        self.collision_event_count = 0
        self.last_collision_intensity = None
        self.last_collision_wall_time = None
        self.collision_reset_count = 0
        self.last_state = None
        self.last_terminal_log_time = None
        self.last_rejection_key = None
        self.trajectory_rejection_count = 0
        self.recovery_active = False
        self.recovery_curvature_per_m = 0.0

        self.command_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter('output_command_topic').value),
            10,
        )
        self.stop_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter('emergency_stop_topic').value),
            10,
        )
        self.nearest_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter('nearest_obstacle_topic').value),
            10,
        )
        self.state_publisher = self.create_publisher(
            String,
            str(self.get_parameter('state_topic').value),
            10,
        )
        self.point_count_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('point_count_topic').value),
            10,
        )
        self.trajectory_rejection_publisher = self.create_publisher(
            String,
            str(self.get_parameter('trajectory_rejection_topic').value),
            10,
        )
        self.trigger_points_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('trigger_points_topic').value),
            10,
        )
        self.checked_trajectory_publisher = self.create_publisher(
            PathMessage,
            str(self.get_parameter('checked_trajectory_topic').value),
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('input_cloud_topic').value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter('input_command_topic').value),
            self._on_command,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter('raw_command_topic').value),
            self._on_raw_command,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter('collision_topic').value),
            self._on_collision,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('collision_reset_topic').value),
            self._on_collision_reset,
            10,
        )
        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_topic').value),
            self._on_waypoint_route,
            latched_qos,
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish_safe_command,
        )

        self.safety_logger = None
        if bool(self.get_parameter('log_safety').value):
            self.safety_logger = SafetyCsvLogger(
                str(self.get_parameter('log_directory').value)
            )
            self.get_logger().info(
                'Safety CSV: {}'.format(self.safety_logger.path)
            )
        self.get_logger().info(
            'LiDAR emergency stop ready: stop <= {:.1f} m, clear > {:.1f} m, '
            'corridor width {:.2f} m, points >= {}, mode={}'.format(
                self.stop_distance_m,
                self.clear_distance_m,
                2.0 * self.corridor_half_width_m,
                minimum_points,
                'commanded_trajectory'
                if self.use_commanded_trajectory else 'fixed_forward',
            )
        )

    def _on_cloud(self, message):
        try:
            points = point_cloud2.read_points_numpy(
                message,
                field_names=['x', 'y', 'z'],
                skip_nans=True,
            )
            raw_points = np.asarray(points)[:, :3].astype(
                np.float32, copy=True
            )
            ground_result = extract_ground_relative_obstacles(
                raw_points,
                fixed_minimum_z_m=self.minimum_z_m,
                fixed_maximum_z_m=self.maximum_z_m,
                **self.ground_filter_parameters,
            )
            self.latest_points = raw_points[ground_result.obstacle_mask]
            self.latest_cloud_header = message.header
            self.cloud_sequence += 1
            self.cloud_stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            self._update_observation(allow_clear_progress=True)
            self._publish_safety_debug()
            self.cloud_valid = True
            self.cloud_error = None
        except Exception as error:
            error_text = str(error)
            if error_text != self.cloud_error:
                self.get_logger().error(
                    'LiDAR safety cloud processing failed: {}'.format(
                        error_text
                    )
                )
            self.cloud_valid = False
            self.cloud_error = error_text
        self.cloud_wall_time = time.monotonic()

    def _on_command(self, message):
        self.latest_command = message
        self.command_wall_time = time.monotonic()
        self.command_sequence += 1
        # Validate a newly selected curve immediately against the latest cloud.
        # Reusing a cloud may trigger a stop, but cannot consume additional
        # clear-hysteresis scans.
        if self.cloud_valid:
            try:
                self._update_observation(allow_clear_progress=False)
            except Exception as error:
                error_text = str(error)
                if error_text != self.cloud_error:
                    self.get_logger().error(
                        'LiDAR safety command recheck failed: {}'.format(
                            error_text
                        )
                    )
                self.cloud_valid = False
                self.cloud_error = error_text

    def _on_raw_command(self, message):
        self.latest_raw_command = message
        self.raw_command_sequence += 1

    def _on_collision(self, message):
        if not self.collision_latch_enabled:
            return
        intensity = float(message.data)
        was_latched = self.collision_latch.latched
        if not self.collision_latch.observe(intensity):
            return
        self.collision_event_count += 1
        self.last_collision_intensity = intensity
        self.last_collision_wall_time = time.monotonic()
        if not was_latched:
            self.get_logger().error(
                'Collision stop LATCHED: intensity={:.2f}. Start a new '
                'route or publish true to /safety/reset_collision before '
                'motion can resume.'.format(intensity)
            )

    def _reset_collision_latch(self, source):
        if not self.collision_latch.latched:
            return
        self.collision_latch.reset()
        self.collision_reset_count += 1
        self.get_logger().warn(
            'Collision stop reset by {} (reset_count={})'.format(
                source,
                self.collision_reset_count,
            )
        )

    def _on_collision_reset(self, message):
        if bool(message.data):
            self._reset_collision_latch('reset topic')

    def _on_waypoint_route(self, message):
        if not self.reset_collision_on_new_route:
            return
        try:
            route = parse_waypoint_route_json(message.data)
        except ValueError:
            return
        if route.start:
            self._reset_collision_latch('new route start')
            self.recovery_hold.invalidate()

    def _publish_trajectory_rejection(self):
        if (
            not self.use_commanded_trajectory
            or abs(float(self.latest_command.linear.x)) <= 0.02
            or self.nominal_observation.stop_point_count
            < self.minimum_obstacle_points
        ):
            return
        key = (
            self.cloud_stamp_ns or self.cloud_sequence,
            round(self.commanded_curvature_per_m, 5),
        )
        if key == self.last_rejection_key:
            return
        payload = {
            'reason': 'obstacle_stop',
            'cloud_stamp_ns': self.cloud_stamp_ns,
            'cloud_sequence': self.cloud_sequence,
            'command_sequence': self.command_sequence,
            'curvature_per_m': self.commanded_curvature_per_m,
            'nominal_stop_point_count': (
                self.nominal_observation.stop_point_count
            ),
            'nominal_clear_point_count': (
                self.nominal_observation.clear_point_count
            ),
            'nearest_distance_m': (
                self.nominal_observation.nearest_distance_m
                if math.isfinite(
                    self.nominal_observation.nearest_distance_m
                ) else None
            ),
        }
        message = String()
        message.data = json.dumps(payload, separators=(',', ':'))
        self.trajectory_rejection_publisher.publish(message)
        self.last_rejection_key = key
        self.trajectory_rejection_count += 1

    def _update_observation(self, allow_clear_progress=True):
        if self.use_commanded_trajectory:
            # The nominal corridor is the footprint of the path the local
            # planner actually selected. Only this corridor may trigger a
            # hard stop; otherwise zeroing the steering command can prevent
            # the vehicle from ever entering its safe avoidance path.
            result = observe_commanded_trajectory(
                self.latest_points,
                float(self.latest_command.linear.x),
                float(self.latest_command.angular.z),
                self.previous_commanded_curvature_per_m,
                self.maximum_curvature_per_m,
                self.trajectory_curvature_uncertainty_per_m,
                self.minimum_x_m,
                self.stop_distance_m,
                self.clear_distance_m,
                self.corridor_half_width_m,
                self.trajectory_safety_extra_width_m,
                self.vehicle_front_m,
                self.vehicle_rear_m,
                self.vehicle_width_m,
                self.filtered_minimum_z_m,
                self.filtered_maximum_z_m,
                self.trajectory_sample_spacing_m,
            )
            self.nominal_observation = result.nominal
            self.nominal_stop_points = result.nominal_stop_points
            self.observation = result.envelope
            self.commanded_curvature_per_m = (
                result.commanded_curvature_per_m
            )
            self.tested_curvature_count = result.tested_curvature_count
            self.caution_active = (
                self.observation.stop_point_count
                >= self.minimum_obstacle_points
                and self.nominal_observation.stop_point_count
                < self.minimum_obstacle_points
            )
            self.previous_commanded_curvature_per_m = (
                self.commanded_curvature_per_m
            )
            if (
                self.hold_last_safe_trajectory
                and abs(float(self.latest_command.linear.x)) > 0.02
                and self.nominal_observation.stop_point_count
                < self.minimum_obstacle_points
            ):
                self.recovery_hold.observe_safe(
                    self.commanded_curvature_per_m,
                    time.monotonic(),
                )
        else:
            self.commanded_curvature_per_m = 0.0
            self.observation = observe_forward_corridor(
                self.latest_points,
                self.minimum_x_m,
                self.stop_distance_m,
                self.clear_distance_m,
                self.corridor_half_width_m,
                -math.inf,
                math.inf,
            )
            self.nominal_observation = self.observation
            self.nominal_stop_points = np.empty((0, 3), dtype=np.float32)
            self.caution_active = False
        self.monitor.update(
            self.nominal_observation,
            allow_clear_progress=allow_clear_progress,
        )
        self._publish_trajectory_rejection()

    def _publish_safety_debug(self):
        """Publish the exact raw points and arc used by the hard-stop test."""
        if self.latest_cloud_header is None:
            return
        trigger_cloud = point_cloud2.create_cloud_xyz32(
            self.latest_cloud_header,
            self.nominal_stop_points.tolist(),
        )
        self.trigger_points_publisher.publish(trigger_cloud)

        checked_path = PathMessage()
        checked_path.header = self.latest_cloud_header
        samples = sample_constant_curvature_path(
            self.commanded_curvature_per_m,
            self.stop_distance_m,
            self.trajectory_sample_spacing_m,
        )
        for x_m, y_m, yaw_rad in samples:
            pose = PoseStamped()
            pose.header = self.latest_cloud_header
            pose.pose.position.x = float(x_m)
            pose.pose.position.y = float(y_m)
            pose.pose.orientation.z = math.sin(0.5 * float(yaw_rad))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw_rad))
            checked_path.poses.append(pose)
        self.checked_trajectory_publisher.publish(checked_path)

    def _copy_latest_command(self):
        output = Twist()
        output.linear.x = self.latest_command.linear.x
        output.linear.y = self.latest_command.linear.y
        output.linear.z = self.latest_command.linear.z
        output.angular.x = self.latest_command.angular.x
        output.angular.y = self.latest_command.angular.y
        output.angular.z = self.latest_command.angular.z
        return output

    def _caution_command(self):
        """Slow down while preserving the selected path curvature."""
        output = self._copy_latest_command()
        input_speed = float(output.linear.x)
        if input_speed <= self.caution_speed_mps:
            return output
        output.linear.x = self.caution_speed_mps
        output.angular.z = (
            self.caution_speed_mps * self.commanded_curvature_per_m
        )
        return output

    def _validate_recovery_command(self, now):
        """Return a slow, revalidated held curve or None for a hard stop."""
        if not self.hold_last_safe_trajectory:
            return None
        curvature = self.recovery_hold.activate_or_current(now)
        if curvature is None:
            return None
        recovery_speed = self.caution_speed_mps
        result = observe_commanded_trajectory(
            self.latest_points,
            recovery_speed,
            recovery_speed * curvature,
            curvature,
            self.maximum_curvature_per_m,
            self.trajectory_curvature_uncertainty_per_m,
            self.minimum_x_m,
            self.stop_distance_m,
            self.clear_distance_m,
            self.corridor_half_width_m,
            self.trajectory_safety_extra_width_m,
            self.vehicle_front_m,
            self.vehicle_rear_m,
            self.vehicle_width_m,
            self.filtered_minimum_z_m,
            self.filtered_maximum_z_m,
            self.trajectory_sample_spacing_m,
        )
        if result.nominal.stop_point_count >= self.minimum_obstacle_points:
            self.recovery_hold.invalidate()
            return None
        output = Twist()
        output.linear.x = recovery_speed
        output.angular.z = recovery_speed * curvature
        self.recovery_active = True
        self.recovery_curvature_per_m = curvature
        return output

    def _safety_state(self, now):
        if self.collision_latch_enabled and self.collision_latch.latched:
            return 'collision_stop'
        if self.cloud_wall_time is None:
            return 'waiting_for_lidar'
        if not self.cloud_valid:
            return 'invalid_lidar'
        if now - self.cloud_wall_time > self.lidar_timeout_s:
            return 'lidar_stale'
        if self.command_wall_time is None:
            return 'waiting_for_command'
        if now - self.command_wall_time > self.command_timeout_s:
            return 'command_stale'
        if self.monitor.stopped:
            return 'obstacle_stop'
        if self.caution_active:
            return 'obstacle_caution'
        return 'clear'

    def _publish_diagnostics(self, state, emergency_stop):
        stop_message = Bool()
        stop_message.data = emergency_stop
        self.stop_publisher.publish(stop_message)
        nearest_message = Float32()
        nearest_message.data = self.observation.nearest_distance_m
        self.nearest_publisher.publish(nearest_message)
        state_message = String()
        state_message.data = state
        self.state_publisher.publish(state_message)
        count_message = UInt32()
        count_message.data = self.observation.clear_point_count
        self.point_count_publisher.publish(count_message)

    def _write_log(self, now, state, emergency_stop, output):
        if self.safety_logger is None:
            return
        cloud_age = (
            now - self.cloud_wall_time
            if self.cloud_wall_time is not None else math.inf
        )
        command_age = (
            now - self.command_wall_time
            if self.command_wall_time is not None else math.inf
        )
        nearest = self.observation.nearest_distance_m
        raw_speed = float(self.latest_raw_command.linear.x)
        raw_yaw_rate = float(self.latest_raw_command.angular.z)
        raw_curvature = (
            raw_yaw_rate / raw_speed if abs(raw_speed) > 0.02 else 0.0
        )
        output_speed = float(output.linear.x)
        output_yaw_rate = float(output.angular.z)
        output_curvature = (
            output_yaw_rate / output_speed
            if abs(output_speed) > 0.02 else 0.0
        )
        if self.nominal_stop_points.shape[0] > 0:
            trigger_min = np.min(self.nominal_stop_points, axis=0)
            trigger_max = np.max(self.nominal_stop_points, axis=0)
            trigger_mean = np.mean(self.nominal_stop_points, axis=0)
            trigger_values = {
                'trigger_x_min_m': '{:.6f}'.format(trigger_min[0]),
                'trigger_x_max_m': '{:.6f}'.format(trigger_max[0]),
                'trigger_y_min_m': '{:.6f}'.format(trigger_min[1]),
                'trigger_y_max_m': '{:.6f}'.format(trigger_max[1]),
                'trigger_z_min_m': '{:.6f}'.format(trigger_min[2]),
                'trigger_z_max_m': '{:.6f}'.format(trigger_max[2]),
                'trigger_x_mean_m': '{:.6f}'.format(trigger_mean[0]),
                'trigger_y_mean_m': '{:.6f}'.format(trigger_mean[1]),
                'trigger_z_mean_m': '{:.6f}'.format(trigger_mean[2]),
            }
        else:
            trigger_values = {
                name: '' for name in (
                    'trigger_x_min_m',
                    'trigger_x_max_m',
                    'trigger_y_min_m',
                    'trigger_y_max_m',
                    'trigger_z_min_m',
                    'trigger_z_max_m',
                    'trigger_x_mean_m',
                    'trigger_y_mean_m',
                    'trigger_z_mean_m',
                )
            }
        log_values = {
            'wall_time_iso': datetime.now().astimezone().isoformat(
                timespec='milliseconds'
            ),
            'state': state,
            'emergency_stop': int(emergency_stop),
            'cloud_age_s': '{:.6f}'.format(cloud_age),
            'command_age_s': '{:.6f}'.format(command_age),
            'cloud_stamp_ns': self.cloud_stamp_ns,
            'command_sequence': self.command_sequence,
            'stop_point_count': self.observation.stop_point_count,
            'clear_point_count': self.observation.clear_point_count,
            'nearest_distance_m': (
                '{:.6f}'.format(nearest) if math.isfinite(nearest) else ''
            ),
            'raw_command_sequence': self.raw_command_sequence,
            'raw_input_speed_mps': '{:.6f}'.format(raw_speed),
            'raw_input_yaw_rate_rps': '{:.6f}'.format(raw_yaw_rate),
            'raw_input_curvature_per_m': '{:.6f}'.format(raw_curvature),
            'input_speed_mps': '{:.6f}'.format(
                self.latest_command.linear.x
            ),
            'input_yaw_rate_rps': '{:.6f}'.format(
                self.latest_command.angular.z
            ),
            'output_speed_mps': '{:.6f}'.format(output_speed),
            'output_yaw_rate_rps': '{:.6f}'.format(output_yaw_rate),
            'output_curvature_per_m': '{:.6f}'.format(output_curvature),
            'corridor_mode': (
                'commanded_trajectory'
                if self.use_commanded_trajectory else 'fixed_forward'
            ),
            'commanded_curvature_per_m': '{:.6f}'.format(
                self.commanded_curvature_per_m
            ),
            'tested_curvature_count': self.tested_curvature_count,
            'nominal_stop_point_count': (
                self.nominal_observation.stop_point_count
            ),
            'nominal_clear_point_count': (
                self.nominal_observation.clear_point_count
            ),
            'caution_active': int(self.caution_active),
            'collision_latched': int(self.collision_latch.latched),
            'collision_event_count': self.collision_event_count,
            'last_collision_intensity': (
                '{:.6f}'.format(self.last_collision_intensity)
                if self.last_collision_intensity is not None else ''
            ),
            'last_collision_age_s': (
                '{:.6f}'.format(now - self.last_collision_wall_time)
                if self.last_collision_wall_time is not None else ''
            ),
            'collision_reset_count': self.collision_reset_count,
            'trajectory_rejection_count': self.trajectory_rejection_count,
            'recovery_active': int(self.recovery_active),
            'recovery_curvature_per_m': '{:.6f}'.format(
                self.recovery_curvature_per_m
            ),
        }
        log_values.update(trigger_values)
        self.safety_logger.write(log_values)

    def _terminal_log(self, now, state):
        changed = state != self.last_state
        elapsed = (
            self.last_terminal_log_time is None
            or self.terminal_log_period_s == 0.0
            or now - self.last_terminal_log_time
            >= self.terminal_log_period_s
        )
        if not changed and not elapsed:
            return
        nearest = self.observation.nearest_distance_m
        nearest_text = (
            '{:.2f}'.format(nearest) if math.isfinite(nearest) else 'none'
        )
        self.get_logger().info(
            'safety_state={} nearest={} m fan_stop_points={} '
            'nominal_stop_points={} clear_points={} input_speed={:.2f} m/s '
            'curvature={:.3f} 1/m collision_latched={}'.format(
                state,
                nearest_text,
                self.observation.stop_point_count,
                self.nominal_observation.stop_point_count,
                self.observation.clear_point_count,
                self.latest_command.linear.x,
                self.commanded_curvature_per_m,
                self.collision_latch.latched,
            )
        )
        if state == 'invalid_lidar' and self.cloud_error:
            self.get_logger().error(
                'PointCloud2 parsing failed: {}'.format(self.cloud_error)
            )
        self.last_state = state
        self.last_terminal_log_time = now

    def _publish_safe_command(self):
        now = time.monotonic()
        state = self._safety_state(now)
        self.recovery_active = False
        self.recovery_curvature_per_m = 0.0
        recovery_output = None
        if state == 'obstacle_stop':
            recovery_output = self._validate_recovery_command(now)
            if recovery_output is not None:
                state = 'obstacle_recovery'
        elif state == 'clear':
            self.recovery_hold.reset_active()
        emergency_stop = state not in (
            'clear', 'obstacle_caution', 'obstacle_recovery'
        )
        if recovery_output is not None:
            output = recovery_output
        elif emergency_stop:
            output = Twist()
        elif state == 'obstacle_caution':
            output = self._caution_command()
        else:
            output = self._copy_latest_command()
        self.command_publisher.publish(output)
        self._publish_diagnostics(state, emergency_stop)
        self._write_log(now, state, emergency_stop, output)
        self._terminal_log(now, state)

    def destroy_node(self):
        if self.safety_logger is not None:
            self.safety_logger.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LidarEmergencyStopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.command_publisher.publish(Twist())
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
