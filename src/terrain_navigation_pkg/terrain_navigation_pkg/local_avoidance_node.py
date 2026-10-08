"""LiDAR-guided detour around obstacles using odometry for path tracking."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path as FilesystemPath
import time

from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String, UInt32

from .local_avoidance_core import LEFT
from .local_avoidance_core import RIGHT
from .local_avoidance_core import associate_progress_interval
from .local_avoidance_core import Pose2D
from .local_avoidance_core import choose_avoidance_side
from .local_avoidance_core import choose_feasible_trajectory_side
from .local_avoidance_core import clamp_tracked_progress_interval
from .local_avoidance_core import compute_trajectory_planning_horizon
from .local_avoidance_core import detour_lateral_offset
from .local_avoidance_core import local_to_world
from .local_avoidance_core import obstacle_requires_detour
from .local_avoidance_core import observe_commanded_trajectory
from .local_avoidance_core import observe_swept_vehicle_footprint
from .local_avoidance_core import observe_detour_corridor
from .local_avoidance_core import PassageObservation
from .local_avoidance_core import Point2D
from .local_avoidance_core import pure_pursuit_command
from .local_avoidance_core import quintic_smoothstep
from .local_avoidance_core import select_collision_free_trajectory
from .local_avoidance_core import trajectory_handoff_ready
from .local_avoidance_core import trajectory_recommit_required
from .local_avoidance_core import trajectory_side_lock_release_permitted
from .local_avoidance_core import update_trajectory_side_lock
from .local_avoidance_core import measure_path_alignment
from .local_avoidance_core import measure_polyline_alignment
from .local_avoidance_core import polyline_lookahead_target
from .local_avoidance_core import update_lateral_shift_confirmation
from .local_avoidance_core import world_to_local
from .safety_core import ForwardObstacleObservation
from .safety_core import observe_forward_corridor
from .waypoint_route_core import parse_waypoint_route_json


LOG_FIELDS = [
    'wall_time_iso',
    'state',
    'selected_side',
    'waypoint_index',
    'center_points',
    'left_points',
    'right_points',
    'nearest_center_m',
    'navigation_path_points',
    'navigation_nominal_points',
    'nearest_navigation_path_m',
    'navigation_curvature_per_m',
    'navigation_tested_curvature_count',
    'navigation_clear_scan_count',
    'avoidance_episode_count',
    'cloud_age_s',
    'odometry_age_s',
    'command_age_s',
    'vehicle_speed_mps',
    'input_speed_mps',
    'output_speed_mps',
    'output_yaw_rate_rps',
    'vehicle_x_m',
    'vehicle_y_m',
    'vehicle_yaw_deg',
    'target_x_m',
    'target_y_m',
    'goal_distance_m',
    'route_status',
    'route_index',
    'route_size',
    'route_loop',
    'terminal_route_goal',
    'trajectory_horizon_m',
    'path_progress_m',
    'actual_lateral_m',
    'shift_completed',
    'safety_stop_active',
    'avoidance_active_elapsed_s',
    'avoidance_phase',
    'passage_not_passed_points',
    'passage_behind_points',
    'passage_obstacle_seen',
    'passage_rear_seen',
    'passage_clear_count',
    'return_corridor_points',
    'tracked_obstacle_min_progress_m',
    'tracked_obstacle_max_progress_m',
    'tracked_obstacle_passed',
    'tracked_obstacle_extent_frozen',
    'tracked_obstacle_seed_progress_m',
    'planner_mode',
    'committed_trajectory_side',
    'trajectory_side_lock_active',
    'trajectory_side_lock_clear_count',
    'signed_lateral_shift_m',
    'lateral_excursion_limited',
    'desired_curvature_per_m',
    'selected_curvature_per_m',
    'valid_trajectory_count',
    'trajectory_candidate_count',
    'trajectory_collision_points',
    'trajectory_safety_validation_points',
    'trajectory_proximity_points',
    'trajectory_nearest_path_m',
    'trajectory_score',
    'recovery_cross_track_m',
    'recovery_heading_error_deg',
    'recovery_aligned',
    'recovery_reference',
    'route_reset_count',
    'cloud_stamp_ns',
    'active_rejected_curvature_count',
    'last_rejected_curvature_per_m',
    'safety_rejection_count',
    'collision_event_count',
    'last_collision_intensity',
    'last_collision_age_s',
]


class AvoidanceCsvLogger:
    def __init__(self, root_directory):
        root = FilesystemPath(root_directory).expanduser().resolve()
        name = datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        self.run_directory = root / name
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.path = self.run_directory / 'avoidance.csv'
        self.stream = self.path.open(
            'w', encoding='utf-8', newline='', buffering=1
        )
        self.writer = csv.DictWriter(self.stream, fieldnames=LOG_FIELDS)
        self.writer.writeheader()

    def write(self, values):
        self.writer.writerow(values)
        self.stream.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.close()


def yaw_from_quaternion(quaternion):
    sine = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cosine = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(sine, cosine)


class LocalAvoidanceNode(Node):
    """Insert a smooth, LiDAR-terminated detour before the safety gate."""

    def __init__(self):
        super().__init__('local_avoidance_node')
        self.declare_parameter('input_cloud_topic', '/lidar/points')
        self.declare_parameter(
            'input_command_topic', '/cmd_vel_navigation'
        )
        self.declare_parameter(
            'output_command_topic', '/cmd_vel_avoidance'
        )
        self.declare_parameter('odometry_topic', '/vehicle/odometry')
        self.declare_parameter(
            'goal_vector_topic', '/navigation/goal_vector'
        )
        self.declare_parameter('collision_topic', '/vehicle/collision')
        self.declare_parameter(
            'distance_to_goal_topic', '/navigation/distance_to_goal'
        )
        self.declare_parameter(
            'safety_stop_topic', '/safety/emergency_stop'
        )
        self.declare_parameter(
            'safety_trajectory_rejection_topic',
            '/safety/trajectory_rejection',
        )
        self.declare_parameter('state_topic', '/avoidance/state')
        self.declare_parameter(
            'selected_side_topic', '/avoidance/selected_side'
        )
        self.declare_parameter('path_topic', '/avoidance/path')
        self.declare_parameter(
            'center_points_topic', '/avoidance/center_points'
        )
        self.declare_parameter('left_points_topic', '/avoidance/left_points')
        self.declare_parameter('right_points_topic', '/avoidance/right_points')
        self.declare_parameter(
            'actual_lateral_topic', '/avoidance/actual_lateral_offset'
        )
        self.declare_parameter('route_topic', '/navigation/waypoint_route')
        self.declare_parameter(
            'route_status_topic', '/navigation/route_status'
        )
        self.declare_parameter('route_index_topic', '/navigation/route_index')
        self.declare_parameter('route_size_topic', '/navigation/route_size')
        self.declare_parameter(
            'smoothed_path_topic', '/navigation/smoothed_path'
        )
        self.declare_parameter(
            'current_local_topic', '/navigation/current_local'
        )
        self.declare_parameter('avoidance_enabled', True)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('lidar_timeout_s', 1.0)
        self.declare_parameter('odometry_timeout_s', 0.6)
        self.declare_parameter('command_timeout_s', 0.6)
        self.declare_parameter('goal_distance_timeout_s', 1.0)
        self.declare_parameter('minimum_x_m', 2.5)
        self.declare_parameter('detection_distance_m', 12.0)
        self.declare_parameter('center_half_width_m', 1.35)
        self.declare_parameter('vehicle_width_m', 2.0)
        self.declare_parameter('lateral_clearance_m', 0.65)
        self.declare_parameter('vehicle_front_m', 2.4)
        self.declare_parameter('vehicle_rear_m', 2.5)
        self.declare_parameter('minimum_z_m', -1.4)
        self.declare_parameter('maximum_z_m', 1.0)
        self.declare_parameter('minimum_obstacle_points', 20)
        self.declare_parameter('preferred_side', 'left')
        self.declare_parameter('avoidance_speed_mps', 1.0)
        self.declare_parameter('avoidance_entry_speed_mps', 1.2)
        self.declare_parameter('deceleration_required_cycles', 3)
        self.declare_parameter('lateral_offset_m', 3.5)
        self.declare_parameter('entry_forward_m', 7.0)
        self.declare_parameter('return_forward_m', 7.0)
        self.declare_parameter('return_lateral_tolerance_m', 0.3)
        self.declare_parameter('passage_rear_clearance_m', 0.0)
        self.declare_parameter('passage_minimum_lateral_m', 0.6)
        self.declare_parameter('passage_maximum_lateral_m', 7.0)
        self.declare_parameter('passage_clear_required_scans', 2)
        self.declare_parameter('target_initial_half_length_m', 0.75)
        self.declare_parameter('target_association_gap_m', 1.5)
        self.declare_parameter('target_maximum_forward_extent_m', 6.0)
        self.declare_parameter('path_sample_spacing_m', 0.75)
        self.declare_parameter('path_lookahead_m', 4.0)
        self.declare_parameter('waypoint_reached_radius_m', 0.8)
        self.declare_parameter('goal_stop_distance_m', 3.0)
        self.declare_parameter('goal_obstacle_margin_m', 0.5)
        self.declare_parameter('goal_approach_speed_limit_distance_m', 6.0)
        self.declare_parameter('lateral_reached_tolerance_m', 0.2)
        self.declare_parameter('lateral_reached_required_cycles', 3)
        self.declare_parameter('heading_kp', 1.0)
        self.declare_parameter('maximum_yaw_rate_rps', 0.35)
        self.declare_parameter('minimum_heading_speed_ratio', 0.25)
        self.declare_parameter('wheelbase_m', 2.85)
        self.declare_parameter('maximum_steering_angle_deg', 35.0)
        self.declare_parameter('trajectory_planner_enabled', True)
        self.declare_parameter('trajectory_horizon_m', 10.0)
        self.declare_parameter('trajectory_minimum_horizon_m', 8.0)
        self.declare_parameter('trajectory_sample_spacing_m', 0.35)
        self.declare_parameter(
            'navigation_trajectory_sample_spacing_m', 0.25
        )
        self.declare_parameter(
            'navigation_trajectory_curvature_uncertainty_per_m', 0.04
        )
        self.declare_parameter(
            'navigation_trajectory_extra_width_m', 0.25
        )
        self.declare_parameter('trajectory_candidate_count', 13)
        self.declare_parameter('trajectory_maximum_curvature_per_m', 0.18)
        self.declare_parameter('trajectory_proximity_margin_m', 0.8)
        self.declare_parameter('trajectory_voxel_size_m', 0.2)
        self.declare_parameter('trajectory_minimum_obstacle_cells', 1)
        self.declare_parameter('safety_validation_clear_distance_m', 8.0)
        self.declare_parameter(
            'safety_validation_sample_spacing_m', 0.25
        )
        self.declare_parameter(
            'safety_validation_minimum_obstacle_points', 20
        )
        self.declare_parameter('safety_rejection_hold_s', 1.5)
        self.declare_parameter(
            'safety_rejection_curvature_tolerance_per_m', 0.012
        )
        self.declare_parameter('trajectory_clear_required_scans', 5)
        self.declare_parameter(
            'trajectory_side_lock_clear_required_scans', 5
        )
        self.declare_parameter(
            'new_obstacle_clear_required_scans', 5
        )
        self.declare_parameter(
            'new_obstacle_recommit_max_lateral_m', 1.5
        )
        self.declare_parameter(
            'trajectory_side_corridor_advantage_points', 20
        )
        self.declare_parameter(
            'trajectory_side_lock_minimum_lateral_m', 3.0
        )
        self.declare_parameter(
            'trajectory_maximum_lateral_excursion_m', 5.0
        )
        self.declare_parameter('trajectory_commit_distance_m', 10.5)
        self.declare_parameter('recovery_lateral_tolerance_m', 0.75)
        self.declare_parameter('recovery_heading_tolerance_deg', 8.0)
        self.declare_parameter('recovery_required_scans', 5)
        self.declare_parameter('recovery_lookahead_m', 8.0)
        self.declare_parameter('recovery_path_search_ahead', 80)
        self.declare_parameter('recovery_path_search_distance_m', 12.0)
        self.declare_parameter('recovery_speed_mps', 1.5)
        self.declare_parameter('handoff_acceleration_mps2', 0.75)
        self.declare_parameter('handoff_minimum_duration_s', 1.0)
        self.declare_parameter('trajectory_goal_weight', 3.0)
        self.declare_parameter('trajectory_switch_weight', 1.0)
        self.declare_parameter('trajectory_curvature_weight', 0.15)
        self.declare_parameter('trajectory_proximity_weight', 1.0)
        self.declare_parameter('trajectory_opposite_goal_weight', 1.5)
        self.declare_parameter('maximum_avoidance_duration_s', 45.0)
        self.declare_parameter('terminal_log_period_s', 2.0)
        self.declare_parameter('log_avoidance', True)
        self.declare_parameter(
            'log_directory', '~/terrain_nav_data/logs/avoidance'
        )

        self.enabled = bool(
            self.get_parameter('avoidance_enabled').value
        )
        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )
        self.lidar_timeout_s = float(
            self.get_parameter('lidar_timeout_s').value
        )
        self.odometry_timeout_s = float(
            self.get_parameter('odometry_timeout_s').value
        )
        self.command_timeout_s = float(
            self.get_parameter('command_timeout_s').value
        )
        self.goal_distance_timeout_s = float(
            self.get_parameter('goal_distance_timeout_s').value
        )
        self.minimum_x_m = float(
            self.get_parameter('minimum_x_m').value
        )
        self.detection_distance_m = float(
            self.get_parameter('detection_distance_m').value
        )
        self.center_half_width_m = float(
            self.get_parameter('center_half_width_m').value
        )
        self.vehicle_width_m = float(
            self.get_parameter('vehicle_width_m').value
        )
        self.lateral_clearance_m = float(
            self.get_parameter('lateral_clearance_m').value
        )
        self.vehicle_front_m = float(
            self.get_parameter('vehicle_front_m').value
        )
        self.vehicle_rear_m = float(
            self.get_parameter('vehicle_rear_m').value
        )
        self.detour_corridor_half_width_m = (
            0.5 * self.vehicle_width_m + self.lateral_clearance_m
        )
        self.minimum_z_m = float(
            self.get_parameter('minimum_z_m').value
        )
        self.maximum_z_m = float(
            self.get_parameter('maximum_z_m').value
        )
        self.minimum_obstacle_points = int(
            self.get_parameter('minimum_obstacle_points').value
        )
        preferred = str(self.get_parameter('preferred_side').value).lower()
        if preferred not in ('left', 'right'):
            raise ValueError('preferred_side must be left or right')
        self.preferred_side = LEFT if preferred == 'left' else RIGHT
        self.avoidance_speed_mps = float(
            self.get_parameter('avoidance_speed_mps').value
        )
        self.avoidance_entry_speed_mps = float(
            self.get_parameter('avoidance_entry_speed_mps').value
        )
        self.deceleration_required_cycles = int(
            self.get_parameter('deceleration_required_cycles').value
        )
        self.lateral_offset_m = float(
            self.get_parameter('lateral_offset_m').value
        )
        self.entry_forward_m = float(
            self.get_parameter('entry_forward_m').value
        )
        self.return_forward_m = float(
            self.get_parameter('return_forward_m').value
        )
        self.return_lateral_tolerance_m = float(
            self.get_parameter('return_lateral_tolerance_m').value
        )
        self.passage_rear_clearance_m = float(
            self.get_parameter('passage_rear_clearance_m').value
        )
        self.passage_minimum_lateral_m = float(
            self.get_parameter('passage_minimum_lateral_m').value
        )
        self.passage_maximum_lateral_m = float(
            self.get_parameter('passage_maximum_lateral_m').value
        )
        self.passage_clear_required_scans = int(
            self.get_parameter('passage_clear_required_scans').value
        )
        self.target_initial_half_length_m = float(
            self.get_parameter('target_initial_half_length_m').value
        )
        self.target_association_gap_m = float(
            self.get_parameter('target_association_gap_m').value
        )
        self.target_maximum_forward_extent_m = float(
            self.get_parameter('target_maximum_forward_extent_m').value
        )
        self.path_sample_spacing_m = float(
            self.get_parameter('path_sample_spacing_m').value
        )
        self.path_lookahead_m = float(
            self.get_parameter('path_lookahead_m').value
        )
        self.waypoint_radius_m = float(
            self.get_parameter('waypoint_reached_radius_m').value
        )
        self.goal_stop_distance_m = float(
            self.get_parameter('goal_stop_distance_m').value
        )
        self.goal_obstacle_margin_m = float(
            self.get_parameter('goal_obstacle_margin_m').value
        )
        self.goal_approach_speed_limit_distance_m = float(
            self.get_parameter(
                'goal_approach_speed_limit_distance_m'
            ).value
        )
        self.lateral_reached_tolerance_m = float(
            self.get_parameter('lateral_reached_tolerance_m').value
        )
        self.lateral_reached_required_cycles = int(
            self.get_parameter('lateral_reached_required_cycles').value
        )
        self.heading_kp = float(self.get_parameter('heading_kp').value)
        self.maximum_yaw_rate_rps = float(
            self.get_parameter('maximum_yaw_rate_rps').value
        )
        self.minimum_heading_speed_ratio = float(
            self.get_parameter('minimum_heading_speed_ratio').value
        )
        self.wheelbase_m = float(self.get_parameter('wheelbase_m').value)
        self.maximum_steering_angle_rad = math.radians(float(
            self.get_parameter('maximum_steering_angle_deg').value
        ))
        if self.wheelbase_m <= 0.0:
            raise ValueError('wheelbase must be positive')
        if not 0.0 < self.maximum_steering_angle_rad < 0.5 * math.pi:
            raise ValueError('maximum steering angle must be in (0, 90) deg')
        self.maximum_curvature_per_m = (
            math.tan(self.maximum_steering_angle_rad) / self.wheelbase_m
        )
        self.trajectory_planner_enabled = bool(
            self.get_parameter('trajectory_planner_enabled').value
        )
        self.trajectory_horizon_m = float(
            self.get_parameter('trajectory_horizon_m').value
        )
        self.trajectory_minimum_horizon_m = float(
            self.get_parameter('trajectory_minimum_horizon_m').value
        )
        self.trajectory_sample_spacing_m = float(
            self.get_parameter('trajectory_sample_spacing_m').value
        )
        self.navigation_trajectory_sample_spacing_m = float(
            self.get_parameter(
                'navigation_trajectory_sample_spacing_m'
            ).value
        )
        self.navigation_trajectory_curvature_uncertainty_per_m = float(
            self.get_parameter(
                'navigation_trajectory_curvature_uncertainty_per_m'
            ).value
        )
        self.navigation_trajectory_extra_width_m = float(
            self.get_parameter(
                'navigation_trajectory_extra_width_m'
            ).value
        )
        self.trajectory_candidate_count = int(
            self.get_parameter('trajectory_candidate_count').value
        )
        self.trajectory_maximum_curvature_per_m = float(
            self.get_parameter(
                'trajectory_maximum_curvature_per_m'
            ).value
        )
        self.trajectory_proximity_margin_m = float(
            self.get_parameter('trajectory_proximity_margin_m').value
        )
        self.trajectory_voxel_size_m = float(
            self.get_parameter('trajectory_voxel_size_m').value
        )
        self.trajectory_minimum_obstacle_cells = int(
            self.get_parameter(
                'trajectory_minimum_obstacle_cells'
            ).value
        )
        self.safety_validation_clear_distance_m = float(
            self.get_parameter('safety_validation_clear_distance_m').value
        )
        self.safety_validation_sample_spacing_m = float(
            self.get_parameter('safety_validation_sample_spacing_m').value
        )
        self.safety_validation_minimum_obstacle_points = int(
            self.get_parameter(
                'safety_validation_minimum_obstacle_points'
            ).value
        )
        self.safety_rejection_hold_s = float(
            self.get_parameter('safety_rejection_hold_s').value
        )
        self.safety_rejection_curvature_tolerance_per_m = float(
            self.get_parameter(
                'safety_rejection_curvature_tolerance_per_m'
            ).value
        )
        self.trajectory_clear_required_scans = int(
            self.get_parameter('trajectory_clear_required_scans').value
        )
        self.trajectory_side_lock_clear_required_scans = int(
            self.get_parameter(
                'trajectory_side_lock_clear_required_scans'
            ).value
        )
        self.new_obstacle_clear_required_scans = int(
            self.get_parameter(
                'new_obstacle_clear_required_scans'
            ).value
        )
        self.new_obstacle_recommit_max_lateral_m = float(
            self.get_parameter(
                'new_obstacle_recommit_max_lateral_m'
            ).value
        )
        self.trajectory_side_corridor_advantage_points = int(
            self.get_parameter(
                'trajectory_side_corridor_advantage_points'
            ).value
        )
        self.trajectory_side_lock_minimum_lateral_m = float(
            self.get_parameter(
                'trajectory_side_lock_minimum_lateral_m'
            ).value
        )
        self.trajectory_maximum_lateral_excursion_m = float(
            self.get_parameter(
                'trajectory_maximum_lateral_excursion_m'
            ).value
        )
        self.trajectory_commit_distance_m = float(
            self.get_parameter('trajectory_commit_distance_m').value
        )
        self.recovery_lateral_tolerance_m = float(
            self.get_parameter('recovery_lateral_tolerance_m').value
        )
        self.recovery_heading_tolerance_rad = math.radians(float(
            self.get_parameter('recovery_heading_tolerance_deg').value
        ))
        self.recovery_required_scans = int(
            self.get_parameter('recovery_required_scans').value
        )
        self.recovery_lookahead_m = float(
            self.get_parameter('recovery_lookahead_m').value
        )
        self.recovery_path_search_ahead = int(
            self.get_parameter('recovery_path_search_ahead').value
        )
        self.recovery_path_search_distance_m = float(
            self.get_parameter('recovery_path_search_distance_m').value
        )
        self.recovery_speed_mps = float(
            self.get_parameter('recovery_speed_mps').value
        )
        self.handoff_acceleration_mps2 = float(
            self.get_parameter('handoff_acceleration_mps2').value
        )
        self.handoff_minimum_duration_s = float(
            self.get_parameter('handoff_minimum_duration_s').value
        )
        self.trajectory_goal_weight = float(
            self.get_parameter('trajectory_goal_weight').value
        )
        self.trajectory_switch_weight = float(
            self.get_parameter('trajectory_switch_weight').value
        )
        self.trajectory_curvature_weight = float(
            self.get_parameter('trajectory_curvature_weight').value
        )
        self.trajectory_proximity_weight = float(
            self.get_parameter('trajectory_proximity_weight').value
        )
        self.trajectory_opposite_goal_weight = float(
            self.get_parameter('trajectory_opposite_goal_weight').value
        )
        self.maximum_avoidance_duration_s = float(
            self.get_parameter('maximum_avoidance_duration_s').value
        )
        self.terminal_log_period_s = float(
            self.get_parameter('terminal_log_period_s').value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if min(
            self.lidar_timeout_s,
            self.odometry_timeout_s,
            self.command_timeout_s,
            self.goal_distance_timeout_s,
        ) <= 0.0:
            raise ValueError('input timeouts must be positive')
        if self.minimum_obstacle_points < 1:
            raise ValueError('minimum_obstacle_points must be positive')
        if not 0.0 < self.avoidance_speed_mps <= self.avoidance_entry_speed_mps:
            raise ValueError(
                'avoidance speed must be positive and no greater than the '
                'entry-speed threshold'
            )
        if self.deceleration_required_cycles < 1:
            raise ValueError('deceleration required cycles must be positive')
        if self.detection_distance_m <= self.minimum_x_m + 0.2:
            raise ValueError('detection distance is too close to minimum x')
        if self.waypoint_radius_m <= 0.0:
            raise ValueError('waypoint radius must be positive')
        if self.path_sample_spacing_m <= 0.0:
            raise ValueError('path sample spacing must be positive')
        if self.path_lookahead_m <= self.path_sample_spacing_m:
            raise ValueError('path lookahead must exceed sample spacing')
        if self.goal_stop_distance_m < 0.0:
            raise ValueError('goal stop distance must be non-negative')
        if self.goal_obstacle_margin_m < 0.0:
            raise ValueError('goal obstacle margin must be non-negative')
        if (
            self.goal_approach_speed_limit_distance_m
            < self.goal_stop_distance_m
        ):
            raise ValueError(
                'goal approach speed-limit distance must be at least the '
                'goal stop distance'
            )
        if min(
            self.vehicle_width_m,
            self.vehicle_front_m,
            self.vehicle_rear_m,
        ) <= 0.0:
            raise ValueError('vehicle dimensions must be positive')
        if self.lateral_clearance_m < 0.0:
            raise ValueError('lateral clearance must be non-negative')
        if not 0.0 <= self.lateral_reached_tolerance_m < self.lateral_offset_m:
            raise ValueError('lateral reached tolerance is invalid')
        if not 0.0 < self.return_lateral_tolerance_m < self.lateral_offset_m:
            raise ValueError('return lateral tolerance is invalid')
        if self.lateral_reached_required_cycles < 1:
            raise ValueError('lateral reached cycles must be positive')
        if self.passage_rear_clearance_m < 0.0:
            raise ValueError('passage rear clearance must be non-negative')
        if (
            self.passage_minimum_lateral_m < 0.0
            or self.passage_maximum_lateral_m
            <= self.passage_minimum_lateral_m
        ):
            raise ValueError('passage lateral limits are invalid')
        if self.passage_clear_required_scans < 1:
            raise ValueError('passage clear scans must be positive')
        if min(
            self.target_initial_half_length_m,
            self.target_association_gap_m,
            self.target_maximum_forward_extent_m,
        ) <= 0.0:
            raise ValueError('target association distances must be positive')
        if self.maximum_avoidance_duration_s <= 0.0:
            raise ValueError('maximum avoidance duration must be positive')
        if (
            self.trajectory_horizon_m
            <= self.minimum_x_m + self.trajectory_sample_spacing_m
        ):
            raise ValueError('trajectory horizon is too short')
        if not (
            self.minimum_x_m + 2.0 * self.trajectory_sample_spacing_m
            <= self.trajectory_minimum_horizon_m
            <= self.trajectory_horizon_m
        ):
            raise ValueError(
                'minimum trajectory horizon must cover the initial vehicle '
                'corridor and cannot exceed the configured horizon'
            )
        if self.trajectory_sample_spacing_m <= 0.0:
            raise ValueError('trajectory sample spacing must be positive')
        if self.navigation_trajectory_sample_spacing_m <= 0.0:
            raise ValueError(
                'navigation trajectory sample spacing must be positive'
            )
        if (
            self.navigation_trajectory_curvature_uncertainty_per_m < 0.0
            or self.navigation_trajectory_extra_width_m < 0.0
        ):
            raise ValueError(
                'navigation trajectory uncertainty must be non-negative'
            )
        if (
            self.trajectory_candidate_count < 3
            or self.trajectory_candidate_count % 2 == 0
        ):
            raise ValueError(
                'trajectory candidate count must be odd and at least 3'
            )
        if not (
            0.0 < self.trajectory_maximum_curvature_per_m
            <= self.maximum_curvature_per_m
        ):
            raise ValueError(
                'trajectory curvature must respect the Ackermann limit'
            )
        if self.trajectory_proximity_margin_m < 0.0:
            raise ValueError('trajectory proximity margin must be non-negative')
        if self.trajectory_voxel_size_m <= 0.0:
            raise ValueError('trajectory voxel size must be positive')
        if self.trajectory_minimum_obstacle_cells < 1:
            raise ValueError('trajectory obstacle cells must be positive')
        if (
            self.safety_validation_clear_distance_m
            <= self.minimum_x_m
            + self.safety_validation_sample_spacing_m
        ):
            raise ValueError('safety validation distance is too short')
        if (
            self.safety_validation_sample_spacing_m <= 0.0
            or self.safety_validation_minimum_obstacle_points < 1
            or self.safety_rejection_hold_s <= 0.0
            or self.safety_rejection_curvature_tolerance_per_m < 0.0
        ):
            raise ValueError('safety validation feedback parameters invalid')
        if self.trajectory_clear_required_scans < 1:
            raise ValueError('trajectory clear scans must be positive')
        if self.trajectory_side_lock_clear_required_scans < 1:
            raise ValueError('trajectory side-lock scans must be positive')
        if self.new_obstacle_clear_required_scans < 1:
            raise ValueError('new-obstacle clear scans must be positive')
        if self.new_obstacle_recommit_max_lateral_m <= 0.0:
            raise ValueError('new-obstacle recommit lateral limit invalid')
        if self.trajectory_side_corridor_advantage_points < 1:
            raise ValueError('side corridor advantage must be positive')
        if not (
            0.0 < self.trajectory_side_lock_minimum_lateral_m
            < self.trajectory_maximum_lateral_excursion_m
        ):
            raise ValueError('trajectory lateral limits are invalid')
        if not (
            self.minimum_x_m < self.trajectory_commit_distance_m
            <= self.detection_distance_m
        ):
            raise ValueError('trajectory commit distance is invalid')
        if self.recovery_lateral_tolerance_m <= 0.0:
            raise ValueError('recovery lateral tolerance must be positive')
        if not 0.0 < self.recovery_heading_tolerance_rad < math.pi:
            raise ValueError('recovery heading tolerance is invalid')
        if self.recovery_required_scans < 1:
            raise ValueError('recovery required scans must be positive')
        if self.recovery_lookahead_m <= 0.0:
            raise ValueError('recovery lookahead must be positive')
        if self.recovery_path_search_ahead < 1:
            raise ValueError('recovery path search ahead must be positive')
        if self.recovery_path_search_distance_m <= 0.0:
            raise ValueError('recovery path search distance must be positive')
        if self.recovery_speed_mps <= 0.0:
            raise ValueError('recovery speed must be positive')
        if self.handoff_acceleration_mps2 <= 0.0:
            raise ValueError('handoff acceleration must be positive')
        if self.handoff_minimum_duration_s < 0.0:
            raise ValueError('handoff duration must be non-negative')
        if min(
            self.trajectory_goal_weight,
            self.trajectory_switch_weight,
            self.trajectory_curvature_weight,
            self.trajectory_proximity_weight,
            self.trajectory_opposite_goal_weight,
        ) < 0.0:
            raise ValueError('trajectory weights must be non-negative')

        self.mode = 'following_goal'
        self.latest_command = Twist()
        self.current_pose = None
        self.current_z_m = 0.0
        self.current_speed_mps = None
        self.cloud_wall_time = None
        self.odometry_wall_time = None
        self.command_wall_time = None
        self.goal_distance_wall_time = None
        self.goal_distance_m = None
        self.cloud_valid = False
        self.cloud_error = None
        self.cloud_stamp_ns = 0
        self.cloud_sequence = 0
        self.center_observation = ForwardObstacleObservation(0, 0, math.inf)
        self.navigation_path_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.navigation_nominal_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.navigation_commanded_curvature_per_m = 0.0
        self.previous_navigation_curvature_per_m = 0.0
        self.navigation_tested_curvature_count = 1
        self.left_observation = ForwardObstacleObservation(0, 0, math.inf)
        self.right_observation = ForwardObstacleObservation(0, 0, math.inf)
        self.selected_side = None
        self.detour_waypoints = []
        self.avoidance_path_visible = False
        self.detour_origin = None
        self.avoidance_phase = None
        self.return_start_progress_m = None
        self.return_start_lateral_m = None
        self.path_progress_m = None
        self.actual_lateral_m = None
        self.shift_completed = False
        self.lateral_reached_count = 0
        self.passage_observation = PassageObservation(0, 0)
        self.return_corridor_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.passage_obstacle_seen = False
        self.passage_rear_seen = False
        self.passage_clear_count = 0
        self.return_ready = False
        self.tracked_obstacle_min_progress_m = None
        self.tracked_obstacle_max_progress_m = None
        self.tracked_obstacle_passed = False
        self.tracked_obstacle_extent_frozen = False
        self.tracked_obstacle_seed_progress_m = None
        self.deceleration_ready_count = 0
        self.waypoint_index = 0
        self.avoidance_start_wall_time = None
        self.avoidance_active_elapsed_s = 0.0
        self.avoidance_clock_wall_time = None
        self.trajectory_plan = None
        self.previous_trajectory_curvature = None
        self.committed_trajectory_side = None
        self.trajectory_side_lock_active = False
        self.trajectory_side_lock_clear_count = 0
        self.signed_lateral_shift_m = None
        self.lateral_excursion_limited = False
        self.navigation_clear_scan_count = 0
        self.avoidance_episode_count = 0
        self.trajectory_clear_count = 0
        self.trajectory_committed = False
        self.trajectory_desired_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.recovery_line_origin = None
        self.recovery_line_yaw_rad = None
        self.recovery_alignment = None
        self.recovery_aligned = False
        self.handoff_start_wall_time = None
        self.goal_heading_rad = None
        self.collision_event_count = 0
        self.last_collision_intensity = None
        self.last_collision_wall_time = None
        self.safety_stop_active = False
        self.rejected_trajectories = []
        self.last_rejected_curvature_per_m = None
        self.safety_rejection_count = 0
        self.last_state = None
        self.last_terminal_log_time = None
        self.goal_guard_active = False
        self.latest_points = np.empty((0, 3), dtype=np.float32)
        self.current_local_map = None
        self.map_to_odom_offset = None
        self.smoothed_path_source = []
        self.smoothed_path_source_frame = ''
        self.smoothed_path_odom = []
        self.smoothed_path_progress_segment = 0
        self.recovery_path_segment = 0
        self.recovery_uses_smoothed_path = False
        self.route_reset_count = 0
        self.route_status = 'idle'
        self.route_index = 0
        self.route_size = 0
        self.route_loop = False

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.command_publisher = self.create_publisher(
            Twist,
            str(self.get_parameter('output_command_topic').value),
            10,
        )
        self.state_publisher = self.create_publisher(
            String,
            str(self.get_parameter('state_topic').value),
            10,
        )
        self.side_publisher = self.create_publisher(
            String,
            str(self.get_parameter('selected_side_topic').value),
            10,
        )
        self.path_publisher = self.create_publisher(
            Path,
            str(self.get_parameter('path_topic').value),
            1,
        )
        self.center_points_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('center_points_topic').value),
            10,
        )
        self.left_points_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('left_points_topic').value),
            10,
        )
        self.right_points_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('right_points_topic').value),
            10,
        )
        self.actual_lateral_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter('actual_lateral_topic').value),
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('input_cloud_topic').value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self._on_odometry,
            10,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('goal_vector_topic').value),
            self._on_goal_vector,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter('collision_topic').value),
            self._on_collision,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter('distance_to_goal_topic').value),
            self._on_goal_distance,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('safety_stop_topic').value),
            self._on_safety_stop,
            10,
        )
        self.create_subscription(
            String,
            str(
                self.get_parameter(
                    'safety_trajectory_rejection_topic'
                ).value
            ),
            self._on_trajectory_rejection,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter('input_command_topic').value),
            self._on_command,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_topic').value),
            self._on_waypoint_route,
            latched_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_status_topic').value),
            self._on_route_status,
            latched_qos,
        )
        self.create_subscription(
            UInt32,
            str(self.get_parameter('route_index_topic').value),
            self._on_route_index,
            latched_qos,
        )
        self.create_subscription(
            UInt32,
            str(self.get_parameter('route_size_topic').value),
            self._on_route_size,
            latched_qos,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter('smoothed_path_topic').value),
            self._on_smoothed_path,
            latched_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('current_local_topic').value),
            self._on_current_local,
            10,
        )
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._publish_command,
        )

        self.avoidance_logger = None
        if bool(self.get_parameter('log_avoidance').value):
            self.avoidance_logger = AvoidanceCsvLogger(
                str(self.get_parameter('log_directory').value)
            )
            self.get_logger().info(
                'Avoidance CSV: {}'.format(self.avoidance_logger.path)
            )
        self.get_logger().info(
            'Local avoidance ready: enabled={}, detect <= {:.1f} m, '
            'offset={:.1f} m, preferred={}, planner={}'.format(
                self.enabled,
                self.detection_distance_m,
                self.lateral_offset_m,
                preferred,
                'trajectory'
                if self.trajectory_planner_enabled else 'legacy_detour',
            )
        )

    def _on_command(self, message):
        self.latest_command = message
        self.command_wall_time = time.monotonic()
        # Keep obstacle activation geometry current when the route controller
        # changes steering between LiDAR scans. Candidate selection remains
        # tied to new clouds, avoiding duplicate side-lock/handoff counters.
        if self.cloud_valid:
            try:
                self._update_navigation_path_observation(self.latest_points)
            except Exception as error:
                self.cloud_valid = False
                self.cloud_error = str(error)

    def _on_waypoint_route(self, message):
        try:
            route = parse_waypoint_route_json(message.data)
        except ValueError:
            return
        if not route.start:
            return
        self.route_loop = bool(route.loop)
        was_active = self.mode not in ('following_goal', 'handoff')
        self._reset_detour()
        self.route_reset_count += 1
        self.smoothed_path_progress_segment = 0
        self._update_smoothed_path_progress()
        self.get_logger().info(
            'New waypoint route: cleared previous avoidance state '
            '(was_active={}, reset_count={})'.format(
                was_active,
                self.route_reset_count,
            )
        )

    def _on_route_status(self, message):
        self.route_status = str(message.data)

    def _on_route_index(self, message):
        self.route_index = int(message.data)

    def _on_route_size(self, message):
        self.route_size = int(message.data)

    def _current_goal_is_terminal(self):
        """Return whether the current GNSS goal is the final route stop."""
        if self.route_status not in (
            'navigating',
            'waiting_for_goal_manager',
        ):
            # A standalone F6 goal can arrive after an older route completed;
            # stale route index/size metadata must not classify it as an
            # intermediate waypoint.
            return True
        if self.route_size <= 0:
            # A standalone F6 goal has no waypoint-route metadata.
            return True
        if self.route_loop:
            return False
        return self.route_index >= self.route_size

    def _on_current_local(self, message):
        values = (message.point.x, message.point.y)
        if not all(math.isfinite(float(value)) for value in values):
            return
        self.current_local_map = Point2D(
            float(message.point.x),
            float(message.point.y),
        )
        if self.current_pose is not None and self.map_to_odom_offset is None:
            self.map_to_odom_offset = Point2D(
                self.current_pose.x_m - self.current_local_map.x_m,
                self.current_pose.y_m - self.current_local_map.y_m,
            )
            self._refresh_smoothed_path_odom()

    def _on_smoothed_path(self, message):
        points = []
        for pose in message.poses:
            x_value = float(pose.pose.position.x)
            y_value = float(pose.pose.position.y)
            if math.isfinite(x_value) and math.isfinite(y_value):
                points.append(Point2D(x_value, y_value))
        if len(points) < 2:
            self.get_logger().warning('Ignored empty smoothed route path')
            return
        self.smoothed_path_source = points
        self.smoothed_path_source_frame = str(message.header.frame_id)
        self.smoothed_path_progress_segment = 0
        self._refresh_smoothed_path_odom()
        self._update_smoothed_path_progress()
        self.get_logger().info(
            'Smoothed recovery path received: points={} frame={}'.format(
                len(points),
                self.smoothed_path_source_frame or '<empty>',
            )
        )

    def _refresh_smoothed_path_odom(self):
        if len(self.smoothed_path_source) < 2:
            self.smoothed_path_odom = []
            return
        if self.smoothed_path_source_frame == 'odom':
            self.smoothed_path_odom = list(self.smoothed_path_source)
            return
        if (
            self.smoothed_path_source_frame == 'map'
            and self.map_to_odom_offset is not None
        ):
            self.smoothed_path_odom = [
                Point2D(
                    point.x_m + self.map_to_odom_offset.x_m,
                    point.y_m + self.map_to_odom_offset.y_m,
                )
                for point in self.smoothed_path_source
            ]
            return
        self.smoothed_path_odom = []

    def _update_smoothed_path_progress(self):
        if self.current_pose is None or len(self.smoothed_path_odom) < 2:
            return None
        try:
            alignment = measure_polyline_alignment(
                self.current_pose,
                self.smoothed_path_odom,
                self.smoothed_path_progress_segment,
                self.recovery_path_search_ahead,
                self.recovery_path_search_distance_m,
            )
        except ValueError:
            return None
        self.smoothed_path_progress_segment = max(
            self.smoothed_path_progress_segment,
            alignment.segment_index,
        )
        return alignment

    def _on_odometry(self, message):
        position = message.pose.pose.position
        self.current_pose = Pose2D(
            float(position.x),
            float(position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        self.current_z_m = float(position.z)
        velocity = message.twist.twist.linear
        self.current_speed_mps = math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )
        self.odometry_wall_time = time.monotonic()
        if self.current_local_map is not None and self.map_to_odom_offset is None:
            self.map_to_odom_offset = Point2D(
                self.current_pose.x_m - self.current_local_map.x_m,
                self.current_pose.y_m - self.current_local_map.y_m,
            )
            self._refresh_smoothed_path_odom()
        if self.mode in ('following_goal', 'handoff'):
            self._update_smoothed_path_progress()

    def _on_goal_vector(self, message):
        east = float(message.vector.x)
        north = float(message.vector.y)
        if math.isfinite(east) and math.isfinite(north):
            if math.hypot(east, north) > 0.1:
                self.goal_heading_rad = math.atan2(north, east)

    def _on_collision(self, message):
        self.collision_event_count += 1
        self.last_collision_intensity = float(message.data)
        self.last_collision_wall_time = time.monotonic()
        self.get_logger().error(
            'CARLA collision event #{}: intensity={:.2f}'.format(
                self.collision_event_count,
                self.last_collision_intensity,
            )
        )

    def _on_goal_distance(self, message):
        distance = float(message.data)
        self.goal_distance_m = distance if math.isfinite(distance) else None
        self.goal_distance_wall_time = time.monotonic()

    def _on_safety_stop(self, message):
        self.safety_stop_active = bool(message.data)

    def _on_trajectory_rejection(self, message):
        """Temporarily exclude a curve rejected by the final safety gate."""
        try:
            payload = json.loads(message.data)
            curvature = float(payload['curvature_per_m'])
            reason = str(payload.get('reason', 'obstacle_stop'))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if not math.isfinite(curvature) or reason != 'obstacle_stop':
            return

        now = time.monotonic()
        expiry = now + self.safety_rejection_hold_s
        tolerance = self.safety_rejection_curvature_tolerance_per_m
        retained = []
        replaced = False
        for rejected, old_expiry in self.rejected_trajectories:
            if old_expiry <= now:
                continue
            if abs(rejected - curvature) <= tolerance:
                retained.append((curvature, max(expiry, old_expiry)))
                replaced = True
            else:
                retained.append((rejected, old_expiry))
        if not replaced:
            retained.append((curvature, expiry))
        self.rejected_trajectories = retained
        self.last_rejected_curvature_per_m = curvature
        self.safety_rejection_count += 1

        if (
            self.trajectory_plan is not None
            and abs(self.trajectory_plan.curvature_per_m - curvature)
            <= tolerance
        ):
            self.trajectory_plan = None
        self.get_logger().warning(
            'Safety rejected curvature {:.3f} 1/m; excluding it for {:.1f} s '
            '(nominal_stop_points={})'.format(
                curvature,
                self.safety_rejection_hold_s,
                payload.get('nominal_stop_point_count', '?'),
            )
        )

    def _active_rejected_curvatures(self, now=None):
        if now is None:
            now = time.monotonic()
        self.rejected_trajectories = [
            (curvature, expiry)
            for curvature, expiry in self.rejected_trajectories
            if expiry > now
        ]
        return tuple(
            curvature for curvature, _expiry in self.rejected_trajectories
        )

    def _observe(self, points, center_y, half_width):
        return observe_forward_corridor(
            points,
            self.minimum_x_m,
            self.detection_distance_m - 0.1,
            self.detection_distance_m,
            half_width,
            self.minimum_z_m,
            self.maximum_z_m,
            center_y_m=center_y,
        )

    def _observe_detour(self, points, side):
        return observe_detour_corridor(
            points,
            side,
            self.minimum_x_m,
            self.detection_distance_m,
            self.lateral_offset_m,
            self.entry_forward_m,
            self.detour_corridor_half_width_m,
            self.vehicle_front_m,
            self.minimum_z_m,
            self.maximum_z_m,
        )

    def _observe_return_corridor(self, points):
        if self.detour_origin is None or self.selected_side is None:
            return ForwardObstacleObservation(0, 0, math.inf)
        current_point = Point2D(
            self.current_pose.x_m,
            self.current_pose.y_m,
        )
        current_local = world_to_local(self.detour_origin, current_point)
        lateral_distance = abs(current_local.y_m)
        if lateral_distance < 0.2:
            return ForwardObstacleObservation(0, 0, math.inf)
        return observe_detour_corridor(
            points,
            -self.selected_side,
            self.minimum_x_m,
            self.detection_distance_m,
            lateral_distance,
            self.return_forward_m,
            self.detour_corridor_half_width_m,
            self.vehicle_front_m,
            self.minimum_z_m,
            self.maximum_z_m,
        )

    def _update_lidar_passage_state(self, points):
        if self.mode != 'avoiding' or self.selected_side is None:
            return
        if self.avoidance_phase == 'decelerate':
            return
        xyz = np.asarray(points)[:, :3].astype(np.float64, copy=False)
        current_point = Point2D(
            self.current_pose.x_m,
            self.current_pose.y_m,
        )
        current_local = world_to_local(self.detour_origin, current_point)
        yaw_delta = self.current_pose.yaw_rad - self.detour_origin.yaw_rad
        cosine = math.cos(yaw_delta)
        sine = math.sin(yaw_delta)
        progress = (
            current_local.x_m + cosine * xyz[:, 0] - sine * xyz[:, 1]
        )
        lateral = (
            current_local.y_m + sine * xyz[:, 0] + cosine * xyz[:, 1]
        )
        path_lateral = self.selected_side * self.lateral_offset_m
        inside_lateral = -self.selected_side * (lateral - path_lateral)
        candidate = (
            np.isfinite(xyz).all(axis=1)
            & (inside_lateral >= self.passage_minimum_lateral_m)
            & (inside_lateral <= self.passage_maximum_lateral_m)
            & (xyz[:, 2] >= self.minimum_z_m)
            & (xyz[:, 2] <= self.maximum_z_m)
        )

        if not self.tracked_obstacle_extent_frozen:
            (
                self.tracked_obstacle_min_progress_m,
                self.tracked_obstacle_max_progress_m,
                _associated,
            ) = associate_progress_interval(
                progress,
                candidate,
                self.tracked_obstacle_min_progress_m,
                self.tracked_obstacle_max_progress_m,
                self.target_association_gap_m,
            )
            (
                self.tracked_obstacle_min_progress_m,
                self.tracked_obstacle_max_progress_m,
            ) = clamp_tracked_progress_interval(
                self.tracked_obstacle_min_progress_m,
                self.tracked_obstacle_max_progress_m,
                self.tracked_obstacle_seed_progress_m,
                self.target_initial_half_length_m,
                self.target_maximum_forward_extent_m,
            )
            associated = (
                candidate
                & (
                    progress >= self.tracked_obstacle_min_progress_m
                    - self.target_association_gap_m
                )
                & (
                    progress <= self.tracked_obstacle_max_progress_m
                    + self.target_association_gap_m
                )
            )
        else:
            associated = (
                candidate
                & (
                    progress >= self.tracked_obstacle_min_progress_m
                    - self.target_association_gap_m
                )
                & (
                    progress <= self.tracked_obstacle_max_progress_m
                    + self.target_association_gap_m
                )
            )

        rear_boundary_progress = (
            current_local.x_m
            - self.vehicle_rear_m
            - self.passage_rear_clearance_m
        )
        not_passed_count = int(np.count_nonzero(
            associated & (progress >= rear_boundary_progress)
        ))
        behind_count = int(np.count_nonzero(
            associated & (progress < rear_boundary_progress)
        ))
        self.passage_observation = PassageObservation(
            not_passed_count,
            behind_count,
        )
        self.tracked_obstacle_passed = (
            self.tracked_obstacle_max_progress_m < rear_boundary_progress
        )
        if (
            behind_count >= self.minimum_obstacle_points
            or self.tracked_obstacle_passed
        ):
            self.tracked_obstacle_extent_frozen = True
        self.passage_obstacle_seen = True
        self.passage_rear_seen = (
            self.tracked_obstacle_extent_frozen
            or self.tracked_obstacle_passed
        )

        self.return_corridor_observation = self._observe_return_corridor(
            points
        )
        return_clear = (
            self.return_corridor_observation.clear_point_count
            < self.minimum_obstacle_points
        )
        if (
            self.shift_completed
            and self.passage_obstacle_seen
            and self.passage_rear_seen
            and self.tracked_obstacle_passed
            and return_clear
        ):
            self.passage_clear_count += 1
        else:
            self.passage_clear_count = 0
        self.return_ready = (
            self.passage_clear_count
            >= self.passage_clear_required_scans
        )

    def _on_cloud(self, message):
        try:
            points = point_cloud2.read_points_numpy(
                message,
                field_names=['x', 'y', 'z'],
                skip_nans=True,
            )
            self.center_observation = self._observe(
                points, 0.0, self.center_half_width_m
            )
            self.left_observation = self._observe_detour(points, LEFT)
            self.right_observation = self._observe_detour(points, RIGHT)
            self.latest_points = np.asarray(points)[:, :3].astype(
                np.float32, copy=True
            )
            self.cloud_sequence += 1
            self.cloud_stamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            self._update_navigation_path_observation(points)
            self.cloud_valid = True
            self.cloud_error = None
            if self.trajectory_planner_enabled:
                self._update_trajectory_plan(points)
            else:
                self._consider_new_detour()
                self._update_lidar_passage_state(points)
        except Exception as error:
            self.cloud_valid = False
            self.cloud_error = str(error)
        self.cloud_wall_time = time.monotonic()

    def _update_navigation_path_observation(self, points):
        result = observe_commanded_trajectory(
            points,
            float(self.latest_command.linear.x),
            float(self.latest_command.angular.z),
            self.previous_navigation_curvature_per_m,
            self.maximum_curvature_per_m,
            self.navigation_trajectory_curvature_uncertainty_per_m,
            self.minimum_x_m,
            self.detection_distance_m - 0.1,
            self.detection_distance_m,
            self.detour_corridor_half_width_m,
            self.navigation_trajectory_extra_width_m,
            self.vehicle_front_m,
            self.vehicle_rear_m,
            self.vehicle_width_m,
            self.minimum_z_m,
            self.maximum_z_m,
            self.navigation_trajectory_sample_spacing_m,
        )
        self.navigation_nominal_observation = result.nominal
        self.navigation_path_observation = result.envelope
        self.navigation_commanded_curvature_per_m = (
            result.commanded_curvature_per_m
        )
        self.navigation_tested_curvature_count = (
            result.tested_curvature_count
        )
        self.previous_navigation_curvature_per_m = (
            result.commanded_curvature_per_m
        )

    def _desired_navigation_curvature(self):
        speed = abs(float(self.latest_command.linear.x))
        if speed <= 0.05:
            return 0.0
        curvature = float(self.latest_command.angular.z) / speed
        return max(
            -self.trajectory_maximum_curvature_per_m,
            min(self.trajectory_maximum_curvature_per_m, curvature),
        )

    def _measure_recovery_alignment(self):
        if self.current_pose is None:
            return None
        if (
            self.recovery_uses_smoothed_path
            and len(self.smoothed_path_odom) >= 2
        ):
            try:
                alignment = measure_polyline_alignment(
                    self.current_pose,
                    self.smoothed_path_odom,
                    self.recovery_path_segment,
                    self.recovery_path_search_ahead,
                    self.recovery_path_search_distance_m,
                )
            except ValueError:
                return None
            self.recovery_path_segment = max(
                self.recovery_path_segment,
                alignment.segment_index,
            )
            self.smoothed_path_progress_segment = max(
                self.smoothed_path_progress_segment,
                alignment.segment_index,
            )
            return alignment
        if (
            self.recovery_line_origin is not None
            and self.recovery_line_yaw_rad is not None
        ):
            return measure_path_alignment(
                self.current_pose,
                self.recovery_line_origin,
                self.recovery_line_yaw_rad,
            )
        return None

    def _desired_recovery_curvature(self):
        """Steer to a nearby point on the active smooth route or goal line."""
        if (
            self.recovery_uses_smoothed_path
            and len(self.smoothed_path_odom) >= 2
        ):
            alignment = self._measure_recovery_alignment()
            if alignment is None:
                return self._desired_navigation_curvature()
            target = polyline_lookahead_target(
                self.smoothed_path_odom,
                alignment,
                self.recovery_lookahead_m,
            )
            command = pure_pursuit_command(
                self.current_pose,
                target,
                speed_mps=1.0,
                maximum_yaw_rate_rps=self.maximum_yaw_rate_rps,
                minimum_heading_speed_ratio=self.minimum_heading_speed_ratio,
                maximum_curvature_per_m=(
                    self.trajectory_maximum_curvature_per_m
                ),
            )
            if command.speed_mps <= 0.05:
                return self._desired_navigation_curvature()
            return max(
                -self.trajectory_maximum_curvature_per_m,
                min(
                    self.trajectory_maximum_curvature_per_m,
                    command.yaw_rate_rps / command.speed_mps,
                ),
            )
        if (
            self.current_pose is None
            or self.recovery_line_origin is None
            or self.recovery_line_yaw_rad is None
        ):
            return self._desired_navigation_curvature()
        alignment = measure_path_alignment(
            self.current_pose,
            self.recovery_line_origin,
            self.recovery_line_yaw_rad,
        )
        target_progress = alignment.progress_m + self.recovery_lookahead_m
        target = Point2D(
            self.recovery_line_origin.x_m
            + math.cos(self.recovery_line_yaw_rad) * target_progress,
            self.recovery_line_origin.y_m
            + math.sin(self.recovery_line_yaw_rad) * target_progress,
        )
        command = pure_pursuit_command(
            self.current_pose,
            target,
            speed_mps=1.0,
            maximum_yaw_rate_rps=self.maximum_yaw_rate_rps,
            minimum_heading_speed_ratio=self.minimum_heading_speed_ratio,
            maximum_curvature_per_m=(
                self.trajectory_maximum_curvature_per_m
            ),
        )
        if command.speed_mps <= 0.05:
            return self._desired_navigation_curvature()
        return max(
            -self.trajectory_maximum_curvature_per_m,
            min(
                self.trajectory_maximum_curvature_per_m,
                command.yaw_rate_rps / command.speed_mps,
            ),
        )

    def _trajectory_planning_horizon(self):
        now = time.monotonic()
        goal_distance = None
        if (
            self.goal_distance_m is not None
            and self.goal_distance_wall_time is not None
            and now - self.goal_distance_wall_time
            <= self.goal_distance_timeout_s
        ):
            goal_distance = self.goal_distance_m
        return compute_trajectory_planning_horizon(
            configured_horizon_m=self.trajectory_horizon_m,
            minimum_horizon_m=self.trajectory_minimum_horizon_m,
            goal_distance_m=goal_distance,
            goal_stop_distance_m=self.goal_stop_distance_m,
            goal_obstacle_margin_m=self.goal_obstacle_margin_m,
            terminal_goal=self._current_goal_is_terminal(),
        )

    def _publish_trajectory_path(self):
        if self.trajectory_plan is None or self.current_pose is None:
            return
        self.detour_waypoints = [
            local_to_world(self.current_pose, point.x_m, point.y_m)
            for point in self.trajectory_plan.path_points
        ]
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        for point in self.detour_waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point.x_m
            pose.pose.position.y = point.y_m
            pose.pose.position.z = self.current_z_m
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_publisher.publish(path)
        self.avoidance_path_visible = True

    def _clear_published_avoidance_path(self):
        if not self.avoidance_path_visible:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        self.path_publisher.publish(path)
        self.avoidance_path_visible = False

    def _select_validated_trajectory(
        self,
        points,
        desired_curvature,
        horizon,
        allowed_side=None,
    ):
        """Run the existing scorer, then apply the final safety contract."""
        return select_collision_free_trajectory(
            points,
            desired_curvature_per_m=desired_curvature,
            previous_curvature_per_m=self.previous_trajectory_curvature,
            maximum_curvature_per_m=(
                self.trajectory_maximum_curvature_per_m
            ),
            candidate_count=self.trajectory_candidate_count,
            horizon_m=horizon,
            sample_spacing_m=self.trajectory_sample_spacing_m,
            minimum_path_m=self.minimum_x_m,
            corridor_half_width_m=self.detour_corridor_half_width_m,
            proximity_margin_m=self.trajectory_proximity_margin_m,
            minimum_obstacle_points=(
                self.trajectory_minimum_obstacle_cells
            ),
            minimum_z_m=self.minimum_z_m,
            maximum_z_m=self.maximum_z_m,
            preferred_side=self.preferred_side,
            goal_weight=self.trajectory_goal_weight,
            switch_weight=self.trajectory_switch_weight,
            curvature_weight=self.trajectory_curvature_weight,
            proximity_weight=self.trajectory_proximity_weight,
            voxel_size_m=self.trajectory_voxel_size_m,
            vehicle_front_m=self.vehicle_front_m,
            vehicle_rear_m=self.vehicle_rear_m,
            ego_half_width_m=0.5 * self.vehicle_width_m,
            opposite_goal_weight=self.trajectory_opposite_goal_weight,
            allowed_side=allowed_side,
            validation_clear_path_m=(
                self.safety_validation_clear_distance_m
            ),
            validation_sample_spacing_m=(
                self.safety_validation_sample_spacing_m
            ),
            validation_minimum_obstacle_points=(
                self.safety_validation_minimum_obstacle_points
            ),
            excluded_curvatures=self._active_rejected_curvatures(),
            excluded_curvature_tolerance_per_m=(
                self.safety_rejection_curvature_tolerance_per_m
            ),
        )

    def _update_trajectory_plan(self, points):
        if not self.enabled or self.current_pose is None:
            return
        # Only the nominal Ackermann path may take control away from route
        # following. The wider steering-lag envelope is intentionally kept
        # for diagnostics and cautious speed limiting by the safety gate; it
        # must not manufacture a detour around geometry the vehicle is not
        # actually commanded to traverse.
        center_blocked = (
            self.navigation_nominal_observation.clear_point_count
            >= self.minimum_obstacle_points
        )
        clear_scans_before_update = self.navigation_clear_scan_count
        if center_blocked:
            self.navigation_clear_scan_count = 0
        else:
            self.navigation_clear_scan_count += 1
        # A clear handoff remains intentionally short. If an obstacle appears
        # during it, however, the new geometry must take control immediately.
        if self.mode == 'handoff' and not center_blocked:
            return
        if self.mode != 'trajectory' and not center_blocked:
            self.goal_guard_active = False
            return
        if self.mode != 'trajectory' and self.latest_command.linear.x <= 0.02:
            return

        nearest = self.navigation_nominal_observation.nearest_distance_m
        now = time.monotonic()
        recommit_alignment = (
            self._measure_recovery_alignment()
            if self.mode == 'trajectory' else None
        )
        new_obstacle_episode = trajectory_recommit_required(
            center_blocked=center_blocked,
            preceding_clear_scans=clear_scans_before_update,
            clear_required_scans=self.new_obstacle_clear_required_scans,
            handoff_active=self.mode == 'handoff',
            cross_track_m=(
                recommit_alignment.cross_track_m
                if recommit_alignment is not None else None
            ),
            maximum_cross_track_m=(
                self.new_obstacle_recommit_max_lateral_m
            ),
        ) and self.mode in ('trajectory', 'handoff')
        if new_obstacle_episode:
            self.mode = 'trajectory'
            self.avoidance_phase = 'trajectory'
            self.avoidance_start_wall_time = now
            self.avoidance_clock_wall_time = now
            self.avoidance_active_elapsed_s = 0.0
            self.trajectory_clear_count = 0
            self.trajectory_committed = (
                math.isfinite(nearest)
                and nearest <= self.trajectory_commit_distance_m
            )
            self.trajectory_plan = None
            self.previous_trajectory_curvature = None
            self.committed_trajectory_side = None
            self.trajectory_side_lock_active = False
            self.trajectory_side_lock_clear_count = 0
            self.selected_side = None
            self.rejected_trajectories = []
            self.last_rejected_curvature_per_m = None
            self.avoidance_episode_count += 1
            self.get_logger().warning(
                'New obstacle after {} clear scans: re-arming trajectory '
                'commitment at nearest={:.2f} m (episode={})'.format(
                    clear_scans_before_update,
                    nearest,
                    self.avoidance_episode_count,
                )
            )
        goal_distance_fresh = (
            self.goal_distance_m is not None
            and self.goal_distance_wall_time is not None
            and now - self.goal_distance_wall_time
            <= self.goal_distance_timeout_s
        )
        if (
            self.mode != 'trajectory'
            and center_blocked
            and self._current_goal_is_terminal()
            and goal_distance_fresh
            and math.isfinite(nearest)
            and not obstacle_requires_detour(
                nearest,
                self.goal_distance_m,
                self.goal_stop_distance_m,
                self.goal_obstacle_margin_m,
            )
        ):
            self.goal_guard_active = True
            return

        if self.mode != 'trajectory':
            self.mode = 'trajectory'
            self.avoidance_phase = 'trajectory'
            self.avoidance_start_wall_time = now
            self.avoidance_clock_wall_time = now
            self.avoidance_active_elapsed_s = 0.0
            self.trajectory_clear_count = 0
            self.avoidance_episode_count += 1
            self.trajectory_committed = (
                math.isfinite(nearest)
                and nearest <= self.trajectory_commit_distance_m
            )
            self.goal_guard_active = False
            self.recovery_uses_smoothed_path = (
                len(self.smoothed_path_odom) >= 2
            )
            if self.recovery_uses_smoothed_path:
                self.recovery_path_segment = (
                    self.smoothed_path_progress_segment
                )
                self.recovery_line_origin = None
                self.recovery_line_yaw_rad = None
                self.recovery_alignment = (
                    self._measure_recovery_alignment()
                )
            else:
                self.recovery_line_origin = Point2D(
                    self.current_pose.x_m,
                    self.current_pose.y_m,
                )
                self.recovery_line_yaw_rad = (
                    self.goal_heading_rad
                    if self.goal_heading_rad is not None
                    else self.current_pose.yaw_rad
                )
                self.recovery_alignment = measure_path_alignment(
                    self.current_pose,
                    self.recovery_line_origin,
                    self.recovery_line_yaw_rad,
                )
            self.recovery_aligned = True
            self.get_logger().warn(
                'Local trajectory planning started: nearest={:.2f} m, '
                'recovery={}'.format(
                    nearest,
                    'smoothed_path'
                    if self.recovery_uses_smoothed_path else 'goal_line',
                )
            )

        if not self.trajectory_committed:
            if not center_blocked:
                self.mode = 'handoff'
                self.handoff_start_wall_time = now
                return
            if (
                math.isfinite(nearest)
                and nearest <= self.trajectory_commit_distance_m
            ):
                self.trajectory_committed = True
                self.previous_trajectory_curvature = None
                self.committed_trajectory_side = None
                self.trajectory_side_lock_active = False
                self.trajectory_side_lock_clear_count = 0
                self.get_logger().info(
                    'Obstacle within {:.1f} m; committing avoidance direction'
                    .format(self.trajectory_commit_distance_m)
                )
            else:
                self.avoidance_phase = 'trajectory_observing'
                self.trajectory_plan = None
                return

        horizon = self._trajectory_planning_horizon()
        desired_curvature = (
            self._desired_recovery_curvature()
            if not center_blocked else self._desired_navigation_curvature()
        )
        initial_side_plan = None
        if (
            self.trajectory_committed
            and self.committed_trajectory_side is None
        ):
            left_plan = self._select_validated_trajectory(
                points,
                desired_curvature,
                horizon,
                allowed_side=LEFT,
            )
            right_plan = self._select_validated_trajectory(
                points,
                desired_curvature,
                horizon,
                allowed_side=RIGHT,
            )
            self.committed_trajectory_side = choose_feasible_trajectory_side(
                left_plan,
                right_plan,
                self.preferred_side,
                left_corridor_points=(
                    self.left_observation.clear_point_count
                ),
                right_corridor_points=(
                    self.right_observation.clear_point_count
                ),
                corridor_advantage_points=(
                    self.trajectory_side_corridor_advantage_points
                ),
            )
            if self.committed_trajectory_side is None:
                # No Ackermann candidate on either side passed the exact same
                # raw-cloud validation used by the final safety gate.
                self.trajectory_plan = None
                self.selected_side = None
                self.trajectory_side_lock_active = False
                self.trajectory_side_lock_clear_count = 0
                return
            initial_side_plan = (
                left_plan
                if self.committed_trajectory_side == LEFT else right_plan
            )
            self.trajectory_side_lock_active = True
            self.trajectory_side_lock_clear_count = 0
            self.get_logger().info(
                'LiDAR committed feasible avoidance side={}: '
                'left_valid={} right_valid={} left_proximity={} '
                'right_proximity={} left_corridor={} '
                'right_corridor={}'.format(
                    'left'
                    if self.committed_trajectory_side == LEFT else 'right',
                    left_plan.valid_candidate_count if left_plan else 0,
                    right_plan.valid_candidate_count if right_plan else 0,
                    left_plan.proximity_point_count if left_plan else -1,
                    right_plan.proximity_point_count if right_plan else -1,
                    self.left_observation.clear_point_count,
                    self.right_observation.clear_point_count,
                )
            )

        allowed_side = None
        self.signed_lateral_shift_m = None
        self.lateral_excursion_limited = False
        if self.committed_trajectory_side is not None:
            current_alignment = self._measure_recovery_alignment()
            if current_alignment is not None:
                self.signed_lateral_shift_m = (
                    self.committed_trajectory_side
                    * current_alignment.cross_track_m
                )
            obstacle_side_observation = (
                self.right_observation
                if self.committed_trajectory_side == LEFT
                else self.left_observation
            )
            (
                self.trajectory_side_lock_clear_count,
                self.trajectory_side_lock_active,
            ) = update_trajectory_side_lock(
                lock_active=self.trajectory_side_lock_active,
                clear_count=self.trajectory_side_lock_clear_count,
                obstacle_point_count=(
                    obstacle_side_observation.clear_point_count
                ),
                minimum_obstacle_points=self.minimum_obstacle_points,
                clear_required_scans=(
                    self.trajectory_side_lock_clear_required_scans
                ),
                release_permitted=(
                    current_alignment is not None
                    and trajectory_side_lock_release_permitted(
                        center_blocked=center_blocked,
                        committed_side=self.committed_trajectory_side,
                        cross_track_m=current_alignment.cross_track_m,
                        minimum_lateral_shift_m=(
                            self.trajectory_side_lock_minimum_lateral_m
                        ),
                    )
                ),
            )
            if self.trajectory_side_lock_active:
                allowed_side = self.committed_trajectory_side
            if (
                self.signed_lateral_shift_m is not None
                and self.signed_lateral_shift_m
                >= self.trajectory_maximum_lateral_excursion_m
            ):
                # Never continue steering away from the saved route beyond
                # the configured envelope. The opposite family still passes
                # through the exact swept-footprint validator; if none is
                # safe the vehicle stops instead of drifting farther away.
                allowed_side = -self.committed_trajectory_side
                self.trajectory_side_lock_active = False
                self.trajectory_side_lock_clear_count = 0
                self.lateral_excursion_limited = True
        if (
            initial_side_plan is not None
            and allowed_side == self.committed_trajectory_side
        ):
            self.trajectory_plan = initial_side_plan
        else:
            self.trajectory_plan = self._select_validated_trajectory(
                points,
                desired_curvature,
                horizon,
                allowed_side=allowed_side,
            )
        if (
            self.trajectory_plan is None
            and allowed_side in (LEFT, RIGHT)
            and self.safety_stop_active
            and self.current_speed_mps is not None
            and self.current_speed_mps <= 0.2
        ):
            # The final gate may reject a curve on a newer scan than the one
            # used for the initial commitment. Once the vehicle is confirmed
            # stopped, it is safe to test the opposite family rather than
            # deadlocking forever behind the direction lock.
            opposite_side = RIGHT if allowed_side == LEFT else LEFT
            opposite_plan = self._select_validated_trajectory(
                points,
                desired_curvature,
                horizon,
                allowed_side=opposite_side,
            )
            if opposite_plan is not None:
                self.committed_trajectory_side = opposite_side
                self.trajectory_side_lock_active = True
                self.trajectory_side_lock_clear_count = 0
                self.trajectory_plan = opposite_plan
                self.get_logger().warning(
                    'Safety-stop replanning switched avoidance side to {}'
                    .format('left' if opposite_side == LEFT else 'right')
                )
        stop_path = horizon - self.trajectory_sample_spacing_m
        self.trajectory_desired_observation = observe_swept_vehicle_footprint(
            points,
            desired_curvature,
            self.minimum_x_m,
            stop_path,
            horizon,
            self.detour_corridor_half_width_m,
            self.vehicle_front_m,
            self.vehicle_rear_m,
            self.minimum_z_m,
            self.maximum_z_m,
            self.trajectory_sample_spacing_m,
            ego_half_width_m=0.5 * self.vehicle_width_m,
        )
        direct_clear = (
            self.trajectory_desired_observation.clear_point_count
            < self.minimum_obstacle_points
        )
        self.recovery_alignment = self._measure_recovery_alignment()
        self.recovery_aligned = (
            self.recovery_alignment is not None
            and abs(self.recovery_alignment.cross_track_m)
            <= self.recovery_lateral_tolerance_m
            and abs(self.recovery_alignment.heading_error_rad)
            <= self.recovery_heading_tolerance_rad
        )
        handoff_ready = trajectory_handoff_ready(
            direct_path_clear=direct_clear,
            recovery_aligned=self.recovery_aligned,
            obstacle_side_lock_active=self.trajectory_side_lock_active,
        )
        if handoff_ready:
            self.trajectory_clear_count += 1
        else:
            self.trajectory_clear_count = 0
        if direct_clear and not self.recovery_aligned:
            self.avoidance_phase = 'trajectory_recovering'
        else:
            self.avoidance_phase = 'trajectory'
        if (
            self.trajectory_clear_count
            >= max(
                self.trajectory_clear_required_scans,
                self.recovery_required_scans,
            )
        ):
            self.get_logger().info(
                'Navigation corridor clear, obstacle passed, and path '
                'alignment restored; starting smooth control handoff'
            )
            self.mode = 'handoff'
            self.handoff_start_wall_time = now
            return

        if self.trajectory_plan is None:
            self.selected_side = None
            self.detour_waypoints = []
            self._clear_published_avoidance_path()
            return
        curvature = self.trajectory_plan.curvature_per_m
        self.previous_trajectory_curvature = curvature
        if (
            self.committed_trajectory_side is None
            and abs(curvature) > 1e-4
        ):
            self.committed_trajectory_side = (
                LEFT if curvature > 0.0 else RIGHT
            )
            self.trajectory_side_lock_active = True
            self.trajectory_side_lock_clear_count = 0
        if curvature > 1e-4:
            self.selected_side = LEFT
        elif curvature < -1e-4:
            self.selected_side = RIGHT
        self._publish_trajectory_path()

    def _trajectory_command(self, now):
        if self.avoidance_clock_wall_time is None:
            self.avoidance_clock_wall_time = now
        elapsed = max(0.0, now - self.avoidance_clock_wall_time)
        self.avoidance_clock_wall_time = now
        if not self.safety_stop_active:
            self.avoidance_active_elapsed_s += elapsed
        if self.avoidance_phase == 'trajectory_observing':
            speed = min(
                self.avoidance_speed_mps,
                max(0.0, float(self.latest_command.linear.x)),
            )
            curvature = self._desired_navigation_curvature()
            output = Twist()
            output.linear.x = speed
            output.angular.z = max(
                -self.maximum_yaw_rate_rps,
                min(self.maximum_yaw_rate_rps, speed * curvature),
            )
            return output, 'trajectory_observing', None
        if self.trajectory_plan is None:
            return Twist(), 'trajectory_blocked', None

        speed = (
            self.recovery_speed_mps
            if self.avoidance_phase == 'trajectory_recovering'
            else self.avoidance_speed_mps
        )
        if self.latest_command.linear.x >= 0.0:
            speed = min(speed, float(self.latest_command.linear.x))
        output = Twist()
        output.linear.x = max(0.0, speed)
        output.angular.z = max(
            -self.maximum_yaw_rate_rps,
            min(
                self.maximum_yaw_rate_rps,
                output.linear.x * self.trajectory_plan.curvature_per_m,
            ),
        )
        lookahead_index = min(
            len(self.trajectory_plan.path_points) - 1,
            max(
                1,
                int(self.path_lookahead_m / self.trajectory_sample_spacing_m),
            ),
        )
        local_target = self.trajectory_plan.path_points[lookahead_index]
        target = local_to_world(
            self.current_pose,
            local_target.x_m,
            local_target.y_m,
        )
        self.waypoint_index = lookahead_index
        if self.avoidance_phase == 'trajectory_recovering':
            state = 'trajectory_recovering_path'
        else:
            state = 'trajectory_avoiding_{}'.format(self._side_name())
        return output, state, target

    def _handoff_command(self, now):
        """Increase speed smoothly while preserving GNSS path curvature."""
        if self.handoff_start_wall_time is None:
            self.handoff_start_wall_time = now
        elapsed = max(0.0, now - self.handoff_start_wall_time)
        navigation_speed = max(0.0, float(self.latest_command.linear.x))
        speed_limit = (
            self.avoidance_speed_mps
            + self.handoff_acceleration_mps2 * elapsed
        )
        output = Twist()
        output.linear.x = min(navigation_speed, speed_limit)
        if navigation_speed > 0.05:
            navigation_curvature = (
                float(self.latest_command.angular.z) / navigation_speed
            )
            navigation_curvature = max(
                -self.maximum_curvature_per_m,
                min(self.maximum_curvature_per_m, navigation_curvature),
            )
            output.angular.z = max(
                -self.maximum_yaw_rate_rps,
                min(
                    self.maximum_yaw_rate_rps,
                    output.linear.x * navigation_curvature,
                ),
            )
        reached_navigation_speed = output.linear.x >= navigation_speed - 0.02
        if (
            reached_navigation_speed
            and elapsed >= self.handoff_minimum_duration_s
        ):
            completed = self._copy_navigation_command()
            self._reset_detour()
            return completed, 'following_goal', None
        return output, 'trajectory_handoff', None

    def _consider_new_detour(self):
        if not self.enabled or self.current_pose is None:
            return
        if self.mode == 'avoiding':
            return
        self.goal_guard_active = False
        center_blocked = (
            self.center_observation.clear_point_count
            >= self.minimum_obstacle_points
        )
        if not center_blocked:
            if self.mode == 'blocked_no_route':
                self.mode = 'following_goal'
            return
        if self.latest_command.linear.x <= 0.02:
            return

        nearest = self.center_observation.nearest_distance_m
        if not math.isfinite(nearest):
            return
        now = time.monotonic()
        goal_distance_fresh = (
            self.goal_distance_m is not None
            and self.goal_distance_wall_time is not None
            and now - self.goal_distance_wall_time
            <= self.goal_distance_timeout_s
        )
        if goal_distance_fresh and self._current_goal_is_terminal():
            if not obstacle_requires_detour(
                nearest,
                self.goal_distance_m,
                self.goal_stop_distance_m,
                self.goal_obstacle_margin_m,
            ):
                self.goal_guard_active = True
                return
        side = choose_avoidance_side(
            self.left_observation,
            self.right_observation,
            self.minimum_obstacle_points,
            self.preferred_side,
        )
        if side is None:
            self.mode = 'blocked_no_route'
            return
        self.selected_side = side
        self.detour_origin = self.current_pose
        self.avoidance_phase = 'decelerate'
        self.return_start_progress_m = None
        self.return_start_lateral_m = None
        self.detour_waypoints = []
        self.waypoint_index = 0
        self.avoidance_start_wall_time = time.monotonic()
        self.avoidance_clock_wall_time = self.avoidance_start_wall_time
        self.avoidance_active_elapsed_s = 0.0
        self.mode = 'avoiding'
        self.path_progress_m = 0.0
        self.actual_lateral_m = 0.0
        self.shift_completed = False
        self.lateral_reached_count = 0
        self.deceleration_ready_count = 0
        self.passage_observation = PassageObservation(0, 0)
        self.return_corridor_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.passage_obstacle_seen = True
        self.passage_rear_seen = False
        self.passage_clear_count = 0
        self.return_ready = False
        self.tracked_obstacle_min_progress_m = (
            nearest - self.target_initial_half_length_m
        )
        self.tracked_obstacle_max_progress_m = (
            nearest + self.target_initial_half_length_m
        )
        self.tracked_obstacle_seed_progress_m = nearest
        self.tracked_obstacle_passed = False
        self.tracked_obstacle_extent_frozen = False
        self._publish_detour_path()
        self.get_logger().warn(
            'Detour started: side={} nearest={:.2f} m left_points={} '
            'right_points={} path_points={}'.format(
                self._side_name(),
                nearest,
                self.left_observation.clear_point_count,
                self.right_observation.clear_point_count,
                len(self.detour_waypoints),
            )
        )

    def _desired_lateral_at_progress(self, progress_m):
        if self.avoidance_phase == 'decelerate':
            return 0.0
        if self.avoidance_phase == 'return':
            delta = max(0.0, progress_m - self.return_start_progress_m)
            ratio = delta / self.return_forward_m
            return self.return_start_lateral_m * (
                1.0 - quintic_smoothstep(ratio)
            )
        return detour_lateral_offset(
            progress_m,
            self.selected_side,
            self.lateral_offset_m,
            self.entry_forward_m,
        )

    def _publish_detour_path(self):
        if self.detour_origin is None or self.selected_side is None:
            return
        if self.avoidance_phase == 'return':
            start_progress = self.return_start_progress_m
            end_progress = start_progress + self.return_forward_m
        else:
            start_progress = 0.0
            end_progress = max(
                self.detection_distance_m,
                (self.path_progress_m or 0.0) + self.detection_distance_m,
            )
        sample_count = max(
            1,
            int(math.ceil(
                (end_progress - start_progress)
                / self.path_sample_spacing_m
            )),
        )
        self.detour_waypoints = []
        for index in range(sample_count + 1):
            progress = min(
                start_progress + index * self.path_sample_spacing_m,
                end_progress,
            )
            self.detour_waypoints.append(local_to_world(
                self.detour_origin,
                progress,
                self._desired_lateral_at_progress(progress),
            ))

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        for point in self.detour_waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point.x_m
            pose.pose.position.y = point.y_m
            pose.pose.position.z = self.current_z_m
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_publisher.publish(path)
        self.avoidance_path_visible = True

    def _reset_detour(self):
        self._clear_published_avoidance_path()
        self.mode = 'following_goal'
        self.selected_side = None
        self.detour_waypoints = []
        self.detour_origin = None
        self.avoidance_phase = None
        self.return_start_progress_m = None
        self.return_start_lateral_m = None
        self.path_progress_m = None
        self.actual_lateral_m = None
        self.shift_completed = False
        self.lateral_reached_count = 0
        self.passage_observation = PassageObservation(0, 0)
        self.return_corridor_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.passage_obstacle_seen = False
        self.passage_rear_seen = False
        self.passage_clear_count = 0
        self.return_ready = False
        self.tracked_obstacle_min_progress_m = None
        self.tracked_obstacle_max_progress_m = None
        self.tracked_obstacle_passed = False
        self.tracked_obstacle_extent_frozen = False
        self.tracked_obstacle_seed_progress_m = None
        self.deceleration_ready_count = 0
        self.waypoint_index = 0
        self.avoidance_start_wall_time = None
        self.avoidance_active_elapsed_s = 0.0
        self.avoidance_clock_wall_time = None
        self.trajectory_plan = None
        self.previous_trajectory_curvature = None
        self.committed_trajectory_side = None
        self.trajectory_side_lock_active = False
        self.trajectory_side_lock_clear_count = 0
        self.signed_lateral_shift_m = None
        self.lateral_excursion_limited = False
        self.navigation_clear_scan_count = 0
        self.avoidance_episode_count = 0
        self.trajectory_clear_count = 0
        self.trajectory_committed = False
        self.trajectory_desired_observation = ForwardObstacleObservation(
            0, 0, math.inf
        )
        self.rejected_trajectories = []
        self.last_rejected_curvature_per_m = None
        self.recovery_line_origin = None
        self.recovery_line_yaw_rad = None
        self.recovery_alignment = None
        self.recovery_aligned = False
        self.recovery_path_segment = self.smoothed_path_progress_segment
        self.recovery_uses_smoothed_path = False
        self.handoff_start_wall_time = None

    def _side_name(self):
        if self.selected_side == LEFT:
            return 'left'
        if self.selected_side == RIGHT:
            return 'right'
        return 'none'

    def _copy_navigation_command(self):
        output = Twist()
        output.linear.x = self.latest_command.linear.x
        output.linear.y = self.latest_command.linear.y
        output.linear.z = self.latest_command.linear.z
        output.angular.x = self.latest_command.angular.x
        output.angular.y = self.latest_command.angular.y
        output.angular.z = self.latest_command.angular.z
        return output

    def _input_state(self, now):
        if self.command_wall_time is None:
            return 'waiting_for_command'
        if now - self.command_wall_time > self.command_timeout_s:
            return 'command_stale'
        if not self.enabled:
            return None
        if self.cloud_wall_time is None:
            return 'waiting_for_lidar'
        if not self.cloud_valid:
            return 'invalid_lidar'
        if now - self.cloud_wall_time > self.lidar_timeout_s:
            return 'lidar_stale'
        if self.odometry_wall_time is None:
            return 'waiting_for_odometry'
        if now - self.odometry_wall_time > self.odometry_timeout_s:
            return 'odometry_stale'
        return None

    def _avoidance_command(self, now):
        if self.avoidance_clock_wall_time is None:
            self.avoidance_clock_wall_time = now
        elapsed = max(0.0, now - self.avoidance_clock_wall_time)
        self.avoidance_clock_wall_time = now
        if not self.safety_stop_active:
            self.avoidance_active_elapsed_s += elapsed
        if (
            self.avoidance_start_wall_time is None
            or self.avoidance_active_elapsed_s
            > self.maximum_avoidance_duration_s
        ):
            self.mode = 'avoidance_timeout'
            return Twist(), 'avoidance_timeout_unconfirmed', None

        current_point = Point2D(
            self.current_pose.x_m,
            self.current_pose.y_m,
        )
        current_local = world_to_local(self.detour_origin, current_point)
        self.path_progress_m = max(0.0, current_local.x_m)
        self.actual_lateral_m = self.selected_side * current_local.y_m
        if self.avoidance_phase == 'decelerate':
            if (
                self.current_speed_mps is not None
                and self.current_speed_mps <= self.avoidance_entry_speed_mps
            ):
                self.deceleration_ready_count += 1
            else:
                self.deceleration_ready_count = 0
            if (
                self.deceleration_ready_count
                >= self.deceleration_required_cycles
            ):
                old_origin = self.detour_origin
                new_origin = self.current_pose
                for attribute in (
                    'tracked_obstacle_min_progress_m',
                    'tracked_obstacle_max_progress_m',
                    'tracked_obstacle_seed_progress_m',
                ):
                    old_progress = getattr(self, attribute)
                    old_point = local_to_world(old_origin, old_progress, 0.0)
                    setattr(
                        self,
                        attribute,
                        world_to_local(new_origin, old_point).x_m,
                    )
                self.detour_origin = new_origin
                self.path_progress_m = 0.0
                self.actual_lateral_m = 0.0
                self.avoidance_phase = 'shift_out'
                self.get_logger().info(
                    'Avoidance entry speed confirmed at {:.2f} m/s; '
                    'starting lateral shift'.format(self.current_speed_mps)
                )
                self._publish_detour_path()
            else:
                output = Twist()
                output.linear.x = self.avoidance_speed_mps
                return output, 'avoiding_decelerate', None
        if not self.shift_completed:
            (
                self.lateral_reached_count,
                self.shift_completed,
            ) = update_lateral_shift_confirmation(
                self.actual_lateral_m,
                self.lateral_offset_m,
                self.lateral_reached_tolerance_m,
                self.lateral_reached_count,
                self.lateral_reached_required_cycles,
            )
            if self.shift_completed:
                self.avoidance_phase = 'pass'
                self.get_logger().info(
                    'Lateral shift confirmed at {:.2f} m'.format(
                        self.actual_lateral_m
                    )
                )

        if self.avoidance_phase == 'pass' and self.return_ready:
            self.avoidance_phase = 'return'
            self.return_start_progress_m = self.path_progress_m
            self.return_start_lateral_m = current_local.y_m
            self.get_logger().info(
                'LiDAR confirmed obstacle behind vehicle; starting return'
            )
            self._publish_detour_path()

        if self.avoidance_phase == 'return':
            return_end_progress = (
                self.return_start_progress_m + self.return_forward_m
            )
            if (
                self.path_progress_m >= return_end_progress
                and abs(current_local.y_m)
                <= self.return_lateral_tolerance_m
            ):
                self.get_logger().info(
                    'LiDAR-guided detour complete; returning to GNSS guidance'
                )
                self._reset_detour()
                return self._copy_navigation_command(), 'following_goal', None

        lookahead_progress = self.path_progress_m + self.path_lookahead_m
        target = local_to_world(
            self.detour_origin,
            lookahead_progress,
            self._desired_lateral_at_progress(lookahead_progress),
        )
        self.waypoint_index = int(
            lookahead_progress / self.path_sample_spacing_m
        )
        command = pure_pursuit_command(
            self.current_pose,
            target,
            self.avoidance_speed_mps,
            self.maximum_yaw_rate_rps,
            self.minimum_heading_speed_ratio,
            self.maximum_curvature_per_m,
        )
        output = Twist()
        output.linear.x = command.speed_mps
        output.angular.z = command.yaw_rate_rps
        goal_distance_fresh = (
            self.goal_distance_m is not None
            and self.goal_distance_wall_time is not None
            and now - self.goal_distance_wall_time
            <= self.goal_distance_timeout_s
        )
        if (
            goal_distance_fresh
            and self.goal_distance_m
            <= self.goal_approach_speed_limit_distance_m
        ):
            # Do not let an active detour override the GNSS controller's
            # arrival slowdown.  Previously the return curve commanded about
            # 1 m/s even when the goal controller had slowed to 0.07 m/s,
            # carrying the vehicle past the destination.
            output.linear.x = min(
                output.linear.x,
                max(0.0, self.latest_command.linear.x),
            )
        if self.avoidance_phase == 'decelerate':
            state = 'avoiding_decelerate'
        elif self.avoidance_phase == 'shift_out':
            state = 'avoiding_shift_out'
        elif self.avoidance_phase == 'pass':
            state = 'avoiding_pass'
        else:
            state = 'avoiding_return'
        return output, state, target

    def _publish_diagnostics(self, state):
        state_message = String()
        state_message.data = state
        self.state_publisher.publish(state_message)
        side_message = String()
        side_message.data = self._side_name()
        self.side_publisher.publish(side_message)
        for publisher, count in (
            (
                self.center_points_publisher,
                self.center_observation.clear_point_count,
            ),
            (
                self.left_points_publisher,
                self.left_observation.clear_point_count,
            ),
            (
                self.right_points_publisher,
                self.right_observation.clear_point_count,
            ),
        ):
            message = UInt32()
            message.data = count
            publisher.publish(message)
        lateral_message = Float32()
        lateral_message.data = float(
            self.actual_lateral_m
            if self.actual_lateral_m is not None else 0.0
        )
        self.actual_lateral_publisher.publish(lateral_message)

    def _write_log(self, now, state, output, target):
        if self.avoidance_logger is None:
            return

        def age(timestamp):
            return now - timestamp if timestamp is not None else math.inf

        nearest = self.center_observation.nearest_distance_m
        navigation_nearest = (
            self.navigation_path_observation.nearest_distance_m
        )
        pose = self.current_pose
        self.avoidance_logger.write({
            'wall_time_iso': datetime.now().astimezone().isoformat(
                timespec='milliseconds'
            ),
            'state': state,
            'selected_side': self._side_name(),
            'waypoint_index': self.waypoint_index,
            'center_points': self.center_observation.clear_point_count,
            'left_points': self.left_observation.clear_point_count,
            'right_points': self.right_observation.clear_point_count,
            'nearest_center_m': (
                '{:.6f}'.format(nearest) if math.isfinite(nearest) else ''
            ),
            'navigation_path_points': (
                self.navigation_path_observation.clear_point_count
            ),
            'navigation_nominal_points': (
                self.navigation_nominal_observation.clear_point_count
            ),
            'nearest_navigation_path_m': (
                '{:.6f}'.format(navigation_nearest)
                if math.isfinite(navigation_nearest) else ''
            ),
            'navigation_curvature_per_m': '{:.6f}'.format(
                self.navigation_commanded_curvature_per_m
            ),
            'navigation_tested_curvature_count': (
                self.navigation_tested_curvature_count
            ),
            'navigation_clear_scan_count': self.navigation_clear_scan_count,
            'avoidance_episode_count': self.avoidance_episode_count,
            'cloud_age_s': '{:.6f}'.format(age(self.cloud_wall_time)),
            'odometry_age_s': '{:.6f}'.format(age(self.odometry_wall_time)),
            'command_age_s': '{:.6f}'.format(age(self.command_wall_time)),
            'vehicle_speed_mps': (
                '{:.6f}'.format(self.current_speed_mps)
                if self.current_speed_mps is not None else ''
            ),
            'input_speed_mps': '{:.6f}'.format(
                self.latest_command.linear.x
            ),
            'output_speed_mps': '{:.6f}'.format(output.linear.x),
            'output_yaw_rate_rps': '{:.6f}'.format(output.angular.z),
            'vehicle_x_m': '{:.6f}'.format(pose.x_m) if pose else '',
            'vehicle_y_m': '{:.6f}'.format(pose.y_m) if pose else '',
            'vehicle_yaw_deg': (
                '{:.6f}'.format(math.degrees(pose.yaw_rad)) if pose else ''
            ),
            'target_x_m': '{:.6f}'.format(target.x_m) if target else '',
            'target_y_m': '{:.6f}'.format(target.y_m) if target else '',
            'goal_distance_m': (
                '{:.6f}'.format(self.goal_distance_m)
                if self.goal_distance_m is not None else ''
            ),
            'route_status': self.route_status,
            'route_index': self.route_index,
            'route_size': self.route_size,
            'route_loop': int(self.route_loop),
            'terminal_route_goal': int(self._current_goal_is_terminal()),
            'trajectory_horizon_m': '{:.6f}'.format(
                self._trajectory_planning_horizon()
            ),
            'path_progress_m': (
                '{:.6f}'.format(self.path_progress_m)
                if self.path_progress_m is not None else ''
            ),
            'actual_lateral_m': (
                '{:.6f}'.format(self.actual_lateral_m)
                if self.actual_lateral_m is not None else ''
            ),
            'shift_completed': int(self.shift_completed),
            'safety_stop_active': int(self.safety_stop_active),
            'avoidance_active_elapsed_s': '{:.6f}'.format(
                self.avoidance_active_elapsed_s
            ),
            'avoidance_phase': self.avoidance_phase or '',
            'passage_not_passed_points': (
                self.passage_observation.not_passed_point_count
            ),
            'passage_behind_points': (
                self.passage_observation.behind_point_count
            ),
            'passage_obstacle_seen': int(self.passage_obstacle_seen),
            'passage_rear_seen': int(self.passage_rear_seen),
            'passage_clear_count': self.passage_clear_count,
            'return_corridor_points': (
                self.return_corridor_observation.clear_point_count
            ),
            'tracked_obstacle_min_progress_m': (
                '{:.6f}'.format(self.tracked_obstacle_min_progress_m)
                if self.tracked_obstacle_min_progress_m is not None else ''
            ),
            'tracked_obstacle_max_progress_m': (
                '{:.6f}'.format(self.tracked_obstacle_max_progress_m)
                if self.tracked_obstacle_max_progress_m is not None else ''
            ),
            'tracked_obstacle_passed': int(self.tracked_obstacle_passed),
            'tracked_obstacle_extent_frozen': int(
                self.tracked_obstacle_extent_frozen
            ),
            'tracked_obstacle_seed_progress_m': (
                '{:.6f}'.format(self.tracked_obstacle_seed_progress_m)
                if self.tracked_obstacle_seed_progress_m is not None else ''
            ),
            'planner_mode': (
                'trajectory'
                if self.trajectory_planner_enabled else 'legacy_detour'
            ),
            'committed_trajectory_side': (
                'left' if self.committed_trajectory_side == LEFT
                else 'right' if self.committed_trajectory_side == RIGHT
                else 'none'
            ),
            'trajectory_side_lock_active': int(
                self.trajectory_side_lock_active
            ),
            'trajectory_side_lock_clear_count': (
                self.trajectory_side_lock_clear_count
            ),
            'signed_lateral_shift_m': (
                '{:.6f}'.format(self.signed_lateral_shift_m)
                if self.signed_lateral_shift_m is not None else ''
            ),
            'lateral_excursion_limited': int(
                self.lateral_excursion_limited
            ),
            'desired_curvature_per_m': (
                '{:.6f}'.format(
                    self.trajectory_plan.desired_curvature_per_m
                ) if self.trajectory_plan is not None else ''
            ),
            'selected_curvature_per_m': (
                '{:.6f}'.format(self.trajectory_plan.curvature_per_m)
                if self.trajectory_plan is not None else ''
            ),
            'valid_trajectory_count': (
                self.trajectory_plan.valid_candidate_count
                if self.trajectory_plan is not None else 0
            ),
            'trajectory_candidate_count': (
                self.trajectory_plan.candidate_count
                if self.trajectory_plan is not None
                else self.trajectory_candidate_count
            ),
            'trajectory_collision_points': (
                self.trajectory_plan.collision_point_count
                if self.trajectory_plan is not None else ''
            ),
            'trajectory_safety_validation_points': (
                self.trajectory_plan.safety_validation_point_count
                if self.trajectory_plan is not None else ''
            ),
            'trajectory_proximity_points': (
                self.trajectory_plan.proximity_point_count
                if self.trajectory_plan is not None else ''
            ),
            'trajectory_nearest_path_m': (
                '{:.6f}'.format(self.trajectory_plan.nearest_path_m)
                if (
                    self.trajectory_plan is not None
                    and math.isfinite(self.trajectory_plan.nearest_path_m)
                ) else ''
            ),
            'trajectory_score': (
                '{:.6f}'.format(self.trajectory_plan.score)
                if self.trajectory_plan is not None else ''
            ),
            'recovery_cross_track_m': (
                '{:.6f}'.format(self.recovery_alignment.cross_track_m)
                if self.recovery_alignment is not None else ''
            ),
            'recovery_heading_error_deg': (
                '{:.6f}'.format(math.degrees(
                    self.recovery_alignment.heading_error_rad
                )) if self.recovery_alignment is not None else ''
            ),
            'recovery_aligned': int(self.recovery_aligned),
            'recovery_reference': (
                'smoothed_path'
                if self.recovery_uses_smoothed_path else 'goal_line'
            ),
            'route_reset_count': self.route_reset_count,
            'cloud_stamp_ns': self.cloud_stamp_ns,
            'active_rejected_curvature_count': len(
                self._active_rejected_curvatures(now)
            ),
            'last_rejected_curvature_per_m': (
                '{:.6f}'.format(self.last_rejected_curvature_per_m)
                if self.last_rejected_curvature_per_m is not None else ''
            ),
            'safety_rejection_count': self.safety_rejection_count,
            'collision_event_count': self.collision_event_count,
            'last_collision_intensity': (
                '{:.6f}'.format(self.last_collision_intensity)
                if self.last_collision_intensity is not None else ''
            ),
            'last_collision_age_s': (
                '{:.6f}'.format(age(self.last_collision_wall_time))
                if self.last_collision_wall_time is not None else ''
            ),
        })

    def _terminal_log(self, now, state, output):
        changed = state != self.last_state
        elapsed = (
            self.last_terminal_log_time is None
            or self.terminal_log_period_s == 0.0
            or now - self.last_terminal_log_time
            >= self.terminal_log_period_s
        )
        if not changed and not elapsed:
            return
        nearest = self.navigation_nominal_observation.nearest_distance_m
        nearest_text = (
            '{:.2f}'.format(nearest) if math.isfinite(nearest) else 'none'
        )
        tracked_max_text = (
            '{:.2f}'.format(self.tracked_obstacle_max_progress_m)
            if self.tracked_obstacle_max_progress_m is not None else 'none'
        )
        selected_curvature_text = (
            '{:.3f}'.format(self.trajectory_plan.curvature_per_m)
            if self.trajectory_plan is not None else 'none'
        )
        valid_trajectory_count = (
            self.trajectory_plan.valid_candidate_count
            if self.trajectory_plan is not None else 0
        )
        self.get_logger().info(
            'avoidance_state={} side={} waypoint={}/{} nearest={} m '
            'points(N/C/L/R)={}/{}/{}/{} lateral={:.2f}/{:.2f} m '
            'shift={} passage(front/rear)={}/{} clear={}/{} '
            'return_corridor={} tracked_max={} frozen={} passed={} '
            'safety_stop={} active={:.1f} s '
            'output={:.2f},{:.2f} planner={} curvature={} valid={}/{} '
            'recovery(cte/heading/aligned)={:.2f}/{:.1f}/{} collisions={}'
            .format(
                state,
                self._side_name(),
                self.waypoint_index,
                len(self.detour_waypoints),
                nearest_text,
                self.navigation_path_observation.clear_point_count,
                self.center_observation.clear_point_count,
                self.left_observation.clear_point_count,
                self.right_observation.clear_point_count,
                self.actual_lateral_m or 0.0,
                self.lateral_offset_m,
                self.shift_completed,
                self.passage_observation.not_passed_point_count,
                self.passage_observation.behind_point_count,
                self.passage_clear_count,
                self.passage_clear_required_scans,
                self.return_corridor_observation.clear_point_count,
                tracked_max_text,
                self.tracked_obstacle_extent_frozen,
                self.tracked_obstacle_passed,
                self.safety_stop_active,
                self.avoidance_active_elapsed_s,
                output.linear.x,
                output.angular.z,
                'trajectory'
                if self.trajectory_planner_enabled else 'legacy_detour',
                selected_curvature_text,
                valid_trajectory_count,
                self.trajectory_candidate_count,
                (
                    self.recovery_alignment.cross_track_m
                    if self.recovery_alignment is not None else 0.0
                ),
                (
                    math.degrees(self.recovery_alignment.heading_error_rad)
                    if self.recovery_alignment is not None else 0.0
                ),
                self.recovery_aligned,
                self.collision_event_count,
            )
        )
        if state == 'invalid_lidar' and self.cloud_error:
            self.get_logger().error(
                'PointCloud2 parsing failed: {}'.format(self.cloud_error)
            )
        self.last_state = state
        self.last_terminal_log_time = now

    def _publish_command(self):
        now = time.monotonic()
        input_problem = self._input_state(now)
        target = None
        goal_distance_fresh = (
            self.goal_distance_m is not None
            and self.goal_distance_wall_time is not None
            and now - self.goal_distance_wall_time
            <= self.goal_distance_timeout_s
        )
        goal_stop_requested = (
            goal_distance_fresh
            and self.goal_distance_m <= self.goal_stop_distance_m + 0.05
            and self.latest_command.linear.x <= 0.02
        )
        if input_problem is not None:
            output = Twist()
            state = input_problem
        elif goal_stop_requested:
            # Arrival takes precedence over finishing the remaining lateral
            # return.  Stopping a little off the original centerline is safer
            # than driving through the goal and turning back toward it.
            self._reset_detour()
            output = Twist()
            state = 'navigation_stop'
        elif self.mode == 'trajectory':
            output, state, target = self._trajectory_command(now)
        elif self.mode == 'handoff':
            output, state, target = self._handoff_command(now)
        elif self.mode in ('avoiding', 'avoidance_timeout'):
            output, state, target = self._avoidance_command(now)
        elif self.latest_command.linear.x <= 0.02:
            self._reset_detour()
            output = Twist()
            state = 'navigation_stop'
        elif not self.enabled:
            output = self._copy_navigation_command()
            state = 'disabled_passthrough'
        elif self.mode == 'blocked_no_route':
            output = Twist()
            state = 'blocked_no_route'
        else:
            output = self._copy_navigation_command()
            state = (
                'following_goal_goal_guard'
                if self.goal_guard_active else 'following_goal'
            )

        self.command_publisher.publish(output)
        self._publish_diagnostics(state)
        self._write_log(now, state, output, target)
        self._terminal_log(now, state, output)

    def destroy_node(self):
        if self.avoidance_logger is not None:
            self.avoidance_logger.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LocalAvoidanceNode()
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
