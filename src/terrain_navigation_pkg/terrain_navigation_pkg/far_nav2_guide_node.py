"""FAR-inspired online guide feeding short Ackermann goals to Nav2."""

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearCostmapAroundRobot
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMessage
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import Bool, String

from .far_guide_core import build_guide_grid
from .far_guide_core import assess_path_efficiency
from .far_guide_core import assess_active_replan
from .far_guide_core import bounded_heading_preference
from .far_guide_core import can_accept_bounded_topology_escape
from .far_guide_core import can_accept_length_only_detour
from .far_guide_core import is_actionable_safety_stop
from .far_guide_core import nearest_polyline_tangent
from .far_guide_core import plan_online_guide
from .far_guide_core import polylines_similar
from .far_guide_core import prefix_polyline_to_point
from .far_guide_core import remaining_polyline_length
from .far_guide_core import retry_lookahead_distance
from .far_guide_core import SafetyReplanHold
from .far_guide_core import SafetyRejectionMonitor
from .far_guide_core import select_direction_continuity
from .far_guide_core import select_subgoal
from .far_guide_core import should_request_costmap_recovery


class FarNav2GuideNode(Node):
    """Plan a long online guide, then send only short goals to Smac Hybrid.

    This is the integration layer inspired by FAR Planner's separation of a
    long-range topological guide and a local kinematically feasible planner.
    It is intentionally independent from the existing direct F9/F10 bridge so
    both modes can be regression-tested without silently changing one another.
    """

    def __init__(self):
        super().__init__('far_nav2_guide_node')
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('goal_local_topic', '/navigation/goal_local')
        self.declare_parameter(
            'goal_tangent_topic', '/navigation/goal_tangent'
        )
        self.declare_parameter(
            'current_local_topic', '/navigation/current_local'
        )
        self.declare_parameter('odometry_topic', '/vehicle/odometry')
        self.declare_parameter(
            'goal_reached_topic', '/navigation/goal_reached'
        )
        self.declare_parameter('action_name', '/navigate_to_pose')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('behavior_tree', '')
        self.declare_parameter('planning_rate_hz', 2.0)
        self.declare_parameter('input_timeout_s', 1.5)
        self.declare_parameter('segment_lookahead_m', 12.0)
        self.declare_parameter('final_goal_distance_m', 10.0)
        self.declare_parameter('goal_tangent_max_bias_deg', 20.0)
        self.declare_parameter('goal_change_tolerance_m', 0.25)
        self.declare_parameter('goal_hold_after_success_s', 1.5)
        self.declare_parameter('retry_delay_s', 1.0)
        # After a failed Hybrid segment, retain its local guide tangent only
        # as a short-lived A* cost preference. This suppresses arbitrary
        # opposite-side P-turns without making a newly required detour illegal.
        self.declare_parameter('retry_direction_continuity_enabled', True)
        self.declare_parameter('retry_direction_hold_s', 12.0)
        self.declare_parameter('retry_direction_distance_m', 8.0)
        self.declare_parameter('retry_direction_cost_weight', 0.8)
        # A successful 12 m segment is stronger evidence than a single noisy
        # rolling-grid snapshot. Preserve that segment's departure direction
        # for the rest of the current mission waypoint so adjacent replans do
        # not arbitrarily switch to the opposite homotopy.
        self.declare_parameter('leg_direction_continuity_enabled', True)
        self.declare_parameter('leg_direction_distance_m', 12.0)
        self.declare_parameter('leg_direction_cost_weight', 2.0)
        # Controller feedback closes a gap the coarse guide cannot predict:
        # RPP may cut inside a valid guide and collide. Remember that executed
        # prefix briefly, penalize it, and hard-block it only when a duplicate
        # retry would otherwise be sent.
        self.declare_parameter('failed_corridor_enabled', True)
        self.declare_parameter('failed_corridor_hold_s', 20.0)
        self.declare_parameter('failed_corridor_radius_m', 1.5)
        self.declare_parameter('failed_corridor_cost_weight', 12.0)
        self.declare_parameter('failed_corridor_hard_radius_m', 0.75)
        self.declare_parameter('failed_corridor_similarity_m', 0.75)
        self.declare_parameter('failed_corridor_max_count', 4)
        self.declare_parameter('blocked_retry_s', 5.0)
        self.declare_parameter('active_replanning_enabled', True)
        self.declare_parameter(
            'active_replan_allow_valid_path_optimization', False
        )
        self.declare_parameter('active_replan_min_interval_s', 2.0)
        self.declare_parameter('active_replan_confirmations', 3)
        self.declare_parameter('active_replan_subgoal_change_m', 3.0)
        self.declare_parameter('active_replan_min_improvement_m', 3.0)
        self.declare_parameter('active_replan_endpoint_tolerance_m', 2.0)
        self.declare_parameter('active_replan_consistency_m', 1.5)
        self.declare_parameter(
            'path_hard_valid_topic',
            '/navigation/path_clearance/hard_valid',
        )
        self.declare_parameter('path_validity_timeout_s', 1.5)
        # Inspect Nav2's actual Smac Hybrid path.  A collision-free Dubins
        # loop is technically valid but is not a useful 12 m guide segment.
        self.declare_parameter('nav2_plan_topic', '/plan')
        self.declare_parameter('path_efficiency_enabled', True)
        self.declare_parameter('path_efficiency_max_length_ratio', 1.9)
        self.declare_parameter('path_efficiency_max_abs_turn_deg', 360.0)
        self.declare_parameter('path_efficiency_max_signed_turn_deg', 220.0)
        self.declare_parameter('path_efficiency_min_reference_m', 4.0)
        self.declare_parameter('path_efficiency_min_excess_m', 4.0)
        self.declare_parameter('path_efficiency_endpoint_tolerance_m', 2.0)
        self.declare_parameter('path_efficiency_start_tolerance_m', 3.0)
        self.declare_parameter('path_efficiency_retry_yaw_hold_s', 15.0)
        self.declare_parameter(
            'path_efficiency_allow_length_only_detour', True
        )
        self.declare_parameter(
            'path_efficiency_length_only_detour_after_retries', 1
        )
        self.declare_parameter('path_efficiency_max_retries', 6)
        self.declare_parameter(
            'path_efficiency_retry_min_lookahead_m', 5.0
        )
        self.declare_parameter(
            'path_efficiency_retry_lookahead_step_m', 1.5
        )
        self.declare_parameter('path_efficiency_blocked_retry_s', 5.0)
        self.declare_parameter('path_efficiency_escape_lookahead_m', 12.0)
        self.declare_parameter(
            'path_efficiency_escape_max_yaw_change_deg', 35.0
        )
        self.declare_parameter(
            'path_efficiency_allow_bounded_topology_escape', True
        )
        self.declare_parameter(
            'path_efficiency_topology_escape_after_retries', 6
        )
        self.declare_parameter(
            'path_efficiency_topology_escape_max_length_ratio', 3.1
        )
        self.declare_parameter(
            'path_efficiency_topology_escape_max_abs_turn_deg', 350.0
        )
        self.declare_parameter(
            'path_efficiency_topology_escape_max_signed_turn_deg', 330.0
        )
        # A repeated controller abort can leave a transient lethal cell under
        # the vehicle in the rolling costmap. Clear only a small neighborhood,
        # and only while two independent safety checks report a fresh clear
        # state. This never bypasses the raw-LiDAR emergency stop.
        self.declare_parameter('stuck_costmap_recovery_enabled', True)
        self.declare_parameter('stuck_costmap_recovery_abort_count', 2)
        self.declare_parameter(
            'stuck_costmap_recovery_reset_distance_m', 3.0
        )
        self.declare_parameter('stuck_costmap_recovery_max_speed_mps', 0.1)
        self.declare_parameter('stuck_costmap_recovery_cooldown_s', 10.0)
        self.declare_parameter('safety_state_topic', '/safety/state')
        self.declare_parameter('safety_state_timeout_s', 1.5)
        self.declare_parameter(
            'safety_trajectory_rejection_topic',
            '/safety/trajectory_rejection',
        )
        self.declare_parameter('safety_rejection_replanning_enabled', True)
        self.declare_parameter('safety_rejection_confirmations', 3)
        self.declare_parameter('safety_rejection_window_s', 1.0)
        self.declare_parameter('safety_rejection_minimum_goal_age_s', 0.5)
        self.declare_parameter('safety_replan_clear_confirmations', 3)
        self.declare_parameter('safety_replan_maximum_hold_s', 1.0)
        self.declare_parameter(
            'local_costmap_clear_service',
            '/local_costmap/clear_around_local_costmap',
        )
        self.declare_parameter(
            'global_costmap_clear_service',
            '/global_costmap/clear_around_global_costmap',
        )
        self.declare_parameter('grid_stride', 2)
        self.declare_parameter('lethal_cost_threshold', 99.0)
        self.declare_parameter('unknown_cost', 0.15)
        self.declare_parameter('cost_weight', 4.0)
        self.declare_parameter('boundary_margin_m', 3.0)
        self.declare_parameter('start_clearance_m', 2.0)
        self.declare_parameter('goal_search_radius_m', 4.0)
        self.declare_parameter('maximum_expansions', 120000)
        self.declare_parameter(
            'log_directory', '~/terrain_nav_data/logs/far_guide'
        )

        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.behavior_tree = str(
            self.get_parameter('behavior_tree').value
        )
        self.planning_rate_hz = float(
            self.get_parameter('planning_rate_hz').value
        )
        self.input_timeout_s = float(
            self.get_parameter('input_timeout_s').value
        )
        self.segment_lookahead_m = float(
            self.get_parameter('segment_lookahead_m').value
        )
        self.final_goal_distance_m = float(
            self.get_parameter('final_goal_distance_m').value
        )
        self.goal_tangent_max_bias_rad = math.radians(float(
            self.get_parameter('goal_tangent_max_bias_deg').value
        ))
        self.goal_change_tolerance_m = float(
            self.get_parameter('goal_change_tolerance_m').value
        )
        self.goal_hold_after_success_s = float(
            self.get_parameter('goal_hold_after_success_s').value
        )
        self.retry_delay_s = float(
            self.get_parameter('retry_delay_s').value
        )
        self.retry_direction_continuity_enabled = bool(
            self.get_parameter(
                'retry_direction_continuity_enabled'
            ).value
        )
        self.retry_direction_hold_s = float(
            self.get_parameter('retry_direction_hold_s').value
        )
        self.retry_direction_distance_m = float(
            self.get_parameter('retry_direction_distance_m').value
        )
        self.retry_direction_cost_weight = float(
            self.get_parameter('retry_direction_cost_weight').value
        )
        self.leg_direction_continuity_enabled = bool(
            self.get_parameter('leg_direction_continuity_enabled').value
        )
        self.leg_direction_distance_m = float(
            self.get_parameter('leg_direction_distance_m').value
        )
        self.leg_direction_cost_weight = float(
            self.get_parameter('leg_direction_cost_weight').value
        )
        self.failed_corridor_enabled = bool(
            self.get_parameter('failed_corridor_enabled').value
        )
        self.failed_corridor_hold_s = float(
            self.get_parameter('failed_corridor_hold_s').value
        )
        self.failed_corridor_radius_m = float(
            self.get_parameter('failed_corridor_radius_m').value
        )
        self.failed_corridor_cost_weight = float(
            self.get_parameter('failed_corridor_cost_weight').value
        )
        self.failed_corridor_hard_radius_m = float(
            self.get_parameter('failed_corridor_hard_radius_m').value
        )
        self.failed_corridor_similarity_m = float(
            self.get_parameter('failed_corridor_similarity_m').value
        )
        self.failed_corridor_max_count = int(
            self.get_parameter('failed_corridor_max_count').value
        )
        self.blocked_retry_s = float(
            self.get_parameter('blocked_retry_s').value
        )
        self.active_replanning_enabled = bool(
            self.get_parameter('active_replanning_enabled').value
        )
        self.active_replan_allow_valid_path_optimization = bool(
            self.get_parameter(
                'active_replan_allow_valid_path_optimization'
            ).value
        )
        self.active_replan_min_interval_s = float(
            self.get_parameter('active_replan_min_interval_s').value
        )
        self.active_replan_confirmations = int(
            self.get_parameter('active_replan_confirmations').value
        )
        self.active_replan_subgoal_change_m = float(
            self.get_parameter('active_replan_subgoal_change_m').value
        )
        self.active_replan_min_improvement_m = float(
            self.get_parameter('active_replan_min_improvement_m').value
        )
        self.active_replan_endpoint_tolerance_m = float(
            self.get_parameter('active_replan_endpoint_tolerance_m').value
        )
        self.active_replan_consistency_m = float(
            self.get_parameter('active_replan_consistency_m').value
        )
        self.path_validity_timeout_s = float(
            self.get_parameter('path_validity_timeout_s').value
        )
        self.path_efficiency_enabled = bool(
            self.get_parameter('path_efficiency_enabled').value
        )
        self.path_efficiency_max_length_ratio = float(
            self.get_parameter('path_efficiency_max_length_ratio').value
        )
        self.path_efficiency_max_abs_turn_rad = math.radians(float(
            self.get_parameter('path_efficiency_max_abs_turn_deg').value
        ))
        self.path_efficiency_max_signed_turn_rad = math.radians(float(
            self.get_parameter('path_efficiency_max_signed_turn_deg').value
        ))
        self.path_efficiency_min_reference_m = float(
            self.get_parameter('path_efficiency_min_reference_m').value
        )
        self.path_efficiency_min_excess_m = float(
            self.get_parameter('path_efficiency_min_excess_m').value
        )
        self.path_efficiency_endpoint_tolerance_m = float(
            self.get_parameter(
                'path_efficiency_endpoint_tolerance_m'
            ).value
        )
        self.path_efficiency_start_tolerance_m = float(
            self.get_parameter('path_efficiency_start_tolerance_m').value
        )
        self.path_efficiency_retry_yaw_hold_s = float(
            self.get_parameter('path_efficiency_retry_yaw_hold_s').value
        )
        self.path_efficiency_allow_length_only_detour = bool(
            self.get_parameter(
                'path_efficiency_allow_length_only_detour'
            ).value
        )
        self.path_efficiency_length_only_detour_after_retries = int(
            self.get_parameter(
                'path_efficiency_length_only_detour_after_retries'
            ).value
        )
        self.path_efficiency_max_retries = int(
            self.get_parameter('path_efficiency_max_retries').value
        )
        self.path_efficiency_retry_min_lookahead_m = float(
            self.get_parameter(
                'path_efficiency_retry_min_lookahead_m'
            ).value
        )
        self.path_efficiency_retry_lookahead_step_m = float(
            self.get_parameter(
                'path_efficiency_retry_lookahead_step_m'
            ).value
        )
        self.path_efficiency_blocked_retry_s = float(
            self.get_parameter('path_efficiency_blocked_retry_s').value
        )
        self.path_efficiency_escape_lookahead_m = float(
            self.get_parameter('path_efficiency_escape_lookahead_m').value
        )
        self.path_efficiency_escape_max_yaw_change_rad = math.radians(float(
            self.get_parameter(
                'path_efficiency_escape_max_yaw_change_deg'
            ).value
        ))
        self.path_efficiency_allow_bounded_topology_escape = bool(
            self.get_parameter(
                'path_efficiency_allow_bounded_topology_escape'
            ).value
        )
        self.path_efficiency_topology_escape_after_retries = int(
            self.get_parameter(
                'path_efficiency_topology_escape_after_retries'
            ).value
        )
        self.path_efficiency_topology_escape_max_length_ratio = float(
            self.get_parameter(
                'path_efficiency_topology_escape_max_length_ratio'
            ).value
        )
        self.path_efficiency_topology_escape_max_abs_turn_rad = math.radians(
            float(self.get_parameter(
                'path_efficiency_topology_escape_max_abs_turn_deg'
            ).value)
        )
        self.path_efficiency_topology_escape_max_signed_turn_rad = (
            math.radians(float(self.get_parameter(
                'path_efficiency_topology_escape_max_signed_turn_deg'
            ).value))
        )
        self.stuck_costmap_recovery_enabled = bool(
            self.get_parameter('stuck_costmap_recovery_enabled').value
        )
        self.stuck_costmap_recovery_abort_count = int(
            self.get_parameter('stuck_costmap_recovery_abort_count').value
        )
        self.stuck_costmap_recovery_reset_distance_m = float(
            self.get_parameter(
                'stuck_costmap_recovery_reset_distance_m'
            ).value
        )
        self.stuck_costmap_recovery_max_speed_mps = float(
            self.get_parameter(
                'stuck_costmap_recovery_max_speed_mps'
            ).value
        )
        self.stuck_costmap_recovery_cooldown_s = float(
            self.get_parameter('stuck_costmap_recovery_cooldown_s').value
        )
        self.safety_state_timeout_s = float(
            self.get_parameter('safety_state_timeout_s').value
        )
        self.safety_rejection_replanning_enabled = bool(
            self.get_parameter(
                'safety_rejection_replanning_enabled'
            ).value
        )
        self.safety_rejection_confirmations = int(
            self.get_parameter('safety_rejection_confirmations').value
        )
        self.safety_rejection_window_s = float(
            self.get_parameter('safety_rejection_window_s').value
        )
        self.safety_rejection_minimum_goal_age_s = float(
            self.get_parameter(
                'safety_rejection_minimum_goal_age_s'
            ).value
        )
        self.safety_replan_clear_confirmations = int(
            self.get_parameter(
                'safety_replan_clear_confirmations'
            ).value
        )
        self.safety_replan_maximum_hold_s = float(
            self.get_parameter('safety_replan_maximum_hold_s').value
        )
        self.grid_stride = int(self.get_parameter('grid_stride').value)
        self.lethal_cost_threshold = float(
            self.get_parameter('lethal_cost_threshold').value
        )
        self.unknown_cost = float(
            self.get_parameter('unknown_cost').value
        )
        self.cost_weight = float(
            self.get_parameter('cost_weight').value
        )
        self.boundary_margin_m = float(
            self.get_parameter('boundary_margin_m').value
        )
        self.start_clearance_m = float(
            self.get_parameter('start_clearance_m').value
        )
        self.goal_search_radius_m = float(
            self.get_parameter('goal_search_radius_m').value
        )
        self.maximum_expansions = int(
            self.get_parameter('maximum_expansions').value
        )
        if self.planning_rate_hz <= 0.0:
            raise ValueError('planning_rate_hz must be positive')
        if self.input_timeout_s <= 0.0:
            raise ValueError('input_timeout_s must be positive')
        if self.segment_lookahead_m <= 0.0:
            raise ValueError('segment_lookahead_m must be positive')
        if self.final_goal_distance_m <= 0.0:
            raise ValueError('final_goal_distance_m must be positive')
        if self.goal_tangent_max_bias_rad < 0.0:
            raise ValueError('goal_tangent_max_bias_deg cannot be negative')
        if min(
            self.retry_direction_hold_s,
            self.retry_direction_distance_m,
            self.retry_direction_cost_weight,
        ) < 0.0:
            raise ValueError('retry direction parameters cannot be negative')
        if min(
            self.leg_direction_distance_m,
            self.leg_direction_cost_weight,
        ) < 0.0:
            raise ValueError(
                'mission-leg direction parameters cannot be negative'
            )
        if min(
            self.failed_corridor_hold_s,
            self.failed_corridor_radius_m,
            self.failed_corridor_cost_weight,
            self.failed_corridor_hard_radius_m,
            self.failed_corridor_similarity_m,
            self.blocked_retry_s,
        ) < 0.0:
            raise ValueError('failed corridor parameters cannot be negative')
        if self.failed_corridor_max_count < 1:
            raise ValueError('failed corridor max count must be positive')
        if self.grid_stride < 1:
            raise ValueError('grid_stride must be positive')
        if self.maximum_expansions < 1:
            raise ValueError('maximum_expansions must be positive')
        if self.active_replan_min_interval_s < 0.0:
            raise ValueError(
                'active replan minimum interval cannot be negative'
            )
        if self.active_replan_confirmations < 1:
            raise ValueError('active replan confirmations must be positive')
        if self.active_replan_subgoal_change_m < 0.0:
            raise ValueError('active replan subgoal change cannot be negative')
        if self.active_replan_min_improvement_m < 0.0:
            raise ValueError('active replan improvement cannot be negative')
        if self.active_replan_endpoint_tolerance_m < 0.0:
            raise ValueError(
                'active replan endpoint tolerance cannot be negative'
            )
        if self.active_replan_consistency_m < 0.0:
            raise ValueError('active replan consistency cannot be negative')
        if self.path_validity_timeout_s <= 0.0:
            raise ValueError('path validity timeout must be positive')
        if min(
            self.path_efficiency_max_length_ratio,
            self.path_efficiency_max_abs_turn_rad,
            self.path_efficiency_max_signed_turn_rad,
        ) <= 0.0:
            raise ValueError('path efficiency limits must be positive')
        if min(
            self.path_efficiency_min_reference_m,
            self.path_efficiency_min_excess_m,
            self.path_efficiency_endpoint_tolerance_m,
            self.path_efficiency_start_tolerance_m,
            self.path_efficiency_retry_yaw_hold_s,
            self.path_efficiency_retry_lookahead_step_m,
            self.path_efficiency_blocked_retry_s,
            self.path_efficiency_escape_lookahead_m,
            self.path_efficiency_escape_max_yaw_change_rad,
        ) < 0.0:
            raise ValueError('path efficiency minimums cannot be negative')
        if self.path_efficiency_max_retries < 1:
            raise ValueError('path efficiency max retries must be positive')
        if self.path_efficiency_length_only_detour_after_retries < 0:
            raise ValueError(
                'length-only detour retry threshold cannot be negative'
            )
        if self.path_efficiency_topology_escape_after_retries < 1:
            raise ValueError('topology escape retry threshold must be positive')
        if min(
            self.path_efficiency_topology_escape_max_length_ratio,
            self.path_efficiency_topology_escape_max_abs_turn_rad,
            self.path_efficiency_topology_escape_max_signed_turn_rad,
        ) <= 0.0:
            raise ValueError('topology escape limits must be positive')
        if self.stuck_costmap_recovery_abort_count < 1:
            raise ValueError('costmap recovery abort count must be positive')
        if min(
            self.stuck_costmap_recovery_reset_distance_m,
            self.stuck_costmap_recovery_max_speed_mps,
            self.stuck_costmap_recovery_cooldown_s,
        ) < 0.0:
            raise ValueError('costmap recovery parameters cannot be negative')
        if self.stuck_costmap_recovery_reset_distance_m <= 0.0:
            raise ValueError('costmap recovery distance must be positive')
        if self.safety_state_timeout_s <= 0.0:
            raise ValueError('safety state timeout must be positive')
        if self.safety_rejection_confirmations < 1:
            raise ValueError(
                'safety rejection confirmations must be positive'
            )
        if self.safety_rejection_window_s <= 0.0:
            raise ValueError('safety rejection window must be positive')
        if self.safety_rejection_minimum_goal_age_s < 0.0:
            raise ValueError('safety rejection minimum goal age is invalid')
        if self.safety_replan_clear_confirmations < 1:
            raise ValueError(
                'safety replan clear confirmations must be positive'
            )
        if self.safety_replan_maximum_hold_s <= 0.0:
            raise ValueError('safety replan maximum hold must be positive')
        if not (
            0.0 < self.path_efficiency_retry_min_lookahead_m
            <= self.segment_lookahead_m
        ):
            raise ValueError(
                'path efficiency retry minimum lookahead must be positive '
                'and no greater than segment lookahead'
            )

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        costmap_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.path_publisher = self.create_publisher(
            PathMessage, '/navigation/far_guide_path', latched_qos
        )
        self.subgoal_publisher = self.create_publisher(
            PoseStamped, '/navigation/far_subgoal', latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, '/navigation/far_guide_status', latched_qos
        )
        # Preserve the status contract expected by the waypoint manager and UI.
        self.nav2_status_publisher = self.create_publisher(
            String, '/navigation/nav2_status', latched_qos
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('costmap_topic').value),
            self._on_costmap,
            costmap_qos,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('goal_tangent_topic').value),
            self._on_goal_tangent,
            latched_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('goal_local_topic').value),
            self._on_goal_local,
            latched_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('current_local_topic').value),
            self._on_current_local,
            reliable_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self._on_odometry,
            sensor_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('goal_reached_topic').value),
            self._on_goal_reached,
            latched_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter('path_hard_valid_topic').value),
            self._on_path_hard_valid,
            reliable_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('safety_state_topic').value),
            self._on_safety_state,
            reliable_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter(
                'safety_trajectory_rejection_topic'
            ).value),
            self._on_safety_trajectory_rejection,
            reliable_qos,
        )
        self.create_subscription(
            PathMessage,
            str(self.get_parameter('nav2_plan_topic').value),
            self._on_nav2_plan,
            reliable_qos,
        )
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter('action_name').value),
        )
        self.local_costmap_clear_client = self.create_client(
            ClearCostmapAroundRobot,
            str(self.get_parameter('local_costmap_clear_service').value),
        )
        self.global_costmap_clear_client = self.create_client(
            ClearCostmapAroundRobot,
            str(self.get_parameter('global_costmap_clear_service').value),
        )

        self.latest_costmap = None
        self.latest_costmap_wall_time = None
        self.latest_goal_local = None
        self.latest_goal_tangent_yaw_rad = None
        self.latest_goal_tangent_index = None
        self.latest_current_local = None
        self.latest_current_local_wall_time = None
        self.latest_odometry = None
        self.latest_odometry_wall_time = None
        self.goal_reached = False
        self.goal_generation = 0
        self.active_goal_handle = None
        self.active_goal_started_wall_time = None
        self.active_guide_path = None
        self.active_subgoal_xy = None
        self.goal_request_in_flight = False
        self.cancel_in_flight = False
        self.next_plan_wall_time = 0.0
        self.hold_until_wall_time = 0.0
        self.last_mission_goal_xy = None
        self.last_subgoal_xy = None
        self.latest_path_hard_valid = None
        self.latest_path_validity_wall_time = None
        self.latest_safety_state = None
        self.latest_safety_state_wall_time = None
        self.safety_rejection_monitor = SafetyRejectionMonitor(
            self.safety_rejection_confirmations,
            self.safety_rejection_window_s,
        )
        self.safety_replan_hold = SafetyReplanHold(
            self.safety_replan_clear_confirmations,
            self.safety_replan_maximum_hold_s,
        )
        self.pending_replan_reason = None
        self.pending_replan_subgoal_xy = None
        self.pending_replan_count = 0
        self.retry_direction_heading_rad = None
        self.retry_direction_until_wall_time = 0.0
        self.leg_direction_heading_rad = None
        self.failed_corridors = []
        self.canceled_goal_handles = []
        self.path_efficiency_retry_count = 0
        self.relaxed_subgoal_yaw_until_wall_time = 0.0
        self.path_efficiency_blocked_until_wall_time = 0.0
        self.path_efficiency_escape_mode = False
        self.path_efficiency_topology_escape_used = False
        self.active_inefficient_plan_signatures = set()
        self.consecutive_segment_aborts = 0
        self.last_costmap_recovery_wall_time = float('-inf')
        self.costmap_recovery_pending = 0
        self.log_stream, self.log_writer = self._create_logger(
            str(self.get_parameter('log_directory').value)
        )
        self.create_timer(1.0 / self.planning_rate_hz, self._tick)
        self._publish_status('waiting_for_inputs')
        self.get_logger().info(
            'FAR-style guide ready: long online guide -> {:.1f} m Smac '
            'Hybrid segments; hard cost >= {:.0f}; '
            'active replanning={}; valid-path optimization={}; '
            'Smac path efficiency gate={}'.format(
                self.segment_lookahead_m,
                self.lethal_cost_threshold,
                self.active_replanning_enabled,
                self.active_replan_allow_valid_path_optimization,
                self.path_efficiency_enabled,
            )
        )

    def _create_logger(self, directory):
        root = Path(directory).expanduser()
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        run_directory = root / ('run_' + stamp)
        run_directory.mkdir(parents=True, exist_ok=True)
        path = run_directory / 'far_guide.csv'
        stream = path.open('w', newline='', encoding='utf-8')
        fields = [
            'wall_time_iso', 'event', 'mission_goal_x_m',
            'mission_goal_y_m', 'subgoal_x_m', 'subgoal_y_m',
            'guide_vertices', 'mission_distance_m', 'action_status',
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        self.get_logger().info('FAR guide CSV: {}'.format(path))
        return stream, writer

    def _log(self, event, guide_vertices='', action_status=''):
        mission_x = mission_y = subgoal_x = subgoal_y = distance = ''
        if self.last_mission_goal_xy is not None:
            mission_x, mission_y = self.last_mission_goal_xy
        if self.last_subgoal_xy is not None:
            subgoal_x, subgoal_y = self.last_subgoal_xy
        if (
            self.last_mission_goal_xy is not None
            and self.latest_odometry is not None
        ):
            position = self.latest_odometry.pose.pose.position
            distance = math.hypot(
                self.last_mission_goal_xy[0] - float(position.x),
                self.last_mission_goal_xy[1] - float(position.y),
            )
        self.log_writer.writerow({
            'wall_time_iso': datetime.now().astimezone().isoformat(
                timespec='milliseconds'
            ),
            'event': event,
            'mission_goal_x_m': mission_x,
            'mission_goal_y_m': mission_y,
            'subgoal_x_m': subgoal_x,
            'subgoal_y_m': subgoal_y,
            'guide_vertices': guide_vertices,
            'mission_distance_m': distance,
            'action_status': action_status,
        })
        self.log_stream.flush()

    def _publish_status(self, status):
        message = String()
        message.data = str(status)
        self.status_publisher.publish(message)

    def _publish_nav2_status(self, status):
        message = String()
        message.data = str(status)
        self.nav2_status_publisher.publish(message)

    def _on_costmap(self, message):
        self.latest_costmap = message
        self.latest_costmap_wall_time = time.monotonic()

    def _on_current_local(self, message):
        values = float(message.point.x), float(message.point.y)
        if not all(math.isfinite(value) for value in values):
            return
        self.latest_current_local = values
        self.latest_current_local_wall_time = time.monotonic()

    def _on_odometry(self, message):
        self.latest_odometry = message
        self.latest_odometry_wall_time = time.monotonic()

    def _on_goal_tangent(self, message):
        x_value = float(message.vector.x)
        y_value = float(message.vector.y)
        magnitude = math.hypot(x_value, y_value)
        if not (
            math.isfinite(x_value)
            and math.isfinite(y_value)
            and magnitude > 0.5
        ):
            self.latest_goal_tangent_yaw_rad = None
        else:
            self.latest_goal_tangent_yaw_rad = math.atan2(
                y_value, x_value
            )
        index_value = float(message.vector.z)
        self.latest_goal_tangent_index = (
            int(round(index_value)) if math.isfinite(index_value) else None
        )

    def _on_goal_local(self, message):
        values = float(message.point.x), float(message.point.y)
        if not all(math.isfinite(value) for value in values):
            return
        changed = (
            self.latest_goal_local is None
            or math.hypot(
                values[0] - self.latest_goal_local[0],
                values[1] - self.latest_goal_local[1],
            ) > self.goal_change_tolerance_m
        )
        self.latest_goal_local = values
        if not changed:
            return
        self.goal_generation += 1
        self.goal_reached = False
        self.next_plan_wall_time = 0.0
        self.hold_until_wall_time = 0.0
        self.last_subgoal_xy = None
        self._clear_retry_direction()
        self._clear_leg_direction()
        self._clear_failed_corridors()
        self._reset_path_efficiency_retry(reset_topology_escape=True)
        self._reset_safety_rejection_monitor()
        self.safety_replan_hold.reset()
        self.consecutive_segment_aborts = 0
        tangent_status = 'tangent=undefined'
        if self.latest_goal_tangent_yaw_rad is not None:
            tangent_status = 'tangent={:.1f}deg;index={}'.format(
                math.degrees(self.latest_goal_tangent_yaw_rad),
                self.latest_goal_tangent_index,
            )
        self._log('mission_goal_changed', action_status=tangent_status)
        self._cancel_active_goal('mission_goal_changed')

    def _on_goal_reached(self, message):
        reached = bool(message.data)
        if reached and self.goal_reached:
            return
        self.goal_reached = reached
        if not reached:
            return
        self._clear_retry_direction()
        self._clear_leg_direction()
        self._clear_failed_corridors()
        self._reset_path_efficiency_retry(reset_topology_escape=True)
        self._reset_safety_rejection_monitor()
        self.safety_replan_hold.reset()
        self.consecutive_segment_aborts = 0
        self._publish_status('mission_waypoint_reached')
        self._log('mission_waypoint_reached')
        self._cancel_active_goal('mission_waypoint_reached')

    def _on_path_hard_valid(self, message):
        self.latest_path_hard_valid = bool(message.data)
        self.latest_path_validity_wall_time = time.monotonic()

    def _on_safety_state(self, message):
        self.latest_safety_state = str(message.data)
        self.latest_safety_state_wall_time = time.monotonic()
        self.safety_replan_hold.observe_state(self.latest_safety_state)

    def _reset_safety_rejection_monitor(self):
        self.safety_rejection_monitor.reset()

    def _on_safety_trajectory_rejection(self, message):
        """Replace a persistently unsafe controller trajectory promptly.

        The raw-LiDAR node remains authoritative and continues commanding
        zero speed.  This callback only feeds that already-enforced stop back
        to FAR so it can abandon the rejected guide before Nav2's 15 second
        progress timeout and search a different corridor.
        """

        if not self.safety_rejection_replanning_enabled:
            return
        if (
            self.active_goal_handle is None
            or self.active_goal_started_wall_time is None
            or self.cancel_in_flight
        ):
            self._reset_safety_rejection_monitor()
            return
        now = time.monotonic()
        safety_state_age_s = (
            float('inf')
            if self.latest_safety_state_wall_time is None
            else now - self.latest_safety_state_wall_time
        )
        # A trajectory-rejection message is also emitted while the safety
        # controller deliberately crawls through obstacle_recovery.  Only a
        # fresh, actual obstacle_stop is allowed to discard a Nav2 segment;
        # caution/recovery are already safe, usable commands.
        if not is_actionable_safety_stop(
            self.latest_safety_state,
            safety_state_age_s,
            self.safety_state_timeout_s,
        ):
            self._reset_safety_rejection_monitor()
            return
        if (
            now - self.active_goal_started_wall_time
            < self.safety_rejection_minimum_goal_age_s
        ):
            return
        try:
            payload = json.loads(message.data)
            if payload.get('reason') != 'obstacle_stop':
                return
            cloud_stamp_ns = int(payload.get('cloud_stamp_ns', 0))
            cloud_sequence = int(payload.get('cloud_sequence', 0))
            evidence_id = (
                cloud_stamp_ns
                if cloud_stamp_ns > 0
                else -cloud_sequence
            )
            if evidence_id == 0:
                return
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not self.safety_rejection_monitor.observe(now, evidence_id):
            return
        diagnostic = (
            'confirmations={};window_s={:.2f};cloud_stamp_ns={};'
            'curvature_per_m={};stop_points={}'.format(
                self.safety_rejection_monitor.count,
                self.safety_rejection_window_s,
                cloud_stamp_ns,
                payload.get('curvature_per_m', ''),
                payload.get('nominal_stop_point_count', ''),
            )
        )
        self._publish_status('safety_rejection_replanning')
        self._log(
            'safety_trajectory_rejection_replan',
            action_status=diagnostic,
        )
        self.get_logger().warning(
            'Persistent swept-trajectory rejection; replacing FAR segment: '
            + diagnostic
        )
        self._reset_safety_rejection_monitor()
        self._cancel_active_goal('safety_trajectory_rejection')

    def _vehicle_speed_mps(self):
        if self.latest_odometry is None:
            return float('inf')
        velocity = self.latest_odometry.twist.twist.linear
        return math.hypot(float(velocity.x), float(velocity.y))

    def _costmap_recovery_complete(self, future, layer):
        self.costmap_recovery_pending = max(
            0, self.costmap_recovery_pending - 1
        )
        try:
            future.result()
        except Exception as error:  # pragma: no cover - ROS service failure
            self._log(
                'costmap_recovery_failed',
                action_status='{}:{}'.format(layer, error),
            )
            self.get_logger().warning(
                '{} costmap recovery failed: {}'.format(layer, error)
            )
            return
        self._log('costmap_recovery_complete', action_status=layer)
        if self.costmap_recovery_pending == 0:
            # Let at least one fresh obstacle/clearing scan repopulate genuine
            # surroundings before asking the planners for another segment.
            self.next_plan_wall_time = max(
                self.next_plan_wall_time, time.monotonic() + 0.3
            )

    def _maybe_request_costmap_recovery(self):
        now = time.monotonic()
        path_validity_fresh = (
            self.latest_path_validity_wall_time is not None
            and now - self.latest_path_validity_wall_time
            <= self.path_validity_timeout_s
        )
        safety_state_fresh = (
            self.latest_safety_state_wall_time is not None
            and now - self.latest_safety_state_wall_time
            <= self.safety_state_timeout_s
        )
        cooldown_ready = (
            self.costmap_recovery_pending == 0
            and now - self.last_costmap_recovery_wall_time
            >= self.stuck_costmap_recovery_cooldown_s
        )
        if not self.stuck_costmap_recovery_enabled or not (
            should_request_costmap_recovery(
                self.consecutive_segment_aborts,
                self.stuck_costmap_recovery_abort_count,
                self._vehicle_speed_mps(),
                self.stuck_costmap_recovery_max_speed_mps,
                self.latest_path_hard_valid,
                path_validity_fresh,
                self.latest_safety_state,
                safety_state_fresh,
                cooldown_ready,
            )
        ):
            return False

        ready_clients = [
            ('local', self.local_costmap_clear_client),
            ('global', self.global_costmap_clear_client),
        ]
        ready_clients = [
            item for item in ready_clients if item[1].service_is_ready()
        ]
        if not ready_clients:
            self._log(
                'costmap_recovery_unavailable',
                action_status='clear_services_not_ready',
            )
            return False

        self.last_costmap_recovery_wall_time = now
        self.costmap_recovery_pending = len(ready_clients)
        for layer, client in ready_clients:
            request = ClearCostmapAroundRobot.Request()
            request.reset_distance = (
                self.stuck_costmap_recovery_reset_distance_m
            )
            future = client.call_async(request)
            future.add_done_callback(
                lambda completed, name=layer: self._costmap_recovery_complete(
                    completed, name
                )
            )
        diagnostic = 'aborts={};distance={:.1f};layers={}'.format(
            self.consecutive_segment_aborts,
            self.stuck_costmap_recovery_reset_distance_m,
            '+'.join(layer for layer, _ in ready_clients),
        )
        self._publish_status('clearing_transient_costmap_near_vehicle')
        self._log('costmap_recovery_requested', action_status=diagnostic)
        self.get_logger().warning(
            'Requesting safety-gated costmap recovery: {}'.format(diagnostic)
        )
        return True

    def _reset_path_efficiency_retry(self, reset_topology_escape=False):
        self.path_efficiency_retry_count = 0
        self.relaxed_subgoal_yaw_until_wall_time = 0.0
        self.path_efficiency_blocked_until_wall_time = 0.0
        self.path_efficiency_escape_mode = False
        self.active_inefficient_plan_signatures = set()
        if reset_topology_escape:
            self.path_efficiency_topology_escape_used = False

    def _on_nav2_plan(self, message):
        """Reject a newly published Hybrid path if it contains a large loop."""
        if (
            not self.path_efficiency_enabled
            or self.active_goal_handle is None
            or self.cancel_in_flight
            or self.active_goal_started_wall_time is None
            or self.active_guide_path is None
            or self.active_subgoal_xy is None
            or self.latest_odometry is None
            or len(message.poses) < 2
        ):
            return
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in message.poses
        ]
        if not all(
            math.isfinite(value)
            for point in points for value in point
        ):
            return
        endpoint_error = math.hypot(
            points[-1][0] - self.active_subgoal_xy[0],
            points[-1][1] - self.active_subgoal_xy[1],
        )
        position = self.latest_odometry.pose.pose.position
        current = float(position.x), float(position.y)
        start_error = math.hypot(
            points[0][0] - current[0], points[0][1] - current[1]
        )
        # `/plan` is shared.  Endpoint/start matching prevents a stale plan
        # from the preceding waypoint from canceling the new action.
        if (
            endpoint_error > self.path_efficiency_endpoint_tolerance_m
            or start_error > self.path_efficiency_start_tolerance_m
        ):
            return
        guide_prefix = prefix_polyline_to_point(
            self.active_guide_path, self.active_subgoal_xy
        )
        reference_length = remaining_polyline_length(
            guide_prefix, current
        )
        assessment = assess_path_efficiency(
            points,
            reference_length,
            self.path_efficiency_max_length_ratio,
            self.path_efficiency_max_abs_turn_rad,
            self.path_efficiency_max_signed_turn_rad,
            self.path_efficiency_min_reference_m,
            self.path_efficiency_min_excess_m,
        )
        if assessment is None or not assessment.inefficient:
            return
        # A long path is not necessarily a loop.  The observed failure mode
        # was: reject one genuine signed loop, then reject every safe detour
        # merely because it was longer than the coarse FAR prefix.  Keep
        # signed_loop/winding as hard topology failures, but permit a
        # length-ratio-only result after a prior retry.  Nav2 costmaps, the
        # path-clearance gate, and the emergency-stop node remain authoritative
        # for collision safety while this path is executed.
        if (
            self.path_efficiency_allow_length_only_detour
            and can_accept_length_only_detour(
                assessment,
                self.path_efficiency_retry_count,
                self.path_efficiency_length_only_detour_after_retries,
            )
        ):
            diagnostic = (
                'reasons=length_ratio;length={:.2f};reference={:.2f};'
                'ratio={:.2f};abs_turn_deg={:.1f};'
                'signed_turn_deg={:.1f};prior_retries={}'
            ).format(
                assessment.path_length_m,
                assessment.reference_length_m,
                assessment.length_ratio,
                math.degrees(assessment.absolute_turn_rad),
                math.degrees(assessment.signed_turn_rad),
                self.path_efficiency_retry_count,
            )
            self.path_efficiency_escape_mode = False
            self.path_efficiency_blocked_until_wall_time = 0.0
            self._publish_status('smac_length_only_detour_accepted')
            self._publish_nav2_status('navigating')
            self._log(
                'smac_length_only_detour_accepted',
                action_status=diagnostic,
            )
            self.get_logger().info(
                'Following non-looping Smac detour after topology retry: '
                '{}'.format(diagnostic)
            )
            return
        # Forward-only Ackermann motion can occasionally require a single
        # wide P-turn.  Do not accept it immediately: first exhaust ordinary
        # short-goal/yaw retries, then allow at most one path per mission
        # waypoint inside a second, finite topology envelope.  Nav2 costmaps,
        # the independent path-validity gate, and raw-LiDAR emergency stop
        # continue to enforce collision safety while it is followed.
        if (
            self.path_efficiency_allow_bounded_topology_escape
            and can_accept_bounded_topology_escape(
                assessment,
                self.path_efficiency_retry_count,
                self.path_efficiency_topology_escape_after_retries,
                self.path_efficiency_topology_escape_used,
                self.path_efficiency_topology_escape_max_length_ratio,
                self.path_efficiency_topology_escape_max_abs_turn_rad,
                self.path_efficiency_topology_escape_max_signed_turn_rad,
            )
        ):
            diagnostic = (
                'reasons={};length={:.2f};reference={:.2f};ratio={:.2f};'
                'abs_turn_deg={:.1f};signed_turn_deg={:.1f};'
                'prior_retries={}'
            ).format(
                '+'.join(assessment.reasons),
                assessment.path_length_m,
                assessment.reference_length_m,
                assessment.length_ratio,
                math.degrees(assessment.absolute_turn_rad),
                math.degrees(assessment.signed_turn_rad),
                self.path_efficiency_retry_count,
            )
            self.path_efficiency_topology_escape_used = True
            self.path_efficiency_escape_mode = False
            self.path_efficiency_blocked_until_wall_time = 0.0
            self._publish_status('smac_bounded_topology_escape_accepted')
            self._publish_nav2_status('navigating')
            self._log(
                'smac_bounded_topology_escape_accepted',
                action_status=diagnostic,
            )
            self.get_logger().warning(
                'Following one bounded Smac P-turn after ordinary retries: '
                '{}'.format(diagnostic)
            )
            return
        signature = (
            round(assessment.path_length_m, 1),
            round(assessment.signed_turn_rad, 1),
        )
        repeated_signature = signature in self.active_inefficient_plan_signatures
        self.active_inefficient_plan_signatures.add(signature)
        self.path_efficiency_retry_count = min(
            self.path_efficiency_retry_count + 1,
            self.path_efficiency_max_retries,
        )
        self.relaxed_subgoal_yaw_until_wall_time = (
            time.monotonic() + self.path_efficiency_retry_yaw_hold_s
        )
        exhausted = (
            self.path_efficiency_retry_count
            >= self.path_efficiency_max_retries
        )
        if exhausted:
            self.path_efficiency_escape_mode = True
            self.path_efficiency_blocked_until_wall_time = (
                time.monotonic() + self.path_efficiency_blocked_retry_s
            )
        diagnostic = (
            'reasons={};length={:.2f};reference={:.2f};ratio={:.2f};'
            'abs_turn_deg={:.1f};signed_turn_deg={:.1f};retry={};repeat={}'
        ).format(
            '+'.join(assessment.reasons),
            assessment.path_length_m,
            assessment.reference_length_m,
            assessment.length_ratio,
            math.degrees(assessment.absolute_turn_rad),
            math.degrees(assessment.signed_turn_rad),
            self.path_efficiency_retry_count,
            int(repeated_signature),
        )
        self._publish_status(
            'smac_path_inefficient_blocked'
            if exhausted else 'smac_path_inefficient_replanning'
        )
        if exhausted:
            self._publish_nav2_status('blocked_inefficient_path')
        self._log('smac_path_inefficient', action_status=diagnostic)
        self.get_logger().warning(
            'Rejecting inefficient Smac Hybrid path: {}'.format(diagnostic)
        )
        if exhausted:
            self.get_logger().warning(
                'Hybrid loop retry limit reached; holding {:.1f} s before '
                'one forward-feasible {:.1f} m escape probe'.format(
                    self.path_efficiency_blocked_retry_s,
                    self.path_efficiency_escape_lookahead_m,
                )
            )
        if exhausted:
            reason = 'inefficient_path_exhausted'
        elif self.path_efficiency_retry_count >= 2:
            reason = 'inefficient_path_alternate'
        else:
            reason = 'inefficient_path'
        self._cancel_active_goal(reason)

    def _reset_replan_monitor(self):
        self.pending_replan_reason = None
        self.pending_replan_subgoal_xy = None
        self.pending_replan_count = 0

    def _clear_retry_direction(self):
        self.retry_direction_heading_rad = None
        self.retry_direction_until_wall_time = 0.0

    def _remember_retry_direction(self, guide_path, source):
        if (
            not self.retry_direction_continuity_enabled
            or not guide_path
            or self.latest_odometry is None
        ):
            return
        position = self.latest_odometry.pose.pose.position
        heading = nearest_polyline_tangent(
            guide_path,
            (float(position.x), float(position.y)),
        )
        if heading is None:
            return
        self.retry_direction_heading_rad = heading
        self.retry_direction_until_wall_time = (
            time.monotonic() + self.retry_direction_hold_s
        )
        diagnostic = 'source={};heading_deg={:.1f};hold_s={:.1f}'.format(
            source,
            math.degrees(heading),
            self.retry_direction_hold_s,
        )
        self._log('retry_direction_armed', action_status=diagnostic)
        self.get_logger().info(
            'Retry direction continuity armed: {}'.format(diagnostic)
        )

    def _active_retry_heading(self, now):
        if (
            self.retry_direction_heading_rad is None
            or now > self.retry_direction_until_wall_time
        ):
            self._clear_retry_direction()
            return None
        return self.retry_direction_heading_rad

    def _clear_leg_direction(self):
        self.leg_direction_heading_rad = None

    def _remember_leg_direction(self, guide_path, source):
        if (
            not self.leg_direction_continuity_enabled
            or not guide_path
            or self.latest_odometry is None
        ):
            return
        position = self.latest_odometry.pose.pose.position
        heading = nearest_polyline_tangent(
            guide_path,
            (float(position.x), float(position.y)),
        )
        if heading is None:
            return
        self.leg_direction_heading_rad = heading
        diagnostic = 'source={};heading_deg={:.1f}'.format(
            source,
            math.degrees(heading),
        )
        self._log('leg_direction_updated', action_status=diagnostic)
        self.get_logger().info(
            'Mission-leg direction continuity updated: {}'.format(
                diagnostic
            )
        )

    def _planning_direction_continuity(self, now):
        retry_heading = self._active_retry_heading(now)
        preference = select_direction_continuity(
            retry_heading,
            self.leg_direction_heading_rad,
            self.retry_direction_distance_m,
            self.retry_direction_cost_weight,
            self.leg_direction_distance_m,
            self.leg_direction_cost_weight,
        )
        return preference

    def _clear_failed_corridors(self):
        self.failed_corridors = []

    def _active_failed_corridors(self, now):
        self.failed_corridors = [
            entry for entry in self.failed_corridors
            if float(entry['until']) >= now
        ]
        return [entry['path'] for entry in self.failed_corridors]

    def _remember_failed_corridor(
        self, guide_path, subgoal, source
    ):
        if (
            not self.failed_corridor_enabled
            or not guide_path
            or subgoal is None
        ):
            return
        failed_prefix = tuple(prefix_polyline_to_point(guide_path, subgoal))
        if len(failed_prefix) < 2:
            return
        now = time.monotonic()
        self._active_failed_corridors(now)
        matching = None
        for entry in self.failed_corridors:
            if polylines_similar(
                failed_prefix,
                entry['path'],
                self.failed_corridor_similarity_m,
            ):
                matching = entry
                break
        if matching is None:
            matching = {
                'path': failed_prefix,
                'until': now + self.failed_corridor_hold_s,
                'count': 1,
            }
            self.failed_corridors.append(matching)
            if len(self.failed_corridors) > self.failed_corridor_max_count:
                self.failed_corridors.pop(0)
        else:
            matching['path'] = failed_prefix
            matching['until'] = now + self.failed_corridor_hold_s
            matching['count'] = int(matching['count']) + 1
        diagnostic = (
            'source={};count={};vertices={};hold_s={:.1f}'.format(
                source,
                matching['count'],
                len(failed_prefix),
                self.failed_corridor_hold_s,
            )
        )
        self._log('failed_corridor_armed', action_status=diagnostic)
        self.get_logger().warning(
            'Controller-rejected corridor remembered: {}'.format(diagnostic)
        )

    def _candidate_repeats_failed(self, candidate, failed_paths):
        return any(
            polylines_similar(
                candidate, failed, self.failed_corridor_similarity_m
            )
            for failed in failed_paths
        )

    def _cancel_active_goal(self, reason):
        if self.active_goal_handle is None or self.cancel_in_flight:
            return
        if reason in (
            'active_replan',
            'safety_trajectory_rejection',
            'inefficient_path',
            'inefficient_path_alternate',
            'inefficient_path_exhausted',
        ):
            self._remember_retry_direction(
                self.active_guide_path, reason
            )
        if reason in (
            'active_replan',
            'safety_trajectory_rejection',
            'inefficient_path_alternate',
            'inefficient_path_exhausted',
        ):
            self._remember_failed_corridor(
                self.active_guide_path,
                self.active_subgoal_xy,
                reason,
            )
        self.cancel_in_flight = True
        handle = self.active_goal_handle
        self.canceled_goal_handles.append(handle)
        future = handle.cancel_goal_async()
        future.add_done_callback(
            lambda completed, requested_handle=handle,
            requested_reason=reason: self._cancel_complete(
                completed, requested_handle, requested_reason
            )
        )

    def _cancel_complete(self, future, handle, reason):
        try:
            future.result()
        except Exception as error:  # pragma: no cover - ROS transport failure
            self.get_logger().warning(
                'FAR segment cancellation failed: {}'.format(error)
            )
        if handle is self.active_goal_handle:
            self.active_goal_handle = None
            self.active_goal_started_wall_time = None
            self.active_guide_path = None
            self.active_subgoal_xy = None
        self._reset_replan_monitor()
        self._reset_safety_rejection_monitor()
        self.cancel_in_flight = False
        now = time.monotonic()
        if reason == 'safety_trajectory_rejection':
            self.safety_replan_hold.arm(now)
            self.next_plan_wall_time = now
            diagnostic = 'clear_confirmations={};maximum_hold_s={:.2f}'.format(
                self.safety_replan_clear_confirmations,
                self.safety_replan_maximum_hold_s,
            )
            self._publish_status('safety_replan_waiting_for_clear')
            self._log('safety_replan_hold_started', action_status=diagnostic)
        elif reason == 'inefficient_path_exhausted':
            self.safety_replan_hold.reset()
            self.next_plan_wall_time = max(
                now,
                self.path_efficiency_blocked_until_wall_time,
            )
        else:
            self.safety_replan_hold.reset()
            self.next_plan_wall_time = now + 0.1
        self._log('segment_canceled_' + reason)

    def _inputs_ready(self):
        now = time.monotonic()
        if (
            self.latest_costmap is None
            or self.latest_goal_local is None
            or self.latest_current_local is None
            or self.latest_odometry is None
        ):
            return False
        ages = (
            now - self.latest_costmap_wall_time,
            now - self.latest_current_local_wall_time,
            now - self.latest_odometry_wall_time,
        )
        return all(age <= self.input_timeout_s for age in ages)

    def _mission_goal_in_odom(self):
        position = self.latest_odometry.pose.pose.position
        return (
            float(position.x)
            + self.latest_goal_local[0] - self.latest_current_local[0],
            float(position.y)
            + self.latest_goal_local[1] - self.latest_current_local[1],
        )

    def _costmap_grid(self):
        message = self.latest_costmap
        width = int(message.info.width)
        height = int(message.info.height)
        values = np.asarray(message.data, dtype=np.float32)
        if values.size != width * height:
            raise ValueError('costmap data size does not match metadata')
        return build_guide_grid(
            values.reshape(height, width),
            float(message.info.resolution),
            float(message.info.origin.position.x),
            float(message.info.origin.position.y),
            stride=self.grid_stride,
            lethal_threshold=self.lethal_cost_threshold,
            unknown_cost=self.unknown_cost,
        )

    def _publish_guide(self, world_path):
        now = self.get_clock().now().to_msg()
        path = PathMessage()
        path.header.stamp = now
        path.header.frame_id = self.odom_frame
        for index, point in enumerate(world_path):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            if index + 1 < len(world_path):
                following = world_path[index + 1]
                yaw = math.atan2(
                    following[1] - point[1], following[0] - point[0]
                )
            elif index > 0:
                previous = world_path[index - 1]
                yaw = math.atan2(
                    point[1] - previous[1], point[0] - previous[0]
                )
            else:
                yaw = 0.0
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path.poses.append(pose)
        self.path_publisher.publish(path)

    def _plan_world_path(
        self, grid, start, mission_goal, now, failed_paths,
        hard_block_radius_m=0.0,
    ):
        continuity_heading, continuity_distance, continuity_weight, _ = (
            self._planning_direction_continuity(now)
        )
        return plan_online_guide(
            grid,
            start,
            mission_goal,
            boundary_margin_m=self.boundary_margin_m,
            start_clearance_m=self.start_clearance_m,
            goal_search_radius_m=self.goal_search_radius_m,
            cost_weight=self.cost_weight,
            maximum_expansions=self.maximum_expansions,
            continuity_heading_rad=continuity_heading,
            continuity_distance_m=continuity_distance,
            continuity_cost_weight=continuity_weight,
            failed_corridors=failed_paths,
            failed_corridor_radius_m=self.failed_corridor_radius_m,
            failed_corridor_cost_weight=self.failed_corridor_cost_weight,
            failed_corridor_start_release_m=self.start_clearance_m,
            failed_corridor_hard_block_radius_m=hard_block_radius_m,
        )

    def _select_segment_target(
        self, world_path, mission_goal, mission_distance, start, now
    ):
        if self.path_efficiency_escape_mode:
            retry_lookahead = self.path_efficiency_escape_lookahead_m
        else:
            retry_lookahead = retry_lookahead_distance(
                self.segment_lookahead_m,
                self.path_efficiency_retry_min_lookahead_m,
                self.path_efficiency_retry_lookahead_step_m,
                self.path_efficiency_retry_count,
            )
        lookahead = min(retry_lookahead, mission_distance)
        subgoal, yaw, path_end = select_subgoal(world_path, lookahead)
        direct_final_goal = (
            mission_distance <= self.final_goal_distance_m
            and (
                self.path_efficiency_retry_count == 0
                or mission_distance
                <= self.path_efficiency_retry_min_lookahead_m
            )
        )
        if direct_final_goal:
            subgoal = mission_goal
            guide_yaw = yaw
            if self.latest_goal_tangent_yaw_rad is not None:
                # Route tangent is a preference, not a hard terminal pose.
                # Smac Hybrid + DUBIN must otherwise make a full circle when
                # the next-leg tangent differs sharply from the feasible
                # arrival heading of this short guide.
                yaw = bounded_heading_preference(
                    guide_yaw,
                    self.latest_goal_tangent_yaw_rad,
                    self.goal_tangent_max_bias_rad,
                )
            elif len(world_path) >= 2:
                previous = world_path[-2]
                yaw = math.atan2(
                    mission_goal[1] - previous[1],
                    mission_goal[0] - previous[0],
                )
            path_end = True
        # Only after the actual Hybrid path has been proven inefficient,
        # replace the guide-tangent terminal yaw with line-of-sight yaw.  This
        # preserves the normal Ackermann route while removing the pose
        # constraint that produced the observed full Dubins loop.
        if (
            now <= self.relaxed_subgoal_yaw_until_wall_time
        ):
            dx = float(subgoal[0]) - float(start[0])
            dy = float(subgoal[1]) - float(start[1])
            if math.hypot(dx, dy) > 0.5:
                yaw = math.atan2(dy, dx)
        if self.path_efficiency_escape_mode and self.latest_odometry is not None:
            orientation = self.latest_odometry.pose.pose.orientation
            current_yaw = math.atan2(
                2.0 * (
                    float(orientation.w) * float(orientation.z)
                    + float(orientation.x) * float(orientation.y)
                ),
                1.0 - 2.0 * (
                    float(orientation.y) ** 2
                    + float(orientation.z) ** 2
                ),
            )
            # An exhausted retry must change the Hybrid-A* pose problem.
            # Keep the goal farther ahead and bound its terminal heading from
            # the current Ackermann heading instead of asking for a nearby,
            # sharply rotated pose that can only be reached with a full loop.
            yaw = bounded_heading_preference(
                current_yaw,
                yaw,
                self.path_efficiency_escape_max_yaw_change_rad,
            )
        return subgoal, yaw, path_end

    def _tick(self):
        now = time.monotonic()
        if self.goal_reached:
            return
        if (
            self.goal_request_in_flight
            or self.cancel_in_flight
            or self.costmap_recovery_pending > 0
        ):
            return
        if self.safety_replan_hold.active:
            release_reason = self.safety_replan_hold.release_reason(now)
            if release_reason is None:
                return
            diagnostic = 'reason={};clear_count={};elapsed_s={:.3f}'.format(
                release_reason,
                self.safety_replan_hold.clear_count,
                now - self.safety_replan_hold.started_s,
            )
            self.safety_replan_hold.reset()
            self._publish_status('safety_replan_hold_released')
            self._log(
                'safety_replan_hold_released', action_status=diagnostic
            )
        if now < self.next_plan_wall_time or now < self.hold_until_wall_time:
            return
        if now < self.path_efficiency_blocked_until_wall_time:
            return
        if (
            self.active_goal_handle is None
            and self.path_efficiency_retry_count
            >= self.path_efficiency_max_retries
        ):
            # A 5 m goal with a large yaw change was the source of repeated
            # full Dubins loops.  Keep rejected signatures and use a farther,
            # heading-bounded escape pose after the cooldown.
            self.path_efficiency_escape_mode = True
            self._publish_status('smac_path_inefficient_escape_probe')
            self._log(
                'smac_path_inefficient_escape_probe',
                action_status='lookahead={:.2f};retry={}'.format(
                    self.path_efficiency_escape_lookahead_m,
                    self.path_efficiency_retry_count,
                ),
            )
        if not self._inputs_ready():
            self._publish_status('waiting_for_fresh_inputs')
            return
        if not self.action_client.server_is_ready():
            self.action_client.wait_for_server(timeout_sec=0.0)
            self._publish_status('waiting_for_nav2')
            return
        position = self.latest_odometry.pose.pose.position
        start = float(position.x), float(position.y)
        mission_goal = self._mission_goal_in_odom()
        self.last_mission_goal_xy = mission_goal
        mission_distance = math.hypot(
            mission_goal[0] - start[0], mission_goal[1] - start[1]
        )
        try:
            grid = self._costmap_grid()
            blocked_percent = (
                100.0 * float(np.count_nonzero(grid.blocked))
                / float(grid.blocked.size)
            )
            failed_paths = self._active_failed_corridors(now)
            world_path = self._plan_world_path(
                grid, start, mission_goal, now, failed_paths
            )
        except ValueError as error:
            self._publish_status('guide_input_invalid')
            self._log('guide_input_invalid')
            self.get_logger().error(
                'FAR guide input invalid: {}'.format(error)
            )
            self.next_plan_wall_time = now + self.retry_delay_s
            return
        if not world_path:
            self._publish_status('guide_not_found')
            diagnostic = 'threshold={:.0f};blocked={:.2f}%'.format(
                self.lethal_cost_threshold,
                blocked_percent,
            )
            self._log('guide_not_found', action_status=diagnostic)
            self.get_logger().warning(
                'No online guide to current mission waypoint; {}. '
                'Retrying'.format(diagnostic)
            )
            self.next_plan_wall_time = now + self.retry_delay_s
            return
        subgoal, yaw, path_end = self._select_segment_target(
            world_path, mission_goal, mission_distance, start, now
        )
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.odom_frame
        pose.pose.position.x = float(subgoal[0])
        pose.pose.position.y = float(subgoal[1])
        pose.pose.position.z = float(position.z)
        pose.pose.orientation.z = math.sin(0.5 * yaw)
        pose.pose.orientation.w = math.cos(0.5 * yaw)
        if self.active_goal_handle is not None:
            self._publish_guide(world_path)
            self._monitor_active_segment(
                now, world_path, subgoal, pose, mission_distance
            )
            return
        candidate_prefix = prefix_polyline_to_point(world_path, subgoal)
        if self._candidate_repeats_failed(candidate_prefix, failed_paths):
            diagnostic = 'failed_count={}'.format(len(failed_paths))
            self._log(
                'duplicate_guide_rejected', action_status=diagnostic
            )
            self.get_logger().warning(
                'Duplicate controller-rejected guide suppressed; '
                'searching another corridor ({})'.format(diagnostic)
            )
            try:
                world_path = self._plan_world_path(
                    grid,
                    start,
                    mission_goal,
                    now,
                    failed_paths,
                    hard_block_radius_m=(
                        self.failed_corridor_hard_radius_m
                    ),
                )
            except ValueError as error:
                world_path = None
                self.get_logger().warning(
                    'Alternate guide input invalid: {}'.format(error)
                )
            if world_path:
                subgoal, yaw, path_end = self._select_segment_target(
                    world_path, mission_goal, mission_distance, start, now
                )
                candidate_prefix = prefix_polyline_to_point(
                    world_path, subgoal
                )
            if (
                not world_path
                or self._candidate_repeats_failed(
                    candidate_prefix, failed_paths
                )
            ):
                self._publish_status('blocked_no_alternative')
                self._publish_nav2_status('blocked_no_alternative')
                self._log(
                    'blocked_no_alternative', action_status=diagnostic
                )
                self.get_logger().warning(
                    'No distinct controller-compatible corridor is '
                    'currently visible; holding instead of resending the '
                    'same failed action'
                )
                self.next_plan_wall_time = now + self.blocked_retry_s
                return
            pose.pose.position.x = float(subgoal[0])
            pose.pose.position.y = float(subgoal[1])
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            self._log(
                'alternate_guide_selected',
                guide_vertices=len(world_path),
                action_status=diagnostic,
            )
        self._publish_guide(world_path)
        self.last_subgoal_xy = subgoal
        self.subgoal_publisher.publish(pose)
        self._send_segment(pose, path_end, world_path)

    def _active_path_is_blocked(self, now):
        if (
            self.latest_path_hard_valid is None
            or self.latest_path_validity_wall_time is None
            or self.active_goal_started_wall_time is None
        ):
            return False
        if (
            self.latest_path_validity_wall_time
            < self.active_goal_started_wall_time
        ):
            return False
        if (
            now - self.latest_path_validity_wall_time
            > self.path_validity_timeout_s
        ):
            return False
        return not self.latest_path_hard_valid

    def _monitor_active_segment(
        self, now, world_path, subgoal, pose, mission_distance
    ):
        if not self.active_replanning_enabled:
            return
        if (
            self.active_goal_started_wall_time is None
            or self.active_guide_path is None
            or self.active_subgoal_xy is None
        ):
            return
        if (
            now - self.active_goal_started_wall_time
            < self.active_replan_min_interval_s
        ):
            self._reset_replan_monitor()
            return
        position = self.latest_odometry.pose.pose.position
        assessment = assess_active_replan(
            self.active_guide_path,
            self.active_subgoal_xy,
            world_path,
            subgoal,
            (float(position.x), float(position.y)),
            self._active_path_is_blocked(now),
            self.active_replan_subgoal_change_m,
            self.active_replan_min_improvement_m,
            self.active_replan_endpoint_tolerance_m,
            self.active_replan_allow_valid_path_optimization,
        )
        if assessment is None:
            self._reset_replan_monitor()
            return
        consistent = (
            assessment.reason == self.pending_replan_reason
            and self.pending_replan_subgoal_xy is not None
            and math.hypot(
                subgoal[0] - self.pending_replan_subgoal_xy[0],
                subgoal[1] - self.pending_replan_subgoal_xy[1],
            ) <= self.active_replan_consistency_m
        )
        if consistent:
            self.pending_replan_count += 1
        else:
            self.pending_replan_reason = assessment.reason
            self.pending_replan_subgoal_xy = subgoal
            self.pending_replan_count = 1
        self._publish_status(
            'active_replan_candidate_{}_of_{}'.format(
                self.pending_replan_count,
                self.active_replan_confirmations,
            )
        )
        if self.pending_replan_count < self.active_replan_confirmations:
            return
        diagnostic = (
            '{};change={:.2f};improvement={:.2f};distance={:.2f}'.format(
                assessment.reason,
                assessment.subgoal_change_m,
                assessment.improvement_m,
                mission_distance,
            )
        )
        self.last_subgoal_xy = subgoal
        self.subgoal_publisher.publish(pose)
        self._publish_status('active_replan_replacing_segment')
        self._log('active_replan_triggered', action_status=diagnostic)
        self._cancel_active_goal('active_replan')

    def _send_segment(self, pose, path_end, world_path):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        if self.behavior_tree:
            goal.behavior_tree = self.behavior_tree
        generation = self.goal_generation
        self.goal_request_in_flight = True
        self._publish_status('sending_far_segment')
        self._publish_nav2_status('goal_sent')
        position = self.latest_odometry.pose.pose.position
        target_distance = math.hypot(
            float(pose.pose.position.x) - float(position.x),
            float(pose.pose.position.y) - float(position.y),
        )
        self._log(
            'segment_sent',
            guide_vertices=len(world_path),
            action_status='retry={};target_distance={:.2f}'.format(
                self.path_efficiency_retry_count,
                target_distance,
            ),
        )
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed, requested_generation=generation,
            requested_path_end=path_end,
            requested_path=tuple(world_path),
            requested_subgoal=self.last_subgoal_xy: self._goal_response(
                completed, requested_generation, requested_path_end,
                requested_path, requested_subgoal,
            )
        )

    def _goal_response(
        self, future, generation, path_end, guide_path, subgoal
    ):
        self.goal_request_in_flight = False
        try:
            handle = future.result()
        except Exception as error:  # pragma: no cover - ROS transport failure
            self._publish_status('segment_send_failed')
            self._publish_nav2_status('send_failed')
            self._log('segment_send_failed')
            self.get_logger().error(
                'FAR segment send failed: {}'.format(error)
            )
            self.next_plan_wall_time = time.monotonic() + self.retry_delay_s
            return
        if not handle.accepted:
            self._publish_status('segment_rejected')
            self._publish_nav2_status('goal_rejected')
            self._log('segment_rejected')
            self.next_plan_wall_time = time.monotonic() + self.retry_delay_s
            return
        self.active_goal_handle = handle
        # A GNSS waypoint can advance while the action request is travelling
        # to Nav2. Never let an accepted segment for the previous waypoint
        # regain command authority after that handoff.
        if generation != self.goal_generation or self.goal_reached:
            self._cancel_active_goal('stale_segment')
            return
        self.active_goal_started_wall_time = time.monotonic()
        self.active_guide_path = guide_path
        self.active_subgoal_xy = subgoal
        # Preserve rejected signatures across retries for this mission
        # waypoint. A repeated `/plan` must be canceled again, not silently
        # accepted merely because it has already been observed once.
        self.latest_path_hard_valid = None
        self.latest_path_validity_wall_time = None
        self._reset_replan_monitor()
        self._reset_safety_rejection_monitor()
        self._publish_status('following_far_segment')
        self._publish_nav2_status('navigating')
        self._log('segment_accepted')
        result = handle.get_result_async()
        result.add_done_callback(
            lambda completed, completed_handle=handle,
            completed_generation=generation,
            completed_path_end=path_end: self._goal_result(
                completed,
                completed_handle,
                completed_generation,
                completed_path_end,
            )
        )

    def _goal_result(self, future, handle, generation, path_end):
        try:
            status = int(future.result().status)
        except Exception as error:  # pragma: no cover - ROS transport failure
            status = -1
            self.get_logger().error(
                'FAR segment result failed: {}'.format(error)
            )
        canceled = next(
            (item for item in self.canceled_goal_handles if item is handle),
            None,
        )
        if canceled is not None:
            self.canceled_goal_handles.remove(canceled)
            self._log(
                'canceled_segment_result_ignored', action_status=status
            )
            return
        completed_guide_path = None
        completed_subgoal_xy = None
        if handle is self.active_goal_handle:
            completed_guide_path = self.active_guide_path
            completed_subgoal_xy = self.active_subgoal_xy
            self.active_goal_handle = None
            self.active_goal_started_wall_time = None
            self.active_guide_path = None
            self.active_subgoal_xy = None
        self._reset_replan_monitor()
        self._reset_safety_rejection_monitor()
        self.cancel_in_flight = False
        self._publish_nav2_status('nav2_result_{}'.format(status))
        self._log('segment_result', action_status=status)
        if generation != self.goal_generation or self.goal_reached:
            self.next_plan_wall_time = time.monotonic() + 0.1
            return
        if status == 4:
            # Continue the homotopy established by this successful short
            # action.  Retry continuity is intentionally cleared below, but
            # mission-leg continuity remains until the waypoint changes.
            self._remember_leg_direction(
                completed_guide_path, 'segment_succeeded'
            )
            self.consecutive_segment_aborts = 0
            self._reset_path_efficiency_retry()
            self._clear_retry_direction()
            self._clear_failed_corridors()
            self._publish_status(
                'waiting_for_mission_arrival'
                if path_end else 'far_segment_complete'
            )
            delay = self.goal_hold_after_success_s if path_end else 0.2
            self.hold_until_wall_time = time.monotonic() + delay
            self.next_plan_wall_time = self.hold_until_wall_time
        else:
            # Nav2 status 6 is ABORTED (typically failed progress or no valid
            # control). Preserve the corridor tangent before clearing the
            # failed short action so the next online A* retry does not choose
            # the symmetric opposite direction merely due to grid jitter.
            recovery_requested = False
            if status == 6:
                self.consecutive_segment_aborts += 1
                self._remember_retry_direction(
                    completed_guide_path, 'segment_aborted'
                )
                self._remember_failed_corridor(
                    completed_guide_path,
                    completed_subgoal_xy,
                    'segment_aborted',
                )
                recovery_requested = self._maybe_request_costmap_recovery()
            else:
                self.consecutive_segment_aborts = 0
            self._publish_status('far_segment_failed_replanning')
            retry_delay = 0.5 if recovery_requested else self.retry_delay_s
            self.next_plan_wall_time = time.monotonic() + retry_delay

    def destroy_node(self):
        if hasattr(self, 'log_stream'):
            try:
                self.log_stream.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FarNav2GuideNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
