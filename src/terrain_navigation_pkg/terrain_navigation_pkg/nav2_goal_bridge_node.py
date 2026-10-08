"""Forward GNSS goals and execute rolling routes one Hybrid segment at a time."""

import csv
from datetime import datetime
import math
from pathlib import Path
import time

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, UInt32

from .nav2_goal_bridge_core import bounded_virtual_preview
from .nav2_goal_bridge_core import goal_signature_changed
from .nav2_goal_bridge_core import distance_bounded_rolling_horizon_count
from .nav2_goal_bridge_core import evaluate_waypoint_passage
from .nav2_goal_bridge_core import odom_route_from_enu_points
from .nav2_goal_bridge_core import odom_target_from_goal_vector
from .nav2_goal_bridge_core import rolling_horizon_global_remaining
from .nav2_goal_bridge_core import rolling_real_poses_remaining
from .nav2_goal_bridge_core import rolling_horizon_window
from .nav2_goal_bridge_core import rolling_initial_start_index
from .nav2_goal_bridge_core import rolling_waypoint_capture_radius
from .nav2_goal_bridge_core import rolling_waypoint_turn_angle_deg
from .nav2_goal_bridge_core import should_focus_rolling_waypoint
from .nav2_goal_bridge_core import should_promote_near_goal_route_abort
from .nav2_goal_bridge_core import should_retry_aborted_rolling_segment
from .nav2_goal_bridge_core import should_retry_rolling_preview_as_current_only
from .navigation_core import GeodeticPoint, geodetic_to_enu
from .waypoint_route_core import parse_waypoint_route_json


class Nav2GoalBridgeNode(Node):
    """Send single goals or sequential Hybrid mission segments to Nav2."""

    def __init__(self):
        super().__init__('nav2_goal_bridge_node')
        self.declare_parameter('goal_local_topic', '/navigation/goal_local')
        self.declare_parameter('goal_vector_topic', '/navigation/goal_vector')
        self.declare_parameter('odometry_topic', '/vehicle/odometry')
        self.declare_parameter('status_topic', '/navigation/nav2_status')
        self.declare_parameter(
            'route_remaining_topic',
            '/navigation/nav2_route_poses_remaining',
        )
        self.declare_parameter('goal_pose_odom_topic', '/navigation/nav2_goal')
        self.declare_parameter('route_topic', '/navigation/waypoint_route')
        self.declare_parameter('origin_topic', '/navigation/origin_gnss')
        self.declare_parameter(
            'current_local_topic', '/navigation/current_local'
        )
        self.declare_parameter('action_name', '/navigate_to_pose')
        self.declare_parameter(
            'route_action_name', '/navigate_through_poses'
        )
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('goal_change_tolerance_m', 0.25)
        self.declare_parameter('input_timeout_s', 1.0)
        self.declare_parameter('safety_state_topic', '/safety/state')
        self.declare_parameter('route_abort_success_radius_m', 1.5)
        self.declare_parameter('goal_rejection_retry_s', 1.0)
        self.declare_parameter('rolling_segment_abort_retry_s', 1.0)
        self.declare_parameter('rolling_horizon_window_size', 2)
        self.declare_parameter('rolling_horizon_max_distance_m', 45.0)
        self.declare_parameter('rolling_waypoint_capture_radius_m', 1.0)
        self.declare_parameter('rolling_corner_capture_radius_m', 1.35)
        self.declare_parameter(
            'rolling_waypoint_crossing_lateral_limit_m', 1.0
        )
        self.declare_parameter('rolling_waypoint_crossing_arm_m', 1.0)
        self.declare_parameter('rolling_waypoint_focus_distance_m', 6.0)
        self.declare_parameter('rolling_waypoint_focus_max_turn_deg', 15.0)
        self.declare_parameter('rolling_virtual_preview_distance_m', 10.0)
        self.declare_parameter(
            'rolling_virtual_preview_boundary_margin_m', 2.0
        )
        self.declare_parameter(
            'rolling_virtual_preview_min_distance_m', 2.0
        )
        self.declare_parameter('rolling_behavior_tree', '')
        self.declare_parameter(
            'log_directory', '~/terrain_nav_data/logs/nav2_bridge'
        )

        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.goal_change_tolerance_m = float(
            self.get_parameter('goal_change_tolerance_m').value
        )
        self.input_timeout_s = float(
            self.get_parameter('input_timeout_s').value
        )
        self.route_abort_success_radius_m = float(
            self.get_parameter('route_abort_success_radius_m').value
        )
        self.goal_rejection_retry_s = float(
            self.get_parameter('goal_rejection_retry_s').value
        )
        self.rolling_segment_abort_retry_s = float(
            self.get_parameter('rolling_segment_abort_retry_s').value
        )
        self.rolling_horizon_window_size = int(
            self.get_parameter('rolling_horizon_window_size').value
        )
        self.rolling_horizon_max_distance_m = float(
            self.get_parameter('rolling_horizon_max_distance_m').value
        )
        self.rolling_waypoint_capture_radius_m = float(
            self.get_parameter('rolling_waypoint_capture_radius_m').value
        )
        self.rolling_corner_capture_radius_m = float(
            self.get_parameter('rolling_corner_capture_radius_m').value
        )
        self.rolling_waypoint_crossing_lateral_limit_m = float(
            self.get_parameter(
                'rolling_waypoint_crossing_lateral_limit_m'
            ).value
        )
        self.rolling_waypoint_crossing_arm_m = float(
            self.get_parameter('rolling_waypoint_crossing_arm_m').value
        )
        self.rolling_waypoint_focus_distance_m = float(
            self.get_parameter('rolling_waypoint_focus_distance_m').value
        )
        self.rolling_waypoint_focus_max_turn_deg = float(
            self.get_parameter('rolling_waypoint_focus_max_turn_deg').value
        )
        self.rolling_virtual_preview_distance_m = float(
            self.get_parameter('rolling_virtual_preview_distance_m').value
        )
        self.rolling_virtual_preview_boundary_margin_m = float(
            self.get_parameter(
                'rolling_virtual_preview_boundary_margin_m'
            ).value
        )
        self.rolling_virtual_preview_min_distance_m = float(
            self.get_parameter(
                'rolling_virtual_preview_min_distance_m'
            ).value
        )
        self.rolling_behavior_tree = str(
            self.get_parameter('rolling_behavior_tree').value
        )
        if self.rolling_behavior_tree:
            self.rolling_behavior_tree = str(
                Path(self.rolling_behavior_tree).expanduser()
            )
        if self.goal_change_tolerance_m <= 0.0:
            raise ValueError('goal_change_tolerance_m must be positive')
        if self.input_timeout_s <= 0.0:
            raise ValueError('input_timeout_s must be positive')
        if self.route_abort_success_radius_m <= 0.0:
            raise ValueError('route_abort_success_radius_m must be positive')
        if self.goal_rejection_retry_s <= 0.0:
            raise ValueError('goal_rejection_retry_s must be positive')
        if self.rolling_segment_abort_retry_s <= 0.0:
            raise ValueError(
                'rolling_segment_abort_retry_s must be positive'
            )
        if self.rolling_horizon_window_size < 2:
            raise ValueError(
                'rolling_horizon_window_size must be at least 2'
            )
        if self.rolling_horizon_max_distance_m <= 0.0:
            raise ValueError(
                'rolling_horizon_max_distance_m must be positive'
            )
        if self.rolling_waypoint_capture_radius_m <= 0.0:
            raise ValueError(
                'rolling_waypoint_capture_radius_m must be positive'
            )
        if (
            self.rolling_corner_capture_radius_m
            < self.rolling_waypoint_capture_radius_m
        ):
            raise ValueError(
                'rolling corner capture radius must be at least normal radius'
            )
        if self.rolling_waypoint_crossing_lateral_limit_m <= 0.0:
            raise ValueError(
                'rolling waypoint crossing lateral limit must be positive'
            )
        if self.rolling_waypoint_crossing_arm_m <= 0.0:
            raise ValueError(
                'rolling_waypoint_crossing_arm_m must be positive'
            )
        if (
            self.rolling_waypoint_focus_distance_m
            <= self.rolling_waypoint_capture_radius_m
        ):
            raise ValueError(
                'rolling waypoint focus distance must exceed capture radius'
            )
        if not 0.0 <= self.rolling_waypoint_focus_max_turn_deg <= 180.0:
            raise ValueError('rolling focus maximum turn must be 0-180 deg')
        if self.rolling_virtual_preview_distance_m <= 0.0:
            raise ValueError(
                'rolling virtual preview distance must be positive'
            )
        if self.rolling_virtual_preview_boundary_margin_m < 0.0:
            raise ValueError(
                'rolling virtual preview margin cannot be negative'
            )
        if self.rolling_virtual_preview_min_distance_m <= 0.0:
            raise ValueError(
                'rolling virtual preview minimum must be positive'
            )

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            latched_qos,
        )
        self.route_remaining_publisher = self.create_publisher(
            UInt32,
            str(self.get_parameter('route_remaining_topic').value),
            10,
        )
        self.goal_pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter('goal_pose_odom_topic').value),
            latched_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('goal_local_topic').value),
            self._on_goal_local,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('safety_state_topic').value),
            self._on_safety_state,
            10,
        )
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter('goal_vector_topic').value),
            self._on_goal_vector,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self._on_odometry,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('route_topic').value),
            self._on_waypoint_route,
            latched_qos,
        )
        self.create_subscription(
            NavSatFix,
            str(self.get_parameter('origin_topic').value),
            self._on_origin,
            latched_qos,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('current_local_topic').value),
            self._on_current_local,
            10,
        )
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter('action_name').value),
        )
        self.route_action_client = ActionClient(
            self,
            NavigateThroughPoses,
            str(self.get_parameter('route_action_name').value),
        )
        self.latest_odometry = None
        self.latest_odometry_wall_time = None
        self.latest_goal_vector = None
        self.latest_goal_vector_wall_time = None
        self.last_route_poses_remaining = None
        self.route_feedback_generation = 0
        self.latest_safety_state = 'unknown'
        self.latest_origin = None
        self.latest_current_local = None
        self.latest_route = None
        self.route_mode = False
        self.route_rolling_horizon = False
        self.rolling_route_poses = None
        self.rolling_route_enu = None
        self.rolling_route_total_count = 0
        self.rolling_route_start_xy = None
        self.rolling_crossing_armed = set()
        self.rolling_geometry_passed = set()
        self.rolling_focused_waypoints = set()
        self.rolling_preview_preserved_waypoints = set()
        self.pending_route_poses = None
        self.pending_route_signature = None
        self.pending_route_count = 0
        self.pending_route_real_count = 0
        self.pending_route_final_xy = None
        self.pending_route_window_start = None
        self.pending_route_total_count = 0
        self.current_goal_signature = None
        self.sent_goal_signature = None
        self.pending_pose = None
        self.pending_signature = None
        self.goal_handle = None
        self.goal_request_in_flight = False
        self.cancel_in_flight = False
        self.active_goal_signature = None
        self.active_goal_mode = None
        self.active_route_count = 0
        self.active_route_real_count = 0
        self.active_route_final_xy = None
        self.active_route_window_start = None
        self.active_route_total_count = 0
        self.next_goal_send_wall_time = 0.0
        self.goal_sequence = 0
        self.event_logger = self._create_event_logger(
            str(self.get_parameter('log_directory').value)
        )
        self.create_timer(0.2, self._try_send_pending)
        self._publish_status('waiting_for_goal')
        self.get_logger().info(
            (
                'Nav2 goal bridge ready: GNSS local/vector -> {} '
                'and complete routes -> {} in frame {}'
            ).format(
                self.get_parameter('action_name').value,
                self.get_parameter('route_action_name').value,
                self.odom_frame,
            )
        )

    def _create_event_logger(self, directory):
        root = Path(directory).expanduser()
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        run_directory = root / ('run_' + stamp)
        run_directory.mkdir(parents=True, exist_ok=True)
        path = run_directory / 'nav2_goal_bridge.csv'
        stream = path.open('w', newline='', encoding='utf-8')
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                'wall_time_iso',
                'event',
                'goal_mode',
                'goal_sequence',
                'route_pose_count',
                'goal_east_m',
                'goal_north_m',
                'active_goal_east_m',
                'active_goal_north_m',
                'pending_goal_east_m',
                'pending_goal_north_m',
                'nav2_result_status',
            ],
        )
        writer.writeheader()
        stream.flush()
        self.get_logger().info('Nav2 bridge CSV: {}'.format(path))
        return (stream, writer)

    @staticmethod
    def _signature_values(signature):
        if signature is None:
            return ('', '')
        return ('{:.6f}'.format(signature[0]), '{:.6f}'.format(signature[1]))

    def _log_event(
        self,
        event,
        signature=None,
        result_status='',
        goal_mode=None,
        route_pose_count=None,
    ):
        stream, writer = self.event_logger
        goal_east, goal_north = self._signature_values(signature)
        active_east, active_north = self._signature_values(
            self.active_goal_signature
        )
        pending_east, pending_north = self._signature_values(
            self.pending_signature
        )
        writer.writerow({
            'wall_time_iso': datetime.now().astimezone().isoformat(
                timespec='milliseconds'
            ),
            'event': event,
            'goal_mode': goal_mode or self.active_goal_mode or (
                'route' if self.pending_route_poses is not None else 'single'
            ),
            'goal_sequence': self.goal_sequence,
            'route_pose_count': (
                self.active_route_count
                if route_pose_count is None else route_pose_count
            ),
            'goal_east_m': goal_east,
            'goal_north_m': goal_north,
            'active_goal_east_m': active_east,
            'active_goal_north_m': active_north,
            'pending_goal_east_m': pending_east,
            'pending_goal_north_m': pending_north,
            'nav2_result_status': result_status,
        })
        stream.flush()

    def _publish_status(self, status):
        message = String()
        message.data = status
        self.status_publisher.publish(message)

    def _on_odometry(self, message):
        self.latest_odometry = message
        self.latest_odometry_wall_time = time.monotonic()
        self._prepare_pending_route()
        self._check_rolling_waypoint_passage()

    def _rolling_approach_segment(self, waypoint_index):
        if self.rolling_route_poses is None:
            return None
        if waypoint_index < 0 or waypoint_index >= len(
            self.rolling_route_poses
        ):
            return None
        target = self.rolling_route_poses[waypoint_index].pose.position
        if waypoint_index == 0:
            previous_xy = self.rolling_route_start_xy
        else:
            previous = self.rolling_route_poses[
                waypoint_index - 1
            ].pose.position
            previous_xy = (float(previous.x), float(previous.y))
        if previous_xy is None:
            return None
        return (
            previous_xy,
            (float(target.x), float(target.y)),
        )

    def _publish_rolling_remaining(self, remaining):
        message = UInt32()
        message.data = int(remaining)
        self.route_remaining_publisher.publish(message)

    def _check_rolling_waypoint_passage(self):
        """Advance a rolling F9/F10 route after capture or gate crossing."""
        if (
            not self.route_rolling_horizon
            or self.latest_odometry is None
            or self.active_route_window_start is None
            or self.rolling_route_poses is None
            or self.pending_route_poses is not None
            or self.cancel_in_flight
        ):
            return
        waypoint_index = int(self.active_route_window_start)
        if waypoint_index in self.rolling_geometry_passed:
            return
        segment = self._rolling_approach_segment(waypoint_index)
        if segment is None:
            return
        current = self.latest_odometry.pose.pose.position
        current_xy = (float(current.x), float(current.y))
        turn_angle = self._rolling_turn_angle_deg(waypoint_index)
        capture_radius = rolling_waypoint_capture_radius(
            turn_angle,
            self.rolling_waypoint_capture_radius_m,
            self.rolling_corner_capture_radius_m,
            self.rolling_waypoint_focus_max_turn_deg,
        )
        passage = evaluate_waypoint_passage(
            segment[0],
            segment[1],
            current_xy,
            capture_radius,
            self.rolling_waypoint_crossing_lateral_limit_m,
            waypoint_index in self.rolling_crossing_armed,
        )
        if passage.along_track_m <= -self.rolling_waypoint_crossing_arm_m:
            self.rolling_crossing_armed.add(waypoint_index)
        if not passage.passed:
            return

        self.rolling_geometry_passed.add(waypoint_index)
        remaining = max(
            0, self.rolling_route_total_count - waypoint_index - 1
        )
        self._publish_rolling_remaining(remaining)
        passage_reason = passage.reason
        if (
            passage.reason == 'radius'
            and capture_radius > self.rolling_waypoint_capture_radius_m
        ):
            passage_reason = 'corner_radius'
        self._log_event(
            'rolling_waypoint_passed_' + passage_reason,
            self.rolling_route_enu[waypoint_index],
            goal_mode='rolling_route',
            route_pose_count=self.active_route_count,
        )
        self.get_logger().info(
            'Rolling waypoint {} passed by {}: distance={:.2f} m, '
            'cross-track={:.2f} m; {} pose(s) remaining'.format(
                waypoint_index + 1,
                passage_reason,
                passage.distance_m,
                passage.cross_track_m,
                remaining,
            )
        )

        next_start = waypoint_index + 1
        if next_start < self.rolling_route_total_count:
            if self._queue_rolling_segment(next_start):
                self._log_event(
                    'rolling_segment_handoff_queued',
                    self.rolling_route_enu[next_start],
                    goal_mode='rolling_segment',
                    route_pose_count=1,
                )
                self.get_logger().info(
                    'Waypoint {} passed; canceling its Hybrid segment and '
                    'planning waypoint {} as a new segment'.format(
                        waypoint_index + 1,
                        next_start + 1,
                    )
                )
                self._publish_status('rolling_segment_handoff')
                self._try_send_pending()
                return

        # The last segment remains active until Nav2's position-only goal
        # checker reports success. Its yaw tolerance is pi, so no parking
        # orientation is imposed at the mission endpoint.
        self._log_event(
            'rolling_final_waypoint_captured',
            self.rolling_route_enu[waypoint_index],
            goal_mode='rolling_segment',
            route_pose_count=1,
        )
        self._publish_status('rolling_final_waypoint_captured')

    def _rolling_window_count(self, start_index, maximum_window_size=None):
        if self.rolling_route_poses is None or self.latest_odometry is None:
            return 0
        if maximum_window_size is None:
            maximum_window_size = self.rolling_horizon_window_size
        current = self.latest_odometry.pose.pose.position
        points_xy = [
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
            )
            for pose in self.rolling_route_poses
        ]
        return distance_bounded_rolling_horizon_count(
            points_xy,
            start_index,
            (float(current.x), float(current.y)),
            int(maximum_window_size),
            self.rolling_horizon_max_distance_m,
        )

    def _maybe_focus_current_rolling_waypoint(self):
        """Focus only nearly straight WPs; preserve corner preview tangents."""
        if (
            not self.route_rolling_horizon
            or self.latest_odometry is None
            or self.active_route_window_start is None
            or self.rolling_route_poses is None
            or self.pending_route_poses is not None
            or self.goal_request_in_flight
            or self.cancel_in_flight
            or self.goal_handle is not None
        ):
            return
        start_index = int(self.active_route_window_start)
        if start_index in self.rolling_geometry_passed:
            return
        target = self.rolling_route_poses[start_index].pose.position
        current = self.latest_odometry.pose.pose.position
        distance = math.hypot(
            float(target.x) - float(current.x),
            float(target.y) - float(current.y),
        )
        if not should_focus_rolling_waypoint(
            distance,
            self.active_route_count,
            start_index in self.rolling_focused_waypoints,
            self.rolling_waypoint_focus_distance_m,
        ):
            return
        turn_angle = self._rolling_turn_angle_deg(start_index)
        if (
            turn_angle is not None
            and turn_angle > self.rolling_waypoint_focus_max_turn_deg
        ):
            if start_index not in self.rolling_preview_preserved_waypoints:
                self.rolling_preview_preserved_waypoints.add(start_index)
                self._log_event(
                    'rolling_corner_preview_preserved',
                    self.rolling_route_enu[start_index],
                    goal_mode='rolling_route',
                    route_pose_count=self.active_route_count,
                )
                self.get_logger().info(
                    'Rolling waypoint {} turns {:.1f} deg; preserving its '
                    'successor preview instead of focusing to one pose'.format(
                        start_index + 1,
                        turn_angle,
                    )
                )
            return
        self.rolling_focused_waypoints.add(start_index)
        if not self._queue_rolling_window(
            start_index,
            maximum_window_size=1,
        ):
            self.rolling_focused_waypoints.discard(start_index)
            return
        self._log_event(
            'rolling_current_waypoint_focus',
            self.rolling_route_enu[start_index],
            goal_mode='rolling_route',
            route_pose_count=1,
        )
        self.get_logger().info(
            'Rolling waypoint {} is {:.2f} m away; replacing the preview '
            'window with a current-waypoint-only action'.format(
                start_index + 1,
                distance,
            )
        )
        self._publish_status('rolling_current_waypoint_focus')
        self._try_send_pending()

    def _rolling_turn_angle_deg(self, waypoint_index):
        """Return the real mission-route turn at one rolling waypoint."""
        if (
            self.rolling_route_poses is None
            or waypoint_index < 0
            or waypoint_index + 1 >= len(self.rolling_route_poses)
        ):
            return None
        segment = self._rolling_approach_segment(waypoint_index)
        if segment is None:
            return None
        following = self.rolling_route_poses[
            waypoint_index + 1
        ].pose.position
        return rolling_waypoint_turn_angle_deg(
            segment[0],
            segment[1],
            (float(following.x), float(following.y)),
        )

    def _maybe_extend_rolling_window(self):
        """Add a preview pose once it enters the rolling costmap horizon."""
        if (
            not self.route_rolling_horizon
            or self.latest_odometry is None
            or self.active_route_window_start is None
            or self.rolling_route_poses is None
            or self.pending_route_poses is not None
            or self.goal_request_in_flight
            or self.cancel_in_flight
            or self.goal_handle is not None
        ):
            return
        start_index = int(self.active_route_window_start)
        if start_index in self.rolling_focused_waypoints:
            return
        desired_count = self._rolling_window_count(start_index)
        if desired_count <= self.active_route_count:
            return
        if not self._queue_rolling_window(start_index):
            return
        final_index = start_index + desired_count - 1
        self._log_event(
            'rolling_preview_added_in_range',
            self.rolling_route_enu[final_index],
            goal_mode='rolling_route',
            route_pose_count=desired_count,
        )
        self.get_logger().info(
            'Rolling preview pose {} entered the {:.1f} m planning horizon; '
            'expanding active window to {} pose(s)'.format(
                final_index + 1,
                self.rolling_horizon_max_distance_m,
                desired_count,
            )
        )
        self._publish_status('rolling_horizon_preview_added')
        self._try_send_pending()

    def _on_origin(self, message):
        values = (
            float(message.latitude),
            float(message.longitude),
            float(message.altitude),
        )
        if not all(math.isfinite(value) for value in values):
            return
        self.latest_origin = GeodeticPoint(*values)
        self._prepare_pending_route()

    def _on_current_local(self, message):
        values = (float(message.point.x), float(message.point.y))
        if not all(math.isfinite(value) for value in values):
            return
        self.latest_current_local = values
        self._prepare_pending_route()

    def _on_safety_state(self, message):
        self.latest_safety_state = str(message.data)

    def _on_waypoint_route(self, message):
        try:
            route = parse_waypoint_route_json(message.data)
        except ValueError as error:
            self.get_logger().error(
                'Nav2 bridge rejected waypoint route: {}'.format(error)
            )
            return
        if not route.start:
            return
        if route.rolling_horizon and len(route.waypoints) < 2:
            self.get_logger().error(
                'Rolling route requires at least 2 navigation targets'
            )
            self._publish_status('invalid_rolling_route')
            return
        if route.rolling_horizon and not self.rolling_behavior_tree:
            self.get_logger().error(
                'Rolling route requires rolling_behavior_tree'
            )
            self._publish_status('invalid_rolling_route')
            return
        # F9 and F10 both use the saved WP1 -> ... -> WPn prefix. F10 adds one
        # final WP1 target. Each target is sent as its own Hybrid
        # NavigateToPose action, so a passed waypoint can never remain inside
        # the action that controls the following segment.
        self.latest_route = route
        self.route_mode = True
        self.route_rolling_horizon = bool(route.rolling_horizon)
        self.rolling_route_poses = None
        self.rolling_route_enu = None
        self.rolling_route_total_count = 0
        self.rolling_route_start_xy = None
        self.rolling_crossing_armed = set()
        self.rolling_geometry_passed = set()
        self.rolling_focused_waypoints = set()
        self.rolling_preview_preserved_waypoints = set()
        self.pending_pose = None
        self.pending_signature = None
        self.sent_goal_signature = None
        self.pending_route_poses = None
        self.pending_route_signature = None
        self.pending_route_count = 0
        self.pending_route_real_count = 0
        self.pending_route_final_xy = None
        self.pending_route_window_start = None
        self.pending_route_total_count = 0
        self._prepare_pending_route()

    def _prepare_pending_route(self):
        if (
            not self.route_mode
            or self.latest_route is None
            or self.latest_origin is None
            or self.latest_current_local is None
            or self.latest_odometry is None
        ):
            return
        route = self.latest_route
        waypoint_enu = [
            geodetic_to_enu(point, self.latest_origin)
            for point in route.waypoints
        ]
        odometry = self.latest_odometry.pose.pose.position
        route_poses = odom_route_from_enu_points(
            float(odometry.x),
            float(odometry.y),
            self.latest_current_local[0],
            self.latest_current_local[1],
            [(point.east_m, point.north_m) for point in waypoint_enu],
            # Intermediate pose yaws bisect their incoming and outgoing route
            # segments.  Smac Hybrid can then connect the waypoint and its
            # preview with one continuous Ackermann curve instead of being
            # asked to change heading instantaneously at the waypoint.
            use_closed_route_tangents=False,
            use_corner_tangents=True,
        )
        poses = []
        for target in route_poses:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.odom_frame
            pose.pose.position.x = target.x_m
            pose.pose.position.y = target.y_m
            pose.pose.position.z = float(odometry.z)
            pose.pose.orientation.z = math.sin(0.5 * target.yaw_rad)
            pose.pose.orientation.w = math.cos(0.5 * target.yaw_rad)
            poses.append(pose)
        if route.rolling_horizon:
            self.rolling_route_poses = poses
            self.rolling_route_enu = [
                (point.east_m, point.north_m) for point in waypoint_enu
            ]
            self.rolling_route_total_count = len(poses)
            self.rolling_route_start_xy = (
                float(odometry.x), float(odometry.y)
            )
            points_xy = [
                (
                    float(pose.pose.position.x),
                    float(pose.pose.position.y),
                )
                for pose in poses
            ]
            start_index = rolling_initial_start_index(
                points_xy,
                self.rolling_route_start_xy,
                self.rolling_waypoint_capture_radius_m,
            )
            if start_index == 1:
                self.rolling_geometry_passed.add(0)
                remaining = self.rolling_route_total_count - 1
                self._publish_rolling_remaining(remaining)
                self._log_event(
                    'rolling_initial_waypoint_skipped',
                    self.rolling_route_enu[0],
                    goal_mode='rolling_route',
                    route_pose_count=0,
                )
                self.get_logger().info(
                    'Initial WP1 is already inside the {:.2f} m capture '
                    'radius; preserving it as WP2 approach geometry and '
                    'starting motion at waypoint 2'.format(
                        self.rolling_waypoint_capture_radius_m,
                    )
                )
            self._queue_rolling_segment(start_index)
        else:
            final_enu = waypoint_enu[-1]
            signature = (final_enu.east_m, final_enu.north_m)
            self.pending_route_poses = poses
            self.pending_route_signature = signature
            self.pending_route_count = len(poses)
            self.pending_route_real_count = len(poses)
            self.pending_route_final_xy = (
                float(poses[-1].pose.position.x),
                float(poses[-1].pose.position.y),
            )
            self.pending_route_window_start = None
            self.pending_route_total_count = len(poses)
            self.goal_pose_publisher.publish(poses[-1])
            self._log_event(
                'route_queued',
                signature,
                goal_mode='route',
                route_pose_count=len(poses),
            )
        self.latest_route = None
        if self.goal_handle is not None:
            self._publish_status('canceling_previous_goal')
        else:
            self._publish_status('waiting_for_nav2_route')
        self._try_send_pending()

    def _queue_rolling_segment(self, waypoint_index):
        """Queue exactly one mission waypoint for Hybrid NavigateToPose."""
        if self.rolling_route_poses is None or self.rolling_route_enu is None:
            raise RuntimeError('rolling route is not prepared')
        if self.latest_odometry is None:
            return False
        if not 0 <= waypoint_index < self.rolling_route_total_count:
            return False
        target_pose = self.rolling_route_poses[waypoint_index]
        target = target_pose.pose.position
        current = self.latest_odometry.pose.pose.position
        distance = math.hypot(
            float(target.x) - float(current.x),
            float(target.y) - float(current.y),
        )
        if distance > self.rolling_horizon_max_distance_m:
            self._publish_status('rolling_target_out_of_range')
            self._log_event(
                'rolling_target_out_of_range',
                self.rolling_route_enu[waypoint_index],
                goal_mode='rolling_segment',
                route_pose_count=0,
            )
            self.get_logger().error(
                'Rolling waypoint {} is {:.1f} m away, outside the {:.1f} m '
                'Hybrid planning horizon; add a closer mission waypoint'.format(
                    waypoint_index + 1,
                    distance,
                    self.rolling_horizon_max_distance_m,
                )
            )
            return False
        signature = self.rolling_route_enu[waypoint_index]
        self.pending_route_poses = [target_pose]
        self.pending_route_signature = signature
        self.pending_route_count = 1
        self.pending_route_real_count = 1
        self.pending_route_final_xy = (
            float(target_pose.pose.position.x),
            float(target_pose.pose.position.y),
        )
        self.pending_route_window_start = waypoint_index
        self.pending_route_total_count = self.rolling_route_total_count
        self.goal_pose_publisher.publish(target_pose)
        self._log_event(
            'rolling_segment_queued',
            signature,
            goal_mode='rolling_segment',
            route_pose_count=1,
        )
        self.get_logger().info(
            'Hybrid rolling segment queued: waypoint {} of {}, distance '
            '{:.1f} m'.format(
                waypoint_index + 1,
                self.rolling_route_total_count,
                distance,
            )
        )
        return True

    def _queue_rolling_window(self, start_index, maximum_window_size=None):
        if self.rolling_route_poses is None or self.rolling_route_enu is None:
            raise RuntimeError('rolling route is not prepared')
        if maximum_window_size is None:
            maximum_window_size = self.rolling_horizon_window_size
        window_count = self._rolling_window_count(
            start_index,
            maximum_window_size=maximum_window_size,
        )
        if window_count < 1:
            target = self.rolling_route_poses[start_index].pose.position
            current = self.latest_odometry.pose.pose.position
            distance = math.hypot(
                float(target.x) - float(current.x),
                float(target.y) - float(current.y),
            )
            self._publish_status('rolling_target_out_of_range')
            self._log_event(
                'rolling_target_out_of_range',
                self.rolling_route_enu[start_index],
                goal_mode='rolling_route',
                route_pose_count=0,
            )
            self.get_logger().error(
                'Rolling target pose {} is {:.1f} m away, outside the {:.1f} '
                'm planning horizon; add a closer waypoint'.format(
                    start_index + 1,
                    distance,
                    self.rolling_horizon_max_distance_m,
                )
            )
            return False
        poses = rolling_horizon_window(
            self.rolling_route_poses,
            start_index,
            window_count,
        )
        actual_pose_count = len(poses)
        end_index = start_index + actual_pose_count - 1
        final_enu = self.rolling_route_enu[end_index]
        actual_final_pose = poses[-1]
        next_index = end_index + 1
        virtual_preview_xy = None
        if (
            maximum_window_size >= self.rolling_horizon_window_size
            and actual_pose_count < self.rolling_horizon_window_size
            and next_index < self.rolling_route_total_count
        ):
            current = self.latest_odometry.pose.pose.position
            waypoint = actual_final_pose.pose.position
            following = self.rolling_route_poses[next_index].pose.position
            virtual_preview_xy = bounded_virtual_preview(
                (float(current.x), float(current.y)),
                (float(waypoint.x), float(waypoint.y)),
                (float(following.x), float(following.y)),
                self.rolling_virtual_preview_distance_m,
                self.rolling_horizon_max_distance_m,
                self.rolling_virtual_preview_boundary_margin_m,
                self.rolling_virtual_preview_min_distance_m,
            )
            if virtual_preview_xy is not None:
                yaw = math.atan2(
                    float(following.y) - float(waypoint.y),
                    float(following.x) - float(waypoint.x),
                )
                virtual_pose = PoseStamped()
                virtual_pose.header.stamp = self.get_clock().now().to_msg()
                virtual_pose.header.frame_id = self.odom_frame
                virtual_pose.pose.position.x = virtual_preview_xy[0]
                virtual_pose.pose.position.y = virtual_preview_xy[1]
                virtual_pose.pose.position.z = float(waypoint.z)
                virtual_pose.pose.orientation.z = math.sin(0.5 * yaw)
                virtual_pose.pose.orientation.w = math.cos(0.5 * yaw)
                poses.append(virtual_pose)
        self.pending_route_poses = poses
        self.pending_route_signature = final_enu
        self.pending_route_count = len(poses)
        self.pending_route_real_count = actual_pose_count
        self.pending_route_final_xy = (
            float(actual_final_pose.pose.position.x),
            float(actual_final_pose.pose.position.y),
        )
        self.pending_route_window_start = start_index
        self.pending_route_total_count = self.rolling_route_total_count
        self.goal_pose_publisher.publish(poses[-1])
        self._log_event(
            'rolling_window_queued',
            final_enu,
            goal_mode='rolling_route',
            route_pose_count=len(poses),
        )
        self.get_logger().info(
            'Nav2 rolling route window queued: poses {}-{} of {} '
            '(distance horizon {:.1f} m)'.format(
                start_index + 1,
                end_index + 1,
                self.rolling_route_total_count,
                self.rolling_horizon_max_distance_m,
            )
        )
        if (
            maximum_window_size >= self.rolling_horizon_window_size
            and actual_pose_count < self.rolling_horizon_window_size
            and next_index < self.rolling_route_total_count
        ):
            next_pose = self.rolling_route_poses[next_index].pose.position
            current = self.latest_odometry.pose.pose.position
            next_distance = math.hypot(
                float(next_pose.x) - float(current.x),
                float(next_pose.y) - float(current.y),
            )
            self._log_event(
                'rolling_preview_deferred_out_of_range',
                self.rolling_route_enu[next_index],
                goal_mode='rolling_route',
                route_pose_count=len(poses),
            )
            self.get_logger().info(
                'Deferring rolling preview pose {}: {:.1f} m exceeds the '
                '{:.1f} m planning horizon'.format(
                    next_index + 1,
                    next_distance,
                    self.rolling_horizon_max_distance_m,
                )
            )
            if virtual_preview_xy is not None:
                self._log_event(
                    'rolling_virtual_preview_added',
                    final_enu,
                    goal_mode='rolling_route',
                    route_pose_count=len(poses),
                )
                self.get_logger().info(
                    'Added a virtual {:.1f} m direction preview after '
                    'waypoint {} toward out-of-range waypoint {}'.format(
                        math.hypot(
                            virtual_preview_xy[0] - float(
                                actual_final_pose.pose.position.x
                            ),
                            virtual_preview_xy[1] - float(
                                actual_final_pose.pose.position.y
                            ),
                        ),
                        end_index + 1,
                        next_index + 1,
                    )
                )
        return True

    def _on_goal_local(self, message):
        signature = (float(message.point.x), float(message.point.y))
        if not all(math.isfinite(value) for value in signature):
            return
        self.current_goal_signature = signature

    def _on_goal_vector(self, message):
        vector = (float(message.vector.x), float(message.vector.y))
        if not all(math.isfinite(value) for value in vector):
            return
        self.latest_goal_vector = vector
        self.latest_goal_vector_wall_time = time.monotonic()
        self._prepare_pending_goal()

    def _inputs_fresh(self):
        now = time.monotonic()
        return bool(
            self.latest_odometry is not None
            and self.latest_goal_vector is not None
            and self.current_goal_signature is not None
            and self.latest_odometry_wall_time is not None
            and self.latest_goal_vector_wall_time is not None
            and now - self.latest_odometry_wall_time <= self.input_timeout_s
            and now - self.latest_goal_vector_wall_time <= self.input_timeout_s
        )

    def _prepare_pending_goal(self):
        # Intermediate GNSS goals continue to drive arrival/status reporting,
        # but must not replace the previewable NavigateThroughPoses action.
        if self.route_mode:
            return
        if not self._inputs_fresh():
            return
        if not goal_signature_changed(
            self.sent_goal_signature,
            self.current_goal_signature,
            self.goal_change_tolerance_m,
        ):
            return
        odometry = self.latest_odometry.pose.pose.position
        vector = self.latest_goal_vector
        target = odom_target_from_goal_vector(
            float(odometry.x),
            float(odometry.y),
            vector[0],
            vector[1],
        )
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.odom_frame
        pose.pose.position.x = target.x_m
        pose.pose.position.y = target.y_m
        pose.pose.position.z = float(odometry.z)
        pose.pose.orientation.z = math.sin(0.5 * target.yaw_rad)
        pose.pose.orientation.w = math.cos(0.5 * target.yaw_rad)
        self.pending_pose = pose
        self.pending_signature = self.current_goal_signature
        self.goal_pose_publisher.publish(pose)
        self._log_event('goal_queued', self.pending_signature)
        if self.goal_handle is not None:
            self._publish_status('canceling_previous_goal')
        else:
            self._publish_status('waiting_for_nav2')
        self._try_send_pending()

    def _try_send_pending(self):
        if time.monotonic() < self.next_goal_send_wall_time:
            return
        has_route = self.pending_route_poses is not None
        if not has_route and self.pending_pose is None:
            return
        if self.goal_request_in_flight or self.cancel_in_flight:
            return
        if self.goal_handle is not None:
            self.cancel_in_flight = True
            signature = self.active_goal_signature
            self._publish_status('canceling_previous_goal')
            self._log_event('cancel_requested', signature)
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(
                lambda completed, canceled_signature=signature:
                self._cancel_response(completed, canceled_signature)
            )
            return
        is_rolling_segment = bool(
            has_route
            and self.route_rolling_horizon
            and self.pending_route_window_start is not None
        )
        if is_rolling_segment:
            if not self.action_client.server_is_ready():
                return
            action_goal = NavigateToPose.Goal()
            action_goal.pose = self.pending_route_poses[0]
            action_goal.behavior_tree = self.rolling_behavior_tree
            sent_route_poses = list(self.pending_route_poses)
            signature = self.pending_route_signature
            route_count = self.pending_route_count
            route_real_count = self.pending_route_real_count
            route_final_xy = self.pending_route_final_xy
            route_window_start = self.pending_route_window_start
            route_total_count = self.pending_route_total_count
            self.goal_request_in_flight = True
            self.last_route_poses_remaining = None
            future = self.action_client.send_goal_async(action_goal)
            future.add_done_callback(
                lambda completed,
                sent_signature=signature,
                sent_count=route_count,
                sent_real_count=route_real_count,
                sent_poses=sent_route_poses,
                sent_final_xy=route_final_xy,
                sent_window_start=route_window_start,
                sent_total_count=route_total_count: self._goal_response(
                    completed,
                    sent_signature,
                    goal_mode='rolling_segment',
                    route_pose_count=sent_count,
                    route_real_pose_count=sent_real_count,
                    route_poses=sent_poses,
                    route_final_xy=sent_final_xy,
                    route_window_start=sent_window_start,
                    route_total_count=sent_total_count,
                )
            )
            self.pending_route_poses = None
            self.pending_route_signature = None
            self.pending_route_count = 0
            self.pending_route_real_count = 0
            self.pending_route_final_xy = None
            self.pending_route_window_start = None
            self.pending_route_total_count = 0
            self.sent_goal_signature = signature
            self._publish_status('rolling_segment_sent')
            self._log_event(
                'rolling_segment_sent',
                signature,
                goal_mode='rolling_segment',
                route_pose_count=1,
            )
            self.get_logger().info(
                'Nav2 Hybrid segment sent: waypoint {} of {}'.format(
                    route_window_start + 1,
                    route_total_count,
                )
            )
            return
        if has_route:
            if not self.route_action_client.server_is_ready():
                return
            action_goal = NavigateThroughPoses.Goal()
            action_goal.poses = self.pending_route_poses
            sent_route_poses = list(self.pending_route_poses)
            signature = self.pending_route_signature
            route_count = self.pending_route_count
            route_real_count = self.pending_route_real_count
            route_final_xy = self.pending_route_final_xy
            route_window_start = self.pending_route_window_start
            route_total_count = self.pending_route_total_count
            if route_window_start is not None:
                action_goal.behavior_tree = self.rolling_behavior_tree
            self.goal_request_in_flight = True
            self.last_route_poses_remaining = None
            self.route_feedback_generation += 1
            feedback_generation = self.route_feedback_generation
            future = self.route_action_client.send_goal_async(
                action_goal,
                feedback_callback=(
                    lambda feedback,
                    generation=feedback_generation,
                    window_start=route_window_start,
                    action_length=route_count,
                    real_length=route_real_count,
                    total_count=route_total_count: self._route_feedback(
                        feedback,
                        generation,
                        window_start,
                        action_length,
                        real_length,
                        total_count,
                    )
                ),
            )
            future.add_done_callback(
                lambda completed,
                sent_signature=signature,
                sent_count=route_count,
                sent_real_count=route_real_count,
                sent_poses=sent_route_poses,
                sent_final_xy=route_final_xy,
                sent_window_start=route_window_start,
                sent_total_count=route_total_count: self._goal_response(
                    completed,
                    sent_signature,
                    goal_mode='route',
                    route_pose_count=sent_count,
                    route_real_pose_count=sent_real_count,
                    route_poses=sent_poses,
                    route_final_xy=sent_final_xy,
                    route_window_start=sent_window_start,
                    route_total_count=sent_total_count,
                )
            )
            self.pending_route_poses = None
            self.pending_route_signature = None
            self.pending_route_count = 0
            self.pending_route_real_count = 0
            self.pending_route_final_xy = None
            self.pending_route_window_start = None
            self.pending_route_total_count = 0
            self.sent_goal_signature = signature
            self._publish_status('route_sent')
            self._log_event(
                'route_sent',
                signature,
                goal_mode='route',
                route_pose_count=route_count,
            )
            self.get_logger().info(
                'Nav2 {} sent: {} pose(s)'.format(
                    'rolling route window'
                    if route_window_start is not None else 'complete route',
                    route_count,
                )
            )
            return
        if not self.action_client.server_is_ready():
            return
        action_goal = NavigateToPose.Goal()
        action_goal.pose = self.pending_pose
        signature = self.pending_signature
        self.goal_request_in_flight = True
        future = self.action_client.send_goal_async(action_goal)
        future.add_done_callback(
            lambda completed, sent_signature=signature: self._goal_response(
                completed,
                sent_signature,
                goal_mode='single',
                route_pose_count=0,
            )
        )
        self.pending_pose = None
        self.pending_signature = None
        self.sent_goal_signature = signature
        self._publish_status('goal_sent')
        self._log_event('goal_sent', signature)
        self.get_logger().info(
            'Nav2 goal sent: local ENU ({:.2f}, {:.2f})'.format(
                signature[0], signature[1]
            )
        )

    def _route_feedback(
        self,
        feedback_message,
        feedback_generation,
        route_window_start,
        route_action_length,
        route_real_length,
        route_total_count,
    ):
        # Feedback may arrive before send_goal_async's goal-response callback,
        # so active_route_* is not yet guaranteed to describe this action.
        # Use metadata captured when the request was sent and reject delayed
        # feedback from a superseded rolling window.
        if feedback_generation != self.route_feedback_generation:
            return
        remaining = int(feedback_message.feedback.number_of_poses_remaining)
        if remaining < 0 or remaining == self.last_route_poses_remaining:
            return
        self.last_route_poses_remaining = remaining
        published_remaining = remaining
        if route_window_start is not None:
            real_remaining = rolling_real_poses_remaining(
                route_action_length,
                route_real_length,
                remaining,
            )
            published_remaining = rolling_horizon_global_remaining(
                route_total_count,
                route_window_start,
                route_real_length,
                real_remaining,
            )
        message = UInt32()
        message.data = published_remaining
        self.route_remaining_publisher.publish(message)
        self.get_logger().info(
            'Nav2 route progress: {} pose(s) remaining{}'.format(
                published_remaining,
                ' (rolling window has {})'.format(remaining)
                if route_window_start is not None else '',
            )
        )
        # Feedback updates progress only.  The previous implementation
        # canceled a still-valid action as soon as its first pose was removed
        # and replanned from a costmap corner.  Window handoff now happens
        # only after the active action returns a result.

    def _cancel_response(self, future, signature):
        try:
            response = future.result()
            canceled_count = len(response.goals_canceling)
        except Exception as error:  # pragma: no cover - ROS transport failure
            self.cancel_in_flight = False
            self._publish_status('cancel_failed')
            self._log_event('cancel_failed', signature)
            self.get_logger().error(
                'Nav2 previous-goal cancellation failed: {}'.format(error)
            )
            return
        if canceled_count < 1:
            # The action may have completed while the cancel request was in
            # flight. Its result callback will clear the handle and retry.
            # Keep the handoff lock set so the timer cannot issue duplicate
            # cancellation requests in the meantime.
            self._log_event('cancel_not_accepted', signature)
            self.get_logger().warning(
                'Nav2 previous goal was not cancelable; waiting for result'
            )
            return
        self._publish_status('previous_goal_canceling')
        self._log_event('cancel_accepted', signature)
        self.get_logger().info(
            'Nav2 previous goal cancellation accepted before goal handoff'
        )

    def _goal_response(
        self,
        future,
        signature,
        goal_mode='single',
        route_pose_count=0,
        route_real_pose_count=0,
        route_poses=None,
        route_final_xy=None,
        route_window_start=None,
        route_total_count=0,
    ):
        self.goal_request_in_flight = False
        try:
            goal_handle = future.result()
        except Exception as error:  # pragma: no cover - ROS transport failure
            self.sent_goal_signature = None
            self._publish_status('send_failed')
            self._log_event(
                'send_failed',
                signature,
                goal_mode=goal_mode,
                route_pose_count=route_pose_count,
            )
            self.get_logger().error('Nav2 goal send failed: {}'.format(error))
            return
        if not goal_handle.accepted:
            self.sent_goal_signature = None
            if (
                goal_mode in ('route', 'rolling_segment')
                and route_poses
                and self.pending_route_poses is None
            ):
                # The action server can be discoverable just before Nav2's
                # lifecycle activation completes.  Preserve the one-shot UI
                # route and retry instead of requiring another F9 press.
                self.pending_route_poses = list(route_poses)
                self.pending_route_signature = signature
                self.pending_route_count = route_pose_count
                self.pending_route_real_count = route_real_pose_count
                self.pending_route_final_xy = route_final_xy
                self.pending_route_window_start = route_window_start
                self.pending_route_total_count = route_total_count
                self.next_goal_send_wall_time = (
                    time.monotonic() + self.goal_rejection_retry_s
                )
                self._publish_status('waiting_for_nav2_activation')
            else:
                self._publish_status('goal_rejected')
            self._log_event(
                'goal_rejected',
                signature,
                goal_mode=goal_mode,
                route_pose_count=route_pose_count,
            )
            self.get_logger().warning(
                'Nav2 rejected the navigation goal; retrying after '
                '{:.1f} s'.format(self.goal_rejection_retry_s)
                if goal_mode in ('route', 'rolling_segment') and route_poses
                else 'Nav2 rejected the navigation goal'
            )
            self._prepare_pending_goal()
            return
        self.next_goal_send_wall_time = 0.0
        self.goal_handle = goal_handle
        self.active_goal_signature = signature
        self.active_goal_mode = goal_mode
        self.active_route_count = route_pose_count
        self.active_route_real_count = route_real_pose_count
        self.active_route_final_xy = route_final_xy
        self.active_route_window_start = route_window_start
        self.active_route_total_count = route_total_count
        self.goal_sequence += 1
        self._publish_status('navigating')
        self._log_event(
            'route_accepted'
            if goal_mode in ('route', 'rolling_segment')
            else 'goal_accepted',
            signature,
            goal_mode=goal_mode,
            route_pose_count=route_pose_count,
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, completed_handle=goal_handle,
            completed_signature=signature,
            completed_mode=goal_mode,
            completed_count=route_pose_count,
            completed_real_count=route_real_pose_count,
            completed_window_start=route_window_start,
            completed_total_count=route_total_count: self._goal_result(
                completed,
                completed_handle,
                completed_signature,
                completed_mode,
                completed_count,
                completed_real_count,
                completed_window_start,
                completed_total_count,
            )
        )
        self._try_send_pending()

    def _goal_result(
        self,
        future,
        completed_handle,
        signature,
        goal_mode='single',
        route_pose_count=0,
        route_real_pose_count=0,
        route_window_start=None,
        route_total_count=0,
    ):
        try:
            wrapped_result = future.result()
            status = int(wrapped_result.status)
        except Exception as error:  # pragma: no cover - ROS transport failure
            self._publish_status('result_failed')
            self._log_event('result_failed', signature)
            self.get_logger().error('Nav2 result failed: {}'.format(error))
            return
        is_current = completed_handle is self.goal_handle
        odometry_age_s = None
        route_goal_vector = None
        if self.latest_odometry_wall_time is not None:
            odometry_age_s = time.monotonic() - self.latest_odometry_wall_time
        if (
            self.latest_odometry is not None
            and self.active_route_final_xy is not None
        ):
            current = self.latest_odometry.pose.pose.position
            route_goal_vector = (
                self.active_route_final_xy[0] - float(current.x),
                self.active_route_final_xy[1] - float(current.y),
            )
        safe_for_arrival = self.latest_safety_state in (
            'clear',
            'command_stale',
            'waiting_for_command',
        )
        if (
            is_current
            and goal_mode in ('route', 'rolling_segment')
            and safe_for_arrival
            and should_promote_near_goal_route_abort(
                status,
                self.last_route_poses_remaining,
                route_goal_vector,
                odometry_age_s,
                self.route_abort_success_radius_m,
                self.input_timeout_s,
            )
        ):
            self._log_event(
                'route_near_goal_abort_promoted',
                signature,
                status,
                goal_mode=goal_mode,
                route_pose_count=route_pose_count,
            )
            self.get_logger().warning(
                'Promoting Nav2 route abort to success: final pose is within '
                '{:.2f} m and safety state is {}'.format(
                    self.route_abort_success_radius_m,
                    self.latest_safety_state,
                )
            )
            status = 4
        if is_current:
            self.goal_handle = None
            self.active_goal_signature = None
            self.active_goal_mode = None
            self.active_route_count = 0
            self.active_route_real_count = 0
            self.active_route_final_xy = None
            self.active_route_window_start = None
            self.active_route_total_count = 0
            self.cancel_in_flight = False
        retry_waypoint_index = route_window_start
        if (
            route_window_start is not None
            and route_window_start in self.rolling_geometry_passed
            and route_window_start + 1 < route_total_count
        ):
            retry_waypoint_index = route_window_start + 1
        retry_current_only = bool(
            is_current
            and goal_mode == 'route'
            and self.route_rolling_horizon
            and self.rolling_route_poses is not None
            and route_window_start is not None
            and self.pending_route_poses is None
            and should_retry_rolling_preview_as_current_only(
                status,
                route_pose_count,
                route_window_start,
                retry_waypoint_index in self.rolling_focused_waypoints,
            )
        )
        if retry_current_only:
            # A preview pose can be outside the currently connected free
            # space even though the ordered current waypoint is reachable.
            # Do not let that future pose prevent all motion toward the
            # current waypoint.  Marking it focused also prevents the normal
            # preview-extension timer from immediately recreating the failed
            # two-pose request.
            self.rolling_focused_waypoints.add(retry_waypoint_index)
            if self._queue_rolling_window(
                retry_waypoint_index,
                maximum_window_size=1,
            ):
                self._log_event(
                    'rolling_preview_abort_current_only_retry',
                    self.rolling_route_enu[retry_waypoint_index],
                    status,
                    goal_mode='rolling_route',
                    route_pose_count=1,
                )
                self.get_logger().warning(
                    'Rolling preview window at waypoint {} aborted; '
                    'retrying the reachable current waypoint without '
                    'preview pose {}'.format(
                        retry_waypoint_index + 1,
                        retry_waypoint_index + 2,
                    )
                )
                self._publish_status(
                    'rolling_preview_abort_current_only_retry'
                )
        retry_aborted_segment = should_retry_aborted_rolling_segment(
            status,
            is_current,
            goal_mode,
            route_window_start,
            self.rolling_route_poses is not None,
            (
                self.pending_route_poses is not None
                or self.pending_pose is not None
            ),
        )
        if retry_aborted_segment and self._queue_rolling_segment(
            retry_waypoint_index
        ):
            self.next_goal_send_wall_time = (
                time.monotonic() + self.rolling_segment_abort_retry_s
            )
            self._log_event(
                'rolling_segment_abort_retry_queued',
                self.rolling_route_enu[retry_waypoint_index],
                status,
                goal_mode='rolling_segment',
                route_pose_count=1,
            )
            self.get_logger().warning(
                'Rolling segment at waypoint {} aborted; vehicle remains '
                'stopped and the same mission segment will be replanned '
                'after {:.1f} s'.format(
                    retry_waypoint_index + 1,
                    self.rolling_segment_abort_retry_s,
                )
            )
            self._publish_status('rolling_segment_abort_retry_wait')
        if (
            is_current
            and goal_mode == 'rolling_segment'
            and status == 4
            and route_window_start is not None
            and self.pending_route_poses is None
            and route_window_start + 1 < route_total_count
        ):
            # Nav2's position-only goal checker may complete before the
            # odometry passage callback observes the same waypoint. Advance
            # the mission exactly once and never leave the completed target
            # inside a still-active multi-pose action.
            self.rolling_geometry_passed.add(route_window_start)
            next_start = route_window_start + 1
            self._publish_rolling_remaining(
                max(0, route_total_count - next_start)
            )
            self._queue_rolling_segment(next_start)
        self._log_event(
            'goal_result_current' if is_current else 'goal_result_superseded',
            signature,
            status,
            goal_mode=goal_mode,
            route_pose_count=route_pose_count,
        )
        if (
            self.pending_route_poses is not None
            or self.pending_pose is not None
        ):
            # A canceled goal must not overwrite the status of the goal that
            # is about to replace it.  This was the source of intermittent
            # waypoint handoff stops.
            self._publish_status(
                'rolling_segment_abort_retry_wait'
                if retry_aborted_segment
                else 'handoff_to_next_goal'
            )
            self._try_send_pending()
            return
        if (
            is_current
            and goal_mode in ('route', 'rolling_segment')
            and status == 4
        ):
            self.route_mode = False
            self.route_rolling_horizon = False
            self.rolling_route_poses = None
            self.rolling_route_enu = None
            self.rolling_route_total_count = 0
            self.rolling_route_start_xy = None
            self.rolling_crossing_armed = set()
            self.rolling_geometry_passed = set()
            self.rolling_focused_waypoints = set()
            self.rolling_preview_preserved_waypoints = set()
        elif is_current and goal_mode in ('route', 'rolling_segment'):
            # Do not silently fall back to sequential NavigateToPose goals.
            # That fallback hid route-action failures and recreated the late
            # Ackermann turn that whole-route preview is meant to prevent.
            self.route_mode = True
        self._publish_status('nav2_result_{}'.format(status))
        self.get_logger().info(
            'Nav2 action finished with status={}'.format(status)
        )

    def destroy_node(self):
        if hasattr(self, 'event_logger'):
            try:
                self.event_logger[0].close()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Nav2GoalBridgeNode()
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
