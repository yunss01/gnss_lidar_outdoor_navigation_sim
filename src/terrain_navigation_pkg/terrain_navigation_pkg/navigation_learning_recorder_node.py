#!/usr/bin/env python3
"""
Record synchronized navigation observations without affecting control.

The node is intentionally subscriber-only.  It captures the current LiDAR,
costmaps, plans, teacher commands and mission state while F9/F10 is active,
then writes compressed samples from a background thread.
"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import threading
import time

from geometry_msgs.msg import PointStamped, PoseStamped, Twist, Vector3Stamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMessage
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32, String, UInt32

from .navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
    occupancy_grid_to_vehicle_bev,
    quaternion_to_yaw,
    world_to_vehicle_xy,
)
from .traversability_evidence_core import VALID_DISPOSITIONS


CSV_FIELDS = [
    'sample_id', 'wall_time_iso', 'ros_time_s', 'cloud_stamp_s',
    'cloud_age_s', 'semantic_cloud_stamp_s', 'semantic_alignment_delta_s',
    'imu_age_s', 'odom_age_s', 'file', 'route_status', 'route_index',
    'route_size', 'vehicle_x', 'vehicle_y', 'vehicle_z', 'vehicle_yaw',
    'speed_mps', 'goal_vehicle_x', 'goal_vehicle_y', 'goal_vehicle_z',
    'teacher_subgoal_x', 'teacher_subgoal_y', 'nav2_cmd_speed',
    'nav2_cmd_yaw_rate', 'output_cmd_speed', 'output_cmd_yaw_rate',
    'safety_state', 'safety_obstacle_points', 'path_hard_valid',
    'nav2_status', 'far_guide_status', 'path_clearance_status',
    'nav2_plan_points', 'far_guide_points', 'raw_lidar_points',
    'collision_event_count', 'collision_max_intensity',
]


def path_hard_valid_subscription_qos():
    """Match the clearance validator's reliable, volatile publisher QoS."""

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def perception_capture_readiness(
    latest,
    latest_wall,
    *,
    now,
    maximum_cloud_age_s,
    maximum_odometry_age_s,
    require_semantic_cloud,
    maximum_semantic_alignment_s,
):
    """Return whether every input required to save a sample is ready."""

    cloud = latest.get('cloud')
    if cloud is None:
        return False, 'waiting_for_geometric_cloud'
    odometry = latest.get('odom')
    if odometry is None:
        return False, 'waiting_for_odometry'
    cloud_age = now - latest_wall.get('cloud', -math.inf)
    if cloud_age > maximum_cloud_age_s:
        return False, 'geometric_cloud_stale'
    odometry_age = now - latest_wall.get('odom', -math.inf)
    if odometry_age > maximum_odometry_age_s:
        return False, 'odometry_stale'
    if not require_semantic_cloud:
        return True, 'ready'
    semantic_cloud = latest.get('semantic_cloud')
    if semantic_cloud is None:
        return False, 'waiting_for_semantic_cloud'
    semantic_age = now - latest_wall.get('semantic_cloud', -math.inf)
    if semantic_age > maximum_cloud_age_s:
        return False, 'semantic_cloud_stale'
    alignment_delta = abs(
        _stamp_seconds(semantic_cloud.header.stamp)
        - _stamp_seconds(cloud.header.stamp)
    )
    if alignment_delta > maximum_semantic_alignment_s:
        return False, 'semantic_cloud_not_aligned'
    return True, 'ready'


def perception_capture_completion_result(queued_samples):
    """Never report a zero-sample perception session as completed."""

    if int(queued_samples) <= 0:
        return 'perception_capture_failed_no_samples'
    return 'perception_capture_completed'


def _json_write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    temporary.replace(path)


class NavigationLearningRecorder(Node):
    """Create one learning-data session for each active F9/F10 mission."""

    def __init__(self):
        super().__init__('navigation_learning_recorder_node')
        self._declare_parameters()
        self.enabled = bool(self.get_parameter('enabled').value)
        self.output_root = Path(
            str(self.get_parameter('output_directory').value)
        ).expanduser()
        self.record_only_active = bool(
            self.get_parameter('record_only_when_route_active').value
        )
        self.maximum_cloud_age = float(
            self.get_parameter('maximum_cloud_age_s').value
        )
        self.maximum_odom_age = float(
            self.get_parameter('maximum_odometry_age_s').value
        )
        self.save_raw_points = bool(
            self.get_parameter('save_raw_points').value
        )
        self.save_sample_files = bool(
            self.get_parameter('save_sample_files').value
        )
        self.save_semantic_labels = bool(
            self.get_parameter('save_semantic_labels').value
        )
        self.maximum_semantic_alignment_s = float(
            self.get_parameter('maximum_semantic_alignment_s').value
        )
        self.controlled_traversability_actors = (
            self._controlled_traversability_actor_policy()
        )
        self.maximum_imu_age_s = float(
            self.get_parameter('maximum_imu_age_s').value
        )
        if self.save_semantic_labels:
            if not self.save_sample_files or not self.save_raw_points:
                raise ValueError(
                    'semantic labels require sample files and raw points'
                )
            if self.maximum_semantic_alignment_s <= 0.0:
                raise ValueError(
                    'maximum_semantic_alignment_s must be positive'
                )
        elif self.controlled_traversability_actors:
            raise ValueError(
                'controlled traversability actor policy requires '
                'save_semantic_labels=true'
            )
        # A short stationary capture is kept separate from the mission
        # recorder: it starts only after fresh LiDAR and odometry arrive, then
        # closes itself after the requested duration.  This lets perception
        # validation reuse one raw scan across B0/B1/Proposed offline.
        self.perception_capture_on_start = bool(
            self.get_parameter('perception_capture_on_start').value
        )
        self.perception_capture_duration_s = float(
            self.get_parameter('perception_capture_duration_s').value
        )
        self.perception_capture_root = Path(str(
            self.get_parameter('perception_capture_output_directory').value
        )).expanduser()
        self.perception_capture_label = str(
            self.get_parameter('perception_capture_label').value
        ).strip()
        if self.perception_capture_on_start:
            if self.perception_capture_duration_s <= 0.0:
                raise ValueError('perception_capture_duration_s must be positive')
            if not self.save_sample_files or not self.save_raw_points:
                raise ValueError(
                    'perception capture requires save_sample_files and '
                    'save_raw_points'
                )
        self.evaluation_variant = str(
            self.get_parameter('evaluation_variant').value
        ).strip()
        self.geometry = BevGeometry(
            x_min_m=float(self.get_parameter('bev_x_min_m').value),
            x_max_m=float(self.get_parameter('bev_x_max_m').value),
            y_min_m=float(self.get_parameter('bev_y_min_m').value),
            y_max_m=float(self.get_parameter('bev_y_max_m').value),
            resolution_m=float(self.get_parameter('bev_resolution_m').value),
            z_min_m=float(self.get_parameter('bev_z_min_m').value),
            z_max_m=float(self.get_parameter('bev_z_max_m').value),
        )

        self._lock = threading.Lock()
        self._latest = {}
        self._latest_wall = {}
        self._route_json = ''
        self._route_status = 'idle'
        self._route_index = 0
        self._route_size = 0
        self._active_session = None
        self._sample_sequence = 0
        self._last_cloud_key = None
        self._queued_samples = 0
        self._dropped_samples = 0
        self._collision_event_count = 0
        self._collision_max_intensity = 0.0
        self._perception_capture_state = 'waiting'
        self._perception_capture_deadline = None

        queue_size = int(self.get_parameter('writer_queue_size').value)
        self._writer_queue = queue.Queue(maxsize=max(4, queue_size))
        self._writer_errors = queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name='navigation-learning-writer',
            daemon=True,
        )
        self._writer_thread.start()

        self._create_subscriptions()
        rate = max(0.2, float(self.get_parameter('record_rate_hz').value))
        self.create_timer(1.0 / rate, self._capture)
        self.create_timer(1.0, self._report_writer_errors)
        if self.enabled and self.perception_capture_on_start:
            self.create_timer(0.1, self._manage_perception_capture)
        self.get_logger().info(
            'Navigation learning recorder ready: enabled=%s, rate=%.1f Hz, '
            'BEV=%dx%d, variant=%s, output=%s, perception_capture=%s' % (
                self.enabled, rate, self.geometry.width,
                self.geometry.height, self.evaluation_variant,
                self.output_root, self.perception_capture_on_start,
            )
        )

    def _declare_parameters(self):
        defaults = {
            'enabled': True,
            'evaluation_variant': 'proposed',
            'output_directory': '~/terrain_nav_data/learning/raw',
            'record_rate_hz': 5.0,
            'record_only_when_route_active': True,
            'maximum_cloud_age_s': 0.7,
            'maximum_odometry_age_s': 0.7,
            'writer_queue_size': 64,
            'save_raw_points': True,
            'save_sample_files': True,
            'save_semantic_labels': False,
            'maximum_semantic_alignment_s': 0.03,
            'controlled_traversability_actor_id': -1,
            'controlled_traversability_disposition': '',
            'controlled_traversability_blueprint': '',
            'controlled_traversability_policy_source': (
                'vehicle_clearance_policy'
            ),
            'maximum_imu_age_s': 0.2,
            'perception_capture_on_start': False,
            'perception_capture_duration_s': 15.0,
            'perception_capture_output_directory': (
                '~/terrain_nav_data/perception_validation/raw'
            ),
            'perception_capture_label': '',
            'bev_x_min_m': -10.0,
            'bev_x_max_m': 30.0,
            'bev_y_min_m': -20.0,
            'bev_y_max_m': 20.0,
            'bev_z_min_m': -2.0,
            'bev_z_max_m': 3.0,
            'bev_resolution_m': 0.25,
            'point_cloud_topic': '/lidar/points',
            'semantic_cloud_topic': '/lidar/semantic_points',
            'imu_topic': '/vectornav/imu',
            'odometry_topic': '/vehicle/odometry',
            'route_topic': '/navigation/waypoint_route',
            'route_status_topic': '/navigation/route_status',
            'route_index_topic': '/navigation/route_index',
            'route_size_topic': '/navigation/route_size',
            'goal_local_topic': '/navigation/goal_local',
            'current_local_topic': '/navigation/current_local',
            'goal_vector_topic': '/navigation/goal_vector',
            'far_subgoal_topic': '/navigation/far_subgoal',
            'far_guide_path_topic': '/navigation/far_guide_path',
            'nav2_plan_topic': '/plan',
            'local_costmap_topic': '/local_costmap/costmap',
            'global_costmap_topic': '/global_costmap/costmap',
            'nav2_command_topic': '/cmd_vel_nav2',
            'output_command_topic': '/cmd_vel',
            'safety_state_topic': '/safety/state',
            'safety_obstacle_points_topic': '/safety/obstacle_points',
            'path_hard_valid_topic': '/navigation/path_clearance/hard_valid',
            'path_clearance_status_topic': '/navigation/path_clearance/status',
            'nav2_status_topic': '/navigation/nav2_status',
            'far_guide_status_topic': '/navigation/far_guide_status',
            'collision_topic': '/vehicle/collision',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _controlled_traversability_actor_policy(self):
        """Return one optional controlled actor entry for session metadata."""
        actor_id = int(
            self.get_parameter('controlled_traversability_actor_id').value
        )
        disposition = str(self.get_parameter(
            'controlled_traversability_disposition'
        ).value).strip().lower()
        blueprint = str(self.get_parameter(
            'controlled_traversability_blueprint'
        ).value).strip()
        policy_source = str(self.get_parameter(
            'controlled_traversability_policy_source'
        ).value).strip()
        if actor_id < 0:
            if disposition or blueprint:
                raise ValueError(
                    'controlled actor disposition/blueprint was set without '
                    'a non-negative controlled_traversability_actor_id'
                )
            return []
        if disposition not in VALID_DISPOSITIONS:
            raise ValueError(
                'controlled_traversability_disposition must be one of: '
                + ', '.join(sorted(VALID_DISPOSITIONS))
            )
        return [{
            'actor_id': actor_id,
            'disposition': disposition,
            'blueprint': blueprint,
            'policy_source': policy_source,
        }]

    def _create_subscriptions(self):
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        path_hard_valid_qos = path_hard_valid_subscription_qos()

        def topic(name):
            return str(self.get_parameter(name).value)

        self.create_subscription(
            PointCloud2, topic('point_cloud_topic'),
            lambda msg: self._remember('cloud', msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2, topic('semantic_cloud_topic'),
            lambda msg: self._remember('semantic_cloud', msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu, topic('imu_topic'),
            lambda msg: self._remember('imu', msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry, topic('odometry_topic'),
            lambda msg: self._remember('odom', msg), qos_profile_sensor_data,
        )
        self.create_subscription(
            String, topic('route_topic'), self._on_route, latched
        )
        self.create_subscription(
            String, topic('route_status_topic'), self._on_route_status, latched
        )
        self.create_subscription(
            UInt32, topic('route_index_topic'), self._on_route_index, latched
        )
        self.create_subscription(
            UInt32, topic('route_size_topic'), self._on_route_size, latched
        )
        self.create_subscription(
            Float32, topic('collision_topic'), self._on_collision, reliable
        )
        for message_type, parameter, key, qos in [
            (PointStamped, 'goal_local_topic', 'goal_local', latched),
            (PointStamped, 'current_local_topic', 'current_local', latched),
            (Vector3Stamped, 'goal_vector_topic', 'goal_vector', reliable),
            (PoseStamped, 'far_subgoal_topic', 'far_subgoal', latched),
            (PathMessage, 'far_guide_path_topic', 'far_guide_path', reliable),
            (PathMessage, 'nav2_plan_topic', 'nav2_plan', reliable),
            (OccupancyGrid, 'local_costmap_topic', 'local_costmap', latched),
            (OccupancyGrid, 'global_costmap_topic', 'global_costmap', latched),
            (Twist, 'nav2_command_topic', 'nav2_command', reliable),
            (Twist, 'output_command_topic', 'output_command', reliable),
            (String, 'safety_state_topic', 'safety_state', reliable),
            (UInt32, 'safety_obstacle_points_topic', 'safety_points', reliable),
            # The clearance validator publishes reliable/volatile Bool data.
            # Requesting transient-local durability here is QoS-incompatible
            # and silently leaves every recorded sample at its false default.
            (
                Bool,
                'path_hard_valid_topic',
                'path_hard_valid',
                path_hard_valid_qos,
            ),
            (String, 'path_clearance_status_topic', 'clearance_status', reliable),
            (String, 'nav2_status_topic', 'nav2_status', reliable),
            (String, 'far_guide_status_topic', 'far_guide_status', reliable),
        ]:
            self.create_subscription(
                message_type, topic(parameter),
                lambda msg, item=key: self._remember(item, msg), qos,
            )

    def _remember(self, key, message):
        with self._lock:
            self._latest[key] = message
            self._latest_wall[key] = time.monotonic()

    def _on_route(self, message):
        with self._lock:
            self._route_json = message.data

    def _on_route_index(self, message):
        with self._lock:
            self._route_index = int(message.data)

    def _on_route_size(self, message):
        with self._lock:
            self._route_size = int(message.data)

    def _on_collision(self, message):
        intensity = max(0.0, float(message.data))
        if intensity <= 0.0:
            return
        with self._lock:
            if self._active_session is not None:
                self._collision_event_count += 1
                self._collision_max_intensity = max(
                    self._collision_max_intensity, intensity
                )

    def _on_route_status(self, message):
        status = message.data.strip().lower()
        with self._lock:
            previous = self._route_status
            self._route_status = status
            active = self._active_session is not None
        if not self.enabled:
            return
        # A stationary perception capture has a deliberate lifecycle of its
        # own and must not be interrupted by a latched idle/ready route state.
        if self.perception_capture_on_start:
            return
        if status == 'navigating' and not active:
            self._start_session()
        elif active and status in {
            'completed', 'final_nav2_failed', 'invalid_route',
        }:
            self._close_session(status)
        elif active and status in {'idle', 'ready'} and previous not in {
            'idle', 'ready', 'waiting_for_goal_manager',
        }:
            self._close_session('interrupted_' + status)

    def _start_session(self):
        now = datetime.now(timezone.utc).astimezone()
        session_name = now.strftime('session_%Y%m%d_%H%M%S_%f')
        output_root = (
            self.perception_capture_root
            if self.perception_capture_on_start else self.output_root
        )
        session_path = output_root / session_name
        with self._lock:
            if self._active_session is not None:
                return
            self._active_session = session_path
            self._sample_sequence = 0
            self._last_cloud_key = None
            self._queued_samples = 0
            self._dropped_samples = 0
            self._collision_event_count = 0
            self._collision_max_intensity = 0.0
            route_json = self._route_json
            route_size = self._route_size
        metadata = {
            'schema_version': 1,
            'session': session_name,
            'started_at': now.isoformat(),
            'purpose': (
                'perception validation capture'
                if self.perception_capture_on_start
                else (
                    'local-navigation policy learning'
                    if self.save_sample_files
                    else 'navigation performance evaluation'
                )
            ),
            'control_effect': 'none; subscriber-only recorder',
            'evaluation_variant': self.evaluation_variant,
            'perception_capture_label': self.perception_capture_label,
            'route_json': route_json,
            'route_size': route_size,
            'bev': {
                'x_min_m': self.geometry.x_min_m,
                'x_max_m': self.geometry.x_max_m,
                'y_min_m': self.geometry.y_min_m,
                'y_max_m': self.geometry.y_max_m,
                'z_min_m': self.geometry.z_min_m,
                'z_max_m': self.geometry.z_max_m,
                'resolution_m': self.geometry.resolution_m,
                'height': self.geometry.height,
                'width': self.geometry.width,
                'lidar_channels': [
                    'occupancy', 'log_density', 'maximum_height',
                    'height_span',
                ],
                'orientation': 'row 0 forward, column 0 vehicle left',
            },
            'privileged_supervision': {
                'semantic_labels_saved': self.save_semantic_labels,
                'semantic_cloud_topic': str(
                    self.get_parameter('semantic_cloud_topic').value
                ),
                'model_input_policy': (
                    'semantic tags are training targets only; deployed model '
                    'must consume geometric LiDAR, GNSS guidance, and state'
                ),
            },
            'controlled_traversability_actors': (
                self.controlled_traversability_actors
            ),
        }
        self._writer_queue.put(('start', session_path, metadata))
        self.get_logger().info('Learning session started: %s' % session_path)

    def _manage_perception_capture(self):
        """Open/close one sensor-ready, non-driving capture session."""
        if not self.enabled or not self.perception_capture_on_start:
            return
        now = time.monotonic()
        with self._lock:
            state = self._perception_capture_state
            ready, readiness_reason = perception_capture_readiness(
                self._latest,
                self._latest_wall,
                now=now,
                maximum_cloud_age_s=self.maximum_cloud_age,
                maximum_odometry_age_s=self.maximum_odom_age,
                require_semantic_cloud=self.save_semantic_labels,
                maximum_semantic_alignment_s=(
                    self.maximum_semantic_alignment_s
                ),
            )
            deadline = self._perception_capture_deadline
            queued_samples = self._queued_samples
        if state == 'waiting':
            if ready:
                self._start_session()
                with self._lock:
                    self._perception_capture_state = 'recording'
                    self._perception_capture_deadline = (
                        now + self.perception_capture_duration_s
                    )
                self.get_logger().info(
                    'Perception capture started: label=%s duration=%.1f s' % (
                        self.perception_capture_label or 'unlabelled',
                        self.perception_capture_duration_s,
                    )
                )
        elif state == 'recording' and deadline is not None and now >= deadline:
            result = perception_capture_completion_result(queued_samples)
            self._close_session(result)
            with self._lock:
                self._perception_capture_state = (
                    'completed'
                    if result == 'perception_capture_completed'
                    else 'failed'
                )
            if result == 'perception_capture_completed':
                self.get_logger().info('Perception capture completed')
            else:
                self.get_logger().error(
                    'Perception capture failed: no samples were saved. '
                    'Check geometric/semantic LiDAR alignment and freshness; '
                    f'last readiness={readiness_reason}'
                )

    def _close_session(self, result):
        with self._lock:
            path = self._active_session
            if path is None:
                return
            summary = {
                'result': result,
                'route_status': self._route_status,
                'route_index': self._route_index,
                'route_size': self._route_size,
                'route_json': self._route_json,
                'queued_samples': self._queued_samples,
                'dropped_samples': self._dropped_samples,
                'collision_event_count': self._collision_event_count,
                'collision_max_intensity': self._collision_max_intensity,
                'ended_at': datetime.now(timezone.utc).astimezone().isoformat(),
            }
            self._active_session = None
        self._writer_queue.put(('close', path, summary))
        self.get_logger().info(
            'Learning session queued for close: result=%s, samples=%d, '
            'dropped=%d' % (
                result, summary['queued_samples'], summary['dropped_samples']
            )
        )

    def _capture(self):
        if not self.enabled:
            return
        now_wall = time.monotonic()
        with self._lock:
            session = self._active_session
            if session is None and self.record_only_active:
                return
            cloud = self._latest.get('cloud')
            odom = self._latest.get('odom')
            cloud_age = now_wall - self._latest_wall.get('cloud', -math.inf)
            odom_age = now_wall - self._latest_wall.get('odom', -math.inf)
            if cloud is None or odom is None:
                return
            if cloud_age > self.maximum_cloud_age or odom_age > self.maximum_odom_age:
                return
            semantic_cloud = self._latest.get('semantic_cloud')
            semantic_stamp_s = math.nan
            semantic_alignment_delta_s = math.nan
            if self.save_semantic_labels:
                if semantic_cloud is None:
                    return
                semantic_stamp_s = _stamp_seconds(
                    semantic_cloud.header.stamp
                )
                semantic_alignment_delta_s = abs(
                    semantic_stamp_s - _stamp_seconds(cloud.header.stamp)
                )
                if (
                    semantic_alignment_delta_s
                    > self.maximum_semantic_alignment_s
                ):
                    return
            imu_age = now_wall - self._latest_wall.get('imu', -math.inf)
            cloud_key = (
                cloud.header.stamp.sec, cloud.header.stamp.nanosec,
                cloud.width, cloud.row_step,
            )
            if cloud_key == self._last_cloud_key:
                return
            self._last_cloud_key = cloud_key
            self._sample_sequence += 1
            sample_id = self._sample_sequence
            snapshot = dict(self._latest)
            snapshot_wall = dict(self._latest_wall)
            context = {
                'session': session,
                'sample_id': sample_id,
                'wall_time_iso': datetime.now(
                    timezone.utc
                ).astimezone().isoformat(),
                'ros_time_s': self.get_clock().now().nanoseconds * 1.0e-9,
                'cloud_age_s': cloud_age,
                'semantic_cloud_stamp_s': semantic_stamp_s,
                'semantic_alignment_delta_s': semantic_alignment_delta_s,
                'imu_age_s': imu_age,
                'odom_age_s': odom_age,
                'route_status': self._route_status,
                'route_index': self._route_index,
                'route_size': self._route_size,
                'route_json': self._route_json,
                'snapshot_wall': snapshot_wall,
                'collision_event_count': self._collision_event_count,
                'collision_max_intensity': self._collision_max_intensity,
            }
        try:
            self._writer_queue.put_nowait(('sample', snapshot, context))
            with self._lock:
                self._queued_samples += 1
        except queue.Full:
            with self._lock:
                self._dropped_samples += 1

    @staticmethod
    def _cloud_xyz(message):
        try:
            values = point_cloud2.read_points_numpy(
                message, field_names=['x', 'y', 'z'], skip_nans=True
            )
            array = np.asarray(values)
            if array.dtype.names:
                return np.column_stack([
                    array['x'], array['y'], array['z']
                ]).astype(np.float32, copy=False)
            return np.asarray(array, dtype=np.float32).reshape((-1, 3))
        except (AssertionError, ValueError, TypeError):
            values = point_cloud2.read_points(
                message, field_names=['x', 'y', 'z'], skip_nans=True
            )
            array = np.asarray(values)
            if array.dtype.names:
                return np.column_stack([
                    array['x'], array['y'], array['z']
                ]).astype(np.float32, copy=False)
            return np.asarray(array, dtype=np.float32).reshape((-1, 3))

    @staticmethod
    def _semantic_cloud_arrays(message):
        """Read the CARLA-only semantic label cloud without float-casting IDs."""
        if message is None:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint32),
                np.empty((0,), dtype=np.uint32),
            )
        values = point_cloud2.read_points(
            message,
            field_names=[
                'x', 'y', 'z', 'cos_incidence',
                'object_idx', 'object_tag',
            ],
            skip_nans=True,
        )
        array = np.asarray(values)
        if not array.dtype.names:
            raise ValueError('semantic PointCloud2 must preserve named fields')
        xyz = np.column_stack([
            array['x'], array['y'], array['z'],
        ]).astype(np.float32, copy=False)
        return (
            xyz,
            np.asarray(array['cos_incidence'], dtype=np.float32),
            np.asarray(array['object_idx'], dtype=np.uint32),
            np.asarray(array['object_tag'], dtype=np.uint32),
        )

    @staticmethod
    def _path_vehicle(message, pose):
        if message is None or not message.poses:
            return np.empty((0, 2), dtype=np.float32)
        points = np.asarray([
            [item.pose.position.x, item.pose.position.y]
            for item in message.poses
        ], dtype=np.float32)
        return world_to_vehicle_xy(points, pose[0], pose[1], pose[3])

    def _costmap_vehicle(self, message, pose):
        if message is None or message.info.width == 0:
            return np.full(
                (self.geometry.height, self.geometry.width), -1, dtype=np.int8
            )
        grid = np.asarray(message.data, dtype=np.int16).reshape(
            (message.info.height, message.info.width)
        )
        origin = message.info.origin
        yaw = quaternion_to_yaw(
            origin.orientation.x, origin.orientation.y,
            origin.orientation.z, origin.orientation.w,
        )
        return occupancy_grid_to_vehicle_bev(
            grid, message.info.resolution,
            origin.position.x, origin.position.y, yaw,
            pose[0], pose[1], pose[3], self.geometry,
        )

    def _process_sample(self, snapshot, context):
        cloud = snapshot['cloud']
        odom = snapshot['odom']
        points = self._cloud_xyz(cloud)
        keep = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= self.geometry.x_min_m)
            & (points[:, 0] < self.geometry.x_max_m)
            & (points[:, 1] >= self.geometry.y_min_m)
            & (points[:, 1] < self.geometry.y_max_m)
            & (points[:, 2] >= self.geometry.z_min_m)
            & (points[:, 2] <= self.geometry.z_max_m)
        )
        points = points[keep].astype(np.float32, copy=False)
        lidar_bev = None
        if self.save_sample_files:
            lidar_bev = build_lidar_bev(points, self.geometry)
        semantic_xyz = np.empty((0, 3), dtype=np.float32)
        semantic_cosine = np.empty((0,), dtype=np.float32)
        semantic_indices = np.empty((0,), dtype=np.uint32)
        semantic_tags = np.empty((0,), dtype=np.uint32)
        if self.save_semantic_labels:
            (
                semantic_xyz,
                semantic_cosine,
                semantic_indices,
                semantic_tags,
            ) = self._semantic_cloud_arrays(snapshot.get('semantic_cloud'))

        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        yaw = quaternion_to_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        pose = (position.x, position.y, position.z, yaw)
        plan = self._path_vehicle(snapshot.get('nav2_plan'), pose)
        guide = self._path_vehicle(snapshot.get('far_guide_path'), pose)
        local_costmap = global_costmap = None
        if self.save_sample_files:
            local_costmap = self._costmap_vehicle(
                snapshot.get('local_costmap'), pose
            )
            global_costmap = self._costmap_vehicle(
                snapshot.get('global_costmap'), pose
            )

        goal = np.full(3, np.nan, dtype=np.float32)
        vector = snapshot.get('goal_vector')
        if vector is not None:
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            goal[0] = cosine * vector.vector.x + sine * vector.vector.y
            goal[1] = -sine * vector.vector.x + cosine * vector.vector.y
            goal[2] = vector.vector.z
        else:
            goal_local = snapshot.get('goal_local')
            current_local = snapshot.get('current_local')
            if goal_local is not None and current_local is not None:
                delta = np.asarray([[
                    goal_local.point.x - current_local.point.x,
                    goal_local.point.y - current_local.point.y,
                ]], dtype=np.float32)
                goal[:2] = world_to_vehicle_xy(delta, 0.0, 0.0, yaw)[0]
                goal[2] = goal_local.point.z - current_local.point.z

        subgoal = np.full(2, np.nan, dtype=np.float32)
        subgoal_message = snapshot.get('far_subgoal')
        if subgoal_message is not None:
            subgoal[:] = world_to_vehicle_xy(np.asarray([[
                subgoal_message.pose.position.x,
                subgoal_message.pose.position.y,
            ]]), pose[0], pose[1], pose[3])[0]

        nav2_command = snapshot.get('nav2_command') or Twist()
        output_command = snapshot.get('output_command') or Twist()
        twist = odom.twist.twist
        imu_acceleration = np.full(3, np.nan, dtype=np.float32)
        imu_angular_velocity = np.full(3, np.nan, dtype=np.float32)
        imu_orientation = np.full(4, np.nan, dtype=np.float32)
        imu = snapshot.get('imu')
        if imu is not None and context['imu_age_s'] <= self.maximum_imu_age_s:
            imu_acceleration[:] = [
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ]
            imu_angular_velocity[:] = [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ]
            imu_orientation[:] = [
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
                imu.orientation.w,
            ]
        sample_name = 'sample_%06d.npz' % context['sample_id']
        sample_path = context['session'] / 'samples' / sample_name
        arrays = {
            'lidar_bev': (
                lidar_bev.astype(np.float16) if lidar_bev is not None
                else np.empty((0,), dtype=np.float16)
            ),
            'lidar_points_xyz': (
                points if self.save_raw_points
                else np.empty((0, 3), dtype=np.float32)
            ),
            'semantic_lidar_points_xyz': semantic_xyz,
            'semantic_lidar_cos_incidence': semantic_cosine,
            'semantic_lidar_object_idx': semantic_indices,
            'semantic_lidar_object_tag': semantic_tags,
            'semantic_alignment_delta_s': np.asarray(
                context['semantic_alignment_delta_s'], dtype=np.float32
            ),
            'local_costmap_bev': local_costmap,
            'global_costmap_bev': global_costmap,
            'nav2_plan_vehicle_xy': plan,
            'far_guide_path_vehicle_xy': guide,
            'teacher_subgoal_vehicle_xy': subgoal,
            'goal_vehicle_xyz': goal,
            'vehicle_pose_odom_xyzyaw': np.asarray(pose, dtype=np.float64),
            'vehicle_twist_xyz_rpy': np.asarray([
                twist.linear.x, twist.linear.y, twist.linear.z,
                twist.angular.x, twist.angular.y, twist.angular.z,
            ], dtype=np.float32),
            'imu_linear_acceleration_xyz': imu_acceleration,
            'imu_angular_velocity_xyz': imu_angular_velocity,
            'imu_orientation_xyzw': imu_orientation,
            'cmd_vel_nav2': np.asarray([
                nav2_command.linear.x, nav2_command.angular.z
            ], dtype=np.float32),
            'cmd_vel_output': np.asarray([
                output_command.linear.x, output_command.angular.z
            ], dtype=np.float32),
        }
        if self.save_sample_files:
            np.savez_compressed(sample_path, **arrays)
        speed = math.sqrt(
            twist.linear.x ** 2 + twist.linear.y ** 2 + twist.linear.z ** 2
        )

        def text_value(key, default=''):
            message = snapshot.get(key)
            return message.data if message is not None else default

        row = {
            'sample_id': context['sample_id'],
            'wall_time_iso': context['wall_time_iso'],
            'ros_time_s': '%.9f' % context['ros_time_s'],
            'cloud_stamp_s': '%.9f' % _stamp_seconds(cloud.header.stamp),
            'cloud_age_s': '%.4f' % context['cloud_age_s'],
            'semantic_cloud_stamp_s': (
                '%.9f' % context['semantic_cloud_stamp_s']
                if math.isfinite(context['semantic_cloud_stamp_s']) else ''
            ),
            'semantic_alignment_delta_s': (
                '%.5f' % context['semantic_alignment_delta_s']
                if math.isfinite(context['semantic_alignment_delta_s']) else ''
            ),
            'imu_age_s': (
                '%.4f' % context['imu_age_s']
                if math.isfinite(context['imu_age_s']) else ''
            ),
            'odom_age_s': '%.4f' % context['odom_age_s'],
            'file': ('samples/' + sample_name) if self.save_sample_files else '',
            'route_status': context['route_status'],
            'route_index': context['route_index'],
            'route_size': context['route_size'],
            'vehicle_x': '%.6f' % pose[0],
            'vehicle_y': '%.6f' % pose[1],
            'vehicle_z': '%.6f' % pose[2],
            'vehicle_yaw': '%.7f' % pose[3],
            'speed_mps': '%.5f' % speed,
            'goal_vehicle_x': '%.5f' % goal[0],
            'goal_vehicle_y': '%.5f' % goal[1],
            'goal_vehicle_z': '%.5f' % goal[2],
            'teacher_subgoal_x': '%.5f' % subgoal[0],
            'teacher_subgoal_y': '%.5f' % subgoal[1],
            'nav2_cmd_speed': '%.5f' % nav2_command.linear.x,
            'nav2_cmd_yaw_rate': '%.5f' % nav2_command.angular.z,
            'output_cmd_speed': '%.5f' % output_command.linear.x,
            'output_cmd_yaw_rate': '%.5f' % output_command.angular.z,
            'safety_state': text_value('safety_state'),
            'safety_obstacle_points': int(
                getattr(snapshot.get('safety_points'), 'data', 0)
            ),
            'path_hard_valid': int(bool(
                getattr(snapshot.get('path_hard_valid'), 'data', False)
            )),
            'nav2_status': text_value('nav2_status'),
            'far_guide_status': text_value('far_guide_status'),
            'path_clearance_status': text_value('clearance_status'),
            'nav2_plan_points': int(plan.shape[0]),
            'far_guide_points': int(guide.shape[0]),
            'raw_lidar_points': int(points.shape[0]),
            'collision_event_count': context['collision_event_count'],
            'collision_max_intensity': '%.5f' % context[
                'collision_max_intensity'
            ],
        }
        return row

    def _writer_loop(self):
        sessions = {}
        while True:
            item = self._writer_queue.get()
            try:
                if item is None:
                    break
                action = item[0]
                if action == 'start':
                    _, path, metadata = item
                    path.mkdir(parents=True, exist_ok=True)
                    if self.save_sample_files:
                        (path / 'samples').mkdir(parents=True, exist_ok=True)
                    _json_write(path / 'metadata.json', metadata)
                    stream = (path / 'frames.csv').open(
                        'w', newline='', encoding='utf-8'
                    )
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    stream.flush()
                    sessions[path] = {
                        'metadata': metadata, 'stream': stream,
                        'writer': writer, 'written': 0,
                    }
                elif action == 'sample':
                    _, snapshot, context = item
                    state = sessions.get(context['session'])
                    if state is None:
                        raise RuntimeError('sample arrived before session start')
                    row = self._process_sample(snapshot, context)
                    state['writer'].writerow(row)
                    state['stream'].flush()
                    state['written'] += 1
                elif action == 'close':
                    _, path, summary = item
                    state = sessions.pop(path, None)
                    if state is not None:
                        state['metadata'].update(summary)
                        state['metadata']['written_samples'] = state['written']
                        _json_write(path / 'metadata.json', state['metadata'])
                        state['stream'].close()
            except Exception as error:  # keep later sessions recordable
                self._writer_errors.put(repr(error))
            finally:
                self._writer_queue.task_done()
        for state in sessions.values():
            state['stream'].close()

    def _report_writer_errors(self):
        while True:
            try:
                error = self._writer_errors.get_nowait()
            except queue.Empty:
                return
            self.get_logger().error('Learning recorder writer error: ' + error)

    def shutdown(self):
        if self._active_session is not None:
            self._close_session('node_shutdown')
        try:
            self._writer_queue.put(None, timeout=2.0)
            self._writer_thread.join(timeout=30.0)
        except Exception as error:
            self.get_logger().error('Recorder shutdown error: %r' % error)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationLearningRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
