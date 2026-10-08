#!/usr/bin/env python3
"""Evaluate shadow traversability against CARLA-only semantic supervision."""

from __future__ import annotations

from collections import OrderedDict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from nav_msgs.msg import OccupancyGrid, Path as PathMessage
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
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String, UInt32

from .navigation_learning_recorder_core import BevGeometry, quaternion_to_yaw
from .traversability_learning_core import (
    EgoFootprint,
    build_semantic_traversability_targets,
)
from .traversability_online_evaluation_core import (
    COUNT_FIELDS,
    OnlineTraversabilityAccumulator,
    build_swept_footprint_mask,
    finite_alignment_summary,
    largest_connected_component,
    occupancy_grid_to_bev,
)


CSV_FIELDS = [
    'frame_index', 'wall_time_iso', 'source_stamp_s',
    'semantic_stamp_s', 'alignment_delta_s', 'route_index', 'route_size',
] + list(COUNT_FIELDS) + [
    'raw_accuracy', 'raw_obstacle_iou', 'raw_obstacle_precision',
    'raw_obstacle_recall', 'raw_false_free_rate',
    'raw_false_obstacle_rate', 'selective_coverage',
    'selective_accepted_accuracy', 'selective_obstacle_recall',
    'selective_false_free_rate', 'selective_false_obstacle_rate',
    'selective_safe_retention_rate',
    'corridor_source', 'corridor_alignment_delta_s',
    'corridor_path_points', 'corridor_cells',
] + ['corridor_' + name for name in COUNT_FIELDS] + [
    'corridor_raw_false_free_rate',
    'corridor_raw_false_obstacle_rate',
    'corridor_selective_coverage',
    'corridor_selective_accepted_accuracy',
    'corridor_selective_false_free_rate',
    'corridor_selective_false_obstacle_rate',
    'corridor_selective_safe_retention_rate',
    'corridor_false_free_cluster_cells',
    'corridor_false_free_cluster_forward_span_m',
    'corridor_false_free_cluster_lateral_span_m',
    'consecutive_corridor_false_free_frames',
    'error_snapshot_file',
]


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    temporary.replace(path)


class TraversabilityShadowEvaluator(Node):
    """Subscriber-only online evaluator with no navigation authority."""

    def __init__(self):
        super().__init__('traversability_shadow_evaluator_node')
        self._declare_parameters()
        self.geometry = BevGeometry(
            x_min_m=float(self.get_parameter('bev_x_min_m').value),
            x_max_m=float(self.get_parameter('bev_x_max_m').value),
            y_min_m=float(self.get_parameter('bev_y_min_m').value),
            y_max_m=float(self.get_parameter('bev_y_max_m').value),
            resolution_m=float(self.get_parameter('bev_resolution_m').value),
            z_min_m=float(self.get_parameter('bev_z_min_m').value),
            z_max_m=float(self.get_parameter('bev_z_max_m').value),
        )
        self.footprint = EgoFootprint(
            rear_m=float(self.get_parameter('ego_rear_m').value),
            front_m=float(self.get_parameter('ego_front_m').value),
            half_width_m=float(self.get_parameter('ego_half_width_m').value),
        )
        self.maximum_alignment_ns = int(round(
            float(self.get_parameter('maximum_semantic_alignment_s').value)
            * 1.0e9
        ))
        self.maximum_pending_frames = int(
            self.get_parameter('maximum_pending_frames').value
        )
        if self.maximum_alignment_ns <= 0 or self.maximum_pending_frames < 2:
            raise ValueError(
                'alignment and pending-frame limits must be positive'
            )
        self.output_root = Path(str(
            self.get_parameter('output_directory').value
        )).expanduser()
        self.corridor_wait_s = float(
            self.get_parameter('corridor_wait_s').value
        )
        self.maximum_trajectory_alignment_ns = int(round(
            float(self.get_parameter(
                'maximum_trajectory_alignment_s'
            ).value) * 1.0e9
        ))
        self.corridor_front_m = float(
            self.get_parameter('corridor_vehicle_front_m').value
        )
        self.corridor_rear_m = float(
            self.get_parameter('corridor_vehicle_rear_m').value
        )
        self.corridor_half_width_m = float(
            self.get_parameter('corridor_half_width_m').value
        )
        self.corridor_horizon_m = float(
            self.get_parameter('corridor_horizon_m').value
        )
        self.corridor_spacing_m = float(
            self.get_parameter('corridor_spacing_m').value
        )
        self.maximum_error_snapshots = int(
            self.get_parameter('maximum_error_snapshots_per_route').value
        )
        if min(
            self.corridor_wait_s,
            self.corridor_front_m,
            self.corridor_rear_m,
            self.corridor_half_width_m,
            self.corridor_horizon_m,
            self.corridor_spacing_m,
        ) <= 0.0:
            raise ValueError('corridor parameters must be positive')
        if self.maximum_error_snapshots < 1:
            raise ValueError('maximum error snapshots must be positive')

        self.predictions = OrderedDict()
        self.semantic_clouds = OrderedDict()
        self.checked_trajectories = OrderedDict()
        self.route_status = 'idle'
        self.route_index = 0
        self.route_size = 0
        self.active_path = None
        self.csv_file = None
        self.csv_writer = None
        self.accumulator = OnlineTraversabilityAccumulator()
        self.corridor_accumulator = OnlineTraversabilityAccumulator()
        self.alignment_deltas = []
        self.frame_index = 0
        self.dropped_prediction_frames = 0
        self.dropped_semantic_frames = 0
        self.started_at = None
        self.error_snapshot_count = 0
        self.consecutive_corridor_false_free_frames = 0
        self.maximum_consecutive_corridor_false_free_frames = 0
        self.maximum_corridor_false_free_cluster_cells = 0
        self.maximum_corridor_false_free_forward_span_m = 0.0
        self.maximum_corridor_false_free_lateral_span_m = 0.0
        self.corridor_source_counts = {}

        status_topic = str(self.get_parameter('evaluation_status_topic').value)
        self.status_publisher = self.create_publisher(String, status_topic, 10)
        self._create_subscriptions()
        self.match_timer = self.create_timer(0.05, self._try_matches)
        self.get_logger().info(
            'Traversability evaluator ready: semantic labels are privileged '
            'evaluation targets only; NO model input or control authority'
        )

    def _declare_parameters(self):
        defaults = {
            'semantic_cloud_topic': '/lidar/semantic_points',
            'probability_topic': '/learning/traversability_probability',
            'decision_topic': '/learning/traversability_shadow_decision',
            'evaluation_status_topic': (
                '/learning/traversability_shadow_evaluation'
            ),
            'route_status_topic': '/navigation/route_status',
            'route_index_topic': '/navigation/route_index',
            'route_size_topic': '/navigation/route_size',
            'checked_trajectory_topic': '/safety/checked_trajectory',
            'output_directory': (
                '~/terrain_nav_data/learning/shadow_evaluations'
            ),
            'maximum_semantic_alignment_s': 0.03,
            'maximum_pending_frames': 32,
            'maximum_trajectory_alignment_s': 0.03,
            'corridor_wait_s': 0.12,
            'corridor_vehicle_front_m': 2.4,
            'corridor_vehicle_rear_m': 2.5,
            'corridor_half_width_m': 1.35,
            'corridor_horizon_m': 4.0,
            'corridor_spacing_m': 0.25,
            'maximum_error_snapshots_per_route': 256,
            'minimum_surface_points': 2,
            'obstacle_vertical_span_m': 0.15,
            'bev_x_min_m': -10.0,
            'bev_x_max_m': 30.0,
            'bev_y_min_m': -20.0,
            'bev_y_max_m': 20.0,
            'bev_resolution_m': 0.25,
            'bev_z_min_m': -2.0,
            'bev_z_max_m': 3.0,
            'ego_rear_m': 2.5,
            'ego_front_m': 2.4,
            'ego_half_width_m': 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _create_subscriptions(self):
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        def topic(name):
            return str(self.get_parameter(name).value)

        self.create_subscription(
            PointCloud2, topic('semantic_cloud_topic'),
            self._on_semantic_cloud, qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid, topic('probability_topic'),
            lambda message: self._on_grid('probability', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid, topic('decision_topic'),
            lambda message: self._on_grid('decision', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PathMessage,
            topic('checked_trajectory_topic'),
            self._on_checked_trajectory,
            10,
        )
        self.create_subscription(
            String, topic('route_status_topic'), self._on_route_status, latched
        )
        self.create_subscription(
            UInt32, topic('route_index_topic'),
            lambda message: setattr(self, 'route_index', int(message.data)),
            latched,
        )
        self.create_subscription(
            UInt32, topic('route_size_topic'),
            lambda message: setattr(self, 'route_size', int(message.data)),
            latched,
        )

    def _on_grid(self, name, message):
        stamp = _stamp_ns(message.header.stamp)
        entry = self.predictions.setdefault(stamp, {})
        entry[name] = message
        self.predictions.move_to_end(stamp)
        self._trim_pending(self.predictions, 'prediction')
        self._try_matches()

    def _on_semantic_cloud(self, message):
        stamp = _stamp_ns(message.header.stamp)
        self.semantic_clouds[stamp] = message
        self.semantic_clouds.move_to_end(stamp)
        self._trim_pending(self.semantic_clouds, 'semantic')
        self._try_matches()

    def _on_checked_trajectory(self, message):
        stamp = _stamp_ns(message.header.stamp)
        self.checked_trajectories[stamp] = message
        self.checked_trajectories.move_to_end(stamp)
        while len(self.checked_trajectories) > self.maximum_pending_frames:
            self.checked_trajectories.popitem(last=False)
        self._try_matches()

    def _trim_pending(self, queue, kind):
        while len(queue) > self.maximum_pending_frames:
            queue.popitem(last=False)
            if kind == 'prediction':
                self.dropped_prediction_frames += 1
            else:
                self.dropped_semantic_frames += 1

    def _nearest_semantic_stamp(self, prediction_stamp):
        if prediction_stamp in self.semantic_clouds:
            return prediction_stamp
        if not self.semantic_clouds:
            return None
        candidate = min(
            self.semantic_clouds,
            key=lambda stamp: abs(stamp - prediction_stamp),
        )
        if abs(candidate - prediction_stamp) <= self.maximum_alignment_ns:
            return candidate
        return None

    def _nearest_trajectory_stamp(self, prediction_stamp):
        if prediction_stamp in self.checked_trajectories:
            return prediction_stamp
        if not self.checked_trajectories:
            return None
        candidate = min(
            self.checked_trajectories,
            key=lambda stamp: abs(stamp - prediction_stamp),
        )
        if (
            abs(candidate - prediction_stamp)
            <= self.maximum_trajectory_alignment_ns
        ):
            return candidate
        return None

    def _straight_fallback_trajectory(self):
        count = max(
            1,
            int(np.ceil(
                self.corridor_horizon_m / self.corridor_spacing_m
            )),
        )
        forward = np.linspace(0.0, self.corridor_horizon_m, count + 1)
        return np.column_stack((
            forward,
            np.zeros_like(forward),
            np.zeros_like(forward),
        ))

    @staticmethod
    def _trajectory_array(message):
        if message is None or not message.poses:
            raise ValueError('checked trajectory is empty')
        values = []
        for item in message.poses:
            position = item.pose.position
            orientation = item.pose.orientation
            values.append((
                float(position.x),
                float(position.y),
                quaternion_to_yaw(
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ),
            ))
        return np.asarray(values, dtype=np.float64)

    def _try_matches(self):
        ready = [
            stamp for stamp, values in self.predictions.items()
            if 'probability' in values and 'decision' in values
        ]
        for prediction_stamp in ready:
            semantic_stamp = self._nearest_semantic_stamp(prediction_stamp)
            if semantic_stamp is None:
                continue
            trajectory_stamp = self._nearest_trajectory_stamp(
                prediction_stamp
            )
            values = self.predictions[prediction_stamp]
            if trajectory_stamp is None:
                waiting_since = values.setdefault(
                    '_corridor_waiting_since', time.monotonic()
                )
                if time.monotonic() - waiting_since < self.corridor_wait_s:
                    continue
                trajectory = self._straight_fallback_trajectory()
                trajectory_source = 'straight_fallback'
                trajectory_delta = None
            else:
                trajectory_message = self.checked_trajectories.pop(
                    trajectory_stamp
                )
                try:
                    trajectory = self._trajectory_array(trajectory_message)
                    trajectory_source = 'safety_checked_trajectory'
                    trajectory_delta = (
                        abs(prediction_stamp - trajectory_stamp) * 1.0e-9
                    )
                except ValueError:
                    trajectory = self._straight_fallback_trajectory()
                    trajectory_source = 'empty_trajectory_fallback'
                    trajectory_delta = None
            messages = self.predictions.pop(prediction_stamp)
            semantic = self.semantic_clouds.pop(semantic_stamp)
            self._evaluate(
                messages['probability'], messages['decision'], semantic,
                abs(prediction_stamp - semantic_stamp) * 1.0e-9,
                trajectory,
                trajectory_source,
                trajectory_delta,
            )

    @staticmethod
    def _semantic_arrays(message):
        values = point_cloud2.read_points(
            message,
            field_names=[
                'x', 'y', 'z', 'cos_incidence', 'object_idx', 'object_tag',
            ],
            skip_nans=True,
        )
        array = np.asarray(values)
        if not array.dtype.names:
            raise ValueError('semantic PointCloud2 must preserve named fields')
        xyz = np.column_stack([
            array['x'], array['y'], array['z'],
        ]).astype(np.float32, copy=False)
        tags = np.asarray(array['object_tag'], dtype=np.uint32)
        return xyz, tags

    def _grid_bev(self, message):
        width = int(message.info.width)
        height = int(message.info.height)
        expected_shape = (self.geometry.width, self.geometry.height)
        if (height, width) != expected_shape:
            raise ValueError(
                'grid shape {} does not match expected {}'.format(
                    (height, width), expected_shape
                )
            )
        resolution_error = abs(
            float(message.info.resolution) - self.geometry.resolution_m
        )
        if resolution_error > 1e-6:
            raise ValueError(
                'grid resolution does not match evaluator geometry'
            )
        values = np.asarray(message.data, dtype=np.int16)
        if values.size != width * height:
            raise ValueError('OccupancyGrid data length is inconsistent')
        return occupancy_grid_to_bev(values.reshape((height, width)))

    def _evaluate(
        self,
        probability_message,
        decision_message,
        semantic,
        delta,
        trajectory,
        trajectory_source,
        trajectory_delta,
    ):
        try:
            probability = self._grid_bev(probability_message)
            decision = self._grid_bev(decision_message)
            xyz, tags = self._semantic_arrays(semantic)
            targets = build_semantic_traversability_targets(
                xyz,
                tags,
                self.geometry,
                minimum_surface_points=int(self.get_parameter(
                    'minimum_surface_points'
                ).value),
                obstacle_vertical_span_m=float(self.get_parameter(
                    'obstacle_vertical_span_m'
                ).value),
                ego_footprint=self.footprint,
            )
            frame = self.accumulator.update(
                targets.labels, probability, decision
            )
            corridor_mask = build_swept_footprint_mask(
                self.geometry,
                trajectory,
                vehicle_front_m=self.corridor_front_m,
                vehicle_rear_m=self.corridor_rear_m,
                half_width_m=self.corridor_half_width_m,
            )
            corridor_frame = self.corridor_accumulator.update(
                targets.labels,
                probability,
                decision,
                evaluation_mask=corridor_mask,
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                'evaluation frame rejected: ' + str(error)
            )
            return

        self.frame_index += 1
        self.alignment_deltas.append(float(delta))
        selective_false_free_mask = (
            (targets.labels == 1) & (decision == 0)
        )
        raw_false_free_mask = (
            (targets.labels == 1)
            & (probability >= 0)
            & (probability < 50)
        )
        corridor_false_free_mask = (
            selective_false_free_mask & corridor_mask
        )
        component = largest_connected_component(
            corridor_false_free_mask
        )
        cluster_forward_span_m = (
            component['row_span_cells'] * self.geometry.resolution_m
        )
        cluster_lateral_span_m = (
            component['column_span_cells'] * self.geometry.resolution_m
        )
        if corridor_frame['selective_false_free'] > 0:
            self.consecutive_corridor_false_free_frames += 1
        else:
            self.consecutive_corridor_false_free_frames = 0
        self.maximum_consecutive_corridor_false_free_frames = max(
            self.maximum_consecutive_corridor_false_free_frames,
            self.consecutive_corridor_false_free_frames,
        )
        self.maximum_corridor_false_free_cluster_cells = max(
            self.maximum_corridor_false_free_cluster_cells,
            component['cell_count'],
        )
        self.maximum_corridor_false_free_forward_span_m = max(
            self.maximum_corridor_false_free_forward_span_m,
            cluster_forward_span_m,
        )
        self.maximum_corridor_false_free_lateral_span_m = max(
            self.maximum_corridor_false_free_lateral_span_m,
            cluster_lateral_span_m,
        )
        self.corridor_source_counts[trajectory_source] = (
            self.corridor_source_counts.get(trajectory_source, 0) + 1
        )
        row = dict(frame)
        row.update({
            'corridor_' + name: value
            for name, value in corridor_frame.items()
            if name in COUNT_FIELDS
        })
        row.update({
            'frame_index': self.frame_index,
            'wall_time_iso': datetime.now(
                timezone.utc
            ).astimezone().isoformat(),
            'source_stamp_s': _stamp_seconds(probability_message.header.stamp),
            'semantic_stamp_s': _stamp_seconds(semantic.header.stamp),
            'alignment_delta_s': float(delta),
            'route_index': self.route_index,
            'route_size': self.route_size,
            'corridor_source': trajectory_source,
            'corridor_alignment_delta_s': (
                trajectory_delta if trajectory_delta is not None else ''
            ),
            'corridor_path_points': int(trajectory.shape[0]),
            'corridor_cells': int(np.count_nonzero(corridor_mask)),
            'corridor_raw_false_free_rate': corridor_frame[
                'raw_false_free_rate'
            ],
            'corridor_raw_false_obstacle_rate': corridor_frame[
                'raw_false_obstacle_rate'
            ],
            'corridor_selective_coverage': corridor_frame[
                'selective_coverage'
            ],
            'corridor_selective_accepted_accuracy': corridor_frame[
                'selective_accepted_accuracy'
            ],
            'corridor_selective_false_free_rate': corridor_frame[
                'selective_false_free_rate'
            ],
            'corridor_selective_false_obstacle_rate': corridor_frame[
                'selective_false_obstacle_rate'
            ],
            'corridor_selective_safe_retention_rate': corridor_frame[
                'selective_safe_retention_rate'
            ],
            'corridor_false_free_cluster_cells': component['cell_count'],
            'corridor_false_free_cluster_forward_span_m': (
                cluster_forward_span_m
            ),
            'corridor_false_free_cluster_lateral_span_m': (
                cluster_lateral_span_m
            ),
            'consecutive_corridor_false_free_frames': (
                self.consecutive_corridor_false_free_frames
            ),
            'error_snapshot_file': '',
        })
        if np.any(selective_false_free_mask):
            row['error_snapshot_file'] = self._save_error_snapshot(
                row,
                targets,
                probability,
                decision,
                corridor_mask,
                raw_false_free_mask,
                selective_false_free_mask,
                trajectory,
            )
        if self.csv_writer is not None:
            self.csv_writer.writerow({
                name: row.get(name, '') for name in CSV_FIELDS
            })
            self.csv_file.flush()

        output = String()
        output.data = json.dumps({
            'mode': 'privileged_semantic_evaluation_only',
            'control_effect': 'none',
            'frame': row,
            'cumulative': self.accumulator.compute(),
            'corridor_cumulative': self.corridor_accumulator.compute(),
        }, sort_keys=True)
        self.status_publisher.publish(output)

    def _save_error_snapshot(
        self,
        row,
        targets,
        probability,
        decision,
        corridor_mask,
        raw_false_free_mask,
        selective_false_free_mask,
        trajectory,
    ):
        if (
            self.active_path is None
            or self.error_snapshot_count >= self.maximum_error_snapshots
        ):
            return ''
        self.error_snapshot_count += 1
        filename = 'frame_{:06d}_wp{:02d}_false_free.npz'.format(
            self.frame_index, self.route_index
        )
        relative = Path('error_snapshots') / filename
        destination = self.active_path / relative
        temporary = destination.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(
                stream,
                target_labels=targets.labels,
                target_free_point_count=targets.free_point_count,
                target_obstacle_point_count=targets.obstacle_point_count,
                target_vertical_span_m=targets.vertical_span_m,
                probability_percent=probability,
                selective_decision=decision,
                corridor_mask=corridor_mask,
                raw_false_free_mask=raw_false_free_mask,
                selective_false_free_mask=selective_false_free_mask,
                trajectory_xy_yaw=trajectory,
                status_json=np.asarray(json.dumps(row, sort_keys=True)),
            )
        temporary.replace(destination)
        return str(relative)

    def _on_route_status(self, message):
        status = str(message.data).strip().lower()
        previous = self.route_status
        self.route_status = status
        if status == 'navigating' and self.active_path is None:
            self._start_session()
        elif self.active_path is not None and status in {
            'completed', 'final_nav2_failed', 'invalid_route',
        }:
            self._close_session(status)
        elif (
            self.active_path is not None
            and status in {'idle', 'ready'}
            and previous not in {'idle', 'ready', 'waiting_for_goal_manager'}
        ):
            self._close_session('interrupted_' + status)

    def _start_session(self):
        now = datetime.now(timezone.utc).astimezone()
        name = now.strftime('evaluation_%Y%m%d_%H%M%S_%f')
        path = self.output_root / name
        path.mkdir(parents=True, exist_ok=False)
        (path / 'error_snapshots').mkdir()
        csv_file = (path / 'frames.csv').open(
            'w', newline='', encoding='utf-8'
        )
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        self.active_path = path
        self.csv_file = csv_file
        self.csv_writer = writer
        self.accumulator = OnlineTraversabilityAccumulator()
        self.corridor_accumulator = OnlineTraversabilityAccumulator()
        self.predictions.clear()
        self.semantic_clouds.clear()
        self.checked_trajectories.clear()
        self.alignment_deltas = []
        self.frame_index = 0
        self.dropped_prediction_frames = 0
        self.dropped_semantic_frames = 0
        self.started_at = now.isoformat()
        self.error_snapshot_count = 0
        self.consecutive_corridor_false_free_frames = 0
        self.maximum_consecutive_corridor_false_free_frames = 0
        self.maximum_corridor_false_free_cluster_cells = 0
        self.maximum_corridor_false_free_forward_span_m = 0.0
        self.maximum_corridor_false_free_lateral_span_m = 0.0
        self.corridor_source_counts = {}
        _write_json(path / 'metadata.json', {
            'schema_version': 1,
            'session': name,
            'started_at': self.started_at,
            'purpose': 'online shadow accuracy against CARLA semantic LiDAR',
            'privileged_input': '/lidar/semantic_points',
            'model_input_effect': 'none',
            'navigation_control_effect': 'none',
            'unit_of_analysis': (
                'directly supervised cell-observations across aligned frames'
            ),
            'maximum_semantic_alignment_s': (
                self.maximum_alignment_ns * 1.0e-9
            ),
            'target_builder': {
                'minimum_surface_points': int(self.get_parameter(
                    'minimum_surface_points'
                ).value),
                'obstacle_vertical_span_m': float(self.get_parameter(
                    'obstacle_vertical_span_m'
                ).value),
                'ego_rear_m': self.footprint.rear_m,
                'ego_front_m': self.footprint.front_m,
                'ego_half_width_m': self.footprint.half_width_m,
            },
            'bev_geometry': {
                'x_min_m': self.geometry.x_min_m,
                'x_max_m': self.geometry.x_max_m,
                'y_min_m': self.geometry.y_min_m,
                'y_max_m': self.geometry.y_max_m,
                'resolution_m': self.geometry.resolution_m,
                'z_min_m': self.geometry.z_min_m,
                'z_max_m': self.geometry.z_max_m,
            },
            'corridor': {
                'topic': str(self.get_parameter(
                    'checked_trajectory_topic'
                ).value),
                'vehicle_front_m': self.corridor_front_m,
                'vehicle_rear_m': self.corridor_rear_m,
                'half_width_m': self.corridor_half_width_m,
                'path_horizon_m': self.corridor_horizon_m,
                'fallback_spacing_m': self.corridor_spacing_m,
                'wait_s': self.corridor_wait_s,
                'maximum_trajectory_alignment_s': (
                    self.maximum_trajectory_alignment_ns * 1.0e-9
                ),
            },
            'error_snapshot_policy': {
                'trigger': 'any selective false-free cell in full BEV',
                'maximum_per_route': self.maximum_error_snapshots,
            },
            'navigation_metric_warning': (
                'The extra semantic sensor and evaluator add simulation '
                'load. Do not mix this run with navigation timing or success '
                'experiments.'
            ),
        })
        self.get_logger().info('Semantic evaluation started: ' + str(path))

    def _close_session(self, result):
        if self.active_path is None:
            return
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
        summary = self.accumulator.compute()
        summary.update({
            'result': result,
            'started_at': self.started_at,
            'ended_at': datetime.now(timezone.utc).astimezone().isoformat(),
            'route_index': self.route_index,
            'route_size': self.route_size,
            'timestamp_alignment': finite_alignment_summary(
                self.alignment_deltas
            ),
            'dropped_prediction_frames': self.dropped_prediction_frames,
            'dropped_semantic_frames': self.dropped_semantic_frames,
            'corridor': self.corridor_accumulator.compute(),
            'corridor_source_counts': self.corridor_source_counts,
            'error_snapshot_count': self.error_snapshot_count,
            'maximum_consecutive_corridor_false_free_frames': (
                self.maximum_consecutive_corridor_false_free_frames
            ),
            'maximum_corridor_false_free_cluster_cells': (
                self.maximum_corridor_false_free_cluster_cells
            ),
            'maximum_corridor_false_free_forward_span_m': (
                self.maximum_corridor_false_free_forward_span_m
            ),
            'maximum_corridor_false_free_lateral_span_m': (
                self.maximum_corridor_false_free_lateral_span_m
            ),
            'interpretation': (
                'Counts are repeated cell-observations, not independent map '
                'cells. Unknown targets are excluded; selective unknowns are '
                'reported as abstentions and are never counted as free.'
            ),
        })
        _write_json(self.active_path / 'summary.json', summary)
        self.get_logger().info(
            'Semantic evaluation closed: result=%s frames=%d '
            'false_free=%d path=%s' % (
                result, summary['frames'], summary['selective_false_free'],
                self.active_path,
            )
        )
        self.active_path = None
        self.csv_file = None
        self.csv_writer = None
        self.predictions.clear()
        self.semantic_clouds.clear()
        self.checked_trajectories.clear()

    def close(self):
        if self.active_path is not None:
            self._close_session('node_shutdown')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityShadowEvaluator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
