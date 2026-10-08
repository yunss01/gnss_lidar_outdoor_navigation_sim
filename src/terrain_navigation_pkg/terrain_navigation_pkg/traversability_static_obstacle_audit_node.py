#!/usr/bin/env python3
"""Record a fixed-duration, no-authority static-obstacle candidate audit."""

from __future__ import annotations

from collections import OrderedDict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import String

from .navigation_learning_recorder_core import BevGeometry, build_lidar_bev
from .traversability_inference_core import occupancy_grid_to_bev
from .traversability_learning_core import (
    EgoFootprint,
    build_semantic_traversability_targets,
)
from .traversability_online_evaluation_core import (
    finite_alignment_summary,
    largest_connected_component,
)
from .traversability_static_obstacle_audit_core import (
    COUNT_FIELDS,
    RATE_FIELDS,
    StaticObstacleAuditAccumulator,
    comparison_masks,
    force_controlled_obstacle_tag,
    semantic_obstacle_instance_metrics,
)


CSV_FIELDS = (
    'frame_index',
    'wall_time_iso',
    'source_stamp_s',
    'semantic_stamp_s',
    'semantic_alignment_delta_s',
    'status_attached',
    'diagnostic_attached',
) + COUNT_FIELDS + RATE_FIELDS + (
    'removed_obstacle_cluster_cells',
    'removed_obstacle_forward_span_m',
    'removed_obstacle_lateral_span_m',
    'added_free_cluster_cells',
    'added_free_forward_span_m',
    'added_free_lateral_span_m',
    'status_baseline_point_count',
    'status_candidate_point_count',
    'status_candidate_added_cell_count',
    'status_candidate_removed_cell_count',
    'corridor_obstacle_instances',
    'corridor_baseline_detected_instances',
    'corridor_candidate_detected_instances',
    'corridor_fully_removed_instances',
)


INSTANCE_CSV_FIELDS = (
    'frame_index',
    'source_stamp_s',
    'object_idx',
    'object_tag',
    'target_cells',
    'baseline_hits',
    'candidate_hits',
    'baseline_only_cells',
    'candidate_only_cells',
    'baseline_detected',
    'candidate_detected',
    'fully_removed',
    'candidate_to_baseline_retention',
    'minimum_forward_m',
    'maximum_forward_m',
    'minimum_left_m',
    'maximum_left_m',
)


DIAGNOSTIC_NAMES = (
    'probability',
    'entropy',
    'variance',
    'hard_mask',
    'decision',
)


def _stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _stamp_s(stamp_ns):
    return float(stamp_ns) * 1.0e-9


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    temporary.replace(path)


def _safe_label(value):
    label = re.sub(r'[^A-Za-z0-9가-힣_.-]+', '_', str(value)).strip('_')
    return label or 'unlabeled'


class TraversabilityStaticObstacleAuditNode(Node):
    """Compare baseline and candidate clouds against semantic supervision."""

    def __init__(self):
        super().__init__('traversability_static_obstacle_audit_node')
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
        self.duration_s = float(self.get_parameter('duration_s').value)
        self.warmup_s = float(self.get_parameter('warmup_s').value)
        self.maximum_alignment_ns = int(round(
            float(self.get_parameter('maximum_semantic_alignment_s').value)
            * 1.0e9
        ))
        self.cache_size = int(self.get_parameter('cache_size').value)
        self.maximum_status_wait_s = float(
            self.get_parameter('maximum_status_wait_s').value
        )
        self.minimum_frames = int(self.get_parameter('minimum_frames').value)
        self.minimum_candidate_rate_hz = float(
            self.get_parameter('minimum_candidate_rate_hz').value
        )
        self.maximum_candidate_amplification_ratio = float(
            self.get_parameter(
                'maximum_candidate_amplification_ratio'
            ).value
        )
        self.minimum_diagnostic_attached_fraction = float(
            self.get_parameter(
                'minimum_diagnostic_attached_fraction'
            ).value
        )
        self.instance_corridor_minimum_x_m = float(self.get_parameter(
            'instance_corridor_minimum_x_m'
        ).value)
        self.instance_corridor_maximum_x_m = float(self.get_parameter(
            'instance_corridor_maximum_x_m'
        ).value)
        self.instance_corridor_half_width_m = float(self.get_parameter(
            'instance_corridor_half_width_m'
        ).value)
        self.controlled_obstacle_actor_id = int(self.get_parameter(
            'controlled_obstacle_actor_id'
        ).value)
        if min(
            self.duration_s,
            self.maximum_alignment_ns,
            self.cache_size,
            self.maximum_status_wait_s,
            self.minimum_frames,
            self.minimum_candidate_rate_hz,
            self.maximum_candidate_amplification_ratio,
        ) <= 0:
            raise ValueError('audit duration, limits, and rates must be positive')
        if not 0.0 <= self.warmup_s < self.duration_s:
            raise ValueError('warmup_s must be in [0, duration_s)')
        if not 0.0 <= self.minimum_diagnostic_attached_fraction <= 1.0:
            raise ValueError(
                'minimum_diagnostic_attached_fraction must be in [0, 1]'
            )
        if not (
            self.instance_corridor_minimum_x_m
            < self.instance_corridor_maximum_x_m
            and self.instance_corridor_half_width_m > 0.0
        ):
            raise ValueError('instance corridor limits are invalid')

        self.started_wall = datetime.now(timezone.utc).astimezone()
        self.started_monotonic = time.monotonic()
        self.scenario_label = _safe_label(
            self.get_parameter('scenario_label').value
        )
        output_root = Path(str(
            self.get_parameter('output_directory').value
        )).expanduser()
        session_name = '{}_{}'.format(
            self.started_wall.strftime('static_audit_%Y%m%d_%H%M%S_%f'),
            self.scenario_label,
        )
        self.output_path = output_root / session_name
        self.output_path.mkdir(parents=True, exist_ok=False)
        (self.output_path / 'snapshots').mkdir()
        self.csv_file = (self.output_path / 'frames.csv').open(
            'w', newline='', encoding='utf-8'
        )
        self.csv_writer = csv.DictWriter(
            self.csv_file, fieldnames=CSV_FIELDS
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()
        self.instance_csv_file = (
            self.output_path / 'instances.csv'
        ).open('w', newline='', encoding='utf-8')
        self.instance_csv_writer = csv.DictWriter(
            self.instance_csv_file, fieldnames=INSTANCE_CSV_FIELDS
        )
        self.instance_csv_writer.writeheader()
        self.instance_csv_file.flush()

        self.raw_clouds = OrderedDict()
        self.baseline_clouds = OrderedDict()
        self.candidate_clouds = OrderedDict()
        self.candidate_arrival = {}
        self.semantic_clouds = OrderedDict()
        self.status_by_stamp = OrderedDict()
        self.diagnostics = {
            name: OrderedDict() for name in DIAGNOSTIC_NAMES
        }
        self.cache_evictions = {
            'raw': 0,
            'baseline': 0,
            'candidate': 0,
            'semantic': 0,
            'status': 0,
            **{name: 0 for name in DIAGNOSTIC_NAMES},
        }
        self.received_candidates_after_warmup = 0
        self.frame_index = 0
        self.status_attached_frames = 0
        self.diagnostic_attached_frames = 0
        self.status_authority_violations = 0
        self.semantic_alignment_deltas = []
        self.accumulator = StaticObstacleAuditAccumulator()
        self.maximum_occupied_ratio = 0.0
        self.maximum_removed_obstacle_cluster = {
            'cell_count': 0, 'row_span_cells': 0, 'column_span_cells': 0,
        }
        self.maximum_added_free_cluster = {
            'cell_count': 0, 'row_span_cells': 0, 'column_span_cells': 0,
        }
        self.best_snapshot_scores = {
            'representative': -1,
            'worst_removed_obstacle': -1,
            'worst_added_free': -1,
        }
        self.instance_observations = 0
        self.baseline_detected_instance_observations = 0
        self.candidate_detected_instance_observations = 0
        self.fully_removed_instance_observations = 0
        self.fully_removed_object_ids = set()
        self.minimum_instance_retention = None
        self.controlled_actor_frames_observed = 0
        self.controlled_actor_point_observations = 0
        self.finished = False

        self._write_metadata()
        self._create_subscriptions()
        self.match_timer = self.create_timer(0.05, self._on_timer)
        self.get_logger().info(
            'Static obstacle audit started: label={} duration={:.1f}s '
            'warmup={:.1f}s; candidate has NO control authority; output={}'
            .format(
                self.scenario_label,
                self.duration_s,
                self.warmup_s,
                self.output_path,
            )
        )

    def _declare_parameters(self):
        defaults = {
            'scenario_label': 'unlabeled',
            'obstacle_type': 'unspecified',
            'obstacle_distance_m': -1.0,
            'obstacle_lateral_position': 'unspecified',
            'map_name': 'Town10HD_Opt',
            'operator_note': '',
            'duration_s': 20.0,
            'warmup_s': 3.0,
            'output_directory': (
                '~/terrain_nav_data/learning/static_obstacle_audits'
            ),
            'baseline_cloud_topic': '/lidar/nav2_obstacles',
            'raw_cloud_topic': '/lidar/points',
            'candidate_cloud_topic': (
                '/learning/nav2_obstacles_candidate'
            ),
            'semantic_cloud_topic': '/lidar/semantic_points',
            'probability_float_topic': (
                '/learning/traversability_probability_float'
            ),
            'entropy_float_topic': (
                '/learning/traversability_entropy_float'
            ),
            'mc_variance_float_topic': (
                '/learning/traversability_mc_variance_float'
            ),
            'hard_obstacle_mask_topic': (
                '/learning/traversability_hard_obstacle_mask'
            ),
            'decision_topic': (
                '/learning/traversability_shadow_decision'
            ),
            'candidate_status_topic': (
                '/learning/obstacle_candidate_status'
            ),
            'maximum_semantic_alignment_s': 0.03,
            'maximum_status_wait_s': 0.15,
            'cache_size': 64,
            'minimum_frames': 40,
            'minimum_candidate_rate_hz': 4.0,
            'maximum_candidate_amplification_ratio': 1.25,
            'minimum_diagnostic_attached_fraction': 0.95,
            'instance_corridor_minimum_x_m': 0.0,
            'instance_corridor_maximum_x_m': 15.0,
            'instance_corridor_half_width_m': 4.0,
            'controlled_obstacle_actor_id': -1,
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

    def _write_metadata(self):
        names = [
            'scenario_label', 'obstacle_type', 'obstacle_distance_m',
            'obstacle_lateral_position', 'map_name', 'operator_note',
            'duration_s', 'warmup_s', 'baseline_cloud_topic',
            'raw_cloud_topic', 'candidate_cloud_topic',
            'semantic_cloud_topic', 'probability_float_topic',
            'entropy_float_topic', 'mc_variance_float_topic',
            'hard_obstacle_mask_topic', 'decision_topic',
            'candidate_status_topic', 'maximum_semantic_alignment_s',
            'minimum_frames', 'minimum_candidate_rate_hz',
            'maximum_candidate_amplification_ratio',
            'minimum_diagnostic_attached_fraction',
            'instance_corridor_minimum_x_m',
            'instance_corridor_maximum_x_m',
            'instance_corridor_half_width_m',
            'controlled_obstacle_actor_id',
        ]
        metadata = {
            'schema_version': 3,
            'purpose': 'no-authority static obstacle candidate audit',
            'started_at': self.started_wall.isoformat(),
            'control_effect': 'none; recorder and candidate are passive',
            'unit_of_analysis': (
                'repeated semantic cell-observations across aligned frames'
            ),
            'parameters': {
                name: self.get_parameter(name).value for name in names
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
            'provisional_gate_warning': (
                'Rate and amplification thresholds are engineering gates for '
                'the first passive audit, not validated safety guarantees.'
            ),
        }
        _write_json(self.output_path / 'metadata.json', metadata)

    def _create_subscriptions(self):
        def topic(parameter):
            return str(self.get_parameter(parameter).value)

        self.create_subscription(
            PointCloud2,
            topic('raw_cloud_topic'),
            lambda message: self._store_cloud('raw', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            topic('baseline_cloud_topic'),
            lambda message: self._store_cloud('baseline', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            topic('candidate_cloud_topic'),
            lambda message: self._store_cloud('candidate', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            topic('semantic_cloud_topic'),
            lambda message: self._store_cloud('semantic', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            topic('candidate_status_topic'),
            self._store_status,
            10,
        )
        for name, parameter in (
            ('probability', 'probability_float_topic'),
            ('entropy', 'entropy_float_topic'),
            ('variance', 'mc_variance_float_topic'),
            ('hard_mask', 'hard_obstacle_mask_topic'),
        ):
            self.create_subscription(
                Image,
                topic(parameter),
                lambda message, key=name: self._store_diagnostic(
                    key, message
                ),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            OccupancyGrid,
            topic('decision_topic'),
            lambda message: self._store_diagnostic('decision', message),
            qos_profile_sensor_data,
        )

    def _cache(self, kind):
        return {
            'raw': self.raw_clouds,
            'baseline': self.baseline_clouds,
            'candidate': self.candidate_clouds,
            'semantic': self.semantic_clouds,
        }[kind]

    def _store_cloud(self, kind, message):
        if self.finished:
            return
        stamp = _stamp_ns(message)
        cache = self._cache(kind)
        cache[stamp] = message
        cache.move_to_end(stamp)
        if kind == 'candidate':
            self.candidate_arrival[stamp] = time.monotonic()
            if self._recording_active():
                self.received_candidates_after_warmup += 1
        self._trim(cache, kind)

    def _store_diagnostic(self, kind, message):
        if self.finished:
            return
        stamp = _stamp_ns(message)
        cache = self.diagnostics[kind]
        cache[stamp] = message
        cache.move_to_end(stamp)
        self._trim(cache, kind)

    def _store_status(self, message):
        if self.finished:
            return
        try:
            value = json.loads(message.data)
            stamp = int(value['source_stamp_ns'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning('invalid candidate status message')
            return
        self.status_by_stamp[stamp] = value
        self.status_by_stamp.move_to_end(stamp)
        self._trim(self.status_by_stamp, 'status')

    def _trim(self, cache, kind):
        while len(cache) > self.cache_size:
            stamp, _ = cache.popitem(last=False)
            if kind == 'candidate':
                self.candidate_arrival.pop(stamp, None)
            self.cache_evictions[kind] += 1

    def _recording_active(self):
        elapsed = time.monotonic() - self.started_monotonic
        return self.warmup_s <= elapsed < self.duration_s

    def _nearest_semantic_stamp(self, source_stamp):
        if source_stamp in self.semantic_clouds:
            return source_stamp
        if not self.semantic_clouds:
            return None
        stamp = min(
            self.semantic_clouds,
            key=lambda value: abs(value - source_stamp),
        )
        if abs(stamp - source_stamp) <= self.maximum_alignment_ns:
            return stamp
        return None

    def _on_timer(self):
        if self.finished:
            return
        self._try_matches()
        if time.monotonic() - self.started_monotonic >= self.duration_s:
            self._finish('completed_duration')

    def _try_matches(self):
        now = time.monotonic()
        for stamp in list(self.candidate_clouds):
            if stamp not in self.baseline_clouds:
                continue
            semantic_stamp = self._nearest_semantic_stamp(stamp)
            if semantic_stamp is None:
                continue
            arrival = self.candidate_arrival.get(stamp, now)
            diagnostics_ready = (
                stamp in self.raw_clouds
                and all(
                    stamp in self.diagnostics[name]
                    for name in DIAGNOSTIC_NAMES
                )
            )
            if (
                (
                    stamp not in self.status_by_stamp
                    or not diagnostics_ready
                )
                and now - arrival < self.maximum_status_wait_s
            ):
                continue
            raw = self.raw_clouds.pop(stamp, None)
            baseline = self.baseline_clouds.pop(stamp)
            candidate = self.candidate_clouds.pop(stamp)
            semantic = self.semantic_clouds.pop(semantic_stamp)
            status = self.status_by_stamp.pop(stamp, None)
            diagnostics = {
                name: self.diagnostics[name].pop(stamp, None)
                for name in DIAGNOSTIC_NAMES
            }
            self.candidate_arrival.pop(stamp, None)
            arrival_elapsed = arrival - self.started_monotonic
            if self.warmup_s <= arrival_elapsed < self.duration_s:
                self._evaluate(
                    stamp, semantic_stamp, baseline, candidate, semantic,
                    status, raw, diagnostics,
                )

    @staticmethod
    def _float32_view(message, name):
        field = next(
            (item for item in message.fields if item.name == name), None
        )
        if field is None or field.datatype != PointField.FLOAT32:
            raise ValueError('{} FLOAT32 field is required'.format(name))
        dtype = np.dtype('>f4' if message.is_bigendian else '<f4')
        count = int(message.width) * int(message.height)
        return np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=message.data,
            offset=int(field.offset),
            strides=(int(message.point_step),),
        )

    @staticmethod
    def _uint32_view(message, name):
        field = next(
            (item for item in message.fields if item.name == name), None
        )
        if field is None or field.datatype != PointField.UINT32:
            raise ValueError('{} UINT32 field is required'.format(name))
        dtype = np.dtype('>u4' if message.is_bigendian else '<u4')
        count = int(message.width) * int(message.height)
        return np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=message.data,
            offset=int(field.offset),
            strides=(int(message.point_step),),
        )

    @classmethod
    def _xyz(cls, message):
        return np.column_stack((
            cls._float32_view(message, 'x'),
            cls._float32_view(message, 'y'),
            cls._float32_view(message, 'z'),
        )).astype(np.float32, copy=False)

    @classmethod
    def _semantic_arrays(cls, message):
        xyz = cls._xyz(message)
        indices = cls._uint32_view(message, 'object_idx')
        tags = cls._uint32_view(message, 'object_tag')
        finite = np.isfinite(xyz).all(axis=1)
        return xyz[finite], indices[finite], tags[finite]

    def _image_bev(self, message, *, mask=False):
        shape = (int(message.height), int(message.width))
        expected = (self.geometry.height, self.geometry.width)
        if shape != expected:
            raise ValueError(
                'diagnostic image shape does not match BEV geometry'
            )
        if mask:
            if message.encoding != 'mono8' or int(message.step) != shape[1]:
                raise ValueError('hard mask image must be packed mono8')
            dtype = np.dtype('u1')
        else:
            if (
                message.encoding != '32FC1'
                or int(message.step) != shape[1] * 4
            ):
                raise ValueError(
                    'diagnostic float image must be packed 32FC1'
                )
            endian = '>' if message.is_bigendian else '<'
            dtype = np.dtype(endian + 'f4')
        value = np.frombuffer(message.data, dtype=dtype)
        if value.size != shape[0] * shape[1]:
            raise ValueError('diagnostic image data length is inconsistent')
        return value.reshape(shape).copy()

    def _decision_bev(self, message):
        width = int(message.info.width)
        height = int(message.info.height)
        if (height, width) != (
            self.geometry.width, self.geometry.height
        ):
            raise ValueError('decision grid shape does not match geometry')
        if abs(
            float(message.info.resolution) - self.geometry.resolution_m
        ) > 1.0e-6:
            raise ValueError('decision grid resolution does not match')
        values = np.asarray(message.data, dtype=np.int16)
        if values.size != width * height:
            raise ValueError('decision grid data length is inconsistent')
        return occupancy_grid_to_bev(
            values.reshape((height, width))
        ).astype(np.int8, copy=False)

    def _evaluate(
        self,
        source_stamp,
        semantic_stamp,
        baseline_message,
        candidate_message,
        semantic_message,
        status,
        raw_message,
        diagnostic_messages,
    ):
        try:
            baseline_xyz = self._xyz(baseline_message)
            candidate_xyz = self._xyz(candidate_message)
            semantic_xyz, semantic_indices, semantic_tags = (
                self._semantic_arrays(semantic_message)
            )
            effective_semantic_tags, controlled_point_count = (
                force_controlled_obstacle_tag(
                    semantic_indices,
                    semantic_tags,
                    self.controlled_obstacle_actor_id,
                )
            )
            if controlled_point_count > 0:
                self.controlled_actor_frames_observed += 1
                self.controlled_actor_point_observations += (
                    controlled_point_count
                )
            targets = build_semantic_traversability_targets(
                semantic_xyz,
                effective_semantic_tags,
                self.geometry,
                minimum_surface_points=int(self.get_parameter(
                    'minimum_surface_points'
                ).value),
                obstacle_vertical_span_m=float(self.get_parameter(
                    'obstacle_vertical_span_m'
                ).value),
                ego_footprint=self.footprint,
            )
            baseline_occupied = (
                build_lidar_bev(baseline_xyz, self.geometry)[0] > 0.5
            )
            candidate_occupied = (
                build_lidar_bev(candidate_xyz, self.geometry)[0] > 0.5
            )
            instances = semantic_obstacle_instance_metrics(
                semantic_xyz,
                semantic_indices,
                effective_semantic_tags,
                targets.labels,
                baseline_occupied,
                candidate_occupied,
                self.geometry,
                corridor_minimum_x_m=(
                    self.instance_corridor_minimum_x_m
                ),
                corridor_maximum_x_m=(
                    self.instance_corridor_maximum_x_m
                ),
                corridor_half_width_m=(
                    self.instance_corridor_half_width_m
                ),
            )
            frame = self.accumulator.update(
                targets.labels, baseline_occupied, candidate_occupied
            )
            masks = comparison_masks(
                targets.labels, baseline_occupied, candidate_occupied
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning('audit frame rejected: ' + str(error))
            return

        raw_xyz = None
        raw_bev = None
        diagnostics = None
        if raw_message is not None and all(
            diagnostic_messages.get(name) is not None
            for name in DIAGNOSTIC_NAMES
        ):
            try:
                raw_xyz = self._xyz(raw_message)
                raw_bev = build_lidar_bev(raw_xyz, self.geometry)
                diagnostics = {
                    'probability': self._image_bev(
                        diagnostic_messages['probability']
                    ),
                    'entropy': self._image_bev(
                        diagnostic_messages['entropy']
                    ),
                    'variance': self._image_bev(
                        diagnostic_messages['variance']
                    ),
                    'hard_mask': self._image_bev(
                        diagnostic_messages['hard_mask'], mask=True
                    ).astype(bool),
                    'decision': self._decision_bev(
                        diagnostic_messages['decision']
                    ),
                }
            except (TypeError, ValueError) as error:
                self.get_logger().warning(
                    'audit diagnostics rejected: ' + str(error)
                )

        diagnostic_attached = diagnostics is not None
        if diagnostic_attached:
            self.diagnostic_attached_frames += 1

        self.frame_index += 1
        alignment_delta_s = abs(source_stamp - semantic_stamp) * 1.0e-9
        self.semantic_alignment_deltas.append(alignment_delta_s)
        ratio = frame['candidate_to_baseline_occupied_ratio']
        if ratio is not None:
            self.maximum_occupied_ratio = max(
                self.maximum_occupied_ratio, float(ratio)
            )

        removed_component = largest_connected_component(
            masks['removed_target_obstacle']
        )
        added_free_component = largest_connected_component(
            masks['added_target_free']
        )
        if (
            removed_component['cell_count']
            > self.maximum_removed_obstacle_cluster['cell_count']
        ):
            self.maximum_removed_obstacle_cluster = removed_component
        if (
            added_free_component['cell_count']
            > self.maximum_added_free_cluster['cell_count']
        ):
            self.maximum_added_free_cluster = added_free_component

        if status is not None:
            self.status_attached_frames += 1
            if (
                status.get('navigation_control_effect') != 'none'
                or status.get('safety_control_effect') != 'none'
            ):
                self.status_authority_violations += 1

        baseline_detected_instances = sum(
            item['baseline_detected'] for item in instances
        )
        candidate_detected_instances = sum(
            item['candidate_detected'] for item in instances
        )
        fully_removed_instances = sum(
            item['fully_removed'] for item in instances
        )
        self.instance_observations += len(instances)
        self.baseline_detected_instance_observations += (
            baseline_detected_instances
        )
        self.candidate_detected_instance_observations += (
            candidate_detected_instances
        )
        self.fully_removed_instance_observations += fully_removed_instances
        for item in instances:
            if item['fully_removed']:
                self.fully_removed_object_ids.add(item['object_idx'])
            retention = item['candidate_to_baseline_retention']
            if retention is not None:
                if self.minimum_instance_retention is None:
                    self.minimum_instance_retention = retention
                else:
                    self.minimum_instance_retention = min(
                        self.minimum_instance_retention, retention
                    )

        row = {
            'frame_index': self.frame_index,
            'wall_time_iso': datetime.now(
                timezone.utc
            ).astimezone().isoformat(),
            'source_stamp_s': _stamp_s(source_stamp),
            'semantic_stamp_s': _stamp_s(semantic_stamp),
            'semantic_alignment_delta_s': alignment_delta_s,
            'status_attached': int(status is not None),
            'diagnostic_attached': int(diagnostic_attached),
            **frame,
            'removed_obstacle_cluster_cells': removed_component[
                'cell_count'
            ],
            'removed_obstacle_forward_span_m': (
                removed_component['row_span_cells']
                * self.geometry.resolution_m
            ),
            'removed_obstacle_lateral_span_m': (
                removed_component['column_span_cells']
                * self.geometry.resolution_m
            ),
            'added_free_cluster_cells': added_free_component['cell_count'],
            'added_free_forward_span_m': (
                added_free_component['row_span_cells']
                * self.geometry.resolution_m
            ),
            'added_free_lateral_span_m': (
                added_free_component['column_span_cells']
                * self.geometry.resolution_m
            ),
            'status_baseline_point_count': (
                status.get('baseline_point_count', '') if status else ''
            ),
            'status_candidate_point_count': (
                status.get('candidate_point_count', '') if status else ''
            ),
            'status_candidate_added_cell_count': (
                status.get('candidate_added_cell_count', '')
                if status else ''
            ),
            'status_candidate_removed_cell_count': (
                status.get('candidate_removed_cell_count', '')
                if status else ''
            ),
            'corridor_obstacle_instances': len(instances),
            'corridor_baseline_detected_instances': (
                baseline_detected_instances
            ),
            'corridor_candidate_detected_instances': (
                candidate_detected_instances
            ),
            'corridor_fully_removed_instances': fully_removed_instances,
        }
        self.csv_writer.writerow({
            name: row.get(name, '') for name in CSV_FIELDS
        })
        self.csv_file.flush()
        for item in instances:
            instance_row = {
                'frame_index': self.frame_index,
                'source_stamp_s': _stamp_s(source_stamp),
                **item,
            }
            self.instance_csv_writer.writerow({
                name: instance_row.get(name, '')
                for name in INSTANCE_CSV_FIELDS
            })
        self.instance_csv_file.flush()
        self._update_snapshots(
            row,
            targets,
            baseline_occupied,
            candidate_occupied,
            masks,
            raw_xyz,
            raw_bev,
            semantic_xyz,
            semantic_indices,
            semantic_tags,
            effective_semantic_tags,
            diagnostics,
            instances,
        )

    def _update_snapshots(
        self,
        row,
        targets,
        baseline,
        candidate,
        masks,
        raw_xyz,
        raw_bev,
        semantic_xyz,
        semantic_indices,
        semantic_tags,
        effective_semantic_tags,
        diagnostics,
        instances,
    ):
        scores = {
            'representative': 1 if self.frame_index == 1 else 0,
            'worst_removed_obstacle': int(row['removed_target_obstacle']),
            'worst_added_free': int(row['added_target_free']),
        }
        for name, score in scores.items():
            if score <= self.best_snapshot_scores[name]:
                continue
            if name != 'representative' and score <= 0:
                continue
            self.best_snapshot_scores[name] = score
            destination = self.output_path / 'snapshots' / (name + '.npz')
            temporary = destination.with_suffix('.npz.tmp')
            payload = {
                'target_labels': targets.labels.astype(np.int8),
                'target_free_point_count': (
                    targets.free_point_count.astype(np.int32)
                ),
                'target_obstacle_point_count': (
                    targets.obstacle_point_count.astype(np.int32)
                ),
                'target_observed_point_count': (
                    targets.observed_point_count.astype(np.int32)
                ),
                'target_vertical_span_m': (
                    targets.vertical_span_m.astype(np.float32)
                ),
                'baseline_occupied': baseline.astype(np.uint8),
                'candidate_occupied': candidate.astype(np.uint8),
                'removed_target_obstacle': masks[
                    'removed_target_obstacle'
                ].astype(np.uint8),
                'removed_target_free': masks[
                    'removed_target_free'
                ].astype(np.uint8),
                'added_target_obstacle': masks[
                    'added_target_obstacle'
                ].astype(np.uint8),
                'added_target_free': masks[
                    'added_target_free'
                ].astype(np.uint8),
                'raw_points_xyz': (
                    raw_xyz.astype(np.float32)
                    if raw_xyz is not None
                    else np.empty((0, 3), dtype=np.float32)
                ),
                'raw_lidar_bev': (
                    raw_bev.astype(np.float32)
                    if raw_bev is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                'semantic_points_xyz': semantic_xyz.astype(np.float32),
                'semantic_object_idx': semantic_indices.astype(np.uint32),
                'semantic_object_tag': semantic_tags.astype(np.uint32),
                'semantic_effective_object_tag': (
                    effective_semantic_tags.astype(np.uint32)
                ),
                'model_obstacle_probability': (
                    diagnostics['probability'].astype(np.float32)
                    if diagnostics is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                'model_predictive_entropy': (
                    diagnostics['entropy'].astype(np.float32)
                    if diagnostics is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                'model_mc_variance': (
                    diagnostics['variance'].astype(np.float32)
                    if diagnostics is not None
                    else np.empty((0,), dtype=np.float32)
                ),
                'hard_obstacle_mask': (
                    diagnostics['hard_mask'].astype(np.uint8)
                    if diagnostics is not None
                    else np.empty((0,), dtype=np.uint8)
                ),
                'selective_decision': (
                    diagnostics['decision'].astype(np.int8)
                    if diagnostics is not None
                    else np.empty((0,), dtype=np.int8)
                ),
                'instance_json': np.asarray(json.dumps(
                    instances, sort_keys=True
                )),
                'frame_json': np.asarray(json.dumps(row, sort_keys=True)),
            }
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **payload)
            temporary.replace(destination)

    def _finish(self, result):
        if self.finished:
            return
        self.finished = True
        elapsed_s = max(0.0, time.monotonic() - self.started_monotonic)
        effective_duration_s = max(0.0, elapsed_s - self.warmup_s)
        summary = self.accumulator.compute()
        candidate_rate_hz = (
            float(self.frame_index) / effective_duration_s
            if effective_duration_s > 0.0 else 0.0
        )
        alignment = finite_alignment_summary(
            self.semantic_alignment_deltas
        )
        status_fraction = (
            float(self.status_attached_frames) / float(self.frame_index)
            if self.frame_index else 0.0
        )
        diagnostic_fraction = (
            float(self.diagnostic_attached_frames) / float(self.frame_index)
            if self.frame_index else 0.0
        )
        gates = {
            'minimum_frames': self.frame_index >= self.minimum_frames,
            'candidate_rate': (
                candidate_rate_hz >= self.minimum_candidate_rate_hz
            ),
            'semantic_alignment': (
                alignment['maximum_s'] is not None
                and alignment['maximum_s']
                <= float(self.get_parameter(
                    'maximum_semantic_alignment_s'
                ).value)
            ),
            'hard_obstacle_false_clearing': (
                summary['removed_target_obstacle'] == 0
            ),
            'candidate_amplification': (
                self.maximum_occupied_ratio
                <= self.maximum_candidate_amplification_ratio
            ),
            'status_attached': status_fraction >= 0.95,
            'diagnostics_attached': (
                diagnostic_fraction
                >= self.minimum_diagnostic_attached_fraction
            ),
            'no_complete_obstacle_instance_loss': (
                self.fully_removed_instance_observations == 0
            ),
            'obstacle_instances_observed': (
                self.baseline_detected_instance_observations > 0
            ),
            'no_control_authority': (
                self.status_authority_violations == 0
                and self.status_attached_frames > 0
            ),
            'candidate_cache_drop': self.cache_evictions['candidate'] == 0,
            'controlled_obstacle_observed': (
                self.controlled_obstacle_actor_id < 0
                or self.controlled_actor_frames_observed > 0
            ),
        }
        summary.update({
            'result': result,
            'scenario_label': self.scenario_label,
            'started_at': self.started_wall.isoformat(),
            'ended_at': datetime.now(
                timezone.utc
            ).astimezone().isoformat(),
            'elapsed_s': elapsed_s,
            'effective_duration_s': effective_duration_s,
            'received_candidates_after_warmup': (
                self.received_candidates_after_warmup
            ),
            'matched_frames': self.frame_index,
            'candidate_rate_hz': candidate_rate_hz,
            'status_attached_frames': self.status_attached_frames,
            'status_attached_fraction': status_fraction,
            'diagnostic_attached_frames': (
                self.diagnostic_attached_frames
            ),
            'diagnostic_attached_fraction': diagnostic_fraction,
            'status_authority_violations': (
                self.status_authority_violations
            ),
            'semantic_alignment': alignment,
            'cache_evictions': dict(self.cache_evictions),
            'maximum_candidate_to_baseline_occupied_ratio': (
                self.maximum_occupied_ratio
            ),
            'maximum_removed_obstacle_cluster_cells': (
                self.maximum_removed_obstacle_cluster['cell_count']
            ),
            'maximum_removed_obstacle_cluster_forward_span_m': (
                self.maximum_removed_obstacle_cluster['row_span_cells']
                * self.geometry.resolution_m
            ),
            'maximum_removed_obstacle_cluster_lateral_span_m': (
                self.maximum_removed_obstacle_cluster['column_span_cells']
                * self.geometry.resolution_m
            ),
            'maximum_added_free_cluster_cells': (
                self.maximum_added_free_cluster['cell_count']
            ),
            'instance_evaluation': {
                'corridor_minimum_x_m': (
                    self.instance_corridor_minimum_x_m
                ),
                'corridor_maximum_x_m': (
                    self.instance_corridor_maximum_x_m
                ),
                'corridor_half_width_m': (
                    self.instance_corridor_half_width_m
                ),
                'instance_observations': self.instance_observations,
                'baseline_detected_instance_observations': (
                    self.baseline_detected_instance_observations
                ),
                'candidate_detected_instance_observations': (
                    self.candidate_detected_instance_observations
                ),
                'fully_removed_instance_observations': (
                    self.fully_removed_instance_observations
                ),
                'fully_removed_object_ids': sorted(
                    self.fully_removed_object_ids
                ),
                'minimum_candidate_to_baseline_retention': (
                    self.minimum_instance_retention
                ),
            },
            'controlled_obstacle_evaluation': {
                'actor_id': self.controlled_obstacle_actor_id,
                'frames_observed': self.controlled_actor_frames_observed,
                'point_observations': (
                    self.controlled_actor_point_observations
                ),
                'tag_override': (
                    'Static props (20)'
                    if self.controlled_obstacle_actor_id >= 0 else None
                ),
            },
            'gates': gates,
            'pass': bool(all(gates.values())),
            'interpretation': (
                'Counts are repeated cell-observations, not independent '
                'physical obstacles. This passive audit does not grant Nav2 '
                'or safety authority to the candidate cloud.'
            ),
        })
        self.csv_file.flush()
        self.csv_file.close()
        self.instance_csv_file.flush()
        self.instance_csv_file.close()
        _write_json(self.output_path / 'summary.json', summary)
        self.get_logger().info(
            'Static obstacle audit finished: frames={} rate={:.2f}Hz '
            'removed_target_obstacle={} max_ratio={:.3f} pass={} output={}'
            .format(
                self.frame_index,
                candidate_rate_hz,
                summary['removed_target_obstacle'],
                self.maximum_occupied_ratio,
                summary['pass'],
                self.output_path,
            )
        )

    def destroy_node(self):
        if not self.finished:
            self._finish('node_shutdown')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityStaticObstacleAuditNode()
        # Keep ownership of the process lifetime in main.  Calling
        # rclpy.shutdown() from a timer callback can leave spin() waiting even
        # after the audit has written its summary.  A bounded spin_once loop
        # lets the callback mark the audit complete and then exits cleanly.
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
