#!/usr/bin/env python3
"""Select the Nav2 obstacle cloud with fail-safe baseline fallback."""

from __future__ import annotations

import json
import time

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String

from .navigation_learning_recorder_core import BevGeometry
from .traversability_inference_core import occupancy_grid_to_bev
from .traversability_obstacle_fusion_core import (
    build_add_only_fusion_masks,
    novel_voxel_indices,
)


VALID_AUTHORITY_MODES = frozenset({'baseline', 'passive', 'add_only'})


def _stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class TraversabilityObstacleAuthorityNode(Node):
    """Publish one selected obstacle cloud; never weaken the baseline."""

    def __init__(self):
        super().__init__('traversability_obstacle_authority_node')
        self.declare_parameter('authority_mode', 'baseline')
        self.declare_parameter('raw_cloud_topic', '/lidar/points')
        self.declare_parameter(
            'baseline_cloud_topic', '/lidar/nav2_obstacles_baseline'
        )
        self.declare_parameter(
            'decision_topic', '/learning/evidence_v2/decision'
        )
        self.declare_parameter(
            'selected_cloud_topic', '/lidar/nav2_obstacles'
        )
        self.declare_parameter(
            'candidate_cloud_topic',
            '/learning/evidence_v2/nav2_obstacles_candidate',
        )
        self.declare_parameter(
            'added_cloud_topic',
            '/learning/evidence_v2/added_obstacles',
        )
        self.declare_parameter(
            'status_topic', '/learning/evidence_v2/authority_status'
        )
        self.declare_parameter('cache_size', 32)
        self.declare_parameter('maximum_ai_silence_s', 0.50)
        self.declare_parameter('maximum_exact_match_delay_s', 0.50)
        self.declare_parameter('minimum_range_m', 1.5)
        self.declare_parameter('maximum_range_m', 12.0)
        self.declare_parameter('obstacle_maximum_z_m', 1.0)
        self.declare_parameter('obstacle_cell_top_band_m', 0.05)
        self.declare_parameter('addition_minimum_relative_height_m', 0.07)
        self.declare_parameter('addition_local_ground_radius_m', 0.75)
        self.declare_parameter('addition_local_ground_quantile', 0.25)
        self.declare_parameter(
            'addition_local_ground_minimum_support_cells', 4
        )
        self.declare_parameter('voxel_size_m', 0.20)
        self.declare_parameter('ego_front_m', 2.55)
        self.declare_parameter('ego_rear_m', 2.65)
        self.declare_parameter('ego_half_width_m', 1.15)
        self.declare_parameter('bev_x_min_m', -10.0)
        self.declare_parameter('bev_x_max_m', 30.0)
        self.declare_parameter('bev_y_min_m', -20.0)
        self.declare_parameter('bev_y_max_m', 20.0)
        self.declare_parameter('bev_resolution_m', 0.25)
        self.declare_parameter('bev_z_min_m', -2.0)
        self.declare_parameter('bev_z_max_m', 3.0)

        self.authority_mode = str(
            self.get_parameter('authority_mode').value
        ).strip().lower()
        if self.authority_mode not in VALID_AUTHORITY_MODES:
            raise ValueError(
                'authority_mode must be one of: '
                + ', '.join(sorted(VALID_AUTHORITY_MODES))
            )
        self.geometry = BevGeometry(
            x_min_m=float(self.get_parameter('bev_x_min_m').value),
            x_max_m=float(self.get_parameter('bev_x_max_m').value),
            y_min_m=float(self.get_parameter('bev_y_min_m').value),
            y_max_m=float(self.get_parameter('bev_y_max_m').value),
            resolution_m=float(
                self.get_parameter('bev_resolution_m').value
            ),
            z_min_m=float(self.get_parameter('bev_z_min_m').value),
            z_max_m=float(self.get_parameter('bev_z_max_m').value),
        )
        self.cache_size = int(self.get_parameter('cache_size').value)
        self.maximum_ai_silence_s = float(
            self.get_parameter('maximum_ai_silence_s').value
        )
        self.maximum_exact_match_delay_s = float(
            self.get_parameter('maximum_exact_match_delay_s').value
        )
        self.minimum_range_m = float(
            self.get_parameter('minimum_range_m').value
        )
        self.maximum_range_m = float(
            self.get_parameter('maximum_range_m').value
        )
        self.obstacle_maximum_z_m = float(
            self.get_parameter('obstacle_maximum_z_m').value
        )
        self.obstacle_cell_top_band_m = float(
            self.get_parameter('obstacle_cell_top_band_m').value
        )
        self.addition_minimum_relative_height_m = float(
            self.get_parameter(
                'addition_minimum_relative_height_m'
            ).value
        )
        self.addition_local_ground_radius_m = float(
            self.get_parameter('addition_local_ground_radius_m').value
        )
        self.addition_local_ground_quantile = float(
            self.get_parameter('addition_local_ground_quantile').value
        )
        self.addition_local_ground_minimum_support_cells = int(
            self.get_parameter(
                'addition_local_ground_minimum_support_cells'
            ).value
        )
        self.voxel_size_m = float(
            self.get_parameter('voxel_size_m').value
        )
        self.ego_front_m = float(
            self.get_parameter('ego_front_m').value
        )
        self.ego_rear_m = float(self.get_parameter('ego_rear_m').value)
        self.ego_half_width_m = float(
            self.get_parameter('ego_half_width_m').value
        )
        if self.cache_size < 3:
            raise ValueError('cache_size must be at least 3')
        if self.maximum_ai_silence_s <= 0.0:
            raise ValueError('maximum_ai_silence_s must be positive')
        if self.maximum_exact_match_delay_s <= 0.0:
            raise ValueError('maximum_exact_match_delay_s must be positive')
        if not 0.0 <= self.minimum_range_m < self.maximum_range_m:
            raise ValueError('LiDAR range limits are invalid')
        if self.voxel_size_m <= 0.0:
            raise ValueError('voxel_size_m must be positive')
        if self.addition_minimum_relative_height_m <= 0.0:
            raise ValueError(
                'addition_minimum_relative_height_m must be positive'
            )
        if self.addition_local_ground_radius_m <= 0.0:
            raise ValueError(
                'addition_local_ground_radius_m must be positive'
            )
        if not 0.0 <= self.addition_local_ground_quantile <= 1.0:
            raise ValueError(
                'addition_local_ground_quantile must be in [0, 1]'
            )
        if self.addition_local_ground_minimum_support_cells < 1:
            raise ValueError(
                'addition_local_ground_minimum_support_cells must be '
                'positive'
            )

        self.raw_clouds = {}
        self.baseline_clouds = {}
        self.decisions = {}
        self.enriched_stamps = set()
        self.last_successful_ai_wall_s = None
        self.selected_sequence = 0
        self.enriched_sequence = 0
        self.cache_evictions = {'raw': 0, 'baseline': 0, 'decision': 0}

        self.selected_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('selected_cloud_topic').value),
            qos_profile_sensor_data,
        )
        self.candidate_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('candidate_cloud_topic').value),
            qos_profile_sensor_data,
        )
        self.added_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('added_cloud_topic').value),
            qos_profile_sensor_data,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter('status_topic').value),
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('raw_cloud_topic').value),
            self._on_raw,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('baseline_cloud_topic').value),
            self._on_baseline,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('decision_topic').value),
            self._on_decision,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'Obstacle authority ready: mode={}, baseline={}, selected={}; '
            'AI add range={:.1f} m, local-ground rise>={:.2f} m; '
            'baseline deletion is impossible'.format(
                self.authority_mode,
                self.get_parameter('baseline_cloud_topic').value,
                self.get_parameter('selected_cloud_topic').value,
                self.maximum_range_m,
                self.addition_minimum_relative_height_m,
            )
        )

    def _cache(self, kind):
        return {
            'raw': self.raw_clouds,
            'baseline': self.baseline_clouds,
            'decision': self.decisions,
        }[kind]

    def _store(self, kind, message, received_s):
        cache = self._cache(kind)
        cache[_stamp_ns(message)] = (message, float(received_s))
        while len(cache) > self.cache_size:
            oldest = min(cache)
            del cache[oldest]
            self.cache_evictions[kind] += 1

    def _ai_is_fresh(self, now_s):
        return (
            self.last_successful_ai_wall_s is not None
            and float(now_s) - self.last_successful_ai_wall_s
            <= self.maximum_ai_silence_s
        )

    def _on_raw(self, message):
        now_s = time.monotonic()
        self._store('raw', message, now_s)
        self._try_enriched(_stamp_ns(message), now_s)

    def _on_baseline(self, message):
        now_s = time.monotonic()
        stamp = _stamp_ns(message)
        self._store('baseline', message, now_s)
        if self.authority_mode in {'baseline', 'passive'}:
            self._publish_baseline(message, 'configured_baseline')
        elif not self._ai_is_fresh(now_s):
            reason = (
                'ai_not_ready'
                if self.last_successful_ai_wall_s is None
                else 'ai_stale'
            )
            self._publish_baseline(message, reason)
        self._try_enriched(stamp, now_s)

    def _on_decision(self, message):
        now_s = time.monotonic()
        try:
            self._decision_bev(message)
        except ValueError as error:
            self.get_logger().error(
                'rejected invalid evidence decision: ' + str(error)
            )
            return
        self._store('decision', message, now_s)
        self._try_enriched(_stamp_ns(message), now_s)

    @staticmethod
    def _field_signature(message):
        return tuple(
            (field.name, field.offset, field.datatype, field.count)
            for field in message.fields
        )

    @staticmethod
    def _float32_view(message, name):
        field = next(
            (item for item in message.fields if item.name == name), None
        )
        if field is None or field.datatype != PointField.FLOAT32:
            raise ValueError('{} FLOAT32 field is required'.format(name))
        endian = '>f4' if message.is_bigendian else '<f4'
        count = int(message.width) * int(message.height)
        return np.ndarray(
            shape=(count,),
            dtype=np.dtype(endian),
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
        ))

    @staticmethod
    def _records(message):
        count = int(message.width) * int(message.height)
        if count == 0:
            return np.empty((0, int(message.point_step)), dtype=np.uint8)
        return np.frombuffer(message.data, dtype=np.uint8).reshape(
            count, int(message.point_step)
        )

    def _decision_bev(self, message):
        expected_width = self.geometry.height
        expected_height = self.geometry.width
        if (
            int(message.info.width) != expected_width
            or int(message.info.height) != expected_height
        ):
            raise ValueError('decision grid dimensions do not match BEV')
        if not np.isclose(
            float(message.info.resolution), self.geometry.resolution_m
        ):
            raise ValueError('decision grid resolution does not match BEV')
        if not np.isclose(
            float(message.info.origin.position.x), self.geometry.x_min_m
        ) or not np.isclose(
            float(message.info.origin.position.y), self.geometry.y_min_m
        ):
            raise ValueError('decision grid origin does not match BEV')
        expected = expected_width * expected_height
        if len(message.data) != expected:
            raise ValueError('decision grid data length is invalid')
        occupancy = np.asarray(message.data, dtype=np.int8).reshape(
            (expected_height, expected_width)
        )
        return occupancy_grid_to_bev(occupancy)

    def _try_enriched(self, stamp, now_s):
        if self.authority_mode == 'baseline':
            return
        if stamp in self.enriched_stamps:
            return
        if not all(
            stamp in cache
            for cache in (
                self.raw_clouds, self.baseline_clouds, self.decisions
            )
        ):
            return
        raw, raw_received_s = self.raw_clouds.pop(stamp)
        baseline, baseline_received_s = self.baseline_clouds.pop(stamp)
        decision, decision_received_s = self.decisions.pop(stamp)
        delay_s = max(
            raw_received_s, baseline_received_s, decision_received_s
        ) - min(raw_received_s, baseline_received_s, decision_received_s)
        if delay_s > self.maximum_exact_match_delay_s:
            if self.authority_mode == 'add_only':
                self._publish_baseline(baseline, 'exact_match_too_late')
            return
        try:
            output, added_output, counts = self._build_add_only(
                raw, baseline, decision
            )
        except (ValueError, RuntimeError) as error:
            self.get_logger().error(
                'add-only fusion failed for stamp {}: {}'.format(
                    stamp, error
                )
            )
            if self.authority_mode == 'add_only':
                self._publish_baseline(baseline, 'fusion_error')
            return

        self.enriched_stamps.add(stamp)
        if len(self.enriched_stamps) > 4 * self.cache_size:
            self.enriched_stamps = set(
                sorted(self.enriched_stamps)[-self.cache_size:]
            )
        self.last_successful_ai_wall_s = float(now_s)
        self.candidate_publisher.publish(output)
        if self.authority_mode == 'add_only':
            self.added_publisher.publish(added_output)
            self.selected_publisher.publish(output)
            self.selected_sequence += 1
        else:
            self.added_publisher.publish(
                self._empty_cloud_like(baseline)
            )
        self.enriched_sequence += 1
        self._publish_status({
            'phase': 'ai_enriched',
            'fallback_reason': '',
            'source_stamp_ns': stamp,
            'exact_match_delay_s': delay_s,
            'baseline_point_count': counts['baseline_point_count'],
            'raw_ai_obstacle_point_count': (
                counts['raw_ai_obstacle_point_count']
            ),
            'raw_model_obstacle_point_count': (
                counts['raw_model_obstacle_point_count']
            ),
            'ground_gate_rejected_point_count': (
                counts['ground_gate_rejected_point_count']
            ),
            'added_point_count': counts['added_point_count'],
            'selected_point_count': counts['selected_point_count'],
            'baseline_removed_point_count': 0,
            'selected_output_published': (
                self.authority_mode == 'add_only'
            ),
            'candidate_output_published': True,
            'ai_output_usable': True,
        })

    def _build_add_only(self, raw, baseline, decision_message):
        if raw.header.frame_id != baseline.header.frame_id:
            raise ValueError('raw and baseline frames do not match')
        if raw.header.frame_id != decision_message.header.frame_id:
            raise ValueError('raw and decision frames do not match')
        if raw.point_step != baseline.point_step:
            raise ValueError('raw and baseline point steps do not match')
        if raw.is_bigendian != baseline.is_bigendian:
            raise ValueError('raw and baseline endianness does not match')
        if self._field_signature(raw) != self._field_signature(baseline):
            raise ValueError('raw and baseline point fields do not match')

        raw_xyz = self._xyz(raw)
        baseline_xyz = self._xyz(baseline)
        decision = self._decision_bev(decision_message)
        masks = build_add_only_fusion_masks(
            raw_xyz,
            decision,
            self.geometry,
            minimum_range_m=self.minimum_range_m,
            maximum_range_m=self.maximum_range_m,
            obstacle_maximum_z_m=self.obstacle_maximum_z_m,
            ego_front_m=self.ego_front_m,
            ego_rear_m=self.ego_rear_m,
            ego_half_width_m=self.ego_half_width_m,
            obstacle_cell_top_band_m=self.obstacle_cell_top_band_m,
            minimum_relative_height_m=(
                self.addition_minimum_relative_height_m
            ),
            local_ground_radius_m=self.addition_local_ground_radius_m,
            local_ground_quantile=self.addition_local_ground_quantile,
            local_ground_minimum_support_cells=(
                self.addition_local_ground_minimum_support_cells
            ),
        )
        candidate_indices = np.flatnonzero(masks.raw_ai_obstacle_mask)
        raw_candidates_xyz = raw_xyz[candidate_indices]
        novel = novel_voxel_indices(
            baseline_xyz, raw_candidates_xyz, self.voxel_size_m
        )
        add_indices = candidate_indices[novel]
        baseline_records = self._records(baseline)
        added_records = self._records(raw)[add_indices]
        selected_records = np.ascontiguousarray(np.concatenate(
            (baseline_records, added_records), axis=0
        ))

        output = self._cloud_from_records(
            baseline,
            selected_records,
            is_dense=bool(baseline.is_dense and raw.is_dense),
        )
        added_output = self._cloud_from_records(
            raw,
            np.ascontiguousarray(added_records),
            is_dense=bool(raw.is_dense),
        )
        return output, added_output, {
            'baseline_point_count': int(baseline_records.shape[0]),
            'raw_model_obstacle_point_count': int(np.count_nonzero(
                masks.raw_obstacle_before_ground_gate_mask
            )),
            'ground_gate_rejected_point_count': int(np.count_nonzero(
                masks.raw_ground_gate_rejected_mask
            )),
            'raw_ai_obstacle_point_count': int(candidate_indices.size),
            'added_point_count': int(add_indices.size),
            'selected_point_count': int(output.width),
        }

    @staticmethod
    def _cloud_from_records(template, records, *, is_dense):
        record_array = np.ascontiguousarray(records, dtype=np.uint8)
        output = PointCloud2()
        output.header = template.header
        output.height = 1
        output.width = int(record_array.shape[0])
        output.fields = template.fields
        output.is_bigendian = template.is_bigendian
        output.point_step = template.point_step
        output.row_step = output.point_step * output.width
        output.is_dense = bool(is_dense)
        output.data = record_array.tobytes()
        return output

    @classmethod
    def _empty_cloud_like(cls, template):
        records = np.empty((0, int(template.point_step)), dtype=np.uint8)
        return cls._cloud_from_records(
            template,
            records,
            is_dense=bool(template.is_dense),
        )

    def _publish_baseline(self, baseline, reason):
        # Clear the diagnostic contribution cloud on every fallback so RViz
        # cannot retain additions from a previously enriched frame.
        self.added_publisher.publish(self._empty_cloud_like(baseline))
        self.selected_publisher.publish(baseline)
        self.selected_sequence += 1
        ai_age = (
            None
            if self.last_successful_ai_wall_s is None
            else max(0.0, time.monotonic() - self.last_successful_ai_wall_s)
        )
        self._publish_status({
            'phase': 'baseline_fallback',
            'fallback_reason': str(reason),
            'source_stamp_ns': _stamp_ns(baseline),
            'baseline_point_count': int(baseline.width * baseline.height),
            'raw_model_obstacle_point_count': 0,
            'ground_gate_rejected_point_count': 0,
            'raw_ai_obstacle_point_count': 0,
            'added_point_count': 0,
            'selected_point_count': int(baseline.width * baseline.height),
            'baseline_removed_point_count': 0,
            'selected_output_published': True,
            'candidate_output_published': False,
            'ai_output_usable': self._ai_is_fresh(time.monotonic()),
            'last_successful_ai_age_s': ai_age,
        })

    def _publish_status(self, values):
        status = {
            'authority_mode': self.authority_mode,
            'navigation_control_effect': (
                'add_obstacles_only'
                if self.authority_mode == 'add_only'
                else 'baseline_only'
            ),
            'safety_control_effect': 'none_raw_lidar_independent',
            'unknown_policy': 'retain_baseline',
            'passable_policy': 'retain_baseline',
            'baseline_deletion_allowed': False,
            'selected_sequence': self.selected_sequence,
            'enriched_sequence': self.enriched_sequence,
            'cache_evictions': dict(self.cache_evictions),
            'maximum_ai_silence_s': self.maximum_ai_silence_s,
            'addition_maximum_range_m': self.maximum_range_m,
            'addition_minimum_relative_height_m': (
                self.addition_minimum_relative_height_m
            ),
        }
        status.update(values)
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self.status_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityObstacleAuthorityNode()
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
