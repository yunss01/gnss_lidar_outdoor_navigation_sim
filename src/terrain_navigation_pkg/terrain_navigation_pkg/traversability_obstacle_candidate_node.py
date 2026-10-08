#!/usr/bin/env python3
"""Publish a passive AI/baseline obstacle-cloud candidate with no authority."""

from __future__ import annotations

import json

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String

from .navigation_learning_recorder_core import BevGeometry, build_lidar_bev
from .traversability_inference_core import occupancy_grid_to_bev
from .traversability_obstacle_fusion_core import (
    build_candidate_fusion_masks,
    voxel_unique_indices,
)


def _stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class TraversabilityObstacleCandidateNode(Node):
    """Fuse learned decisions with baseline points on an isolated topic."""

    def __init__(self):
        super().__init__('traversability_obstacle_candidate_node')
        self.declare_parameter('raw_cloud_topic', '/lidar/points')
        self.declare_parameter(
            'baseline_cloud_topic', '/lidar/nav2_obstacles'
        )
        self.declare_parameter(
            'decision_topic', '/learning/traversability_shadow_decision'
        )
        self.declare_parameter(
            'candidate_cloud_topic',
            '/learning/nav2_obstacles_candidate',
        )
        self.declare_parameter(
            'status_topic', '/learning/obstacle_candidate_status'
        )
        self.declare_parameter('cache_size', 32)
        self.declare_parameter('minimum_range_m', 1.5)
        self.declare_parameter('maximum_range_m', 20.0)
        self.declare_parameter('obstacle_maximum_z_m', 1.0)
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
        self.declare_parameter('hard_obstacle_minimum_z_m', -1.40)
        self.declare_parameter('hard_obstacle_vertical_span_m', 0.15)

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
        self.minimum_range_m = float(
            self.get_parameter('minimum_range_m').value
        )
        self.maximum_range_m = float(
            self.get_parameter('maximum_range_m').value
        )
        self.obstacle_maximum_z_m = float(
            self.get_parameter('obstacle_maximum_z_m').value
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
        self.hard_obstacle_minimum_z_m = float(
            self.get_parameter('hard_obstacle_minimum_z_m').value
        )
        self.hard_obstacle_vertical_span_m = float(self.get_parameter(
            'hard_obstacle_vertical_span_m'
        ).value)
        if self.cache_size < 3:
            raise ValueError('cache_size must be at least 3')
        if not 0.0 <= self.minimum_range_m < self.maximum_range_m:
            raise ValueError('LiDAR range limits are invalid')
        if self.voxel_size_m <= 0.0:
            raise ValueError('voxel_size_m must be positive')

        self.raw_clouds = {}
        self.baseline_clouds = {}
        self.decisions = {}
        self.emitted_stamps = set()
        self.cache_evictions = {
            'raw': 0,
            'baseline': 0,
            'decision': 0,
        }
        self.frame_sequence = 0

        self.candidate_publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter('candidate_cloud_topic').value),
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
            lambda message: self._store('raw', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('baseline_cloud_topic').value),
            lambda message: self._store('baseline', message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('decision_topic').value),
            lambda message: self._store('decision', message),
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            'Passive AI obstacle candidate ready: output={} has NO Nav2 or '
            'safety authority'.format(
                self.get_parameter('candidate_cloud_topic').value
            )
        )

    def _cache(self, kind):
        return {
            'raw': self.raw_clouds,
            'baseline': self.baseline_clouds,
            'decision': self.decisions,
        }[kind]

    def _store(self, kind, message):
        stamp = _stamp_ns(message)
        cache = self._cache(kind)
        cache[stamp] = message
        while len(cache) > self.cache_size:
            oldest = min(cache)
            del cache[oldest]
            self.cache_evictions[kind] += 1
        self._try_emit(stamp)

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

    def _try_emit(self, stamp):
        if stamp in self.emitted_stamps:
            return
        if not all(
            stamp in cache
            for cache in (
                self.raw_clouds,
                self.baseline_clouds,
                self.decisions,
            )
        ):
            return
        raw = self.raw_clouds.pop(stamp)
        baseline = self.baseline_clouds.pop(stamp)
        decision_message = self.decisions.pop(stamp)
        self.emitted_stamps.add(stamp)
        if len(self.emitted_stamps) > 4 * self.cache_size:
            self.emitted_stamps = set(sorted(self.emitted_stamps)[-32:])
        try:
            self._publish_candidate(raw, baseline, decision_message)
        except (ValueError, RuntimeError) as error:
            self.get_logger().error(
                'candidate fusion failed for stamp {}: {}'.format(
                    stamp, error
                )
            )

    def _publish_candidate(self, raw, baseline, decision_message):
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
        masks = build_candidate_fusion_masks(
            raw_xyz,
            baseline_xyz,
            decision,
            self.geometry,
            minimum_range_m=self.minimum_range_m,
            maximum_range_m=self.maximum_range_m,
            obstacle_maximum_z_m=self.obstacle_maximum_z_m,
            ego_front_m=self.ego_front_m,
            ego_rear_m=self.ego_rear_m,
            ego_half_width_m=self.ego_half_width_m,
            hard_obstacle_minimum_z_m=(
                self.hard_obstacle_minimum_z_m
            ),
            hard_obstacle_vertical_span_m=(
                self.hard_obstacle_vertical_span_m
            ),
        )
        baseline_records = self._records(baseline)[
            masks.baseline_keep_mask
        ]
        raw_records = self._records(raw)[masks.raw_ai_obstacle_mask]
        combined_records = np.concatenate(
            (baseline_records, raw_records), axis=0
        )
        combined_xyz = np.concatenate((
            baseline_xyz[masks.baseline_keep_mask],
            raw_xyz[masks.raw_ai_obstacle_mask],
        ), axis=0)
        unique = voxel_unique_indices(combined_xyz, self.voxel_size_m)
        candidate_records = np.ascontiguousarray(combined_records[unique])
        candidate_xyz = combined_xyz[unique]
        baseline_occupied_cells = (
            build_lidar_bev(baseline_xyz, self.geometry)[0] > 0.5
        )
        candidate_occupied_cells = (
            build_lidar_bev(candidate_xyz, self.geometry)[0] > 0.5
        )

        output = PointCloud2()
        output.header = raw.header
        output.height = 1
        output.width = int(candidate_records.shape[0])
        output.fields = raw.fields
        output.is_bigendian = raw.is_bigendian
        output.point_step = raw.point_step
        output.row_step = output.point_step * output.width
        output.is_dense = True
        output.data = candidate_records.tobytes()
        self.candidate_publisher.publish(output)

        self.frame_sequence += 1
        status = {
            'mode': 'passive_candidate_no_navigation_authority',
            'frame_sequence': self.frame_sequence,
            'source_stamp_ns': _stamp_ns(raw),
            'raw_point_count': int(raw_xyz.shape[0]),
            'baseline_point_count': int(baseline_xyz.shape[0]),
            'baseline_kept_count': int(np.count_nonzero(
                masks.baseline_keep_mask
            )),
            'baseline_cleared_by_ai_count': int(np.count_nonzero(
                masks.baseline_cleared_by_ai_mask
            )),
            'raw_ai_obstacle_point_count': int(np.count_nonzero(
                masks.raw_ai_obstacle_mask
            )),
            'candidate_point_count': int(output.width),
            'baseline_occupied_cell_count': int(np.count_nonzero(
                baseline_occupied_cells
            )),
            'candidate_occupied_cell_count': int(np.count_nonzero(
                candidate_occupied_cells
            )),
            'candidate_added_cell_count': int(np.count_nonzero(
                candidate_occupied_cells & ~baseline_occupied_cells
            )),
            'candidate_removed_cell_count': int(np.count_nonzero(
                baseline_occupied_cells & ~candidate_occupied_cells
            )),
            'ai_free_cell_count': int(np.count_nonzero(decision == 0)),
            'ai_obstacle_cell_count': int(np.count_nonzero(
                decision == 100
            )),
            'ai_unknown_cell_count': int(np.count_nonzero(decision < 0)),
            'cache_evictions': dict(self.cache_evictions),
            'navigation_control_effect': 'none',
            'safety_control_effect': 'none',
            'unknown_policy': 'retain_baseline',
            'ai_change_minimum_range_m': self.minimum_range_m,
            'ai_change_maximum_range_m': self.maximum_range_m,
        }
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self.status_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TraversabilityObstacleCandidateNode()
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
