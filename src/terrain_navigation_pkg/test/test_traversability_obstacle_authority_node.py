import json

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from terrain_navigation_pkg.traversability_inference_core import (
    bev_to_occupancy_grid,
)
from terrain_navigation_pkg.traversability_obstacle_authority_node import (
    TraversabilityObstacleAuthorityNode,
)


class _CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _header(seconds=10):
    header = Header()
    header.frame_id = 'lidar_frame'
    header.stamp.sec = int(seconds)
    return header


def _cloud(points, seconds=10):
    return point_cloud2.create_cloud_xyz32(_header(seconds), points)


def _decision(seconds=10):
    bev = np.full((4, 4), -1, dtype=np.int8)
    bev[1, 1] = 0
    bev[2, 2] = 100
    stored = bev_to_occupancy_grid(bev)
    message = OccupancyGrid()
    message.header = _header(seconds)
    message.info.width = int(stored.shape[1])
    message.info.height = int(stored.shape[0])
    message.info.resolution = 1.0
    message.info.origin.position.x = 0.0
    message.info.origin.position.y = -2.0
    message.info.origin.orientation.w = 1.0
    message.data = stored.ravel().tolist()
    return message


def test_add_only_falls_back_then_enriches_without_baseline_deletion():
    rclpy.init(args=[
        '--ros-args',
        '-p', 'authority_mode:=add_only',
        '-p', 'bev_x_min_m:=0.0',
        '-p', 'bev_x_max_m:=4.0',
        '-p', 'bev_y_min_m:=-2.0',
        '-p', 'bev_y_max_m:=2.0',
        '-p', 'bev_resolution_m:=1.0',
        '-p', 'minimum_range_m:=0.1',
        '-p', 'maximum_range_m:=10.0',
        '-p', 'obstacle_maximum_z_m:=1.0',
        '-p', 'obstacle_cell_top_band_m:=0.05',
        '-p', 'addition_minimum_relative_height_m:=0.07',
        '-p', 'addition_local_ground_radius_m:=1.0',
        '-p', 'addition_local_ground_minimum_support_cells:=1',
        '-p', 'voxel_size_m:=0.1',
        '-p', 'ego_front_m:=0.1',
        '-p', 'ego_rear_m:=0.1',
        '-p', 'ego_half_width_m:=0.1',
    ])
    node = None
    try:
        node = TraversabilityObstacleAuthorityNode()
        selected = _CapturePublisher()
        candidate = _CapturePublisher()
        added = _CapturePublisher()
        statuses = _CapturePublisher()
        node.selected_publisher = selected
        node.candidate_publisher = candidate
        node.added_publisher = added
        node.status_publisher = statuses

        baseline = _cloud([
            (2.5, 0.5, 0.0),
            (0.5, 1.5, 0.0),
        ])
        raw = _cloud([
            (2.5, 0.5, 0.0),
            (1.5, -0.5, -1.6),
            (1.5, -0.5, -1.2),
            (0.5, 1.5, 0.0),
        ])

        node._on_baseline(baseline)
        assert selected.messages == [baseline]
        assert len(added.messages) == 1
        assert added.messages[-1].width == 0
        assert added.messages[-1].header.stamp == baseline.header.stamp
        first_status = json.loads(statuses.messages[-1].data)
        assert first_status['fallback_reason'] == 'ai_not_ready'
        node._on_raw(raw)
        node._on_decision(_decision())

        assert len(selected.messages) == 2
        assert len(candidate.messages) == 1
        assert len(added.messages) == 2
        enriched = selected.messages[-1]
        points = list(point_cloud2.read_points(
            enriched, field_names=('x', 'y', 'z'), skip_nans=True
        ))
        assert len(points) == 3
        assert [(p[0], p[1], p[2]) for p in points[:2]] == [
            (2.5, 0.5, 0.0),
            (0.5, 1.5, 0.0),
        ]
        assert tuple(round(value, 1) for value in points[2]) == (
            1.5, -0.5, -1.2
        )
        added_points = list(point_cloud2.read_points(
            added.messages[-1],
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        ))
        assert len(added_points) == 1
        assert tuple(round(value, 1) for value in added_points[0]) == (
            1.5, -0.5, -1.2
        )
        assert added.messages[-1].header == raw.header
        assert added.messages[-1].data == enriched.data[
            2 * enriched.point_step:
        ]
        status = json.loads(statuses.messages[-1].data)
        assert status['phase'] == 'ai_enriched'
        assert status['baseline_removed_point_count'] == 0
        assert status['baseline_point_count'] == 2
        assert status['raw_model_obstacle_point_count'] == 1
        assert status['ground_gate_rejected_point_count'] == 0
        assert status['added_point_count'] == 1
        assert status['navigation_control_effect'] == 'add_obstacles_only'
        assert status['safety_control_effect'] == (
            'none_raw_lidar_independent'
        )
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_add_only_returns_to_baseline_after_ai_silence():
    rclpy.init(args=[
        '--ros-args',
        '-p', 'authority_mode:=add_only',
        '-p', 'maximum_ai_silence_s:=0.1',
    ])
    node = None
    try:
        node = TraversabilityObstacleAuthorityNode()
        selected = _CapturePublisher()
        added = _CapturePublisher()
        statuses = _CapturePublisher()
        node.selected_publisher = selected
        node.added_publisher = added
        node.status_publisher = statuses
        node.last_successful_ai_wall_s = 0.0
        baseline = _cloud([(3.0, 2.0, 0.0)], seconds=20)
        node._on_baseline(baseline)
        assert selected.messages == [baseline]
        assert len(added.messages) == 1
        assert added.messages[-1].width == 0
        assert bytes(added.messages[-1].data) == b''
        status = json.loads(statuses.messages[-1].data)
        assert status['fallback_reason'] == 'ai_stale'
        assert status['baseline_removed_point_count'] == 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_passive_candidate_is_not_reported_as_actual_addition():
    rclpy.init(args=[
        '--ros-args',
        '-p', 'authority_mode:=passive',
        '-p', 'bev_x_min_m:=0.0',
        '-p', 'bev_x_max_m:=4.0',
        '-p', 'bev_y_min_m:=-2.0',
        '-p', 'bev_y_max_m:=2.0',
        '-p', 'bev_resolution_m:=1.0',
        '-p', 'minimum_range_m:=0.1',
        '-p', 'maximum_range_m:=10.0',
        '-p', 'obstacle_maximum_z_m:=1.0',
        '-p', 'addition_local_ground_radius_m:=1.0',
        '-p', 'addition_local_ground_minimum_support_cells:=1',
        '-p', 'voxel_size_m:=0.1',
        '-p', 'ego_front_m:=0.1',
        '-p', 'ego_rear_m:=0.1',
        '-p', 'ego_half_width_m:=0.1',
    ])
    node = None
    try:
        node = TraversabilityObstacleAuthorityNode()
        selected = _CapturePublisher()
        candidate = _CapturePublisher()
        added = _CapturePublisher()
        statuses = _CapturePublisher()
        node.selected_publisher = selected
        node.candidate_publisher = candidate
        node.added_publisher = added
        node.status_publisher = statuses

        baseline = _cloud([(2.5, 0.5, 0.0)])
        raw = _cloud([
            (2.5, 0.5, 0.0),
            (1.5, -0.5, -1.6),
            (1.5, -0.5, -1.2),
        ])
        node._on_baseline(baseline)
        node._on_raw(raw)
        node._on_decision(_decision())

        assert selected.messages == [baseline]
        assert len(candidate.messages) == 1
        assert candidate.messages[-1].width == 2
        assert len(added.messages) == 2
        assert all(message.width == 0 for message in added.messages)
        status = json.loads(statuses.messages[-1].data)
        assert status['phase'] == 'ai_enriched'
        assert status['added_point_count'] == 1
        assert status['navigation_control_effect'] == 'baseline_only'
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
