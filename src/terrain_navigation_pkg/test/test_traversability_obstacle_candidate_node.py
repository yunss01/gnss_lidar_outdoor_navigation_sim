import json

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from terrain_navigation_pkg.traversability_inference_core import (
    bev_to_occupancy_grid,
)
from terrain_navigation_pkg.traversability_obstacle_candidate_node import (
    TraversabilityObstacleCandidateNode,
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


def test_candidate_node_matches_stamp_and_has_no_control_authority():
    rclpy.init(args=[
        '--ros-args',
        '-p', 'bev_x_min_m:=0.0',
        '-p', 'bev_x_max_m:=4.0',
        '-p', 'bev_y_min_m:=-2.0',
        '-p', 'bev_y_max_m:=2.0',
        '-p', 'bev_resolution_m:=1.0',
        '-p', 'minimum_range_m:=0.1',
        '-p', 'maximum_range_m:=10.0',
        '-p', 'obstacle_maximum_z_m:=1.0',
        '-p', 'voxel_size_m:=0.1',
        '-p', 'ego_front_m:=0.1',
        '-p', 'ego_rear_m:=0.1',
        '-p', 'ego_half_width_m:=0.1',
        '-p', 'hard_obstacle_minimum_z_m:=1.0',
    ])
    node = None
    try:
        node = TraversabilityObstacleCandidateNode()
        candidate_capture = _CapturePublisher()
        status_capture = _CapturePublisher()
        node.candidate_publisher = candidate_capture
        node.status_publisher = status_capture
        raw = _cloud([
            (2.5, 0.5, 0.0),
            (1.5, -0.5, 0.0),
            (0.5, 1.5, 0.0),
        ])
        baseline = _cloud([
            (2.5, 0.5, 0.0),
            (0.5, 1.5, 0.0),
        ])

        # Exercise callback-order independence.
        node._store('decision', _decision())
        node._store('raw', raw)
        assert candidate_capture.messages == []
        node._store('baseline', baseline)

        assert len(candidate_capture.messages) == 1
        candidate = candidate_capture.messages[0]
        points = list(point_cloud2.read_points(
            candidate, field_names=('x', 'y', 'z'), skip_nans=True
        ))
        assert len(points) == 2
        assert {(round(p[0], 1), round(p[1], 1)) for p in points} == {
            (1.5, -0.5),
            (0.5, 1.5),
        }
        status = json.loads(status_capture.messages[0].data)
        assert status['baseline_cleared_by_ai_count'] == 1
        assert status['raw_ai_obstacle_point_count'] == 1
        assert status['candidate_point_count'] == 2
        assert status['baseline_occupied_cell_count'] == 2
        assert status['candidate_occupied_cell_count'] == 2
        assert status['candidate_added_cell_count'] == 1
        assert status['candidate_removed_cell_count'] == 1
        assert status['navigation_control_effect'] == 'none'
        assert status['safety_control_effect'] == 'none'
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
