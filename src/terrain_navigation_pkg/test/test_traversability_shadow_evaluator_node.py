import csv
import json

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMessage
import numpy as np
import rclpy
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String

from terrain_navigation_pkg.traversability_inference_core import (
    bev_to_occupancy_grid,
)
from terrain_navigation_pkg.traversability_shadow_evaluator_node import (
    TraversabilityShadowEvaluator,
)


def _stamp(message, seconds):
    message.header.stamp.sec = int(seconds)
    message.header.stamp.nanosec = int((seconds % 1.0) * 1.0e9)
    return message


def _grid(bev, seconds):
    stored = bev_to_occupancy_grid(np.asarray(bev, dtype=np.int8))
    message = _stamp(OccupancyGrid(), seconds)
    message.info.width = int(stored.shape[1])
    message.info.height = int(stored.shape[0])
    message.info.resolution = 1.0
    message.data = stored.ravel().tolist()
    return message


def _semantic_cloud(seconds):
    fields = [
        PointField(
            name='x', offset=0, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name='y', offset=4, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name='z', offset=8, datatype=PointField.FLOAT32, count=1
        ),
        PointField(
            name='cos_incidence', offset=12,
            datatype=PointField.FLOAT32, count=1,
        ),
        PointField(
            name='object_idx', offset=16,
            datatype=PointField.UINT32, count=1,
        ),
        PointField(
            name='object_tag', offset=20,
            datatype=PointField.UINT32, count=1,
        ),
    ]
    header = Header()
    header.frame_id = 'lidar_frame'
    header.stamp.sec = int(seconds)
    header.stamp.nanosec = int((seconds % 1.0) * 1.0e9)
    points = [
        (1.5, 1.5, 0.00, 1.0, 1, 1),
        (1.5, 1.5, 0.01, 1.0, 2, 1),
        (0.5, 0.5, 0.50, 1.0, 3, 3),
    ]
    return point_cloud2.create_cloud(header, fields, points)


def _trajectory(seconds):
    message = _stamp(PathMessage(), seconds)
    for x_m, y_m in ((0.0, 0.5), (0.5, 0.5), (1.0, 0.5)):
        pose = PoseStamped()
        pose.pose.position.x = x_m
        pose.pose.position.y = y_m
        pose.pose.orientation.w = 1.0
        message.poses.append(pose)
    return message


def test_aligned_semantic_frame_is_written_and_summarized(tmp_path):
    rclpy.init(args=[
        '--ros-args',
        '-p', 'output_directory:=' + str(tmp_path),
        '-p', 'bev_x_min_m:=0.0',
        '-p', 'bev_x_max_m:=2.0',
        '-p', 'bev_y_min_m:=0.0',
        '-p', 'bev_y_max_m:=2.0',
        '-p', 'bev_resolution_m:=1.0',
        '-p', 'ego_rear_m:=0.1',
        '-p', 'ego_front_m:=0.1',
        '-p', 'ego_half_width_m:=0.1',
    ])
    node = None
    try:
        node = TraversabilityShadowEvaluator()
        node.route_size = 4
        node._on_route_status(String(data='navigating'))

        # Exercise callback-order independence across all three inputs.
        node._on_grid('decision', _grid([[0, -1], [-1, 100]], 12.5))
        node._on_semantic_cloud(_semantic_cloud(12.5))
        node._on_checked_trajectory(_trajectory(12.5))
        node._on_grid('probability', _grid([[10, -1], [-1, 90]], 12.5))
        node._on_route_status(String(data='completed'))

        sessions = list(tmp_path.glob('evaluation_*'))
        assert len(sessions) == 1
        with (sessions[0] / 'frames.csv').open(
            newline='', encoding='utf-8'
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert int(rows[0]['target_known']) == 2
        assert int(rows[0]['selective_false_free']) == 0
        assert rows[0]['corridor_source'] == 'safety_checked_trajectory'
        assert int(rows[0]['corridor_target_known']) == 2

        summary = json.loads(
            (sessions[0] / 'summary.json').read_text(encoding='utf-8')
        )
        assert summary['result'] == 'completed'
        assert summary['frames'] == 1
        assert summary['raw_obstacle_iou'] == 1.0
        assert summary['selective_accepted_accuracy'] == 1.0
        assert summary['timestamp_alignment']['maximum_s'] == 0.0
        assert summary['corridor']['selective_false_free'] == 0
        assert summary['error_snapshot_count'] == 0
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_false_free_frame_saves_auditable_corridor_snapshot(tmp_path):
    rclpy.init(args=[
        '--ros-args',
        '-p', 'output_directory:=' + str(tmp_path),
        '-p', 'bev_x_min_m:=0.0',
        '-p', 'bev_x_max_m:=2.0',
        '-p', 'bev_y_min_m:=0.0',
        '-p', 'bev_y_max_m:=2.0',
        '-p', 'bev_resolution_m:=1.0',
        '-p', 'ego_rear_m:=0.1',
        '-p', 'ego_front_m:=0.1',
        '-p', 'ego_half_width_m:=0.1',
        '-p', 'corridor_vehicle_front_m:=0.5',
        '-p', 'corridor_vehicle_rear_m:=0.5',
        '-p', 'corridor_half_width_m:=0.6',
    ])
    node = None
    try:
        node = TraversabilityShadowEvaluator()
        node._on_route_status(String(data='navigating'))
        node._on_checked_trajectory(_trajectory(20.0))
        node._on_semantic_cloud(_semantic_cloud(20.0))
        node._on_grid('probability', _grid([[10, -1], [-1, 10]], 20.0))
        node._on_grid('decision', _grid([[0, -1], [-1, 0]], 20.0))
        node._on_route_status(String(data='completed'))

        session = next(tmp_path.glob('evaluation_*'))
        with (session / 'frames.csv').open(
            newline='', encoding='utf-8'
        ) as stream:
            row = next(csv.DictReader(stream))
        snapshot = session / row['error_snapshot_file']
        assert snapshot.is_file()
        with np.load(snapshot) as archive:
            assert archive['selective_false_free_mask'].sum() == 1
            assert archive['corridor_mask'][1, 1]

        summary = json.loads(
            (session / 'summary.json').read_text(encoding='utf-8')
        )
        assert summary['selective_false_free'] == 1
        assert summary['corridor']['selective_false_free'] == 1
        assert summary['error_snapshot_count'] == 1
        assert summary['maximum_corridor_false_free_cluster_cells'] == 1
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
