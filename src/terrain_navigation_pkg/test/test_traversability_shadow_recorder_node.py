import csv
import json

from nav_msgs.msg import OccupancyGrid
import rclpy
from std_msgs.msg import String

from terrain_navigation_pkg.traversability_shadow_recorder_node import (
    TraversabilityShadowRecorder,
)


def _grid(stamp_s):
    message = OccupancyGrid()
    message.header.stamp.sec = int(stamp_s)
    message.header.stamp.nanosec = int((stamp_s % 1.0) * 1.0e9)
    message.info.width = 2
    message.info.height = 2
    message.data = [0, 100, -1, 0]
    return message


def test_completed_route_writes_csv_summary_and_snapshot(tmp_path):
    rclpy.init(args=[
        '--ros-args', '-p', 'output_directory:=' + str(tmp_path),
    ])
    node = None
    try:
        node = TraversabilityShadowRecorder()
        node.route_size = 4
        node._on_route_status(String(data='navigating'))
        for name in ('probability', 'uncertainty', 'decision'):
            node._on_grid(name, _grid(12.5))
        node._on_shadow_status(String(data=json.dumps({
            'frame_sequence': 1,
            'source_stamp_s': 12.5,
            'source_frame': 'lidar_frame',
            'checkpoint': '/tmp/model.pt',
            'device': 'cpu',
            'mc_samples': 4,
            'scan_points': 100,
            'observed_cells': 10,
            'learned_free_cells': 6,
            'learned_obstacle_cells': 1,
            'hard_obstacle_cells': 2,
            'uncertain_cells': 1,
            'mean_entropy': 0.1,
            'mean_mc_variance': 0.01,
            'inference_ms': 20.0,
        })))
        node._on_route_status(String(data='completed'))

        sessions = list(tmp_path.glob('shadow_*'))
        assert len(sessions) == 1
        with (sessions[0] / 'frames.csv').open(
            newline='', encoding='utf-8'
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert float(rows[0]['known_coverage']) == 0.9
        assert rows[0]['snapshot_file'].endswith('route_start.npz')
        assert (sessions[0] / rows[0]['snapshot_file']).is_file()
        summary = json.loads(
            (sessions[0] / 'summary.json').read_text(encoding='utf-8')
        )
        assert summary['result'] == 'completed'
        assert summary['frame_count'] == 1
        assert summary['snapshot_count'] == 1
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
