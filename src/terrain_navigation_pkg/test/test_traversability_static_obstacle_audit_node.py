import csv
import json

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from terrain_navigation_pkg.traversability_inference_core import (
    bev_to_occupancy_grid,
)
from terrain_navigation_pkg.traversability_shadow_node import (
    TraversabilityShadowNode,
)
from terrain_navigation_pkg.traversability_static_obstacle_audit_node import (
    TraversabilityStaticObstacleAuditNode,
)


def _header(seconds=12.5):
    value = Header()
    value.frame_id = 'lidar_frame'
    value.stamp.sec = int(seconds)
    value.stamp.nanosec = int((seconds % 1.0) * 1.0e9)
    return value


def _xyz_cloud(points):
    return point_cloud2.create_cloud_xyz32(_header(), points)


def _semantic_cloud():
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
    return point_cloud2.create_cloud(
        _header(),
        fields,
        [
            (2.5, 0.5, 0.4, 1.0, 101, 20),
            (1.5, -0.5, 0.4, 1.0, 202, 20),
        ],
    )


def _decision_grid(bev):
    stored = bev_to_occupancy_grid(np.asarray(bev, dtype=np.int8))
    message = OccupancyGrid()
    message.header = _header()
    message.info.width = int(stored.shape[1])
    message.info.height = int(stored.shape[0])
    message.info.resolution = 1.0
    message.data = stored.ravel().tolist()
    return message


def test_v2_snapshot_preserves_diagnostics_and_instance_loss(tmp_path):
    rclpy.init(args=[
        '--ros-args',
        '-p', 'output_directory:=' + str(tmp_path),
        '-p', 'duration_s:=1.0',
        '-p', 'warmup_s:=0.0',
        '-p', 'minimum_frames:=1',
        '-p', 'bev_x_min_m:=0.0',
        '-p', 'bev_x_max_m:=4.0',
        '-p', 'bev_y_min_m:=-2.0',
        '-p', 'bev_y_max_m:=2.0',
        '-p', 'bev_resolution_m:=1.0',
        '-p', 'ego_rear_m:=0.1',
        '-p', 'ego_front_m:=0.1',
        '-p', 'ego_half_width_m:=0.1',
        '-p', 'instance_corridor_maximum_x_m:=4.0',
        '-p', 'instance_corridor_half_width_m:=2.0',
    ])
    node = None
    try:
        node = TraversabilityStaticObstacleAuditNode()
        raw = _xyz_cloud([
            (2.5, 0.5, 0.4),
            (1.5, -0.5, 0.4),
        ])
        baseline = _xyz_cloud([
            (2.5, 0.5, 0.4),
            (1.5, -0.5, 0.4),
        ])
        candidate = _xyz_cloud([(2.5, 0.5, 0.4)])
        probability = np.zeros((4, 4), dtype=np.float32)
        entropy = np.zeros((4, 4), dtype=np.float32)
        variance = np.zeros((4, 4), dtype=np.float32)
        hard_mask = np.zeros((4, 4), dtype=np.uint8)
        diagnostics = {
            'probability': TraversabilityShadowNode._image_message(
                probability, raw
            ),
            'entropy': TraversabilityShadowNode._image_message(entropy, raw),
            'variance': TraversabilityShadowNode._image_message(
                variance, raw
            ),
            'hard_mask': TraversabilityShadowNode._image_message(
                hard_mask, raw, mask=True
            ),
            'decision': _decision_grid(np.zeros((4, 4), dtype=np.int8)),
        }
        stamp = 12_500_000_000
        node._evaluate(
            stamp,
            stamp,
            baseline,
            candidate,
            _semantic_cloud(),
            {
                'navigation_control_effect': 'none',
                'safety_control_effect': 'none',
            },
            raw,
            diagnostics,
        )
        node._finish('test_complete')

        session = next(tmp_path.glob('static_audit_*'))
        snapshot = np.load(session / 'snapshots' / 'representative.npz')
        assert snapshot['raw_points_xyz'].shape == (2, 3)
        assert snapshot['semantic_object_idx'].tolist() == [101, 202]
        assert snapshot['model_mc_variance'].shape == (4, 4)
        assert snapshot['hard_obstacle_mask'].shape == (4, 4)

        with (session / 'instances.csv').open(
            newline='', encoding='utf-8'
        ) as stream:
            instances = list(csv.DictReader(stream))
        assert len(instances) == 2
        removed = [
            item for item in instances if int(item['fully_removed']) == 1
        ]
        assert [int(item['object_idx']) for item in removed] == [202]

        summary = json.loads(
            (session / 'summary.json').read_text(encoding='utf-8')
        )
        assert summary['diagnostic_attached_fraction'] == 1.0
        assert summary['instance_evaluation'][
            'fully_removed_instance_observations'
        ] == 1
        assert not summary['gates']['no_complete_obstacle_instance_loss']
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
