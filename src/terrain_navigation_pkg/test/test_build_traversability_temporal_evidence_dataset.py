import csv
import json

import numpy as np

from terrain_navigation_pkg import (
    build_traversability_temporal_evidence_dataset as temporal_builder,
)
from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
)
from terrain_navigation_pkg.traversability_evidence_dataset import (
    TraversabilityEvidenceDataset,
)


def _moving_session(root):
    name = 'session_20260921_200000_000001'
    session = root / name
    samples = session / 'samples'
    samples.mkdir(parents=True)
    geometry = BevGeometry(
        x_min_m=0.0,
        x_max_m=4.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        z_min_m=-2.0,
        z_max_m=2.0,
        resolution_m=0.5,
    )
    metadata = {
        'result': 'perception_capture_completed',
        'privileged_supervision': {'semantic_labels_saved': True},
        'bev': {
            'x_min_m': geometry.x_min_m,
            'x_max_m': geometry.x_max_m,
            'y_min_m': geometry.y_min_m,
            'y_max_m': geometry.y_max_m,
            'z_min_m': geometry.z_min_m,
            'z_max_m': geometry.z_max_m,
            'resolution_m': geometry.resolution_m,
        },
        'controlled_traversability_actors': [{
            'actor_id': 7,
            'blueprint': 'static.prop.motorhelmet',
            'disposition': 'obstacle',
            'policy_source': 'test',
        }],
    }
    (session / 'metadata.json').write_text(json.dumps(metadata))
    fields = [
        'sample_id', 'file', 'ros_time_s', 'route_index', 'route_status',
        'semantic_alignment_delta_s', 'collision_event_count',
    ]
    with (session / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(6):
            sample_id = index + 1
            vehicle_x = 0.1 * index
            ground_world = np.asarray([
                [forward, left, -1.0]
                for forward in np.arange(1.25, 4.0, 0.5)
                for left in np.arange(-1.25, 1.5, 0.5)
            ], dtype=np.float32)
            ground = ground_world.copy()
            ground[:, 0] -= vehicle_x
            actor = np.asarray(
                [[3.25 - vehicle_x, 0.25, -0.7]], dtype=np.float32
            )
            raw = np.concatenate([ground, actor], axis=0)
            semantic = np.concatenate([ground[:8], actor], axis=0)
            np.savez_compressed(
                samples / ('sample_%06d.npz' % sample_id),
                lidar_bev=build_lidar_bev(raw, geometry).astype(np.float16),
                lidar_points_xyz=raw,
                semantic_lidar_points_xyz=semantic,
                semantic_lidar_object_tag=np.asarray(
                    [1] * 8 + [20], dtype=np.uint32
                ),
                semantic_lidar_object_idx=np.asarray(
                    [0] * 8 + [7], dtype=np.uint32
                ),
                semantic_alignment_delta_s=np.asarray(0.01),
                vehicle_pose_odom_xyzyaw=np.asarray(
                    [vehicle_x, 0.0, 0.0, 0.0], dtype=np.float64
                ),
            )
            writer.writerow({
                'sample_id': sample_id,
                'file': 'samples/sample_%06d.npz' % sample_id,
                'ros_time_s': 10.0 + 0.2 * index,
                'route_index': 0,
                'route_status': 'idle',
                'semantic_alignment_delta_s': 0.01,
                'collision_event_count': 0,
            })
    return session


def test_temporal_builder_writes_17_channel_non_destructive_samples(tmp_path):
    raw = tmp_path / 'raw'
    session = _moving_session(raw)
    output = tmp_path / 'temporal'
    raw_sample = session / 'samples' / 'sample_000006.npz'
    before = raw_sample.read_bytes()
    summary = temporal_builder.build_temporal_evidence_dataset(
        raw,
        output,
        session_names=(session.name,),
        history_size=3,
        ego_rear_m=0.1,
        ego_front_m=0.1,
        ego_half_width_m=0.1,
        visibility_angular_bin_count=36,
    )
    assert summary['written_samples'] == 4
    assert summary['skip_reasons'] == {'insufficient_unique_history': 2}
    assert raw_sample.read_bytes() == before
    derived = output / 'samples' / session.name / 'sample_000006.npz'
    with np.load(derived, allow_pickle=False) as arrays:
        assert arrays['lidar_evidence_bev'].shape == (17, 8, 8)
        assert arrays['current_lidar_evidence_bev'].shape == (8, 8, 8)
        assert arrays['temporal_lidar_evidence_bev'].shape == (8, 8, 8)
        assert arrays['temporal_scan_support_fraction'].shape == (8, 8)
        assert arrays['temporal_source_sample_ids'].tolist() == [4, 5, 6]
        assert np.max(arrays['temporal_scan_support_fraction']) <= 1.0
        assert np.count_nonzero(
            arrays['target_controlled_obstacle_point_count']
        ) == 1
    row = {
        'scene_id': 'scene_test',
        'object_family': 'motorhelmet',
        'role': 'controlled',
        'source_session': session.name,
        'source_sample_id': '6',
        'derived_sample_path': str(derived),
    }
    item = TraversabilityEvidenceDataset([row])[0]
    assert tuple(item['lidar_evidence_bev'].shape) == (17, 8, 8)
    current_item = TraversabilityEvidenceDataset(
        [row], input_variant='current_only'
    )[0]
    temporal_item = TraversabilityEvidenceDataset(
        [row], input_variant='temporal'
    )[0]
    assert tuple(current_item['lidar_evidence_bev'].shape) == (8, 8, 8)
    assert tuple(temporal_item['lidar_evidence_bev'].shape) == (17, 8, 8)
