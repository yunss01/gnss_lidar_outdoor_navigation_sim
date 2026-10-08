import csv
import json

import numpy as np

from terrain_navigation_pkg.build_traversability_dataset import build_dataset


def _write_session(root, semantic=True, collision_event_count=0):
    session = root / 'session_20260914_120000_000001'
    samples = session / 'samples'
    samples.mkdir(parents=True)
    metadata = {
        'privileged_supervision': {'semantic_labels_saved': True},
        'bev': {
            'x_min_m': 0.0, 'x_max_m': 2.0,
            'y_min_m': -1.0, 'y_max_m': 1.0,
            'z_min_m': -1.0, 'z_max_m': 2.0,
            'resolution_m': 1.0,
        },
    }
    (session / 'metadata.json').write_text(json.dumps(metadata))
    with (session / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'sample_id', 'file', 'ros_time_s', 'route_index',
            'route_status', 'semantic_alignment_delta_s',
            'collision_event_count',
        ])
        writer.writeheader()
        writer.writerow({
            'sample_id': 1,
            'file': 'samples/sample_000001.npz',
            'ros_time_s': 10.0,
            'route_index': 2,
            'route_status': 'navigating',
            'semantic_alignment_delta_s': 0.01,
            'collision_event_count': collision_event_count,
        })
    arrays = {
        'lidar_bev': np.zeros((4, 2, 2), dtype=np.float16),
        'semantic_lidar_points_xyz': (
            np.asarray([[1.5, 0.5, 0.0], [1.4, 0.4, 0.01]])
            if semantic else np.empty((0, 3))
        ),
        'semantic_lidar_object_tag': (
            np.asarray([1, 2], dtype=np.uint32)
            if semantic else np.empty((0,), dtype=np.uint32)
        ),
        'semantic_alignment_delta_s': np.asarray(0.01),
        'goal_vehicle_xyz': np.asarray([5.0, 0.0, 0.0]),
    }
    np.savez_compressed(samples / 'sample_000001.npz', **arrays)
    return session


def test_build_dataset_writes_input_target_pair_and_manifest(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'derived'
    _write_session(raw)

    summary = build_dataset(
        raw, output,
        ego_rear_m=0.25, ego_front_m=0.25, ego_half_width_m=0.25,
    )

    assert summary['written_samples'] == 1
    assert summary['skipped_samples'] == 0
    derived = (
        output / 'samples' / 'session_20260914_120000_000001'
        / 'sample_000001.npz'
    )
    with np.load(derived) as arrays:
        assert arrays['lidar_bev'].shape == (4, 2, 2)
        assert arrays['target_labels'].shape == (2, 2)
        assert np.count_nonzero(arrays['target_labels'] == 0) == 1
        assert arrays['visibility_free_mask'].shape == (2, 2)
        assert arrays['visibility_ray_count'].shape == (2, 2)
        assert arrays['goal_vehicle_xyz'].tolist() == [5.0, 0.0, 0.0]
        assert 'semantic_lidar_object_tag' not in arrays.files
    rows = list(csv.DictReader((output / 'manifest.csv').open()))
    assert rows[0]['status'] == 'written'
    assert rows[0]['free_cell_count'] == '1'


def test_build_dataset_audits_sample_without_semantic_points(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'derived'
    _write_session(raw, semantic=False)

    summary = build_dataset(
        raw, output,
        ego_rear_m=0.25, ego_front_m=0.25, ego_half_width_m=0.25,
    )

    assert summary['written_samples'] == 0
    assert summary['skipped_samples'] == 1
    assert summary['skip_reasons'] == {'missing_semantic_points': 1}


def test_build_dataset_excludes_collision_and_later_frames(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'derived'
    _write_session(raw, collision_event_count=1)

    summary = build_dataset(
        raw, output,
        ego_rear_m=0.25, ego_front_m=0.25, ego_half_width_m=0.25,
    )

    assert summary['written_samples'] == 0
    assert summary['skipped_samples'] == 1
    assert summary['skip_reasons'] == {'collision_or_after': 1}
    rows = list(csv.DictReader((output / 'manifest.csv').open()))
    assert rows[0]['collision_event_count'] == '1'
