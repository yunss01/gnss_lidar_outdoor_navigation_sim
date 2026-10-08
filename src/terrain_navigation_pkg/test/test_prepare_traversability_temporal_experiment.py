import csv
import json

import numpy as np

from terrain_navigation_pkg.prepare_traversability_temporal_experiment import (
    prepare_temporal_experiment,
)


def _temporal_archive(root, scene_name, x_offset):
    archive = root / scene_name
    sample_root = archive / 'samples'
    sample_root.mkdir(parents=True)
    fields = [
        'status', 'derived_sample_path', 'source_session',
        'source_sample_id', 'source_sample_path', 'scan_fingerprint',
        'controlled_actor_return_count',
    ]
    rows = []
    for role in ('control', 'controlled'):
        session = scene_name + '_' + role
        session_root = sample_root / session
        session_root.mkdir()
        for sample_id, x_value in enumerate((0.0, 1.0), start=1):
            path = session_root / ('sample_%06d.npz' % sample_id)
            policy = []
            controlled = np.zeros((4, 4), dtype=np.int32)
            if role == 'controlled':
                policy = [{
                    'actor_id': 7,
                    'blueprint': 'static.prop.motorhelmet',
                    'disposition': 'obstacle',
                    'policy_source': 'test',
                }]
                controlled[2, 2] = 2
            pose_x = x_offset + x_value
            if role == 'controlled':
                pose_x += 0.02
            np.savez_compressed(
                path,
                lidar_evidence_bev=np.zeros((17, 4, 4), dtype=np.float16),
                current_lidar_evidence_bev=np.zeros(
                    (8, 4, 4), dtype=np.float16
                ),
                vehicle_pose_odom_xyzyaw=np.asarray(
                    [pose_x, 0.0, 0.0, 0.0], dtype=np.float64
                ),
                controlled_actor_policy_json=np.asarray(json.dumps(policy)),
                target_controlled_obstacle_point_count=controlled,
            )
            rows.append({
                'status': 'written',
                'derived_sample_path': str(path),
                'source_session': session,
                'source_sample_id': sample_id,
                'source_sample_path': 'raw/' + path.name,
                'scan_fingerprint': session + str(sample_id),
                'controlled_actor_return_count': int(controlled.sum()),
            })
    with (archive / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return archive


def test_temporal_experiment_pairs_each_frame_with_nearest_control(tmp_path):
    scene_a = _temporal_archive(tmp_path, 'scene_a', 0.0)
    scene_b = _temporal_archive(tmp_path, 'scene_b', 10.0)
    output = tmp_path / 'experiment'
    summary = prepare_temporal_experiment(
        [('scene_a', scene_a), ('scene_b', scene_b)],
        output,
        maximum_pair_position_error_m=0.10,
        maximum_pair_yaw_error_deg=1.0,
    )
    assert summary['sample_rows'] == 8
    assert summary['control_rows'] == 4
    assert summary['controlled_rows'] == 4
    assert summary['frozen_test_accessed'] is False
    with (output / 'split_manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    controls = {}
    for row in rows:
        key = (row['scene_id'], row['pair_group'])
        controls[key] = controls.get(key, 0) + int(row['role'] == 'control')
        if row['role'] == 'controlled':
            assert float(row['pair_position_error_m']) == 0.02
    assert set(controls.values()) == {1}


def test_temporal_experiment_excludes_position_mismatch(tmp_path):
    scene_a = _temporal_archive(tmp_path, 'scene_a', 0.0)
    scene_b = _temporal_archive(tmp_path, 'scene_b', 10.0)
    controlled = next(
        (scene_b / 'samples' / 'scene_b_controlled').glob('*.npz')
    )
    with np.load(controlled, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    values['vehicle_pose_odom_xyzyaw'][0] += 5.0
    np.savez_compressed(controlled, **values)
    summary = prepare_temporal_experiment(
        [('scene_a', scene_a), ('scene_b', scene_b)],
        tmp_path / 'experiment',
        maximum_pair_position_error_m=0.10,
        maximum_pair_yaw_error_deg=1.0,
    )
    report = summary['scenes']['scene_b']
    assert report['controlled_rows_excluded_position'] == 1
    assert report['controlled_rows_matched'] == 1
