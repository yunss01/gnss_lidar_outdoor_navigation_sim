import csv
import json

import numpy as np

from terrain_navigation_pkg.navigation_learning_manifest import (
    assess_frame,
    balanced_session_splits,
    build_manifests,
    deterministic_split,
    route_signature,
    verify_sample_file,
)


def _valid_row(file_name='samples/sample_000001.npz'):
    return {
        'sample_id': '1',
        'file': file_name,
        'route_status': 'navigating',
        'route_index': '1',
        'speed_mps': '1.0',
        'goal_vehicle_x': '8.0',
        'goal_vehicle_y': '1.0',
        'teacher_subgoal_x': '6.0',
        'teacher_subgoal_y': '0.5',
        'nav2_plan_points': '12',
        'raw_lidar_points': '1000',
        'nav2_status': 'navigating',
        'far_guide_status': 'following_far_segment',
        'safety_state': 'clear',
        'collision_event_count': '0',
    }


def _write_sample(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lidar_bev=np.zeros((4, 8, 8), dtype=np.float16),
        lidar_points_xyz=np.zeros((10, 3), dtype=np.float32),
        nav2_plan_vehicle_xy=np.zeros((4, 2), dtype=np.float32),
        goal_vehicle_xyz=np.zeros(3, dtype=np.float32),
        vehicle_pose_odom_xyzyaw=np.zeros(4, dtype=np.float64),
        vehicle_twist_xyz_rpy=np.zeros(6, dtype=np.float32),
    )


def test_route_signature_is_stable_and_order_sensitive():
    first = {
        'route_json': json.dumps({
            'waypoints': [
                {'latitude': 1.0, 'longitude': 2.0, 'altitude': 3.0},
                {'latitude': 4.0, 'longitude': 5.0, 'altitude': 6.0},
            ],
            'start': True,
        })
    }
    reformatted = {
        'route_json': {
            'start': False,
            'waypoints': [
                {'longitude': 2, 'latitude': 1, 'altitude': 3},
                {'longitude': 5, 'latitude': 4, 'altitude': 6},
            ],
        }
    }
    reversed_route = {
        'route_json': {
            'waypoints': list(reversed(
                reformatted['route_json']['waypoints']
            ))
        }
    }
    assert route_signature(first) == route_signature(reformatted)
    assert route_signature(first) != route_signature(reversed_route)


def test_deterministic_split_keeps_group_together():
    result = deterministic_split(
        'session_a', seed=7, train_fraction=0.7,
        validation_fraction=0.15,
    )
    assert result in {'train', 'validation', 'test'}
    assert result == deterministic_split(
        'session_a', seed=7, train_fraction=0.7,
        validation_fraction=0.15,
    )


def test_balanced_session_split_keeps_clean_data_in_each_split():
    rows = []
    for index, weight in enumerate((100, 90, 80, 70, 60, 50)):
        rows.append({
            'session': 'session_{}'.format(index),
            'route_signature': 'route_a',
            'clean_imitation_frames': weight,
        })
    assignment = balanced_session_splits(
        rows, seed=9, train_fraction=0.7, validation_fraction=0.15,
    )
    assert set(assignment.values()) == {'train', 'validation', 'test'}
    assert len(assignment) == len(rows)


def test_assess_frame_separates_clean_and_recovery_data():
    clean = assess_frame(
        _valid_row(), session_result='completed', sample_exists=True,
    )
    assert clean['initial_bc_eligible'] == 1
    assert clean['quality_tier'] == 'clean_imitation'

    recovery_row = _valid_row()
    recovery_row['nav2_status'] = 'blocked_inefficient_path'
    recovery = assess_frame(
        recovery_row, session_result='node_shutdown', sample_exists=True,
    )
    assert recovery['initial_bc_eligible'] == 0
    assert recovery['recovery_candidate'] == 1
    assert recovery['quality_tier'] == 'recovery_or_negative'

    missing = assess_frame(
        _valid_row(), session_result='completed', sample_exists=False,
    )
    assert missing['base_sample_valid'] == 0
    assert 'sample_file_missing' in missing['reason_codes']


def test_verify_sample_file_checks_required_shapes(tmp_path):
    valid = tmp_path / 'valid.npz'
    _write_sample(valid)
    assert verify_sample_file(valid) == 'passed'

    invalid = tmp_path / 'invalid.npz'
    np.savez_compressed(invalid, lidar_bev=np.zeros((4, 8, 8)))
    assert verify_sample_file(invalid).startswith('failed:missing_arrays=')


def test_build_manifests_does_not_modify_raw_session(tmp_path):
    raw_root = tmp_path / 'raw'
    session = raw_root / 'session_001'
    sample = session / 'samples' / 'sample_000001.npz'
    _write_sample(sample)
    metadata = {
        'session': 'session_001',
        'result': 'completed',
        'route_size': 2,
        'route_json': json.dumps({
            'waypoints': [
                {'latitude': 1, 'longitude': 2, 'altitude': 0},
                {'latitude': 3, 'longitude': 4, 'altitude': 0},
            ]
        }),
    }
    (session / 'metadata.json').write_text(json.dumps(metadata))
    row = _valid_row()
    with (session / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    output = tmp_path / 'manifests'
    summary = build_manifests(
        raw_root, output, seed=3, verify_samples_per_session=1,
    )

    assert sample.is_file()
    assert summary['sessions'] == 1
    assert summary['sample_bearing_sessions'] == 1
    assert summary['existing_sample_files'] == 1
    assert summary['verified_samples'] == 1
    assert summary['verification_failures'] == 0
    with (output / 'manifest_all.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['initial_bc_eligible'] == '1'
    assert rows[0]['npz_verification'] == 'passed'
