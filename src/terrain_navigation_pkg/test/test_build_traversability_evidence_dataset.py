import csv
import json

import numpy as np
import pytest

from terrain_navigation_pkg.build_traversability_evidence_dataset import (
    build_evidence_dataset,
)


def _write_session(
    root,
    *,
    include_object_ids=True,
    session_name='session_20260918_120000_000001',
):
    session = root / session_name
    samples = session / 'samples'
    samples.mkdir(parents=True)
    metadata = {
        'privileged_supervision': {'semantic_labels_saved': True},
        'bev': {
            'x_min_m': 0.0,
            'x_max_m': 4.0,
            'y_min_m': -2.0,
            'y_max_m': 2.0,
            'z_min_m': -2.0,
            'z_max_m': 2.0,
            'resolution_m': 0.5,
        },
        'controlled_traversability_actors': [
            {
                'actor_id': 167,
                'blueprint': 'static.prop.motorhelmet',
                'disposition': 'obstacle',
                'policy_source': 'vehicle_clearance_policy',
                'bbox_extent_xyz_m': [0.15, 0.12, 0.10],
            },
            {
                'actor_id': 168,
                'blueprint': 'static.prop.low_curb_sample',
                'disposition': 'passable',
                'policy_source': 'vehicle_clearance_policy',
            },
        ],
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
            'route_index': 0,
            'route_status': 'idle',
            'semantic_alignment_delta_s': 0.01,
            'collision_event_count': 0,
        })

    points = np.asarray([
        [3.25, 1.25, -1.00],
        [3.20, 1.20, -0.99],
        [3.25, 0.25, -0.80],  # Controlled obstacle actor 167.
        [3.25, -0.75, -0.90],  # Controlled passable actor 168.
    ], dtype=np.float32)
    # Populate raw LiDAR ground support plus the two controlled returns.
    raw_points = []
    for forward in np.arange(1.25, 4.0, 0.5):
        for left in np.arange(-1.25, 1.5, 0.5):
            raw_points.append([forward, left, -1.0])
    raw_points.extend(points[2:].tolist())
    from terrain_navigation_pkg.navigation_learning_recorder_core import (
        BevGeometry,
        build_lidar_bev,
    )
    geometry = BevGeometry(
        x_min_m=0.0, x_max_m=4.0,
        y_min_m=-2.0, y_max_m=2.0,
        z_min_m=-2.0, z_max_m=2.0,
        resolution_m=0.5,
    )
    arrays = {
        'lidar_bev': build_lidar_bev(
            np.asarray(raw_points, dtype=np.float32), geometry
        ).astype(np.float16),
        'semantic_lidar_points_xyz': points,
        'semantic_lidar_object_tag': np.asarray(
            [1, 1, 20, 20], dtype=np.uint32
        ),
        'semantic_alignment_delta_s': np.asarray(0.01),
        'goal_vehicle_xyz': np.asarray([5.0, 0.0, 0.0]),
    }
    if include_object_ids:
        arrays['semantic_lidar_object_idx'] = np.asarray(
            [0, 0, 167, 168], dtype=np.uint32
        )
    np.savez_compressed(samples / 'sample_000001.npz', **arrays)
    return session


def _append_duplicate_frame(session, source_sample_id=1, new_sample_id=2):
    source = session / 'samples' / ('sample_%06d.npz' % source_sample_id)
    duplicate = session / 'samples' / ('sample_%06d.npz' % new_sample_id)
    duplicate.write_bytes(source.read_bytes())
    with (session / 'frames.csv').open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'sample_id', 'file', 'ros_time_s', 'route_index',
            'route_status', 'semantic_alignment_delta_s',
            'collision_event_count',
        ])
        writer.writerow({
            'sample_id': new_sample_id,
            'file': 'samples/sample_%06d.npz' % new_sample_id,
            'ros_time_s': 11.0,
            'route_index': 0,
            'route_status': 'idle',
            'semantic_alignment_delta_s': 0.01,
            'collision_event_count': 0,
        })


def test_v2_builder_writes_independent_targets_and_instance_provenance(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'v2'
    session = _write_session(raw)
    raw_sample = session / 'samples' / 'sample_000001.npz'
    before = raw_sample.read_bytes()

    summary = build_evidence_dataset(
        raw,
        output,
        ego_rear_m=0.1,
        ego_front_m=0.1,
        ego_half_width_m=0.1,
        visibility_angular_bin_count=36,
        local_ground_radii_m=(0.75, 1.5),
    )

    assert summary['schema_version'] == 3
    assert summary['written_samples'] == 1
    assert summary['controlled_sessions_scanned'] == 1
    assert raw_sample.read_bytes() == before
    derived = (
        output / 'samples' / session.name / 'sample_000001.npz'
    )
    with np.load(derived, allow_pickle=False) as arrays:
        assert arrays['lidar_bev'].shape == (4, 8, 8)
        assert arrays['lidar_evidence_bev'].shape == (8, 8, 8)
        assert arrays['target_passable_surface_mask'].shape == (8, 8)
        assert arrays['target_obstacle_evidence_mask'].shape == (8, 8)
        assert arrays['target_ambiguous_observed_mask'].shape == (8, 8)
        assert arrays['controlled_actor_ids'].tolist() == [167, 168]
        assert arrays['controlled_actor_disposition_code'].tolist() == [1, 0]
        assert 167 in arrays['target_obstacle_instance_id']
        assert np.count_nonzero(
            arrays['target_controlled_passable_point_count']
        ) == 1
        assert 'semantic_lidar_object_tag' not in arrays.files
    rows = list(csv.DictReader((output / 'manifest.csv').open()))
    assert rows[0]['status'] == 'written'
    assert rows[0]['controlled_actor_return_count'] == '2'


def test_controlled_policy_requires_object_instance_ids(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'v2'
    _write_session(raw, include_object_ids=False)

    summary = build_evidence_dataset(
        raw,
        output,
        ego_rear_m=0.1,
        ego_front_m=0.1,
        ego_half_width_m=0.1,
    )
    assert summary['written_samples'] == 0
    assert summary['skip_reasons'] == {
        'controlled_policy_requires_semantic_object_idx': 1
    }


def test_v2_builder_refuses_output_inside_raw_root(tmp_path):
    raw = tmp_path / 'raw'
    _write_session(raw)
    with pytest.raises(ValueError):
        build_evidence_dataset(raw, raw / 'derived_v2')


def test_v2_builder_deduplicates_exact_scans_within_session(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'v2'
    session = _write_session(raw)
    _append_duplicate_frame(session)

    summary = build_evidence_dataset(raw, output)

    assert summary['written_samples'] == 1
    assert summary['skipped_samples'] == 1
    assert summary['duplicate_samples_skipped'] == 1
    assert summary['skip_reasons'] == {'duplicate_identical_scan': 1}
    rows = list(csv.DictReader((output / 'manifest.csv').open()))
    assert rows[0]['status'] == 'written'
    assert len(rows[0]['scan_fingerprint']) == 64
    assert rows[1]['reason'] == 'duplicate_identical_scan'
    assert rows[1]['scan_fingerprint'] == rows[0]['scan_fingerprint']
    assert rows[1]['duplicate_of_source_sample_id'] == '1'


def test_v2_builder_filters_sessions_and_reports_empty_session(tmp_path):
    raw = tmp_path / 'raw'
    output = tmp_path / 'v2'
    included = _write_session(
        raw, session_name='session_20260918_120000_000001'
    )
    _write_session(raw, session_name='session_20260918_120100_000001')
    empty = _write_session(
        raw, session_name='session_20260918_120200_000001'
    )
    (empty / 'frames.csv').write_text(
        'sample_id,file,ros_time_s,route_index,route_status,'
        'semantic_alignment_delta_s,collision_event_count\n'
    )

    summary = build_evidence_dataset(
        raw,
        output,
        session_names=(included.name, empty.name),
    )

    assert summary['requested_sessions'] == [included.name, empty.name]
    assert summary['sessions_scanned'] == 2
    assert summary['written_samples'] == 1
    assert summary['empty_session_count'] == 1
    assert summary['empty_sessions'] == [empty.name]


def test_v2_builder_rejects_missing_requested_session(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    with pytest.raises(ValueError, match='requested sessions were not found'):
        build_evidence_dataset(
            raw,
            tmp_path / 'v2',
            session_names=('session_missing',),
        )
