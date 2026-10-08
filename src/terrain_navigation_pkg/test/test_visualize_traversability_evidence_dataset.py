import csv

import numpy as np
import pytest

from terrain_navigation_pkg.visualize_traversability_evidence_dataset import (
    audit_evidence_dataset,
    audit_sample,
    read_written_rows,
    render_sample_tile,
)


def _write_fixture(tmp_path, *, invalid_controlled_mask=False):
    sample = tmp_path / 'sample_000001.npz'
    shape = (6, 6)
    passable = np.zeros(shape, dtype=bool)
    obstacle = np.zeros(shape, dtype=bool)
    ambiguous = np.zeros(shape, dtype=bool)
    passable[1, 1] = True
    obstacle[2, 3] = True
    controlled_obstacle_count = np.zeros(shape, dtype=np.int32)
    controlled_obstacle_count[2, 3] = 1
    if invalid_controlled_mask:
        obstacle[2, 3] = False
    instance_ids = np.full(shape, -1, dtype=np.int64)
    instance_ids[2, 3] = 41
    zeros = np.zeros(shape, dtype=np.int32)
    np.savez_compressed(
        sample,
        lidar_bev=np.zeros((4,) + shape, dtype=np.float16),
        local_ground_relative_max_height_m=np.zeros(
            (2,) + shape, dtype=np.float16
        ),
        target_passable_surface_mask=passable,
        target_obstacle_evidence_mask=obstacle,
        target_ambiguous_observed_mask=ambiguous,
        target_controlled_passable_point_count=zeros,
        target_controlled_obstacle_point_count=controlled_obstacle_count,
        target_controlled_ambiguous_point_count=zeros,
        target_obstacle_instance_id=instance_ids,
        target_ego_exclusion_mask=np.zeros(shape, dtype=bool),
        visibility_free_mask=np.zeros(shape, dtype=bool),
        controlled_actor_ids=np.asarray([41], dtype=np.int64),
        controlled_actor_disposition_code=np.asarray([1], dtype=np.int8),
    )
    row = {
        'source_session': 'session_controlled',
        'source_sample_id': '1',
        'derived_sample_path': str(sample),
        'status': 'written',
        'controlled_actor_return_count': '1',
    }
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return manifest, row


def test_v2_evidence_audit_and_render_pass(tmp_path):
    manifest, row = _write_fixture(tmp_path)

    rows = read_written_rows(manifest)
    audit = audit_sample(rows[0])
    tile = render_sample_tile(row, cell_scale=2)

    assert audit['audit_passed']
    assert audit['controlled_obstacle_cell_count'] == 1
    assert audit['controlled_obstacle_cells_rc'] == '[[2, 3]]'
    assert audit['instance_ids_in_obstacle_target'] == '[41]'
    assert tile.width > 24
    assert tile.height > 24


def test_v2_evidence_audit_writes_outputs(tmp_path):
    manifest, _ = _write_fixture(tmp_path)
    output = tmp_path / 'audit'

    summary = audit_evidence_dataset(manifest, output)

    assert summary['audited_samples'] == 1
    assert summary['passed_samples'] == 1
    assert summary['failed_samples'] == 0
    assert (output / 'audit.csv').is_file()
    assert (output / 'controlled_evidence_01.png').is_file()
    assert (output / 'controlled_evidence.pdf').is_file()
    assert (output / 'summary.json').is_file()


def test_v2_evidence_audit_fails_if_controlled_obstacle_is_not_target(tmp_path):
    manifest, _ = _write_fixture(tmp_path, invalid_controlled_mask=True)

    with pytest.raises(RuntimeError, match='failed invariants'):
        audit_evidence_dataset(manifest, tmp_path / 'audit')

    audit = audit_sample(read_written_rows(manifest)[0])
    assert not audit['audit_passed']
    assert 'controlled_obstacle_not_in_obstacle_target' in audit['violations']


def test_v2_evidence_counterfactual_rejects_background_obstacle(tmp_path):
    manifest, row = _write_fixture(tmp_path)
    control_sample = tmp_path / 'control_sample.npz'
    with np.load(row['derived_sample_path'], allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values['controlled_actor_ids'] = np.empty(0, dtype=np.int64)
    values['controlled_actor_disposition_code'] = np.empty(0, dtype=np.int8)
    values['target_controlled_obstacle_point_count'] = np.zeros(
        (6, 6), dtype=np.int32
    )
    np.savez_compressed(control_sample, **values)
    control_row = {
        'source_session': 'session_control',
        'source_sample_id': '1',
        'derived_sample_path': str(control_sample),
        'status': 'written',
        'controlled_actor_return_count': '0',
    }

    audit = audit_sample(row, control_row=control_row)

    assert not audit['audit_passed']
    assert audit['counterfactual_evaluated']
    assert audit['controlled_cells_obstacle_in_control'] == 1
    assert 'controlled_cell_already_obstacle_in_control' in audit['violations']


def test_v2_evidence_dataset_uses_named_counterfactual_control(tmp_path):
    manifest, row = _write_fixture(tmp_path)
    control_sample = tmp_path / 'control_sample.npz'
    with np.load(row['derived_sample_path'], allow_pickle=False) as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values['target_obstacle_evidence_mask'] = np.zeros((6, 6), dtype=bool)
    values['target_obstacle_point_count'] = np.zeros((6, 6), dtype=np.int32)
    values['target_obstacle_instance_id'] = np.full(
        (6, 6), -1, dtype=np.int64
    )
    values['target_controlled_obstacle_point_count'] = np.zeros(
        (6, 6), dtype=np.int32
    )
    values['controlled_actor_ids'] = np.empty(0, dtype=np.int64)
    values['controlled_actor_disposition_code'] = np.empty(0, dtype=np.int8)
    np.savez_compressed(control_sample, **values)
    control_row = {
        'source_session': 'session_control',
        'source_sample_id': '1',
        'derived_sample_path': str(control_sample),
        'status': 'written',
        'controlled_actor_return_count': '0',
    }
    with manifest.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writerow(control_row)

    summary = audit_evidence_dataset(
        manifest,
        tmp_path / 'audit',
        control_session='session_control',
    )

    assert summary['failed_samples'] == 0
    assert summary['counterfactual_evaluated_samples'] == 1
    assert summary['controlled_cells_obstacle_in_control'] == 0
