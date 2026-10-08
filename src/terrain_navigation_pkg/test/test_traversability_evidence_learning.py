import csv
import json
import math

import numpy as np
import pytest
import torch

from terrain_navigation_pkg import (
    cross_validate_traversability_evidence_model as evidence_cv,
)
from terrain_navigation_pkg import (
    train_traversability_evidence_model as evidence_train,
)
from terrain_navigation_pkg.prepare_traversability_evidence_experiment import (
    _validate_spec,
    prepare_experiment,
)
from terrain_navigation_pkg.traversability_evidence_dataset import (
    PairedTraversabilityEvidenceDataset,
    TraversabilityEvidenceDataset,
    family_balanced_sample_weights,
    normalize_lidar_evidence_bev,
)
from terrain_navigation_pkg.traversability_evidence_metrics import (
    TraversabilityEvidenceMetricAccumulator,
)
from terrain_navigation_pkg.traversability_evidence_model import (
    BevTraversabilityDecoupledEvidenceUNet,
    BevTraversabilityEvidenceUNet,
    BevTraversabilityLocalEvidenceNet,
    build_traversability_evidence_model,
    independent_evidence_loss,
)


def _sample(path, blueprint=''):
    shape = (8, 8)
    evidence = np.zeros((8,) + shape, dtype=np.float32)
    evidence[0, 2:6, 2:6] = 1.0
    evidence[6:8, 2:6, 2:6] = 1.0
    passable = np.zeros(shape, dtype=bool)
    obstacle = np.zeros(shape, dtype=bool)
    ambiguous = np.zeros(shape, dtype=bool)
    observed = np.zeros(shape, dtype=bool)
    passable[2:4, 2:6] = True
    observed[2:6, 2:6] = True
    controlled = np.zeros(shape, dtype=np.int32)
    policy = []
    if blueprint:
        obstacle[5, 4] = True
        controlled[5, 4] = 2
        policy = [{
            'actor_id': 10,
            'blueprint': 'static.prop.' + blueprint,
            'disposition': 'obstacle',
            'policy_source': 'test',
        }]
    else:
        obstacle[5, 5] = True
    ambiguous[4, 4] = True
    np.savez_compressed(
        path,
        lidar_bev=evidence[:4],
        lidar_evidence_bev=evidence,
        local_ground_relative_max_height_m=evidence[4:6],
        local_ground_valid_mask=np.ones((2,) + shape, dtype=bool),
        target_passable_surface_mask=passable,
        target_obstacle_evidence_mask=obstacle,
        target_ambiguous_observed_mask=ambiguous,
        target_observed_mask=observed,
        target_controlled_obstacle_point_count=controlled,
        target_obstacle_instance_id=np.full(shape, -1, dtype=np.int64),
        target_ego_exclusion_mask=np.zeros(shape, dtype=bool),
        controlled_actor_policy_json=np.asarray(json.dumps(policy)),
    )


def _row(
    path,
    session='session_a',
    family='background',
    role='control',
    pair_group=None,
):
    row = {
        'split': 'train',
        'scene_id': 'scene_a',
        'object_family': family,
        'role': role,
        'evaluation_slice': 'control' if role == 'control' else 'seen_family',
        'source_session': session,
        'source_sample_id': '1',
        'derived_sample_path': str(path),
    }
    if pair_group is not None:
        row['pair_group'] = pair_group
    return row


def test_dataset_preserves_unknown_and_independent_targets(tmp_path):
    path = tmp_path / 'sample.npz'
    _sample(path, 'motorhelmet')
    item = TraversabilityEvidenceDataset([
        _row(path, family='motorhelmet', role='controlled')
    ])[0]
    assert tuple(item['lidar_evidence_bev'].shape) == (8, 8, 8)
    assert item['passable_target'][2, 2] == 1
    assert item['obstacle_target'][5, 4] == 1
    assert not item['known_evidence_mask'][4, 4]
    assert item['ambiguous_mask'][4, 4]
    assert item['controlled_obstacle_mask'][5, 4]


def test_temporal_17_channel_input_normalizes_two_evidence_blocks():
    evidence = np.zeros((17, 4, 4), dtype=np.float32)
    evidence[0, 1, 1] = 1.0
    evidence[2, 1, 1] = 0.5
    evidence[8, 2, 2] = 1.0
    evidence[10, 2, 2] = 1.0
    evidence[16] = 0.6
    normalized = normalize_lidar_evidence_bev(evidence)
    assert normalized.shape == (17, 4, 4)
    assert normalized[0, 1, 1] == 1.0
    assert normalized[8, 2, 2] == 1.0
    assert normalized[16, 0, 0] == pytest.approx(0.6)


def test_decoupled_model_accepts_temporal_17_channel_input():
    model = BevTraversabilityDecoupledEvidenceUNet(
        input_channels=17, base_channels=4
    )
    outputs = model(torch.randn(2, 17, 32, 32))
    assert outputs['passable_logits'].shape == (2, 32, 32)
    assert outputs['obstacle_logits'].shape == (2, 32, 32)


def test_independent_loss_does_not_supervise_unknown_cells():
    passable_logits = torch.zeros((1, 2, 2), requires_grad=True)
    obstacle_logits = torch.zeros((1, 2, 2), requires_grad=True)
    passable = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    obstacle = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    known = torch.tensor([[[True, True], [False, False]]])
    controlled = torch.tensor([[[False, True], [False, False]]])
    loss, parts = independent_evidence_loss(
        {
            'passable_logits': passable_logits,
            'obstacle_logits': obstacle_logits,
        },
        passable,
        obstacle,
        known,
        controlled,
        obstacle_positive_weight=2.0,
    )
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert float(parts['controlled_obstacle_loss'].detach()) > 0
    assert passable_logits.grad[0, 1, 0] == 0
    assert obstacle_logits.grad[0, 1, 1] == 0


def test_paired_loss_rewards_actor_specific_obstacle_contrast():
    target = torch.tensor([[[0.0, 1.0]]])
    known = torch.tensor([[[True, True]]])
    controlled = torch.tensor([[[False, True]]])
    common = {
        'passable_logits': torch.tensor([[[2.0, -2.0]]]),
        'obstacle_logits': torch.tensor([[[-2.0, 2.0]]]),
    }
    no_contrast, no_contrast_parts = independent_evidence_loss(
        common,
        1.0 - target,
        target,
        known,
        controlled,
        paired_control_obstacle_logits=torch.tensor([[[-2.0, 2.0]]]),
        paired_counterfactual_weight=1.0,
        paired_counterfactual_margin=2.0,
    )
    good_contrast, good_contrast_parts = independent_evidence_loss(
        common,
        1.0 - target,
        target,
        known,
        controlled,
        paired_control_obstacle_logits=torch.tensor([[[-2.0, -2.0]]]),
        paired_counterfactual_weight=1.0,
        paired_counterfactual_margin=2.0,
    )
    assert good_contrast < no_contrast
    assert (
        good_contrast_parts['paired_counterfactual_loss']
        < no_contrast_parts['paired_counterfactual_loss']
    )


def test_paired_dataset_uses_same_scene_control_and_flip(tmp_path):
    control_path = tmp_path / 'control.npz'
    object_path = tmp_path / 'object.npz'
    _sample(control_path)
    _sample(object_path, 'motorhelmet')
    rows = [
        _row(control_path, session='control'),
        _row(
            object_path, session='object', family='motorhelmet',
            role='controlled',
        ),
    ]
    dataset = PairedTraversabilityEvidenceDataset(
        rows, horizontal_flip_probability=1.0
    )
    item = dataset[1]
    assert item['controlled_obstacle_mask'][5, 3]
    assert item['paired_control_lidar_evidence_bev'][0, 2, 5] == 1.0


def test_paired_dataset_supports_multiple_pose_controls_in_one_scene(
    tmp_path,
):
    control_a = tmp_path / 'control_a.npz'
    control_b = tmp_path / 'control_b.npz'
    object_a = tmp_path / 'object_a.npz'
    _sample(control_a)
    _sample(control_b)
    _sample(object_a, 'motorhelmet')
    with np.load(control_b, allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    values['lidar_evidence_bev'][0, 0, 0] = 1.0
    np.savez_compressed(control_b, **values)
    rows = [
        _row(control_a, session='control_a', pair_group='pose_a'),
        _row(control_b, session='control_b', pair_group='pose_b'),
        _row(
            object_a,
            session='object_a',
            family='motorhelmet',
            role='controlled',
            pair_group='pose_a',
        ),
    ]
    dataset = PairedTraversabilityEvidenceDataset(rows)
    item = dataset[2]
    assert item['paired_control_lidar_evidence_bev'][0, 0, 0] == 0.0


def test_paired_dataset_rejects_duplicate_control_in_pair_group(tmp_path):
    first = tmp_path / 'first.npz'
    second = tmp_path / 'second.npz'
    _sample(first)
    _sample(second)
    rows = [
        _row(first, session='first', pair_group='pose_a'),
        _row(second, session='second', pair_group='pose_a'),
    ]
    with pytest.raises(ValueError, match='multiple controls'):
        PairedTraversabilityEvidenceDataset(rows)


def test_model_has_independent_heads_and_supports_backward():
    model = BevTraversabilityEvidenceUNet(base_channels=4)
    outputs = model(torch.randn(2, 8, 32, 32))
    assert outputs['passable_logits'].shape == (2, 32, 32)
    assert outputs['obstacle_logits'].shape == (2, 32, 32)
    total = (
        outputs['passable_logits'].mean()
        + outputs['obstacle_logits'].mean()
    )
    total.backward()


def test_local_model_has_bounded_context_without_coordinates():
    model = BevTraversabilityLocalEvidenceNet(base_channels=4)
    outputs = model(torch.randn(2, 8, 32, 32))
    assert outputs['passable_logits'].shape == (2, 32, 32)
    assert outputs['obstacle_logits'].shape == (2, 32, 32)
    assert not model.use_coordinate_channels


def test_model_factory_keeps_legacy_checkpoint_compatibility():
    legacy = build_traversability_evidence_model({
        'input_channels': 8,
        'base_channels': 4,
        'dropout_probability': 0.0,
        'use_coordinate_channels': True,
    })
    local = build_traversability_evidence_model({
        'architecture': 'local_evidence',
        'input_channels': 8,
        'base_channels': 4,
        'dropout_probability': 0.0,
        'use_coordinate_channels': False,
    })
    assert isinstance(legacy, BevTraversabilityEvidenceUNet)
    assert isinstance(local, BevTraversabilityLocalEvidenceNet)


def test_decoupled_evidence_model_uses_independent_networks():
    model = build_traversability_evidence_model({
        'architecture': 'decoupled_unet_context',
        'input_channels': 8,
        'base_channels': 4,
        'dropout_probability': 0.0,
        'use_coordinate_channels': True,
    })
    outputs = model(torch.randn(2, 8, 32, 32))
    assert isinstance(model, BevTraversabilityDecoupledEvidenceUNet)
    assert outputs['passable_logits'].shape == (2, 32, 32)
    assert outputs['obstacle_logits'].shape == (2, 32, 32)
    passable_ids = {id(value) for value in model.passable_network.parameters()}
    obstacle_ids = {id(value) for value in model.obstacle_network.parameters()}
    assert passable_ids.isdisjoint(obstacle_ids)


def _selection_metrics(unsafe, paired_false, recall, obstacle_iou):
    return {
        'controlled_unsafe_passable_instance_rate': unsafe,
        'paired_control_false_obstacle_rate': paired_false,
        'controlled_instance_any_detection_rate': recall,
        'obstacle_head': {'iou': obstacle_iou},
    }


def test_obstacle_selection_prioritizes_safety_gates_before_iou():
    unsafe_high_iou = evidence_train._obstacle_selection_key(
        _selection_metrics(0.25, 0.0, 1.0, 0.99),
        0.1,
        minimum_controlled_instance_recall=0.95,
    )
    safe_lower_iou = evidence_train._obstacle_selection_key(
        _selection_metrics(0.0, 0.0, 0.95, 0.70),
        1.0,
        minimum_controlled_instance_recall=0.95,
    )
    assert safe_lower_iou > unsafe_high_iou


def test_obstacle_selection_uses_iou_after_recall_gate():
    higher_recall = evidence_train._obstacle_selection_key(
        _selection_metrics(0.0, 0.0, 1.0, 0.70),
        0.1,
        minimum_controlled_instance_recall=0.95,
    )
    higher_iou = evidence_train._obstacle_selection_key(
        _selection_metrics(0.0, 0.0, 0.95, 0.80),
        1.0,
        minimum_controlled_instance_recall=0.95,
    )
    assert higher_iou > higher_recall


def test_obstacle_selection_minimizes_recall_shortfall_before_iou():
    low_recall_high_iou = evidence_train._obstacle_selection_key(
        _selection_metrics(0.0, 0.0, 0.50, 0.99),
        0.1,
        minimum_controlled_instance_recall=0.95,
    )
    near_gate_lower_iou = evidence_train._obstacle_selection_key(
        _selection_metrics(0.0, 0.0, 0.90, 0.70),
        1.0,
        minimum_controlled_instance_recall=0.95,
    )
    assert near_gate_lower_iou > low_recall_high_iou


def test_metrics_report_controlled_instance_detection():
    outputs = {
        'passable_logits': torch.tensor([[[-4.0, 4.0]]]),
        'obstacle_logits': torch.tensor([[[4.0, -4.0]]]),
    }
    metric = TraversabilityEvidenceMetricAccumulator()
    metric.update(
        outputs,
        torch.tensor([[[0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[True, True]]]),
        torch.tensor([[[True, False]]]),
        torch.tensor([[[True, True]]]),
        torch.tensor([[[1.0, 1.0]]]),
    )
    result = metric.compute()
    assert result['obstacle_head']['f1'] == pytest.approx(1.0)
    assert result['controlled_obstacle_cell_recall'] == pytest.approx(1.0)
    assert (
        result['controlled_instance_any_detection_rate']
        == pytest.approx(1.0)
    )
    assert result['controlled_unsafe_passable_cells'] == 0
    assert result['controlled_unsafe_passable_instance_rate'] == 0.0


def test_metrics_separate_obstacle_miss_from_unsafe_passable():
    metric = TraversabilityEvidenceMetricAccumulator()
    metric.update(
        {
            'passable_logits': torch.tensor([[[8.0, 8.0]]]),
            'obstacle_logits': torch.tensor([[[-8.0, 0.0]]]),
        },
        torch.tensor([[[0.0, 0.0]]]),
        torch.tensor([[[1.0, 1.0]]]),
        torch.tensor([[[True, True]]]),
        torch.tensor([[[True, True]]]),
        torch.tensor([[[True, True]]]),
        torch.tensor([[[1.0, 1.0]]]),
    )
    result = metric.compute()
    assert result['controlled_obstacle_cell_recall'] == 0.5
    assert result['controlled_unsafe_passable_cells'] == 1
    assert result['controlled_unsafe_passable_cell_rate'] == 0.5
    assert result['controlled_unsafe_passable_instance_rate'] == 0.0


def test_metrics_report_unsafe_passable_only_when_instance_is_undetected():
    metric = TraversabilityEvidenceMetricAccumulator()
    metric.update(
        {
            'passable_logits': torch.tensor([[[8.0]]]),
            'obstacle_logits': torch.tensor([[[-8.0]]]),
        },
        torch.tensor([[[0.0]]]),
        torch.tensor([[[1.0]]]),
        torch.tensor([[[True]]]),
        torch.tensor([[[True]]]),
        torch.tensor([[[True]]]),
        torch.tensor([[[1.0]]]),
    )
    result = metric.compute()
    assert result['controlled_obstacle_cell_recall'] == 0.0
    assert result['controlled_unsafe_passable_cell_rate'] == 1.0
    assert result['controlled_unsafe_passable_instance_rate'] == 1.0


def test_family_balancing_gives_each_family_equal_mass():
    rows = [
        {'object_family': 'background'},
        {'object_family': 'motorhelmet'},
        {'object_family': 'motorhelmet'},
    ]
    weights = family_balanced_sample_weights(rows)
    assert weights[0] == pytest.approx(weights[1] + weights[2])


def test_spec_rejects_scene_leakage():
    spec = {
        'schema_version': 1,
        'target_contract': 'independent_passable_obstacle_evidence',
        'sessions': [
            {
                'source_session': 'a', 'scene_id': 'same', 'split': 'train',
                'object_family': 'background', 'role': 'control',
                'evaluation_slice': 'control',
            },
            {
                'source_session': 'b', 'scene_id': 'same',
                'split': 'validation', 'object_family': 'motorhelmet',
                'role': 'controlled', 'evaluation_slice': 'seen_family',
            },
        ],
    }
    with pytest.raises(ValueError, match='scene leakage'):
        _validate_spec(spec)


def test_spec_accepts_multiple_pair_groups_in_one_scene():
    def entry(session, scene, split, role, pair_group):
        return {
            'source_session': session,
            'scene_id': scene,
            'pair_group': pair_group,
            'split': split,
            'object_family': (
                'background' if role == 'control' else 'motorhelmet'
            ),
            'role': role,
            'evaluation_slice': (
                'control' if role == 'control' else 'seen_family'
            ),
        }

    spec = {
        'schema_version': 1,
        'target_contract': 'independent_passable_obstacle_evidence',
        'sessions': [
            entry('train_a_control', 'train_scene', 'train', 'control', 'a'),
            entry(
                'train_a_object', 'train_scene', 'train',
                'controlled', 'a',
            ),
            entry('train_b_control', 'train_scene', 'train', 'control', 'b'),
            entry('val_control', 'val_scene', 'validation', 'control', 'a'),
            entry('test_control', 'test_scene', 'test', 'control', 'a'),
        ],
    }
    assert len(_validate_spec(spec)) == 5


def test_prepare_experiment_freezes_explicit_splits(tmp_path):
    sessions = [
        (
            'train_control', 'train_scene', 'train', 'background',
            'control', '',
        ),
        (
            'train_object', 'train_scene', 'train', 'motorhelmet',
            'controlled', 'motorhelmet',
        ),
        (
            'val_control', 'val_scene', 'validation', 'background',
            'control', '',
        ),
        (
            'val_object', 'val_scene', 'validation', 'motorhelmet',
            'controlled', 'motorhelmet',
        ),
        (
            'test_control', 'test_scene', 'test', 'background',
            'control', '',
        ),
        (
            'test_object', 'test_scene', 'test', 'trashcan01',
            'controlled', 'trashcan01',
        ),
    ]
    manifest_rows = []
    spec_rows = []
    for index, values in enumerate(sessions):
        session, scene, split, family, role, blueprint = values
        sample = tmp_path / f'{session}.npz'
        _sample(sample, blueprint)
        manifest_rows.append({
            'source_session': session,
            'source_sample_id': '1',
            'source_sample_path': str(sample),
            'derived_sample_path': str(sample),
            'status': 'written',
            'scan_fingerprint': str(index),
            'controlled_actor_return_count': '1' if blueprint else '0',
        })
        spec_rows.append({
            'source_session': session,
            'scene_id': scene,
            'split': split,
            'object_family': family,
            'role': role,
            'evaluation_slice': (
                'family_holdout' if family == 'trashcan01'
                else ('control' if role == 'control' else 'seen_family')
            ),
        })
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'schema_version': 1,
        'target_contract': 'independent_passable_obstacle_evidence',
        'sessions': spec_rows,
    }))
    summary = prepare_experiment(manifest, spec, tmp_path / 'prepared')
    assert summary['sample_counts'] == {
        'train': 2, 'validation': 2, 'test': 2,
    }
    assert summary['scene_leakage_count'] == 0
    assert summary['family_holdout_families'] == ['trashcan01']
    assert summary['pair_group_counts'] == {
        'train': 1, 'validation': 1, 'test': 1,
    }
    with (tmp_path / 'prepared' / 'split_manifest.csv').open(
        newline=''
    ) as stream:
        written = list(csv.DictReader(stream))
    assert all(row['pair_group'] == row['scene_id'] for row in written)


def test_cross_validation_fold_keeps_frozen_test_untouched(tmp_path):
    rows = [
        {
            'split': 'train', 'scene_id': 'scene_a',
            'source_session': 'a',
        },
        {
            'split': 'train', 'scene_id': 'scene_b',
            'source_session': 'b',
        },
        {
            'split': 'validation', 'scene_id': 'scene_c',
            'source_session': 'c',
        },
        {
            'split': 'test', 'scene_id': 'scene_test',
            'source_session': 'test',
        },
    ]
    manifest = tmp_path / 'fold.csv'
    counts = evidence_cv._write_fold_manifest(rows, 'scene_b', manifest)
    with manifest.open(newline='') as stream:
        written = list(csv.DictReader(stream))
    by_scene = {row['scene_id']: row['split'] for row in written}
    assert by_scene == {
        'scene_a': 'train',
        'scene_b': 'validation',
        'scene_c': 'train',
        'scene_test': 'test',
    }
    assert counts == {'train': 2, 'validation': 1, 'test': 1}


def test_consumed_test_promotion_is_explicit_and_keeps_untouched_test():
    rows = [
        {
            'split': 'test', 'scene_id': 'scene_consumed',
            'source_session': 'consumed_control',
            'evaluation_slice': 'consumed_test_do_not_use',
        },
        {
            'split': 'test', 'scene_id': 'scene_consumed',
            'source_session': 'consumed_actor',
            'evaluation_slice': 'consumed_test_do_not_use',
        },
        {
            'split': 'test', 'scene_id': 'scene_untouched',
            'source_session': 'untouched',
            'evaluation_slice': 'untouched_test',
        },
    ]
    prepared, scenes, sessions = (
        evidence_cv._promote_consumed_test_rows(rows)
    )
    assert [row['split'] for row in prepared] == [
        'train', 'train', 'test',
    ]
    assert prepared[0]['evaluation_slice'] == (
        'promoted_consumed_test_development'
    )
    assert scenes == ['scene_consumed']
    assert sessions == ['consumed_actor', 'consumed_control']
