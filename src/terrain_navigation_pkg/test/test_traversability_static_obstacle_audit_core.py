import numpy as np
import pytest

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
)
from terrain_navigation_pkg.traversability_static_obstacle_audit_core import (
    StaticObstacleAuditAccumulator,
    comparison_masks,
    evaluate_frame,
    force_controlled_obstacle_tag,
    semantic_obstacle_instance_metrics,
)


def test_controlled_actor_unknown_tag_is_overridden_without_mutation():
    indices = np.asarray([0, 43, 43, 99], dtype=np.uint32)
    tags = np.asarray([1, 0, 0, 6], dtype=np.uint32)

    effective, count = force_controlled_obstacle_tag(indices, tags, 43)

    assert count == 2
    assert effective.tolist() == [1, 20, 20, 6]
    assert tags.tolist() == [1, 0, 0, 6]


def test_disabled_controlled_actor_override_reuses_tags():
    indices = np.asarray([43], dtype=np.uint32)
    tags = np.asarray([0], dtype=np.uint32)

    effective, count = force_controlled_obstacle_tag(indices, tags, -1)

    assert count == 0
    assert effective is tags


def test_static_audit_distinguishes_safe_and_unsafe_changes():
    target = np.asarray([
        [-1, 0, 1],
        [0, 1, 0],
    ], dtype=np.int8)
    baseline = np.asarray([
        [0, 1, 1],
        [0, 1, 0],
    ], dtype=bool)
    candidate = np.asarray([
        [0, 0, 0],
        [0, 1, 1],
    ], dtype=bool)

    metrics = evaluate_frame(target, baseline, candidate)

    assert metrics['target_known'] == 5
    assert metrics['target_obstacle'] == 2
    assert metrics['removed_cells'] == 2
    assert metrics['removed_target_free'] == 1
    assert metrics['removed_target_obstacle'] == 1
    assert metrics['added_target_free'] == 1
    assert metrics['added_target_obstacle'] == 0
    assert metrics['baseline_obstacle_recall'] == 1.0
    assert metrics['candidate_obstacle_recall'] == 0.5
    assert metrics['removed_target_obstacle_rate'] == 0.5


def test_static_audit_accumulates_cell_observations():
    target = np.asarray([[0, 1]], dtype=np.int8)
    baseline = np.asarray([[0, 1]], dtype=bool)
    candidate = np.asarray([[0, 1]], dtype=bool)
    accumulator = StaticObstacleAuditAccumulator()

    accumulator.update(target, baseline, candidate)
    accumulator.update(target, baseline, candidate)
    summary = accumulator.compute()

    assert summary['frames'] == 2
    assert summary['target_known'] == 4
    assert summary['candidate_true_obstacle'] == 2
    assert summary['candidate_obstacle_iou'] == 1.0


def test_static_audit_rejects_shape_and_label_errors():
    target = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError, match='equal shape'):
        comparison_masks(target, np.zeros((2, 3)), np.zeros((2, 2)))
    target[0, 0] = 7
    with pytest.raises(ValueError, match='unknown, free, or obstacle'):
        comparison_masks(target, np.zeros((2, 2)), np.zeros((2, 2)))


def test_semantic_instances_distinguish_partial_and_complete_removal():
    geometry = BevGeometry(
        x_min_m=0.0,
        x_max_m=4.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        resolution_m=1.0,
        z_min_m=-2.0,
        z_max_m=2.0,
    )
    xyz = np.asarray([
        [2.5, 0.5, 0.0],
        [2.5, -0.5, 0.0],
        [1.5, 0.5, 0.0],
    ], dtype=np.float32)
    indices = np.asarray([101, 101, 202], dtype=np.uint32)
    tags = np.asarray([20, 20, 6], dtype=np.uint32)
    target = np.zeros((4, 4), dtype=np.int8)
    target[1, 1:3] = 1
    target[2, 1] = 1
    baseline = target == 1
    candidate = baseline.copy()
    candidate[1, 2] = False
    candidate[2, 1] = False

    metrics = semantic_obstacle_instance_metrics(
        xyz,
        indices,
        tags,
        target,
        baseline,
        candidate,
        geometry,
        corridor_minimum_x_m=0.0,
        corridor_maximum_x_m=4.0,
        corridor_half_width_m=2.0,
    )

    assert [item['object_idx'] for item in metrics] == [101, 202]
    assert metrics[0]['baseline_hits'] == 2
    assert metrics[0]['candidate_hits'] == 1
    assert metrics[0]['fully_removed'] == 0
    assert metrics[0]['candidate_to_baseline_retention'] == 0.5
    assert metrics[1]['baseline_hits'] == 1
    assert metrics[1]['candidate_hits'] == 0
    assert metrics[1]['fully_removed'] == 1


def test_semantic_instances_ignore_free_tags_and_outside_corridor():
    geometry = BevGeometry(
        x_min_m=0.0,
        x_max_m=4.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        resolution_m=1.0,
        z_min_m=-2.0,
        z_max_m=2.0,
    )
    target = np.ones((4, 4), dtype=np.int8)
    occupied = np.ones((4, 4), dtype=bool)
    metrics = semantic_obstacle_instance_metrics(
        np.asarray([[2.5, 0.5, 0.0], [2.5, 1.5, 0.0]]),
        np.asarray([1, 2]),
        np.asarray([1, 20]),
        target,
        occupied,
        occupied,
        geometry,
        corridor_minimum_x_m=0.0,
        corridor_maximum_x_m=4.0,
        corridor_half_width_m=1.0,
    )
    assert metrics == []
