import numpy as np
import pytest

from terrain_navigation_pkg.traversability_inference_core import (
    bev_to_occupancy_grid,
)
from terrain_navigation_pkg.traversability_online_evaluation_core import (
    OnlineTraversabilityAccumulator,
    build_swept_footprint_mask,
    evaluate_traversability_frame,
    finite_alignment_summary,
    largest_connected_component,
    occupancy_grid_to_bev,
)
from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
)


def test_occupancy_grid_conversion_is_exact_inverse():
    bev = np.arange(12, dtype=np.int16).reshape((3, 4))
    assert np.array_equal(
        occupancy_grid_to_bev(bev_to_occupancy_grid(bev)), bev
    )


def test_frame_metrics_separate_abstention_from_unsafe_free():
    target = np.asarray([[1, 1, 0, 0, -1]], dtype=np.int8)
    probability = np.asarray([[90, 10, 80, 10, 90]], dtype=np.int16)
    decision = np.asarray([[100, -1, 100, 0, 100]], dtype=np.int16)

    result = evaluate_traversability_frame(
        target, probability, decision
    )

    assert result['target_known'] == 4
    assert result['raw_true_obstacle'] == 1
    assert result['raw_false_free'] == 1
    assert result['raw_false_obstacle'] == 1
    assert result['raw_true_free'] == 1
    assert result['raw_obstacle_iou'] == pytest.approx(1.0 / 3.0)
    assert result['selective_coverage'] == pytest.approx(0.75)
    assert result['selective_false_free'] == 0
    assert result['selective_abstained_obstacle'] == 1
    assert result['selective_safe_retention_rate'] == pytest.approx(1.0)


def test_geometric_override_is_audited_and_accumulated():
    target = np.asarray([[1, 0]], dtype=np.int8)
    probability = np.asarray([[10, 10]], dtype=np.int16)
    decision = np.asarray([[100, 0]], dtype=np.int16)
    accumulator = OnlineTraversabilityAccumulator()

    frame = accumulator.update(target, probability, decision)
    accumulator.update(target, probability, decision)
    total = accumulator.compute()

    assert frame['geometric_override_obstacle'] == 1
    assert total['frames'] == 2
    assert total['geometric_override_obstacle'] == 2
    assert total['selective_accepted_accuracy'] == pytest.approx(1.0)


def test_alignment_summary_ignores_non_finite_values():
    result = finite_alignment_summary([0.0, 0.01, float('nan')])
    assert result['mean_s'] == pytest.approx(0.005)
    assert result['maximum_s'] == pytest.approx(0.01)


def test_invalid_selective_decision_is_rejected():
    with pytest.raises(ValueError, match='decision values'):
        evaluate_traversability_frame(
            np.asarray([[0]], dtype=np.int8),
            np.asarray([[10]], dtype=np.int16),
            np.asarray([[50]], dtype=np.int16),
        )


def test_evaluation_mask_limits_all_target_counts():
    target = np.asarray([[1, 1], [0, 0]], dtype=np.int8)
    probability = np.asarray([[10, 90], [90, 10]], dtype=np.int16)
    decision = np.asarray([[0, 100], [100, 0]], dtype=np.int16)
    selected = np.asarray([[True, False], [True, False]])

    result = evaluate_traversability_frame(
        target, probability, decision, evaluation_mask=selected
    )

    assert result['target_known'] == 2
    assert result['target_obstacle'] == 1
    assert result['selective_false_free'] == 1
    assert result['selective_false_obstacle'] == 1


def test_swept_footprint_mask_follows_oriented_path():
    geometry = BevGeometry(
        x_min_m=-2.0,
        x_max_m=6.0,
        y_min_m=-4.0,
        y_max_m=4.0,
        resolution_m=1.0,
    )
    path = np.asarray([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, np.pi / 2.0],
    ])
    mask = build_swept_footprint_mask(
        geometry,
        path,
        vehicle_front_m=1.0,
        vehicle_rear_m=1.0,
        half_width_m=0.6,
    )

    assert mask.shape == (8, 8)
    assert np.count_nonzero(mask) > 0
    # Far-left corner lies outside every sampled oriented rectangle.
    assert not mask[0, 0]


def test_largest_component_uses_diagonal_connectivity():
    value = np.asarray([
        [True, False, False, False],
        [False, True, False, True],
        [False, False, True, True],
    ])
    result = largest_connected_component(value)
    assert result == {
        'cell_count': 5,
        'row_span_cells': 3,
        'column_span_cells': 4,
    }
