import numpy as np
import pytest

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
)
from terrain_navigation_pkg.traversability_obstacle_fusion_core import (
    build_add_only_fusion_masks,
    build_candidate_fusion_masks,
    novel_voxel_indices,
    point_decisions,
    voxel_unique_indices,
)


def _geometry():
    return BevGeometry(
        x_min_m=-1.0,
        x_max_m=3.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        resolution_m=1.0,
        z_min_m=-2.0,
        z_max_m=2.0,
    )


def _dense_geometry():
    return BevGeometry(
        x_min_m=0.0,
        x_max_m=2.0,
        y_min_m=-1.0,
        y_max_m=1.0,
        resolution_m=0.25,
        z_min_m=-2.0,
        z_max_m=2.0,
    )


def _road_points(geometry, slope=0.0):
    xs = np.arange(
        geometry.x_min_m + 0.125, geometry.x_max_m, 0.25
    )
    ys = np.arange(
        geometry.y_min_m + 0.125, geometry.y_max_m, 0.25
    )
    return np.asarray([
        [x, y, -1.6 + float(slope) * x]
        for x in xs
        for y in ys
    ], dtype=np.float32)


def test_point_decisions_preserve_unknown_outside_bev():
    decision = np.full((4, 4), -1, dtype=np.int8)
    decision[1, 1] = 0
    decision[2, 2] = 100
    points = np.asarray([
        [1.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [5.0, 0.0, 0.0],
    ])
    sampled, inside = point_decisions(points, decision, _geometry())
    assert sampled.tolist() == [0, 100, -1]
    assert inside.tolist() == [True, True, False]


def test_candidate_fusion_clears_free_adds_obstacle_and_keeps_unknown():
    geometry = _geometry()
    decision = np.full((4, 4), -1, dtype=np.int8)
    decision[1, 1] = 0
    decision[2, 2] = 100
    raw = np.asarray([
        [1.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, 1.5, 0.0],
        [0.0, 0.0, 0.0],
    ])
    baseline = np.asarray([
        [1.5, 0.5, 0.0],
        [-0.5, 1.5, 0.0],
    ])
    result = build_candidate_fusion_masks(
        raw,
        baseline,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=10.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.2,
        ego_rear_m=0.2,
        ego_half_width_m=0.2,
        hard_obstacle_minimum_z_m=1.0,
        hard_obstacle_vertical_span_m=0.15,
    )
    assert result.baseline_cleared_by_ai_mask.tolist() == [True, False]
    assert result.baseline_keep_mask.tolist() == [False, True]
    assert result.raw_ai_obstacle_mask.tolist() == [False, True, False, False]


def test_candidate_fusion_does_not_add_ego_or_out_of_range_points():
    geometry = _geometry()
    decision = np.full((4, 4), 100, dtype=np.int8)
    raw = np.asarray([
        [0.0, 0.0, 0.0],
        [0.4, 1.5, 0.0],
        [2.5, 1.5, 0.0],
    ])
    result = build_candidate_fusion_masks(
        raw,
        np.empty((0, 3)),
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=2.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.3,
        ego_rear_m=0.3,
        ego_half_width_m=0.3,
        hard_obstacle_minimum_z_m=1.0,
        hard_obstacle_vertical_span_m=0.15,
    )
    assert result.raw_ai_obstacle_mask.tolist() == [False, True, False]


def test_candidate_fusion_retains_baseline_outside_ai_change_range():
    geometry = _geometry()
    decision = np.zeros((4, 4), dtype=np.int8)
    baseline = np.asarray([
        [1.5, 0.5, 0.0],
        [2.5, 1.5, 0.0],
    ])
    result = build_candidate_fusion_masks(
        np.empty((0, 3)),
        baseline,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=2.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.2,
        ego_rear_m=0.2,
        ego_half_width_m=0.2,
        hard_obstacle_minimum_z_m=1.0,
        hard_obstacle_vertical_span_m=0.15,
    )
    assert result.baseline_cleared_by_ai_mask.tolist() == [True, False]
    assert result.baseline_keep_mask.tolist() == [False, True]


def test_add_only_selects_obstacle_cell_top_without_free_or_unknown():
    geometry = _geometry()
    decision = np.full((4, 4), -1, dtype=np.int8)
    decision[1, 1] = 0
    decision[2, 2] = 100
    raw = np.asarray([
        [1.5, 0.5, -1.5],
        [0.5, -0.5, -1.6],
        [0.5, -0.5, -1.2],
        [-0.5, 1.5, 0.0],
    ])
    result = build_add_only_fusion_masks(
        raw,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=10.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.2,
        ego_rear_m=0.2,
        ego_half_width_m=0.2,
        obstacle_cell_top_band_m=0.05,
        minimum_relative_height_m=0.07,
        local_ground_radius_m=1.0,
        local_ground_quantile=0.25,
        local_ground_minimum_support_cells=1,
    )
    assert result.raw_ai_obstacle_mask.tolist() == [False, False, True, False]


def test_add_only_rejects_flat_road_even_when_model_calls_it_obstacle():
    geometry = _dense_geometry()
    raw = _road_points(geometry)
    decision = np.full(
        (geometry.height, geometry.width), 100, dtype=np.int8
    )
    result = build_add_only_fusion_masks(
        raw,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=12.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.05,
        ego_rear_m=0.05,
        ego_half_width_m=0.05,
        obstacle_cell_top_band_m=0.05,
        minimum_relative_height_m=0.07,
        local_ground_radius_m=0.75,
        local_ground_quantile=0.25,
        local_ground_minimum_support_cells=4,
    )
    assert np.count_nonzero(
        result.raw_obstacle_before_ground_gate_mask
    ) > 0
    assert not np.any(result.raw_ai_obstacle_mask)
    assert np.array_equal(
        result.raw_ground_gate_rejected_mask,
        result.raw_obstacle_before_ground_gate_mask,
    )


def test_add_only_rejects_ordinary_sloped_road():
    geometry = _dense_geometry()
    raw = _road_points(geometry, slope=0.04)
    decision = np.full(
        (geometry.height, geometry.width), 100, dtype=np.int8
    )
    result = build_add_only_fusion_masks(
        raw,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=12.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.05,
        ego_rear_m=0.05,
        ego_half_width_m=0.05,
        minimum_relative_height_m=0.07,
        local_ground_radius_m=0.75,
        local_ground_quantile=0.25,
        local_ground_minimum_support_cells=4,
    )
    assert not np.any(result.raw_ai_obstacle_mask)


def test_add_only_keeps_raised_return_with_supported_local_ground():
    geometry = _dense_geometry()
    road = _road_points(geometry)
    target = np.asarray([[1.125, 0.125, -1.50]], dtype=np.float32)
    raw = np.concatenate((road, target), axis=0)
    decision = np.full(
        (geometry.height, geometry.width), -1, dtype=np.int8
    )
    rows = int(np.floor(
        (geometry.x_max_m - target[0, 0]) / geometry.resolution_m
    ))
    columns = int(np.floor(
        (geometry.y_max_m - target[0, 1]) / geometry.resolution_m
    ))
    decision[rows, columns] = 100
    result = build_add_only_fusion_masks(
        raw,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=12.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.05,
        ego_rear_m=0.05,
        ego_half_width_m=0.05,
        minimum_relative_height_m=0.07,
        local_ground_radius_m=0.75,
        local_ground_quantile=0.25,
        local_ground_minimum_support_cells=4,
    )
    assert result.raw_ai_obstacle_mask[-1]
    assert np.count_nonzero(result.raw_ai_obstacle_mask) == 1


def test_add_only_rejects_raised_return_outside_ai_range():
    geometry = _dense_geometry()
    road = _road_points(geometry)
    target = np.asarray([[1.875, 0.125, -1.50]], dtype=np.float32)
    raw = np.concatenate((road, target), axis=0)
    decision = np.full(
        (geometry.height, geometry.width), 100, dtype=np.int8
    )
    result = build_add_only_fusion_masks(
        raw,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=1.5,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.05,
        ego_rear_m=0.05,
        ego_half_width_m=0.05,
        minimum_relative_height_m=0.07,
        local_ground_radius_m=0.75,
        local_ground_quantile=0.25,
        local_ground_minimum_support_cells=4,
    )
    assert not result.raw_ai_obstacle_mask[-1]


def test_add_only_rejects_candidate_without_valid_local_ground():
    geometry = _dense_geometry()
    raw = np.asarray([[1.125, 0.125, -1.50]], dtype=np.float32)
    decision = np.full(
        (geometry.height, geometry.width), 100, dtype=np.int8
    )
    result = build_add_only_fusion_masks(
        raw,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=12.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.05,
        ego_rear_m=0.05,
        ego_half_width_m=0.05,
        minimum_relative_height_m=0.07,
        local_ground_radius_m=0.75,
        local_ground_quantile=0.25,
        local_ground_minimum_support_cells=4,
    )
    assert not result.raw_ai_obstacle_mask[0]
    assert result.raw_ground_gate_rejected_mask[0]


def test_novel_voxels_preserve_every_existing_point_and_stable_candidates():
    existing = np.asarray([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],
    ])
    candidates = np.asarray([
        [0.05, 0.05, 0.05],
        [0.21, 0.01, 0.01],
        [0.22, 0.02, 0.02],
        [0.41, 0.01, 0.01],
    ])
    assert novel_voxel_indices(existing, candidates, 0.2).tolist() == [1, 3]


def test_hard_geometry_veto_keeps_baseline_without_bulk_raw_addition():
    geometry = _geometry()
    decision = np.full((4, 4), -1, dtype=np.int8)
    decision[2, 2] = 100
    raw = np.asarray([[0.5, -0.5, 0.0]])
    baseline = raw.copy()
    result = build_candidate_fusion_masks(
        raw,
        baseline,
        decision,
        geometry,
        minimum_range_m=0.1,
        maximum_range_m=10.0,
        obstacle_maximum_z_m=1.0,
        ego_front_m=0.2,
        ego_rear_m=0.2,
        ego_half_width_m=0.2,
        hard_obstacle_minimum_z_m=-0.5,
        hard_obstacle_vertical_span_m=0.15,
    )
    assert result.baseline_keep_mask.tolist() == [True]
    assert result.raw_ai_obstacle_mask.tolist() == [False]


def test_invalid_decision_and_voxel_size_are_rejected():
    decision = np.zeros((4, 4), dtype=np.int8)
    decision[0, 0] = 5
    with pytest.raises(ValueError, match='decision values'):
        point_decisions(np.zeros((1, 3)), decision, _geometry())
    with pytest.raises(ValueError, match='voxel_size'):
        voxel_unique_indices(np.zeros((1, 3)), 0.0)


def test_voxel_unique_indices_keep_first_record_stably():
    points = np.asarray([
        [0.01, 0.01, 0.01],
        [0.05, 0.05, 0.05],
        [0.21, 0.01, 0.01],
    ])
    assert voxel_unique_indices(points, 0.2).tolist() == [0, 2]
