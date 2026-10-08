import numpy as np
import pytest

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
)
from terrain_navigation_pkg.traversability_learning_core import (
    EgoFootprint,
    FREE_LABEL,
    OBSTACLE_LABEL,
    UNKNOWN_LABEL,
    build_conservative_visibility_evidence,
    build_semantic_traversability_targets,
)


def _geometry():
    return BevGeometry(
        x_min_m=0.0,
        x_max_m=4.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        resolution_m=1.0,
        z_min_m=-2.0,
        z_max_m=2.0,
    )


def test_surface_types_are_equal_free_space():
    points = np.asarray([
        [3.5, 1.5, -1.0],  # Road
        [3.4, 1.4, -0.99],
        [2.5, 0.5, -1.0],  # Sidewalk
        [2.4, 0.4, -0.99],
        [1.5, -0.5, -1.0],  # Terrain / grass
        [1.4, -0.4, -0.99],
        [0.5, -1.5, -1.0],  # Ground
        [0.4, -1.4, -0.99],
    ], dtype=np.float32)
    result = build_semantic_traversability_targets(
        points, np.asarray([1, 1, 2, 2, 10, 10, 25, 25]), _geometry()
    )
    assert np.count_nonzero(result.labels == FREE_LABEL) == 4
    assert np.count_nonzero(result.labels == OBSTACLE_LABEL) == 0


def test_hard_obstacle_wins_over_surface_return():
    points = np.asarray([
        [3.5, 1.5, -1.0],
        [3.4, 1.4, -0.99],
        [3.45, 1.45, 0.5],
    ], dtype=np.float32)
    result = build_semantic_traversability_targets(
        points, np.asarray([1, 1, 3]), _geometry()
    )
    assert result.labels[0, 0] == OBSTACLE_LABEL


def test_vertical_discontinuity_preserves_curb_inside_surface_tag():
    points = np.asarray([
        [3.5, 1.5, -1.0],
        [3.4, 1.4, -0.78],
    ], dtype=np.float32)
    result = build_semantic_traversability_targets(
        points, np.asarray([2, 2]), _geometry(),
        obstacle_vertical_span_m=0.15,
    )
    assert result.labels[0, 0] == OBSTACLE_LABEL


def test_unsupported_or_ambiguous_tags_remain_unknown():
    points = np.asarray([[3.5, 1.5, -1.0]], dtype=np.float32)
    result = build_semantic_traversability_targets(
        points, np.asarray([22]), _geometry()
    )
    assert result.labels[0, 0] == UNKNOWN_LABEL


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        build_semantic_traversability_targets(
            np.zeros((2, 4)), np.zeros(2), _geometry()
        )


def _cell(geometry, forward, left):
    row = int((geometry.x_max_m - forward) // geometry.resolution_m)
    column = int((geometry.y_max_m - left) // geometry.resolution_m)
    return row, column


def test_ego_vehicle_returns_are_masked_as_unknown():
    geometry = BevGeometry(
        x_min_m=-3.0, x_max_m=5.0,
        y_min_m=-3.0, y_max_m=3.0,
        resolution_m=0.5, z_min_m=-2.0, z_max_m=2.0,
    )
    points = np.asarray([
        [1.0, 0.0, -0.5],  # Ego vehicle return.
        [3.25, 0.25, -1.0],  # Road support outside the body.
        [3.20, 0.20, -0.99],
    ], dtype=np.float32)
    footprint = EgoFootprint(rear_m=2.5, front_m=2.4, half_width_m=1.0)

    result = build_semantic_traversability_targets(
        points, np.asarray([14, 1, 1]), geometry,
        ego_footprint=footprint,
    )

    assert result.labels[_cell(geometry, 1.0, 0.0)] == UNKNOWN_LABEL
    assert result.labels[_cell(geometry, 3.25, 0.25)] == FREE_LABEL
    assert result.ego_exclusion_mask[_cell(geometry, 1.0, 0.0)]


def test_visibility_stops_before_obstacle_and_stays_separate_from_labels():
    geometry = BevGeometry(
        x_min_m=0.0, x_max_m=6.0,
        y_min_m=-3.0, y_max_m=3.0,
        resolution_m=1.0, z_min_m=-2.0, z_max_m=2.0,
    )
    points = np.asarray([
        [3.5, 0.1, 0.0],   # Wall limits the ray.
        [5.5, 0.1, -1.0],  # A higher ray reaches ground behind it.
        [5.4, 0.0, -0.99],
    ], dtype=np.float32)
    footprint = EgoFootprint(rear_m=0.4, front_m=0.4, half_width_m=0.4)
    targets = build_semantic_traversability_targets(
        points, np.asarray([3, 1, 1]), geometry,
        ego_footprint=footprint,
    )

    visibility = build_conservative_visibility_evidence(
        points, geometry, targets,
        ego_footprint=footprint,
        angular_bin_count=360,
    )

    assert visibility.free_mask[_cell(geometry, 2.0, 0.05)]
    assert not visibility.free_mask[_cell(geometry, 3.5, 0.1)]
    assert not visibility.free_mask[_cell(geometry, 4.5, 0.12)]
    assert targets.labels[_cell(geometry, 2.0, 0.05)] == UNKNOWN_LABEL
