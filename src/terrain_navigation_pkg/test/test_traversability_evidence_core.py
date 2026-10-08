import numpy as np

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
)
from terrain_navigation_pkg.traversability_evidence_core import (
    AMBIGUOUS_DISPOSITION,
    OBSTACLE_DECISION,
    OBSTACLE_DISPOSITION,
    PASSABLE_DECISION,
    PASSABLE_DISPOSITION,
    UNKNOWN_DECISION,
    build_conservative_evidence_decision,
    build_local_ground_evidence,
    build_vehicle_clearance_obstacle_mask,
    build_vehicle_evidence_targets,
)
from terrain_navigation_pkg.traversability_learning_core import EgoFootprint


def _geometry():
    return BevGeometry(
        x_min_m=0.0,
        x_max_m=4.0,
        y_min_m=-2.0,
        y_max_m=2.0,
        resolution_m=0.5,
        z_min_m=-2.0,
        z_max_m=2.0,
    )


def _cell(geometry, forward, left):
    row = int((geometry.x_max_m - forward) // geometry.resolution_m)
    column = int((geometry.y_max_m - left) // geometry.resolution_m)
    return row, column


def _tiny_footprint():
    return EgoFootprint(rear_m=0.1, front_m=0.1, half_width_m=0.1)


def test_low_obstacle_probability_alone_never_declares_passable():
    result = build_conservative_evidence_decision(
        passable_probability=np.asarray([[0.05]], dtype=np.float32),
        obstacle_probability=np.asarray([[0.01]], dtype=np.float32),
        observed_mask=np.asarray([[True]]),
        passable_support=np.asarray([[1.0]], dtype=np.float32),
    )
    assert result.decision[0, 0] == UNKNOWN_DECISION
    assert result.unknown_mask[0, 0]


def test_passable_requires_positive_evidence_and_support():
    result = build_conservative_evidence_decision(
        passable_probability=np.asarray([[0.98, 0.98]], dtype=np.float32),
        obstacle_probability=np.asarray([[0.01, 0.01]], dtype=np.float32),
        observed_mask=np.asarray([[True, True]]),
        passable_support=np.asarray([[0.8, 0.0]], dtype=np.float32),
    )
    assert result.decision[0, 0] == PASSABLE_DECISION
    assert result.decision[0, 1] == UNKNOWN_DECISION


def test_obstacle_evidence_wins_probability_conflict():
    result = build_conservative_evidence_decision(
        passable_probability=np.asarray([[0.99]], dtype=np.float32),
        obstacle_probability=np.asarray([[0.99]], dtype=np.float32),
        observed_mask=np.asarray([[True]]),
        passable_support=np.asarray([[1.0]], dtype=np.float32),
    )
    assert result.decision[0, 0] == OBSTACLE_DECISION
    assert result.obstacle_accepted[0, 0]
    assert not result.passable_accepted[0, 0]


def test_vehicle_clearance_hard_obstacle_cannot_be_learned_free():
    result = build_conservative_evidence_decision(
        passable_probability=np.asarray([[0.99]], dtype=np.float32),
        obstacle_probability=np.asarray([[0.01]], dtype=np.float32),
        observed_mask=np.asarray([[True]]),
        passable_support=np.asarray([[1.0]], dtype=np.float32),
        hard_obstacle_mask=np.asarray([[True]]),
    )
    assert result.decision[0, 0] == OBSTACLE_DECISION
    assert result.hard_obstacle_mask[0, 0]
    assert not result.learned_obstacle_mask[0, 0]
    assert not result.passable_accepted[0, 0]


def test_vehicle_clearance_mask_uses_local_ground_prominence():
    lidar = np.zeros((4, 3, 3), dtype=np.float32)
    lidar[0, 1, 1] = 1.0
    lidar[2, 1, 1] = -1.65
    relative = np.zeros((2, 3, 3), dtype=np.float32)
    relative[:, 1, 1] = 0.22
    valid = np.ones((2, 3, 3), dtype=bool)
    hard = build_vehicle_clearance_obstacle_mask(
        lidar, relative, valid
    )
    assert hard[1, 1]


def test_controlled_actor_disposition_overrides_static_prop_taxonomy():
    geometry = _geometry()
    points = np.asarray(
        [
            [3.25, 0.25, -0.95],
            [2.75, 0.25, -0.95],
            [2.25, 0.25, -0.95],
        ],
        dtype=np.float32,
    )
    tags = np.asarray([20, 20, 20], dtype=np.int32)
    actor_ids = np.asarray([10, 11, 12], dtype=np.int64)
    result = build_vehicle_evidence_targets(
        points,
        tags,
        geometry,
        _tiny_footprint(),
        semantic_object_ids=actor_ids,
        actor_dispositions={
            10: PASSABLE_DISPOSITION,
            11: OBSTACLE_DISPOSITION,
            12: AMBIGUOUS_DISPOSITION,
        },
    )

    passable_cell = _cell(geometry, 3.25, 0.25)
    obstacle_cell = _cell(geometry, 2.75, 0.25)
    ambiguous_cell = _cell(geometry, 2.25, 0.25)
    assert result.passable_surface_mask[passable_cell]
    assert not result.obstacle_evidence_mask[passable_cell]
    assert result.obstacle_evidence_mask[obstacle_cell]
    assert result.obstacle_instance_id[obstacle_cell] == 11
    assert result.ambiguous_observed_mask[ambiguous_cell]
    assert not result.passable_surface_mask[ambiguous_cell]
    assert not result.obstacle_evidence_mask[ambiguous_cell]


def test_different_obstacle_in_same_cell_wins_over_passable_override():
    geometry = _geometry()
    points = np.asarray(
        [
            [3.25, 0.25, -0.95],
            [3.24, 0.24, -0.60],
        ],
        dtype=np.float32,
    )
    result = build_vehicle_evidence_targets(
        points,
        np.asarray([20, 6], dtype=np.int32),
        geometry,
        _tiny_footprint(),
        semantic_object_ids=np.asarray([10, 99], dtype=np.int64),
        actor_dispositions={10: PASSABLE_DISPOSITION},
    )
    cell = _cell(geometry, 3.25, 0.25)
    assert result.obstacle_evidence_mask[cell]
    assert not result.passable_surface_mask[cell]


def test_local_ground_prominence_detects_neighboring_cell_protrusion():
    geometry = BevGeometry(
        x_min_m=0.0,
        x_max_m=5.0,
        y_min_m=-2.5,
        y_max_m=2.5,
        resolution_m=0.5,
        z_min_m=-2.0,
        z_max_m=2.0,
    )
    ground = []
    for forward in np.arange(1.25, 4.0, 0.5):
        for left in np.arange(-1.25, 1.5, 0.5):
            ground.append([forward, left, -1.0])
    obstacle = [3.25, 0.25, -0.65]
    bev = build_lidar_bev(np.asarray(ground + [obstacle]), geometry)
    local = build_local_ground_evidence(
        bev,
        geometry,
        radii_m=(1.0,),
        minimum_support_cells=4,
    )
    cell = _cell(geometry, obstacle[0], obstacle[1])
    assert local.valid_mask[0][cell]
    assert local.relative_max_height[0][cell] > 0.30


def test_sparse_local_ground_stays_unsupported_instead_of_guessing():
    geometry = _geometry()
    point = np.asarray([[3.25, 0.25, -0.5]], dtype=np.float32)
    bev = build_lidar_bev(point, geometry)
    local = build_local_ground_evidence(
        bev,
        geometry,
        radii_m=(0.75,),
        minimum_support_cells=4,
    )
    cell = _cell(geometry, 3.25, 0.25)
    assert not local.valid_mask[0][cell]
    assert local.relative_max_height[0][cell] == 0.0
