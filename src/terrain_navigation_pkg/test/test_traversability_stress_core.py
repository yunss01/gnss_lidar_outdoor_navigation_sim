import numpy as np

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
)
from terrain_navigation_pkg.traversability_stress_core import (
    apply_bev_stress,
    apply_random_sensor_augmentation,
)


def _example():
    geometry = BevGeometry(
        x_min_m=0.0, x_max_m=2.0,
        y_min_m=-1.0, y_max_m=1.0,
        resolution_m=1.0,
    )
    bev = np.zeros((4, 2, 2), dtype=np.float32)
    bev[0, 0, 0] = 1.0
    bev[1, 0, 0] = 0.8
    bev[2, 0, 0] = -1.8
    targets = np.asarray([[0, -1], [-1, 1]], dtype=np.int64)
    return geometry, bev, targets


def test_height_bias_changes_only_observed_height():
    geometry, bev, targets = _example()
    shifted, shifted_targets = apply_bev_stress(
        bev, targets, geometry, 'height_bias_plus_15cm', 1
    )
    assert shifted[2, 0, 0] == np.float32(-1.65)
    assert shifted[2, 0, 1] == 0.0
    assert np.array_equal(shifted_targets, targets)


def test_cell_dropout_removes_corresponding_targets():
    geometry, bev, targets = _example()
    bev[0] = 1.0
    bev[1] = 0.8
    dropped, valid_targets = apply_bev_stress(
        bev, targets, geometry, 'cell_dropout_15pct', 4
    )
    removed = dropped[0] == 0.0
    assert np.any(removed)
    assert np.all(valid_targets[removed] == -1)


def test_random_sensor_augmentation_is_seeded_and_keeps_empty_cells_empty():
    geometry, bev, targets = _example()
    first_bev, first_target = apply_random_sensor_augmentation(
        bev, targets, geometry, 19
    )
    second_bev, second_target = apply_random_sensor_augmentation(
        bev, targets, geometry, 19
    )
    assert np.array_equal(first_bev, second_bev)
    assert np.array_equal(first_target, second_target)
    assert np.all(first_bev[:, bev[0] == 0.0] == 0.0)
