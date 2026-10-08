import math

import numpy as np

from terrain_navigation_pkg.navigation_trajectory_core import (
    analyze_nav2_plan,
    forward_polyline_from_vehicle,
    initial_bc_trajectory_decision,
    median_nearest_distance,
    resample_polyline,
    transform_vehicle_points,
)


def test_initial_bc_decision_excludes_unusable_targets_and_winding_paths():
    eligible, reasons = initial_bc_trajectory_decision([
        'short_horizon', 'excessive_curvature',
    ])
    assert eligible
    assert reasons == ()

    eligible, reasons = initial_bc_trajectory_decision([
        'winding_path', 'excessive_curvature',
    ])
    assert not eligible
    assert reasons == ('winding_path',)

    eligible, reasons = initial_bc_trajectory_decision([
        'short_horizon', 'no_valid_target',
    ])
    assert not eligible
    assert reasons == ('no_valid_target',)


def test_analysis_flags_path_shorter_than_first_target():
    result = analyze_nav2_plan(
        [[0.0, 0.0], [0.5, 0.0]], [0.5, 0.0],
        spacing_m=0.75, target_count=12,
    )
    assert 'no_valid_target' in result.flags


def test_forward_polyline_projects_origin_onto_closest_segment():
    path, distance = forward_polyline_from_vehicle(np.array([
        [-2.0, 1.0],
        [2.0, 1.0],
        [5.0, 1.0],
    ]))
    np.testing.assert_allclose(path[0], [0.0, 1.0])
    np.testing.assert_allclose(path[-1], [5.0, 1.0])
    assert math.isclose(distance, 1.0)


def test_resample_polyline_uses_arc_length_and_padding_mask():
    target, mask = resample_polyline(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]),
        spacing_m=1.0,
        target_count=5,
    )
    np.testing.assert_allclose(target[:4], [
        [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0],
    ])
    np.testing.assert_allclose(target[4], [2.0, 2.0])
    np.testing.assert_array_equal(mask, [True, True, True, True, False])


def test_analysis_flags_winding_and_excessive_curvature():
    # Start at the vehicle and turn through more than 1.5 revolutions of
    # heading without returning to the origin.  A closed circle would make
    # the closest-point projection ambiguous at its end.
    angle = np.linspace(0.0, 1.75 * math.pi, 80)
    loop = np.column_stack((
        2.0 * np.sin(angle),
        2.0 - 2.0 * np.cos(angle),
    ))
    result = analyze_nav2_plan(
        loop, [10.0, 0.0], maximum_curvature_per_m=0.25,
    )
    assert 'winding_path' in result.flags
    assert 'excessive_curvature' in result.flags


def test_transform_vehicle_points_accounts_for_translation_and_yaw():
    transformed = transform_vehicle_points(
        [[2.0, 0.0]],
        source_pose=[10.0, 5.0, 0.0, math.pi / 2.0],
        target_pose=[10.0, 6.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(transformed, [[0.0, 1.0]], atol=1.0e-7)


def test_median_nearest_distance_is_zero_for_matching_vertices():
    first = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert median_nearest_distance(first, first) == 0.0
