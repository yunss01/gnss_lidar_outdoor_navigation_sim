import math

import numpy as np
import pytest

from terrain_navigation_pkg.far_guide_core import build_guide_grid
from terrain_navigation_pkg.far_guide_core import assess_path_efficiency
from terrain_navigation_pkg.far_guide_core import assess_active_replan
from terrain_navigation_pkg.far_guide_core import add_direction_continuity_cost
from terrain_navigation_pkg.far_guide_core import add_failed_corridor_cost
from terrain_navigation_pkg.far_guide_core import bounded_heading_preference
from terrain_navigation_pkg.far_guide_core import can_accept_bounded_topology_escape
from terrain_navigation_pkg.far_guide_core import can_accept_length_only_detour
from terrain_navigation_pkg.far_guide_core import is_actionable_safety_stop
from terrain_navigation_pkg.far_guide_core import PathEfficiencyAssessment
from terrain_navigation_pkg.far_guide_core import line_of_sight
from terrain_navigation_pkg.far_guide_core import nearest_polyline_tangent
from terrain_navigation_pkg.far_guide_core import plan_online_guide
from terrain_navigation_pkg.far_guide_core import polylines_similar
from terrain_navigation_pkg.far_guide_core import prefix_polyline_to_point
from terrain_navigation_pkg.far_guide_core import remaining_polyline_length
from terrain_navigation_pkg.far_guide_core import retry_lookahead_distance
from terrain_navigation_pkg.far_guide_core import SafetyRejectionMonitor
from terrain_navigation_pkg.far_guide_core import SafetyReplanHold
from terrain_navigation_pkg.far_guide_core import select_direction_continuity
from terrain_navigation_pkg.far_guide_core import select_subgoal
from terrain_navigation_pkg.far_guide_core import (
    should_request_costmap_recovery,
)


def _grid(occupancy, resolution=1.0, stride=1):
    return build_guide_grid(
        np.asarray(occupancy, dtype=np.float32),
        resolution,
        0.0,
        0.0,
        stride=stride,
        lethal_threshold=65.0,
        unknown_cost=0.15,
    )


def test_bounded_heading_preference_limits_a_sharp_route_tangent():
    value = bounded_heading_preference(
        math.radians(40.0), math.radians(100.0), math.radians(20.0)
    )

    assert math.degrees(value) == pytest.approx(60.0)


def test_bounded_heading_preference_keeps_a_nearby_route_tangent():
    value = bounded_heading_preference(
        math.radians(40.0), math.radians(52.0), math.radians(20.0)
    )

    assert math.degrees(value) == pytest.approx(52.0)


def test_bounded_heading_preference_wraps_across_pi():
    value = bounded_heading_preference(
        math.radians(175.0), math.radians(-165.0), math.radians(10.0)
    )

    assert abs(math.degrees(value)) == pytest.approx(175.0)


def test_inefficient_path_retries_shorten_and_bound_the_subgoal():
    values = [
        retry_lookahead_distance(12.0, 5.0, 1.5, retry)
        for retry in range(8)
    ]

    assert values == [12.0, 10.5, 9.0, 7.5, 6.0, 5.0, 5.0, 5.0]


def test_straight_open_space_is_reduced_to_one_visible_segment():
    grid = _grid(np.zeros((30, 30), dtype=np.float32))
    path = plan_online_guide(
        grid,
        (2.5, 2.5),
        (25.5, 2.5),
        boundary_margin_m=1.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=10000,
    )

    assert path == [(2.5, 2.5), (25.5, 2.5)]


def test_long_guide_routes_around_a_wall_and_preserves_clearance():
    occupancy = np.zeros((30, 30), dtype=np.float32)
    occupancy[4:26, 14:16] = 100.0
    grid = _grid(occupancy)

    path = plan_online_guide(
        grid,
        (4.5, 15.5),
        (25.5, 15.5),
        boundary_margin_m=1.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=20000,
    )

    assert path is not None
    assert len(path) >= 3
    cells = [grid.world_to_cell(point) for point in path]
    assert all(
        line_of_sight(grid, first, second)
        for first, second in zip(cells, cells[1:])
    )
    assert any(point[1] < 4.5 or point[1] > 25.5 for point in path[1:-1])


def test_visibility_pruning_never_cuts_through_a_cost_99_band():
    occupancy = np.zeros((30, 30), dtype=np.float32)
    occupancy[4:26, 14:16] = 99.0
    grid = build_guide_grid(
        occupancy,
        resolution_m=1.0,
        origin_x_m=0.0,
        origin_y_m=0.0,
        stride=1,
        lethal_threshold=99.0,
        unknown_cost=0.15,
    )

    path = plan_online_guide(
        grid,
        (4.5, 15.5),
        (25.5, 15.5),
        boundary_margin_m=1.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=20000,
    )

    assert path is not None
    assert len(path) >= 3
    cells = [grid.world_to_cell(point) for point in path]
    assert all(
        line_of_sight(grid, first, second)
        for first, second in zip(cells, cells[1:])
    )


def test_distant_goal_is_clipped_to_the_rolling_grid_boundary():
    grid = _grid(np.zeros((20, 20), dtype=np.float32))
    path = plan_online_guide(
        grid,
        (10.5, 10.5),
        (100.0, 10.5),
        boundary_margin_m=2.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=10000,
    )

    assert path is not None
    assert math.isclose(path[-1][0], 17.5, abs_tol=0.51)
    assert math.isclose(path[-1][1], 10.5, abs_tol=0.51)


def test_distant_disconnected_goal_advances_to_reachable_frontier():
    occupancy = np.zeros((30, 30), dtype=np.float32)
    # A wall crosses the observed rolling map. The unseen distant goal lies
    # beyond it, but the vehicle can safely move to a side frontier and reveal
    # more space instead of remaining in guide_not_found forever.
    occupancy[14:16, :] = 100.0
    grid = _grid(occupancy)
    path = plan_online_guide(
        grid,
        (15.5, 6.5),
        (15.5, 100.0),
        boundary_margin_m=2.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=20000,
    )

    assert path is not None
    assert len(path) >= 2
    endpoint = grid.world_to_cell(path[-1])
    assert endpoint[0] <= 2 or endpoint[0] >= grid.width - 3
    assert endpoint[1] < 14
    assert math.hypot(
        path[-1][0] - path[0][0], path[-1][1] - path[0][1]
    ) >= 2.0


def test_inside_grid_disconnected_goal_advances_to_reachable_frontier():
    occupancy = np.zeros((30, 30), dtype=np.float32)
    occupancy[14:16, :] = 100.0
    grid = _grid(occupancy)
    path = plan_online_guide(
        grid,
        (15.5, 6.5),
        (15.5, 22.5),
        boundary_margin_m=2.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=20000,
    )

    assert path is not None
    assert len(path) >= 2
    endpoint = grid.world_to_cell(path[-1])
    assert endpoint[0] <= 2 or endpoint[0] >= grid.width - 3
    assert endpoint[1] < 14
    assert grid.world_to_cell(path[-1]) != grid.world_to_cell((15.5, 22.5))
    assert math.hypot(
        path[-1][0] - path[0][0], path[-1][1] - path[0][1]
    ) >= 2.0


def test_subgoal_interpolates_at_requested_lookahead():
    subgoal, yaw, reached_end = select_subgoal(
        [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0)],
        8.0,
    )

    assert subgoal == (5.0, 3.0)
    assert math.isclose(yaw, math.pi / 2.0)
    assert reached_end is False


def test_unknown_space_remains_traversable_but_is_not_lethal():
    occupancy = np.full((10, 10), -1.0, dtype=np.float32)
    grid = _grid(occupancy)

    assert not np.any(grid.blocked)
    assert np.allclose(grid.costs, 0.15)


def test_default_threshold_keeps_inscribed_and_lethal_costs_hard():
    occupancy = np.asarray([[0.0, 80.0, 99.0, 100.0]], dtype=np.float32)
    grid = build_guide_grid(
        occupancy,
        resolution_m=1.0,
        origin_x_m=0.0,
        origin_y_m=0.0,
        stride=1,
        unknown_cost=0.15,
    )

    assert grid.blocked.tolist() == [[False, False, True, True]]
    assert np.isclose(grid.costs[0, 1], 0.8)
    assert np.isclose(grid.costs[0, 2], 0.99)


def test_online_guide_can_escape_a_start_local_inscribed_cost_ring():
    occupancy = np.zeros((15, 15), dtype=np.float32)
    # Reproduce a vehicle pose enclosed by Nav2's value-99 inscribed band.
    # The bounded start release may open cost 99 around the valid pose.
    occupancy[5:10, 5] = 99.0
    occupancy[5:10, 9] = 99.0
    occupancy[5, 5:10] = 99.0
    occupancy[9, 5:10] = 99.0
    grid = build_guide_grid(
        occupancy,
        resolution_m=1.0,
        origin_x_m=0.0,
        origin_y_m=0.0,
        stride=1,
        lethal_threshold=99.0,
        unknown_cost=0.15,
    )

    path = plan_online_guide(
        grid,
        (7.5, 7.5),
        (12.5, 7.5),
        boundary_margin_m=0.0,
        start_clearance_m=2.0,
        goal_search_radius_m=1.0,
        cost_weight=4.0,
        maximum_expansions=10000,
    )

    assert path is not None
    assert path[-1] == (12.5, 7.5)


def test_start_release_never_opens_a_truly_lethal_wall():
    occupancy = np.zeros((15, 15), dtype=np.float32)
    occupancy[:, 6] = 100.0
    grid = build_guide_grid(
        occupancy,
        resolution_m=1.0,
        origin_x_m=0.0,
        origin_y_m=0.0,
        stride=1,
        lethal_threshold=99.0,
        unknown_cost=0.15,
    )

    path = plan_online_guide(
        grid,
        (4.5, 7.5),
        (10.5, 7.5),
        boundary_margin_m=0.0,
        start_clearance_m=3.0,
        goal_search_radius_m=1.0,
        cost_weight=4.0,
        maximum_expansions=10000,
    )

    # The mapless frontier fallback may advance along the reachable side of
    # the wall, but it must never cross the truly lethal column to the goal.
    assert path is not None
    assert all(point[0] < 6.0 for point in path)
    assert path[-1] != (10.5, 7.5)


def test_remaining_length_projects_vehicle_onto_old_guide():
    remaining = remaining_polyline_length(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        (6.0, 1.0),
    )

    assert math.isclose(remaining, 15.0)


def test_nearest_polyline_tangent_uses_vehicle_local_segment():
    heading = nearest_polyline_tangent(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        (10.2, 6.0),
    )

    assert math.isclose(heading, math.pi / 2.0)


def test_direction_continuity_is_soft_and_fades_with_distance():
    grid = _grid(np.zeros((21, 21), dtype=np.float32))
    preferred = add_direction_continuity_cost(
        grid,
        start_world=(10.5, 10.5),
        heading_rad=0.0,
        distance_m=8.0,
        cost_weight=1.0,
    )

    ahead = preferred.costs[10, 13]
    sideways = preferred.costs[13, 10]
    behind = preferred.costs[10, 7]
    far_sideways = preferred.costs[19, 10]
    assert ahead < sideways < behind
    assert math.isclose(far_sideways, grid.costs[19, 10])
    assert not np.any(preferred.blocked)


def test_direction_preference_never_opens_a_blocked_cell():
    occupancy = np.zeros((15, 15), dtype=np.float32)
    occupancy[7, 9] = 100.0
    grid = _grid(occupancy)
    preferred = add_direction_continuity_cost(
        grid,
        start_world=(7.5, 7.5),
        heading_rad=0.0,
        distance_m=8.0,
        cost_weight=1.0,
    )

    assert preferred.blocked[7, 9]


def test_direction_preference_does_not_forbid_required_opposite_route():
    grid = _grid(np.zeros((20, 20), dtype=np.float32))
    path = plan_online_guide(
        grid,
        (10.5, 10.5),
        (2.5, 10.5),
        boundary_margin_m=1.0,
        start_clearance_m=0.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=10000,
        continuity_heading_rad=0.0,
        continuity_distance_m=8.0,
        continuity_cost_weight=0.8,
    )

    assert path is not None
    assert path[-1] == (2.5, 10.5)


def test_successful_mission_leg_direction_is_used_without_a_retry():
    preference = select_direction_continuity(
        retry_heading_rad=None,
        leg_heading_rad=0.7,
        retry_distance_m=8.0,
        retry_cost_weight=0.8,
        leg_distance_m=12.0,
        leg_cost_weight=2.0,
    )

    assert preference == (0.7, 12.0, 2.0, 'mission_leg')


def test_immediate_retry_direction_overrides_mission_leg_direction():
    preference = select_direction_continuity(
        retry_heading_rad=-0.4,
        leg_heading_rad=0.7,
        retry_distance_m=8.0,
        retry_cost_weight=0.8,
        leg_distance_m=12.0,
        leg_cost_weight=2.0,
    )

    assert preference == (-0.4, 8.0, 0.8, 'retry')


def test_direction_continuity_is_empty_before_the_first_segment():
    preference = select_direction_continuity(
        retry_heading_rad=None,
        leg_heading_rad=None,
        retry_distance_m=8.0,
        retry_cost_weight=0.8,
        leg_distance_m=12.0,
        leg_cost_weight=2.0,
    )

    assert preference == (None, 0.0, 0.0, 'none')


def test_failed_corridor_soft_cost_penalizes_center_without_blocking_it():
    grid = _grid(np.zeros((21, 21), dtype=np.float32))
    penalized = add_failed_corridor_cost(
        grid,
        failed_paths=[[(5.5, 10.5), (15.5, 10.5)]],
        start_world=(5.5, 10.5),
        radius_m=2.0,
        cost_weight=10.0,
        start_release_m=1.0,
    )

    assert penalized.costs[10, 12] > penalized.costs[13, 12]
    assert not np.any(penalized.blocked)


def test_failed_corridor_hard_block_preserves_a_start_release_exit():
    grid = _grid(np.zeros((21, 21), dtype=np.float32))
    blocked = add_failed_corridor_cost(
        grid,
        failed_paths=[[(5.5, 10.5), (15.5, 10.5)]],
        start_world=(5.5, 10.5),
        radius_m=2.0,
        cost_weight=10.0,
        start_release_m=1.5,
        hard_block_radius_m=0.75,
    )

    assert not blocked.blocked[10, 5]
    assert blocked.blocked[10, 10]


def test_failed_corridor_similarity_allows_a_shifted_retry_start():
    failed = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
    same_corridor = [(2.0, 0.2), (6.0, 0.1), (10.0, 0.0)]
    alternative = [(2.0, 0.2), (5.0, 3.0), (10.0, 3.0)]

    assert polylines_similar(same_corridor, failed, 0.5)
    assert not polylines_similar(alternative, failed, 0.5)


def test_action_prefix_stops_at_the_subgoal_instead_of_full_guide_end():
    prefix = prefix_polyline_to_point(
        [(0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (20.0, 10.0)],
        (5.0, 4.0),
    )

    assert prefix == [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0)]


def test_hard_failed_corridor_forces_a_distinct_alternate_guide():
    grid = _grid(np.zeros((31, 31), dtype=np.float32))
    failed = [(3.5, 15.5), (15.5, 15.5)]
    path = plan_online_guide(
        grid,
        (3.5, 15.5),
        (27.5, 15.5),
        boundary_margin_m=1.0,
        start_clearance_m=1.5,
        goal_search_radius_m=2.0,
        cost_weight=4.0,
        maximum_expansions=20000,
        failed_corridors=[failed],
        failed_corridor_radius_m=2.0,
        failed_corridor_cost_weight=10.0,
        failed_corridor_start_release_m=1.5,
        failed_corridor_hard_block_radius_m=0.75,
    )

    assert path is not None
    assert any(abs(point[1] - 15.5) > 0.75 for point in path[1:])
    assert not polylines_similar(path, failed, 0.75)


def test_shorter_changed_corridor_requests_active_replan():
    assessment = assess_active_replan(
        active_path=[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
        active_subgoal=(0.0, 10.0),
        candidate_path=[(0.0, 0.0), (10.0, 10.0)],
        candidate_subgoal=(8.0, 8.0),
        current=(0.0, 0.0),
        path_blocked=False,
        minimum_subgoal_change_m=3.0,
        minimum_improvement_m=3.0,
        endpoint_tolerance_m=2.0,
    )

    assert assessment is not None
    assert assessment.reason == 'shorter_guide'
    assert assessment.improvement_m > 5.0


def test_valid_active_path_stays_sticky_when_optimization_is_disabled():
    assessment = assess_active_replan(
        active_path=[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)],
        active_subgoal=(0.0, 10.0),
        candidate_path=[(0.0, 0.0), (10.0, 10.0)],
        candidate_subgoal=(8.0, 8.0),
        current=(0.0, 0.0),
        path_blocked=False,
        minimum_subgoal_change_m=3.0,
        minimum_improvement_m=3.0,
        endpoint_tolerance_m=2.0,
        allow_valid_path_optimization=False,
    )

    assert assessment is None


def test_small_guide_jitter_does_not_request_active_replan():
    assessment = assess_active_replan(
        active_path=[(0.0, 0.0), (10.0, 0.0)],
        active_subgoal=(8.0, 0.0),
        candidate_path=[(0.0, 0.0), (10.0, 0.0)],
        candidate_subgoal=(8.4, 0.2),
        current=(1.0, 0.0),
        path_blocked=False,
        minimum_subgoal_change_m=3.0,
        minimum_improvement_m=3.0,
        endpoint_tolerance_m=2.0,
    )

    assert assessment is None


def test_blocked_path_jitter_does_not_cancel_the_active_action():
    assessment = assess_active_replan(
        active_path=[(0.0, 0.0), (10.0, 0.0)],
        active_subgoal=(8.0, 0.0),
        candidate_path=[(0.0, 0.0), (10.0, 0.0)],
        candidate_subgoal=(9.5, 0.0),
        current=(1.0, 0.0),
        path_blocked=True,
        minimum_subgoal_change_m=3.0,
        minimum_improvement_m=3.0,
        endpoint_tolerance_m=2.0,
        allow_valid_path_optimization=False,
    )

    assert assessment is None


def test_blocked_path_accepts_a_materially_different_corridor():
    assessment = assess_active_replan(
        active_path=[(0.0, 0.0), (10.0, 0.0)],
        active_subgoal=(8.0, 0.0),
        candidate_path=[(0.0, 0.0), (0.0, 8.0), (10.0, 0.0)],
        candidate_subgoal=(0.0, 8.0),
        current=(1.0, 0.0),
        path_blocked=True,
        minimum_subgoal_change_m=3.0,
        minimum_improvement_m=3.0,
        endpoint_tolerance_m=2.0,
        allow_valid_path_optimization=False,
    )

    assert assessment is not None
    assert assessment.reason == 'path_blocked'


def test_path_efficiency_accepts_a_normal_right_angle_detour():
    assessment = assess_path_efficiency(
        [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0)],
        reference_length_m=12.0,
        maximum_length_ratio=1.9,
        maximum_absolute_turn_rad=math.radians(360.0),
        maximum_signed_turn_rad=math.radians(220.0),
    )

    assert assessment is not None
    assert not assessment.inefficient
    assert math.isclose(
        math.degrees(assessment.signed_turn_rad), 90.0, abs_tol=1.0
    )


def test_path_efficiency_rejects_a_full_dubins_loop():
    path = []
    radius = 5.0
    for degree in range(0, 541, 10):
        angle = math.radians(degree)
        path.append((
            radius * math.sin(angle),
            radius * (1.0 - math.cos(angle)),
        ))
    assessment = assess_path_efficiency(
        path,
        reference_length_m=12.0,
        maximum_length_ratio=1.9,
        maximum_absolute_turn_rad=math.radians(360.0),
        maximum_signed_turn_rad=math.radians(220.0),
    )

    assert assessment is not None
    assert assessment.inefficient
    assert 'signed_loop' in assessment.reasons
    assert assessment.length_ratio > 1.9


def test_length_only_detour_is_accepted_after_one_topology_retry():
    assessment = assess_path_efficiency(
        [(0.0, 0.0), (0.0, 8.0), (6.0, 8.0), (6.0, 0.0)],
        reference_length_m=7.0,
        maximum_length_ratio=1.9,
        maximum_absolute_turn_rad=math.radians(360.0),
        maximum_signed_turn_rad=math.radians(220.0),
    )

    assert assessment is not None
    assert assessment.reasons == ('length_ratio',)
    assert not can_accept_length_only_detour(assessment, 0, 1)
    assert can_accept_length_only_detour(assessment, 1, 1)


def test_signed_loop_is_never_accepted_as_length_only_detour():
    path = []
    for degree in range(0, 541, 10):
        angle = math.radians(degree)
        path.append((
            5.0 * math.sin(angle),
            5.0 * (1.0 - math.cos(angle)),
        ))
    assessment = assess_path_efficiency(
        path,
        reference_length_m=12.0,
        maximum_length_ratio=1.9,
        maximum_absolute_turn_rad=math.radians(360.0),
        maximum_signed_turn_rad=math.radians(220.0),
    )

    assert assessment is not None
    assert 'signed_loop' in assessment.reasons
    assert not can_accept_length_only_detour(assessment, 10, 1)


def test_one_bounded_p_turn_is_accepted_only_after_retries():
    assessment = PathEfficiencyAssessment(
        path_length_m=28.0,
        reference_length_m=10.0,
        length_ratio=2.8,
        absolute_turn_rad=math.radians(325.0),
        signed_turn_rad=math.radians(320.0),
        inefficient=True,
        reasons=('length_ratio', 'signed_loop'),
    )
    limits = dict(
        minimum_previous_rejections=6,
        maximum_length_ratio=3.1,
        maximum_absolute_turn_rad=math.radians(350.0),
        maximum_signed_turn_rad=math.radians(330.0),
    )
    assert not can_accept_bounded_topology_escape(
        assessment, previous_rejections=5, already_used=False, **limits
    )
    assert can_accept_bounded_topology_escape(
        assessment, previous_rejections=6, already_used=False, **limits
    )
    assert not can_accept_bounded_topology_escape(
        assessment, previous_rejections=6, already_used=True, **limits
    )


def test_excessive_winding_is_not_a_bounded_p_turn_escape():
    assessment = PathEfficiencyAssessment(
        path_length_m=30.0,
        reference_length_m=10.0,
        length_ratio=3.0,
        absolute_turn_rad=math.radians(390.0),
        signed_turn_rad=math.radians(370.0),
        inefficient=True,
        reasons=('length_ratio', 'winding', 'signed_loop'),
    )
    assert not can_accept_bounded_topology_escape(
        assessment,
        previous_rejections=6,
        minimum_previous_rejections=6,
        already_used=False,
        maximum_length_ratio=3.1,
        maximum_absolute_turn_rad=math.radians(350.0),
        maximum_signed_turn_rad=math.radians(330.0),
    )


def test_costmap_recovery_requires_all_independent_clear_signals():
    arguments = dict(
        abort_count=2,
        minimum_abort_count=2,
        speed_mps=0.01,
        maximum_speed_mps=0.10,
        path_hard_valid=True,
        path_validity_fresh=True,
        safety_state='clear',
        safety_state_fresh=True,
        cooldown_ready=True,
    )
    assert should_request_costmap_recovery(**arguments)
    assert not should_request_costmap_recovery(
        **dict(arguments, safety_state='obstacle_stop')
    )
    assert not should_request_costmap_recovery(
        **dict(arguments, path_validity_fresh=False)
    )
    assert not should_request_costmap_recovery(
        **dict(arguments, speed_mps=0.5)
    )


def test_safety_rejection_monitor_requires_distinct_scans_in_window():
    monitor = SafetyRejectionMonitor(3, 1.0)

    assert not monitor.observe(10.0, 100)
    assert not monitor.observe(10.1, 100)
    assert not monitor.observe(10.2, 101)
    assert monitor.observe(10.3, 102)
    assert monitor.count == 3


def test_safety_rejection_monitor_restarts_after_window_expires():
    monitor = SafetyRejectionMonitor(2, 0.5)

    assert not monitor.observe(10.0, 100)
    assert not monitor.observe(10.7, 101)
    assert monitor.count == 1
    assert monitor.observe(10.8, 102)


def test_safety_rejection_monitor_reset_discards_old_evidence():
    monitor = SafetyRejectionMonitor(2, 1.0)
    assert not monitor.observe(10.0, 100)

    monitor.reset()

    assert not monitor.observe(10.1, 101)
    assert monitor.count == 1


def test_only_fresh_obstacle_stop_authorizes_safety_replan():
    assert is_actionable_safety_stop('obstacle_stop', 0.05, 1.5)
    assert not is_actionable_safety_stop('obstacle_recovery', 0.05, 1.5)
    assert not is_actionable_safety_stop('obstacle_caution', 0.05, 1.5)
    assert not is_actionable_safety_stop('clear', 0.05, 1.5)
    assert not is_actionable_safety_stop('obstacle_stop', 1.51, 1.5)


def test_safety_replan_hold_requires_consecutive_clear_states():
    hold = SafetyReplanHold(3, 1.0)
    hold.arm(10.0)

    hold.observe_state('clear')
    hold.observe_state('obstacle_recovery')
    hold.observe_state('clear')
    hold.observe_state('clear')
    assert hold.release_reason(10.5) is None

    hold.observe_state('clear')
    assert hold.release_reason(10.6) == 'clear_confirmed'


def test_safety_replan_hold_has_bounded_timeout():
    hold = SafetyReplanHold(3, 1.0)
    hold.arm(20.0)
    hold.observe_state('obstacle_stop')

    assert hold.release_reason(20.99) is None
    assert hold.release_reason(21.0) == 'timeout'

    hold.reset()
    assert not hold.active
    assert hold.release_reason(22.0) == 'inactive'


def test_path_efficiency_uses_far_reference_not_blocked_direct_distance():
    # A legitimate obstacle detour can be much longer than endpoint distance,
    # but it is not inefficient when it agrees with the FAR guide length.
    detour = [(0.0, 0.0), (0.0, 8.0), (10.0, 8.0), (10.0, 0.0)]
    assessment = assess_path_efficiency(
        detour,
        reference_length_m=26.0,
        maximum_length_ratio=1.9,
        maximum_absolute_turn_rad=math.radians(360.0),
        maximum_signed_turn_rad=math.radians(220.0),
    )

    assert assessment is not None
    assert not assessment.inefficient
    assert math.isclose(assessment.length_ratio, 1.0)
