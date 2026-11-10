"""Exercise real FAR selection methods without ROS sockets or actuation."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from terrain_navigation_pkg.far_guide_core import (
    GuideGrid, terminal_approach_target,
)
from terrain_navigation_pkg.far_nav2_guide_node import FarNav2GuideNode


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, -1.596])
def test_rejoin_target_is_on_the_route_eight_metres_before_goal(yaw):
    direction = math.cos(yaw), math.sin(yaw)
    goal = (40 * direction[0], 40 * direction[1])
    target, active, passed = terminal_approach_target(
        (0, 0), goal, yaw, 8, 1)
    assert target == pytest.approx((32 * direction[0], 32 * direction[1]))
    assert active and not passed


def test_disabled_mode_does_not_need_tangent_or_change_goal():
    assert terminal_approach_target((0, 0), (40, 0), None, 0, 1) == (
        (40, 0), False, False)


@pytest.mark.parametrize('start', [(31, 0), (35, 3), (42, 0)])
def test_checkpoint_plane_never_requests_a_backward_target(start):
    assert terminal_approach_target(start, (40, 0), 0, 8, 1) == (
        (40, 0), False, True)
    # A replan/pose jitter must not resurrect an already passed checkpoint.
    assert terminal_approach_target((29, 0), (40, 0), 0, 8, 1, True) == (
        (40, 0), False, True)


@pytest.mark.parametrize('yaw,distance,capture', [
    (None, 8, 1), (float('nan'), 8, 1), (0, -1, 1),
    (0, float('inf'), 1), (0, 8, 0), (0, 8, 9),
])
def test_invalid_rejoin_input_fails_closed(yaw, distance, capture):
    with pytest.raises(ValueError):
        terminal_approach_target((0, 0), (40, 0), yaw, distance, capture)


def fixture(distance=8.0):
    phases = []
    node = SimpleNamespace(
        terminal_approach_distance_m=distance,
        terminal_approach_capture_m=1.0,
        terminal_approach_passed=False,
        terminal_approach_active=False,
        terminal_approach_target_xy=None,
        terminal_approach_phase=None,
        latest_goal_tangent_yaw_rad=0.0,
        terminal_approach_publisher=SimpleNamespace(
            publish=lambda msg: phases.append(msg.data)),
        _log=lambda *_args, **_kwargs: None,
        _planning_direction_continuity=lambda _now: (None, 0, 0, ''),
        boundary_margin_m=3.0, start_clearance_m=2.0,
        goal_search_radius_m=4.0, cost_weight=4.0, maximum_expansions=120000,
        failed_corridor_radius_m=1.5, failed_corridor_cost_weight=12.0,
        path_efficiency_escape_mode=False, segment_lookahead_m=12.0,
        path_efficiency_retry_min_lookahead_m=5.0,
        path_efficiency_retry_lookahead_step_m=1.5,
        path_efficiency_retry_count=0, final_goal_distance_m=10.0,
        goal_tangent_max_bias_rad=math.radians(20),
        relaxed_subgoal_yaw_until_wall_time=0.0,
    )
    grid = GuideGrid(np.zeros((100, 120)), np.zeros((100, 120), dtype=bool),
                     0.5, -10, -20)
    return node, grid, phases


def test_actual_far_methods_plan_an_early_forward_rejoin_pose_then_final():
    node, grid, phases = fixture()
    start, goal = (25, 3), (40, 0)
    path = FarNav2GuideNode._plan_world_path(node, grid, start, goal, 10, [])
    assert path[-1] == pytest.approx((32, 0))
    target, yaw, final = FarNav2GuideNode._select_segment_target(
        node, path, goal, math.dist(start, goal), start, 10)
    assert target == pytest.approx((32, 0))
    assert yaw == 0 and not final
    assert phases == ['approach']
    # Efficiency retries cannot silently discard the treatment's heading.
    node.relaxed_subgoal_yaw_until_wall_time = 100.0
    assert FarNav2GuideNode._select_segment_target(
        node, path, goal, math.dist(start, goal), start, 10)[1] == 0
    start = (31.2, 0.2)
    path = FarNav2GuideNode._plan_world_path(node, grid, start, goal, 11, [])
    assert path[-1] == pytest.approx(goal)
    assert phases == ['approach', 'final']
    assert node.terminal_approach_passed


def test_blocked_rejoin_cell_is_not_relocated_or_cleared():
    node, grid, _phases = fixture()
    x, y = grid.world_to_cell((32, 0))
    grid.blocked[y, x] = True
    path = FarNav2GuideNode._plan_world_path(
        node, grid, (25, 3), (40, 0), 10, [])
    assert path is None
    assert grid.blocked[y, x]  # Original lethal cell is still hard.


def test_bent_guide_does_not_stop_short_and_lose_rejoin_heading():
    node, _grid, _phases = fixture()
    node.terminal_approach_active = True
    node.terminal_approach_target_xy = (32, 0)
    path = [(25, 3), (28, 3), (32, 0)]
    target, yaw, final = FarNav2GuideNode._select_segment_target(
        node, path, (40, 0), math.dist((25, 3), (40, 0)), (25, 3), 10)
    assert target == pytest.approx((32, 0))
    assert yaw == 0 and not final


def test_baseline_keeps_original_mission_goal():
    node, grid, phases = fixture(0.0)
    node.latest_goal_tangent_yaw_rad = None
    path = FarNav2GuideNode._plan_world_path(
        node, grid, (25, 3), (40, 0), 10, [])
    assert path[-1] == pytest.approx((40, 0))
    assert phases == ['disabled']
