import math

import pytest

from terrain_navigation_pkg.nav2_goal_bridge_core import (
    bounded_virtual_preview,
    distance_bounded_rolling_horizon_count,
    evaluate_waypoint_passage,
    goal_signature_changed,
    odom_route_from_enu_points,
    odom_target_from_goal_vector,
    rolling_horizon_global_remaining,
    rolling_horizon_next_start,
    rolling_real_poses_remaining,
    rolling_horizon_result_next_start,
    rolling_horizon_window,
    rolling_initial_start_index,
    rolling_waypoint_capture_radius,
    rolling_waypoint_turn_angle_deg,
    should_focus_rolling_waypoint,
    should_handoff_virtual_preview_after_waypoint,
    should_promote_near_goal_route_abort,
    should_retry_aborted_rolling_segment,
    should_retry_rolling_preview_as_current_only,
)


def test_waypoint_passage_accepts_precise_radius_without_crossing_arm():
    result = evaluate_waypoint_passage(
        (0.0, 0.0), (10.0, 0.0), (9.4, 0.7), 1.0, 2.25, False
    )

    assert result.passed
    assert result.reason == 'radius'


def test_waypoint_passage_accepts_armed_forward_gate_crossing():
    result = evaluate_waypoint_passage(
        (0.0, 0.0), (10.0, 0.0), (10.2, 1.97), 1.0, 2.25, True
    )

    assert result.passed
    assert result.reason == 'directed_crossing'
    assert result.cross_track_m == pytest.approx(1.97)


def test_waypoint_passage_rejects_unarmed_or_wide_crossing():
    unarmed = evaluate_waypoint_passage(
        (0.0, 0.0), (10.0, 0.0), (10.2, 1.97), 1.0, 2.25, False
    )
    wide = evaluate_waypoint_passage(
        (0.0, 0.0), (10.0, 0.0), (10.2, 2.30), 1.0, 2.25, True
    )

    assert not unarmed.passed
    assert not wide.passed


def test_goal_vector_is_anchored_to_current_odom_pose():
    target = odom_target_from_goal_vector(150.0, 380.0, 12.0, -5.0)

    assert target.x_m == pytest.approx(162.0)
    assert target.y_m == pytest.approx(375.0)
    assert target.yaw_rad == pytest.approx(math.atan2(-5.0, 12.0))


def test_goal_signature_filters_repeated_publications():
    assert goal_signature_changed(None, (10.0, 20.0), 0.25)
    assert not goal_signature_changed((10.0, 20.0), (10.1, 20.1), 0.25)
    assert goal_signature_changed((10.0, 20.0), (10.3, 20.0), 0.25)


def test_complete_route_is_anchored_and_oriented_along_approach_segments():
    poses = odom_route_from_enu_points(
        150.0,
        380.0,
        2.0,
        -1.0,
        [(7.0, 0.0), (7.0, -10.0), (20.0, -10.0)],
    )

    assert [(pose.x_m, pose.y_m) for pose in poses] == pytest.approx([
        (155.0, 381.0),
        (155.0, 371.0),
        (168.0, 371.0),
    ])
    assert poses[0].yaw_rad == pytest.approx(math.atan2(1.0, 5.0))
    assert poses[1].yaw_rad == pytest.approx(-math.pi / 2.0)
    assert poses[2].yaw_rad == pytest.approx(0.0)


def test_nearby_first_waypoint_keeps_its_reachable_approach_heading():
    poses = odom_route_from_enu_points(
        150.0,
        380.0,
        0.0,
        0.0,
        [(3.37, -2.29), (32.13, -15.65)],
    )

    assert poses[0].yaw_rad == pytest.approx(math.atan2(-2.29, 3.37))
    assert poses[1].yaw_rad == pytest.approx(
        math.atan2(-15.65 + 2.29, 32.13 - 3.37)
    )


def test_open_route_corner_tangents_join_hybrid_segments_continuously():
    poses = odom_route_from_enu_points(
        0.0,
        0.0,
        0.0,
        0.0,
        [(10.0, 0.0), (10.0, 10.0), (20.0, 10.0)],
        use_corner_tangents=True,
    )

    assert poses[0].yaw_rad == pytest.approx(math.pi / 4.0)
    assert poses[1].yaw_rad == pytest.approx(math.pi / 4.0)
    assert poses[2].yaw_rad == pytest.approx(0.0)


def test_open_route_corner_tangent_keeps_straight_preview_straight():
    poses = odom_route_from_enu_points(
        2.0,
        -1.0,
        0.0,
        0.0,
        [(5.0, 0.0), (15.0, 0.0)],
        use_corner_tangents=True,
    )

    assert poses[0].yaw_rad == pytest.approx(0.0)
    assert poses[1].yaw_rad == pytest.approx(0.0)


def test_final_route_abort_is_promoted_only_near_fresh_final_goal():
    assert should_promote_near_goal_route_abort(
        6, 1, (1.2, 0.2), 0.1, 1.5, 1.0
    )
    assert not should_promote_near_goal_route_abort(
        6, 2, (1.2, 0.2), 0.1, 1.5, 1.0
    )
    assert not should_promote_near_goal_route_abort(
        6, 1, (2.0, 0.0), 0.1, 1.5, 1.0
    )
    assert not should_promote_near_goal_route_abort(
        6, 1, (1.0, 0.0), 1.1, 1.5, 1.0
    )
    assert not should_promote_near_goal_route_abort(
        4, 1, (1.0, 0.0), 0.1, 1.5, 1.0
    )


def test_closed_route_keeps_a_finite_final_heading():
    poses = odom_route_from_enu_points(
        0.0, 0.0, 0.0, 0.0, [(5.0, 0.0), (5.0, 5.0), (5.0, 0.0)]
    )

    assert poses[-1].yaw_rad == pytest.approx(-math.pi / 2.0)


def test_closed_route_tangents_round_each_corner_and_repeat_first_yaw():
    poses = odom_route_from_enu_points(
        0.0,
        0.0,
        0.0,
        0.0,
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)],
        use_closed_route_tangents=True,
    )

    assert poses[0].yaw_rad == pytest.approx(-3.0 * math.pi / 8.0)
    assert poses[1].yaw_rad == pytest.approx(math.pi / 4.0)
    assert poses[2].yaw_rad == pytest.approx(7.0 * math.pi / 8.0)
    assert poses[-1].yaw_rad == pytest.approx(poses[0].yaw_rad)


def test_rolling_horizon_advances_with_one_pose_overlap():
    route = ['wp2', 'wp3', 'wp4', 'wp1-finish']

    assert rolling_horizon_window(route, 0) == ['wp2', 'wp3']
    assert rolling_horizon_global_remaining(4, 0, 2, 2) == 4
    assert rolling_horizon_global_remaining(4, 0, 2, 1) == 3
    assert rolling_horizon_next_start(4, 0, 2, 1) == 1
    assert rolling_horizon_window(route, 1) == ['wp3', 'wp4']
    assert rolling_horizon_global_remaining(4, 1, 2, 1) == 2
    assert rolling_horizon_next_start(4, 2, 2, 1) is None


def test_distance_bounded_window_defers_preview_outside_local_map():
    points = [(30.0, 0.0), (53.0, 0.0), (20.0, 0.0)]

    assert distance_bounded_rolling_horizon_count(
        points, 0, (0.0, 0.0), 2, 45.0
    ) == 1
    # Once the vehicle approaches, the same ordered preview enters range.
    assert distance_bounded_rolling_horizon_count(
        points, 0, (10.0, 0.0), 2, 45.0
    ) == 2


def test_distance_bounded_window_never_skips_an_out_of_range_target():
    points = [(50.0, 0.0), (10.0, 0.0)]

    assert distance_bounded_rolling_horizon_count(
        points, 0, (0.0, 0.0), 2, 45.0
    ) == 0


def test_single_pose_window_advances_after_result_without_resending_it():
    assert rolling_horizon_next_start(4, 1, 1, 1) is None
    assert rolling_horizon_result_next_start(4, 1, 1) == 2
    assert rolling_horizon_result_next_start(5, 1, 2) == 3
    assert rolling_horizon_result_next_start(4, 2, 2) is None


def test_virtual_preview_is_not_counted_as_mission_route_progress():
    assert rolling_real_poses_remaining(3, 2, 3) == 2
    assert rolling_real_poses_remaining(3, 2, 2) == 1
    assert rolling_real_poses_remaining(3, 2, 1) == 0
    assert rolling_real_poses_remaining(2, 2, 1) == 1


def test_closed_lap_preserves_open_prefix_and_finishes_at_wp1():
    route = ['wp1', 'wp2', 'wp3', 'wp4', 'wp1-finish']

    first_window = rolling_horizon_window(route, 0)
    assert first_window == ['wp1', 'wp2']
    assert rolling_horizon_global_remaining(5, 0, 2, 2) == 5
    assert rolling_horizon_next_start(5, 0, 2, 1) == 1


def test_initial_wp_is_skipped_only_when_vehicle_is_already_there():
    route = [(0.0, 0.0), (10.0, 0.0), (0.0, 0.0)]

    assert rolling_initial_start_index(route, (0.6, 0.0), 1.0) == 1
    assert rolling_initial_start_index(route, (1.1, 0.0), 1.0) == 0


def test_open_rolling_route_starts_at_wp1_and_stops_at_final_wp():
    route = ['wp1', 'wp2']

    assert rolling_horizon_window(route, 0) == ['wp1', 'wp2']
    assert rolling_horizon_global_remaining(2, 0, 2, 2) == 2
    # The first two-pose action already contains the final waypoint, so its
    # successful result is the completion authority; no duplicate final
    # window is required.
    assert rolling_horizon_next_start(2, 0, 2, 1) is None
    assert rolling_horizon_result_next_start(2, 0, 2) is None


def test_nearby_preview_window_focuses_only_once():
    assert should_focus_rolling_waypoint(6.0, 2, False, 6.0)
    assert not should_focus_rolling_waypoint(6.01, 2, False, 6.0)
    assert not should_focus_rolling_waypoint(5.0, 1, False, 6.0)
    assert not should_focus_rolling_waypoint(5.0, 2, True, 6.0)


@pytest.mark.parametrize(
    'values',
    [
        (-0.1, 2, False, 6.0),
        (1.0, 0, False, 6.0),
        (1.0, 2, False, 0.0),
        (float('nan'), 2, False, 6.0),
    ],
)
def test_rolling_focus_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        should_focus_rolling_waypoint(*values)


def test_rolling_waypoint_turn_angle_distinguishes_straight_and_corner():
    assert rolling_waypoint_turn_angle_deg(
        (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)
    ) == pytest.approx(0.0)
    assert rolling_waypoint_turn_angle_deg(
        (0.0, 0.0), (10.0, 0.0), (10.0, 10.0)
    ) == pytest.approx(90.0)


def test_rolling_capture_radius_widens_only_for_real_corners():
    assert rolling_waypoint_capture_radius(
        0.0, 1.0, 1.35, 15.0
    ) == pytest.approx(1.0)
    assert rolling_waypoint_capture_radius(
        15.0, 1.0, 1.35, 15.0
    ) == pytest.approx(1.0)
    assert rolling_waypoint_capture_radius(
        90.0, 1.0, 1.35, 15.0
    ) == pytest.approx(1.35)


def test_virtual_preview_preserves_outgoing_direction_inside_horizon():
    preview = bounded_virtual_preview(
        (0.0, 0.0),
        (35.0, 0.0),
        (35.0, 30.0),
        desired_extension_m=10.0,
        maximum_distance_m=45.0,
        boundary_margin_m=2.0,
        minimum_extension_m=2.0,
    )
    assert preview == pytest.approx((35.0, 10.0))
    assert math.hypot(*preview) < 43.0


def test_virtual_preview_is_shortened_at_costmap_boundary():
    preview = bounded_virtual_preview(
        (0.0, 0.0),
        (40.0, 0.0),
        (60.0, 0.0),
        desired_extension_m=10.0,
        maximum_distance_m=45.0,
        boundary_margin_m=2.0,
        minimum_extension_m=2.0,
    )
    assert preview == pytest.approx((43.0, 0.0))


def test_virtual_preview_is_omitted_without_minimum_horizon_room():
    assert bounded_virtual_preview(
        (0.0, 0.0),
        (42.0, 0.0),
        (60.0, 0.0),
        desired_extension_m=10.0,
        maximum_distance_m=45.0,
        boundary_margin_m=2.0,
        minimum_extension_m=2.0,
    ) is None


def test_aborted_rolling_preview_retries_current_waypoint_only_once():
    assert should_retry_rolling_preview_as_current_only(6, 2, 3, False)
    assert not should_retry_rolling_preview_as_current_only(6, 2, 3, True)
    assert not should_retry_rolling_preview_as_current_only(6, 1, 3, False)
    assert not should_retry_rolling_preview_as_current_only(5, 2, 3, False)
    assert not should_retry_rolling_preview_as_current_only(
        6, 2, None, False
    )


def test_aborted_current_rolling_segment_preserves_route_for_retry():
    assert should_retry_aborted_rolling_segment(
        6, True, 'rolling_segment', 2, True, False
    )
    assert not should_retry_aborted_rolling_segment(
        5, True, 'rolling_segment', 2, True, False
    )
    assert not should_retry_aborted_rolling_segment(
        6, False, 'rolling_segment', 2, True, False
    )
    assert not should_retry_aborted_rolling_segment(
        6, True, 'rolling_segment', 2, True, True
    )


def test_virtual_preview_hands_off_after_its_only_real_waypoint():
    assert should_handoff_virtual_preview_after_waypoint(
        route_pose_count=2,
        route_real_pose_count=1,
        route_window_start=2,
        passed_waypoint_index=2,
        route_total_count=5,
    )


def test_real_only_window_and_noncurrent_passage_do_not_force_handoff():
    assert not should_handoff_virtual_preview_after_waypoint(
        2, 2, 2, 2, 5
    )
    assert not should_handoff_virtual_preview_after_waypoint(
        2, 1, 2, 3, 5
    )
    assert not should_handoff_virtual_preview_after_waypoint(
        2, 1, 4, 4, 5
    )


def test_rolling_route_geometry_preserves_saved_waypoint_order():
    poses = odom_route_from_enu_points(
        0.0, 0.0, 0.0, 0.0, [(5.0, 0.0), (5.0, 5.0)]
    )

    assert [pose.yaw_rad for pose in poses] == pytest.approx([
        0.0,
        math.pi / 2.0,
    ])


@pytest.mark.parametrize(
    'values',
    [
        (float('nan'), 0.0, 1.0, 2.0),
        (0.0, 0.0, float('inf'), 2.0),
    ],
)
def test_goal_vector_rejects_nonfinite_values(values):
    with pytest.raises(ValueError):
        odom_target_from_goal_vector(*values)
