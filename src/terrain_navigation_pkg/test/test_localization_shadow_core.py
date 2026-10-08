import math

import pytest

from terrain_navigation_pkg.localization_shadow_core import align_planar_pose
from terrain_navigation_pkg.localization_shadow_core import align_enu_pose
from terrain_navigation_pkg.localization_shadow_core import (
    assess_localization_transition,
)
from terrain_navigation_pkg.localization_shadow_core import (
    base_position_from_antenna_delta,
)
from terrain_navigation_pkg.localization_shadow_core import circular_mean_angle
from terrain_navigation_pkg.localization_shadow_core import (
    planar_error_components,
)
from terrain_navigation_pkg.localization_shadow_core import (
    relative_translation_error,
)
from terrain_navigation_pkg.localization_shadow_core import summarize_errors
from terrain_navigation_pkg.localization_shadow_core import (
    summarize_signed_errors,
)
from terrain_navigation_pkg.localization_shadow_core import wrap_angle


def test_align_planar_pose_removes_initial_translation_and_yaw():
    aligned = align_planar_pose(
        estimate=(12.0, 5.0, math.radians(100.0)),
        estimate_origin=(10.0, 5.0, math.radians(90.0)),
        reference_origin=(3.0, 4.0, 0.0),
    )

    assert aligned[0] == pytest.approx(3.0)
    assert aligned[1] == pytest.approx(2.0)
    assert math.degrees(aligned[2]) == pytest.approx(10.0)


def test_align_enu_pose_does_not_rotate_absolute_position_axes():
    aligned = align_enu_pose(
        estimate=(12.0, 5.0, math.radians(100.0)),
        estimate_origin=(10.0, 5.0, math.radians(90.0)),
        reference_origin=(3.0, 4.0, 0.0),
    )

    assert aligned[0] == pytest.approx(5.0)
    assert aligned[1] == pytest.approx(4.0)
    assert math.degrees(aligned[2]) == pytest.approx(10.0)


def test_summarize_errors_reports_nearest_rank_p95_and_rmse():
    summary = summarize_errors([1.0, 2.0, 3.0, 4.0])

    assert summary['count'] == 4
    assert summary['latest'] == 4.0
    assert summary['rmse'] == pytest.approx(math.sqrt(7.5))
    assert summary['p95'] == 4.0
    assert summary['maximum'] == 4.0


def test_summarize_signed_errors_preserves_bias_direction():
    summary = summarize_signed_errors([-2.0, -1.0, 1.0, 4.0])

    assert summary['mean'] == pytest.approx(0.5)
    assert summary['median'] == pytest.approx(0.0)
    assert summary['p95_absolute'] == pytest.approx(4.0)
    assert summary['maximum_absolute'] == pytest.approx(4.0)


def test_transition_gate_separates_shadow_quality_from_control_plumbing():
    readiness = assess_localization_transition(
        sample_count=2000,
        estimate_age_s=0.05,
        gnss_age_s=0.10,
        position_p95_m=0.15,
        longitudinal_p95_m=0.12,
        lateral_p95_m=0.08,
        absolute_longitudinal_bias_m=0.03,
        relative_translation_p95_m=0.04,
        pose_jump_count=0,
        minimum_sample_count=1000,
        maximum_estimate_age_s=0.15,
        maximum_gnss_age_s=0.25,
        maximum_position_p95_m=0.30,
        maximum_longitudinal_p95_m=0.25,
        maximum_lateral_p95_m=0.20,
        maximum_absolute_longitudinal_bias_m=0.10,
        maximum_relative_translation_p95_m=0.08,
        maximum_pose_jump_count=0,
        active_navigation_tf_ready=False,
        independent_velocity_ready=False,
    )

    assert readiness['shadow_quality_ready'] is True
    assert readiness['control_trial_ready'] is False
    assert readiness['blockers'] == [
        'sensor_navigation_tf_not_ready',
        'independent_velocity_not_ready',
    ]


def test_transition_gate_rejects_bad_position_tail():
    readiness = assess_localization_transition(
        sample_count=2000,
        estimate_age_s=0.05,
        gnss_age_s=0.10,
        position_p95_m=0.35,
        longitudinal_p95_m=0.12,
        lateral_p95_m=0.08,
        absolute_longitudinal_bias_m=0.03,
        relative_translation_p95_m=0.04,
        pose_jump_count=0,
        minimum_sample_count=1000,
        maximum_estimate_age_s=0.15,
        maximum_gnss_age_s=0.25,
        maximum_position_p95_m=0.30,
        maximum_longitudinal_p95_m=0.25,
        maximum_lateral_p95_m=0.20,
        maximum_absolute_longitudinal_bias_m=0.10,
        maximum_relative_translation_p95_m=0.08,
        maximum_pose_jump_count=0,
        active_navigation_tf_ready=True,
        independent_velocity_ready=True,
    )

    assert readiness['shadow_quality_ready'] is False
    assert readiness['control_trial_ready'] is False
    assert readiness['blockers'] == ['position_p95_exceeds_limit']


def test_wrap_angle_uses_closed_open_interval():
    assert wrap_angle(math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)


def test_circular_mean_crosses_negative_pi_seam():
    mean = circular_mean_angle([
        math.radians(179.0),
        math.radians(-179.0),
    ])

    assert abs(math.degrees(mean)) == pytest.approx(180.0)


def test_antenna_lever_arm_does_not_report_rotation_as_translation():
    # A base at (5, 0) rotates from 0 to +90 degrees.  A one-metre forward
    # antenna therefore moves from (1, 0) to (5, 1), an ENU delta of (4, 1).
    position = base_position_from_antenna_delta(
        antenna_east_m=4.0,
        antenna_north_m=1.0,
        current_yaw_rad=math.pi / 2.0,
        reference_yaw_rad=0.0,
        antenna_forward_m=1.0,
        antenna_left_m=0.0,
    )

    assert position == pytest.approx((5.0, 0.0))


def test_planar_error_components_follow_vehicle_axes():
    longitudinal, lateral = planar_error_components(
        estimate_xy=(8.0, 7.0),
        reference_pose=(5.0, 5.0, math.pi / 2.0),
    )

    assert longitudinal == pytest.approx(2.0)
    assert lateral == pytest.approx(-3.0)


def test_relative_translation_error_removes_reference_motion():
    error = relative_translation_error(
        current_pair=(2.2, 1.8, 12.0, 7.0),
        previous_pair=(1.0, 1.0, 11.0, 6.0),
    )

    assert error == pytest.approx(math.hypot(0.2, -0.2))
