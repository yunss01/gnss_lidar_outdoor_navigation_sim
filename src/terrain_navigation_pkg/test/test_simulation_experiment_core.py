import math

import pytest

from terrain_navigation_pkg.simulation_experiment_core import (
    evaluate_trial,
    polygon_clearance,
    position_fault,
    PROFILES,
    quaternion_yaw,
    rectangle,
    rotate_quaternion_yaw,
    scenario_offsets,
    SteeringFault,
    validate_anchor,
)


def anchor():
    return {
        'schema_version': 1,
        'map_name': 'Town',
        'vehicle_blueprint': 'vehicle.lincoln.mkz_2020',
        'start_carla': dict(x=0, y=0, z=1, yaw=0, pitch=0, roll=0),
        'route_length_m': 40,
        'obstacle_distance_m': 20,
        'corridor_half_width_m': 7,
        'waypoints': [
            dict(latitude=0, longitude=0, altitude=0),
            dict(latitude=0, longitude=0.001, altitude=0)
        ],
    }


def row(t, along=0, cross=0, speed=1, **overrides):
    result = {
        'sim_time_s': t,
        'truth_x': along,
        'truth_y': cross,
        'along_track_m': along,
        'cross_track_m': cross,
        'heading_error_deg': 0,
        'goal_error_m': abs(40 - along),
        'speed_mps': speed,
        'collision_count': 0,
        'bbox_clearance_m': 1,
        'corridor_violation': 0,
        'safety_state': 'clear',
        'path_gate_state': 'path_valid',
        'tf_consistent': 1,
        'lidar_age_s': 0.05,
        'bridge_status_age_s': 0.05,
    }
    result.update(overrides)
    return result


def meta(scenario='S1', termination='goal_stopped'):
    return {
        'run_id': 'test',
        'scenario': scenario,
        'profile': 'baseline',
        'termination': termination,
        'obstacle_passage_s_m': 26
    }


def recovered_frames():
    return [
        row(i / 10, along=26 + i / 5, speed=0 if i >= 60 else 1)
        for i in range(71)
    ]


def test_anchor_validates_before_world_mutation():
    assert validate_anchor(anchor())['route_length_m'] == 40
    for key, value in [('route_length_m', 15), ('obstacle_distance_m', 39),
                       ('corridor_half_width_m', 2)]:
        invalid = dict(anchor(), **{key: value})
        with pytest.raises(ValueError):
            validate_anchor(invalid)


def test_nonfinite_anchor_is_rejected():
    invalid = anchor()
    invalid['start_carla']['yaw'] = math.nan
    with pytest.raises(ValueError):
        validate_anchor(invalid)


def test_pose_bias_is_off_at_anchor_then_ramps_in():
    profile = PROFILES['position_left']
    assert position_fault(profile, 3, 0) == (0, 0, 0)
    assert position_fault(profile, 3.5, 0)[1] == pytest.approx(0.15)
    assert position_fault(profile, 4, 0)[1] == pytest.approx(0.3)
    assert position_fault(profile, 10, math.pi / 2)[0] == pytest.approx(-0.3)


def test_yaw_bias_rotates_quaternion_without_moving_position():
    delta = position_fault(PROFILES['yaw_right'], 4, 0)[2]
    quaternion = rotate_quaternion_yaw((0, 0, 0, 1), delta)
    assert quaternion_yaw(quaternion) == pytest.approx(math.radians(-3))
    assert sum(value * value for value in quaternion) == pytest.approx(1)


def test_baseline_steering_is_identity():
    fault = SteeringFault(PROFILES['baseline'])
    for time, value in enumerate([-1, -0.17, 0, 0.24, 1]):
        assert fault.apply(value, time, time) == value


def test_fifteen_steering_steps_are_symmetric():
    fault = SteeringFault(PROFILES['steering_steps'])
    outputs = {fault.apply(i / 70, i + 100, 10) for i in range(-70, 71)}
    assert len(outputs) == 15
    assert outputs == {-value for value in outputs}
    assert fault.apply(0.5 / 7, 1000, 10) == pytest.approx(1 / 7)


def test_delay_uses_simulation_seconds_not_sample_count():
    fault = SteeringFault(PROFILES['steering_delay'])
    assert fault.apply(0.7, 10, 5) == 0
    assert fault.apply(0.1, 10.1, 5.1) == 0
    assert fault.apply(-0.7, 10.21, 5.21) == 0.7


def test_stop_discards_queued_steering_and_never_replays_it():
    fault = SteeringFault(PROFILES['steering_both'])
    fault.apply(1, 10, 5)
    assert fault.apply(1, 10.3, 5.3) == 1
    assert fault.apply(1, 10.31, 5.31, stopping=True) == 0
    assert not fault.history
    assert fault.apply(-1, 10.4, 5.4) == 0


def test_clock_rewind_clears_history():
    fault = SteeringFault(PROFILES['steering_delay'])
    fault.apply(1, 10, 5)
    assert fault.apply(1, 1, 0) == 0


def test_scene_sides_and_blocked_corridor():
    assert scenario_offsets('S0', 20, 2) == []
    assert scenario_offsets('S1', 20, 2) == [(20, 0)]
    assert all(right <= 0 for _, right in scenario_offsets('S2', 20, 2))
    assert min(right for _, right in scenario_offsets('S2', 20, 2)) < -6
    blocked = scenario_offsets('S3', 20, 2)
    assert min(right for _, right in blocked) < -6
    assert max(right for _, right in blocked) > 6
    # All gaps are < vehicle width: a Lincoln cannot pass between them.
    positions = sorted(right for _, right in blocked)
    assert max(b - a - 2 for a, b in zip(positions, positions[1:])) < 2


def test_clearance_is_body_box_not_center_or_planned_path():
    first = rectangle(0, 0, 0, 2, 1)
    assert polygon_clearance(first, rectangle(5, 0, 0, 2,
                                              1)) == pytest.approx(1)
    assert polygon_clearance(first, rectangle(3, 0, 0, 2, 1)) == 0
    assert polygon_clearance(first, rectangle(0, 5, math.pi / 2, 2,
                                              1)) == pytest.approx(2)


def test_success_requires_truth_goal_and_sustained_recovery():
    summary = evaluate_trial(meta(), recovered_frames())
    assert summary['success'] and summary['recovered']
    assert summary['recovery_time_s'] == 0
    frames = [row(i / 10, along=39, cross=1) for i in range(31)]
    summary = evaluate_trial(meta(), frames)
    assert summary['goal_arrived'] and not summary['recovered']
    assert not summary['success']


def test_route_completed_label_cannot_mask_failed_recovery():
    frames = [
        row(i / 10, along=39, cross=2, route_status='completed')
        for i in range(31)
    ]
    assert not evaluate_trial(meta(), frames)['success']


def test_recovery_window_resets_on_heading_departure():
    frames = [
        row(i / 10, along=27, heading_error_deg=8 if i == 15 else 0)
        for i in range(31)
    ]
    assert not evaluate_trial(meta(), frames)['recovered']


@pytest.mark.parametrize('override', [
    {
        'collision_count': 1
    },
    {
        'bbox_clearance_m': 0
    },
    {
        'corridor_violation': 1
    },
    {
        'tf_consistent': 0
    },
    {
        'lidar_age_s': 2
    },
    {
        'bridge_status_age_s': 1
    },
])
def test_goal_reached_does_not_mask_safety_or_recording_failure(override):
    frames = recovered_frames()
    frames[-1].update(override)
    assert not evaluate_trial(meta(), frames)['success']


def test_sparse_records_and_duplicate_time_cannot_pass():
    frames = recovered_frames()
    frames[3]['sim_time_s'] = frames[2]['sim_time_s']
    assert not evaluate_trial(meta(), frames)['metrics_valid']
    assert not evaluate_trial(meta(), [])['success']


def test_expected_blocked_stop_is_not_goal_success():
    frames = [
        row(i / 10, along=12, speed=0, safety_state='obstacle_stop')
        for i in range(121)
    ]
    summary = evaluate_trial(meta('S3', 'blocked_stopped'), frames)
    assert summary['success'] and summary['expected_stop']
    assert not summary['goal_arrived']


@pytest.mark.parametrize('change', [
    {
        'along_track_m': 0
    },
    {
        'safety_state': 'command_stale'
    },
    {
        'safety_state': 'sensor_stale'
    },
    {
        'safety_state': 'clear',
        'path_gate_state': 'path_valid'
    },
])
def test_inert_or_broken_system_is_not_a_successful_blocked_stop(change):
    frames = [
        row(i / 10, along=12, speed=0, safety_state='obstacle_stop', **change)
        if 'safety_state' not in change else row(
            i / 10, along=12, speed=0, **change) for i in range(121)
    ]
    assert not evaluate_trial(meta('S3', 'blocked_stopped'), frames)['success']


def test_interrupted_attempt_is_retained_as_failure():
    assert not evaluate_trial(meta(termination='interrupted'),
                              recovered_frames())['success']


def test_missing_fault_effect_cannot_be_reported_as_valid_fault_trial():
    metadata = meta()
    metadata['profile'] = 'position_left'
    summary = evaluate_trial(metadata, recovered_frames())
    assert not summary['pose_fault_observed']
    assert not summary['success']


def test_nonfinite_recording_is_retained_without_nan_json():
    frames = recovered_frames()
    frames[0]['truth_x'] = math.nan
    summary = evaluate_trial(meta(), frames)
    assert summary['verdict'] == 'invalid_recording'
    assert not summary['success']


@pytest.mark.parametrize('age_field', [
    'command_age_s',
    'safety_state_age_s',
])
def test_watchdog_stop_or_stale_safety_cannot_mask_a_dead_controller(
        age_field):
    frames = [
        row(i / 10, along=12, speed=0, safety_state='obstacle_stop')
        for i in range(121)
    ]
    for frame in frames:
        frame[age_field] = 2.0
    assert not evaluate_trial(meta('S3', 'blocked_stopped'), frames)['success']


def test_unchanged_gate_event_is_not_a_stale_periodic_sensor():
    frames = recovered_frames()
    for frame in frames:
        frame['path_gate_age_s'] = 60.0
    summary = evaluate_trial(meta(), frames)
    assert summary['success'] and summary['metrics_valid']
    assert summary['recording_issues'] == {}


def test_s0_initial_wait_can_precede_the_first_gate_event():
    frames = [
        row(0, speed=0, path_gate_state='', path_gate_age_s=23000),
        row(0.1, speed=0, path_gate_state='validating_path'),
        row(0.2, along=39, speed=0, path_gate_age_s=60),
    ]
    summary = evaluate_trial(meta('S0'), frames)
    assert summary['success'] and not summary['recovered']


def test_missing_gate_state_during_motion_is_invalid():
    frames = recovered_frames()
    frames[1]['path_gate_state'] = ''
    summary = evaluate_trial(meta(), frames)
    assert not summary['metrics_valid'] and not summary['success']
    assert summary['recording_issues'] == {
        'path_gate_state_missing_while_moving': 1
    }


def test_missing_final_gate_state_is_invalid_even_when_parked():
    frames = recovered_frames()
    frames[-1]['path_gate_state'] = ''
    summary = evaluate_trial(meta(), frames)
    assert not summary['success']
    assert summary['recording_issues']['final_path_gate_state_missing'] == 1


def test_blocked_stop_accepts_unchanged_gate_event_with_live_safety():
    frames = [
        row(i / 10, along=12, speed=0, safety_state='obstacle_stop',
            path_gate_age_s=60) for i in range(121)
    ]
    summary = evaluate_trial(meta('S3', 'blocked_stopped'), frames)
    assert summary['success'] and summary['expected_stop']
