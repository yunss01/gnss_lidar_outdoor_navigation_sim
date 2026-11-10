"""Check busy callbacks, stale data and wrong-session shutdown responses."""

from types import SimpleNamespace

import pytest

from terrain_navigation_pkg.simulation_experiment import (
    _halt_client,
    _parser,
    _verify_command_mode,
    _verify_navigation_variant,
    _verify_path_trace_stack,
    report,
)
from terrain_navigation_pkg.simulation_experiment_core import (
    navigation_variant_parameters, )
from terrain_navigation_pkg.simulation_experiment_scenario import write_json


def truth(speed):
    return SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(
        linear=SimpleNamespace(x=speed, y=0, z=0))))


@pytest.mark.parametrize('case', [
    'delayed_ack',
    'no_ack',
    'stale_ack',
    'wrong_session',
    'wrong_run',
    'cached_truth',
    'moving_truth',
    'nan_speed',
    'pre_ack_replay',
])
def test_halt_requires_fresh_matching_ack_and_new_stable_truth(case):
    now = [10.0]
    calls, sends = [], []
    node = SimpleNamespace(bridge_sequence=1,
                           bridge_wall=9.9,
                           truth_wall=9.9,
                           bridge=dict(adapter_session='session',
                                       run_id='run',
                                       armed=True,
                                       sim_time_s=0.0),
                           truth={0: truth(0)},
                           publish_event=lambda *args: sends.append(
                               (now[0], args)))

    def spin(_node, timeout_sec):
        assert timeout_sec <= .05
        calls.append(now[0])
        # Always-ready callbacks return immediately. Thirty iterations are
        # only 30ms here, NOT a guaranteed 1.5-second wait.
        now[0] += .001
        elapsed = now[0] - 10
        stamp = int(elapsed / .05) * 50_000_000
        if elapsed >= .6 and case != 'stale_ack':
            node.bridge_sequence += 1
            node.bridge_wall = now[0]
            node.bridge.update(
                armed=case == 'no_ack',
                sim_time_s=100 if case == 'pre_ack_replay' else stamp / 1e9,
                adapter_session=('other'
                                 if case == 'wrong_session' else 'session'),
                run_id='other' if case == 'wrong_run' else 'run')
        if case == 'stale_ack':
            node.bridge['armed'] = False
        if case != 'cached_truth' and stamp not in node.truth:
            speed = (1.0 if case == 'moving_truth' else
                     float('nan') if case == 'nan_speed' else 0.0)
            node.truth[stamp] = truth(speed)
            node.truth_wall = now[0]

    result = _halt_client(node,
                          'run',
                          'baseline',
                          SimpleNamespace(spin_once=spin),
                          timeout_s=1.0,
                          clock=lambda: now[0])
    assert len(calls) > 30
    assert sends
    assert all(args == ('stop', 'run', 'baseline') for _, args in sends)
    assert result['halt_confirmation_complete'] == (case == 'delayed_ack')
    if case == 'delayed_ack':
        assert result['halt_confirmation_wall_s'] >= .7
    else:
        assert result['halt_confirmation_wall_s'] >= 1.0


@pytest.mark.parametrize('variant', ['baseline', 'rejoin'])
def test_cli_and_expected_variant_parameters(variant):
    options = _parser().parse_args([
        'run', '--anchor', 'anchor.json', '--scenario', 'S1',
        '--navigation-variant', variant, '--arrival-radius-m', '1.0'
    ])
    assert options.navigation_variant == variant
    assert navigation_variant_parameters(variant) == {
        'terminal_approach_distance_m': 8.0 if variant == 'rejoin' else 0.0,
        'terminal_approach_capture_m': 1.0
    }


@pytest.mark.parametrize('mode', ['legacy', 'latest'])
def test_command_mode_cli_and_preflight(mode):
    options = _parser().parse_args([
        'run', '--anchor', 'anchor.json', '--scenario', 'S1', '--command-mode',
        mode
    ])
    assert options.command_mode == mode
    bridge = dict(command_mode=mode, command_telemetry_version=1)
    _verify_command_mode(bridge, mode)
    for invalid in ({}, dict(bridge, command_mode='wrong'),
                    dict(bridge, command_telemetry_version=0)):
        with pytest.raises(RuntimeError, match='command mode/telemetry'):
            _verify_command_mode(invalid, mode)


@pytest.mark.parametrize('case', [
    'ok',
    'mismatch',
    'missing',
    'nan',
    'unavailable',
    'timeout',
])
def test_actual_far_parameters_must_match_requested_variant(case):
    from rcl_interfaces.msg import ParameterValue
    expected = navigation_variant_parameters('rejoin')
    values = [
        ParameterValue(type=3, double_value=value)
        for value in expected.values()
    ]
    if case == 'mismatch':
        values[0].double_value = 0.0
    if case == 'nan':
        values[0].double_value = float('nan')
    if case == 'missing':
        values = []
    requests, destroyed = [], []

    def call(request):
        requests.append(request.names)
        return SimpleNamespace(done=lambda: case != 'timeout',
                               result=lambda: SimpleNamespace(values=values))

    client = SimpleNamespace(
        wait_for_service=lambda **_kw: case != 'unavailable', call_async=call)
    node = SimpleNamespace(create_client=lambda *_args: client,
                           destroy_client=destroyed.append,
                           active_navigation_parameters=None)
    ros = SimpleNamespace(spin_until_future_complete=lambda *_a, **_kw: None)
    if case == 'ok':
        assert _verify_navigation_variant(node, 'rejoin', ros) == expected
    else:
        with pytest.raises(RuntimeError, match='preflight:'):
            _verify_navigation_variant(node, 'rejoin', ros)
    assert destroyed == [client]
    if case != 'unavailable':
        assert requests == [list(expected)]


def test_report_separates_navigation_treatments_with_same_source(
        tmp_path, capsys):
    for variant in ('baseline', 'rejoin'):
        directory = tmp_path / ('run_' + variant)
        directory.mkdir()
        write_json(
            directory / 'summary.json',
            dict(run_id=directory.name,
                 scenario='S1',
                 profile='baseline',
                 metrics_valid=True,
                 success=False))
        write_json(
            directory / 'metadata.json',
            dict(anchor_sha256='same',
                 navigation_code_sha256='same',
                 navigation_variant=variant,
                 navigation_variant_verified=True,
                 active_navigation_parameters=navigation_variant_parameters(
                     variant)))
    report(SimpleNamespace(output_root=str(tmp_path)))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert any('variant=baseline verified=True' in line for line in lines)
    assert any('variant=rejoin verified=True' in line for line in lines)


def test_report_separates_command_modes_with_same_navigation(tmp_path, capsys):
    for mode in ('legacy', 'latest'):
        directory = tmp_path / ('run_' + mode)
        directory.mkdir()
        write_json(
            directory / 'summary.json',
            dict(run_id=directory.name,
                 scenario='S1',
                 profile='baseline',
                 metrics_valid=True,
                 success=True,
                 aligned_stop_success=False))
        write_json(
            directory / 'metadata.json',
            dict(anchor_sha256='same',
                 navigation_code_sha256='same',
                 navigation_variant='rejoin',
                 command_mode=mode,
                 requested_command_mode=mode,
                 command_mode_verified=True))
    report(SimpleNamespace(output_root=str(tmp_path)))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert any('command_mode=legacy verified=True' in line for line in lines)
    assert any('command_mode=latest verified=True' in line for line in lines)


def test_report_groups_control_parameters_not_unique_log_paths(
        tmp_path, capsys):
    for name, penalty in (('a', 3.), ('b', 3.), ('c', 5.)):
        directory = tmp_path / ('run_' + name)
        directory.mkdir()
        write_json(directory / 'summary.json', dict(
            run_id=directory.name, scenario='S1', profile='baseline',
            metrics_valid=True, success=True,
            path_trace=dict(verdict='supported', critical_coverage=1.)))
        write_json(directory / 'metadata.json', dict(
            anchor_sha256='same', navigation_code_sha256='same',
            path_trace_enabled=True, path_trace_protocol_version=1,
            path_trace_configuration=dict(
                session_log_root='/logs/session_' + name,
                parameters={
                    'planner_server': {'GridBased.cost_penalty': penalty},
                    'path_clearance_validator_node': {
                        'log_directory': (
                            '/logs/session_' + name + '/validator'),
                        'hard_clearance_m': .35}})))
    report(SimpleNamespace(output_root=str(tmp_path)))
    lines = capsys.readouterr().out.splitlines()
    # Same control settings in two unique launches stay together; changing a
    # control parameter is a separate group even with identical source code.
    assert len(lines) == 3
    assert any('2/2 pass;' in line for line in lines)
    assert any('1/1 pass;' in line for line in lines)


@pytest.mark.parametrize('case', [
    'ok',
    'missing_publisher',
    'wrong_controller',
    'missing_parameter',
    'nan_parameter',
    'other_plugin',
    'old_log_directory',
    'mixed_sessions',
])
def test_trace_preflight_is_read_only_and_rejects_incompatible_stack(case):
    from rcl_interfaces.msg import ParameterValue
    values = {
        'FollowPath.plugin': ('nav2_regulated_pure_pursuit_controller::'
                              'RegulatedPurePursuitController'),
        'FollowPath.lookahead_dist':
        3.5,
        'FollowPath.desired_linear_vel':
        1.,
        'FollowPath.use_velocity_scaled_lookahead_dist':
        False,
        'GridBased.cost_penalty':
        3.,
        'GridBased.minimum_turning_radius':
        4.1,
        'vehicle_front_m':
        2.4,
        'vehicle_rear_m':
        2.5,
        'vehicle_half_width_m':
        1.,
        'hard_clearance_m':
        .35,
        'preferred_clearance_m':
        .75,
    }
    if case == 'nan_parameter':
        values['FollowPath.lookahead_dist'] = float('nan')
    if case == 'other_plugin':
        values['FollowPath.plugin'] = 'different_controller'
    queries, destroyed = [], []

    def client_factory(service, name):
        assert service.__name__ == 'GetParameters'
        queries.append(name)
        node_name = name.split('/')[1]

        def call(request):
            payload = []
            for field in request.names:
                value = values.get(field)
                if field == 'log_directory':
                    root = ('/tmp/node_logs'
                            if case == 'old_log_directory' else
                            '/tmp/session_other' if case == 'mixed_sessions'
                            and node_name == 'far_nav2_guide_node' else
                            '/tmp/session_trace_test')
                    value = root + '/' + node_name
                if isinstance(value, bool):
                    p = ParameterValue(type=1, bool_value=value)
                elif isinstance(value, float):
                    p = ParameterValue(type=3, double_value=value)
                else:
                    p = ParameterValue(type=4, string_value=value)
                payload.append(p)
            if case == 'missing_parameter':
                payload = []
            return SimpleNamespace(
                done=lambda: True,
                result=lambda: SimpleNamespace(values=payload))

        return SimpleNamespace(wait_for_service=lambda **_kw: True,
                               call_async=call)

    node = SimpleNamespace(count_publishers=lambda _topic: 0
                           if case == 'missing_publisher' else 1,
                           get_publishers_info_by_topic=lambda _topic: [
                               SimpleNamespace(node_name='wrong'
                                               if case == 'wrong_controller'
                                               else 'controller_server')
                           ],
                           create_client=client_factory,
                           destroy_client=destroyed.append)
    ros = SimpleNamespace(spin_until_future_complete=lambda *_a, **_kw: None)
    if case == 'ok':
        result = _verify_path_trace_stack(node, ros)
        assert result['parameters']['controller_server'][
            'FollowPath.lookahead_dist'] == 3.5
        assert result['session_log_root'] == '/tmp/session_trace_test'
        assert len(queries) == len(destroyed) == 5
    else:
        with pytest.raises(RuntimeError, match='preflight:'):
            _verify_path_trace_stack(node, ros)
