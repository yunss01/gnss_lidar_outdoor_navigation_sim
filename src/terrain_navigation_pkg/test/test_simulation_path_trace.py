"""Passive tracing: geometry, stale/future paths and unsupported evidence."""

import csv
import copy
import hashlib
import json
import math
import os
import threading
import time
from types import SimpleNamespace

import pytest

from terrain_navigation_pkg.simulation_path_trace import (
    analyze_tracking,
    body_polygon,
    new_log_session,
    PathTraceObserver,
    project_controller_window,
    reanalyze_trial,
    _transform_pose,
)
from terrain_navigation_pkg.simulation_experiment import _parser


def anchor():
    return dict(ego_box=dict(half_length_m=1.,
                             half_width_m=1.,
                             offset_forward_m=0.,
                             offset_left_m=0.,
                             yaw_offset_rad=0.))


def state():
    from terrain_navigation_pkg.simulation_experiment_core import rectangle
    return dict(
        obstacles=[dict(actor_id=42, polygon_ros=rectangle(0, -3, 0, .5, .5))])


def frame(t=10.1, **updates):
    result = dict(sim_time_s=t,
                  elapsed_s=t - 10,
                  truth_x=0.,
                  truth_y=0.,
                  truth_yaw_rad=0.,
                  speed_mps=1.,
                  tf_consistent=1,
                  pose_bias_m=0.,
                  yaw_bias_deg=0.,
                  steer_applied=.2,
                  accepted_command_speed_mps=1.,
                  accepted_command_yaw_rate_rps=.1,
                  brake=0.)
    result.update(updates)
    return result


def window(**updates):
    result = dict(path_id='controller_1',
                  received_sim_time_s=10.,
                  stamp_s=10.,
                  poses=[[-1, 1, 0], [2, 1, 0], [5, 1, 0]],
                  candidate_at_receipt='candidate_1',
                  controller_goal_id='goal')
    result.update(updates)
    return result


def status(t=9.9, goal='goal', value=2):
    return dict(received_sim_time_s=t,
                goals=[dict(goal_id=goal, status=value)])


def test_same_station_clearance_not_straight_route_error(tmp_path):
    result = analyze_tracking(anchor(), state(), [frame()], [window()],
                              [status()], tmp_path)
    assert result['verdict'] == 'supported'
    assert result['critical_coverage'] == 1
    row = result['at_actual_minimum']
    assert row['actual_bbox_clearance_m'] == pytest.approx(1.5)
    assert row['same_station_planned_clearance_m'] == pytest.approx(2.5)
    assert row['same_station_clearance_loss_m'] == pytest.approx(1.)
    assert row['tracking_cross_m'] == pytest.approx(-1.)
    assert row['tracking_heading_error_deg'] == 0.
    assert row['accepted_command_curvature_per_m'] == .1
    assert list(csv.DictReader((tmp_path / 'path_tracking.csv').open()))


@pytest.mark.parametrize('case,reason', [
    ('absent', 'no_controller_window'),
    ('future_arrival', 'no_controller_window'),
    ('future_stamp', 'stale_or_future_controller_window'),
    ('stale', 'stale_or_future_controller_window'),
    ('cancel', 'controller_goal_not_confirmed_active'),
    ('replacement', 'controller_goal_not_confirmed_active'),
    ('unknown_action', 'controller_goal_not_confirmed_active'),
    ('ambiguous_action', 'controller_goal_not_confirmed_active'),
    ('tf_mismatch', 'navigation_tf_mismatch'),
    ('pose_bias', 'biased_navigation_frame'),
    ('invalid_new_window', 'invalid_controller_window'),
    ('missing_transform', 'transform_unavailable'),
])
def test_missing_evidence_never_becomes_zero_tracking_error(
        tmp_path, case, reason):
    windows, events, f = [window()], [status()], frame()
    if case == 'absent':
        windows = []
    if case == 'future_arrival':
        windows = [window(received_sim_time_s=10.2)]
    if case == 'future_stamp':
        windows = [window(stamp_s=10.2)]
    if case == 'stale':
        windows = [window(stamp_s=9.)]
    if case == 'cancel':
        events.append(status(t=10.05, value=3))
    if case == 'replacement':
        events.append(status(t=10.05, goal='other'))
    if case == 'unknown_action':
        events = []
    if case == 'ambiguous_action':
        events[0]['goals'].append(dict(goal_id='other', status=2))
    if case == 'tf_mismatch':
        f['tf_consistent'] = 0
    if case == 'pose_bias':
        f['pose_bias_m'] = .3
    if case in ('invalid_new_window', 'missing_transform'):
        windows.append(
            window(received_sim_time_s=10.05,
                   poses=[],
                   observation_error=reason))
    result = analyze_tracking(anchor(), state(), [f], windows, events,
                              tmp_path)
    assert result['verdict'] == 'unsupported'
    assert result['maximum_abs_tracking_cross_m'] is None
    assert result['at_actual_minimum']['observation'] == reason
    assert result['at_actual_minimum']['tracking_cross_m'] == ''


def test_one_missing_critical_sample_prevents_full_support(tmp_path):
    result = analyze_tracking(anchor(), state(),
                              [frame(10.1), frame(10.4)], [window()],
                              [status()], tmp_path)
    assert result['verdict'] == 'partial'
    assert result['critical_coverage'] == .5


@pytest.mark.parametrize('value,context', [
    (3, 'canceling'), (4, 'succeeded'), (5, 'canceled'), (6, 'aborted'),
])
@pytest.mark.parametrize('speed,motion', [
    (1., 'moving'), (.1, 'low_speed'), (0., 'stationary'),
])
def test_terminal_goal_motion_is_descriptive_not_tracking_support(
        tmp_path, value, context, speed, motion):
    result = analyze_tracking(
        anchor(), state(), [frame(speed_mps=speed, brake=.2)], [window()],
        [status(), status(10.05, value=value)], tmp_path)
    row = result['at_actual_minimum']
    assert row['goal_context'] == context + '_goal_' + motion
    assert row['controller_goal_status'] == value
    assert row['active_goal_count'] == 0
    assert row['observation'] == 'controller_goal_not_confirmed_active'
    assert row['same_station_planned_clearance_m'] == ''
    assert result['verdict'] == 'unsupported'
    if motion == 'moving':
        assert result['critical_samples'] == 1
        assert result['critical_coverage'] == 0
        assert result['unsupported_critical_goal_contexts'] == {
            context + '_goal_' + motion: 1}


@pytest.mark.parametrize('case,context', [
    ('overlap', 'overlapping_executing_goals'),
    ('new_goal', 'active_goal_changed'),
    ('no_link', 'active_goal_without_linked_window'),
    ('future_status', 'no_status_observed'),
])
def test_transition_context_does_not_backfill_from_future_status(
        tmp_path, case, context):
    events, w = [status()], window()
    if case == 'overlap':
        events[0]['goals'].append(dict(goal_id='new', status=2))
    elif case == 'new_goal':
        events = [status(goal='new')]
    elif case == 'no_link':
        w['controller_goal_id'] = None
    elif case == 'future_status':
        events = [status(10.2)]
    result = analyze_tracking(anchor(), state(), [frame()], [w], events,
                              tmp_path)
    assert result['at_actual_minimum']['goal_context'] == context
    assert result['critical_samples'] == 1
    assert result['critical_coverage'] == 0
    assert result['same_station_critical_coverage'] == 0


def test_speed_change_and_braking_do_not_reactivate_finished_goal(tmp_path):
    result = analyze_tracking(
        anchor(), state(),
        [frame(10.05), frame(10.1, speed_mps=.8, brake=.2,
                             accepted_command_speed_mps=0.)],
        [window()], [status(), status(10.08, value=4)], tmp_path)
    rows = list(csv.DictReader((tmp_path / 'path_tracking.csv').open()))
    assert float(rows[1]['speed_derivative_mps2']) == pytest.approx(-4.)
    assert rows[1]['goal_context'] == 'succeeded_goal_moving'
    assert rows[1]['accepted_command_speed_mps'] == '0.0'
    assert result['critical_coverage'] == .5
    assert result['same_station_critical_coverage'] == .5


@pytest.mark.parametrize('poses,reason', [
    ([(.15, 1, 0), (2, 1, 0)], 'projection_clamped_window_start'),
    ([(-2, 1, 0), (-.15, 1, 0)], 'projection_clamped_window_end'),
    ([(-4, 1, 0), (2, 1, 0)], 'projection_clamped_prefix_end'),
])
def test_endpoint_gaps_are_not_same_station_tracking_loss(
        tmp_path, poses, reason):
    result = analyze_tracking(anchor(), state(), [frame()],
                              [window(poses=poses)], [status()], tmp_path)
    row = result['at_actual_minimum']
    assert row['observation'] == 'supported'
    assert row['same_station_observation'] == reason
    assert abs(row['projection_longitudinal_offset_m']) > .1
    assert row['projected_planned_clearance_m'] != ''
    assert row['projected_clearance_difference_m'] != ''
    assert row['same_station_planned_clearance_m'] == ''
    assert row['same_station_clearance_loss_m'] == ''
    assert result['association_verdict'] == 'supported'
    assert result['same_station_verdict'] == 'unsupported'
    assert result['verdict'] == 'partial'
    assert result['critical_coverage'] == 1
    assert result['same_station_critical_coverage'] == 0


def test_exact_perpendicular_at_start_remains_a_same_station_match():
    projection, _ = project_controller_window([(0, 1, 0), (2, 1, 0)], 0, 0)
    assert projection['region'] == 'window_start'
    assert projection['longitudinal'] == 0
    assert projection['same_station_observation'] == 'supported'


def test_vertex_clamp_is_not_an_interior_projection():
    projection, _ = project_controller_window(
        [(0, 0, 0), (1, 0, 0), (1, 1, math.pi / 2)], 1.1, -.1)
    assert projection['region'] == 'path_vertex'
    assert projection['same_station_observation'] == (
        'projection_clamped_path_vertex')


def saved_trial(tmp_path, extra_events=()):
    trial = tmp_path / 'run_saved'
    trial.mkdir()
    metadata = dict(run_id='saved', anchor=anchor(), prepared=state(),
                    path_trace_protocol_version=1)
    (trial / 'metadata.json').write_text(json.dumps(metadata))
    (trial / 'summary.json').write_text(json.dumps(dict(
        success=True, path_trace=dict(schema_version=1))))
    f = frame()
    with (trial / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(f))
        writer.writeheader()
        writer.writerow(f)
    events = [dict(status(), event='follow_path_status'),
              dict(window(poses=None), event='path_snapshot',
                   source='controller', frame_id='odom'),
              dict(event='path_resolved', received_sim_time_s=10.,
                   path_id='controller_1',
                   odom_poses_xy_yaw=window()['poses'])]
    events.extend(extra_events)
    for sequence, event in enumerate(events, start=1):
        event.update(sequence=sequence, schema_version=1, run_id='saved')
    (trial / 'path_trace.jsonl').write_text(
        ''.join(json.dumps(e) + '\n' for e in events))
    return trial


def test_v1_reanalysis_is_offline_unique_and_preserves_all_originals(tmp_path):
    trial = saved_trial(tmp_path)
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in trial.iterdir() if p.is_file()}
    output, summary = reanalyze_trial(trial)
    second, _ = reanalyze_trial(trial)
    assert output != second
    assert output.parent == trial
    assert summary['schema_version'] == 2
    assert summary['source_trace_versions'] == [1]
    assert summary['verdict'] == 'supported'
    assert summary['same_station_critical_coverage'] == 1
    assert summary['run_id'] == 'saved'
    provenance = json.loads((output / 'analysis_metadata.json').read_text())
    assert provenance['source_files_sha256'] == original
    assert len(provenance['analysis_code_sha256']) == 64
    assert provenance['recorded_tf_only'] is True
    for name, digest in original.items():
        actual = hashlib.sha256((trial / name).read_bytes()).hexdigest()
        assert actual == digest


@pytest.mark.parametrize('kind,reason', [
    ('rejected', 'invalid_controller_window'),
    ('unresolved', 'unresolved_controller_window'),
    ('failed_tf', 'transform_unavailable'),
])
def test_replay_never_reuses_old_window_after_invalid_new_evidence(
        tmp_path, kind, reason):
    extra = [dict(window(path_id='controller_new', received_sim_time_s=10.05),
                  event='path_snapshot', source='controller')]
    if kind == 'rejected':
        extra = [dict(event='path_rejected', source='controller',
                      received_sim_time_s=10.05, reason='empty_or_short_path')]
    elif kind == 'failed_tf':
        extra.append(dict(event='path_transform_unavailable',
                          path_id='controller_new', received_sim_time_s=10.6))
    _, summary = reanalyze_trial(saved_trial(tmp_path, extra))
    assert summary['at_actual_minimum']['observation'] == reason
    assert summary['at_actual_minimum']['same_station_clearance_loss_m'] == ''
    assert summary['critical_coverage'] == 0


@pytest.mark.parametrize('kind', ['run_id', 'version', 'sequence', 'nan'])
def test_replay_rejects_mixed_or_corrupt_sources_before_creating_output(
        tmp_path, kind):
    trial = saved_trial(tmp_path)
    path = trial / 'path_trace.jsonl'
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if kind == 'run_id':
        events[-1]['run_id'] = 'another_run'
    elif kind == 'version':
        events[-1]['schema_version'] = 999
    elif kind == 'sequence':
        events[-1]['sequence'] = 1
    elif kind == 'nan':
        events[-1]['odom_poses_xy_yaw'][0][0] = float('nan')
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    with pytest.raises(ValueError):
        reanalyze_trial(trial)
    assert not list(trial.glob('path_reanalysis_*'))


def test_trace_report_cli_does_not_connect_or_re_evaluate_motion(
        tmp_path, monkeypatch, capsys):
    from terrain_navigation_pkg import simulation_experiment as command
    trial = saved_trial(tmp_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail('offline command must not connect or re-score motion')

    monkeypatch.setattr(command, 'connect', forbidden)
    monkeypatch.setattr(command, '_make_node', forbidden)
    monkeypatch.setattr(command, 'evaluate_trial', forbidden)
    assert command.main(['trace-report', '--trial', str(trial)]) == 0
    assert 'Path reanalysis:' in capsys.readouterr().out


def test_projection_does_not_jump_to_distant_future_branch():
    poses = [(0, 0, 0), (4, 0, 0), (4, 10, 0), (0, 10, 0), (0, 1, 0)]
    projected, reason = project_controller_window(poses, 0, 1)
    assert reason == 'supported'
    assert projected['index'] == 0
    assert projected['distance'] == 1.


def test_ambiguous_near_crossing_is_not_a_good_match():
    poses = [(0, 0, 0), (1, 0, 0), (1, .5, 0), (0, .5, 0), (0, 0, 0)]
    assert project_controller_window(poses, 0, 0)[1] == 'ambiguous_path_branch'
    assert project_controller_window([(0, 0, 0)], 0, 0)[0] is None


def test_full_tf_rotation_and_physical_obb_offset():
    half = math.sqrt(.5)
    result = _transform_pose([1, 0, 0, 0, 0, 0, 1],
                             [10, 5, 0, 0, 0, half, half])
    assert result == pytest.approx([10, 6, math.pi / 2])
    roll = _transform_pose([0, 1, 0, 0, 0, 0, 1], [0, 0, 0, half, 0, 0, half])
    assert roll[1] == pytest.approx(0)
    box = dict(anchor()['ego_box'], offset_forward_m=1.)
    polygon = body_polygon(0, 0, math.pi / 2, box)
    assert sum(y for x, y in polygon) / 4 == pytest.approx(1.)


def path_message(frame_id='odom', stamp=10., invalid=None):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = round((stamp - int(stamp)) * 1e9)
    for x in (-1., 2., 5.):
        p = PoseStamped()
        p.header = copy.deepcopy(msg.header)
        p.pose.position.x, p.pose.position.y = x, 1.
        p.pose.orientation.w = 1.
        msg.poses.append(p)
    if invalid == 'mixed_frames':
        msg.poses[0].header.frame_id = 'map'
    if invalid == 'nan':
        msg.poses[0].pose.position.x = float('nan')
    if invalid == 'empty':
        msg.poses.clear()
    return msg


class TraceNode:

    def __init__(self):
        self.time_s = 10.
        self.buffer = SimpleNamespace(lookup_transform=self.lookup)
        self.lookup_calls = []

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=round(self.time_s * 1e9)))

    def lookup(self, target, source, stamp):
        self.lookup_calls.append((target, source, stamp.nanoseconds))
        raise RuntimeError('no exact-stamp TF')

    def create_subscription(self, *args):
        return args

    def destroy_subscription(self, _subscription):
        return True


def test_observer_raw_candidate_and_controller_are_not_equivalent(tmp_path):
    from action_msgs.msg import GoalStatus, GoalStatusArray
    node = TraceNode()
    observer = PathTraceObserver(node, anchor(), state(), tmp_path, 'test')
    s = GoalStatus()
    s.status = 2
    observer.on_status(GoalStatusArray(status_list=[s]))
    observer.on_path(path_message(), 'candidate')
    observer.on_path(path_message(), 'controller')
    observer.resolve()
    result = observer.finish([frame()])
    raw = [
        json.loads(line)
        for line in (tmp_path / 'path_trace.jsonl').read_text().splitlines()
    ]
    snapshots = [r for r in raw if r['event'] == 'path_snapshot']
    assert len(snapshots) == 2
    assert snapshots[0]['evidence'] == 'candidate_not_acceptance'
    assert snapshots[1]['evidence'] == 'rpp_computation_window_not_actuation'
    assert snapshots[1]['candidate_at_receipt'] == snapshots[0]['path_id']
    assert result['verdict'] == 'supported'
    assert all(r['run_id'] == 'test' for r in raw)
    assert node.lookup_calls == []  # Already odom, no latest-TF query.
    with pytest.raises(FileExistsError):
        PathTraceObserver(node, anchor(), state(), tmp_path, 'overwrite')


@pytest.mark.parametrize('invalid', ['mixed_frames', 'nan', 'empty'])
def test_bad_new_path_invalidates_old_observation(tmp_path, invalid):
    node = TraceNode()
    observer = PathTraceObserver(node, anchor(), state(), tmp_path, 'test')
    observer.on_path(path_message(), 'controller')
    observer.resolve()
    node.time_s = 10.05
    observer.on_path(path_message(invalid=invalid), 'controller')
    result = observer.finish([frame()])
    assert result['at_actual_minimum'][
        'observation'] == 'invalid_controller_window'


def test_missing_exact_tf_is_not_replaced_with_latest_tf(tmp_path):
    node = TraceNode()
    observer = PathTraceObserver(node, anchor(), state(), tmp_path, 'test')
    observer.on_path(path_message(frame_id='base_link'), 'controller')
    observer.resolve()
    assert observer.pending
    node.time_s = 10.6
    result = observer.finish([frame()])
    assert result['snapshot_issues']['transform_unavailable'] == 1
    assert result['at_actual_minimum'][
        'observation'] == 'transform_unavailable'
    assert all(stamp == 10_000_000_000 for _, _, stamp in node.lookup_calls)


def test_clock_zero_does_not_collide_launch_logs(tmp_path):
    first = new_log_session(tmp_path)
    second = new_log_session(tmp_path)
    assert first != second
    for path in (first, second):
        (path / 'run_0').mkdir()
        with (path / 'run_0' / 'path_clearance.csv').open('x') as stream:
            stream.write(path.name)
    assert (first / 'run_0' / 'path_clearance.csv').read_text() == first.name


def test_trace_is_opt_in_and_cli_preserves_control_defaults():
    common = ['run', '--anchor', 'anchor.json', '--scenario', 'S1']
    options = _parser().parse_args(common)
    assert not options.trace_paths
    assert options.command_mode == 'legacy'
    assert options.arrival_radius_m == 3.
    assert _parser().parse_args(common + ['--trace-paths']).trace_paths


@pytest.mark.skipif(os.environ.get('TERRAIN_TEST_ROS_LOOPBACK') != '1',
                    reason='explicit isolated DDS trace test, domain114')
def test_real_dds_path_observation_exact_tf_and_parameter_snapshot(tmp_path):
    import rclpy
    from action_msgs.msg import GoalStatus, GoalStatusArray
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Path as PathMessage
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
    from terrain_navigation_pkg.simulation_experiment import (
        _verify_path_trace_stack, )

    rclpy.init(domain_id=114)
    names = ('controller_server', 'planner_server',
             'path_clearance_validator_node', 'far_nav2_guide_node',
             'lidar_emergency_stop_node')
    nodes = {name: Node(name) for name in names}
    source, planner, validator = (nodes[name] for name in names[:3])
    session = new_log_session(tmp_path)
    params = {
        'controller_server': {
            'FollowPath.plugin': ('nav2_regulated_pure_pursuit_controller::'
                                  'RegulatedPurePursuitController'),
            'FollowPath.lookahead_dist':
            3.5,
            'FollowPath.desired_linear_vel':
            1.,
            'FollowPath.use_velocity_scaled_lookahead_dist':
            False
        },
        'planner_server': {
            'GridBased.cost_penalty': 3.,
            'GridBased.minimum_turning_radius': 4.1
        },
        'path_clearance_validator_node': {
            'log_directory': str(session / names[2]),
            'vehicle_front_m': 2.4,
            'vehicle_rear_m': 2.5,
            'vehicle_half_width_m': 1.,
            'hard_clearance_m': .35,
            'preferred_clearance_m': .75
        },
        'far_nav2_guide_node': {
            'log_directory': str(session / names[3])
        },
        'lidar_emergency_stop_node': {
            'log_directory': str(session / names[4])
        },
    }
    for name, values in params.items():
        for key, value in values.items():
            nodes[name].declare_parameter(key, value)
    probe = Node('path_trace_dds_probe',
                 parameter_overrides=[
                     Parameter('use_sim_time', Parameter.Type.BOOL, True)
                 ])
    probe.buffer = Buffer(node=probe)
    listener = TransformListener(probe.buffer, probe)
    observer = PathTraceObserver(probe, anchor(), state(), tmp_path,
                                 'dds_test')
    path_pub = source.create_publisher(PathMessage, '/received_global_plan',
                                       10)
    candidate_pub = planner.create_publisher(PathMessage, '/plan', 10)
    status_pub = source.create_publisher(
        GoalStatusArray, '/follow_path/_action/status',
        QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    clearance_pub = validator.create_publisher(
        String, '/navigation/path_clearance/status', 10)
    clock_pub = source.create_publisher(Clock, '/clock', 10)
    broadcaster = StaticTransformBroadcaster(source)
    executor = SingleThreadedExecutor()
    for node in nodes.values():
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        clock = Clock()
        clock.clock.sec = 10
        transform = TransformStamped()
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.rotation.w = 1.
        broadcaster.sendTransform(transform)
        active = GoalStatus()
        active.status = 2
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            clock_pub.publish(clock)
            status_pub.publish(GoalStatusArray(status_list=[active]))
            candidate_pub.publish(path_message())
            path_pub.publish(path_message(frame_id='base_link'))
            clearance_pub.publish(String(data='{"hard_valid":true}'))
            rclpy.spin_once(probe, timeout_sec=.01)
            observer.resolve()
            if observer.windows and observer.windows[-1].get(
                    'controller_goal_id'):
                break
        assert observer.windows, 'real controller path subscription/TF missing'
        actual = _verify_path_trace_stack(probe, rclpy)
        assert actual['session_log_root'] == str(session)
        assert actual['parameters']['controller_server'][
            'FollowPath.lookahead_dist'] == 3.5
        result = observer.finish([frame(10.2)])
        assert result['verdict'] == 'supported'
        assert result['at_actual_minimum'][
            'tracking_cross_m'] == pytest.approx(-1.)
        assert (tmp_path / 'path_trace_summary.json').is_file()
        assert listener is not None
    finally:
        observer.close()
        executor.shutdown()
        thread.join(timeout=2)
        probe.destroy_node()
        for node in nodes.values():
            node.destroy_node()
        rclpy.shutdown()
