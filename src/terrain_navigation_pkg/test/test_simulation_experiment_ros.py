"""Opt-in localhost DDS smoke test, not a CARLA driving validation.

Run with TERRAIN_TEST_ROS_LOOPBACK=1 in the isolated experiment environment.
The fixture uses domain 114, separate from the live experiment domain 113.
"""

import copy
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TERRAIN_TEST_ROS_LOOPBACK') != '1',
    reason='explicit localhost DDS test; not required for pure regression')


def test_latest_dds_discards_backlogged_motion_before_stop():
    """Real middleware queue, no CARLA, shared graph or vehicle actuation."""
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from terrain_navigation_pkg.simulation_experiment_bridge import (
        LatestCommandReceiver,
    )
    rclpy.init(domain_id=114)
    source = Node('command_queue_fixture')
    legacy = Node('legacy_command_probe')
    topic = '/experiment_test/queue_probe'
    old, new = [], []
    publisher = source.create_publisher(Twist, topic, 10)
    subscription = legacy.create_subscription(
        Twist, topic, lambda msg: old.append(msg.linear.x), 10)
    receiver = LatestCommandReceiver(
        lambda msg: new.append(msg.linear.x), source.context, topic)
    executor = SingleThreadedExecutor()
    executor.add_node(legacy)
    try:
        deadline = time.monotonic() + 3
        while publisher.get_subscription_count() < 2:
            assert time.monotonic() < deadline, 'DDS discovery timed out'
            time.sleep(.01)
        assert receiver.subscription.qos_profile.depth == 1
        # No consumer spins during the burst. Keep ten samples in legacy,
        # but only the newest setpoint in the command-only receiver.
        for _ in range(9):
            move = Twist()
            move.linear.x = 1.0
            publisher.publish(move)
        publisher.publish(Twist())
        time.sleep(.1)  # Delivery, deliberately no callback execution.
        executor.spin_once(timeout_sec=.2)
        receiver.poll()
        assert old == [1.0]
        assert new == [0.0]
        # No new packet is fabricated when publication stops.
        for _ in range(3):
            receiver.poll()
        assert new == [0.0]
    finally:
        receiver.close()
        executor.shutdown()
        legacy.destroy_subscription(subscription)
        legacy.destroy_node()
        source.destroy_node()
        rclpy.shutdown()


def test_real_dds_preflight_tf_pairing_and_recording():
    import rclpy
    from geometry_msgs.msg import PointStamped, TransformStamped, Twist
    from lifecycle_msgs.srv import GetState
    from nav_msgs.msg import Odometry
    from rcl_interfaces.msg import ParameterValue
    from rcl_interfaces.srv import GetParameters
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import NavSatFix, PointCloud2
    from std_msgs.msg import Bool, Float32, String
    from tf2_ros import TransformBroadcaster
    from terrain_navigation_pkg.simulation_experiment import (
        _halt_client,
        _make_node,
        _preflight,
    )

    rclpy.init(domain_id=114)
    assert rclpy.get_default_context().get_domain_id() == 114
    source = Node('experiment_localhost_fixture')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publishers = {
        'truth':
        source.create_publisher(Odometry, '/experiment/truth_odometry', 10),
        'nav':
        source.create_publisher(Odometry, '/vehicle/odometry', 10),
        'clock':
        source.create_publisher(Clock, '/clock', 10),
        'bridge':
        source.create_publisher(String, '/experiment/bridge_status', 10),
        'cloud':
        source.create_publisher(PointCloud2, '/lidar/points',
                                qos_profile_sensor_data),
        'origin':
        source.create_publisher(NavSatFix, '/navigation/origin_gnss', latched),
        'current':
        source.create_publisher(PointStamped, '/navigation/current_local', 10),
        'safety':
        source.create_publisher(String, '/safety/state', 10),
        'gate':
        source.create_publisher(String,
                                '/navigation/path_validity_gate/status',
                                latched),
        'cmd':
        source.create_publisher(Twist, '/cmd_vel', 10),
        'distance':
        source.create_publisher(Float32, '/navigation/distance_to_goal', 10),
        'reached':
        source.create_publisher(Bool, '/navigation/goal_reached', latched),
        'approach': source.create_publisher(
            String, '/navigation/far_terminal_approach/status', latched),
    }
    tf = TransformBroadcaster(source)
    tick = [200]
    position_after_goal = [False]
    armed = [False]
    stop_requested = [None]
    # Retain one change event before the trial subscriber exists, as the
    # real gate does; a healthy unchanged status has no periodic heartbeat.
    publishers['gate'].publish(String(data='path_valid'))

    def active(_request, response):
        response.current_state.id = 3
        response.current_state.label = 'active'
        return response

    services = [
        source.create_service(GetState, f'/{name}/get_state', active)
        for name in ('controller_server', 'planner_server', 'bt_navigator')
    ]
    arrival_radius = [3.0]

    def get_parameters(request, response):
        assert request.names == ['arrival_radius_m']
        response.values = [
            ParameterValue(type=3, double_value=arrival_radius[0])
        ]
        return response

    services.append(
        source.create_service(GetParameters,
                              '/gnss_goal_manager_node/get_parameters',
                              get_parameters))
    approach_distance = [0.0]

    def get_far_parameters(request, response):
        assert request.names == ['terminal_approach_distance_m',
                                 'terminal_approach_capture_m']
        response.values = [ParameterValue(type=3, double_value=value)
                           for value in (approach_distance[0], 1.0)]
        return response

    services.append(source.create_service(
        GetParameters, '/far_nav2_guide_node/get_parameters',
        get_far_parameters))

    def on_control(message):
        data = json.loads(message.data)
        if data.get('event') == 'stop' and data.get('run_id') == 'fixture_run':
            if stop_requested[0] is None:
                stop_requested[0] = time.monotonic()

    control_subscription = source.create_subscription(
        String, '/experiment/control', on_control, 10)

    def publish():
        tick[0] += 1
        stamp = tick[0] * 50_000_000
        truth = Odometry()
        truth.header.stamp.sec = stamp // 1_000_000_000
        truth.header.stamp.nanosec = stamp % 1_000_000_000
        truth.header.frame_id = 'experiment_truth_world'
        truth.pose.pose.position.x = (tick[0] - 200) / 20
        truth.pose.pose.orientation.w = 1.0
        if (stop_requested[0] is not None
                and time.monotonic() - stop_requested[0] >= 0.15):
            armed[0] = False
        truth.twist.twist.linear.x = 1.0 if armed[0] else 0.0
        nav = copy.deepcopy(truth)
        nav.header.frame_id = 'odom'
        nav.child_frame_id = 'base_link'
        nav.pose.pose.position.y = 0.3
        transform = TransformStamped()
        transform.header = nav.header
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = nav.pose.pose.position.x
        transform.transform.translation.y = nav.pose.pose.position.y
        transform.transform.rotation = nav.pose.pose.orientation
        clock = Clock()
        clock.clock = truth.header.stamp
        publishers['clock'].publish(clock)
        publishers['truth'].publish(truth)
        publishers['nav'].publish(nav)
        tf.sendTransform(transform)
        publishers['bridge'].publish(
            String(data=json.dumps({
                'profile': 'position_left',
                'armed': armed[0],
                'adapter_session': 'fixture_session',
                'run_id': 'fixture_run',
                'sim_time_s': stamp / 1e9,
                'truth_has_navigation_tf': False,
                'steer_requested': 0.1,
                'steer_applied': 0.1,
                'brake': 0,
                'command_mode': 'legacy',
                'command_telemetry_version': 1,
                'accepted_command_speed_mps': 0.75,
                'accepted_command_yaw_rate_rps': .03,
                'accepted_command_wall_age_s': .02,
                'control_sequence': tick[0],
                'control_sim_time_s': stamp / 1e9 - .05,
                'command_rx_count': tick[0],
                'throttle': .2,
            })))
        cloud = PointCloud2()
        cloud.header = truth.header
        publishers['cloud'].publish(cloud)
        publishers['origin'].publish(NavSatFix())
        if position_after_goal[0]:
            publishers['current'].publish(PointStamped())
            publishers['distance'].publish(Float32(data=0.8))
            publishers['reached'].publish(Bool(data=True))
        publishers['safety'].publish(String(data='clear'))
        publishers['cmd'].publish(Twist())
        publishers['approach'].publish(String(data=(
            'approach' if approach_distance[0] > 0 else 'disabled')))

    timer = source.create_timer(0.05, publish)
    executor = SingleThreadedExecutor()
    executor.add_node(source)
    thread = threading.Thread(target=executor.spin, daemon=True)
    anchor = {
        'start_ros':
        dict(x=0, y=0, yaw_rad=0),
        'route_length_m':
        40,
        'corridor_half_width_m':
        7,
        'ego_box':
        dict(half_length_m=2.4,
             half_width_m=1,
             offset_forward_m=0,
             offset_left_m=0,
             yaw_offset_rad=0),
    }
    trial = _make_node(anchor, {'obstacles': []})
    thread.start()
    try:
        options = SimpleNamespace(profile='position_left',
                                  arrival_radius_m=3.0,
                                  navigation_variant='baseline',
                                  command_mode='legacy')
        assert _preflight(trial, options, rclpy) == 3.0
        options.command_mode = 'latest'
        with pytest.raises(RuntimeError, match='command mode/telemetry'):
            _preflight(trial, options, rclpy)
        assert not trial.recording
        options.command_mode = 'legacy'
        assert trial.current_local is None
        options.arrival_radius_m = 1.0
        with pytest.raises(RuntimeError, match='arrival radius mismatch'):
            _preflight(trial, options, rclpy)
        assert not trial.recording
        arrival_radius[0] = 1.0
        assert _preflight(trial, options, rclpy) == 1.0
        options.navigation_variant = 'rejoin'
        with pytest.raises(RuntimeError, match='navigation variant mismatch'):
            _preflight(trial, options, rclpy)
        assert not trial.recording
        approach_distance[0] = 8.0
        assert _preflight(trial, options, rclpy) == 1.0
        assert trial.active_navigation_parameters[
            'terminal_approach_distance_m'] == 8.0
        position_after_goal[0] = True
        trial.start_time_s = float(trial.bridge['sim_time_s'])
        trial.recording = True
        deadline = time.monotonic() + 3
        while len(trial.frames) < 10 and time.monotonic() < deadline:
            rclpy.spin_once(trial, timeout_sec=0.05)
        assert len(trial.frames) >= 10
        assert trial.current_local is not None
        assert all(row['path_gate_state'] == 'path_valid'
                   for row in trial.frames)
        assert all(row['tf_consistent'] for row in trial.frames)
        assert all(
            abs(row['pose_bias_m'] - 0.3) < 1e-9 for row in trial.frames)
        assert all(row['truth_y'] == 0 for row in trial.frames)
        assert all(row['accepted_command_speed_mps'] == .75
                   for row in trial.frames)
        assert all(row['accepted_command_yaw_rate_rps'] == .03
                   for row in trial.frames)
        assert all(row['throttle'] == .2 for row in trial.frames)
        assert any(row['navigation_goal_reached'] == 1 for row in trial.frames)
        assert any(
            abs(row['navigation_goal_distance_m'] - 0.8) < 1e-6
            for row in trial.frames if row['navigation_goal_distance_m'] != '')
        assert any(row['terminal_approach_phase'] == 'approach'
                   for row in trial.frames)
        trial.recording = False
        armed[0] = True
        deadline = time.monotonic() + 2
        while not trial.bridge.get('armed') and time.monotonic() < deadline:
            rclpy.spin_once(trial, timeout_sec=0.05)
        assert trial.bridge['armed']
        initial_stamp = max(trial.truth)
        halted = _halt_client(trial, 'fixture_run', 'position_left', rclpy,
                              timeout_s=3.0)
        assert halted['halt_acknowledged']
        assert halted['vehicle_stopped_after_halt']
        assert halted['halt_confirmation_complete']
        assert max(trial.truth) > initial_stamp
    finally:
        trial.destroy_node()
        executor.shutdown()
        thread.join(timeout=2)
        source.destroy_timer(timer)
        source.destroy_subscription(control_subscription)
        for service in services:
            source.destroy_service(service)
        source.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('variant,distance', [
    ('baseline', 0.0), ('rejoin', 8.0),
])
def test_real_far_node_parameters_and_planning(tmp_path, variant, distance):
    import math
    import numpy as np
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from terrain_navigation_pkg.far_guide_core import GuideGrid
    from terrain_navigation_pkg.far_nav2_guide_node import FarNav2GuideNode
    from terrain_navigation_pkg.simulation_experiment import (
        _verify_navigation_variant,
    )

    rclpy.init(domain_id=114, args=[
        '--ros-args', '-p', f'log_directory:={tmp_path}',
        '-p', f'terminal_approach_distance_m:={distance}',
    ])
    far = FarNav2GuideNode()
    probe = Node('variant_probe')
    probe.active_navigation_parameters = None
    executor = SingleThreadedExecutor()
    executor.add_node(far)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        assert _verify_navigation_variant(probe, variant, rclpy) == {
            'terminal_approach_distance_m': distance,
            'terminal_approach_capture_m': 1.0}
        other = 'baseline' if variant == 'rejoin' else 'rejoin'
        with pytest.raises(RuntimeError, match='navigation variant mismatch'):
            _verify_navigation_variant(probe, other, rclpy)
        far.latest_goal_tangent_yaw_rad = 0.0
        grid = GuideGrid(np.zeros((100, 120)),
                         np.zeros((100, 120), dtype=bool), 0.5, -10, -20)
        start, goal = (25, 3), (40, 0)
        path = far._plan_world_path(grid, start, goal, time.monotonic(), [])
        assert path[-1] == pytest.approx((32, 0) if distance else goal)
        subgoal, yaw, final = far._select_segment_target(
            path, goal, math.dist(start, goal), start, time.monotonic())
        if distance:
            assert subgoal == pytest.approx((32, 0))
            assert yaw == 0.0 and not final
    finally:
        executor.shutdown()
        thread.join(timeout=2)
        probe.destroy_node()
        far.destroy_node()
        rclpy.shutdown()
