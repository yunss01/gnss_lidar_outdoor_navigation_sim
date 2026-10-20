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


def test_real_dds_preflight_tf_pairing_and_recording():
    import rclpy
    from geometry_msgs.msg import PointStamped, TransformStamped, Twist
    from lifecycle_msgs.srv import GetState
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import NavSatFix, PointCloud2
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster
    from terrain_navigation_pkg.simulation_experiment import _make_node, _preflight

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
    }
    tf = TransformBroadcaster(source)
    tick = [200]
    position_after_goal = [False]
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

    def publish():
        tick[0] += 1
        stamp = tick[0] * 50_000_000
        truth = Odometry()
        truth.header.stamp.sec = stamp // 1_000_000_000
        truth.header.stamp.nanosec = stamp % 1_000_000_000
        truth.header.frame_id = 'experiment_truth_world'
        truth.pose.pose.position.x = (tick[0] - 200) / 20
        truth.pose.pose.orientation.w = 1.0
        truth.twist.twist.linear.x = 1.0
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
                'armed': False,
                'sim_time_s': stamp / 1e9,
                'truth_has_navigation_tf': False,
                'steer_requested': 0.1,
                'steer_applied': 0.1,
                'brake': 0,
            })))
        cloud = PointCloud2()
        cloud.header = truth.header
        publishers['cloud'].publish(cloud)
        publishers['origin'].publish(NavSatFix())
        if position_after_goal[0]:
            publishers['current'].publish(PointStamped())
        publishers['safety'].publish(String(data='clear'))
        publishers['cmd'].publish(Twist())

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
        _preflight(trial, SimpleNamespace(profile='position_left'), rclpy)
        assert trial.current_local is None
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
    finally:
        trial.destroy_node()
        executor.shutdown()
        thread.join(timeout=2)
        source.destroy_timer(timer)
        for service in services:
            source.destroy_service(service)
        source.destroy_node()
        rclpy.shutdown()
