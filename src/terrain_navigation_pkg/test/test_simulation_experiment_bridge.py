"""Execute the ORIGINAL bridge/control method bodies with a fake CARLA world.

No ROS graph, simulator, GPU or actuation is needed for this contract test.
"""

import argparse
import ast
import copy
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from terrain_navigation_pkg.simulation_experiment_bridge import (
    inspect_client,
    install_adapter,
    RealtimeClock,
)
from terrain_navigation_pkg.simulation_experiment_core import (
    PROFILES,
    quaternion_yaw,
)
from terrain_navigation_pkg import simulation_experiment_bridge as bridge_cli
from terrain_navigation_pkg import simulation_experiment_scenario as scenario
from terrain_navigation_pkg import controlled_carla_obstacle

CLIENT = Path.home() / 'carla/0.10.0/PythonAPI/examples/manual_control.py'


class Collector:

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(copy.deepcopy(message))

    def sendTransform(self, messages):
        self.messages.extend(
            copy.deepcopy(
                messages if isinstance(messages, list) else [messages]))


class FakeNode:

    context = None

    def create_publisher(self, *_args):
        return Collector()

    def create_subscription(self, *_args):
        return None

    def destroy_subscription(self, _subscription):
        return True

    def get_logger(self):
        return SimpleNamespace(error=lambda _text: None)


class Control:

    def __init__(self,
                 throttle=0,
                 steer=0,
                 brake=0,
                 hand_brake=False,
                 reverse=False,
                 manual_gear_shift=False,
                 gear=1):
        self.throttle, self.steer, self.brake = throttle, steer, brake
        self.hand_brake, self.reverse = hand_brake, reverse
        self.manual_gear_shift, self.gear = manual_gear_shift, gear


def module_fixture():
    if not CLIENT.exists():
        pytest.skip('requires the documented CARLA manual client')
    from geometry_msgs.msg import TransformStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Float32
    namespace = {
        'math': math,
        'time': time,
        'TransformStamped': TransformStamped,
        'Odometry': Odometry,
        'TwistStamped': TwistStamped,
        'Float32': Float32,
        'Clock': Clock,
        'VEHICLE_FRAME_ID': 'base_link',
        'ODOMETRY_FRAME_ID': 'odom',
        'EXTERNAL_CONTROL_WHEELBASE_M': 2.85,
        'EXTERNAL_CONTROL_MAX_SPEED_KMH': 12,
        'EXTERNAL_CONTROL_MAX_STEERING_ANGLE_RAD': math.radians(35),
        'EXTERNAL_CONTROL_TIMEOUT_SECONDS': 0.5,
    }
    tree = ast.parse(CLIENT.read_text())
    names = {
        '_carla_rotation_to_ros_quaternion', '_stamp_from_sim_time',
        '_carla_transform_to_ros_message'
    }
    bodies = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    for class_name, method_name in [
        ('CarlaRos2Publisher', 'publish_vehicle_speed'),
        ('CarlaRos2Publisher', '_on_command'),
        ('CarlaRos2Publisher', 'external_command'),
        ('KeyboardControl', '_apply_external_control')
    ]:
        class_ast = next(
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == class_name)
        bodies.append(
            next(item for item in class_ast.body
                 if isinstance(item, ast.FunctionDef)
                 and item.name == method_name))
    exec(
        compile(ast.Module(body=bodies, type_ignores=[]), str(CLIENT), 'exec'),
        namespace)

    class BaseBridge:
        enabled = True

        def __init__(self, **_kwargs):
            self.node = FakeNode()
            self.external_control_enabled = True
            self.command = (1.0, 0.2, 0)
            self.command_subscription = object()
            for key in ('odometry_publisher', 'velocity_publisher',
                        'speed_publisher', 'clock_publisher',
                        'tf_broadcaster'):
                setattr(self, key, Collector())

        def external_command(self):
            return self.command

        def _on_command(self, message):
            self.command = (message.linear.x, message.angular.z, 0.0)

        def close(self):
            self.closed = True

    BaseBridge.publish_vehicle_speed = namespace['publish_vehicle_speed']
    control_class = type(
        'KeyboardControl', (),
        {'_apply_external_control': namespace['_apply_external_control']})
    return SimpleNamespace(CarlaRos2Publisher=BaseBridge,
                           KeyboardControl=control_class,
                           carla=SimpleNamespace(VehicleControl=Control),
                           EXTERNAL_CONTROL_TOPIC='/cmd_vel',
                           raw_on_command=namespace['_on_command'],
                           raw_external_command=namespace['external_command'],
                           Odometry=Odometry,
                           ODOMETRY_FRAME_ID='odom',
                           VEHICLE_FRAME_ID='base_link')


class Vehicle:

    def __init__(self):
        self.controls = []

    def get_transform(self):
        return SimpleNamespace(
            location=SimpleNamespace(x=10.0, y=5.0, z=1.0),
            rotation=SimpleNamespace(yaw=0, pitch=0, roll=0),
            get_forward_vector=lambda: SimpleNamespace(x=1, y=0, z=0),
            get_right_vector=lambda: SimpleNamespace(x=0, y=1, z=0),
            get_up_vector=lambda: SimpleNamespace(x=0, y=0, z=1))

    def get_velocity(self):
        return SimpleNamespace(x=1.0, y=0.0, z=0.0)

    def apply_control(self, control):
        self.controls.append(copy.deepcopy(control))


def start(bridge, profile):
    import json
    from std_msgs.msg import String
    bridge._experiment_control(
        String(data=json.dumps({
            'event': 'start',
            'run_id': 'trial',
            'profile': profile,
            'route_yaw_rad': 0,
        })))


@pytest.mark.parametrize('profile', ['baseline', 'position_left', 'yaw_left'])
def test_original_pose_fault_keeps_tf_and_truth_consistent(profile):
    module = module_fixture()
    install_adapter(module, PROFILES[profile], inspect_client(CLIENT))
    bridge = module.CarlaRos2Publisher()
    vehicle = Vehicle()
    bridge.publish_vehicle_speed(vehicle, 10)
    assert bridge.external_command() is None
    start(bridge, profile)
    bridge.publish_vehicle_speed(vehicle, 14.5)
    truth = bridge._experiment_truth_publisher.messages[-1]
    nav = bridge.odometry_publisher.publisher.messages[-1]
    tf = bridge.tf_broadcaster.broadcaster.messages[-1]
    assert truth.header.frame_id == 'experiment_truth_world'
    assert truth.pose.pose.position.x == 10
    assert truth.pose.pose.position.y == -5
    assert tf.header.frame_id == 'odom' and tf.child_frame_id == 'base_link'
    assert tf.transform.translation.x == nav.pose.pose.position.x
    assert tf.transform.translation.y == nav.pose.pose.position.y
    assert tf.transform.rotation == nav.pose.pose.orientation
    assert truth.pose.pose.orientation.z == 0
    if profile == 'position_left':
        assert nav.pose.pose.position.y == pytest.approx(-4.7)
    if profile == 'yaw_left':
        q = nav.pose.pose.orientation
        assert quaternion_yaw(
            (q.x, q.y, q.z, q.w)) == pytest.approx(math.radians(3))


def test_original_control_brake_is_immediate_despite_steering_delay():
    module = module_fixture()
    install_adapter(module, PROFILES['steering_both'], inspect_client(CLIENT))
    bridge = module.CarlaRos2Publisher()
    vehicle = Vehicle()
    controller = module.KeyboardControl()
    controller._control = Control()
    controller._external_speed_controller = SimpleNamespace(
        reset=lambda: None, run_step=lambda _speed, _dt: (0.2, 0))
    world = SimpleNamespace(
        sensor_ros=bridge,
        player=vehicle,
        world=SimpleNamespace(
            get_settings=lambda: SimpleNamespace(fixed_delta_seconds=0.05)))
    bridge.publish_vehicle_speed(vehicle, 10)
    start(bridge, 'steering_both')
    for timestamp in (14, 14.1, 14.21):
        bridge.publish_vehicle_speed(vehicle, timestamp)
        controller._apply_external_control(world, None)
    assert vehicle.controls[0].steer == 0
    assert abs(vehicle.controls[-1].steer) > 0
    assert vehicle.controls[-1].throttle == 0.2
    bridge.command = None  # ORIGINAL watchdog/stale-command path.
    controller._apply_external_control(world, None)
    assert vehicle.controls[-1].brake == 1
    assert vehicle.controls[-1].steer == 0
    assert not bridge._experiment_steering.history


def test_stop_event_latches_client_against_residual_nav_commands():
    import json
    from std_msgs.msg import String
    module = module_fixture()
    install_adapter(module, PROFILES['baseline'], inspect_client(CLIENT))
    bridge = module.CarlaRos2Publisher()
    bridge._experiment_time_s = 10
    start(bridge, 'baseline')
    assert bridge.external_command() is not None
    bridge._experiment_control(
        String(data=json.dumps({
            'event': 'stop',
            'run_id': 'trial'
        })))
    assert bridge.external_command() is None
    assert bridge.command is not None


def test_latest_polls_before_original_control_and_stop_cannot_resume(
        monkeypatch):
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    receivers = []

    class Receiver:
        def __init__(self, callback, context, topic):
            assert topic == '/cmd_vel'
            self.callback, self.pending, self.closed = callback, None, False
            receivers.append(self)

        def poll(self):
            if self.pending is not None:
                self.callback(self.pending)
                self.pending = None

        def close(self):
            self.closed = True

    monkeypatch.setattr(bridge_cli, 'LatestCommandReceiver', Receiver)
    module = module_fixture()
    install_adapter(module, PROFILES['steering_both'], inspect_client(CLIENT),
                    command_mode='latest')
    bridge = module.CarlaRos2Publisher()
    vehicle = Vehicle()
    controller = module.KeyboardControl()
    controller._control = Control()
    controller._external_speed_controller = SimpleNamespace(
        reset=lambda: None, run_step=lambda *_args: (.2, 0))
    world = SimpleNamespace(
        sensor_ros=bridge, player=vehicle,
        world=SimpleNamespace(get_settings=lambda: SimpleNamespace(
            fixed_delta_seconds=.05)))
    start(bridge, 'steering_both')
    bridge.publish_vehicle_speed(vehicle, 10)
    receivers[0].pending = Twist()  # Fresh STOP replaces queued motion.
    controller._apply_external_control(world, None)
    assert vehicle.controls[-1].brake == 1.0
    assert vehicle.controls[-1].throttle == 0
    assert vehicle.controls[-1].steer == 0
    assert bridge._experiment_command_rx_count == 1
    bridge.publish_vehicle_speed(vehicle, 10.05)
    status = json.loads(bridge._experiment_status_publisher.messages[-1].data)
    assert status['command_mode'] == 'latest'
    assert status['command_telemetry_version'] == 1
    assert status['accepted_command_speed_mps'] == 0
    assert status['accepted_command_yaw_rate_rps'] == 0
    assert status['control_sequence'] == 1
    assert status['control_sim_time_s'] == 10
    bridge._experiment_control(String(data=json.dumps(
        dict(event='stop', run_id='trial'))))
    move = Twist()
    move.linear.x = 1.0
    receivers[0].pending = move
    controller._apply_external_control(world, None)
    assert vehicle.controls[-1].brake == 1
    assert bridge._experiment_accepted_command is None
    bridge.close()
    bridge.close()  # Dedicated executor closed once, before base shutdown.
    assert receivers[0].closed and bridge.closed


def test_original_command_watchdog_preserves_receipt_age(monkeypatch):
    from geometry_msgs.msg import Twist
    module = module_fixture()
    now = [10.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    state = SimpleNamespace(external_control_enabled=True,
                            command_wall_time=None)
    assert module.raw_external_command(state) is None
    message = Twist()
    message.linear.x, message.angular.z = 1.0, .2
    module.raw_on_command(state, message)
    assert module.raw_external_command(state) == (1.0, .2, 0)
    now[0] = 10.51
    assert module.raw_external_command(state) is None


def test_latest_control_still_brakes_on_original_watchdog(monkeypatch):
    from geometry_msgs.msg import Twist
    now = [10.0]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(bridge_cli, 'LatestCommandReceiver',
                        lambda *_args: SimpleNamespace(poll=lambda: None))
    module = module_fixture()
    # Execute the original receipt/watchdog bodies, not the fixture's fake
    # constant command. Only the DDS transport is replaced in this unit test.
    module.CarlaRos2Publisher._on_command = module.raw_on_command
    module.CarlaRos2Publisher.external_command = module.raw_external_command
    install_adapter(module, PROFILES['steering_delay'], inspect_client(CLIENT),
                    command_mode='latest')
    bridge = module.CarlaRos2Publisher()
    bridge.command_wall_time = None
    start(bridge, 'steering_delay')
    vehicle = Vehicle()
    controller = module.KeyboardControl()
    controller._control = Control()
    controller._external_speed_controller = SimpleNamespace(
        reset=lambda: None, run_step=lambda *_args: (.2, 0))
    world = SimpleNamespace(
        sensor_ros=bridge, player=vehicle,
        world=SimpleNamespace(get_settings=lambda: SimpleNamespace(
            fixed_delta_seconds=.05)))
    move = Twist()
    move.linear.x, move.angular.z = 1.0, .2
    bridge._on_command(move)
    controller._apply_external_control(world, None)
    assert vehicle.controls[-1].throttle == .2
    now[0] += .51  # Polling/logging must NOT renew receipt age.
    controller._apply_external_control(world, None)
    assert vehicle.controls[-1].throttle == 0
    assert vehicle.controls[-1].brake == 1
    assert vehicle.controls[-1].steer == 0
    assert bridge._experiment_accepted_command is None


def test_wrong_profile_cannot_arm_the_client():
    module = module_fixture()
    install_adapter(module, PROFILES['baseline'], inspect_client(CLIENT))
    bridge = module.CarlaRos2Publisher()
    start(bridge, 'position_left')
    assert not bridge._experiment_armed


def test_incompatible_manual_client_is_rejected(tmp_path):
    path = tmp_path / 'manual.py'
    path.write_text('class CarlaRos2Publisher:\n    pass\n')
    with pytest.raises(RuntimeError):
        inspect_client(path)


def test_experiment_clock_does_not_fast_forward_physics():
    rates = []
    clock = SimpleNamespace(tick_busy_loop=lambda rate: rates.append(rate),
                            get_time=lambda: 50)
    wrapped = RealtimeClock(clock, 20)
    wrapped.tick_busy_loop(100)
    wrapped.tick_busy_loop(10)
    assert rates == [20, 10]
    assert wrapped.get_time() == 50


def test_experiment_ui_does_not_reuse_normal_saved_routes():
    module = module_fixture()

    class NormalWorld:

        def _load_waypoint_route(self):
            return self.waypoint_route_path

        def _load_navigation_goal(self):
            return self.navigation_goal_path

    module.World = NormalWorld
    install_adapter(module, PROFILES['baseline'], inspect_client(CLIENT))
    world = module.World()
    assert '/experiments/ui/' in world._load_waypoint_route()
    assert '/experiments/ui/' in world._load_navigation_goal()


@pytest.mark.parametrize('start_mode', ['auto', 'anchor'])
def test_cli_negative_spawn_reaches_original_argument_parser(
        start_mode, tmp_path, monkeypatch):
    if not CLIENT.exists():
        pytest.skip('requires the documented CARLA manual client')
    # Execute the original parser through parse_args, without its GUI/world.
    main_ast = next(
        item for item in ast.parse(CLIENT.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == 'main')
    parser_body = []
    for statement in main_ast.body:
        parser_body.append(statement)
        if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'args'
                        for target in statement.targets)):
            break
    captured = {}

    def parse_original():
        namespace = {
            'argparse': argparse,
            'SIMULATION_FIXED_DELTA_SECONDS': 0.01,
        }
        exec(compile(ast.Module(body=parser_body, type_ignores=[]),
                     str(CLIENT), 'exec'), namespace)
        captured.update(vars(namespace['args']))

    manual = SimpleNamespace(
        pygame=SimpleNamespace(time=SimpleNamespace(Clock=lambda: object())),
        main=parse_original)
    monkeypatch.setattr(
        bridge_cli.importlib.util, 'spec_from_file_location',
        lambda *_args: SimpleNamespace(
            loader=SimpleNamespace(exec_module=lambda _module: None)))
    monkeypatch.setattr(bridge_cli.importlib.util, 'module_from_spec',
                        lambda _spec: manual)
    monkeypatch.setattr(bridge_cli, 'install_adapter', lambda *_args: None)
    monkeypatch.setattr(bridge_cli.sys, 'argv', list(bridge_cli.sys.argv))
    monkeypatch.setattr(bridge_cli.sys, 'path', list(bridge_cli.sys.path))
    monkeypatch.setenv('ROS_DOMAIN_ID', '113')
    client = SimpleNamespace(set_timeout=lambda _time: None,
                             get_world=lambda: object())
    monkeypatch.setattr(
        controlled_carla_obstacle, 'import_carla',
        lambda _root: SimpleNamespace(Client=lambda *_args: client))
    transform = SimpleNamespace(
        location=SimpleNamespace(x=-103.179, y=-14.435, z=0.6),
        rotation=SimpleNamespace(yaw=-89.358))
    monkeypatch.setattr(scenario, 'straight_start_candidates',
                        lambda *_args: [transform])
    arguments = ['--profile', 'baseline', '--gnss-quality', 'rtk_fixed']
    if start_mode == 'auto':
        arguments.append('--auto-straight-start')
    else:
        anchor = tmp_path / 'anchor.json'
        anchor.write_text(json.dumps({
            'schema_version': 1,
            'map_name': 'Town',
            'vehicle_blueprint': 'vehicle.lincoln.mkz',
            'start_carla': dict(x=-103.179, y=-14.435, z=0.35,
                                yaw=-89.358, pitch=0, roll=0),
            'route_length_m': 40,
            'obstacle_distance_m': 20,
            'corridor_half_width_m': 7,
            'waypoints': [dict(latitude=37, longitude=127, altitude=0)] * 2,
        }))
        arguments += ['--anchor', str(anchor)]
    assert bridge_cli.main(arguments) == 0
    assert [float(value) for value in captured['spawn_point'].split(',')] == (
        pytest.approx([-103.179, -14.435, 0.6, -89.358]))
    assert captured['sync'] and captured['external_control']
    assert captured['gnss_quality'] == 'rtk_fixed'
