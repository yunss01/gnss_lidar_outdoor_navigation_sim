"""Process-local adapter around the existing CARLA client; no file patching.

Normal manual_control.py and F9/F10 are untouched. Navigation still uses a
synthetic oracle pose, optionally biased; this is NOT a GNSS/IMU estimator.
"""

import argparse
import ast
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

from .simulation_experiment_core import (
    position_fault,
    PROFILES,
    rotate_quaternion_yaw,
    SteeringFault,
    validate_anchor,
)


def inspect_client(path):
    """Refuse incompatible clients instead of silently retaining truth TF."""
    source = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(source)
    required = {
        'CarlaRos2Publisher': {'publish_vehicle_speed', 'external_command'},
        'KeyboardControl': {'_apply_external_control'},
        'World': {'_load_waypoint_route', '_load_navigation_goal'},
    }
    for class_name, methods in required.items():
        classes = [
            item for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == class_name
        ]
        available = {
            item.name
            for item in classes[0].body if isinstance(item, ast.FunctionDef)
        } if classes else set()
        if not methods <= available:
            raise RuntimeError(f'incompatible CARLA client: {class_name}')
    controls = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == 'KeyboardControl')
    publish = next(item for item in controls.body
                   if isinstance(item, ast.FunctionDef)
                   and item.name == '_apply_external_control')
    if [arg.arg for arg in publish.args.args] != ['self', 'world', 'clock']:
        raise RuntimeError('incompatible external-control method signature')
    return hashlib.sha256(source.encode()).hexdigest()


class PublisherTap:

    def __init__(self, publisher, transform):
        self.publisher, self.transform = publisher, transform

    def publish(self, message):
        self.publisher.publish(self.transform(message))

    def __getattr__(self, name):
        return getattr(self.publisher, name)


class TfTap:
    """The existing sole dynamic TF publisher emits the SAME biased pose."""

    def __init__(self, broadcaster, bridge, module):
        self.broadcaster = broadcaster
        self.bridge, self.module = bridge, module

    def sendTransform(self, message):
        values = message if isinstance(message, list) else [message]
        output = []
        for transform in values:
            if (transform.header.frame_id == self.module.ODOMETRY_FRAME_ID and
                    transform.child_frame_id == self.module.VEHICLE_FRAME_ID):
                odometry = self.bridge._experiment_nav_odom
                if (odometry is None
                        or odometry.header.stamp != transform.header.stamp):
                    raise RuntimeError('experiment odometry/TF stamp mismatch')
                transform = copy.deepcopy(transform)
                position = odometry.pose.pose.position
                transform.transform.translation.x = position.x
                transform.transform.translation.y = position.y
                transform.transform.translation.z = position.z
                transform.transform.rotation = copy.deepcopy(
                    odometry.pose.pose.orientation)
            output.append(transform)
        self.broadcaster.sendTransform(
            output if isinstance(message, list) else output[0])


class PlayerTap:

    def __init__(self, player, bridge, module):
        self.player, self.bridge, self.module = player, bridge, module

    def __getattr__(self, name):
        return getattr(self.player, name)

    def apply_control(self, control):
        bridge = self.bridge
        requested = float(control.steer)
        stopping = control.brake >= 0.99 or not bridge._experiment_armed
        elapsed = bridge._experiment_time_s - bridge._experiment_started_s
        steer = bridge._experiment_steering.apply(requested,
                                                  bridge._experiment_time_s,
                                                  elapsed,
                                                  stopping=stopping)
        applied = self.module.carla.VehicleControl(
            throttle=control.throttle,
            steer=steer,
            brake=control.brake,
            hand_brake=control.hand_brake,
            reverse=control.reverse,
            manual_gear_shift=control.manual_gear_shift,
            gear=control.gear)
        # Throttle and braking are bit-for-bit the original decision. Never
        # enqueue a Twist, brake, watchdog result or the entire control packet.
        bridge._experiment_steer_requested = requested
        bridge._experiment_steer_applied = steer
        bridge._experiment_brake = float(applied.brake)
        return self.player.apply_control(applied)


class WorldTap:

    def __init__(self, world, player):
        self.original, self.player = world, player

    def __getattr__(self, name):
        return getattr(self.original, name)


class RealtimeClock:
    """Do not drive a 20 Hz physics experiment at the client's 100 FPS cap."""

    def __init__(self, clock, maximum_fps):
        self.clock, self.maximum_fps = clock, maximum_fps

    def tick_busy_loop(self, requested_fps):
        return self.clock.tick_busy_loop(min(requested_fps, self.maximum_fps))

    def __getattr__(self, name):
        return getattr(self.clock, name)


def install_adapter(module, profile, source_sha):
    original_bridge = module.CarlaRos2Publisher
    original_control = module.KeyboardControl._apply_external_control

    class ExperimentBridge(original_bridge):

        def __init__(self, *args, **kwargs):
            self._experiment_armed = False
            self._experiment_run_id = ''
            self._experiment_time_s = 0.0
            self._experiment_started_s = 0.0
            self._experiment_route_yaw = 0.0
            self._experiment_nav_odom = None
            self._experiment_steering = SteeringFault(profile)
            self._experiment_steer_requested = 0.0
            self._experiment_steer_applied = 0.0
            self._experiment_brake = 1.0
            self._experiment_session = uuid.uuid4().hex
            sensor = getattr(module, 'SENSOR_PROFILE', None)
            self._experiment_sensor_profile = (asdict(sensor)
                                               if is_dataclass(sensor) else {})
            super().__init__(*args, **kwargs)
            if not self.enabled or not self.external_control_enabled:
                raise RuntimeError(
                    'experiment requires the ROS external-control bridge')
            from std_msgs.msg import String
            self._experiment_truth_publisher = self.node.create_publisher(
                module.Odometry, '/experiment/truth_odometry', 10)
            self._experiment_status_publisher = self.node.create_publisher(
                String, '/experiment/bridge_status', 10)
            self._experiment_subscription = self.node.create_subscription(
                String, '/experiment/control', self._experiment_control, 10)
            self.odometry_publisher = PublisherTap(self.odometry_publisher,
                                                   self._experiment_odometry)
            self.tf_broadcaster = TfTap(self.tf_broadcaster, self, module)

        def _experiment_control(self, message):
            try:
                data = json.loads(message.data)
                if data.get('event') == 'stop':
                    if data.get('run_id') == self._experiment_run_id:
                        self._experiment_armed = False
                        self._experiment_steering.reset()
                    return
                if data.get('event') != 'start' or self._experiment_armed:
                    return
                if data.get(
                        'profile') != profile.name or not data.get('run_id'):
                    raise ValueError('profile/run ID mismatch')
                yaw = float(data['route_yaw_rad'])
                if not math.isfinite(yaw):
                    raise ValueError('invalid route yaw')
                self._experiment_route_yaw = yaw
                self._experiment_run_id = str(data['run_id'])
                self._experiment_started_s = self._experiment_time_s
                self._experiment_steering.reset()
                self._experiment_armed = True
            except (ValueError, KeyError, TypeError) as error:
                self._experiment_armed = False
                self.node.get_logger().error(
                    f'experiment control rejected: {error}')

        def external_command(self):
            # Explicit start is required; after stop, residual Nav2 commands
            # cannot make the car resume. Original stale-command brake remains.
            if not self._experiment_armed:
                return None
            return super().external_command()

        def _experiment_odometry(self, message):
            truth = copy.deepcopy(message)
            truth.header.frame_id = 'experiment_truth_world'
            truth.child_frame_id = 'experiment_truth_vehicle'
            self._experiment_truth_publisher.publish(truth)
            nav = copy.deepcopy(message)
            dx = dy = yaw_bias = 0.0
            elapsed = self._experiment_time_s - self._experiment_started_s
            if self._experiment_armed:
                dx, dy, yaw_bias = position_fault(profile, elapsed,
                                                  self._experiment_route_yaw)
                nav.pose.pose.position.x += dx
                nav.pose.pose.position.y += dy
                q = nav.pose.pose.orientation
                q.x, q.y, q.z, q.w = rotate_quaternion_yaw(
                    (q.x, q.y, q.z, q.w), yaw_bias)
            self._experiment_nav_odom = nav
            from std_msgs.msg import String
            self._experiment_status_publisher.publish(
                String(data=json.dumps({
                    'schema_version': 1,
                    'adapter_session': self._experiment_session,
                    'profile': profile.name,
                    'profile_parameters': profile.to_dict(),
                    'manual_client_sha256': source_sha,
                    'sensor_profile': self._experiment_sensor_profile,
                    'armed': self._experiment_armed,
                    'run_id': self._experiment_run_id,
                    'sim_time_s': self._experiment_time_s,
                    'elapsed_s': elapsed if self._experiment_armed else 0.0,
                    'position_bias_x_m': dx,
                    'position_bias_y_m': dy,
                    'yaw_bias_deg': math.degrees(yaw_bias),
                    'steer_requested': self._experiment_steer_requested,
                    'steer_applied': self._experiment_steer_applied,
                    'brake': self._experiment_brake,
                    'navigation_pose_source': (
                        'synthetic_oracle_with_optional_bias'),
                    'truth_has_navigation_tf': False,
                })))
            return nav

        def publish_vehicle_speed(self, vehicle, timestamp):
            self._experiment_time_s = float(timestamp)
            super().publish_vehicle_speed(vehicle, timestamp)

    def experiment_control(controller, world, clock):
        tapped = WorldTap(world,
                          PlayerTap(world.player, world.sensor_ros, module))
        return original_control(controller, tapped, clock)

    module.CarlaRos2Publisher = ExperimentBridge
    module.KeyboardControl._apply_external_control = experiment_control
    original_world = getattr(module, 'World', None)
    if original_world is not None:

        class ExperimentWorld(original_world):

            def _load_navigation_goal(self):
                self.navigation_goal_path = str(
                    Path.home() /
                    'terrain_nav_data/experiments/ui/saved_goal.json')
                return super()._load_navigation_goal()

            def _load_waypoint_route(self):
                self.waypoint_route_path = str(
                    Path.home() /
                    'terrain_nav_data/experiments/ui/waypoints.json')
                return super()._load_waypoint_route()

        module.World = ExperimentWorld


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carla-root',
                        default=str(Path.home() / 'carla/0.10.0'))
    parser.add_argument('--profile',
                        choices=sorted(PROFILES),
                        default='baseline')
    parser.add_argument('--anchor', help='previously captured anchor JSON')
    parser.add_argument('--auto-straight-start', action='store_true')
    parser.add_argument('--start-candidate', type=int, default=1)
    parser.add_argument('--inspect-only', action='store_true')
    options, remaining = parser.parse_known_args(argv)
    if '--' in remaining:
        remaining.remove('--')
    root = Path(options.carla_root).expanduser()
    client_file = root / 'PythonAPI/examples/manual_control.py'
    source_sha = inspect_client(client_file)
    if options.inspect_only:
        print(
            json.dumps(
                {
                    'compatible': True,
                    'manual_client_sha256': source_sha,
                    'profile': PROFILES[options.profile].to_dict()
                },
                indent=2))
        return 0
    if os.environ.get('ROS_DOMAIN_ID') != '113':
        parser.error(
            'use scripts/experiment.sh (isolated experiment domain 113)')
    if '--autopilot' in remaining or '-a' in remaining:
        parser.error('autopilot bypasses the experiment/safety contract')

    def has_option(name):
        return any(arg == name or arg.startswith(name + '=')
                   for arg in remaining)

    if options.anchor:
        anchor = validate_anchor(
            json.loads(Path(options.anchor).expanduser().read_text()))
        if has_option('--spawn-point') or options.auto_straight_start:
            parser.error('--anchor cannot be combined with --spawn-point')
        if has_option('--filter'):
            parser.error(
                '--anchor selects its recorded blueprint; omit --filter')
        remaining += ['--filter', anchor['vehicle_blueprint']]
        start = anchor['start_carla']
        remaining.append(
            '--spawn-point=' + ','.join(
                str(value) for value in (start['x'], start['y'],
                                         start['z'] + 0.25, start['yaw'])))
    if not has_option('--external-control'):
        remaining.append('--external-control')
    if not has_option('--sync'):
        remaining.append('--sync')
    if not has_option('--fixed-delta-seconds'):
        remaining += ['--fixed-delta-seconds', '0.05']
    if not has_option('--filter'):
        remaining += ['--filter', 'vehicle.lincoln.mkz*']
    sys.path.insert(0, str(client_file.parent))
    from .controlled_carla_obstacle import import_carla
    carla = import_carla(root)
    if options.auto_straight_start:
        if has_option('--spawn-point'):
            parser.error('auto start cannot be combined with --spawn-point')
        from .simulation_experiment_scenario import straight_start_candidates
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10)
        candidates = straight_start_candidates(carla, client.get_world())
        if not 1 <= options.start_candidate <= len(candidates):
            parser.error(
                'start-candidate is outside the available candidate list')
        transform = candidates[options.start_candidate - 1]
        values = (transform.location.x, transform.location.y,
                  transform.location.z, transform.rotation.yaw)
        # argparse treats a separate negative CSV as another option.
        remaining.append('--spawn-point=' + ','.join(map(str, values)))
        print('Straight-road candidate:',
              values,
              '(inspect scene before capture/run)',
              flush=True)
    spec = importlib.util.spec_from_file_location('experiment_carla_client',
                                                  client_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_adapter(module, PROFILES[options.profile], source_sha)
    original_clock = module.pygame.time.Clock
    module.pygame.time.Clock = lambda: RealtimeClock(original_clock(), 20)
    sys.argv = [str(client_file)] + remaining
    module.main()
    return 0
