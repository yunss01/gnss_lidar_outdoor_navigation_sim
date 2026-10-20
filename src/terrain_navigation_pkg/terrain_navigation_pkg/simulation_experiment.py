"""Prepare, record and report isolated simulation sensitivity trials."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from .simulation_experiment_core import (
    evaluate_trial,
    polygon_clearance,
    PROFILES,
    quaternion_yaw,
    rectangle,
    route_coordinates,
    SCENARIOS,
    wrap_angle,
)
from .simulation_experiment_scenario import (
    capture,
    cleanup,
    connect,
    find_start,
    fingerprint,
    prepare,
    read_anchor,
    validate_prepared,
    write_json,
)

FRAME_FIELDS = [
    'sim_time_s',
    'elapsed_s',
    'truth_x',
    'truth_y',
    'truth_yaw_rad',
    'nav_x',
    'nav_y',
    'nav_yaw_rad',
    'pose_bias_m',
    'yaw_bias_deg',
    'along_track_m',
    'cross_track_m',
    'heading_error_deg',
    'goal_error_m',
    'speed_mps',
    'command_speed_mps',
    'command_yaw_rate_rps',
    'steer_requested',
    'steer_applied',
    'brake',
    'bbox_clearance_m',
    'corridor_violation',
    'collision_count',
    'safety_state',
    'path_gate_state',
    'route_status',
    'tf_consistent',
    'lidar_age_s',
    'bridge_status_age_s',
    'command_age_s',
    'safety_state_age_s',
    'path_gate_age_s',
]


def _stamp(message):
    return (message.header.stamp.sec * 1_000_000_000 +
            message.header.stamp.nanosec)


def _yaw(message):
    q = message.pose.pose.orientation
    return quaternion_yaw((q.x, q.y, q.z, q.w))


def navigation_fingerprint():
    """Include uncommitted source/config changes, not just a git commit ID."""
    workspace = Path(__file__).resolve().parents[3]
    directories = (
        workspace / 'src/terrain_navigation_pkg/terrain_navigation_pkg',
        workspace / 'src/config_pkg/config',
        workspace / 'src/launch_pkg/launch',
    )
    digest = hashlib.sha256()
    count = 0
    for directory in directories:
        for path in sorted(directory.rglob('*')):
            if path.suffix not in ('.py', '.yaml',
                                   '.xml') or not path.is_file():
                continue
            digest.update(str(path.relative_to(workspace)).encode())
            digest.update(path.read_bytes())
            count += 1
    if not count:
        raise RuntimeError('cannot fingerprint navigation source/config files')
    return digest.hexdigest()


def _make_node(anchor, state):
    import rclpy
    from geometry_msgs.msg import PointStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import NavSatFix, PointCloud2
    from std_msgs.msg import Float32, String
    from tf2_ros import Buffer, TransformListener

    class TrialNode(Node):

        def __init__(self):
            super().__init__('simulation_experiment_trial',
                             parameter_overrides=[
                                 Parameter('use_sim_time', Parameter.Type.BOOL,
                                           True)
                             ])
            self.bridge = None
            self.bridge_wall = 0.0
            self.lidar_wall = 0.0
            self.truth = {}
            self.nav = {}
            self.last_logged = -1
            self.frames = []
            self.recording = False
            self.writer = None
            self.origin = None
            self.current_local = None
            self.route_status = ''
            self.safety_state = ''
            self.path_state = ''
            self.command = Twist()
            self.command_wall = 0.0
            self.safety_state_wall = 0.0
            self.path_state_wall = 0.0
            self.collision_count = 0
            self.start_time_s = 0.0
            latched = QoSProfile(depth=1,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Odometry, '/experiment/truth_odometry',
                                     lambda msg: self._cache(self.truth, msg),
                                     10)
            self.create_subscription(Odometry, '/vehicle/odometry',
                                     lambda msg: self._cache(self.nav, msg),
                                     10)
            self.create_subscription(String, '/experiment/bridge_status',
                                     self._bridge, 10)
            self.create_subscription(PointCloud2, '/lidar/points', self._lidar,
                                     qos_profile_sensor_data)
            self.create_subscription(
                String, '/navigation/route_status',
                lambda msg: setattr(self, 'route_status', msg.data), 10)
            self.create_subscription(
                String, '/safety/state',
                lambda msg: self._remember('safety_state', msg.data), 10)
            self.create_subscription(
                String, '/navigation/path_validity_gate/status',
                lambda msg: self._remember('path_state', msg.data), latched)
            self.create_subscription(
                Twist, '/cmd_vel', lambda msg: self._remember('command', msg),
                10)
            self.create_subscription(Float32, '/vehicle/collision',
                                     self._collision, 10)
            self.create_subscription(NavSatFix, '/navigation/origin_gnss',
                                     lambda msg: setattr(self, 'origin', msg),
                                     latched)
            self.create_subscription(
                PointStamped, '/navigation/current_local',
                lambda msg: setattr(self, 'current_local', msg), 10)
            self.control_publisher = self.create_publisher(
                String, '/experiment/control', 10)
            self.route_publisher = self.create_publisher(
                String, '/navigation/waypoint_route', latched)
            self.buffer = Buffer(node=self)
            self.listener = TransformListener(self.buffer, self)
            self.create_timer(0.05, self._collect)

        def _remember(self, name, value):
            setattr(self, name, value)
            setattr(self, name + '_wall', time.monotonic())

        def _cache(self, cache, message):
            cache[_stamp(message)] = message
            while len(cache) > 64:
                del cache[min(cache)]

        def _bridge(self, message):
            self.bridge = json.loads(message.data)
            self.bridge_wall = time.monotonic()

        def _lidar(self, _message):
            self.lidar_wall = time.monotonic()

        def _collision(self, _message):
            if self.recording:
                self.collision_count += 1

        def tf_matches(self, nav):
            try:
                tf = self.buffer.lookup_transform(
                    'odom', 'base_link',
                    rclpy.time.Time.from_msg(nav.header.stamp))
            except Exception:
                return None
            position = nav.pose.pose.position
            translation = tf.transform.translation
            q = nav.pose.pose.orientation
            rotation = tf.transform.rotation
            delta = min(
                sum((a - b)**2
                    for a, b in zip((q.x, q.y, q.z,
                                     q.w), (rotation.x, rotation.y, rotation.z,
                                            rotation.w))),
                sum((a + b)**2
                    for a, b in zip((q.x, q.y, q.z,
                                     q.w), (rotation.x, rotation.y, rotation.z,
                                            rotation.w))))
            return (math.hypot(position.x - translation.x,
                               position.y - translation.y) < 1e-5
                    and delta < 1e-10)

        def publish_event(self, event, run_id, profile):
            message = String(data=json.dumps({
                'event': event,
                'run_id': run_id,
                'profile': profile,
                'route_yaw_rad': anchor['start_ros']['yaw_rad'],
            }))
            self.control_publisher.publish(message)

        def _collect(self):
            if not self.recording or self.bridge is None:
                return
            for key in sorted(set(self.truth) & set(self.nav)):
                if key <= self.last_logged or key / 1e9 < self.start_time_s:
                    continue
                truth, nav = self.truth[key], self.nav[key]
                match = self.tf_matches(nav)
                if match is None and (self.get_clock().now().nanoseconds -
                                      key) < 500_000_000:
                    continue
                p = truth.pose.pose.position
                nav_p = nav.pose.pose.position
                yaw = _yaw(truth)
                start = anchor['start_ros']
                along, cross = route_coordinates(p.x, p.y, start['x'],
                                                 start['y'], start['yaw_rad'])
                ego = anchor['ego_box']
                center_x = (p.x + math.cos(yaw) * ego['offset_forward_m'] -
                            math.sin(yaw) * ego['offset_left_m'])
                center_y = (p.y + math.sin(yaw) * ego['offset_forward_m'] +
                            math.cos(yaw) * ego['offset_left_m'])
                polygon = rectangle(center_x, center_y,
                                    yaw + ego['yaw_offset_rad'],
                                    ego['half_length_m'], ego['half_width_m'])
                distances = [
                    polygon_clearance(polygon, item['polygon_ros'])
                    for item in state['obstacles']
                ]
                corridor_violation = any(
                    abs(
                        route_coordinates(x, y, start['x'], start['y'],
                                          start['yaw_rad'])[1]) >
                    anchor['corridor_half_width_m'] for x, y in polygon)
                velocity = truth.twist.twist.linear
                bridge_age = abs(key / 1e9 - float(self.bridge['sim_time_s']))
                row = {
                    'sim_time_s': key / 1e9,
                    'elapsed_s': key / 1e9 - self.start_time_s,
                    'truth_x': p.x,
                    'truth_y': p.y,
                    'truth_yaw_rad': yaw,
                    'nav_x': nav_p.x,
                    'nav_y': nav_p.y,
                    'nav_yaw_rad': _yaw(nav),
                    'pose_bias_m': math.hypot(nav_p.x - p.x, nav_p.y - p.y),
                    'yaw_bias_deg': math.degrees(wrap_angle(_yaw(nav) - yaw)),
                    'along_track_m': along,
                    'cross_track_m': cross,
                    'heading_error_deg': math.degrees(
                        wrap_angle(yaw - start['yaw_rad'])),
                    'goal_error_m': math.hypot(
                        along - anchor['route_length_m'], cross),
                    'speed_mps': math.sqrt(velocity.x**2 + velocity.y**2 +
                                           velocity.z**2),
                    'command_speed_mps': self.command.linear.x,
                    'command_yaw_rate_rps': self.command.angular.z,
                    'steer_requested': self.bridge['steer_requested'],
                    'steer_applied': self.bridge['steer_applied'],
                    'brake': self.bridge['brake'],
                    'bbox_clearance_m': min(distances) if distances else '',
                    'corridor_violation': int(corridor_violation),
                    'collision_count': self.collision_count,
                    'safety_state': self.safety_state,
                    'path_gate_state': self.path_state,
                    'route_status': self.route_status,
                    'tf_consistent': int(bool(match)),
                    'lidar_age_s': time.monotonic() - self.lidar_wall,
                    'bridge_status_age_s': bridge_age,
                    'command_age_s': time.monotonic() - self.command_wall,
                    'safety_state_age_s': (time.monotonic() -
                                           self.safety_state_wall),
                    # Event age: the gate publishes only when state changes.
                    'path_gate_age_s': (
                        time.monotonic() - self.path_state_wall
                        if self.path_state_wall else ''),
                }
                self.frames.append(row)
                if self.writer is not None:
                    self.writer.writerow(row)
                self.last_logged = key

    return TrialNode()


def _preflight_missing(node, options):
    now = time.monotonic()
    bridge = node.bridge or {}
    checks = (
        ('experiment_bridge', bool(node.bridge)),
        ('matching_profile', bridge.get('profile') == options.profile),
        ('unarmed_client', bool(node.bridge) and not bridge.get('armed')),
        ('gnss_origin', node.origin is not None),
        ('truth_odometry', bool(node.truth)),
        ('navigation_odometry', bool(node.nav)),
        ('safety_state', bool(node.safety_state)),
        ('fresh_lidar', now - node.lidar_wall < 0.5),
        ('fresh_bridge', now - node.bridge_wall < 0.5),
    )
    # current_local is published only after a goal arrives. The route is
    # submitted AFTER preflight, so requiring it here would deadlock startup.
    return [name for name, ready in checks if not ready]


def _preflight(node, options, rclpy):
    from lifecycle_msgs.srv import GetState
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if not _preflight_missing(node, options):
            break
    else:
        raise RuntimeError(
            'preflight: need matching experiment client, fresh LiDAR, '
            'GNSS origin and navigation stack; missing=' +
            ','.join(_preflight_missing(node, options)) + '; do not press F9')
    for topic in ('/vehicle/odometry', '/experiment/truth_odometry',
                  '/cmd_vel', '/tf'):
        if node.count_publishers(topic) != 1:
            raise RuntimeError(
                f'preflight: {topic} must have exactly one publisher')
    if node.bridge.get('truth_has_navigation_tf') is not False:
        raise RuntimeError('ground truth must not own navigation TF')
    if node.safety_state == 'collision_stop':
        raise RuntimeError(
            'collision latch is active; restart the trial stack')
    if node.tf_matches(node.nav[max(node.nav)]) is not True:
        # Allow the matching TF packet to arrive after the odometry packet.
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.tf_matches(node.nav[max(node.nav)]) is True:
                break
        else:
            raise RuntimeError('navigation odometry and TF are inconsistent')
    for name in ('controller_server', 'planner_server', 'bt_navigator'):
        client = node.create_client(GetState, f'/{name}/get_state')
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError(f'preflight: {name} unavailable')
        future = client.call_async(GetState.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
        if not future.done() or future.result().current_state.id != 3:
            raise RuntimeError(f'preflight: {name} is not ACTIVE')
        node.destroy_client(client)


def run_trial(options):
    import rclpy
    from std_msgs.msg import String
    anchor = read_anchor(options.anchor)
    _carla, world = connect(options)
    state = validate_prepared(world, anchor, options.anchor, options.scenario)
    settings = world.get_settings()
    if (not settings.synchronous_mode or settings.fixed_delta_seconds is None
            or abs(settings.fixed_delta_seconds - 0.05) > 1e-8):
        raise RuntimeError('experiment requires synchronous 0.05 s physics')
    run_id = datetime.now().strftime(
        'run_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
    output = Path(options.output_root).expanduser() / run_id
    output.mkdir(parents=True, exist_ok=False)
    start = anchor['start_ros']
    obstacle_ends = [
        route_coordinates(x, y, start['x'], start['y'], start['yaw_rad'])[0]
        for item in state['obstacles'] for x, y in item['polygon_ros']
    ]
    metadata = {
        'schema_version': 1,
        'run_id': run_id,
        'scenario': options.scenario,
        'profile': options.profile,
        'profile_parameters': PROFILES[options.profile].to_dict(),
        'anchor_sha256': fingerprint(anchor),
        'navigation_code_sha256': navigation_fingerprint(),
        'anchor': anchor,
        'prepared': state,
        'termination': 'preflight',
        'obstacle_passage_s_m': (max(obstacle_ends) +
                                 anchor['ego_box']['half_length_m'] +
                                 1.0 if obstacle_ends else None),
        'created_at': datetime.now().astimezone().isoformat(),
        'maximum_sim_time_s': options.max_sim_seconds,
        'navigation_pose_source': 'synthetic_oracle_with_optional_bias',
        'not_sensor_localization_validation': True,
    }
    write_json(output / 'metadata.json', metadata, exclusive=True)
    print(f'Trial output: {output}', flush=True)
    rclpy.init()
    node = _make_node(anchor, state)
    started = False
    try:
        with (output / 'frames.csv').open('x', newline='') as stream:
            node.writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
            node.writer.writeheader()
            _preflight(node, options, rclpy)
            metadata['adapter_session'] = node.bridge['adapter_session']
            metadata['manual_client_sha256'] = node.bridge[
                'manual_client_sha256']
            metadata['sensor_profile'] = node.bridge.get('sensor_profile', {})
            metadata['initial_gnss_origin'] = {
                'latitude': node.origin.latitude,
                'longitude': node.origin.longitude,
                'altitude': node.origin.altitude,
            }
            node.publish_event('start', run_id, options.profile)
            started = True
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.bridge.get('armed') and node.bridge.get(
                        'run_id') == run_id:
                    break
            else:
                raise RuntimeError(
                    'experiment bridge did not acknowledge start')
            node.start_time_s = float(node.bridge['sim_time_s'])
            node.recording = True
            node.route_publisher.publish(
                String(data=json.dumps({
                    'waypoints': anchor['waypoints'],
                    'loop': False,
                    'start': True,
                    'rolling_horizon': True,
                })))
            metadata['termination'] = 'running'
            write_json(output / 'metadata.json', metadata)
            wall_deadline = time.monotonic() + options.max_wall_seconds
            last_report = -1.0
            goal_stop_since = None
            while time.monotonic() < wall_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if time.monotonic() - node.bridge_wall > 2:
                    metadata['termination'] = 'bridge_stale'
                    break
                if not node.frames:
                    continue
                row = node.frames[-1]
                elapsed = row['elapsed_s']
                if node.collision_count:
                    metadata['termination'] = 'collision'
                    break
                if row['corridor_violation'] or row['bbox_clearance_m'] == 0.0:
                    metadata['termination'] = ('corridor_violation'
                                               if row['corridor_violation']
                                               else 'bbox_overlap')
                    break
                if row['goal_error_m'] <= 3 and row['speed_mps'] <= 0.05:
                    if goal_stop_since is None:
                        goal_stop_since = row['sim_time_s']
                    if row['sim_time_s'] - goal_stop_since >= 2:
                        metadata['termination'] = 'goal_stopped'
                        break
                else:
                    goal_stop_since = None
                if elapsed - last_report >= 1:
                    last_report = elapsed
                    stream.flush()
                    snapshot = evaluate_trial(metadata, node.frames)
                    print(
                        f"{elapsed:6.1f}s s={row['along_track_m']:5.1f}m "
                        f"lateral={row['cross_track_m']:+.2f}m "
                        f"speed={row['speed_mps']:.2f}m/s "
                        f"safety={row['safety_state']} "
                        f"recovered={snapshot['recovered']}",
                        flush=True)
                    if options.scenario == 'S3' and snapshot[
                            'explained_blocked_stop_s'] >= 10:
                        metadata['termination'] = 'blocked_stopped'
                        break
                if elapsed >= options.max_sim_seconds:
                    metadata['termination'] = 'simulation_timeout'
                    break
            else:
                metadata['termination'] = 'wall_timeout'
    except KeyboardInterrupt:
        metadata['termination'] = 'interrupted'
    except Exception as error:
        metadata['error'] = str(error)
        metadata[
            'termination'] = 'runtime_error' if started else 'preflight_failed'
        print(f'ERROR: {error}', flush=True)
    finally:
        node.recording = False
        if started:
            # Halt the guarded client, not /cmd_vel. Original safety remains
            # the sole final Twist publisher and braking is not delayed.
            for _ in range(30):
                node.publish_event('stop', run_id, options.profile)
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.bridge and not node.bridge.get('armed'):
                    break
            metadata['halt_acknowledged'] = bool(
                node.bridge and not node.bridge.get('armed'))
            stop_deadline = time.monotonic() + 5
            metadata['vehicle_stopped_after_halt'] = False
            while time.monotonic() < stop_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.truth:
                    velocity = node.truth[max(node.truth)].twist.twist.linear
                    if math.hypot(velocity.x, velocity.y) <= 0.05:
                        metadata['vehicle_stopped_after_halt'] = True
                        break
            if not (metadata['halt_acknowledged']
                    and metadata['vehicle_stopped_after_halt']):
                print(
                    'WARNING: halt/stationary confirmation missing; '
                    'close the client.',
                    flush=True)
        metadata['ended_at'] = datetime.now().astimezone().isoformat()
        summary = evaluate_trial(metadata, node.frames)
        write_json(output / 'metadata.json', metadata)
        write_json(output / 'summary.json', summary, exclusive=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if summary['success'] else 1


def report(options):
    root = Path(options.output_root).expanduser()
    summaries = []
    for path in sorted(root.glob('run_*/summary.json')):
        result = json.loads(path.read_text())
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        result['anchor_sha256'] = metadata['anchor_sha256']
        result['manual_client_sha256'] = metadata.get('manual_client_sha256',
                                                      '')
        result['navigation_code_sha256'] = metadata.get(
            'navigation_code_sha256', '')
        summaries.append(result)
    if not summaries:
        print(
            'No experiment results. Existing F9/F10 registry is not consulted.'
        )
        return
    fields = [
        'run_id', 'anchor_sha256', 'manual_client_sha256',
        'navigation_code_sha256', 'scenario', 'profile', 'verdict',
        'termination', 'metrics_valid', 'success', 'collision',
        'expected_stop', 'recovered', 'recovery_time_s', 'final_goal_error_m',
        'cross_track_p95_m', 'maximum_stop_s', 'minimum_bbox_clearance_m',
        'duration_sim_s'
    ]
    destination = root / ('report_' + datetime.now().strftime('%Y%m%d_%H%M%S_')
                          + uuid.uuid4().hex[:6] + '.csv')
    with destination.open('x', newline='') as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    print(f'All {len(summaries)} attempts (including failures): {destination}')
    groups = {}
    for row in summaries:
        key = (row['anchor_sha256'], row['manual_client_sha256'],
               row['navigation_code_sha256'], row['scenario'], row['profile'])
        groups.setdefault(key, []).append(row)
    for key, rows in sorted(groups.items()):
        valid = sum(bool(row['metrics_valid']) for row in rows)
        passed = sum(bool(row['success']) for row in rows)
        print(
            f'{key[0][:8]} code={key[2][:8]} {key[3]} {key[4]}: '
            f'{passed}/{len(rows)} pass; {valid}/{len(rows)} valid recordings')


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carla-root',
                        default=str(Path.home() / 'carla/0.10.0'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('find-start')
    for name in ('capture', 'prepare', 'cleanup', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--anchor', required=True)
        if name == 'capture':
            command.add_argument('--route-length-m', type=float, default=40.0)
            command.add_argument('--obstacle-distance-m',
                                 type=float,
                                 default=20.0)
            command.add_argument('--corridor-half-width-m',
                                 type=float,
                                 default=7.0)
        if name in ('prepare', 'run'):
            command.add_argument('--scenario',
                                 choices=SCENARIOS,
                                 required=True)
        if name == 'run':
            command.add_argument('--profile',
                                 choices=sorted(PROFILES),
                                 default='baseline')
            command.add_argument('--output-root',
                                 default='~/terrain_nav_data/experiments/runs')
            command.add_argument('--max-sim-seconds',
                                 type=float,
                                 default=120.0)
            command.add_argument('--max-wall-seconds',
                                 type=float,
                                 default=900.0)
    summary = commands.add_parser('report')
    summary.add_argument('--output-root',
                         default='~/terrain_nav_data/experiments/runs')
    return parser


def main(argv=None):
    parser = _parser()
    options = parser.parse_args(argv)
    if options.command == 'run' and os.environ.get('ROS_DOMAIN_ID') != '113':
        parser.error('run through scripts/experiment.sh (ROS domain 113)')
    if options.command == 'run' and (
            not math.isfinite(options.max_sim_seconds)
            or options.max_sim_seconds <= 0
            or not math.isfinite(options.max_wall_seconds)
            or options.max_wall_seconds <= 0):
        parser.error('timeouts must be finite and positive')
    try:
        if options.command == 'run':
            return run_trial(options)
        {
            'find-start': find_start,
            'capture': capture,
            'prepare': prepare,
            'cleanup': cleanup,
            'report': report
        }[options.command](options)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
