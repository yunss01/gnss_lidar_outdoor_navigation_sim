"""Read-only path/vehicle diagnostics for the isolated CARLA experiment.

Never a controller input. /plan is only a candidate; Humble RPP publishes
its pruned, base-frame computation window on /received_global_plan.
Source: navigation2/humble, regulated_pure_pursuit_controller.cpp,
transformGlobalPlan(). A computation window does NOT prove actuation.
"""

from collections import Counter
from datetime import datetime
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from .simulation_experiment_core import (
    finite,
    polygon_clearance,
    quaternion_yaw,
    rectangle,
    wrap_angle,
)

TRACE_VERSION = 2
TRACE_FIELDS = (
    'sim_time_s',
    'elapsed_s',
    'obstacle_id',
    'actual_bbox_clearance_m',
    'critical_zone',
    'observation',
    'goal_context',
    'controller_goal_status',
    'active_goal_count',
    'controller_path_id',
    'candidate_at_receipt',
    'controller_goal_id',
    'path_age_s',
    'projection_x',
    'projection_y',
    'projection_yaw_rad',
    'path_tangent_rad',
    'tracking_cross_m',
    'tracking_heading_error_deg',
    'projection_distance_m',
    'projection_region',
    'projection_arc_m',
    'projection_longitudinal_offset_m',
    'projected_planned_clearance_m',
    'projected_clearance_difference_m',
    'same_station_observation',
    'same_station_planned_clearance_m',
    'same_station_clearance_loss_m',
    'speed_mps',
    'speed_derivative_mps2',
    'accepted_command_speed_mps',
    'accepted_command_curvature_per_m',
    'steer_applied',
    'steer_saturated',
    'brake',
)


def new_log_session(root):
    """Do not name a launch's logs using a not-yet-received /clock (zero)."""
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    path = Path(root).expanduser() / ('session_' + stamp + '_' +
                                      uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    return path


def body_polygon(x, y, yaw, box):
    """Same physical CARLA OBB for planned and executed poses, no padding."""
    cx = x + math.cos(yaw) * box['offset_forward_m'] - math.sin(
        yaw) * box['offset_left_m']
    cy = y + math.sin(yaw) * box['offset_forward_m'] + math.cos(
        yaw) * box['offset_left_m']
    return rectangle(cx, cy, yaw + box['yaw_offset_rad'], box['half_length_m'],
                     box['half_width_m'])


def project_controller_window(poses, x, y, prefix_m=3.0):
    """Project only onto the already-pruned controller window's near prefix.

    No matching against an arbitrary distant future branch or a new /plan.
    Nonadjacent, near-equal crossing matches are explicitly ambiguous.
    Stored pose yaw determines body orientation; segment tangent determines
    tracking error. These are intentionally different measurements.
    """
    candidates = []
    cumulative = 0.0
    for i, (a, b) in enumerate(zip(poses, poses[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            continue
        if cumulative >= prefix_m:
            break
        raw_fraction = ((x - a[0]) * dx + (y - a[1]) * dy) / length**2
        upper_fraction = min(1.0, (prefix_m - cumulative) / length)
        fraction = max(
            0.0, min(upper_fraction, raw_fraction))
        px, py = a[0] + fraction * dx, a[1] + fraction * dy
        tangent = math.atan2(dy, dx)
        yaw = wrap_angle(a[2] + fraction * wrap_angle(b[2] - a[2]))
        arc = cumulative + fraction * length
        longitudinal = ((x - px) * dx + (y - py) * dy) / length
        region = 'segment_interior'
        if arc <= 1e-8:
            region = 'window_start'
        elif abs(arc - prefix_m) <= 1e-8:
            region = 'prefix_end'
        elif fraction >= 1 - 1e-8 and i == len(poses) - 2:
            region = 'window_end'
        elif fraction <= 1e-8 or fraction >= 1 - 1e-8:
            region = 'path_vertex'
        comparison = 'supported'
        # A nonzero along-tangent residual means the perpendicular foot is
        # outside the retained segment. Do not extrapolate a pruned path or
        # call a displaced endpoint the vehicle's same-progress reference.
        if abs(longitudinal) > 1e-6:
            comparison = 'projection_clamped_' + region
        candidates.append(
            dict(index=i,
                 distance=math.hypot(x - px, y - py),
                 x=px,
                 y=py,
                 yaw=yaw,
                 tangent=tangent,
                 cross=((x - px) * (-dy) + (y - py) * dx) / length,
                 arc=arc,
                 region=region,
                 longitudinal=longitudinal,
                 same_station_observation=comparison))
        cumulative += length
    if not candidates:
        return None, 'degenerate_controller_path'
    candidates.sort(key=lambda c: c['distance'])
    best = candidates[0]
    if any(
            abs(c['index'] - best['index']) > 1 and abs(c['arc'] - best['arc'])
            > 1.0 and c['distance'] - best['distance'] < .05
            for c in candidates[1:]):
        return None, 'ambiguous_path_branch'
    return best, 'supported'


def _goal_context(goals, goal_id, speed):
    """Describe received action state, never infer acceptance from a new plan.

    Terminal/canceling goals with a moving vehicle are NOT valid references
    for tracking, even when braking has begun. Keep these samples in coverage.
    """
    active = [g['goal_id'] for g in goals if g['status'] == 2]
    matched = next((g['status'] for g in goals
                    if g['goal_id'] == goal_id), None)
    if not goals:
        context = 'no_status_observed'
    elif len(active) > 1:
        context = 'overlapping_executing_goals'
    elif active:
        context = ('active_goal_matched' if active == [goal_id] else
                   'active_goal_changed' if goal_id else
                   'active_goal_without_linked_window')
    elif matched in (3, 4, 5, 6):
        terminal = {
            3: 'canceling', 4: 'succeeded', 5: 'canceled', 6: 'aborted'}
        motion = ('moving' if abs(speed) >= .2 else
                  'stationary' if abs(speed) <= .05 else 'low_speed')
        context = terminal[matched] + '_goal_' + motion
    else:
        context = 'no_executing_goal'
    return context, matched, active


def _plain_pose(pose):
    p, q = pose.position, pose.orientation
    values = [finite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)]
    if sum(v * v for v in values[3:]) < 1e-10:
        raise ValueError('zero quaternion')
    return values


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec / 1e9


def _transform_pose(values, transform):
    """Full quaternion TF, then planar projection (not yaw-only TF)."""
    x, y, z, qx, qy, qz, qw = values
    tx, ty, tz, rx, ry, rz, rw = transform
    norm = math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw)
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('invalid transform quaternion')
    rx, ry, rz, rw = (v / norm for v in (rx, ry, rz, rw))
    # Quaternion rotation of position.
    ux, uy, uz = 2 * (ry * z - rz * y), 2 * (rz * x - rx * z), 2 * (rx * y -
                                                                    ry * x)
    ox = tx + x + rw * ux + ry * uz - rz * uy
    oy = ty + y + rw * uy + rz * ux - rx * uz
    # TF rotation left-multiplies pose orientation.
    q = (rw * qx + rx * qw + ry * qz - rz * qy,
         rw * qy - rx * qz + ry * qw + rz * qx,
         rw * qz + rx * qy - ry * qx + rz * qw,
         rw * qw - rx * qx - ry * qy - rz * qz)
    qnorm = math.sqrt(sum(v * v for v in q))
    if qnorm < 1e-8:
        raise ValueError('invalid pose orientation')
    return [
        finite(ox),
        finite(oy),
        quaternion_yaw(tuple(v / qnorm for v in q))
    ]


class PathTraceObserver:
    """Passive subscriber + append-only snapshots; geometry runs after halt."""

    def __init__(self, node, anchor, state, output, run_id):
        from action_msgs.msg import GoalStatusArray
        from nav_msgs.msg import Path as PathMessage
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        self.node, self.anchor, self.state = node, anchor, state
        self.output, self.run_id = Path(output), run_id
        self.stream = (self.output / 'path_trace.jsonl').open('x',
                                                              encoding='utf-8')
        self.sequence = 0
        self.pending, self.windows = [], []
        self.events, self.follow_goals = [], []
        self.last_candidate = None
        self.invalid = Counter()
        self.closed = False
        self.subscriptions = [
            node.create_subscription(PathMessage, '/plan',
                                     lambda m: self.on_path(m, 'candidate'),
                                     10),
            node.create_subscription(PathMessage, '/received_global_plan',
                                     lambda m: self.on_path(m, 'controller'),
                                     10),
            node.create_subscription(
                GoalStatusArray, '/follow_path/_action/status', self.on_status,
                QoSProfile(depth=10,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            node.create_subscription(String,
                                     '/navigation/path_clearance/status',
                                     self.on_clearance, 10),
        ]
        self.emit('trace_opened',
                  topics=[
                      '/plan', '/received_global_plan',
                      '/follow_path/_action/status',
                      '/navigation/path_clearance/status'
                  ])

    def now(self):
        return self.node.get_clock().now().nanoseconds / 1e9

    def emit(self, event, **payload):
        self.sequence += 1
        record = dict(schema_version=TRACE_VERSION,
                      run_id=self.run_id,
                      sequence=self.sequence,
                      event=event,
                      received_sim_time_s=self.now(),
                      received_monotonic_s=time.monotonic(),
                      **payload)
        self.stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record

    def on_status(self, message):
        goals = [
            dict(goal_id=bytes(s.goal_info.goal_id.uuid).hex(),
                 status=int(s.status),
                 goal_stamp_s=s.goal_info.stamp.sec +
                 s.goal_info.stamp.nanosec / 1e9) for s in message.status_list
        ]
        if goals == self.follow_goals:
            return
        self.follow_goals = goals
        record = self.emit('follow_path_status', goals=goals)
        self.events.append(record)

    def on_clearance(self, message):
        try:
            payload = json.loads(message.data)
            self.emit('lidar_path_clearance', payload=payload)
        except (ValueError, TypeError):
            self.invalid['invalid_clearance_status'] += 1

    def on_path(self, message, source):
        try:
            stamp = finite(_stamp(message.header))
            frame = str(message.header.frame_id)
            if not frame or stamp <= 0:
                raise ValueError('missing_frame_or_stamp')
            if any(p.header.frame_id not in ('', frame)
                   for p in message.poses):
                raise ValueError('mixed_pose_frames')
            raw = [_plain_pose(p.pose) for p in message.poses]
            if len(raw) < 2:
                raise ValueError('empty_or_short_path')
        except (ValueError, TypeError) as error:
            self.invalid['invalid_' + source + '_path'] += 1
            record = self.emit('path_rejected',
                               source=source,
                               reason=str(error))
            if source == 'controller':
                # An invalid NEW observation must not silently reuse an old
                # valid path. This is diagnostic invalidation, not control.
                self.windows.append(
                    dict(record,
                         path_id='invalid_' + str(record['sequence']),
                         stamp_s=record['received_sim_time_s'],
                         observation_error='invalid_controller_window',
                         poses=[]))
            return
        geometry = hashlib.sha256(json.dumps([frame,
                                              raw]).encode()).hexdigest()
        path_id = source + '_' + str(self.sequence + 1)
        executing = [
            g['goal_id'] for g in self.follow_goals
            if g['status'] == 2 and g['goal_stamp_s'] <= stamp + 1e-6
        ]
        goal_id = executing[0] if len(executing) == 1 else None
        if source == 'candidate':
            self.last_candidate = path_id
        record = self.emit(
            'path_snapshot',
            source=source,
            path_id=path_id,
            geometry_sha256=geometry,
            frame_id=frame,
            stamp_s=stamp,
            raw_poses_xyz_xyzw=raw,
            controller_goal_id=goal_id,
            candidate_at_receipt=self.last_candidate,
            evidence=('rpp_computation_window_not_actuation' if source
                      == 'controller' else 'candidate_not_acceptance'))
        self.pending.append(record)

    def resolve(self, final=False):
        """Non-blocking exact-stamp TF; never fall back to 'latest' TF."""
        from rclpy.time import Time
        pending = []
        for record in self.pending:
            try:
                if record['frame_id'] == 'odom':
                    transform = [0, 0, 0, 0, 0, 0, 1]
                else:
                    stamp_ns = round(record['stamp_s'] * 1e9)
                    tf = self.node.buffer.lookup_transform(
                        'odom', record['frame_id'], Time(nanoseconds=stamp_ns))
                    t, q = tf.transform.translation, tf.transform.rotation
                    transform = [
                        finite(v) for v in (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
                    ]
                poses = [
                    _transform_pose(p, transform)
                    for p in record['raw_poses_xyz_xyzw']
                ]
            except Exception as error:
                if not final and self.now(
                ) - record['received_sim_time_s'] < .5:
                    pending.append(record)
                else:
                    self.invalid['transform_unavailable'] += 1
                    self.emit('path_transform_unavailable',
                              path_id=record['path_id'],
                              reason=str(error))
                    if record['source'] == 'controller':
                        self.windows.append(
                            dict(record,
                                 poses=[],
                                 observation_error='transform_unavailable'))
                continue
            self.emit('path_resolved',
                      path_id=record['path_id'],
                      odom_from_source_xyz_xyzw=transform,
                      odom_poses_xy_yaw=poses)
            if record['source'] == 'controller':
                self.windows.append(dict(record, poses=poses))
        self.pending = pending
        self.stream.flush()

    def finish(self, frames):
        self.resolve(final=True)
        summary = analyze_tracking(self.anchor, self.state, frames,
                                   self.windows, self.events, self.output)
        summary['snapshot_issues'] = dict(self.invalid)
        summary['run_id'] = self.run_id
        summary['raw_trace_file'] = 'path_trace.jsonl'
        summary['tracking_file'] = 'path_tracking.csv'
        with (self.output / 'path_trace_summary.json').open('x') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        self.close()
        return summary

    def close(self):
        if not self.closed:
            for subscription in self.subscriptions:
                self.node.destroy_subscription(subscription)
            self.stream.close()
            self.closed = True


def analyze_tracking(anchor, state, frames, windows, events, output):
    """Offline, time-causal matching; never choose a future candidate path."""
    windows = sorted(windows, key=lambda w: w['received_sim_time_s'])
    arrivals = [w['received_sim_time_s'] for w in windows]
    events = sorted(events, key=lambda e: e['received_sim_time_s'])
    event_times = [e['received_sim_time_s'] for e in events]
    obstacles = state['obstacles']
    rows = []
    reasons = Counter()
    comparison_reasons = Counter()
    contexts, unsupported_contexts = Counter(), Counter()
    previous = None
    for frame in frames:
        t = float(frame['sim_time_s'])
        x, y, yaw = (float(frame[k])
                     for k in ('truth_x', 'truth_y', 'truth_yaw_rad'))
        actual = body_polygon(x, y, yaw, anchor['ego_box'])
        distances = [
            polygon_clearance(actual, o['polygon_ros']) for o in obstacles
        ]
        obstacle_index = min(range(len(distances)),
                             key=distances.__getitem__) if distances else None
        clearance = distances[obstacle_index] if distances else None
        critical = bool(clearance is not None and clearance <= 3.0
                        and abs(float(frame['speed_mps'])) >= .2)
        row = {key: '' for key in TRACE_FIELDS}
        row.update(
            sim_time_s=t,
            elapsed_s=frame.get('elapsed_s', ''),
            obstacle_id=(obstacles[obstacle_index].get('actor_id', '')
                         if obstacle_index is not None else ''),
            actual_bbox_clearance_m=clearance if clearance is not None else '',
            critical_zone=int(critical),
            speed_mps=frame['speed_mps'],
            accepted_command_speed_mps=frame.get(
                'accepted_command_speed_mps', ''),
            steer_applied=frame.get('steer_applied', ''),
            brake=frame.get('brake', ''))
        speed = float(frame['speed_mps'])
        if previous is not None and 0 < t - previous[0] <= .25:
            row['speed_derivative_mps2'] = (
                speed - previous[1]) / (t - previous[0])
        previous = (t, speed)
        try:
            row['steer_saturated'] = int(
                abs(finite(frame['steer_applied'])) >= .98)
            v = finite(frame['accepted_command_speed_mps'])
            if v >= .2:
                row['accepted_command_curvature_per_m'] = finite(
                    frame['accepted_command_yaw_rate_rps']) / v
        except (KeyError, ValueError, TypeError):
            pass
        reason = 'no_controller_window'
        i = bisect.bisect_right(arrivals, t) - 1
        event_i = bisect.bisect_right(event_times, t) - 1
        goals = events[event_i]['goals'] if event_i >= 0 else []
        window = windows[i] if i >= 0 else None
        context, goal_status, active = _goal_context(
            goals, window.get('controller_goal_id') if window else None,
            speed)
        row.update(goal_context=context,
                   controller_goal_status=(goal_status
                                           if goal_status is not None else ''),
                   active_goal_count=len(active))
        if i >= 0:
            row.update(controller_path_id=window['path_id'],
                       candidate_at_receipt=window.get('candidate_at_receipt')
                       or '',
                       controller_goal_id=window.get('controller_goal_id')
                       or '',
                       path_age_s=t - window['stamp_s'])
            if window.get('observation_error'):
                reason = window['observation_error']
            elif not 0 <= t - window['stamp_s'] <= .25:
                reason = 'stale_or_future_controller_window'
            elif not window.get('controller_goal_id') or active != [
                    window['controller_goal_id']
            ]:
                reason = 'controller_goal_not_confirmed_active'
            elif not int(frame.get('tf_consistent', 0)):
                reason = 'navigation_tf_mismatch'
            elif (abs(float(frame.get('pose_bias_m', 0))) > 1e-5
                  or abs(float(frame.get('yaw_bias_deg', 0))) > 1e-5):
                reason = 'biased_navigation_frame'
            else:
                projection, reason = project_controller_window(
                    window['poses'], x, y)
                if projection is not None:
                    planned = body_polygon(projection['x'], projection['y'],
                                           projection['yaw'],
                                           anchor['ego_box'])
                    pc = (polygon_clearance(
                        planned, obstacles[obstacle_index]['polygon_ros'])
                          if obstacle_index is not None else None)
                    row.update(
                        projection_x=projection['x'],
                        projection_y=projection['y'],
                        projection_yaw_rad=projection['yaw'],
                        path_tangent_rad=projection['tangent'],
                        tracking_cross_m=projection['cross'],
                        tracking_heading_error_deg=math.degrees(
                            wrap_angle(yaw - projection['tangent'])),
                        projection_distance_m=projection['distance'],
                        projection_region=projection['region'],
                        projection_arc_m=projection['arc'],
                        projection_longitudinal_offset_m=(
                            projection['longitudinal']),
                        projected_planned_clearance_m=(
                            pc if pc is not None else ''),
                        projected_clearance_difference_m=(
                            pc - clearance if pc is not None else ''),
                        same_station_observation=(
                            projection['same_station_observation']))
                    if row['same_station_observation'] == 'supported':
                        row['same_station_planned_clearance_m'] = (
                            pc if pc is not None else '')
                        row['same_station_clearance_loss_m'] = (
                            pc - clearance if pc is not None else '')
        row['observation'] = reason
        if reason != 'supported':
            row['same_station_observation'] = reason
        if critical:
            reasons[reason] += 1
            comparison_reasons[row['same_station_observation']] += 1
            contexts[context] += 1
            if reason != 'supported':
                unsupported_contexts[context] += 1
        rows.append(row)
    with (Path(output) / 'path_tracking.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    critical_rows = [r for r in rows if r['critical_zone']]
    supported = [r for r in critical_rows if r['observation'] == 'supported']
    coverage = len(supported) / len(critical_rows) if critical_rows else None
    comparable = [r for r in critical_rows
                  if r['same_station_observation'] == 'supported']
    comparison_coverage = (len(comparable) / len(critical_rows)
                           if critical_rows else None)
    minimum = min((r for r in rows if r['actual_bbox_clearance_m'] != ''),
                  key=lambda r: r['actual_bbox_clearance_m'],
                  default=None)
    minimum_supported = minimum is not None and minimum[
        'observation'] == 'supported'
    minimum_comparable = minimum is not None and minimum[
        'same_station_observation'] == 'supported'
    association_verdict = ('supported' if coverage == 1 and minimum_supported
                           else 'partial' if supported else 'unsupported')
    comparison_verdict = ('supported' if comparison_coverage == 1
                          and minimum_comparable else
                          'partial' if comparable else 'unsupported')
    return dict(
        schema_version=TRACE_VERSION,
        verdict=('supported' if association_verdict == comparison_verdict ==
                 'supported' else 'partial' if supported else 'unsupported'),
        association_verdict=association_verdict,
        same_station_verdict=comparison_verdict,
        critical_zone_definition='actual OBB gap <=3m and speed >=0.2m/s',
        critical_samples=len(critical_rows),
        supported_critical_samples=len(supported),
        critical_coverage=coverage,
        critical_observation_reasons=dict(reasons),
        critical_goal_contexts=dict(contexts),
        unsupported_critical_goal_contexts=dict(unsupported_contexts),
        same_station_critical_samples=len(comparable),
        same_station_critical_coverage=comparison_coverage,
        same_station_critical_reasons=dict(comparison_reasons),
        maximum_abs_tracking_cross_m=(max(
            abs(r['tracking_cross_m'])
            for r in supported) if supported else None),
        maximum_abs_tracking_heading_deg=(max(
            abs(r['tracking_heading_error_deg'])
            for r in supported) if supported else None),
        at_actual_minimum=minimum,
        limitation=(
         'Sampled, read-only RPP computation windows, not an actuation '
         'or planner-root-cause proof. Candidate IDs are not acceptance IDs. '
         'Clamped projections retain only projected (not same-station) gaps. '
         'Gap differences combine position and body heading, not pure lateral '
         'tracking loss. Transitions remain in the coverage denominator. '
         'LiDAR future-horizon minima are a separate measurement.'))


def reanalyze_trial(trial):
    """Replay saved evidence only; no ROS, CARLA, new TF or original writes.

    Versions 1 and 2 have the same raw snapshot/status/TF event contract.
    Unresolved/rejected NEW windows invalidate older evidence as in live use.
    This does not repair missing association evidence using a future status.
    """
    trial = Path(trial).expanduser().resolve()
    names = ['metadata.json', 'frames.csv', 'path_trace.jsonl']
    if (trial / 'summary.json').is_file():
        names.append('summary.json')
    original = {name: (trial / name).read_bytes() for name in names}
    metadata = json.loads(original['metadata.json'])
    frames = list(csv.DictReader(
        original['frames.csv'].decode('utf-8').splitlines()))
    snapshots, resolved, events, rejected = {}, {}, [], []
    failed_transforms = set()
    issues = Counter()
    versions = set()
    previous_sequence = 0
    for line in original['path_trace.jsonl'].decode('utf-8').splitlines():
        record = json.loads(line)
        if record.get('run_id') != metadata['run_id']:
            raise ValueError('trace contains a different run_id')
        version = record.get('schema_version')
        if version not in (1, TRACE_VERSION):
            raise ValueError('unsupported raw trace schema version')
        versions.add(version)
        sequence = record['sequence']
        if sequence <= previous_sequence:
            raise ValueError('trace sequence is not strictly increasing')
        previous_sequence = sequence
        finite(record['received_sim_time_s'])
        kind = record['event']
        if kind == 'path_snapshot':
            key = record['path_id']
            if key in snapshots:
                raise ValueError('duplicate path snapshot ID')
            snapshots[key] = record
        elif kind == 'path_resolved':
            key = record['path_id']
            if key not in snapshots or key in resolved:
                raise ValueError('unmatched or duplicate path resolution')
            poses = record['odom_poses_xy_yaw']
            if len(poses) < 2 or any(len(p) != 3 for p in poses):
                raise ValueError('invalid saved resolved path')
            resolved[key] = [[finite(v) for v in p] for p in poses]
        elif kind == 'follow_path_status':
            events.append(record)
        elif kind == 'path_rejected':
            issues['invalid_' + record['source'] + '_path'] += 1
            if record['source'] == 'controller':
                rejected.append(dict(
                    record, path_id='invalid_' + str(sequence), poses=[],
                    stamp_s=record['received_sim_time_s'],
                    observation_error='invalid_controller_window'))
        elif kind == 'path_transform_unavailable':
            issues['transform_unavailable'] += 1
            failed_transforms.add(record['path_id'])
    windows = rejected
    for key, record in snapshots.items():
        if record['source'] != 'controller':
            continue
        if key in resolved:
            windows.append(dict(record, poses=resolved[key]))
        else:
            reason = ('transform_unavailable' if key in failed_transforms else
                      'unresolved_controller_window')
            windows.append(dict(record, poses=[], observation_error=reason))
    windows.sort(key=lambda r: (r['received_sim_time_s'], r['sequence']))
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    output = trial / ('path_reanalysis_' + stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    summary = analyze_tracking(
        metadata['anchor'], metadata['prepared'], frames, windows, events,
        output)
    summary.update(run_id=metadata['run_id'], source_trial=str(trial),
                   source_trace_versions=sorted(versions),
                   snapshot_issues=dict(issues),
                   raw_trace_file='../path_trace.jsonl',
                   tracking_file='path_tracking.csv')
    provenance = dict(
        analysis_version=TRACE_VERSION, source_trial=str(trial),
        analysis_code_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        source_files_sha256={name: hashlib.sha256(data).hexdigest()
                             for name, data in original.items()},
        recorded_tf_only=True, original_results_modified=False)
    for name, payload in [('path_trace_summary.json', summary),
                          ('analysis_metadata.json', provenance)]:
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    return output, summary
