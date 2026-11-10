"""Pure geometry, fault models and honest metrics for opt-in CARLA trials.

These are sensitivity experiments, not a calibrated model of the real car.
Ground truth is an evaluation input, never a sensor-localization claim.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math

SCHEMA_VERSION = 1
SCENARIOS = ('S0', 'S1', 'S2', 'S3')
NAVIGATION_VARIANTS = ('baseline', 'rejoin')
COMMAND_MODES = ('legacy', 'latest')


def navigation_variant_parameters(name):
    """One explicit treatment; ordinary navigation defaults remain disabled."""
    if name not in NAVIGATION_VARIANTS:
        raise ValueError('unknown navigation variant: ' + str(name))
    return {
        'terminal_approach_distance_m': 8.0 if name == 'rejoin' else 0.0,
        'terminal_approach_capture_m': 1.0,
    }


@dataclass(frozen=True)
class FaultProfile:
    name: str
    steering_half_steps: int = 0
    steering_delay_s: float = 0.0
    lateral_bias_m: float = 0.0
    yaw_bias_deg: float = 0.0
    activation_delay_s: float = 3.0
    bias_ramp_s: float = 1.0

    def to_dict(self):
        return asdict(self)


PROFILES = {
    'baseline':
    FaultProfile('baseline'),
    'steering_steps':
    FaultProfile('steering_steps', steering_half_steps=7),
    'steering_delay':
    FaultProfile('steering_delay', steering_delay_s=0.2),
    'steering_both':
    FaultProfile('steering_both', steering_half_steps=7, steering_delay_s=0.2),
    'position_left':
    FaultProfile('position_left', lateral_bias_m=0.3),
    'position_right':
    FaultProfile('position_right', lateral_bias_m=-0.3),
    'yaw_left':
    FaultProfile('yaw_left', yaw_bias_deg=3.0),
    'yaw_right':
    FaultProfile('yaw_right', yaw_bias_deg=-3.0),
}


def finite(value, name='value'):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def validate_arrival_radius(value):
    """Bound this experiment's control setting, not its scoring threshold."""
    radius = finite(value, 'arrival_radius_m')
    if not 1.0 <= radius <= 3.0:
        raise ValueError('arrival_radius_m must be in [1.0, 3.0] m')
    return radius


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_yaw(values):
    x, y, z, w = values
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotate_quaternion_yaw(values, delta):
    """Left-multiply by a world-Z rotation; retain the true roll/pitch."""
    x, y, z, w = values
    sine, cosine = math.sin(delta / 2), math.cos(delta / 2)
    return (cosine * x - sine * y, cosine * y + sine * x,
            cosine * z + sine * w, cosine * w - sine * z)


def route_coordinates(x, y, start_x, start_y, yaw):
    dx, dy = x - start_x, y - start_y
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


def position_fault(profile, elapsed_s, route_yaw):
    """Bias is zero at anchoring and ramps in only AFTER route start."""
    elapsed = finite(elapsed_s)
    amount = max(
        0.0,
        min(1.0, (elapsed - profile.activation_delay_s) / profile.bias_ramp_s))
    bias = profile.lateral_bias_m * amount
    return (-math.sin(route_yaw) * bias, math.cos(route_yaw) * bias,
            math.radians(profile.yaw_bias_deg) * amount)


class SteeringFault:
    """Delay/quantize steering ONLY; brakes and watchdog are never queued."""

    def __init__(self, profile):
        self.profile = profile
        self.history = deque()
        self.last_time = None

    def reset(self):
        self.history.clear()
        self.last_time = None

    def apply(self, steer, sim_time, elapsed_s, stopping=False):
        value = max(-1.0, min(1.0, finite(steer, 'steer')))
        now = finite(sim_time, 'sim_time')
        if stopping or (self.last_time is not None and now < self.last_time):
            self.reset()
            return 0.0
        self.last_time = now
        self.history.append((now, value))
        active = elapsed_s >= self.profile.activation_delay_s
        delay = self.profile.steering_delay_s if active else 0.0
        target_time = now - delay
        while len(self.history) > 1 and self.history[1][0] <= target_time:
            self.history.popleft()
        delayed = self.history[0][1] if self.history[0][
            0] <= target_time else 0.0
        steps = self.profile.steering_half_steps if active else 0
        if steps:
            # Symmetric nearest-step rounding, including half-way ties.
            integer = math.floor(abs(delayed) * steps + 0.5)
            delayed = math.copysign(integer / steps, delayed)
        return delayed


def validate_anchor(anchor):
    if anchor.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported anchor schema')
    if not anchor.get('map_name') or not anchor.get('vehicle_blueprint'):
        raise ValueError('anchor needs map and vehicle blueprint')
    for key in ('x', 'y', 'z', 'yaw', 'pitch', 'roll'):
        finite(anchor['start_carla'][key], key)
    length = finite(anchor['route_length_m'], 'route length')
    distance = finite(anchor['obstacle_distance_m'], 'obstacle distance')
    if length < 30 or not 10 <= distance <= length - 10:
        raise ValueError('need >=30 m route and >=10 m before/after obstacle')
    if abs(anchor['start_carla']['pitch']) > 3 or abs(
            anchor['start_carla']['roll']) > 3:
        raise ValueError('first experiment requires a nearly level start')
    if finite(anchor['corridor_half_width_m']) < 6:
        raise ValueError('corridor half-width must be >=6 m for Lincoln')
    if len(anchor['waypoints']) != 2:
        raise ValueError('straight experiment requires exactly A and B')
    for point in anchor['waypoints']:
        latitude = finite(point['latitude'])
        longitude = finite(point['longitude'])
        finite(point['altitude'])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError('invalid WGS84 waypoint')
    return anchor


def scenario_offsets(scenario,
                     obstacle_distance,
                     vehicle_width,
                     corridor_half_width=7.0):
    """Return frozen-car lateral offsets (CARLA right positive).

    S3 fills the approved corridor, not the entire infinite CARLA world.
    Leaving that corridor is a failure, never a successful detour.
    """
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    if scenario == 'S0':
        return []
    if scenario == 'S1':
        return [(obstacle_distance, 0.0)]
    spacing = finite(vehicle_width) + 0.6
    count = int(math.ceil((corridor_half_width - vehicle_width / 2) / spacing))
    if scenario == 'S2':
        return [(obstacle_distance, -spacing * index)
                for index in range(count + 1)]
    return [(obstacle_distance, spacing * index)
            for index in range(-count, count + 1)]


def rectangle(x, y, yaw, half_length, half_width):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [(x + cosine * a - sine * b, y + sine * a + cosine * b)
            for a, b in ((half_length, half_width), (-half_length, half_width),
                         (-half_length, -half_width), (half_length,
                                                       -half_width))]


def _segment_distance(point, first, second):
    dx, dy = second[0] - first[0], second[1] - first[1]
    length_sq = dx * dx + dy * dy
    ratio = 0.0 if not length_sq else max(
        0.0,
        min(1.0, ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) /
            length_sq))
    return math.hypot(point[0] - first[0] - ratio * dx,
                      point[1] - first[1] - ratio * dy)


def polygon_clearance(first, second):
    """2D oriented bounding-box clearance proxy, NOT exact mesh distance."""
    separated = False
    for polygon in (first, second):
        for index, point in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            axis = (-(following[1] - point[1]), following[0] - point[0])
            left = [p[0] * axis[0] + p[1] * axis[1] for p in first]
            right = [p[0] * axis[0] + p[1] * axis[1] for p in second]
            if max(left) < min(right) or max(right) < min(left):
                separated = True
    if not separated:
        return 0.0
    return min(
        _segment_distance(point, polygon[index], polygon[(index + 1) %
                                                         len(polygon)])
        for points, polygon in ((first, second), (second, first))
        for point in points for index in range(len(polygon)))


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(math.ceil((len(ordered) - 1) * fraction))]


def evaluate_trial(metadata, frames):
    """Keep collision, expected stop, goal arrival and recovery separate."""
    result = {
        'schema_version':
        SCHEMA_VERSION,
        'run_id':
        metadata['run_id'],
        'scenario':
        metadata['scenario'],
        'profile':
        metadata['profile'],
        'termination':
        metadata['termination'],
        'success':
        False,
        'collision':
        False,
        'expected_stop':
        False,
        'goal_arrived':
        False,
        'recovered':
        False,
        'metrics_valid':
        False,
        'samples':
        len(frames),
        'requested_arrival_radius_m':
        metadata.get('requested_arrival_radius_m'),
        'active_arrival_radius_m':
        metadata.get('active_arrival_radius_m'),
        'arrival_radius_verified':
        metadata.get('arrival_radius_verified', False),
        'navigation_variant': metadata.get('navigation_variant', 'unrecorded'),
        'navigation_variant_verified': metadata.get(
            'navigation_variant_verified', False),
        'command_mode': metadata.get('command_mode', 'unrecorded'),
        'command_mode_verified': metadata.get('command_mode_verified', False),
        'halt_acknowledged': metadata.get('halt_acknowledged'),
        'vehicle_stopped_after_halt': metadata.get(
            'vehicle_stopped_after_halt'),
        'shutdown_confirmed': metadata.get('halt_confirmation_complete'),
        'limitations': [
            'clearance uses oriented bounding boxes, not mesh',
            'synthetic faults are not calibrated real sensors'
        ],
    }
    if len(frames) < 2:
        result['verdict'] = 'insufficient_data'
        return result
    required_numbers = (
        'sim_time_s',
        'truth_x',
        'truth_y',
        'along_track_m',
        'cross_track_m',
        'heading_error_deg',
        'goal_error_m',
        'speed_mps',
    )
    try:
        for row in frames:
            for name in required_numbers:
                finite(row[name], name)
    except (ValueError, TypeError, KeyError):
        result['verdict'] = 'invalid_recording'
        return result
    times = [finite(row['sim_time_s']) for row in frames]
    gaps = [b - a for a, b in zip(times, times[1:])]
    # Gate status is a retained change event, not a periodic heartbeat.
    # Require its state while driving, but never interpret event age as loss.
    issues = {
        'sample_gap':
        sum(not 0 < gap <= 0.5 for gap in gaps),
        'tf_mismatch':
        sum(not bool(int(row.get('tf_consistent', 1))) for row in frames),
        'path_gate_state_missing_while_moving':
        sum(not row.get('path_gate_state') and (
            abs(float(row['speed_mps'])) > 0.05
            or abs(float(row.get('command_speed_mps', 0))) > 0.05)
            for row in frames),
        'final_path_gate_state_missing':
        int(not frames[-1].get('path_gate_state')),
    }
    for name, limit in (('lidar_age_s', 0.5), ('command_age_s', 0.5),
                        ('safety_state_age_s', 0.5), ('bridge_status_age_s',
                                                      0.2)):
        issues[name] = sum(not float(row.get(name, 0)) <= limit
                           for row in frames)
    result['recording_issues'] = {
        name: count
        for name, count in issues.items() if count
    }
    data_valid = not result['recording_issues']
    result['metrics_valid'] = data_valid
    result['maximum_sample_gap_s'] = max(gaps)
    result['duration_sim_s'] = times[-1] - times[0]
    result['collision'] = any(
        int(row['collision_count']) > 0 for row in frames)
    result['corridor_violation'] = any(
        bool(int(row['corridor_violation'])) for row in frames)
    result['cross_track_p95_m'] = _percentile(
        [abs(float(row['cross_track_m'])) for row in frames], 0.95)
    result['cross_track_max_m'] = max(
        abs(float(row['cross_track_m'])) for row in frames)
    result['final_goal_error_m'] = float(frames[-1]['goal_error_m'])
    result['final_heading_error_deg'] = abs(
        float(frames[-1]['heading_error_deg']))
    # Diagnostics only: keep the historical truth-based 3 m scoring contract.
    # GNSS uses the antenna, not the vehicle body; do not compare these
    # distances as though they had the same measurement origin.
    navigation_distances = []
    navigation_reached = False
    for row in frames:
        # A also emits reached=True near zero metres. Do not confuse that
        # initial waypoint with arrival near the final waypoint B.
        if float(row['goal_error_m']) > 3.0:
            continue
        try:
            distance = finite(row['navigation_goal_distance_m'])
            distance_age = finite(row['navigation_goal_distance_age_s'])
            reached_age = finite(row['navigation_goal_reached_age_s'])
            if distance >= 0 and 0 <= distance_age <= 0.5:
                navigation_distances.append(distance)
            if 0 <= reached_age <= 0.5:
                navigation_reached |= str(
                    row['navigation_goal_reached']) == '1'
        except (KeyError, TypeError, ValueError):
            continue
    result['minimum_navigation_goal_distance_m'] = (
        min(navigation_distances) if navigation_distances else None)
    result['navigation_goal_reached_observed'] = navigation_reached
    result['route_completed_observed'] = any(
        row.get('route_status') == 'completed' for row in frames)
    result['maximum_progress_m'] = max(
        float(row['along_track_m']) for row in frames)
    result['maximum_observed_pose_bias_m'] = max(
        float(row.get('pose_bias_m', 0)) for row in frames)
    result['maximum_observed_yaw_bias_deg'] = max(
        abs(float(row.get('yaw_bias_deg', 0))) for row in frames)
    result['maximum_steering_change'] = max(
        abs(
            float(row.get('steer_requested', 0)) -
            float(row.get('steer_applied', 0))) for row in frames)
    profile = PROFILES[metadata['profile']]
    result['pose_fault_observed'] = (
        (not profile.lateral_bias_m or result['maximum_observed_pose_bias_m']
         >= abs(profile.lateral_bias_m) * 0.95) and
        (not profile.yaw_bias_deg or result['maximum_observed_yaw_bias_deg']
         >= abs(profile.yaw_bias_deg) * 0.95))
    result['steering_effect_observed'] = result[
        'maximum_steering_change'] > 1e-5
    clearances = [
        float(row['bbox_clearance_m']) for row in frames
        if row.get('bbox_clearance_m') not in ('', None)
    ]
    result['minimum_bbox_clearance_m'] = min(
        clearances) if clearances else None
    result['bbox_overlap'] = bool(clearances and min(clearances) <= 0)
    result['path_length_m'] = sum(
        math.hypot(
            float(b['truth_x']) - float(a['truth_x']),
            float(b['truth_y']) - float(a['truth_y']))
        for a, b in zip(frames, frames[1:]))
    stop_total = stop_longest = stop_run = 0.0
    recovery_start = passage_time = recovery_confirmed = None
    recovery_confirmed_at = moving_start = None
    moving_confirmed_at = None
    moving_recovery_before_completion = False
    completed_at = next((float(row['sim_time_s']) for row in frames
                         if row.get('route_status') == 'completed'), None)
    blocked_run = blocked_longest = 0.0
    post_clearance = metadata.get('obstacle_passage_s_m')
    for index, row in enumerate(frames):
        dt = gaps[index - 1] if index else 0.0
        dt = dt if 0 < dt <= 0.5 else 0.0
        stopped = abs(float(row['speed_mps'])) <= 0.05
        stop_run = stop_run + dt if stopped else 0.0
        stop_total += dt if stopped else 0.0
        stop_longest = max(stop_longest, stop_run)
        beyond = post_clearance is not None and float(
            row['along_track_m']) >= post_clearance
        if beyond and passage_time is None:
            passage_time = float(row['sim_time_s'])
        aligned = beyond and abs(float(row['cross_track_m'])) <= 0.5 and abs(
            float(row['heading_error_deg'])) <= 5
        if aligned:
            if recovery_start is None:
                recovery_start = float(row['sim_time_s'])
            if float(row['sim_time_s']) - recovery_start >= 2:
                if recovery_confirmed is None:
                    recovery_confirmed = recovery_start
                    recovery_confirmed_at = float(row['sim_time_s'])
        else:
            recovery_start = None
        moving_aligned = (
            aligned and abs(float(row['speed_mps'])) >= 0.2
            and completed_at is not None
            and float(row['sim_time_s']) < completed_at
            and (index == 0 or 0 < gaps[index - 1] <= 0.5))
        if moving_aligned:
            if moving_start is None:
                moving_start = float(row['sim_time_s'])
            if float(row['sim_time_s']) - moving_start >= 2.0:
                moving_recovery_before_completion = True
                if moving_confirmed_at is None:
                    moving_confirmed_at = float(row['sim_time_s'])
        else:
            moving_start = None
        reason = str(row.get('path_gate_state', '')).lower()
        explainable = str(row.get(
            'safety_state', '')) == 'obstacle_stop' or ('invalid' in reason
                                                        or 'blocked' in reason)
        healthy = row.get('safety_state') not in ('sensor_stale',
                                                  'command_stale',
                                                  'waiting_for_cloud',
                                                  'invalid_cloud')
        blocked = (stopped and explainable and healthy
                   and float(row['along_track_m']) >= 8
                   and metadata['scenario'] == 'S3')
        blocked_run = blocked_run + dt if blocked else 0.0
        blocked_longest = max(blocked_longest, blocked_run)
    result['total_stop_s'] = stop_total
    result['maximum_stop_s'] = stop_longest
    result['explained_blocked_stop_s'] = blocked_longest
    result['recovered'] = recovery_confirmed is not None
    result['recovery_time_s'] = (recovery_confirmed -
                                 passage_time if result['recovered'] else None)
    result['recovery_confirmed_before_route_completion'] = (
        recovery_confirmed_at is not None and completed_at is not None
        and recovery_confirmed_at < completed_at)
    result['moving_recovery_before_route_completion'] = (
        moving_recovery_before_completion)
    result['terminal_approach_observed'] = any(
        row.get('terminal_approach_phase') == 'approach' for row in frames)
    result['goal_arrived'] = result['final_goal_error_m'] <= 3.0
    finished = metadata['termination'] == 'goal_stopped'
    recovery_needed = metadata['scenario'] in ('S1', 'S2')
    safe = (data_valid and result['pose_fault_observed']
            and not result['collision'] and not result['bbox_overlap']
            and not result['corridor_violation'])
    if metadata['scenario'] == 'S3':
        result['expected_stop'] = safe and blocked_longest >= 10 and (
            metadata['termination'] == 'blocked_stopped')
        result['success'] = result['expected_stop']
    else:
        result['success'] = (safe and finished and result['goal_arrived']
                             and (not recovery_needed or result['recovered']))
    result['verdict'] = ('expected_stop' if result['expected_stop'] else
                         'pass' if result['success'] else 'fail')
    # Historical motion criteria are preserved. New runs additionally must
    # confirm safe protocol shutdown; old summaries/files are never rewritten.
    result['motion_success'] = result['success']
    if (metadata.get('halt_protocol_version', 1) >= 2
            and not metadata.get('halt_confirmation_complete', False)):
        result['success'] = False
        if result['motion_success']:
            result['verdict'] = 'shutdown_unconfirmed'
    result.update(_terminal_alignment_diagnostics(
        metadata, frames, completed_at, moving_confirmed_at, result))
    return result


def _terminal_alignment_diagnostics(metadata, frames, completed_at,
                                    moving_confirmed_at, result):
    """Stricter, separate diagnostics: never silently redefine legacy pass.

    These are sampled observations (up to sample-gap uncertainty), NOT exact
    transport latencies. In particular an unstamped Twist has no source age.
    """
    def aligned(row):
        return (abs(float(row['cross_track_m'])) <= .5
                and abs(float(row['heading_error_deg'])) <= 5)

    last = frames[-1]
    final_aligned = aligned(last) and abs(float(last['speed_mps'])) <= .05
    tail = [row for row in frames if moving_confirmed_at is not None
            and float(row['sim_time_s']) >= moving_confirmed_at]
    gaps_valid = all(0 < float(b['sim_time_s']) - float(a['sim_time_s']) <= .5
                     for a, b in zip(tail, tail[1:]))
    violations = sum(not aligned(row) for row in tail) if tail else None
    retained = bool(tail and gaps_valid and not violations and final_aligned
                    and metadata['termination'] == 'goal_stopped')
    post = [row for row in frames if completed_at is not None
            and float(row['sim_time_s']) >= completed_at]

    def first_delay(field, predicate):
        for row in post:
            try:
                value = finite(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if predicate(value):
                return float(row['sim_time_s']) - completed_at
        return None

    distance = None
    if post:
        distance = sum(
            math.hypot(float(b['truth_x']) - float(a['truth_x']),
                       float(b['truth_y']) - float(a['truth_y']))
            for a, b in zip(post, post[1:]))
    return {
        'terminal_diagnostics_version': 1,
        'final_alignment_within_bounds': final_aligned,
        'moving_recovery_confirmed_elapsed_s': (
            moving_confirmed_at - float(frames[0]['sim_time_s'])
            if moving_confirmed_at is not None else None),
        'post_recovery_alignment_violation_samples': violations,
        'recovery_retained_until_stop': retained,
        'aligned_stop_success': (
            bool(result['success'] and retained)
            if metadata['scenario'] in ('S1', 'S2') else None),
        'heading_error_at_route_completion_deg': (
            float(post[0]['heading_error_deg']) if post else None),
        'maximum_heading_error_after_route_completion_deg': (
            max(abs(float(row['heading_error_deg'])) for row in post)
            if post else None),
        'completion_to_full_brake_s': first_delay('brake', lambda x: x >= .99),
        'completion_to_accepted_stop_s': first_delay(
            'accepted_command_speed_mps', lambda x: 0 <= x <= .02),
        'completion_to_standstill_s': first_delay(
            'speed_mps', lambda x: abs(x) <= .05),
        'post_completion_distance_m': distance,
    }
