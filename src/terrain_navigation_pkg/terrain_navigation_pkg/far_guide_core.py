"""Pure helpers for a FAR-inspired online long-range guide.

The guide deliberately does not command the vehicle.  It searches the rolling
occupancy grid for a coarse collision-free route, removes unnecessary grid
corners with line-of-sight tests, and selects a short target for the
kinematically constrained Nav2 planner.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence, Tuple

import numpy as np


GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]


def select_direction_continuity(
    retry_heading_rad,
    leg_heading_rad,
    retry_distance_m: float,
    retry_cost_weight: float,
    leg_distance_m: float,
    leg_cost_weight: float,
):
    """Select the soft direction preference for the next online guide.

    A short-lived retry heading represents immediate controller feedback and
    therefore takes precedence.  Otherwise, keep the heading established by
    the last successful segment of the current mission leg.  Returning a
    *cost preference*, rather than a blocked corridor, lets A* change sides
    when the remembered direction is genuinely obstructed.
    """
    candidates = (
        (
            retry_heading_rad,
            float(retry_distance_m),
            float(retry_cost_weight),
            'retry',
        ),
        (
            leg_heading_rad,
            float(leg_distance_m),
            float(leg_cost_weight),
            'mission_leg',
        ),
    )
    for heading, distance, weight, source in candidates:
        if heading is None:
            continue
        values = float(heading), distance, weight
        if not all(math.isfinite(value) for value in values):
            continue
        if distance < 0.0 or weight < 0.0:
            raise ValueError(
                'direction continuity parameters cannot be negative'
            )
        return values[0], distance, weight, source
    return None, 0.0, 0.0, 'none'


def bounded_heading_preference(
    reference_yaw_rad: float,
    preferred_yaw_rad: float,
    maximum_bias_rad: float,
):
    """Bias one feasible path heading toward a route preference.

    A Hybrid-A* pose goal treats yaw as a hard terminal constraint.  Applying
    an unconstrained next-leg tangent to a nearby waypoint can therefore make
    a forward-only Dubins vehicle drive a full circle just to match that yaw.
    Keep the already feasible guide-arrival heading as the reference and move
    it toward the route tangent by at most ``maximum_bias_rad``.
    """
    reference = float(reference_yaw_rad)
    preferred = float(preferred_yaw_rad)
    maximum = float(maximum_bias_rad)
    if not all(math.isfinite(value) for value in (reference, preferred, maximum)):
        raise ValueError('heading values must be finite')
    if maximum < 0.0:
        raise ValueError('maximum heading bias cannot be negative')
    difference = math.atan2(
        math.sin(preferred - reference),
        math.cos(preferred - reference),
    )
    applied = max(-maximum, min(maximum, difference))
    return math.atan2(
        math.sin(reference + applied),
        math.cos(reference + applied),
    )


@dataclass(frozen=True)
class GuideGrid:
    """A coarse planning grid in an odometry-aligned world frame."""

    costs: np.ndarray
    blocked: np.ndarray
    resolution_m: float
    origin_x_m: float
    origin_y_m: float

    def __post_init__(self):
        if self.costs.ndim != 2 or self.blocked.ndim != 2:
            raise ValueError('guide grid arrays must be two-dimensional')
        if self.costs.shape != self.blocked.shape:
            raise ValueError('guide grid cost and blocked shapes must match')
        if self.resolution_m <= 0.0:
            raise ValueError('guide grid resolution must be positive')

    @property
    def height(self):
        return int(self.costs.shape[0])

    @property
    def width(self):
        return int(self.costs.shape[1])

    def contains(self, cell: GridCell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def world_to_cell(self, point: WorldPoint):
        x = int(math.floor(
            (float(point[0]) - self.origin_x_m) / self.resolution_m
        ))
        y = int(math.floor(
            (float(point[1]) - self.origin_y_m) / self.resolution_m
        ))
        return x, y

    def cell_to_world(self, cell: GridCell):
        return (
            self.origin_x_m + (float(cell[0]) + 0.5) * self.resolution_m,
            self.origin_y_m + (float(cell[1]) + 0.5) * self.resolution_m,
        )


@dataclass(frozen=True)
class ActiveReplanAssessment:
    """Evidence for replacing one currently executing short segment."""

    reason: str
    subgoal_change_m: float
    active_remaining_m: float
    candidate_length_m: float
    improvement_m: float


@dataclass(frozen=True)
class PathEfficiencyAssessment:
    """Geometric evidence that one Smac path contains a needless loop."""

    path_length_m: float
    reference_length_m: float
    length_ratio: float
    absolute_turn_rad: float
    signed_turn_rad: float
    inefficient: bool
    reasons: Tuple[str, ...]


@dataclass
class SafetyRejectionMonitor:
    """Debounce independent LiDAR rejections of one active Nav2 segment.

    One rejected scan can be a transient return.  Counting only distinct
    cloud identifiers inside a bounded wall-time window prevents duplicate
    publications from triggering a replan while still reacting much sooner
    than Nav2's progress-checker timeout.
    """

    required_confirmations: int
    window_s: float
    count: int = 0
    window_started_s: float | None = None
    last_evidence_id: int | None = None

    def __post_init__(self):
        self.required_confirmations = int(self.required_confirmations)
        self.window_s = float(self.window_s)
        if self.required_confirmations < 1:
            raise ValueError('safety rejection confirmations must be positive')
        if not math.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError('safety rejection window must be positive')

    def reset(self):
        self.count = 0
        self.window_started_s = None
        self.last_evidence_id = None

    def observe(self, now_s: float, evidence_id: int):
        """Return true once enough distinct evidence arrives in the window."""

        now = float(now_s)
        identifier = int(evidence_id)
        if not math.isfinite(now):
            raise ValueError('safety rejection time must be finite')
        if identifier == 0:
            raise ValueError('safety rejection evidence id cannot be zero')
        if identifier == self.last_evidence_id:
            return False
        if (
            self.window_started_s is None
            or now < self.window_started_s
            or now - self.window_started_s > self.window_s
        ):
            self.count = 0
            self.window_started_s = now
        self.last_evidence_id = identifier
        self.count += 1
        return self.count >= self.required_confirmations


def is_actionable_safety_stop(
    state: str,
    state_age_s: float,
    timeout_s: float,
):
    """Return whether a safety state authorizes discarding the active path."""

    age = float(state_age_s)
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError('safety state timeout must be positive')
    return (
        str(state) == 'obstacle_stop'
        and math.isfinite(age)
        and 0.0 <= age <= timeout
    )


@dataclass
class SafetyReplanHold:
    """Bound replanning until the raw-LiDAR safety state settles.

    A safety-triggered Nav2 cancellation must not immediately replace the
    stopped segment while the same obstacle scan is still active.  Consecutive
    clear states release the hold early; the finite timeout prevents a missing
    state message from deadlocking navigation.  This class never changes the
    safety command itself.
    """

    required_clear_confirmations: int
    maximum_hold_s: float
    active: bool = False
    started_s: float | None = None
    clear_count: int = 0

    def __post_init__(self):
        self.required_clear_confirmations = int(
            self.required_clear_confirmations
        )
        self.maximum_hold_s = float(self.maximum_hold_s)
        if self.required_clear_confirmations < 1:
            raise ValueError(
                'safety replan clear confirmations must be positive'
            )
        if (
            not math.isfinite(self.maximum_hold_s)
            or self.maximum_hold_s <= 0.0
        ):
            raise ValueError('safety replan maximum hold must be positive')

    def arm(self, now_s: float):
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError('safety replan hold time must be finite')
        self.active = True
        self.started_s = now
        self.clear_count = 0

    def reset(self):
        self.active = False
        self.started_s = None
        self.clear_count = 0

    def observe_state(self, state: str):
        if not self.active:
            return
        if str(state) == 'clear':
            self.clear_count += 1
        else:
            self.clear_count = 0

    def release_reason(self, now_s: float):
        """Return the release reason, or ``None`` while still holding."""

        if not self.active:
            return 'inactive'
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError('safety replan hold time must be finite')
        if self.clear_count >= self.required_clear_confirmations:
            return 'clear_confirmed'
        if self.started_s is None or now - self.started_s >= self.maximum_hold_s:
            return 'timeout'
        return None


def can_accept_length_only_detour(
    assessment: PathEfficiencyAssessment,
    previous_rejections: int,
    minimum_previous_rejections: int = 1,
):
    """Return whether a long, non-looping Hybrid path may be followed.

    Path length is only a soft efficiency signal: a legitimate obstacle
    detour can be substantially longer than the FAR guide prefix.  Signed
    loops and excessive winding are topology failures and must remain hard
    rejects.  Requiring at least one prior rejection keeps the first-pass
    efficiency preference, while allowing the next collision-checked Smac
    path when its sole issue is length.
    """
    rejected = int(previous_rejections)
    minimum = int(minimum_previous_rejections)
    if rejected < 0 or minimum < 0:
        raise ValueError('path rejection counts cannot be negative')
    return (
        assessment is not None
        and assessment.inefficient
        and assessment.reasons == ('length_ratio',)
        and rejected >= minimum
    )


def can_accept_bounded_topology_escape(
    assessment: PathEfficiencyAssessment,
    previous_rejections: int,
    minimum_previous_rejections: int,
    already_used: bool,
    maximum_length_ratio: float,
    maximum_absolute_turn_rad: float,
    maximum_signed_turn_rad: float,
):
    """Allow one bounded P-turn after ordinary Hybrid retries are exhausted.

    A forward-only Ackermann vehicle can occasionally have only a wide
    P-turn available.  Treating every path above the normal signed-turn limit
    as a permanent failure caused the observed infinite hold.  This helper is
    intentionally strict: the path must contain a topology violation, fit
    within a second set of finite bounds, and may be accepted only once per
    mission waypoint after the configured number of rejected alternatives.
    """
    rejected = int(previous_rejections)
    minimum = int(minimum_previous_rejections)
    limits = (
        float(maximum_length_ratio),
        float(maximum_absolute_turn_rad),
        float(maximum_signed_turn_rad),
    )
    if rejected < 0 or minimum < 0:
        raise ValueError('path rejection counts cannot be negative')
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError('bounded topology escape limits must be positive')
    if assessment is None or not assessment.inefficient or already_used:
        return False
    topology_failure = any(
        reason in ('signed_loop', 'winding')
        for reason in assessment.reasons
    )
    return (
        topology_failure
        and rejected >= minimum
        and assessment.length_ratio <= limits[0]
        and assessment.absolute_turn_rad <= limits[1]
        and abs(assessment.signed_turn_rad) <= limits[2]
    )


def should_request_costmap_recovery(
    abort_count: int,
    minimum_abort_count: int,
    speed_mps: float,
    maximum_speed_mps: float,
    path_hard_valid: bool,
    path_validity_fresh: bool,
    safety_state: str,
    safety_state_fresh: bool,
    cooldown_ready: bool,
):
    """Gate a small costmap clear using independent safety evidence.

    The recovery is for transient lethal/inscribed cells around a stationary
    vehicle, not for bypassing a real obstacle.  It is therefore permitted
    only after repeated action aborts while both the raw-LiDAR safety state
    and the separately evaluated path report clear/fresh evidence.
    """
    aborts = int(abort_count)
    threshold = int(minimum_abort_count)
    speed = float(speed_mps)
    maximum_speed = float(maximum_speed_mps)
    if aborts < 0 or threshold < 1:
        raise ValueError('costmap recovery abort counts are invalid')
    if not math.isfinite(speed) or not math.isfinite(maximum_speed):
        return False
    if maximum_speed < 0.0:
        raise ValueError('costmap recovery speed cannot be negative')
    return (
        aborts >= threshold
        and abs(speed) <= maximum_speed
        and bool(path_hard_valid)
        and bool(path_validity_fresh)
        and str(safety_state) == 'clear'
        and bool(safety_state_fresh)
        and bool(cooldown_ready)
    )


def retry_lookahead_distance(
    base_lookahead_m: float,
    minimum_lookahead_m: float,
    reduction_step_m: float,
    retry_count: int,
):
    """Return a bounded, progressively shorter Hybrid-planner target.

    Re-sending the same distant pose after Smac Hybrid produces a needless
    Dubins loop cannot change the kinematic problem. Moving the subgoal toward
    the vehicle gives the planner a genuinely different, reachable short
    action while preserving the same long-range guide corridor.
    """
    base = float(base_lookahead_m)
    minimum = float(minimum_lookahead_m)
    step = float(reduction_step_m)
    retries = int(retry_count)
    if base <= 0.0 or minimum <= 0.0:
        raise ValueError('retry lookahead distances must be positive')
    if minimum > base:
        raise ValueError('minimum retry lookahead cannot exceed base')
    if step < 0.0 or retries < 0:
        raise ValueError('retry lookahead step/count cannot be negative')
    return max(minimum, base - step * retries)


def polyline_length(path: Sequence[WorldPoint]):
    """Return the planar length of one world-frame polyline."""
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(path, path[1:])
    )


def assess_path_efficiency(
    path: Sequence[WorldPoint],
    reference_length_m: float,
    maximum_length_ratio: float,
    maximum_absolute_turn_rad: float,
    maximum_signed_turn_rad: float,
    minimum_reference_length_m: float = 4.0,
    minimum_excess_length_m: float = 4.0,
):
    """Detect a long or winding Smac path relative to its FAR guide prefix.

    FAR's coarse guide already contains the positional detour required by the
    rolling costmap.  Smac Hybrid may nevertheless satisfy the same subgoal
    pose with a full Dubins loop.  Comparing the actual Hybrid path with the
    guide prefix distinguishes that kinematic artefact from a legitimate
    obstacle detour.  The turn metrics are calculated after regular spatial
    resampling so dense planner output does not receive extra weight.
    """
    if len(path) < 2:
        return None
    if min(
        maximum_length_ratio,
        maximum_absolute_turn_rad,
        maximum_signed_turn_rad,
    ) <= 0.0:
        raise ValueError('path efficiency limits must be positive')
    if minimum_reference_length_m < 0.0 or minimum_excess_length_m < 0.0:
        raise ValueError('path efficiency minimums cannot be negative')

    cleaned = []
    for point in path:
        value = float(point[0]), float(point[1])
        if not all(math.isfinite(component) for component in value):
            continue
        if not cleaned or math.hypot(
            value[0] - cleaned[-1][0], value[1] - cleaned[-1][1]
        ) >= 0.02:
            cleaned.append(value)
    if len(cleaned) < 2:
        return None

    path_length = polyline_length(cleaned)
    direct_length = math.hypot(
        cleaned[-1][0] - cleaned[0][0],
        cleaned[-1][1] - cleaned[0][1],
    )
    reference_length = max(
        float(reference_length_m), direct_length, 1e-6
    )
    length_ratio = path_length / reference_length

    sampled = _sample_polyline(cleaned, 0.5)
    headings = []
    for first, second in zip(sampled, sampled[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if math.hypot(dx, dy) >= 0.05:
            headings.append(math.atan2(dy, dx))
    turns = []
    for first, second in zip(headings, headings[1:]):
        turns.append(math.atan2(
            math.sin(second - first), math.cos(second - first)
        ))
    absolute_turn = sum(abs(value) for value in turns)
    signed_turn = sum(turns)

    reasons = []
    enough_reference = reference_length >= minimum_reference_length_m
    enough_excess = path_length - reference_length >= minimum_excess_length_m
    if enough_reference and enough_excess and length_ratio > maximum_length_ratio:
        reasons.append('length_ratio')
    if (
        enough_reference
        and abs(signed_turn) > maximum_signed_turn_rad
    ):
        reasons.append('signed_loop')
    if (
        enough_reference
        and length_ratio > 1.2
        and absolute_turn > maximum_absolute_turn_rad
    ):
        reasons.append('winding')
    return PathEfficiencyAssessment(
        path_length,
        reference_length,
        length_ratio,
        absolute_turn,
        signed_turn,
        bool(reasons),
        tuple(reasons),
    )


def _point_segment_distance(point, first, second):
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    squared = dx * dx + dy * dy
    if squared < 1e-12:
        return math.hypot(
            float(point[0]) - float(first[0]),
            float(point[1]) - float(first[1]),
        )
    ratio = min(1.0, max(0.0, (
        (float(point[0]) - float(first[0])) * dx
        + (float(point[1]) - float(first[1])) * dy
    ) / squared))
    projection_x = float(first[0]) + ratio * dx
    projection_y = float(first[1]) + ratio * dy
    return math.hypot(
        float(point[0]) - projection_x,
        float(point[1]) - projection_y,
    )


def point_polyline_distance(point, path: Sequence[WorldPoint]):
    """Return the shortest planar distance from a point to a polyline."""
    if not path:
        return float('inf')
    if len(path) == 1:
        return math.hypot(
            float(point[0]) - float(path[0][0]),
            float(point[1]) - float(path[0][1]),
        )
    return min(
        _point_segment_distance(point, first, second)
        for first, second in zip(path, path[1:])
    )


def _sample_polyline(path: Sequence[WorldPoint], spacing_m: float):
    if not path:
        return []
    samples = [(float(path[0][0]), float(path[0][1]))]
    spacing = max(0.05, float(spacing_m))
    for first, second in zip(path, path[1:]):
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        count = max(1, int(math.ceil(length / spacing)))
        for index in range(1, count + 1):
            ratio = float(index) / float(count)
            samples.append((
                float(first[0]) + ratio * dx,
                float(first[1]) + ratio * dy,
            ))
    return samples


def polylines_similar(
    candidate: Sequence[WorldPoint],
    failed_reference: Sequence[WorldPoint],
    maximum_distance_m: float,
):
    """Return whether a candidate repeats one previously failed corridor.

    The comparison is intentionally directed from the new action prefix to
    the failed prefix.  The vehicle can have advanced slightly before Nav2
    reports ABORTED, so requiring the old, already-travelled start to match
    the new path would incorrectly classify the same retry as different.
    """
    if not candidate or not failed_reference or maximum_distance_m < 0.0:
        return False
    spacing = max(0.1, min(0.5, 0.5 * maximum_distance_m))
    return all(
        point_polyline_distance(point, failed_reference)
        <= maximum_distance_m
        for point in _sample_polyline(candidate, spacing)
    )


def prefix_polyline_to_point(
    path: Sequence[WorldPoint],
    endpoint: WorldPoint,
):
    """Cut a guide at the point actually submitted to the Nav2 action."""
    if not path:
        return []
    if len(path) == 1:
        return [(float(endpoint[0]), float(endpoint[1]))]
    best_index = 0
    best_distance = float('inf')
    for index, (first, second) in enumerate(zip(path, path[1:])):
        distance = _point_segment_distance(endpoint, first, second)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    result = [
        (float(point[0]), float(point[1]))
        for point in path[:best_index + 1]
    ]
    endpoint_value = (float(endpoint[0]), float(endpoint[1]))
    if not result or math.hypot(
        endpoint_value[0] - result[-1][0],
        endpoint_value[1] - result[-1][1],
    ) > 1e-9:
        result.append(endpoint_value)
    return result


def remaining_polyline_length(
    path: Sequence[WorldPoint],
    current: WorldPoint,
):
    """Estimate distance remaining after projecting ``current`` onto a path.

    Smac Hybrid does not follow the coarse FAR polyline exactly.  Projection
    lets the replanning monitor compare the old guide with a newly observed
    guide without incorrectly charging already travelled path length.
    """
    if not path:
        return float('inf')
    if len(path) == 1:
        return math.hypot(path[0][0] - current[0], path[0][1] - current[1])

    suffix = [0.0] * len(path)
    for index in range(len(path) - 2, -1, -1):
        suffix[index] = suffix[index + 1] + math.hypot(
            path[index + 1][0] - path[index][0],
            path[index + 1][1] - path[index][1],
        )

    best_distance = float('inf')
    best_remaining = float('inf')
    cx, cy = float(current[0]), float(current[1])
    for index, (first, second) in enumerate(zip(path, path[1:])):
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        squared = dx * dx + dy * dy
        if squared < 1e-12:
            ratio = 0.0
        else:
            ratio = min(1.0, max(0.0, (
                (cx - float(first[0])) * dx
                + (cy - float(first[1])) * dy
            ) / squared))
        px = float(first[0]) + ratio * dx
        py = float(first[1]) + ratio * dy
        offset = math.hypot(cx - px, cy - py)
        segment_remaining = (1.0 - ratio) * math.sqrt(squared)
        remaining = offset + segment_remaining + suffix[index + 1]
        if offset < best_distance:
            best_distance = offset
            best_remaining = remaining
    return best_remaining


def nearest_polyline_tangent(
    path: Sequence[WorldPoint],
    current: WorldPoint,
):
    """Return the tangent of the path segment nearest ``current``.

    A failed short Nav2 action can end between two FAR vertices.  Remembering
    the local tangent, rather than the direction from the original segment
    start, gives the next online search a useful indication of the corridor
    the vehicle was already following.
    """
    if len(path) < 2:
        return None
    cx, cy = float(current[0]), float(current[1])
    best_distance = float('inf')
    best_yaw = None
    for first, second in zip(path, path[1:]):
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        squared = dx * dx + dy * dy
        if squared < 1e-12:
            continue
        ratio = min(1.0, max(0.0, (
            (cx - float(first[0])) * dx
            + (cy - float(first[1])) * dy
        ) / squared))
        px = float(first[0]) + ratio * dx
        py = float(first[1]) + ratio * dy
        distance = math.hypot(cx - px, cy - py)
        if distance < best_distance:
            best_distance = distance
            best_yaw = math.atan2(dy, dx)
    return best_yaw


def add_direction_continuity_cost(
    grid: GuideGrid,
    start_world: WorldPoint,
    heading_rad: float,
    distance_m: float,
    cost_weight: float,
):
    """Softly prefer continuing along a previous guide direction.

    The preference is intentionally not a hard mask.  Close to the current
    pose, cells behind or sideways from the remembered tangent receive more
    cost than cells ahead of it; the penalty fades to zero at ``distance_m``.
    Therefore a newly observed wall can still force the vehicle to choose the
    opposite corridor, while equal-cost left/right retries no longer tend to
    alternate and create a large P-turn.
    """
    if distance_m <= 0.0 or cost_weight <= 0.0:
        return grid
    if not math.isfinite(heading_rad):
        return grid
    costs = np.array(grid.costs, copy=True)
    cosine = math.cos(float(heading_rad))
    sine = math.sin(float(heading_rad))
    sx, sy = float(start_world[0]), float(start_world[1])
    for y_index in range(grid.height):
        for x_index in range(grid.width):
            if grid.blocked[y_index, x_index]:
                continue
            wx, wy = grid.cell_to_world((x_index, y_index))
            dx, dy = wx - sx, wy - sy
            distance = math.hypot(dx, dy)
            if distance < 1e-9 or distance >= distance_m:
                continue
            forward = dx * cosine + dy * sine
            alignment = max(-1.0, min(1.0, forward / distance))
            angular_penalty = 0.5 * (1.0 - alignment)
            fade = 1.0 - distance / distance_m
            costs[y_index, x_index] += (
                float(cost_weight) * angular_penalty * fade
            )
    return GuideGrid(
        costs,
        np.array(grid.blocked, copy=True),
        grid.resolution_m,
        grid.origin_x_m,
        grid.origin_y_m,
    )


def add_failed_corridor_cost(
    grid: GuideGrid,
    failed_paths: Sequence[Sequence[WorldPoint]],
    start_world: WorldPoint,
    radius_m: float,
    cost_weight: float,
    start_release_m: float,
    hard_block_radius_m: float = 0.0,
):
    """Penalize or block controller-rejected action corridors.

    A coarse FAR line can be valid while the controller's pursuit arc cuts
    inside that line and collides.  Nav2's ABORT result is therefore fed back
    as a short-lived corridor constraint.  A release disk around the current
    vehicle pose guarantees that remembering a path starting under the
    vehicle never traps the start cell.
    """
    if not failed_paths or radius_m <= 0.0:
        return grid
    if cost_weight <= 0.0 and hard_block_radius_m <= 0.0:
        return grid
    valid_paths = [path for path in failed_paths if path]
    if not valid_paths:
        return grid
    costs = np.array(grid.costs, copy=True)
    blocked = np.array(grid.blocked, copy=True)
    sx, sy = float(start_world[0]), float(start_world[1])
    world_x = (
        grid.origin_x_m
        + (np.arange(grid.width, dtype=np.float64) + 0.5)
        * grid.resolution_m
    )[None, :]
    world_y = (
        grid.origin_y_m
        + (np.arange(grid.height, dtype=np.float64) + 0.5)
        * grid.resolution_m
    )[:, None]
    minimum_distance = np.full(grid.costs.shape, np.inf, dtype=np.float64)
    for path in valid_paths:
        if len(path) == 1:
            distance = np.hypot(
                world_x - float(path[0][0]),
                world_y - float(path[0][1]),
            )
            minimum_distance = np.minimum(minimum_distance, distance)
            continue
        for first, second in zip(path, path[1:]):
            dx = float(second[0]) - float(first[0])
            dy = float(second[1]) - float(first[1])
            squared = dx * dx + dy * dy
            if squared < 1e-12:
                distance = np.hypot(
                    world_x - float(first[0]),
                    world_y - float(first[1]),
                )
            else:
                ratio = np.clip((
                    (world_x - float(first[0])) * dx
                    + (world_y - float(first[1])) * dy
                ) / squared, 0.0, 1.0)
                distance = np.hypot(
                    world_x - (float(first[0]) + ratio * dx),
                    world_y - (float(first[1]) + ratio * dy),
                )
            minimum_distance = np.minimum(minimum_distance, distance)
    released = np.hypot(world_x - sx, world_y - sy) <= start_release_m
    if hard_block_radius_m > 0.0:
        blocked |= (
            (minimum_distance <= hard_block_radius_m) & ~released
        )
    if cost_weight > 0.0:
        penalized = (
            (minimum_distance < radius_m) & ~blocked & ~released
        )
        costs[penalized] += float(cost_weight) * (
            1.0 - minimum_distance[penalized] / radius_m
        )
    return GuideGrid(
        costs,
        blocked,
        grid.resolution_m,
        grid.origin_x_m,
        grid.origin_y_m,
    )


def assess_active_replan(
    active_path: Sequence[WorldPoint],
    active_subgoal: WorldPoint,
    candidate_path: Sequence[WorldPoint],
    candidate_subgoal: WorldPoint,
    current: WorldPoint,
    path_blocked: bool,
    minimum_subgoal_change_m: float,
    minimum_improvement_m: float,
    endpoint_tolerance_m: float,
    allow_valid_path_optimization: bool = True,
):
    """Decide whether a fresh guide is materially better than the active one.

    A persistent hard-invalid Smac path is eligible for external replacement
    only when the fresh guide selects a materially different subgoal.  Nav2's
    behavior tree already replans an invalid path and keeps the current path
    when a replacement cannot be computed, so cancelling an action for guide
    jitter merely creates command gaps.  Replacing a still-valid path merely
    because a shorter guide appeared is optional: online rolling costmaps can
    alternately reveal left and right corridors and otherwise cause
    action-cancel/steering chatter.  When that experimental optimization is
    enabled, both guides must end at the same rolling-map target, the new
    subgoal must select a different corridor, and the remaining route must be
    shorter by a configured margin.  Temporal confirmation is deliberately
    handled by the ROS node so this helper stays deterministic.
    """
    if not active_path or not candidate_path:
        return None
    subgoal_change = math.hypot(
        float(candidate_subgoal[0]) - float(active_subgoal[0]),
        float(candidate_subgoal[1]) - float(active_subgoal[1]),
    )
    active_remaining = remaining_polyline_length(active_path, current)
    candidate_length = polyline_length(candidate_path)
    improvement = active_remaining - candidate_length
    if path_blocked:
        if subgoal_change < minimum_subgoal_change_m:
            return None
        return ActiveReplanAssessment(
            'path_blocked', subgoal_change, active_remaining,
            candidate_length, improvement,
        )
    if not allow_valid_path_optimization:
        return None
    endpoint_change = math.hypot(
        float(candidate_path[-1][0]) - float(active_path[-1][0]),
        float(candidate_path[-1][1]) - float(active_path[-1][1]),
    )
    if endpoint_change > endpoint_tolerance_m:
        return None
    if subgoal_change < minimum_subgoal_change_m:
        return None
    if improvement < minimum_improvement_m:
        return None
    return ActiveReplanAssessment(
        'shorter_guide', subgoal_change, active_remaining,
        candidate_length, improvement,
    )


def build_guide_grid(
    occupancy: np.ndarray,
    resolution_m: float,
    origin_x_m: float,
    origin_y_m: float,
    stride: int = 2,
    lethal_threshold: float = 99.0,
    unknown_cost: float = 0.15,
):
    """Coarsen one Nav2 OccupancyGrid while preserving lethal cells.

    ``occupancy`` uses the ROS convention: -1 unknown, 0 free, 100 occupied.
    Unknown space remains traversable because this is an online, mapless
    guide, but it receives a small cost so observed free space is preferred.
    Nav2 publishes truly lethal cells as 100 and its inscribed collision band
    as 99.  Both are hard obstacles for an ordinary FAR guide segment.  A
    narrowly bounded start release is applied later by ``plan_online_guide``
    so coarse-grid pooling cannot trap an otherwise valid current pose.
    """
    values = np.asarray(occupancy, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError('occupancy grid must be two-dimensional')
    if stride < 1:
        raise ValueError('guide grid stride must be positive')
    if not 0.0 <= unknown_cost <= 1.0:
        raise ValueError('unknown cost must be between zero and one')
    if not 0.0 < lethal_threshold <= 100.0:
        raise ValueError('lethal threshold must be in (0, 100]')

    height, width = values.shape
    padded_height = int(math.ceil(height / stride) * stride)
    padded_width = int(math.ceil(width / stride) * stride)
    padded = np.full((padded_height, padded_width), 100.0, dtype=np.float32)
    padded[:height, :width] = values
    blocks = padded.reshape(
        padded_height // stride,
        stride,
        padded_width // stride,
        stride,
    ).transpose(0, 2, 1, 3)

    unknown = np.any(blocks < 0.0, axis=(2, 3))
    known = np.where(blocks < 0.0, 0.0, blocks)
    maximum = np.max(known, axis=(2, 3))
    blocked = maximum >= float(lethal_threshold)
    costs = np.clip(maximum / 100.0, 0.0, 1.0)
    costs = np.where(
        unknown & ~blocked,
        np.maximum(costs, unknown_cost),
        costs,
    )
    return GuideGrid(
        costs=costs,
        blocked=blocked,
        resolution_m=float(resolution_m) * stride,
        origin_x_m=float(origin_x_m),
        origin_y_m=float(origin_y_m),
    )


def _disk_cells(center: GridCell, radius_cells: int):
    cx, cy = center
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                yield cx + dx, cy + dy


def clear_start_footprint(
    grid: GuideGrid,
    start: GridCell,
    radius_m: float,
):
    """Release non-lethal start cells without erasing real obstacles.

    Cost 99 may surround the current valid pose after the 0.25 m Nav2 grid is
    max-pooled into the coarser FAR grid.  Releasing that band only near the
    current pose lets the guide leave the artefact.  Cost 100 remains blocked,
    and released cells retain their high cost so A* crosses as little of the
    band as possible.
    """
    blocked = np.array(grid.blocked, copy=True)
    costs = np.array(grid.costs, copy=True)
    radius_cells = max(0, int(math.ceil(radius_m / grid.resolution_m)))
    for cell in _disk_cells(start, radius_cells):
        if grid.contains(cell):
            if costs[cell[1], cell[0]] < 1.0:
                blocked[cell[1], cell[0]] = False
    return GuideGrid(
        costs,
        blocked,
        grid.resolution_m,
        grid.origin_x_m,
        grid.origin_y_m,
    )


def nearest_free_cell(
    grid: GuideGrid,
    requested: GridCell,
    maximum_radius_m: float,
):
    """Find the nearest unblocked cell to a requested grid location."""
    if not grid.contains(requested):
        return None
    if not grid.blocked[requested[1], requested[0]]:
        return requested
    maximum = max(0, int(math.ceil(
        maximum_radius_m / grid.resolution_m
    )))
    best = None
    best_squared = None
    for radius in range(1, maximum + 1):
        for cell in _disk_cells(requested, radius):
            if not grid.contains(cell):
                continue
            dx = cell[0] - requested[0]
            dy = cell[1] - requested[1]
            squared = dx * dx + dy * dy
            if squared > radius * radius:
                continue
            if grid.blocked[cell[1], cell[0]]:
                continue
            if best_squared is None or squared < best_squared:
                best = cell
                best_squared = squared
        if best is not None:
            return best
    return None


def clip_goal_to_grid(
    grid: GuideGrid,
    start_world: WorldPoint,
    goal_world: WorldPoint,
    boundary_margin_m: float,
):
    """Clip a distant mission goal to the current rolling-grid boundary."""
    half_cell = 0.5 * grid.resolution_m
    minimum_x = grid.origin_x_m + boundary_margin_m + half_cell
    minimum_y = grid.origin_y_m + boundary_margin_m + half_cell
    maximum_x = (
        grid.origin_x_m + grid.width * grid.resolution_m
        - boundary_margin_m - half_cell
    )
    maximum_y = (
        grid.origin_y_m + grid.height * grid.resolution_m
        - boundary_margin_m - half_cell
    )
    if minimum_x >= maximum_x or minimum_y >= maximum_y:
        raise ValueError('guide grid is smaller than its boundary margins')
    gx, gy = float(goal_world[0]), float(goal_world[1])
    if minimum_x <= gx <= maximum_x and minimum_y <= gy <= maximum_y:
        return gx, gy
    sx, sy = float(start_world[0]), float(start_world[1])
    dx, dy = gx - sx, gy - sy
    if abs(dx) + abs(dy) < 1e-9:
        return (
            min(max(gx, minimum_x), maximum_x),
            min(max(gy, minimum_y), maximum_y),
        )
    scales = [1.0]
    if dx > 0.0:
        scales.append((maximum_x - sx) / dx)
    elif dx < 0.0:
        scales.append((minimum_x - sx) / dx)
    if dy > 0.0:
        scales.append((maximum_y - sy) / dy)
    elif dy < 0.0:
        scales.append((minimum_y - sy) / dy)
    positive = [scale for scale in scales if 0.0 <= scale <= 1.0]
    scale = min(positive) if positive else 0.0
    return sx + scale * dx, sy + scale * dy


_NEIGHBORS = (
    (-1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
)


def _heuristic(first: GridCell, second: GridCell):
    return math.hypot(second[0] - first[0], second[1] - first[1])


def astar_grid_path(
    grid: GuideGrid,
    start: GridCell,
    goal: GridCell,
    cost_weight: float = 3.0,
    maximum_expansions: int = 120000,
    reachable_frontier_margin_cells: int = 0,
    minimum_frontier_progress_cells: float = 0.0,
):
    """Find a coarse 8-connected route through the online costmap.

    A rolling map can show the distant goal in a free component that is not
    connected to the vehicle yet.  When ``reachable_frontier_margin_cells``
    is positive and the exact goal cannot be reached, return a path to the
    reachable map frontier that is closest to the requested goal.  This lets
    the vehicle reveal the next part of a mapless route instead of reporting
    ``guide_not_found`` forever.  The fallback is deliberately disabled by
    default and is only used for goals clipped to a rolling-map boundary.
    """
    if not grid.contains(start) or not grid.contains(goal):
        return None
    if grid.blocked[start[1], start[0]] or grid.blocked[goal[1], goal[0]]:
        return None
    if start == goal:
        return [start]
    frontier = [(float(_heuristic(start, goal)), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    expansions = 0

    def reconstruct(last):
        path = [last]
        while last in came_from:
            last = came_from[last]
            path.append(last)
        path.reverse()
        return path

    while frontier and expansions < maximum_expansions:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > g_score.get(current, float('inf')) + 1e-9:
            continue
        if current == goal:
            return reconstruct(current)
        expansions += 1
        for dx, dy, step in _NEIGHBORS:
            neighbor = current[0] + dx, current[1] + dy
            if not grid.contains(neighbor):
                continue
            if grid.blocked[neighbor[1], neighbor[0]]:
                continue
            if dx != 0 and dy != 0:
                if (
                    grid.blocked[current[1], neighbor[0]]
                    or grid.blocked[neighbor[1], current[0]]
                ):
                    continue
            penalty = float(grid.costs[neighbor[1], neighbor[0]])
            tentative = current_cost + step * (1.0 + cost_weight * penalty)
            if tentative + 1e-9 >= g_score.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            priority = tentative + _heuristic(neighbor, goal)
            heapq.heappush(frontier, (priority, tentative, neighbor))

    margin = int(reachable_frontier_margin_cells)
    if margin <= 0:
        return None
    margin = min(margin, max(0, (min(grid.width, grid.height) - 1) // 2))
    minimum_progress = max(0.0, float(minimum_frontier_progress_cells))
    candidates = []
    for cell, travel_cost in g_score.items():
        if _heuristic(start, cell) < minimum_progress:
            continue
        x, y = cell
        on_frontier = (
            x <= margin
            or y <= margin
            or x >= grid.width - 1 - margin
            or y >= grid.height - 1 - margin
        )
        if not on_frontier:
            continue
        # Goal proximity selects the correct side of the reachable component;
        # the small travel-cost term breaks ties in favour of a clean route.
        score = _heuristic(cell, goal) + 0.05 * float(travel_cost)
        candidates.append((score, float(travel_cost), cell))
    if candidates:
        _, _, best = min(candidates)
        return reconstruct(best)
    return None


def _bresenham(first: GridCell, second: GridCell) -> Iterable[GridCell]:
    x0, y0 = first
    x1, y1 = second
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def line_of_sight(grid: GuideGrid, first: GridCell, second: GridCell):
    return all(
        grid.contains(cell) and not grid.blocked[cell[1], cell[0]]
        for cell in _bresenham(first, second)
    )


def visibility_prune(grid: GuideGrid, path: Sequence[GridCell]):
    """Reduce a dense A* path to mutually visible turning vertices."""
    if len(path) <= 2:
        return list(path)
    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        farthest = anchor + 1
        for candidate in range(anchor + 2, len(path)):
            if not line_of_sight(grid, path[anchor], path[candidate]):
                break
            farthest = candidate
        result.append(path[farthest])
        anchor = farthest
    return result


def select_subgoal(
    world_path: Sequence[WorldPoint],
    lookahead_m: float,
):
    """Interpolate a subgoal and tangent on a world-frame guide path."""
    if not world_path:
        raise ValueError('cannot select a subgoal from an empty path')
    if lookahead_m <= 0.0:
        raise ValueError('subgoal lookahead must be positive')
    if len(world_path) == 1:
        return world_path[0], 0.0, True
    remaining = float(lookahead_m)
    for index in range(len(world_path) - 1):
        first = world_path[index]
        second = world_path[index + 1]
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        yaw = math.atan2(dy, dx)
        if remaining <= length:
            ratio = remaining / length
            return (
                (first[0] + ratio * dx, first[1] + ratio * dy),
                yaw,
                False,
            )
        remaining -= length
    final = world_path[-1]
    previous = world_path[-2]
    yaw = math.atan2(final[1] - previous[1], final[0] - previous[0])
    return final, yaw, True


def plan_online_guide(
    grid: GuideGrid,
    start_world: WorldPoint,
    goal_world: WorldPoint,
    boundary_margin_m: float,
    start_clearance_m: float,
    goal_search_radius_m: float,
    cost_weight: float,
    maximum_expansions: int,
    continuity_heading_rad=None,
    continuity_distance_m: float = 0.0,
    continuity_cost_weight: float = 0.0,
    failed_corridors=(),
    failed_corridor_radius_m: float = 0.0,
    failed_corridor_cost_weight: float = 0.0,
    failed_corridor_start_release_m: float = 0.0,
    failed_corridor_hard_block_radius_m: float = 0.0,
    reachable_frontier_minimum_progress_m: float = 2.0,
):
    """Plan and visibility-prune one online long-range guide."""
    clipped_goal = clip_goal_to_grid(
        grid, start_world, goal_world, boundary_margin_m
    )
    start_cell = grid.world_to_cell(start_world)
    if not grid.contains(start_cell):
        return None
    planning_grid = clear_start_footprint(
        grid, start_cell, start_clearance_m
    )
    if continuity_heading_rad is not None:
        planning_grid = add_direction_continuity_cost(
            planning_grid,
            start_world,
            float(continuity_heading_rad),
            continuity_distance_m,
            continuity_cost_weight,
        )
    planning_grid = add_failed_corridor_cost(
        planning_grid,
        failed_corridors,
        start_world,
        failed_corridor_radius_m,
        failed_corridor_cost_weight,
        failed_corridor_start_release_m,
        failed_corridor_hard_block_radius_m,
    )
    requested_goal_cell = planning_grid.world_to_cell(clipped_goal)
    goal_cell = requested_goal_cell
    goal_cell = nearest_free_cell(
        planning_grid, goal_cell, goal_search_radius_m
    )
    if goal_cell is None:
        return None
    # The exact goal can lie inside a rolling costmap while still belonging
    # to a free-space component that is not connected to the vehicle.  This
    # happens in mapless navigation when a building/wall divides the current
    # observation: the far side is represented in the costmap, but the route
    # around the end has not been revealed yet.  Trying only the exact goal in
    # that case leaves the vehicle in ``guide_not_found`` forever.
    #
    # Always make the reachable-frontier fallback available.  A* still tries
    # the exact (possibly clipped) goal first, so connected F9/F10 routes are
    # unchanged.  The frontier is used only after exact-goal search fails and
    # only when the vehicle's free component actually reaches the rolling-map
    # boundary by at least the configured minimum progress.
    frontier_margin_cells = max(1, int(math.ceil(
        boundary_margin_m / planning_grid.resolution_m
    )))
    dense = astar_grid_path(
        planning_grid,
        start_cell,
        goal_cell,
        cost_weight=cost_weight,
        maximum_expansions=maximum_expansions,
        reachable_frontier_margin_cells=frontier_margin_cells,
        minimum_frontier_progress_cells=(
            reachable_frontier_minimum_progress_m
            / planning_grid.resolution_m
        ),
    )
    if not dense:
        return None
    sparse = visibility_prune(planning_grid, dense)
    world = [planning_grid.cell_to_world(cell) for cell in sparse]
    world[0] = (float(start_world[0]), float(start_world[1]))
    if dense[-1] == goal_cell and math.hypot(
        clipped_goal[0] - goal_world[0], clipped_goal[1] - goal_world[1]
    ) < planning_grid.resolution_m and goal_cell == requested_goal_cell:
        world[-1] = (float(goal_world[0]), float(goal_world[1]))
    return world
