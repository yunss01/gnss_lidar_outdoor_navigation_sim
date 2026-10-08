"""Pure geometry helpers for the GNSS-to-Nav2 goal bridge."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple, TypeVar


T = TypeVar('T')


@dataclass(frozen=True)
class PlanarPose:
    """A planar target pose expressed in the odom frame."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class WaypointPassage:
    """Geometric relationship between a vehicle and an ordered waypoint."""

    passed: bool
    reason: str
    distance_m: float
    along_track_m: float
    cross_track_m: float


def evaluate_waypoint_passage(
    previous_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    current_xy: Tuple[float, float],
    capture_radius_m: float,
    crossing_lateral_limit_m: float,
    crossing_armed: bool,
) -> WaypointPassage:
    """Accept a waypoint by radius or by a correctly directed crossing.

    ``along_track_m`` is negative before the plane through the waypoint and
    positive after it.  The crossing alternative is only valid after the
    caller observed the vehicle on the approach side (``crossing_armed``),
    preventing a route that starts beyond a waypoint from skipping it.
    """
    values = (
        *previous_xy,
        *target_xy,
        *current_xy,
        capture_radius_m,
        crossing_lateral_limit_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('waypoint passage inputs must be finite')
    if capture_radius_m <= 0.0:
        raise ValueError('capture radius must be positive')
    if crossing_lateral_limit_m <= 0.0:
        raise ValueError('crossing lateral limit must be positive')

    segment_x = float(target_xy[0]) - float(previous_xy[0])
    segment_y = float(target_xy[1]) - float(previous_xy[1])
    segment_length = math.hypot(segment_x, segment_y)
    if segment_length <= 1.0e-6:
        raise ValueError('waypoint approach segment must have length')
    unit_x = segment_x / segment_length
    unit_y = segment_y / segment_length
    relative_x = float(current_xy[0]) - float(target_xy[0])
    relative_y = float(current_xy[1]) - float(target_xy[1])
    distance = math.hypot(relative_x, relative_y)
    along_track = relative_x * unit_x + relative_y * unit_y
    cross_track = abs(relative_x * unit_y - relative_y * unit_x)

    if distance <= float(capture_radius_m):
        reason = 'radius'
        passed = True
    elif (
        crossing_armed
        and along_track >= 0.0
        and cross_track <= float(crossing_lateral_limit_m)
    ):
        reason = 'directed_crossing'
        passed = True
    else:
        reason = 'approaching' if along_track < 0.0 else 'outside_corridor'
        passed = False
    return WaypointPassage(
        passed=passed,
        reason=reason,
        distance_m=distance,
        along_track_m=along_track,
        cross_track_m=cross_track,
    )


def rolling_initial_start_index(
    waypoint_xy: Sequence[Tuple[float, float]],
    current_xy: Tuple[float, float],
    capture_radius_m: float,
) -> int:
    """Skip only an initial waypoint already occupied by the vehicle.

    F9 and F10 share the saved ``WP1 -> ... -> WPn`` prefix.  A closed lap
    appends WP1 at the end, so a vehicle staged at WP1 must not be asked to
    execute a near-zero first maneuver.  Returning index one preserves WP1 as
    the geometric predecessor of WP2 while making WP2 the first motion target.
    """
    if len(waypoint_xy) < 2:
        raise ValueError('rolling route requires at least two waypoints')
    values = (*current_xy, capture_radius_m)
    values += tuple(value for point in waypoint_xy for value in point)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('rolling route start inputs must be finite')
    if capture_radius_m <= 0.0:
        raise ValueError('initial capture radius must be positive')
    first = waypoint_xy[0]
    distance = math.hypot(
        float(first[0]) - float(current_xy[0]),
        float(first[1]) - float(current_xy[1]),
    )
    return 1 if distance <= float(capture_radius_m) else 0


def odom_target_from_goal_vector(
    odom_x_m: float,
    odom_y_m: float,
    goal_east_m: float,
    goal_north_m: float,
) -> PlanarPose:
    """Convert an ENU goal vector at the current pose into an odom target."""
    values = (odom_x_m, odom_y_m, goal_east_m, goal_north_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('odometry and goal vector values must be finite')
    return PlanarPose(
        x_m=odom_x_m + goal_east_m,
        y_m=odom_y_m + goal_north_m,
        yaw_rad=math.atan2(goal_north_m, goal_east_m),
    )


def goal_signature_changed(
    previous_xy,
    current_xy,
    tolerance_m: float,
) -> bool:
    """Return true when a new local ENU goal should replace the Nav2 goal."""
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError('goal signature tolerance must be positive')
    if previous_xy is None:
        return True
    if len(previous_xy) != 2 or len(current_xy) != 2:
        raise ValueError('goal signatures must contain x and y')
    return math.hypot(
        float(current_xy[0]) - float(previous_xy[0]),
        float(current_xy[1]) - float(previous_xy[1]),
    ) > tolerance_m


def should_promote_near_goal_route_abort(
    status: int,
    poses_remaining,
    goal_vector,
    goal_vector_age_s,
    success_radius_m: float,
    input_timeout_s: float,
) -> bool:
    """Recognize a planner replan abort after the GPS goal was achieved."""
    if int(status) != 6 or poses_remaining is None:
        return False
    if int(poses_remaining) > 1 or goal_vector is None:
        return False
    if goal_vector_age_s is None or float(goal_vector_age_s) > input_timeout_s:
        return False
    try:
        distance = math.hypot(float(goal_vector[0]), float(goal_vector[1]))
    except (TypeError, ValueError, IndexError):
        return False
    return math.isfinite(distance) and distance <= float(success_radius_m)


def should_retry_rolling_preview_as_current_only(
    status: int,
    route_pose_count: int,
    route_window_start,
    already_focused: bool,
) -> bool:
    """Retry an aborted preview window without its next waypoint.

    A ``NavigateThroughPoses`` request containing the current waypoint and a
    preview waypoint is rejected as a whole when the preview cannot yet be
    connected in the rolling costmap.  The current waypoint can still be
    reachable.  Retry only planner/controller aborts of multi-pose rolling
    windows, and only once for a given current waypoint.
    """
    if route_window_start is None:
        return False
    if int(status) != 6 or int(route_pose_count) <= 1:
        return False
    return not bool(already_focused)


def should_retry_aborted_rolling_segment(
    status: int,
    is_current: bool,
    goal_mode: str,
    route_window_start,
    route_available: bool,
    pending_goal_exists: bool,
) -> bool:
    """Keep an F9/F10 mission alive after one Nav2 segment aborts.

    A failed required replan must stop the controller, but it must not discard
    the saved route.  Only retry the currently authoritative one-pose rolling
    segment.  Canceled/superseded goals and explicit waypoint handoffs already
    have a pending replacement and must never be resurrected here.
    """
    return bool(
        int(status) == 6
        and is_current
        and goal_mode == 'rolling_segment'
        and route_window_start is not None
        and route_available
        and not pending_goal_exists
    )


def should_handoff_virtual_preview_after_waypoint(
    route_pose_count: int,
    route_real_pose_count: int,
    route_window_start,
    passed_waypoint_index: int,
    route_total_count: int,
) -> bool:
    """Discard a direction-only preview after its real waypoint is passed.

    A virtual preview gives an Ackermann planner an outgoing tangent while the
    next mission waypoint is outside the rolling horizon.  It is not itself a
    mission waypoint.  Once the sole real pose in that action is passed, the
    bridge must replace the action with a window beginning at the next real
    waypoint instead of forcing the vehicle to reach the synthetic endpoint.
    """
    if route_window_start is None:
        return False
    values = (
        route_pose_count,
        route_real_pose_count,
        route_window_start,
        passed_waypoint_index,
        route_total_count,
    )
    if not all(isinstance(value, int) for value in values):
        raise ValueError('rolling route indices and counts must be integers')
    if route_pose_count < 1 or route_real_pose_count < 1:
        raise ValueError('rolling route pose counts must be positive')
    if route_real_pose_count > route_pose_count:
        raise ValueError('real pose count cannot exceed action pose count')
    if not 0 <= route_window_start < route_total_count:
        raise ValueError('rolling route start index is out of range')
    if not 0 <= passed_waypoint_index < route_total_count:
        raise ValueError('passed waypoint index is out of range')
    return bool(
        route_pose_count > route_real_pose_count
        and route_real_pose_count == 1
        and passed_waypoint_index == route_window_start
        and passed_waypoint_index + 1 < route_total_count
    )


def odom_route_from_enu_points(
    odom_x_m: float,
    odom_y_m: float,
    current_east_m: float,
    current_north_m: float,
    waypoint_enu: Iterable[Tuple[float, float]],
    use_closed_route_tangents: bool = False,
    use_corner_tangents: bool = False,
) -> List[PlanarPose]:
    """Anchor a complete ENU waypoint route in the current odom frame.

    The GNSS manager's local ENU frame and CARLA/vehicle odometry may have
    different origins.  Their axes are aligned, so one current-position pair
    is sufficient to translate every future waypoint into odometry.  Each
    waypoint yaw follows the direction in which the vehicle approaches that
    point.  A waypoint is a positional guide, not a demand to instantaneously
    assume the following segment's heading.  The latter makes a nearby first
    waypoint kinematically expensive for a forward-only Ackermann planner.
    """
    anchor = (odom_x_m, odom_y_m, current_east_m, current_north_m)
    if not all(math.isfinite(float(value)) for value in anchor):
        raise ValueError('route anchor values must be finite')
    points = [(float(east), float(north)) for east, north in waypoint_enu]
    if not points:
        raise ValueError('waypoint route must not be empty')
    if not all(
        math.isfinite(east) and math.isfinite(north)
        for east, north in points
    ):
        raise ValueError('waypoint ENU coordinates must be finite')

    odom_points = [
        (
            float(odom_x_m) + east - float(current_east_m),
            float(odom_y_m) + north - float(current_north_m),
        )
        for east, north in points
    ]
    if use_closed_route_tangents:
        return _closed_route_tangent_poses(odom_points)
    if use_corner_tangents:
        return _open_route_tangent_poses(
            odom_points,
            (float(odom_x_m), float(odom_y_m)),
        )

    poses = []
    previous_x = float(odom_x_m)
    previous_y = float(odom_y_m)
    for index, (x_value, y_value) in enumerate(odom_points):
        if index == 0:
            delta_x = x_value - float(odom_x_m)
            delta_y = y_value - float(odom_y_m)
        else:
            delta_x = x_value - previous_x
            delta_y = y_value - previous_y
        if math.hypot(delta_x, delta_y) <= 1.0e-6:
            # A closed lap can repeat WP1 as its final point.  Preserve the
            # previous segment direction instead of producing atan2(0, 0).
            if poses:
                yaw = poses[-1].yaw_rad
            else:
                yaw = 0.0
        else:
            yaw = math.atan2(delta_y, delta_x)
        poses.append(PlanarPose(x_value, y_value, yaw))
        previous_x = x_value
        previous_y = y_value
    return poses


def _open_route_tangent_poses(
    odom_points: Sequence[Tuple[float, float]],
    start_xy: Tuple[float, float],
) -> List[PlanarPose]:
    """Assign continuous Ackermann headings to an ordered open route.

    A Hybrid-A* through-poses plan treats every pose orientation as a real
    kinematic constraint.  Giving a corner waypoint only its incoming heading
    and the following preview pose its outgoing heading demands an impossible
    instantaneous heading change at the shared waypoint.  The normalized
    incoming/outgoing bisector rounds that corner while preserving the manual
    GNSS waypoint position.  The final waypoint keeps its incoming heading.
    """
    poses = []
    for index, (x_value, y_value) in enumerate(odom_points):
        previous = start_xy if index == 0 else odom_points[index - 1]
        incoming = (x_value - previous[0], y_value - previous[1])
        incoming_length = math.hypot(*incoming)
        if incoming_length <= 1.0e-6:
            if poses:
                incoming = (
                    math.cos(poses[-1].yaw_rad),
                    math.sin(poses[-1].yaw_rad),
                )
                incoming_length = 1.0
            elif len(odom_points) > 1:
                incoming = (
                    odom_points[1][0] - x_value,
                    odom_points[1][1] - y_value,
                )
                incoming_length = math.hypot(*incoming)
            if incoming_length <= 1.0e-6:
                incoming = (1.0, 0.0)
                incoming_length = 1.0

        tangent_x = incoming[0] / incoming_length
        tangent_y = incoming[1] / incoming_length
        if index + 1 < len(odom_points):
            following = odom_points[index + 1]
            outgoing = (following[0] - x_value, following[1] - y_value)
            outgoing_length = math.hypot(*outgoing)
            if outgoing_length > 1.0e-6:
                bisector_x = tangent_x + outgoing[0] / outgoing_length
                bisector_y = tangent_y + outgoing[1] / outgoing_length
                if math.hypot(bisector_x, bisector_y) > 1.0e-6:
                    tangent_x = bisector_x
                    tangent_y = bisector_y

        poses.append(PlanarPose(
            x_m=x_value,
            y_m=y_value,
            yaw_rad=math.atan2(tangent_y, tangent_x),
        ))
    return poses


def _closed_route_tangent_poses(
    odom_points: Sequence[Tuple[float, float]],
) -> List[PlanarPose]:
    """Assign continuous corner tangents to a route ending at its start.

    Manually captured GNSS waypoints describe positions, not parking poses.
    For a one-lap route, using only the incoming segment heading at every
    corner forces a forward-only planner to reach the corner and then change
    direction abruptly.  The normalized incoming/outgoing angle bisector is
    the tangent of the intended rounded corner and gives the Ackermann
    planner useful look-ahead without changing waypoint positions.
    """
    if len(odom_points) < 4:
        raise ValueError('closed tangent route requires at least 3 waypoints')
    if math.hypot(
        odom_points[-1][0] - odom_points[0][0],
        odom_points[-1][1] - odom_points[0][1],
    ) > 1.0e-3:
        raise ValueError('closed tangent route must repeat its first point')

    unique_points = list(odom_points[:-1])
    poses = []
    for index, (x_value, y_value) in enumerate(unique_points):
        previous = unique_points[(index - 1) % len(unique_points)]
        following = unique_points[(index + 1) % len(unique_points)]
        incoming = (x_value - previous[0], y_value - previous[1])
        outgoing = (following[0] - x_value, following[1] - y_value)
        incoming_length = math.hypot(*incoming)
        outgoing_length = math.hypot(*outgoing)
        if incoming_length <= 1.0e-6 or outgoing_length <= 1.0e-6:
            raise ValueError('closed tangent route has duplicate neighbors')
        tangent_x = (
            incoming[0] / incoming_length + outgoing[0] / outgoing_length
        )
        tangent_y = (
            incoming[1] / incoming_length + outgoing[1] / outgoing_length
        )
        if math.hypot(tangent_x, tangent_y) <= 1.0e-6:
            # A 180-degree cusp has no unique angle bisector. Preserve the
            # outgoing direction; the planner can still report infeasibility.
            tangent_x, tangent_y = outgoing
        poses.append(PlanarPose(
            x_m=x_value,
            y_m=y_value,
            yaw_rad=math.atan2(tangent_y, tangent_x),
        ))
    poses.append(PlanarPose(
        x_m=poses[0].x_m,
        y_m=poses[0].y_m,
        yaw_rad=poses[0].yaw_rad,
    ))
    return poses


def rolling_horizon_window(
    values: Sequence[T],
    start_index: int,
    window_size: int = 2,
) -> List[T]:
    """Return the current and next route items for bounded Nav2 planning."""
    if window_size < 1:
        raise ValueError('rolling horizon window size must be at least 1')
    if start_index < 0 or start_index >= len(values):
        raise ValueError('rolling horizon start index is out of range')
    return list(values[start_index:start_index + window_size])


def should_focus_rolling_waypoint(
    distance_m: float,
    active_window_count: int,
    already_focused: bool,
    focus_distance_m: float,
) -> bool:
    """Return whether a preview window should become current-WP-only.

    A pure-pursuit controller can cut an intermediate waypoint when it sees
    the following route segment in the same action.  Near that waypoint, a
    single-pose window makes the current waypoint the actual Nav2 endpoint.
    """
    values = (distance_m, focus_distance_m)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('rolling focus distances must be finite')
    if distance_m < 0.0:
        raise ValueError('rolling focus distance cannot be negative')
    if focus_distance_m <= 0.0:
        raise ValueError('rolling focus threshold must be positive')
    if active_window_count < 1:
        raise ValueError('active rolling window must contain a pose')
    return bool(
        active_window_count >= 2
        and not already_focused
        and distance_m <= focus_distance_m
    )


def rolling_waypoint_turn_angle_deg(
    previous_xy: Tuple[float, float],
    waypoint_xy: Tuple[float, float],
    next_xy: Tuple[float, float],
) -> float:
    """Return the unsigned heading change at an ordered waypoint.

    Zero degrees is a straight route and 180 degrees is a reversal.  The
    bridge uses this value to keep the successor preview on real corners;
    converting a corner to a single-pose goal destroys the outgoing tangent
    that an Ackermann controller needs before it reaches the waypoint.
    """
    values = (*previous_xy, *waypoint_xy, *next_xy)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('rolling waypoint geometry must be finite')
    incoming = (
        float(waypoint_xy[0]) - float(previous_xy[0]),
        float(waypoint_xy[1]) - float(previous_xy[1]),
    )
    outgoing = (
        float(next_xy[0]) - float(waypoint_xy[0]),
        float(next_xy[1]) - float(waypoint_xy[1]),
    )
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if incoming_length <= 1.0e-6 or outgoing_length <= 1.0e-6:
        raise ValueError('rolling waypoint neighbors must be distinct')
    cosine = (
        incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    ) / (incoming_length * outgoing_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def rolling_waypoint_capture_radius(
    turn_angle_deg: float | None,
    normal_radius_m: float,
    corner_radius_m: float,
    corner_threshold_deg: float,
) -> float:
    """Choose a slightly wider capture radius only on real corners."""
    values = (normal_radius_m, corner_radius_m, corner_threshold_deg)
    if turn_angle_deg is not None:
        values = (*values, turn_angle_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('rolling capture-radius inputs must be finite')
    if normal_radius_m <= 0.0 or corner_radius_m < normal_radius_m:
        raise ValueError('corner radius must be at least the normal radius')
    if not 0.0 <= corner_threshold_deg <= 180.0:
        raise ValueError('corner threshold must be 0-180 deg')
    if turn_angle_deg is not None and not 0.0 <= turn_angle_deg <= 180.0:
        raise ValueError('turn angle must be 0-180 deg')
    return float(
        corner_radius_m
        if turn_angle_deg is not None
        and turn_angle_deg > corner_threshold_deg
        else normal_radius_m
    )


def bounded_virtual_preview(
    current_xy: Tuple[float, float],
    waypoint_xy: Tuple[float, float],
    next_xy: Tuple[float, float],
    desired_extension_m: float,
    maximum_distance_m: float,
    boundary_margin_m: float,
    minimum_extension_m: float,
) -> Tuple[float, float] | None:
    """Extend a waypoint toward its successor inside a rolling horizon.

    This is a direction-only preview, not a new mission waypoint.  It gives
    the planner an outgoing tangent when the real successor is outside the
    local costmap.  The returned point is bounded by
    ``maximum_distance_m - boundary_margin_m`` from the current vehicle.
    """
    values = (
        *current_xy,
        *waypoint_xy,
        *next_xy,
        desired_extension_m,
        maximum_distance_m,
        boundary_margin_m,
        minimum_extension_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('virtual preview inputs must be finite')
    if desired_extension_m <= 0.0 or minimum_extension_m <= 0.0:
        raise ValueError('virtual preview extensions must be positive')
    if maximum_distance_m <= 0.0 or boundary_margin_m < 0.0:
        raise ValueError('virtual preview horizon is invalid')
    usable_radius = float(maximum_distance_m) - float(boundary_margin_m)
    if usable_radius <= 0.0:
        raise ValueError('virtual preview margin consumes the horizon')

    direction = (
        float(next_xy[0]) - float(waypoint_xy[0]),
        float(next_xy[1]) - float(waypoint_xy[1]),
    )
    direction_length = math.hypot(*direction)
    if direction_length <= 1.0e-6:
        return None
    unit = (
        direction[0] / direction_length,
        direction[1] / direction_length,
    )
    offset = (
        float(waypoint_xy[0]) - float(current_xy[0]),
        float(waypoint_xy[1]) - float(current_xy[1]),
    )
    if math.hypot(*offset) >= usable_radius:
        return None

    # Positive ray/circle intersection measured from waypoint_xy.
    projection = offset[0] * unit[0] + offset[1] * unit[1]
    discriminant = (
        projection * projection
        + usable_radius * usable_radius
        - offset[0] * offset[0]
        - offset[1] * offset[1]
    )
    maximum_extension = -projection + math.sqrt(max(0.0, discriminant))
    extension = min(
        float(desired_extension_m),
        direction_length,
        maximum_extension,
    )
    if extension < float(minimum_extension_m):
        return None
    return (
        float(waypoint_xy[0]) + extension * unit[0],
        float(waypoint_xy[1]) + extension * unit[1],
    )


def distance_bounded_rolling_horizon_count(
    points_xy: Sequence[Tuple[float, float]],
    start_index: int,
    current_xy: Tuple[float, float],
    maximum_window_size: int,
    maximum_distance_m: float,
) -> int:
    """Count contiguous route targets that fit inside a local map horizon.

    The first target is not forced into the result when it is outside the
    horizon.  Sending even one out-of-bounds target makes Nav2 reject the
    complete ``NavigateThroughPoses`` request, so the caller must wait or
    report that the ordered route is too sparse.
    """
    if start_index < 0 or start_index >= len(points_xy):
        raise ValueError('rolling horizon start index is out of range')
    if maximum_window_size < 1:
        raise ValueError(
            'rolling horizon maximum window size must be positive'
        )
    if (
        not math.isfinite(float(maximum_distance_m))
        or maximum_distance_m <= 0.0
    ):
        raise ValueError('rolling horizon maximum distance must be positive')
    values = (*current_xy,)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('rolling horizon current position must be finite')

    count = 0
    stop_index = min(len(points_xy), start_index + maximum_window_size)
    for index in range(start_index, stop_index):
        point = points_xy[index]
        if len(point) != 2 or not all(
            math.isfinite(float(value)) for value in point
        ):
            raise ValueError('rolling horizon target position must be finite')
        distance = math.hypot(
            float(point[0]) - float(current_xy[0]),
            float(point[1]) - float(current_xy[1]),
        )
        if distance > float(maximum_distance_m):
            break
        count += 1
    return count


def rolling_horizon_global_remaining(
    total_count: int,
    start_index: int,
    window_length: int,
    window_remaining: int,
) -> int:
    """Convert one rolling action's feedback to whole-route progress."""
    if total_count < 1:
        raise ValueError('rolling route must contain at least one pose')
    if start_index < 0 or start_index >= total_count:
        raise ValueError('rolling horizon start index is out of range')
    if window_length < 1 or start_index + window_length > total_count:
        raise ValueError('rolling horizon length is out of range')
    if window_remaining < 0 or window_remaining > window_length:
        raise ValueError('rolling horizon remaining count is out of range')
    passed_in_window = window_length - window_remaining
    return max(0, total_count - start_index - passed_in_window)


def rolling_horizon_next_start(
    total_count: int,
    start_index: int,
    window_length: int,
    window_remaining: int,
) -> int | None:
    """Advance after the first pose passes while retaining one-pose overlap."""
    rolling_horizon_global_remaining(
        total_count,
        start_index,
        window_length,
        window_remaining,
    )
    if window_length < 2 or window_remaining != 1:
        return None
    next_start = start_index + window_length - 1
    if next_start + 1 >= total_count:
        # The active action already contains the final pose. Let it finish so
        # the final Nav2 result remains the route completion authority.
        return None
    return next_start


def rolling_horizon_result_next_start(
    total_count: int,
    start_index: int,
    window_length: int,
) -> int | None:
    """Choose the next window after one rolling action finishes.

    A successful action has reached every *real* waypoint in its window.
    Therefore the next action starts after the entire completed window.  It
    must never resend the final completed waypoint: doing so can ask a
    forward-only Ackermann vehicle to turn back toward a point behind it.
    """
    if total_count < 1:
        raise ValueError('rolling route must contain at least one pose')
    if start_index < 0 or start_index >= total_count:
        raise ValueError('rolling horizon start index is out of range')
    if window_length < 1 or start_index + window_length > total_count:
        raise ValueError('rolling horizon length is out of range')
    if start_index + window_length >= total_count:
        return None
    next_start = start_index + window_length
    return next_start


def rolling_real_poses_remaining(
    action_pose_count: int,
    real_pose_count: int,
    action_poses_remaining: int,
) -> int:
    """Remove direction-only virtual previews from Nav2 feedback counts.

    Virtual previews are always appended after the real mission waypoints.
    ``NavigateThroughPoses`` includes them in ``number_of_poses_remaining``,
    while route progress must count only real waypoints.
    """
    if action_pose_count < 1:
        raise ValueError('rolling action must contain at least one pose')
    if real_pose_count < 1 or real_pose_count > action_pose_count:
        raise ValueError('rolling real-pose count is out of range')
    if (
        action_poses_remaining < 0
        or action_poses_remaining > action_pose_count
    ):
        raise ValueError('rolling action remaining count is out of range')
    virtual_pose_count = action_pose_count - real_pose_count
    return max(0, action_poses_remaining - virtual_pose_count)
