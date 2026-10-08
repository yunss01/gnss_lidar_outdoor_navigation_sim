"""Small, ROS-independent helpers for shadow localization evaluation."""

import math


def wrap_angle(angle):
    """Return an angle in the closed-open interval [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_yaw(quaternion):
    """Extract planar yaw from an object with ROS quaternion attributes."""
    x_value = float(quaternion.x)
    y_value = float(quaternion.y)
    z_value = float(quaternion.z)
    w_value = float(quaternion.w)
    sin_yaw = 2.0 * (w_value * z_value + x_value * y_value)
    cos_yaw = 1.0 - 2.0 * (y_value * y_value + z_value * z_value)
    return math.atan2(sin_yaw, cos_yaw)


def circular_mean_angle(angles):
    """Return the circular mean of one or more finite angles in radians."""
    values = [float(angle) for angle in angles]
    if not values:
        raise ValueError('at least one angle is required')
    if not all(math.isfinite(value) for value in values):
        raise ValueError('angles must be finite')
    sine = sum(math.sin(value) for value in values)
    cosine = sum(math.cos(value) for value in values)
    if math.hypot(sine, cosine) < 1e-12:
        raise ValueError('angles do not have a unique circular mean')
    return math.atan2(sine, cosine)


def base_position_from_antenna_delta(
        antenna_east_m, antenna_north_m,
        current_yaw_rad, reference_yaw_rad,
        antenna_forward_m, antenna_left_m):
    """Remove a planar GNSS antenna lever arm from a relative ENU position.

    The input ENU position is relative to the antenna's startup location.
    Offsets follow REP-103: forward is +x and left is +y in ``base_link``.
    Adding the startup lever arm after removing the current lever arm keeps
    the returned base position anchored at (0, 0) at initialization.
    """
    values = (
        antenna_east_m, antenna_north_m,
        current_yaw_rad, reference_yaw_rad,
        antenna_forward_m, antenna_left_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('antenna correction inputs must be finite')

    def rotate(forward_m, left_m, yaw_rad):
        cosine = math.cos(yaw_rad)
        sine = math.sin(yaw_rad)
        return (
            cosine * forward_m - sine * left_m,
            sine * forward_m + cosine * left_m,
        )

    current_offset = rotate(
        antenna_forward_m, antenna_left_m, current_yaw_rad)
    reference_offset = rotate(
        antenna_forward_m, antenna_left_m, reference_yaw_rad)
    return (
        float(antenna_east_m) - current_offset[0] + reference_offset[0],
        float(antenna_north_m) - current_offset[1] + reference_offset[1],
    )


def align_planar_pose(estimate, estimate_origin, reference_origin):
    """SE(2)-align an estimate to the first paired reference pose."""
    estimate_x, estimate_y, estimate_yaw = map(float, estimate)
    origin_x, origin_y, origin_yaw = map(float, estimate_origin)
    reference_x, reference_y, reference_yaw = map(float, reference_origin)
    yaw_offset = wrap_angle(reference_yaw - origin_yaw)
    cosine = math.cos(yaw_offset)
    sine = math.sin(yaw_offset)
    delta_x = estimate_x - origin_x
    delta_y = estimate_y - origin_y
    return (
        reference_x + cosine * delta_x - sine * delta_y,
        reference_y + sine * delta_x + cosine * delta_y,
        wrap_angle(estimate_yaw + yaw_offset),
    )


def align_enu_pose(estimate, estimate_origin, reference_origin):
    """Align an absolute-axis ENU pose without rotating its position axes.

    A GNSS-derived ENU filter already shares east/north axes with the
    reference frame.  Its initial translation is arbitrary, while an AHRS yaw
    can have a constant heading bias.  Applying that yaw correction to x/y
    would rotate an otherwise correct GNSS trajectory and manufacture a
    distance-dependent position error.  Position and heading therefore need
    independent alignment here.
    """
    estimate_x, estimate_y, estimate_yaw = map(float, estimate)
    origin_x, origin_y, origin_yaw = map(float, estimate_origin)
    reference_x, reference_y, reference_yaw = map(float, reference_origin)
    values = (
        estimate_x, estimate_y, estimate_yaw,
        origin_x, origin_y, origin_yaw,
        reference_x, reference_y, reference_yaw,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('ENU alignment inputs must be finite')
    return (
        reference_x + estimate_x - origin_x,
        reference_y + estimate_y - origin_y,
        wrap_angle(estimate_yaw + reference_yaw - origin_yaw),
    )


def planar_error_components(estimate_xy, reference_pose):
    """Return signed longitudinal/lateral position error in vehicle axes."""
    estimate_x, estimate_y = map(float, estimate_xy)
    reference_x, reference_y, reference_yaw = map(float, reference_pose)
    values = (
        estimate_x, estimate_y,
        reference_x, reference_y, reference_yaw,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('planar error inputs must be finite')
    delta_x = estimate_x - reference_x
    delta_y = estimate_y - reference_y
    cosine = math.cos(reference_yaw)
    sine = math.sin(reference_yaw)
    return (
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
    )


def relative_translation_error(current_pair, previous_pair):
    """Return error between estimate and reference translation increments."""
    estimate_x, estimate_y, reference_x, reference_y = map(
        float, current_pair)
    previous_estimate_x, previous_estimate_y, previous_reference_x, \
        previous_reference_y = map(float, previous_pair)
    values = (
        estimate_x, estimate_y, reference_x, reference_y,
        previous_estimate_x, previous_estimate_y,
        previous_reference_x, previous_reference_y,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('relative translation inputs must be finite')
    estimate_delta_x = estimate_x - previous_estimate_x
    estimate_delta_y = estimate_y - previous_estimate_y
    reference_delta_x = reference_x - previous_reference_x
    reference_delta_y = reference_y - previous_reference_y
    return math.hypot(
        estimate_delta_x - reference_delta_x,
        estimate_delta_y - reference_delta_y,
    )


def summarize_errors(values):
    """Return deterministic latest/RMSE/p95/max metrics for finite errors."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            'count': 0,
            'latest': None,
            'rmse': None,
            'p95': None,
            'maximum': None,
        }
    ordered = sorted(finite)
    percentile_index = int(math.ceil(0.95 * len(ordered))) - 1
    return {
        'count': len(finite),
        'latest': finite[-1],
        'rmse': math.sqrt(
            sum(value * value for value in finite) / len(finite)
        ),
        'p95': ordered[max(0, percentile_index)],
        'maximum': ordered[-1],
    }


def summarize_signed_errors(values):
    """Summarize signed residuals while reporting absolute tail magnitudes."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            'count': 0,
            'latest': None,
            'mean': None,
            'median': None,
            'rmse': None,
            'p95_absolute': None,
            'maximum_absolute': None,
        }
    ordered = sorted(finite)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])
    absolute = sorted(abs(value) for value in finite)
    percentile_index = int(math.ceil(0.95 * count)) - 1
    return {
        'count': count,
        'latest': finite[-1],
        'mean': sum(finite) / count,
        'median': median,
        'rmse': math.sqrt(
            sum(value * value for value in finite) / count
        ),
        'p95_absolute': absolute[max(0, percentile_index)],
        'maximum_absolute': absolute[-1],
    }


def assess_localization_transition(
        sample_count, estimate_age_s, gnss_age_s,
        position_p95_m, longitudinal_p95_m, lateral_p95_m,
        absolute_longitudinal_bias_m, relative_translation_p95_m,
        pose_jump_count, minimum_sample_count,
        maximum_estimate_age_s, maximum_gnss_age_s,
        maximum_position_p95_m, maximum_longitudinal_p95_m,
        maximum_lateral_p95_m, maximum_absolute_longitudinal_bias_m,
        maximum_relative_translation_p95_m, maximum_pose_jump_count,
        active_navigation_tf_ready=False,
        independent_velocity_ready=False):
    """Return explicit shadow-quality and active-control readiness gates."""
    blockers = []
    if int(sample_count) < int(minimum_sample_count):
        blockers.append('insufficient_samples')
    if estimate_age_s is None or estimate_age_s > maximum_estimate_age_s:
        blockers.append('estimate_stale')
    if gnss_age_s is None or gnss_age_s > maximum_gnss_age_s:
        blockers.append('gnss_odometry_stale')
    if position_p95_m is None or position_p95_m > maximum_position_p95_m:
        blockers.append('position_p95_exceeds_limit')
    if (
        longitudinal_p95_m is None
        or longitudinal_p95_m > maximum_longitudinal_p95_m
    ):
        blockers.append('longitudinal_p95_exceeds_limit')
    if lateral_p95_m is None or lateral_p95_m > maximum_lateral_p95_m:
        blockers.append('lateral_p95_exceeds_limit')
    if (
        absolute_longitudinal_bias_m is None
        or absolute_longitudinal_bias_m
        > maximum_absolute_longitudinal_bias_m
    ):
        blockers.append('longitudinal_bias_exceeds_limit')
    if (
        relative_translation_p95_m is None
        or relative_translation_p95_m
        > maximum_relative_translation_p95_m
    ):
        blockers.append('relative_translation_p95_exceeds_limit')
    if int(pose_jump_count) > int(maximum_pose_jump_count):
        blockers.append('pose_jump_count_exceeds_limit')
    metric_blockers = list(blockers)
    if not active_navigation_tf_ready:
        blockers.append('sensor_navigation_tf_not_ready')
    if not independent_velocity_ready:
        blockers.append('independent_velocity_not_ready')
    return {
        'shadow_quality_ready': not metric_blockers,
        'control_trial_ready': not blockers,
        'blockers': blockers,
    }
