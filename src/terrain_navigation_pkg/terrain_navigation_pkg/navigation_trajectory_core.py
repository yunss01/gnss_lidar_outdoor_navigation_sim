"""Pure helpers for building fixed-horizon local trajectory targets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


INITIAL_BC_EXCLUSION_FLAGS = frozenset({
    'no_valid_target',
    'winding_path',
})


@dataclass(frozen=True)
class TrajectoryAnalysis:
    """Fixed-spacing target and diagnostic values for one Nav2 path."""

    forward_path: np.ndarray
    target_points: np.ndarray
    target_mask: np.ndarray
    source_point_count: int
    forward_path_length_m: float
    vehicle_to_path_m: float
    maximum_source_gap_m: float
    maximum_curvature_per_m: float
    absolute_heading_change_rad: float
    backward_target_fraction: float
    goal_direction_error_rad: float
    flags: tuple[str, ...]


def initial_bc_trajectory_decision(flags) -> tuple[bool, tuple[str, ...]]:
    """Return the conservative first-stage imitation eligibility decision."""
    reasons = tuple(sorted(
        set(flags).intersection(INITIAL_BC_EXCLUSION_FLAGS)
    ))
    return not reasons, reasons


def _points_xy(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError('points must have shape (N, 2+)')
    array = array[:, :2]
    array = array[np.isfinite(array).all(axis=1)]
    if array.shape[0] < 2:
        return array
    keep = np.ones(array.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(array, axis=0), axis=1) > 1.0e-4
    return array[keep]


def forward_polyline_from_vehicle(points) -> tuple[np.ndarray, float]:
    """Project the vehicle origin onto a path and retain its forward suffix."""
    path = _points_xy(points)
    if path.shape[0] == 0:
        return path, math.inf
    if path.shape[0] == 1:
        return path.copy(), float(np.linalg.norm(path[0]))

    segment = np.diff(path, axis=0)
    length_sq = np.sum(segment * segment, axis=1)
    usable = length_sq > 1.0e-12
    projection = path[:-1].copy()
    parameter = np.zeros(segment.shape[0], dtype=np.float64)
    parameter[usable] = np.clip(
        -np.sum(path[:-1][usable] * segment[usable], axis=1)
        / length_sq[usable],
        0.0,
        1.0,
    )
    projection += parameter[:, None] * segment
    distances = np.linalg.norm(projection, axis=1)
    index = int(np.argmin(distances))
    suffix = np.vstack((projection[index], path[index + 1:]))
    suffix = _points_xy(suffix)
    return suffix, float(distances[index])


def polyline_length(points) -> float:
    path = _points_xy(points)
    if path.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def resample_polyline(
    points,
    *,
    spacing_m: float,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a path by arc length and pad unavailable targets with its end."""
    if spacing_m <= 0.0:
        raise ValueError('spacing_m must be positive')
    if target_count < 1:
        raise ValueError('target_count must be positive')
    path = _points_xy(points)
    targets = np.full((target_count, 2), np.nan, dtype=np.float64)
    mask = np.zeros(target_count, dtype=bool)
    if path.shape[0] == 0:
        return targets, mask
    if path.shape[0] == 1:
        targets[:] = path[0]
        return targets, mask

    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    requested = spacing_m * np.arange(1, target_count + 1)
    mask = requested <= total + 1.0e-9
    query = np.minimum(requested, total)
    targets[:, 0] = np.interp(query, cumulative, path[:, 0])
    targets[:, 1] = np.interp(query, cumulative, path[:, 1])
    return targets, mask


def _path_shape_metrics(points) -> tuple[float, float, float]:
    path = _points_xy(points)
    if path.shape[0] < 2:
        return 0.0, 0.0, 0.0
    delta = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(delta, axis=1)
    maximum_gap = float(segment_lengths.max(initial=0.0))
    usable = segment_lengths > 1.0e-4
    headings = np.arctan2(delta[usable, 1], delta[usable, 0])
    if headings.size < 2:
        return maximum_gap, 0.0, 0.0
    changes = np.arctan2(
        np.sin(np.diff(headings)), np.cos(np.diff(headings))
    )
    adjacent_length = 0.5 * (
        segment_lengths[usable][1:] + segment_lengths[usable][:-1]
    )
    curvature = np.abs(changes) / np.maximum(adjacent_length, 1.0e-4)
    return (
        maximum_gap,
        float(curvature.max(initial=0.0)),
        float(np.abs(changes).sum()),
    )


def _angle_difference(first: float, second: float) -> float:
    return float(abs(math.atan2(
        math.sin(first - second), math.cos(first - second)
    )))


def analyze_nav2_plan(
    points,
    goal_vehicle_xy,
    *,
    spacing_m: float = 0.75,
    target_count: int = 12,
    maximum_curvature_per_m: float = 0.25,
) -> TrajectoryAnalysis:
    """Create one fixed-spacing target and conservative review flags."""
    if maximum_curvature_per_m <= 0.0:
        raise ValueError('maximum_curvature_per_m must be positive')
    source = _points_xy(points)
    forward, vehicle_to_path = forward_polyline_from_vehicle(source)
    targets, target_mask = resample_polyline(
        forward, spacing_m=spacing_m, target_count=target_count,
    )
    length = polyline_length(forward)
    maximum_gap, maximum_curvature, heading_change = _path_shape_metrics(
        forward
    )
    valid_targets = targets[target_mask]
    backward_fraction = (
        float(np.mean(valid_targets[:, 0] < -0.5))
        if valid_targets.size else 0.0
    )
    goal = np.asarray(goal_vehicle_xy, dtype=np.float64).reshape(-1)
    goal_direction_error = math.nan
    if (
        goal.size >= 2 and np.isfinite(goal[:2]).all()
        and valid_targets.shape[0] > 0
        and np.linalg.norm(goal[:2]) > 1.0e-3
        and np.linalg.norm(valid_targets[-1]) > 1.0e-3
    ):
        goal_direction_error = _angle_difference(
            math.atan2(valid_targets[-1, 1], valid_targets[-1, 0]),
            math.atan2(goal[1], goal[0]),
        )

    flags = []
    minimum_useful_targets = max(2, int(math.ceil(0.5 * target_count)))
    goal_distance = (
        float(np.linalg.norm(goal[:2]))
        if goal.size >= 2 and np.isfinite(goal[:2]).all() else math.inf
    )
    if (
        np.count_nonzero(target_mask) < minimum_useful_targets
        and goal_distance > length + 2.0
    ):
        flags.append('short_horizon')
    if vehicle_to_path > 2.0:
        flags.append('vehicle_far_from_plan')
    if maximum_gap > 2.0:
        flags.append('large_path_gap')
    if maximum_curvature > maximum_curvature_per_m * 1.10:
        flags.append('excessive_curvature')
    if heading_change > 1.5 * math.pi:
        flags.append('winding_path')
    if backward_fraction > 0.25:
        flags.append('backward_path')
    if (
        math.isfinite(goal_direction_error)
        and goal_direction_error > math.radians(100.0)
    ):
        flags.append('goal_direction_mismatch')
    if source.shape[0] < 2:
        flags.append('insufficient_path')
    if not np.any(target_mask):
        flags.append('no_valid_target')

    return TrajectoryAnalysis(
        forward_path=forward.astype(np.float32),
        target_points=targets.astype(np.float32),
        target_mask=target_mask,
        source_point_count=int(source.shape[0]),
        forward_path_length_m=length,
        vehicle_to_path_m=vehicle_to_path,
        maximum_source_gap_m=maximum_gap,
        maximum_curvature_per_m=maximum_curvature,
        absolute_heading_change_rad=heading_change,
        backward_target_fraction=backward_fraction,
        goal_direction_error_rad=goal_direction_error,
        flags=tuple(flags),
    )


def transform_vehicle_points(points, source_pose, target_pose) -> np.ndarray:
    """Transform XY points between two vehicle frames through odom/world."""
    local = _points_xy(points)
    source = np.asarray(source_pose, dtype=np.float64).reshape(-1)
    target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
    if source.size < 4 or target.size < 4:
        raise ValueError('poses must contain x, y, z, yaw')
    source_cosine = math.cos(source[3])
    source_sine = math.sin(source[3])
    world_x = (
        source[0] + source_cosine * local[:, 0]
        - source_sine * local[:, 1]
    )
    world_y = (
        source[1] + source_sine * local[:, 0]
        + source_cosine * local[:, 1]
    )
    dx = world_x - target[0]
    dy = world_y - target[1]
    target_cosine = math.cos(target[3])
    target_sine = math.sin(target[3])
    return np.column_stack((
        target_cosine * dx + target_sine * dy,
        -target_sine * dx + target_cosine * dy,
    ))


def median_nearest_distance(first, second) -> float:
    """Return the median nearest-vertex distance between two polylines."""
    first_array = _points_xy(first)
    second_array = _points_xy(second)
    if first_array.shape[0] == 0 or second_array.shape[0] == 0:
        return math.nan
    distance = np.linalg.norm(
        first_array[:, None, :] - second_array[None, :, :], axis=2
    )
    return float(np.median(np.min(distance, axis=1)))
