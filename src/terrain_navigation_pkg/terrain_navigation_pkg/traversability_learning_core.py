"""Pure helpers for LiDAR-only traversability-learning targets.

The deployed model consumes only geometric LiDAR features.  CARLA semantic
tags are privileged supervision used while creating simulation targets; they
are not model inputs and are unavailable during deployment by design.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .navigation_learning_recorder_core import BevGeometry


UNKNOWN_LABEL = np.int8(-1)
FREE_LABEL = np.int8(0)
OBSTACLE_LABEL = np.int8(1)

# CARLA 0.10.0 CityObjectLabel values, verified from the installed Python API.
FREE_SURFACE_TAGS = frozenset({
    1,   # Roads
    2,   # Sidewalks / paved pedestrian surfaces
    10,  # Terrain: grass, soil, sand, ground-level vegetation
    24,  # RoadLines
    25,  # Ground
})
OBSTACLE_TAGS = frozenset({
    3,   # Buildings
    4,   # Walls
    5,   # Fences
    6,   # Poles
    7,   # TrafficLight
    8,   # TrafficSigns
    9,   # Vegetation: trees, hedges, vertical vegetation
    12,  # Pedestrians
    13,  # Rider
    14,  # Car
    15,  # Truck
    16,  # Bus
    17,  # Train
    18,  # Motorcycle
    19,  # Bicycle
    20,  # Static props
    21,  # Dynamic props
    23,  # Water
    28,  # GuardRail
})


@dataclass(frozen=True)
class TraversabilityTargets:
    """Vehicle-centric dense labels with auditable supporting counts."""

    labels: np.ndarray
    free_point_count: np.ndarray
    obstacle_point_count: np.ndarray
    observed_point_count: np.ndarray
    vertical_span_m: np.ndarray
    ego_exclusion_mask: np.ndarray


@dataclass(frozen=True)
class EgoFootprint:
    """Nominal vehicle body relative to the LiDAR frame."""

    rear_m: float = 2.5
    front_m: float = 2.4
    half_width_m: float = 1.0

    def __post_init__(self):
        if self.rear_m <= 0.0 or self.front_m <= 0.0:
            raise ValueError('ego footprint lengths must be positive')
        if self.half_width_m <= 0.0:
            raise ValueError('ego footprint half width must be positive')


@dataclass(frozen=True)
class VisibilityEvidence:
    """Ray-cleared evidence kept separate from semantic surface labels."""

    free_mask: np.ndarray
    ray_count: np.ndarray


def ego_footprint_mask(geometry: BevGeometry, footprint: EgoFootprint):
    """Return cells covered by the nominal, unpadded vehicle body."""
    rows = np.arange(geometry.height, dtype=np.float32)
    columns = np.arange(geometry.width, dtype=np.float32)
    forward = geometry.x_max_m - (
        rows + 0.5
    ) * geometry.resolution_m
    left = geometry.y_max_m - (
        columns + 0.5
    ) * geometry.resolution_m
    return (
        (forward[:, None] >= -footprint.rear_m)
        & (forward[:, None] <= footprint.front_m)
        & (np.abs(left[None, :]) <= footprint.half_width_m)
    )


def _validate_points(points, columns, name):
    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f'{name} must have shape (N, {columns})')
    return array


def build_semantic_traversability_targets(
    semantic_xyz,
    object_tags,
    geometry: BevGeometry,
    minimum_surface_points: int = 2,
    obstacle_vertical_span_m: float = 0.15,
    ego_footprint: EgoFootprint | None = None,
) -> TraversabilityTargets:
    """Rasterize CARLA-only semantic supervision into free/obstacle/unknown.

    A hard-object return always wins over a free-surface return in the same
    cell.  A large vertical span also marks an otherwise free-surface cell as
    an obstacle, preserving curbs and steps whose CARLA tag may be Sidewalk or
    Terrain.  Unknown and unsupported tags remain ignored by the training loss.
    """
    xyz = _validate_points(semantic_xyz, 3, 'semantic_xyz').astype(
        np.float32, copy=False
    )
    tags = np.asarray(object_tags)
    if tags.shape != (xyz.shape[0],):
        raise ValueError('object_tags must have shape (N,)')
    if minimum_surface_points < 1:
        raise ValueError('minimum_surface_points must be positive')
    if obstacle_vertical_span_m <= 0.0:
        raise ValueError('obstacle_vertical_span_m must be positive')

    shape = (geometry.height, geometry.width)
    observed_count = np.zeros(shape, dtype=np.int32)
    free_count = np.zeros(shape, dtype=np.int32)
    obstacle_count = np.zeros(shape, dtype=np.int32)
    minimum_z = np.full(shape, np.inf, dtype=np.float32)
    maximum_z = np.full(shape, -np.inf, dtype=np.float32)

    finite = np.all(np.isfinite(xyz), axis=1)
    inside = (
        finite
        & (xyz[:, 0] >= geometry.x_min_m)
        & (xyz[:, 0] < geometry.x_max_m)
        & (xyz[:, 1] >= geometry.y_min_m)
        & (xyz[:, 1] < geometry.y_max_m)
        & (xyz[:, 2] >= geometry.z_min_m)
        & (xyz[:, 2] <= geometry.z_max_m)
    )
    if np.any(inside):
        selected = xyz[inside]
        selected_tags = tags[inside].astype(np.int64, copy=False)
        rows = np.floor(
            (geometry.x_max_m - selected[:, 0]) / geometry.resolution_m
        ).astype(np.int32)
        columns = np.floor(
            (geometry.y_max_m - selected[:, 1]) / geometry.resolution_m
        ).astype(np.int32)
        rows = np.clip(rows, 0, geometry.height - 1)
        columns = np.clip(columns, 0, geometry.width - 1)

        np.add.at(observed_count, (rows, columns), 1)
        free_mask = np.isin(selected_tags, tuple(FREE_SURFACE_TAGS))
        obstacle_mask = np.isin(selected_tags, tuple(OBSTACLE_TAGS))
        np.add.at(free_count, (rows[free_mask], columns[free_mask]), 1)
        np.add.at(
            obstacle_count,
            (rows[obstacle_mask], columns[obstacle_mask]),
            1,
        )
        np.minimum.at(minimum_z, (rows, columns), selected[:, 2])
        np.maximum.at(maximum_z, (rows, columns), selected[:, 2])

    vertical_span = np.zeros(shape, dtype=np.float32)
    observed = observed_count > 0
    vertical_span[observed] = maximum_z[observed] - minimum_z[observed]

    labels = np.full(shape, UNKNOWN_LABEL, dtype=np.int8)
    supported_free = free_count >= minimum_surface_points
    labels[supported_free] = FREE_LABEL
    labels[
        (obstacle_count > 0)
        | (supported_free & (vertical_span >= obstacle_vertical_span_m))
    ] = OBSTACLE_LABEL
    ego_mask = np.zeros(shape, dtype=bool)
    if ego_footprint is not None:
        ego_mask = ego_footprint_mask(geometry, ego_footprint)
        labels[ego_mask] = UNKNOWN_LABEL
        observed_count[ego_mask] = 0
        free_count[ego_mask] = 0
        obstacle_count[ego_mask] = 0
        vertical_span[ego_mask] = 0.0
    return TraversabilityTargets(
        labels=labels,
        free_point_count=free_count,
        obstacle_point_count=obstacle_count,
        observed_point_count=observed_count,
        vertical_span_m=vertical_span,
        ego_exclusion_mask=ego_mask,
    )


def build_conservative_visibility_evidence(
    semantic_xyz,
    geometry: BevGeometry,
    targets: TraversabilityTargets,
    *,
    ego_footprint: EgoFootprint,
    angular_bin_count: int = 720,
    endpoint_margin_m: float = 0.25,
    obstacle_margin_m: float = 0.50,
    obstacle_angular_dilation_bins: int = 1,
) -> VisibilityEvidence:
    """Build conservative 2-D ray evidence without changing class targets.

    Each azimuth bin is cleared only up to its farthest observed return and is
    shortened by the nearest obstacle return in that bin or an adjacent bin.
    This evidence is useful for later occupancy-map fusion, but it is not
    merged into ``target_labels`` because a 3-D beam can pass above an unseen
    low obstacle.
    """
    xyz = _validate_points(semantic_xyz, 3, 'semantic_xyz').astype(
        np.float32, copy=False
    )
    if targets.labels.shape != (geometry.height, geometry.width):
        raise ValueError('targets do not match BEV geometry')
    if angular_bin_count < 4:
        raise ValueError('angular_bin_count must be at least four')
    if endpoint_margin_m < 0.0 or obstacle_margin_m < 0.0:
        raise ValueError('ray margins must be non-negative')
    if obstacle_angular_dilation_bins < 0:
        raise ValueError('obstacle dilation bins must be non-negative')

    finite = np.all(np.isfinite(xyz), axis=1)
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    outside_ego = ~(
        (xyz[:, 0] >= -ego_footprint.rear_m)
        & (xyz[:, 0] <= ego_footprint.front_m)
        & (np.abs(xyz[:, 1]) <= ego_footprint.half_width_m)
    )
    usable = finite & outside_ego & (radius > geometry.resolution_m)
    shape = (geometry.height, geometry.width)
    ray_count = np.zeros(shape, dtype=np.uint16)
    if not np.any(usable):
        return VisibilityEvidence(ray_count > 0, ray_count)

    selected = xyz[usable]
    selected_radius = radius[usable]
    angle = np.arctan2(selected[:, 1], selected[:, 0])
    bins = np.floor(
        (angle + np.pi) * angular_bin_count / (2.0 * np.pi)
    ).astype(np.int32)
    bins = np.mod(bins, angular_bin_count)

    farthest = np.zeros(angular_bin_count, dtype=np.float32)
    np.maximum.at(farthest, bins, selected_radius)
    nearest_obstacle = np.full(
        angular_bin_count, np.inf, dtype=np.float32
    )

    inside = (
        (selected[:, 0] >= geometry.x_min_m)
        & (selected[:, 0] < geometry.x_max_m)
        & (selected[:, 1] >= geometry.y_min_m)
        & (selected[:, 1] < geometry.y_max_m)
        & (selected[:, 2] >= geometry.z_min_m)
        & (selected[:, 2] <= geometry.z_max_m)
    )
    selected_rows = np.zeros(selected.shape[0], dtype=np.int32)
    selected_columns = np.zeros(selected.shape[0], dtype=np.int32)
    selected_rows[inside] = np.floor(
        (geometry.x_max_m - selected[inside, 0]) / geometry.resolution_m
    ).astype(np.int32)
    selected_columns[inside] = np.floor(
        (geometry.y_max_m - selected[inside, 1]) / geometry.resolution_m
    ).astype(np.int32)
    selected_rows = np.clip(selected_rows, 0, geometry.height - 1)
    selected_columns = np.clip(selected_columns, 0, geometry.width - 1)
    obstacle_endpoint = np.zeros(selected.shape[0], dtype=bool)
    obstacle_endpoint[inside] = (
        targets.labels[
            selected_rows[inside], selected_columns[inside]
        ] == OBSTACLE_LABEL
    )
    np.minimum.at(
        nearest_obstacle,
        bins[obstacle_endpoint],
        selected_radius[obstacle_endpoint],
    )
    dilated_obstacle = nearest_obstacle.copy()
    for offset in range(1, obstacle_angular_dilation_bins + 1):
        dilated_obstacle = np.minimum(
            dilated_obstacle, np.roll(nearest_obstacle, offset)
        )
        dilated_obstacle = np.minimum(
            dilated_obstacle, np.roll(nearest_obstacle, -offset)
        )

    clear_limit = np.maximum(0.0, farthest - endpoint_margin_m)
    finite_obstacle = np.isfinite(dilated_obstacle)
    clear_limit[finite_obstacle] = np.minimum(
        clear_limit[finite_obstacle],
        np.maximum(
            0.0,
            dilated_obstacle[finite_obstacle] - obstacle_margin_m,
        ),
    )
    step = 0.5 * geometry.resolution_m
    for index in np.flatnonzero(clear_limit > step):
        distance = np.arange(step, clear_limit[index], step)
        angle_center = (
            (index + 0.5) * 2.0 * np.pi / angular_bin_count - np.pi
        )
        forward = distance * math.cos(angle_center)
        left = distance * math.sin(angle_center)
        ray_inside = (
            (forward >= geometry.x_min_m)
            & (forward < geometry.x_max_m)
            & (left >= geometry.y_min_m)
            & (left < geometry.y_max_m)
        )
        if not np.any(ray_inside):
            continue
        rows = np.floor(
            (geometry.x_max_m - forward[ray_inside])
            / geometry.resolution_m
        ).astype(np.int32)
        columns = np.floor(
            (geometry.y_max_m - left[ray_inside])
            / geometry.resolution_m
        ).astype(np.int32)
        rows = np.clip(rows, 0, geometry.height - 1)
        columns = np.clip(columns, 0, geometry.width - 1)
        flat = np.unique(rows * geometry.width + columns)
        flat_counts = ray_count.reshape(-1)
        can_increment = flat_counts[flat] < np.iinfo(np.uint16).max
        flat_counts[flat[can_increment]] += 1

    ego_mask = ego_footprint_mask(geometry, ego_footprint)
    ray_count[ego_mask] = 0
    ray_count[targets.labels == OBSTACLE_LABEL] = 0
    return VisibilityEvidence(ray_count > 0, ray_count)
