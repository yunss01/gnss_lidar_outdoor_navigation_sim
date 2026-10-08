"""Pure helpers for passive learned/baseline obstacle-cloud fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .navigation_learning_recorder_core import BevGeometry, build_lidar_bev
from .traversability_evidence_core import build_local_ground_evidence
from .traversability_inference_core import (
    FREE_DECISION,
    OBSTACLE_DECISION,
    UNKNOWN_DECISION,
)


@dataclass(frozen=True)
class CandidateFusionMasks:
    """Point masks used to construct one passive candidate cloud."""

    baseline_keep_mask: np.ndarray
    baseline_cleared_by_ai_mask: np.ndarray
    raw_ai_obstacle_mask: np.ndarray
    raw_in_bev_mask: np.ndarray


@dataclass(frozen=True)
class AddOnlyFusionMasks:
    """Raw endpoints that may be appended without removing baseline data."""

    raw_ai_obstacle_mask: np.ndarray
    raw_obstacle_before_ground_gate_mask: np.ndarray
    raw_ground_gate_rejected_mask: np.ndarray
    raw_in_bev_mask: np.ndarray


def point_decisions(points_xyz, decision, geometry: BevGeometry):
    """Sample a far/left-first BEV decision at each finite 3-D point."""
    points = np.asarray(points_xyz, dtype=np.float32)
    grid = np.asarray(decision)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points_xyz must have shape (N, 3+)')
    if grid.shape != (geometry.height, geometry.width):
        raise ValueError('decision shape does not match BEV geometry')
    allowed = np.isin(
        grid,
        (UNKNOWN_DECISION, FREE_DECISION, OBSTACLE_DECISION),
    )
    if not np.all(allowed):
        raise ValueError('decision values must be -1, 0, or 100')

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    inside = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (x >= geometry.x_min_m)
        & (x < geometry.x_max_m)
        & (y >= geometry.y_min_m)
        & (y < geometry.y_max_m)
        & (z >= geometry.z_min_m)
        & (z <= geometry.z_max_m)
    )
    sampled = np.full(points.shape[0], UNKNOWN_DECISION, dtype=np.int8)
    if np.any(inside):
        rows = np.floor(
            (geometry.x_max_m - x[inside]) / geometry.resolution_m
        ).astype(np.int32)
        columns = np.floor(
            (geometry.y_max_m - y[inside]) / geometry.resolution_m
        ).astype(np.int32)
        rows = np.clip(rows, 0, geometry.height - 1)
        columns = np.clip(columns, 0, geometry.width - 1)
        sampled[inside] = grid[rows, columns].astype(np.int8)
    return sampled, inside


def build_candidate_fusion_masks(
    raw_points_xyz,
    baseline_points_xyz,
    decision,
    geometry: BevGeometry,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    obstacle_maximum_z_m: float,
    ego_front_m: float,
    ego_rear_m: float,
    ego_half_width_m: float,
    hard_obstacle_minimum_z_m: float,
    hard_obstacle_vertical_span_m: float,
):
    """Fuse selective AI cells with the already-running baseline cloud.

    Confident learned-free cells may remove baseline obstacle points. Learned
    obstacle cells add the corresponding raw returns. Unknown/out-of-BEV cells
    retain the baseline result. The upstream selective policy already forces
    fixed-height and vertical-span hazards to ``OBSTACLE_DECISION``.
    """
    raw = np.asarray(raw_points_xyz, dtype=np.float32)
    baseline = np.asarray(baseline_points_xyz, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError('raw_points_xyz must have shape (N, 3+)')
    if baseline.ndim != 2 or baseline.shape[1] < 3:
        raise ValueError('baseline_points_xyz must have shape (N, 3+)')
    if not 0.0 <= minimum_range_m < maximum_range_m:
        raise ValueError('LiDAR range limits are invalid')
    if min(ego_front_m, ego_rear_m, ego_half_width_m) <= 0.0:
        raise ValueError('ego extents must be positive')
    if hard_obstacle_vertical_span_m <= 0.0:
        raise ValueError('hard obstacle span must be positive')
    if not geometry.z_min_m < obstacle_maximum_z_m <= geometry.z_max_m:
        raise ValueError('obstacle maximum z is outside BEV geometry')

    raw_decision, raw_in_bev = point_decisions(raw, decision, geometry)
    baseline_decision, _ = point_decisions(
        baseline, decision, geometry
    )
    baseline_range_sq = baseline[:, 0] ** 2 + baseline[:, 1] ** 2
    baseline_in_range = (
        (baseline_range_sq >= float(minimum_range_m) ** 2)
        & (baseline_range_sq <= float(maximum_range_m) ** 2)
    )
    # Learned decisions may modify the established baseline only inside the
    # validated AI fusion radius. Outside it, retain baseline obstacles even
    # if the network emits a confident free decision.
    baseline_cleared = (
        (baseline_decision == FREE_DECISION) & baseline_in_range
    )
    baseline_keep = ~baseline_cleared

    x = raw[:, 0]
    y = raw[:, 1]
    z = raw[:, 2]
    range_sq = x * x + y * y
    in_range = (
        (range_sq >= float(minimum_range_m) ** 2)
        & (range_sq <= float(maximum_range_m) ** 2)
    )
    ego_body = (
        (x >= -float(ego_rear_m))
        & (x <= float(ego_front_m))
        & (np.abs(y) <= float(ego_half_width_m))
    )
    raw_bev = build_lidar_bev(raw[:, :3], geometry)
    hard_cells = (
        (raw_bev[2] >= float(hard_obstacle_minimum_z_m))
        | (raw_bev[3] >= float(hard_obstacle_vertical_span_m))
    )
    hard_grid = np.full(hard_cells.shape, UNKNOWN_DECISION, dtype=np.int8)
    hard_grid[hard_cells] = OBSTACLE_DECISION
    hard_point_decision, _ = point_decisions(raw, hard_grid, geometry)
    # Hard geometry is a veto against learned clearing, not permission to
    # bulk-add every raw return from tall buildings or overhanging objects.
    # The established baseline remains responsible for those cells. Only a
    # network obstacle that is not already a hard-geometry veto may add a new
    # low-height endpoint during this passive integration stage.
    learned_obstacle = (
        (raw_decision == OBSTACLE_DECISION)
        & (hard_point_decision != OBSTACLE_DECISION)
    )
    raw_ai_obstacle = (
        raw_in_bev
        & in_range
        & ~ego_body
        & (z <= float(obstacle_maximum_z_m))
        & learned_obstacle
    )
    return CandidateFusionMasks(
        baseline_keep_mask=baseline_keep,
        baseline_cleared_by_ai_mask=baseline_cleared,
        raw_ai_obstacle_mask=raw_ai_obstacle,
        raw_in_bev_mask=raw_in_bev,
    )


def build_add_only_fusion_masks(
    raw_points_xyz,
    decision,
    geometry: BevGeometry,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    obstacle_maximum_z_m: float,
    ego_front_m: float,
    ego_rear_m: float,
    ego_half_width_m: float,
    obstacle_cell_top_band_m: float = 0.05,
    minimum_relative_height_m: float = 0.07,
    local_ground_radius_m: float = 0.75,
    local_ground_quantile: float = 0.25,
    local_ground_minimum_support_cells: int = 4,
):
    """Select physically raised endpoints in v2 obstacle cells.

    Passable and unknown decisions never remove or add anything.  Within an
    obstacle cell, only returns close to that cell's maximum height are
    selected.  A candidate must also be sufficiently high above a supported
    local-ground estimate.  This prevents a learned false-positive on a flat
    road cell from promoting that cell's highest (and possibly only) road
    return into a Nav2 obstacle.
    """

    raw = np.asarray(raw_points_xyz, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError('raw_points_xyz must have shape (N, 3+)')
    if not 0.0 <= minimum_range_m < maximum_range_m:
        raise ValueError('LiDAR range limits are invalid')
    if min(ego_front_m, ego_rear_m, ego_half_width_m) <= 0.0:
        raise ValueError('ego extents must be positive')
    if not geometry.z_min_m < obstacle_maximum_z_m <= geometry.z_max_m:
        raise ValueError('obstacle maximum z is outside BEV geometry')
    if obstacle_cell_top_band_m < 0.0:
        raise ValueError('obstacle_cell_top_band_m must be non-negative')
    if minimum_relative_height_m <= 0.0:
        raise ValueError('minimum_relative_height_m must be positive')
    if local_ground_radius_m <= 0.0:
        raise ValueError('local_ground_radius_m must be positive')
    if not 0.0 <= local_ground_quantile <= 1.0:
        raise ValueError('local_ground_quantile must be in [0, 1]')
    if local_ground_minimum_support_cells < 1:
        raise ValueError(
            'local_ground_minimum_support_cells must be positive'
        )

    raw_decision, raw_in_bev = point_decisions(raw, decision, geometry)
    x = raw[:, 0]
    y = raw[:, 1]
    z = raw[:, 2]
    range_sq = x * x + y * y
    in_range = (
        (range_sq >= float(minimum_range_m) ** 2)
        & (range_sq <= float(maximum_range_m) ** 2)
    )
    ego_body = (
        (x >= -float(ego_rear_m))
        & (x <= float(ego_front_m))
        & (np.abs(y) <= float(ego_half_width_m))
    )

    selected = (
        raw_in_bev
        & in_range
        & ~ego_body
        & (z <= float(obstacle_maximum_z_m))
        & (raw_decision == OBSTACLE_DECISION)
    )
    lidar_bev = build_lidar_bev(raw[:, :3], geometry)
    # Match the recorder/model preprocessing boundary so the authority and
    # the model use the same quantized local-ground representation.
    lidar_bev = lidar_bev.astype(np.float16).astype(np.float32)
    local_ground = build_local_ground_evidence(
        lidar_bev,
        geometry,
        radii_m=(float(local_ground_radius_m),),
        ground_quantile=float(local_ground_quantile),
        minimum_support_cells=int(local_ground_minimum_support_cells),
    )
    local_ground_height = (
        local_ground.ground_height.astype(np.float16).astype(np.float32)
    )

    if np.any(selected):
        rows = np.floor(
            (geometry.x_max_m - x[selected]) / geometry.resolution_m
        ).astype(np.int32)
        columns = np.floor(
            (geometry.y_max_m - y[selected]) / geometry.resolution_m
        ).astype(np.int32)
        rows = np.clip(rows, 0, geometry.height - 1)
        columns = np.clip(columns, 0, geometry.width - 1)
        cell_maximum = lidar_bev[2]
        close_to_top = z[selected] >= (
            cell_maximum[rows, columns]
            - float(obstacle_cell_top_band_m)
        )
        selected_indices = np.flatnonzero(selected)
        selected[selected_indices[~close_to_top]] = False

    before_ground_gate = selected.copy()
    if np.any(selected):
        selected_indices = np.flatnonzero(selected)
        rows = np.floor(
            (geometry.x_max_m - x[selected]) / geometry.resolution_m
        ).astype(np.int32)
        columns = np.floor(
            (geometry.y_max_m - y[selected]) / geometry.resolution_m
        ).astype(np.int32)
        rows = np.clip(rows, 0, geometry.height - 1)
        columns = np.clip(columns, 0, geometry.width - 1)
        ground_valid = local_ground.valid_mask[0, rows, columns]
        relative_height = (
            z[selected] - local_ground_height[0, rows, columns]
        )
        physically_raised = (
            ground_valid
            & (relative_height >= float(minimum_relative_height_m))
        )
        selected[selected_indices[~physically_raised]] = False

    return AddOnlyFusionMasks(
        raw_ai_obstacle_mask=selected,
        raw_obstacle_before_ground_gate_mask=before_ground_gate,
        raw_ground_gate_rejected_mask=(before_ground_gate & ~selected),
        raw_in_bev_mask=raw_in_bev,
    )


def voxel_unique_indices(points_xyz, voxel_size_m: float):
    """Return stable first-point indices after 3-D voxel deduplication."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points_xyz must have shape (N, 3+)')
    if voxel_size_m <= 0.0:
        raise ValueError('voxel_size_m must be positive')
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    keys = np.floor(points[:, :3] / float(voxel_size_m)).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices).astype(np.int64, copy=False)


def novel_voxel_indices(
    existing_points_xyz, candidate_points_xyz, voxel_size_m
):
    """Return stable candidate indices whose voxels are absent from existing.

    Existing records are never deduplicated or rewritten.  This is the key
    byte-level invariant for add-only fusion: every baseline point remains in
    the output, and only novel candidate voxels are appended.
    """

    existing = np.asarray(existing_points_xyz, dtype=np.float32)
    candidates = np.asarray(candidate_points_xyz, dtype=np.float32)
    if existing.ndim != 2 or existing.shape[1] < 3:
        raise ValueError('existing_points_xyz must have shape (N, 3+)')
    if candidates.ndim != 2 or candidates.shape[1] < 3:
        raise ValueError('candidate_points_xyz must have shape (N, 3+)')
    if voxel_size_m <= 0.0:
        raise ValueError('voxel_size_m must be positive')
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    scale = float(voxel_size_m)
    existing_keys = {
        tuple(value)
        for value in np.floor(existing[:, :3] / scale).astype(np.int32)
    }
    accepted = []
    for index, key in enumerate(
        np.floor(candidates[:, :3] / scale).astype(np.int32)
    ):
        voxel = tuple(key)
        if voxel in existing_keys:
            continue
        existing_keys.add(voxel)
        accepted.append(index)
    return np.asarray(accepted, dtype=np.int64)
