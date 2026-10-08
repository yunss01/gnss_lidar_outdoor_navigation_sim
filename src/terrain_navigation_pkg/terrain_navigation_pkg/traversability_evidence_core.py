"""Vehicle-aware evidence targets and conservative decision rules.

This module defines the v2 traversability contract.  Unlike the legacy binary
target, obstacle evidence and passable-surface evidence are independent.  A
cell for which neither kind of positive evidence exists remains unknown; low
obstacle probability alone is never sufficient to declare free space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .navigation_learning_recorder_core import BevGeometry
from .traversability_learning_core import (
    FREE_SURFACE_TAGS,
    OBSTACLE_TAGS,
    EgoFootprint,
    ego_footprint_mask,
)


PASSABLE_DISPOSITION = "passable"
OBSTACLE_DISPOSITION = "obstacle"
AMBIGUOUS_DISPOSITION = "ambiguous"
VALID_DISPOSITIONS = frozenset(
    {PASSABLE_DISPOSITION, OBSTACLE_DISPOSITION, AMBIGUOUS_DISPOSITION}
)

NO_INSTANCE_ID = -1
MULTIPLE_INSTANCE_IDS = -2

UNKNOWN_DECISION = np.int8(-1)
PASSABLE_DECISION = np.int8(0)
OBSTACLE_DECISION = np.int8(100)


@dataclass(frozen=True)
class VehicleEvidenceTargets:
    """Independent cell-level evidence used by the v2 learning target."""

    passable_surface_mask: np.ndarray
    obstacle_evidence_mask: np.ndarray
    ambiguous_observed_mask: np.ndarray
    observed_mask: np.ndarray
    passable_point_count: np.ndarray
    obstacle_point_count: np.ndarray
    ambiguous_point_count: np.ndarray
    controlled_passable_point_count: np.ndarray
    controlled_obstacle_point_count: np.ndarray
    controlled_ambiguous_point_count: np.ndarray
    vertical_span: np.ndarray
    obstacle_instance_id: np.ndarray
    ego_mask: np.ndarray


@dataclass(frozen=True)
class LocalGroundEvidence:
    """Multi-scale local-ground-relative features derived from a raw BEV."""

    ground_height: np.ndarray
    relative_max_height: np.ndarray
    support_count: np.ndarray
    support_confidence: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class EvidenceDecision:
    """Conservative inference result for independent evidence predictions."""

    decision: np.ndarray
    hard_obstacle_mask: np.ndarray
    learned_obstacle_mask: np.ndarray
    obstacle_accepted: np.ndarray
    passable_accepted: np.ndarray
    unknown_mask: np.ndarray


def _validate_points(
    semantic_xyz: np.ndarray,
    semantic_tags: np.ndarray,
    semantic_object_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(semantic_xyz, dtype=np.float32)
    tags = np.asarray(semantic_tags).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("semantic_xyz must have shape (N, >=3)")
    if tags.shape[0] != xyz.shape[0]:
        raise ValueError("semantic_tags length must match semantic_xyz")

    if semantic_object_ids is None:
        object_ids = np.full((xyz.shape[0],), NO_INSTANCE_ID, dtype=np.int64)
    else:
        object_ids = (
            np.asarray(semantic_object_ids).reshape(-1).astype(np.int64)
        )
        if object_ids.shape[0] != xyz.shape[0]:
            raise ValueError(
                "semantic_object_ids length must match semantic_xyz"
            )
    return xyz[:, :3], tags.astype(np.int32), object_ids


def _raster_indices(
    xyz: np.ndarray,
    geometry: BevGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(xyz), axis=1)
    safe_xyz = np.where(finite[:, None], xyz, 0.0)
    rows = np.floor(
        (geometry.x_max_m - safe_xyz[:, 0]) / geometry.resolution_m
    ).astype(np.int64)
    cols = np.floor(
        (geometry.y_max_m - safe_xyz[:, 1]) / geometry.resolution_m
    ).astype(np.int64)
    inside = (
        finite
        & (safe_xyz[:, 0] >= geometry.x_min_m)
        & (safe_xyz[:, 0] < geometry.x_max_m)
        & (safe_xyz[:, 1] >= geometry.y_min_m)
        & (safe_xyz[:, 1] < geometry.y_max_m)
        & (safe_xyz[:, 2] >= geometry.z_min_m)
        & (safe_xyz[:, 2] <= geometry.z_max_m)
        & (rows >= 0)
        & (rows < geometry.height)
        & (cols >= 0)
        & (cols < geometry.width)
    )
    return rows, cols, inside


def _count_points(
    rows: np.ndarray,
    cols: np.ndarray,
    point_mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.int32)
    np.add.at(result, (rows[point_mask], cols[point_mask]), 1)
    return result


def _validate_actor_dispositions(
    actor_dispositions: Mapping[int, str] | None,
) -> dict[int, str]:
    if not actor_dispositions:
        return {}
    validated: dict[int, str] = {}
    for actor_id, raw_disposition in actor_dispositions.items():
        disposition = str(raw_disposition).strip().lower()
        if disposition not in VALID_DISPOSITIONS:
            raise ValueError(
                f"actor {actor_id} has unsupported disposition "
                f"{raw_disposition!r}; "
                f"expected one of {sorted(VALID_DISPOSITIONS)}"
            )
        validated[int(actor_id)] = disposition
    return validated


def build_vehicle_evidence_targets(
    semantic_xyz: np.ndarray,
    semantic_tags: np.ndarray,
    geometry: BevGeometry,
    footprint: EgoFootprint,
    *,
    semantic_object_ids: np.ndarray | None = None,
    actor_dispositions: Mapping[int, str] | None = None,
    minimum_surface_points: int = 2,
    minimum_controlled_passable_points: int = 1,
    obstacle_height_span: float = 0.15,
) -> VehicleEvidenceTargets:
    """Build v2 cell targets from semantic returns and explicit vehicle policy.

    CARLA semantic tags provide a conservative default.  Optional actor-level
    dispositions express the vehicle policy for controlled experiments and
    override those semantic defaults per return.  ``ambiguous`` deliberately
    suppresses both positive targets.  Obstacle evidence always wins when
    different objects conflict in one cell.
    """

    if minimum_surface_points < 1 or minimum_controlled_passable_points < 1:
        raise ValueError("minimum point thresholds must be positive")
    if obstacle_height_span <= 0.0:
        raise ValueError("obstacle_height_span must be positive")

    xyz, tags, object_ids = _validate_points(
        semantic_xyz, semantic_tags, semantic_object_ids
    )
    rows_all, cols_all, inside = _raster_indices(xyz, geometry)
    xyz = xyz[inside]
    tags = tags[inside]
    object_ids = object_ids[inside]
    rows = rows_all[inside]
    cols = cols_all[inside]

    shape = (geometry.height, geometry.width)
    ego_mask = ego_footprint_mask(geometry, footprint)
    dispositions = _validate_actor_dispositions(actor_dispositions)

    semantic_passable_point = np.isin(tags, tuple(FREE_SURFACE_TAGS))
    semantic_obstacle_point = np.isin(tags, tuple(OBSTACLE_TAGS))
    controlled_passable_point = np.zeros(tags.shape, dtype=bool)
    controlled_obstacle_point = np.zeros(tags.shape, dtype=bool)
    controlled_ambiguous_point = np.zeros(tags.shape, dtype=bool)

    for actor_id, disposition in dispositions.items():
        selected = object_ids == actor_id
        if not np.any(selected):
            continue
        # Explicit vehicle policy takes precedence over the simulator taxonomy.
        semantic_passable_point[selected] = False
        semantic_obstacle_point[selected] = False
        if disposition == PASSABLE_DISPOSITION:
            controlled_passable_point[selected] = True
        elif disposition == OBSTACLE_DISPOSITION:
            controlled_obstacle_point[selected] = True
        else:
            controlled_ambiguous_point[selected] = True

    passable_point = semantic_passable_point | controlled_passable_point
    obstacle_point = semantic_obstacle_point | controlled_obstacle_point
    classified_point = passable_point | obstacle_point
    ambiguous_point = controlled_ambiguous_point | ~classified_point

    passable_point_count = _count_points(rows, cols, passable_point, shape)
    obstacle_point_count = _count_points(rows, cols, obstacle_point, shape)
    ambiguous_point_count = _count_points(rows, cols, ambiguous_point, shape)
    controlled_passable_count = _count_points(
        rows, cols, controlled_passable_point, shape
    )
    controlled_obstacle_count = _count_points(
        rows, cols, controlled_obstacle_point, shape
    )
    controlled_ambiguous_count = _count_points(
        rows, cols, controlled_ambiguous_point, shape
    )

    observed_count = np.zeros(shape, dtype=np.int32)
    np.add.at(observed_count, (rows, cols), 1)
    observed_mask = observed_count > 0

    min_z = np.full(shape, np.inf, dtype=np.float32)
    max_z = np.full(shape, -np.inf, dtype=np.float32)
    if xyz.shape[0]:
        np.minimum.at(min_z, (rows, cols), xyz[:, 2])
        np.maximum.at(max_z, (rows, cols), xyz[:, 2])
    vertical_span = np.zeros(shape, dtype=np.float32)
    vertical_span[observed_mask] = (
        max_z[observed_mask] - min_z[observed_mask]
    )

    semantic_surface_count = _count_points(
        rows, cols, semantic_passable_point, shape
    )
    passable_surface_mask = (
        (semantic_surface_count >= int(minimum_surface_points))
        | (
            controlled_passable_count
            >= int(minimum_controlled_passable_points)
        )
    )

    # An explicitly passable controlled actor is the vehicle-policy label for
    # that object.  Do not turn it back into an obstacle solely because its
    # points increase same-cell span.  A different obstacle in the cell still
    # wins below.
    geometry_obstacle = (
        (semantic_surface_count >= int(minimum_surface_points))
        & (vertical_span >= float(obstacle_height_span))
        & (controlled_passable_count == 0)
    )
    obstacle_evidence_mask = (obstacle_point_count > 0) | geometry_obstacle
    passable_surface_mask &= ~obstacle_evidence_mask

    ambiguous_observed_mask = (
        observed_mask & ~passable_surface_mask & ~obstacle_evidence_mask
    )

    obstacle_instance_id = np.full(shape, NO_INSTANCE_ID, dtype=np.int64)
    for row, col, object_id in zip(
        rows[obstacle_point], cols[obstacle_point], object_ids[obstacle_point]
    ):
        current = obstacle_instance_id[row, col]
        if current == NO_INSTANCE_ID:
            obstacle_instance_id[row, col] = int(object_id)
        elif current != int(object_id):
            obstacle_instance_id[row, col] = MULTIPLE_INSTANCE_IDS

    # The ego body is sensor self-return/occlusion, never supervision.
    for array in (
        passable_surface_mask,
        obstacle_evidence_mask,
        ambiguous_observed_mask,
        observed_mask,
    ):
        array[ego_mask] = False
    for array in (
        passable_point_count,
        obstacle_point_count,
        ambiguous_point_count,
        controlled_passable_count,
        controlled_obstacle_count,
        controlled_ambiguous_count,
        vertical_span,
    ):
        array[ego_mask] = 0
    obstacle_instance_id[ego_mask] = NO_INSTANCE_ID

    return VehicleEvidenceTargets(
        passable_surface_mask=passable_surface_mask,
        obstacle_evidence_mask=obstacle_evidence_mask,
        ambiguous_observed_mask=ambiguous_observed_mask,
        observed_mask=observed_mask,
        passable_point_count=passable_point_count,
        obstacle_point_count=obstacle_point_count,
        ambiguous_point_count=ambiguous_point_count,
        controlled_passable_point_count=controlled_passable_count,
        controlled_obstacle_point_count=controlled_obstacle_count,
        controlled_ambiguous_point_count=controlled_ambiguous_count,
        vertical_span=vertical_span,
        obstacle_instance_id=obstacle_instance_id,
        ego_mask=ego_mask,
    )


def _shift_without_wrap(values: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Return output[r,c] = values[r+dr,c+dc], padding with +inf."""

    height, width = values.shape
    shifted = np.full(values.shape, np.inf, dtype=np.float32)
    src_r0 = max(0, dr)
    src_r1 = min(height, height + dr)
    src_c0 = max(0, dc)
    src_c1 = min(width, width + dc)
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return shifted
    dst_r0 = max(0, -dr)
    dst_c0 = max(0, -dc)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    shifted[dst_r0:dst_r1, dst_c0:dst_c1] = values[
        src_r0:src_r1, src_c0:src_c1
    ]
    return shifted


def build_local_ground_evidence(
    lidar_bev: np.ndarray,
    geometry: BevGeometry,
    *,
    radii_m: Sequence[float] = (0.75, 1.50),
    ground_quantile: float = 0.25,
    minimum_support_cells: int = 4,
) -> LocalGroundEvidence:
    """Estimate local ground and object prominence at multiple spatial scales.

    The estimate uses the lower height of occupied cells around each location.
    This exposes protrusions whose LiDAR returns fall into neighboring cells,
    avoiding dependence on same-cell vertical span alone.
    """

    bev = np.asarray(lidar_bev, dtype=np.float32)
    if bev.ndim != 3 or bev.shape[0] < 4:
        raise ValueError("lidar_bev must have shape (>=4, H, W)")
    if bev.shape[1:] != (geometry.height, geometry.width):
        raise ValueError("lidar_bev spatial shape does not match geometry")
    if not radii_m or any(float(radius) <= 0.0 for radius in radii_m):
        raise ValueError("radii_m must contain positive radii")
    if not 0.0 <= float(ground_quantile) <= 1.0:
        raise ValueError("ground_quantile must be in [0, 1]")
    if minimum_support_cells < 1:
        raise ValueError("minimum_support_cells must be positive")

    occupied = bev[0] > 0.5
    max_z = bev[2]
    vertical_span = np.maximum(bev[3], 0.0)
    lower_height = np.where(occupied, max_z - vertical_span, np.inf).astype(
        np.float32
    )

    ground_layers: list[np.ndarray] = []
    relative_layers: list[np.ndarray] = []
    count_layers: list[np.ndarray] = []
    confidence_layers: list[np.ndarray] = []
    valid_layers: list[np.ndarray] = []

    for radius_m in radii_m:
        radius_cells = max(
            1, int(np.ceil(float(radius_m) / geometry.resolution_m))
        )
        offsets = [
            (dr, dc)
            for dr in range(-radius_cells, radius_cells + 1)
            for dc in range(-radius_cells, radius_cells + 1)
            if dr * dr + dc * dc <= radius_cells * radius_cells
        ]
        neighborhood = np.stack(
            [_shift_without_wrap(lower_height, dr, dc) for dr, dc in offsets],
            axis=0,
        )
        finite = np.isfinite(neighborhood)
        support_count = finite.sum(axis=0).astype(np.int16)
        ordered = np.sort(neighborhood, axis=0)
        quantile_index = np.floor(
            float(ground_quantile) * np.maximum(support_count - 1, 0)
        ).astype(np.int64)
        ground = np.take_along_axis(
            ordered, quantile_index[None, ...], axis=0
        )[0]
        valid = support_count >= int(minimum_support_cells)
        ground = np.where(valid, ground, 0.0).astype(np.float32)
        relative = np.where(valid & occupied, max_z - ground, 0.0).astype(
            np.float32
        )
        confidence = np.clip(
            support_count.astype(np.float32) / float(len(offsets)), 0.0, 1.0
        )

        ground_layers.append(ground)
        relative_layers.append(relative)
        count_layers.append(support_count)
        confidence_layers.append(confidence)
        valid_layers.append(valid)

    return LocalGroundEvidence(
        ground_height=np.stack(ground_layers, axis=0),
        relative_max_height=np.stack(relative_layers, axis=0),
        support_count=np.stack(count_layers, axis=0),
        support_confidence=np.stack(confidence_layers, axis=0),
        valid_mask=np.stack(valid_layers, axis=0),
    )


def build_evidence_bev(
    lidar_bev: np.ndarray,
    geometry: BevGeometry,
    *,
    radii_m: Sequence[float] = (0.75, 1.50),
    ground_quantile: float = 0.25,
    minimum_support_cells: int = 4,
) -> tuple[np.ndarray, LocalGroundEvidence]:
    """Append local prominence/support to the legacy 4-channel BEV."""

    local = build_local_ground_evidence(
        lidar_bev,
        geometry,
        radii_m=radii_m,
        ground_quantile=ground_quantile,
        minimum_support_cells=minimum_support_cells,
    )
    evidence_bev = np.concatenate(
        [
            np.asarray(lidar_bev, dtype=np.float32),
            local.relative_max_height.astype(np.float32),
            local.support_confidence.astype(np.float32),
        ],
        axis=0,
    )
    return evidence_bev, local


def build_vehicle_clearance_obstacle_mask(
    lidar_bev: np.ndarray,
    local_ground_relative_height_m: np.ndarray,
    local_ground_valid_mask: np.ndarray,
    *,
    minimum_relative_height_m: float = 0.15,
    minimum_vertical_span_m: float = 0.15,
    minimum_absolute_height_m: float = -1.40,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build obstacle evidence that a learned model may never erase.

    The relative-height threshold is a vehicle-clearance policy, not an
    object-family heuristic.  Multi-scale local ground catches sparse objects
    whose return and road return fall in different cells; same-cell span and
    the established absolute sensor-height boundary remain fallbacks.
    """
    bev = np.asarray(lidar_bev, dtype=np.float32)
    relative = np.asarray(
        local_ground_relative_height_m, dtype=np.float32
    )
    valid = np.asarray(local_ground_valid_mask, dtype=bool)
    if bev.ndim != 3 or bev.shape[0] < 4:
        raise ValueError('lidar_bev must have shape (>=4, H, W)')
    if relative.ndim == 2:
        relative = relative[None, ...]
    if valid.ndim == 2:
        valid = valid[None, ...]
    if relative.shape != valid.shape or relative.shape[1:] != bev.shape[1:]:
        raise ValueError('local-ground arrays do not match lidar_bev')
    if minimum_relative_height_m <= 0.0:
        raise ValueError('minimum relative height must be positive')
    if minimum_vertical_span_m <= 0.0:
        raise ValueError('minimum vertical span must be positive')
    if not np.isfinite(minimum_absolute_height_m):
        raise ValueError('minimum absolute height must be finite')
    excluded = (
        np.zeros(bev.shape[1:], dtype=bool)
        if exclusion_mask is None
        else np.asarray(exclusion_mask, dtype=bool)
    )
    if excluded.shape != bev.shape[1:]:
        raise ValueError('exclusion_mask does not match lidar_bev')
    occupied = bev[0] > 0.5
    local_prominence = np.any(
        valid & (relative >= float(minimum_relative_height_m)), axis=0
    )
    hard = occupied & (
        local_prominence
        | (bev[3] >= float(minimum_vertical_span_m))
        | (bev[2] >= float(minimum_absolute_height_m))
    )
    return hard & ~excluded


def build_conservative_evidence_decision(
    passable_probability: np.ndarray,
    obstacle_probability: np.ndarray,
    observed_mask: np.ndarray,
    passable_support: np.ndarray,
    *,
    passable_uncertainty: np.ndarray | None = None,
    obstacle_uncertainty: np.ndarray | None = None,
    hard_obstacle_mask: np.ndarray | None = None,
    obstacle_probability_threshold: float = 0.50,
    passable_probability_threshold: float = 0.90,
    maximum_obstacle_probability_for_passable: float = 0.10,
    minimum_passable_support: float = 0.20,
    maximum_uncertainty: float = 0.25,
) -> EvidenceDecision:
    """Fuse independent predictions while preserving the unknown state.

    The key safety invariant is explicit: a low obstacle probability does not
    make a cell passable.  Passable output requires positive passable evidence,
    geometric/visibility support, and acceptable uncertainty.
    """

    passable_probability = np.asarray(passable_probability, dtype=np.float32)
    obstacle_probability = np.asarray(obstacle_probability, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    support = np.asarray(passable_support, dtype=np.float32)
    if not (
        passable_probability.shape
        == obstacle_probability.shape
        == observed.shape
        == support.shape
    ):
        raise ValueError("all evidence arrays must have the same shape")
    hard_obstacle = (
        np.zeros(observed.shape, dtype=bool)
        if hard_obstacle_mask is None
        else np.asarray(hard_obstacle_mask, dtype=bool)
    )
    if hard_obstacle.shape != observed.shape:
        raise ValueError('hard_obstacle_mask shape mismatch')
    hard_obstacle = hard_obstacle & observed

    passable_confident = np.ones(observed.shape, dtype=bool)
    obstacle_confident = np.ones(observed.shape, dtype=bool)
    if passable_uncertainty is not None:
        uncertainty = np.asarray(passable_uncertainty, dtype=np.float32)
        if uncertainty.shape != observed.shape:
            raise ValueError("passable_uncertainty shape mismatch")
        passable_confident &= uncertainty <= float(maximum_uncertainty)
    if obstacle_uncertainty is not None:
        uncertainty = np.asarray(obstacle_uncertainty, dtype=np.float32)
        if uncertainty.shape != observed.shape:
            raise ValueError("obstacle_uncertainty shape mismatch")
        obstacle_confident &= uncertainty <= float(maximum_uncertainty)

    learned_obstacle = (
        observed
        & obstacle_confident
        & (obstacle_probability >= float(obstacle_probability_threshold))
    )
    obstacle_accepted = hard_obstacle | learned_obstacle
    passable_accepted = (
        observed
        & passable_confident
        & (support >= float(minimum_passable_support))
        & (passable_probability >= float(passable_probability_threshold))
        & (
            obstacle_probability
            <= float(maximum_obstacle_probability_for_passable)
        )
        & ~obstacle_accepted
    )

    decision = np.full(observed.shape, UNKNOWN_DECISION, dtype=np.int8)
    decision[passable_accepted] = PASSABLE_DECISION
    decision[obstacle_accepted] = OBSTACLE_DECISION
    unknown_mask = decision == UNKNOWN_DECISION
    return EvidenceDecision(
        decision=decision,
        hard_obstacle_mask=hard_obstacle,
        learned_obstacle_mask=learned_obstacle & ~hard_obstacle,
        obstacle_accepted=obstacle_accepted,
        passable_accepted=passable_accepted,
        unknown_mask=unknown_mask,
    )
