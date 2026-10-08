"""Pure metrics for no-authority static-obstacle candidate audits."""

from __future__ import annotations

import numpy as np

from .traversability_learning_core import OBSTACLE_TAGS


COUNT_FIELDS = (
    'frames',
    'target_known',
    'target_obstacle',
    'target_free',
    'baseline_occupied',
    'baseline_true_obstacle',
    'baseline_true_free',
    'baseline_false_obstacle',
    'baseline_false_free',
    'candidate_occupied',
    'candidate_true_obstacle',
    'candidate_true_free',
    'candidate_false_obstacle',
    'candidate_false_free',
    'removed_cells',
    'added_cells',
    'removed_target_obstacle',
    'removed_target_free',
    'added_target_obstacle',
    'added_target_free',
)

RATE_FIELDS = (
    'baseline_obstacle_iou',
    'baseline_obstacle_precision',
    'baseline_obstacle_recall',
    'baseline_false_free_rate',
    'baseline_false_obstacle_rate',
    'candidate_obstacle_iou',
    'candidate_obstacle_precision',
    'candidate_obstacle_recall',
    'candidate_false_free_rate',
    'candidate_false_obstacle_rate',
    'removed_target_obstacle_rate',
    'removed_target_free_rate',
    'added_target_obstacle_rate',
    'added_target_free_rate',
    'candidate_to_baseline_occupied_ratio',
)


def _ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def force_controlled_obstacle_tag(
    object_indices,
    object_tags,
    controlled_actor_id,
    obstacle_tag=20,
):
    """Return tags with one spawned actor treated as obstacle ground truth."""
    indices = np.asarray(object_indices)
    tags = np.asarray(object_tags)
    if indices.ndim != 1 or tags.ndim != 1 or indices.shape != tags.shape:
        raise ValueError('object indices and tags must be equal 1-D arrays')
    actor_id = int(controlled_actor_id)
    if actor_id < 0:
        return tags, 0
    actor_mask = indices.astype(np.int64, copy=False) == actor_id
    count = int(np.count_nonzero(actor_mask))
    effective_tags = tags.copy()
    effective_tags[actor_mask] = int(obstacle_tag)
    return effective_tags, count


def comparison_masks(target_labels, baseline_occupied, candidate_occupied):
    """Return boolean masks used to audit one aligned static frame."""
    target = np.asarray(target_labels)
    baseline = np.asarray(baseline_occupied, dtype=bool)
    candidate = np.asarray(candidate_occupied, dtype=bool)
    if target.ndim != 2:
        raise ValueError('target_labels must be two-dimensional')
    if baseline.shape != target.shape or candidate.shape != target.shape:
        raise ValueError('target and occupancy grids must have equal shape')
    if not np.all(np.isin(target, (-1, 0, 1))):
        raise ValueError('target labels must be unknown, free, or obstacle')

    known = target >= 0
    obstacle = target == 1
    free = target == 0
    removed = baseline & ~candidate
    added = candidate & ~baseline
    return {
        'known': known,
        'obstacle': obstacle,
        'free': free,
        'removed': removed,
        'added': added,
        'removed_target_obstacle': removed & obstacle,
        'removed_target_free': removed & free,
        'added_target_obstacle': added & obstacle,
        'added_target_free': added & free,
    }


def summarize_counts(counts):
    """Add interpretable rates to accumulated cell-observation counts."""
    values = {name: int(counts.get(name, 0)) for name in COUNT_FIELDS}
    baseline_union = (
        values['baseline_true_obstacle']
        + values['baseline_false_obstacle']
        + values['baseline_false_free']
    )
    candidate_union = (
        values['candidate_true_obstacle']
        + values['candidate_false_obstacle']
        + values['candidate_false_free']
    )
    values.update({
        'baseline_obstacle_iou': _ratio(
            values['baseline_true_obstacle'], baseline_union
        ),
        'baseline_obstacle_precision': _ratio(
            values['baseline_true_obstacle'],
            values['baseline_true_obstacle']
            + values['baseline_false_obstacle'],
        ),
        'baseline_obstacle_recall': _ratio(
            values['baseline_true_obstacle'], values['target_obstacle']
        ),
        'baseline_false_free_rate': _ratio(
            values['baseline_false_free'], values['target_obstacle']
        ),
        'baseline_false_obstacle_rate': _ratio(
            values['baseline_false_obstacle'], values['target_free']
        ),
        'candidate_obstacle_iou': _ratio(
            values['candidate_true_obstacle'], candidate_union
        ),
        'candidate_obstacle_precision': _ratio(
            values['candidate_true_obstacle'],
            values['candidate_true_obstacle']
            + values['candidate_false_obstacle'],
        ),
        'candidate_obstacle_recall': _ratio(
            values['candidate_true_obstacle'], values['target_obstacle']
        ),
        'candidate_false_free_rate': _ratio(
            values['candidate_false_free'], values['target_obstacle']
        ),
        'candidate_false_obstacle_rate': _ratio(
            values['candidate_false_obstacle'], values['target_free']
        ),
        'removed_target_obstacle_rate': _ratio(
            values['removed_target_obstacle'], values['target_obstacle']
        ),
        'removed_target_free_rate': _ratio(
            values['removed_target_free'], values['target_free']
        ),
        'added_target_obstacle_rate': _ratio(
            values['added_target_obstacle'], values['target_obstacle']
        ),
        'added_target_free_rate': _ratio(
            values['added_target_free'], values['target_free']
        ),
        'candidate_to_baseline_occupied_ratio': _ratio(
            values['candidate_occupied'], values['baseline_occupied']
        ),
    })
    return values


def evaluate_frame(target_labels, baseline_occupied, candidate_occupied):
    """Evaluate one aligned baseline/candidate/semantic BEV frame."""
    baseline = np.asarray(baseline_occupied, dtype=bool)
    candidate = np.asarray(candidate_occupied, dtype=bool)
    masks = comparison_masks(target_labels, baseline, candidate)
    known = masks['known']
    obstacle = masks['obstacle']
    free = masks['free']

    counts = {name: 0 for name in COUNT_FIELDS}
    counts.update({
        'frames': 1,
        'target_known': int(np.count_nonzero(known)),
        'target_obstacle': int(np.count_nonzero(obstacle)),
        'target_free': int(np.count_nonzero(free)),
        'baseline_occupied': int(np.count_nonzero(baseline)),
        'baseline_true_obstacle': int(np.count_nonzero(
            baseline & obstacle
        )),
        'baseline_true_free': int(np.count_nonzero(
            ~baseline & free
        )),
        'baseline_false_obstacle': int(np.count_nonzero(
            baseline & free
        )),
        'baseline_false_free': int(np.count_nonzero(
            ~baseline & obstacle
        )),
        'candidate_occupied': int(np.count_nonzero(candidate)),
        'candidate_true_obstacle': int(np.count_nonzero(
            candidate & obstacle
        )),
        'candidate_true_free': int(np.count_nonzero(
            ~candidate & free
        )),
        'candidate_false_obstacle': int(np.count_nonzero(
            candidate & free
        )),
        'candidate_false_free': int(np.count_nonzero(
            ~candidate & obstacle
        )),
        'removed_cells': int(np.count_nonzero(masks['removed'])),
        'added_cells': int(np.count_nonzero(masks['added'])),
        'removed_target_obstacle': int(np.count_nonzero(
            masks['removed_target_obstacle']
        )),
        'removed_target_free': int(np.count_nonzero(
            masks['removed_target_free']
        )),
        'added_target_obstacle': int(np.count_nonzero(
            masks['added_target_obstacle']
        )),
        'added_target_free': int(np.count_nonzero(
            masks['added_target_free']
        )),
    })
    return summarize_counts(counts)


class StaticObstacleAuditAccumulator:
    """Accumulate repeated cell-observation counts across static frames."""

    def __init__(self):
        self.counts = {name: 0 for name in COUNT_FIELDS}

    def update(self, target_labels, baseline_occupied, candidate_occupied):
        frame = evaluate_frame(
            target_labels, baseline_occupied, candidate_occupied
        )
        for name in COUNT_FIELDS:
            self.counts[name] += int(frame[name])
        return frame

    def compute(self):
        return summarize_counts(self.counts)


def semantic_obstacle_instance_metrics(
    semantic_xyz,
    object_indices,
    object_tags,
    target_labels,
    baseline_occupied,
    candidate_occupied,
    geometry,
    *,
    corridor_minimum_x_m=0.0,
    corridor_maximum_x_m=15.0,
    corridor_half_width_m=4.0,
):
    """
    Measure obstacle retention per CARLA semantic object instance.

    Only semantic obstacle returns whose rasterized cell is an obstacle target
    are used.  This avoids treating free-surface returns belonging to the same
    CARLA actor as obstacle evidence.  Results are limited to instances with
    at least one target cell in the configured forward diagnostic corridor.
    """
    xyz = np.asarray(semantic_xyz, dtype=np.float32)
    indices = np.asarray(object_indices)
    tags = np.asarray(object_tags)
    target = np.asarray(target_labels)
    baseline = np.asarray(baseline_occupied, dtype=bool)
    candidate = np.asarray(candidate_occupied, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('semantic_xyz must have shape (N, 3)')
    if indices.shape != (xyz.shape[0],):
        raise ValueError('object_indices must have shape (N,)')
    if tags.shape != (xyz.shape[0],):
        raise ValueError('object_tags must have shape (N,)')
    expected_shape = (geometry.height, geometry.width)
    if any(
        value.shape != expected_shape
        for value in (target, baseline, candidate)
    ):
        raise ValueError('target and occupancy grids must match geometry')
    if not (
        corridor_minimum_x_m < corridor_maximum_x_m
        and corridor_half_width_m > 0.0
    ):
        raise ValueError('instance corridor limits are invalid')

    finite = np.isfinite(xyz).all(axis=1)
    obstacle_return = np.isin(tags, tuple(OBSTACLE_TAGS))
    inside = (
        finite
        & obstacle_return
        & (xyz[:, 0] >= geometry.x_min_m)
        & (xyz[:, 0] < geometry.x_max_m)
        & (xyz[:, 1] >= geometry.y_min_m)
        & (xyz[:, 1] < geometry.y_max_m)
        & (xyz[:, 2] >= geometry.z_min_m)
        & (xyz[:, 2] <= geometry.z_max_m)
    )
    if not np.any(inside):
        return []

    selected_xyz = xyz[inside]
    selected_indices = indices[inside].astype(np.uint64, copy=False)
    selected_tags = tags[inside].astype(np.uint32, copy=False)
    rows = np.floor(
        (geometry.x_max_m - selected_xyz[:, 0]) / geometry.resolution_m
    ).astype(np.int32)
    columns = np.floor(
        (geometry.y_max_m - selected_xyz[:, 1]) / geometry.resolution_m
    ).astype(np.int32)
    rows = np.clip(rows, 0, geometry.height - 1)
    columns = np.clip(columns, 0, geometry.width - 1)

    result = []
    for object_index in np.unique(selected_indices):
        match = selected_indices == object_index
        cells = np.unique(
            np.column_stack((rows[match], columns[match])), axis=0
        )
        if cells.size == 0:
            continue
        cell_rows = cells[:, 0]
        cell_columns = cells[:, 1]
        obstacle_cells = target[cell_rows, cell_columns] == 1
        cell_rows = cell_rows[obstacle_cells]
        cell_columns = cell_columns[obstacle_cells]
        if cell_rows.size == 0:
            continue
        forward = geometry.x_max_m - (
            cell_rows.astype(np.float32) + 0.5
        ) * geometry.resolution_m
        left = geometry.y_max_m - (
            cell_columns.astype(np.float32) + 0.5
        ) * geometry.resolution_m
        corridor = (
            (forward >= float(corridor_minimum_x_m))
            & (forward <= float(corridor_maximum_x_m))
            & (np.abs(left) <= float(corridor_half_width_m))
        )
        if not np.any(corridor):
            continue
        cell_rows = cell_rows[corridor]
        cell_columns = cell_columns[corridor]
        forward = forward[corridor]
        left = left[corridor]
        baseline_hits = int(np.count_nonzero(
            baseline[cell_rows, cell_columns]
        ))
        candidate_hits = int(np.count_nonzero(
            candidate[cell_rows, cell_columns]
        ))
        baseline_only = int(np.count_nonzero(
            baseline[cell_rows, cell_columns]
            & ~candidate[cell_rows, cell_columns]
        ))
        candidate_only = int(np.count_nonzero(
            ~baseline[cell_rows, cell_columns]
            & candidate[cell_rows, cell_columns]
        ))
        instance_tags = selected_tags[match]
        unique_tags, tag_counts = np.unique(
            instance_tags, return_counts=True
        )
        dominant_tag = int(unique_tags[np.argmax(tag_counts)])
        retention = (
            float(candidate_hits) / float(baseline_hits)
            if baseline_hits else None
        )
        result.append({
            'object_idx': int(object_index),
            'object_tag': dominant_tag,
            'target_cells': int(cell_rows.size),
            'baseline_hits': baseline_hits,
            'candidate_hits': candidate_hits,
            'baseline_only_cells': baseline_only,
            'candidate_only_cells': candidate_only,
            'baseline_detected': int(baseline_hits > 0),
            'candidate_detected': int(candidate_hits > 0),
            'fully_removed': int(
                baseline_hits > 0 and candidate_hits == 0
            ),
            'candidate_to_baseline_retention': retention,
            'minimum_forward_m': float(np.min(forward)),
            'maximum_forward_m': float(np.max(forward)),
            'minimum_left_m': float(np.min(left)),
            'maximum_left_m': float(np.max(left)),
        })
    return sorted(result, key=lambda value: value['object_idx'])
