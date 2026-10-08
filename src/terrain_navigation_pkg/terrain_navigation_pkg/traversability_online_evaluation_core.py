"""Pure helpers for online shadow evaluation with privileged labels."""

from __future__ import annotations

import numpy as np


COUNT_FIELDS = (
    'frames',
    'target_known',
    'target_obstacle',
    'target_free',
    'raw_evaluated',
    'raw_true_obstacle',
    'raw_true_free',
    'raw_false_obstacle',
    'raw_false_free',
    'selective_accepted',
    'selective_correct',
    'selective_true_obstacle',
    'selective_true_free',
    'selective_false_obstacle',
    'selective_false_free',
    'selective_abstained',
    'selective_abstained_obstacle',
    'selective_abstained_free',
    'geometric_override_obstacle',
)


def occupancy_grid_to_bev(grid):
    """Invert ``bev_to_occupancy_grid`` without changing cell values."""
    value = np.asarray(grid)
    if value.ndim != 2:
        raise ValueError('occupancy grid must be two-dimensional')
    return np.flip(value.T, axis=(0, 1)).copy()


def _ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def summarize_evaluation_counts(counts):
    """Return rates that keep abstention distinct from classification error."""
    values = {name: int(counts.get(name, 0)) for name in COUNT_FIELDS}
    raw_union = (
        values['raw_true_obstacle']
        + values['raw_false_obstacle']
        + values['raw_false_free']
    )
    raw_correct = values['raw_true_obstacle'] + values['raw_true_free']
    raw_obstacle_predictions = (
        values['raw_true_obstacle']
        + values['raw_false_obstacle']
    )
    values.update({
        'raw_accuracy': _ratio(raw_correct, values['raw_evaluated']),
        'raw_obstacle_iou': _ratio(
            values['raw_true_obstacle'], raw_union
        ),
        'raw_obstacle_precision': _ratio(
            values['raw_true_obstacle'], raw_obstacle_predictions
        ),
        'raw_obstacle_recall': _ratio(
            values['raw_true_obstacle'], values['target_obstacle']
        ),
        'raw_false_free_rate': _ratio(
            values['raw_false_free'], values['target_obstacle']
        ),
        'raw_false_obstacle_rate': _ratio(
            values['raw_false_obstacle'], values['target_free']
        ),
        'selective_coverage': _ratio(
            values['selective_accepted'], values['target_known']
        ),
        'selective_accepted_accuracy': _ratio(
            values['selective_correct'], values['selective_accepted']
        ),
        'selective_obstacle_recall': _ratio(
            values['selective_true_obstacle'], values['target_obstacle']
        ),
        'selective_false_free_rate': _ratio(
            values['selective_false_free'], values['target_obstacle']
        ),
        'selective_false_obstacle_rate': _ratio(
            values['selective_false_obstacle'], values['target_free']
        ),
        'selective_safe_retention_rate': _ratio(
            values['target_obstacle'] - values['selective_false_free'],
            values['target_obstacle'],
        ),
    })
    return values


def evaluate_traversability_frame(
    target_labels,
    probability_percent,
    selective_decision,
    *,
    obstacle_probability_threshold_percent=50,
    evaluation_mask=None,
):
    """Count raw and selective predictions for one aligned cell grid.

    Unknown target cells never enter an accuracy denominator.  A selective
    unknown is an abstention, not a free prediction.  This distinction is
    necessary because deployment keeps unknown space conservative.
    """
    target = np.asarray(target_labels)
    probability = np.asarray(probability_percent)
    decision = np.asarray(selective_decision)
    if target.ndim != 2 or probability.shape != target.shape:
        raise ValueError('target and probability grids must have equal shape')
    if decision.shape != target.shape:
        raise ValueError('target and decision grids must have equal shape')
    if not 0 <= obstacle_probability_threshold_percent <= 100:
        raise ValueError('probability threshold must be in [0, 100]')
    if not np.all(np.isin(target, (-1, 0, 1))):
        raise ValueError('target values must be unknown, free, or obstacle')
    if np.any((probability < -1) | (probability > 100)):
        raise ValueError('probability values must be -1 or in [0, 100]')
    if not np.all(np.isin(decision, (-1, 0, 100))):
        raise ValueError('decision values must be unknown, free, or obstacle')

    selected = (
        np.ones(target.shape, dtype=bool)
        if evaluation_mask is None
        else np.asarray(evaluation_mask, dtype=bool)
    )
    if selected.shape != target.shape:
        raise ValueError('evaluation mask must match the target grid')

    target_known = (target >= 0) & selected
    target_obstacle = (target == 1) & selected
    target_free = (target == 0) & selected
    probability_known = probability >= 0
    raw_mask = target_known & probability_known
    raw_obstacle = probability >= obstacle_probability_threshold_percent
    raw_free = probability_known & ~raw_obstacle

    accepted = target_known & (decision >= 0)
    predicted_obstacle = decision == 100
    predicted_free = decision == 0
    abstained = target_known & (decision < 0)
    geometric_override = (
        target_known
        & predicted_obstacle
        & probability_known
        & ~raw_obstacle
    )

    counts = {name: 0 for name in COUNT_FIELDS}
    counts.update({
        'frames': 1,
        'target_known': int(np.count_nonzero(target_known)),
        'target_obstacle': int(np.count_nonzero(target_obstacle)),
        'target_free': int(np.count_nonzero(target_free)),
        'raw_evaluated': int(np.count_nonzero(raw_mask)),
        'raw_true_obstacle': int(np.count_nonzero(
            raw_mask & target_obstacle & raw_obstacle
        )),
        'raw_true_free': int(np.count_nonzero(
            raw_mask & target_free & raw_free
        )),
        'raw_false_obstacle': int(np.count_nonzero(
            raw_mask & target_free & raw_obstacle
        )),
        'raw_false_free': int(np.count_nonzero(
            raw_mask & target_obstacle & raw_free
        )),
        'selective_accepted': int(np.count_nonzero(accepted)),
        'selective_correct': int(np.count_nonzero(
            accepted & (
                (target_obstacle & predicted_obstacle)
                | (target_free & predicted_free)
            )
        )),
        'selective_true_obstacle': int(np.count_nonzero(
            target_obstacle & predicted_obstacle
        )),
        'selective_true_free': int(np.count_nonzero(
            target_free & predicted_free
        )),
        'selective_false_obstacle': int(np.count_nonzero(
            target_free & predicted_obstacle
        )),
        'selective_false_free': int(np.count_nonzero(
            target_obstacle & predicted_free
        )),
        'selective_abstained': int(np.count_nonzero(abstained)),
        'selective_abstained_obstacle': int(np.count_nonzero(
            target_obstacle & abstained
        )),
        'selective_abstained_free': int(np.count_nonzero(
            target_free & abstained
        )),
        'geometric_override_obstacle': int(np.count_nonzero(
            geometric_override
        )),
    })
    return summarize_evaluation_counts(counts)


class OnlineTraversabilityAccumulator:
    """Accumulate cell-observation counts over aligned route frames."""

    def __init__(self):
        self.counts = {name: 0 for name in COUNT_FIELDS}

    def update(
        self,
        target_labels,
        probability_percent,
        selective_decision,
        *,
        evaluation_mask=None,
    ):
        frame = evaluate_traversability_frame(
            target_labels,
            probability_percent,
            selective_decision,
            evaluation_mask=evaluation_mask,
        )
        for name in COUNT_FIELDS:
            self.counts[name] += int(frame[name])
        return frame

    def compute(self):
        return summarize_evaluation_counts(self.counts)


def build_swept_footprint_mask(
    geometry,
    trajectory_xy_yaw,
    *,
    vehicle_front_m,
    vehicle_rear_m,
    half_width_m,
):
    """Rasterize oriented vehicle rectangles along a sampled trajectory."""
    path = np.asarray(trajectory_xy_yaw, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 1:
        raise ValueError('trajectory must have shape (N, 3)')
    if not np.all(np.isfinite(path)):
        raise ValueError('trajectory values must be finite')
    if min(vehicle_front_m, vehicle_rear_m, half_width_m) <= 0.0:
        raise ValueError('vehicle footprint dimensions must be positive')

    rows = np.arange(geometry.height, dtype=np.float64)
    columns = np.arange(geometry.width, dtype=np.float64)
    forward = geometry.x_max_m - (
        rows + 0.5
    ) * geometry.resolution_m
    left = geometry.y_max_m - (
        columns + 0.5
    ) * geometry.resolution_m
    cell_x, cell_y = np.meshgrid(forward, left, indexing='ij')
    mask = np.zeros(cell_x.shape, dtype=bool)
    for x_m, y_m, yaw_rad in path:
        dx = cell_x - x_m
        dy = cell_y - y_m
        cosine = np.cos(yaw_rad)
        sine = np.sin(yaw_rad)
        longitudinal = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        mask |= (
            (longitudinal >= -vehicle_rear_m)
            & (longitudinal <= vehicle_front_m)
            & (np.abs(lateral) <= half_width_m)
        )
    return mask


def largest_connected_component(mask):
    """Return the largest 8-connected component size and cell spans."""
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError('component mask must be two-dimensional')
    visited = np.zeros(value.shape, dtype=bool)
    best = (0, 0, 0)
    height, width = value.shape
    for start_row, start_column in np.argwhere(value):
        if visited[start_row, start_column]:
            continue
        stack = [(int(start_row), int(start_column))]
        visited[start_row, start_column] = True
        rows = []
        columns = []
        while stack:
            row, column = stack.pop()
            rows.append(row)
            columns.append(column)
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor_row = row + row_offset
                    neighbor_column = column + column_offset
                    if not (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    ):
                        continue
                    if (
                        value[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))
        candidate = (
            len(rows),
            max(rows) - min(rows) + 1,
            max(columns) - min(columns) + 1,
        )
        if candidate[0] > best[0]:
            best = candidate
    return {
        'cell_count': int(best[0]),
        'row_span_cells': int(best[1]),
        'column_span_cells': int(best[2]),
    }


def finite_alignment_summary(values):
    """Summarize finite timestamp deltas in seconds."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {'mean_s': None, 'maximum_s': None, 'p95_s': None}
    return {
        'mean_s': float(np.mean(array)),
        'maximum_s': float(np.max(array)),
        'p95_s': float(np.percentile(array, 95.0)),
    }
