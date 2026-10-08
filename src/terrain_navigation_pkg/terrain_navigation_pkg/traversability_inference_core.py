"""Pure conservative fusion helpers for learned traversability inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UNKNOWN_DECISION = np.int8(-1)
FREE_DECISION = np.int8(0)
OBSTACLE_DECISION = np.int8(100)


def inference_rate_limit_allows(
    now_s,
    last_inference_s,
    minimum_interval_s,
    *,
    tolerance_s=0.005,
):
    """Return whether a periodic sensor callback may run inference.

    Sensor callbacks nominally separated by the requested interval can arrive
    a few milliseconds early in wall-clock time.  Without a small tolerance,
    a 10 Hz stream throttled to 5 Hz can select every third frame (3.33 Hz)
    instead of every second frame.  The tolerance is deliberately much
    smaller than one 10 Hz input period, so it cannot admit adjacent scans in
    the intended deployment stream.
    """
    now = float(now_s)
    last = float(last_inference_s)
    interval = float(minimum_interval_s)
    tolerance = float(tolerance_s)
    if not np.isfinite(now):
        raise ValueError('now_s must be finite')
    if interval <= 0.0 or not np.isfinite(interval):
        raise ValueError('minimum_interval_s must be positive and finite')
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError('tolerance_s must be non-negative and finite')
    if not np.isfinite(last) or now < last:
        return True
    return now - last + tolerance >= interval


@dataclass(frozen=True)
class SelectiveTraversabilityDecision:
    """Selective AI result plus geometry that cannot be cleared by AI."""

    decision: np.ndarray
    hard_obstacle_mask: np.ndarray
    learned_obstacle_mask: np.ndarray
    learned_free_mask: np.ndarray
    uncertain_mask: np.ndarray


class ShadowPolicyAccumulator:
    """Summarize selective decisions against privileged validation labels."""

    def __init__(self):
        self.counts = {
            key: 0 for key in (
                'known', 'obstacle_targets', 'free_targets', 'accepted',
                'correct', 'false_free', 'false_obstacle', 'observed',
                'hard_obstacle', 'free_decision', 'obstacle_decision',
                'uncertain_observed',
            )
        }

    def update(self, result, target_labels, observed_mask, exclusion_mask):
        target = np.asarray(target_labels)
        observed = np.asarray(observed_mask, dtype=bool)
        excluded = np.asarray(exclusion_mask, dtype=bool)
        if target.shape != result.decision.shape:
            raise ValueError('target does not match selective decision')
        if observed.shape != target.shape or excluded.shape != target.shape:
            raise ValueError('observed/exclusion mask does not match target')
        known = target >= 0
        accepted = known & (result.decision >= 0)
        prediction = result.decision == OBSTACLE_DECISION
        available = observed & ~excluded
        counts = self.counts
        counts['known'] += int(np.count_nonzero(known))
        counts['obstacle_targets'] += int(np.count_nonzero(target == 1))
        counts['free_targets'] += int(np.count_nonzero(target == 0))
        counts['accepted'] += int(np.count_nonzero(accepted))
        counts['correct'] += int(np.count_nonzero(
            accepted & (prediction == (target == 1))
        ))
        counts['false_free'] += int(np.count_nonzero(
            accepted & (target == 1) & ~prediction
        ))
        counts['false_obstacle'] += int(np.count_nonzero(
            accepted & (target == 0) & prediction
        ))
        counts['observed'] += int(np.count_nonzero(available))
        counts['hard_obstacle'] += int(np.count_nonzero(
            result.hard_obstacle_mask
        ))
        counts['free_decision'] += int(np.count_nonzero(
            result.learned_free_mask
        ))
        counts['obstacle_decision'] += int(np.count_nonzero(
            result.decision == OBSTACLE_DECISION
        ))
        counts['uncertain_observed'] += int(np.count_nonzero(
            result.uncertain_mask
        ))

    @staticmethod
    def _ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else float('nan')

    def compute(self):
        values = dict(self.counts)
        values.update({
            'abstained_known': values['known'] - values['accepted'],
            'known_coverage': self._ratio(
                values['accepted'], values['known']
            ),
            'accepted_accuracy': self._ratio(
                values['correct'], values['accepted']
            ),
            'false_free_fraction_of_obstacle_targets': self._ratio(
                values['false_free'], values['obstacle_targets']
            ),
            'false_obstacle_fraction_of_free_targets': self._ratio(
                values['false_obstacle'], values['free_targets']
            ),
            'observed_free_fraction': self._ratio(
                values['free_decision'], values['observed']
            ),
            'observed_obstacle_fraction': self._ratio(
                values['obstacle_decision'], values['observed']
            ),
            'observed_uncertain_fraction': self._ratio(
                values['uncertain_observed'], values['observed']
            ),
        })
        return values


def binary_predictive_entropy(obstacle_probability) -> np.ndarray:
    """Return normalized binary entropy in [0, 1]."""
    probability = np.asarray(obstacle_probability, dtype=np.float32)
    if np.any(~np.isfinite(probability)):
        raise ValueError('obstacle_probability must be finite')
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError('obstacle_probability must be in [0, 1]')
    epsilon = np.finfo(np.float32).eps
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / np.log(2.0)


def build_selective_decision(
    obstacle_probability,
    predictive_entropy,
    observed_mask,
    maximum_height_m,
    vertical_span_m,
    *,
    free_probability_threshold: float = 0.99,
    obstacle_probability_threshold: float = 0.50,
    maximum_entropy: float = 0.15,
    maximum_mc_variance: float = 0.02,
    mc_variance=None,
    hard_obstacle_minimum_z_m: float = -1.40,
    hard_obstacle_vertical_span_m: float = 0.15,
    exclusion_mask=None,
) -> SelectiveTraversabilityDecision:
    """Keep geometric hazards and reject uncertain learned classifications.

    The learned model may classify a directly observed return as free only
    when confidence and uncertainty checks pass. Returns above the fixed
    hard-height boundary or cells with a large vertical span remain obstacles
    even if the network predicts free. This function does not ray-clear empty
    cells and therefore cannot convert unobserved space into free space.
    """
    probability = np.asarray(obstacle_probability, dtype=np.float32)
    entropy = np.asarray(predictive_entropy, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    maximum_height = np.asarray(maximum_height_m, dtype=np.float32)
    vertical_span = np.asarray(vertical_span_m, dtype=np.float32)
    arrays = (entropy, observed, maximum_height, vertical_span)
    if any(value.shape != probability.shape for value in arrays):
        raise ValueError('all traversability grids must have equal shape')
    if not 0.5 < free_probability_threshold <= 1.0:
        raise ValueError('free_probability_threshold must be in (0.5, 1]')
    if not 0.0 <= obstacle_probability_threshold < 0.5 + 1.0e-9:
        raise ValueError(
            'obstacle_probability_threshold must be in [0, 0.5]'
        )
    if not 0.0 <= maximum_entropy <= 1.0:
        raise ValueError('maximum_entropy must be in [0, 1]')
    if hard_obstacle_vertical_span_m <= 0.0:
        raise ValueError('hard obstacle span must be positive')
    if np.any(~np.isfinite(probability)) or np.any(~np.isfinite(entropy)):
        raise ValueError('probability and entropy grids must be finite')

    excluded = (
        np.zeros(probability.shape, dtype=bool)
        if exclusion_mask is None
        else np.asarray(exclusion_mask, dtype=bool)
    )
    if excluded.shape != probability.shape:
        raise ValueError('exclusion_mask does not match probability grid')
    usable = observed & ~excluded
    confident = entropy <= float(maximum_entropy)
    if mc_variance is not None:
        variance = np.asarray(mc_variance, dtype=np.float32)
        if variance.shape != probability.shape:
            raise ValueError('mc_variance does not match probability grid')
        if maximum_mc_variance < 0.0:
            raise ValueError('maximum_mc_variance must be non-negative')
        confident &= variance <= float(maximum_mc_variance)

    hard_obstacle = usable & (
        (maximum_height >= float(hard_obstacle_minimum_z_m))
        | (vertical_span >= float(hard_obstacle_vertical_span_m))
    )
    learned_obstacle = (
        usable
        & confident
        & (probability >= float(obstacle_probability_threshold))
    )
    learned_free = (
        usable
        & confident
        & (probability <= 1.0 - float(free_probability_threshold))
        & ~hard_obstacle
    )
    obstacle = hard_obstacle | learned_obstacle
    uncertain = usable & ~obstacle & ~learned_free
    decision = np.full(probability.shape, UNKNOWN_DECISION, dtype=np.int8)
    decision[learned_free] = FREE_DECISION
    decision[obstacle] = OBSTACLE_DECISION
    return SelectiveTraversabilityDecision(
        decision=decision,
        hard_obstacle_mask=hard_obstacle,
        learned_obstacle_mask=learned_obstacle & ~hard_obstacle,
        learned_free_mask=learned_free,
        uncertain_mask=uncertain,
    )


def bev_to_occupancy_grid(bev_grid) -> np.ndarray:
    """Convert far/left-first BEV cells to ROS x/y OccupancyGrid layout."""
    grid = np.asarray(bev_grid)
    if grid.ndim != 2:
        raise ValueError('bev_grid must be two-dimensional')
    return np.flip(grid, axis=(0, 1)).T.copy()


def occupancy_grid_to_bev(occupancy_grid) -> np.ndarray:
    """Convert ROS x/y OccupancyGrid layout back to far/left-first BEV."""
    grid = np.asarray(occupancy_grid)
    if grid.ndim != 2:
        raise ValueError('occupancy_grid must be two-dimensional')
    return np.flip(grid.T, axis=(0, 1)).copy()
