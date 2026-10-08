"""Metrics that keep v2 passable and obstacle evidence independent."""

from __future__ import annotations

import math

import torch


def _ratio(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else math.nan


class _BinaryEvidenceAccumulator:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        self.brier_sum = 0.0
        self.count = 0

    def update(self, probability, target, mask) -> None:
        mask = mask.to(dtype=torch.bool)
        if not torch.any(mask):
            return
        probability = probability.detach()[mask].double().cpu()
        target = target.detach()[mask].to(dtype=torch.bool).cpu()
        prediction = probability >= self.threshold
        self.true_positive += int(torch.count_nonzero(prediction & target))
        self.false_positive += int(torch.count_nonzero(prediction & ~target))
        self.true_negative += int(torch.count_nonzero(~prediction & ~target))
        self.false_negative += int(torch.count_nonzero(~prediction & target))
        self.brier_sum += float(torch.sum(
            (probability - target.double()) ** 2
        ))
        self.count += int(target.numel())

    def compute(self) -> dict[str, float | int]:
        tp = self.true_positive
        fp = self.false_positive
        tn = self.true_negative
        fn = self.false_negative
        return {
            'known_cells': self.count,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn,
            'precision': _ratio(tp, tp + fp),
            'recall': _ratio(tp, tp + fn),
            'f1': _ratio(2 * tp, 2 * tp + fp + fn),
            'iou': _ratio(tp, tp + fp + fn),
            'brier_score': _ratio(self.brier_sum, self.count),
        }


class TraversabilityEvidenceMetricAccumulator:
    """Accumulate head metrics and safety-critical controlled-cell recall."""

    def __init__(
        self,
        *,
        passable_threshold: float = 0.90,
        obstacle_threshold: float = 0.50,
        maximum_obstacle_probability_for_passable: float = 0.10,
        minimum_passable_support: float = 0.20,
    ):
        self.passable_threshold = float(passable_threshold)
        self.obstacle_threshold = float(obstacle_threshold)
        self.maximum_obstacle_probability_for_passable = float(
            maximum_obstacle_probability_for_passable
        )
        self.minimum_passable_support = float(minimum_passable_support)
        self.passable = _BinaryEvidenceAccumulator(self.passable_threshold)
        self.obstacle = _BinaryEvidenceAccumulator(self.obstacle_threshold)
        self.controlled_obstacle_cells = 0
        self.controlled_obstacle_detected_cells = 0
        self.controlled_instances = 0
        self.controlled_instances_any_detected = 0
        self.controlled_instances_all_detected = 0
        self.controlled_unsafe_passable_cells = 0
        self.controlled_instances_unsafe_passable = 0
        self.observed_cells = 0
        self.passable_decisions = 0
        self.obstacle_decisions = 0
        self.unknown_decisions = 0

    def update(
        self,
        outputs,
        passable_target,
        obstacle_target,
        known_evidence_mask,
        controlled_obstacle_mask,
        observed_mask,
        passable_support,
    ) -> None:
        passable_probability = torch.sigmoid(
            outputs['passable_logits'].detach()
        )
        obstacle_probability = torch.sigmoid(
            outputs['obstacle_logits'].detach()
        )
        known = known_evidence_mask.to(dtype=torch.bool)
        self.passable.update(passable_probability, passable_target, known)
        self.obstacle.update(obstacle_probability, obstacle_target, known)

        controlled = controlled_obstacle_mask.to(dtype=torch.bool)
        detected = obstacle_probability >= self.obstacle_threshold
        self.controlled_obstacle_cells += int(torch.count_nonzero(controlled))
        self.controlled_obstacle_detected_cells += int(torch.count_nonzero(
            controlled & detected
        ))
        for sample_index in range(controlled.shape[0]):
            sample_mask = controlled[sample_index]
            if not torch.any(sample_mask):
                continue
            sample_detected = detected[sample_index][sample_mask]
            self.controlled_instances += 1
            self.controlled_instances_any_detected += int(torch.any(
                sample_detected
            ))
            self.controlled_instances_all_detected += int(torch.all(
                sample_detected
            ))

        observed = observed_mask.to(dtype=torch.bool)
        obstacle_accepted = observed & detected
        passable_accepted = (
            observed
            & ~obstacle_accepted
            & (passable_probability >= self.passable_threshold)
            & (
                obstacle_probability
                <= self.maximum_obstacle_probability_for_passable
            )
            & (passable_support >= self.minimum_passable_support)
        )
        unsafe_controlled_passable = controlled & passable_accepted
        self.controlled_unsafe_passable_cells += int(torch.count_nonzero(
            unsafe_controlled_passable
        ))
        for sample_index in range(controlled.shape[0]):
            sample_mask = controlled[sample_index]
            if not torch.any(sample_mask):
                continue
            sample_detected = torch.any(
                detected[sample_index][sample_mask]
            )
            sample_passable = torch.any(
                unsafe_controlled_passable[sample_index][sample_mask]
            )
            self.controlled_instances_unsafe_passable += int(
                not sample_detected and sample_passable
            )
        observed_count = int(torch.count_nonzero(observed))
        obstacle_count = int(torch.count_nonzero(obstacle_accepted))
        passable_count = int(torch.count_nonzero(passable_accepted))
        self.observed_cells += observed_count
        self.obstacle_decisions += obstacle_count
        self.passable_decisions += passable_count
        self.unknown_decisions += (
            observed_count - obstacle_count - passable_count
        )

    def compute(self) -> dict:
        return {
            'passable_head': self.passable.compute(),
            'obstacle_head': self.obstacle.compute(),
            'controlled_obstacle_cells': self.controlled_obstacle_cells,
            'controlled_obstacle_detected_cells': (
                self.controlled_obstacle_detected_cells
            ),
            'controlled_obstacle_cell_recall': _ratio(
                self.controlled_obstacle_detected_cells,
                self.controlled_obstacle_cells,
            ),
            'controlled_instances': self.controlled_instances,
            'controlled_instance_any_detection_rate': _ratio(
                self.controlled_instances_any_detected,
                self.controlled_instances,
            ),
            'controlled_instance_all_cells_detection_rate': _ratio(
                self.controlled_instances_all_detected,
                self.controlled_instances,
            ),
            'controlled_unsafe_passable_cells': (
                self.controlled_unsafe_passable_cells
            ),
            'controlled_unsafe_passable_cell_rate': _ratio(
                self.controlled_unsafe_passable_cells,
                self.controlled_obstacle_cells,
            ),
            'controlled_instances_unsafe_passable': (
                self.controlled_instances_unsafe_passable
            ),
            'controlled_unsafe_passable_instance_rate': _ratio(
                self.controlled_instances_unsafe_passable,
                self.controlled_instances,
            ),
            'decision_observed_cells': self.observed_cells,
            'decision_passable_cells': self.passable_decisions,
            'decision_obstacle_cells': self.obstacle_decisions,
            'decision_unknown_cells': self.unknown_decisions,
            'decision_unknown_fraction': _ratio(
                self.unknown_decisions, self.observed_cells
            ),
        }
