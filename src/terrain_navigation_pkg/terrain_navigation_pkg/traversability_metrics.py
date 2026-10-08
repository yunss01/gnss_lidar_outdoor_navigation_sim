"""Auditable cell-level metrics for binary traversability prediction."""

from __future__ import annotations

import math

import torch


class TraversabilityMetricAccumulator:
    """Accumulate confusion, calibration, and uncertainty statistics."""

    def __init__(self, calibration_bins: int = 15):
        if calibration_bins < 2:
            raise ValueError('calibration_bins must be at least 2')
        self.calibration_bins = int(calibration_bins)
        self.confusion = torch.zeros((2, 2), dtype=torch.int64)
        self.bin_count = torch.zeros(calibration_bins, dtype=torch.int64)
        self.bin_confidence_sum = torch.zeros(
            calibration_bins, dtype=torch.float64
        )
        self.bin_correct_sum = torch.zeros(
            calibration_bins, dtype=torch.float64
        )
        self.brier_sum = 0.0
        self.entropy_sum = 0.0
        self.correct_entropy_sum = 0.0
        self.error_entropy_sum = 0.0
        self.known_count = 0
        self.correct_count = 0
        self.error_count = 0

    def update(self, logits, targets) -> None:
        probabilities = torch.softmax(logits.detach(), dim=1)
        known = targets >= 0
        if not torch.any(known):
            return
        obstacle_probability = probabilities[:, 1][known].double().cpu()
        target = targets[known].long().cpu()
        prediction = (obstacle_probability >= 0.5).long()

        indices = target * 2 + prediction
        self.confusion += torch.bincount(indices, minlength=4).reshape(2, 2)
        correctness = prediction.eq(target)
        confidence = torch.maximum(
            obstacle_probability, 1.0 - obstacle_probability
        )
        bin_index = torch.clamp(
            (confidence * self.calibration_bins).long(),
            max=self.calibration_bins - 1,
        )
        self.bin_count += torch.bincount(
            bin_index, minlength=self.calibration_bins
        )
        self.bin_confidence_sum.scatter_add_(
            0, bin_index, confidence
        )
        self.bin_correct_sum.scatter_add_(
            0, bin_index, correctness.double()
        )

        entropy = -(
            obstacle_probability.clamp_min(1e-12).log()
            * obstacle_probability
            + (1.0 - obstacle_probability).clamp_min(1e-12).log()
            * (1.0 - obstacle_probability)
        ) / math.log(2.0)
        self.brier_sum += float(torch.sum(
            (obstacle_probability - target.double()) ** 2
        ))
        self.entropy_sum += float(torch.sum(entropy))
        self.correct_entropy_sum += float(torch.sum(entropy[correctness]))
        self.error_entropy_sum += float(torch.sum(entropy[~correctness]))
        count = int(target.numel())
        correct = int(torch.count_nonzero(correctness))
        self.known_count += count
        self.correct_count += correct
        self.error_count += count - correct

    @staticmethod
    def _ratio(numerator, denominator) -> float:
        return float(numerator / denominator) if denominator else math.nan

    def compute(self) -> dict[str, float | int]:
        true_free = int(self.confusion[0, 0])
        false_obstacle = int(self.confusion[0, 1])
        false_free = int(self.confusion[1, 0])
        true_obstacle = int(self.confusion[1, 1])
        free_iou = self._ratio(
            true_free, true_free + false_obstacle + false_free
        )
        obstacle_iou = self._ratio(
            true_obstacle, true_obstacle + false_obstacle + false_free
        )
        free_recall = self._ratio(true_free, true_free + false_obstacle)
        obstacle_recall = self._ratio(
            true_obstacle, true_obstacle + false_free
        )
        obstacle_precision = self._ratio(
            true_obstacle, true_obstacle + false_obstacle
        )
        obstacle_f1 = self._ratio(
            2 * true_obstacle,
            2 * true_obstacle + false_obstacle + false_free,
        )
        calibration_error = 0.0
        if self.known_count:
            for index in range(self.calibration_bins):
                count = int(self.bin_count[index])
                if not count:
                    continue
                confidence = float(self.bin_confidence_sum[index]) / count
                accuracy = float(self.bin_correct_sum[index]) / count
                calibration_error += count / self.known_count * abs(
                    accuracy - confidence
                )
        valid_ious = [value for value in (free_iou, obstacle_iou)
                      if math.isfinite(value)]
        valid_recalls = [value for value in (free_recall, obstacle_recall)
                         if math.isfinite(value)]
        return {
            'known_cells': self.known_count,
            'true_free': true_free,
            'false_obstacle': false_obstacle,
            'false_free': false_free,
            'true_obstacle': true_obstacle,
            'accuracy': self._ratio(self.correct_count, self.known_count),
            'balanced_accuracy': (
                sum(valid_recalls) / len(valid_recalls)
                if valid_recalls else math.nan
            ),
            'free_iou': free_iou,
            'obstacle_iou': obstacle_iou,
            'mean_iou': (
                sum(valid_ious) / len(valid_ious)
                if valid_ious else math.nan
            ),
            'obstacle_precision': obstacle_precision,
            'obstacle_recall': obstacle_recall,
            'obstacle_f1': obstacle_f1,
            'brier_score': self._ratio(self.brier_sum, self.known_count),
            'expected_calibration_error': calibration_error,
            'mean_entropy': self._ratio(
                self.entropy_sum, self.known_count
            ),
            'correct_mean_entropy': self._ratio(
                self.correct_entropy_sum, self.correct_count
            ),
            'error_mean_entropy': self._ratio(
                self.error_entropy_sum, self.error_count
            ),
        }


class SelectivePredictionAccumulator:
    """Measure residual errors after rejecting low-confidence predictions."""

    def __init__(self, thresholds=(0.50, 0.70, 0.80, 0.90, 0.95, 0.99)):
        values = tuple(float(value) for value in thresholds)
        if not values or any(not 0.5 <= value < 1.0 for value in values):
            raise ValueError('confidence thresholds must be in [0.5, 1.0)')
        self.thresholds = values
        self.known_cells = 0
        self.obstacle_cells = 0
        self.counts = {
            value: {
                'accepted': 0,
                'correct': 0,
                'false_free': 0,
                'false_obstacle': 0,
            }
            for value in values
        }

    def update(self, logits, targets) -> None:
        probabilities = torch.softmax(logits.detach(), dim=1)
        known = targets >= 0
        if not torch.any(known):
            return
        class_probabilities = probabilities.permute(0, 2, 3, 1)[known]
        confidence, prediction = class_probabilities.max(dim=1)
        target = targets[known]
        self.known_cells += int(target.numel())
        self.obstacle_cells += int(torch.count_nonzero(target == 1))
        for threshold in self.thresholds:
            accepted = confidence >= threshold
            selected_prediction = prediction[accepted]
            selected_target = target[accepted]
            values = self.counts[threshold]
            values['accepted'] += int(torch.count_nonzero(accepted))
            values['correct'] += int(torch.count_nonzero(
                selected_prediction == selected_target
            ))
            values['false_free'] += int(torch.count_nonzero(
                (selected_target == 1) & (selected_prediction == 0)
            ))
            values['false_obstacle'] += int(torch.count_nonzero(
                (selected_target == 0) & (selected_prediction == 1)
            ))

    @staticmethod
    def _ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else math.nan

    def compute(self) -> list[dict]:
        output = []
        for threshold in self.thresholds:
            values = self.counts[threshold]
            output.append({
                'confidence_threshold': threshold,
                'accepted_cells': values['accepted'],
                'rejected_cells': self.known_cells - values['accepted'],
                'coverage': self._ratio(
                    values['accepted'], self.known_cells
                ),
                'accepted_accuracy': self._ratio(
                    values['correct'], values['accepted']
                ),
                'residual_false_free': values['false_free'],
                'residual_false_free_fraction_of_all_obstacles': self._ratio(
                    values['false_free'], self.obstacle_cells
                ),
                'residual_false_obstacle': values['false_obstacle'],
            })
        return output
