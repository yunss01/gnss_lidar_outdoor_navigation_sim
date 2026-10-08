import pytest
import torch

from terrain_navigation_pkg.traversability_metrics import (
    SelectivePredictionAccumulator,
    TraversabilityMetricAccumulator,
)


def test_metrics_compute_known_binary_confusion_and_ignore_unknown():
    probabilities = torch.tensor([[
        [[0.9, 0.2, 0.4, 0.1, 0.9]],
        [[0.1, 0.8, 0.6, 0.9, 0.1]],
    ]])
    targets = torch.tensor([[[-1, 1, 0, 1, 0]]])
    metrics = TraversabilityMetricAccumulator(calibration_bins=5)
    metrics.update(probabilities.log(), targets)
    result = metrics.compute()
    assert result['known_cells'] == 4
    assert result['true_free'] == 1
    assert result['false_obstacle'] == 1
    assert result['false_free'] == 0
    assert result['true_obstacle'] == 2
    assert result['accuracy'] == pytest.approx(0.75)
    assert result['obstacle_recall'] == pytest.approx(1.0)
    assert result['obstacle_precision'] == pytest.approx(2.0 / 3.0)


def test_selective_metrics_count_residual_high_confidence_errors():
    probabilities = torch.tensor([[
        [[0.99, 0.40, 0.20, 0.95]],
        [[0.01, 0.60, 0.80, 0.05]],
    ]])
    targets = torch.tensor([[[0, 0, 1, 1]]])
    metrics = SelectivePredictionAccumulator(thresholds=(0.5, 0.9))
    metrics.update(probabilities.log(), targets)
    all_predictions, high_confidence = metrics.compute()
    assert all_predictions['coverage'] == pytest.approx(1.0)
    assert all_predictions['residual_false_free'] == 1
    assert all_predictions['residual_false_obstacle'] == 1
    assert high_confidence['coverage'] == pytest.approx(0.5)
    assert high_confidence['residual_false_free'] == 1
    assert high_confidence['residual_false_obstacle'] == 0
