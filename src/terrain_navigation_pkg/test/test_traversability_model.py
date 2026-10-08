import math

import pytest
import torch

from terrain_navigation_pkg.traversability_model import (
    BevTraversabilityUNet,
    masked_cross_entropy,
    normalized_predictive_entropy,
)


def test_model_preserves_bev_shape_and_supports_backward():
    model = BevTraversabilityUNet(base_channels=4)
    lidar_bev = torch.randn(2, 4, 32, 40)
    targets = torch.full((2, 32, 40), -1, dtype=torch.long)
    targets[:, 4:10, 5:12] = 0
    targets[:, 15:20, 20:25] = 1
    logits = model(lidar_bev)
    loss = masked_cross_entropy(logits, targets)
    loss.backward()
    assert logits.shape == (2, 2, 32, 40)
    assert math.isfinite(float(loss.detach()))


def test_masked_loss_rejects_batch_without_known_targets():
    logits = torch.randn(1, 2, 4, 4)
    targets = torch.full((1, 4, 4), -1)
    with pytest.raises(ValueError, match='no known'):
        masked_cross_entropy(logits, targets)


def test_normalized_entropy_has_expected_limits():
    certain = torch.tensor([[[[1.0]], [[0.0]]]])
    uniform = torch.tensor([[[[0.5]], [[0.5]]]])
    assert normalized_predictive_entropy(certain).item() < 1e-5
    assert normalized_predictive_entropy(uniform).item() == pytest.approx(1.0)
