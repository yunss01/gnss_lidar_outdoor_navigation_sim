import pytest

torch = pytest.importorskip('torch')

from terrain_navigation_pkg.navigation_learning_model import (  # noqa: E402
    BevTrajectoryPolicy,
    masked_trajectory_mse,
)


def test_policy_returns_fixed_horizon_targets():
    model = BevTrajectoryPolicy(target_count=12)
    prediction = model(
        torch.zeros(2, 4, 160, 160), torch.zeros(2, 2)
    )
    assert prediction.shape == (2, 12, 2)
    assert torch.isfinite(prediction).all()


def test_spatial_policy_accepts_coordinate_channels():
    model = BevTrajectoryPolicy(
        target_count=12,
        spatial_pool_size=5,
        use_coordinate_channels=True,
    )
    prediction = model(
        torch.zeros(2, 4, 160, 160), torch.zeros(2, 2)
    )
    assert prediction.shape == (2, 12, 2)
    assert model.bev_encoder[0].in_channels == 6
    assert model.head[0].in_features == 192 * 25 + 64


def test_masked_mse_ignores_invalid_padded_targets():
    prediction = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    target[0, 2] = 100.0
    mask = torch.tensor([[True, True, False]])
    assert masked_trajectory_mse(prediction, target, mask).item() == 0.0
