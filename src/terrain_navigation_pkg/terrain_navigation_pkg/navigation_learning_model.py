"""Small BEV-conditioned local trajectory policy for the first baseline."""

from __future__ import annotations

import torch
from torch import nn


class BevTrajectoryPolicy(nn.Module):
    """Predict fixed-horizon XY targets from LiDAR BEV and a relative goal."""

    def __init__(
        self,
        target_count: int = 12,
        spatial_pool_size: int = 1,
        use_coordinate_channels: bool = False,
    ):
        super().__init__()
        if target_count < 1:
            raise ValueError('target_count must be positive')
        if spatial_pool_size < 1:
            raise ValueError('spatial_pool_size must be positive')
        self.target_count = int(target_count)
        self.spatial_pool_size = int(spatial_pool_size)
        self.use_coordinate_channels = bool(use_coordinate_channels)
        input_channels = 6 if self.use_coordinate_channels else 4
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=5, stride=2, padding=2
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 192),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(self.spatial_pool_size),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(
                192 * self.spatial_pool_size ** 2 + 64, 256
            ),
            nn.SiLU(),
            nn.Linear(256, self.target_count * 2),
        )

    def forward(self, lidar_bev: torch.Tensor, goal_xy: torch.Tensor):
        if lidar_bev.ndim != 4 or lidar_bev.shape[1] != 4:
            raise ValueError('lidar_bev must have shape (B, 4, H, W)')
        if goal_xy.ndim != 2 or goal_xy.shape[1] != 2:
            raise ValueError('goal_xy must have shape (B, 2)')
        if lidar_bev.shape[0] != goal_xy.shape[0]:
            raise ValueError('lidar_bev and goal_xy batch sizes differ')
        if self.use_coordinate_channels:
            batch, _, height, width = lidar_bev.shape
            forward = torch.linspace(
                1.0, -1.0, height,
                dtype=lidar_bev.dtype, device=lidar_bev.device,
            )
            left = torch.linspace(
                1.0, -1.0, width,
                dtype=lidar_bev.dtype, device=lidar_bev.device,
            )
            forward_grid, left_grid = torch.meshgrid(
                forward, left, indexing='ij'
            )
            coordinates = torch.stack(
                (forward_grid, left_grid), dim=0
            ).unsqueeze(0).expand(batch, -1, -1, -1)
            lidar_bev = torch.cat((lidar_bev, coordinates), dim=1)
        bev_features = self.bev_encoder(lidar_bev).flatten(1)
        goal_distance = torch.linalg.vector_norm(
            goal_xy, dim=1, keepdim=True
        )
        goal_features = self.goal_encoder(
            torch.cat((goal_xy, goal_distance), dim=1)
        )
        output = self.head(torch.cat((bev_features, goal_features), dim=1))
        return output.view(-1, self.target_count, 2)


def masked_trajectory_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only on targets that exist in the teacher path."""
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes differ')
    if target_mask.shape != prediction.shape[:2]:
        raise ValueError('target_mask shape does not match trajectory')
    mask = target_mask.to(dtype=prediction.dtype).unsqueeze(-1)
    squared_error = (prediction - target).square() * mask
    denominator = mask.sum() * prediction.shape[-1]
    if denominator.item() <= 0.0:
        raise ValueError('target_mask contains no valid target')
    return squared_error.sum() / denominator
