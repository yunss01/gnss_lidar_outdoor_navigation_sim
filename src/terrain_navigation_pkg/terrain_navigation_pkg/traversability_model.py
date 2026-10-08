"""Compact LiDAR-BEV endpoint classifier with uncertainty-ready dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_probability):
        super().__init__()
        layers = [
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3, padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
        ]
        if dropout_probability > 0.0:
            layers.append(nn.Dropout2d(dropout_probability))
        self.layers = nn.Sequential(*layers)

    def forward(self, value):
        return self.layers(value)


class _UpBlock(nn.Module):
    def __init__(
        self, input_channels, skip_channels, output_channels,
        dropout_probability,
    ):
        super().__init__()
        self.block = _ConvBlock(
            input_channels + skip_channels,
            output_channels,
            dropout_probability,
        )

    def forward(self, value, skip):
        value = F.interpolate(
            value, size=skip.shape[-2:], mode='bilinear', align_corners=False
        )
        return self.block(torch.cat((value, skip), dim=1))


class BevTraversabilityUNet(nn.Module):
    """Predict free/obstacle logits for every cell of a geometric LiDAR BEV.

    Semantic labels are never model inputs. Normalized forward/left coordinate
    channels are generated internally because beam density and sensor support
    vary systematically with ego-relative position.
    """

    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 16,
        dropout_probability: float = 0.10,
        use_coordinate_channels: bool = True,
        output_channels: int = 2,
    ):
        super().__init__()
        if input_channels < 1 or base_channels < 4 or output_channels < 1:
            raise ValueError('model channel counts are invalid')
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError('dropout_probability must be in [0, 1)')
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.dropout_probability = float(dropout_probability)
        self.use_coordinate_channels = bool(use_coordinate_channels)
        self.output_channels = int(output_channels)
        effective_input = self.input_channels + (
            2 if self.use_coordinate_channels else 0
        )
        widths = [self.base_channels * factor for factor in (1, 2, 4, 8)]
        self.encoder_1 = _ConvBlock(effective_input, widths[0], 0.0)
        self.encoder_2 = _ConvBlock(widths[0], widths[1], 0.0)
        self.encoder_3 = _ConvBlock(
            widths[1], widths[2], self.dropout_probability * 0.5
        )
        self.bottleneck = _ConvBlock(
            widths[2], widths[3], self.dropout_probability
        )
        self.pool = nn.MaxPool2d(2)
        self.decoder_3 = _UpBlock(
            widths[3], widths[2], widths[2], self.dropout_probability
        )
        self.decoder_2 = _UpBlock(
            widths[2], widths[1], widths[1], self.dropout_probability * 0.5
        )
        self.decoder_1 = _UpBlock(widths[1], widths[0], widths[0], 0.0)
        self.output = nn.Conv2d(
            widths[0], self.output_channels, kernel_size=1
        )

    @staticmethod
    def _coordinate_channels(value):
        batch, _, height, width = value.shape
        forward = torch.linspace(
            1.0, -1.0, height, dtype=value.dtype, device=value.device
        )
        left = torch.linspace(
            1.0, -1.0, width, dtype=value.dtype, device=value.device
        )
        forward_grid, left_grid = torch.meshgrid(
            forward, left, indexing='ij'
        )
        coordinates = torch.stack(
            (forward_grid, left_grid), dim=0
        ).unsqueeze(0)
        return coordinates.expand(batch, -1, -1, -1)

    def forward(self, lidar_bev):
        if lidar_bev.ndim != 4 or lidar_bev.shape[1] != self.input_channels:
            raise ValueError(
                'lidar_bev must have shape (B, {}, H, W)'.format(
                    self.input_channels
                )
            )
        value = lidar_bev
        if self.use_coordinate_channels:
            value = torch.cat(
                (value, self._coordinate_channels(value)), dim=1
            )
        skip_1 = self.encoder_1(value)
        skip_2 = self.encoder_2(self.pool(skip_1))
        skip_3 = self.encoder_3(self.pool(skip_2))
        value = self.bottleneck(self.pool(skip_3))
        value = self.decoder_3(value, skip_3)
        value = self.decoder_2(value, skip_2)
        value = self.decoder_1(value, skip_1)
        return self.output(value)


def masked_cross_entropy(logits, targets, class_weights=None):
    """Cross entropy over direct semantic targets; -1 remains ignored."""
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError('logits must have shape (B, 2, H, W)')
    if targets.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('targets do not match logits')
    known = targets >= 0
    if not torch.any(known):
        raise ValueError('batch contains no known target cells')
    return F.cross_entropy(
        logits,
        targets.to(dtype=torch.long),
        weight=class_weights,
        ignore_index=-1,
    )


def normalized_predictive_entropy(probabilities):
    """Return two-class entropy in [0, 1]."""
    if probabilities.ndim < 2 or probabilities.shape[1] != 2:
        raise ValueError('probabilities must have class dimension 2')
    values = probabilities.clamp_min(torch.finfo(probabilities.dtype).eps)
    return -(values * values.log()).sum(dim=1) / torch.log(
        torch.tensor(2.0, dtype=values.dtype, device=values.device)
    )
