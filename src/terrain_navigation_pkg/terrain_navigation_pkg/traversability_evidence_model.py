"""Independent-head neural model and loss for v2 traversability evidence."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .traversability_model import BevTraversabilityUNet


class BevTraversabilityEvidenceUNet(nn.Module):
    """Predict independent sigmoid passable and obstacle evidence."""

    def __init__(
        self,
        input_channels: int = 8,
        base_channels: int = 16,
        dropout_probability: float = 0.10,
        use_coordinate_channels: bool = True,
    ):
        super().__init__()
        if input_channels < 8:
            raise ValueError('v2 evidence model requires at least 8 channels')
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.dropout_probability = float(dropout_probability)
        self.use_coordinate_channels = bool(use_coordinate_channels)
        self.network = BevTraversabilityUNet(
            input_channels=self.input_channels,
            base_channels=self.base_channels,
            dropout_probability=self.dropout_probability,
            use_coordinate_channels=self.use_coordinate_channels,
        )

    def forward(self, evidence_bev) -> dict[str, torch.Tensor]:
        logits = self.network(evidence_bev)
        return {
            'passable_logits': logits[:, 0],
            'obstacle_logits': logits[:, 1],
        }


class BevTraversabilityDecoupledEvidenceUNet(nn.Module):
    """Use independent U-Nets so the two evidence tasks share no gradients.

    The obstacle head receives controlled-instance and paired-counterfactual
    losses that can conflict with passable-surface learning. Fully separate
    feature extractors make the independent-evidence contract structural,
    rather than merely using two channels of one shared decoder.
    """

    def __init__(
        self,
        input_channels: int = 8,
        base_channels: int = 16,
        dropout_probability: float = 0.10,
        use_coordinate_channels: bool = True,
    ):
        super().__init__()
        if input_channels < 8:
            raise ValueError('v2 evidence model requires at least 8 channels')
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.dropout_probability = float(dropout_probability)
        self.use_coordinate_channels = bool(use_coordinate_channels)
        config = {
            'input_channels': self.input_channels,
            'base_channels': self.base_channels,
            'dropout_probability': self.dropout_probability,
            'use_coordinate_channels': self.use_coordinate_channels,
            'output_channels': 1,
        }
        self.passable_network = BevTraversabilityUNet(**config)
        self.obstacle_network = BevTraversabilityUNet(**config)

    def forward(self, evidence_bev) -> dict[str, torch.Tensor]:
        return {
            'passable_logits': self.passable_network(evidence_bev)[:, 0],
            'obstacle_logits': self.obstacle_network(evidence_bev)[:, 0],
        }


class BevTraversabilityLocalEvidenceNet(nn.Module):
    """Classify each cell from a bounded local geometric neighbourhood.

    The network deliberately omits absolute coordinate channels and pooling.
    Four dilated 3x3 convolutions give a 17x17-cell receptive field, preventing
    distant buildings and roadside layout from becoming object shortcuts.
    """

    def __init__(
        self,
        input_channels: int = 8,
        base_channels: int = 16,
        dropout_probability: float = 0.10,
        use_coordinate_channels: bool = False,
    ):
        super().__init__()
        if input_channels != 8:
            raise ValueError(
                'v2 local evidence model requires exactly 8 input channels'
            )
        if base_channels < 4:
            raise ValueError('base_channels must be at least 4')
        if use_coordinate_channels:
            raise ValueError(
                'local evidence model intentionally forbids coordinates'
            )
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError('dropout_probability must be in [0, 1)')
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.dropout_probability = float(dropout_probability)
        self.use_coordinate_channels = False
        layers = []
        input_width = self.input_channels
        group_count = (
            4 if self.base_channels % 4 == 0
            else (2 if self.base_channels % 2 == 0 else 1)
        )
        for dilation in (1, 1, 2, 4):
            layers.extend([
                nn.Conv2d(
                    input_width,
                    self.base_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(group_count, self.base_channels),
                nn.SiLU(),
            ])
            input_width = self.base_channels
        if self.dropout_probability > 0.0:
            layers.append(nn.Dropout2d(self.dropout_probability))
        self.features = nn.Sequential(*layers)
        self.output = nn.Conv2d(self.base_channels, 2, kernel_size=1)

    def forward(self, evidence_bev) -> dict[str, torch.Tensor]:
        if evidence_bev.ndim != 4 or evidence_bev.shape[1] != 8:
            raise ValueError('evidence_bev must have shape (B, 8, H, W)')
        logits = self.output(self.features(evidence_bev))
        return {
            'passable_logits': logits[:, 0],
            'obstacle_logits': logits[:, 1],
        }


def build_traversability_evidence_model(model_config: dict) -> nn.Module:
    """Build a v2 model while remaining compatible with older checkpoints."""
    config = dict(model_config)
    architecture = config.pop('architecture', 'unet_context')
    if architecture == 'unet_context':
        return BevTraversabilityEvidenceUNet(**config)
    if architecture == 'decoupled_unet_context':
        return BevTraversabilityDecoupledEvidenceUNet(**config)
    if architecture == 'local_evidence':
        return BevTraversabilityLocalEvidenceNet(**config)
    raise ValueError('unknown v2 evidence architecture: ' + architecture)


def _masked_balanced_binary_loss(
    logits,
    targets,
    mask,
    *,
    positive_weight: float,
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise ValueError('evidence head contains no supervised cells')
    target = targets.to(dtype=logits.dtype)
    point_loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction='none'
    )
    weights = torch.where(
        target > 0.5,
        torch.as_tensor(
            positive_weight, dtype=logits.dtype, device=logits.device
        ),
        torch.ones((), dtype=logits.dtype, device=logits.device),
    )
    return (point_loss * weights)[mask].mean()


def _instance_balanced_controlled_obstacle_loss(
    obstacle_logits,
    controlled_obstacle_mask,
) -> torch.Tensor:
    """Average each controlled instance/sample before averaging the batch."""
    losses = []
    for sample_index in range(obstacle_logits.shape[0]):
        mask = controlled_obstacle_mask[sample_index].to(dtype=torch.bool)
        if torch.any(mask):
            loss = F.softplus(-obstacle_logits[sample_index][mask]).mean()
            losses.append(loss)
    if not losses:
        return obstacle_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _paired_counterfactual_obstacle_loss(
    obstacle_logits,
    paired_control_obstacle_logits,
    controlled_obstacle_mask,
    *,
    margin: float,
) -> torch.Tensor:
    """Require actor-present evidence to exceed its same-scene control.

    The first term ranks present above absent with a logit margin.  The second
    term explicitly suppresses obstacle probability in the actor-absent scan
    at the exact controlled-object cells.  Each object/sample has equal mass.
    """
    if paired_control_obstacle_logits.shape != obstacle_logits.shape:
        raise ValueError('paired control logits do not match obstacle logits')
    losses = []
    for sample_index in range(obstacle_logits.shape[0]):
        mask = controlled_obstacle_mask[sample_index].to(dtype=torch.bool)
        if not torch.any(mask):
            continue
        present = obstacle_logits[sample_index][mask]
        absent = paired_control_obstacle_logits[sample_index][mask]
        ranking = F.softplus(float(margin) - (present - absent)).mean()
        absent_suppression = F.softplus(absent).mean()
        losses.append(ranking + absent_suppression)
    if not losses:
        return obstacle_logits.sum() * 0.0
    return torch.stack(losses).mean()


def independent_evidence_loss(
    outputs,
    passable_target,
    obstacle_target,
    known_evidence_mask,
    controlled_obstacle_mask,
    *,
    paired_control_obstacle_logits=None,
    passable_positive_weight: float = 1.0,
    obstacle_positive_weight: float = 1.0,
    controlled_obstacle_auxiliary_weight: float = 1.0,
    paired_counterfactual_weight: float = 0.0,
    paired_counterfactual_margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train independent heads without making unknown a negative label."""
    passable_logits = outputs['passable_logits']
    obstacle_logits = outputs['obstacle_logits']
    expected = passable_target.shape
    if not (
        passable_logits.shape == obstacle_logits.shape == expected
        and obstacle_target.shape == expected
        and known_evidence_mask.shape == expected
        and controlled_obstacle_mask.shape == expected
    ):
        raise ValueError('evidence logits and target shapes do not match')
    passable_loss = _masked_balanced_binary_loss(
        passable_logits,
        passable_target,
        known_evidence_mask,
        positive_weight=passable_positive_weight,
    )
    obstacle_loss = _masked_balanced_binary_loss(
        obstacle_logits,
        obstacle_target,
        known_evidence_mask,
        positive_weight=obstacle_positive_weight,
    )
    controlled_loss = _instance_balanced_controlled_obstacle_loss(
        obstacle_logits, controlled_obstacle_mask
    )
    if paired_control_obstacle_logits is None:
        paired_loss = obstacle_logits.sum() * 0.0
    else:
        paired_loss = _paired_counterfactual_obstacle_loss(
            obstacle_logits,
            paired_control_obstacle_logits,
            controlled_obstacle_mask,
            margin=paired_counterfactual_margin,
        )
    total = (
        passable_loss + obstacle_loss
        + float(controlled_obstacle_auxiliary_weight) * controlled_loss
        + float(paired_counterfactual_weight) * paired_loss
    )
    return total, {
        'passable_loss': passable_loss,
        'obstacle_loss': obstacle_loss,
        'controlled_obstacle_loss': controlled_loss,
        'paired_counterfactual_loss': paired_loss,
    }
