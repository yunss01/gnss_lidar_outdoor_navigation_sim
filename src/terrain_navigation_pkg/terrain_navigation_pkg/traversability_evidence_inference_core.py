"""Pure online preprocessing and inference for current-only v2 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
)
from .traversability_evidence_core import (
    LocalGroundEvidence,
    build_evidence_bev,
    build_vehicle_clearance_obstacle_mask,
)
from .traversability_evidence_dataset import (
    normalize_lidar_evidence_bev,
)
from .traversability_evidence_model import (
    build_traversability_evidence_model,
)
from .traversability_learning_core import EgoFootprint, ego_footprint_mask


@dataclass(frozen=True)
class CurrentEvidenceInput:
    """One raw scan transformed exactly like an 8-channel training input."""

    lidar_bev: np.ndarray
    evidence_bev: np.ndarray
    normalized_evidence_bev: np.ndarray
    local_ground: LocalGroundEvidence
    observed_mask: np.ndarray
    passable_support: np.ndarray
    hard_obstacle_mask: np.ndarray
    ego_exclusion_mask: np.ndarray


@dataclass(frozen=True)
class EvidencePrediction:
    """Independent probabilities and optional MC-dropout variances."""

    passable_probability: np.ndarray
    obstacle_probability: np.ndarray
    passable_variance: np.ndarray
    obstacle_variance: np.ndarray


def build_current_evidence_input(
    points_xyz: np.ndarray,
    geometry: BevGeometry,
    footprint: EgoFootprint,
    *,
    local_ground_radii_m=(0.75, 1.50),
    local_ground_quantile: float = 0.25,
    local_ground_minimum_support_cells: int = 4,
    hard_obstacle_minimum_relative_height_m: float = 0.15,
    hard_obstacle_minimum_vertical_span_m: float = 0.15,
    hard_obstacle_minimum_absolute_height_m: float = -1.40,
) -> CurrentEvidenceInput:
    """Build the exact current-only view consumed by the v2 dataset.

    The recorder stores the four-channel LiDAR BEV as float16 before the v2
    dataset builder reads it.  The builder then stores the resulting
    eight-channel evidence as float16 as well.  Reproducing both boundaries
    here is intentional: otherwise an online scan and its archived training
    counterpart can cross local-ground and decision thresholds differently.
    Arrays are converted back to float32 after each boundary so the model and
    safety rules still execute in float32, exactly as they do when a saved
    sample is loaded by the training dataset.
    """

    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('points_xyz must have shape (N, 3+)')
    raw_lidar_bev = build_lidar_bev(points[:, :3], geometry)
    lidar_bev = raw_lidar_bev.astype(np.float16).astype(np.float32)
    raw_evidence_bev, raw_local_ground = build_evidence_bev(
        lidar_bev,
        geometry,
        radii_m=tuple(float(value) for value in local_ground_radii_m),
        ground_quantile=float(local_ground_quantile),
        minimum_support_cells=int(
            local_ground_minimum_support_cells
        ),
    )
    evidence_bev = raw_evidence_bev.astype(np.float16).astype(np.float32)
    local_ground = LocalGroundEvidence(
        ground_height=(
            raw_local_ground.ground_height
            .astype(np.float16).astype(np.float32)
        ),
        relative_max_height=(
            raw_local_ground.relative_max_height
            .astype(np.float16).astype(np.float32)
        ),
        support_count=raw_local_ground.support_count,
        support_confidence=(
            raw_local_ground.support_confidence
            .astype(np.float16).astype(np.float32)
        ),
        valid_mask=raw_local_ground.valid_mask,
    )
    ego_mask = ego_footprint_mask(geometry, footprint)
    observed = (lidar_bev[0] > 0.5) & ~ego_mask
    hard_obstacle = build_vehicle_clearance_obstacle_mask(
        lidar_bev,
        local_ground.relative_max_height,
        local_ground.valid_mask,
        minimum_relative_height_m=float(
            hard_obstacle_minimum_relative_height_m
        ),
        minimum_vertical_span_m=float(
            hard_obstacle_minimum_vertical_span_m
        ),
        minimum_absolute_height_m=float(
            hard_obstacle_minimum_absolute_height_m
        ),
        exclusion_mask=ego_mask,
    )
    normalized = normalize_lidar_evidence_bev(evidence_bev)
    support = np.max(evidence_bev[6:8], axis=0).astype(np.float32)
    return CurrentEvidenceInput(
        lidar_bev=lidar_bev,
        evidence_bev=evidence_bev,
        normalized_evidence_bev=normalized,
        local_ground=local_ground,
        observed_mask=observed,
        passable_support=support,
        hard_obstacle_mask=hard_obstacle,
        ego_exclusion_mask=ego_mask,
    )


def load_current_only_evidence_checkpoint(path, device):
    """Load only a validated 8-channel independent-evidence checkpoint."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'evidence checkpoint not found: ' + str(checkpoint_path)
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    training = checkpoint.get('training_config', {})
    if training.get('target_contract') != (
        'independent_passable_obstacle_evidence'
    ):
        raise ValueError(
            'checkpoint is not an independent v2 evidence model'
        )
    input_variant = training.get('evidence_input_variant', 'current_only')
    if input_variant != 'current_only':
        raise ValueError(
            'online node requires a current_only checkpoint, got '
            + str(input_variant)
        )
    model_config = dict(checkpoint.get('model_config', {}))
    if int(model_config.get('input_channels', 0)) != 8:
        raise ValueError('online node requires exactly 8 input channels')
    model = build_traversability_evidence_model(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint, checkpoint_path


def predict_evidence(
    model,
    normalized_evidence_bev: np.ndarray,
    device,
    *,
    mc_samples: int = 1,
) -> EvidencePrediction:
    """Predict independent evidence without changing spatial conventions."""

    evidence = np.asarray(normalized_evidence_bev, dtype=np.float32)
    if evidence.ndim != 3 or evidence.shape[0] != 8:
        raise ValueError('normalized evidence must have shape (8, H, W)')
    if not np.all(np.isfinite(evidence)):
        raise ValueError('normalized evidence contains non-finite values')
    if mc_samples < 1:
        raise ValueError('mc_samples must be positive')

    tensor = torch.from_numpy(evidence).unsqueeze(0).to(device)
    passable = []
    obstacle = []
    model.eval()
    with torch.no_grad():
        if mc_samples > 1:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout2d):
                    module.train()
        for _ in range(mc_samples):
            outputs = model(tensor)
            passable.append(torch.sigmoid(outputs['passable_logits'][0]))
            obstacle.append(torch.sigmoid(outputs['obstacle_logits'][0]))
    model.eval()

    passable_stack = torch.stack(passable)
    obstacle_stack = torch.stack(obstacle)
    return EvidencePrediction(
        passable_probability=(
            passable_stack.mean(dim=0).cpu().numpy().astype(np.float32)
        ),
        obstacle_probability=(
            obstacle_stack.mean(dim=0).cpu().numpy().astype(np.float32)
        ),
        passable_variance=(
            passable_stack.var(dim=0, unbiased=False)
            .cpu().numpy().astype(np.float32)
        ),
        obstacle_variance=(
            obstacle_stack.var(dim=0, unbiased=False)
            .cpu().numpy().astype(np.float32)
        ),
    )
