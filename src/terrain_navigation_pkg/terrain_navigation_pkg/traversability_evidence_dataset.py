"""Dataset utilities for independent v2 traversability evidence.

The v2 contract deliberately differs from the legacy two-class dataset.  A
cell is supervised only when it has direct passable-surface or obstacle
evidence.  Ambiguous and unobserved cells remain ignored; low obstacle
evidence is never converted into a passable target.
"""

from __future__ import annotations

import csv
from pathlib import Path
import random

import numpy as np

from .traversability_evidence_core import (
    build_vehicle_clearance_obstacle_mask,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep the ROS package importable without PyTorch.
    torch = None

    class Dataset:  # type: ignore
        pass


REQUIRED_ARRAYS = (
    'lidar_bev',
    'lidar_evidence_bev',
    'local_ground_relative_max_height_m',
    'local_ground_valid_mask',
    'target_passable_surface_mask',
    'target_obstacle_evidence_mask',
    'target_ambiguous_observed_mask',
    'target_observed_mask',
    'target_controlled_obstacle_point_count',
)

EVIDENCE_INPUT_VARIANTS = ('auto', 'current_only', 'temporal')


def evidence_row_key(row) -> tuple[str, int]:
    """Return the stable identity of one derived frame."""
    return str(row['source_session']), int(row['source_sample_id'])


def select_lidar_evidence_bev(arrays, input_variant: str) -> np.ndarray:
    """Select a current-only or temporal view without copying the archive."""
    variant = str(input_variant)
    if variant not in EVIDENCE_INPUT_VARIANTS:
        raise ValueError('unknown evidence input variant: ' + variant)
    primary = np.asarray(arrays['lidar_evidence_bev'], dtype=np.float32)
    if variant == 'auto':
        return primary
    if variant == 'current_only':
        if 'current_lidar_evidence_bev' in arrays:
            current = np.asarray(
                arrays['current_lidar_evidence_bev'], dtype=np.float32
            )
        elif primary.ndim == 3 and primary.shape[0] == 8:
            current = primary
        else:
            raise ValueError(
                'current_only input needs current_lidar_evidence_bev'
            )
        if current.ndim != 3 or current.shape[0] != 8:
            raise ValueError(
                'current_lidar_evidence_bev must have shape (8, H, W)'
            )
        return current
    if primary.ndim != 3 or primary.shape[0] != 17:
        raise ValueError(
            'temporal input needs lidar_evidence_bev shape (17, H, W)'
        )
    return primary


def evidence_pair_key(row) -> tuple[str, str]:
    """Return the physical-scene and matched-pose pairing identity.

    Older frozen manifests have no ``pair_group`` column.  For those files,
    the scene remains the pair group so existing checkpoints and experiments
    retain their original behaviour.
    """
    scene = str(row['scene_id'])
    pair_group = str(row.get('pair_group') or scene)
    return scene, pair_group


def read_evidence_split_manifest(path) -> list[dict]:
    """Read a prepared split manifest and reject incomplete rows."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('split manifest not found: ' + str(path))
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('split manifest contains no rows')
    required = {
        'split', 'scene_id', 'object_family', 'role', 'source_session',
        'source_sample_id', 'derived_sample_path',
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            'split manifest is missing columns: ' + ', '.join(sorted(missing))
        )
    valid_splits = {'train', 'validation', 'test'}
    for row in rows:
        if row['split'] not in valid_splits:
            raise ValueError('invalid split value: ' + row['split'])
        sample = Path(row['derived_sample_path']).expanduser()
        if not sample.is_file():
            raise FileNotFoundError('derived sample not found: ' + str(sample))
    return rows


def split_evidence_rows(rows) -> dict[str, list[dict]]:
    """Group already-frozen rows without making a random split."""
    output = {'train': [], 'validation': [], 'test': []}
    for row in rows:
        output[row['split']].append(row)
    if not output['train']:
        raise ValueError('split manifest contains no training rows')
    if not output['validation']:
        raise ValueError('split manifest contains no validation rows')
    return output


def normalize_lidar_evidence_bev(value) -> np.ndarray:
    """Normalize single-scan or temporal geometric evidence channels.

    Channels 0--3 retain the established legacy normalization.  Channels
    4--5 are local-ground-relative maximum height in metres and are clipped
    to +/-2 m before scaling.  Channels 6--7 are support confidence in [0, 1].
    The temporal variant concatenates current 8, aligned-history 8, and one
    distinct-scan support fraction channel, for 17 channels total.
    """
    bev = np.asarray(value, dtype=np.float32)
    if bev.ndim != 3 or bev.shape[0] not in (8, 17):
        raise ValueError(
            'lidar_evidence_bev must have shape (8, H, W) or (17, H, W)'
        )

    def normalize_eight(channels: np.ndarray) -> np.ndarray:
        output = np.empty_like(channels, dtype=np.float32)
        occupancy = np.clip(channels[0], 0.0, 1.0)
        output[0] = occupancy
        output[1] = np.clip(channels[1], 0.0, 1.0)
        height = np.clip(channels[2], -2.0, 3.0)
        output[2] = np.where(
            occupancy > 0.5,
            2.0 * (height + 2.0) / 5.0 - 1.0,
            0.0,
        )
        output[3] = np.clip(channels[3] / 5.0, 0.0, 1.0)
        output[4:6] = np.clip(channels[4:6] / 2.0, -1.0, 1.0)
        output[6:8] = np.clip(channels[6:8], 0.0, 1.0)
        return output

    if bev.shape[0] == 8:
        return normalize_eight(bev)
    output = np.empty_like(bev, dtype=np.float32)
    output[:8] = normalize_eight(bev[:8])
    output[8:16] = normalize_eight(bev[8:16])
    output[16] = np.clip(bev[16], 0.0, 1.0)
    return output


def family_balanced_sample_weights(rows) -> np.ndarray:
    """Give each object/background family equal total sampling mass."""
    counts: dict[str, int] = {}
    for row in rows:
        family = row.get('object_family') or 'background'
        counts[family] = counts.get(family, 0) + 1
    return np.asarray(
        [
            1.0 / counts[row.get('object_family') or 'background']
            for row in rows
        ],
        dtype=np.float64,
    )


class TraversabilityEvidenceDataset(Dataset):
    """Load 8-channel inputs and independent positive-evidence targets."""

    def __init__(
        self,
        rows,
        *,
        horizontal_flip_probability: float = 0.0,
        return_metadata: bool = False,
        input_variant: str = 'auto',
    ):
        if torch is None:
            raise RuntimeError('PyTorch is required for this dataset')
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError('horizontal_flip_probability must be in [0, 1]')
        self.rows = list(rows)
        if not self.rows:
            raise ValueError('dataset rows cannot be empty')
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.return_metadata = bool(return_metadata)
        if input_variant not in EVIDENCE_INPUT_VARIANTS:
            raise ValueError(
                'unknown evidence input variant: ' + str(input_variant)
            )
        self.input_variant = str(input_variant)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        sample_path = Path(row['derived_sample_path']).expanduser()
        with np.load(sample_path, allow_pickle=False) as arrays:
            missing = [key for key in REQUIRED_ARRAYS if key not in arrays]
            if missing:
                raise ValueError(
                    'v2 sample is missing arrays: ' + ', '.join(missing)
                )
            evidence = select_lidar_evidence_bev(
                arrays, self.input_variant
            ).copy()
            passable = np.asarray(
                arrays['target_passable_surface_mask'], dtype=bool
            ).copy()
            obstacle = np.asarray(
                arrays['target_obstacle_evidence_mask'], dtype=bool
            ).copy()
            ambiguous = np.asarray(
                arrays['target_ambiguous_observed_mask'], dtype=bool
            ).copy()
            observed = np.asarray(
                arrays['target_observed_mask'], dtype=bool
            ).copy()
            controlled_obstacle = np.asarray(
                arrays['target_controlled_obstacle_point_count'] > 0,
                dtype=bool,
            ).copy()
            obstacle_instance_id = (
                np.asarray(
                    arrays['target_obstacle_instance_id'], dtype=np.int64
                )
                .copy()
                if 'target_obstacle_instance_id' in arrays else
                np.full(passable.shape, -1, dtype=np.int64)
            )
            vehicle_clearance_hard_obstacle = (
                build_vehicle_clearance_obstacle_mask(
                    arrays['lidar_bev'],
                    arrays['local_ground_relative_max_height_m'],
                    arrays['local_ground_valid_mask'],
                    exclusion_mask=(
                        arrays['target_ego_exclusion_mask']
                        if 'target_ego_exclusion_mask' in arrays else None
                    ),
                )
            )

        shape = evidence.shape[1:]
        for name, value in (
            ('passable', passable), ('obstacle', obstacle),
            ('ambiguous', ambiguous), ('observed', observed),
            ('controlled_obstacle', controlled_obstacle),
        ):
            if value.shape != shape:
                raise ValueError(name + ' target shape does not match input')
        if np.any(passable & obstacle):
            raise ValueError('passable and obstacle evidence overlap')
        if np.any(ambiguous & (passable | obstacle)):
            raise ValueError('ambiguous evidence overlaps a positive target')
        known = passable | obstacle
        if not np.any(known):
            raise ValueError('sample contains no direct evidence targets')

        evidence = normalize_lidar_evidence_bev(evidence)
        if self.horizontal_flip_probability > 0.0 and (
            random.random() < self.horizontal_flip_probability
        ):
            evidence = np.flip(evidence, axis=2).copy()
            passable = np.flip(passable, axis=1).copy()
            obstacle = np.flip(obstacle, axis=1).copy()
            ambiguous = np.flip(ambiguous, axis=1).copy()
            observed = np.flip(observed, axis=1).copy()
            known = np.flip(known, axis=1).copy()
            controlled_obstacle = np.flip(
                controlled_obstacle, axis=1
            ).copy()
            obstacle_instance_id = np.flip(
                obstacle_instance_id, axis=1
            ).copy()
            vehicle_clearance_hard_obstacle = np.flip(
                vehicle_clearance_hard_obstacle, axis=1
            ).copy()

        item = {
            'lidar_evidence_bev': torch.from_numpy(evidence),
            'passable_target': torch.from_numpy(passable.astype(np.float32)),
            'obstacle_target': torch.from_numpy(obstacle.astype(np.float32)),
            'known_evidence_mask': torch.from_numpy(known),
            'observed_mask': torch.from_numpy(observed),
            'ambiguous_mask': torch.from_numpy(ambiguous),
            'controlled_obstacle_mask': torch.from_numpy(controlled_obstacle),
            'obstacle_instance_id': torch.from_numpy(obstacle_instance_id),
            'passable_support': torch.from_numpy(
                np.max(evidence[6:8], axis=0).astype(np.float32)
            ),
            'vehicle_clearance_hard_obstacle_mask': torch.from_numpy(
                vehicle_clearance_hard_obstacle
            ),
        }
        if self.return_metadata:
            item.update({
                'scene_id': row['scene_id'],
                'pair_group': row.get('pair_group') or row['scene_id'],
                'object_family': row['object_family'],
                'role': row['role'],
                'evaluation_slice': row.get('evaluation_slice', ''),
                'source_session': row['source_session'],
                'source_sample_id': int(row['source_sample_id']),
                'derived_sample_path': str(sample_path),
            })
        return item


class PairedTraversabilityEvidenceDataset(Dataset):
    """Return each sample with its actor-absent same-pose control.

    The paired input supports a causal counterfactual loss: obstacle evidence
    should rise where an actor was added, rather than because a coordinate or
    background texture happened to correlate with the target.
    """

    def __init__(
        self,
        rows,
        *,
        horizontal_flip_probability: float = 0.0,
        input_variant: str = 'auto',
    ):
        if torch is None:
            raise RuntimeError('PyTorch is required for this dataset')
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError('horizontal_flip_probability must be in [0, 1]')
        self.rows = list(rows)
        if not self.rows:
            raise ValueError('dataset rows cannot be empty')
        controls: dict[tuple[str, str], dict] = {}
        for row in self.rows:
            if row.get('role') != 'control':
                continue
            pair_key = evidence_pair_key(row)
            if pair_key in controls:
                raise ValueError(
                    'multiple controls for scene/pair_group: '
                    + '/'.join(pair_key)
                )
            controls[pair_key] = row
        missing = sorted({
            evidence_pair_key(row) for row in self.rows
            if evidence_pair_key(row) not in controls
        })
        if missing:
            raise ValueError(
                'paired training needs one control for every '
                'scene/pair_group: '
                + ', '.join('/'.join(key) for key in missing)
            )
        self.base = TraversabilityEvidenceDataset(
            self.rows,
            horizontal_flip_probability=0.0,
            input_variant=input_variant,
        )
        self.control_datasets = {
            pair_key: TraversabilityEvidenceDataset(
                [row],
                horizontal_flip_probability=0.0,
                input_variant=input_variant,
            )
            for pair_key, row in controls.items()
        }
        self.horizontal_flip_probability = float(
            horizontal_flip_probability
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _flip_item(item: dict) -> None:
        spatial_keys = (
            'lidar_evidence_bev', 'passable_target', 'obstacle_target',
            'known_evidence_mask', 'observed_mask', 'ambiguous_mask',
            'controlled_obstacle_mask', 'obstacle_instance_id',
            'passable_support',
            'vehicle_clearance_hard_obstacle_mask',
        )
        for key in spatial_keys:
            item[key] = torch.flip(item[key], dims=(-1,))

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        item = self.base[index]
        control = self.control_datasets[evidence_pair_key(row)][0]
        if item['lidar_evidence_bev'].shape != control[
            'lidar_evidence_bev'
        ].shape:
            raise ValueError('sample/control BEV shapes do not match')
        if self.horizontal_flip_probability > 0.0 and (
            random.random() < self.horizontal_flip_probability
        ):
            self._flip_item(item)
            self._flip_item(control)
        item['paired_control_lidar_evidence_bev'] = control[
            'lidar_evidence_bev'
        ]
        return item
