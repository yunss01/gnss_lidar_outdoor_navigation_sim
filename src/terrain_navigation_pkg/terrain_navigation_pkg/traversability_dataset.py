"""Session-isolated dataset utilities for LiDAR-BEV traversability."""

from __future__ import annotations

import csv
from pathlib import Path
import random

import numpy as np

from .navigation_learning_recorder_core import BevGeometry
from .traversability_stress_core import apply_random_sensor_augmentation

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep the ROS package importable without PyTorch.
    torch = None

    class Dataset:  # type: ignore
        pass


def normalize_lidar_bev(lidar_bev) -> np.ndarray:
    """Normalize the four recorder BEV channels without semantic inputs."""
    bev = np.asarray(lidar_bev, dtype=np.float32)
    if bev.ndim != 3 or bev.shape[0] != 4:
        raise ValueError('lidar_bev must have shape (4, H, W)')

    output = np.empty_like(bev, dtype=np.float32)
    occupancy = np.clip(bev[0], 0.0, 1.0)
    output[0] = occupancy
    output[1] = np.clip(bev[1], 0.0, 1.0)

    # Recorder geometry uses [-2, 3] m. Empty-cell height values are masked so
    # zero input cannot be confused with an observed point at sensor height.
    height = np.clip(bev[2], -2.0, 3.0)
    output[2] = np.where(
        occupancy > 0.5, 2.0 * (height + 2.0) / 5.0 - 1.0, 0.0
    )
    output[3] = np.clip(bev[3] / 5.0, 0.0, 1.0)
    return output


def read_written_manifest_rows(manifest_path) -> list[dict]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError('manifest not found: ' + str(manifest_path))
    with manifest_path.open(newline='', encoding='utf-8') as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get('status') == 'written'
        ]
    if not rows:
        raise ValueError('manifest contains no written samples')
    return rows


def split_rows_by_session(
    rows,
    *,
    validation_sessions,
    test_sessions=(),
) -> dict[str, list[dict]]:
    """Split whole sessions; frame-level random splitting is prohibited."""
    validation = set(validation_sessions)
    test = set(test_sessions)
    overlap = validation.intersection(test)
    if overlap:
        raise ValueError(
            'validation and test sessions overlap: ' + ', '.join(overlap)
        )
    available = {row['source_session'] for row in rows}
    missing = (validation | test).difference(available)
    if missing:
        raise ValueError(
            'requested split sessions are absent: '
            + ', '.join(sorted(missing))
        )

    output = {'train': [], 'validation': [], 'test': []}
    for row in rows:
        session = row['source_session']
        if session in test:
            output['test'].append(row)
        elif session in validation:
            output['validation'].append(row)
        else:
            output['train'].append(row)
    if not output['train']:
        raise ValueError('session split leaves no training samples')
    if not output['validation']:
        raise ValueError('session split leaves no validation samples')
    return output


def label_counts(rows) -> tuple[int, int]:
    free = sum(int(row.get('free_cell_count', 0)) for row in rows)
    obstacle = sum(int(row.get('obstacle_cell_count', 0)) for row in rows)
    return free, obstacle


class TraversabilityDataset(Dataset):
    """Load geometric LiDAR BEV input and free/obstacle/unknown targets."""

    def __init__(
        self,
        rows,
        *,
        horizontal_flip_probability: float = 0.0,
        sensor_augmentation_probability: float = 0.0,
        sensor_augmentation_parameters: dict | None = None,
        return_raw_bev: bool = False,
        return_metadata: bool = False,
    ):
        if torch is None:
            raise RuntimeError('PyTorch is required for TraversabilityDataset')
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError('horizontal_flip_probability must be in [0, 1]')
        if not 0.0 <= sensor_augmentation_probability <= 1.0:
            raise ValueError(
                'sensor_augmentation_probability must be in [0, 1]'
            )
        self.rows = list(rows)
        if not self.rows:
            raise ValueError('dataset rows cannot be empty')
        self.horizontal_flip_probability = float(
            horizontal_flip_probability
        )
        self.sensor_augmentation_probability = float(
            sensor_augmentation_probability
        )
        self.sensor_augmentation_parameters = dict(
            sensor_augmentation_parameters or {}
        )
        self.geometry = BevGeometry()
        self.return_metadata = bool(return_metadata)
        self.return_raw_bev = bool(return_raw_bev)

        for row in self.rows:
            path = Path(row['derived_sample_path']).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    'derived sample not found: ' + str(path)
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        sample_path = Path(row['derived_sample_path']).expanduser()
        with np.load(sample_path, allow_pickle=False) as arrays:
            lidar_bev = np.asarray(
                arrays['lidar_bev'], dtype=np.float32
            ).copy()
            target_labels = np.asarray(
                arrays['target_labels'], dtype=np.int64
            ).copy()
            ego_mask = (
                np.asarray(
                    arrays['target_ego_exclusion_mask'], dtype=bool
                ).copy()
                if self.return_raw_bev else None
            )

        if target_labels.shape != lidar_bev.shape[1:]:
            raise ValueError('target shape does not match LiDAR BEV')
        if self.sensor_augmentation_probability > 0.0 and (
            random.random() < self.sensor_augmentation_probability
        ):
            lidar_bev, target_labels = apply_random_sensor_augmentation(
                lidar_bev,
                target_labels,
                self.geometry,
                random.randrange(2 ** 32),
                **self.sensor_augmentation_parameters,
            )
        raw_lidar_bev = lidar_bev
        lidar_bev = normalize_lidar_bev(raw_lidar_bev)
        if self.horizontal_flip_probability > 0.0 and (
            random.random() < self.horizontal_flip_probability
        ):
            lidar_bev = np.flip(lidar_bev, axis=2).copy()
            target_labels = np.flip(target_labels, axis=1).copy()

        item = {
            'lidar_bev': torch.from_numpy(lidar_bev),
            'target_labels': torch.from_numpy(target_labels),
        }
        if self.return_raw_bev:
            item['raw_lidar_bev'] = torch.from_numpy(raw_lidar_bev)
            item['ego_exclusion_mask'] = torch.from_numpy(ego_mask)
        if self.return_metadata:
            item.update({
                'source_session': row['source_session'],
                'source_sample_id': int(row['source_sample_id']),
                'derived_sample_path': str(sample_path),
            })
        return item
