"""PyTorch dataset for clean LiDAR-BEV local-trajectory imitation data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .navigation_trajectory_core import INITIAL_BC_EXCLUSION_FLAGS

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Keep the non-learning ROS stack usable without PyTorch.
    torch = None

    class Dataset:  # type: ignore
        pass


def normalize_lidar_bev(
    lidar_bev,
    *,
    z_min_m: float = -2.0,
    z_max_m: float = 3.0,
) -> np.ndarray:
    """Normalize the recorder's four LiDAR BEV channels for a CNN."""
    bev = np.asarray(lidar_bev, dtype=np.float32)
    if bev.ndim != 3 or bev.shape[0] != 4:
        raise ValueError('lidar_bev must have shape (4, H, W)')
    if z_max_m <= z_min_m:
        raise ValueError('z_max_m must exceed z_min_m')

    output = np.empty_like(bev, dtype=np.float32)
    occupancy = np.clip(bev[0], 0.0, 1.0)
    output[0] = occupancy
    output[1] = np.clip(bev[1], 0.0, 1.0)

    height = np.clip(bev[2], z_min_m, z_max_m)
    height = 2.0 * (height - z_min_m) / (z_max_m - z_min_m) - 1.0
    output[2] = np.where(occupancy > 0.5, height, 0.0)
    output[3] = np.clip(
        bev[3] / (z_max_m - z_min_m), 0.0, 1.0
    )
    return output


def _flag_set(value) -> set[str]:
    return set(filter(None, str(value).split(';')))


class NavigationTrajectoryDataset(Dataset):
    """Load one LiDAR BEV, relative goal, and masked local trajectory."""

    def __init__(
        self,
        target_archive,
        *,
        normalize_bev: bool = True,
        goal_scale_m: float = 30.0,
        return_metadata: bool = False,
        allow_excluded: bool = False,
        verify_sample_paths: bool = True,
    ):
        if torch is None:
            raise RuntimeError(
                'PyTorch is required for NavigationTrajectoryDataset'
            )
        if goal_scale_m <= 0.0:
            raise ValueError('goal_scale_m must be positive')
        self.archive_path = Path(target_archive).expanduser().resolve()
        self.normalize_bev = bool(normalize_bev)
        self.goal_scale_m = float(goal_scale_m)
        self.return_metadata = bool(return_metadata)

        required = {
            'sample_path', 'session', 'sample_id', 'speed_mps',
            'target_points', 'target_mask', 'goal_vehicle_xy',
            'quality_flags', 'trajectory_horizon_m',
        }
        with np.load(self.archive_path, allow_pickle=False) as arrays:
            missing = sorted(required.difference(arrays.files))
            if missing:
                raise ValueError(
                    'target archive is missing: {}'.format(', '.join(missing))
                )
            self.sample_paths = np.asarray(arrays['sample_path']).astype(str)
            self.sessions = np.asarray(arrays['session']).astype(str)
            self.sample_ids = np.asarray(
                arrays['sample_id'], dtype=np.int64
            ).copy()
            self.speed_mps = np.asarray(
                arrays['speed_mps'], dtype=np.float32
            ).copy()
            self.target_points = np.asarray(
                arrays['target_points'], dtype=np.float32
            ).copy()
            self.target_mask = np.asarray(
                arrays['target_mask'], dtype=bool
            ).copy()
            self.goal_vehicle_xy = np.asarray(
                arrays['goal_vehicle_xy'], dtype=np.float32
            ).copy()
            self.quality_flags = np.asarray(
                arrays['quality_flags']
            ).astype(str)
            self.trajectory_horizon_m = float(
                np.asarray(arrays['trajectory_horizon_m']).item()
            )

        count = self.sample_paths.shape[0]
        expected_lengths = {
            'session': self.sessions.shape[0],
            'sample_id': self.sample_ids.shape[0],
            'speed_mps': self.speed_mps.shape[0],
            'target_points': self.target_points.shape[0],
            'target_mask': self.target_mask.shape[0],
            'goal_vehicle_xy': self.goal_vehicle_xy.shape[0],
            'quality_flags': self.quality_flags.shape[0],
        }
        mismatched = {
            name: length for name, length in expected_lengths.items()
            if length != count
        }
        if mismatched:
            raise ValueError('inconsistent archive lengths: {}'.format(
                mismatched
            ))
        if self.target_points.ndim != 3:
            raise ValueError('target_points must have shape (N, T, 2)')
        if self.target_points.shape[2] != 2:
            raise ValueError('target_points must contain XY coordinates')
        if self.target_mask.shape != self.target_points.shape[:2]:
            raise ValueError('target_mask shape does not match target_points')
        if self.goal_vehicle_xy.shape != (count, 2):
            raise ValueError('goal_vehicle_xy must have shape (N, 2)')
        empty_targets = np.flatnonzero(~self.target_mask.any(axis=1))
        if empty_targets.size:
            raise ValueError(
                '{} samples contain no valid trajectory target; regenerate '
                'the clean target archive'.format(empty_targets.size)
            )

        excluded = [
            index for index, flags in enumerate(self.quality_flags)
            if _flag_set(flags).intersection(INITIAL_BC_EXCLUSION_FLAGS)
        ]
        if excluded and not allow_excluded:
            raise ValueError(
                '{} samples contain initial-BC exclusion flags; use a clean '
                'target archive'.format(len(excluded))
            )
        if verify_sample_paths:
            missing_paths = [
                path for path in self.sample_paths if not Path(path).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    'raw sample does not exist: {}'.format(missing_paths[0])
                )

    def __len__(self) -> int:
        return int(self.sample_paths.shape[0])

    def __getitem__(self, index: int) -> dict:
        sample_path = Path(self.sample_paths[index])
        with np.load(sample_path, allow_pickle=False) as arrays:
            if 'lidar_bev' not in arrays.files:
                raise ValueError(
                    'raw sample has no lidar_bev: {}'.format(sample_path)
                )
            lidar_bev = np.asarray(
                arrays['lidar_bev'], dtype=np.float32
            )
        if self.normalize_bev:
            lidar_bev = normalize_lidar_bev(lidar_bev)
        else:
            lidar_bev = lidar_bev.copy()

        goal = self.goal_vehicle_xy[index].copy()
        targets = self.target_points[index].copy()
        item = {
            'lidar_bev': torch.from_numpy(lidar_bev),
            'goal_vehicle_xy_m': torch.from_numpy(goal),
            'goal_vehicle_xy_normalized': torch.from_numpy(np.clip(
                goal / self.goal_scale_m, -1.0, 1.0
            ).astype(np.float32)),
            'target_points_m': torch.from_numpy(targets),
            'target_points_normalized': torch.from_numpy(
                (targets / self.trajectory_horizon_m).astype(np.float32)
            ),
            'target_mask': torch.from_numpy(
                self.target_mask[index].copy()
            ),
            'speed_mps': torch.tensor(
                self.speed_mps[index], dtype=torch.float32
            ),
        }
        if self.return_metadata:
            item.update({
                'sample_path': str(sample_path),
                'session': str(self.sessions[index]),
                'sample_id': int(self.sample_ids[index]),
                'quality_flags': str(self.quality_flags[index]),
            })
        return item
