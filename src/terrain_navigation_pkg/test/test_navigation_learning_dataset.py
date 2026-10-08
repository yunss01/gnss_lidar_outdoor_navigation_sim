from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from terrain_navigation_pkg.navigation_learning_dataset import (  # noqa:E402
    NavigationTrajectoryDataset,
    normalize_lidar_bev,
)


def _write_archive(path: Path, sample_path: Path, flags: str = '') -> None:
    np.savez_compressed(
        path,
        sample_path=np.asarray([str(sample_path)]),
        session=np.asarray(['session_test']),
        sample_id=np.asarray([7], dtype=np.int32),
        speed_mps=np.asarray([1.25], dtype=np.float32),
        target_points=np.asarray([[
            [0.75, 0.0], [1.5, 0.2], [2.25, 0.3],
        ]], dtype=np.float32),
        target_mask=np.asarray([[True, True, False]]),
        goal_vehicle_xy=np.asarray([[15.0, -7.5]], dtype=np.float32),
        quality_flags=np.asarray([flags]),
        trajectory_horizon_m=np.asarray(9.0, dtype=np.float32),
    )


def test_normalize_lidar_bev_preserves_empty_height_as_neutral():
    bev = np.zeros((4, 2, 2), dtype=np.float32)
    bev[0, 0, 0] = 1.0
    bev[1, 0, 0] = 0.5
    bev[2, 0, 0] = 3.0
    bev[2, 1, 1] = -2.0
    bev[3, 0, 0] = 2.5
    result = normalize_lidar_bev(bev)
    assert result.dtype == np.float32
    assert result[2, 0, 0] == 1.0
    assert result[2, 1, 1] == 0.0
    assert result[3, 0, 0] == 0.5


def test_dataset_loads_clean_masked_trajectory(tmp_path):
    raw_path = tmp_path / 'sample_000007.npz'
    np.savez_compressed(
        raw_path, lidar_bev=np.zeros((4, 4, 5), dtype=np.float16)
    )
    archive = tmp_path / 'trajectory_targets_clean_train.npz'
    _write_archive(archive, raw_path)
    dataset = NavigationTrajectoryDataset(
        archive, return_metadata=True
    )
    item = dataset[0]
    assert len(dataset) == 1
    assert item['lidar_bev'].shape == (4, 4, 5)
    assert item['lidar_bev'].dtype == torch.float32
    torch.testing.assert_close(
        item['goal_vehicle_xy_normalized'],
        torch.tensor([0.5, -0.25]),
    )
    torch.testing.assert_close(
        item['target_points_normalized'][0],
        torch.tensor([0.75 / 9.0, 0.0]),
    )
    assert item['target_mask'].tolist() == [True, True, False]
    assert item['session'] == 'session_test'
    assert item['sample_id'] == 7


def test_dataset_rejects_winding_teacher_archive(tmp_path):
    raw_path = tmp_path / 'sample_000007.npz'
    np.savez_compressed(
        raw_path, lidar_bev=np.zeros((4, 4, 5), dtype=np.float16)
    )
    archive = tmp_path / 'trajectory_targets_train.npz'
    _write_archive(archive, raw_path, flags='winding_path')
    with pytest.raises(ValueError, match='exclusion flags'):
        NavigationTrajectoryDataset(archive)
