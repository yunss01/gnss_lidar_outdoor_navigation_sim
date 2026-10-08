import csv

import numpy as np

from terrain_navigation_pkg.traversability_dataset import (
    TraversabilityDataset,
    normalize_lidar_bev,
    read_written_manifest_rows,
    split_rows_by_session,
)


def _sample(path):
    bev = np.zeros((4, 4, 6), dtype=np.float32)
    bev[0, 1, 2] = 1.0
    bev[1, 1, 2] = 0.5
    bev[2, 1, 2] = 3.0
    bev[3, 1, 2] = 2.5
    labels = np.full((4, 6), -1, dtype=np.int8)
    labels[1, 2] = 1
    labels[2, 4] = 0
    np.savez_compressed(path, lidar_bev=bev, target_labels=labels)


def _row(path, session, sample_id):
    return {
        'status': 'written',
        'source_session': session,
        'source_sample_id': str(sample_id),
        'derived_sample_path': str(path),
        'free_cell_count': '1',
        'obstacle_cell_count': '1',
    }


def test_normalize_lidar_bev_masks_empty_cell_heights():
    bev = np.zeros((4, 2, 2), dtype=np.float32)
    bev[2] = 3.0
    bev[0, 0, 1] = 1.0
    normalized = normalize_lidar_bev(bev)
    assert normalized[2, 0, 0] == 0.0
    assert normalized[2, 0, 1] == 1.0


def test_dataset_loads_and_flips_bev_with_target(tmp_path):
    path = tmp_path / 'sample.npz'
    _sample(path)
    dataset = TraversabilityDataset(
        [_row(path, 'session_train', 1)],
        horizontal_flip_probability=1.0,
    )
    item = dataset[0]
    assert tuple(item['lidar_bev'].shape) == (4, 4, 6)
    assert item['lidar_bev'][0, 1, 3] == 1.0
    assert item['target_labels'][1, 3] == 1
    assert item['target_labels'][2, 1] == 0


def test_manifest_split_keeps_whole_sessions_isolated(tmp_path):
    paths = []
    rows = []
    for index, session in enumerate(('train_a', 'train_a', 'validation_b')):
        path = tmp_path / ('sample_{}.npz'.format(index))
        _sample(path)
        paths.append(path)
        rows.append(_row(path, session, index))
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = read_written_manifest_rows(manifest)
    splits = split_rows_by_session(
        loaded, validation_sessions=['validation_b']
    )
    assert len(splits['train']) == 2
    assert len(splits['validation']) == 1
    assert {row['source_session'] for row in splits['train']} == {'train_a'}
