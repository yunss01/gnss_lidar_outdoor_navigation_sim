import csv

import numpy as np

from terrain_navigation_pkg.visualize_traversability_dataset import (
    generate_visualizations,
    random_selection,
    read_manifest,
    render_sample_tile,
    sequence_selection,
)


def _write_fixture(tmp_path, count=5):
    sample_directory = tmp_path / 'samples'
    sample_directory.mkdir()
    rows = []
    for index in range(1, count + 1):
        path = sample_directory / ('sample_%06d.npz' % index)
        labels = np.full((4, 4), -1, dtype=np.int8)
        labels[0, 0] = 0
        labels[0, 1] = 1
        ego = np.zeros((4, 4), dtype=bool)
        ego[3, 1:3] = True
        visibility = np.zeros((4, 4), dtype=bool)
        visibility[1:3, 1:3] = True
        np.savez_compressed(
            path,
            lidar_bev=np.zeros((4, 4, 4), dtype=np.float16),
            target_labels=labels,
            target_obstacle_point_count=(labels == 1).astype(np.int32),
            target_vertical_span_m=np.zeros((4, 4), dtype=np.float32),
            target_ego_exclusion_mask=ego,
            visibility_free_mask=visibility,
            visibility_ray_count=visibility.astype(np.uint16),
        )
        rows.append({
            'source_session': 'session_test',
            'source_session_result': 'completed',
            'source_sample_id': index,
            'derived_sample_path': str(path),
            'status': 'written',
            'route_index': 1,
            'free_cell_count': 1,
            'obstacle_cell_count': 1,
            'visibility_free_cell_count': 4,
            'ego_excluded_cell_count': 2,
        })
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, rows


def test_selection_and_tile_rendering_are_deterministic(tmp_path):
    manifest, source_rows = _write_fixture(tmp_path)
    rows = read_manifest(manifest, {'completed'})

    assert len(rows) == len(source_rows)
    assert random_selection(rows, 3, 42) == random_selection(rows, 3, 42)
    assert [row['source_sample_id'] for row in sequence_selection(rows, 3)] == [
        '1', '3', '5',
    ]
    tile = render_sample_tile(rows[0], cell_scale=2)
    assert tile.width > 16
    assert tile.height > 16


def test_generate_visualizations_writes_png_pdf_and_summary(tmp_path):
    manifest, _ = _write_fixture(tmp_path, count=2)
    output = tmp_path / 'visualizations'

    summary = generate_visualizations(
        manifest, output, count=2, seed=7, dpi=300
    )

    assert summary['eligible_samples'] == 2
    assert (output / 'random_contact_sheet_01.png').is_file()
    assert (output / 'random_contact_sheet.pdf').is_file()
    assert (output / 'sequence_contact_sheet_01.png').is_file()
    assert (output / 'sequence_contact_sheet.pdf').is_file()
    assert (output / 'summary.json').is_file()
