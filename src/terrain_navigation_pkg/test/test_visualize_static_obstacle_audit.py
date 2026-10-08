import json

import numpy as np
from PIL import Image

from terrain_navigation_pkg.visualize_static_obstacle_audit import (
    compute_spatial_diagnostics,
    compute_v2_model_diagnostics,
    generate_diagnostic,
)


def _fixture(tmp_path):
    audit = tmp_path / 'static_audit_test'
    snapshots = audit / 'snapshots'
    snapshots.mkdir(parents=True)
    metadata = {
        'bev_geometry': {
            'x_min_m': 0.0,
            'x_max_m': 4.0,
            'y_min_m': -2.0,
            'y_max_m': 2.0,
            'resolution_m': 1.0,
        }
    }
    summary = {'scenario_label': 'test_scene', 'pass': False}
    (audit / 'metadata.json').write_text(json.dumps(metadata))
    (audit / 'summary.json').write_text(json.dumps(summary))
    target = np.asarray([
        [-1, -1, 1, 1],
        [-1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int8)
    baseline = np.asarray([
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)
    candidate = np.asarray([
        [1, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)
    np.savez_compressed(
        snapshots / 'representative.npz',
        target_labels=target,
        baseline_occupied=baseline,
        candidate_occupied=candidate,
        frame_json=np.asarray(json.dumps({'frame_index': 3})),
    )
    return audit, target, baseline, candidate, metadata['bev_geometry']


def test_spatial_diagnostics_distinguish_added_cell_labels(tmp_path):
    _, target, baseline, candidate, geometry = _fixture(tmp_path)
    values = compute_spatial_diagnostics(
        target, baseline, candidate, geometry
    )['whole_bev']

    assert values['added_total'] == 2
    assert values['added_target_obstacle'] == 1
    assert values['added_target_free'] == 0
    assert values['added_unknown'] == 1
    assert values['removed_target_free'] == 1
    assert values['removed_target_obstacle'] == 0


def test_generate_diagnostic_writes_300_dpi_png_and_json(tmp_path):
    audit, *_ = _fixture(tmp_path)
    result = generate_diagnostic(audit, dpi=300)

    image_path = audit / 'diagnostics' / 'representative_spatial_diagnostic.png'
    json_path = audit / 'diagnostics' / 'representative_spatial_diagnostic.json'
    assert image_path.is_file()
    assert json_path.is_file()
    assert result['spatial_diagnostics']['whole_bev']['added_total'] == 2
    with Image.open(image_path) as image:
        assert round(image.info['dpi'][0]) == 300


def test_v2_model_diagnostics_explain_removed_obstacle_cell():
    shape = (2, 2)
    arrays = {
        'target_labels': np.asarray([[1, 1], [0, 0]], dtype=np.int8),
        'baseline_occupied': np.asarray([[1, 1], [0, 0]], dtype=np.uint8),
        'candidate_occupied': np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
        'raw_lidar_bev': np.zeros((4,) + shape, dtype=np.float32),
        'model_obstacle_probability': np.asarray(
            [[0.01, 0.9], [0.0, 0.0]], dtype=np.float32
        ),
        'model_predictive_entropy': np.asarray(
            [[0.02, 0.1], [0.0, 0.0]], dtype=np.float32
        ),
        'model_mc_variance': np.asarray(
            [[0.001, 0.002], [0.0, 0.0]], dtype=np.float32
        ),
        'hard_obstacle_mask': np.asarray(
            [[0, 1], [0, 0]], dtype=np.uint8
        ),
        'selective_decision': np.asarray(
            [[0, 100], [-1, -1]], dtype=np.int8
        ),
    }
    arrays['raw_lidar_bev'][2, 0, 0] = -1.7
    arrays['raw_lidar_bev'][3, 0, 0] = 0.05

    result = compute_v2_model_diagnostics(arrays)
    removed = result['removed_target_obstacle']

    assert removed['cells'] == 1
    assert np.isclose(removed['obstacle_probability']['mean'], 0.01)
    assert removed['hard_obstacle_cells'] == 0
    assert removed['free_decision_cells'] == 1
