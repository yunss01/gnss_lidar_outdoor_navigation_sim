import numpy as np
import pytest
import torch

from terrain_navigation_pkg.navigation_learning_recorder_core import (
    BevGeometry,
    build_lidar_bev,
)
from terrain_navigation_pkg.traversability_evidence_core import (
    build_evidence_bev,
)
from terrain_navigation_pkg.traversability_evidence_dataset import (
    normalize_lidar_evidence_bev,
)
from terrain_navigation_pkg.traversability_evidence_inference_core import (
    build_current_evidence_input,
    load_current_only_evidence_checkpoint,
    predict_evidence,
)
from terrain_navigation_pkg.traversability_evidence_model import (
    build_traversability_evidence_model,
)
from terrain_navigation_pkg.traversability_learning_core import EgoFootprint


def _geometry():
    return BevGeometry(
        x_min_m=0.0,
        x_max_m=8.0,
        y_min_m=-4.0,
        y_max_m=4.0,
        resolution_m=0.25,
        z_min_m=-2.0,
        z_max_m=3.0,
    )


def _footprint():
    return EgoFootprint(rear_m=0.2, front_m=0.2, half_width_m=0.2)


def _model_config():
    return {
        'architecture': 'decoupled_unet_context',
        'input_channels': 8,
        'base_channels': 4,
        'dropout_probability': 0.0,
        'use_coordinate_channels': True,
    }


def test_online_preprocessing_matches_training_pipeline():
    geometry = _geometry()
    points = []
    for forward in np.arange(1.0, 7.0, 0.25):
        for left in np.arange(-2.0, 2.1, 0.5):
            points.append([forward, left, -1.6])
    points.append([4.0, 0.0, -1.2])
    points = np.asarray(points, dtype=np.float32)

    online = build_current_evidence_input(
        points, geometry, _footprint()
    )
    expected_lidar = (
        build_lidar_bev(points, geometry)
        .astype(np.float16).astype(np.float32)
    )
    expected_evidence, expected_local = build_evidence_bev(
        expected_lidar, geometry
    )
    expected_evidence = (
        expected_evidence.astype(np.float16).astype(np.float32)
    )

    assert np.array_equal(online.lidar_bev, expected_lidar)
    assert np.array_equal(online.evidence_bev, expected_evidence)
    assert np.array_equal(
        online.normalized_evidence_bev,
        normalize_lidar_evidence_bev(expected_evidence),
    )
    assert np.array_equal(
        online.observed_mask,
        (expected_lidar[0] > 0.5) & ~online.ego_exclusion_mask,
    )
    assert np.array_equal(
        online.local_ground.relative_max_height,
        expected_local.relative_max_height
        .astype(np.float16).astype(np.float32),
    )
    assert online.hard_obstacle_mask.shape == (32, 32)


def test_online_prediction_matches_direct_model_forward():
    torch.manual_seed(7)
    model = build_traversability_evidence_model(_model_config())
    evidence = np.random.default_rng(7).normal(
        size=(8, 32, 32)
    ).astype(np.float32)
    prediction = predict_evidence(
        model, evidence, torch.device('cpu')
    )
    model.eval()
    with torch.no_grad():
        direct = model(torch.from_numpy(evidence).unsqueeze(0))
    assert np.allclose(
        prediction.passable_probability,
        torch.sigmoid(direct['passable_logits'][0]).numpy(),
    )
    assert np.allclose(
        prediction.obstacle_probability,
        torch.sigmoid(direct['obstacle_logits'][0]).numpy(),
    )
    assert np.count_nonzero(prediction.passable_variance) == 0
    assert np.count_nonzero(prediction.obstacle_variance) == 0


def test_checkpoint_loader_accepts_current_only_decoupled_model(tmp_path):
    model = build_traversability_evidence_model(_model_config())
    path = tmp_path / 'best.pt'
    torch.save({
        'model_config': _model_config(),
        'model_state_dict': model.state_dict(),
        'training_config': {
            'target_contract': 'independent_passable_obstacle_evidence',
            'evidence_input_variant': 'current_only',
        },
    }, path)
    restored, checkpoint, restored_path = (
        load_current_only_evidence_checkpoint(path, torch.device('cpu'))
    )
    assert restored_path == path.resolve()
    assert checkpoint['model_config'] == _model_config()
    assert restored.training is False


def test_checkpoint_loader_rejects_temporal_input(tmp_path):
    config = dict(_model_config(), input_channels=17)
    model = build_traversability_evidence_model(config)
    path = tmp_path / 'temporal.pt'
    torch.save({
        'model_config': config,
        'model_state_dict': model.state_dict(),
        'training_config': {
            'target_contract': 'independent_passable_obstacle_evidence',
            'evidence_input_variant': 'temporal',
        },
    }, path)
    with pytest.raises(ValueError, match='current_only'):
        load_current_only_evidence_checkpoint(path, torch.device('cpu'))
