#!/usr/bin/env python3
"""Compare archived offline evidence with the online preprocessing path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .navigation_learning_recorder_core import BevGeometry
from .traversability_evidence_dataset import normalize_lidar_evidence_bev
from .traversability_evidence_inference_core import (
    build_current_evidence_input,
    load_current_only_evidence_checkpoint,
    predict_evidence,
)
from .traversability_learning_core import EgoFootprint


def validate_parity(
    source_sample,
    derived_sample,
    checkpoint,
    *,
    device='cpu',
    maximum_evidence_error=1.0e-3,
    maximum_probability_error=1.0e-3,
):
    source_path = Path(source_sample).expanduser().resolve()
    derived_path = Path(derived_sample).expanduser().resolve()
    torch_device = torch.device(device)
    with np.load(source_path, allow_pickle=False) as source:
        points = np.asarray(source['lidar_points_xyz'], dtype=np.float32)
    with np.load(derived_path, allow_pickle=False) as derived:
        archived_lidar = np.asarray(derived['lidar_bev'], dtype=np.float32)
        archived_evidence = np.asarray(
            derived['current_lidar_evidence_bev'], dtype=np.float32
        )

    geometry = BevGeometry(
        x_min_m=-10.0,
        x_max_m=30.0,
        y_min_m=-20.0,
        y_max_m=20.0,
        resolution_m=0.25,
        z_min_m=-2.0,
        z_max_m=3.0,
    )
    footprint = EgoFootprint(
        rear_m=2.5,
        front_m=2.4,
        half_width_m=1.0,
    )
    online = build_current_evidence_input(points, geometry, footprint)
    model, _, checkpoint_path = load_current_only_evidence_checkpoint(
        checkpoint, torch_device
    )
    online_prediction = predict_evidence(
        model, online.normalized_evidence_bev, torch_device
    )
    archived_prediction = predict_evidence(
        model,
        normalize_lidar_evidence_bev(archived_evidence),
        torch_device,
    )

    lidar_error = float(np.max(np.abs(
        online.lidar_bev - archived_lidar
    )))
    evidence_error = float(np.max(np.abs(
        online.evidence_bev - archived_evidence
    )))
    passable_error = float(np.max(np.abs(
        online_prediction.passable_probability
        - archived_prediction.passable_probability
    )))
    obstacle_error = float(np.max(np.abs(
        online_prediction.obstacle_probability
        - archived_prediction.obstacle_probability
    )))
    passed = (
        lidar_error <= float(maximum_evidence_error)
        and evidence_error <= float(maximum_evidence_error)
        and passable_error <= float(maximum_probability_error)
        and obstacle_error <= float(maximum_probability_error)
    )
    return {
        'schema_version': 1,
        'passed': bool(passed),
        'source_sample': str(source_path),
        'derived_sample': str(derived_path),
        'checkpoint': str(checkpoint_path),
        'device': str(torch_device),
        'maximum_errors': {
            'lidar_bev': lidar_error,
            'current_evidence_bev': evidence_error,
            'passable_probability': passable_error,
            'obstacle_probability': obstacle_error,
        },
        'tolerances': {
            'lidar_bev': float(maximum_evidence_error),
            'current_evidence_bev': float(maximum_evidence_error),
            'probability': float(maximum_probability_error),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sample', required=True)
    parser.add_argument('--derived-sample', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--maximum-evidence-error', type=float, default=1e-3)
    parser.add_argument(
        '--maximum-probability-error', type=float, default=1e-3
    )
    args = parser.parse_args()
    result = validate_parity(
        args.source_sample,
        args.derived_sample,
        args.checkpoint,
        device=args.device,
        maximum_evidence_error=args.maximum_evidence_error,
        maximum_probability_error=args.maximum_probability_error,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
