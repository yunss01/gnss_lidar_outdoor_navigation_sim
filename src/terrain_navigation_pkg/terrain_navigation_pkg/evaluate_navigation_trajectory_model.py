#!/usr/bin/env python3
"""Evaluate a trained BEV trajectory baseline in metric units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .navigation_learning_dataset import NavigationTrajectoryDataset
from .navigation_learning_model import BevTrajectoryPolicy


DEFAULT_DATA_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/visualizations/v1'
)
DEFAULT_CHECKPOINT = Path(
    '/home/sukja/terrain_nav_data/learning/models/trajectory_baseline/v3/'
    'best.pt'
)


def evaluate(args) -> dict:
    device = torch.device(
        args.device if args.device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    archive = args.data_directory.expanduser().resolve() / (
        'trajectory_targets_clean_{}.npz'.format(args.split)
    )
    dataset = NavigationTrajectoryDataset(archive)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    model = BevTrajectoryPolicy(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    horizon = float(dataset.trajectory_horizon_m)
    squared_sum = 0.0
    point_distance_sum = 0.0
    endpoint_distance_sum = 0.0
    point_distance_values = []
    endpoint_distance_values = []
    prediction_chunks = []
    sample_point_error_chunks = []
    sample_endpoint_error_chunks = []
    valid_count = 0
    sample_count = 0
    endpoint_sample_count = 0
    with torch.no_grad():
        for batch in loader:
            lidar = batch['lidar_bev'].to(device, non_blocking=True)
            goal = batch['goal_vehicle_xy_normalized'].to(
                device, non_blocking=True
            )
            target = batch['target_points_m'].to(
                device, non_blocking=True
            )
            mask = batch['target_mask'].to(device, non_blocking=True)
            prediction = model(lidar, goal) * horizon
            error = prediction - target
            valid = mask.bool()
            squared_sum += float(error.square()[valid].sum().cpu())
            distances = torch.linalg.vector_norm(error, dim=-1)
            point_distance_sum += float(distances[valid].sum().cpu())
            point_distance_values.extend(distances[valid].cpu().tolist())
            valid_count += int(valid.sum().item())

            valid_samples = mask.any(dim=1)
            valid_rows = torch.nonzero(
                valid_samples, as_tuple=False
            ).flatten()
            last_index = (
                mask[valid_samples].sum(dim=1).to(torch.long) - 1
            )
            endpoint = distances[valid_rows, last_index]
            endpoint_distance_sum += float(endpoint.sum().cpu())
            endpoint_distance_values.extend(endpoint.cpu().tolist())
            prediction_chunks.append(prediction.cpu().numpy())
            per_sample_distance = (
                (distances * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1)
            )
            sample_point_error_chunks.append(
                per_sample_distance.cpu().numpy()
            )
            sample_endpoint_error_chunks.append(endpoint.cpu().numpy())
            sample_count += mask.shape[0]
            endpoint_sample_count += int(valid_samples.sum().item())

    if valid_count == 0 or sample_count == 0 or endpoint_sample_count == 0:
        raise RuntimeError('validation archive contains no usable samples')
    point_values = torch.tensor(point_distance_values)
    endpoint_values = torch.tensor(endpoint_distance_values)
    metrics = {
        'split': args.split,
        'checkpoint': str(args.checkpoint.expanduser().resolve()),
        'device': str(device),
        'samples': sample_count,
        'valid_target_points': valid_count,
        'valid_target_fraction': valid_count / float(
            sample_count * dataset.target_points.shape[1]
        ),
        'coordinate_rmse_m': (squared_sum / (valid_count * 2.0)) ** 0.5,
        'point_distance_mean_m': point_distance_sum / valid_count,
        'point_distance_median_m': float(point_values.median()),
        'samples_with_valid_targets': endpoint_sample_count,
        'endpoint_distance_mean_m': (
            endpoint_distance_sum / endpoint_sample_count
        ),
        'endpoint_distance_median_m': float(endpoint_values.median()),
    }
    output = args.output.expanduser().resolve() if args.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    predictions_output = (
        args.predictions_output.expanduser().resolve()
        if args.predictions_output else None
    )
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_output,
            sample_path=dataset.sample_paths,
            session=dataset.sessions,
            sample_id=dataset.sample_ids,
            goal_vehicle_xy=dataset.goal_vehicle_xy,
            teacher_points_m=dataset.target_points,
            target_mask=dataset.target_mask,
            predicted_points_m=np.concatenate(
                prediction_chunks, axis=0
            ).astype(np.float32),
            mean_point_error_m=np.concatenate(
                sample_point_error_chunks, axis=0
            ).astype(np.float32),
            endpoint_error_m=np.concatenate(
                sample_endpoint_error_chunks, axis=0
            ).astype(np.float32),
            checkpoint=np.asarray(metrics['checkpoint']),
            split=np.asarray(args.split),
        )
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate a trained BEV trajectory policy.'
    )
    parser.add_argument('--data-directory', type=Path,
                        default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument('--checkpoint', type=Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument('--split', choices=('train', 'validation'),
                        default='validation')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                        default='auto')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--predictions-output', type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
