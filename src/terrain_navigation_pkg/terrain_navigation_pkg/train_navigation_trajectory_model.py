#!/usr/bin/env python3
"""Train the first LiDAR-BEV local-trajectory imitation baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .navigation_learning_dataset import NavigationTrajectoryDataset
from .navigation_learning_model import (
    BevTrajectoryPolicy,
    masked_trajectory_mse,
)


DEFAULT_DATA_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/visualizations/v1'
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/models/trajectory_baseline/v3'
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    max_batches: int | None = None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    valid_targets = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        lidar = batch['lidar_bev'].to(device, non_blocking=True)
        goal = batch['goal_vehicle_xy_normalized'].to(
            device, non_blocking=True
        )
        target = batch['target_points_normalized'].to(
            device, non_blocking=True
        )
        mask = batch['target_mask'].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(lidar, goal)
            loss = masked_trajectory_mse(prediction, target, mask)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        valid_targets += int(mask.sum().item())
        batches += 1
    if batches == 0:
        raise RuntimeError('no batches were processed')
    return {
        'loss': total_loss / batches,
        'batches': batches,
        'valid_targets': valid_targets,
    }


def train(args) -> dict:
    _seed_everything(args.seed)
    device = torch.device(
        args.device if args.device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    data_directory = args.data_directory.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    train_dataset = NavigationTrajectoryDataset(
        data_directory / 'trajectory_targets_clean_train.npz'
    )
    validation_dataset = NavigationTrajectoryDataset(
        data_directory / 'trajectory_targets_clean_validation.npz'
    )
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )
    model = BevTrajectoryPolicy(
        target_count=12,
        spatial_pool_size=args.spatial_pool_size,
        use_coordinate_channels=args.coordinate_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, device, optimizer,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            validation_metrics = _run_epoch(
                model, validation_loader, device,
                max_batches=args.max_validation_batches,
            )
        record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'validation_loss': validation_metrics['loss'],
            'train_batches': train_metrics['batches'],
            'validation_batches': validation_metrics['batches'],
            'device': str(device),
        }
        history.append(record)
        checkpoint = {
            'model_state': model.state_dict(),
            'model_config': {
                'target_count': 12,
                'spatial_pool_size': args.spatial_pool_size,
                'use_coordinate_channels': args.coordinate_channels,
            },
            'epoch': epoch,
            'validation_loss': validation_metrics['loss'],
            'train_args': vars(args),
        }
        torch.save(checkpoint, output_directory / 'last.pt')
        if validation_metrics['loss'] < best_loss:
            best_loss = validation_metrics['loss']
            torch.save(checkpoint, output_directory / 'best.pt')
        print(json.dumps(record, ensure_ascii=False))
    history_path = output_directory / 'history.csv'
    with history_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = {
        'device': str(device),
        'train_samples': len(train_dataset),
        'validation_samples': len(validation_dataset),
        'best_validation_loss': best_loss,
        'epochs': args.epochs,
        'output_directory': str(output_directory),
    }
    (output_directory / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Train the first LiDAR-BEV trajectory baseline.'
    )
    parser.add_argument('--data-directory', type=Path,
                        default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument('--output-directory', type=Path,
                        default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1.0e-3)
    parser.add_argument('--weight-decay', type=float, default=1.0e-4)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--spatial-pool-size', type=int, default=5)
    parser.add_argument(
        '--coordinate-channels',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                        default='auto')
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-validation-batches', type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(train(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
