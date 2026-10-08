#!/usr/bin/env python3
"""Train the first session-isolated LiDAR-BEV traversability pilot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .traversability_dataset import (
    TraversabilityDataset,
    label_counts,
    read_written_manifest_rows,
    split_rows_by_session,
)
from .traversability_metrics import TraversabilityMetricAccumulator
from .traversability_model import BevTraversabilityUNet, masked_cross_entropy


DEFAULT_MANIFEST = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/v1/manifest.csv'
)
DEFAULT_OUTPUT = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_pilot/v1'
)
DEFAULT_VALIDATION_SESSION = 'session_20260915_131243_896751'


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _class_weights(rows) -> tuple[torch.Tensor, dict]:
    free, obstacle = label_counts(rows)
    if free <= 0 or obstacle <= 0:
        raise ValueError('training split must contain both target classes')
    counts = np.asarray([free, obstacle], dtype=np.float64)
    weights = np.sqrt(counts.sum() / (2.0 * counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32), {
        'free_cells': free,
        'obstacle_cells': obstacle,
        'free_weight': float(weights[0]),
        'obstacle_weight': float(weights[1]),
    }


def _limited_batches(loader, maximum_batches):
    for index, batch in enumerate(loader):
        if maximum_batches is not None and index >= maximum_batches:
            break
        yield batch


def _run_epoch(
    model,
    loader,
    device,
    class_weights,
    *,
    optimizer=None,
    maximum_batches=None,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    metric = TraversabilityMetricAccumulator()
    total_loss = 0.0
    batch_count = 0
    amp_enabled = device.type == 'cuda'

    for batch in _limited_batches(loader, maximum_batches):
        lidar_bev = batch['lidar_bev'].to(
            device, non_blocking=amp_enabled
        )
        targets = batch['target_labels'].to(
            device, non_blocking=amp_enabled
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(lidar_bev)
                loss = masked_cross_entropy(
                    logits, targets, class_weights=class_weights
                )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.detach())
        batch_count += 1
        metric.update(logits, targets)
    if batch_count == 0:
        raise ValueError('data loader produced no batches')
    return total_loss / batch_count, metric.compute()


def _write_split_manifest(path: Path, splits: dict) -> None:
    fields = ['split', 'source_session', 'source_sample_id',
              'derived_sample_path']
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split_name in ('train', 'validation', 'test'):
            for row in splits[split_name]:
                writer.writerow({
                    'split': split_name,
                    'source_session': row['source_session'],
                    'source_sample_id': row['source_sample_id'],
                    'derived_sample_path': row['derived_sample_path'],
                })
    temporary.replace(path)


def train(args) -> dict:
    _set_seed(args.seed)
    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    rows = read_written_manifest_rows(args.manifest)
    splits = split_rows_by_session(
        rows,
        validation_sessions=args.validation_session,
        test_sessions=args.test_session,
    )
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_split_manifest(output / 'split_manifest.csv', splits)

    train_dataset = TraversabilityDataset(
        splits['train'],
        horizontal_flip_probability=args.flip_probability,
        sensor_augmentation_probability=(
            args.sensor_augmentation_probability
        ),
        sensor_augmentation_parameters={
            'maximum_height_bias_m': args.maximum_height_bias_m,
            'maximum_tilt_deg': args.maximum_tilt_deg,
            'height_noise_standard_deviation_m': args.height_noise_std_m,
            'minimum_density_scale': args.minimum_density_scale,
            'maximum_cell_dropout_fraction': args.maximum_cell_dropout,
        },
    )
    validation_dataset = TraversabilityDataset(splits['validation'])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
    )

    class_weights, class_summary = _class_weights(splits['train'])
    class_weights = class_weights.to(device)
    model_config = {
        'input_channels': 4,
        'base_channels': args.base_channels,
        'dropout_probability': args.dropout_probability,
        'use_coordinate_channels': True,
    }
    model = BevTraversabilityUNet(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    config = {
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'purpose': 'Town10 feasibility pilot; not a final generalization test',
        'manifest': str(Path(args.manifest).expanduser().resolve()),
        'output_directory': str(output),
        'device': str(device),
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'flip_probability': args.flip_probability,
        'sensor_augmentation': {
            'probability': args.sensor_augmentation_probability,
            'maximum_height_bias_m': args.maximum_height_bias_m,
            'maximum_tilt_deg': args.maximum_tilt_deg,
            'height_noise_standard_deviation_m': args.height_noise_std_m,
            'minimum_density_scale': args.minimum_density_scale,
            'maximum_cell_dropout_fraction': args.maximum_cell_dropout,
        },
        'validation_sessions': list(args.validation_session),
        'test_sessions': list(args.test_session),
        'split_sample_counts': {
            name: len(values) for name, values in splits.items()
        },
        'training_class_balance': class_summary,
        'model': model_config,
        'input_policy': 'geometric lidar_bev only',
        'target_policy': (
            'CARLA semantic LiDAR free/obstacle labels; unknown=-1 ignored'
        ),
        'maximum_train_batches': args.max_train_batches,
        'maximum_validation_batches': args.max_validation_batches,
    }
    _atomic_json(output / 'training_config.json', config)

    history = []
    best_loss = math.inf
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            optimizer=optimizer,
            maximum_batches=args.max_train_batches,
        )
        with torch.no_grad():
            validation_loss, validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                class_weights,
                maximum_batches=args.max_validation_batches,
            )
        scheduler.step(validation_loss)
        record = {
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'validation_loss': validation_loss,
        }
        record.update({
            'train_' + key: value for key, value in train_metrics.items()
        })
        record.update({
            'validation_' + key: value
            for key, value in validation_metrics.items()
        })
        history.append(record)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model_config,
            'training_config': config,
            'class_weights': class_weights.detach().cpu(),
            'validation_metrics': validation_metrics,
            'validation_loss': validation_loss,
        }
        _atomic_checkpoint(output / 'last.pt', checkpoint)
        improved = validation_loss < best_loss - args.minimum_improvement
        if improved:
            best_loss = validation_loss
            stale_epochs = 0
            _atomic_checkpoint(output / 'best.pt', checkpoint)
        else:
            stale_epochs += 1

        print(
            'epoch={:02d} train_loss={:.5f} val_loss={:.5f} '
            'val_obstacle_iou={:.4f} val_recall={:.4f} lr={:.2e}'.format(
                epoch,
                train_loss,
                validation_loss,
                validation_metrics['obstacle_iou'],
                validation_metrics['obstacle_recall'],
                optimizer.param_groups[0]['lr'],
            ),
            flush=True,
        )
        if stale_epochs >= args.early_stopping_patience:
            print('early stopping after epoch {}'.format(epoch), flush=True)
            break

    history_path = output / 'history.csv'
    with history_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {
        'device': str(device),
        'epochs_completed': len(history),
        'best_validation_loss': best_loss,
        'best_checkpoint': str(output / 'best.pt'),
        'history': str(history_path),
        'split_sample_counts': config['split_sample_counts'],
    }
    _atomic_json(output / 'result.json', result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--output-directory', type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        '--validation-session', action='append', default=None,
        help='whole session reserved for validation; may be repeated',
    )
    parser.add_argument(
        '--test-session', action='append', default=[],
        help='optional untouched test session; may be repeated',
    )
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda']
    )
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--dropout-probability', type=float, default=0.10)
    parser.add_argument('--flip-probability', type=float, default=0.5)
    parser.add_argument(
        '--sensor-augmentation-probability', type=float, default=0.0
    )
    parser.add_argument('--maximum-height-bias-m', type=float, default=0.20)
    parser.add_argument('--maximum-tilt-deg', type=float, default=4.0)
    parser.add_argument('--height-noise-std-m', type=float, default=0.03)
    parser.add_argument('--minimum-density-scale', type=float, default=0.50)
    parser.add_argument('--maximum-cell-dropout', type=float, default=0.20)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=6)
    parser.add_argument('--minimum-improvement', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-validation-batches', type=int)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.validation_session is None:
        args.validation_session = [DEFAULT_VALIDATION_SESSION]
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError('epochs/batch-size/workers arguments are invalid')
    if not 0.0 <= args.sensor_augmentation_probability <= 1.0:
        raise ValueError('sensor augmentation probability must be in [0, 1]')
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
