#!/usr/bin/env python3
"""Audit a traversability checkpoint under synthetic LiDAR domain shifts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .navigation_learning_recorder_core import BevGeometry
from .traversability_dataset import (
    normalize_lidar_bev,
    read_written_manifest_rows,
)
from .traversability_metrics import TraversabilityMetricAccumulator
from .traversability_model import BevTraversabilityUNet
from .traversability_stress_core import STRESS_CASES, apply_bev_stress


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def evaluate_stress(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = BevTraversabilityUNet(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    sessions = args.session or checkpoint[
        'training_config'
    ]['validation_sessions']
    rows = [
        row for row in read_written_manifest_rows(args.manifest)
        if row['source_session'] in set(sessions)
    ]
    if not rows:
        raise ValueError('no samples found for requested sessions')
    geometry = BevGeometry()
    metrics = {
        case: TraversabilityMetricAccumulator() for case in STRESS_CASES
    }

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            raw_bevs = []
            raw_targets = []
            for row in chunk:
                with np.load(
                    row['derived_sample_path'], allow_pickle=False
                ) as arrays:
                    raw_bevs.append(np.asarray(
                        arrays['lidar_bev'], dtype=np.float32
                    ))
                    raw_targets.append(np.asarray(
                        arrays['target_labels'], dtype=np.int64
                    ))
            for case_index, case in enumerate(STRESS_CASES):
                inputs = []
                targets = []
                for item_index, (bev, target) in enumerate(zip(
                    raw_bevs, raw_targets
                )):
                    sample_id = int(chunk[item_index]['source_sample_id'])
                    perturbed, valid_target = apply_bev_stress(
                        bev,
                        target,
                        geometry,
                        case,
                        args.seed + case_index * 1000003 + sample_id,
                    )
                    inputs.append(normalize_lidar_bev(perturbed))
                    targets.append(valid_target)
                input_tensor = torch.from_numpy(np.stack(inputs)).to(device)
                target_tensor = torch.from_numpy(np.stack(targets)).to(device)
                logits = model(input_tensor)
                metrics[case].update(logits, target_tensor)

    clean = metrics['clean'].compute()
    results = []
    for case in STRESS_CASES:
        values = metrics[case].compute()
        results.append({
            'case': case,
            'known_cells': values['known_cells'],
            'accuracy': values['accuracy'],
            'free_iou': values['free_iou'],
            'obstacle_iou': values['obstacle_iou'],
            'obstacle_precision': values['obstacle_precision'],
            'obstacle_recall': values['obstacle_recall'],
            'false_free': values['false_free'],
            'false_obstacle': values['false_obstacle'],
            'obstacle_iou_change': (
                values['obstacle_iou'] - clean['obstacle_iou']
            ),
            'obstacle_recall_change': (
                values['obstacle_recall'] - clean['obstacle_recall']
            ),
        })
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'checkpoint': str(checkpoint_path),
        'manifest': str(Path(args.manifest).expanduser().resolve()),
        'sessions': sessions,
        'samples': len(rows),
        'device': str(device),
        'seed': args.seed,
        'note': (
            'Synthetic stress tests diagnose sensitivity; they are not a '
            'substitute for a held-out map or real-vehicle evaluation.'
        ),
        'results': results,
    }
    _atomic_json(output, report)
    csv_path = output.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--session', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                        default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260915)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    print(json.dumps(evaluate_stress(args), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
