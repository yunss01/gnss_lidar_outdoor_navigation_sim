#!/usr/bin/env python3
"""Evaluate a traversability checkpoint on explicit held-out sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .traversability_dataset import (
    TraversabilityDataset,
    read_written_manifest_rows,
)
from .traversability_metrics import (
    SelectivePredictionAccumulator,
    TraversabilityMetricAccumulator,
)
from .traversability_inference_core import (
    ShadowPolicyAccumulator,
    binary_predictive_entropy,
    build_selective_decision,
)
from .traversability_model import BevTraversabilityUNet


def _enable_dropout(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout2d):
            module.train()


def evaluate(args) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    checkpoint = torch.load(
        Path(args.checkpoint).expanduser(), map_location=device,
        weights_only=False,
    )
    model = BevTraversabilityUNet(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sessions = list(args.session)
    if not sessions:
        sessions = checkpoint['training_config']['validation_sessions']
    rows = [
        row for row in read_written_manifest_rows(args.manifest)
        if row['source_session'] in set(sessions)
    ]
    found = {row['source_session'] for row in rows}
    missing = set(sessions).difference(found)
    if missing:
        raise ValueError(
            'sessions absent from manifest: ' + ', '.join(missing)
        )
    dataset = TraversabilityDataset(rows, return_raw_bev=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
    )
    metrics = TraversabilityMetricAccumulator()
    selective = SelectivePredictionAccumulator()
    shadow_policy = ShadowPolicyAccumulator()
    mc_variance_sum = 0.0
    mc_variance_cells = 0
    with torch.no_grad():
        for batch in loader:
            lidar_bev = batch['lidar_bev'].to(device)
            targets = batch['target_labels'].to(device)
            if args.mc_samples > 1:
                _enable_dropout(model)
                probabilities = torch.stack([
                    torch.softmax(model(lidar_bev), dim=1)
                    for _ in range(args.mc_samples)
                ])
                mean_probability = probabilities.mean(dim=0)
                obstacle_variance = probabilities[:, :, 1].var(
                    dim=0, unbiased=False
                )
                known = targets >= 0
                mc_variance_sum += float(obstacle_variance[known].sum())
                mc_variance_cells += int(torch.count_nonzero(known))
                logits = mean_probability.clamp_min(1e-8).log()
                model.eval()
            else:
                logits = model(lidar_bev)
                mean_probability = torch.softmax(logits, dim=1)
                obstacle_variance = torch.zeros_like(
                    mean_probability[:, 1]
                )
            metrics.update(logits, targets)
            selective.update(logits, targets)
            raw_bevs = batch['raw_lidar_bev'].numpy()
            ego_masks = batch['ego_exclusion_mask'].numpy()
            probability_values = mean_probability[:, 1].cpu().numpy()
            variance_values = obstacle_variance.cpu().numpy()
            target_values = targets.cpu().numpy()
            for index in range(raw_bevs.shape[0]):
                entropy = binary_predictive_entropy(
                    probability_values[index]
                )
                raw_bev = raw_bevs[index]
                decision = build_selective_decision(
                    probability_values[index],
                    entropy,
                    raw_bev[0] > 0.5,
                    raw_bev[2],
                    raw_bev[3],
                    free_probability_threshold=(
                        args.shadow_free_probability
                    ),
                    obstacle_probability_threshold=(
                        args.shadow_obstacle_probability
                    ),
                    maximum_entropy=args.shadow_maximum_entropy,
                    maximum_mc_variance=(
                        args.shadow_maximum_mc_variance
                    ),
                    mc_variance=variance_values[index],
                    hard_obstacle_minimum_z_m=(
                        args.shadow_hard_minimum_z_m
                    ),
                    hard_obstacle_vertical_span_m=(
                        args.shadow_hard_vertical_span_m
                    ),
                    exclusion_mask=ego_masks[index],
                )
                shadow_policy.update(
                    decision,
                    target_values[index],
                    raw_bev[0] > 0.5,
                    ego_masks[index],
                )
    result = {
        'checkpoint': str(Path(args.checkpoint).expanduser().resolve()),
        'manifest': str(Path(args.manifest).expanduser().resolve()),
        'sessions': sessions,
        'samples': len(dataset),
        'device': str(device),
        'mc_samples': args.mc_samples,
        'seed': args.seed,
        'metrics': metrics.compute(),
        'selective_prediction': selective.compute(),
        'shadow_policy_parameters': {
            'free_probability_threshold': args.shadow_free_probability,
            'obstacle_probability_threshold': (
                args.shadow_obstacle_probability
            ),
            'maximum_entropy': args.shadow_maximum_entropy,
            'maximum_mc_variance': args.shadow_maximum_mc_variance,
            'hard_obstacle_minimum_z_m': args.shadow_hard_minimum_z_m,
            'hard_obstacle_vertical_span_m': (
                args.shadow_hard_vertical_span_m
            ),
        },
        'shadow_policy': shadow_policy.compute(),
    }
    if mc_variance_cells:
        result['mean_mc_obstacle_probability_variance'] = (
            mc_variance_sum / mc_variance_cells
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--session', action='append', default=[])
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda']
    )
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--mc-samples', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--shadow-free-probability', type=float, default=0.99)
    parser.add_argument(
        '--shadow-obstacle-probability', type=float, default=0.50
    )
    parser.add_argument('--shadow-maximum-entropy', type=float, default=0.15)
    parser.add_argument(
        '--shadow-maximum-mc-variance', type=float, default=0.02
    )
    parser.add_argument(
        '--shadow-hard-minimum-z-m', type=float, default=-1.40
    )
    parser.add_argument(
        '--shadow-hard-vertical-span-m', type=float, default=0.15
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.mc_samples < 1:
        raise ValueError('mc-samples must be positive')
    result = evaluate(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(rendered, encoding='utf-8')
        temporary.replace(output)
    print(rendered, end='')


if __name__ == '__main__':
    main()
