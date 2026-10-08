#!/usr/bin/env python3
"""Evaluate a v2 evidence checkpoint on a frozen scene split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .traversability_evidence_dataset import (
    TraversabilityEvidenceDataset,
    evidence_pair_key,
    evidence_row_key,
    read_evidence_split_manifest,
)
from .traversability_evidence_metrics import (
    TraversabilityEvidenceMetricAccumulator,
)
from .traversability_evidence_model import (
    build_traversability_evidence_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _device(args):
    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def _load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get('training_config', {}).get('target_contract') != (
        'independent_passable_obstacle_evidence'
    ):
        raise ValueError('checkpoint is not an independent v2 evidence model')
    model = build_traversability_evidence_model(
        checkpoint['model_config']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def _predict_rows(
    model, rows, device, workers: int, input_variant: str = 'auto'
) -> dict[tuple[str, int], dict]:
    dataset = TraversabilityEvidenceDataset(
        rows, return_metadata=True, input_variant=input_variant
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == 'cuda',
    )
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            evidence = batch['lidar_evidence_bev'].to(device)
            outputs = model(evidence)
            row_key = (
                batch['source_session'][0],
                int(batch['source_sample_id'][0]),
            )
            predictions[row_key] = {
                'passable_probability': torch.sigmoid(
                    outputs['passable_logits'][0]
                ).cpu(),
                'obstacle_probability': torch.sigmoid(
                    outputs['obstacle_logits'][0]
                ).cpu(),
                'passable_target': batch['passable_target'][0],
                'obstacle_target': batch['obstacle_target'][0],
                'known_evidence_mask': batch['known_evidence_mask'][0],
                'controlled_obstacle_mask': (
                    batch['controlled_obstacle_mask'][0]
                ),
                'observed_mask': batch['observed_mask'][0],
                'passable_support': batch['passable_support'][0],
                'vehicle_clearance_hard_obstacle_mask': (
                    batch['vehicle_clearance_hard_obstacle_mask'][0]
                ),
            }
    return predictions


def _family_report(rows, predictions, control_by_pair, obstacle_threshold):
    totals = defaultdict(lambda: {
        'instances': 0,
        'controlled_cells': 0,
        'detected_cells': 0,
        'any_detected_instances': 0,
        'positive_probability_sum': 0.0,
        'control_probability_sum': 0.0,
        'paired_delta_sum': 0.0,
    })
    for row in rows:
        if row['role'] != 'controlled':
            continue
        prediction = predictions[evidence_row_key(row)]
        mask = prediction['controlled_obstacle_mask'].bool()
        cells = int(torch.count_nonzero(mask))
        if cells <= 0:
            continue
        probability = prediction['obstacle_probability'][mask]
        detected = probability >= obstacle_threshold
        control_row = control_by_pair[evidence_pair_key(row)]
        control_probability = predictions[evidence_row_key(control_row)][
            'obstacle_probability'
        ][mask]
        family = row['object_family']
        value = totals[family]
        value['instances'] += 1
        value['controlled_cells'] += cells
        value['detected_cells'] += int(torch.count_nonzero(detected))
        value['any_detected_instances'] += int(torch.any(detected))
        value['positive_probability_sum'] += float(probability.sum())
        value['control_probability_sum'] += float(control_probability.sum())
        value['paired_delta_sum'] += float(
            (probability - control_probability).sum()
        )
    result = {}
    for family, value in sorted(totals.items()):
        cells = value['controlled_cells']
        instances = value['instances']
        result[family] = {
            'instances': instances,
            'controlled_cells': cells,
            'controlled_cell_recall': _ratio(
                value['detected_cells'], cells
            ),
            'instance_any_detection_rate': _ratio(
                value['any_detected_instances'], instances
            ),
            'mean_positive_obstacle_probability': _ratio(
                value['positive_probability_sum'], cells
            ),
            'mean_paired_control_obstacle_probability': _ratio(
                value['control_probability_sum'], cells
            ),
            'mean_paired_obstacle_probability_delta': _ratio(
                value['paired_delta_sum'], cells
            ),
        }
    return result


def _vehicle_clearance_fusion_report(
    rows,
    predictions,
    control_by_pair,
    obstacle_threshold,
    passable_threshold,
    maximum_obstacle_probability_for_passable,
    minimum_passable_support,
):
    counts = {
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0,
        'controlled_cells': 0,
        'controlled_detected_cells': 0,
        'controlled_instances': 0,
        'controlled_instances_detected': 0,
        'paired_control_cells': 0,
        'paired_control_false_obstacle_cells': 0,
        'controlled_unsafe_passable_cells': 0,
        'controlled_instances_unsafe_passable': 0,
    }
    for row in rows:
        prediction = predictions[evidence_row_key(row)]
        obstacle = prediction['obstacle_target'].bool()
        known = prediction['known_evidence_mask'].bool()
        learned = (
            prediction['obstacle_probability'] >= obstacle_threshold
        )
        fused = learned | prediction[
            'vehicle_clearance_hard_obstacle_mask'
        ].bool()
        counts['true_positive'] += int(torch.count_nonzero(
            known & fused & obstacle
        ))
        counts['false_positive'] += int(torch.count_nonzero(
            known & fused & ~obstacle
        ))
        counts['true_negative'] += int(torch.count_nonzero(
            known & ~fused & ~obstacle
        ))
        counts['false_negative'] += int(torch.count_nonzero(
            known & ~fused & obstacle
        ))
        controlled = prediction['controlled_obstacle_mask'].bool()
        cells = int(torch.count_nonzero(controlled))
        if cells <= 0:
            continue
        detected = int(torch.count_nonzero(controlled & fused))
        counts['controlled_cells'] += cells
        counts['controlled_detected_cells'] += detected
        counts['controlled_instances'] += 1
        counts['controlled_instances_detected'] += int(detected > 0)
        passable = (
            prediction['observed_mask'].bool()
            & ~fused
            & (
                prediction['passable_probability'] >= passable_threshold
            )
            & (
                prediction['obstacle_probability']
                <= maximum_obstacle_probability_for_passable
            )
            & (
                prediction['passable_support'] >= minimum_passable_support
            )
        )
        unsafe_passable = controlled & passable
        unsafe_cells = int(torch.count_nonzero(unsafe_passable))
        counts['controlled_unsafe_passable_cells'] += unsafe_cells
        counts['controlled_instances_unsafe_passable'] += int(
            detected == 0 and unsafe_cells > 0
        )
        control = predictions[evidence_row_key(
            control_by_pair[evidence_pair_key(row)]
        )]
        control_fused = (
            control['obstacle_probability'] >= obstacle_threshold
        ) | control['vehicle_clearance_hard_obstacle_mask'].bool()
        counts['paired_control_cells'] += cells
        counts['paired_control_false_obstacle_cells'] += int(
            torch.count_nonzero(controlled & control_fused)
        )
    tp = counts['true_positive']
    fp = counts['false_positive']
    fn = counts['false_negative']
    counts.update({
        'obstacle_iou': _ratio(tp, tp + fp + fn),
        'obstacle_precision': _ratio(tp, tp + fp),
        'obstacle_recall': _ratio(tp, tp + fn),
        'controlled_cell_recall': _ratio(
            counts['controlled_detected_cells'],
            counts['controlled_cells'],
        ),
        'controlled_instance_detection_rate': _ratio(
            counts['controlled_instances_detected'],
            counts['controlled_instances'],
        ),
        'paired_control_false_obstacle_rate': _ratio(
            counts['paired_control_false_obstacle_cells'],
            counts['paired_control_cells'],
        ),
        'controlled_unsafe_passable_cell_rate': _ratio(
            counts['controlled_unsafe_passable_cells'],
            counts['controlled_cells'],
        ),
        'controlled_unsafe_passable_instance_rate': _ratio(
            counts['controlled_instances_unsafe_passable'],
            counts['controlled_instances'],
        ),
    })
    return counts


def evaluate(args) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    model, checkpoint = _load_model(checkpoint_path, device)
    input_variant = checkpoint.get('training_config', {}).get(
        'evidence_input_variant', 'auto'
    )
    all_rows = read_evidence_split_manifest(args.split_manifest)
    split_rows = [row for row in all_rows if row['split'] == args.split]
    if not split_rows:
        raise ValueError('requested split contains no rows: ' + args.split)
    consumed_test = (
        args.split == 'test'
        and any(
            row.get('evaluation_slice') == 'consumed_test_do_not_use'
            for row in split_rows
        )
    )
    if consumed_test and not getattr(args, 'allow_consumed_test', False):
        raise ValueError(
            'test split is marked consumed_test_do_not_use; refusing to '
            'evaluate it without --allow-consumed-test'
        )
    control_by_pair = {}
    for row in split_rows:
        if row['role'] != 'control':
            continue
        pair_key = evidence_pair_key(row)
        if pair_key in control_by_pair:
            raise ValueError(
                'split has multiple controls for scene/pair_group: '
                + '/'.join(pair_key)
            )
        control_by_pair[pair_key] = row
    missing_controls = sorted({
        evidence_pair_key(row) for row in split_rows
        if evidence_pair_key(row) not in control_by_pair
    })
    if missing_controls:
        raise ValueError(
            'split lacks paired controls for scene/pair_group: '
            + ', '.join('/'.join(key) for key in missing_controls)
        )
    selected_rows = split_rows
    if args.evaluation_slice:
        selected_rows = [
            row for row in split_rows
            if row['evaluation_slice'] == args.evaluation_slice
        ]
        if not selected_rows:
            raise ValueError(
                'evaluation slice contains no rows: ' + args.evaluation_slice
            )
    prediction_rows = list(selected_rows)
    present = {evidence_row_key(row) for row in prediction_rows}
    for row in selected_rows:
        control = control_by_pair[evidence_pair_key(row)]
        control_key = evidence_row_key(control)
        if control_key not in present:
            prediction_rows.append(control)
            present.add(control_key)
    predictions = _predict_rows(
        model,
        prediction_rows,
        device,
        args.workers,
        input_variant=input_variant,
    )

    metrics = TraversabilityEvidenceMetricAccumulator(
        passable_threshold=args.passable_threshold,
        obstacle_threshold=args.obstacle_threshold,
        maximum_obstacle_probability_for_passable=(
            args.maximum_obstacle_probability_for_passable
        ),
        minimum_passable_support=args.minimum_passable_support,
    )
    for row in selected_rows:
        prediction = predictions[evidence_row_key(row)]
        outputs = {
            'passable_logits': torch.logit(
                prediction['passable_probability'].clamp(1e-6, 1 - 1e-6)
            ).unsqueeze(0),
            'obstacle_logits': torch.logit(
                prediction['obstacle_probability'].clamp(1e-6, 1 - 1e-6)
            ).unsqueeze(0),
        }
        metrics.update(
            outputs,
            prediction['passable_target'].unsqueeze(0),
            prediction['obstacle_target'].unsqueeze(0),
            prediction['known_evidence_mask'].unsqueeze(0),
            prediction['controlled_obstacle_mask'].unsqueeze(0),
            prediction['observed_mask'].unsqueeze(0),
            prediction['passable_support'].unsqueeze(0),
        )
    family_metrics = _family_report(
        selected_rows,
        predictions,
        control_by_pair,
        args.obstacle_threshold,
    )
    vehicle_clearance_fusion = _vehicle_clearance_fusion_report(
        selected_rows,
        predictions,
        control_by_pair,
        args.obstacle_threshold,
        args.passable_threshold,
        args.maximum_obstacle_probability_for_passable,
        args.minimum_passable_support,
    )
    result = {
        'schema_version': 1,
        'deployable': False,
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': _sha256(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'split_manifest': str(
            Path(args.split_manifest).expanduser().resolve()
        ),
        'split': args.split,
        'evaluation_slice': args.evaluation_slice or 'all',
        'samples': len(selected_rows),
        'scenes': sorted({row['scene_id'] for row in selected_rows}),
        'device': str(device),
        'evidence_input_variant': input_variant,
        'seed': args.seed,
        'thresholds': {
            'passable_probability': args.passable_threshold,
            'obstacle_probability': args.obstacle_threshold,
            'maximum_obstacle_probability_for_passable': (
                args.maximum_obstacle_probability_for_passable
            ),
            'minimum_passable_support': args.minimum_passable_support,
        },
        'metrics': metrics.compute(),
        'family_metrics': family_metrics,
        'vehicle_clearance_fusion': vehicle_clearance_fusion,
        'vehicle_clearance_policy': {
            'minimum_relative_height_m': 0.15,
            'minimum_vertical_span_m': 0.15,
            'minimum_absolute_height_m': -1.40,
            'fusion': 'hard_obstacle OR learned_obstacle',
        },
        'family_holdout_note': (
            'A family with one controlled sample is a sentinel, not a '
            'statistically reliable generalization estimate.'
        ),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument(
        '--split', choices=['validation', 'test'], default='validation'
    )
    parser.add_argument('--evaluation-slice', default='')
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda']
    )
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--passable-threshold', type=float, default=0.90)
    parser.add_argument('--obstacle-threshold', type=float, default=0.50)
    parser.add_argument(
        '--maximum-obstacle-probability-for-passable',
        type=float,
        default=0.10,
    )
    parser.add_argument('--minimum-passable-support', type=float, default=0.20)
    parser.add_argument(
        '--allow-consumed-test',
        action='store_true',
        help=(
            'Explicitly override the guard on a previously consumed test. '
            'Never use this for model selection.'
        ),
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    result = evaluate(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        _atomic_json(Path(args.output).expanduser().resolve(), result)
    print(rendered, end='')


if __name__ == '__main__':
    main()
