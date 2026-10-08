#!/usr/bin/env python3
"""Compare v2 architectures with leave-one-scene-out development folds.

Rows that still have an untouched ``test`` role are copied into fold manifests
but never loaded or evaluated.  A formerly consumed test may be explicitly
promoted to development once a new untouched test is reserved; that transition
is recorded in the summary rather than silently reusing a test result.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .evaluate_traversability_evidence_model import evaluate
from .train_traversability_evidence_model import train
from .traversability_evidence_dataset import read_evidence_split_manifest


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)


def _write_fold_manifest(
    rows: list[dict],
    validation_scene: str,
    output_path: Path,
) -> dict[str, int]:
    development_scenes = {
        row['scene_id'] for row in rows if row['split'] != 'test'
    }
    if validation_scene not in development_scenes:
        raise ValueError('unknown development scene: ' + validation_scene)
    prepared = []
    for source in rows:
        row = dict(source)
        if source['split'] != 'test':
            row['split'] = (
                'validation'
                if source['scene_id'] == validation_scene
                else 'train'
            )
        prepared.append(row)
    train_scenes = {
        row['scene_id'] for row in prepared if row['split'] == 'train'
    }
    validation_scenes = {
        row['scene_id'] for row in prepared
        if row['split'] == 'validation'
    }
    test_scenes = {
        row['scene_id'] for row in prepared if row['split'] == 'test'
    }
    if train_scenes & validation_scenes:
        raise ValueError('scene leakage between train and validation')
    if (train_scenes | validation_scenes) & test_scenes:
        raise ValueError('development scene leaks into frozen test')
    if validation_scenes != {validation_scene}:
        raise ValueError('fold must contain exactly one validation scene')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prepared[0]))
        writer.writeheader()
        writer.writerows(prepared)
    temporary.replace(output_path)
    return {
        split: sum(row['split'] == split for row in prepared)
        for split in ('train', 'validation', 'test')
    }


def _finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def _promote_consumed_test_rows(rows: list[dict]) -> tuple[
    list[dict], list[str], list[str]
]:
    """Explicitly turn consumed historical test rows into development rows."""
    prepared = []
    promoted_sessions = []
    promoted_scenes = set()
    for source in rows:
        row = dict(source)
        if (
            row.get('split') == 'test'
            and row.get('evaluation_slice') == 'consumed_test_do_not_use'
        ):
            row['split'] = 'train'
            row['evaluation_slice'] = 'promoted_consumed_test_development'
            promoted_sessions.append(row['source_session'])
            promoted_scenes.add(row['scene_id'])
        prepared.append(row)
    mixed = sorted(
        scene for scene in promoted_scenes
        if any(
            row['scene_id'] == scene and row['split'] == 'test'
            for row in prepared
        )
    )
    if mixed:
        raise ValueError(
            'cannot partially promote consumed test scenes: '
            + ', '.join(mixed)
        )
    return prepared, sorted(promoted_scenes), sorted(promoted_sessions)


def _read_source_manifests(args) -> tuple[
    list[dict], list[Path], list[str], list[str]
]:
    paths = [Path(args.split_manifest).expanduser().resolve()]
    paths.extend(
        Path(path).expanduser().resolve()
        for path in getattr(args, 'additional_development_manifest', [])
    )
    rows = []
    seen_samples = set()
    for path in paths:
        for row in read_evidence_split_manifest(path):
            sample = str(
                Path(row['derived_sample_path']).expanduser().resolve()
            )
            if sample in seen_samples:
                raise ValueError(
                    'duplicate derived sample across manifests: ' + sample
                )
            seen_samples.add(sample)
            rows.append(row)
    promoted_scenes = []
    promoted_sessions = []
    if getattr(args, 'promote_consumed_test_to_development', False):
        rows, promoted_scenes, promoted_sessions = (
            _promote_consumed_test_rows(rows)
        )
    return rows, paths, promoted_scenes, promoted_sessions


def _aggregate(folds: list[dict]) -> dict:
    metric_paths = {
        'obstacle_iou': ('metrics', 'obstacle_head', 'iou'),
        'obstacle_precision': ('metrics', 'obstacle_head', 'precision'),
        'obstacle_recall': ('metrics', 'obstacle_head', 'recall'),
        'passable_iou': ('metrics', 'passable_head', 'iou'),
        'controlled_cell_recall': (
            'metrics', 'controlled_obstacle_cell_recall',
        ),
        'controlled_instance_detection_rate': (
            'metrics', 'controlled_instance_any_detection_rate',
        ),
        'controlled_unsafe_passable_cell_rate': (
            'metrics', 'controlled_unsafe_passable_cell_rate',
        ),
        'controlled_unsafe_passable_instance_rate': (
            'metrics', 'controlled_unsafe_passable_instance_rate',
        ),
        'fused_obstacle_iou': (
            'vehicle_clearance_fusion', 'obstacle_iou',
        ),
        'fused_obstacle_precision': (
            'vehicle_clearance_fusion', 'obstacle_precision',
        ),
        'fused_obstacle_recall': (
            'vehicle_clearance_fusion', 'obstacle_recall',
        ),
        'fused_controlled_cell_recall': (
            'vehicle_clearance_fusion', 'controlled_cell_recall',
        ),
        'fused_controlled_instance_detection_rate': (
            'vehicle_clearance_fusion',
            'controlled_instance_detection_rate',
        ),
        'fused_paired_control_false_obstacle_rate': (
            'vehicle_clearance_fusion',
            'paired_control_false_obstacle_rate',
        ),
        'fused_controlled_unsafe_passable_cell_rate': (
            'vehicle_clearance_fusion',
            'controlled_unsafe_passable_cell_rate',
        ),
        'fused_controlled_unsafe_passable_instance_rate': (
            'vehicle_clearance_fusion',
            'controlled_unsafe_passable_instance_rate',
        ),
    }
    summary = {}
    for name, path in metric_paths.items():
        values = []
        for fold in folds:
            value = fold['evaluation']
            for key in path:
                value = value[key]
            values.append(value)
        finite = _finite(values)
        summary[name] = {
            'mean': sum(finite) / len(finite),
            'minimum': min(finite),
            'maximum': max(finite),
            'values': values,
        }
    deltas = []
    control_probabilities = []
    for fold in folds:
        for family in fold['evaluation']['family_metrics'].values():
            deltas.append(
                family['mean_paired_obstacle_probability_delta']
            )
            control_probabilities.append(
                family['mean_paired_control_obstacle_probability']
            )
    finite_deltas = _finite(deltas)
    finite_controls = _finite(control_probabilities)
    summary['paired_family_delta'] = {
        'mean': sum(finite_deltas) / len(finite_deltas),
        'minimum': min(finite_deltas),
        'values': deltas,
    }
    summary['paired_control_probability'] = {
        'mean': sum(finite_controls) / len(finite_controls),
        'maximum': max(finite_controls),
        'values': control_probabilities,
    }
    return summary


def cross_validate(args) -> dict:
    (
        rows,
        source_manifests,
        promoted_scenes,
        promoted_sessions,
    ) = _read_source_manifests(args)
    development_scenes = sorted({
        row['scene_id'] for row in rows if row['split'] != 'test'
    })
    if len(development_scenes) < 2:
        raise ValueError(
            'cross-validation needs at least two development scenes'
        )
    output_root = Path(args.output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold_index, validation_scene in enumerate(development_scenes):
        fold_directory = output_root / ('holdout_' + validation_scene)
        manifest = fold_directory / 'split_manifest.csv'
        counts = _write_fold_manifest(rows, validation_scene, manifest)
        if getattr(args, 'evaluation_only', False):
            checkpoint = fold_directory / 'model' / 'best.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    'evaluation-only fold lacks checkpoint: '
                    + str(checkpoint)
                )
            train_result = {'best_checkpoint': str(checkpoint)}
        else:
            train_result = train(argparse.Namespace(
                split_manifest=manifest,
                output_directory=fold_directory / 'model',
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                workers=args.workers,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                architecture=args.architecture,
                input_variant=getattr(args, 'input_variant', 'auto'),
                dropout_probability=args.dropout_probability,
                flip_probability=args.flip_probability,
                controlled_obstacle_auxiliary_weight=(
                    args.controlled_obstacle_auxiliary_weight
                ),
                paired_counterfactual_weight=(
                    args.paired_counterfactual_weight
                ),
                paired_counterfactual_margin=(
                    args.paired_counterfactual_margin
                ),
                minimum_improvement=args.minimum_improvement,
                early_stopping_patience=args.early_stopping_patience,
                passable_threshold=args.passable_threshold,
                obstacle_threshold=args.obstacle_threshold,
                maximum_obstacle_probability_for_passable=(
                    args.maximum_obstacle_probability_for_passable
                ),
                minimum_passable_support=args.minimum_passable_support,
                minimum_controlled_instance_recall=(
                    args.minimum_controlled_instance_recall
                ),
                max_train_batches=None,
                max_validation_batches=None,
                seed=args.seed + fold_index,
            ))
        evaluation = evaluate(argparse.Namespace(
            checkpoint=Path(train_result['best_checkpoint']),
            split_manifest=manifest,
            split='validation',
            evaluation_slice='',
            output=fold_directory / 'validation_metrics.json',
            device=args.device,
            workers=args.workers,
            seed=args.seed + fold_index,
            passable_threshold=args.passable_threshold,
            obstacle_threshold=args.obstacle_threshold,
            maximum_obstacle_probability_for_passable=(
                args.maximum_obstacle_probability_for_passable
            ),
            minimum_passable_support=args.minimum_passable_support,
        ))
        _atomic_json(fold_directory / 'validation_metrics.json', evaluation)
        folds.append({
            'validation_scene': validation_scene,
            'sample_counts': counts,
            'model_directory': str(fold_directory / 'model'),
            'training_result': train_result,
            'evaluation': evaluation,
        })
    summary = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'architecture': args.architecture,
        'source_split_manifest': str(
            Path(args.split_manifest).expanduser().resolve()
        ),
        'source_split_manifests': [str(path) for path in source_manifests],
        'development_scenes': development_scenes,
        'frozen_test_accessed': False,
        'untouched_test_accessed': False,
        'consumed_test_promoted_to_development': bool(promoted_sessions),
        'promoted_consumed_test_scenes': promoted_scenes,
        'promoted_consumed_test_sessions': promoted_sessions,
        'test_policy': (
            'Rows still assigned to test are never loaded or evaluated. '
            'Consumed historical tests are development data only when '
            'explicitly promoted; the next untouched test remains unseen.'
        ),
        'fold_count': len(folds),
        'evaluation_only': bool(getattr(args, 'evaluation_only', False)),
        'folds': folds,
        'aggregate': _aggregate(folds),
    }
    _atomic_json(output_root / 'cross_validation_summary.json', summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument(
        '--additional-development-manifest',
        type=Path,
        action='append',
        default=[],
        help=(
            'Additional prepared manifest whose non-test scenes join '
            'development; may be repeated.'
        ),
    )
    parser.add_argument(
        '--promote-consumed-test-to-development',
        action='store_true',
        help=(
            'Explicitly reclassify rows marked consumed_test_do_not_use as '
            'development after reserving a new untouched test.'
        ),
    )
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument(
        '--evaluation-only',
        action='store_true',
        help=(
            'Reuse each existing fold model/best.pt and rebuild only the '
            'evaluation reports and aggregate summary.'
        ),
    )
    parser.add_argument(
        '--architecture',
        choices=[
            'unet_context', 'decoupled_unet_context', 'local_evidence',
        ],
        required=True,
    )
    parser.add_argument(
        '--device', choices=['auto', 'cpu', 'cuda'], default='auto'
    )
    parser.add_argument(
        '--input-variant',
        choices=['auto', 'current_only', 'temporal'],
        default='auto',
    )
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--base-channels', type=int, default=8)
    parser.add_argument('--dropout-probability', type=float, default=0.10)
    parser.add_argument('--flip-probability', type=float, default=0.50)
    parser.add_argument(
        '--controlled-obstacle-auxiliary-weight', type=float, default=2.0
    )
    parser.add_argument(
        '--paired-counterfactual-weight', type=float, default=1.0
    )
    parser.add_argument(
        '--paired-counterfactual-margin', type=float, default=2.0
    )
    parser.add_argument('--minimum-improvement', type=float, default=1e-4)
    parser.add_argument('--early-stopping-patience', type=int, default=8)
    parser.add_argument('--passable-threshold', type=float, default=0.90)
    parser.add_argument('--obstacle-threshold', type=float, default=0.50)
    parser.add_argument(
        '--maximum-obstacle-probability-for-passable',
        type=float,
        default=0.10,
    )
    parser.add_argument('--minimum-passable-support', type=float, default=0.20)
    parser.add_argument(
        '--minimum-controlled-instance-recall',
        type=float,
        default=0.95,
    )
    parser.add_argument('--seed', type=int, default=20260921)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    result = cross_validate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
