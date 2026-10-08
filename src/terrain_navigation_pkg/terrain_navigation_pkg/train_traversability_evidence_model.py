#!/usr/bin/env python3
"""Train the independent-head v2 traversability evidence pilot model."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler

from .traversability_evidence_dataset import (
    PairedTraversabilityEvidenceDataset,
    family_balanced_sample_weights,
    read_evidence_split_manifest,
    select_lidar_evidence_bev,
    split_evidence_rows,
)
from .traversability_evidence_metrics import (
    TraversabilityEvidenceMetricAccumulator,
)
from .traversability_evidence_model import (
    build_traversability_evidence_model,
    independent_evidence_loss,
)


DEFAULT_OUTPUT = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/pilot_20260918'
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value) -> None:
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


def _positive_weights(rows) -> tuple[dict[str, float], dict[str, int]]:
    passable_cells = 0
    obstacle_cells = 0
    controlled_cells = 0
    for row in rows:
        with np.load(row['derived_sample_path'], allow_pickle=False) as arrays:
            passable = np.asarray(
                arrays['target_passable_surface_mask'], dtype=bool
            )
            obstacle = np.asarray(
                arrays['target_obstacle_evidence_mask'], dtype=bool
            )
            passable_cells += int(np.count_nonzero(passable))
            obstacle_cells += int(np.count_nonzero(obstacle))
            controlled_cells += int(np.count_nonzero(
                arrays['target_controlled_obstacle_point_count'] > 0
            ))
    if passable_cells <= 0 or obstacle_cells <= 0:
        raise ValueError('training split needs passable and obstacle evidence')
    weights = {
        'passable': max(1.0, math.sqrt(obstacle_cells / passable_cells)),
        'obstacle': max(1.0, math.sqrt(passable_cells / obstacle_cells)),
    }
    counts = {
        'passable_cells': passable_cells,
        'obstacle_cells': obstacle_cells,
        'controlled_obstacle_cells': controlled_cells,
    }
    return weights, counts


def _input_channel_count(rows, input_variant: str = 'auto') -> int:
    """Require one consistent evidence representation across all splits."""

    counts = set()
    for row in rows:
        with np.load(row['derived_sample_path'], allow_pickle=False) as arrays:
            evidence = select_lidar_evidence_bev(arrays, input_variant)
            if evidence.ndim != 3:
                raise ValueError(
                    'lidar_evidence_bev must be three-dimensional'
                )
            counts.add(int(evidence.shape[0]))
    if len(counts) != 1:
        raise ValueError(
            'split manifest mixes evidence input channel counts: '
            + ', '.join(str(value) for value in sorted(counts))
        )
    channels = counts.pop()
    if channels not in (8, 17):
        raise ValueError(
            f'unsupported evidence input channel count: {channels}'
        )
    return channels


def _limited_batches(loader, maximum_batches):
    for index, batch in enumerate(loader):
        if maximum_batches is not None and index >= maximum_batches:
            break
        yield batch


def _to_device(batch, device):
    non_blocking = device.type == 'cuda'
    keys = (
        'lidar_evidence_bev', 'passable_target', 'obstacle_target',
        'known_evidence_mask', 'controlled_obstacle_mask', 'observed_mask',
        'passable_support',
    )
    return {
        key: batch[key].to(device, non_blocking=non_blocking)
        for key in keys
    }


def _run_epoch(
    model,
    loader,
    device,
    positive_weights,
    controlled_auxiliary_weight,
    paired_counterfactual_weight,
    paired_counterfactual_margin,
    *,
    optimizers=None,
    active_branches=('passable', 'obstacle'),
    passable_threshold=0.90,
    obstacle_threshold=0.50,
    maximum_obstacle_probability_for_passable=0.10,
    minimum_passable_support=0.20,
    maximum_batches=None,
) -> tuple[float, dict, dict]:
    training = optimizers is not None
    active_branches = frozenset(active_branches)
    if not active_branches <= {'passable', 'obstacle'}:
        raise ValueError('unknown active evidence branch')
    model.train(training)
    metrics = TraversabilityEvidenceMetricAccumulator(
        passable_threshold=passable_threshold,
        obstacle_threshold=obstacle_threshold,
        maximum_obstacle_probability_for_passable=(
            maximum_obstacle_probability_for_passable
        ),
        minimum_passable_support=minimum_passable_support,
    )
    total_loss = 0.0
    component_totals = {
        'passable_loss': 0.0,
        'obstacle_loss': 0.0,
        'controlled_obstacle_loss': 0.0,
        'paired_counterfactual_loss': 0.0,
    }
    paired_control_cells = 0
    paired_control_false_obstacle_cells = 0
    batch_count = 0
    for raw_batch in _limited_batches(loader, maximum_batches):
        batch = _to_device(raw_batch, device)
        if training:
            for optimizer in {id(value): value for value in (
                optimizers.values()
            )}.values():
                optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            paired_control = raw_batch[
                'paired_control_lidar_evidence_bev'
            ].to(device, non_blocking=device.type == 'cuda')
            combined = torch.cat(
                (batch['lidar_evidence_bev'], paired_control), dim=0
            )
            combined_outputs = model(combined)
            batch_size = batch['lidar_evidence_bev'].shape[0]
            outputs = {
                key: value[:batch_size]
                for key, value in combined_outputs.items()
            }
            paired_control_obstacle_logits = combined_outputs[
                'obstacle_logits'
            ][batch_size:]
            loss, components = independent_evidence_loss(
                outputs,
                batch['passable_target'],
                batch['obstacle_target'],
                batch['known_evidence_mask'],
                batch['controlled_obstacle_mask'],
                paired_control_obstacle_logits=(
                    paired_control_obstacle_logits
                ),
                passable_positive_weight=positive_weights['passable'],
                obstacle_positive_weight=positive_weights['obstacle'],
                controlled_obstacle_auxiliary_weight=(
                    controlled_auxiliary_weight
                ),
                paired_counterfactual_weight=(
                    paired_counterfactual_weight
                ),
                paired_counterfactual_margin=paired_counterfactual_margin,
            )
            if training:
                passable_objective = components['passable_loss']
                obstacle_objective = (
                    components['obstacle_loss']
                    + float(controlled_auxiliary_weight)
                    * components['controlled_obstacle_loss']
                    + float(paired_counterfactual_weight)
                    * components['paired_counterfactual_loss']
                )
                active_objectives = []
                if 'passable' in active_branches:
                    active_objectives.append(passable_objective)
                if 'obstacle' in active_branches:
                    active_objectives.append(obstacle_objective)
                if active_objectives:
                    torch.stack(active_objectives).sum().backward()
                    if 'joint' in optimizers:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 5.0
                        )
                        optimizers['joint'].step()
                    else:
                        for name in active_branches:
                            parameters = optimizers[name].param_groups[0][
                                'params'
                            ]
                            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                            optimizers[name].step()
        total_loss += float(loss.detach())
        for key in component_totals:
            component_totals[key] += float(components[key].detach())
        batch_count += 1
        metrics.update(
            outputs,
            batch['passable_target'],
            batch['obstacle_target'],
            batch['known_evidence_mask'],
            batch['controlled_obstacle_mask'],
            batch['observed_mask'],
            batch['passable_support'],
        )
        controlled = batch['controlled_obstacle_mask'].bool()
        paired_control_cells += int(torch.count_nonzero(controlled))
        paired_control_false_obstacle_cells += int(torch.count_nonzero(
            controlled
            & (
                torch.sigmoid(paired_control_obstacle_logits)
                >= obstacle_threshold
            )
        ))
    if batch_count == 0:
        raise ValueError('data loader produced no batches')
    computed_metrics = metrics.compute()
    computed_metrics['paired_control_cells'] = paired_control_cells
    computed_metrics['paired_control_false_obstacle_cells'] = (
        paired_control_false_obstacle_cells
    )
    computed_metrics['paired_control_false_obstacle_rate'] = (
        float(paired_control_false_obstacle_cells / paired_control_cells)
        if paired_control_cells else math.nan
    )
    return (
        total_loss / batch_count,
        {key: value / batch_count for key, value in component_totals.items()},
        computed_metrics,
    )


def _flatten(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(prefix + key + '_', child, output)
    elif isinstance(value, (int, float)):
        output[prefix[:-1]] = value


def _cpu_state_dict(module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _obstacle_objective(components, args) -> float:
    return (
        float(components['obstacle_loss'])
        + float(args.controlled_obstacle_auxiliary_weight)
        * float(components['controlled_obstacle_loss'])
        + float(args.paired_counterfactual_weight)
        * float(components['paired_counterfactual_loss'])
    )


def _finite_or(value, fallback: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else fallback


def _obstacle_selection_key(
    metrics: dict,
    objective_loss: float,
    *,
    minimum_controlled_instance_recall: float,
) -> tuple:
    """Rank assembled checkpoints by safety gates, then obstacle IoU.

    A lower nonzero violation is preferred when no candidate reaches a gate.
    Once a gate is satisfied, later criteria decide.  This makes the fallback
    deterministic without pretending that a failed gate passed.
    """
    unsafe = _finite_or(
        metrics['controlled_unsafe_passable_instance_rate'], math.inf
    )
    paired_false = _finite_or(
        metrics['paired_control_false_obstacle_rate'], math.inf
    )
    controlled_recall = _finite_or(
        metrics['controlled_instance_any_detection_rate'], -math.inf
    )
    obstacle_iou = _finite_or(
        metrics['obstacle_head']['iou'], -math.inf
    )
    recall_gate_passed = (
        controlled_recall >= minimum_controlled_instance_recall
    )
    return (
        int(unsafe <= 0.0),
        -unsafe,
        int(paired_false <= 0.0),
        -paired_false,
        int(recall_gate_passed),
        0.0 if recall_gate_passed else controlled_recall,
        obstacle_iou,
        controlled_recall,
        -float(objective_loss),
    )


def _selection_summary(metrics: dict, objective_loss: float) -> dict:
    return {
        'controlled_unsafe_passable_instance_rate': metrics[
            'controlled_unsafe_passable_instance_rate'
        ],
        'paired_control_false_obstacle_rate': metrics[
            'paired_control_false_obstacle_rate'
        ],
        'controlled_instance_detection_rate': metrics[
            'controlled_instance_any_detection_rate'
        ],
        'obstacle_iou': metrics['obstacle_head']['iou'],
        'obstacle_objective_loss': float(objective_loss),
    }


def train(args) -> dict:
    _set_seed(args.seed)
    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    rows = read_evidence_split_manifest(args.split_manifest)
    splits = split_evidence_rows(rows)
    input_variant = str(getattr(args, 'input_variant', 'auto'))
    input_channels = _input_channel_count(rows, input_variant)
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    train_dataset = PairedTraversabilityEvidenceDataset(
        splits['train'],
        horizontal_flip_probability=args.flip_probability,
        input_variant=input_variant,
    )
    validation_dataset = PairedTraversabilityEvidenceDataset(
        splits['validation'], input_variant=input_variant
    )
    generator = torch.Generator().manual_seed(args.seed)
    sample_weights = torch.as_tensor(
        family_balanced_sample_weights(splits['train']), dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
    )

    positive_weights, target_counts = _positive_weights(splits['train'])
    model_config = {
        'architecture': args.architecture,
        'input_channels': input_channels,
        'base_channels': args.base_channels,
        'dropout_probability': args.dropout_probability,
        'use_coordinate_channels': args.architecture in (
            'unet_context', 'decoupled_unet_context',
        ),
    }
    model = build_traversability_evidence_model(model_config).to(device)
    independent_controller = (
        args.architecture == 'decoupled_unet_context'
    )
    if independent_controller:
        optimizers = {
            'passable': torch.optim.AdamW(
                model.passable_network.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
            'obstacle': torch.optim.AdamW(
                model.obstacle_network.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
        }
        schedulers = {
            name: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
            for name, optimizer in optimizers.items()
        }
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizers = {'joint': optimizer}
        schedulers = {
            'joint': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            )
        }
    config = {
        'schema_version': 2 if independent_controller else 1,
        'created_at': datetime.now(timezone.utc).astimezone().isoformat(),
        'purpose': (
            'controlled v2 evidence pilot; validates the learning contract '
            'and is not deployable'
        ),
        'deployable': False,
        'split_manifest': str(
            Path(args.split_manifest).expanduser().resolve()
        ),
        'output_directory': str(output),
        'target_contract': 'independent_passable_obstacle_evidence',
        'evidence_input_variant': input_variant,
        'input_variant': (
            'current_plus_ego_motion_temporal_5scan'
            if input_channels == 17 else 'single_scan'
        ),
        'unknown_policy': 'ambiguous and unobserved cells are ignored',
        'test_access_policy': 'test rows are never loaded by this trainer',
        'device': str(device),
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'flip_probability': args.flip_probability,
        'controlled_obstacle_auxiliary_weight': (
            args.controlled_obstacle_auxiliary_weight
        ),
        'paired_counterfactual_weight': args.paired_counterfactual_weight,
        'paired_counterfactual_margin': args.paired_counterfactual_margin,
        'training_controller_policy': (
            'branch_independent'
            if independent_controller else 'legacy_joint'
        ),
        'checkpoint_selection_policy': (
            'independent_passable_loss_then_safety_constrained_obstacle'
            if independent_controller else 'minimum_combined_validation_loss'
        ),
        'minimum_controlled_instance_recall': (
            args.minimum_controlled_instance_recall
        ),
        'selection_thresholds': {
            'passable_threshold': args.passable_threshold,
            'obstacle_threshold': args.obstacle_threshold,
            'maximum_obstacle_probability_for_passable': (
                args.maximum_obstacle_probability_for_passable
            ),
            'minimum_passable_support': args.minimum_passable_support,
        },
        'positive_weights': positive_weights,
        'training_target_counts': target_counts,
        'split_sample_counts': {
            name: len(split_rows) for name, split_rows in splits.items()
        },
        'training_families': sorted({
            row['object_family'] for row in splits['train']
        }),
        'validation_scenes': sorted({
            row['scene_id'] for row in splits['validation']
        }),
        'model': model_config,
        'maximum_train_batches': args.max_train_batches,
        'maximum_validation_batches': args.max_validation_batches,
    }
    _atomic_json(output / 'training_config.json', config)

    history = []
    threshold_arguments = {
        'passable_threshold': args.passable_threshold,
        'obstacle_threshold': args.obstacle_threshold,
        'maximum_obstacle_probability_for_passable': (
            args.maximum_obstacle_probability_for_passable
        ),
        'minimum_passable_support': args.minimum_passable_support,
    }
    if independent_controller:
        active = {'passable': True, 'obstacle': True}
        stale = {'passable': 0, 'obstacle': 0}
        best_branch_loss = {'passable': math.inf, 'obstacle': math.inf}
        best_passable_state = None
        best_passable_epoch = None
        obstacle_candidates = []
        for epoch in range(1, args.epochs + 1):
            trained_branches = tuple(
                name for name, enabled in active.items() if enabled
            )
            train_loss, train_components, train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                positive_weights,
                args.controlled_obstacle_auxiliary_weight,
                args.paired_counterfactual_weight,
                args.paired_counterfactual_margin,
                optimizers=optimizers,
                active_branches=trained_branches,
                maximum_batches=args.max_train_batches,
                **threshold_arguments,
            )
            with torch.no_grad():
                (
                    validation_loss,
                    validation_components,
                    validation_metrics,
                ) = _run_epoch(
                    model,
                    validation_loader,
                    device,
                    positive_weights,
                    args.controlled_obstacle_auxiliary_weight,
                    args.paired_counterfactual_weight,
                    args.paired_counterfactual_margin,
                    maximum_batches=args.max_validation_batches,
                    **threshold_arguments,
                )
            validation_branch_loss = {
                'passable': float(
                    validation_components['passable_loss']
                ),
                'obstacle': _obstacle_objective(
                    validation_components, args
                ),
            }
            for name in trained_branches:
                schedulers[name].step(validation_branch_loss[name])

            record = {
                'epoch': epoch,
                'learning_rate': min(
                    optimizer.param_groups[0]['lr']
                    for optimizer in optimizers.values()
                ),
                'learning_rate_passable': (
                    optimizers['passable'].param_groups[0]['lr']
                ),
                'learning_rate_obstacle': (
                    optimizers['obstacle'].param_groups[0]['lr']
                ),
                'passable_branch_active': int('passable' in trained_branches),
                'obstacle_branch_active': int('obstacle' in trained_branches),
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'validation_passable_objective': validation_branch_loss[
                    'passable'
                ],
                'validation_obstacle_objective': validation_branch_loss[
                    'obstacle'
                ],
            }
            _flatten('train_component_', train_components, record)
            _flatten(
                'validation_component_', validation_components, record
            )
            _flatten('train_metric_', train_metrics, record)
            _flatten('validation_metric_', validation_metrics, record)
            history.append(record)

            if 'passable' in trained_branches:
                improved = (
                    validation_branch_loss['passable']
                    < best_branch_loss['passable']
                    - args.minimum_improvement
                )
                if improved:
                    best_branch_loss['passable'] = (
                        validation_branch_loss['passable']
                    )
                    best_passable_state = _cpu_state_dict(
                        model.passable_network
                    )
                    best_passable_epoch = epoch
                    stale['passable'] = 0
                else:
                    stale['passable'] += 1
            if 'obstacle' in trained_branches:
                obstacle_candidates.append({
                    'epoch': epoch,
                    'state_dict': _cpu_state_dict(model.obstacle_network),
                    'objective_loss': validation_branch_loss['obstacle'],
                })
                improved = (
                    validation_branch_loss['obstacle']
                    < best_branch_loss['obstacle']
                    - args.minimum_improvement
                )
                if improved:
                    best_branch_loss['obstacle'] = (
                        validation_branch_loss['obstacle']
                    )
                    stale['obstacle'] = 0
                else:
                    stale['obstacle'] += 1

            checkpoint = {
                'schema_version': 2,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dicts': {
                    name: optimizer.state_dict()
                    for name, optimizer in optimizers.items()
                },
                'model_config': model_config,
                'training_config': config,
                'positive_weights': positive_weights,
                'validation_metrics': validation_metrics,
                'validation_loss': validation_loss,
                'validation_branch_loss': validation_branch_loss,
                'active_branches': list(trained_branches),
            }
            _atomic_checkpoint(output / 'last.pt', checkpoint)

            print(
                'epoch={:03d} train_loss={:.5f} val_passable={:.5f} '
                'val_obstacle={:.5f} val_obstacle_iou={:.4f} '
                'val_controlled_instance={:.4f} lr_passable={:.2e} '
                'lr_obstacle={:.2e}'.format(
                    epoch,
                    train_loss,
                    validation_branch_loss['passable'],
                    validation_branch_loss['obstacle'],
                    validation_metrics['obstacle_head']['iou'],
                    validation_metrics[
                        'controlled_instance_any_detection_rate'
                    ],
                    optimizers['passable'].param_groups[0]['lr'],
                    optimizers['obstacle'].param_groups[0]['lr'],
                ),
                flush=True,
            )
            for name in trained_branches:
                if stale[name] >= args.early_stopping_patience:
                    active[name] = False
                    print(
                        '{} branch early stopping after epoch {}'.format(
                            name, epoch
                        ),
                        flush=True,
                    )
            if not any(active.values()):
                break

        if best_passable_state is None or not obstacle_candidates:
            raise RuntimeError('branch-independent training made no candidate')
        model.passable_network.load_state_dict(best_passable_state)
        selected_obstacle = None
        selection_records = []
        for candidate in obstacle_candidates:
            model.obstacle_network.load_state_dict(candidate['state_dict'])
            with torch.no_grad():
                candidate_loss, candidate_components, candidate_metrics = (
                    _run_epoch(
                        model,
                        validation_loader,
                        device,
                        positive_weights,
                        args.controlled_obstacle_auxiliary_weight,
                        args.paired_counterfactual_weight,
                        args.paired_counterfactual_margin,
                        maximum_batches=args.max_validation_batches,
                        **threshold_arguments,
                    )
                )
            candidate_objective = _obstacle_objective(
                candidate_components, args
            )
            key = _obstacle_selection_key(
                candidate_metrics,
                candidate_objective,
                minimum_controlled_instance_recall=(
                    args.minimum_controlled_instance_recall
                ),
            )
            selection_record = {
                'epoch': candidate['epoch'],
                **_selection_summary(
                    candidate_metrics, candidate_objective
                ),
            }
            selection_records.append(selection_record)
            if (
                selected_obstacle is None
                or key > selected_obstacle['selection_key']
            ):
                selected_obstacle = {
                    **candidate,
                    'selection_key': key,
                    'validation_loss': candidate_loss,
                    'validation_components': candidate_components,
                    'validation_metrics': candidate_metrics,
                    'selection_record': selection_record,
                }

        model.obstacle_network.load_state_dict(
            selected_obstacle['state_dict']
        )
        assembled_validation_loss = selected_obstacle['validation_loss']
        assembled_validation_metrics = selected_obstacle[
            'validation_metrics'
        ]
        selection_gate_passed = (
            selected_obstacle['selection_record'][
                'controlled_unsafe_passable_instance_rate'
            ] <= 0.0
            and selected_obstacle['selection_record'][
                'paired_control_false_obstacle_rate'
            ] <= 0.0
            and selected_obstacle['selection_record'][
                'controlled_instance_detection_rate'
            ] >= args.minimum_controlled_instance_recall
        )
        branch_selection = {
            'policy': (
                'minimum passable loss; with that branch fixed, rank every '
                'obstacle epoch by unsafe-passable zero, paired-control '
                'false-obstacle zero, controlled-instance recall gate, then '
                'obstacle IoU'
            ),
            'passable': {
                'epoch': best_passable_epoch,
                'validation_objective_loss': best_branch_loss['passable'],
            },
            'obstacle': selected_obstacle['selection_record'],
            'minimum_controlled_instance_recall': (
                args.minimum_controlled_instance_recall
            ),
            'safety_gates_passed': selection_gate_passed,
            'obstacle_candidates': selection_records,
        }
        best_checkpoint = {
            'schema_version': 2,
            'epoch': max(
                best_passable_epoch, selected_obstacle['epoch']
            ),
            'branch_epochs': {
                'passable': best_passable_epoch,
                'obstacle': selected_obstacle['epoch'],
            },
            'assembled_from_independent_branches': True,
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'training_config': config,
            'positive_weights': positive_weights,
            'validation_metrics': assembled_validation_metrics,
            'validation_loss': assembled_validation_loss,
            'branch_selection': branch_selection,
        }
        _atomic_checkpoint(output / 'best.pt', best_checkpoint)
        _atomic_checkpoint(output / 'best_passable_branch.pt', {
            'schema_version': 1,
            'branch': 'passable',
            'epoch': best_passable_epoch,
            'state_dict': best_passable_state,
            'validation_objective_loss': best_branch_loss['passable'],
        })
        _atomic_checkpoint(output / 'best_obstacle_branch.pt', {
            'schema_version': 1,
            'branch': 'obstacle',
            'epoch': selected_obstacle['epoch'],
            'state_dict': selected_obstacle['state_dict'],
            'selection': selected_obstacle['selection_record'],
        })
        _atomic_json(output / 'branch_selection.json', branch_selection)
        best_loss = assembled_validation_loss
    else:
        best_loss = math.inf
        stale_epochs = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_components, train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                positive_weights,
                args.controlled_obstacle_auxiliary_weight,
                args.paired_counterfactual_weight,
                args.paired_counterfactual_margin,
                optimizers=optimizers,
                maximum_batches=args.max_train_batches,
                **threshold_arguments,
            )
            with torch.no_grad():
                (
                    validation_loss,
                    validation_components,
                    validation_metrics,
                ) = _run_epoch(
                    model,
                    validation_loader,
                    device,
                    positive_weights,
                    args.controlled_obstacle_auxiliary_weight,
                    args.paired_counterfactual_weight,
                    args.paired_counterfactual_margin,
                    maximum_batches=args.max_validation_batches,
                    **threshold_arguments,
                )
            schedulers['joint'].step(validation_loss)
            record = {
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'validation_loss': validation_loss,
            }
            _flatten('train_component_', train_components, record)
            _flatten(
                'validation_component_', validation_components, record
            )
            _flatten('train_metric_', train_metrics, record)
            _flatten('validation_metric_', validation_metrics, record)
            history.append(record)
            checkpoint = {
                'schema_version': 1,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'training_config': config,
                'positive_weights': positive_weights,
                'validation_metrics': validation_metrics,
                'validation_loss': validation_loss,
            }
            _atomic_checkpoint(output / 'last.pt', checkpoint)
            improved = (
                validation_loss
                < best_loss - args.minimum_improvement
            )
            if improved:
                best_loss = validation_loss
                stale_epochs = 0
                _atomic_checkpoint(output / 'best.pt', checkpoint)
            else:
                stale_epochs += 1
            print(
                'epoch={:03d} train_loss={:.5f} val_loss={:.5f} '
                'val_obstacle_iou={:.4f} val_controlled_recall={:.4f} '
                'lr={:.2e}'.format(
                    epoch,
                    train_loss,
                    validation_loss,
                    validation_metrics['obstacle_head']['iou'],
                    validation_metrics['controlled_obstacle_cell_recall'],
                    optimizer.param_groups[0]['lr'],
                ),
                flush=True,
            )
            if stale_epochs >= args.early_stopping_patience:
                print(
                    'early stopping after epoch {}'.format(epoch),
                    flush=True,
                )
                break

    history_path = output / 'history.csv'
    with history_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _atomic_json(output / 'history.json', history)
    result = {
        'deployable': False,
        'device': str(device),
        'epochs_completed': len(history),
        'best_validation_loss': best_loss,
        'best_checkpoint': str(output / 'best.pt'),
        'training_controller_policy': config[
            'training_controller_policy'
        ],
        'history': str(history_path),
        'split_sample_counts': config['split_sample_counts'],
    }
    if independent_controller:
        result.update({
            'best_passable_branch': str(
                output / 'best_passable_branch.pt'
            ),
            'best_obstacle_branch': str(
                output / 'best_obstacle_branch.pt'
            ),
            'branch_selection': str(output / 'branch_selection.json'),
            'passable_epoch': best_passable_epoch,
            'obstacle_epoch': selected_obstacle['epoch'],
            'selection_safety_gates_passed': selection_gate_passed,
        })
    _atomic_json(output / 'result.json', result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument(
        '--output-directory', type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        '--device', default='auto', choices=['auto', 'cpu', 'cuda']
    )
    parser.add_argument(
        '--input-variant',
        choices=['auto', 'current_only', 'temporal'],
        default='auto',
        help=(
            'Select the archived evidence view. current_only and temporal '
            'allow an exact-sample representation comparison.'
        ),
    )
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument(
        '--architecture',
        choices=[
            'unet_context', 'decoupled_unet_context', 'local_evidence',
        ],
        default='unet_context',
    )
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
    parser.add_argument(
        '--minimum-passable-support', type=float, default=0.20
    )
    parser.add_argument(
        '--minimum-controlled-instance-recall',
        type=float,
        default=0.95,
    )
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-validation-batches', type=int)
    parser.add_argument('--seed', type=int, default=20260918)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('epochs and batch-size must be positive')
    probabilities = (
        args.passable_threshold,
        args.obstacle_threshold,
        args.maximum_obstacle_probability_for_passable,
        args.minimum_passable_support,
        args.minimum_controlled_instance_recall,
    )
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError('selection thresholds must be in [0, 1]')
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
