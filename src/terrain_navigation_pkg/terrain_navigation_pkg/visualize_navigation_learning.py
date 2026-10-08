#!/usr/bin/env python3
"""Audit and visualize fixed-horizon targets from navigation recordings."""

from __future__ import annotations

import os
import sys

# Ubuntu's system Matplotlib is compiled against NumPy 1.x.  Do not mix it
# with a user-site NumPy 2.x installation when this console script starts.
if os.environ.get('PYTHONNOUSERSITE') != '1':
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            '-m',
            'terrain_navigation_pkg.visualize_navigation_learning',
            *sys.argv[1:],
        ],
        environment,
    )

import argparse  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from .navigation_trajectory_core import (  # noqa: E402
    INITIAL_BC_EXCLUSION_FLAGS,
    analyze_nav2_plan,
    initial_bc_trajectory_decision,
    median_nearest_distance,
    transform_vehicle_points,
)


DEFAULT_MANIFEST = Path(
    '/home/sukja/terrain_nav_data/learning/manifests/v1/manifest_all.csv'
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/visualizations/v1'
)
DEFAULT_SPLITS = ('train', 'validation')
TRAJECTORY_COLOR = '#2C7FB8'
TARGET_COLOR = '#D95F02'
GOAL_COLOR = '#7B3294'
VEHICLE_COLOR = '#202020'


QUALITY_FIELDS = [
    'session', 'sample_id', 'split', 'sample_path', 'speed_mps',
    'route_index', 'route_size', 'source_point_count',
    'forward_path_length_m', 'valid_target_count', 'vehicle_to_path_m',
    'maximum_source_gap_m', 'maximum_curvature_per_m',
    'absolute_heading_change_deg', 'backward_target_fraction',
    'goal_direction_error_deg', 'temporal_path_jump_m', 'quality_flags',
    'trajectory_bc_eligible', 'trajectory_exclusion_reason',
]


def _float(value, default=math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _read_manifest(path: Path, splits: set[str]) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get('split') in splits
            and row.get('initial_bc_eligible') == '1'
            and row.get('sample_exists') == '1'
        ]
    rows.sort(key=lambda row: (
        row.get('session', ''), _integer(row.get('sample_id'))
    ))
    return rows


def _load_target(row: dict, spacing_m: float, target_count: int) -> dict:
    path = Path(row['sample_path'])
    with np.load(path, allow_pickle=False) as arrays:
        plan = np.asarray(arrays['nav2_plan_vehicle_xy'], dtype=np.float64)
        goal = np.asarray(arrays['goal_vehicle_xyz'], dtype=np.float64)
        pose = np.asarray(
            arrays['vehicle_pose_odom_xyzyaw'], dtype=np.float64
        )
        analysis = analyze_nav2_plan(
            plan,
            goal[:2],
            spacing_m=spacing_m,
            target_count=target_count,
        )
    output = dict(row)
    output.update({
        'analysis': analysis,
        'goal_xy': goal[:2].astype(np.float32),
        'vehicle_pose': pose,
        'temporal_path_jump_m': math.nan,
        'quality_flags': list(analysis.flags),
    })
    return output


def _temporal_diagnostics(samples: list[dict]) -> None:
    previous_by_session = {}
    for sample in samples:
        session = sample['session']
        previous = previous_by_session.get(session)
        if previous is not None:
            sample_gap = (
                _integer(sample['sample_id'])
                - _integer(previous['sample_id'])
            )
            if 0 < sample_gap <= 2:
                previous_analysis = previous['analysis']
                current_analysis = sample['analysis']
                previous_points = previous_analysis.target_points[
                    previous_analysis.target_mask
                ]
                current_points = current_analysis.target_points[
                    current_analysis.target_mask
                ]
                aligned_previous = transform_vehicle_points(
                    previous_points,
                    previous['vehicle_pose'],
                    sample['vehicle_pose'],
                )
                jump = median_nearest_distance(
                    current_points, aligned_previous
                )
                sample['temporal_path_jump_m'] = jump
                if math.isfinite(jump) and jump > 1.5:
                    sample['quality_flags'].append('temporal_path_jump')
        previous_by_session[session] = sample

    for sample in samples:
        eligible, reasons = initial_bc_trajectory_decision(
            sample['quality_flags']
        )
        sample['trajectory_bc_eligible'] = eligible
        sample['trajectory_exclusion_reason'] = list(reasons)


def _quality_row(sample: dict) -> dict:
    analysis = sample['analysis']
    goal_error = analysis.goal_direction_error_rad
    return {
        'session': sample['session'],
        'sample_id': sample['sample_id'],
        'split': sample['split'],
        'sample_path': sample['sample_path'],
        'speed_mps': sample.get('speed_mps', ''),
        'route_index': sample.get('route_index', ''),
        'route_size': sample.get('route_size', ''),
        'source_point_count': analysis.source_point_count,
        'forward_path_length_m': '{:.5f}'.format(
            analysis.forward_path_length_m
        ),
        'valid_target_count': int(np.count_nonzero(analysis.target_mask)),
        'vehicle_to_path_m': '{:.5f}'.format(analysis.vehicle_to_path_m),
        'maximum_source_gap_m': '{:.5f}'.format(
            analysis.maximum_source_gap_m
        ),
        'maximum_curvature_per_m': '{:.5f}'.format(
            analysis.maximum_curvature_per_m
        ),
        'absolute_heading_change_deg': '{:.3f}'.format(math.degrees(
            analysis.absolute_heading_change_rad
        )),
        'backward_target_fraction': '{:.5f}'.format(
            analysis.backward_target_fraction
        ),
        'goal_direction_error_deg': (
            '{:.3f}'.format(math.degrees(goal_error))
            if math.isfinite(goal_error) else ''
        ),
        'temporal_path_jump_m': (
            '{:.5f}'.format(sample['temporal_path_jump_m'])
            if math.isfinite(sample['temporal_path_jump_m']) else ''
        ),
        'quality_flags': ';'.join(sample['quality_flags']),
        'trajectory_bc_eligible': int(sample['trajectory_bc_eligible']),
        'trajectory_exclusion_reason': ';'.join(
            sample['trajectory_exclusion_reason']
        ),
    }


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_target_archive(
    path: Path,
    samples: list[dict],
    *,
    target_count: int,
    spacing_m: float,
) -> None:
    if not samples:
        return
    np.savez_compressed(
        path,
        sample_path=np.asarray([
            sample['sample_path'] for sample in samples
        ]),
        session=np.asarray([sample['session'] for sample in samples]),
        sample_id=np.asarray([
            _integer(sample['sample_id']) for sample in samples
        ], dtype=np.int32),
        speed_mps=np.asarray([
            _float(sample.get('speed_mps')) for sample in samples
        ], dtype=np.float32),
        target_points=np.stack([
            sample['analysis'].target_points for sample in samples
        ]).astype(np.float32),
        target_mask=np.stack([
            sample['analysis'].target_mask for sample in samples
        ]),
        goal_vehicle_xy=np.stack([
            sample['goal_xy'] for sample in samples
        ]).astype(np.float32),
        quality_flags=np.asarray([
            ';'.join(sample['quality_flags']) for sample in samples
        ]),
        target_count=np.asarray(target_count, dtype=np.int32),
        target_spacing_m=np.asarray(spacing_m, dtype=np.float32),
        trajectory_horizon_m=np.asarray(
            target_count * spacing_m, dtype=np.float32
        ),
    )


def _save_target_archives(
    samples: list[dict],
    output_directory: Path,
    target_count: int,
    spacing_m: float,
) -> dict[str, dict[str, int]]:
    by_split = defaultdict(list)
    for sample in samples:
        by_split[sample['split']].append(sample)
    counts = {}
    for split, split_samples in by_split.items():
        clean_samples = [
            sample for sample in split_samples
            if sample['trajectory_bc_eligible']
        ]
        _save_target_archive(
            output_directory / 'trajectory_targets_{}.npz'.format(split),
            split_samples,
            target_count=target_count,
            spacing_m=spacing_m,
        )
        _save_target_archive(
            output_directory
            / 'trajectory_targets_clean_{}.npz'.format(split),
            clean_samples,
            target_count=target_count,
            spacing_m=spacing_m,
        )
        counts[split] = {
            'all': len(split_samples),
            'clean': len(clean_samples),
            'excluded': len(split_samples) - len(clean_samples),
        }
    return counts


def _bev_geometry(sample_path: Path) -> dict:
    metadata_path = sample_path.parents[1] / 'metadata.json'
    fallback = {
        'x_min_m': -10.0,
        'x_max_m': 30.0,
        'y_min_m': -20.0,
        'y_max_m': 20.0,
    }
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        geometry = metadata.get('bev', {})
        return {
            key: float(geometry.get(key, value))
            for key, value in fallback.items()
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def _draw_sample(axis, sample: dict) -> None:
    path = Path(sample['sample_path'])
    with np.load(path, allow_pickle=False) as arrays:
        bev = np.asarray(arrays['lidar_bev'], dtype=np.float32)
        raw_path = np.asarray(
            arrays['nav2_plan_vehicle_xy'], dtype=np.float32
        )
    geometry = _bev_geometry(path)
    density = bev[1] if bev.shape[0] > 1 else bev[0]
    display = np.flip(density.T, axis=1)
    axis.imshow(
        display,
        extent=(
            geometry['x_min_m'], geometry['x_max_m'],
            geometry['y_min_m'], geometry['y_max_m'],
        ),
        origin='upper', cmap='Greys', vmin=0.0, vmax=1.0,
        interpolation='nearest', alpha=0.72,
    )
    if raw_path.ndim == 2 and raw_path.shape[1] >= 2:
        axis.plot(
            raw_path[:, 0], raw_path[:, 1], '--',
            color=TRAJECTORY_COLOR, linewidth=1.0, alpha=0.58,
            label='Raw Nav2 path',
        )
    analysis = sample['analysis']
    target = analysis.target_points[analysis.target_mask]
    if target.size:
        axis.plot(
            target[:, 0], target[:, 1], '-o', color=TARGET_COLOR,
            linewidth=1.4, markersize=3.0, label='Resampled target',
        )
        for index, point in enumerate(target, start=1):
            if index in {1, len(target)}:
                axis.text(
                    point[0], point[1] + 0.45, str(index),
                    color=TARGET_COLOR, fontsize=5.7, ha='center',
                )
    goal = sample['goal_xy']
    if np.isfinite(goal).all():
        axis.scatter(
            goal[0], goal[1], marker='*', s=65, color=GOAL_COLOR,
            edgecolors='white', linewidths=0.45, zorder=6,
            label='Relative goal',
        )
    axis.add_patch(Rectangle(
        (-2.65, -1.15), 5.20, 2.30, fill=False,
        edgecolor=VEHICLE_COLOR, linewidth=0.9, zorder=5,
    ))
    axis.scatter(0.0, 0.0, marker='^', s=24, color=VEHICLE_COLOR, zorder=6)
    axis.annotate(
        '', xy=(3.7, 0.0), xytext=(0.0, 0.0),
        arrowprops={
            'arrowstyle': '->', 'color': VEHICLE_COLOR, 'lw': 0.8,
        },
    )
    flags = sample['quality_flags'] or ['none']
    session_label = sample['session'].removeprefix('session_')
    title = '{} | {} | sample {}'.format(
        sample['split'], session_label, sample['sample_id']
    )
    axis.set_title(title, fontsize=7.0, pad=3)
    info = (
        'L={:.1f} m  targets={}/{}  kmax={:.2f}\nflags: {}'
    ).format(
        analysis.forward_path_length_m,
        int(np.count_nonzero(analysis.target_mask)),
        analysis.target_mask.size,
        analysis.maximum_curvature_per_m,
        ', '.join(flags),
    )
    axis.text(
        0.015, 0.015, info, transform=axis.transAxes,
        fontsize=5.8, va='bottom', ha='left',
        bbox={
            'facecolor': 'white', 'edgecolor': '#B0B0B0',
            'alpha': 0.82, 'pad': 2,
        },
    )
    axis.set_xlim(-5.0, 16.0)
    axis.set_ylim(-12.0, 12.0)
    axis.set_aspect('equal', adjustable='box')
    axis.grid(alpha=0.16, linewidth=0.4)
    axis.tick_params(labelsize=5.8)
    axis.set_xlabel('forward x (m)', fontsize=6.3)
    axis.set_ylabel('left y (m)', fontsize=6.3)


def _diverse_selection(
    samples: list[dict], count: int, *, rng: random.Random,
    ranked: bool,
) -> list[dict]:
    candidates = list(samples)
    if ranked:
        candidates.sort(key=lambda sample: (
            -len(sample['quality_flags']),
            -(
                sample['temporal_path_jump_m']
                if math.isfinite(sample['temporal_path_jump_m']) else -1.0
            ),
            -sample['analysis'].absolute_heading_change_rad,
            -sample['analysis'].maximum_curvature_per_m,
        ))
    else:
        rng.shuffle(candidates)
    selected = []
    selected_ids = defaultdict(list)
    for sample in candidates:
        sample_id = _integer(sample['sample_id'])
        if any(
            abs(sample_id - existing) < 15
            for existing in selected_ids[sample['session']]
        ):
            continue
        selected.append(sample)
        selected_ids[sample['session']].append(sample_id)
        if len(selected) >= count:
            break
    return selected


def _write_contact_sheet(
    samples: list[dict], output_stem: Path, *, dpi: int,
    page_size: int = 12,
) -> list[Path]:
    outputs = []
    if not samples:
        return outputs
    pdf_path = output_stem.with_suffix('.pdf')
    with PdfPages(pdf_path) as pdf:
        for page_number, start in enumerate(
            range(0, len(samples), page_size), start=1
        ):
            page = samples[start:start + page_size]
            columns = 3
            rows = int(math.ceil(len(page) / columns))
            figure, axes = plt.subplots(
                rows, columns, figsize=(12.0, 3.85 * rows), squeeze=False,
            )
            for axis, sample in zip(axes.ravel(), page):
                _draw_sample(axis, sample)
            for axis in axes.ravel()[len(page):]:
                axis.set_visible(False)
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            figure.legend(
                handles, labels, loc='lower center', ncol=3,
                frameon=False, fontsize=8,
            )
            figure.subplots_adjust(
                left=0.055, right=0.99, top=0.97, bottom=0.065,
                wspace=0.19, hspace=0.24,
            )
            png_path = output_stem.with_name(
                '{}_{:02d}.png'.format(output_stem.name, page_number)
            )
            figure.savefig(png_path, dpi=dpi, bbox_inches='tight')
            pdf.savefig(figure, bbox_inches='tight')
            outputs.append(png_path)
            plt.close(figure)
    outputs.append(pdf_path)
    return outputs


def _sequence_selection(samples: list[dict], length: int) -> list[dict]:
    if not samples or length <= 0:
        return []
    anchor = max(samples, key=lambda sample: (
        len(sample['quality_flags']),
        sample['temporal_path_jump_m']
        if math.isfinite(sample['temporal_path_jump_m']) else -1.0,
        sample['analysis'].absolute_heading_change_rad,
    ))
    session_samples = [
        sample for sample in samples if sample['session'] == anchor['session']
    ]
    session_samples.sort(key=lambda sample: _integer(sample['sample_id']))
    anchor_index = session_samples.index(anchor)
    start = max(0, anchor_index - length // 2)
    start = min(start, max(0, len(session_samples) - length))
    return session_samples[start:start + length]


def generate_visualizations(
    manifest: Path,
    output_directory: Path,
    *,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    spacing_m: float = 0.75,
    target_count: int = 12,
    random_count: int = 24,
    flagged_count: int = 24,
    sequence_length: int = 8,
    seed: int = 20260904,
    dpi: int = 220,
) -> dict:
    """Build diagnostics and contact sheets without changing raw data."""
    manifest = manifest.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest(manifest, set(splits))
    samples = [
        _load_target(row, spacing_m, target_count) for row in rows
    ]
    _temporal_diagnostics(samples)
    quality_rows = [_quality_row(sample) for sample in samples]
    _atomic_write_csv(
        output_directory / 'trajectory_quality.csv', quality_rows
    )
    target_archive_counts = _save_target_archives(
        samples, output_directory, target_count, spacing_m
    )

    rng = random.Random(seed)
    random_samples = _diverse_selection(
        samples, random_count, rng=rng, ranked=False
    )
    flagged_pool = [sample for sample in samples if sample['quality_flags']]
    flagged_samples = _diverse_selection(
        flagged_pool, flagged_count, rng=rng, ranked=True
    )
    sequence_samples = _sequence_selection(samples, sequence_length)

    outputs = []
    outputs.extend(_write_contact_sheet(
        random_samples, output_directory / 'random_contact_sheet', dpi=dpi
    ))
    outputs.extend(_write_contact_sheet(
        flagged_samples, output_directory / 'flagged_contact_sheet', dpi=dpi
    ))
    outputs.extend(_write_contact_sheet(
        sequence_samples, output_directory / 'sequence_contact_sheet',
        dpi=dpi, page_size=max(1, sequence_length),
    ))

    flag_counts = Counter()
    for sample in samples:
        flag_counts.update(sample['quality_flags'])
    summary = {
        'manifest': str(manifest),
        'output_directory': str(output_directory),
        'splits': list(splits),
        'spacing_m': spacing_m,
        'target_count': target_count,
        'trajectory_horizon_m': spacing_m * target_count,
        'samples_analyzed': len(samples),
        'samples_without_flags': sum(
            not sample['quality_flags'] for sample in samples
        ),
        'samples_with_flags': sum(
            bool(sample['quality_flags']) for sample in samples
        ),
        'initial_bc_exclusion_flags': sorted(INITIAL_BC_EXCLUSION_FLAGS),
        'target_archive_counts': target_archive_counts,
        'initial_bc_eligible_samples': sum(
            sample['trajectory_bc_eligible'] for sample in samples
        ),
        'initial_bc_excluded_samples': sum(
            not sample['trajectory_bc_eligible'] for sample in samples
        ),
        'flag_counts': dict(sorted(flag_counts.items())),
        'random_examples': len(random_samples),
        'flagged_examples': len(flagged_samples),
        'sequence_examples': len(sequence_samples),
        'generated_files': [str(path) for path in outputs],
        'important_note': (
            'Flags request human review; they do not automatically delete '
            'or relabel a training sample. Test split is excluded by default.'
        ),
    }
    summary_path = output_directory / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    guide = output_directory / 'HOW_TO_REVIEW.txt'
    guide.write_text(
        '1. random_contact_sheet_*.png에서 BEV, 차량, 상대 목표와 경로의 '
        '좌표가 맞는지 확인한다.\n'
        '2. flagged_contact_sheet_*.png에서 각 flag가 실제로 잘못된 '
        'teacher 경로인지, 정상적인 회피 또는 회전인지 판단한다.\n'
        '3. sequence_contact_sheet_*.png에서 연속 프레임의 경로 변화가 '
        '자연스러운지 확인한다.\n'
        '4. 원본 NPZ를 수정하거나 삭제하지 않는다. 이상한 그림은 제목의 '
        'session과 sample 번호를 기록해 품질 규칙을 재현 가능하게 '
        '수정한다.\n'
        '5. test split은 시각화하지 않았으며 모델 개발 중 그대로 '
        '보존한다.\n',
        encoding='utf-8',
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Visualize and audit fixed-horizon Nav2 teacher paths.'
    )
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--output-directory', type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        '--splits', nargs='+', choices=('train', 'validation', 'test'),
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument('--spacing-m', type=float, default=0.75)
    parser.add_argument('--target-count', type=int, default=12)
    parser.add_argument('--random-count', type=int, default=24)
    parser.add_argument('--flagged-count', type=int, default=24)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260904)
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args(argv)
    summary = generate_visualizations(
        args.manifest,
        args.output_directory,
        splits=tuple(args.splits),
        spacing_m=args.spacing_m,
        target_count=args.target_count,
        random_count=args.random_count,
        flagged_count=args.flagged_count,
        sequence_length=args.sequence_length,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
