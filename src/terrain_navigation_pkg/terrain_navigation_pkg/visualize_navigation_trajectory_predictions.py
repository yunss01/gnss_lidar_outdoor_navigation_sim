#!/usr/bin/env python3
"""Visualize offline trajectory predictions without loading PyTorch."""

from __future__ import annotations

import os
import sys

# The system Matplotlib requires the system NumPy. Model inference is exported
# to NPZ first so this plotting process does not need the user-site PyTorch.
if os.environ.get('PYTHONNOUSERSITE') != '1':
    environment = os.environ.copy()
    environment['PYTHONNOUSERSITE'] = '1'
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            '-m',
            'terrain_navigation_pkg.'
            'visualize_navigation_trajectory_predictions',
            *sys.argv[1:],
        ],
        environment,
    )

import argparse  # noqa: E402
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


DEFAULT_PREDICTIONS = Path(
    '/home/sukja/terrain_nav_data/learning/models/trajectory_baseline/v3/'
    'validation_predictions.npz'
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/models/trajectory_baseline/v3/'
    'prediction_visualizations'
)
RAW_PATH_COLOR = '#4A72A8'
TEACHER_COLOR = '#C1633A'
PREDICTION_COLOR = '#2A9D8F'
GOAL_COLOR = '#7B3294'
VEHICLE_COLOR = '#202020'


def _bev_geometry(sample_path: Path) -> dict:
    fallback = {
        'x_min_m': -10.0,
        'x_max_m': 30.0,
        'y_min_m': -20.0,
        'y_max_m': 20.0,
    }
    metadata_path = sample_path.parents[1] / 'metadata.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        geometry = metadata.get('bev', {})
        return {
            key: float(geometry.get(key, value))
            for key, value in fallback.items()
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def _load_predictions(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            'sample_path', 'session', 'sample_id', 'goal_vehicle_xy',
            'teacher_points_m', 'target_mask', 'predicted_points_m',
            'mean_point_error_m', 'endpoint_error_m',
        }
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise ValueError(
                'prediction archive is missing: {}'.format(
                    ', '.join(missing)
                )
            )
        count = len(arrays['sample_id'])
        samples = []
        for index in range(count):
            samples.append({
                'sample_path': str(arrays['sample_path'][index]),
                'session': str(arrays['session'][index]),
                'sample_id': int(arrays['sample_id'][index]),
                'goal_xy': np.asarray(
                    arrays['goal_vehicle_xy'][index], dtype=np.float32
                ),
                'teacher': np.asarray(
                    arrays['teacher_points_m'][index], dtype=np.float32
                ),
                'mask': np.asarray(
                    arrays['target_mask'][index], dtype=bool
                ),
                'prediction': np.asarray(
                    arrays['predicted_points_m'][index], dtype=np.float32
                ),
                'mean_error_m': float(
                    arrays['mean_point_error_m'][index]
                ),
                'endpoint_error_m': float(
                    arrays['endpoint_error_m'][index]
                ),
            })
    return samples


def _draw_sample(axis, sample: dict) -> None:
    sample_path = Path(sample['sample_path'])
    with np.load(sample_path, allow_pickle=False) as arrays:
        bev = np.asarray(arrays['lidar_bev'], dtype=np.float32)
        raw_path = np.asarray(
            arrays['nav2_plan_vehicle_xy'], dtype=np.float32
        )
    geometry = _bev_geometry(sample_path)
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
            color=RAW_PATH_COLOR, linewidth=1.0, alpha=0.48,
            label='Raw Nav2 path',
        )
    mask = sample['mask']
    teacher = sample['teacher'][mask]
    prediction = sample['prediction'][mask]
    axis.plot(
        teacher[:, 0], teacher[:, 1], '-o', color=TEACHER_COLOR,
        linewidth=1.5, markersize=2.8, label='Teacher target',
    )
    axis.plot(
        prediction[:, 0], prediction[:, 1], '-o',
        color=PREDICTION_COLOR, linewidth=1.5, markersize=2.8,
        label='Model prediction',
    )
    goal = sample['goal_xy']
    axis.scatter(
        goal[0], goal[1], marker='*', s=65, color=GOAL_COLOR,
        edgecolors='white', linewidths=0.45, zorder=6,
        label='Relative goal',
    )
    axis.add_patch(Rectangle(
        (-2.65, -1.15), 5.20, 2.30, fill=False,
        edgecolor=VEHICLE_COLOR, linewidth=0.9, zorder=5,
    ))
    axis.scatter(
        0.0, 0.0, marker='^', s=24, color=VEHICLE_COLOR, zorder=6
    )
    axis.annotate(
        '', xy=(3.7, 0.0), xytext=(0.0, 0.0),
        arrowprops={
            'arrowstyle': '->', 'color': VEHICLE_COLOR, 'lw': 0.8,
        },
    )
    session = sample['session'].removeprefix('session_')
    axis.set_title(
        '{} | sample {}'.format(session, sample['sample_id']),
        fontsize=7.0, pad=3,
    )
    info = 'mean={:.2f} m  end={:.2f} m  targets={}'.format(
        sample['mean_error_m'], sample['endpoint_error_m'],
        int(mask.sum()),
    )
    axis.text(
        0.015, 0.015, info, transform=axis.transAxes,
        fontsize=6.0, va='bottom', ha='left',
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


def _diverse_selection(samples, count: int, *, seed: int, worst: bool):
    candidates = list(samples)
    if worst:
        candidates.sort(
            key=lambda item: (
                item['mean_error_m'] + 0.5 * item['endpoint_error_m']
            ),
            reverse=True,
        )
    else:
        random.Random(seed).shuffle(candidates)
    selected = []
    selected_ids = {}
    for sample in candidates:
        ids = selected_ids.setdefault(sample['session'], [])
        if any(abs(sample['sample_id'] - item) < 15 for item in ids):
            continue
        selected.append(sample)
        ids.append(sample['sample_id'])
        if len(selected) >= count:
            break
    return selected


def _write_contact_sheet(samples, output_stem: Path, *, dpi: int) -> list:
    outputs = []
    page_size = 12
    pdf_path = output_stem.with_suffix('.pdf')
    with PdfPages(pdf_path) as pdf:
        for page_number, start in enumerate(
            range(0, len(samples), page_size), start=1
        ):
            page = samples[start:start + page_size]
            rows = int(math.ceil(len(page) / 3))
            figure, axes = plt.subplots(
                rows, 3, figsize=(12.0, 3.85 * rows), squeeze=False
            )
            for axis, sample in zip(axes.ravel(), page):
                _draw_sample(axis, sample)
            for axis in axes.ravel()[len(page):]:
                axis.set_visible(False)
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            figure.legend(
                handles, labels, loc='lower center', ncol=4,
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


def generate(args) -> dict:
    predictions = args.predictions.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    samples = _load_predictions(predictions)
    random_samples = _diverse_selection(
        samples, args.random_count, seed=args.seed, worst=False
    )
    worst_samples = _diverse_selection(
        samples, args.worst_count, seed=args.seed, worst=True
    )
    outputs = []
    outputs.extend(_write_contact_sheet(
        random_samples, output_directory / 'prediction_random', dpi=args.dpi
    ))
    outputs.extend(_write_contact_sheet(
        worst_samples, output_directory / 'prediction_worst', dpi=args.dpi
    ))
    summary = {
        'predictions': str(predictions),
        'samples': len(samples),
        'random_examples': len(random_samples),
        'worst_examples': len(worst_samples),
        'generated_files': [str(path) for path in outputs],
    }
    (output_directory / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='Visualize teacher and predicted local trajectories.'
    )
    parser.add_argument('--predictions', type=Path,
                        default=DEFAULT_PREDICTIONS)
    parser.add_argument('--output-directory', type=Path,
                        default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument('--random-count', type=int, default=24)
    parser.add_argument('--worst-count', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
