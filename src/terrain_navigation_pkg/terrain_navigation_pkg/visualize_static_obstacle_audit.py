#!/usr/bin/env python3
"""Render spatial diagnostics for one static-obstacle audit snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(
    '/home/sukja/terrain_nav_data/learning/static_obstacle_audits'
)

UNKNOWN_COLOR = (250, 250, 250)
FREE_COLOR = (237, 242, 238)
TARGET_OBSTACLE_COLOR = (95, 95, 95)
BASELINE_COLOR = (74, 114, 168)
CANDIDATE_COLOR = (92, 138, 92)
ADDED_OBSTACLE_COLOR = (45, 125, 70)
ADDED_FREE_COLOR = (196, 45, 115)
ADDED_UNKNOWN_COLOR = (230, 145, 55)
REMOVED_OBSTACLE_COLOR = (205, 45, 45)
REMOVED_FREE_COLOR = (45, 145, 205)
REMOVED_UNKNOWN_COLOR = (135, 90, 170)
GRID_COLOR = (170, 170, 170)
CORRIDOR_COLOR = (0, 145, 175)
VEHICLE_COLOR = (20, 20, 20)


def _font(size: int, bold: bool = False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _latest_audit(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError('no static audit directory below ' + str(root))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _geometry(metadata: dict, shape: tuple[int, int]) -> dict:
    value = dict(metadata.get('bev_geometry', {}))
    value.setdefault('x_min_m', -10.0)
    value.setdefault('x_max_m', 30.0)
    value.setdefault('y_min_m', -20.0)
    value.setdefault('y_max_m', 20.0)
    value.setdefault(
        'resolution_m',
        (float(value['x_max_m']) - float(value['x_min_m'])) / shape[0],
    )
    return {key: float(item) for key, item in value.items()}


def _cell_coordinates(shape: tuple[int, int], geometry: dict):
    rows = np.arange(shape[0], dtype=np.float32)[:, None]
    columns = np.arange(shape[1], dtype=np.float32)[None, :]
    resolution = geometry['resolution_m']
    forward = geometry['x_max_m'] - (rows + 0.5) * resolution
    left = geometry['y_max_m'] - (columns + 0.5) * resolution
    return (
        np.broadcast_to(forward, shape),
        np.broadcast_to(left, shape),
    )


def compute_spatial_diagnostics(
    target_labels,
    baseline_occupied,
    candidate_occupied,
    geometry,
):
    """Summarize snapshot changes by radial range and planning corridor."""
    target = np.asarray(target_labels, dtype=np.int8)
    baseline = np.asarray(baseline_occupied, dtype=bool)
    candidate = np.asarray(candidate_occupied, dtype=bool)
    if baseline.shape != target.shape or candidate.shape != target.shape:
        raise ValueError('target and occupancy grids must have equal shape')
    forward, left = _cell_coordinates(target.shape, geometry)
    radial = np.hypot(forward, left)

    def summarize(mask):
        added = mask & candidate & ~baseline
        removed = mask & baseline & ~candidate
        baseline_count = int(np.count_nonzero(mask & baseline))
        candidate_count = int(np.count_nonzero(mask & candidate))
        return {
            'baseline_occupied': baseline_count,
            'candidate_occupied': candidate_count,
            'candidate_to_baseline_ratio': (
                float(candidate_count / baseline_count)
                if baseline_count else None
            ),
            'added_total': int(np.count_nonzero(added)),
            'added_target_obstacle': int(np.count_nonzero(
                added & (target == 1)
            )),
            'added_target_free': int(np.count_nonzero(
                added & (target == 0)
            )),
            'added_unknown': int(np.count_nonzero(added & (target < 0))),
            'removed_total': int(np.count_nonzero(removed)),
            'removed_target_obstacle': int(np.count_nonzero(
                removed & (target == 1)
            )),
            'removed_target_free': int(np.count_nonzero(
                removed & (target == 0)
            )),
            'removed_unknown': int(np.count_nonzero(
                removed & (target < 0)
            )),
        }

    radial_bins = []
    for lower, upper in (
        (0, 5), (5, 10), (10, 20), (20, 30), (30, None)
    ):
        mask = radial >= lower
        if upper is not None:
            mask &= radial < upper
        values = summarize(mask)
        values.update({'minimum_m': lower, 'maximum_m': upper})
        radial_bins.append(values)
    corridor = (
        (forward >= 0.0)
        & (forward <= 15.0)
        & (np.abs(left) <= 3.0)
    )
    near_field = radial < 20.0
    return {
        'radial_bins': radial_bins,
        'planning_corridor_x_0_15_y_plus_minus_3': summarize(corridor),
        'near_field_radial_below_20_m': summarize(near_field),
        'far_field_radial_20_m_or_more': summarize(~near_field),
        'whole_bev': summarize(np.ones(target.shape, dtype=bool)),
    }


def compute_v2_model_diagnostics(arrays):
    """Summarize exact model and geometry values saved by audit schema v2."""
    required = {
        'target_labels',
        'baseline_occupied',
        'candidate_occupied',
        'raw_lidar_bev',
        'model_obstacle_probability',
        'model_predictive_entropy',
        'model_mc_variance',
        'hard_obstacle_mask',
        'selective_decision',
    }
    if not required.issubset(arrays):
        return None
    target = np.asarray(arrays['target_labels'], dtype=np.int8)
    baseline = np.asarray(arrays['baseline_occupied'], dtype=bool)
    candidate = np.asarray(arrays['candidate_occupied'], dtype=bool)
    raw_bev = np.asarray(arrays['raw_lidar_bev'], dtype=np.float32)
    probability = np.asarray(
        arrays['model_obstacle_probability'], dtype=np.float32
    )
    entropy = np.asarray(
        arrays['model_predictive_entropy'], dtype=np.float32
    )
    variance = np.asarray(arrays['model_mc_variance'], dtype=np.float32)
    hard_mask = np.asarray(arrays['hard_obstacle_mask'], dtype=bool)
    decision = np.asarray(arrays['selective_decision'], dtype=np.int8)
    if any(
        value.shape != target.shape
        for value in (probability, entropy, variance, hard_mask, decision)
    ) or raw_bev.shape != (4,) + target.shape:
        return None

    masks = {
        'removed_target_obstacle': (
            (target == 1) & baseline & ~candidate
        ),
        'retained_baseline_target_obstacle': (
            (target == 1) & baseline & candidate
        ),
    }

    def numeric(values, mask):
        selected = np.asarray(values)[mask]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            return {'count': 0, 'minimum': None, 'mean': None, 'maximum': None}
        return {
            'count': int(selected.size),
            'minimum': float(np.min(selected)),
            'mean': float(np.mean(selected)),
            'maximum': float(np.max(selected)),
        }

    result = {}
    for name, mask in masks.items():
        result[name] = {
            'cells': int(np.count_nonzero(mask)),
            'obstacle_probability': numeric(probability, mask),
            'predictive_entropy': numeric(entropy, mask),
            'mc_variance': numeric(variance, mask),
            'maximum_height_m': numeric(raw_bev[2], mask),
            'vertical_span_m': numeric(raw_bev[3], mask),
            'hard_obstacle_cells': int(np.count_nonzero(
                hard_mask & mask
            )),
            'free_decision_cells': int(np.count_nonzero(
                (decision == 0) & mask
            )),
            'obstacle_decision_cells': int(np.count_nonzero(
                (decision == 100) & mask
            )),
            'unknown_decision_cells': int(np.count_nonzero(
                (decision < 0) & mask
            )),
        }
    if 'instance_json' in arrays:
        result['instances'] = json.loads(str(arrays['instance_json'].item()))
    return result


def _base_image(target: np.ndarray) -> np.ndarray:
    image = np.empty(target.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[target == 0] = FREE_COLOR
    image[target == 1] = TARGET_OBSTACLE_COLOR
    return image


def _occupancy_image(target, occupied, color) -> np.ndarray:
    image = _base_image(target)
    image[np.asarray(occupied, dtype=bool)] = color
    return image


def _change_image(target, baseline, candidate) -> np.ndarray:
    image = _base_image(target)
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    added = candidate & ~baseline
    removed = baseline & ~candidate
    image[added & (target == 1)] = ADDED_OBSTACLE_COLOR
    image[added & (target == 0)] = ADDED_FREE_COLOR
    image[added & (target < 0)] = ADDED_UNKNOWN_COLOR
    image[removed & (target == 1)] = REMOVED_OBSTACLE_COLOR
    image[removed & (target == 0)] = REMOVED_FREE_COLOR
    image[removed & (target < 0)] = REMOVED_UNKNOWN_COLOR
    return image


def _cell_pixel(forward_m, left_m, geometry, scale):
    row = (geometry['x_max_m'] - forward_m) / geometry['resolution_m']
    column = (geometry['y_max_m'] - left_m) / geometry['resolution_m']
    return int(round(column * scale)), int(round(row * scale))


def _overlay_geometry(image: Image.Image, geometry: dict, scale: int):
    draw = ImageDraw.Draw(image)
    origin = _cell_pixel(0.0, 0.0, geometry, scale)
    for radius_m in (5.0, 10.0, 20.0, 30.0):
        radius = int(round(
            radius_m / geometry['resolution_m'] * scale
        ))
        draw.ellipse(
            (
                origin[0] - radius, origin[1] - radius,
                origin[0] + radius, origin[1] + radius,
            ),
            outline=GRID_COLOR,
            width=max(1, scale // 2),
        )
    corridor_top_left = _cell_pixel(15.0, 3.0, geometry, scale)
    corridor_bottom_right = _cell_pixel(0.0, -3.0, geometry, scale)
    draw.rectangle(
        (*corridor_top_left, *corridor_bottom_right),
        outline=CORRIDOR_COLOR,
        width=max(2, scale),
    )
    size = max(8, 3 * scale)
    draw.polygon(
        [
            (origin[0], origin[1] - 2 * size),
            (origin[0] - size, origin[1] + size),
            (origin[0] + size, origin[1] + size),
        ],
        fill=VEHICLE_COLOR,
        outline=(255, 255, 255),
    )


def _panel(source, title, geometry, scale):
    height, width = source.shape[:2]
    title_height = 54
    panel = Image.new(
        'RGB', (width * scale, height * scale + title_height), 'white'
    )
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(20, 20, 20), font=_font(28, True))
    grid = Image.fromarray(source).resize(
        (width * scale, height * scale), Image.Resampling.NEAREST
    )
    _overlay_geometry(grid, geometry, scale)
    panel.paste(grid, (0, title_height))
    return panel


def _format_ratio(value):
    return '--' if value is None else '{:.2f}'.format(value)


def render_diagnostic(
    target,
    baseline,
    candidate,
    geometry,
    diagnostics,
    title,
    *,
    scale=4,
):
    """Render a three-panel, 300-DPI diagnostic image."""
    panels = [
        _panel(
            _occupancy_image(target, baseline, BASELINE_COLOR),
            'Baseline obstacle cells', geometry, scale,
        ),
        _panel(
            _occupancy_image(target, candidate, CANDIDATE_COLOR),
            'AI-fused candidate cells', geometry, scale,
        ),
        _panel(
            _change_image(target, baseline, candidate),
            'Candidate changes from baseline', geometry, scale,
        ),
    ]
    gap = 28
    margin = 36
    heading_height = 105
    footer_height = 385
    width = 2 * margin + sum(panel.width for panel in panels) + 2 * gap
    height = heading_height + panels[0].height + footer_height
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, fill=(15, 15, 15), font=_font(38, True))
    draw.text(
        (margin, 65),
        'Top = vehicle forward; cyan box = x 0-15 m, |y| <= 3 m corridor',
        fill=(55, 55, 55),
        font=_font(20),
    )
    x_offset = margin
    for panel in panels:
        canvas.paste(panel, (x_offset, heading_height))
        x_offset += panel.width + gap

    footer_y = heading_height + panels[0].height + 22
    legend = [
        ('Semantic obstacle', TARGET_OBSTACLE_COLOR),
        ('Added true obstacle', ADDED_OBSTACLE_COLOR),
        ('Added known-free (ghost)', ADDED_FREE_COLOR),
        ('Added unknown', ADDED_UNKNOWN_COLOR),
        ('Removed true obstacle', REMOVED_OBSTACLE_COLOR),
        ('Removed known-free', REMOVED_FREE_COLOR),
    ]
    legend_column_width = (width - 2 * margin) // 3
    for index, (label, color) in enumerate(legend):
        row = index // 3
        column = index % 3
        lx = margin + column * legend_column_width
        ly = footer_y + row * 32
        draw.rectangle((lx, ly, lx + 22, ly + 22), fill=color)
        draw.text(
            (lx + 30, ly - 2), label, fill=(30, 30, 30),
            font=_font(17),
        )

    table_y = footer_y + 82
    headers = [
        'Range', 'Baseline', 'Candidate', 'Ratio', 'Added',
        '+true', '+free', '+unknown', '-true', '-free',
    ]
    column_widths = [150, 130, 140, 100, 105, 100, 100, 125, 100]
    positions = [margin]
    for column_width in column_widths:
        positions.append(positions[-1] + column_width)
    for position, header in zip(positions, headers):
        draw.text(
            (position, table_y), header, fill=(20, 20, 20),
            font=_font(19, True),
        )
    table_y += 34
    table_rows = []
    for values in diagnostics['radial_bins']:
        upper = values['maximum_m']
        label = '{}-{} m'.format(
            values['minimum_m'], upper if upper is not None else 'max'
        )
        table_rows.append((label, values))
    table_rows.extend([
        ('Corridor', diagnostics[
            'planning_corridor_x_0_15_y_plus_minus_3'
        ]),
        ('Whole BEV', diagnostics['whole_bev']),
    ])
    for label, values in table_rows:
        cells = [
            label,
            values['baseline_occupied'],
            values['candidate_occupied'],
            _format_ratio(values['candidate_to_baseline_ratio']),
            values['added_total'],
            values['added_target_obstacle'],
            values['added_target_free'],
            values['added_unknown'],
            values['removed_target_obstacle'],
            values['removed_target_free'],
        ]
        for position, value in zip(positions, cells):
            draw.text(
                (position, table_y), str(value), fill=(35, 35, 35),
                font=_font(18),
            )
        table_y += 31
    return canvas


def generate_diagnostic(
    audit_directory,
    *,
    snapshot='representative',
    output_directory=None,
    dpi=300,
):
    audit = Path(audit_directory).expanduser().resolve()
    metadata = json.loads((audit / 'metadata.json').read_text())
    summary = json.loads((audit / 'summary.json').read_text())
    snapshot_path = audit / 'snapshots' / (snapshot + '.npz')
    if not snapshot_path.is_file():
        raise FileNotFoundError('snapshot not found: ' + str(snapshot_path))
    with np.load(snapshot_path, allow_pickle=False) as arrays:
        target = np.asarray(arrays['target_labels'], dtype=np.int8)
        baseline = np.asarray(arrays['baseline_occupied'], dtype=bool)
        candidate = np.asarray(arrays['candidate_occupied'], dtype=bool)
        frame = json.loads(str(arrays['frame_json'].item()))
        v2_diagnostics = compute_v2_model_diagnostics(arrays)
    geometry = _geometry(metadata, target.shape)
    diagnostics = compute_spatial_diagnostics(
        target, baseline, candidate, geometry
    )
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory
        else audit / 'diagnostics'
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = '{}_spatial_diagnostic'.format(snapshot)
    image_path = output / (stem + '.png')
    title = '{} | {} | frame {}'.format(
        summary.get('scenario_label', audit.name), snapshot,
        frame.get('frame_index', '?'),
    )
    image = render_diagnostic(
        target, baseline, candidate, geometry, diagnostics, title
    )
    image.save(image_path, dpi=(dpi, dpi), optimize=True)
    result = {
        'audit_directory': str(audit),
        'snapshot': snapshot,
        'snapshot_path': str(snapshot_path),
        'output_image': str(image_path),
        'dpi': int(dpi),
        'frame': frame,
        'geometry': geometry,
        'spatial_diagnostics': diagnostics,
        'v2_model_diagnostics': v2_diagnostics,
        'audit_summary': summary,
        'interpretation_limit': (
            'Added unknown cells have no synchronized semantic cell label in '
            'this snapshot. Their physical class cannot be proven from this '
            'NPZ alone.'
        ),
    }
    summary_path = output / (stem + '.json')
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Visualize one static obstacle audit snapshot.'
    )
    parser.add_argument(
        '--audit-directory', type=Path, default=None,
        help='Audit directory; defaults to the newest audit.',
    )
    parser.add_argument(
        '--audit-root', type=Path, default=DEFAULT_ROOT,
        help='Root used when --audit-directory is omitted.',
    )
    parser.add_argument(
        '--snapshot', default='representative',
        help='Snapshot basename below snapshots/ without .npz.',
    )
    parser.add_argument('--output-directory', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    audit = (
        args.audit_directory.expanduser()
        if args.audit_directory
        else _latest_audit(args.audit_root.expanduser())
    )
    result = generate_diagnostic(
        audit,
        snapshot=args.snapshot,
        output_directory=args.output_directory,
        dpi=args.dpi,
    )
    print('Saved image: ' + result['output_image'])
    print('Saved JSON: ' + str(
        Path(result['output_image']).with_suffix('.json')
    ))


if __name__ == '__main__':
    main()
