#!/usr/bin/env python3
"""Audit and visualize independent v2 traversability evidence targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MANIFEST = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_motorhelmet_pilot_20260918/manifest.csv'
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'visualizations/v2_motorhelmet_pilot_20260918'
)

UNKNOWN_COLOR = (29, 33, 40)
PASSABLE_COLOR = (76, 150, 92)
OBSTACLE_COLOR = (210, 75, 58)
AMBIGUOUS_COLOR = (230, 181, 64)
CONTROLLED_OBSTACLE_COLOR = (238, 42, 183)
CONTROLLED_PASSABLE_COLOR = (37, 200, 197)
CONTROLLED_AMBIGUOUS_COLOR = (150, 92, 210)
VISIBILITY_COLOR = (58, 113, 182)
EGO_COLOR = (55, 55, 55)

REQUIRED_ARRAYS = {
    'lidar_bev',
    'local_ground_relative_max_height_m',
    'target_passable_surface_mask',
    'target_obstacle_evidence_mask',
    'target_ambiguous_observed_mask',
    'target_controlled_passable_point_count',
    'target_controlled_obstacle_point_count',
    'target_controlled_ambiguous_point_count',
    'target_obstacle_instance_id',
    'target_ego_exclusion_mask',
    'visibility_free_mask',
    'controlled_actor_ids',
    'controlled_actor_disposition_code',
}

AUDIT_FIELDS = [
    'source_session', 'source_sample_id', 'derived_sample_path',
    'controlled_actor_ids', 'controlled_return_count_manifest',
    'controlled_return_count_derived', 'passable_cell_count',
    'obstacle_cell_count', 'ambiguous_cell_count',
    'controlled_passable_cell_count', 'controlled_obstacle_cell_count',
    'controlled_ambiguous_cell_count', 'instance_ids_in_obstacle_target',
    'controlled_obstacle_cells_rc', 'counterfactual_control_session',
    'controlled_cells_obstacle_in_control', 'counterfactual_evaluated',
    'audit_passed', 'violations',
]


def _integer(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _font(size: int, bold: bool = False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def read_written_rows(manifest: Path) -> list[dict]:
    """Read written samples in a deterministic session/sample order."""

    path = Path(manifest).expanduser().resolve()
    with path.open(newline='', encoding='utf-8') as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get('status') == 'written'
            and row.get('derived_sample_path')
        ]
    rows.sort(key=lambda row: (
        row.get('source_session', ''),
        _integer(row.get('source_sample_id')),
    ))
    return rows


def _bool_array(arrays, name: str, shape=None) -> np.ndarray:
    value = np.asarray(arrays[name], dtype=bool)
    if shape is not None and value.shape != shape:
        raise ValueError(f'{name} shape {value.shape} != {shape}')
    return value


def audit_sample(row: dict, control_row: dict | None = None) -> dict:
    """Check target invariants and controlled-instance provenance."""

    path = Path(row['derived_sample_path'])
    violations: list[str] = []
    with np.load(path, allow_pickle=False) as arrays:
        missing = sorted(REQUIRED_ARRAYS.difference(arrays.files))
        if missing:
            raise ValueError(f'{path} missing {", ".join(missing)}')
        passable = _bool_array(arrays, 'target_passable_surface_mask')
        shape = passable.shape
        obstacle = _bool_array(
            arrays, 'target_obstacle_evidence_mask', shape
        )
        ambiguous = _bool_array(
            arrays, 'target_ambiguous_observed_mask', shape
        )
        controlled_passable_count = np.asarray(
            arrays['target_controlled_passable_point_count']
        )
        controlled_obstacle_count = np.asarray(
            arrays['target_controlled_obstacle_point_count']
        )
        controlled_ambiguous_count = np.asarray(
            arrays['target_controlled_ambiguous_point_count']
        )
        for name, value in (
            ('target_controlled_passable_point_count', controlled_passable_count),
            ('target_controlled_obstacle_point_count', controlled_obstacle_count),
            ('target_controlled_ambiguous_point_count', controlled_ambiguous_count),
        ):
            if value.shape != shape:
                raise ValueError(f'{name} shape {value.shape} != {shape}')
        controlled_passable = controlled_passable_count > 0
        controlled_obstacle = controlled_obstacle_count > 0
        controlled_ambiguous = controlled_ambiguous_count > 0
        controlled_obstacle_cells = [
            [int(row_index), int(column_index)]
            for row_index, column_index in np.argwhere(controlled_obstacle)
        ]
        instance_ids = np.asarray(arrays['target_obstacle_instance_id'])
        if instance_ids.shape != shape:
            raise ValueError('target_obstacle_instance_id shape mismatch')
        actor_ids = np.asarray(
            arrays['controlled_actor_ids'], dtype=np.int64
        ).reshape(-1)
        disposition_codes = np.asarray(
            arrays['controlled_actor_disposition_code'], dtype=np.int8
        ).reshape(-1)
        if actor_ids.shape != disposition_codes.shape:
            raise ValueError('controlled actor ID/disposition shape mismatch')

        if np.any(passable & obstacle):
            violations.append('passable_obstacle_overlap')
        if np.any(ambiguous & (passable | obstacle)):
            violations.append('ambiguous_positive_overlap')
        if np.any(controlled_obstacle & ~obstacle):
            violations.append('controlled_obstacle_not_in_obstacle_target')
        if np.any(controlled_passable & ~passable):
            violations.append('controlled_passable_not_in_passable_target')
        if np.any(controlled_ambiguous & ~ambiguous):
            violations.append('controlled_ambiguous_not_in_ambiguous_target')

        controlled_return_count = int(
            controlled_passable_count.sum()
            + controlled_obstacle_count.sum()
            + controlled_ambiguous_count.sum()
        )
        manifest_returns = _integer(
            row.get('controlled_actor_return_count'), -1
        )
        if manifest_returns != controlled_return_count:
            violations.append('controlled_return_count_mismatch')

        obstacle_actor_ids = actor_ids[disposition_codes == 1]
        present_instance_ids = sorted(
            int(value) for value in np.unique(instance_ids[obstacle])
            if int(value) >= 0
        )
        if controlled_obstacle.any():
            for actor_id in obstacle_actor_ids:
                if int(actor_id) not in present_instance_ids:
                    violations.append(
                        f'controlled_actor_{int(actor_id)}_missing_instance_target'
                    )

    control_session = ''
    controlled_cells_obstacle_in_control = 0
    counterfactual_evaluated = False
    if control_row is not None and controlled_obstacle_cells:
        control_path = Path(control_row['derived_sample_path'])
        control_session = control_row.get('source_session', '')
        with np.load(control_path, allow_pickle=False) as control_arrays:
            control_obstacle = _bool_array(
                control_arrays, 'target_obstacle_evidence_mask', shape
            )
            controlled_cells_obstacle_in_control = sum(
                bool(control_obstacle[row_index, column_index])
                for row_index, column_index in controlled_obstacle_cells
            )
        counterfactual_evaluated = True
        if controlled_cells_obstacle_in_control:
            violations.append('controlled_cell_already_obstacle_in_control')

    return {
        'source_session': row.get('source_session', ''),
        'source_sample_id': _integer(row.get('source_sample_id'), -1),
        'derived_sample_path': str(path),
        'controlled_actor_ids': json.dumps(actor_ids.tolist()),
        'controlled_return_count_manifest': manifest_returns,
        'controlled_return_count_derived': controlled_return_count,
        'passable_cell_count': int(np.count_nonzero(passable)),
        'obstacle_cell_count': int(np.count_nonzero(obstacle)),
        'ambiguous_cell_count': int(np.count_nonzero(ambiguous)),
        'controlled_passable_cell_count': int(np.count_nonzero(
            controlled_passable
        )),
        'controlled_obstacle_cell_count': int(np.count_nonzero(
            controlled_obstacle
        )),
        'controlled_ambiguous_cell_count': int(np.count_nonzero(
            controlled_ambiguous
        )),
        'instance_ids_in_obstacle_target': json.dumps(present_instance_ids),
        'controlled_obstacle_cells_rc': json.dumps(
            controlled_obstacle_cells
        ),
        'counterfactual_control_session': control_session,
        'controlled_cells_obstacle_in_control': (
            controlled_cells_obstacle_in_control
        ),
        'counterfactual_evaluated': counterfactual_evaluated,
        'audit_passed': not violations,
        'violations': ';'.join(violations),
    }


def _raw_image(bev: np.ndarray) -> np.ndarray:
    density = np.asarray(bev[1] if bev.shape[0] > 1 else bev[0])
    level = np.clip(density, 0.0, 1.0)
    value = (246.0 - 218.0 * level).astype(np.uint8)
    return np.repeat(value[:, :, None], 3, axis=2)


def _target_image(arrays) -> np.ndarray:
    passable = _bool_array(arrays, 'target_passable_surface_mask')
    obstacle = _bool_array(arrays, 'target_obstacle_evidence_mask')
    ambiguous = _bool_array(arrays, 'target_ambiguous_observed_mask')
    ego = _bool_array(arrays, 'target_ego_exclusion_mask')
    image = np.empty(passable.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[passable] = PASSABLE_COLOR
    image[ambiguous] = AMBIGUOUS_COLOR
    image[obstacle] = OBSTACLE_COLOR
    image[ego] = EGO_COLOR
    return image


def _controlled_image(arrays) -> np.ndarray:
    occupied = np.asarray(arrays['lidar_bev'][0]) > 0
    ego = _bool_array(arrays, 'target_ego_exclusion_mask')
    controlled_passable = (
        np.asarray(arrays['target_controlled_passable_point_count']) > 0
    )
    controlled_obstacle = (
        np.asarray(arrays['target_controlled_obstacle_point_count']) > 0
    )
    controlled_ambiguous = (
        np.asarray(arrays['target_controlled_ambiguous_point_count']) > 0
    )
    image = np.empty(occupied.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[occupied] = (125, 130, 137)
    image[controlled_passable] = CONTROLLED_PASSABLE_COLOR
    image[controlled_ambiguous] = CONTROLLED_AMBIGUOUS_COLOR
    image[controlled_obstacle] = CONTROLLED_OBSTACLE_COLOR
    image[ego] = EGO_COLOR
    return image


def _support_image(arrays) -> np.ndarray:
    relative = np.asarray(
        arrays['local_ground_relative_max_height_m'], dtype=np.float32
    )
    if relative.ndim == 3:
        relative = relative[0]
    level = np.clip(relative / 0.50, 0.0, 1.0)
    image = np.zeros(relative.shape + (3,), dtype=np.uint8)
    image[..., 0] = (255.0 * level).astype(np.uint8)
    image[..., 1] = (160.0 * np.sqrt(level)).astype(np.uint8)
    image[..., 2] = (45.0 * (1.0 - level)).astype(np.uint8)
    image[_bool_array(arrays, 'visibility_free_mask')] = VISIBILITY_COLOR
    image[_bool_array(arrays, 'target_ego_exclusion_mask')] = EGO_COLOR
    return image


def _vehicle_marker(image: Image.Image, scale: int) -> None:
    draw = ImageDraw.Draw(image)
    center_x = image.width // 2
    origin_y = int(round(0.75 * image.height))
    radius = max(3, scale + 1)
    draw.polygon([
        (center_x, origin_y - 3 * radius),
        (center_x - 2 * radius, origin_y + 2 * radius),
        (center_x + 2 * radius, origin_y + 2 * radius),
    ], fill=(20, 90, 220), outline=(255, 255, 255))


def _outline_cells(
    image: Image.Image,
    mask: np.ndarray,
    scale: int,
    *,
    color=(255, 255, 255),
    width: int = 1,
) -> None:
    draw = ImageDraw.Draw(image)
    for row_index, column_index in np.argwhere(mask):
        left = int(column_index) * scale
        top = int(row_index) * scale
        draw.rectangle(
            (left, top, left + scale - 1, top + scale - 1),
            outline=color,
            width=width,
        )


def _zoom_bounds(mask: np.ndarray, *, radius_cells: int = 10):
    rows, columns = mask.shape
    locations = np.argwhere(mask)
    if locations.size:
        center_row, center_column = np.rint(
            locations.mean(axis=0)
        ).astype(int)
    else:
        # The controlled 10 m reference location for the default 40 m BEV is
        # near its center.  This also gives an informative fixed crop for the
        # matched actor-absent control sample.
        center_row, center_column = rows // 2, columns // 2
    row_start = max(0, center_row - radius_cells)
    row_stop = min(rows, center_row + radius_cells + 1)
    column_start = max(0, center_column - radius_cells)
    column_stop = min(columns, center_column + radius_cells + 1)
    return row_start, row_stop, column_start, column_stop


def render_sample_tile(
    row: dict,
    *,
    cell_scale: int = 2,
    control_row: dict | None = None,
) -> Image.Image:
    """Render geometry, independent targets, controlled evidence, support."""

    sample_path = Path(row['derived_sample_path'])
    with np.load(sample_path, allow_pickle=False) as arrays:
        missing = sorted(REQUIRED_ARRAYS.difference(arrays.files))
        if missing:
            raise ValueError(
                f'{sample_path} missing {", ".join(missing)}'
            )
        bev = np.asarray(arrays['lidar_bev'], dtype=np.float32)
        controlled_mask = (
            np.asarray(arrays['target_controlled_obstacle_point_count']) > 0
        )
        full_sources = [
            _raw_image(bev),
            _target_image(arrays),
            _controlled_image(arrays),
            _support_image(arrays),
        ]
        zoom_bounds = _zoom_bounds(controlled_mask)
        row_start, row_stop, column_start, column_stop = zoom_bounds
        zoom_sources = [
            _controlled_image(arrays)[
                row_start:row_stop, column_start:column_stop
            ],
            _target_image(arrays)[
                row_start:row_stop, column_start:column_stop
            ],
        ]
        zoom_mask = controlled_mask[
            row_start:row_stop, column_start:column_stop
        ]
    shape = full_sources[0].shape[:2]
    if any(source.shape[:2] != shape for source in full_sources):
        raise ValueError('incompatible target shapes: ' + str(sample_path))

    titles = [
        'Geometric LiDAR',
        'Independent evidence target',
        'Controlled-actor evidence',
        'Local-height / visibility support',
        'Controlled evidence zoom r{}:{} c{}:{}'.format(*zoom_bounds),
        'Target zoom r{}:{} c{}:{}'.format(*zoom_bounds),
    ]
    audit = audit_sample(row, control_row=control_row)
    panel_width = shape[1] * cell_scale
    panel_height = shape[0] * cell_scale
    title_height = 24
    info_height = 52
    gap = 8
    tile = Image.new(
        'RGB',
        (2 * panel_width + gap, info_height + 3 * (title_height + panel_height)),
        'white',
    )
    draw = ImageDraw.Draw(tile)
    draw.text(
        (5, 3),
        f'{row["source_session"]} | sample {row["source_sample_id"]}',
        fill=(20, 20, 20), font=_font(12, True),
    )
    draw.text(
        (5, 26),
        'actor={} returns={} controlled obstacle cells={} audit={}'.format(
            audit['controlled_actor_ids'],
            audit['controlled_return_count_derived'],
            audit['controlled_obstacle_cell_count'],
            'PASS' if audit['audit_passed'] else 'FAIL',
        ),
        fill=(50, 50, 50), font=_font(11),
    )
    for index, (source, title) in enumerate(
        zip(full_sources + zoom_sources, titles)
    ):
        column = index % 2
        row_index = index // 2
        x = column * (panel_width + gap)
        y = info_height + row_index * (title_height + panel_height)
        draw.text((x + 4, y + 3), title, fill=(20, 20, 20),
                  font=_font(11, True))
        panel = Image.fromarray(source).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        if index < 4:
            _vehicle_marker(panel, cell_scale)
            if index in (1, 2):
                _outline_cells(panel, controlled_mask, cell_scale)
        else:
            zoom_scale = max(1, panel_width // source.shape[1])
            _outline_cells(
                panel, zoom_mask, zoom_scale,
                color=(255, 255, 255), width=max(1, zoom_scale // 8),
            )
        tile.paste(panel, (x, y + title_height))
    return tile


def _legend(width: int) -> Image.Image:
    entries = [
        ('Unknown', UNKNOWN_COLOR), ('Passable', PASSABLE_COLOR),
        ('Obstacle', OBSTACLE_COLOR), ('Ambiguous', AMBIGUOUS_COLOR),
        ('Controlled obstacle', CONTROLLED_OBSTACLE_COLOR),
        ('Ego', EGO_COLOR),
    ]
    image = Image.new('RGB', (width, 42), 'white')
    draw = ImageDraw.Draw(image)
    x = 8
    font = _font(10)
    for label, color in entries:
        draw.rectangle((x, 12, x + 15, 27), fill=color, outline=(70, 70, 70))
        x += 20
        draw.text((x, 12), label, fill=(25, 25, 25), font=font)
        x += int(draw.textlength(label, font=font)) + 15
    return image


def write_contact_sheet(
    rows: list[dict],
    output_directory: Path,
    *,
    dpi: int = 300,
    samples_per_page: int = 3,
    control_row: dict | None = None,
) -> list[Path]:
    if not rows:
        raise ValueError('no written samples to visualize')
    output_directory.mkdir(parents=True, exist_ok=True)
    tiles = [
        render_sample_tile(row, control_row=control_row) for row in rows
    ]
    pages: list[tuple[Image.Image, Path]] = []
    for start in range(0, len(tiles), samples_per_page):
        selected = tiles[start:start + samples_per_page]
        width = max(tile.width for tile in selected)
        legend = _legend(width)
        page = Image.new(
            'RGB', (width, legend.height + sum(tile.height for tile in selected)),
            'white',
        )
        page.paste(legend, (0, 0))
        y = legend.height
        for tile in selected:
            page.paste(tile, ((width - tile.width) // 2, y))
            y += tile.height
        path = output_directory / f'controlled_evidence_{len(pages) + 1:02d}.png'
        page.save(path, format='PNG', dpi=(dpi, dpi))
        pages.append((page, path))
    pdf = output_directory / 'controlled_evidence.pdf'
    pages[0][0].save(
        pdf, format='PDF', save_all=True,
        append_images=[page for page, _ in pages[1:]], resolution=float(dpi),
    )
    return [path for _, path in pages] + [pdf]


def audit_evidence_dataset(
    manifest: Path,
    output_directory: Path,
    *,
    count: int = 0,
    dpi: int = 300,
    fail_on_violation: bool = True,
    control_session: str = '',
) -> dict:
    """Audit every written row and create deterministic contact sheets."""

    rows = read_written_rows(manifest)
    if not rows:
        raise ValueError('manifest contains no written samples')
    if count > 0:
        rows = rows[:count]
    control_row = None
    if control_session:
        matches = [
            row for row in rows
            if row.get('source_session') == control_session
        ]
        if len(matches) != 1:
            raise ValueError(
                f'control session {control_session!r} must identify exactly '
                f'one written sample, found {len(matches)}'
            )
        control_row = matches[0]
    audits = [
        audit_sample(
            row,
            control_row=(
                control_row
                if control_row is not None and row is not control_row
                else None
            ),
        )
        for row in rows
    ]
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_csv = output / 'audit.csv'
    temporary_csv = audit_csv.with_suffix('.csv.tmp')
    with temporary_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audits)
    temporary_csv.replace(audit_csv)
    outputs = write_contact_sheet(
        rows, output, dpi=dpi, control_row=control_row
    )
    failed = [item for item in audits if not item['audit_passed']]
    summary = {
        'manifest': str(Path(manifest).expanduser().resolve()),
        'audited_samples': len(audits),
        'passed_samples': len(audits) - len(failed),
        'failed_samples': len(failed),
        'controlled_samples': sum(
            item['controlled_return_count_derived'] > 0 for item in audits
        ),
        'total_controlled_returns': sum(
            item['controlled_return_count_derived'] for item in audits
        ),
        'total_controlled_obstacle_cells': sum(
            item['controlled_obstacle_cell_count'] for item in audits
        ),
        'counterfactual_control_session': control_session,
        'counterfactual_evaluated_samples': sum(
            item['counterfactual_evaluated'] for item in audits
        ),
        'controlled_cells_obstacle_in_control': sum(
            item['controlled_cells_obstacle_in_control'] for item in audits
        ),
        'violations': [
            {
                'source_session': item['source_session'],
                'source_sample_id': item['source_sample_id'],
                'violations': item['violations'],
            }
            for item in failed
        ],
        'audit_csv': str(audit_csv),
        'outputs': [str(path) for path in outputs],
    }
    summary_path = output / 'summary.json'
    temporary_summary = summary_path.with_suffix('.json.tmp')
    temporary_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary_summary.replace(summary_path)
    if failed and fail_on_violation:
        raise RuntimeError(
            f'{len(failed)} evidence samples failed invariants; '
            f'see {audit_csv}'
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--output-directory', type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument(
        '--control-session', default='',
        help=(
            'Actor-absent written session used to verify that controlled '
            'obstacle cells are not already obstacle in the background.'
        ),
    )
    parser.add_argument(
        '--fail-on-violation', action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = audit_evidence_dataset(
        args.manifest,
        args.output_directory,
        count=args.count,
        dpi=args.dpi,
        fail_on_violation=args.fail_on_violation,
        control_session=args.control_session,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
