#!/usr/bin/env python3
"""Create 300-DPI contact sheets for traversability-target auditing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MANIFEST = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/v1/manifest.csv'
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'visualizations/v1'
)

UNKNOWN_COLOR = (35, 39, 46)
FREE_COLOR = (92, 138, 92)
OBSTACLE_COLOR = (193, 99, 58)
VISIBILITY_COLOR = (74, 114, 168)
SPAN_OVERRIDE_COLOR = (242, 193, 78)
EGO_COLOR = (45, 45, 45)
RAW_COLOR = (35, 35, 35)
RAW_BACKGROUND = (248, 248, 248)

REQUIRED_ARRAYS = {
    'lidar_bev', 'target_labels', 'target_obstacle_point_count',
    'target_vertical_span_m', 'target_ego_exclusion_mask',
    'visibility_free_mask', 'visibility_ray_count',
}


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def read_manifest(path: Path, session_results: set[str]) -> list[dict]:
    """Load valid derived samples with deterministic ordering."""
    path = Path(path).expanduser()
    with path.open(newline='', encoding='utf-8') as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get('status') == 'written'
            and row.get('derived_sample_path')
            and (
                not session_results
                or row.get('source_session_result') in session_results
            )
        ]
    rows.sort(key=lambda row: (
        row.get('source_session', ''),
        _integer(row.get('source_sample_id')),
    ))
    return rows


def random_selection(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Choose reproducible samples while avoiding near-duplicate frames."""
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    selected = []
    ids_by_session = {}
    for row in candidates:
        session = row['source_session']
        sample_id = _integer(row['source_sample_id'])
        used = ids_by_session.setdefault(session, [])
        if any(abs(sample_id - previous) < 15 for previous in used):
            continue
        selected.append(row)
        used.append(sample_id)
        if len(selected) >= count:
            break
    return selected


def sequence_selection(
    rows: list[dict], count: int, session: str = ''
) -> list[dict]:
    """Choose evenly spaced samples from one completed sequence."""
    by_session = {}
    for row in rows:
        by_session.setdefault(row['source_session'], []).append(row)
    if session:
        candidates = by_session.get(session, [])
        if not candidates:
            raise ValueError('sequence session not found: ' + session)
    else:
        if not by_session:
            return []
        completed = {
            name: items for name, items in by_session.items()
            if items[0].get('source_session_result') == 'completed'
        }
        pool = completed or by_session
        name = max(pool, key=lambda key: (len(pool[key]), key))
        candidates = pool[name]
    if len(candidates) <= count:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[index] for index in indices]


def _font(size: int, bold: bool = False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _raw_image(bev: np.ndarray) -> np.ndarray:
    density = np.asarray(bev[1] if bev.shape[0] > 1 else bev[0])
    level = np.clip(density, 0.0, 1.0)
    value = (248.0 - 220.0 * level).astype(np.uint8)
    return np.repeat(value[:, :, None], 3, axis=2)


def _target_image(labels: np.ndarray, ego: np.ndarray) -> np.ndarray:
    image = np.empty(labels.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[labels == 0] = FREE_COLOR
    image[labels == 1] = OBSTACLE_COLOR
    image[ego] = EGO_COLOR
    return image


def _visibility_image(
    visibility: np.ndarray,
    labels: np.ndarray,
    ego: np.ndarray,
) -> np.ndarray:
    image = np.empty(labels.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[visibility] = VISIBILITY_COLOR
    image[labels == 1] = OBSTACLE_COLOR
    image[ego] = EGO_COLOR
    return image


def _audit_image(
    bev: np.ndarray,
    labels: np.ndarray,
    obstacle_count: np.ndarray,
    vertical_span: np.ndarray,
    ego: np.ndarray,
) -> np.ndarray:
    image = np.empty(labels.shape + (3,), dtype=np.uint8)
    image[:] = RAW_BACKGROUND
    image[np.asarray(bev[0]) > 0] = (190, 190, 190)
    image[labels == 0] = FREE_COLOR
    image[labels == 1] = OBSTACLE_COLOR
    span_override = (
        (labels == 1)
        & (obstacle_count == 0)
        & (vertical_span >= 0.15)
    )
    image[span_override] = SPAN_OVERRIDE_COLOR
    image[ego] = EGO_COLOR
    return image


def _vehicle_marker(image: Image.Image, scale: int) -> None:
    """Mark LiDAR origin and forward direction on a 160-cell default BEV."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    center_x = width // 2
    origin_y = int(round(0.75 * height))
    radius = max(3, scale + 1)
    draw.polygon([
        (center_x, origin_y - 3 * radius),
        (center_x - 2 * radius, origin_y + 2 * radius),
        (center_x + 2 * radius, origin_y + 2 * radius),
    ], fill=(20, 90, 220), outline=(255, 255, 255))


def render_sample_tile(
    row: dict,
    *,
    cell_scale: int = 2,
) -> Image.Image:
    """Render four complementary views for one derived sample."""
    sample_path = Path(row['derived_sample_path'])
    with np.load(sample_path, allow_pickle=False) as arrays:
        missing = sorted(REQUIRED_ARRAYS.difference(arrays.files))
        if missing:
            raise ValueError(
                '{} missing {}'.format(sample_path, ', '.join(missing))
            )
        bev = np.asarray(arrays['lidar_bev'], dtype=np.float32)
        labels = np.asarray(arrays['target_labels'], dtype=np.int8)
        obstacle_count = np.asarray(
            arrays['target_obstacle_point_count']
        )
        vertical_span = np.asarray(arrays['target_vertical_span_m'])
        ego = np.asarray(arrays['target_ego_exclusion_mask'], dtype=bool)
        visibility = np.asarray(arrays['visibility_free_mask'], dtype=bool)
    if bev.ndim != 3 or labels.shape != bev.shape[1:]:
        raise ValueError('incompatible BEV and target shapes: ' + str(sample_path))

    sources = [
        _raw_image(bev),
        _target_image(labels, ego),
        _visibility_image(visibility, labels, ego),
        _audit_image(bev, labels, obstacle_count, vertical_span, ego),
    ]
    titles = [
        'Geometric LiDAR',
        'Direct class target',
        'Ray visibility evidence',
        'Target audit overlay',
    ]
    panel_width = labels.shape[1] * cell_scale
    panel_height = labels.shape[0] * cell_scale
    title_height = 25
    info_height = 49
    gap = 8
    tile = Image.new(
        'RGB',
        (2 * panel_width + gap, info_height + 2 * (title_height + panel_height)),
        'white',
    )
    draw = ImageDraw.Draw(tile)
    session = row['source_session'].removeprefix('session_')
    heading = '{} | sample {} | WP {} | {}'.format(
        session,
        row['source_sample_id'],
        row.get('route_index', '?'),
        row.get('source_session_result', ''),
    )
    counts = 'direct free={}  obstacle={}  ray-visible={}  ego-mask={}'.format(
        row.get('free_cell_count', '?'),
        row.get('obstacle_cell_count', '?'),
        row.get('visibility_free_cell_count', '?'),
        row.get('ego_excluded_cell_count', '?'),
    )
    draw.text((5, 3), heading, fill=(20, 20, 20), font=_font(13, True))
    draw.text((5, 25), counts, fill=(55, 55, 55), font=_font(11))
    for index, (source, title) in enumerate(zip(sources, titles)):
        column = index % 2
        row_index = index // 2
        x = column * (panel_width + gap)
        y = info_height + row_index * (title_height + panel_height)
        draw.text((x + 4, y + 3), title, fill=(20, 20, 20), font=_font(12, True))
        panel = Image.fromarray(source).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        _vehicle_marker(panel, cell_scale)
        tile.paste(panel, (x, y + title_height))
    return tile


def _legend(width: int) -> Image.Image:
    height = 42
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    entries = [
        ('Unknown', UNKNOWN_COLOR), ('Direct free', FREE_COLOR),
        ('Obstacle', OBSTACLE_COLOR), ('Ray-visible', VISIBILITY_COLOR),
        ('Height-span override', SPAN_OVERRIDE_COLOR), ('Ego mask', EGO_COLOR),
    ]
    x = 8
    for label, color in entries:
        draw.rectangle((x, 12, x + 17, 29), fill=color, outline=(80, 80, 80))
        x += 23
        draw.text((x, 12), label, fill=(25, 25, 25), font=_font(11))
        x += int(draw.textlength(label, font=_font(11))) + 21
    return image


def write_contact_sheet(
    rows: list[dict],
    output_stem: Path,
    *,
    dpi: int = 300,
    samples_per_page: int = 4,
) -> list[Path]:
    """Write PNG pages and one multi-page PDF."""
    if not rows:
        raise ValueError('no samples selected for visualization')
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    tiles = [render_sample_tile(row) for row in rows]
    pages = []
    for start in range(0, len(tiles), samples_per_page):
        page_tiles = tiles[start:start + samples_per_page]
        width = max(tile.width for tile in page_tiles)
        legend = _legend(width)
        height = legend.height + sum(tile.height for tile in page_tiles)
        page = Image.new('RGB', (width, height), 'white')
        page.paste(legend, (0, 0))
        y = legend.height
        for tile in page_tiles:
            page.paste(tile, ((width - tile.width) // 2, y))
            y += tile.height
        page_number = len(pages) + 1
        png_path = output_stem.with_name(
            '{}_{:02d}.png'.format(output_stem.name, page_number)
        )
        page.save(png_path, format='PNG', dpi=(dpi, dpi))
        pages.append((page, png_path))
    pdf_path = output_stem.with_suffix('.pdf')
    pages[0][0].save(
        pdf_path,
        format='PDF',
        save_all=True,
        append_images=[item[0] for item in pages[1:]],
        resolution=float(dpi),
    )
    return [item[1] for item in pages] + [pdf_path]


def generate_visualizations(
    manifest: Path,
    output_directory: Path,
    *,
    count: int = 12,
    seed: int = 20260915,
    session_results: set[str] | None = None,
    sequence_session: str = '',
    dpi: int = 300,
) -> dict:
    """Generate random and temporal contact sheets plus an audit summary."""
    results = session_results if session_results is not None else {'completed'}
    rows = read_manifest(manifest, results)
    if not rows:
        raise ValueError('manifest contains no matching written samples')
    count = min(max(1, int(count)), len(rows))
    random_rows = random_selection(rows, count, seed)
    sequence_rows = sequence_selection(rows, count, sequence_session)
    output_directory = Path(output_directory).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    random_outputs = write_contact_sheet(
        random_rows, output_directory / 'random_contact_sheet', dpi=dpi
    )
    sequence_outputs = write_contact_sheet(
        sequence_rows, output_directory / 'sequence_contact_sheet', dpi=dpi
    )
    summary = {
        'manifest': str(Path(manifest).expanduser().resolve()),
        'session_results': sorted(results),
        'eligible_samples': len(rows),
        'seed': seed,
        'dpi': dpi,
        'random_samples': [
            [row['source_session'], _integer(row['source_sample_id'])]
            for row in random_rows
        ],
        'sequence_samples': [
            [row['source_session'], _integer(row['source_sample_id'])]
            for row in sequence_rows
        ],
        'outputs': [str(path) for path in random_outputs + sequence_outputs],
    }
    path = output_directory / 'summary.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--output-directory', type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument(
        '--session-result', action='append', default=None,
        help='source session result to include; may be repeated',
    )
    parser.add_argument('--sequence-session', default='')
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    summary = generate_visualizations(
        args.manifest,
        args.output_directory,
        count=args.count,
        seed=args.seed,
        session_results=set(args.session_result or ['completed']),
        sequence_session=args.sequence_session,
        dpi=args.dpi,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
