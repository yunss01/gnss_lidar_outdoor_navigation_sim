#!/usr/bin/env python3
"""Visualize held-out traversability predictions and their cell errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from .traversability_dataset import (
    TraversabilityDataset,
    read_written_manifest_rows,
)
from .traversability_model import BevTraversabilityUNet
from .visualize_traversability_dataset import (
    EGO_COLOR,
    FREE_COLOR,
    OBSTACLE_COLOR,
    UNKNOWN_COLOR,
    VISIBILITY_COLOR,
    _font,
    _raw_image,
    _vehicle_marker,
)


FALSE_FREE_COLOR = (184, 62, 120)
CORRECT_COLOR = (205, 205, 205)


def _score_predictions(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            lidar_bev = batch['lidar_bev'].to(device)
            targets = batch['target_labels'].to(device)
            predictions = model(lidar_bev).argmax(dim=1)
            for index in range(targets.shape[0]):
                target = targets[index]
                prediction = predictions[index]
                obstacle = target == 1
                free = target == 0
                false_free = int(torch.count_nonzero(
                    obstacle & (prediction == 0)
                ))
                false_obstacle = int(torch.count_nonzero(
                    free & (prediction == 1)
                ))
                obstacle_count = int(torch.count_nonzero(obstacle))
                records.append({
                    'row_index': offset + index,
                    'false_free': false_free,
                    'false_obstacle': false_obstacle,
                    'false_free_rate': (
                        false_free / obstacle_count if obstacle_count else 0.0
                    ),
                })
            offset += targets.shape[0]
    return records


def _spaced_selection(records, rows, count, key):
    ordered = sorted(
        records,
        key=lambda item: (
            item[key], item['false_free'], item['false_obstacle']
        ),
        reverse=True,
    )
    selected = []
    used = {}
    for record in ordered:
        row = rows[record['row_index']]
        session = row['source_session']
        sample_id = int(row['source_sample_id'])
        previous = used.setdefault(session, [])
        if any(abs(sample_id - value) < 15 for value in previous):
            continue
        selected.append(record)
        previous.append(sample_id)
        if len(selected) == count:
            break
    return selected


def _target_image(labels, ego):
    image = np.empty(labels.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[labels == 0] = FREE_COLOR
    image[labels == 1] = OBSTACLE_COLOR
    image[ego] = EGO_COLOR
    return image


def _prediction_image(prediction, known, ego):
    image = np.empty(prediction.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[known & (prediction == 0)] = FREE_COLOR
    image[known & (prediction == 1)] = OBSTACLE_COLOR
    image[ego] = EGO_COLOR
    return image


def _error_image(prediction, labels, ego):
    known = labels >= 0
    image = np.empty(labels.shape + (3,), dtype=np.uint8)
    image[:] = UNKNOWN_COLOR
    image[known & (prediction == labels)] = CORRECT_COLOR
    image[(labels == 1) & (prediction == 0)] = FALSE_FREE_COLOR
    image[(labels == 0) & (prediction == 1)] = VISIBILITY_COLOR
    image[ego] = EGO_COLOR
    return image


def _render_tile(model, dataset, row, record, device, scale=2):
    item = dataset[record['row_index']]
    lidar_bev = item['lidar_bev'].unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(lidar_bev), dim=1)[0]
    prediction = probability.argmax(dim=0).cpu().numpy()
    labels = item['target_labels'].numpy()
    known = labels >= 0
    with np.load(row['derived_sample_path'], allow_pickle=False) as arrays:
        raw_bev = np.asarray(arrays['lidar_bev'], dtype=np.float32)
        ego = np.asarray(arrays['target_ego_exclusion_mask'], dtype=bool)

    sources = [
        _raw_image(raw_bev),
        _target_image(labels, ego),
        _prediction_image(prediction, known, ego),
        _error_image(prediction, labels, ego),
    ]
    titles = ['Geometric LiDAR', 'Semantic target',
              'Prediction on known cells', 'Classification errors']
    panel_width = labels.shape[1] * scale
    panel_height = labels.shape[0] * scale
    title_height = 25
    info_height = 48
    gap = 8
    tile = Image.new(
        'RGB',
        (2 * panel_width + gap,
         info_height + 2 * (title_height + panel_height)),
        'white',
    )
    draw = ImageDraw.Draw(tile)
    heading = '{} | sample {}'.format(
        row['source_session'].removeprefix('session_'),
        row['source_sample_id'],
    )
    counts = 'missed obstacle={}  false obstacle={}  miss rate={:.2%}'.format(
        record['false_free'], record['false_obstacle'],
        record['false_free_rate'],
    )
    draw.text((5, 3), heading, fill=(20, 20, 20), font=_font(13, True))
    draw.text((5, 25), counts, fill=(55, 55, 55), font=_font(11))
    for index, (source, title) in enumerate(zip(sources, titles)):
        column = index % 2
        row_index = index // 2
        x = column * (panel_width + gap)
        y = info_height + row_index * (title_height + panel_height)
        draw.text((x + 4, y + 3), title, fill=(20, 20, 20),
                  font=_font(12, True))
        panel = Image.fromarray(source).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        _vehicle_marker(panel, scale)
        tile.paste(panel, (x, y + title_height))
    return tile


def _write_sheet(tiles, output_stem, dpi=300):
    pages = []
    for start in range(0, len(tiles), 4):
        current = tiles[start:start + 4]
        width = max(tile.width for tile in current)
        legend_height = 45
        page = Image.new(
            'RGB',
            (width, legend_height + sum(tile.height for tile in current)),
            'white',
        )
        draw = ImageDraw.Draw(page)
        entries = [
            ('Correct', CORRECT_COLOR),
            ('Missed obstacle (unsafe)', FALSE_FREE_COLOR),
            ('False obstacle', VISIBILITY_COLOR),
            ('Unknown/ignored', UNKNOWN_COLOR),
        ]
        x = 8
        for label, color in entries:
            draw.rectangle((x, 13, x + 17, 30), fill=color)
            x += 23
            draw.text((x, 12), label, fill=(25, 25, 25), font=_font(11))
            x += int(draw.textlength(label, font=_font(11))) + 20
        y = legend_height
        for tile in current:
            page.paste(tile, (0, y))
            y += tile.height
        path = output_stem.with_name(
            '{}_{:02d}.png'.format(output_stem.name, len(pages) + 1)
        )
        page.save(path, format='PNG', dpi=(dpi, dpi))
        pages.append((page, path))
    pdf = output_stem.with_suffix('.pdf')
    pages[0][0].save(
        pdf, format='PDF', save_all=True,
        append_images=[item[0] for item in pages[1:]], resolution=float(dpi),
    )
    return [str(item[1]) for item in pages] + [str(pdf)]


def generate(args):
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
    sessions = args.session or checkpoint[
        'training_config'
    ]['validation_sessions']
    rows = [
        row for row in read_written_manifest_rows(args.manifest)
        if row['source_session'] in set(sessions)
    ]
    if not rows:
        raise ValueError('no samples found for requested sessions')
    dataset = TraversabilityDataset(rows)
    records = _score_predictions(
        model, dataset, device, args.batch_size
    )
    count = min(args.count, len(records))
    worst = _spaced_selection(records, rows, count, 'false_free_rate')
    random_records = records.copy()
    random.Random(args.seed).shuffle(random_records)
    random_records = random_records[:count]
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = {'worst_missed_obstacle': worst, 'random': random_records}
    outputs = []
    for name, selected in groups.items():
        tiles = [
            _render_tile(
                model, dataset, rows[record['row_index']], record, device
            )
            for record in selected
        ]
        outputs.extend(_write_sheet(tiles, output / name, args.dpi))
    summary = {
        'checkpoint': str(Path(args.checkpoint).expanduser().resolve()),
        'sessions': sessions,
        'evaluated_samples': len(rows),
        'outputs': outputs,
        'worst_samples': [
            {
                'session': rows[item['row_index']]['source_session'],
                'sample_id': int(
                    rows[item['row_index']]['source_sample_id']
                ),
                'false_free': item['false_free'],
                'false_obstacle': item['false_obstacle'],
                'false_free_rate': item['false_free_rate'],
            }
            for item in worst
        ],
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--session', action='append', default=[])
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.count < 1:
        raise ValueError('count must be positive')
    print(json.dumps(generate(args), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
