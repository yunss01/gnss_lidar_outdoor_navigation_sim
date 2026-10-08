#!/usr/bin/env python3
"""Render a portfolio-ready LiDAR BEV/evidence figure from real data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from terrain_navigation_pkg.traversability_evidence_core import (
    build_conservative_evidence_decision,
)
from terrain_navigation_pkg.traversability_evidence_dataset import (
    normalize_lidar_evidence_bev,
)
from terrain_navigation_pkg.traversability_evidence_inference_core import (
    load_current_only_evidence_checkpoint,
    predict_evidence,
)


DEFAULT_SAMPLE = Path(
    '/home/sukja/terrain_nav_data/learning/traversability/'
    'v2_temporal_scene11_poseA_20260921/samples/'
    'session_20260921_203128_250684/sample_000031.npz'
)
DEFAULT_CHECKPOINT = Path(
    '/home/sukja/terrain_nav_data/learning/models/'
    'traversability_evidence_v2/'
    'cv_temporal_representation_current_only_scenes09_11_20260921/'
    'holdout_scene11/model/best.pt'
)
DEFAULT_SCENE = Path(
    '/home/sukja/Pictures/스크린샷/'
    '스크린샷 2026-09-22 15-49-26.png'
)
DEFAULT_OUTPUT = Path(
    '/home/sukja/terrain_nav_data/evaluation/portfolio/'
    'fig_lidar_evidence_perception_v5.png'
)

WIDTH = 3000
HEIGHT = 1000
X_MIN_M = -10.0
X_MAX_M = 30.0
Y_MIN_M = -20.0
Y_MAX_M = 20.0
RESOLUTION_M = 0.25

BACKGROUND = '#F4F7FB'
CARD = '#FFFFFF'
INK = '#172033'
MUTED = '#657086'
ACCENT = '#16B8C4'
UNKNOWN = '#3D4656'
PASSABLE = '#24A978'
OBSTACLE = '#F06449'


def _font(size: int, bold: bool = False):
    filename = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / filename
    return ImageFont.truetype(str(path), size=size)


def _rgb(value: str):
    value = value.lstrip('#')
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _actor_center(mask: np.ndarray) -> tuple[float, float]:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError('sample contains no controlled-obstacle cells')
    forward = X_MAX_M - (rows.astype(float) + 0.5) * RESOLUTION_M
    left = Y_MAX_M - (columns.astype(float) + 0.5) * RESOLUTION_M
    return float(np.mean(forward)), float(np.mean(left))


def _density_rgb(density: np.ndarray) -> np.ndarray:
    value = np.power(
        np.clip(np.asarray(density, dtype=np.float32), 0.0, 1.0), 0.52
    )
    low = np.asarray(_rgb('#0B111B'), dtype=np.float32)
    high = np.asarray(_rgb('#E8FBFF'), dtype=np.float32)
    cyan = np.asarray(_rgb('#6EE7F0'), dtype=np.float32)
    image = low + value[..., None] * (high - low)
    glow = np.clip((value - 0.55) / 0.45, 0.0, 1.0)[..., None]
    image = image * (1.0 - 0.16 * glow) + cyan * 0.16 * glow
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _decision_rgb(decision: np.ndarray) -> np.ndarray:
    image = np.empty(decision.shape + (3,), dtype=np.uint8)
    image[:] = _rgb(UNKNOWN)
    image[decision == 0] = _rgb(PASSABLE)
    image[decision == 100] = _rgb(OBSTACLE)
    return image


def _crop_indices(
    forward_min: float,
    forward_max: float,
    left_min: float,
    left_max: float,
) -> tuple[int, int, int, int]:
    row_start = int(np.floor((X_MAX_M - forward_max) / RESOLUTION_M))
    row_end = int(np.ceil((X_MAX_M - forward_min) / RESOLUTION_M))
    col_start = int(np.floor((Y_MAX_M - left_max) / RESOLUTION_M))
    col_end = int(np.ceil((Y_MAX_M - left_min) / RESOLUTION_M))
    return (
        max(0, row_start), min(160, row_end),
        max(0, col_start), min(160, col_end),
    )


def _paste_crop(
    canvas: Image.Image,
    array: np.ndarray,
    box: tuple[int, int, int, int],
    physical_bounds: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    indices = _crop_indices(*physical_bounds)
    row_start, row_end, col_start, col_end = indices
    crop = array[row_start:row_end, col_start:col_end]
    image = Image.fromarray(crop, mode='RGB').resize(
        (box[2] - box[0], box[3] - box[1]),
        Image.Resampling.NEAREST,
    )
    canvas.paste(image, (box[0], box[1]))
    return indices


def _data_pixel(
    forward: float,
    left: float,
    box: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    forward_min, forward_max, left_min, left_max = bounds
    x = box[0] + (left_max - left) / (left_max - left_min) * (
        box[2] - box[0]
    )
    y = box[1] + (forward_max - forward) / (
        forward_max - forward_min
    ) * (box[3] - box[1])
    return x, y


def _rounded_card(draw, box):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        (x0 + 8, y0 + 12, x1 + 8, y1 + 12),
        radius=34, fill='#DCE3EE',
    )
    draw.rounded_rectangle(
        box, radius=34, fill=CARD, outline='#DCE3EC', width=2,
    )


def _draw_main_plot(
    canvas, draw, image_array, box, actor_forward, actor_left,
    *, dark: bool,
):
    bounds = (-2.0, 20.0, -11.0, 11.0)
    _paste_crop(canvas, image_array, box, bounds)
    draw.rectangle(box, outline='#C9D2DE', width=2)

    center_x, center_y = _data_pixel(0.0, 0.0, box, bounds)
    pixels_per_metre = (box[2] - box[0]) / 22.0
    ring_color = '#D8FBFD' if dark else '#E9EDF3'
    for radius in (5.0, 10.0, 15.0, 20.0):
        r = radius * pixels_per_metre
        draw.ellipse(
            (center_x - r, center_y - r, center_x + r, center_y + r),
            outline=ring_color, width=1,
        )

    vehicle_left_top = _data_pixel(1.25, 0.62, box, bounds)
    vehicle_right_bottom = _data_pixel(-1.25, -0.62, box, bounds)
    draw.rounded_rectangle(
        (
            vehicle_left_top[0], vehicle_left_top[1],
            vehicle_right_bottom[0], vehicle_right_bottom[1],
        ),
        radius=8, fill='#111827', outline='#FFFFFF', width=2,
    )
    arrow_top = _data_pixel(0.95, 0.0, box, bounds)
    arrow_bottom = _data_pixel(0.10, 0.0, box, bounds)
    draw.line((arrow_bottom, arrow_top), fill=ACCENT, width=4)
    draw.polygon([
        (arrow_top[0], arrow_top[1] - 8),
        (arrow_top[0] - 7, arrow_top[1] + 4),
        (arrow_top[0] + 7, arrow_top[1] + 4),
    ], fill=ACCENT)

    focus_top_left = _data_pixel(
        actor_forward + 0.85, actor_left + 0.85, box, bounds
    )
    focus_bottom_right = _data_pixel(
        actor_forward - 0.85, actor_left - 0.85, box, bounds
    )
    draw.rectangle(
        (
            focus_top_left[0], focus_top_left[1],
            focus_bottom_right[0], focus_bottom_right[1],
        ),
        outline=ACCENT, width=5,
    )

    small = _font(18)
    tiny = _font(16)
    for forward in (0, 5, 10, 15, 20):
        x, y = _data_pixel(forward, 0, box, bounds)
        label = str(forward)
        label_box = draw.textbbox((0, 0), label, font=tiny)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (box[0] - label_width - 12, y - 9), label,
            font=tiny, fill=MUTED,
        )
    for left in (10, 5, 0, -5, -10):
        x, _ = _data_pixel(-2.0, left, box, bounds)
        label = str(abs(left)) if left else '0'
        prefix = 'L' if left > 0 else ('R' if left < 0 else '')
        text = prefix + label
        text_box = draw.textbbox((0, 0), text, font=tiny)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (x - text_width / 2, box[3] + 7), text,
            font=tiny, fill=MUTED,
        )
    draw.text((box[0] - 62, box[1] - 25), 'm', font=small, fill=MUTED)
    return bounds


def _draw_inset(
    canvas, draw, image_array, box, actor_forward, actor_left, actor_mask,
    *, forward_radius=2.25, left_radius=2.25,
):
    bounds = (
        actor_forward - forward_radius, actor_forward + forward_radius,
        actor_left - left_radius, actor_left + left_radius,
    )
    indices = _paste_crop(canvas, image_array, box, bounds)
    draw.rectangle(box, outline='#C9D2DE', width=2)
    row_start, row_end, col_start, col_end = indices
    width = box[2] - box[0]
    height = box[3] - box[1]
    rows = max(1, row_end - row_start)
    columns = max(1, col_end - col_start)
    for row, column in np.argwhere(actor_mask):
        if not (
            row_start <= row < row_end and col_start <= column < col_end
        ):
            continue
        x0 = box[0] + (column - col_start) / columns * width
        x1 = box[0] + (column + 1 - col_start) / columns * width
        y0 = box[1] + (row - row_start) / rows * height
        y1 = box[1] + (row + 1 - row_start) / rows * height
        draw.rectangle((x0, y0, x1, y1), outline='#FFFFFF', width=4)
    return bounds


def _legend_item(draw, x, y, color, label):
    draw.rounded_rectangle((x, y, x + 38, y + 38), radius=7, fill=color)
    draw.text((x + 54, y - 3), label, font=_font(32, bold=True), fill=INK)


def _draw_label_chip(
    draw,
    x,
    y,
    text,
    font,
    *,
    fill='#111827',
    text_fill='#FFFFFF',
    padding_x=20,
    padding_y=10,
    radius=14,
    swatch_color=None,
    swatch_size=0,
    swatch_gap=14,
):
    """Draw a content-sized chip with precisely centered label text."""
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    swatch_width = swatch_size + swatch_gap if swatch_color else 0
    content_height = max(text_height, swatch_size if swatch_color else 0)
    width = padding_x * 2 + swatch_width + text_width
    height = padding_y * 2 + content_height
    box = (
        int(round(x)), int(round(y)),
        int(round(x + width)), int(round(y + height)),
    )
    draw.rounded_rectangle(box, radius=radius, fill=fill)

    cursor_x = box[0] + padding_x
    if swatch_color:
        swatch_y = box[1] + (height - swatch_size) / 2
        draw.rounded_rectangle(
            (
                cursor_x, swatch_y,
                cursor_x + swatch_size, swatch_y + swatch_size,
            ),
            radius=max(3, int(swatch_size * 0.22)), fill=swatch_color,
        )
        cursor_x += swatch_width

    text_y = box[1] + (height - text_height) / 2 - text_bbox[1]
    draw.text((cursor_x, text_y), text, font=font, fill=text_fill)
    return box


def render(
    sample_path: Path,
    checkpoint_path: Path,
    scene_path: Path,
    output_path: Path,
):
    with np.load(sample_path, allow_pickle=False) as arrays:
        lidar_bev = np.asarray(arrays['lidar_bev'], dtype=np.float32)
        evidence_bev = np.asarray(
            arrays['current_lidar_evidence_bev'], dtype=np.float32
        )
        observed = np.asarray(arrays['target_observed_mask'], dtype=bool)
        actor_mask = np.asarray(
            arrays['target_controlled_obstacle_point_count'] > 0,
            dtype=bool,
        )

    device = torch.device('cpu')
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    model, checkpoint, _ = load_current_only_evidence_checkpoint(
        checkpoint_path, device
    )
    prediction = predict_evidence(
        model,
        normalize_lidar_evidence_bev(evidence_bev),
        device,
        mc_samples=1,
    )
    thresholds = checkpoint['training_config']['selection_thresholds']
    decision = build_conservative_evidence_decision(
        prediction.passable_probability,
        prediction.obstacle_probability,
        observed,
        np.max(evidence_bev[6:8], axis=0),
        hard_obstacle_mask=None,
        obstacle_probability_threshold=float(
            thresholds['obstacle_threshold']
        ),
        passable_probability_threshold=float(
            thresholds['passable_threshold']
        ),
        maximum_obstacle_probability_for_passable=float(
            thresholds['maximum_obstacle_probability_for_passable']
        ),
        minimum_passable_support=float(
            thresholds['minimum_passable_support']
        ),
    )

    actor_forward, actor_left = _actor_center(actor_mask)
    actor_scores = prediction.obstacle_probability[actor_mask]
    actor_cell_count = int(np.count_nonzero(actor_mask))
    detected_cells = int(np.count_nonzero(
        actor_scores >= float(thresholds['obstacle_threshold'])
    ))

    prediction_image = _decision_rgb(decision.decision)
    canvas = Image.new('RGB', (WIDTH, HEIGHT), _rgb(BACKGROUND))
    draw = ImageDraw.Draw(canvas)

    _rounded_card(draw, (35, 35, 2965, 955))

    # The evidence output is the hero visual.
    main_box = (90, 90, 930, 930)
    zoom_box = (1070, 90, 2900, 750)
    _draw_main_plot(
        canvas, draw, prediction_image, main_box,
        actor_forward, actor_left, dark=False,
    )

    zoom_bounds = _draw_inset(
        canvas, draw, prediction_image, zoom_box,
        actor_forward, actor_left, actor_mask,
        forward_radius=2.0, left_radius=5.55,
    )

    # Minimal semantic tags replace paragraph-style titles and subtitles.
    _draw_label_chip(
        draw, 115, 112, 'MODEL OUTPUT  ·  TRAVERSABILITY',
        _font(22, bold=True), padding_x=20, padding_y=10, radius=14,
    )

    actor_zoom_x, actor_zoom_y = _data_pixel(
        actor_forward, actor_left, zoom_box, zoom_bounds
    )
    callout_box = _draw_label_chip(
        draw, actor_zoom_x - 580, actor_zoom_y - 190,
        'MOTORHELMET  ·  9.0 m', _font(27, bold=True),
        padding_x=20, padding_y=11, radius=17,
        swatch_color=OBSTACLE, swatch_size=29, swatch_gap=14,
    )
    draw.line(
        (callout_box[2] - 18, callout_box[3],
         actor_zoom_x - 16, actor_zoom_y - 22),
        fill=OBSTACLE, width=5,
    )
    draw.polygon(
        [
            (actor_zoom_x - 5, actor_zoom_y - 5),
            (actor_zoom_x - 30, actor_zoom_y - 16),
            (actor_zoom_x - 14, actor_zoom_y - 34),
        ],
        fill=OBSTACLE,
    )

    # Connect the highlighted region in the global BEV to the magnified view.
    draw.line((930, 355, 1070, 90), fill='#81DCE2', width=3)
    draw.line((930, 430, 1070, 750), fill='#81DCE2', width=3)

    # A representative CARLA camera view explains what the tiny obstacle is.
    # It is deliberately labelled as a test scene rather than the same frame.
    scene_source = Image.open(scene_path).convert('RGB')
    scene_crop_bounds = (300, 320, 920, 680)
    scene_crop = scene_source.crop(scene_crop_bounds)
    scene_box = (2250, 100, 2870, 460)
    scene_shadow = (2268, 118, 2888, 478)
    draw.rounded_rectangle(scene_shadow, radius=20, fill='#1B2432')
    draw.rounded_rectangle(
        (scene_box[0] - 10, scene_box[1] - 10,
         scene_box[2] + 10, scene_box[3] + 10),
        radius=18, fill='#FFFFFF', outline='#CFD8E4', width=2,
    )
    scene_image = scene_crop.resize(
        (scene_box[2] - scene_box[0], scene_box[3] - scene_box[1]),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(scene_image, (scene_box[0], scene_box[1]))
    draw.rectangle(scene_box, outline='#FFFFFF', width=4)
    _draw_label_chip(
        draw, 2275, 122, 'CARLA LiDAR TEST SCENE',
        _font(20, bold=True), padding_x=18, padding_y=9, radius=12,
    )
    helmet_source_x = 724
    helmet_source_y = 515
    helmet_x = scene_box[0] + (
        (helmet_source_x - scene_crop_bounds[0]) /
        (scene_crop_bounds[2] - scene_crop_bounds[0])
    ) * (scene_box[2] - scene_box[0])
    helmet_y = scene_box[1] + (
        (helmet_source_y - scene_crop_bounds[1]) /
        (scene_crop_bounds[3] - scene_crop_bounds[1])
    ) * (scene_box[3] - scene_box[1])
    draw.ellipse(
        (helmet_x - 27, helmet_y - 27, helmet_x + 27, helmet_y + 27),
        outline=OBSTACLE, width=6,
    )
    helmet_chip = _draw_label_chip(
        draw, helmet_x + 38, helmet_y - 30, 'HELMET',
        _font(19, bold=True), padding_x=17, padding_y=8, radius=11,
    )
    draw.line(
        (helmet_x + 27, helmet_y - 5,
         helmet_chip[0] + 1, (helmet_chip[1] + helmet_chip[3]) / 2),
        fill=OBSTACLE, width=5,
    )

    # A single legend and a terse numerical callout are the only captions.
    legend_y = 812
    _legend_item(draw, 1085, legend_y, PASSABLE, 'Passable')
    _legend_item(draw, 1385, legend_y, OBSTACLE, 'Obstacle')
    _legend_item(draw, 1705, legend_y, UNKNOWN, 'Unknown')

    metric_text = '{:.1f} m   •   OBSTACLE {:.2f}   •   {}/{} CELLS'.format(
        actor_forward, float(np.mean(actor_scores)),
        detected_cells, actor_cell_count,
    )
    metric_font = _font(25, bold=True)
    metric_bbox = draw.textbbox((0, 0), metric_text, font=metric_font)
    metric_width = metric_bbox[2] - metric_bbox[0]
    metric_x = 2870 - metric_width
    draw.rounded_rectangle(
        (metric_x - 28, 806, 2900, 866),
        radius=22, fill='#E4FAFB', outline='#A8E8EC', width=2,
    )
    draw.text(
        (metric_x, 822), metric_text,
        font=metric_font, fill='#0B6870',
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=96, dpi=(200, 200))
    canvas.save(output_path.with_suffix('.pdf'), 'PDF', resolution=200.0)
    return {
        'output': str(output_path),
        'pdf': str(output_path.with_suffix('.pdf')),
        'sample': str(sample_path),
        'checkpoint': str(checkpoint_path),
        'representative_scene': str(scene_path),
        'actor_forward_m': actor_forward,
        'actor_left_m': actor_left,
        'actor_obstacle_score_mean': float(np.mean(actor_scores)),
        'actor_detected_cells': detected_cells,
        'actor_cell_count': actor_cell_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(
        args.sample.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.scene.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    for key, value in result.items():
        print('{}: {}'.format(key, value))


if __name__ == '__main__':
    main()
