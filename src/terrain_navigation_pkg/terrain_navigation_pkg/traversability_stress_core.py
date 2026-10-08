"""Sensor-domain perturbations for traversability robustness auditing."""

from __future__ import annotations

import math

import numpy as np


STRESS_CASES = (
    'clean',
    'height_bias_plus_15cm',
    'height_bias_minus_15cm',
    'height_bias_plus_30cm',
    'height_bias_minus_30cm',
    'height_noise_5cm',
    'pitch_plus_2deg',
    'roll_plus_2deg',
    'pitch_plus_5deg',
    'roll_plus_5deg',
    'density_60pct',
    'cell_dropout_15pct',
    'cell_dropout_30pct',
    'combined_sensor_shift',
    'combined_severe_shift',
)


def _coordinate_grids(geometry):
    rows = np.arange(geometry.height, dtype=np.float32)
    columns = np.arange(geometry.width, dtype=np.float32)
    forward = geometry.x_max_m - (
        rows + 0.5
    ) * geometry.resolution_m
    left = geometry.y_max_m - (
        columns + 0.5
    ) * geometry.resolution_m
    return np.meshgrid(forward, left, indexing='ij')


def apply_bev_stress(lidar_bev, target_labels, geometry, case, seed):
    """Return deterministically perturbed input and its valid target."""
    if case not in STRESS_CASES:
        raise ValueError('unknown stress case: ' + str(case))
    bev = np.asarray(lidar_bev, dtype=np.float32).copy()
    targets = np.asarray(target_labels, dtype=np.int64).copy()
    if bev.shape != (4, geometry.height, geometry.width):
        raise ValueError('lidar_bev does not match geometry')
    if targets.shape != bev.shape[1:]:
        raise ValueError('target_labels does not match lidar_bev')
    if case == 'clean':
        return bev, targets

    occupied = bev[0] > 0.5
    rng = np.random.default_rng(int(seed))

    def add_height_offset(offset):
        bev[2, occupied] += np.asarray(offset)[occupied]

    if case == 'height_bias_plus_15cm':
        bev[2, occupied] += 0.15
    elif case == 'height_bias_minus_15cm':
        bev[2, occupied] -= 0.15
    elif case == 'height_bias_plus_30cm':
        bev[2, occupied] += 0.30
    elif case == 'height_bias_minus_30cm':
        bev[2, occupied] -= 0.30
    elif case == 'height_noise_5cm':
        noise = rng.normal(0.0, 0.05, targets.shape).astype(np.float32)
        bev[2, occupied] += noise[occupied]
        bev[3, occupied] = np.maximum(
            0.0,
            bev[3, occupied]
            + rng.normal(0.0, 0.02, np.count_nonzero(occupied)),
        )
    elif case in {
        'pitch_plus_2deg', 'roll_plus_2deg',
        'pitch_plus_5deg', 'roll_plus_5deg',
    }:
        forward, left = _coordinate_grids(geometry)
        magnitude = 5.0 if '5deg' in case else 2.0
        pitch = math.radians(magnitude) if 'pitch' in case else 0.0
        roll = math.radians(magnitude) if 'roll' in case else 0.0
        add_height_offset(forward * math.tan(pitch) + left * math.tan(roll))
    elif case == 'density_60pct':
        bev[1] *= 0.60
    elif case == 'cell_dropout_15pct':
        dropped = occupied & (rng.random(targets.shape) < 0.15)
        bev[:, dropped] = 0.0
        targets[dropped] = -1
    elif case == 'cell_dropout_30pct':
        dropped = occupied & (rng.random(targets.shape) < 0.30)
        bev[:, dropped] = 0.0
        targets[dropped] = -1
    elif case == 'combined_sensor_shift':
        forward, left = _coordinate_grids(geometry)
        pitch = math.radians(rng.uniform(-2.0, 2.0))
        roll = math.radians(rng.uniform(-2.0, 2.0))
        bias = rng.uniform(-0.10, 0.10)
        noise = rng.normal(0.0, 0.03, targets.shape).astype(np.float32)
        height_change = (
            bias + forward * math.tan(pitch) + left * math.tan(roll) + noise
        )
        add_height_offset(height_change)
        bev[1] *= rng.uniform(0.60, 0.85)
        dropped = occupied & (rng.random(targets.shape) < 0.10)
        bev[:, dropped] = 0.0
        targets[dropped] = -1
    elif case == 'combined_severe_shift':
        forward, left = _coordinate_grids(geometry)
        pitch = math.radians(rng.uniform(-5.0, 5.0))
        roll = math.radians(rng.uniform(-5.0, 5.0))
        bias = rng.uniform(-0.20, 0.20)
        noise = rng.normal(0.0, 0.05, targets.shape).astype(np.float32)
        height_change = (
            bias + forward * math.tan(pitch) + left * math.tan(roll) + noise
        )
        add_height_offset(height_change)
        bev[1] *= rng.uniform(0.40, 0.70)
        dropped = occupied & (rng.random(targets.shape) < 0.25)
        bev[:, dropped] = 0.0
        targets[dropped] = -1
    return bev, targets


def apply_random_sensor_augmentation(
    lidar_bev,
    target_labels,
    geometry,
    seed,
    *,
    maximum_height_bias_m=0.20,
    maximum_tilt_deg=4.0,
    height_noise_standard_deviation_m=0.03,
    minimum_density_scale=0.50,
    maximum_cell_dropout_fraction=0.20,
):
    """Apply bounded geometric and sampling variation during training."""
    if maximum_height_bias_m < 0.0 or maximum_tilt_deg < 0.0:
        raise ValueError('height bias and tilt bounds must be non-negative')
    if height_noise_standard_deviation_m < 0.0:
        raise ValueError('height noise must be non-negative')
    if not 0.0 < minimum_density_scale <= 1.0:
        raise ValueError('minimum_density_scale must be in (0, 1]')
    if not 0.0 <= maximum_cell_dropout_fraction < 1.0:
        raise ValueError('maximum dropout must be in [0, 1)')
    bev = np.asarray(lidar_bev, dtype=np.float32).copy()
    targets = np.asarray(target_labels, dtype=np.int64).copy()
    if bev.shape != (4, geometry.height, geometry.width):
        raise ValueError('lidar_bev does not match geometry')
    if targets.shape != bev.shape[1:]:
        raise ValueError('target_labels does not match lidar_bev')

    rng = np.random.default_rng(int(seed))
    occupied = bev[0] > 0.5
    forward, left = _coordinate_grids(geometry)
    pitch = math.radians(rng.uniform(-maximum_tilt_deg, maximum_tilt_deg))
    roll = math.radians(rng.uniform(-maximum_tilt_deg, maximum_tilt_deg))
    bias = rng.uniform(-maximum_height_bias_m, maximum_height_bias_m)
    noise = rng.normal(
        0.0, height_noise_standard_deviation_m, targets.shape
    ).astype(np.float32)
    height_change = (
        bias + forward * math.tan(pitch) + left * math.tan(roll) + noise
    )
    bev[2, occupied] += height_change[occupied]
    density_scale = rng.uniform(minimum_density_scale, 1.0)
    bev[1] *= density_scale
    dropout_fraction = rng.uniform(0.0, maximum_cell_dropout_fraction)
    dropped = occupied & (rng.random(targets.shape) < dropout_fraction)
    bev[:, dropped] = 0.0
    targets[dropped] = -1
    return bev, targets
