"""Pure helpers for route-level traversability shadow summaries."""

from __future__ import annotations

import math

import numpy as np


SUMMARY_METRICS = (
    'inference_ms',
    'mean_entropy',
    'mean_mc_variance',
    'known_coverage',
    'uncertain_fraction',
    'obstacle_fraction',
    'observed_cells',
)


def enrich_shadow_status(status):
    """Add comparable fractions without treating unknown as free space."""
    output = dict(status)
    observed = max(0, int(output.get('observed_cells', 0)))
    learned_free = max(0, int(output.get('learned_free_cells', 0)))
    learned_obstacle = max(
        0, int(output.get('learned_obstacle_cells', 0))
    )
    hard_obstacle = max(0, int(output.get('hard_obstacle_cells', 0)))
    uncertain = max(0, int(output.get('uncertain_cells', 0)))
    obstacle = learned_obstacle + hard_obstacle
    known = learned_free + obstacle
    denominator = float(observed) if observed else math.nan
    output.update({
        'known_cells': known,
        'obstacle_cells': obstacle,
        'known_coverage': known / denominator,
        'free_fraction': learned_free / denominator,
        'obstacle_fraction': obstacle / denominator,
        'uncertain_fraction': uncertain / denominator,
    })
    return output


def summarize_shadow_rows(rows):
    """Return aggregate statistics for status rows from one route."""
    summary = {'frame_count': len(rows)}
    for name in SUMMARY_METRICS:
        values = np.asarray([
            float(row[name]) for row in rows
            if row.get(name) is not None
            and math.isfinite(float(row[name]))
        ], dtype=np.float64)
        if values.size == 0:
            summary[name] = {
                'mean': None, 'minimum': None,
                'maximum': None, 'p95': None,
            }
            continue
        summary[name] = {
            'mean': float(np.mean(values)),
            'minimum': float(np.min(values)),
            'maximum': float(np.max(values)),
            'p95': float(np.percentile(values, 95.0)),
        }
    return summary
