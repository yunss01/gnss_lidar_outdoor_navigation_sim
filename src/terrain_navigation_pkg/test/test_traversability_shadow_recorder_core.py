import math

import pytest

from terrain_navigation_pkg.traversability_shadow_recorder_core import (
    enrich_shadow_status,
    summarize_shadow_rows,
)


def test_enrich_shadow_status_keeps_unknown_separate():
    result = enrich_shadow_status({
        'observed_cells': 100,
        'learned_free_cells': 60,
        'learned_obstacle_cells': 10,
        'hard_obstacle_cells': 20,
        'uncertain_cells': 10,
    })
    assert result['known_cells'] == 90
    assert result['obstacle_cells'] == 30
    assert result['known_coverage'] == pytest.approx(0.9)
    assert result['free_fraction'] == pytest.approx(0.6)
    assert result['obstacle_fraction'] == pytest.approx(0.3)
    assert result['uncertain_fraction'] == pytest.approx(0.1)


def test_enrich_shadow_status_handles_empty_scan():
    result = enrich_shadow_status({'observed_cells': 0})
    assert math.isnan(result['known_coverage'])


def test_summarize_shadow_rows_reports_mean_maximum_and_p95():
    rows = []
    for value in (1.0, 2.0, 3.0):
        rows.append({name: value for name in (
            'inference_ms', 'mean_entropy', 'mean_mc_variance',
            'known_coverage', 'uncertain_fraction', 'obstacle_fraction',
            'observed_cells',
        )})
    result = summarize_shadow_rows(rows)
    assert result['frame_count'] == 3
    assert result['inference_ms']['mean'] == pytest.approx(2.0)
    assert result['inference_ms']['minimum'] == pytest.approx(1.0)
    assert result['inference_ms']['maximum'] == pytest.approx(3.0)
    assert result['inference_ms']['p95'] == pytest.approx(2.9)
