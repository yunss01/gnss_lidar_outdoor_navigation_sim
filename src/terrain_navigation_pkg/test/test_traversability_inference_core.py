import numpy as np
import pytest

from terrain_navigation_pkg.traversability_inference_core import (
    ShadowPolicyAccumulator,
    bev_to_occupancy_grid,
    binary_predictive_entropy,
    build_selective_decision,
    inference_rate_limit_allows,
    occupancy_grid_to_bev,
)


def test_rate_limit_accepts_nominal_boundary_with_callback_jitter():
    interval = 0.2
    assert not inference_rate_limit_allows(10.19, 10.0, interval)
    assert inference_rate_limit_allows(10.199, 10.0, interval)
    assert inference_rate_limit_allows(10.2, 10.0, interval)


def test_rate_limit_accepts_first_frame_and_clock_reset():
    assert inference_rate_limit_allows(10.0, -np.inf, 0.2)
    assert inference_rate_limit_allows(1.0, 10.0, 0.2)


def test_rate_limit_selects_every_second_timestamp_from_10_hz_stream():
    accepted = []
    last = -np.inf
    for stamp in np.arange(0.0, 1.0, 0.1):
        if inference_rate_limit_allows(stamp, last, 0.2):
            accepted.append(float(stamp))
            last = stamp
    assert accepted == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])


@pytest.mark.parametrize(
    'now,last,interval,tolerance',
    [
        (np.nan, 0.0, 0.2, 0.005),
        (1.0, 0.0, 0.0, 0.005),
        (1.0, 0.0, 0.2, -0.001),
    ],
)
def test_rate_limit_rejects_invalid_configuration(
    now, last, interval, tolerance
):
    with pytest.raises(ValueError):
        inference_rate_limit_allows(
            now, last, interval, tolerance_s=tolerance
        )


def test_binary_entropy_has_expected_limits():
    entropy = binary_predictive_entropy(
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    )
    assert entropy[0] < 1e-5
    assert entropy[1] == pytest.approx(1.0)
    assert entropy[2] < 1e-5


def test_selective_decision_preserves_hard_geometry_and_uncertainty():
    probability = np.asarray([[0.01, 0.60, 0.01, 0.01, 0.90]])
    entropy = np.asarray([[0.05, 0.97, 0.05, 0.05, 0.05]])
    observed = np.ones((1, 5), dtype=bool)
    maximum_height = np.asarray([[-1.8, -1.8, -1.0, -1.8, -1.8]])
    vertical_span = np.asarray([[0.0, 0.0, 0.0, 0.2, 0.0]])

    result = build_selective_decision(
        probability,
        entropy,
        observed,
        maximum_height,
        vertical_span,
    )

    assert result.decision.tolist() == [[0, -1, 100, 100, 100]]
    assert result.learned_free_mask.tolist() == [
        [True, False, False, False, False]
    ]
    assert result.uncertain_mask.tolist() == [
        [False, True, False, False, False]
    ]
    assert result.hard_obstacle_mask.tolist() == [
        [False, False, True, True, False]
    ]


def test_excluded_and_unobserved_cells_remain_unknown():
    shape = (2, 2)
    result = build_selective_decision(
        np.full(shape, 0.01),
        np.zeros(shape),
        np.asarray([[True, False], [True, True]]),
        np.full(shape, -1.8),
        np.zeros(shape),
        exclusion_mask=np.asarray([[False, False], [True, False]]),
    )
    assert result.decision.tolist() == [[0, -1], [-1, 0]]


def test_bev_to_occupancy_grid_reorders_forward_and_left_axes():
    bev = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    converted = bev_to_occupancy_grid(bev)
    assert converted.tolist() == [[6, 3], [5, 2], [4, 1]]


def test_occupancy_grid_conversion_round_trip_supports_rectangles():
    bev = np.arange(15, dtype=np.int8).reshape((3, 5))
    occupancy = bev_to_occupancy_grid(bev)
    assert occupancy.shape == (5, 3)
    assert np.array_equal(occupancy_grid_to_bev(occupancy), bev)


def test_shadow_policy_accumulator_reports_abstention_and_errors():
    probability = np.asarray([[0.01, 0.90, 0.40, 0.01]])
    entropy = np.asarray([[0.01, 0.01, 0.90, 0.01]])
    observed = np.ones((1, 4), dtype=bool)
    result = build_selective_decision(
        probability,
        entropy,
        observed,
        np.asarray([[-1.8, -1.8, -1.8, -1.0]]),
        np.zeros((1, 4)),
    )
    metrics = ShadowPolicyAccumulator()
    metrics.update(
        result,
        np.asarray([[0, 1, 1, 0]]),
        observed,
        np.zeros((1, 4), dtype=bool),
    )
    values = metrics.compute()
    assert values['known_coverage'] == pytest.approx(0.75)
    assert values['accepted_accuracy'] == pytest.approx(2.0 / 3.0)
    assert values['false_obstacle'] == 1
    assert values['abstained_known'] == 1
