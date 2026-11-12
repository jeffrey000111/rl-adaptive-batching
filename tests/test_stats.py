from __future__ import annotations

import numpy as np
import pytest

from adaptive_batching.stats import (
    block_ids,
    classify,
    condition_verdict,
    paired_block_bootstrap,
)


def test_block_ids_are_consecutive_and_balanced() -> None:
    ids = block_ids(103, 20)
    assert ids[0] == 0 and ids[-1] == 19 and (np.diff(ids) >= 0).all()
    counts = np.bincount(ids)
    assert counts.sum() == 103 and counts.max() - counts.min() <= 1


@pytest.mark.parametrize(
    "low, high, expected",
    [(-0.2, -0.01, "rl_better"), (0.01, 0.2, "rl_worse"),
     (-0.03, 0.04, "equivalent"), (-0.2, 0.04, "inconclusive")],
)
def test_classify_uses_the_position_of_the_interval(low: float, high: float, expected: str) -> None:
    assert classify(low, high, 0.05) == expected


@pytest.mark.parametrize(
    "latency, slo, expected",
    [("rl_better", "equivalent", "rl_better"), ("rl_better", "rl_worse", "trade_off"),
     ("rl_worse", "rl_better", "rl_worse"), ("equivalent", "rl_worse", "rl_worse"),
     ("inconclusive", "rl_better", "no_difference"), ("equivalent", "equivalent", "no_difference")],
)
def test_condition_verdict(latency: str, slo: str, expected: str) -> None:
    assert condition_verdict(latency, slo) == expected


def test_identical_policies_give_a_zero_effect() -> None:
    rng = np.random.default_rng(0)
    latency = rng.exponential(0.1, size=2000)
    out = paired_block_bootstrap(np.stack([latency] * 3), latency, slo_s=0.3, num_resamples=500)
    for interval in out.values():
        # Summing blocks in a different order leaves rounding noise only.
        assert interval.estimate == 0.0
        assert abs(interval.low) < 1e-12 and abs(interval.high) < 1e-12
        assert interval.verdict == "equivalent"


def test_a_uniform_ten_percent_improvement_is_detected_exactly() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.exponential(0.1, size=2000)
    out = paired_block_bootstrap(np.stack([0.9 * baseline] * 5), baseline, slo_s=10.0,
                                 num_resamples=500)
    latency = out["mean_latency_relative_difference"]
    # Pairing removes the block-to-block noise: every replicate gives exactly -10%.
    assert latency.estimate == pytest.approx(-0.10)
    assert latency.low == pytest.approx(-0.10) and latency.high == pytest.approx(-0.10)
    assert latency.verdict == "rl_better"
    assert out["slo_violation_difference_points"].estimate == 0.0


def test_seed_to_seed_variation_widens_the_interval() -> None:
    rng = np.random.default_rng(2)
    baseline = rng.exponential(0.1, size=2000)
    seeds = np.stack([f * baseline for f in (0.7, 0.9, 1.0, 1.1, 1.3)])
    out = paired_block_bootstrap(seeds, baseline, slo_s=10.0, num_resamples=2000)
    latency = out["mean_latency_relative_difference"]
    assert latency.estimate == pytest.approx(0.0, abs=1e-12)
    assert latency.low < -0.1 and latency.high > 0.1
    assert latency.verdict == "inconclusive"


def test_slo_difference_is_in_percentage_points() -> None:
    baseline = np.full(1000, 0.1)
    learned = baseline.copy()
    learned[::10] = 1.0  # 10% of requests become late
    out = paired_block_bootstrap(learned[None, :], baseline, slo_s=0.5, num_resamples=300)
    slo = out["slo_violation_difference_points"]
    assert slo.estimate == pytest.approx(10.0) and slo.verdict == "rl_worse"


def test_unfinished_requests_are_rejected() -> None:
    with pytest.raises(ValueError):
        paired_block_bootstrap(np.array([[0.1, np.nan]]), np.array([0.1, 0.2]), slo_s=1.0,
                               num_blocks=2)
