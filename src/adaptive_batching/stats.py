"""Paired block bootstrap for comparing a learned policy with a baseline.

Requests are grouped into consecutive blocks of equal request count. A
bootstrap replicate resamples blocks with replacement, the same blocks for
both policies (paired), pools the requests of the chosen blocks and computes
the metric. For the learned policy, which exists as several training seeds,
the seeds are resampled as well and the metric is averaged over them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

LATENCY_MARGIN = 0.05  # +/-5% relative difference counts as equivalent
SLO_MARGIN_POINTS = 1.0  # +/-1 percentage point counts as equivalent


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    verdict: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def block_ids(num_requests: int, num_blocks: int) -> NDArray[np.int64]:
    """Block index of each request: consecutive blocks of (almost) equal count."""
    if not 0 < num_blocks <= num_requests:
        raise ValueError("need 1 <= num_blocks <= num_requests")
    return (np.arange(num_requests) * num_blocks) // num_requests


def classify(low: float, high: float, margin: float) -> str:
    """Pre-registered verdict from the position of a confidence interval."""
    if high < 0:
        return "rl_better"
    if low > 0:
        return "rl_worse"
    if -margin <= low and high <= margin:
        return "equivalent"
    return "inconclusive"


def condition_verdict(latency: str, slo: str) -> str:
    """Pre-registered verdict for a condition from its two metric verdicts."""
    if latency == "rl_better":
        return "trade_off" if slo == "rl_worse" else "rl_better"
    if latency == "rl_worse" or slo == "rl_worse":
        return "rl_worse"
    return "no_difference"


def paired_block_bootstrap(
    learned_latency_s: NDArray[np.float64],
    baseline_latency_s: NDArray[np.float64],
    slo_s: float,
    num_blocks: int = 20,
    num_resamples: int = 10_000,
    seed: int = 20261111,
    confidence: float = 0.95,
) -> dict[str, Interval]:
    """Compare per-request latencies of a learned policy and a baseline.

    ``learned_latency_s`` has shape ``(seeds, requests)``;
    ``baseline_latency_s`` has shape ``(requests,)``; both in arrival order
    over the same requests. Returns intervals for the relative difference in
    mean latency and the difference in SLO violation rate (percentage
    points), learned minus baseline.
    """
    learned = np.atleast_2d(np.asarray(learned_latency_s, dtype=np.float64))
    baseline = np.asarray(baseline_latency_s, dtype=np.float64)
    if learned.shape[1] != baseline.shape[0]:
        raise ValueError("both policies must cover the same requests")
    if np.isnan(learned).any() or np.isnan(baseline).any():
        raise ValueError("every request must have completed")

    seeds, requests = learned.shape
    blocks = block_ids(requests, num_blocks)
    counts = np.bincount(blocks, minlength=num_blocks).astype(np.float64)

    def block_sums(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.bincount(blocks, weights=values, minlength=num_blocks)

    def effects(
        base_lat: NDArray[np.float64],
        base_late: NDArray[np.float64],
        rl_lat: NDArray[np.float64],
        rl_late: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (rl_lat - base_lat) / base_lat, 100.0 * (rl_late - base_late)

    base_lat_blocks = block_sums(baseline)
    base_late_blocks = block_sums((baseline > slo_s).astype(np.float64))
    rl_lat_blocks = np.stack([block_sums(row) for row in learned])
    rl_late_blocks = np.stack([block_sums((row > slo_s).astype(np.float64)) for row in learned])

    point = effects(
        np.array(base_lat_blocks.sum() / requests),
        np.array(base_late_blocks.sum() / requests),
        np.array(rl_lat_blocks.sum(axis=1).mean() / requests),
        np.array(rl_late_blocks.sum(axis=1).mean() / requests),
    )

    rng = np.random.default_rng(seed)
    chosen_blocks = rng.integers(0, num_blocks, size=(num_resamples, num_blocks))
    chosen_seeds = rng.integers(0, seeds, size=(num_resamples, seeds))
    totals = counts[chosen_blocks].sum(axis=1)

    def resampled_baseline(sums: NDArray[np.float64]) -> NDArray[np.float64]:
        return sums[chosen_blocks].sum(axis=1) / totals

    def resampled_learned(sums: NDArray[np.float64]) -> NDArray[np.float64]:
        per_seed = sums[:, chosen_blocks].sum(axis=2) / totals  # (seeds, resamples)
        picked = np.take_along_axis(per_seed.T, chosen_seeds, axis=1)
        return picked.mean(axis=1)

    boot = effects(
        resampled_baseline(base_lat_blocks),
        resampled_baseline(base_late_blocks),
        resampled_learned(rl_lat_blocks),
        resampled_learned(rl_late_blocks),
    )
    tail = 100.0 * (1.0 - confidence) / 2.0
    out = {}
    for name, estimate, samples, margin in (
        ("mean_latency_relative_difference", point[0], boot[0], LATENCY_MARGIN),
        ("slo_violation_difference_points", point[1], boot[1], SLO_MARGIN_POINTS),
    ):
        low, high = np.percentile(samples, [tail, 100.0 - tail])
        out[name] = Interval(float(estimate), float(low), float(high),
                             classify(float(low), float(high), margin))
    return out
