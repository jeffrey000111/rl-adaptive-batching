"""Behaviour the simulator must show on the real trace.

These tests encode what went wrong in the original project: there, batches
were free, so the largest batch always won and no trade-off existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_batching.engine import Policy, simulate
from adaptive_batching.load import offered_load, time_scale_for_load
from adaptive_batching.metrics import summarize
from adaptive_batching.policies import GreedyBatch, WaitForBatch
from adaptive_batching.service_time import BATCH_SIZES, BERT_BASE_T4_FP16
from adaptive_batching.traces import Trace, load_trace

DATA = Path(__file__).resolve().parent.parent / "data"
STAGES = [BERT_BASE_T4_FP16]
MAX_BATCH = BATCH_SIZES[-1]


@pytest.fixture(scope="module")
def train() -> Trace:
    return load_trace(DATA / "AzureLLMInferenceTrace_conv.csv", "conv").split()["train"]


def mean_latency(train: Trace, load: float, policy: Policy) -> float:
    scaled = train.time_scaled(time_scale_for_load(train, load, STAGES, MAX_BATCH))
    result = simulate(scaled.relative_arrivals(), scaled.context_tokens, STAGES, policy)
    return summarize(result).mean_latency_s


def test_time_scaling_hits_the_target_load(train: Trace) -> None:
    scaled = train.time_scaled(time_scale_for_load(train, 0.7, STAGES, MAX_BATCH))
    assert offered_load(scaled, STAGES, MAX_BATCH) == pytest.approx(0.7)


def test_no_batching_is_clearly_not_optimal_at_high_load(train: Trace) -> None:
    single = mean_latency(train, 0.9, GreedyBatch(1))
    best = min(mean_latency(train, 0.9, GreedyBatch(b)) for b in BATCH_SIZES)
    assert single > 5 * best


def test_always_waiting_for_the_largest_batch_is_clearly_not_optimal_at_moderate_load(
    train: Trace,
) -> None:
    by_size = {b: mean_latency(train, 0.5, WaitForBatch(b, 1.0)) for b in BATCH_SIZES}
    assert by_size[MAX_BATCH] > 5 * min(by_size.values())
    # Neither extreme is the best full-batch size: the curve is U-shaped.
    best = min(by_size, key=by_size.__getitem__)
    assert BATCH_SIZES[0] < best < MAX_BATCH


def test_best_full_batch_size_grows_with_load(train: Trace) -> None:
    def best_size(load: float) -> int:
        return min(BATCH_SIZES, key=lambda b: mean_latency(train, load, WaitForBatch(b, 1.0)))

    assert best_size(0.5) < best_size(0.7) < best_size(0.9)
