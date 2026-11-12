from __future__ import annotations

import math

import pytest

from adaptive_batching.engine import simulate
from adaptive_batching.policies import DownstreamAwareBatch, GreedyBatch, ThresholdBatch
from adaptive_batching.service_time import ServiceTimeModel

# s(b) = 10 ms + 1 ms * b
MODEL = ServiceTimeModel(fixed_s=0.010, per_token_s=1e-5, fixed_length=100)


def batches(result: object, stage: int = 0) -> list[tuple[float, int]]:
    return [(round(b.start_s, 6), b.size) for b in result.batches if b.stage == stage]  # type: ignore[attr-defined]


def test_threshold_waits_for_k_then_serves_everything_queued() -> None:
    policy = ThresholdBatch(threshold=3, max_wait_s=1.0, max_batch=32)
    result = simulate([0.0, 0.001, 0.002, 0.003], [1] * 4, [MODEL], policy)
    # Third request reaches the threshold at 0.002; the fourth waits and then
    # leaves alone when its own max wait expires (1.003).
    assert batches(result) == [(0.002, 3), (1.003, 1)]


def test_threshold_serves_whole_backlog_up_to_max_batch() -> None:
    policy = ThresholdBatch(threshold=2, max_wait_s=1.0, max_batch=4)
    result = simulate([0.0] * 7, [1] * 7, [MODEL], policy)
    assert [size for _, size in batches(result)] == [4, 3]


def test_threshold_respects_max_wait() -> None:
    policy = ThresholdBatch(threshold=5, max_wait_s=0.050, max_batch=32)
    result = simulate([0.0, 0.010], [1, 1], [MODEL], policy)
    assert batches(result) == [(0.050, 2)]


def test_threshold_one_is_greedy() -> None:
    arrivals = [0.0, 0.001, 0.004, 0.02, 0.021, 0.05]
    a = simulate(arrivals, [1] * 6, [MODEL], ThresholdBatch(1, 0.5, 8))
    b = simulate(arrivals, [1] * 6, [MODEL], GreedyBatch(8))
    assert a.batches == b.batches


def test_downstream_aware_equals_threshold_without_hold() -> None:
    arrivals = [0.0, 0.001, 0.004, 0.02, 0.021, 0.05]
    stages = [MODEL, MODEL]
    a = simulate(arrivals, [1] * 6, stages, DownstreamAwareBatch(2, 0.03, 8, hold_at=math.inf))
    b = simulate(arrivals, [1] * 6, stages, ThresholdBatch(2, 0.03, 8))
    assert a.batches == b.batches


def test_downstream_aware_holds_while_next_stage_is_backed_up() -> None:
    slow = ServiceTimeModel(fixed_s=0.100, per_token_s=1e-5, fixed_length=100)
    upstream = DownstreamAwareBatch(threshold=1, max_wait_s=10.0, max_batch=32, hold_at=1)
    downstream = GreedyBatch(1)
    arrivals = [0.0, 0.001, 0.002, 0.003, 0.004]
    result = simulate(arrivals, [1] * 5, [MODEL, slow], [upstream, downstream])
    # Batch 1 = request 0. Batch 2 = requests 1-4, sent at 0.011 while the
    # next stage is busy but its queue is empty. Had a request been queued
    # downstream, stage 0 would have held.
    assert batches(result, 0) == [(0.0, 1), (0.011, 4)]

    held = simulate(
        [0.0, 0.001, 0.030, 0.031], [1] * 4, [MODEL, slow], [upstream, downstream]
    )
    # Requests 0 and 1 go down in two batches; request 1 queues downstream
    # (stage 1 is busy with request 0 until 0.011 + 0.101 = 0.112), so requests
    # 2 and 3 are held upstream until stage 1 takes request 1 at 0.112.
    assert batches(held, 0) == [(0.0, 1), (0.011, 1), (pytest.approx(0.112), 2)]
