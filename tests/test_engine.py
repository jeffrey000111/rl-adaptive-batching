from __future__ import annotations

import numpy as np
import pytest

from adaptive_batching.engine import (
    Dispatch,
    Engine,
    SimulationStalledError,
    StageView,
    Wait,
    simulate,
)
from adaptive_batching.metrics import summarize
from adaptive_batching.policies import GreedyBatch, WaitForBatch
from adaptive_batching.service_time import ServiceTimeModel

# 10 ms fixed + 1 ms per request at length 100: s(1) = 11 ms, s(2) = 12 ms, s(4) = 14 ms.
MODEL = ServiceTimeModel(fixed_s=0.010, per_token_s=1e-5, fixed_length=100)
SLOW = ServiceTimeModel(fixed_s=0.020, per_token_s=2e-5, fixed_length=100)


def poisson_arrivals(rate: float, n: int, seed: int) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).exponential(1 / rate, size=n))


def test_processing_occupies_the_server() -> None:
    # Two requests at t=0, no batching: the second must wait for the first.
    result = simulate([0.0, 0.0], [1, 1], [MODEL], GreedyBatch(1))
    assert result.latency_s.tolist() == pytest.approx([0.011, 0.022])


def test_a_batch_finishes_together_and_takes_longer_than_a_single_request() -> None:
    result = simulate([0.0, 0.0], [1, 1], [MODEL], GreedyBatch(2))
    assert result.latency_s.tolist() == pytest.approx([0.012, 0.012])


def test_latency_is_queueing_plus_processing() -> None:
    # Request 1 arrives while request 0 is being processed (busy until 0.011).
    result = simulate([0.0, 0.005], [1, 1], [MODEL], GreedyBatch(8))
    waited = result.start_s[0] - result.enqueue_s[0]
    served = result.finish_s[0] - result.start_s[0]
    assert waited.tolist() == pytest.approx([0.0, 0.006])
    assert served.tolist() == pytest.approx([0.011, 0.011])
    assert result.latency_s.tolist() == pytest.approx((waited + served).tolist())


def test_requests_arriving_during_a_batch_form_the_next_batch() -> None:
    arrivals = [0.0, 0.001, 0.002, 0.003]
    result = simulate(arrivals, [1] * 4, [MODEL], GreedyBatch(8))
    assert [b.size for b in result.batches] == [1, 3]
    assert result.batches[1].start_s == pytest.approx(0.011)


def test_wait_for_batch_dispatches_when_full_or_when_the_oldest_request_times_out() -> None:
    policy = WaitForBatch(batch_size=2, max_wait_s=0.050)
    # Requests 0 and 1 fill a batch at t=0.010. Request 2 is alone and leaves at its timeout.
    result = simulate([0.0, 0.010, 0.100], [1, 1, 1], [MODEL], policy)
    assert [(b.start_s, b.size) for b in result.batches] == [
        (pytest.approx(0.010), 2),
        (pytest.approx(0.150), 1),
    ]


def test_pipeline_latency_adds_up_over_stages_in_fifo_order() -> None:
    result = simulate([0.0, 0.0], [1, 1], [MODEL, SLOW], GreedyBatch(1))
    # Stage 0 finishes at 0.011 and 0.022; stage 1 takes 0.022 per request.
    assert result.finish_s[0].tolist() == pytest.approx([0.011, 0.022])
    assert result.finish_s[1].tolist() == pytest.approx([0.033, 0.055])
    assert result.num_stages == 2


@pytest.mark.parametrize("policy", [GreedyBatch(1), GreedyBatch(8), WaitForBatch(4, 0.02)])
def test_requests_are_conserved_at_every_decision_point(policy: object) -> None:
    arrivals = poisson_arrivals(rate=60.0, n=400, seed=1)
    engine = Engine(arrivals, [100] * 400, [MODEL, SLOW, MODEL])
    decisions = 0
    while (view := engine.next_decision()) is not None:
        assert engine.arrived == engine.completed + engine.queued + engine.in_service
        engine.apply(policy.decide(view))  # type: ignore[attr-defined]
        assert engine.arrived == engine.completed + engine.queued + engine.in_service
        decisions += 1
    assert decisions > 0
    assert engine.arrived == engine.completed == 400
    assert engine.queued == engine.in_service == 0
    assert not np.isnan(engine.result().latency_s).any()


def test_identical_inputs_give_identical_results() -> None:
    arrivals = poisson_arrivals(rate=80.0, n=500, seed=7)
    tokens = np.random.default_rng(7).integers(1, 400, size=500)
    model = ServiceTimeModel(fixed_s=0.008, per_token_s=2e-5)
    first = simulate(arrivals, tokens, [model, model], WaitForBatch(8, 0.01))
    second = simulate(arrivals, tokens, [model, model], WaitForBatch(8, 0.01))
    np.testing.assert_array_equal(first.finish_s, second.finish_s)
    assert first.batches == second.batches


def test_server_never_runs_two_batches_at_once() -> None:
    arrivals = poisson_arrivals(rate=200.0, n=300, seed=3)
    result = simulate(arrivals, [100] * 300, [MODEL, SLOW], GreedyBatch(4))
    for stage in range(2):
        batches = [b for b in result.batches if b.stage == stage]
        for earlier, later in zip(batches, batches[1:], strict=False):
            assert later.start_s >= earlier.start_s + earlier.duration_s - 1e-12


def test_dispatch_size_must_fit_the_queue() -> None:
    engine = Engine([0.0], [1], [MODEL])
    assert engine.next_decision() is not None
    with pytest.raises(ValueError):
        engine.apply(Dispatch(2))


def test_a_policy_that_never_dispatches_is_reported_as_stalled() -> None:
    class NeverDispatch:
        def decide(self, view: StageView) -> Wait:
            return Wait(1.0)

    with pytest.raises(SimulationStalledError):
        simulate([0.0], [1], [MODEL], NeverDispatch(), stall_limit_s=10.0)


def test_view_reports_queue_wait_downstream_and_recent_arrivals() -> None:
    engine = Engine([0.0, 0.001, 0.002], [1, 1, 1], [MODEL, SLOW])
    view = engine.next_decision()
    assert view is not None and (view.stage, view.queue_len) == (0, 1)
    engine.apply(Dispatch(1))  # stage 0 busy until 0.011

    # At t=0.011 request 0 reaches stage 1 and stage 0 becomes free. The
    # downstream stage is asked first.
    view = engine.next_decision()
    assert view is not None and view.stage == 1 and view.now_s == pytest.approx(0.011)
    assert view.queue_len_at(0) == 2
    engine.apply(Dispatch(1))  # stage 1 busy for 0.022

    view = engine.next_decision()
    assert view is not None and view.stage == 0 and view.now_s == pytest.approx(0.011)
    assert view.queue_len == 2 and view.oldest_wait_s == pytest.approx(0.010)
    assert view.queue_len_at(1) == 0  # already dispatched downstream
    assert view.busy_remaining_s(1) == pytest.approx(0.022)
    assert view.arrivals_in_last(0.0095) == 1 and view.arrivals_in_last(1.0) == 3
    assert view.batch_time_s(2) == pytest.approx(0.012)


def test_metrics_on_a_hand_computed_run() -> None:
    result = simulate([0.0, 0.0], [1, 1], [MODEL], GreedyBatch(1))
    metrics = summarize(result, slo_s=0.015)
    assert metrics.requests == metrics.completed == 2
    assert metrics.mean_latency_s == pytest.approx(0.0165)
    assert metrics.slo_violation_rate == pytest.approx(0.5)
    assert metrics.busy_time_s == (pytest.approx(0.022),)
    assert metrics.utilization == (pytest.approx(1.0),)
    assert metrics.throughput_rps == pytest.approx(2 / 0.022)
    assert metrics.mean_batch_size == (1.0,)
    assert summarize(result).slo_violation_rate is None


def test_request_seconds_equals_total_latency_when_done() -> None:
    arrivals = poisson_arrivals(rate=90.0, n=300, seed=5)
    engine = Engine(arrivals, [100] * 300, [MODEL, SLOW])
    policy = WaitForBatch(4, 0.02)
    while (view := engine.next_decision()) is not None:
        engine.apply(policy.decide(view))
    assert engine.request_seconds == pytest.approx(engine.result().latency_s.sum())
