from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaptive_batching.traces import Trace, load_trace

DATA = Path(__file__).resolve().parent.parent / "data"


def make_trace(arrivals: list[float], end_s: float = 100.0) -> Trace:
    n = len(arrivals)
    return Trace(
        name="t",
        arrival_s=np.array(arrivals, dtype=np.float64),
        context_tokens=np.arange(n, dtype=np.int64) + 100,
        generated_tokens=np.arange(n, dtype=np.int64) + 10,
        start_s=0.0,
        end_s=end_s,
    )


def test_split_is_a_partition_in_time_order() -> None:
    rng = np.random.default_rng(0)
    trace = make_trace(sorted(rng.uniform(0, 100, size=500).tolist()))
    parts = trace.split((0.6, 0.2, 0.2))

    assert sum(len(p) for p in parts.values()) == len(trace)
    rejoined = np.concatenate([parts[s].arrival_s for s in ("train", "dev", "test")])
    np.testing.assert_array_equal(rejoined, trace.arrival_s)
    assert parts["train"].end_s == parts["dev"].start_s
    assert parts["dev"].end_s == parts["test"].start_s
    assert parts["train"].arrival_s.max() < parts["dev"].arrival_s.min()
    assert parts["dev"].arrival_s.max() < parts["test"].arrival_s.min()


def test_window_is_half_open() -> None:
    trace = make_trace([10.0, 20.0, 30.0])
    window = trace.window(10.0, 30.0)
    assert window.arrival_s.tolist() == [10.0, 20.0]


def test_split_keeps_tokens_aligned_with_arrivals() -> None:
    trace = make_trace([1.0, 50.0, 70.0, 90.0])
    parts = trace.split((0.6, 0.2, 0.2))
    assert parts["dev"].arrival_s.tolist() == [70.0]
    assert parts["dev"].context_tokens.tolist() == [102]
    assert parts["dev"].generated_tokens.tolist() == [12]


def test_slices_are_equal_length_non_overlapping_and_keep_idle_slices() -> None:
    trace = make_trace([1.0, 2.0, 35.0], end_s=45.0)
    slices = trace.slices(10.0)
    assert [(s.start_s, s.end_s) for s in slices] == [(0, 10), (10, 20), (20, 30), (30, 40)]
    assert [len(s) for s in slices] == [2, 0, 0, 1]  # remainder [40, 45) dropped


def test_time_scaling_changes_rate_but_not_requests() -> None:
    trace = make_trace([10.0, 20.0, 40.0])
    fast = trace.time_scaled(0.5)
    assert fast.arrival_s.tolist() == [5.0, 10.0, 20.0]
    assert fast.duration_s == pytest.approx(trace.duration_s * 0.5)
    assert fast.mean_rate == pytest.approx(trace.mean_rate * 2)
    np.testing.assert_array_equal(fast.context_tokens, trace.context_tokens)


def test_relative_arrivals_start_at_window_start() -> None:
    window = make_trace([10.0, 25.0, 30.0]).window(20.0, 40.0)
    assert window.relative_arrivals().tolist() == [5.0, 10.0]


def test_arrays_are_read_only() -> None:
    trace = make_trace([1.0, 2.0])
    with pytest.raises(ValueError):
        trace.arrival_s[0] = 99.0


@pytest.mark.parametrize(
    "arrivals, end_s",
    [([2.0, 1.0], 100.0), ([1.0, 100.0], 100.0)],
    ids=["unsorted", "arrival-at-window-end"],
)
def test_invalid_traces_are_rejected(arrivals: list[float], end_s: float) -> None:
    with pytest.raises(ValueError):
        make_trace(arrivals, end_s=end_s)


def test_bad_split_fractions_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_trace([1.0]).split((0.5, 0.2, 0.2))


@pytest.mark.parametrize("name", ["code", "conv"])
def test_real_traces_load_and_split(name: str) -> None:
    trace = load_trace(DATA / f"AzureLLMInferenceTrace_{name}.csv", name=name)
    assert len(trace) > 1000
    assert trace.arrival_s[0] == 0.0
    assert trace.context_tokens.min() > 0 and trace.generated_tokens.min() > 0
    parts = trace.split()
    assert sum(len(p) for p in parts.values()) == len(trace)
    assert all(len(p) > 0 for p in parts.values())
