from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_batching.service_time import (
    BATCH_SIZES,
    BERT_BASE_T4_FP16,
    ServiceTimeModel,
    fit_two_parameter,
    load_measurements,
)

MEASUREMENTS = (
    Path(__file__).resolve().parent.parent / "data/public_measurements/nvidia_bert_tf_t4_fp16.csv"
)
MODEL = ServiceTimeModel(fixed_s=0.008, per_token_s=2e-5)


def test_larger_batches_take_longer_but_cost_less_per_request() -> None:
    # The lesson from the original project: a batch must cost time.
    times = [MODEL.uniform_batch_time_s(b, 384) for b in BATCH_SIZES]
    per_request = [t / b for t, b in zip(times, BATCH_SIZES, strict=True)]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert per_request == sorted(per_request, reverse=True)


def test_batch_is_padded_to_its_longest_request() -> None:
    mixed = MODEL.batch_time_s([10, 200, 50])
    assert mixed == pytest.approx(MODEL.uniform_batch_time_s(3, 200))
    assert mixed > MODEL.batch_time_s([10, 10, 50])


def test_lengths_are_truncated_to_max_length() -> None:
    assert MODEL.request_length(7000) == 384
    assert MODEL.batch_time_s([7000]) == pytest.approx(MODEL.uniform_batch_time_s(1, 384))


def test_fixed_length_ablation_ignores_tokens() -> None:
    fixed = ServiceTimeModel(fixed_s=0.008, per_token_s=2e-5, fixed_length=128)
    assert fixed.batch_time_s([5, 7000]) == pytest.approx(fixed.uniform_batch_time_s(2, 128))
    assert fixed.reference_length() == 128


def test_batching_gain_definitions() -> None:
    # gain(b) = throughput(b) / throughput(1); its limit is s(1) / per-request cost.
    assert MODEL.batching_gain(1, 384) == pytest.approx(1.0)
    limit = MODEL.asymptotic_batching_gain(384)
    assert MODEL.batching_gain(32, 384) < MODEL.batching_gain(10_000, 384) < limit
    assert MODEL.batching_gain(10_000_000, 384) == pytest.approx(limit, rel=1e-4)


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError):
        MODEL.batch_time_s([])


def test_shipped_constants_match_a_refit_of_the_public_measurements() -> None:
    fit = fit_two_parameter(load_measurements(MEASUREMENTS, "base"))
    assert fit.n_points == 8
    assert BERT_BASE_T4_FP16.fixed_s == pytest.approx(fit.fixed_s, rel=1e-3)
    assert BERT_BASE_T4_FP16.per_token_s == pytest.approx(fit.per_token_s, rel=1e-3)
    # The model reproduces every published point to within 8%.
    assert fit.max_rel_error < 0.08
