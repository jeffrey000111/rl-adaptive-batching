"""Batch processing-time model.

    batch_time = fixed_s + per_token_s * batch_size * padded_length

``fixed_s`` is paid once per model invocation regardless of batch size;
amortising it over more requests is where the benefit of batching comes
from. The second term is the compute, proportional to the number of tokens
processed: a batch is padded to its longest sequence, so the cost is
``batch_size * max(length)``, not the sum of the lengths.

Parameter provenance is documented in docs/decisions.md (D7) and checked by
tests/test_service_time.py against data/public_measurements/.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ServiceTimeModel:
    """Deterministic processing time of one batch on one server."""

    fixed_s: float
    per_token_s: float
    max_length: int = 384
    # If set, every request is treated as having this length (ablation).
    fixed_length: int | None = None

    def __post_init__(self) -> None:
        if self.fixed_s < 0 or self.per_token_s <= 0 or self.max_length <= 0:
            raise ValueError("fixed_s must be >= 0, per_token_s and max_length must be > 0")

    def request_length(self, context_tokens: int) -> int:
        """Sequence length the model sees for a request: its tokens, truncated."""
        if self.fixed_length is not None:
            return self.fixed_length
        return min(int(context_tokens), self.max_length)

    def batch_time_s(self, context_tokens: Sequence[int]) -> float:
        """Processing time of a batch made of requests with these token counts."""
        if len(context_tokens) == 0:
            raise ValueError("a batch must contain at least one request")
        padded = max(self.request_length(tokens) for tokens in context_tokens)
        return self.uniform_batch_time_s(len(context_tokens), padded)

    def uniform_batch_time_s(self, batch_size: int, length: int) -> float:
        """Processing time of ``batch_size`` requests that all have ``length`` tokens."""
        return self.fixed_s + self.per_token_s * batch_size * length

    def throughput_rps(self, batch_size: int, length: int) -> float:
        """Requests per second of a server that runs back-to-back batches of this size."""
        return batch_size / self.uniform_batch_time_s(batch_size, length)

    def batching_gain(self, batch_size: int, length: int) -> float:
        """Throughput at ``batch_size`` divided by throughput at batch size 1."""
        return self.throughput_rps(batch_size, length) / self.throughput_rps(1, length)

    def asymptotic_batching_gain(self, length: int) -> float:
        """Limit of :meth:`batching_gain` as the batch size grows without bound."""
        return self.uniform_batch_time_s(1, length) / (self.per_token_s * length)

    def reference_length(self) -> int:
        """Length used to define capacity: the longest a request can be."""
        return self.fixed_length if self.fixed_length is not None else self.max_length


@dataclass(frozen=True)
class FitReport:
    """Least-squares fit of the two-parameter model to published measurements."""

    fixed_s: float
    per_token_s: float
    n_points: int
    r_squared: float
    max_rel_error: float


def load_measurements(path: str | Path, model: str) -> list[tuple[int, int, float]]:
    """Rows ``(batch_size, seq_len, p50_latency_s)`` for one model from the CSV."""
    with Path(path).open(newline="") as handle:
        return [
            (int(row["batch_size"]), int(row["seq_len"]), float(row["latency_p50_ms"]) / 1e3)
            for row in csv.DictReader(handle)
            if row["model"] == model
        ]


def fit_two_parameter(rows: Sequence[tuple[int, int, float]]) -> FitReport:
    """Fit ``time = fixed + per_token * batch * length`` by ordinary least squares."""
    tokens = np.array([batch * length for batch, length, _ in rows], dtype=np.float64)
    observed = np.array([seconds for _, _, seconds in rows], dtype=np.float64)
    design = np.column_stack([np.ones_like(tokens), tokens])
    (fixed, per_token), *_ = np.linalg.lstsq(design, observed, rcond=None)
    predicted = design @ np.array([fixed, per_token])
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return FitReport(
        fixed_s=float(fixed),
        per_token_s=float(per_token),
        n_points=len(rows),
        r_squared=1.0 - residual / total,
        max_rel_error=float(np.abs(predicted / observed - 1.0).max()),
    )


# BERT-base, NVIDIA T4 16GB, FP16, TensorFlow 20.06 (see docs/decisions.md D7).
# Rounded output of fit_two_parameter on data/public_measurements/; a test
# re-runs the fit and fails if these constants drift from the data.
BERT_BASE_T4_FP16 = ServiceTimeModel(fixed_s=7.616e-3, per_token_s=2.174e-5)
# BERT-large, same source and conditions.
BERT_LARGE_T4_FP16 = ServiceTimeModel(fixed_s=16.59e-3, per_token_s=6.275e-5)

# Batch sizes a policy may dispatch. The source measures 1-8; 16 and 32 are a
# linear extrapolation of the fitted model.
BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
MEASURED_MAX_BATCH = 8
