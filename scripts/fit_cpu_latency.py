"""Fit the processing-time model to the CPU measurements (post-hoc check).

Two fits of ``time = fixed + per_token * batch_size * length``:

* all points, to see how well the two-parameter shape describes them;
* only batch sizes <= 8, then predict 16 and 32: this mirrors how the
  simulator extrapolates the published T4 data, which stops at 8.

As a secondary check, a straight line in batch size is fitted per sequence
length on batch sizes <= 8 and extrapolated the same way, which tests
linearity in batch size without the length coupling.

Writes results/real_latency/fit.json.

Usage:
    python scripts/fit_cpu_latency.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adaptive_batching.service_time import MEASURED_MAX_BATCH, fit_two_parameter

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results/real_latency/cpu_bert_base.json"
OUT = ROOT / "results/real_latency/fit.json"


def predict(fixed: float, per_token: float, batch: int, length: int) -> float:
    return fixed + per_token * batch * length


def main() -> None:
    data = json.loads(SOURCE.read_text())
    points = [(m["batch_size"], m["seq_len"], m["median_s"]) for m in data["measurements"]]

    full = fit_two_parameter(points)
    inside = [p for p in points if p[0] <= MEASURED_MAX_BATCH]
    outside = [p for p in points if p[0] > MEASURED_MAX_BATCH]
    small = fit_two_parameter(inside)

    rows = []
    for batch, length, observed in points:
        full_pred = predict(full.fixed_s, full.per_token_s, batch, length)
        small_pred = predict(small.fixed_s, small.per_token_s, batch, length)
        rows.append({
            "batch_size": batch,
            "seq_len": length,
            "measured_s": observed,
            "full_fit_s": full_pred,
            "full_fit_rel_error": full_pred / observed - 1,
            "small_batch_fit_s": small_pred,
            "small_batch_fit_rel_error": small_pred / observed - 1,
            "extrapolated": batch > MEASURED_MAX_BATCH,
        })

    per_length = {}
    for length in sorted({p[1] for p in points}):
        xs = np.array([b for b, s_len, _ in inside if s_len == length], dtype=float)
        ys = np.array([t for _, s_len, t in inside if s_len == length])
        slope, intercept = np.polyfit(xs, ys, 1)
        errors = {
            str(b): float((intercept + slope * b) / t - 1)
            for b, s_len, t in outside
            if s_len == length
        }
        per_length[str(length)] = {
            "fixed_s": float(intercept),
            "per_request_s": float(slope),
            "extrapolation_rel_error_by_batch": errors,
        }

    def max_abs(key: str, extrapolated: bool | None = None) -> float:
        chosen = [r for r in rows if extrapolated is None or r["extrapolated"] == extrapolated]
        return float(max(abs(r[key]) for r in chosen))

    record = {
        "exploratory": True,
        "measured_max_batch_for_extrapolation": MEASURED_MAX_BATCH,
        "full_fit": {
            "fixed_s": full.fixed_s, "per_token_s": full.per_token_s,
            "n_points": full.n_points, "r_squared": full.r_squared,
            "max_abs_rel_error": max_abs("full_fit_rel_error"),
        },
        "small_batch_fit": {
            "fixed_s": small.fixed_s, "per_token_s": small.per_token_s,
            "n_points": small.n_points, "r_squared": small.r_squared,
            "max_abs_rel_error_inside": max_abs("small_batch_fit_rel_error", False),
            "max_abs_rel_error_extrapolated": max_abs("small_batch_fit_rel_error", True),
        },
        "per_length_linear_in_batch": per_length,
        "points": rows,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "points"}, indent=2))
    for r in rows:
        print(f"b={r['batch_size']:2d} L={r['seq_len']} measured {1e3 * r['measured_s']:8.1f} ms  "
              f"full {100 * r['full_fit_rel_error']:+6.1f}%  b<=8 fit "
              f"{100 * r['small_batch_fit_rel_error']:+6.1f}%")


if __name__ == "__main__":
    main()
