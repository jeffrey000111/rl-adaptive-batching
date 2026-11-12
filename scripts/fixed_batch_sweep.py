"""Latency of fixed-batch-size policies at several load levels, per scenario.

For each trace and nominal load, runs two families of fixed-size policies on
the TRAIN split (never dev or test):

* greedy:    dispatch immediately, at most b requests per batch
* full:      wait until b requests are queued (or the oldest has waited
             FULL_BATCH_MAX_WAIT_S), then dispatch

In the pipeline scenario every stage uses the same policy. Writes
results/fixed_batch_sweep/train.json. Deterministic; takes seconds.

Usage:
    python scripts/fixed_batch_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_batching.engine import Policy, simulate
from adaptive_batching.load import capacity_rps, time_scale_for_load
from adaptive_batching.metrics import slo_violation_rate, summarize
from adaptive_batching.policies import GreedyBatch, WaitForBatch
from adaptive_batching.scenarios import LOADS, MAX_BATCH, SCENARIOS, Scenario
from adaptive_batching.service_time import BATCH_SIZES, MEASURED_MAX_BATCH
from adaptive_batching.traces import Trace, load_trace

ROOT = Path(__file__).resolve().parent.parent
TRACES = ("conv", "code")
# Not tuned: this sweep is a sanity check of the simulator, not a baseline.
FULL_BATCH_MAX_WAIT_S = 1.0
PEAK_WINDOW_S = 60.0  # in original trace time


def peak_to_mean_rate(trace: Trace) -> float:
    """Busiest PEAK_WINDOW_S window's arrival rate divided by the mean rate."""
    return max(s.mean_rate for s in trace.slices(PEAK_WINDOW_S)) / trace.mean_rate


def run(trace: Trace, scenario: Scenario, policy: Policy) -> dict[str, object]:
    result = simulate(trace.relative_arrivals(), trace.context_tokens, scenario.stages, policy)
    metrics = summarize(result, slo_s=scenario.slo_s).to_dict()
    metrics["tight_slo_violation_rate"] = slo_violation_rate(result, scenario.tight_slo_s)
    return metrics


def main() -> None:
    families = {
        "greedy": lambda b: GreedyBatch(b),
        "full": lambda b: WaitForBatch(b, FULL_BATCH_MAX_WAIT_S),
    }
    cells = []
    traces = {
        name: load_trace(ROOT / f"data/AzureLLMInferenceTrace_{name}.csv", name).split()["train"]
        for name in TRACES
    }
    for scenario in SCENARIOS.values():
        for name, train in traces.items():
            peak_ratio = peak_to_mean_rate(train)
            for load in LOADS:
                factor = time_scale_for_load(train, load, scenario.stages, MAX_BATCH)
                scaled = train.time_scaled(factor)
                cell = {
                    "scenario": scenario.name,
                    "trace": name,
                    "split": "train",
                    "nominal_load": load,
                    "peak_load": load * peak_ratio,
                    "requests": len(scaled),
                    "duration_s": scaled.duration_s,
                    "arrival_rate_rps": scaled.mean_rate,
                    "policies": {
                        family: {str(b): run(scaled, scenario, make(b)) for b in BATCH_SIZES}
                        for family, make in families.items()
                    },
                }
                cells.append(cell)
                best = {
                    family: min(runs, key=lambda b, r=runs: r[b]["mean_latency_s"])
                    for family, runs in cell["policies"].items()
                }
                print(f"{scenario.name} {name} load={load}: best mean latency at b={best}")

    out = ROOT / "results/fixed_batch_sweep/train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "scenarios": {
                    s.name: {
                        "description": s.description,
                        "capacity_rps": capacity_rps(s.stages, MAX_BATCH),
                        "slo_s": s.slo_s,
                        "tight_slo_s": s.tight_slo_s,
                    }
                    for s in SCENARIOS.values()
                },
                "batch_sizes": list(BATCH_SIZES),
                "measured_max_batch": MEASURED_MAX_BATCH,
                "full_batch_max_wait_s": FULL_BATCH_MAX_WAIT_S,
                "peak_window_s": PEAK_WINDOW_S,
                "cells": cells,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
