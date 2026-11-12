"""Experimental conditions: (scenario, trace, load) and their scaled data splits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from adaptive_batching.engine import Policy, SimulationResult, simulate
from adaptive_batching.load import offered_load, time_scale_for_load
from adaptive_batching.scenarios import LOADS, MAX_BATCH, SCENARIOS, Scenario
from adaptive_batching.traces import Trace, load_trace

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRACES = ("conv", "code")
PEAK_WINDOW_S = 60.0  # on the unscaled trace clock


@dataclass(frozen=True)
class Condition:
    scenario_name: str
    trace: str
    load: float

    @property
    def scenario(self) -> Scenario:
        return SCENARIOS[self.scenario_name]

    @property
    def id(self) -> str:
        return f"{self.scenario_name}-{self.trace}-load{self.load}"


# Pre-registered grouping (docs/preregistration.md, Section 4).
PRIMARY: tuple[Condition, ...] = tuple(
    Condition(s, "conv", load) for s in SCENARIOS for load in (0.5, 0.7, 0.9)
)
SECONDARY: tuple[Condition, ...] = tuple(
    [Condition(s, "conv", 0.3) for s in SCENARIOS]
    + [Condition(s, "code", load) for s in SCENARIOS for load in LOADS]
)
PRIMARY_COMPARISON = Condition("pipeline", "conv", 0.7)
GROUPS: dict[str, tuple[Condition, ...]] = {"primary": PRIMARY, "secondary": SECONDARY}
BY_ID: dict[str, Condition] = {c.id: c for c in PRIMARY + SECONDARY}


@cache
def _raw_splits(trace: str) -> dict[str, Trace]:
    return load_trace(DATA_DIR / f"AzureLLMInferenceTrace_{trace}.csv", trace).split()


@cache
def load_splits(condition: Condition) -> dict[str, Trace]:
    """Train / dev / test of the condition's trace, time-scaled to its load.

    The scaling factor comes from the train split alone and is applied
    unchanged to dev and test.
    """
    raw = _raw_splits(condition.trace)
    stages = condition.scenario.stages
    factor = time_scale_for_load(raw["train"], condition.load, stages, MAX_BATCH)
    return {name: split.time_scaled(factor) for name, split in raw.items()}


def split_loads(condition: Condition) -> dict[str, dict[str, float]]:
    """Realised load and busiest-minute load of every split of a condition."""
    raw = _raw_splits(condition.trace)
    stages = condition.scenario.stages
    out = {}
    for name, split in load_splits(condition).items():
        peak_ratio = max(s.mean_rate for s in raw[name].slices(PEAK_WINDOW_S)) / raw[name].mean_rate
        realised = offered_load(split, stages, MAX_BATCH)
        out[name] = {"requests": len(split), "load": realised, "peak_load": realised * peak_ratio}
    return out


def run(trace: Trace, scenario: Scenario, policy: Policy | list[Policy]) -> SimulationResult:
    """Simulate one policy (or one per stage) over a whole split, continuously."""
    return simulate(trace.relative_arrivals(), trace.context_tokens, scenario.stages, policy)


def max_in_system(result: SimulationResult) -> int:
    """Largest number of requests simultaneously in the system during a run."""
    times = np.concatenate([result.arrival_s, result.completion_s])
    deltas = np.concatenate([np.ones(len(result.arrival_s)), -np.ones(len(result.arrival_s))])
    # At equal times process completions first, so the count is not overstated.
    order = np.lexsort((deltas, times))
    return int(np.cumsum(deltas[order]).max())


def stress_test_split(condition: Condition) -> Trace:
    """Test split re-scaled by its own arrival rate to the condition's nominal load.

    Post-hoc overload stress test (pre-registration, Deviation 3). The
    registered protocol scales test with the factor computed on train, which
    leaves test lighter than nominal because the trace's rate falls at its end.
    """
    raw = _raw_splits(condition.trace)["test"]
    stages = condition.scenario.stages
    return raw.time_scaled(time_scale_for_load(raw, condition.load, stages, MAX_BATCH))
