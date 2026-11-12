"""Grid search of the rule-based baselines (pre-registered grids, dev split).

A policy configuration is a list with one *spec* per stage. A spec is a
plain dict so that it can be written to JSON and rebuilt later:

    {"family": "greedy", "cap": 8}
    {"family": "full", "batch": 8, "wait_mult": 2}
    {"family": "threshold", "k": 4, "wait_mult": 2}
    {"family": "downstream", "k": 4, "wait_mult": 2, "hold_at": 8}

``wait_mult`` is the maximum queue delay in units of the stage's
single-request processing time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from adaptive_batching.engine import Policy
from adaptive_batching.policies import (
    DownstreamAwareBatch,
    GreedyBatch,
    ThresholdBatch,
    WaitForBatch,
)
from adaptive_batching.scenarios import MAX_BATCH, Scenario
from adaptive_batching.service_time import BATCH_SIZES, ServiceTimeModel

Spec = dict[str, Any]
Score = tuple[float, float]  # (mean latency, SLO violation rate), lower is better

FAMILIES = ("greedy", "full", "threshold", "downstream")
WAIT_MULTIPLIERS = (0.5, 1, 2, 4, 8, 16)
THRESHOLDS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
HOLD_AT = (1, 2, 4, 8, 16, 32)


def build_policy(spec: Spec, model: ServiceTimeModel) -> Policy:
    """Instantiate the policy a spec describes for a stage with this model."""
    family = spec["family"]
    if family == "greedy":
        return GreedyBatch(spec["cap"])
    max_wait_s = spec["wait_mult"] * model.uniform_batch_time_s(1, model.reference_length())
    if family == "full":
        return WaitForBatch(spec["batch"], max_wait_s)
    if family == "threshold":
        return ThresholdBatch(spec["k"], max_wait_s, MAX_BATCH)
    if family == "downstream":
        return DownstreamAwareBatch(spec["k"], max_wait_s, MAX_BATCH, spec["hold_at"])
    raise ValueError(f"unknown family {family!r}")


def build_policies(config: Sequence[Spec], scenario: Scenario) -> list[Policy]:
    return [build_policy(spec, model) for spec, model in zip(config, scenario.stages, strict=True)]


def family_grid(family: str) -> list[Spec]:
    """Every pre-registered parameter combination of one family."""
    if family == "greedy":
        return [{"family": family, "cap": cap} for cap in BATCH_SIZES]
    if family == "full":
        return [
            {"family": family, "batch": b, "wait_mult": w}
            for b in BATCH_SIZES
            for w in WAIT_MULTIPLIERS
        ]
    if family == "threshold":
        return [
            {"family": family, "k": k, "wait_mult": w} for k in THRESHOLDS for w in WAIT_MULTIPLIERS
        ]
    if family == "downstream":
        return [
            {"family": family, "k": k, "wait_mult": w, "hold_at": h}
            for k in THRESHOLDS
            for w in WAIT_MULTIPLIERS
            for h in HOLD_AT
        ]
    raise ValueError(f"unknown family {family!r}")


def _rank(score: Score, spec_families: Sequence[str], index: int) -> tuple[Any, ...]:
    # Ties: lower SLO violation, then simpler family, then grid order.
    return (score[0], score[1], max(FAMILIES.index(f) for f in spec_families), index)


def search(
    scenario: Scenario, evaluate: Callable[[list[Spec]], Score]
) -> dict[str, Any]:
    """Find the strongest baseline configuration for one condition.

    ``evaluate`` maps a per-stage configuration to (mean latency, SLO
    violation rate) on the dev split. Single stage: exhaustive. Pipeline:
    exhaustive with the same spec at all stages, then coordinate descent
    over stages (bottleneck first, two passes) over the union of all grids.
    """
    num_stages = len(scenario.stages)
    families = FAMILIES if num_stages > 1 else FAMILIES[:-1]
    evaluated = 0
    best_by_family: dict[str, dict[str, Any]] = {}
    best: tuple[tuple[Any, ...], list[Spec], Score] | None = None

    for family in families:
        family_best: tuple[tuple[Any, ...], list[Spec], Score] | None = None
        for index, spec in enumerate(family_grid(family)):
            config = [spec] * num_stages
            score = evaluate(config)
            evaluated += 1
            candidate = (_rank(score, [family], index), config, score)
            if family_best is None or candidate[0] < family_best[0]:
                family_best = candidate
        assert family_best is not None
        best_by_family[family] = _record(family_best[1], family_best[2])
        if best is None or family_best[0] < best[0]:
            best = family_best
    assert best is not None
    shared = _record(best[1], best[2])

    if num_stages > 1:
        capacities = [
            m.throughput_rps(MAX_BATCH, m.reference_length()) for m in scenario.stages
        ]
        order = sorted(range(num_stages), key=capacities.__getitem__)
        config, score = list(best[1]), best[2]
        rank = _rank(score, [s["family"] for s in config], 0)
        for _ in range(2):
            for stage in order:
                # The downstream-aware rule only differs from the threshold rule
                # where a next stage exists.
                stage_families = families if stage + 1 < num_stages else FAMILIES[:-1]
                index = 0
                for family in stage_families:
                    for spec in family_grid(family):
                        index += 1
                        trial = [*config[:stage], spec, *config[stage + 1 :]]
                        trial_score = evaluate(trial)
                        evaluated += 1
                        trial_rank = _rank(trial_score, [s["family"] for s in trial], index)
                        if trial_rank < rank:
                            config, score, rank = trial, trial_score, trial_rank
        best = (rank, config, score)

    return {
        "strongest": _record(best[1], best[2]),
        "best_shared_parameters": shared,
        "best_by_family": best_by_family,
        "configurations_evaluated": evaluated,
    }


def _record(config: Sequence[Spec], score: Score) -> dict[str, Any]:
    return {
        "config": [dict(spec) for spec in config],
        "dev_mean_latency_s": score[0],
        "dev_slo_violation_rate": score[1],
    }
