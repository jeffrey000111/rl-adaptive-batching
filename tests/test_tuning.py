from __future__ import annotations

from adaptive_batching.scenarios import PIPELINE_3STAGE, SINGLE_STAGE
from adaptive_batching.tuning import (
    Spec,
    build_policies,
    family_grid,
    search,
)


def test_grid_sizes_match_the_preregistration() -> None:
    assert len(family_grid("greedy")) == 6
    assert len(family_grid("full")) == 36
    assert len(family_grid("threshold")) == 60
    assert len(family_grid("downstream")) == 360


def test_every_grid_spec_builds_a_policy() -> None:
    for family in ("greedy", "full", "threshold", "downstream"):
        for spec in family_grid(family):
            assert len(build_policies([spec] * 3, PIPELINE_3STAGE)) == 3


def test_single_stage_search_is_exhaustive_and_finds_the_minimum() -> None:
    target = {"family": "threshold", "k": 4, "wait_mult": 2}

    def evaluate(config: list[Spec]) -> tuple[float, float]:
        return (0.0 if config[0] == target else 1.0, 0.0)

    out = search(SINGLE_STAGE, evaluate)
    assert out["strongest"]["config"] == [target]
    assert out["configurations_evaluated"] == 6 + 36 + 60
    assert set(out["best_by_family"]) == {"greedy", "full", "threshold"}


def test_ties_go_to_the_simpler_family() -> None:
    out = search(SINGLE_STAGE, lambda config: (1.0, 0.0))
    assert out["strongest"]["config"] == [{"family": "greedy", "cap": 1}]


def test_pipeline_search_refines_stages_separately() -> None:
    wanted = [
        {"family": "greedy", "cap": 32},
        {"family": "threshold", "k": 8, "wait_mult": 4},
        {"family": "greedy", "cap": 4},
    ]

    def evaluate(config: list[Spec]) -> tuple[float, float]:
        return (float(sum(a != b for a, b in zip(config, wanted, strict=True))), 0.0)

    out = search(PIPELINE_3STAGE, evaluate)
    assert out["strongest"]["config"] == wanted
    assert out["strongest"]["dev_mean_latency_s"] == 0.0
    # Shared-parameter search alone cannot reach it.
    assert out["best_shared_parameters"]["dev_mean_latency_s"] == 2.0
