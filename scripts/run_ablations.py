"""Ablations at conv / load 0.7 in both scenarios (pre-registration, Section 11).

Variants, each trained with five seeds of the condition's best RL algorithm:

    slo_penalty_10, slo_penalty_100   reward with an SLO-miss penalty
    no_arrival_rate                   observation without arrival-rate features
    fixed_length_128                  every request 128 tokens; baselines are
                                      re-tuned on dev (tune_baselines.py
                                      --group ablation_len128)
    tight_slo                         main policies re-scored under the tight
                                      SLO; no training

Stages: `train` (train/dev only) and `test` (once; refuses to overwrite).

Usage:
    python scripts/run_ablations.py --stage train [--workers 6]
    python scripts/run_ablations.py --stage test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from adaptive_batching.experiment import (
    ABLATION_BASE,
    ABLATION_LEN128,
    Condition,
    load_splits,
    run,
)
from adaptive_batching.metrics import summarize
from adaptive_batching.rl_env import LearnedPolicy
from adaptive_batching.stats import condition_verdict, paired_block_bootstrap
from adaptive_batching.training import SEEDS, RunSpec, load_model, train_run
from adaptive_batching.tuning import build_policies

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/ablations"
MODELS = ROOT / "artifacts/models"

# variant -> (conditions, environment switches)
TRAINED: dict[str, tuple[tuple[Condition, ...], dict[str, object]]] = {
    "slo_penalty_10": (ABLATION_BASE, {"slo_penalty": 10.0}),
    "slo_penalty_100": (ABLATION_BASE, {"slo_penalty": 100.0}),
    "no_arrival_rate": (ABLATION_BASE, {"include_arrival_rate": False}),
    "fixed_length_128": (ABLATION_LEN128, {}),
}


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def base_condition_id(condition: Condition) -> str:
    """The main-grid condition an ablation condition belongs to."""
    return condition.id.replace("-len128", "")


def algorithm_for(condition: Condition) -> str:
    return read(f"results/selection/{base_condition_id(condition)}.json")["best_rl_algorithm"]


def model_path(variant: str, condition: Condition, algorithm: str, seed: int) -> Path:
    return MODELS / "ablations" / variant / condition.id / f"{algorithm}_seed{seed}.zip"


def specs() -> list[RunSpec]:
    chosen = read("results/rl/hyperparameters.json")
    out = []
    for variant, (conditions, switches) in TRAINED.items():
        run_dir = RESULTS / "runs" / variant
        for condition in conditions:
            algorithm = algorithm_for(condition)
            for seed in SEEDS:
                out.append(
                    RunSpec(
                        condition_id=condition.id,
                        algorithm=algorithm,
                        seed=seed,
                        hyperparameters=chosen[algorithm]["hyperparameters"],
                        out_path=str(run_dir / condition.id / f"{algorithm}_seed{seed}.json"),
                        model_path=str(model_path(variant, condition, algorithm, seed)),
                        total_steps=chosen[algorithm]["total_steps"],
                        extra={"variant": variant},
                        **switches,  # type: ignore[arg-type]
                    )
                )
    return out


def job(spec: RunSpec) -> str:
    record = train_run(spec)
    return (f"{spec.extra['variant']} {spec.condition_id} {spec.algorithm} seed={spec.seed}: "
            f"best dev mean {1e3 * record['selected_metrics']['dev']['mean_latency_s']:.1f} ms")


def compare(
    learned: np.ndarray, baseline: np.ndarray, slo_s: float
) -> dict[str, object]:
    intervals = paired_block_bootstrap(learned, baseline, slo_s)
    verdicts = {name: interval.verdict for name, interval in intervals.items()}
    return {
        "mean_latency_s_by_seed": learned.mean(axis=1).tolist(),
        "slo_violation_rate_by_seed": (learned > slo_s).mean(axis=1).tolist(),
        "intervals": {name: interval.to_dict() for name, interval in intervals.items()},
        "verdict": condition_verdict(
            verdicts["mean_latency_relative_difference"],
            verdicts["slo_violation_difference_points"],
        ),
    }


def test_variant(variant: str, condition: Condition, commit: str) -> dict[str, object]:
    scenario = condition.scenario
    test = load_splits(condition)["test"]
    algorithm = algorithm_for(condition)
    config = read(f"results/baselines/{condition.id}.json")["strongest"]["config"]
    base = run(test, scenario, build_policies(config, scenario))

    if variant == "tight_slo":
        slo_s, include_rate = scenario.tight_slo_s, True
        paths = [MODELS / condition.id / f"{algorithm}_seed{seed}.zip" for seed in SEEDS]
    else:
        slo_s = scenario.slo_s
        include_rate = bool(TRAINED[variant][1].get("include_arrival_rate", True))
        paths = [model_path(variant, condition, algorithm, seed) for seed in SEEDS]
    policies = [LearnedPolicy(load_model(algorithm, p), scenario, include_rate) for p in paths]
    learned = np.stack([run(test, scenario, policy).latency_s for policy in policies])
    return {
        "variant": variant,
        "condition": condition.id,
        "commit": commit,
        "algorithm": algorithm,
        "slo_s": slo_s,
        "strongest_baseline": {
            "config": config,
            "mean_latency_s": summarize(base).mean_latency_s,
            "slo_violation_rate": float((base.latency_s > slo_s).mean()),
        },
        **compare(learned, base.latency_s, slo_s),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["train", "test"], required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.stage == "train":
        todo = [s for s in specs() if not Path(s.out_path).exists()]
        print(f"{len(todo)} runs to do", flush=True)
        started = time.time()
        with get_context("spawn").Pool(args.workers) as pool:
            for done, line in enumerate(pool.imap_unordered(job, todo), start=1):
                print(f"[{done}/{len(todo)} {time.time() - started:.0f}s] {line}", flush=True)
        return

    out_dir = RESULTS / "test"
    if out_dir.exists():
        raise SystemExit("ablation test results already exist; refusing to re-run")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    plan = [(v, c) for v, (conditions, _) in TRAINED.items() for c in conditions]
    plan += [("tight_slo", c) for c in ABLATION_BASE]
    for variant, condition in plan:
        record = test_variant(variant, condition, commit)
        path = out_dir / variant / f"{condition.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
        lat = record["intervals"]["mean_latency_relative_difference"]  # type: ignore[index]
        slo = record["intervals"]["slo_violation_difference_points"]  # type: ignore[index]
        print(f"{variant} {condition.id}: latency {100 * lat['estimate']:+.1f}% "
              f"[{100 * lat['low']:+.1f}, {100 * lat['high']:+.1f}], SLO {slo['estimate']:+.2f} pt "
              f"[{slo['low']:+.2f}, {slo['high']:+.2f}] -> {record['verdict']}", flush=True)


if __name__ == "__main__":
    main()
