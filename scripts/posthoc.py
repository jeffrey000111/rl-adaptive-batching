"""Post-hoc exploratory analyses (pre-registration, Deviation 3).

None of this changes a registered verdict. For each condition of the group:

* per-seed test results, their median, and the number of failed seeds
  (test mean latency above FAILURE_RATIO times the strongest baseline's);
* the best seed by dev against the strongest baseline on test;
* with --stress, the same on the test split re-scaled to its nominal load
  (primary conditions at nominal load 0.7 and 0.9 only), with every policy
  frozen.

Also writes results/posthoc/cost.json. Requires results/test/ to exist for
the group: it never evaluates a condition that has not had its registered
test evaluation.

Usage:
    python scripts/posthoc.py --group primary [--stress]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_batching.experiment import (
    GROUPS,
    Condition,
    load_splits,
    run,
    split_loads,
    stress_test_split,
)
from adaptive_batching.load import offered_load
from adaptive_batching.metrics import summarize
from adaptive_batching.rl_env import LearnedPolicy
from adaptive_batching.scenarios import MAX_BATCH
from adaptive_batching.stats import condition_verdict, paired_block_bootstrap
from adaptive_batching.traces import Trace
from adaptive_batching.training import ALGORITHMS, SEEDS, load_model
from adaptive_batching.tuning import build_policies

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/posthoc"
FAILURE_RATIO = 2.0
STRESS_LOADS = (0.7, 0.9)


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def compare(learned: np.ndarray, baseline: np.ndarray, slo_s: float) -> dict[str, object]:
    intervals = paired_block_bootstrap(learned, baseline, slo_s)
    verdicts = {name: interval.verdict for name, interval in intervals.items()}
    return {
        "intervals": {name: interval.to_dict() for name, interval in intervals.items()},
        "verdict": condition_verdict(
            verdicts["mean_latency_relative_difference"],
            verdicts["slo_violation_difference_points"],
        ),
    }


def analyse(condition: Condition, split: Trace, label: str) -> dict[str, object]:
    scenario = condition.scenario
    selection = read(f"results/selection/{condition.id}.json")
    config = read(f"results/baselines/{condition.id}.json")["strongest"]["config"]
    base = run(split, scenario, build_policies(config, scenario))
    base_metrics = summarize(base, slo_s=scenario.slo_s)
    record: dict[str, object] = {
        "condition": condition.id,
        "evaluated_on": label,
        "exploratory": True,
        "requests": len(split),
        "realised_load": offered_load(split, scenario.stages, MAX_BATCH),
        "failure_ratio": FAILURE_RATIO,
        "best_rl_algorithm": selection["best_rl_algorithm"],
        "strongest_baseline": base_metrics.to_dict(),
        "algorithms": {},
    }
    for algorithm in ALGORITHMS:
        results = [
            run(
                split,
                scenario,
                LearnedPolicy(
                    load_model(
                        algorithm,
                        ROOT / f"artifacts/models/{condition.id}/{algorithm}_seed{seed}.zip",
                    ),
                    scenario,
                ),
            )
            for seed in SEEDS
        ]
        latency = np.stack([r.latency_s for r in results])
        by_seed = latency.mean(axis=1)
        dev = selection["algorithms"][algorithm]["dev_mean_latency_s_by_seed"]
        best_seed = int(np.argmin(dev))
        failed = by_seed > FAILURE_RATIO * base_metrics.mean_latency_s
        record["algorithms"][algorithm] = {  # type: ignore[index]
            "mean_latency_s_by_seed": by_seed.tolist(),
            "slo_violation_rate_by_seed": (latency > scenario.slo_s).mean(axis=1).tolist(),
            "mean_over_seeds_s": float(by_seed.mean()),
            "median_over_seeds_s": float(np.median(by_seed)),
            "failed_seeds": int(failed.sum()),
            "five_seed_average": compare(latency, base.latency_s, scenario.slo_s),
            "best_seed_by_dev": {
                "seed": SEEDS[best_seed],
                "mean_latency_s": float(by_seed[best_seed]),
                **compare(latency[best_seed : best_seed + 1], base.latency_s, scenario.slo_s),
            },
        }
    return record


def cost() -> dict[str, object]:
    def train_seconds(pattern: str) -> float:
        return float(sum(json.loads(p.read_text())["train_seconds"] for p in ROOT.glob(pattern)))

    per_condition = {}
    for path in sorted((ROOT / "results/baselines").glob("*.json")):
        name = path.stem
        per_condition[name] = {
            "baseline_search_seconds": json.loads(path.read_text())["search_seconds"],
            "baseline_configurations": json.loads(path.read_text())["configurations_evaluated"],
            "rl_training_seconds_10_runs": train_seconds(f"results/rl/runs/{name}/*.json"),
        }
    return {
        "note": "wall-clock seconds per process, measured with up to six processes in parallel",
        "rl_tuning_seconds_8_runs": train_seconds("results/rl/tuning/*.json"),
        "rl_tuning_seconds_discarded_first_round": train_seconds(
            "results/rl/tuning_registered_reward/*_[0-9].json"
        ),
        "per_condition": per_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()

    for condition in GROUPS[args.group]:
        if not (ROOT / f"results/test/{condition.id}.json").exists():
            raise SystemExit(f"{condition.id}: run the registered test evaluation first")
        if args.stress and condition.load not in STRESS_LOADS:
            continue
        if args.stress:
            split, label, out = stress_test_split(condition), "test re-scaled to nominal load", \
                OUT / "stress"
        else:
            split, label, out = load_splits(condition)["test"], "test", OUT / "per_seed"
        record = analyse(condition, split, label)
        if not args.stress:
            record["registered_test_load"] = split_loads(condition)["test"]["load"]
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{condition.id}.json").write_text(json.dumps(record, indent=2) + "\n")
        best = record["algorithms"][record["best_rl_algorithm"]]  # type: ignore[index]
        lat = best["five_seed_average"]["intervals"]["mean_latency_relative_difference"]
        one = best["best_seed_by_dev"]["intervals"]["mean_latency_relative_difference"]
        print(f"{condition.id} [{label}, load {record['realised_load']:.2f}] "
              f"{record['best_rl_algorithm']}: failed {best['failed_seeds']}/5; 5-seed "
              f"{100 * lat['estimate']:+.1f}% [{100 * lat['low']:+.1f}, {100 * lat['high']:+.1f}]; "
              f"best-by-dev seed {100 * one['estimate']:+.1f}% "
              f"[{100 * one['low']:+.1f}, {100 * one['high']:+.1f}]", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost.json").write_text(json.dumps(cost(), indent=2) + "\n")


if __name__ == "__main__":
    main()
