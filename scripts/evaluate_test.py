"""Evaluate frozen policies on the TEST split, once per condition.

For every condition of the group: simulates the strongest baseline (chosen
on dev) and every trained seed of both algorithms over the whole test
split, then compares the best RL algorithm (chosen on dev) with the baseline
by a paired block bootstrap. Writes results/test/<condition>.json with the
commit hash it ran from, and refuses to overwrite an existing result.

Usage:
    python scripts/evaluate_test.py --group primary
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from adaptive_batching.experiment import GROUPS, Condition, load_splits, run, split_loads
from adaptive_batching.metrics import slo_violation_rate, summarize
from adaptive_batching.rl_env import LearnedPolicy
from adaptive_batching.stats import condition_verdict, paired_block_bootstrap
from adaptive_batching.training import ALGORITHMS, SEEDS, load_model
from adaptive_batching.tuning import build_policies

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/test"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def evaluate(condition: Condition, commit: str, dirty: bool) -> dict[str, object]:
    scenario = condition.scenario
    test = load_splits(condition)["test"]
    baseline = json.loads((ROOT / f"results/baselines/{condition.id}.json").read_text())
    selection = json.loads((ROOT / f"results/selection/{condition.id}.json").read_text())

    def metrics(result: object) -> dict[str, object]:
        out = summarize(result, slo_s=scenario.slo_s).to_dict()  # type: ignore[arg-type]
        out["tight_slo_violation_rate"] = slo_violation_rate(result, scenario.tight_slo_s)  # type: ignore[arg-type]
        return out

    config = baseline["strongest"]["config"]
    base_result = run(test, scenario, build_policies(config, scenario))
    record: dict[str, object] = {
        "condition": condition.id,
        "commit": commit,
        "working_tree_dirty": dirty,
        "test_split": split_loads(condition)["test"],
        "slo_s": scenario.slo_s,
        "strongest_baseline": {"config": config, "metrics": metrics(base_result)},
        "best_rl_algorithm": selection["best_rl_algorithm"],
        "algorithms": {},
    }
    for algorithm in ALGORITHMS:
        results = [
            run(
                test,
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
        intervals = paired_block_bootstrap(latency, base_result.latency_s, scenario.slo_s)
        verdicts = {name: interval.verdict for name, interval in intervals.items()}
        record["algorithms"][algorithm] = {  # type: ignore[index]
            "role": "best RL (pre-registered comparison)"
            if algorithm == selection["best_rl_algorithm"]
            else "other algorithm (secondary)",
            "metrics_by_seed": [metrics(r) for r in results],
            "mean_latency_s": float(latency.mean()),
            "ratio_to_strongest_baseline": float(latency.mean() / base_result.latency_s.mean()),
            "intervals": {name: interval.to_dict() for name, interval in intervals.items()},
            "verdict": condition_verdict(
                verdicts["mean_latency_relative_difference"],
                verdicts["slo_violation_difference_points"],
            ),
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    args = parser.parse_args()

    if "Status: **approved" not in (ROOT / "docs/preregistration.md").read_text():
        raise SystemExit("the pre-registration is not approved; refusing to touch test")
    existing = [c.id for c in GROUPS[args.group] if (OUT / f"{c.id}.json").exists()]
    if existing:
        raise SystemExit(f"test results already exist for {existing}; refusing to re-run")

    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    OUT.mkdir(parents=True, exist_ok=True)
    for condition in GROUPS[args.group]:
        record = evaluate(condition, commit, dirty)
        (OUT / f"{condition.id}.json").write_text(json.dumps(record, indent=2) + "\n")
        best = record["algorithms"][record["best_rl_algorithm"]]  # type: ignore[index]
        latency = best["intervals"]["mean_latency_relative_difference"]
        print(f"{condition.id}: {record['best_rl_algorithm']} vs baseline, mean latency "
              f"{100 * latency['estimate']:+.1f}% [{100 * latency['low']:+.1f}, "
              f"{100 * latency['high']:+.1f}] -> {best['verdict']}", flush=True)


if __name__ == "__main__":
    main()
