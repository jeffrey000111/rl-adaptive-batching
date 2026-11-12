"""Select the best RL algorithm per condition on dev and run the memorisation check.

Reads results/baselines and results/rl/runs; writes
results/selection/<condition>.json. Never reads a test split.

Usage:
    python scripts/summarize_dev.py --group primary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_batching.experiment import GROUPS
from adaptive_batching.training import ALGORITHMS, SEEDS

ROOT = Path(__file__).resolve().parent.parent
OVERFIT_GAP = 0.10  # pre-registration, Section 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    args = parser.parse_args()
    out_dir = ROOT / "results/selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in GROUPS[args.group]:
        baseline = json.loads((ROOT / f"results/baselines/{condition.id}.json").read_text())
        base = {s: baseline["strongest_metrics"][s]["mean_latency_s"] for s in ("train", "dev")}
        algorithms = {}
        for algorithm in ALGORITHMS:
            run_dir = ROOT / f"results/rl/runs/{condition.id}"
            runs = [
                json.loads((run_dir / f"{algorithm}_seed{seed}.json").read_text())
                for seed in SEEDS
            ]

            def mean_latency(split: str, runs: list[dict] = runs) -> float:
                values = [r["selected_metrics"][split]["mean_latency_s"] for r in runs]
                return float(np.mean(values))

            mean = {split: mean_latency(split) for split in ("train", "dev")}
            ratio = {split: mean[split] / base[split] for split in ("train", "dev")}
            algorithms[algorithm] = {
                "dev_mean_latency_s_by_seed": [
                    r["selected_metrics"]["dev"]["mean_latency_s"] for r in runs
                ],
                "selected_checkpoint_steps_by_seed": [r["selected_checkpoint_steps"] for r in runs],
                "mean_latency_s": mean,
                "ratio_to_strongest_baseline": ratio,
                "possible_overfitting": ratio["train"] < ratio["dev"] - OVERFIT_GAP,
            }
        best = min(ALGORITHMS, key=lambda a: algorithms[a]["mean_latency_s"]["dev"])
        record = {
            "condition": condition.id,
            "best_rl_algorithm": best,
            "strongest_baseline_mean_latency_s": base,
            "algorithms": algorithms,
        }
        (out_dir / f"{condition.id}.json").write_text(json.dumps(record, indent=2) + "\n")
        chosen = algorithms[best]
        print(f"{condition.id}: best RL = {best}; latency ratio to baseline "
              f"train {chosen['ratio_to_strongest_baseline']['train']:.3f}, "
              f"dev {chosen['ratio_to_strongest_baseline']['dev']:.3f}; "
              f"overfit flag {chosen['possible_overfitting']}")


if __name__ == "__main__":
    main()
