"""Training-sufficiency check (pre-registration, Deviation 2). Dev split only.

Compares the better algorithm's selected tuning configuration with the
untuned rule "never wait, cap 32 at every stage" on the dev split of the
primary comparison's condition. Appends the outcome to
results/rl/sufficiency_check.json and prints whether the one allowed
escalation of the training budget is triggered.

Usage:
    python scripts/check_training_sufficiency.py
"""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_batching.experiment import PRIMARY_COMPARISON, load_splits, run
from adaptive_batching.metrics import summarize
from adaptive_batching.policies import GreedyBatch
from adaptive_batching.scenarios import MAX_BATCH

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/rl/sufficiency_check.json"
THRESHOLD = 1.5
ESCALATED_BUDGET = 1_000_000


def main() -> None:
    condition = PRIMARY_COMPARISON
    dev = load_splits(condition)["dev"]
    reference = summarize(run(dev, condition.scenario, GreedyBatch(MAX_BATCH))).mean_latency_s
    chosen = json.loads((ROOT / "results/rl/hyperparameters.json").read_text())
    by_algorithm = {a: c["dev_mean_latency_s"] for a, c in chosen.items()}
    best = min(by_algorithm, key=by_algorithm.__getitem__)
    budget = next(iter(chosen.values()))["total_steps"]

    history = json.loads(OUT.read_text()) if OUT.exists() else []
    # Re-running at the same budget replaces that entry instead of adding one.
    history = [entry for entry in history if entry["training_budget"] != budget]
    passed = by_algorithm[best] <= THRESHOLD * reference
    history.append(
        {
            "condition": condition.id,
            "training_budget": budget,
            "reference_rule": f"GreedyBatch({MAX_BATCH}) at every stage",
            "reference_dev_mean_latency_s": reference,
            "threshold_ratio": THRESHOLD,
            "dev_mean_latency_s_by_algorithm": by_algorithm,
            "best_algorithm": best,
            "ratio": by_algorithm[best] / reference,
            "passed": passed,
        }
    )
    OUT.write_text(json.dumps(history, indent=2) + "\n")
    print(f"budget {budget}: best = {best} at {1e3 * by_algorithm[best]:.1f} ms, reference "
          f"{1e3 * reference:.1f} ms, ratio {by_algorithm[best] / reference:.2f} "
          f"(threshold {THRESHOLD}) -> {'PASSED' if passed else 'FAILED'}")
    if not passed and len(history) == 1:
        print(f"escalation triggered: re-run tuning with --total-steps {ESCALATED_BUDGET}")
    elif not passed:
        print("still failing after the one allowed escalation: proceed and report as such")


if __name__ == "__main__":
    main()
