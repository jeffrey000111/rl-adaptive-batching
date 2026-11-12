"""Grid-search the rule-based baselines on the dev split of each condition.

Writes results/baselines/<condition>.json: the strongest configuration, the
best of each family, and the strongest configuration's metrics on train and
dev. The test split is never read here.

Usage:
    python scripts/tune_baselines.py --group primary [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

from adaptive_batching.experiment import BY_ID, GROUPS, load_splits, run, split_loads
from adaptive_batching.metrics import summarize
from adaptive_batching.tuning import Spec, build_policies, search

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/baselines"


def tune(condition_id: str) -> str:
    condition = BY_ID[condition_id]
    scenario = condition.scenario
    splits = load_splits(condition)
    started = time.time()

    def evaluate(config: list[Spec]) -> tuple[float, float]:
        metrics = summarize(
            run(splits["dev"], scenario, build_policies(config, scenario)), slo_s=scenario.slo_s
        )
        assert metrics.slo_violation_rate is not None
        return metrics.mean_latency_s, metrics.slo_violation_rate

    found = search(scenario, evaluate)
    strongest = build_policies(found["strongest"]["config"], scenario)
    record = {
        "condition": condition_id,
        "selection_split": "dev",
        "splits": {k: v for k, v in split_loads(condition).items() if k != "test"},
        **found,
        "strongest_metrics": {
            split: summarize(run(splits[split], scenario, strongest), scenario.slo_s).to_dict()
            for split in ("train", "dev")
        },
        "search_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{condition_id}.json").write_text(json.dumps(record, indent=2) + "\n")
    best = found["strongest"]
    return (f"{condition_id}: dev mean {1e3 * best['dev_mean_latency_s']:.1f} ms "
            f"with {best['config']} ({found['configurations_evaluated']} configs, "
            f"{record['search_seconds']:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    ids = [c.id for c in GROUPS[args.group]]
    with get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(tune, ids):
            print(line, flush=True)


if __name__ == "__main__":
    main()
