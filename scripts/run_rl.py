"""Run pre-registered training jobs in parallel. Never reads a test split.

Stages:
    tune       hyper-parameter grid on the primary comparison's condition
               (seed 100); writes results/rl/hyperparameters.json
    primary    five seeds x {PPO, DQN} on the six primary conditions
    secondary  the same on the ten secondary conditions

Finished runs are skipped, so an interrupted stage can be resumed.

Usage:
    python scripts/run_rl.py --stage tune [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

from adaptive_batching.experiment import GROUPS, PRIMARY_COMPARISON
from adaptive_batching.training import (
    ALGORITHMS,
    SEEDS,
    TOTAL_STEPS,
    TUNING_GRID,
    TUNING_SEED,
    RunSpec,
    train_run,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/rl"
MODELS = ROOT / "artifacts/models"
HYPERPARAMETERS = RESULTS / "hyperparameters.json"


def tuning_specs(total_steps: int) -> list[RunSpec]:
    condition = PRIMARY_COMPARISON.id
    return [
        RunSpec(
            condition_id=condition,
            algorithm=algorithm,
            seed=TUNING_SEED,
            hyperparameters=hp,
            out_path=str(RESULTS / "tuning" / f"{algorithm}_{index}.json"),
            model_path=str(MODELS / "tuning" / f"{algorithm}_{index}.zip"),
            total_steps=total_steps,
        )
        for algorithm in ALGORITHMS
        for index, hp in enumerate(TUNING_GRID[algorithm])
    ]


def main_specs(group: str) -> list[RunSpec]:
    chosen = json.loads(HYPERPARAMETERS.read_text())
    return [
        RunSpec(
            condition_id=condition.id,
            algorithm=algorithm,
            seed=seed,
            hyperparameters=chosen[algorithm]["hyperparameters"],
            out_path=str(RESULTS / "runs" / condition.id / f"{algorithm}_seed{seed}.json"),
            model_path=str(MODELS / condition.id / f"{algorithm}_seed{seed}.zip"),
            total_steps=chosen[algorithm]["total_steps"],
        )
        for condition in GROUPS[group]
        for algorithm in ALGORITHMS
        for seed in SEEDS
    ]


def job(spec: RunSpec) -> str:
    record = train_run(spec)
    return (f"{spec.condition_id} {spec.algorithm} seed={spec.seed} {spec.hyperparameters}: "
            f"best dev mean {1e3 * record['selected_metrics']['dev']['mean_latency_s']:.1f} ms "
            f"at {record['selected_checkpoint_steps']} steps ({record['train_seconds']:.0f}s)")


def choose_hyperparameters() -> None:
    chosen = {}
    for algorithm in ALGORITHMS:
        paths = sorted((RESULTS / "tuning").glob(f"{algorithm}_*.json"))
        runs = [json.loads(path.read_text()) for path in paths]
        best = min(runs, key=lambda r: r["selected_metrics"]["dev"]["mean_latency_s"])
        chosen[algorithm] = {
            "hyperparameters": {k: best["hyperparameters"][k] for k in TUNING_GRID[algorithm][0]},
            "dev_mean_latency_s": best["selected_metrics"]["dev"]["mean_latency_s"],
            "total_steps": best["total_steps"],
            "tuned_on": best["condition"],
            "tuning_seed": best["seed"],
        }
    HYPERPARAMETERS.write_text(json.dumps(chosen, indent=2) + "\n")
    print(f"chosen: {json.dumps(chosen)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["tune", "primary", "secondary"], required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS,
                        help="training budget for the tune stage (main stages reuse it)")
    args = parser.parse_args()

    specs = tuning_specs(args.total_steps) if args.stage == "tune" else main_specs(args.stage)
    todo = [s for s in specs if not Path(s.out_path).exists()]
    print(f"{len(specs)} runs, {len(todo)} to do, {args.workers} workers", flush=True)
    started = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for done, line in enumerate(pool.imap_unordered(job, todo), start=1):
            print(f"[{done}/{len(todo)} {time.time() - started:.0f}s] {line}", flush=True)
    print(f"stage {args.stage} finished in {time.time() - started:.0f}s", flush=True)
    if args.stage == "tune":
        choose_hyperparameters()


if __name__ == "__main__":
    main()
