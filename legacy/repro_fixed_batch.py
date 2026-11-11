"""Reproduce the flaw in the original multi-stage experiment.

The original report compared PPO against Static(batch=8), Threshold and
Downstream-Aware and claimed a 52% average-JCT win. This script evaluates
every fixed batch size with the *same* protocol as run_multistage.py's
evaluate_agent (3 stages, bursty, 10 episodes, reset(seed=ep+500)) and shows
that fixed batch 32 and 64 beat the reported PPO number.

Root cause: MultiStageServingEnv.step() computes stage_latency and never
uses it, so a batch costs no time and the largest batch is always best.

Usage (from the repository root):
    python legacy/repro_fixed_batch.py

Writes results/legacy_repro/fixed_batch.json. The PPO row is not retrained;
it is copied from legacy/results/multistage/multistage_results.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

LEGACY = Path(__file__).resolve().parent
ROOT = LEGACY.parent
sys.path.insert(0, str(LEGACY / "src"))

from env.multistage_env import (  # noqa: E402
    DownstreamAwareMultiStage,
    MultiStageServingEnv,
    StaticMultiStage,
    ThresholdMultiStage,
)

N_EPISODES = 10
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


def evaluate(agent):
    env = MultiStageServingEnv(num_stages=3, pattern="bursty", seed=99)
    rewards, served, jct, p99 = [], [], [], []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=ep + 500)
        total = 0.0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        served.append(info["served"])
        jct.append(info["avg_jct"])
        p99.append(info["p99_jct"])
    return {
        "reward_mean": float(np.mean(rewards)),
        "served": float(np.mean(served)),
        "avg_jct": float(np.mean(jct)),
        "p99_jct": float(np.mean(p99)),
    }


def main():
    rows = []
    for idx, b in enumerate(BATCH_SIZES):
        rows.append({"policy": f"Fixed batch {b}", "source": "re-run",
                     **evaluate(StaticMultiStage(batch_idx=idx, num_stages=3))})
    rows.append({"policy": "Threshold", "source": "re-run",
                 **evaluate(ThresholdMultiStage(num_stages=3))})
    rows.append({"policy": "Downstream-Aware", "source": "re-run",
                 **evaluate(DownstreamAwareMultiStage(num_stages=3))})

    stored = json.loads((LEGACY / "results/multistage/multistage_results.json").read_text())
    ppo = stored["PPO"]
    rows.append({"policy": "PPO (original run, not retrained)", "source": "stored",
                 "reward_mean": ppo["reward_mean"], "served": ppo["served"],
                 "avg_jct": ppo["avg_jct"], "p99_jct": ppo["p99_jct"]})

    out = ROOT / "results/legacy_repro/fixed_batch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": {
        "env": "legacy MultiStageServingEnv(num_stages=3, pattern='bursty')",
        "episodes": N_EPISODES, "episode_seeds": "ep + 500"}, "rows": rows}, indent=2) + "\n")

    for r in rows:
        print(f"{r['policy']:36s} avg_jct={r['avg_jct']:8.2f}  p99_jct={r['p99_jct']:8.2f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
