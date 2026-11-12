"""Training of learned policies with Stable-Baselines3 (pre-registered protocol).

One *run* trains one algorithm with one seed on the train split of one
condition, evaluates a checkpoint on the dev split every
``CHECKPOINT_EVERY`` decisions, and keeps the checkpoint with the lowest dev
mean latency. The test split is never read here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptive_batching.experiment import BY_ID, load_splits, max_in_system, run
from adaptive_batching.metrics import summarize
from adaptive_batching.policies import GreedyBatch
from adaptive_batching.rl_env import BatchingEnv, LearnedPolicy
from adaptive_batching.scenarios import MAX_BATCH, Scenario
from adaptive_batching.traces import Trace

TOTAL_STEPS = 300_000
CHECKPOINT_EVERY = 30_000
EPISODE_REQUESTS = 512
SEEDS = (1, 2, 3, 4, 5)
TUNING_SEED = 100
ALGORITHMS = ("ppo", "dqn")

FIXED: dict[str, dict[str, Any]] = {
    "ppo": {"n_steps": 2048, "batch_size": 256, "n_epochs": 10},
    "dqn": {
        "buffer_size": 100_000,
        "learning_starts": 5_000,
        "train_freq": 4,
        "target_update_interval": 2_000,
        "exploration_final_eps": 0.02,
    },
}
# Pre-registered tuning grids (dev mean latency, primary comparison's condition).
TUNING_GRID: dict[str, list[dict[str, float]]] = {
    "ppo": [
        {"learning_rate": lr, "ent_coef": ent} for lr in (3e-4, 1e-3) for ent in (0.0, 0.01)
    ],
    "dqn": [
        {"learning_rate": lr, "exploration_fraction": frac}
        for lr in (1e-4, 5e-4)
        for frac in (0.1, 0.3)
    ],
}
GAMMA = 0.999
NET_ARCH = [64, 64]


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines one training run."""

    condition_id: str
    algorithm: str
    seed: int
    hyperparameters: dict[str, float]
    out_path: str
    model_path: str
    total_steps: int = TOTAL_STEPS
    checkpoint_every: int = CHECKPOINT_EVERY
    # Ablation switches; the defaults are the main experiment.
    include_arrival_rate: bool = True
    slo_penalty: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def backlog_max(train: Trace, scenario: Scenario) -> int:
    """Largest number in system under GreedyBatch(32) on train (pre-registration, Sec. 7)."""
    observed = max_in_system(run(train, scenario, GreedyBatch(MAX_BATCH)))
    return max(0, min(observed, len(train) - EPISODE_REQUESTS))


def make_model(spec: RunSpec, env: BatchingEnv) -> Any:  # noqa: ANN401 - SB3 algorithm object
    from stable_baselines3 import DQN, PPO

    cls = {"ppo": PPO, "dqn": DQN}[spec.algorithm]
    return cls(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        policy_kwargs={"net_arch": NET_ARCH},
        seed=spec.seed,
        device="cpu",
        verbose=0,
        **FIXED[spec.algorithm],
        **spec.hyperparameters,
    )


def load_model(algorithm: str, path: str | Path) -> Any:  # noqa: ANN401
    from stable_baselines3 import DQN, PPO

    return {"ppo": PPO, "dqn": DQN}[algorithm].load(str(path), device="cpu")


def evaluate_model(
    model: Any,  # noqa: ANN401
    split: Trace,
    scenario: Scenario,
    include_arrival_rate: bool = True,
) -> dict[str, Any]:
    policy = LearnedPolicy(model, scenario, include_arrival_rate)
    return summarize(run(split, scenario, policy), slo_s=scenario.slo_s).to_dict()


def train_run(spec: RunSpec) -> dict[str, Any]:
    """Train one run, write its record and best checkpoint, return the record."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    condition = BY_ID[spec.condition_id]
    scenario = condition.scenario
    splits = load_splits(condition)
    train, dev = splits["train"], splits["dev"]
    started = time.time()

    max_backlog = backlog_max(train, scenario)
    env = BatchingEnv(
        train,
        scenario,
        episode_requests=EPISODE_REQUESTS,
        include_arrival_rate=spec.include_arrival_rate,
        slo_penalty=spec.slo_penalty,
        backlog_max=max_backlog,
    )
    model = make_model(spec, env)
    curve: list[dict[str, float]] = []
    best = {"dev_mean_latency_s": float("inf")}
    model_path = Path(spec.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    class DevCheckpoint(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % spec.checkpoint_every == 0:
                metrics = evaluate_model(self.model, dev, scenario, spec.include_arrival_rate)
                point = {
                    "steps": self.num_timesteps,
                    "dev_mean_latency_s": metrics["mean_latency_s"],
                    "dev_slo_violation_rate": metrics["slo_violation_rate"],
                }
                curve.append(point)
                if point["dev_mean_latency_s"] < best["dev_mean_latency_s"]:
                    best.update(point)
                    self.model.save(str(model_path))
            return True

    model.learn(total_timesteps=spec.total_steps, callback=DevCheckpoint())
    train_seconds = time.time() - started

    selected = load_model(spec.algorithm, model_path)
    record = {
        "condition": spec.condition_id,
        "algorithm": spec.algorithm,
        "seed": spec.seed,
        "hyperparameters": {**FIXED[spec.algorithm], **spec.hyperparameters,
                            "gamma": GAMMA, "net_arch": NET_ARCH},
        "total_steps": spec.total_steps,
        "episode_requests": EPISODE_REQUESTS,
        "backlog_max": max_backlog,
        "include_arrival_rate": spec.include_arrival_rate,
        "slo_penalty": spec.slo_penalty,
        "curve": curve,
        "selected_checkpoint_steps": best["steps"],
        "selected_metrics": {
            "train": evaluate_model(selected, train, scenario, spec.include_arrival_rate),
            "dev": evaluate_model(selected, dev, scenario, spec.include_arrival_rate),
        },
        "train_seconds": train_seconds,
        **spec.extra,
    }
    out = Path(spec.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    return record
