"""
Multi-Stage Adaptive Batching Experiment

Compares heuristics and PPO on a 3-stage disaggregated serving pipeline,
inspired by vllm-omni's stage abstraction. The metric of interest is
end-to-end Job Completion Time (JCT), which is what vllm-omni's paper
optimizes.

Usage: python run_multistage.py
Takes ~15-20 min on a laptop.
"""

import sys, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.multistage_env import (
    MultiStageServingEnv,
    StaticMultiStage,
    ThresholdMultiStage,
    DownstreamAwareMultiStage,
)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

NUM_STAGES = 3
PATTERN = "bursty"
SEED = 42
TIMESTEPS = 100000
EVAL_EPS = 10


def evaluate_agent(agent, num_stages=NUM_STAGES, pattern=PATTERN, n_eps=EVAL_EPS, is_sb3=False):
    """Run an agent (heuristic or trained model) on the env."""
    env = MultiStageServingEnv(num_stages=num_stages, pattern=pattern, seed=99)
    rewards, served_list, jct_list, p99_list = [], [], [], []

    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep + 500)
        total_reward = 0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        served_list.append(info["served"])
        jct_list.append(info["avg_jct"])
        p99_list.append(info["p99_jct"])

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "served": float(np.mean(served_list)),
        "avg_jct": float(np.mean(jct_list)),
        "p99_jct": float(np.mean(p99_list)),
    }


print("=" * 70)
print(f"  MULTI-STAGE ADAPTIVE BATCHING (vllm-omni inspired)")
print(f"  {NUM_STAGES} stages, pattern={PATTERN}, eval over {EVAL_EPS} eps")
print("=" * 70)

results = {}

# === Baselines ===
print("\n[1/4] Static (batch=8 at every stage)...")
agent = StaticMultiStage(batch_idx=3, num_stages=NUM_STAGES)
results["Static"] = evaluate_agent(agent)
print(f"  reward={results['Static']['reward_mean']:.1f} +/- {results['Static']['reward_std']:.1f}, "
      f"JCT={results['Static']['avg_jct']:.1f}, p99_JCT={results['Static']['p99_jct']:.1f}")

print("\n[2/4] Threshold (independent per-stage)...")
agent = ThresholdMultiStage(num_stages=NUM_STAGES)
results["Threshold"] = evaluate_agent(agent)
print(f"  reward={results['Threshold']['reward_mean']:.1f} +/- {results['Threshold']['reward_std']:.1f}, "
      f"JCT={results['Threshold']['avg_jct']:.1f}, p99_JCT={results['Threshold']['p99_jct']:.1f}")

print("\n[3/4] Downstream-Aware (considers next stage queue)...")
agent = DownstreamAwareMultiStage(num_stages=NUM_STAGES)
results["Downstream-Aware"] = evaluate_agent(agent)
print(f"  reward={results['Downstream-Aware']['reward_mean']:.1f} +/- {results['Downstream-Aware']['reward_std']:.1f}, "
      f"JCT={results['Downstream-Aware']['avg_jct']:.1f}, p99_JCT={results['Downstream-Aware']['p99_jct']:.1f}")

# === PPO ===
print(f"\n[4/4] PPO (training {TIMESTEPS} steps)...")
env = MultiStageServingEnv(num_stages=NUM_STAGES, pattern=PATTERN, seed=SEED)
model = PPO(
    "MlpPolicy", env,
    learning_rate=1e-3,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[128, 128]),
    verbose=0,
    seed=SEED,
    device="cpu",
)
model.learn(total_timesteps=TIMESTEPS)
results["PPO"] = evaluate_agent(model, is_sb3=True)
print(f"  reward={results['PPO']['reward_mean']:.1f} +/- {results['PPO']['reward_std']:.1f}, "
      f"JCT={results['PPO']['avg_jct']:.1f}, p99_JCT={results['PPO']['p99_jct']:.1f}")

# === Save and summarize ===
os.makedirs("results/multistage", exist_ok=True)
with open("results/multistage/multistage_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print(f"{'Method':22s} {'Reward':>10s} {'Std':>8s} {'Served':>8s} {'AvgJCT':>8s} {'P99JCT':>8s}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:22s} {r['reward_mean']:>10.1f} {r['reward_std']:>8.1f} "
          f"{r['served']:>8.0f} {r['avg_jct']:>8.1f} {r['p99_jct']:>8.1f}")
print("=" * 70)
print("\nSaved to results/multistage/multistage_results.json")
