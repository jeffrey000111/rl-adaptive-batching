"""
Multi-Seed Stability Analysis
Trains PPO, SAC, and DQN with 3 different random seeds
to check if results are consistent or just lucky.

Usage: python run_multi_seed.py
Takes ~15-20 min on a laptop.
"""

import sys, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.serving_env import InferenceServingEnv
from agents.sac_agent import ContinuousToDiscreteWrapper
from stable_baselines3 import DQN, PPO, SAC
import numpy as np

SEEDS = [42, 123, 456]
PATTERNS = ["steady", "bursty", "diurnal"]
TIMESTEPS = 50000  # decent training length
EVAL_EPS = 10


def eval_model(model, pattern, use_wrapper=False):
    base = InferenceServingEnv(pattern=pattern, seed=99)
    env = ContinuousToDiscreteWrapper(base) if use_wrapper else base
    rewards, served, violations, latencies = [], [], [], []
    for ep in range(EVAL_EPS):
        obs, _ = env.reset(seed=ep + 500)
        total = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(a)
            total += r
            if info["served"] > 0:
                latencies.append(info["latency_ms"])
            if done:
                break
        rewards.append(total)
        served.append(info["total_served"])
        violations.append(info["violations"])
    lat = np.array(latencies) if latencies else np.array([0])
    return {
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "served": float(np.mean(served)),
        "violations": float(np.mean(violations)),
        "lat_mean": float(np.mean(lat)),
        "lat_p99": float(np.percentile(lat, 99)),
    }


results = []

for pattern in PATTERNS:
    print(f"\n{'='*60}")
    print(f"  PATTERN: {pattern.upper()}")
    print(f"{'='*60}")

    for algo_name in ["DQN", "PPO", "SAC"]:
        seed_results = []

        for seed in SEEDS:
            print(f"  Training {algo_name} seed={seed}...")

            if algo_name == "DQN":
                env = InferenceServingEnv(pattern=pattern, seed=seed)
                model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=10000,
                            learning_starts=500, batch_size=64, gamma=0.99,
                            policy_kwargs=dict(net_arch=[64, 64]),
                            verbose=0, seed=seed, device="cpu")
                model.learn(total_timesteps=TIMESTEPS)
                res = eval_model(model, pattern)

            elif algo_name == "PPO":
                env = InferenceServingEnv(pattern=pattern, seed=seed)
                model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512,
                            batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                            clip_range=0.2, ent_coef=0.01,
                            policy_kwargs=dict(net_arch=[64, 64]),
                            verbose=0, seed=seed, device="cpu")
                model.learn(total_timesteps=TIMESTEPS)
                res = eval_model(model, pattern)

            elif algo_name == "SAC":
                base = InferenceServingEnv(pattern=pattern, seed=seed)
                env = ContinuousToDiscreteWrapper(base)
                model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=10000,
                            learning_starts=500, batch_size=64, gamma=0.99, tau=0.005,
                            ent_coef="auto",
                            policy_kwargs=dict(net_arch=[64, 64]),
                            verbose=0, seed=seed, device="cpu")
                model.learn(total_timesteps=TIMESTEPS)
                res = eval_model(model, pattern, use_wrapper=True)

            seed_results.append(res)
            print(f"    seed={seed}: reward={res['reward']:.1f}")

        # aggregate across seeds
        agg = {
            "algo": algo_name,
            "pattern": pattern,
            "reward_mean": float(np.mean([r["reward"] for r in seed_results])),
            "reward_std_across_seeds": float(np.std([r["reward"] for r in seed_results])),
            "served_mean": float(np.mean([r["served"] for r in seed_results])),
            "lat_mean": float(np.mean([r["lat_mean"] for r in seed_results])),
            "lat_p99": float(np.mean([r["lat_p99"] for r in seed_results])),
            "per_seed": seed_results,
        }
        results.append(agg)
        print(f"  {algo_name} avg: {agg['reward_mean']:.1f} +/- {agg['reward_std_across_seeds']:.1f}")

# save
os.makedirs("results/multi_seed", exist_ok=True)
with open("results/multi_seed/multi_seed_results.json", "w") as f:
    json.dump(results, f, indent=2)

# print summary table
print(f"\n\n{'='*70}")
print(f"{'Algo':8s} {'Pattern':10s} {'Reward (mean)':>14s} {'Std across seeds':>18s} {'Served':>8s}")
print(f"{'-'*70}")
for r in results:
    print(f"{r['algo']:8s} {r['pattern']:10s} {r['reward_mean']:14.1f} {r['reward_std_across_seeds']:18.1f} {r['served_mean']:8.0f}")
print(f"{'='*70}")
print(f"\nSaved to results/multi_seed/multi_seed_results.json")
