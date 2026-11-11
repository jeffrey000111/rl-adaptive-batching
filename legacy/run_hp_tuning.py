"""
Hyperparameter Tuning Experiment
Sweeps learning rate and network architecture for PPO, SAC, and DQN
on bursty traffic (the hardest pattern).

Usage: python run_hp_tuning.py
Takes ~25-30 min on a laptop.
"""

import sys, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.serving_env import InferenceServingEnv
from agents.sac_agent import ContinuousToDiscreteWrapper
from stable_baselines3 import DQN, PPO, SAC
import numpy as np

PATTERN = "bursty"
TIMESTEPS = 50000
EVAL_EPS = 10
SEED = 42

# Hyperparameter grid
LR_OPTIONS = [1e-4, 3e-4, 1e-3]
ARCH_OPTIONS = [[32, 32], [64, 64], [128, 128]]


def eval_model(model, pattern, use_wrapper=False):
    base = InferenceServingEnv(pattern=pattern, seed=99)
    env = ContinuousToDiscreteWrapper(base) if use_wrapper else base
    rewards = []
    for ep in range(EVAL_EPS):
        obs, _ = env.reset(seed=ep + 500)
        total = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(a)
            total += r
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))


results = []
total_configs = len(LR_OPTIONS) * len(ARCH_OPTIONS) * 3  # 3 algos
config_num = 0

for algo_name in ["DQN", "PPO", "SAC"]:
    print(f"\n{'='*60}")
    print(f"  TUNING: {algo_name}")
    print(f"{'='*60}")

    best_reward = -float('inf')
    best_config = None

    for lr in LR_OPTIONS:
        for arch in ARCH_OPTIONS:
            config_num += 1
            arch_str = f"{arch[0]}x{arch[1]}"
            print(f"  [{config_num}/{total_configs}] {algo_name} lr={lr} arch={arch_str}...", end=" ")

            try:
                if algo_name == "DQN":
                    env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
                    model = DQN("MlpPolicy", env, learning_rate=lr, buffer_size=10000,
                                learning_starts=500, batch_size=64, gamma=0.99,
                                policy_kwargs=dict(net_arch=arch),
                                verbose=0, seed=SEED, device="cpu")
                    model.learn(total_timesteps=TIMESTEPS)
                    reward_mean, reward_std = eval_model(model, PATTERN)

                elif algo_name == "PPO":
                    env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
                    model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=512,
                                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                                clip_range=0.2, ent_coef=0.01,
                                policy_kwargs=dict(net_arch=arch),
                                verbose=0, seed=SEED, device="cpu")
                    model.learn(total_timesteps=TIMESTEPS)
                    reward_mean, reward_std = eval_model(model, PATTERN)

                elif algo_name == "SAC":
                    base = InferenceServingEnv(pattern=PATTERN, seed=SEED)
                    env = ContinuousToDiscreteWrapper(base)
                    model = SAC("MlpPolicy", env, learning_rate=lr, buffer_size=10000,
                                learning_starts=500, batch_size=64, gamma=0.99, tau=0.005,
                                ent_coef="auto",
                                policy_kwargs=dict(net_arch=arch),
                                verbose=0, seed=SEED, device="cpu")
                    model.learn(total_timesteps=TIMESTEPS)
                    reward_mean, reward_std = eval_model(model, PATTERN, use_wrapper=True)

                print(f"reward={reward_mean:.1f} +/- {reward_std:.1f}")

                entry = {
                    "algo": algo_name,
                    "lr": lr,
                    "arch": arch_str,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                }
                results.append(entry)

                if reward_mean > best_reward:
                    best_reward = reward_mean
                    best_config = entry

            except Exception as e:
                print(f"FAILED: {e}")
                results.append({
                    "algo": algo_name, "lr": lr, "arch": arch_str,
                    "reward_mean": None, "reward_std": None, "error": str(e)
                })

    if best_config:
        print(f"\n  Best {algo_name}: lr={best_config['lr']} arch={best_config['arch']} reward={best_config['reward_mean']:.1f}")

# Save results
os.makedirs("results/hp_tuning", exist_ok=True)
with open("results/hp_tuning/hp_tuning_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"\n\n{'='*70}")
print(f"{'Algo':6s} {'LR':>8s} {'Arch':>8s} {'Reward':>10s} {'Std':>8s}")
print(f"{'-'*70}")
for r in results:
    if r["reward_mean"] is not None:
        print(f"{r['algo']:6s} {r['lr']:>8.0e} {r['arch']:>8s} {r['reward_mean']:>10.1f} {r['reward_std']:>8.1f}")
print(f"{'='*70}")

# Print best per algo
print(f"\nBest configs:")
for algo in ["DQN", "PPO", "SAC"]:
    algo_results = [r for r in results if r["algo"] == algo and r["reward_mean"] is not None]
    if algo_results:
        best = max(algo_results, key=lambda x: x["reward_mean"])
        print(f"  {algo}: lr={best['lr']}, arch={best['arch']}, reward={best['reward_mean']:.1f}")

print(f"\nSaved to results/hp_tuning/hp_tuning_results.json")
