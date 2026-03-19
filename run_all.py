"""
Run all experiments: baselines + DQN + PPO + SAC
Outputs results to results/ folder.

Usage: python run_all.py
"""

import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.serving_env import InferenceServingEnv
from baselines.heuristics import StaticBatcher, TimeoutBatcher, ThresholdBatcher, run_baseline
from stable_baselines3 import DQN, PPO, SAC
from agents.sac_agent import ContinuousToDiscreteWrapper
import numpy as np
import json

PATTERNS = ["steady", "bursty", "diurnal"]
TIMESTEPS = 30000  # increase for better results
EVAL_EPISODES = 10
RESULTS = []


def eval_rl(model, pattern, name, episodes=EVAL_EPISODES, use_wrapper=False):
    base_env = InferenceServingEnv(pattern=pattern, seed=99)
    env = ContinuousToDiscreteWrapper(base_env) if use_wrapper else base_env
    rewards, served_list, viol_list, latencies = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep + 100)
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(action)
            ep_reward += r
            if info["served"] > 0:
                latencies.append(info["latency_ms"])
            if done:
                break
        rewards.append(ep_reward)
        served_list.append(info["total_served"])
        viol_list.append(info["violations"])

    lat_arr = np.array(latencies) if latencies else np.array([0])
    return {
        "name": name, "pattern": pattern,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "served_mean": float(np.mean(served_list)),
        "violations_mean": float(np.mean(viol_list)),
        "latency_mean": float(np.mean(lat_arr)),
        "latency_p99": float(np.percentile(lat_arr, 99)),
    }


os.makedirs("results", exist_ok=True)

for pattern in PATTERNS:
    print(f"\n{'='*60}")
    print(f"  PATTERN: {pattern.upper()}")
    print(f"{'='*60}")

    env = InferenceServingEnv(pattern=pattern, seed=42)

    # --- heuristic baselines ---
    for agent in [StaticBatcher(3), TimeoutBatcher(), ThresholdBatcher()]:
        res = run_baseline(env, agent, episodes=EVAL_EPISODES)
        res["pattern"] = pattern
        RESULTS.append(res)
        print(f"  {res['name']:20s} reward={res['reward_mean']:8.1f} served={res['served_mean']:6.0f}")

    # --- DQN (baseline RL) ---
    print(f"\n  Training DQN ({TIMESTEPS} steps)...")
    dqn_env = InferenceServingEnv(pattern=pattern, seed=42)
    dqn = DQN("MlpPolicy", dqn_env, learning_rate=1e-3, buffer_size=5000,
              learning_starts=500, batch_size=64, gamma=0.99,
              policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=42, device="cpu")
    dqn.learn(total_timesteps=TIMESTEPS)
    res = eval_rl(dqn, pattern, "DQN")
    RESULTS.append(res)
    print(f"  {'DQN':20s} reward={res['reward_mean']:8.1f} served={res['served_mean']:6.0f}")

    # --- PPO (core) ---
    print(f"  Training PPO ({TIMESTEPS} steps)...")
    ppo_env = InferenceServingEnv(pattern=pattern, seed=42)
    ppo = PPO("MlpPolicy", ppo_env, learning_rate=3e-4, n_steps=512,
              batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.01,
              policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=42, device="cpu")
    ppo.learn(total_timesteps=TIMESTEPS)
    res = eval_rl(ppo, pattern, "PPO")
    RESULTS.append(res)
    print(f"  {'PPO':20s} reward={res['reward_mean']:8.1f} served={res['served_mean']:6.0f}")

    # --- SAC (core) ---
    print(f"  Training SAC ({TIMESTEPS} steps)...")
    sac_base = InferenceServingEnv(pattern=pattern, seed=42)
    sac_env = ContinuousToDiscreteWrapper(sac_base)
    sac = SAC("MlpPolicy", sac_env, learning_rate=3e-4, buffer_size=5000,
              learning_starts=500, batch_size=64, gamma=0.99, tau=0.005,
              ent_coef="auto",
              policy_kwargs=dict(net_arch=[64, 64]), verbose=0, seed=42, device="cpu")
    sac.learn(total_timesteps=TIMESTEPS)
    res = eval_rl(sac, pattern, "SAC", use_wrapper=True)
    RESULTS.append(res)
    print(f"  {'SAC':20s} reward={res['reward_mean']:8.1f} served={res['served_mean']:6.0f}")

# save all results
with open("results/all_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nResults saved to results/all_results.json")
print(f"Total experiments: {len(RESULTS)}")
