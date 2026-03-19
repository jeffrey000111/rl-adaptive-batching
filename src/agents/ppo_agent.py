"""
PPO agent for adaptive batching.
Per professor feedback: PPO is one of two core algorithms (PPO vs SAC).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from env.serving_env import InferenceServingEnv
import numpy as np


def train_ppo(pattern="bursty", total_timesteps=50000, seed=42):
    env = InferenceServingEnv(pattern=pattern, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )

    print(f"Training PPO on {pattern} traffic for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = f"ppo_{pattern}"
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    return model


def eval_ppo(model, pattern="bursty", episodes=10):
    env = InferenceServingEnv(pattern=pattern, seed=99)
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

    lat_arr = np.array(latencies)
    print(f"\nPPO eval ({pattern}, {episodes} eps):")
    print(f"  reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  served: {np.mean(served_list):.0f}")
    print(f"  violations: {np.mean(viol_list):.0f}")
    print(f"  latency mean: {np.mean(lat_arr):.1f}ms, p99: {np.percentile(lat_arr, 99):.1f}ms")

    return {
        "name": "PPO",
        "reward_mean": np.mean(rewards),
        "served_mean": np.mean(served_list),
        "violations_mean": np.mean(viol_list),
        "latency_mean": np.mean(lat_arr),
        "latency_p99": np.percentile(lat_arr, 99),
    }


if __name__ == "__main__":
    model = train_ppo(pattern="bursty", total_timesteps=20000)
    eval_ppo(model, pattern="bursty", episodes=5)
