"""
DQN agent for adaptive batching.
Per professor feedback: DQN is our baseline RL algorithm.
Core investigation is PPO vs SAC.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from env.serving_env import InferenceServingEnv
import numpy as np


def train_dqn(pattern="bursty", total_timesteps=50000, seed=42):
    """train a DQN agent on the serving env"""
    env = InferenceServingEnv(pattern=pattern, seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=250,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )

    print(f"Training DQN on {pattern} traffic for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # save
    save_path = f"dqn_{pattern}"
    model.save(save_path)
    print(f"Saved to {save_path}.zip")

    return model


def eval_dqn(model, pattern="bursty", episodes=10):
    """evaluate trained DQN"""
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
    print(f"\nDQN eval ({pattern}, {episodes} eps):")
    print(f"  reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  served: {np.mean(served_list):.0f}")
    print(f"  violations: {np.mean(viol_list):.0f}")
    print(f"  latency mean: {np.mean(lat_arr):.1f}ms, p99: {np.percentile(lat_arr, 99):.1f}ms")

    return {
        "name": "DQN",
        "reward_mean": np.mean(rewards),
        "served_mean": np.mean(served_list),
        "violations_mean": np.mean(viol_list),
        "latency_mean": np.mean(lat_arr),
        "latency_p99": np.percentile(lat_arr, 99),
    }


if __name__ == "__main__":
    model = train_dqn(pattern="bursty", total_timesteps=20000)
    eval_dqn(model, pattern="bursty", episodes=5)
