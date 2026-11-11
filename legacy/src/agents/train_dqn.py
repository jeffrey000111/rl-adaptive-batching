"""
Train DQN agent on the inference serving environment.
Uses Stable-Baselines3.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from src.env.inference_env import InferenceServingEnv
import numpy as np


def train_dqn(traffic_pattern="steady", total_timesteps=50000, seed=42):
    env = InferenceServingEnv(traffic_pattern=traffic_pattern, seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=1,
        seed=seed,
    )

    print(f"Training DQN on {traffic_pattern} traffic for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # save
    save_path = f"models/dqn_{traffic_pattern}"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # quick eval
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model


def evaluate_dqn(model, env, n_episodes=20):
    all_metrics = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 100)
        done = False
        ep_latencies = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if info["batch_size"] > 0:
                ep_latencies.append(info["latency_ms"])
        metrics = env.get_metrics()
        metrics["p99_latency"] = np.percentile(ep_latencies, 99) if ep_latencies else 0
        all_metrics.append(metrics)
    return {
        "agent": "DQN",
        "avg_latency_ms": round(np.mean([m["avg_latency_ms"] for m in all_metrics]), 2),
        "p99_latency_ms": round(np.mean([m["p99_latency"] for m in all_metrics]), 2),
        "slo_attainment": round(np.mean([m["slo_attainment"] for m in all_metrics]), 4),
        "avg_throughput": round(np.mean([m["total_served"] for m in all_metrics]), 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic", default="steady", choices=["steady", "bursty", "diurnal"])
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()
    train_dqn(traffic_pattern=args.traffic, total_timesteps=args.timesteps)
