"""
SAC (Soft Actor-Critic) agent for adaptive batching.
Per professor feedback: SAC brings the best of DQN and actor-critic together.
Core comparison is PPO vs SAC.

Note: SAC in SB3 normally expects continuous actions. We use a wrapper
that maps continuous output to discrete batch sizes.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from env.serving_env import InferenceServingEnv
import numpy as np


class ContinuousToDiscreteWrapper(gym.ActionWrapper):
    """
    Wraps a discrete-action env so SAC can use it.
    SAC outputs a continuous action in [-1, 1], we map it to one of the
    discrete batch size indices.
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_actions = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, act):
        # map [-1, 1] -> [0, n_actions-1]
        continuous = float(act[0])
        idx = int(np.clip(
            np.round((continuous + 1) / 2 * (self.n_actions - 1)),
            0, self.n_actions - 1
        ))
        return idx


def train_sac(pattern="bursty", total_timesteps=50000, seed=42):
    base_env = InferenceServingEnv(pattern=pattern, seed=seed)
    env = ContinuousToDiscreteWrapper(base_env)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )

    print(f"Training SAC on {pattern} traffic for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = f"sac_{pattern}"
    model.save(save_path)
    print(f"Saved to {save_path}.zip")
    return model


def eval_sac(model, pattern="bursty", episodes=10):
    base_env = InferenceServingEnv(pattern=pattern, seed=99)
    env = ContinuousToDiscreteWrapper(base_env)
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
    print(f"\nSAC eval ({pattern}, {episodes} eps):")
    print(f"  reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  served: {np.mean(served_list):.0f}")
    print(f"  violations: {np.mean(viol_list):.0f}")
    print(f"  latency mean: {np.mean(lat_arr):.1f}ms, p99: {np.percentile(lat_arr, 99):.1f}ms")

    return {
        "name": "SAC",
        "reward_mean": np.mean(rewards),
        "served_mean": np.mean(served_list),
        "violations_mean": np.mean(viol_list),
        "latency_mean": np.mean(lat_arr),
        "latency_p99": np.percentile(lat_arr, 99),
    }


if __name__ == "__main__":
    model = train_sac(pattern="bursty", total_timesteps=20000)
    eval_sac(model, pattern="bursty", episodes=5)
