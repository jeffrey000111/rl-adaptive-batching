"""
Training Curve Comparison
Records episode rewards during training and plots learning curves
for PPO vs SAC vs DQN side by side.

Usage: python run_training_curves.py
Takes ~10 min on a laptop.
"""

import sys, os, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env.serving_env import InferenceServingEnv
from agents.sac_agent import ContinuousToDiscreteWrapper
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt


class RewardTracker(BaseCallback):
    """Records episode rewards during training."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self):
        # SB3 wraps env in Monitor, so we can read episode info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


PATTERN = "bursty"  # hardest pattern, most interesting curves
TIMESTEPS = 80000
SEED = 42

all_curves = {}

# --- DQN ---
print("Training DQN...")
env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
tracker = RewardTracker()
model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=10000,
            learning_starts=500, batch_size=64, gamma=0.99,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0, seed=SEED, device="cpu")
model.learn(total_timesteps=TIMESTEPS, callback=tracker)
all_curves["DQN"] = tracker.episode_rewards
print(f"  DQN: {len(tracker.episode_rewards)} episodes, final avg: {np.mean(tracker.episode_rewards[-5:]):.1f}")

# --- PPO ---
print("Training PPO...")
env = InferenceServingEnv(pattern=PATTERN, seed=SEED)
tracker = RewardTracker()
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512,
            batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0, seed=SEED, device="cpu")
model.learn(total_timesteps=TIMESTEPS, callback=tracker)
all_curves["PPO"] = tracker.episode_rewards
print(f"  PPO: {len(tracker.episode_rewards)} episodes, final avg: {np.mean(tracker.episode_rewards[-5:]):.1f}")

# --- SAC ---
print("Training SAC...")
base = InferenceServingEnv(pattern=PATTERN, seed=SEED)
env = ContinuousToDiscreteWrapper(base)
tracker = RewardTracker()
model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=10000,
            learning_starts=500, batch_size=64, gamma=0.99, tau=0.005,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0, seed=SEED, device="cpu")
model.learn(total_timesteps=TIMESTEPS, callback=tracker)
all_curves["SAC"] = tracker.episode_rewards
print(f"  SAC: {len(tracker.episode_rewards)} episodes, final avg: {np.mean(tracker.episode_rewards[-5:]):.1f}")

# --- Save raw data ---
os.makedirs("results/training_curves", exist_ok=True)
save_data = {}
for name, rewards in all_curves.items():
    save_data[name] = [float(r) for r in rewards]
with open("results/training_curves/training_curves.json", "w") as f:
    json.dump(save_data, f)

# --- Plot ---
def smooth(data, window=5):
    """simple moving average"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"DQN": "#ffcc99", "PPO": "#ff6666", "SAC": "#c299ff"}

# raw rewards
for name, rewards in all_curves.items():
    axes[0].plot(rewards, alpha=0.3, color=colors[name], linewidth=0.5)
    axes[0].plot(smooth(rewards, 5), color=colors[name], linewidth=2, label=name)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Episode Reward")
axes[0].set_title("Training Curves (Bursty Traffic, 80k steps)")
axes[0].legend()
axes[0].axhline(y=52.5, color="green", linestyle="--", alpha=0.5, label="Threshold baseline")
axes[0].axhline(y=0, color="black", linestyle="-", alpha=0.2)

# zoomed in on convergence (last 50%)
for name, rewards in all_curves.items():
    half = len(rewards) // 2
    if half > 0:
        axes[1].plot(range(half, len(rewards)), rewards[half:], alpha=0.3, color=colors[name], linewidth=0.5)
        axes[1].plot(range(half, len(rewards)), smooth(rewards[half:], 3), color=colors[name], linewidth=2, label=name)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Episode Reward")
axes[1].set_title("Convergence (Last 50% of Training)")
axes[1].legend()
axes[1].axhline(y=52.5, color="green", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/training_curves/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: results/training_curves/training_curves.png")
print("Saved: results/training_curves/training_curves.json")
