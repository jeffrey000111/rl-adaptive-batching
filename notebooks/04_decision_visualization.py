"""
Decision Visualization
Shows how each agent picks batch sizes over time in a single episode.
Helps us understand if the RL agents are actually doing something smart.

Usage: python notebooks/04_decision_visualization.py
"""

import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.serving_env import InferenceServingEnv
from baselines.heuristics import ThresholdBatcher
from agents.sac_agent import ContinuousToDiscreteWrapper
from stable_baselines3 import PPO, SAC, DQN
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
PATTERN = "bursty"
STEPS = 300  # just enough to see patterns


def collect_episode(env, agent, name, is_sb3=False, use_wrapper=False):
    """run one episode and record decisions"""
    if use_wrapper:
        run_env = ContinuousToDiscreteWrapper(env)
    else:
        run_env = env

    obs, _ = run_env.reset(seed=42)
    data = {"steps": [], "batch_size": [], "queue": [], "rate": [],
            "latency": [], "gpu_util": [], "reward": []}

    for step in range(STEPS):
        if is_sb3:
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.predict(obs)

        obs, r, done, _, info = run_env.step(action)
        data["steps"].append(step)
        data["batch_size"].append(info["batch_size"])
        data["queue"].append(info["queue"])
        data["rate"].append(info["rate"])
        data["latency"].append(info["latency_ms"])
        data["gpu_util"].append(info["gpu_util"])
        data["reward"].append(r)
        if done:
            break

    return data


def plot_decisions(all_data):
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "plots"), exist_ok=True)
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

    n_agents = len(all_data)
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    colors = {"Threshold": "#66b3ff", "DQN": "#ffcc99", "PPO": "#ff6666", "SAC": "#c299ff"}

    # plot 1: batch size choices over time
    for name, data in all_data.items():
        axes[0].plot(data["steps"], data["batch_size"], label=name,
                     alpha=0.8, linewidth=1.2, color=colors.get(name, "gray"))
    axes[0].set_ylabel("Batch Size")
    axes[0].set_title("Batch Size Decisions Over Time (Bursty Traffic)")
    axes[0].legend(loc="upper right")
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks(BATCH_SIZES)
    axes[0].set_yticklabels(BATCH_SIZES)

    # plot 2: queue length (same for all since env is seeded)
    # just use the first agent's data for queue/rate since env is shared
    first_data = list(all_data.values())[0]
    axes[1].fill_between(first_data["steps"], first_data["queue"],
                         alpha=0.3, color="steelblue")
    axes[1].plot(first_data["steps"], first_data["queue"],
                 linewidth=0.8, color="steelblue")
    axes[1].set_ylabel("Queue Length")
    axes[1].set_title("Queue Length")

    # plot 3: arrival rate
    axes[2].plot(first_data["steps"], first_data["rate"],
                 linewidth=0.8, color="coral")
    axes[2].set_ylabel("Arrival Rate (req/s)")
    axes[2].set_title("Request Arrival Rate")

    # plot 4: cumulative reward
    for name, data in all_data.items():
        cum_reward = np.cumsum(data["reward"])
        axes[3].plot(data["steps"], cum_reward, label=name,
                     alpha=0.8, linewidth=1.2, color=colors.get(name, "gray"))
    axes[3].set_ylabel("Cumulative Reward")
    axes[3].set_xlabel("Step")
    axes[3].set_title("Cumulative Reward")
    axes[3].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "decision_visualization.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: decision_visualization.png")


if __name__ == "__main__":
    print("Collecting decision data...")

    all_data = {}

    # threshold baseline
    env = InferenceServingEnv(pattern=PATTERN, seed=42)
    all_data["Threshold"] = collect_episode(env, ThresholdBatcher(), "Threshold")

    # check if trained models exist, otherwise train quick ones
    for algo_name, AlgoClass, use_wrapper in [("DQN", DQN, False), ("PPO", PPO, False), ("SAC", SAC, True)]:
        model_path = f"{algo_name.lower()}_{PATTERN}.zip"
        if os.path.exists(model_path):
            print(f"  Loading {algo_name} from {model_path}")
            model = AlgoClass.load(model_path)
        else:
            print(f"  Training {algo_name} (quick, 10k steps)...")
            base_env = InferenceServingEnv(pattern=PATTERN, seed=42)
            train_env = ContinuousToDiscreteWrapper(base_env) if use_wrapper else base_env
            if algo_name == "DQN":
                model = DQN("MlpPolicy", train_env, verbose=0, seed=42, device="cpu",
                            buffer_size=5000, learning_starts=200)
            elif algo_name == "PPO":
                model = PPO("MlpPolicy", train_env, verbose=0, seed=42, device="cpu")
            else:
                model = SAC("MlpPolicy", train_env, verbose=0, seed=42, device="cpu",
                            buffer_size=5000, learning_starts=200)
            model.learn(total_timesteps=10000)

        env = InferenceServingEnv(pattern=PATTERN, seed=42)
        all_data[algo_name] = collect_episode(env, model, algo_name,
                                               is_sb3=True, use_wrapper=use_wrapper)

    plot_decisions(all_data)
    print("Done!")
