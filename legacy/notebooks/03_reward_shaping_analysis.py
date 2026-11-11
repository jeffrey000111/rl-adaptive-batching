"""
Reward Shaping Analysis
Experiments with different latency/throughput weight combinations
to understand how reward design affects learned policies.

Usage: python notebooks/03_reward_shaping_analysis.py
"""

import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.serving_env import InferenceServingEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import json

# we test different reward weightings to see how they change behavior
# lat_w + tput_w don't have to sum to 1, they're just relative weights
REWARD_CONFIGS = [
    {"lat_w": 0.9, "tput_w": 0.1, "label": "latency-heavy (0.9/0.1)"},
    {"lat_w": 0.7, "tput_w": 0.3, "label": "latency-leaning (0.7/0.3)"},
    {"lat_w": 0.5, "tput_w": 0.5, "label": "balanced (0.5/0.5)"},
    {"lat_w": 0.3, "tput_w": 0.7, "label": "throughput-leaning (0.3/0.7)"},
    {"lat_w": 0.1, "tput_w": 0.9, "label": "throughput-heavy (0.1/0.9)"},
]

PATTERN = "bursty"
TIMESTEPS = 20000
EVAL_EPS = 5
RESULTS = []


def train_and_eval(lat_w, tput_w, label):
    print(f"\n  Training PPO with {label}...")

    env = InferenceServingEnv(pattern=PATTERN, seed=42, lat_w=lat_w, tput_w=tput_w)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=512,
                batch_size=64, n_epochs=10, gamma=0.99,
                policy_kwargs=dict(net_arch=[64, 64]),
                verbose=0, seed=42, device="cpu")
    model.learn(total_timesteps=TIMESTEPS)

    # eval
    eval_env = InferenceServingEnv(pattern=PATTERN, seed=99, lat_w=lat_w, tput_w=tput_w)
    rewards, latencies, served_list, batch_choices = [], [], [], []

    for ep in range(EVAL_EPS):
        obs, _ = eval_env.reset(seed=ep + 300)
        ep_reward = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = eval_env.step(a)
            ep_reward += r
            batch_choices.append(info["batch_size"])
            if info["served"] > 0:
                latencies.append(info["latency_ms"])
            if done:
                break
        rewards.append(ep_reward)
        served_list.append(info["total_served"])

    lat_arr = np.array(latencies) if latencies else np.array([0])
    batch_arr = np.array(batch_choices)

    result = {
        "label": label,
        "lat_w": lat_w,
        "tput_w": tput_w,
        "reward_mean": float(np.mean(rewards)),
        "served_mean": float(np.mean(served_list)),
        "latency_mean": float(np.mean(lat_arr)),
        "latency_p99": float(np.percentile(lat_arr, 99)),
        "avg_batch_size": float(np.mean(batch_arr)),
        "batch_size_std": float(np.std(batch_arr)),
    }
    print(f"    reward={result['reward_mean']:.1f}, served={result['served_mean']:.0f}, "
          f"lat_mean={result['latency_mean']:.1f}ms, avg_batch={result['avg_batch_size']:.1f}")
    return result


def plot_reward_shaping_results(results):
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "plots"), exist_ok=True)
    plot_dir = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

    labels = [r["label"] for r in results]
    x = range(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # reward
    axes[0, 0].bar(x, [r["reward_mean"] for r in results], color="#4c72b0")
    axes[0, 0].set_title("Average Reward")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    # throughput
    axes[0, 1].bar(x, [r["served_mean"] for r in results], color="#55a868")
    axes[0, 1].set_title("Throughput (requests served)")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    # latency
    axes[1, 0].bar(x, [r["latency_mean"] for r in results], color="#dd8452", label="Mean")
    axes[1, 0].bar(x, [r["latency_p99"] for r in results], alpha=0.4, color="#c44e52", label="P99")
    axes[1, 0].set_title("Latency (ms)")
    axes[1, 0].set_xticks(list(x))
    axes[1, 0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1, 0].legend()

    # avg batch size chosen
    axes[1, 1].bar(x, [r["avg_batch_size"] for r in results], color="#8172b2")
    axes[1, 1].set_title("Average Batch Size Chosen")
    axes[1, 1].set_xticks(list(x))
    axes[1, 1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    plt.suptitle("Effect of Reward Shaping on PPO Policy (Bursty Traffic)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reward_shaping_analysis.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: reward_shaping_analysis.png")


if __name__ == "__main__":
    print("=" * 60)
    print("  REWARD SHAPING ANALYSIS")
    print("=" * 60)

    for cfg in REWARD_CONFIGS:
        result = train_and_eval(cfg["lat_w"], cfg["tput_w"], cfg["label"])
        RESULTS.append(result)

    # save
    os.makedirs("results/reward_shaping", exist_ok=True)
    with open("results/reward_shaping/reward_shaping_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

    plot_reward_shaping_results(RESULTS)

    print("\n\nKey takeaways:")
    print("-" * 50)
    for r in RESULTS:
        print(f"  {r['label']:35s} avg_batch={r['avg_batch_size']:5.1f} lat={r['latency_mean']:5.1f}ms")
    print("\nExpected: latency-heavy => smaller batches, throughput-heavy => bigger batches")
