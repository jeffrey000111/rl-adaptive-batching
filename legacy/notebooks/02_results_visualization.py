"""
Training Results Visualization
Generates all the plots for the final report.

Run this AFTER run_all.py has finished and results/all_results.json exists.
Usage: python notebooks/02_results_visualization.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ---- load results ----
results_path = os.path.join(os.path.dirname(__file__), "..", "results", "all_results.json")

if not os.path.exists(results_path):
    print("No results found. Run 'python run_all.py' first.")
    print("Generating dummy results for testing the plots...")
    # dummy data so you can test the notebook before running full experiments
    results = []
    for pattern in ["steady", "bursty", "diurnal"]:
        results.append({"name": "Static(bs=8)", "pattern": pattern, "reward_mean": -200 if pattern=="bursty" else 30, "served_mean": 5000, "violations_mean": 0, "latency_mean": 20, "latency_p99": 28})
        results.append({"name": "Timeout", "pattern": pattern, "reward_mean": 50, "served_mean": 8000, "violations_mean": 0, "latency_mean": 25, "latency_p99": 45})
        results.append({"name": "Threshold", "pattern": pattern, "reward_mean": 52, "served_mean": 8100, "violations_mean": 0, "latency_mean": 26, "latency_p99": 45})
        results.append({"name": "DQN", "pattern": pattern, "reward_mean": 55, "served_mean": 8200, "violations_mean": 1, "latency_mean": 22, "latency_p99": 40})
        results.append({"name": "PPO", "pattern": pattern, "reward_mean": 62, "served_mean": 8500, "violations_mean": 0, "latency_mean": 21, "latency_p99": 38})
        results.append({"name": "SAC", "pattern": pattern, "reward_mean": 58, "served_mean": 8400, "violations_mean": 1, "latency_mean": 23, "latency_p99": 42})
else:
    with open(results_path) as f:
        results = json.load(f)

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "plots"), exist_ok=True)
plot_dir = os.path.join(os.path.dirname(__file__), "..", "results", "plots")


# ---- plot 1: reward comparison bar chart ----
def plot_reward_comparison():
    patterns = ["steady", "bursty", "diurnal"]
    agents = ["Static(bs=8)", "Timeout", "Threshold", "DQN", "PPO", "SAC"]
    colors = ["#999999", "#66b3ff", "#99ff99", "#ffcc99", "#ff6666", "#c299ff"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for i, pat in enumerate(patterns):
        pat_results = [r for r in results if r.get("pattern") == pat]
        names = []
        rewards = []
        for agent in agents:
            match = [r for r in pat_results if r["name"] == agent]
            if match:
                names.append(agent)
                rewards.append(match[0]["reward_mean"])

        bars = axes[i].bar(range(len(names)), rewards, color=colors[:len(names)])
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        axes[i].set_title(f"{pat.capitalize()} Traffic", fontsize=13)
        axes[i].set_ylabel("Average Reward" if i == 0 else "")
        axes[i].axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    plt.suptitle("Reward Comparison Across Traffic Patterns", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reward_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: reward_comparison.png")


# ---- plot 2: latency comparison ----
def plot_latency_comparison():
    patterns = ["steady", "bursty", "diurnal"]
    agents = ["Timeout", "Threshold", "DQN", "PPO", "SAC"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, pat in enumerate(patterns):
        pat_results = [r for r in results if r.get("pattern") == pat]
        names, means, p99s = [], [], []
        for agent in agents:
            match = [r for r in pat_results if r["name"] == agent]
            if match:
                names.append(agent)
                means.append(match[0].get("latency_mean", 0))
                p99s.append(match[0].get("latency_p99", 0))

        x = np.arange(len(names))
        w = 0.35
        axes[i].bar(x - w/2, means, w, label="Mean Latency", color="#4c72b0")
        axes[i].bar(x + w/2, p99s, w, label="P99 Latency", color="#dd8452")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        axes[i].set_title(f"{pat.capitalize()} Traffic")
        axes[i].set_ylabel("Latency (ms)" if i == 0 else "")
        if i == 0:
            axes[i].legend()

    plt.suptitle("Latency: Mean vs P99", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "latency_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: latency_comparison.png")


# ---- plot 3: throughput comparison ----
def plot_throughput():
    patterns = ["steady", "bursty", "diurnal"]
    agents = ["Static(bs=8)", "Timeout", "Threshold", "DQN", "PPO", "SAC"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(patterns))
    w = 0.12
    for j, agent in enumerate(agents):
        vals = []
        for pat in patterns:
            match = [r for r in results if r["name"] == agent and r.get("pattern") == pat]
            vals.append(match[0]["served_mean"] if match else 0)
        ax.bar(x + j * w, vals, w, label=agent)

    ax.set_xticks(x + w * 2.5)
    ax.set_xticklabels([p.capitalize() for p in patterns])
    ax.set_ylabel("Total Requests Served")
    ax.set_title("Throughput Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "throughput_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: throughput_comparison.png")


# ---- plot 4: PPO vs SAC head-to-head ----
def plot_ppo_vs_sac():
    metrics = ["reward_mean", "served_mean", "latency_mean", "latency_p99", "violations_mean"]
    labels = ["Reward", "Throughput", "Mean Latency\n(ms)", "P99 Latency\n(ms)", "SLO\nViolations"]
    patterns = ["steady", "bursty", "diurnal"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, pat in enumerate(patterns):
        ppo = [r for r in results if r["name"] == "PPO" and r.get("pattern") == pat]
        sac = [r for r in results if r["name"] == "SAC" and r.get("pattern") == pat]

        if not ppo or not sac:
            continue

        ppo_vals = [ppo[0].get(m, 0) for m in metrics]
        sac_vals = [sac[0].get(m, 0) for m in metrics]

        x = np.arange(len(metrics))
        w = 0.35
        axes[i].bar(x - w/2, ppo_vals, w, label="PPO", color="#ff6666")
        axes[i].bar(x + w/2, sac_vals, w, label="SAC", color="#c299ff")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, fontsize=8)
        axes[i].set_title(f"{pat.capitalize()} Traffic")
        if i == 0:
            axes[i].legend()

    plt.suptitle("PPO vs SAC: Head-to-Head", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ppo_vs_sac.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: ppo_vs_sac.png")


# ---- plot 5: summary table ----
def print_summary_table():
    print("\n" + "=" * 90)
    print(f"{'Agent':20s} {'Pattern':10s} {'Reward':>10s} {'Served':>10s} {'Lat Mean':>10s} {'Lat P99':>10s} {'Violations':>10s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x.get("pattern",""), x["name"])):
        print(f"{r['name']:20s} {r.get('pattern','?'):10s} "
              f"{r.get('reward_mean',0):10.1f} {r.get('served_mean',0):10.0f} "
              f"{r.get('latency_mean',0):10.1f} {r.get('latency_p99',0):10.1f} "
              f"{r.get('violations_mean',0):10.0f}")
    print("=" * 90)


if __name__ == "__main__":
    print("Generating plots...")
    plot_reward_comparison()
    plot_latency_comparison()
    plot_throughput()
    plot_ppo_vs_sac()
    print_summary_table()
    print(f"\nAll plots saved to {plot_dir}")
