"""Plot dev mean latency against training steps for a group of conditions.

Usage:
    python scripts/plot_learning_curves.py --group primary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_batching.experiment import GROUPS  # noqa: E402
from adaptive_batching.training import SEEDS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"ppo": ("#2a78d6", "PPO"), "dqn": ("#eb6834", "DQN")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    args = parser.parse_args()
    conditions = GROUPS[args.group]
    scenarios = list(dict.fromkeys(c.scenario_name for c in conditions))
    columns = max(sum(c.scenario_name == s for c in conditions) for s in scenarios)

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(scenarios), columns, figsize=(4.6 * columns, 3.6 * len(scenarios)),
                             facecolor=SURFACE, squeeze=False)
    for row, scenario in enumerate(scenarios):
        row_conditions = [c for c in conditions if c.scenario_name == scenario]
        for col in range(columns):
            ax = axes[row][col]
            if col >= len(row_conditions):
                ax.set_visible(False)
                continue
            condition = row_conditions[col]
            ax.set_facecolor(SURFACE)
            baseline = json.loads((ROOT / f"results/baselines/{condition.id}.json").read_text())
            base_ms = 1e3 * baseline["strongest"]["dev_mean_latency_s"]
            ax.axhline(base_ms, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
            for algorithm, (color, _) in SERIES.items():
                curves = []
                for seed in SEEDS:
                    path = ROOT / f"results/rl/runs/{condition.id}/{algorithm}_seed{seed}.json"
                    curve = json.loads(path.read_text())["curve"]
                    steps = [p["steps"] / 1e3 for p in curve]
                    curves.append([1e3 * p["dev_mean_latency_s"] for p in curve])
                    ax.plot(steps, curves[-1], color=color, linewidth=0.8, alpha=0.35, zorder=3)
                ax.plot(steps, np.mean(curves, axis=0), color=color, linewidth=2, zorder=4)
            ax.set_yscale("log")
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{scenario} · {condition.trace} · load {condition.load}", fontsize=10,
                         loc="left", color=TEXT)
            if col == 0:
                ax.set_ylabel("dev mean latency (ms, log scale)")
            if row == len(scenarios) - 1:
                ax.set_xlabel("training decisions (thousands)")

    handles = [plt.Line2D([], [], color=c, linewidth=2, label=f"{name}: mean of 5 seeds "
                          "(thin lines: each seed)") for c, name in SERIES.values()]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)),
                              label="strongest rule-based baseline on dev"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncols=3,
               frameon=False)
    fig.suptitle("Learning curves on the dev split (checkpoint every 30,000 decisions)",
                 x=0.012, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = ROOT / f"results/rl/learning_curves_{args.group}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
