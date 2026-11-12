"""Summary figure: each seed's test mean latency divided by the strongest rule's.

One point per trained seed of the best RL algorithm (chosen on dev) in every
condition, log scale, with reference lines at 1 (equal to the rule) and at
the post-hoc failure threshold. Reads results/posthoc/per_seed/.

Usage:
    python scripts/plot_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_batching.experiment import PRIMARY, SECONDARY  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POINT = "#2a78d6"
ORDER = [c for scenario in ("single", "pipeline") for c in PRIMARY + SECONDARY
         if c.scenario_name == scenario]


def main() -> None:
    primary_ids = {c.id for c in PRIMARY}
    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    failure_ratio = None
    rng = np.random.default_rng(0)  # horizontal jitter only, for legibility
    labels = []
    for x, condition in enumerate(ORDER):
        record = json.loads(
            (ROOT / f"results/posthoc/per_seed/{condition.id}.json").read_text()
        )
        failure_ratio = record["failure_ratio"]
        base = record["strongest_baseline"]["mean_latency_s"]
        chosen = record["algorithms"][record["best_rl_algorithm"]]
        seeds = np.array(chosen["mean_latency_s_by_seed"])
        ratios = seeds / base
        filled = condition.id in primary_ids
        ax.scatter(x + rng.uniform(-0.18, 0.18, size=len(ratios)), ratios, s=42, zorder=3,
                   facecolors=POINT if filled else SURFACE, edgecolors=POINT, linewidths=1.5)
        labels.append(f"{condition.trace} {condition.load}\n{record['best_rl_algorithm'].upper()}")
    split = sum(c.scenario_name == "single" for c in ORDER) - 0.5
    ax.axvline(split, color=GRID, linewidth=1.2, zorder=1)
    ax.axhline(1.0, color=TEXT, linewidth=1.2, zorder=2)
    ax.axhline(failure_ratio, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)), labels, fontsize=8.5)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel("RL mean latency / strongest rule (test, log scale)")
    top = ax.get_ylim()[1]
    ax.text(split / 2, top, "single stage", ha="center", va="top", color=MUTED)
    ax.text((split + len(ORDER) - 0.5) / 2, top, "three-stage pipeline", ha="center", va="top",
            color=MUTED)
    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=POINT,
                   markeredgecolor=POINT, label="one training seed, primary condition"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                   markeredgecolor=POINT, markeredgewidth=1.5,
                   label="one training seed, secondary condition"),
        plt.Line2D([], [], color=TEXT, linewidth=1.2, label="1: same mean latency as the rule"),
        plt.Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
                   label=f"{failure_ratio:g}: post-hoc threshold for a failed seed"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncols=2,
               frameon=False)
    fig.suptitle("Learned policy relative to the strongest tuned rule, per seed "
                 "(below 1 would mean RL is faster)", x=0.012, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = ROOT / "results/posthoc/summary_ratio_by_seed.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
