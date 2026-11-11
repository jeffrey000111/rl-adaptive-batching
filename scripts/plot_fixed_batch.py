"""Plot mean latency against fixed batch size from results/fixed_batch_sweep/train.json.

Usage:
    python scripts/plot_fixed_batch.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = {
    "greedy": ("#2a78d6", "o", "dispatch immediately, at most b per batch"),
    "full": ("#eb6834", "s", "wait for a full batch of b"),
}


def main() -> None:
    data = json.loads((ROOT / "results/fixed_batch_sweep/train.json").read_text())
    sizes = data["batch_sizes"]
    positions = list(range(len(sizes)))
    first_extrapolated = next(i for i, b in enumerate(sizes) if b > data["measured_max_batch"])
    traces = sorted({c["trace"] for c in data["cells"]}, reverse=True)
    loads = sorted({c["nominal_load"] for c in data["cells"]})
    cells = {(c["trace"], c["nominal_load"]): c for c in data["cells"]}

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(traces), len(loads), figsize=(12, 7.2), sharey="row",
                             facecolor=SURFACE)
    for row, trace in enumerate(traces):
        for col, load in enumerate(loads):
            ax = axes[row][col]
            cell = cells[(trace, load)]
            ax.set_facecolor(SURFACE)
            ax.axvspan(first_extrapolated - 0.5, len(sizes) - 0.5, color="#efeeea", zorder=0)
            for family, (color, marker, _) in SERIES.items():
                values = [1e3 * cell["policies"][family][str(b)]["mean_latency_s"] for b in sizes]
                ax.plot(positions, values, color=color, marker=marker, linewidth=2,
                        markersize=6, zorder=3)
                best = min(positions, key=values.__getitem__)
                ax.plot(best, values[best], marker="o", markersize=15, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.5, zorder=4)
            ax.set_yscale("log")
            ax.set_xticks(positions, [str(b) for b in sizes])
            ax.set_xlim(-0.5, len(sizes) - 0.5)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{trace} trace · load {load} (busiest minute {cell['peak_load']:.2f})",
                         fontsize=10, loc="left", color=TEXT)
            if col == 0:
                ax.set_ylabel("mean latency (ms, log scale)")
            if row == len(traces) - 1:
                ax.set_xlabel("batch size b")
            if row == 0 and col == 0:
                ax.text(len(sizes) - 0.55, 0.97, "b > 8: extrapolated\nprocessing time",
                        transform=ax.get_xaxis_transform(), ha="right", va="top",
                        fontsize=8.5, color=MUTED)

    handles = [plt.Line2D([], [], color=color, marker=marker, linewidth=2, markersize=6,
                          label=label) for color, marker, label in SERIES.values()]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=12,
                              markerfacecolor="none", markeredgecolor=MUTED,
                              markeredgewidth=1.5, label="lowest point of each curve"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.94), ncols=3,
               frameon=False, fontsize=10)
    fig.suptitle("Mean latency of fixed-batch-size policies, train split, one BERT-base T4 server "
                 "(simulated)", x=0.012, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = ROOT / "results/fixed_batch_sweep/latency_vs_batch.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
