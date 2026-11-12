"""Fill generated blocks in Markdown files from files under results/.

No number in README.md or REPORT.md is typed by hand. A document marks a
generated region with

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

and this script replaces the region body with the output of the renderer
registered under ``name``.

Usage:
    python scripts/render_docs.py           # rewrite documents in place
    python scripts/render_docs.py --check   # exit 1 if any document is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ["README.md", "REPORT.md"]

BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def legacy_fixed_batch_table() -> str:
    """Table showing that fixed batch sizes beat the originally reported PPO result."""
    data = json.loads((ROOT / "results/legacy_repro/fixed_batch.json").read_text())
    lines = [
        "| Policy | Avg JCT (steps) | P99 JCT (steps) | Source |",
        "|---|---|---|---|",
    ]
    for row in data["rows"]:
        lines.append(
            f"| {row['policy']} | {row['avg_jct']:.2f} | {row['p99_jct']:.2f} | {row['source']} |"
        )
    return "\n".join(lines) + "\n"


def fixed_batch_sweep_table(scenario: str) -> str:
    """Best fixed batch size per trace and load on the train split, with SLO violations."""
    data = json.loads((ROOT / "results/fixed_batch_sweep/train.json").read_text())

    def cell_text(run: dict[str, float], size: str) -> str:
        return (
            f"b={size}: {1e3 * run['mean_latency_s']:.0f} ms, "
            f"{100 * run['slo_violation_rate']:.1f}%"
        )

    def best(runs: dict[str, dict[str, float]]) -> str:
        size = min(runs, key=lambda b: runs[b]["mean_latency_s"])
        return cell_text(runs[size], size)

    info = data["scenarios"][scenario]
    lines = [
        f"Capacity {info['capacity_rps']:.1f} requests/s; SLO {1e3 * info['slo_s']:.0f} ms. "
        "Each cell: batch size, mean latency, share of requests over the SLO.",
        "",
        "| Trace | Nominal load | Load in busiest minute | No batching | "
        "Largest greedy cap | Best greedy cap | Best full-batch size |",
        "|---|---|---|---|---|---|---|",
    ]
    largest = str(max(data["batch_sizes"]))
    for cell in data["cells"]:
        if cell["scenario"] != scenario:
            continue
        greedy, full = cell["policies"]["greedy"], cell["policies"]["full"]
        lines.append(
            f"| {cell['trace']} | {cell['nominal_load']} | {cell['peak_load']:.2f} | "
            f"{cell_text(greedy['1'], '1')} | {cell_text(greedy[largest], largest)} | "
            f"{best(greedy)} | {best(full)} |"
        )
    return "\n".join(lines) + "\n"


RENDERERS: dict[str, Callable[[], str]] = {
    "legacy-fixed-batch": legacy_fixed_batch_table,
    "fixed-batch-sweep-single": lambda: fixed_batch_sweep_table("single"),
    "fixed-batch-sweep-pipeline": lambda: fixed_batch_sweep_table("pipeline"),
}


def render(text: str) -> str:
    """Return ``text`` with every generated block re-rendered."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in RENDERERS:
            raise KeyError(f"no renderer registered for generated block {name!r}")
        return match.group(1) + RENDERERS[name]() + match.group(4)

    return BLOCK.sub(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if a document is stale")
    args = parser.parse_args()

    stale = []
    for name in DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            continue
        old = path.read_text()
        new = render(old)
        if new != old:
            stale.append(name)
            if not args.check:
                path.write_text(new)
    if args.check and stale:
        print(f"stale generated blocks in: {', '.join(stale)} (run `make docs`)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
