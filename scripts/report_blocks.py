"""Renderers for the generated blocks of REPORT.md (and the README summary).

Every number comes from a file under results/, from data/ via the package's
own loaders, or from a constant defined in the package. Nothing is typed in.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adaptive_batching.experiment import (
    ABLATION_BASE,
    ABLATION_LEN128,
    PRIMARY,
    PRIMARY_COMPARISON,
    SECONDARY,
    Condition,
    _raw_splits,
)
from adaptive_batching.load import capacity_rps
from adaptive_batching.scenarios import LOADS, MAX_BATCH, SCENARIOS
from adaptive_batching.service_time import MEASURED_MAX_BATCH

ROOT = Path(__file__).resolve().parent.parent
ALL = PRIMARY + SECONDARY


def _training_settings() -> tuple[int, int]:
    """(seeds per configuration, decisions per run), read from the results."""
    chosen = json.loads((ROOT / "results/rl/hyperparameters.json").read_text())
    steps = {entry["total_steps"] for entry in chosen.values()}
    if len(steps) != 1:
        raise ValueError("algorithms were trained with different budgets")
    seeds = len(next(iter(json.loads(
        (ROOT / f"results/test/{PRIMARY_COMPARISON.id}.json").read_text()
    )["algorithms"].values()))["metrics_by_seed"])
    return seeds, steps.pop()


NUM_SEEDS, TOTAL_STEPS = _training_settings()
SEEDS = range(NUM_SEEDS)
VERDICTS = {
    "rl_better": "RL better",
    "rl_worse": "RL worse",
    "trade_off": "trade-off, not a win",
    "no_difference": "no difference",
}


def read(path: str) -> Any:  # noqa: ANN401 - parsed JSON
    return json.loads((ROOT / path).read_text())


def ms(seconds: float) -> str:
    value = 1e3 * seconds
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def interval(entry: dict[str, float], scale: float, unit: str, digits: int = 1) -> str:
    return (
        f"{scale * entry['estimate']:+.{digits}f}{unit} "
        f"[{scale * entry['low']:+.{digits}f}, {scale * entry['high']:+.{digits}f}]"
    )


def latency_interval(result: dict[str, Any]) -> str:
    return interval(result["intervals"]["mean_latency_relative_difference"], 100, "%")


def slo_interval(result: dict[str, Any]) -> str:
    return interval(result["intervals"]["slo_violation_difference_points"], 1, " pt", 2)


def verdict(result: dict[str, Any]) -> str:
    text = VERDICTS[result["verdict"]]
    if result["verdict"] == "no_difference":
        text += f" ({result['intervals']['mean_latency_relative_difference']['verdict']})"
    return text


def times(count: int) -> str:
    return "once" if count == 1 else f"{count} times"


def label(condition: Condition) -> str:
    return f"{condition.scenario_name}, {condition.trace}, load {condition.load}"


def spec_text(spec: dict[str, Any]) -> str:
    family = spec["family"]
    if family == "greedy":
        return f"greedy(cap {spec['cap']})"
    if family == "full":
        return f"full(b {spec['batch']}, wait {spec['wait_mult']}x)"
    if family == "threshold":
        return f"threshold(k {spec['k']}, wait {spec['wait_mult']}x)"
    return f"downstream(k {spec['k']}, wait {spec['wait_mult']}x, hold {spec['hold_at']})"


def config_text(config: list[dict[str, Any]]) -> str:
    return " / ".join(spec_text(spec) for spec in config)


def test(condition: Condition) -> dict[str, Any]:
    return read(f"results/test/{condition.id}.json")


def best(condition: Condition) -> dict[str, Any]:
    record = test(condition)
    return record["algorithms"][record["best_rl_algorithm"]]


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- blocks


def conclusion() -> str:
    """Three to four sentences: single stage, pipeline, scope and data limits."""
    verdicts = {c.id: best(c)["verdict"] for c in ALL}
    better = [c for c in ALL if verdicts[c.id] == "rl_better"]
    single = [c for c in ALL if c.scenario_name == "single"]
    single_conv = [c for c in single if c.trace == "conv"]
    single_counts = Counter(verdicts[c.id] for c in single)
    conv_gap = max(
        abs(best(c)["intervals"]["mean_latency_relative_difference"]["estimate"])
        for c in single_conv
    )

    primary = test(PRIMARY_COMPARISON)
    primary_best = best(PRIMARY_COMPARISON)
    seeds = read(f"results/posthoc/per_seed/{PRIMARY_COMPARISON.id}.json")
    seed_info = seeds["algorithms"][primary["best_rl_algorithm"]]
    base_latency = primary["strongest_baseline"]["metrics"]["mean_latency_s"]
    ok = [
        abs(s / base_latency - 1)
        for s in seed_info["mean_latency_s_by_seed"]
        if s <= seeds["failure_ratio"] * base_latency
    ]
    pipeline = [c for c in ALL if c.scenario_name == "pipeline"]
    pipeline_counts = Counter(verdicts[c.id] for c in pipeline)

    conv_test_loads = [test(c)["test_split"]["load"] for c in PRIMARY if c.load == max(LOADS)]
    code_requests = test(next(c for c in SECONDARY if c.trace == "code"))["test_split"]["requests"]
    hours = max(sum(s.duration_s for s in _raw_splits(t).values()) for t in ("conv", "code")) / 3600

    if better:
        raise ValueError("the conclusion text assumes no condition with verdict 'RL better'")
    if primary_best["verdict"] != "rl_worse":
        raise ValueError("the conclusion text assumes the primary comparison is 'RL worse'")

    return (
        f"**Single stage.** With one server the learned policy never beat the tuned rule: in "
        f"{len(single)} single-stage conditions the pre-registered verdict was \"no difference\" "
        f"{times(single_counts['no_difference'])} and \"RL worse\" "
        f"{times(single_counts['rl_worse'])}, and on the conversation trace the best learned "
        f"policy reproduced the rule \"never wait\" almost exactly (largest difference in "
        f"mean latency {100 * conv_gap:.1f}%).\n\n"
        f"**Pipeline.** In the three-stage pipeline the pre-registered primary comparison "
        f"({PRIMARY_COMPARISON.trace}, load {PRIMARY_COMPARISON.load}, mean latency, "
        f"{primary['best_rl_algorithm'].upper()} against the strongest rule) found the learned "
        f"policy **worse**: {latency_interval(primary_best)} on the test split. The cause is "
        f"unstable training rather than a uniformly bad policy: {seed_info['failed_seeds']} of "
        f"{len(SEEDS)} seeds failed (test mean latency above {seeds['failure_ratio']:.0f}x the "
        f"rule's), while the other {len(ok)} were within {100 * max(ok):.1f}% of the rule. "
        f"Over all {len(pipeline)} pipeline conditions the verdict was \"RL worse\" "
        f"{times(pipeline_counts['rl_worse'])} and \"no difference\" "
        f"{times(pipeline_counts['no_difference'])}. **In none of the {len(ALL)} conditions was "
        f"the learned policy better than the tuned rule.**\n\n"
        f"**Scope and data limits.** Arrival times come from about {hours:.0f} hour of one day "
        f"of an LLM-service trace; the test split is lighter than its nominal load (realised "
        f"load {min(conv_test_loads):.2f} at nominal {max(LOADS)} on the conversation trace) and "
        f"the code trace's test split has only {code_requests} requests; the processing-time "
        f"model describes whole-batch inference of an encoder model and is a linear "
        f"extrapolation above batch size {MEASURED_MAX_BATCH}. The result is for DQN and PPO "
        f"with a budget of {TOTAL_STEPS:,} decisions per run.\n"
    )


def setup() -> str:
    rows = []
    for trace in ("conv", "code"):
        for name, split in _raw_splits(trace).items():
            rows.append(
                [trace, name, f"{len(split):,}", f"{split.duration_s / 60:.1f}",
                 f"{split.mean_rate:.2f}"]
            )
    out = "Trace splits (by time, before load scaling):\n\n"
    out += table(["Trace", "Split", "Requests", "Minutes", "Requests/s"], rows)
    out += "\nScenarios:\n\n"
    out += table(
        ["Scenario", "Stages", "Capacity (requests/s)", "Primary SLO (ms)", "Tight SLO (ms)"],
        [
            [s.name, s.description, f"{capacity_rps(s.stages, MAX_BATCH):.1f}",
             f"{1e3 * s.slo_s:.0f}", f"{1e3 * s.tight_slo_s:.0f}"]
            for s in SCENARIOS.values()
        ],
    )
    return out


def results_table(conditions: tuple[Condition, ...]) -> str:
    rows = []
    for c in conditions:
        record, result = test(c), best(c)
        base = record["strongest_baseline"]["metrics"]
        split = record["test_split"]
        rows.append(
            [
                label(c),
                f"{split['load']:.2f} ({split['peak_load']:.2f})",
                f"{ms(base['mean_latency_s'])} ms, {100 * base['slo_violation_rate']:.1f}%",
                f"{record['best_rl_algorithm'].upper()}: {ms(result['mean_latency_s'])} ms",
                latency_interval(result),
                slo_interval(result),
                verdict(result),
            ]
        )
    return table(
        ["Condition", "Realised test load (busiest minute)",
         "Strongest rule: mean latency, SLO misses", "Best RL: mean of 5 seeds",
         "Mean latency, RL vs rule [95% CI]", "SLO misses, RL vs rule [95% CI]", "Verdict"],
        rows,
    )


def primary_comparison() -> str:
    record, result = test(PRIMARY_COMPARISON), best(PRIMARY_COMPARISON)
    base = record["strongest_baseline"]
    seeds = ", ".join(ms(m["mean_latency_s"]) for m in result["metrics_by_seed"])
    return (
        f"- Condition: {label(PRIMARY_COMPARISON)}; test split of "
        f"{record['test_split']['requests']:,} requests.\n"
        f"- Strongest rule (chosen on dev): {config_text(base['config'])}; test mean latency "
        f"{ms(base['metrics']['mean_latency_s'])} ms.\n"
        f"- Best RL (chosen on dev): {record['best_rl_algorithm'].upper()}; test mean latency "
        f"of the five seeds: {seeds} ms (average {ms(result['mean_latency_s'])} ms).\n"
        f"- Mean latency, RL relative to the rule: **{latency_interval(result)}**.\n"
        f"- SLO violation rate, RL minus rule: {slo_interval(result)}.\n"
        f"- Pre-registered verdict: **{verdict(result)}**.\n"
    )


def both_algorithms() -> str:
    rows = []
    for c in ALL:
        record = test(c)
        cells = [label(c), record["best_rl_algorithm"].upper()]
        for algorithm in ("ppo", "dqn"):
            result = record["algorithms"][algorithm]
            cells.append(f"{latency_interval(result)}, {verdict(result)}")
        rows.append(cells)
    return table(["Condition", "Best RL (by dev)", "PPO vs rule", "DQN vs rule"], rows)


def baselines() -> str:
    rows = []
    for c in ALL:
        record = read(f"results/baselines/{c.id}.json")
        families = record["best_by_family"]
        rows.append(
            [label(c), config_text(record["strongest"]["config"]),
             ms(record["strongest"]["dev_mean_latency_s"])]
            + [
                ms(families[f]["dev_mean_latency_s"]) if f in families else "n/a"
                for f in ("greedy", "full", "threshold", "downstream")
            ]
        )
    return table(
        ["Condition", "Strongest rule (per stage)", "Dev mean (ms)", "Best greedy",
         "Best full-batch", "Best threshold", "Best downstream-aware"],
        rows,
    )


def sufficiency() -> str:
    history = read("results/rl/sufficiency_check.json")
    lines = []
    for entry in history:
        values = ", ".join(
            f"{a.upper()} {ms(v)} ms" for a, v in entry["dev_mean_latency_s_by_algorithm"].items()
        )
        lines.append(
            f"- Budget {entry['training_budget']:,} decisions: selected tuning configurations "
            f"reached {values} on dev; the untuned rule \"{entry['reference_rule']}\" reached "
            f"{ms(entry['reference_dev_mean_latency_s'])} ms. Ratio of the better algorithm "
            f"{entry['ratio']:.2f} against a threshold of {entry['threshold_ratio']}: "
            f"**{'passed' if entry['passed'] else 'failed'}**."
        )
    escalated = len(history) > 1
    lines.append(
        f"- Escalation of the training budget: **{'triggered' if escalated else 'not triggered'}**."
    )
    return "\n".join(lines) + "\n"


def memorisation() -> str:
    rows = []
    for c in ALL:
        record = read(f"results/selection/{c.id}.json")
        algorithm = record["best_rl_algorithm"]
        info = record["algorithms"][algorithm]
        ratio = info["ratio_to_strongest_baseline"]
        rows.append(
            [label(c), algorithm.upper(), f"{ratio['train']:.3f}", f"{ratio['dev']:.3f}",
             f"{best(c)['ratio_to_strongest_baseline']:.3f}",
             "flagged" if info["possible_overfitting"] else "no"]
        )
    return table(
        ["Condition", "Best RL", "Train ratio", "Dev ratio", "Test ratio", "Possible overfitting"],
        rows,
    )


def seeds_table() -> str:
    rows = []
    for c in ALL:
        record = read(f"results/posthoc/per_seed/{c.id}.json")
        info = record["algorithms"][record["best_rl_algorithm"]]
        one = info["best_seed_by_dev"]
        rows.append(
            [label(c), ms(record["strongest_baseline"]["mean_latency_s"]),
             record["best_rl_algorithm"].upper(),
             " / ".join(ms(s) for s in info["mean_latency_s_by_seed"]),
             ms(info["mean_over_seeds_s"]), ms(info["median_over_seeds_s"]),
             f"{info['failed_seeds']} of {len(SEEDS)}",
             f"{latency_interval(one)}, {verdict(one)}"]
        )
    return table(
        ["Condition", "Rule (ms)", "Best RL", "Test mean latency per seed (ms)", "Mean", "Median",
         "Failed seeds", "Best seed by dev vs rule [95% CI]"],
        rows,
    )


def stress() -> str:
    rows = []
    for condition in PRIMARY:
        path = ROOT / f"results/posthoc/stress/{condition.id}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        info = record["algorithms"][record["best_rl_algorithm"]]
        registered = read(f"results/test/{record['condition']}.json")["test_split"]["load"]
        rows.append(
            [label(condition), f"{registered:.2f}", f"{record['realised_load']:.2f}",
             ms(record["strongest_baseline"]["mean_latency_s"]),
             " / ".join(ms(s) for s in info["mean_latency_s_by_seed"]),
             f"{info['failed_seeds']} of {len(SEEDS)}",
             latency_interval(info["five_seed_average"]),
             latency_interval(info["best_seed_by_dev"])]
        )
    return table(
        ["Condition", "Registered test load", "Stress-test load", "Rule (ms)",
         "RL per seed (ms)", "Failed seeds", "5-seed average vs rule [95% CI]",
         "Best seed by dev vs rule [95% CI]"],
        rows,
    )


def cost() -> str:
    record = read("results/posthoc/cost.json")
    conditions = [c.id for c in ALL]
    search = sum(record["per_condition"][c]["baseline_search_seconds"] for c in conditions)
    training = sum(record["per_condition"][c]["rl_training_seconds_10_runs"] for c in conditions)
    tuning = record["rl_tuning_seconds_8_runs"]
    rows = [
        [f"Rule grid search on dev, {len(conditions)} conditions", f"{search:,.0f}"],
        ["RL hyper-parameter tuning, 8 runs", f"{tuning:,.0f}"],
        [f"RL training, {len(conditions)} conditions x 2 algorithms x {len(SEEDS)} seeds",
         f"{training:,.0f}"],
        ["RL total", f"{tuning + training:,.0f}"],
        ["Discarded first tuning round (reward before Deviation 1)",
         f"{record['rl_tuning_seconds_discarded_first_round']:,.0f}"],
    ]
    out = table(["Item", "Seconds"], rows)
    out += (
        f"\nRL tuning plus training took {(tuning + training) / search:.0f} times as long as "
        f"the rule search ({record['note']}, on one laptop CPU).\n"
    )
    return out


def ablations() -> str:
    rows = []
    for base in ABLATION_BASE:
        record, result = test(base), best(base)
        rows.append(
            ["main experiment", label(base), record["best_rl_algorithm"].upper(),
             ms(record["strongest_baseline"]["metrics"]["mean_latency_s"]),
             " / ".join(ms(m["mean_latency_s"]) for m in result["metrics_by_seed"]),
             latency_interval(result), slo_interval(result), verdict(result)]
        )
    variants = [
        ("slo_penalty_10", "reward + SLO penalty 10", ABLATION_BASE),
        ("slo_penalty_100", "reward + SLO penalty 100", ABLATION_BASE),
        ("no_arrival_rate", "state without arrival rate", ABLATION_BASE),
        ("fixed_length_128", "fixed sequence length 128 (rules re-tuned)", ABLATION_LEN128),
        ("tight_slo", "main policies scored under the tight SLO", ABLATION_BASE),
    ]
    for variant, name, conditions in variants:
        for c in conditions:
            record = read(f"results/ablations/test/{variant}/{c.id}.json")
            rows.append(
                [name, label(c), record["algorithm"].upper(),
                 ms(record["strongest_baseline"]["mean_latency_s"]),
                 " / ".join(ms(s) for s in record["mean_latency_s_by_seed"]),
                 latency_interval(record), slo_interval(record), verdict(record)]
            )
    return table(
        ["Variant", "Condition", "Algorithm", "Rule (ms)", "RL per seed (ms)",
         "Mean latency, RL vs rule [95% CI]", "SLO misses, RL vs rule [95% CI]", "Verdict"],
        rows,
    )


def limitations() -> str:
    hours = {t: sum(s.duration_s for s in _raw_splits(t).values()) / 3600 for t in ("conv", "code")}
    conv = {c.load: test(c)["test_split"]["load"] for c in PRIMARY if c.scenario_name == "single"}
    code = {
        c.load: test(c)["test_split"]
        for c in SECONDARY
        if c.scenario_name == "single" and c.trace == "code"
    }
    loads = ", ".join(f"{realised:.2f} at nominal {nominal}" for nominal, realised in conv.items())
    code_loads = ", ".join(f"{s['load']:.2f} at nominal {n}" for n, s in code.items())
    code_requests = next(iter(code.values()))["requests"]
    return (
        f"- **One hour of one day.** The traces cover {hours['conv']:.2f} h (conversation) and "
        f"{hours['code']:.2f} h (code) of a single day. Nothing here speaks to daily cycles, "
        f"other days or other services.\n"
        f"- **The test split is light.** Its arrival rate is below the train split's, and the "
        f"load scaling is fixed on train, so the realised test load on the conversation trace "
        f"is {loads}. The registered test never puts the system into overload; the stress test "
        f"in the exploratory section is the only evidence at realised loads of 0.7 and 0.9.\n"
        f"- **The code trace's test split has {code_requests} requests** (realised load "
        f"{code_loads}). Its P99 rests on a handful of requests and its intervals are wide.\n"
        f"- **Processing-time model.** Parameters are fitted to NVIDIA's published BERT "
        f"measurements on a T4 from 2020 (TensorFlow 20.06 container), batch sizes 1 to "
        f"{MEASURED_MAX_BATCH}. Batch sizes above {MEASURED_MAX_BATCH} (up to {MAX_BATCH}) are "
        f"a linear extrapolation that has not been checked against a real server.\n"
        f"- **What the simulator leaves out.** No GPU memory limit, no step-like slowdowns at "
        f"large batches, no random variation of processing time, one model instance per "
        f"stage, and an illustrative pipeline rather than a measured one.\n"
        f"- **Narrow RL coverage.** Only DQN and PPO from Stable-Baselines3, "
        f"{TOTAL_STEPS:,} decisions per run, hyper-parameters tuned on one condition, one "
        f"observation and action design. A different algorithm, a larger budget or a better "
        f"state design might do better; this study cannot rule that out.\n"
        f"- **Dev was used more than planned.** Two tuning rounds and a set of diagnostic "
        f"runs looked at dev (pre-registration, Deviation 1).\n"
    )


RENDERERS: dict[str, Callable[[], str]] = {
    "report-conclusion": conclusion,
    "report-setup": setup,
    "report-primary-comparison": primary_comparison,
    "report-primary-table": lambda: results_table(PRIMARY),
    "report-secondary-table": lambda: results_table(SECONDARY),
    "report-both-algorithms": both_algorithms,
    "report-baselines": baselines,
    "report-sufficiency": sufficiency,
    "report-memorisation": memorisation,
    "report-ablations": ablations,
    "report-posthoc-seeds": seeds_table,
    "report-posthoc-stress": stress,
    "report-posthoc-cost": cost,
    "report-limitations": limitations,
}
