# Adaptive Batching for ML Inference Serving

**Research question:** under real arrival patterns, when does a learned batching
policy beat a well-tuned heuristic?

An inference server groups requests into batches. Larger batches use the
accelerator more efficiently but make every request in the batch wait longer.
This project compares reinforcement-learning policies (DQN and PPO, from
Stable-Baselines3) against tuned rule-based policies for choosing when to
dispatch a batch and how large it should be.

> **Scope.** Arrival times are replayed from a production LLM-service trace,
> but the processing-time model describes whole-batch inference of an
> encoder-type service (embedding, classification, re-ranking). It does not
> represent the continuous, token-level batching used to serve generative LLMs.

## Result

<!-- BEGIN GENERATED: report-conclusion -->
**Single stage.** With one server the learned policy never beat the tuned rule: in 8 single-stage conditions the pre-registered verdict was "no difference" 6 times and "RL worse" 2 times, and on the conversation trace the best learned policy reproduced the rule "never wait" almost exactly (largest difference in mean latency 0.0%).

**Pipeline.** In the three-stage pipeline the pre-registered primary comparison (conv, load 0.7, mean latency, PPO against the strongest rule) found the learned policy **worse**: +399.2% [+0.5, +947.8] on the test split. The cause is unstable training rather than a uniformly bad policy: 2 of 5 seeds failed (test mean latency above 2x the rule's), while the other 3 were within 0.8% of the rule. Over all 8 pipeline conditions the verdict was "RL worse" 7 times and "no difference" once. **In none of the 16 conditions was the learned policy better than the tuned rule.**

**Scope and data limits.** Arrival times come from about 1 hour of one day of an LLM-service trace; the test split is lighter than its nominal load (realised load 0.63 at nominal 0.9 on the conversation trace) and the code trace's test split has only 751 requests; the processing-time model describes whole-batch inference of an encoder model and is a linear extrapolation above batch size 8. The result is for DQN and PPO with a budget of 300,000 decisions per run.
<!-- END GENERATED: report-conclusion -->

Full results, the pre-registered protocol and its deviations, exploratory
analyses and limitations: **[REPORT.md](REPORT.md)**.

## Simulator sanity check: fixed batch sizes

The simulator is discrete-event: a server is occupied while it processes a
batch, and a request's latency is its queueing time plus its processing time.
Batch processing time is `fixed + per_token x batch_size x longest_sequence`,
with both parameters fitted to published BERT-base measurements on an NVIDIA
T4 (see [docs/decisions.md](docs/decisions.md), D7). The published
measurements cover batch sizes 1-8; **batch sizes 16 and 32 are a linear
extrapolation**.

Fixed-batch-size policies on the **train** split (simulated), for two ways
of using a fixed size b: "greedy cap" dispatches immediately with at most b
requests; "full batch" waits until b requests are queued (at most one
second; this value is not tuned). The SLO is twice the processing time of the
largest batch, summed over stages (rule from Nexus, SOSP 2019, Section 4.1).

**One server (BERT-base):**

<!-- BEGIN GENERATED: fixed-batch-sweep-single -->
Capacity 116.5 requests/s; SLO 550 ms. Each cell: batch size, mean latency, share of requests over the SLO.

| Trace | Nominal load | Load in busiest minute | No batching | Largest greedy cap | Best greedy cap | Best full-batch size |
|---|---|---|---|---|---|---|
| conv | 0.3 | 0.43 | b=1: 28 ms, 0.0% | b=32: 24 ms, 0.0% | b=4: 24 ms, 0.0% | b=1: 28 ms, 0.0% |
| conv | 0.5 | 0.72 | b=1: 2026 ms, 40.8% | b=32: 37 ms, 0.0% | b=8: 37 ms, 0.0% | b=2: 69 ms, 0.0% |
| conv | 0.7 | 1.01 | b=1: 13702 ms, 94.1% | b=32: 89 ms, 0.1% | b=32: 89 ms, 0.1% | b=8: 201 ms, 5.2% |
| conv | 0.9 | 1.29 | b=1: 31352 ms, 98.4% | b=32: 1101 ms, 34.7% | b=32: 1101 ms, 34.7% | b=32: 1367 ms, 52.2% |
| code | 0.3 | 1.00 | b=1: 1664 ms, 62.1% | b=32: 463 ms, 30.8% | b=32: 463 ms, 30.8% | b=8: 580 ms, 36.9% |
| code | 0.5 | 1.67 | b=1: 4235 ms, 94.6% | b=32: 1020 ms, 56.4% | b=32: 1020 ms, 56.4% | b=16: 1109 ms, 60.4% |
| code | 0.7 | 2.33 | b=1: 12648 ms, 96.9% | b=32: 1557 ms, 77.0% | b=32: 1557 ms, 77.0% | b=16: 1643 ms, 78.8% |
| code | 0.9 | 3.00 | b=1: 20060 ms, 98.1% | b=32: 2855 ms, 94.2% | b=32: 2855 ms, 94.2% | b=32: 2947 ms, 95.6% |
<!-- END GENERATED: fixed-batch-sweep-single -->

![Single stage: mean latency against batch size](results/fixed_batch_sweep/latency_vs_batch_single.png)

**Three-stage pipeline (BERT-base, BERT-large, BERT-base; every stage uses
the same policy).** This pipeline is an illustrative configuration assembled
from two published per-model profiles, not a measurement of a real system.

<!-- BEGIN GENERATED: fixed-batch-sweep-pipeline -->
Capacity 40.6 requests/s; SLO 2674 ms. Each cell: batch size, mean latency, share of requests over the SLO.

| Trace | Nominal load | Load in busiest minute | No batching | Largest greedy cap | Best greedy cap | Best full-batch size |
|---|---|---|---|---|---|---|
| conv | 0.3 | 0.43 | b=1: 94 ms, 0.0% | b=32: 92 ms, 0.0% | b=2: 92 ms, 0.0% | b=1: 94 ms, 0.0% |
| conv | 0.5 | 0.72 | b=1: 1735 ms, 25.4% | b=32: 132 ms, 0.0% | b=4: 131 ms, 0.0% | b=2: 236 ms, 0.0% |
| conv | 0.7 | 1.01 | b=1: 16547 ms, 49.5% | b=32: 304 ms, 0.0% | b=16: 300 ms, 0.0% | b=8: 702 ms, 0.0% |
| conv | 0.9 | 1.29 | b=1: 58983 ms, 94.7% | b=32: 3295 ms, 29.1% | b=32: 3295 ms, 29.1% | b=16: 4514 ms, 33.7% |
| code | 0.3 | 1.00 | b=1: 3569 ms, 41.8% | b=32: 1486 ms, 15.9% | b=16: 1403 ms, 14.5% | b=8: 1760 ms, 20.7% |
| code | 0.5 | 1.67 | b=1: 7088 ms, 74.3% | b=32: 3149 ms, 42.0% | b=16: 3035 ms, 39.2% | b=8: 3353 ms, 44.8% |
| code | 0.7 | 2.33 | b=1: 24078 ms, 95.0% | b=32: 4725 ms, 66.5% | b=16: 4628 ms, 63.7% | b=16: 4943 ms, 68.4% |
| code | 0.9 | 3.00 | b=1: 41231 ms, 97.4% | b=32: 8480 ms, 92.3% | b=32: 8480 ms, 92.3% | b=32: 9002 ms, 94.1% |
<!-- END GENERATED: fixed-batch-sweep-pipeline -->

![Pipeline: mean latency against batch size](results/fixed_batch_sweep/latency_vs_batch_pipeline.png)

What this shows: no batching is far from optimal except at the lowest
load; waiting for a full batch gives a U-shaped curve whose best size grows
with load; with one server, a policy that never waits is not hurt by a large
cap, while in the pipeline a cap below the largest is slightly better in
several cells. Nominal load is computed from the mean arrival rate, so bursts push the
busiest minute well above it. Reproduce with `make sweep`.

## What was wrong with the original result

The original multi-stage experiment reported that PPO clearly beat the
strongest heuristic on average job completion time (JCT). That conclusion is
not valid.

**How it was found.** The only static baseline in the original comparison was
batch size 8. Evaluating every fixed batch size with the original evaluation
protocol shows that simply always using the largest batch beats the reported
PPO result:

<!-- BEGIN GENERATED: legacy-fixed-batch -->
| Policy | Avg JCT (steps) | P99 JCT (steps) | Source |
|---|---|---|---|
| Fixed batch 1 | 439.32 | 868.63 | re-run |
| Fixed batch 2 | 379.41 | 749.34 | re-run |
| Fixed batch 4 | 260.32 | 516.53 | re-run |
| Fixed batch 8 | 34.30 | 63.70 | re-run |
| Fixed batch 16 | 2.38 | 4.30 | re-run |
| Fixed batch 32 | 2.04 | 3.00 | re-run |
| Fixed batch 64 | 2.00 | 2.00 | re-run |
| Threshold | 4.81 | 7.00 | re-run |
| Downstream-Aware | 5.32 | 7.90 | re-run |
| PPO (original run, not retrained) | 2.30 | 3.10 | stored |
<!-- END GENERATED: legacy-fixed-batch -->

**Why.** In the original multi-stage environment, the processing time of a
batch was computed and then never used: a batch of any size finished within
the same step and never occupied the server. With batching free, the largest
batch is always optimal, and "PPO beats the heuristics" only meant that the
hand-written threshold rule picked small batches when queues were short.

Other problems in the original code, all addressed by the rework:

- Baselines were weak and untuned (a single fixed batch size, hand-picked thresholds).
- The original README said the simulator's parameters came from the Azure LLM
  inference trace. They did not: the trace files were only read by an
  exploratory script, and arrival rates and latency parameters were hand-written.
- In the single-stage environment a step always represented a fixed time
  slice, even when the batch it dispatched took longer than that slice.
- The multi-stage experiment used one training seed.
- There were no tests and no CI.

The original code, scripts and results are kept unmodified in
[legacy/](legacy/) (see [legacy/README.md](legacy/README.md)). Reproduce the
table above with `make legacy-repro`.

## Rework plan

| Phase | Content | State |
|---|---|---|
| 0 | Archive old code, packaging, lint, tests, CI | done |
| 1 | Discrete-event simulator replaying the Azure trace timestamps, with train / dev / test splits | done |
| 2 | Tuned heuristic baselines vs. DQN / PPO, multiple seeds, paired bootstrap confidence intervals, ablations | done |
| 3 | Real batching inference service (FastAPI, CPU model), trace replay load test, simulator-vs-real comparison | not started |

Design decisions and their reasons are recorded in
[docs/decisions.md](docs/decisions.md).

## Development

Requires Python 3.10+.

```bash
make setup   # create .venv and install with dev extras
make check   # ruff + pytest
make sweep      # fixed-batch-size sweep on the train split + figure
make baselines  # rule grid search on dev
make train      # RL tuning, training and dev selection (the longest step)
make docs       # regenerate generated blocks in README.md and REPORT.md
```

Every number in this README and in REPORT.md is written by `scripts/render_docs.py` from files
under `results/`; none is typed by hand.

## Data

`data/` contains two files from the
[Azure LLM Inference Trace 2023](https://github.com/Azure/AzurePublicDataset)
(code and conversation workloads): request timestamps with context and
generated token counts. Each covers roughly one hour of a single day, which
limits how far any conclusion drawn from them can be generalized.

## Attribution

The RL algorithms are used as implemented in
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). This
project's own work is the simulator, the state and reward design, the
baselines, and the experimental method.

## Team (original course project)

- Yumeng Ren — environment design and simulation
- Haoran Jiang — PPO and SAC agent implementation
- Brian Lam — DQN baseline, heuristic baselines, evaluation and reporting

The rework on top of the course project is by Haoran Jiang.
