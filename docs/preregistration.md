# Pre-registration of the Phase 2 comparison

Status: **draft, awaiting the owner's approval.** Once approved, the commit
that marks it approved is the reference point: no policy is evaluated on a
test split before that commit, and nothing above the "Deviations" section is
edited afterwards. Changes after approval are appended to "Deviations" with
a date and a reason.

## 1. Question

Under real arrival patterns, when does a learned batching policy beat a
well-tuned rule-based policy?

## 2. What has been looked at so far

- Train splits: fixed-batch sweeps (`results/fixed_batch_sweep/train.json`)
  and a 20,000-step timing run of untrained-to-barely-trained PPO and DQN
  models, evaluated on the conversation train split only.
- Dev and test splits: request counts, arrival rates and the number of empty
  one-minute slices were inspected when the splits were built (Phase 1). No
  policy of any kind has been simulated on dev or test.

## 3. Prior expectation

- **Single stage:** a learned policy is not expected to beat the strongest
  rule by a detectable margin. On train, "never wait, largest cap" was the
  best or tied-best fixed policy at every load, and batching can raise
  throughput by less than a factor of two (decisions D7, D10).
- **Three-stage pipeline:** this is the main test. Rules that act on one
  stage at a time may interact badly, which leaves room for a policy that
  sees all stages. No direction is predicted.

## 4. Data, scenarios, loads

- Traces: Azure LLM inference traces `conv` and `code`, split by time into
  train / dev / test (60 / 20 / 20 %).
- Scenarios: `single` (one BERT-base server) and `pipeline` (illustrative
  BERT-base, BERT-large, BERT-base); decision D14.
- Nominal loads 0.3, 0.5, 0.7, 0.9. The time-scaling factor is computed on
  train and reused for dev and test (decision D9). Realised load and the
  load of the busiest minute are reported for every split.
- A **condition** is one (scenario, trace, load) triple: 16 in total.

**Primary conditions (6):** `conv`, both scenarios, loads 0.5, 0.7, 0.9.

**Secondary conditions (10):** `conv` at load 0.3 (little contention; both
scenarios), and all eight `code` conditions. The `code` trace is so bursty
that its busiest minute is at or above capacity at every nominal load; on
train every fixed policy had more than 90% SLO violations at load 0.9. It is
therefore not expected to separate policies at high load, and its test split
has only 751 requests. `code` results are reported in full but cannot
upgrade or downgrade the answer to the question.

## 5. Metrics

- **Mean latency** (end-to-end: queueing plus processing).
- **SLO violation rate** under the primary SLO (550 ms single stage,
  2674 ms pipeline; decision D13). A request that never completes counts as
  a violation.

Also reported, without decision rules: P50 / P95 / P99 latency, throughput,
GPU busy time, mean batch size, violation rate under the tight SLO (160 ms /
726 ms). P99 on the `code` test split rests on about 8 requests and is
marked unreliable.

## 6. Rule-based baselines and how the strongest one is chosen

Families (every parameter grid is fixed here):

1. `GreedyBatch(cap)`: cap in {1, 2, 4, 8, 16, 32}.
2. `WaitForBatch(b, max_wait)`: b in {1, 2, 4, 8, 16, 32}.
3. `ThresholdBatch(k, max_wait, max_batch=32)`:
   k in {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}. The threshold form is motivated
   by Deb and Serfozo, "Optimal control of batch service queues", Advances
   in Applied Probability 5(2), 1973, whose abstract states that a
   control-limit rule (serve min(x, Q) when at least M wait) is optimal for
   their model. The paper's exact assumptions were not checked against the
   full text, and no optimality is claimed for this project's setting
   (trace arrivals, batch-size-dependent processing time, several stages).
4. `DownstreamAwareBatch(k, max_wait, max_batch=32, hold_at)` (pipeline
   only): k as above, hold_at in {1, 2, 4, 8, 16, 32}.

`max_wait` grid for families 2-4: {0.5, 1, 2, 4, 8, 16} times the stage's
single-request processing time.

Search, on the **dev** split of each condition, minimising mean latency
(ties: lower SLO violation rate, then the simpler family in the order
above):

- Single stage: exhaustive over each grid.
- Pipeline: first exhaustive with the same parameters at all stages; then
  coordinate descent over stages (bottleneck stage first, two passes), where
  each stage may take any family-and-parameter combination from the grids
  while the others are held fixed.

The **strongest baseline** of a condition is the single configuration with
the lowest dev mean latency. It is frozen before any test evaluation and is
the comparator for both metrics. The best configuration of each family is
also reported.

## 7. Learned policies

Algorithms: DQN and PPO as implemented in Stable-Baselines3. This project's
contribution is the simulator, the observation, action and reward design and
the experimental method, not the algorithms.

- **Actions:** wait, or dispatch at most 1, 2, 4, 8, 16 or 32 queued
  requests. Waiting ends at the next arrival to that queue or after half a
  single-request processing time. If the oldest queued request has already
  waited a whole SLO, "wait" is overridden by a full dispatch.
- **Observation:** which stage is deciding; its queue length; how long its
  oldest request has waited; queue length and remaining busy time of every
  stage; recent external arrival rate over a short and a long window.
- **Reward:** minus the request-seconds accumulated since the previous
  decision. The undiscounted return of an episode is exactly minus the sum
  of its requests' latencies.
- **One policy per condition**, shared by all stages of the pipeline.
- **Episodes:** each episode is a window of 512 consecutive requests whose
  first request is drawn uniformly at random from the (time-scaled) train
  split. The system starts empty; the episode ends when all 512 requests
  have completed. Known bias: backlog carried over from before the window is
  not represented.
- **Budget:** 300,000 decisions per run; a checkpoint every 30,000.
- **Seeds:** 1-5 for every (condition, algorithm).
- **Fixed settings:** discount 0.999, two hidden layers of 64 units, CPU.
  PPO: n_steps 2048, batch size 256, 10 epochs. DQN: buffer 100,000,
  learning starts at 5,000, train every 4 steps, target update every 2,000,
  final exploration rate 0.02.
- **Tuned settings** (dev mean latency, on the primary comparison's
  condition only, with seed 100, then frozen for every condition):
  PPO learning rate in {3e-4, 1e-3} and entropy coefficient in {0, 0.01};
  DQN learning rate in {1e-4, 5e-4} and exploration fraction in {0.1, 0.3}.
  Limitation: settings tuned on one condition are reused on the others.
- **Checkpoint selection:** for each run, the checkpoint with the lowest dev
  mean latency.
- **"Best RL" of a condition:** the algorithm whose five selected
  checkpoints have the lower average dev mean latency. Its value on any
  split is the average over its five seeds. No seed is dropped.

Nothing about a learned policy (algorithm, settings, checkpoint, seed) is
chosen using a test split.

## 8. Evaluation and statistics

- Each frozen policy is simulated once over the whole test split of its
  condition (the simulator is deterministic).
- The test split is cut into 20 consecutive blocks of equal request count;
  each request belongs to the block it arrived in (decision D11).
- **Paired block bootstrap**, 10,000 resamples, fixed random seed: resample
  the 20 blocks with replacement, using the same blocks for both policies;
  for the learned policy also resample its five seeds with replacement;
  pool the requests of the resampled blocks; compute the metric.
- Effect sizes (learned minus baseline):
  mean latency as a relative difference, (L_RL - L_base) / L_base;
  SLO violation rate as a difference in percentage points.
- 95% percentile confidence intervals.

**Classification of each metric in each condition:**

| Verdict | Condition on the 95% CI |
|---|---|
| RL better | entirely below 0 |
| RL worse | entirely above 0 |
| Equivalent | contains 0 and lies inside +/-5% (latency) or +/-1 point (SLO) |
| Inconclusive | contains 0 and is wider than that |

**Verdict for a condition:**

- *RL better*: mean latency "RL better" and SLO violation rate not "RL worse".
- *RL worse*: mean latency "RL worse"; or SLO "RL worse" while latency is
  not "RL better".
- *Trade-off, not a win*: latency "RL better" and SLO "RL worse".
- *No difference*: everything else, stating whether it is "equivalent" or
  "inconclusive".

## 9. Primary comparison and multiplicity

**One primary comparison:** pipeline, `conv`, load 0.7, mean latency, best
RL against the strongest baseline, on the test split.

All other comparisons (the other five primary conditions, the SLO metric,
the ten secondary conditions, per-algorithm results, ablations) are
**secondary or exploratory**. Their intervals are reported uncorrected and
labelled as such; with this many comparisons some intervals will exclude
zero by chance, so none of them alone supports a claim.

**How the question will be answered.**

- "RL beats tuned rules" is claimed only if the primary comparison is "RL
  better". The answer is then qualified by how many of the six primary
  conditions agree.
- If the primary comparison is "equivalent" or "RL worse", the answer is
  that the learned policy does not beat tuned rules in this setting.
- If it is "inconclusive", that is the answer: the test data cannot tell.
- The single-stage results are reported against the prior expectation in
  Section 3 either way.

## 10. Check that the learned policy has not memorised the train arrivals

For every condition and for the best RL policy, compute the ratio of mean
latency, learned policy over strongest baseline, on the full train split and
on the dev split (both simulated in one continuous run). If the train ratio
is more than 0.10 below the dev ratio, the condition is flagged as
"possible overfitting to the train arrival sequence" in the report. Because
checkpoints are selected on dev, the test ratio is reported next to both.
This is a diagnostic; it does not change the verdicts.

## 11. Ablations (exploratory)

Run at `conv`, load 0.7, both scenarios, with the best RL algorithm of that
condition, five seeds each:

- **Load** and **burstiness** are covered by the main grid (four loads; the
  near-Poisson `conv` trace against the very bursty `code` trace).
- **Reward weights:** add a penalty per request that misses the SLO, of 10
  or 100 time units (one unit = single-request processing time of the
  pipeline); the main runs use 0.
- **State without the arrival-rate features.**
- **Fixed sequence length 128** (stronger batching effect); baselines are
  re-tuned on dev for this variant.
- **Tight SLO:** the main policies re-scored under the tight SLO; no
  retraining.

## 12. Procedure

1. This document is committed and approved.
2. Baseline search and all training and model selection run on train and
   dev only. Their outputs are committed.
3. A single script evaluates the frozen policies on test once and writes the
   results with the commit hash it ran from.
4. `REPORT.md` is generated from those files.

If a run fails or a policy turns out badly on dev, it is still evaluated on
test and reported. No setting is changed after test results exist.

## Deviations

None yet.
