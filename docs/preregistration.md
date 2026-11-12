# Pre-registration of the Phase 2 comparison

Status: **approved by the repository owner on 2026-11-11.** The commit that
introduced this line is the reference point: no policy was evaluated on a
test split before it, and nothing above the "Deviations" section is edited
after it. Later changes are appended to "Deviations" with a date and a
reason.

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
- **Reward (main experiments): latency only.** Let N(u) be the number of
  requests in the system at time u (arrived, not yet completed), t_k the
  time of the k-th decision, and tau the unbatched processing time of one
  request through the whole pipeline (the sum over stages of s_i(1) at the
  reference length). The reward for decision k is

      r_k = -(1 / tau) * integral from t_k to t_{k+1} of N(u) du

  The integral of N over an episode equals the sum of the latencies of its
  requests, so the undiscounted return is -(1 / tau) * (sum of latencies).
  This is the same quantity, mean latency, that selects the baselines on
  dev. The main experiments contain no SLO term and no other shaping.
- **Reward (ablation only):** r'_k = r_k - lambda * V_k, where V_k is the
  number of requests that completed in (t_k, t_{k+1}] with latency above the
  primary SLO, and lambda is 10 or 100 (Section 11).
- **One policy per condition**, shared by all stages of the pipeline.
- **Episodes:** each episode plays 512 consecutive requests whose first
  request is drawn uniformly at random from the (time-scaled) train split,
  and ends when every request of the episode has completed.
- **Random backlog at the start of an episode.** Evaluation runs a whole
  split continuously, and under overload a backlog builds up and drains
  over a long time; 512-request episodes that start from an empty system
  would rarely show such states. Two fixes were considered: (a) much longer
  episodes that cover the build-up and drain, or (b) starting episodes from
  a random backlog. **(b) is chosen.** With probability 0.5 an episode
  starts with B extra requests already queued at the first stage at time
  zero, B drawn log-uniformly from 1 to B_max. The extra requests are the B
  requests that precede the window in the trace, with their own token
  counts. B_max is the largest number of requests simultaneously in the
  system when `GreedyBatch(32)` is simulated over the condition's train
  split, so it is set from train data only. Reasons: on the train split the
  overloaded stretch of the conversation trace spans thousands of requests,
  so an episode long enough to contain it would leave only a handful of
  distinct episodes, which gives few independent starts and makes
  memorising the arrival sequence easier; random backlogs expose the policy
  to the whole range of backlog sizes while keeping many distinct episodes.
  Known limitations: a backlog appears only at the first stage (backlogs at
  later stages arise only as the policy pushes it through), the backlogged
  requests start with zero waiting time, and queue-length features are
  clipped at 128 requests, so larger backlogs look alike to the policy.
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
3. Test evaluation is run by one script, once per group of conditions, in
   three groups, each only after every policy of that group is frozen:
   the six primary conditions; the ten secondary conditions; the ablations.
   The script writes the commit hash it ran from and refuses to overwrite an
   existing test result. Hyper-parameters, grids and every other shared
   setting are fixed before the first group's test results exist and are
   not changed afterwards.
4. `REPORT.md` is generated from those files.

If a run fails or a policy turns out badly on dev, it is still evaluated on
test and reported. No setting is changed after test results exist.

## Deviations

### Deviation 1 (2026-11-11): the reward is expressed in a different unit

**Made before any test evaluation.** At the time of this entry no policy had
been simulated on any test split.

**What changes.** The main-experiment reward becomes

    r_k = -(1 / U) * integral from t_k to t_{k+1} of N(u) du,
    U = 32 * (sum over stages of s_i(32) at the reference length)

instead of dividing by tau (Section 7). U is a positive constant per
scenario, so this is the registered reward multiplied by the constant
tau / U: it still contains latency only, the undiscounted return is still
proportional to minus the sum of latencies, and the ranking of policies by
undiscounted return is unchanged. The ablation reward keeps its registered
meaning in the same unit: r'_k = -(1 / U) * (integral + lambda * tau * V_k).

**Why.** With the registered unit the tuning stage (Section 7, all eight
configurations, 300,000 decisions, seed 100, pipeline / conv / load 0.7)
produced policies that choose "wait" almost always: dev mean latency
6.6-8.9 s against 0.23 s for the strongest rule and 0.24 s for the constant
action "dispatch at most 32". Per-step rewards reached about -500 and
episode returns were between about -3,000 and -63,000, a scale the value
networks did not fit. Reporting those policies would compare the rules with
a learner that was never given a usable training signal.

**Evidence** (train and dev only; `results/rl/diagnostics/`,
`results/rl/tuning_registered_reward/`). Selected PPO checkpoint: 31,987 of
31,987 dev decisions were "wait". With the reward multiplied by tau / U and
nothing else changed, 150,000 decisions, seed 100: DQN reached 235 ms and
PPO 551 ms on the pipeline condition (strongest rule 229.7 ms); PPO reached
71 ms on single / conv / load 0.7 (strongest rule 70.8 ms).

**Consequences.**
- The tuning stage is re-run with the same grid and the same seed under the
  new unit; the earlier selection is discarded because it was made among
  collapsed runs.
- While diagnosing, three other changes were tried on dev: discount 1.0,
  `VecNormalize` reward normalisation, and a different PPO learning rate
  with scaling. **None of them is adopted.** Everything else in Section 7
  stays as registered.
- Dev has now been looked at more than the protocol planned (eight collapsed
  tuning runs and eight diagnostic runs on two conditions). Test has not.

### Deviation 2 (2026-11-11): training-sufficiency check with one fixed escalation

**Made before any test evaluation and before the tuning stage is re-run.**
Requested by the repository owner.

**Check.** After re-tuning, on the primary comparison's condition (pipeline,
conv, load 0.7), take the algorithm whose selected tuning configuration has
the lower dev mean latency (the one that would count as "best RL" there).
If that dev mean latency exceeds 1.5 times the dev mean latency of the
untuned rule "never wait, cap 32 at all three stages", training is judged
**not sufficient within the registered budget**. The other algorithm's
value is reported next to it but does not trigger anything.

**Escalation, allowed once.** If the check fails, the training budget goes
from 300,000 to 1,000,000 decisions for every run of the study. Nothing
else changes (same grids, seeds, checkpoint interval, selection rules). The
tuning stage is repeated at the new budget and the check is evaluated again.
There is no second escalation.

**If the check still fails after the escalation,** all runs are carried out
and reported as they are, and the report states the outcome as "within this
training budget the learned policies did not reach a competitive policy",
kept separate from the statement "learned policies are worse than rules".

**Scope.** The check uses the dev split only.

**Why.** Deviation 1 showed that a learner can fail for reasons that have
nothing to do with the question. A threshold and a single remedy fixed in
advance keep that from turning into open-ended tuning on dev.

### Deviation 3 (2026-11-11): exploratory analyses added after the primary test results

**Made after the test results of the six primary conditions were seen**
(`results/test/`, commit `50caa9e`), and before any secondary or ablation
test evaluation. Requested by the repository owner. Everything in this
entry is **post-hoc and exploratory**. None of it changes a registered
verdict, and the registered tables are reported unchanged next to it.

1. **Per-seed reporting.** For every condition, next to the registered
   five-seed average: the median over seeds, each seed's test mean latency,
   and the number of failed seeds. A seed counts as **failed** if its test
   mean latency is more than twice the strongest baseline's test mean
   latency. (This definition was written after the per-seed primary results
   had been seen.)
2. **Best seed by dev.** For the best RL algorithm of each condition, the
   seed with the lowest dev mean latency is compared with the strongest
   baseline on test with the same paired block bootstrap (blocks only; there
   is a single seed).
3. **Overload stress test.** The registered test split is lighter than its
   nominal load because the trace's arrival rate falls in its last fifth
   (realised load 0.49 and 0.63 at nominal 0.7 and 0.9). For the four
   primary conditions at nominal load 0.7 and 0.9, the test split is
   re-scaled by its own mean arrival rate so that its realised load equals
   the nominal load. The policies are the frozen ones of the same condition:
   the strongest baseline with its dev-selected parameters and all trained
   seeds with their dev-selected checkpoints. Nothing is retrained,
   re-selected or re-tuned. Same metrics, blocks and bootstrap as Section 8.
4. **Cost.** Wall-clock seconds of the baseline grid search per condition
   against the summed training time of hyper-parameter tuning plus the five
   seeds of both algorithms, all measured on the same machine while up to six
   processes ran in parallel.

### Deviation 4 (2026-11-12): post-hoc CPU measurement of the processing-time model

**Made after every test evaluation of the study** (primary, secondary and
ablations). Requested by the repository owner. Exploratory; it changes no
registered result.

`bert-base-uncased` is timed on the local CPU at batch sizes 1, 2, 4, 8, 16,
32 and sequence lengths 128 and 384 (fixed thread count, warm-up, median of
repeated shuffled rounds). The two-parameter model is fitted to all points,
and separately to batch sizes up to 8 only, whose predictions for 16 and 32
are compared with the measurements. This tests the shape of the formula and
its extrapolation above batch size 8; it does not measure the T4 parameters
the simulator uses. Scripts: `scripts/measure_cpu_latency.py`,
`scripts/fit_cpu_latency.py`; results: `results/real_latency/`.

