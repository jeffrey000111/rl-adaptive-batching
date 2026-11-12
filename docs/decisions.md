# Design decisions

Each entry records what was decided, why, and what the alternative was.
Entries are append-only; a changed decision gets a new entry that supersedes
the old one.

## D1. Archive the original code instead of fixing it in place

**Decision.** The original environments, scripts and results live unmodified
under `legacy/`. The rework is a new package under `src/adaptive_batching/`.

**Why.** The original environments advance in fixed time steps and treat a
batch as instantaneous. Fixing that means changing the time model itself
(fixed steps to discrete events), which invalidates every stored result.
Keeping the old code runnable lets anyone reproduce the flaw
(`make legacy-repro`) and compare it with the new simulator.

**Alternative.** Patch `stage_latency` into the old step loop. Rejected: a
fixed-step loop cannot represent a batch that takes longer than one step
without ad-hoc bookkeeping, and the old results would silently stop matching
the code.

## D2. Drop SAC

**Decision.** The rework compares DQN and PPO only.

**Why.** The action space is discrete (dispatch a batch of size b, or wait).
Stable-Baselines3 ships SAC for continuous actions only. The original project
worked around this by rounding a continuous action to a discrete index, which
is not the discrete SAC algorithm: the critic learns over an action range in
which large intervals map to the same decision, and the original results were
unstable across seeds. Implementing discrete SAC from scratch would add an
unverified algorithm implementation to a project whose contribution is the
simulator and the experimental method, not RL algorithms.

**Alternative.** Implement discrete SAC (Christodoulou, 2019). Rejected for
the reason above; it can be revisited if a maintained implementation is used.

## D3. Privacy

**Decision.** Student IDs were removed from the README and the four course
PDFs under `docs/` were removed from the tree. Git history is not rewritten
and nothing is force-pushed. `tests/test_repo_hygiene.py` fails if a tracked
text file contains a nine-digit ID, a personal e-mail address or an absolute
path containing a user name, or if any PDF is tracked.

**Why.** The repository is public. PDFs cannot be scanned by the text checks,
so they are banned outright; a report may only come back as Markdown or after
it has been checked by hand for IDs.

**Known limitation.** The IDs remain in earlier commits. The owner's plan is
to publish the finished project from a fresh repository and make the old one
private, rather than rewrite history.

## D4. No claim that the simulator is fitted to real measurements until it is

**Decision.** Until the processing-time model has been fitted to measurements
of a real service (Phase 3) and the fit error is reported, no file in the
repository may describe the simulator as fitted to, or validated against,
real data. A test enforces this for the specific word the original README
used.

**Why.** The original README made that claim although no environment read the
trace. Arrival timestamps replayed from the trace (Phase 1) are real data;
processing-time parameters are assumptions until measured.

## D5. Every reported number is generated

**Decision.** Numbers in `README.md` and `REPORT.md` appear only inside
`<!-- BEGIN GENERATED: name -->` blocks, which `scripts/render_docs.py` fills
from files under `results/`. A test runs the script in `--check` mode, so CI
fails when a document and the result files disagree.

**Why.** The original report's numbers were copied by hand from script
output, so nothing tied a stated number to the run that produced it.

**Alternative.** A templating engine (Jinja2). Rejected: marker blocks keep
the document readable on GitHub as-is and need no extra dependency.

## D6. Dependencies are added when first used

**Decision.** `pyproject.toml` lists only what the current code imports.
Training (`stable-baselines3`) and serving (`fastapi`) dependencies will be
added as optional extras in the phases that introduce them.

**Why.** The simulator and the baselines should be installable and testable
in CI in seconds, without PyTorch.

## D7. Processing-time model and where its parameters come from

**Decision.** One batch takes

    batch_time = fixed + per_token * batch_size * padded_length

where `padded_length` is the longest request in the batch, and a request's
length is `min(ContextTokens, 384)` from the trace. Parameters for the default
server (`BERT_BASE_T4_FP16`): `fixed = 7.616 ms`, `per_token = 0.02174 ms`.

**Source.** Least-squares fit to the eight published median latencies for
BERT-base, FP16, on one NVIDIA T4 16GB (batch sizes 1, 2, 4, 8; sequence
lengths 128 and 384) in the NVIDIA DeepLearningExamples TensorFlow BERT
README. The transcribed rows, the conditions and the link are in
`data/public_measurements/`. `tests/test_service_time.py` re-runs the fit and
fails if the constants in the code drift from the data. The fit reproduces
every published point to within 8% (worst point 7.6%).

**Why this shape.** The fixed term is paid once per model invocation
whatever the batch size; spreading it over more requests is the entire
benefit of batching. The compute term is proportional to the tokens
processed, and a batch is padded to its longest sequence. A linear batch
latency profile is the standard model in the serving literature (Nexus, SOSP
2019; Symphony, arXiv 2308.07470, Section 2.1; Inoue, arXiv 1912.06322).

**Alternative considered.** A three-parameter fit (`fixed + c*batch +
d*batch*length`) lowers the worst error to 2.9% but yields a negative `c`,
which has no physical reading. Rejected.

**Batching gain, defined.** `gain(b) = throughput(b) / throughput(1)
= b * s(1) / s(b)`. Its limit for large b is `s(1) / (per_token * length)
= 1 + fixed / (per_token * length)`. With the adopted parameters: at length
384 the gain is 1.86 at b=32 and at most 1.91; at length 128 it is 3.44 at
b=32 and at most 3.74. (Fitting length 128 alone gives a larger fixed term
and a limit of 4.68; the joint fit is used because one model must serve
mixed lengths.) Batching can therefore raise throughput by less than a
factor of two for the default workload. This is what the data says and it
bounds how much any batching policy, learned or not, can gain.

**Where the assumptions fail.**
1. Batch sizes above 8 are extrapolated. The action set is 1, 2, 4, 8, 16, 32;
   16 and 32 are marked as extrapolated in every figure. Real servers can
   show memory limits or step-like slowdowns at larger batches. Phase 3 must
   measure up to 32 to test this.
2. Lengths outside 128-384 are extrapolated. Only about a fifth of trace
   requests have at most 384 context tokens, so most requests sit at the cap.
3. The measurements are from a 2020 TensorFlow container without TensorRT.
4. It is not a model of generative LLM serving, which batches continuously
   at token level. Arrival times come from an LLM service; the server model
   is an encoder-type service (embedding, classification, re-ranking).
5. Processing time is deterministic (published P99 is about 5% above P50) and
   there is one model instance per stage.
6. A CPU model (Phase 3) will have a different fixed-to-variable ratio.

**Ablation.** `fixed_length=128` treats every request as 128 tokens, where
batching helps more.

## D8. Discrete-event engine driven from outside

**Decision.** Time advances from event to event (arrival, batch completion,
wake-up timer). A decision point exists whenever a stage's server is idle
and its queue is not empty. The caller answers with `Dispatch(size)` or
`Wait(timeout)`; after `Wait` the same stage is asked again on its next
arrival or at the timeout. The engine exposes `next_decision()` / `apply()`
and never calls a policy itself.

**Why.** (1) A batch that takes 275 ms must make the server unavailable for
275 ms; with events this is one scheduled completion, with fixed steps it
needs bookkeeping and a step-size choice that distorts latency. (2) Inverting
control lets rule-based policies and a Gymnasium environment share one
engine: an RL environment must return to the agent at each decision point.
(3) All events at one timestamp are processed before a decision is requested,
so a policy never sees half of a simultaneous arrival group. (4) When several
stages need a decision at the same instant, downstream stages are asked
first, so an upstream stage sees the queue its neighbour is actually left
with rather than requests that are about to be dispatched.

**Consequences.** The simulation has no randomness; identical inputs give
identical outputs, and variation across training seeds comes only from the
learning algorithm. `Dispatch` with a size outside `1..queue_length` raises
instead of being clamped, so policy bugs surface. A policy that waits
although a request has been queued for `stall_limit_s` raises
`SimulationStalledError` instead of looping forever.

## D9. Offered load and time scaling

**Decision.** `load = mean arrival rate / capacity`, where capacity is the
throughput of the slowest stage running back-to-back batches of the largest
allowed size at the reference length (384). A load level is reached by
multiplying all trace timestamps by one factor. The factor is computed from
the **train** split only and applied unchanged to dev and test.

**Why.** Load 1.0 is then the point beyond which no policy is stable, so
levels are comparable across server models. Computing the factor from train
keeps evaluation data out of the workload definition.

**Known consequences, to be reported with every result.**
- The realised load on dev and test differs from the nominal one; both are
  reported.
- Nominal load is an average. The traces are not stationary, so the busiest
  minute is far above it (see the generated table in the README): the system
  is temporarily overloaded at nominal loads well below 1.
- Compressing time assumes higher load means the same arrival pattern played
  faster; real growth would smooth bursts.

## D10. What "fixed batch size" means, and what the sweep showed

**Decision.** Two fixed-size families are distinguished: `GreedyBatch(b)`
(never wait; at most b per batch) and `WaitForBatch(b, max_wait)` (wait for b
requests or until the oldest has waited `max_wait`).

**Finding on the train split, one server**
(`results/fixed_batch_sweep/train.json`).
Waiting for a full batch gives a U-shaped latency curve whose minimum moves
to larger batches as load grows. Never waiting gives a curve that only
falls and flattens: a larger cap is never clearly worse, because the cap is
only reached when a backlog exists. The original expectation that "always
the largest batch" is clearly bad at high load does **not** hold in this
simulator: at high load the largest batch is the best fixed choice in both
families. What is clearly bad is no batching at high load, and waiting for
the largest batch at low or moderate load. `tests/test_lessons.py` asserts
exactly these, not the original expectation.

**Pipeline.** With three stages and the same policy at every stage, the
largest greedy cap is no longer always the best: a smaller cap is slightly
better in several cells (see the generated README table). The differences
are small and measured on train only; they are a lead, not a result.

**Implication.** `GreedyBatch(32)` is a strong, parameter-free baseline for
mean latency. A learned policy can only beat it by sometimes waiting on
purpose or by choosing a smaller batch than the backlog allows.

## D11. Evaluation blocks (protocol to be fixed before any test run)

**Decision.** Time scaling does not change which requests fall in which
block, only how long the block lasts, so blocks are defined on the unscaled
trace. Because equal-time blocks of the code trace are often empty, the
evaluation split is cut into consecutive blocks of **equal request count**.
A policy is simulated once over the whole split; each request is assigned to
the block it arrived in; metrics are computed on requests pooled over blocks;
uncertainty comes from a paired block bootstrap (the same resampled blocks
for both policies being compared).

**Why.** Pooling keeps tail percentiles from being computed on a few dozen
requests per block. Simulating the split in one run avoids restarting from
an empty system at every block boundary, which would bias latency down.

**Limits.** The code test split has 751 requests; its P99 rests on about 8
requests and will be reported as unreliable.

## D12. Load levels

**Decision.** Nominal loads 0.3, 0.5, 0.7 and 0.9. Every result is reported
next to the load of the busiest minute of the same data.

**Why.** Nominal load is an average over a non-stationary trace. On the
train split the busiest minute of the conversation trace reaches load 1.01
at nominal 0.7, and the code trace is already at 1.00 at nominal 0.3. At 0.9
the system spends much of the time draining backlog and latency is measured
in seconds for every policy, so 0.3 was added to have a level where the
system is not overloaded even at its peak (conversation trace).

## D13. Latency SLO

**Decision.** Primary SLO: twice the processing time of the largest batch
at the reference length, summed over the stages (550 ms for one BERT-base
server). Ablation ("tight") SLO: ten times the processing time of a single
request, summed over the stages (160 ms for one server).

**Source of the rule.** Nexus (Shen et al., SOSP 2019), Section 4.1: "the
batch execution cost for a given task type cannot exceed half of the task's
latency SLO; a task that missed being scheduled with a batch would be
executed as part of the next batch, and thus its latency would be twice the
batch execution cost." The linear batch latency model is Equation (1) in
Section 2.2 of the same paper.

**Why two.** Under the primary SLO every allowed batch size is feasible in
principle, so the SLO does not decide the batch size by itself. Under the
tight SLO the largest batch alone exceeds the SLO, so a policy has to hold
batch sizes down, which is a different trade-off.

**Extension to the pipeline** (sum of per-stage budgets) is this project's
choice, not taken from a source.

## D14. Three-stage pipeline

**Decision.** `PIPELINE_3STAGE` is BERT-base, BERT-large, BERT-base, all T4
FP16, loosely "encode, re-rank, classify". BERT-large parameters
(`fixed = 16.59 ms`, `per_token = 0.06275 ms`) are fitted the same way from
the same source as D7; the fit reproduces the eight published points to
within 9% (worst point 8.4%).

**This is an illustrative configuration, not a measurement of a real
system.** Each stage's profile comes from a published single-model
benchmark; no published system with this exact pipeline was measured. Every
request passes through all three stages with the same token count. The
middle stage is the bottleneck and sets the pipeline capacity.

**Why this configuration.** A pipeline is where per-stage greedy rules can
interact badly: a large batch released upstream arrives downstream all at
once. Having a clear bottleneck in the middle makes that interaction
visible.

## D15. Pre-registration

The hypotheses, conditions, baseline search, model selection, statistics and
decision rules for Phase 2 are fixed in
[preregistration.md](preregistration.md) before any policy is evaluated on a
test split. That document is not edited after approval; changes go to its
"Deviations" section.

## D16. Learned-policy interface

**Decision.** One environment step is one engine decision point. Actions:
wait, or dispatch at most 1, 2, 4, 8, 16, 32. The observation includes how
long the oldest queued request has waited. Reward is minus the
request-seconds accrued since the previous decision. Trained models are
evaluated through `LearnedPolicy`, which plugs into the same `simulate()`
loop as the rules.

**Why.** (1) Without a wait action and the oldest-wait feature a policy
cannot learn to delay on purpose, which is the only way to beat a greedy
rule in a single stage (D10). (2) The time integral of the number of
requests in the system equals the sum of latencies, so the undiscounted
return is exactly the reported metric; a test checks this identity. A
hand-weighted reward, as in the original project, can be maximised without
improving the metric. (3) Sharing `observe` / `to_decision` between training
and evaluation removes the chance that the two paths differ.

**Reward unit.** The integral is divided by a constant per scenario (a
full batch of 32 held for the time a full batch takes at every stage), so
rewards are of order one. The first choice, dividing by the single-request
processing time, gave per-step rewards down to about -500 and both
algorithms collapsed to "always wait" (pre-registration, Deviation 1;
`results/rl/diagnostics/`). The unit is a positive constant, so it changes
how well the value network can fit, not which policy is best.

**Trade-offs.** Decisions are unevenly spaced in time, so a discount factor
below one weights cost by decision count rather than by time; the discount
is kept close to one (0.999). "Wait" is overridden by a full dispatch once
the oldest request has waited a whole SLO, so a policy cannot stall an
episode; this slightly limits the policy space.
