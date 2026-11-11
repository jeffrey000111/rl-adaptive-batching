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
