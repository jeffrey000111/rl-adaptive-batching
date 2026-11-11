# Legacy: original course-project code (Spring 2026)

Everything in this directory is the original course project, kept so that the
flawed result can be inspected and reproduced. Nothing here is used by the
reworked project, and none of its results should be cited.

The only changes made after archiving: files were moved here, two comments in
`notebooks/01_eda_azure_traces.py` that claimed the trace was used to set
simulator parameters were corrected, and `repro_fixed_batch.py` was added.

## Known defects

1. `src/env/multistage_env.py`: `step()` computes `stage_latency` and never
   uses it. Batches take no time and do not occupy the server, so the largest
   batch is always best. `repro_fixed_batch.py` demonstrates this.
2. The multi-stage state feature named `arr_rate` is `len(queue) / 20`, i.e. a
   second copy of queue depth, not an arrival rate.
3. `src/env/serving_env.py`: each step is a fixed time slice regardless of how
   long the dispatched batch takes; the reported latency is batch processing
   time only and excludes queueing.
4. Baselines: one fixed batch size (8) and hand-picked thresholds, no tuning.
5. `src/agents/sac_agent.py`: SAC is run through a continuous-to-discrete
   action wrapper and is unstable across seeds.
6. The Azure trace files are read only by `notebooks/01_eda_azure_traces.py`;
   no environment uses them.
7. `run_multistage.py` trains with a single seed.

## Running

```bash
pip install -e ".[legacy]"            # from the repository root
python legacy/repro_fixed_batch.py    # fixed-batch check, no training
```

The original `run_*.py` scripts additionally need `stable-baselines3`,
`pandas` and `matplotlib` (`legacy/requirements.txt`) and must be run with
`legacy/` as the working directory.
