# Diagnostics of the collapsed tuning runs (2026-11-11, train/dev only)

The pre-registered tuning stage (`results/rl/tuning/`, reward exactly as
registered) produced policies that almost always choose "wait" and reach a
dev mean latency of 6.6-8.9 s on pipeline / conv / load 0.7, where the
strongest rule reaches 0.23 s and the constant action "dispatch at most 32"
reaches 0.24 s.

`collapse_diagnosis.py` shows the action counts of the selected checkpoints
and the reward scale seen in training (per-step rewards down to about -500,
episode returns between about -3,000 and -63,000).

`reward_scale_check.py` retrains for 150,000 decisions (seed 100) with the
reward multiplied by a positive constant so that it is of order one. Output
of the run on 2026-11-11 (dev mean latency in ms after every 30,000
decisions):

    dqn static scale                   pipeline-conv-load0.7  [13807, 5318, 7303, 240, 235]
    ppo static scale                   pipeline-conv-load0.7  [619, 694, 631, 588, 551]
    ppo static scale gamma=1           pipeline-conv-load0.7  [5057, 932, 921, 1067, 468]
    ppo static scale lr3e-4            pipeline-conv-load0.7  [827, 1450, 5810, 8554, 5565]
    ppo VecNormalize reward            pipeline-conv-load0.7  [1017, 2288, 2621, 743, 717]
    ppo registered (no scaling)        pipeline-conv-load0.7  [11650, 11212, 13477, 7745, 7981]
    ppo static scale                   single-conv-load0.7    [86, 71, 71, 71, 71]
    dqn static scale                   single-conv-load0.7    [6353, 5605, 5079, 1790, 3685]

Strongest rule on dev: 229.7 ms (pipeline), 70.8 ms (single).

These scripts were run from a scratch directory and are kept here as a
record; they are not part of the pre-registered pipeline. No test split was
read.
