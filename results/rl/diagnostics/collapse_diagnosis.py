"""Diagnose the collapsed tuning runs. Train/dev only."""
import sys, json, collections
import numpy as np, torch
torch.set_num_threads(1)
from adaptive_batching.experiment import BY_ID, load_splits, run
from adaptive_batching.metrics import summarize
from adaptive_batching.policies import GreedyBatch
from adaptive_batching.rl_env import BatchingEnv, LearnedPolicy, NUM_ACTIONS, observe
from adaptive_batching.training import load_model, backlog_max
root = sys.argv[1]
c = BY_ID["pipeline-conv-load0.7"]; sc = c.scenario; sp = load_splits(c)
for cap in (1, 8, 32):
    print(f"dev GreedyBatch({cap}) mean ms", round(1e3 * summarize(run(sp['dev'], sc, GreedyBatch(cap))).mean_latency_s, 1))
class Const:
    def __init__(s, a): s.a = a
    def predict(s, o, deterministic=True): return s.a, None
for a in range(NUM_ACTIONS):
    print(f"dev constant action {a} mean ms", round(1e3 * summarize(run(sp['dev'], sc, LearnedPolicy(Const(a), sc))).mean_latency_s, 1))
class Count:
    def __init__(s, m): s.m = m; s.n = collections.Counter()
    def predict(s, o, deterministic=True):
        a, _ = s.m.predict(o, deterministic=True); s.n[int(a)] += 1; return a, None
for name in ("ppo_0", "ppo_2", "dqn_0"):
    algo = name.split("_")[0]
    m = Count(load_model(algo, f"{root}/artifacts/models/tuning/{name}.zip"))
    r = summarize(run(sp['dev'], sc, LearnedPolicy(m, sc)))
    print(name, "dev mean ms", round(1e3 * r.mean_latency_s), "action counts", dict(sorted(m.n.items())))
# reward scale seen in training
env = BatchingEnv(sp['train'], sc, episode_requests=512, backlog_max=backlog_max(sp['train'], sc))
rs = []; rets = []
for ep in range(20):
    env.reset(seed=ep); tot = 0; n = 0
    while True:
        _, r, term, _, _ = env.step(NUM_ACTIONS - 1); rs.append(r); tot += r; n += 1
        if term: break
    rets.append((tot, n))
print("always-dispatch-32 in training env: per-step reward mean %.1f, min %.1f; episode return mean %.0f, steps mean %.0f" % (np.mean(rs), np.min(rs), np.mean([t for t, _ in rets]), np.mean([n for _, n in rets])))
rs = []; rets = []
for ep in range(5):
    env.reset(seed=ep); tot = 0; n = 0
    while True:
        _, r, term, _, _ = env.step(0); rs.append(r); tot += r; n += 1
        if term: break
    rets.append((tot, n))
print("always-wait in training env: per-step reward mean %.1f; episode return mean %.0f, steps mean %.0f" % (np.mean(rs), np.mean([t for t, _ in rets]), np.mean([n for _, n in rets])))
