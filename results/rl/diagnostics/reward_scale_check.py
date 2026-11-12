"""Does rescaling the reward fix the collapse? Train/dev only, seed 100, 150k steps."""
import sys, time
from multiprocessing import get_context

def job(v):
    import torch, gymnasium as gym
    torch.set_num_threads(1)
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from adaptive_batching.experiment import BY_ID, load_splits, run
    from adaptive_batching.metrics import summarize
    from adaptive_batching.rl_env import BatchingEnv, LearnedPolicy, single_request_time_s
    from adaptive_batching.scenarios import MAX_BATCH
    from adaptive_batching.training import backlog_max, FIXED, NET_ARCH
    name, cond, algo, scale, vecnorm, gamma, hp = v
    c = BY_ID[cond]; sc = c.scenario; sp = load_splits(c)
    env = BatchingEnv(sp["train"], sc, episode_requests=512, backlog_max=backlog_max(sp["train"], sc))
    if scale:
        k = single_request_time_s(sc) / (MAX_BATCH * sum(m.uniform_batch_time_s(MAX_BATCH, m.reference_length()) for m in sc.stages))
        env = gym.wrappers.TransformReward(env, lambda r: r * k)
    venv = DummyVecEnv([lambda: env])
    if vecnorm:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)
    cls = PPO if algo == "ppo" else DQN
    m = cls("MlpPolicy", venv, gamma=gamma, policy_kwargs={"net_arch": NET_ARCH}, seed=100, device="cpu", verbose=0, **FIXED[algo], **hp)
    out = []
    for i in range(5):
        m.learn(30_000, reset_num_timesteps=False)
        out.append(round(1e3 * summarize(run(sp["dev"], sc, LearnedPolicy(m, sc))).mean_latency_s))
    return f"{name:34s} {cond:22s} dev mean ms every 30k: {out}"

if __name__ == "__main__":
    P = {"learning_rate": 1e-3, "ent_coef": 0.0}; P3 = {"learning_rate": 3e-4, "ent_coef": 0.0}
    D = {"learning_rate": 1e-4, "exploration_fraction": 0.1}
    pc, sg = "pipeline-conv-load0.7", "single-conv-load0.7"
    V = [("ppo registered (no scaling)", pc, "ppo", False, False, 0.999, P),
         ("ppo static scale", pc, "ppo", True, False, 0.999, P),
         ("ppo static scale lr3e-4", pc, "ppo", True, False, 0.999, P3),
         ("ppo VecNormalize reward", pc, "ppo", False, True, 0.999, P),
         ("dqn static scale", pc, "dqn", True, False, 0.999, D),
         ("ppo static scale gamma=1", pc, "ppo", True, False, 1.0, P),
         ("ppo static scale", sg, "ppo", True, False, 0.999, P),
         ("dqn static scale", sg, "dqn", True, False, 0.999, D)]
    t = time.time()
    with get_context("spawn").Pool(6) as pool:
        for line in pool.imap_unordered(job, V): print(line, flush=True)
    print("wall", round(time.time() - t), "s")
