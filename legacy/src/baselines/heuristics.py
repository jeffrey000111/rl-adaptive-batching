"""
Baselines for adaptive batching comparison.
Per professor feedback: DQN is a baseline, not a core algorithm.
Epsilon-Greedy and UCB are exploration strategies, not core algorithms.
"""

import numpy as np


class StaticBatcher:
    """always same batch size"""
    def __init__(self, idx=3):
        self.idx = idx
        self.name = f"Static(bs={[1,2,4,8,16,32,64][idx]})"
    def predict(self, obs):
        return self.idx


class TimeoutBatcher:
    """big batch if queue above threshold, small otherwise"""
    def __init__(self, thresh=0.3, big=4, small=2):
        self.thresh = thresh
        self.big = big
        self.small = small
        self.name = "Timeout"
    def predict(self, obs):
        return self.big if obs[0] >= self.thresh else self.small


class ThresholdBatcher:
    """batch size proportional to queue length"""
    def __init__(self):
        self.name = "Threshold"
    def predict(self, obs):
        q = obs[0]
        if q < 0.05: return 0
        if q < 0.1: return 1
        if q < 0.2: return 2
        if q < 0.35: return 3
        if q < 0.5: return 4
        if q < 0.75: return 5
        return 6


def run_baseline(env, agent, episodes=5):
    """evaluate a heuristic baseline over multiple episodes"""
    rewards, served_list, viol_list = [], [], []
    latencies = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0
        ep_latencies = []
        while True:
            a = agent.predict(obs)
            obs, r, done, _, info = env.step(a)
            ep_reward += r
            if info["served"] > 0:
                ep_latencies.append(info["latency_ms"])
            if done:
                break
        rewards.append(ep_reward)
        served_list.append(info["total_served"])
        viol_list.append(info["violations"])
        latencies.extend(ep_latencies)

    lat_arr = np.array(latencies) if latencies else np.array([0])
    return {
        "name": agent.name,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "served_mean": np.mean(served_list),
        "violations_mean": np.mean(viol_list),
        "latency_mean": np.mean(lat_arr),
        "latency_p99": np.percentile(lat_arr, 99) if len(lat_arr) > 0 else 0,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from env.serving_env import InferenceServingEnv

    for pattern in ["steady", "bursty", "diurnal"]:
        env = InferenceServingEnv(pattern=pattern, seed=42)
        print(f"\n=== {pattern.upper()} traffic ===")
        for agent in [StaticBatcher(3), TimeoutBatcher(), ThresholdBatcher()]:
            res = run_baseline(env, agent)
            print(f"  {res['name']:20s} reward={res['reward_mean']:7.1f} "
                  f"served={res['served_mean']:6.0f} violations={res['violations_mean']:3.0f} "
                  f"lat_mean={res['latency_mean']:.1f}ms p99={res['latency_p99']:.1f}ms")
