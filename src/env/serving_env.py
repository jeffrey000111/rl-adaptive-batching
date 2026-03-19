"""
Gymnasium environment for adaptive batching in ML inference serving.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InferenceServingEnv(gym.Env):
    """
    Simulates an ML inference serving queue.

    State (Box, 4 dims):
        [queue_length_norm, avg_wait_norm, arrival_rate_norm, gpu_util]

    Action (Discrete 7):
        index into BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    Reward:
        throughput bonus - latency penalty - SLO violation penalty - queue overflow penalty
    """

    metadata = {"render_modes": ["human"]}
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(self, max_steps=1000, slo_ms=100.0, base_rate=50.0,
                 pattern="steady", lat_w=0.6, tput_w=0.4, seed=None):
        super().__init__()
        self.max_steps = max_steps
        self.slo_ms = slo_ms
        self.base_rate = base_rate
        self.pattern = pattern
        self.lat_w = lat_w
        self.tput_w = tput_w

        self.action_space = spaces.Discrete(len(self.BATCH_SIZES))
        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        )
        self._rng = np.random.default_rng(seed)
        self.reset()

    def _arrival_rate(self):
        t = self.step_count / self.max_steps
        if self.pattern == "steady":
            return self.base_rate
        elif self.pattern == "bursty":
            if self._rng.random() < 0.15:
                return self.base_rate * self._rng.uniform(3, 8)
            return self.base_rate * self._rng.uniform(0.5, 1.5)
        elif self.pattern == "diurnal":
            return max(5, self.base_rate * (1 + 0.8 * np.sin(2 * np.pi * t)))
        return self.base_rate

    def _latency(self, bs):
        base = 5.0
        per_req = 2.0
        overhead = 1.5 * np.log2(bs + 1)
        noise = self._rng.normal(0, 1.0)
        return max(1.0, base + per_req * bs + overhead + noise)

    def _obs(self):
        return np.array([
            min(self.queue / 200, 1),
            min(self.wait / 500, 1),
            min(self.rate / 400, 1),
            self.gpu_util,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.queue = float(self._rng.poisson(self.base_rate * 0.5))
        self.wait = 0.0
        self.rate = self.base_rate
        self.gpu_util = 0.0
        self.served_total = 0
        self.violations = 0
        self.latency_sum = 0.0
        return self._obs(), {}

    def step(self, action):
        bs = self.BATCH_SIZES[action]
        self.step_count += 1

        served = min(bs, int(self.queue))
        lat = self._latency(served) if served > 0 else 0.0
        self.gpu_util = min(served / 64, 1.0) if served > 0 else 0.0
        self.queue -= served

        self.rate = self._arrival_rate()
        arrivals = self._rng.poisson(self.rate * 0.1)
        self.queue += arrivals

        if self.queue > 0:
            self.wait = self.wait * 0.8 + lat * 0.2 + self.queue * 0.3
        else:
            self.wait *= 0.5

        # reward
        r = self.tput_w * (served / 64)
        if served > 0:
            r -= self.lat_w * max(0, lat - self.slo_ms) / self.slo_ms
        if lat > self.slo_ms and served > 0:
            r -= 0.5
            self.violations += 1
        if self.queue > 100:
            r -= 0.3 * (self.queue - 100) / 100
        if self.queue > 20 and served < 4:
            r -= 0.2

        self.served_total += served
        self.latency_sum += lat
        done = self.step_count >= self.max_steps

        info = dict(batch_size=bs, served=served, latency_ms=lat,
                    queue=self.queue, rate=self.rate, gpu_util=self.gpu_util,
                    violations=self.violations, total_served=self.served_total)
        return self._obs(), r, done, False, info


if __name__ == "__main__":
    env = InferenceServingEnv(pattern="bursty", seed=42)
    obs, _ = env.reset()
    tot = 0
    for i in range(100):
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        tot += r
        if i % 25 == 0:
            print(f"step {i}: bs={info['batch_size']}, served={info['served']}, "
                  f"lat={info['latency_ms']:.1f}ms, q={info['queue']:.0f}")
        if done:
            break
    print(f"\ntotal reward: {tot:.2f}, served: {info['total_served']}, violations: {info['violations']}")
