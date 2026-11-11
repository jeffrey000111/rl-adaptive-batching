"""
Gymnasium environment for ML inference serving with adaptive batching.
Simulates a request queue where an RL agent picks batch sizes.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InferenceServingEnv(gym.Env):
    """
    Simulates an inference serving queue.

    State: [queue_length, avg_wait_time, arrival_rate, gpu_utilization]
    Action: index into BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    Reward: weighted combo of negative latency + throughput bonus - SLO penalty
    """

    metadata = {"render_modes": ["human"]}

    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(
        self,
        max_queue=256,
        slo_ms=200,
        base_latency_ms=10,
        per_request_latency_ms=2,
        latency_noise_std=1.0,
        arrival_rate_range=(10, 500),
        episode_length=500,
        latency_weight=0.6,
        throughput_weight=0.3,
        slo_penalty_weight=0.1,
        traffic_pattern="steady",
        seed=None,
    ):
        super().__init__()

        self.max_queue = max_queue
        self.slo_ms = slo_ms
        self.base_latency_ms = base_latency_ms
        self.per_request_latency_ms = per_request_latency_ms
        self.latency_noise_std = latency_noise_std
        self.arrival_rate_range = arrival_rate_range
        self.episode_length = episode_length
        self.latency_weight = latency_weight
        self.throughput_weight = throughput_weight
        self.slo_penalty_weight = slo_penalty_weight
        self.traffic_pattern = traffic_pattern

        # action = index into BATCH_SIZES
        self.action_space = spaces.Discrete(len(self.BATCH_SIZES))

        # state: [norm_queue_len, norm_avg_wait, norm_arrival_rate, gpu_util]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.reset()

    def _get_arrival_rate(self):
        """get current arrival rate based on traffic pattern"""
        lo, hi = self.arrival_rate_range
        t = self.step_count / self.episode_length

        if self.traffic_pattern == "steady":
            return (lo + hi) / 2

        elif self.traffic_pattern == "bursty":
            # random spikes
            base = (lo + hi) / 3
            if self.rng.random() < 0.15:
                return self.rng.uniform(hi * 0.7, hi)
            return base

        elif self.traffic_pattern == "diurnal":
            # sine wave pattern like day/night
            rate = (lo + hi) / 2 + ((hi - lo) / 2) * np.sin(2 * np.pi * t)
            return max(lo, rate)

        return (lo + hi) / 2

    def _simulate_latency(self, batch_size):
        """model latency for a given batch size (in ms)"""
        # latency grows sub-linearly with batch size (GPU parallelism helps)
        latency = self.base_latency_ms + self.per_request_latency_ms * (batch_size ** 0.7)
        noise = self.rng.normal(0, self.latency_noise_std)
        return max(1.0, latency + noise)

    def _get_obs(self):
        norm_queue = min(self.queue_length / self.max_queue, 1.0)
        norm_wait = min(self.avg_wait_ms / self.slo_ms, 1.0) if self.avg_wait_ms > 0 else 0.0
        lo, hi = self.arrival_rate_range
        norm_arrival = (self.current_arrival_rate - lo) / (hi - lo + 1e-8)
        norm_arrival = np.clip(norm_arrival, 0, 1)
        gpu_util = self.gpu_utilization

        return np.array([norm_queue, norm_wait, norm_arrival, gpu_util], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.queue_length = self.rng.integers(5, 30)
        self.avg_wait_ms = 0.0
        self.current_arrival_rate = self._get_arrival_rate()
        self.gpu_utilization = 0.0

        # metrics for tracking
        self.total_requests_served = 0
        self.total_latency = 0.0
        self.slo_violations = 0
        self.total_requests_arrived = 0

        return self._get_obs(), {}

    def step(self, action):
        batch_size = self.BATCH_SIZES[action]
        self.step_count += 1

        # --- process batch ---
        actual_batch = min(batch_size, self.queue_length)

        if actual_batch > 0:
            latency = self._simulate_latency(actual_batch)
            total_latency_for_batch = latency + self.avg_wait_ms
            self.queue_length -= actual_batch
            self.total_requests_served += actual_batch
            self.total_latency += total_latency_for_batch * actual_batch

            # count SLO violations
            if total_latency_for_batch > self.slo_ms:
                self.slo_violations += actual_batch

            # gpu utilization = how full the batch was vs max
            self.gpu_utilization = actual_batch / self.BATCH_SIZES[-1]
        else:
            latency = 0
            total_latency_for_batch = 0
            self.gpu_utilization = 0.0

        # --- new arrivals ---
        self.current_arrival_rate = self._get_arrival_rate()
        # arrivals per step (each step ~ 10ms window)
        new_arrivals = self.rng.poisson(self.current_arrival_rate * 0.01)
        self.queue_length = min(self.queue_length + new_arrivals, self.max_queue)
        self.total_requests_arrived += new_arrivals

        # update avg wait (rough estimate: increases with queue, decreases when served)
        if self.queue_length > 0:
            self.avg_wait_ms = self.avg_wait_ms * 0.9 + (self.queue_length * 0.5) * 0.1
        else:
            self.avg_wait_ms *= 0.5

        # --- reward ---
        reward = 0.0

        # latency penalty (normalized)
        if actual_batch > 0:
            norm_latency = total_latency_for_batch / self.slo_ms
            reward -= self.latency_weight * norm_latency

        # throughput bonus
        throughput_ratio = actual_batch / self.BATCH_SIZES[-1]
        reward += self.throughput_weight * throughput_ratio

        # SLO violation penalty
        if actual_batch > 0 and total_latency_for_batch > self.slo_ms:
            reward -= self.slo_penalty_weight * 2.0

        # small penalty for empty batches (wasted step)
        if actual_batch == 0:
            reward -= 0.05

        # --- done? ---
        terminated = self.step_count >= self.episode_length
        truncated = False

        info = {
            "batch_size": actual_batch,
            "latency_ms": total_latency_for_batch,
            "queue_length": self.queue_length,
            "arrival_rate": self.current_arrival_rate,
            "gpu_utilization": self.gpu_utilization,
            "slo_violation": total_latency_for_batch > self.slo_ms if actual_batch > 0 else False,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_metrics(self):
        """return summary metrics for the episode"""
        avg_lat = self.total_latency / max(self.total_requests_served, 1)
        slo_rate = 1 - (self.slo_violations / max(self.total_requests_served, 1))
        return {
            "total_served": self.total_requests_served,
            "avg_latency_ms": avg_lat,
            "slo_attainment": slo_rate,
            "total_arrived": self.total_requests_arrived,
        }


# register the env so we can use gym.make()
gym.register(
    id="InferenceServing-v0",
    entry_point="src.env.inference_env:InferenceServingEnv",
)
