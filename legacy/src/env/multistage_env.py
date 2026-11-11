"""
Multi-Stage Inference Serving Environment

Extends our single-stage simulator to a multi-stage disaggregated pipeline,
inspired by vllm-omni's stage abstraction (e.g., Qwen-Omni's
Thinker -> Talker -> Code2wav pipeline).

Each request flows through N stages sequentially. The agent picks a batch
size for EACH stage at every step. The reward considers end-to-end Job
Completion Time (JCT), which is the metric vllm-omni optimizes for.

Place this file at: src/env/multistage_env.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class MultiStageServingEnv(gym.Env):
    """
    Multi-stage serving simulator.

    State (per stage): queue depth, avg wait time, arrival rate
    Total state: 3 * num_stages + 1 (last dim is global utilization)

    Action: MultiDiscrete - one batch size choice per stage
            Each stage picks from {1, 2, 4, 8, 16, 32, 64} (7 options)

    Reward: -w_lat * end_to_end_latency_penalty + w_thr * throughput_bonus
            where end-to-end JCT = sum of per-stage processing times
    """

    metadata = {"render_modes": []}

    def __init__(self, num_stages=3, pattern="bursty", seed=42,
                 w_lat=0.5, w_thr=0.5, max_steps=1000):
        super().__init__()

        self.num_stages = num_stages
        self.pattern = pattern
        self.w_lat = w_lat
        self.w_thr = w_thr
        self.max_steps = max_steps

        # Batch size options
        self.batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
        self.num_batch_options = len(self.batch_sizes)

        # Per-stage processing characteristics (mimics Thinker/Talker/Code2wav)
        # Stage 0 (Thinker): heavy compute, slow
        # Stage 1 (Talker): medium compute
        # Stage 2 (Code2wav): light compute, fast
        self.stage_base_latency = [3.0, 2.0, 1.5][:num_stages]
        self.stage_per_req_cost = [0.8, 0.5, 0.3][:num_stages]

        # State and action spaces
        # State: [q_depth, avg_wait, arr_rate] per stage + global utilization
        state_dim = 3 * num_stages + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        # Action: one batch size index per stage
        self.action_space = spaces.MultiDiscrete(
            [self.num_batch_options] * num_stages
        )

        self.rng = np.random.RandomState(seed)
        self._setup_traffic()
        self.reset(seed=seed)

    def _setup_traffic(self):
        """Configure traffic pattern parameters."""
        if self.pattern == "steady":
            self.base_rate = 5.0
            self.spike_prob = 0.0
        elif self.pattern == "bursty":
            self.base_rate = 5.0
            self.spike_prob = 0.15
        elif self.pattern == "diurnal":
            self.base_rate = 5.0
            self.spike_prob = 0.0
            self.diurnal_period = 200
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.step_count = 0
        # Each stage has its own queue (list of (arrival_time, request_id))
        self.queues = [deque() for _ in range(self.num_stages)]
        # Track requests in flight (which stage they're at)
        self.next_request_id = 0
        # Stats
        self.total_served = 0
        self.total_jct = 0.0  # cumulative end-to-end completion time
        self.jct_samples = []
        # Track when each request entered stage 0 (for JCT calculation)
        self.request_start_times = {}

        return self._get_state(), {}

    def _get_arrival_rate(self):
        """Get current arrival rate based on traffic pattern."""
        if self.pattern == "diurnal":
            phase = (self.step_count / self.diurnal_period) * 2 * np.pi
            return self.base_rate * (1.0 + 0.5 * np.sin(phase))

        rate = self.base_rate
        if self.rng.random() < self.spike_prob:
            rate *= self.rng.uniform(3.0, 8.0)
        return rate

    def _get_state(self):
        """Build the state vector."""
        state = []
        max_queue = 100.0  # for normalization
        for q in self.queues:
            depth = min(len(q) / max_queue, 1.0)
            if len(q) > 0:
                avg_wait = np.mean([self.step_count - t for t, _ in q])
                avg_wait = min(avg_wait / 50.0, 1.0)  # normalize
            else:
                avg_wait = 0.0
            # Approximate per-stage arrival rate as recent flow
            arr_rate = min(len(q) / 20.0, 1.0)
            state.extend([depth, avg_wait, arr_rate])

        # Global utilization: total queue depth across all stages
        global_util = sum(len(q) for q in self.queues) / (max_queue * self.num_stages)
        state.append(min(global_util, 1.0))

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        action: array of batch size indices, one per stage
        """
        self.step_count += 1

        # === 1. New requests arrive at stage 0 ===
        rate = self._get_arrival_rate()
        n_arrivals = self.rng.poisson(rate)
        for _ in range(n_arrivals):
            req_id = self.next_request_id
            self.next_request_id += 1
            self.queues[0].append((self.step_count, req_id))
            self.request_start_times[req_id] = self.step_count

        # === 2. Process each stage in reverse order ===
        # (so we don't pass requests forward and process them in same step)
        completed_this_step = 0
        latency_penalty = 0.0

        for stage_idx in reversed(range(self.num_stages)):
            batch_size_idx = action[stage_idx]
            batch_size = self.batch_sizes[batch_size_idx]

            # Compute processing latency for this batch
            base = self.stage_base_latency[stage_idx]
            per_req = self.stage_per_req_cost[stage_idx]
            stage_latency = base + per_req * np.log(batch_size + 1)
            stage_latency += self.rng.normal(0, 0.2)  # noise
            stage_latency = max(stage_latency, 0.1)

            # Pull up to batch_size requests from this stage's queue
            n_to_process = min(batch_size, len(self.queues[stage_idx]))
            processed_reqs = []
            for _ in range(n_to_process):
                arrival_time, req_id = self.queues[stage_idx].popleft()
                # Wait time at this stage = current step - when it joined this stage
                wait_at_stage = self.step_count - arrival_time
                latency_penalty += wait_at_stage * 0.5
                processed_reqs.append(req_id)

            # Move processed requests to next stage (or complete if last stage)
            if stage_idx < self.num_stages - 1:
                for req_id in processed_reqs:
                    self.queues[stage_idx + 1].append((self.step_count, req_id))
            else:
                # Last stage: requests complete here
                for req_id in processed_reqs:
                    if req_id in self.request_start_times:
                        jct = self.step_count - self.request_start_times[req_id]
                        self.total_jct += jct
                        self.jct_samples.append(jct)
                        del self.request_start_times[req_id]
                completed_this_step += n_to_process

        # === 3. Compute reward ===
        # Throughput bonus: completed end-to-end requests
        throughput_bonus = completed_this_step * 2.0
        # Latency penalty includes any queue buildup
        for q in self.queues:
            if len(q) > 30:  # penalize backlog
                latency_penalty += (len(q) - 30) * 0.3

        reward = -self.w_lat * (latency_penalty / 10.0) + self.w_thr * throughput_bonus
        self.total_served += completed_this_step

        # === 4. Done? ===
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "served": self.total_served,
            "queue_depths": [len(q) for q in self.queues],
            "avg_jct": np.mean(self.jct_samples) if self.jct_samples else 0.0,
            "p99_jct": np.percentile(self.jct_samples, 99) if len(self.jct_samples) > 10 else 0.0,
        }

        return self._get_state(), float(reward), terminated, truncated, info


# Helper baselines for multi-stage env
class MultiStageHeuristic:
    """Base class for multi-stage heuristics."""

    def __init__(self, num_stages=3, num_options=7):
        self.num_stages = num_stages
        self.num_options = num_options

    def predict(self, obs, deterministic=True):
        raise NotImplementedError


class StaticMultiStage(MultiStageHeuristic):
    """Fixed batch size at every stage."""

    def __init__(self, batch_idx=3, **kwargs):  # default = batch size 8
        super().__init__(**kwargs)
        self.batch_idx = batch_idx

    def predict(self, obs, deterministic=True):
        return np.array([self.batch_idx] * self.num_stages), None


class ThresholdMultiStage(MultiStageHeuristic):
    """Each stage independently scales batch size with its queue depth."""

    def predict(self, obs, deterministic=True):
        # State layout: [q0, w0, a0, q1, w1, a1, ..., global]
        action = []
        for s in range(self.num_stages):
            depth = obs[s * 3]
            # Map depth (0-1) to batch size index
            if depth < 0.05:
                idx = 1  # batch 2
            elif depth < 0.10:
                idx = 2  # batch 4
            elif depth < 0.20:
                idx = 3  # batch 8
            elif depth < 0.35:
                idx = 4  # batch 16
            elif depth < 0.55:
                idx = 5  # batch 32
            else:
                idx = 6  # batch 64
            action.append(idx)
        return np.array(action), None


class DownstreamAwareMultiStage(MultiStageHeuristic):
    """
    Smarter heuristic: avoid overwhelming downstream stages.
    If next stage has high queue, batch less aggressively here.
    """

    def predict(self, obs, deterministic=True):
        action = []
        for s in range(self.num_stages):
            depth = obs[s * 3]
            # Check downstream queue depth (if not last stage)
            if s < self.num_stages - 1:
                downstream_depth = obs[(s + 1) * 3]
            else:
                downstream_depth = 0.0

            # Reduce batch size if downstream is congested
            congestion_factor = 1.0 - 0.5 * downstream_depth
            effective_depth = depth * congestion_factor

            if effective_depth < 0.05:
                idx = 1
            elif effective_depth < 0.10:
                idx = 2
            elif effective_depth < 0.20:
                idx = 3
            elif effective_depth < 0.35:
                idx = 4
            elif effective_depth < 0.55:
                idx = 5
            else:
                idx = 6
            action.append(idx)
        return np.array(action), None
