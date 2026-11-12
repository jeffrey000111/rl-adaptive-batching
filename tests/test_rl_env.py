from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from adaptive_batching.engine import simulate  # noqa: E402
from adaptive_batching.policies import GreedyBatch  # noqa: E402
from adaptive_batching.rl_env import (  # noqa: E402
    NUM_ACTIONS,
    WAIT_ACTION,
    BatchingEnv,
    LearnedPolicy,
    observation_size,
    reward_unit_s,
    single_request_time_s,
)
from adaptive_batching.scenarios import MAX_BATCH, PIPELINE_3STAGE, SINGLE_STAGE  # noqa: E402
from adaptive_batching.traces import Trace  # noqa: E402


def make_trace(n: int = 300, rate: float = 60.0, seed: int = 0) -> Trace:
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / rate, size=n))
    return Trace(
        name="synthetic",
        arrival_s=arrivals,
        context_tokens=rng.integers(1, 600, size=n).astype(np.int64),
        generated_tokens=np.ones(n, dtype=np.int64),
        start_s=0.0,
        end_s=float(arrivals[-1]) + 1.0,
    )


class Constant:
    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        return self.action, None


def run_episode(env: BatchingEnv, policy: Constant, seed: int = 0) -> tuple[float, int]:
    observation, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        assert env.observation_space.contains(observation)
        action, _ = policy.predict(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps


@pytest.mark.parametrize("scenario", [SINGLE_STAGE, PIPELINE_3STAGE])
def test_return_equals_minus_total_latency(scenario: object) -> None:
    env = BatchingEnv(make_trace(), scenario, episode_requests=None)  # type: ignore[arg-type]
    total, _ = run_episode(env, Constant(NUM_ACTIONS - 1))
    latency = env.result().latency_s
    assert not np.isnan(latency).any()
    assert total == pytest.approx(-latency.sum() / reward_unit_s(scenario), rel=1e-9)  # type: ignore[arg-type]


def test_largest_dispatch_action_matches_the_greedy_rule() -> None:
    trace = make_trace()
    env = BatchingEnv(trace, PIPELINE_3STAGE, episode_requests=None)
    run_episode(env, Constant(NUM_ACTIONS - 1))
    reference = simulate(
        trace.relative_arrivals(), trace.context_tokens, PIPELINE_3STAGE.stages,
        GreedyBatch(MAX_BATCH),
    )
    # The environment shifts the window to start at t=0, so compare latencies.
    np.testing.assert_allclose(env.result().latency_s, reference.latency_s)


def test_learned_policy_adapter_reproduces_the_environment() -> None:
    trace = make_trace()
    env = BatchingEnv(trace, SINGLE_STAGE, episode_requests=None)
    run_episode(env, Constant(3))
    replay = simulate(
        trace.relative_arrivals(), trace.context_tokens, SINGLE_STAGE.stages,
        LearnedPolicy(Constant(3), SINGLE_STAGE),
    )
    np.testing.assert_allclose(env.result().latency_s, replay.latency_s)


def test_always_waiting_cannot_stall_and_is_worse_than_dispatching() -> None:
    env = BatchingEnv(make_trace(n=100), SINGLE_STAGE, episode_requests=None)
    waiting, _ = run_episode(env, Constant(WAIT_ACTION))
    latency = env.result().latency_s
    assert not np.isnan(latency).any()
    # The oldest request waits a full SLO before the override sends the queue.
    assert latency[0] >= SINGLE_STAGE.slo_s
    dispatching, _ = run_episode(env, Constant(NUM_ACTIONS - 1))
    assert waiting < dispatching


def test_episodes_are_random_windows_and_reproducible_by_seed() -> None:
    env = BatchingEnv(make_trace(), SINGLE_STAGE, episode_requests=50)
    _, info_a = env.reset(seed=1)
    _, info_b = env.reset(seed=1)
    _, info_c = env.reset(seed=2)
    assert info_a == info_b != info_c
    _, steps = run_episode(env, Constant(1), seed=1)
    assert len(env.result().latency_s) == 50 and steps >= 50


def test_observation_layout_and_arrival_rate_ablation() -> None:
    with_rate = BatchingEnv(make_trace(), PIPELINE_3STAGE, episode_requests=50)
    without = BatchingEnv(
        make_trace(), PIPELINE_3STAGE, episode_requests=50, include_arrival_rate=False
    )
    assert with_rate.observation_space.shape == (observation_size(PIPELINE_3STAGE),) == (13,)
    assert without.observation_space.shape == (11,)
    observation, _ = with_rate.reset(seed=0)
    assert observation[:3].tolist() == [1.0, 0.0, 0.0]  # stage 0 decides first
    assert observation[3] == pytest.approx(1 / MAX_BATCH)  # one request queued
    assert observation[4] == 0.0  # it has not waited yet


def test_slo_penalty_charges_one_unit_per_late_request() -> None:
    trace = make_trace(n=100)
    plain = BatchingEnv(trace, SINGLE_STAGE, episode_requests=None)
    penalised = BatchingEnv(trace, SINGLE_STAGE, episode_requests=None, slo_penalty=1.0)
    for policy, expect_late in ((Constant(WAIT_ACTION), True), (Constant(NUM_ACTIONS - 1), False)):
        with_penalty = run_episode(penalised, policy)[0]
        late = int((penalised.result().latency_s > SINGLE_STAGE.slo_s).sum())
        assert (late > 0) == expect_late
        per_miss = single_request_time_s(SINGLE_STAGE) / reward_unit_s(SINGLE_STAGE)
        assert with_penalty == pytest.approx(run_episode(plain, policy)[0] - late * per_miss)


def test_backlog_start_queues_extra_requests_at_time_zero() -> None:
    env = BatchingEnv(make_trace(), SINGLE_STAGE, episode_requests=50, backlog_max=100)
    backlogs = []
    for seed in range(40):
        observation, info = env.reset(seed=seed)
        backlogs.append(info["backlog"])
        # Stage 0 sees the backlog plus the window's first request.
        assert observation[1] == pytest.approx(min((info["backlog"] + 1) / MAX_BATCH, 4.0))
        assert info["start_request"] >= info["backlog"]
    assert 0 in backlogs and max(backlogs) > 10 and max(backlogs) <= 100
    assert 0.25 < np.mean([b > 0 for b in backlogs]) < 0.75

    env.reset(seed=int(np.argmax(backlogs)))
    total = 0.0
    while True:
        _, reward, terminated, _, _ = env.step(NUM_ACTIONS - 1)
        total += reward
        if terminated:
            break
    latency = env.result().latency_s
    assert len(latency) == 50 + max(backlogs)
    assert total == pytest.approx(-latency.sum() / reward_unit_s(SINGLE_STAGE))


def test_backlog_needs_windowed_episodes() -> None:
    with pytest.raises(ValueError):
        BatchingEnv(make_trace(), SINGLE_STAGE, episode_requests=None, backlog_max=10)


def test_rewards_are_of_order_one_for_a_sensible_policy() -> None:
    env = BatchingEnv(make_trace(rate=30.0), PIPELINE_3STAGE, episode_requests=None)
    env.reset(seed=0)
    rewards = []
    while True:
        _, reward, terminated, _, _ = env.step(NUM_ACTIONS - 1)
        rewards.append(reward)
        if terminated:
            break
    assert -5.0 < min(rewards) <= max(rewards) <= 0.0
