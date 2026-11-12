"""Gymnasium environment and policy adapter for learned batching policies.

One step of the environment is one decision point of the engine: some
stage's server is idle and its queue is not empty. A single policy controls
every stage; the observation says which stage is deciding.

Actions (``Discrete(1 + len(BATCH_SIZES))``):
    0        wait: do nothing until another request joins this queue or a
             short timer expires
    1..6     dispatch a batch of at most 1, 2, 4, 8, 16, 32 queued requests

Reward: minus the request-seconds accumulated in the system since the
previous decision, in units of the pipeline's single-request processing
time. Summed over an episode this is exactly minus the total latency of the
episode's requests, so the return and the reported metric cannot disagree.

``observe`` and ``to_decision`` are shared by the training environment and
by :class:`LearnedPolicy`, which plugs a trained model into the same
``simulate`` loop that evaluates rule-based policies.
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from adaptive_batching.engine import Decision, Dispatch, Engine, StageView, Wait
from adaptive_batching.load import capacity_rps
from adaptive_batching.scenarios import MAX_BATCH, Scenario
from adaptive_batching.service_time import BATCH_SIZES
from adaptive_batching.traces import Trace

WAIT_ACTION = 0
NUM_ACTIONS = 1 + len(BATCH_SIZES)
_CLIP = 4.0
# Arrival-rate windows, in units of the time capacity needs to serve one request.
_RATE_WINDOWS = (20.0, 200.0)


def single_request_time_s(scenario: Scenario) -> float:
    """Processing time of one request through the whole pipeline, unbatched."""
    return sum(m.uniform_batch_time_s(1, m.reference_length()) for m in scenario.stages)


def observation_size(scenario: Scenario, include_arrival_rate: bool = True) -> int:
    stages = len(scenario.stages)
    return stages + 2 + 2 * stages + (len(_RATE_WINDOWS) if include_arrival_rate else 0)


def observe(
    view: StageView, scenario: Scenario, include_arrival_rate: bool = True
) -> NDArray[np.float32]:
    """Features of a decision point, each scaled to roughly [0, 1] and clipped.

    Layout: one-hot of the deciding stage; its queue length and the time its
    oldest request has waited; queue length and remaining busy time of every
    stage; optionally the recent external arrival rate over two windows.
    """
    stages = scenario.stages
    features = [1.0 if i == view.stage else 0.0 for i in range(len(stages))]
    features.append(view.queue_len / MAX_BATCH)
    features.append(view.oldest_wait_s / scenario.slo_s)
    for i, model in enumerate(stages):
        features.append(view.queue_len_at(i) / MAX_BATCH)
        longest = model.uniform_batch_time_s(MAX_BATCH, model.reference_length())
        features.append(view.busy_remaining_s(i) / longest)
    if include_arrival_rate:
        capacity = capacity_rps(stages, MAX_BATCH)
        for window in _RATE_WINDOWS:
            window_s = window / capacity
            features.append(view.arrivals_in_last(window_s) / window)
    return np.clip(np.array(features, dtype=np.float32), 0.0, _CLIP)


def to_decision(action: int, view: StageView, scenario: Scenario) -> Decision:
    """Translate an action index into an engine decision.

    A dispatch larger than the queue sends the whole queue. Waiting is
    overridden by a full dispatch once the oldest request has already waited
    a whole SLO, so a policy cannot stall the simulation.
    """
    if action == WAIT_ACTION:
        if view.oldest_wait_s >= scenario.slo_s:
            return Dispatch(min(view.queue_len, MAX_BATCH))
        model = scenario.stages[view.stage]
        return Wait(0.5 * model.uniform_batch_time_s(1, model.reference_length()))
    return Dispatch(min(BATCH_SIZES[action - 1], view.queue_len))


class BatchingEnv(gym.Env[NDArray[np.float32], np.int64]):
    """Episodes are windows of consecutive requests cut from one trace.

    ``trace`` must already be scaled to the wanted load. With
    ``episode_requests`` set, each reset picks a uniformly random start
    request and plays that many requests from an empty system; with ``None``
    the whole trace is one episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace: Trace,
        scenario: Scenario,
        episode_requests: int | None = 512,
        include_arrival_rate: bool = True,
        slo_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        if episode_requests is not None and not 0 < episode_requests <= len(trace):
            raise ValueError("episode_requests must be between 1 and the trace length")
        self._arrivals = trace.relative_arrivals()
        self._tokens = trace.context_tokens
        self._scenario = scenario
        self._episode_requests = episode_requests
        self._include_arrival_rate = include_arrival_rate
        self._slo_penalty = slo_penalty
        self._time_unit_s = single_request_time_s(scenario)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(
            0.0, _CLIP, (observation_size(scenario, include_arrival_rate),), np.float32
        )
        self._engine: Engine | None = None
        self._view: StageView | None = None
        self._charged_seconds = 0.0
        self._charged_completions = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if self._episode_requests is None:
            start, stop = 0, len(self._arrivals)
        else:
            last_start = len(self._arrivals) - self._episode_requests
            start = int(self.np_random.integers(0, last_start + 1))
            stop = start + self._episode_requests
        arrivals = self._arrivals[start:stop] - self._arrivals[start]
        self._engine = Engine(arrivals, self._tokens[start:stop], self._scenario.stages)
        self._charged_seconds = 0.0
        self._charged_completions = 0
        self._view = self._engine.next_decision()
        assert self._view is not None  # a non-empty episode always has a first decision
        return self._observe(), {"start_request": start}

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        engine, view = self._engine, self._view
        if engine is None or view is None:
            raise RuntimeError("reset() must be called before step()")
        engine.apply(to_decision(int(action), view, self._scenario))
        self._view = engine.next_decision()

        reward = -(engine.request_seconds - self._charged_seconds) / self._time_unit_s
        self._charged_seconds = engine.request_seconds
        if self._slo_penalty:
            reward -= self._slo_penalty * self._new_slo_violations(engine)

        terminated = self._view is None
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._observe()
        )
        return observation, float(reward), terminated, False, {}

    def result(self) -> Any:  # noqa: ANN401 - SimulationResult; avoids an import cycle in docs
        """Per-request record of the current episode."""
        if self._engine is None:
            raise RuntimeError("reset() must be called first")
        return self._engine.result()

    def _observe(self) -> NDArray[np.float32]:
        assert self._view is not None
        return observe(self._view, self._scenario, self._include_arrival_rate)

    def _new_slo_violations(self, engine: Engine) -> int:
        # Queues are FIFO at every stage, so requests complete in arrival order.
        done = engine.completed
        finished = engine._finish_s[-1, self._charged_completions : done]
        arrived = engine._arrival_s[self._charged_completions : done]
        self._charged_completions = done
        return int(((finished - arrived) > self._scenario.slo_s).sum())


class _Predictor(Protocol):
    def predict(
        self, observation: NDArray[np.float32], deterministic: bool = ...
    ) -> tuple[Any, Any]: ...


class LearnedPolicy:
    """Wraps a trained model (anything with SB3's ``predict``) as an engine policy."""

    def __init__(
        self, model: _Predictor, scenario: Scenario, include_arrival_rate: bool = True
    ) -> None:
        self._model = model
        self._scenario = scenario
        self._include_arrival_rate = include_arrival_rate

    def decide(self, view: StageView) -> Decision:
        observation = observe(view, self._scenario, self._include_arrival_rate)
        action, _ = self._model.predict(observation, deterministic=True)
        return to_decision(int(action), view, self._scenario)
