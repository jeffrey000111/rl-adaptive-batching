"""Discrete-event simulation of a batching inference pipeline.

The pipeline is a chain of stages. Each stage is one server with a FIFO
queue. A server runs one batch at a time and is occupied for the whole
processing time of that batch, so a request's latency is the time it spends
queueing plus the time its batches take, summed over the stages.

Time only advances between events (an arrival, a batch completion, or a
wake-up timer), never in fixed steps. A *decision point* occurs whenever a
stage's server is idle and its queue is not empty. At a decision point the
caller answers with either

* ``Dispatch(size)``: start a batch made of the first ``size`` queued
  requests now, or
* ``Wait(timeout_s)``: do nothing; ask again when another request joins this
  queue or when the timeout expires, whichever comes first.

The engine is driven from outside (``next_decision`` / ``apply``) rather than
calling a policy itself, so the same code serves rule-based policies (see
``simulate``) and a reinforcement-learning environment, which must hand
control back to the agent at every decision point.

The simulation contains no randomness: the same arrivals and the same
decisions always produce the same result.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from adaptive_batching.service_time import ServiceTimeModel

_COMPLETE = 0
_WAKE = 1


@dataclass(frozen=True)
class Dispatch:
    """Start a batch of the first ``size`` queued requests now."""

    size: int


@dataclass(frozen=True)
class Wait:
    """Hold the queue; ask again on the next arrival or after ``timeout_s``."""

    timeout_s: float


Decision = Dispatch | Wait


class SimulationStalledError(RuntimeError):
    """A policy kept waiting although a request had been queued for too long."""


@dataclass(frozen=True)
class Batch:
    """One executed batch, for metrics and debugging."""

    stage: int
    start_s: float
    duration_s: float
    size: int


class StageView:
    """Read-only view of the system offered to a policy at a decision point."""

    def __init__(self, engine: Engine, stage: int) -> None:
        self._engine = engine
        self.stage = stage
        self.now_s = engine.now_s
        self.num_stages = engine.num_stages
        self.queue_len = len(engine._queues[stage])
        head = engine._queues[stage][0]
        self.oldest_wait_s = engine.now_s - engine._enqueue_s[stage][head]

    def queue_len_at(self, stage: int) -> int:
        """Number of requests queued at any stage (not counting the running batch)."""
        return len(self._engine._queues[stage])

    def busy_remaining_s(self, stage: int) -> float:
        """Seconds until the server at ``stage`` is free; zero if it is idle."""
        return max(0.0, self._engine._busy_until_s[stage] - self.now_s)

    def batch_time_s(self, size: int) -> float:
        """Processing time this stage would need for the first ``size`` queued requests."""
        return self._engine._batch_time_s(self.stage, self._engine._head(self.stage, size))

    def arrivals_in_last(self, window_s: float) -> int:
        """External arrivals to the pipeline during the last ``window_s`` seconds."""
        engine = self._engine
        lo = int(np.searchsorted(engine._arrival_s, self.now_s - window_s, side="right"))
        return engine._next_arrival - min(lo, engine._next_arrival)


class Policy(Protocol):
    """Anything that can answer a decision point."""

    def decide(self, view: StageView) -> Decision: ...


@dataclass(frozen=True)
class SimulationResult:
    """Per-request timestamps and per-batch log of one simulation run.

    Arrays are indexed by request in arrival order. Timestamps a request has
    not reached yet are NaN. ``enqueue_s``, ``start_s`` and ``finish_s`` have
    shape ``(num_stages, num_requests)``.
    """

    arrival_s: NDArray[np.float64]
    enqueue_s: NDArray[np.float64]
    start_s: NDArray[np.float64]
    finish_s: NDArray[np.float64]
    batches: tuple[Batch, ...]
    end_s: float

    @property
    def num_stages(self) -> int:
        return self.finish_s.shape[0]

    @property
    def completion_s(self) -> NDArray[np.float64]:
        """Time each request left the last stage (NaN if it never did)."""
        return self.finish_s[-1]

    @property
    def latency_s(self) -> NDArray[np.float64]:
        """End-to-end latency: queueing plus processing, over all stages."""
        return self.completion_s - self.arrival_s


class Engine:
    """Event-driven simulator; see the module docstring for the model."""

    def __init__(
        self,
        arrival_s: Sequence[float] | NDArray[np.float64],
        context_tokens: Sequence[int] | NDArray[np.int64],
        stages: Sequence[ServiceTimeModel],
        stall_limit_s: float = 3600.0,
    ) -> None:
        self._arrival_s = np.asarray(arrival_s, dtype=np.float64)
        self._tokens = [int(t) for t in context_tokens]
        if len(self._tokens) != len(self._arrival_s):
            raise ValueError("arrival_s and context_tokens must have the same length")
        if len(self._arrival_s) and np.any(np.diff(self._arrival_s) < 0):
            raise ValueError("arrivals must be sorted by time")
        if not stages:
            raise ValueError("at least one stage is required")

        self._models = tuple(stages)
        self.num_stages = len(self._models)
        self._stall_limit_s = stall_limit_s
        n = len(self._arrival_s)

        self.now_s = 0.0
        self._next_arrival = 0
        self._queues: list[deque[int]] = [deque() for _ in self._models]
        self._in_service: list[list[int]] = [[] for _ in self._models]
        self._busy_until_s = [0.0] * self.num_stages
        self._needs_decision = [False] * self.num_stages
        self._wake_token = [0] * self.num_stages
        # Heap of (time, sequence number, kind, stage, wake token).
        self._events: list[tuple[float, int, int, int, int]] = []
        self._sequence = 0
        self._pending: int | None = None

        self._enqueue_s = np.full((self.num_stages, n), np.nan)
        self._start_s = np.full((self.num_stages, n), np.nan)
        self._finish_s = np.full((self.num_stages, n), np.nan)
        self._batches: list[Batch] = []
        self._completed = 0

    # ------------------------------------------------------------------ state

    @property
    def arrived(self) -> int:
        return self._next_arrival

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def queued(self) -> int:
        return sum(len(queue) for queue in self._queues)

    @property
    def in_service(self) -> int:
        return sum(len(batch) for batch in self._in_service)

    # ------------------------------------------------------------- driving it

    def next_decision(self) -> StageView | None:
        """Advance time to the next decision point.

        Returns ``None`` when every request has arrived and completed.
        """
        if self._pending is not None:
            raise RuntimeError("apply() must be called before asking for the next decision")
        while True:
            for stage in range(self.num_stages):
                if self._needs_decision[stage]:
                    self._pending = stage
                    return StageView(self, stage)
            if not self._advance():
                return None

    def apply(self, decision: Decision) -> None:
        """Carry out the answer to the decision point returned by ``next_decision``."""
        stage = self._pending
        if stage is None:
            raise RuntimeError("no decision is pending")
        self._pending = None
        self._needs_decision[stage] = False
        self._wake_token[stage] += 1  # any earlier wake-up timer is now stale

        if isinstance(decision, Dispatch):
            self._dispatch(stage, decision.size)
        elif isinstance(decision, Wait):
            self._wait(stage, decision.timeout_s)
        else:
            raise TypeError(f"unknown decision {decision!r}")

    def result(self) -> SimulationResult:
        return SimulationResult(
            arrival_s=self._arrival_s.copy(),
            enqueue_s=self._enqueue_s.copy(),
            start_s=self._start_s.copy(),
            finish_s=self._finish_s.copy(),
            batches=tuple(self._batches),
            end_s=self.now_s,
        )

    # -------------------------------------------------------------- internals

    def _head(self, stage: int, size: int) -> list[int]:
        queue = self._queues[stage]
        if not 1 <= size <= len(queue):
            raise ValueError(f"batch size {size} is not in 1..{len(queue)} (queue length)")
        return [queue[i] for i in range(size)]

    def _batch_time_s(self, stage: int, requests: Sequence[int]) -> float:
        return self._models[stage].batch_time_s([self._tokens[r] for r in requests])

    def _push(self, time_s: float, kind: int, stage: int, token: int = 0) -> None:
        heapq.heappush(self._events, (time_s, self._sequence, kind, stage, token))
        self._sequence += 1

    def _dispatch(self, stage: int, size: int) -> None:
        requests = self._head(stage, size)
        duration = self._batch_time_s(stage, requests)
        for _ in requests:
            self._queues[stage].popleft()
        self._in_service[stage] = requests
        self._start_s[stage, requests] = self.now_s
        self._busy_until_s[stage] = self.now_s + duration
        self._batches.append(Batch(stage, self.now_s, duration, size))
        self._push(self.now_s + duration, _COMPLETE, stage)

    def _wait(self, stage: int, timeout_s: float) -> None:
        if not (timeout_s > 0 and math.isfinite(timeout_s)):
            raise ValueError("Wait.timeout_s must be positive and finite")
        head = self._queues[stage][0]
        if self.now_s - self._enqueue_s[stage][head] > self._stall_limit_s:
            raise SimulationStalledError(
                f"stage {stage}: a request has been queued for more than {self._stall_limit_s} s"
            )
        # Guarantee the timer fires strictly later, even for a tiny timeout.
        wake_s = max(self.now_s + timeout_s, math.nextafter(self.now_s, math.inf))
        self._push(wake_s, _WAKE, stage, self._wake_token[stage])

    def _advance(self) -> bool:
        """Move to the next event time and process every event at that time."""
        has_arrival = self._next_arrival < len(self._arrival_s)
        next_arrival_s = self._arrival_s[self._next_arrival] if has_arrival else math.inf
        next_event_s = self._events[0][0] if self._events else math.inf
        time_s = min(next_arrival_s, next_event_s)
        if math.isinf(time_s):
            return False
        self.now_s = float(time_s)

        while self._events and self._events[0][0] == time_s:
            _, _, kind, stage, token = heapq.heappop(self._events)
            if kind == _COMPLETE:
                self._complete(stage)
            elif token == self._wake_token[stage]:
                self._request_decision(stage)
        while (
            self._next_arrival < len(self._arrival_s)
            and self._arrival_s[self._next_arrival] == time_s
        ):
            self._enqueue(0, self._next_arrival)
            self._next_arrival += 1
        return True

    def _request_decision(self, stage: int) -> None:
        if not self._in_service[stage] and self._queues[stage]:
            self._needs_decision[stage] = True

    def _enqueue(self, stage: int, request: int) -> None:
        self._queues[stage].append(request)
        self._enqueue_s[stage, request] = self.now_s
        self._request_decision(stage)

    def _complete(self, stage: int) -> None:
        requests = self._in_service[stage]
        self._in_service[stage] = []
        self._finish_s[stage, requests] = self.now_s
        if stage + 1 < self.num_stages:
            for request in requests:
                self._enqueue(stage + 1, request)
        else:
            self._completed += len(requests)
        self._request_decision(stage)


def simulate(
    arrival_s: Sequence[float] | NDArray[np.float64],
    context_tokens: Sequence[int] | NDArray[np.int64],
    stages: Sequence[ServiceTimeModel],
    policy: Policy | Sequence[Policy],
    stall_limit_s: float = 3600.0,
) -> SimulationResult:
    """Run a pipeline to completion with one policy, or one policy per stage."""
    engine = Engine(arrival_s, context_tokens, stages, stall_limit_s=stall_limit_s)
    if isinstance(policy, Sequence):
        policies = list(policy)
        if len(policies) != engine.num_stages:
            raise ValueError("need exactly one policy per stage")
    else:
        policies = [policy] * engine.num_stages
    while (view := engine.next_decision()) is not None:
        engine.apply(policies[view.stage].decide(view))
    return engine.result()
