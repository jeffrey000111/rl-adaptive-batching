"""Rule-based batching policies."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_batching.engine import Decision, Dispatch, StageView, Wait

# Slack for comparing accumulated floating-point waiting times with a limit.
_EPS_S = 1e-9


@dataclass(frozen=True)
class GreedyBatch:
    """Never wait: whenever the server is free, send up to ``max_batch`` queued requests.

    ``GreedyBatch(1)`` is "no batching"; a large ``max_batch`` lets the batch
    size follow the backlog.
    """

    max_batch: int

    def decide(self, view: StageView) -> Decision:
        return Dispatch(min(self.max_batch, view.queue_len))


@dataclass(frozen=True)
class WaitForBatch:
    """Wait for a full batch, but no longer than ``max_wait_s``.

    Dispatches exactly ``batch_size`` requests once that many are queued. If
    the oldest queued request has waited ``max_wait_s``, dispatches whatever
    is queued instead. This is the delayed-batching rule of inference servers
    such as Triton (preferred batch size + maximum queue delay).
    """

    batch_size: int
    max_wait_s: float

    def decide(self, view: StageView) -> Decision:
        if view.queue_len >= self.batch_size:
            return Dispatch(self.batch_size)
        remaining = self.max_wait_s - view.oldest_wait_s
        if remaining <= _EPS_S:
            return Dispatch(view.queue_len)
        return Wait(remaining)
