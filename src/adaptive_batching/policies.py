"""Rule-based batching policies."""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class ThresholdBatch:
    """Serve everything queued (up to ``max_batch``) once ``threshold`` requests wait.

    Below the threshold the server idles on purpose, but no request waits
    longer than ``max_wait_s`` in the queue. ``threshold=1`` is
    ``GreedyBatch(max_batch)``. Unlike :class:`WaitForBatch`, the batch is
    not limited to the threshold: a backlog is cleared in large batches.
    """

    threshold: int
    max_wait_s: float
    max_batch: int

    def decide(self, view: StageView) -> Decision:
        if view.queue_len >= self.threshold:
            return Dispatch(min(view.queue_len, self.max_batch))
        remaining = self.max_wait_s - view.oldest_wait_s
        if remaining <= _EPS_S:
            return Dispatch(min(view.queue_len, self.max_batch))
        return Wait(remaining)


@dataclass(frozen=True)
class DownstreamAwareBatch:
    """Threshold rule that also holds back while the next stage is backed up.

    If the next stage is busy and already has ``hold_at`` or more requests
    queued, sending it more work cannot make those requests finish sooner,
    so this stage keeps collecting a larger (cheaper per request) batch and
    looks again when the next stage finishes its current batch. The
    ``max_wait_s`` limit still applies. At the last stage, or with
    ``hold_at = inf``, this is exactly :class:`ThresholdBatch`.
    """

    threshold: int
    max_wait_s: float
    max_batch: int
    hold_at: float = math.inf

    def decide(self, view: StageView) -> Decision:
        remaining = self.max_wait_s - view.oldest_wait_s
        size = min(view.queue_len, self.max_batch)
        if remaining <= _EPS_S or view.queue_len >= self.max_batch:
            return Dispatch(size)

        nxt = view.stage + 1
        if nxt < view.num_stages and view.queue_len_at(nxt) >= self.hold_at:
            downstream_busy_s = view.busy_remaining_s(nxt)
            if downstream_busy_s > 0:
                return Wait(min(remaining, downstream_busy_s))
        if view.queue_len >= self.threshold:
            return Dispatch(size)
        return Wait(remaining)
