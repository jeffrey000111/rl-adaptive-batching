"""Offered load: definition and time scaling of a trace to reach a target load.

Offered load is the arrival rate divided by the pipeline's capacity, where
capacity is the highest throughput the slowest stage can sustain: back-to-back
batches of the largest allowed size, every request at the reference length.

    load = arrival_rate / capacity

Load 1.0 is therefore the point beyond which no policy can keep the queue
bounded. Policies that use smaller batches saturate earlier.
"""

from __future__ import annotations

from collections.abc import Sequence

from adaptive_batching.service_time import ServiceTimeModel
from adaptive_batching.traces import Trace


def capacity_rps(stages: Sequence[ServiceTimeModel], max_batch: int) -> float:
    """Highest sustainable throughput of the pipeline (its slowest stage)."""
    return min(model.throughput_rps(max_batch, model.reference_length()) for model in stages)


def offered_load(trace: Trace, stages: Sequence[ServiceTimeModel], max_batch: int) -> float:
    """Mean arrival rate of ``trace`` divided by pipeline capacity."""
    return trace.mean_rate / capacity_rps(stages, max_batch)


def time_scale_for_load(
    reference: Trace, target_load: float, stages: Sequence[ServiceTimeModel], max_batch: int
) -> float:
    """Factor for ``Trace.time_scaled`` that gives ``reference`` the target load.

    The factor must be computed from the training split only and then applied
    unchanged to dev and test, so that no information about the evaluation
    data leaks into the workload definition. The load actually realised on
    dev and test differs when their arrival rate differs from training.
    """
    if target_load <= 0:
        raise ValueError("target_load must be positive")
    return offered_load(reference, stages, max_batch) / target_load
