"""Summary metrics of a simulation run."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from adaptive_batching.engine import SimulationResult


@dataclass(frozen=True)
class Metrics:
    """All times in seconds. Latency is end-to-end: queueing plus processing."""

    requests: int
    completed: int
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    # Share of requests slower than the SLO; None if no SLO was given.
    slo_violation_rate: float | None
    # Completed requests per second between the first arrival and the last completion.
    throughput_rps: float
    # Seconds each stage's server spent processing batches.
    busy_time_s: tuple[float, ...]
    # busy_time_s divided by the same span as throughput.
    utilization: tuple[float, ...]
    mean_batch_size: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize(result: SimulationResult, slo_s: float | None = None) -> Metrics:
    """Compute metrics over every request of a run that ran to completion."""
    latency = result.latency_s
    done = ~np.isnan(latency)
    completed = int(done.sum())
    if completed == 0:
        raise ValueError("no request completed")
    finished = latency[done]
    p50, p95, p99 = np.percentile(finished, [50, 95, 99])

    span = float(result.completion_s[done].max() - result.arrival_s[0])
    busy = [0.0] * result.num_stages
    sizes: list[list[int]] = [[] for _ in range(result.num_stages)]
    for batch in result.batches:
        busy[batch.stage] += batch.duration_s
        sizes[batch.stage].append(batch.size)

    violation_rate = None
    if slo_s is not None:
        # A request that never completed has also missed its SLO.
        violation_rate = float(((finished > slo_s).sum() + (~done).sum()) / len(latency))

    return Metrics(
        requests=len(latency),
        completed=completed,
        mean_latency_s=float(finished.mean()),
        p50_latency_s=float(p50),
        p95_latency_s=float(p95),
        p99_latency_s=float(p99),
        slo_violation_rate=violation_rate,
        throughput_rps=completed / span,
        busy_time_s=tuple(busy),
        utilization=tuple(b / span for b in busy),
        mean_batch_size=tuple(float(np.mean(s)) if s else 0.0 for s in sizes),
    )
