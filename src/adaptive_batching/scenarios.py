"""Named experiment configurations: pipelines, load levels and latency SLOs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from adaptive_batching.service_time import (
    BATCH_SIZES,
    BERT_BASE_T4_FP16,
    BERT_LARGE_T4_FP16,
    ServiceTimeModel,
)

MAX_BATCH = BATCH_SIZES[-1]

# Nominal offered-load levels (see adaptive_batching.load for the definition).
LOADS: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    stages: tuple[ServiceTimeModel, ...]

    @property
    def slo_s(self) -> float:
        """Primary latency SLO; see :func:`primary_slo_s`."""
        return primary_slo_s(self.stages)

    @property
    def tight_slo_s(self) -> float:
        """Tight latency SLO used as an ablation; see :func:`tight_slo_s`."""
        return tight_slo_s(self.stages)


def primary_slo_s(stages: Sequence[ServiceTimeModel]) -> float:
    """Twice the processing time of the largest batch, summed over stages.

    Rule from Nexus (Shen et al., SOSP 2019, Section 4.1): a batch may take
    at most half of the SLO, because a request that just misses a batch
    waits for that batch and then for its own. Under this SLO every allowed
    batch size is feasible in principle.
    """
    return sum(2 * m.uniform_batch_time_s(MAX_BATCH, m.reference_length()) for m in stages)


def tight_slo_s(stages: Sequence[ServiceTimeModel]) -> float:
    """Ten times the processing time of a single request, summed over stages.

    Under this SLO the largest batch alone already takes longer than the SLO.
    """
    return sum(10 * m.uniform_batch_time_s(1, m.reference_length()) for m in stages)


SINGLE_STAGE = Scenario(
    name="single",
    description="one BERT-base server",
    stages=(BERT_BASE_T4_FP16,),
)

# An illustrative pipeline (encode -> re-rank -> classify) assembled from two
# published per-model profiles. It is not a measurement of any real system.
# The middle stage is the bottleneck.
PIPELINE_3STAGE = Scenario(
    name="pipeline",
    description="illustrative 3-stage pipeline: BERT-base, BERT-large, BERT-base",
    stages=(BERT_BASE_T4_FP16, BERT_LARGE_T4_FP16, BERT_BASE_T4_FP16),
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (SINGLE_STAGE, PIPELINE_3STAGE)}
