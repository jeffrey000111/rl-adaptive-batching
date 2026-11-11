"""Request arrival traces: loading, time-based splitting, slicing and load scaling.

A :class:`Trace` is an immutable view of requests inside a half-open time
window ``[start_s, end_s)``. The window is stored explicitly instead of being
inferred from the first and last request, because idle time at either end of
a window is part of the workload: a ten-minute window with one burst in the
middle is not the same workload as the burst alone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SPLIT_NAMES = ("train", "dev", "test")


@dataclass(frozen=True)
class Trace:
    """Requests that arrive inside ``[start_s, end_s)`` on the trace clock.

    ``arrival_s`` is in seconds on the clock of the file the trace was loaded
    from (zero = first request in the file), so windows cut from the same
    file stay comparable. Use :meth:`relative_arrivals` to get times measured
    from the start of the window.
    """

    name: str
    arrival_s: NDArray[np.float64]
    context_tokens: NDArray[np.int64]
    generated_tokens: NDArray[np.int64]
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        n = len(self.arrival_s)
        if len(self.context_tokens) != n or len(self.generated_tokens) != n:
            raise ValueError("arrival and token arrays must have the same length")
        if not self.end_s > self.start_s:
            raise ValueError("window must have positive duration")
        if n and np.any(np.diff(self.arrival_s) < 0):
            raise ValueError("arrivals must be sorted by time")
        if n and not (self.start_s <= self.arrival_s[0] and self.arrival_s[-1] < self.end_s):
            raise ValueError("arrivals must lie inside [start_s, end_s)")
        for array in (self.arrival_s, self.context_tokens, self.generated_tokens):
            array.setflags(write=False)

    def __len__(self) -> int:
        return len(self.arrival_s)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def mean_rate(self) -> float:
        """Requests per second averaged over the whole window, idle time included."""
        return len(self) / self.duration_s

    def relative_arrivals(self) -> NDArray[np.float64]:
        """Arrival times in seconds since the start of the window."""
        return self.arrival_s - self.start_s

    def window(self, start_s: float, end_s: float, name: str | None = None) -> Trace:
        """Requests with ``start_s <= arrival < end_s``; bounds must lie inside this window."""
        if start_s < self.start_s or end_s > self.end_s:
            raise ValueError("sub-window must lie inside the parent window")
        lo, hi = np.searchsorted(self.arrival_s, [start_s, end_s], side="left")
        return Trace(
            name=name or self.name,
            arrival_s=self.arrival_s[lo:hi].copy(),
            context_tokens=self.context_tokens[lo:hi].copy(),
            generated_tokens=self.generated_tokens[lo:hi].copy(),
            start_s=start_s,
            end_s=end_s,
        )

    def split(self, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, Trace]:
        """Cut the window into consecutive train / dev / test windows by time.

        The cut is by time, not by request count, so the three parts are
        contiguous in time and no request is in two parts.
        """
        if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("fractions must be three positive numbers summing to 1")
        cut1 = self.start_s + fractions[0] * self.duration_s
        cut2 = self.start_s + (fractions[0] + fractions[1]) * self.duration_s
        bounds = [self.start_s, cut1, cut2, self.end_s]
        return {
            split: self.window(bounds[i], bounds[i + 1], name=f"{self.name}/{split}")
            for i, split in enumerate(SPLIT_NAMES)
        }

    def slices(self, length_s: float) -> list[Trace]:
        """Consecutive non-overlapping windows of ``length_s`` seconds.

        A final remainder shorter than ``length_s`` is dropped so that every
        slice covers the same amount of time. Slices with no requests are
        kept: an idle slice is a real observation of the workload.
        """
        if length_s <= 0:
            raise ValueError("length_s must be positive")
        count = int(self.duration_s // length_s)
        return [
            self.window(
                self.start_s + i * length_s,
                self.start_s + (i + 1) * length_s,
                name=f"{self.name}[{i}]",
            )
            for i in range(count)
        ]

    def time_scaled(self, factor: float) -> Trace:
        """Multiply every timestamp by ``factor``.

        ``factor < 1`` compresses time and raises the arrival rate by
        ``1 / factor`` while keeping the order, the relative burst structure
        and each request's token counts unchanged.
        """
        if factor <= 0:
            raise ValueError("factor must be positive")
        start = self.start_s * factor
        arrivals = self.arrival_s * factor
        # Guard the half-open window against floating-point rounding.
        end = max(self.end_s * factor, np.nextafter(arrivals[-1], np.inf) if len(self) else 0.0)
        return Trace(
            name=self.name,
            arrival_s=np.maximum(arrivals, start),
            context_tokens=self.context_tokens.copy(),
            generated_tokens=self.generated_tokens.copy(),
            start_s=start,
            end_s=float(end),
        )


def load_trace(path: str | Path, name: str | None = None) -> Trace:
    """Load an Azure LLM inference trace CSV (TIMESTAMP, ContextTokens, GeneratedTokens).

    The window runs from the first request to just after the last one.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]:
            raise ValueError(f"unexpected header in {path.name}: {header}")
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"{path.name} contains no requests")

    stamps = np.array([row[0].replace(" ", "T") for row in rows], dtype="datetime64[ns]")
    arrival_s = (stamps - stamps[0]) / np.timedelta64(1, "s")
    return Trace(
        name=name or path.stem,
        arrival_s=arrival_s.astype(np.float64),
        context_tokens=np.array([int(row[1]) for row in rows], dtype=np.int64),
        generated_tokens=np.array([int(row[2]) for row in rows], dtype=np.int64),
        start_s=0.0,
        end_s=float(np.nextafter(arrival_s[-1], np.inf)),
    )
