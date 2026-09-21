"""Lightweight timing utilities for latency measurement."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timer:
    """Collects named wall-clock durations in milliseconds."""

    timings_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (
                (time.perf_counter() - start) * 1000.0
            )

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())
