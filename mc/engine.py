from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class SimulationResult:
    values: np.ndarray
    n: int

    @property
    def p10(self) -> float:
        return float(np.percentile(self.values, 10))

    @property
    def p50(self) -> float:
        return float(np.percentile(self.values, 50))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.values, 90))

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def std(self) -> float:
        return float(self.values.std())

    def confidence_interval(self, lower: float = 5.0, upper: float = 95.0) -> tuple[float, float]:
        return (
            float(np.percentile(self.values, lower)),
            float(np.percentile(self.values, upper)),
        )

    def probability_above(self, threshold: float) -> float:
        return float((self.values > threshold).mean())

    def probability_below(self, threshold: float) -> float:
        return float((self.values < threshold).mean())

    def summary(self) -> dict:
        lo, hi = self.confidence_interval()
        return {
            "n": self.n,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "p10": round(self.p10, 2),
            "p50": round(self.p50, 2),
            "p90": round(self.p90, 2),
            "ci_90": (round(lo, 2), round(hi, 2)),
        }


class MonteCarloEngine:
    def __init__(self, n: int = 10_000, seed: int | None = None):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def run(self, model_fn: Callable[[], float]) -> SimulationResult:
        """Run N simulations of model_fn() → returns distribution of outcomes."""
        values = np.array([model_fn() for _ in range(self.n)])
        return SimulationResult(values=values, n=self.n)

    def run_vectorized(self, model_fn: Callable[[int], np.ndarray]) -> SimulationResult:
        """Run vectorized simulation: model_fn receives n, returns ndarray of shape (n,)."""
        values = model_fn(self.n)
        return SimulationResult(values=values, n=self.n)
