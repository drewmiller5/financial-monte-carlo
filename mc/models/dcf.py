from __future__ import annotations
import numpy as np
from mc.distributions import Triangular, Normal
from mc.engine import MonteCarloEngine, SimulationResult


class DCFModel:
    """
    Monte Carlo discounted cash flow valuation.

    Each assumption is a distribution, not a point estimate.
    Outputs a range of enterprise valuations (p10/p50/p90) in $M.

    Simplification: operating_margin is used as an FCF margin proxy
    (revenue × margin = FCF). This is appropriate for asset-light or
    services businesses. Capital-intensive businesses should adjust for
    capex before passing operating_margin.
    """

    def __init__(
        self,
        revenue: float,
        revenue_growth: tuple[float, float, float],
        operating_margin: tuple[float, float, float],
        discount_rate: tuple[float, float],
        terminal_growth: tuple[float, float],
        years: int = 10,
        n: int = 10_000,
        seed: int | None = None,
    ):
        if revenue <= 0:
            raise ValueError(f"revenue must be positive, got {revenue}")
        self.revenue = revenue
        self.revenue_growth = Triangular(*revenue_growth)
        self.operating_margin = Triangular(*operating_margin)
        self.discount_rate = Normal(*discount_rate)
        self.terminal_growth = Normal(*terminal_growth)
        self.years = years
        self.engine = MonteCarloEngine(n=n, seed=seed)

    def run(self) -> dict:
        rng = self.engine.rng
        n = self.engine.n

        g_samples  = self.revenue_growth.sample(rng, n).clip(0.0, 0.80)
        m_samples  = self.operating_margin.sample(rng, n).clip(0.0, 0.60)
        r_samples  = self.discount_rate.sample(rng, n).clip(0.04, 0.35)
        tg_samples = self.terminal_growth.sample(rng, n).clip(0.005, 0.05)

        # Ensure terminal growth < discount rate to avoid infinite terminal value
        tg_samples = np.minimum(tg_samples, r_samples - 0.01)

        # Vectorized: broadcast over (years, n) simultaneously
        yr = np.arange(1, self.years + 1)[:, None]              # (years, 1)
        rev_at_yr = self.revenue * (1.0 + g_samples) ** yr      # (years, n)
        fcf = rev_at_yr * m_samples                              # (years, n)
        discount = (1.0 + r_samples) ** yr                       # (years, n)
        pv = (fcf / discount).sum(axis=0)                        # (n,)

        rev_final = rev_at_yr[-1]                                # (n,)
        terminal = (rev_final * m_samples * (1.0 + tg_samples)) / (r_samples - tg_samples)
        pv += terminal / (1.0 + r_samples) ** self.years

        result = SimulationResult(values=pv, n=n)
        lo, hi = result.confidence_interval(10, 90)

        return {
            "p10":   round(result.p10, 1),
            "p50":   round(result.p50, 1),
            "p90":   round(result.p90, 1),
            "mean":  round(result.mean, 1),
            "std":   round(result.std, 1),
            "ci_90": (round(lo, 1), round(hi, 1)),
            "unit":  "$M",
        }
