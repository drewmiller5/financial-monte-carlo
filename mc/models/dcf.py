from __future__ import annotations
import numpy as np
from mc.distributions import Triangular, Normal
from mc.engine import MonteCarloEngine, SimulationResult


class DCFModel:
    """
    Monte Carlo discounted cash flow valuation.

    Each assumption is a distribution, not a point estimate.
    Outputs a range of enterprise valuations (p10/p50/p90) in $M.

    Single-stage (default): one revenue growth rate applied uniformly across all years.

    Two-stage: pass both ``revenue_growth_phase2`` and ``phase1_years`` to model a
    high-growth phase followed by a mature-growth phase. Neither or both must be set.

    CapEx: pass ``capex_pct_revenue`` (Triangular low/mode/high as a fraction of
    revenue) to deduct CapEx from FCF each year. When None, operating_margin is
    treated as a direct FCF margin proxy (appropriate for asset-light businesses).

    Net debt: pass ``net_debt`` (scalar, $M) to add equity bridge to the output.
    equity_value = enterprise_value − net_debt. A negative net_debt means net cash.

    Two-stage margins: pass ``operating_margin_phase2`` to model margin expansion from
    phase 1 (e.g. heavy R&D, compressed margins) to phase 2 (mature, higher margins).
    When None, ``operating_margin`` applies to both phases. Requires ``phase1_years``.

    Simplification: operating_margin × revenue = FCF (minus capex when provided).
    For full FCFF (D&A, NWC, tax), adjust operating_margin before passing.
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
        revenue_growth_phase2: tuple[float, float, float] | None = None,
        phase1_years: int | None = None,
        capex_pct_revenue: tuple[float, float, float] | None = None,
        net_debt: float = 0.0,
        operating_margin_phase2: tuple[float, float, float] | None = None,
    ):
        if revenue <= 0:
            raise ValueError(f"revenue must be positive, got {revenue}")
        if (phase1_years is None) != (revenue_growth_phase2 is None):
            raise ValueError(
                "phase1_years and revenue_growth_phase2 must both be set, or neither"
            )
        if phase1_years is not None and not (0 < phase1_years < years):
            raise ValueError(
                f"phase1_years must be in (0, {years}), got {phase1_years}"
            )
        if operating_margin_phase2 is not None and phase1_years is None:
            raise ValueError(
                "operating_margin_phase2 requires phase1_years and revenue_growth_phase2 to be set"
            )

        self.revenue = revenue
        self.revenue_growth = Triangular(*revenue_growth)
        self.operating_margin = Triangular(*operating_margin)
        self.discount_rate = Normal(*discount_rate)
        self.terminal_growth = Normal(*terminal_growth)
        self.years = years
        self.phase1_years = phase1_years
        self.revenue_growth_phase2 = (
            Triangular(*revenue_growth_phase2) if revenue_growth_phase2 is not None else None
        )
        self.operating_margin_phase2 = (
            Triangular(*operating_margin_phase2) if operating_margin_phase2 is not None else None
        )
        self.capex_pct_revenue = (
            Triangular(*capex_pct_revenue) if capex_pct_revenue is not None else None
        )
        self.net_debt = net_debt
        self.engine = MonteCarloEngine(n=n, seed=seed)

    def run(self) -> dict:
        rng = self.engine.rng
        n = self.engine.n

        m_samples  = self.operating_margin.sample(rng, n).clip(0.0, 0.60)
        r_samples  = self.discount_rate.sample(rng, n).clip(0.04, 0.35)
        tg_samples = self.terminal_growth.sample(rng, n).clip(0.005, 0.05)

        # Ensure terminal growth < discount rate to avoid infinite terminal value
        tg_samples = np.minimum(tg_samples, r_samples - 0.01)

        capex_samples = (
            self.capex_pct_revenue.sample(rng, n).clip(0.0, 1.0)
            if self.capex_pct_revenue is not None else None
        )

        # FCF margin: operating margin minus capex fraction (when provided)
        fcf_margin = m_samples - capex_samples if capex_samples is not None else m_samples

        if self.phase1_years is not None:
            pv, rev_final = self._run_two_stage(rng, n, r_samples, fcf_margin, capex_samples)
        else:
            pv, rev_final = self._run_single_stage(rng, n, r_samples, fcf_margin)

        terminal = (rev_final * fcf_margin * (1.0 + tg_samples)) / (r_samples - tg_samples)
        pv += terminal / (1.0 + r_samples) ** self.years

        result = SimulationResult(values=pv, n=n)
        lo, hi = result.confidence_interval(10, 90)

        out = {
            "p10":    round(result.p10, 1),
            "p50":    round(result.p50, 1),
            "p90":    round(result.p90, 1),
            "mean":   round(result.mean, 1),
            "std":    round(result.std, 1),
            "ci_90":  (round(lo, 1), round(hi, 1)),
            "unit":   "$M",
            "values": result.values,
        }

        if self.net_debt != 0.0:
            out["equity_p10"] = round(result.p10 - self.net_debt, 1)
            out["equity_p50"] = round(result.p50 - self.net_debt, 1)
            out["equity_p90"] = round(result.p90 - self.net_debt, 1)
            out["net_debt"]   = self.net_debt

        return out

    def _run_single_stage(
        self, rng: np.random.Generator, n: int,
        r_samples: np.ndarray, fcf_margin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        g_samples = self.revenue_growth.sample(rng, n).clip(0.0, 0.80)
        yr = np.arange(1, self.years + 1)[:, None]              # (years, 1)
        rev_at_yr = self.revenue * (1.0 + g_samples) ** yr      # (years, n)
        fcf = rev_at_yr * fcf_margin                             # (years, n)
        discount = (1.0 + r_samples) ** yr                       # (years, n)
        pv = (fcf / discount).sum(axis=0)                        # (n,)
        return pv, rev_at_yr[-1]

    def _run_two_stage(
        self, rng: np.random.Generator, n: int,
        r_samples: np.ndarray, fcf_margin: np.ndarray,
        capex_samples: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        p1 = self.phase1_years
        p2 = self.years - p1

        g1 = self.revenue_growth.sample(rng, n).clip(0.0, 0.80)
        g2 = self.revenue_growth_phase2.sample(rng, n).clip(0.0, 0.80)

        yr1 = np.arange(1, p1 + 1)[:, None]                        # (p1, 1)
        yr2 = np.arange(1, p2 + 1)[:, None]                        # (p2, 1)

        # Phase-specific FCF margins: if operating_margin_phase2 is set, use it for phase 2.
        if self.operating_margin_phase2 is not None:
            m2 = self.operating_margin_phase2.sample(rng, n).clip(0.0, 0.60)
            fcf_margin_p2 = m2 - capex_samples if capex_samples is not None else m2
        else:
            fcf_margin_p2 = fcf_margin

        # Phase 1: years 1 through p1
        rev_p1 = self.revenue * (1.0 + g1) ** yr1                  # (p1, n)
        fcf_p1 = rev_p1 * fcf_margin                                # (p1, n)
        disc_p1 = (1.0 + r_samples) ** yr1                          # (p1, n)

        # Phase 2: years p1+1 through years; discount exponent offsets by p1
        rev_p2 = rev_p1[-1] * (1.0 + g2) ** yr2                    # (p2, n)
        fcf_p2 = rev_p2 * fcf_margin_p2                             # (p2, n)
        disc_p2 = (1.0 + r_samples) ** (yr2 + p1)                  # (p2, n)

        pv = (fcf_p1 / disc_p1).sum(axis=0) + (fcf_p2 / disc_p2).sum(axis=0)
        return pv, rev_p2[-1]
