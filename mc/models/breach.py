from __future__ import annotations
import math
import numpy as np
from mc.distributions import Triangular, Normal, Beta
from mc.engine import MonteCarloEngine, SimulationResult

# Three-point sector risk (low, mode, high) — calibrated to IBM/Ponemon + Verizon DBIR
SECTOR_RISK: dict[str, tuple[float, float, float]] = {
    "Healthcare":            (0.72, 0.87, 0.96),
    "Financial Services":    (0.68, 0.83, 0.94),
    "Finance":               (0.68, 0.83, 0.94),
    "Technology":            (0.52, 0.70, 0.84),
    "Communication Services":(0.50, 0.67, 0.80),
    "Consumer Discretionary":(0.44, 0.60, 0.74),
    "Consumer Staples":      (0.40, 0.55, 0.68),
    "Real Estate":           (0.38, 0.55, 0.68),
    "Energy":                (0.36, 0.52, 0.66),
    "Industrials":           (0.34, 0.50, 0.64),
    "Materials":             (0.30, 0.47, 0.62),
    "Utilities":             (0.28, 0.44, 0.58),
}
DEFAULT_SECTOR_RISK = (0.25, 0.50, 0.70)

WEIGHTS = {
    "sector": 0.30,
    "size":   0.20,
    "peers":  0.25,
    "growth": 0.15,
    "beta":   0.10,
}


class BreachLikelihoodModel:
    """
    Monte Carlo breach likelihood estimator.

    Replaces the deterministic heuristic in BreachIQ likelihood.py.
    Each input factor is sampled from a probability distribution across
    N simulations, producing a score distribution instead of a point estimate.

    Returns p10/p50/p90, confidence interval, and probability of each tier.
    """

    def __init__(
        self,
        company_info: dict,
        industry_rows: list,
        n: int = 10_000,
        seed: int | None = None,
    ):
        self.company_info = company_info
        self.industry_rows = industry_rows
        self.engine = MonteCarloEngine(n=n, seed=seed)

    def run(self) -> dict:
        rng = self.engine.rng
        n = self.engine.n

        sector = self.company_info.get("sector") or ""
        market_cap_m = self.company_info.get("market_cap_m") or 0
        market_beta = float(self.company_info.get("beta") or 1.0)
        rev_growth = float(self.company_info.get("rev_growth") or 0.0)
        peer_tickers = len({r["ticker"] for r in self.industry_rows}) if self.industry_rows else 0

        # Factor 1: Sector base risk — triangular distribution
        low, mode, high = SECTOR_RISK.get(sector, DEFAULT_SECTOR_RISK)
        f_sector = Triangular(low, mode, high).sample(rng, n)

        # Factor 2: Company size — normal uncertainty around log-scale point estimate
        if market_cap_m > 0:
            size_point = min(math.log10(max(market_cap_m, 1)) / 6.0, 1.0)
        else:
            size_point = 0.25
        f_size = Normal(size_point, 0.06).sample(rng, n).clip(0.0, 1.0)

        # Factor 3: Peer breach rate — beta distribution (probability interpretation)
        peer_mode = min(peer_tickers / 5.0, 1.0)
        alpha = max(peer_mode * 5.0, 0.5)
        beta_p = max((1.0 - peer_mode) * 5.0, 0.5)
        f_peers = Beta(alpha, beta_p).sample(rng, n)

        # Factor 4: Revenue growth — triangular with ±5% uncertainty band
        g_low = max(rev_growth - 0.05, 0.0) / 0.30
        g_mode = min(rev_growth / 0.30, 1.0)
        g_high = min((rev_growth + 0.05) / 0.30, 1.0)
        if g_low > g_mode:
            g_low = g_mode
        if g_mode > g_high:
            g_high = g_mode
        # Degenerate case: all three collapse to the same value (e.g., very high growth >= 35%)
        if g_low >= g_high:
            f_growth = np.full(n, g_mode)
        else:
            f_growth = Triangular(g_low, g_mode, g_high).sample(rng, n).clip(0.0, 1.0)

        # Factor 5: Market beta — normal uncertainty around point estimate
        beta_point = min(max(market_beta - 0.5, 0.0) / 1.5, 1.0)
        f_beta = Normal(beta_point, 0.08).sample(rng, n).clip(0.0, 1.0)

        raw = (
            f_sector * WEIGHTS["sector"]
            + f_size   * WEIGHTS["size"]
            + f_peers  * WEIGHTS["peers"]
            + f_growth * WEIGHTS["growth"]
            + f_beta   * WEIGHTS["beta"]
        )
        scores = (raw * 100.0).clip(0.0, 100.0)
        result = SimulationResult(values=scores, n=n)

        p50 = result.p50
        lo, hi = result.confidence_interval(10, 90)
        prob_critical_pct = round(result.probability_above(70) * 100, 1)

        return {
            "p10": round(result.p10),
            "p50": round(p50),
            "p90": round(result.p90),
            "mean": round(result.mean),
            "std": round(result.std, 1),
            "ci_90": (round(lo), round(hi)),
            "tier": _tier(p50),
            "prob_critical_pct": prob_critical_pct,
            "narrative": _narrative(self.company_info, result, peer_tickers, prob_critical_pct),
        }


def _tier(score: float) -> str:
    if score >= 70:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Moderate"
    return "Low"


def _narrative(info: dict, result: SimulationResult, peer_tickers: int, prob_crit: float) -> str:
    name = info.get("name", "This company")
    p50 = round(result.p50)
    p10 = round(result.p10)
    p90 = round(result.p90)
    tier = _tier(result.p50)

    spread = "Wide confidence interval — additional data would reduce uncertainty." if result.std > 14 else ""
    peer_note = f"{peer_tickers} sector peers have disclosed material incidents." if peer_tickers > 0 else "No peer breach disclosures found in this sector."

    return (
        f"{name} median likelihood score: {p50}/100 ({tier} tier). "
        f"90% CI: {p10}–{p90}. "
        f"Probability of Critical tier (70+): {prob_crit}%. "
        f"{peer_note} "
        f"{spread}"
    ).strip()
