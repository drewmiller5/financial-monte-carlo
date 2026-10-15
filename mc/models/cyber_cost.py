from __future__ import annotations
import math
import numpy as np
from mc.engine import MonteCarloEngine, SimulationResult

CALIBRATION_YEAR = 2025  # IBM Cost of a Data Breach 2025 (Ponemon Institute)
# Update _BASE_MU, _BASE_SIGMA, and SECTOR_LOG_ADJ when a new IBM annual report is published.

# Base calibration: IBM Cost of a Data Breach 2025 — Healthcare sector reference.
# Mean breach cost $7.42M (12th consecutive year as highest-cost industry).
# Log-normal parameters: mu=1.414, sigma=1.086 (CV ~1.5 — right-skewed, typical for breach costs).
# mu derived from: ln(7.42) - (1.086^2 / 2) = 2.004 - 0.590 = 1.414
_BASE_MU = 1.414
_BASE_SIGMA = 1.086

# Sector log-scale adjustments (additive to mu; each +1 = ~2.7x median cost).
# Derived from IBM 2025 sector mean costs relative to Healthcare $7.42M baseline.
# Formula: ln(sector_cost / 7.42)
SECTOR_LOG_ADJ: dict[str, float] = {
    "Healthcare":              0.00,   # Reference: $7.42M IBM 2025
    "Financial Services":     -0.29,   # IBM 2025: $5.56M
    "Finance":                -0.29,
    "Industrials":            -0.39,   # IBM 2025: $5.00M (was -1.29 — major shift)
    "Energy":                 -0.43,   # IBM 2025: $4.83M (was -1.11)
    "Technology":             -0.44,   # IBM 2025: $4.79M (was -0.85)
    "Pharmaceuticals":        -0.48,   # IBM 2025: $4.61M
    "Communication Services": -0.68,   # IBM 2025: $3.75M (was -1.05)
    "Consumer Discretionary": -0.69,   # IBM 2025: Consumer $3.72M proxy
    "Consumer Staples":       -0.74,   # IBM 2025: Retail $3.54M proxy
    "Retail":                 -0.74,   # IBM 2025: $3.54M
    "Transportation":         -0.62,   # IBM 2025: $3.98M
    "Real Estate":            -0.80,   # No direct IBM entry; interpolated
    "Materials":              -0.85,   # No direct IBM entry; interpolated
    "Utilities":              -0.90,   # IBM Energy definition includes utilities
}
DEFAULT_SECTOR_LOG_ADJ = -0.55   # IBM 2025 global avg $4.44M vs healthcare $7.42M


def _size_log_adj(market_cap_m: float) -> float:
    """Log-scale cost multiplier by company size tier.

    Larger companies have more records at risk, more complex remediation,
    higher regulatory penalties, and greater business interruption exposure.
    Calibrated to observed mega-breach outcomes:
      Equifax (2017): $1.4B total; Anthem (2015): $115M+ settlement;
      Change Healthcare (2024): $3.09B FY2024 total loss (UNH, Jan 2025).
    """
    if market_cap_m >= 200_000:    # Mega-cap ($200B+)
        return math.log(250)
    if market_cap_m >= 50_000:     # Large-cap ($50B–$200B)
        return math.log(60)
    if market_cap_m >= 10_000:     # Mid-large ($10B–$50B)
        return math.log(15)
    if market_cap_m >= 1_000:      # Mid-cap ($1B–$10B)
        return math.log(5)
    return 0.0                      # Small-cap (<$1B): base IBM cost


class CyberCostModel:
    """
    Monte Carlo cyber breach cost estimator (dollar loss in $M).

    Answers: "If this company suffers a breach, what does it cost?"

    Calibrated to IBM Cost of a Data Breach 2025. Costs follow a log-normal
    distribution: right-skewed because most breaches are moderate in cost
    but a small fraction are catastrophic. Outputs p10/p50/p90 in $M.

    Expected Annual Loss (EAL) = p50 cost * P(breach from BreachLikelihoodModel / 100).
    EAL is the standard input for cyber insurance premium pricing.
    """

    def __init__(
        self,
        market_cap_m: float,
        sector: str = "Healthcare",
        breach_score: float | None = None,
        n: int = 10_000,
        seed: int | None = None,
    ):
        if market_cap_m <= 0:
            raise ValueError(f"market_cap_m must be positive, got {market_cap_m}")
        if breach_score is not None and not (0.0 <= breach_score <= 100.0):
            raise ValueError(f"breach_score must be in [0, 100], got {breach_score}")
        self.market_cap_m = market_cap_m
        self.sector = sector
        self.breach_score = breach_score
        self.engine = MonteCarloEngine(n=n, seed=seed)

    def run(self) -> dict:
        rng = self.engine.rng
        n = self.engine.n

        sector_adj = SECTOR_LOG_ADJ.get(self.sector, DEFAULT_SECTOR_LOG_ADJ)
        size_adj = _size_log_adj(self.market_cap_m)
        mu = _BASE_MU + sector_adj + size_adj

        # High-risk companies face amplified costs: more regulatory exposure,
        # more records at risk, greater reputational damage, longer remediation.
        if self.breach_score is not None:
            if self.breach_score >= 70:
                mu += math.log(1.5)   # +50% median cost
            elif self.breach_score <= 30:
                mu += math.log(0.6)   # -40% for lower-risk, smaller-scope breaches

        cost_samples = rng.lognormal(mu, _BASE_SIGMA, n)
        result = SimulationResult(values=cost_samples, n=n)
        lo, hi = result.confidence_interval(10, 90)

        return {
            "p10":    round(result.p10, 1),
            "p50":    round(result.p50, 1),
            "p90":    round(result.p90, 1),
            "mean":   round(result.mean, 1),
            "std":    round(result.std, 1),
            "ci_90":  (round(lo, 1), round(hi, 1)),
            "unit":   "$M",
            "scores": result.values,
        }
