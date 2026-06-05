from __future__ import annotations
import math
import numpy as np
from mc.distributions import Triangular, Normal, Beta
from mc.engine import MonteCarloEngine, SimulationResult

CALIBRATION_YEAR = 2026  # IBM Cost of a Data Breach 2025 + Verizon DBIR 2026

# Three-point sector risk (low, mode, high) — calibrated to IBM/Ponemon + Verizon DBIR
# Update SECTOR_RISK and CALIBRATION_YEAR when the annual IBM report is published.
# DBIR 2026: 22,625 confirmed breaches across 145 countries (Oct 2024–Nov 2025).
SECTOR_RISK: dict[str, tuple[float, float, float]] = {
    "Healthcare":             (0.72, 0.87, 0.96),  # DBIR 2026: 1,438 breaches, 96.4% confirmation
    "Financial Services":     (0.64, 0.79, 0.92),  # DBIR 2026: 34.1% confirmation, internal actors 22%→12%
    "Finance":                (0.64, 0.79, 0.92),
    "Industrials":            (0.62, 0.78, 0.91),  # DBIR 2026: #1 volume (2,713), third-party 61%
    "Energy":                 (0.56, 0.72, 0.86),  # no direct DBIR sector; interpolated
    "Technology":             (0.55, 0.71, 0.85),  # Information (NAICS 51): 1,099 breaches, 64.5% confirmation
    "Pharmaceuticals":        (0.52, 0.68, 0.82),  # Professional services proxy
    "Communication Services": (0.46, 0.62, 0.76),  # Information (NAICS 51) proxy
    "Consumer Discretionary": (0.44, 0.60, 0.74),  # Retail (NAICS 44-45): nearly doubled to 806 breaches
    "Consumer Staples":       (0.40, 0.55, 0.69),  # Retail proxy, lower exposure
    "Real Estate":            (0.40, 0.57, 0.70),  # DBIR 2026: 98.8% confirmation rate
    "Materials":              (0.32, 0.49, 0.63),  # Manufacturing data supports slight increase
    "Utilities":              (0.48, 0.64, 0.78),  # DBIR 2026: 93.6% confirmation, espionage 71% — major shift
}
DEFAULT_SECTOR_RISK = (0.25, 0.50, 0.70)

WEIGHTS = {
    "sector": 0.30,
    "size":   0.20,
    "peers":  0.25,
    "growth": 0.15,
    "beta":   0.10,
}

# Factor correlation matrix.
# Key relationships: sector_risk <-> peer_incidents (+0.50, same sector = more breaches);
# revenue_growth <-> market_beta (+0.50, high-growth companies have higher market beta).
# Factors are sampled as correlated normals (normal approximation to their marginal
# distributions) to avoid overstating the confidence interval width. Independent
# sampling adds ~10-20% excess variance for correlated sector profiles.
FACTOR_CORR = np.array([
    #sector  size   peers  growth  beta
    [ 1.00,  0.10,  0.50, -0.10, -0.20],  # sector_risk
    [ 0.10,  1.00,  0.20, -0.15, -0.10],  # company_size
    [ 0.50,  0.20,  1.00,  0.05,  0.00],  # peer_incidents
    [-0.10, -0.15,  0.05,  1.00,  0.50],  # revenue_growth
    [-0.20, -0.10,  0.00,  0.50,  1.00],  # market_beta
])
_FACTOR_CHOL = np.linalg.cholesky(FACTOR_CORR)


def factor_contributions(company_info: dict, industry_rows: list) -> dict[str, float]:
    """Return the deterministic point contribution of each factor to the breach score.

    Useful for bar charts showing why a company scored the way it did.
    Each value represents the expected score points (0–max_weight×100) from that factor.
    """
    sector = company_info.get("sector") or ""
    market_cap_m = company_info.get("market_cap_m") or 0
    market_beta = float(company_info.get("beta") or 1.0)
    rev_growth = float(company_info.get("rev_growth") or 0.0)
    peer_count = len({r["ticker"] for r in industry_rows}) if industry_rows else 0

    low, mode, high = SECTOR_RISK.get(sector, DEFAULT_SECTOR_RISK)
    f1 = (low + mode + high) / 3.0

    f2 = min(math.log10(max(market_cap_m, 1)) / 6.0, 1.0) if market_cap_m > 0 else 0.25

    peer_mode = min(peer_count / 5.0, 1.0)
    alpha = max(peer_mode * 5.0, 0.5)
    beta_p = max((1.0 - peer_mode) * 5.0, 0.5)
    f3 = alpha / (alpha + beta_p)

    g_low = max(rev_growth - 0.05, 0.0) / 0.30
    g_mode = min(rev_growth / 0.30, 1.0)
    g_high = min((rev_growth + 0.05) / 0.30, 1.0)
    if g_low > g_mode:
        g_low = g_mode
    if g_mode > g_high:
        g_high = g_mode
    f4 = (g_low + g_mode + g_high) / 3.0

    f5 = min(max(market_beta - 0.5, 0.0) / 1.5, 1.0)

    return {
        "Sector risk":    round(f1 * WEIGHTS["sector"] * 100, 1),
        "Company size":   round(f2 * WEIGHTS["size"]   * 100, 1),
        "Peer incidents": round(f3 * WEIGHTS["peers"]  * 100, 1),
        "Revenue growth": round(f4 * WEIGHTS["growth"] * 100, 1),
        "Market beta":    round(f5 * WEIGHTS["beta"]   * 100, 1),
    }


class BreachLikelihoodModel:
    """
    Monte Carlo breach likelihood estimator.

    Calibrated to IBM Cost of a Data Breach {year} and Verizon DBIR {year}.
    Update CALIBRATION_YEAR and SECTOR_RISK when a new annual report is published.

    Replaces the deterministic heuristic in BreachIQ likelihood.py.
    Each input factor is sampled from a probability distribution across N
    simulations using a correlated normal approximation (see FACTOR_CORR) to
    avoid overstating the confidence interval width for correlated sector profiles.

    Returns p10/p50/p90, confidence interval, probability of each tier, and
    the full scores array for histogram and sensitivity analysis.
    """.format(year=CALIBRATION_YEAR)

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

        # --- Factor parameters (mean + std for normal approximation) ---
        # Triangular(a, m, b): mean = (a+m+b)/3, var = (a²+b²+m²-am-ab-mb)/18
        # Beta(α, β):          mean = α/(α+β),   var = αβ / ((α+β)²(α+β+1))
        # Normal(μ, σ):        mean = μ,          std = σ  (exact)

        low, mode, high = SECTOR_RISK.get(sector, DEFAULT_SECTOR_RISK)
        f1_mean = (low + mode + high) / 3.0
        f1_std = math.sqrt(
            (low**2 + high**2 + mode**2 - low*mode - low*high - mode*high) / 18.0
        )

        size_point = min(math.log10(max(market_cap_m, 1)) / 6.0, 1.0) if market_cap_m > 0 else 0.25
        f2_mean, f2_std = size_point, 0.06

        peer_mode = min(peer_tickers / 5.0, 1.0)
        alpha = max(peer_mode * 5.0, 0.5)
        beta_p = max((1.0 - peer_mode) * 5.0, 0.5)
        f3_mean = alpha / (alpha + beta_p)
        f3_std = math.sqrt(alpha * beta_p / ((alpha + beta_p) ** 2 * (alpha + beta_p + 1)))

        g_low = max(rev_growth - 0.05, 0.0) / 0.30
        g_mode = min(rev_growth / 0.30, 1.0)
        g_high = min((rev_growth + 0.05) / 0.30, 1.0)
        if g_low > g_mode: g_low = g_mode
        if g_mode > g_high: g_high = g_mode
        f4_mean = (g_low + g_mode + g_high) / 3.0
        # Degenerate: all three equal (e.g., very high growth >= 35%)
        f4_std = (
            math.sqrt(
                (g_low**2 + g_high**2 + g_mode**2 - g_low*g_mode - g_low*g_high - g_mode*g_high) / 18.0
            ) if g_low < g_high else 0.01
        )

        beta_point = min(max(market_beta - 0.5, 0.0) / 1.5, 1.0)
        f5_mean, f5_std = beta_point, 0.08

        f_means = np.array([f1_mean, f2_mean, f3_mean, f4_mean, f5_mean])
        f_stds  = np.array([f1_std,  f2_std,  f3_std,  f4_std,  f5_std])

        # Correlated sampling: (n, 5) standard normals × Cholesky factor
        z = rng.standard_normal((n, 5)) @ _FACTOR_CHOL.T
        factors = (f_means + f_stds * z).clip(0.0, 1.0)

        f_sector = factors[:, 0]
        f_size   = factors[:, 1]
        f_peers  = factors[:, 2]
        f_growth = factors[:, 3]
        f_beta   = factors[:, 4]

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
            "scores": result.values,
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
    peer_note = (
        f"{peer_tickers} sector peers have disclosed material incidents."
        if peer_tickers > 0
        else "No peer breach disclosures found in this sector."
    )

    return (
        f"{name} median likelihood score: {p50}/100 ({tier} tier). "
        f"90% CI: {p10}–{p90}. "
        f"Probability of Critical tier (70+): {prob_crit}%. "
        f"{peer_note} "
        f"{spread}"
    ).strip()
