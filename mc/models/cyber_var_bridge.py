from __future__ import annotations
import numpy as np
from mc.engine import MonteCarloEngine, SimulationResult

# Sector-specific breach drawdown distributions (mean, std) calibrated to
# observed post-breach equity drawdowns over the 30 days following announcement.
# Sources: Equifax -35% (2017), Change Healthcare -10% (2024),
#          Anthem -3% (2015), Capital One -6% (2019), MGM Resorts -8% (2023).
SECTOR_BREACH_DRAWDOWN: dict[str, tuple[float, float]] = {
    "Healthcare":             (-0.08, 0.05),
    "Financial Services":     (-0.06, 0.04),
    "Finance":                (-0.06, 0.04),
    "Technology":             (-0.10, 0.06),
    "Communication Services": (-0.07, 0.04),
    "Consumer Discretionary": (-0.08, 0.05),
    "Consumer Staples":       (-0.04, 0.03),
    "Real Estate":            (-0.04, 0.03),
    "Energy":                 (-0.05, 0.04),
    "Industrials":            (-0.05, 0.04),
    "Materials":              (-0.04, 0.03),
    "Utilities":              (-0.03, 0.02),
}
DEFAULT_BREACH_DRAWDOWN = (-0.07, 0.05)


class CyberVaRBridge:
    """
    Cyber-adjusted portfolio VaR.

    Extends standard market-return VaR by injecting a breach-event jump risk
    for each asset. For each simulation:
      1. Draw correlated market returns (Cholesky, identical to VaRModel).
      2. Per asset, draw a Bernoulli(p_breach_horizon) event.
      3. If breach fires, apply a stock shock from the asset's breach
         drawdown distribution (log-normal negative shock, capped at -95%).

    Returns both baseline_var (market risk only) and cyber_var (market + cyber)
    so the incremental cyber risk contribution is directly observable.

    This connects BreachLikelihoodModel (likelihood score 0-100) to VaRModel
    (financial risk) — the bridge between BreachIQ and Iron Standard.

    Usage:
        breach = BreachLikelihoodModel(company_info, rows, n=50_000).run()
        prob = CyberVaRBridge.score_to_prob(breach["p50"])
        bridge = CyberVaRBridge(
            portfolio_value=1_000_000,
            returns_mean=np.array([daily_mean]),
            returns_cov=np.array([[daily_var]]),
            weights=np.array([1.0]),
            annual_breach_probs=np.array([prob]),
        )
        result = bridge.run()
        print(result["cyber_contribution"])  # incremental VaR from cyber risk
    """

    def __init__(
        self,
        portfolio_value: float,
        returns_mean: np.ndarray,
        returns_cov: np.ndarray,
        weights: np.ndarray,
        annual_breach_probs: np.ndarray | float,
        breach_drawdown_mean: np.ndarray | float = -0.08,
        breach_drawdown_std: np.ndarray | float = 0.05,
        horizon_days: int = 30,
        confidence: float = 0.95,
        n: int = 10_000,
        seed: int | None = None,
    ):
        weights = np.asarray(weights, dtype=float)
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {weights.sum():.6f}")

        k = len(weights)
        annual_breach_probs = np.broadcast_to(
            np.asarray(annual_breach_probs, dtype=float), (k,)
        ).copy()
        if np.any((annual_breach_probs < 0.0) | (annual_breach_probs > 1.0)):
            raise ValueError("annual_breach_probs must all be in [0, 1]")

        self.portfolio_value = portfolio_value
        self.returns_mean = np.asarray(returns_mean, dtype=float)
        self.returns_cov = np.asarray(returns_cov, dtype=float)
        self.weights = weights
        self.annual_breach_probs = annual_breach_probs
        self.breach_drawdown_mean = np.broadcast_to(
            np.asarray(breach_drawdown_mean, dtype=float), (k,)
        ).copy()
        self.breach_drawdown_std = np.broadcast_to(
            np.asarray(breach_drawdown_std, dtype=float), (k,)
        ).copy()
        self.horizon_days = horizon_days
        self.confidence = confidence
        self.engine = MonteCarloEngine(n=n, seed=seed)

    @staticmethod
    def score_to_prob(breach_score: float) -> float:
        """Convert BreachLikelihoodModel score (0-100) to annual breach probability.

        The score is calibrated to IBM/Ponemon annual breach rates, so
        score / 100 is a direct probability interpretation.
        """
        return min(max(breach_score / 100.0, 0.0), 1.0)

    @staticmethod
    def sector_drawdown(sector: str) -> tuple[float, float]:
        """Return (mean, std) breach-induced equity drawdown for a sector."""
        return SECTOR_BREACH_DRAWDOWN.get(sector, DEFAULT_BREACH_DRAWDOWN)

    def run(self) -> dict:
        rng = self.engine.rng
        n = self.engine.n
        k = len(self.weights)

        mean_h = self.returns_mean * self.horizon_days
        cov_h = self.returns_cov * self.horizon_days

        try:
            L = np.linalg.cholesky(cov_h)
        except np.linalg.LinAlgError as exc:
            raise ValueError("returns_cov must be positive-definite.") from exc

        # Market returns: same Cholesky approach as VaRModel
        z = rng.standard_normal((n, k))
        market_returns = mean_h + z @ L.T  # (n, k)

        # Baseline P&L: market risk only
        baseline_pnl = (market_returns @ self.weights) * self.portfolio_value

        # Scale annual breach probability to horizon window
        p_horizon = 1.0 - (1.0 - self.annual_breach_probs) ** (self.horizon_days / 365.0)

        # Per-simulation per-asset Bernoulli breach events
        breach_events = rng.uniform(0.0, 1.0, (n, k)) < p_horizon  # (n, k) bool

        # Breach drawdown shocks (negative only; max drawdown -95%)
        shocks = rng.normal(self.breach_drawdown_mean, self.breach_drawdown_std, (n, k))
        shocks = shocks.clip(-0.95, 0.0)

        # Cyber-adjusted returns: market + breach shock where breach fired
        cyber_returns = market_returns + (breach_events * shocks)
        cyber_pnl = (cyber_returns @ self.weights) * self.portfolio_value

        var_pct = (1.0 - self.confidence) * 100.0

        baseline_cutoff = np.percentile(baseline_pnl, var_pct)
        cyber_cutoff = np.percentile(cyber_pnl, var_pct)

        baseline_var = max(0.0, -baseline_cutoff)
        cyber_var = max(0.0, -cyber_cutoff)

        baseline_tail = baseline_pnl[baseline_pnl <= baseline_cutoff]
        cyber_tail = cyber_pnl[cyber_pnl <= cyber_cutoff]

        baseline_cvar = max(0.0, -baseline_tail.mean()) if len(baseline_tail) > 0 else baseline_var
        cyber_cvar = max(0.0, -cyber_tail.mean()) if len(cyber_tail) > 0 else cyber_var

        pct_sims_breach = float(breach_events.any(axis=1).mean()) * 100.0

        return {
            "baseline_var":         round(baseline_var, 2),
            "cyber_var":            round(cyber_var, 2),
            "cyber_contribution":   round(cyber_var - baseline_var, 2),
            "baseline_cvar":        round(baseline_cvar, 2),
            "cyber_cvar":           round(cyber_cvar, 2),
            "confidence":           self.confidence,
            "horizon_days":         self.horizon_days,
            "breach_prob_horizon":  [round(float(p), 4) for p in p_horizon],
            "pct_sims_with_breach": round(pct_sims_breach, 1),
            "baseline_pnl":         baseline_pnl,
            "cyber_pnl":            cyber_pnl,
        }
