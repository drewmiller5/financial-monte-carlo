from __future__ import annotations
import numpy as np
from mc.engine import MonteCarloEngine, SimulationResult


class VaRModel:
    """
    Monte Carlo portfolio Value at Risk (VaR) and Expected Shortfall (CVaR).

    Simulates correlated multi-asset returns via Cholesky decomposition.
    Outputs dollar loss at the specified confidence level over horizon_days.
    """

    def __init__(
        self,
        portfolio_value: float,
        returns_mean: np.ndarray,
        returns_cov: np.ndarray,
        weights: np.ndarray,
        horizon_days: int = 1,
        confidence: float = 0.95,
        n: int = 10_000,
        seed: int | None = None,
    ):
        self.portfolio_value = portfolio_value
        self.returns_mean = np.asarray(returns_mean)
        self.returns_cov = np.asarray(returns_cov)
        self.weights = np.asarray(weights)
        self.horizon_days = horizon_days
        self.confidence = confidence
        self.engine = MonteCarloEngine(n=n, seed=seed)

    def run(self) -> dict:
        rng = self.engine.rng
        n = self.engine.n

        # Scale daily parameters to horizon
        mean_h = self.returns_mean * self.horizon_days
        cov_h  = self.returns_cov  * self.horizon_days

        # Cholesky decomposition for correlated sampling
        L = np.linalg.cholesky(cov_h)
        z = rng.standard_normal((n, len(self.weights)))
        returns_sim = mean_h + z @ L.T

        portfolio_returns = returns_sim @ self.weights
        pnl = portfolio_returns * self.portfolio_value

        # VaR: loss exceeded with (1 - confidence) probability
        var_pct = (1.0 - self.confidence) * 100.0
        var_cutoff = np.percentile(pnl, var_pct)
        var_dollar = abs(var_cutoff)

        # CVaR (Expected Shortfall): expected loss in the tail beyond VaR
        tail = pnl[pnl <= var_cutoff]
        cvar_dollar = abs(tail.mean()) if len(tail) > 0 else var_dollar

        result = SimulationResult(values=pnl, n=n)

        return {
            "var":             round(var_dollar, 2),
            "cvar":            round(cvar_dollar, 2),
            "confidence":      self.confidence,
            "horizon_days":    self.horizon_days,
            "p5_loss":         round(abs(np.percentile(pnl, 5.0)), 2),
            "p1_loss":         round(abs(np.percentile(pnl, 1.0)), 2),
            "prob_loss_pct":   round(result.probability_below(0.0) * 100.0, 1),
        }
