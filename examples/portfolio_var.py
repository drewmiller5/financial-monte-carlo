"""
Example: 3-asset portfolio Monte Carlo VaR.

Demonstrates 95% 1-day and 10-day Value at Risk with
correlated asset returns via Cholesky decomposition.
"""
import numpy as np
from mc.models.var import VaRModel

# Portfolio: 40% SPY-like, 40% bonds, 20% gold
returns_mean = np.array([0.0004, 0.0001, 0.0002])
returns_cov  = np.array([
    [0.000100, 0.000020, 0.000010],
    [0.000020, 0.000025, 0.000005],
    [0.000010, 0.000005, 0.000080],
])
weights = np.array([0.40, 0.40, 0.20])

portfolio_value = 1_000_000.0

print(f"Portfolio: ${portfolio_value:,.0f}")
print(f"Weights: 40% equities / 40% bonds / 20% gold\n")

for horizon, conf in [(1, 0.95), (1, 0.99), (10, 0.95)]:
    result = VaRModel(
        portfolio_value=portfolio_value,
        returns_mean=returns_mean,
        returns_cov=returns_cov,
        weights=weights,
        horizon_days=horizon,
        confidence=conf,
        n=50_000,
        seed=42,
    ).run()
    print(f"{int(conf*100)}% {horizon}-day VaR:  ${result['var']:>10,.2f}   CVaR: ${result['cvar']:>10,.2f}")

print(f"\nProb any loss (1-day): {result['prob_loss_pct']}%")
