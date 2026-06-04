"""Integration tests: model pipeline connections."""
import numpy as np
from mc.models.breach import BreachLikelihoodModel
from mc.models.cyber_cost import CyberCostModel
from mc.models.cyber_var_bridge import CyberVaRBridge

COMPANY = {"name": "TestCo", "sector": "Healthcare", "market_cap_m": 10_000,
           "beta": 1.0, "rev_growth": 0.10}
PEERS = [{"ticker": f"P{i}"} for i in range(4)]


def test_breach_to_cost_pipeline():
    breach = BreachLikelihoodModel(COMPANY, PEERS, n=2_000, seed=42).run()
    cost = CyberCostModel(
        market_cap_m=COMPANY["market_cap_m"],
        sector=COMPANY["sector"],
        breach_score=float(breach["p50"]),
        n=2_000, seed=42,
    ).run()
    assert cost["p50"] > 0.0
    assert cost["p10"] <= cost["p50"] <= cost["p90"]


def test_breach_to_var_bridge_pipeline():
    breach = BreachLikelihoodModel(COMPANY, PEERS, n=2_000, seed=42).run()
    prob = CyberVaRBridge.score_to_prob(breach["p50"])
    ddmean, ddstd = CyberVaRBridge.sector_drawdown(COMPANY["sector"])
    bridge = CyberVaRBridge(
        portfolio_value=1_000_000.0,
        returns_mean=np.array([0.0005]),
        returns_cov=np.array([[0.0001]]),
        weights=np.array([1.0]),
        annual_breach_probs=np.array([prob]),
        breach_drawdown_mean=ddmean,
        breach_drawdown_std=ddstd,
        n=2_000, seed=42,
    ).run()
    assert bridge["cyber_var"] >= bridge["baseline_var"]
    assert bridge["cyber_contribution"] >= 0.0


def test_full_three_model_pipeline():
    breach = BreachLikelihoodModel(COMPANY, PEERS, n=2_000, seed=42).run()
    cost = CyberCostModel(
        market_cap_m=COMPANY["market_cap_m"],
        sector=COMPANY["sector"],
        breach_score=float(breach["p50"]),
        n=2_000, seed=42,
    ).run()
    eal = cost["p50"] * CyberVaRBridge.score_to_prob(breach["p50"])
    assert eal > 0.0
    assert breach["tier"] in {"Low", "Moderate", "High", "Critical"}
