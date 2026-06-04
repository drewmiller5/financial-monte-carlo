import numpy as np
import pytest
from mc.models.cyber_var_bridge import CyberVaRBridge

BASE = dict(
    portfolio_value=100_000.0,
    returns_mean=np.array([0.0005]),
    returns_cov=np.array([[0.0001]]),
    weights=np.array([1.0]),
    annual_breach_probs=np.array([0.70]),
    horizon_days=30,
    confidence=0.95,
    n=5_000,
    seed=42,
)


def test_output_keys():
    r = CyberVaRBridge(**BASE).run()
    assert {"baseline_var","cyber_var","cyber_contribution","baseline_cvar","cyber_cvar",
            "confidence","horizon_days","breach_prob_horizon","pct_sims_with_breach",
            "baseline_pnl","cyber_pnl"}.issubset(r.keys())


def test_cyber_var_gte_baseline():
    r = CyberVaRBridge(**BASE).run()
    assert r["cyber_var"] >= r["baseline_var"]


def test_cyber_contribution_nonneg():
    r = CyberVaRBridge(**BASE).run()
    assert r["cyber_contribution"] >= 0.0


def test_cvar_gte_var():
    r = CyberVaRBridge(**BASE).run()
    assert r["baseline_cvar"] >= r["baseline_var"]
    assert r["cyber_cvar"] >= r["cyber_var"]


def test_zero_breach_prob_equals_baseline():
    r = CyberVaRBridge(**{**BASE, "annual_breach_probs": np.array([0.0])}).run()
    assert r["cyber_contribution"] == 0.0


def test_high_breach_prob_increases_var():
    low  = CyberVaRBridge(**{**BASE, "annual_breach_probs": np.array([0.01]), "n": 10_000, "seed": 42}).run()
    high = CyberVaRBridge(**{**BASE, "annual_breach_probs": np.array([0.90]), "n": 10_000, "seed": 42}).run()
    assert high["cyber_var"] > low["cyber_var"]


def test_breach_prob_horizon_in_0_1():
    r = CyberVaRBridge(**BASE).run()
    for p in r["breach_prob_horizon"]:
        assert 0.0 <= p <= 1.0


def test_pct_sims_with_breach_range():
    r = CyberVaRBridge(**BASE).run()
    assert 0.0 <= r["pct_sims_with_breach"] <= 100.0


def test_weights_not_summing_raises():
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        CyberVaRBridge(**{**BASE, "weights": np.array([0.6])})


def test_invalid_breach_prob_raises():
    with pytest.raises(ValueError, match="annual_breach_probs"):
        CyberVaRBridge(**{**BASE, "annual_breach_probs": np.array([1.5])})


def test_score_to_prob():
    assert CyberVaRBridge.score_to_prob(74.0) == pytest.approx(0.74)
    assert CyberVaRBridge.score_to_prob(0.0)  == pytest.approx(0.0)
    assert CyberVaRBridge.score_to_prob(100.0) == pytest.approx(1.0)


def test_sector_drawdown_returns_tuple():
    mean, std = CyberVaRBridge.sector_drawdown("Healthcare")
    assert mean < 0.0
    assert std > 0.0


def test_multi_asset():
    r = CyberVaRBridge(
        portfolio_value=1_000_000.0,
        returns_mean=np.array([0.0005, 0.0003]),
        returns_cov=np.array([[0.0001, 0.00003],[0.00003, 0.0001]]),
        weights=np.array([0.6, 0.4]),
        annual_breach_probs=np.array([0.70, 0.30]),
        n=5_000, seed=42,
    ).run()
    assert r["cyber_var"] > 0
    assert len(r["breach_prob_horizon"]) == 2
