import numpy as np
import pytest
from mc.models.var import VaRModel


SIMPLE_INPUTS = dict(
    portfolio_value=100_000.0,
    returns_mean=np.array([0.0005, 0.0004, 0.0003]),
    returns_cov=np.array([
        [0.0001,  0.00005, 0.00002],
        [0.00005, 0.0001,  0.00003],
        [0.00002, 0.00003, 0.0001 ],
    ]),
    weights=np.array([0.4, 0.4, 0.2]),
    horizon_days=1,
    confidence=0.95,
    n=5000,
    seed=42,
)


def test_output_has_required_keys():
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert {"var", "cvar", "confidence", "horizon_days", "p5_loss", "p1_loss", "prob_loss_pct"}.issubset(result.keys())


def test_var_is_positive():
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert result["var"] > 0


def test_cvar_gte_var():
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert result["cvar"] >= result["var"]


def test_confidence_returned():
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert result["confidence"] == 0.95


def test_longer_horizon_higher_var():
    r1 = VaRModel(**{**SIMPLE_INPUTS, "horizon_days": 1}).run()
    r10 = VaRModel(**{**SIMPLE_INPUTS, "horizon_days": 10}).run()
    assert r10["var"] > r1["var"]


def test_prob_loss_pct_range():
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert 0.0 <= result["prob_loss_pct"] <= 100.0


def test_var_reasonable_magnitude():
    # 95% 1-day VaR on $100K portfolio should be well under $10K
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert result["var"] < 10_000


def test_weights_not_summing_to_one_raises():
    bad_weights = np.array([0.4, 0.4, 0.4])  # sums to 1.2
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        VaRModel(**{**SIMPLE_INPUTS, "weights": bad_weights})


def test_pnl_scores_present_and_frozen():
    result = VaRModel(**SIMPLE_INPUTS).run()
    assert "pnl_scores" in result
    assert len(result["pnl_scores"]) == SIMPLE_INPUTS["n"]
    assert not result["pnl_scores"].flags.writeable


def test_non_positive_definite_cov_raises():
    singular_cov = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValueError, match="positive-definite"):
        VaRModel(**{**SIMPLE_INPUTS, "returns_cov": singular_cov}).run()
