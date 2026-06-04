import numpy as np
import pytest
from mc.models.cyber_cost import CyberCostModel


BASE = dict(market_cap_m=5_000.0, sector="Healthcare", n=2_000, seed=42)


def test_output_has_required_keys():
    result = CyberCostModel(**BASE).run()
    assert {"p10", "p50", "p90", "mean", "std", "ci_90", "unit", "scores"}.issubset(result.keys())


def test_unit_is_dollars_M():
    assert CyberCostModel(**BASE).run()["unit"] == "$M"


def test_percentile_ordering():
    result = CyberCostModel(**BASE).run()
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_all_costs_positive():
    result = CyberCostModel(**BASE).run()
    assert result["p10"] > 0


def test_scores_array_length():
    result = CyberCostModel(**{**BASE, "n": 500}).run()
    assert len(result["scores"]) == 500


def test_larger_company_higher_cost():
    small = CyberCostModel(market_cap_m=100.0, sector="Healthcare", n=3_000, seed=42).run()
    large = CyberCostModel(market_cap_m=500_000.0, sector="Healthcare", n=3_000, seed=42).run()
    assert large["p50"] > small["p50"]


def test_healthcare_higher_than_utilities():
    health = CyberCostModel(market_cap_m=5_000.0, sector="Healthcare", n=3_000, seed=42).run()
    util  = CyberCostModel(market_cap_m=5_000.0, sector="Utilities",  n=3_000, seed=42).run()
    assert health["p50"] > util["p50"]


def test_high_breach_score_increases_cost():
    base_cost  = CyberCostModel(**BASE).run()
    high_score = CyberCostModel(**{**BASE, "breach_score": 80.0}).run()
    assert high_score["p50"] > base_cost["p50"]


def test_low_breach_score_decreases_cost():
    base_cost = CyberCostModel(**BASE).run()
    low_score = CyberCostModel(**{**BASE, "breach_score": 20.0}).run()
    assert low_score["p50"] < base_cost["p50"]


def test_negative_market_cap_raises():
    with pytest.raises(ValueError, match="market_cap_m must be positive"):
        CyberCostModel(market_cap_m=-1.0)


def test_invalid_breach_score_raises():
    with pytest.raises(ValueError, match="breach_score must be in"):
        CyberCostModel(market_cap_m=1_000.0, breach_score=150.0)


def test_unknown_sector_uses_default():
    result = CyberCostModel(market_cap_m=1_000.0, sector="Alien Industry", n=1_000, seed=0).run()
    assert result["p10"] > 0


def test_ci_90_ordered():
    lo, hi = CyberCostModel(**BASE).run()["ci_90"]
    assert lo < hi


def test_mega_cap_healthcare_range():
    # Change Healthcare calibration check: $485B market cap healthcare
    # Expected p50 ~$1,500M–$2,500M based on IBM/Ponemon scaling
    result = CyberCostModel(
        market_cap_m=485_000.0, sector="Healthcare", breach_score=74.0, n=5_000, seed=42
    ).run()
    assert 500.0 < result["p50"] < 10_000.0
    assert result["p90"] > result["p50"]
