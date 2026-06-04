import pytest
from mc.models.breach import BreachLikelihoodModel


HEALTHCARE_INFO = {
    "name": "TestCo Health",
    "sector": "Healthcare",
    "market_cap_m": 5000,
    "beta": 1.2,
    "rev_growth": 0.10,
}
INDUSTRY_ROWS_MANY = [{"ticker": f"T{i}"} for i in range(6)]
INDUSTRY_ROWS_NONE = []


def test_output_has_required_keys():
    model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    required = {"p10", "p50", "p90", "mean", "std", "ci_90", "tier", "narrative", "prob_critical_pct"}
    assert required.issubset(result.keys())


def test_percentile_ordering():
    model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_scores_bounded_0_100():
    model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert 0 <= result["p10"] <= 100
    assert 0 <= result["p90"] <= 100


def test_healthcare_scores_higher_than_utilities():
    utilities_info = {**HEALTHCARE_INFO, "sector": "Utilities", "name": "TestCo Power"}
    health_model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=2000, seed=42)
    util_model = BreachLikelihoodModel(utilities_info, INDUSTRY_ROWS_NONE, n=2000, seed=42)
    assert health_model.run()["p50"] > util_model.run()["p50"]


def test_more_peers_increases_score():
    few_peers = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=2000, seed=42)
    many_peers = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_MANY, n=2000, seed=42)
    assert many_peers.run()["p50"] > few_peers.run()["p50"]


def test_tier_is_valid():
    model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert result["tier"] in {"Low", "Moderate", "High", "Critical"}


def test_prob_critical_pct_range():
    model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert 0.0 <= result["prob_critical_pct"] <= 100.0


def test_unknown_sector_uses_default():
    unknown_info = {**HEALTHCARE_INFO, "sector": "Alien Industry"}
    model = BreachLikelihoodModel(unknown_info, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert result["p10"] >= 0


def test_narrative_is_nonempty_string():
    model = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert isinstance(result["narrative"], str)
    assert len(result["narrative"]) > 20
