import pytest
from mc.models.breach import BreachLikelihoodModel, factor_contributions


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


def test_extreme_revenue_growth_does_not_crash():
    # Very high growth (>=35%) previously caused degenerate Triangular(1,1,1) ValueError
    high_growth_info = {**HEALTHCARE_INFO, "rev_growth": 0.50}
    model = BreachLikelihoodModel(high_growth_info, INDUSTRY_ROWS_NONE, n=1000, seed=0)
    result = model.run()
    assert result["p10"] >= 0


# ── factor_contributions ──────────────────────────────────────────────────────

def test_factor_contributions_keys():
    contribs = factor_contributions(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE)
    assert set(contribs.keys()) == {"Sector risk", "Company size", "Peer incidents", "Revenue growth", "Market beta"}


def test_factor_contributions_sum_close_to_p50():
    contribs = factor_contributions(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE)
    total = sum(contribs.values())
    result = BreachLikelihoodModel(HEALTHCARE_INFO, INDUSTRY_ROWS_NONE, n=5000, seed=42).run()
    # Deterministic point estimate sum should be within ~10 pts of the median
    assert abs(total - result["p50"]) < 15


def test_factor_contributions_all_nonnegative():
    contribs = factor_contributions(HEALTHCARE_INFO, INDUSTRY_ROWS_MANY)
    assert all(v >= 0 for v in contribs.values())


def test_factor_contributions_high_peers_raises_peer_score():
    few  = factor_contributions(HEALTHCARE_INFO, [])
    many = factor_contributions(HEALTHCARE_INFO, INDUSTRY_ROWS_MANY)
    assert many["Peer incidents"] > few["Peer incidents"]
