import pytest

from mc.models.fitted_cost import FittedCyberCostModel

# Fixed, hand-written params so the test doesn't depend on the shipped fit.
PARAMS = {"use_ibm": True, "theta": [3.9, 0.5, -0.08], "cov": [[0.02, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]}


def run(mcap, sector="Healthcare", params=PARAMS):
    return FittedCyberCostModel(mcap, sector, params=params, n=20_000, seed=1).run()


def test_percentiles_ordered_and_positive():
    r = run(10_000)
    assert 0 < r["p10"] < r["p50"] < r["p90"]


def test_bigger_company_costs_more():
    assert run(100_000)["p50"] > run(1_000)["p50"]


def test_ibm_sector_baseline_applies_only_with_ibm():
    assert run(10_000, "Healthcare")["p50"] > run(10_000, "Retail")["p50"]
    no_ibm = {**PARAMS, "use_ibm": False}
    assert run(10_000, "Healthcare", no_ibm)["p50"] == run(10_000, "Retail", no_ibm)["p50"]


def test_shipped_params_load():
    assert FittedCyberCostModel(10_000, n=1_000, seed=1).run()["p50"] > 0


def test_rejects_bad_market_cap():
    with pytest.raises(ValueError):
        FittedCyberCostModel(0)
