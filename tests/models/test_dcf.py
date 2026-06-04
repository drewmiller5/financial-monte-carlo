import pytest
from mc.models.dcf import DCFModel


BASE_INPUTS = dict(
    revenue=1000.0,
    revenue_growth=(0.05, 0.10, 0.18),
    operating_margin=(0.10, 0.15, 0.22),
    discount_rate=(0.10, 0.02),
    terminal_growth=(0.025, 0.005),
    years=10,
    n=2000,
    seed=42,
)


def test_output_has_required_keys():
    result = DCFModel(**BASE_INPUTS).run()
    assert {"p10", "p50", "p90", "mean", "std", "ci_90", "unit"}.issubset(result.keys())


def test_percentile_ordering():
    result = DCFModel(**BASE_INPUTS).run()
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_valuations_positive():
    result = DCFModel(**BASE_INPUTS).run()
    assert result["p10"] > 0


def test_unit_is_dollars_M():
    result = DCFModel(**BASE_INPUTS).run()
    assert result["unit"] == "$M"


def test_higher_growth_higher_value():
    low_growth = DCFModel(**{**BASE_INPUTS, "revenue_growth": (0.01, 0.03, 0.06)}).run()
    high_growth = DCFModel(**{**BASE_INPUTS, "revenue_growth": (0.15, 0.25, 0.35)}).run()
    assert high_growth["p50"] > low_growth["p50"]


def test_higher_discount_rate_lower_value():
    low_rate = DCFModel(**{**BASE_INPUTS, "discount_rate": (0.06, 0.01)}).run()
    high_rate = DCFModel(**{**BASE_INPUTS, "discount_rate": (0.18, 0.01)}).run()
    assert low_rate["p50"] > high_rate["p50"]


def test_ci_90_ordered():
    result = DCFModel(**BASE_INPUTS).run()
    lo, hi = result["ci_90"]
    assert lo < hi
