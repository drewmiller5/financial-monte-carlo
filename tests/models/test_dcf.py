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


def test_near_zero_spread_does_not_crash():
    # terminal_growth mean nearly equals discount_rate mean — guard must hold
    result = DCFModel(
        **{**BASE_INPUTS, "terminal_growth": (0.09, 0.001), "discount_rate": (0.10, 0.001)}
    ).run()
    assert result["p10"] > 0


# ── Two-stage + CapEx + Net Debt ──────────────────────────────────────────────

TWO_STAGE = dict(
    **BASE_INPUTS,
    revenue_growth_phase2=(0.03, 0.05, 0.08),
    phase1_years=5,
)


def test_two_stage_output_has_required_keys():
    result = DCFModel(**TWO_STAGE).run()
    assert {"p10", "p50", "p90", "mean", "std", "ci_90", "unit"}.issubset(result.keys())


def test_two_stage_percentile_ordering():
    result = DCFModel(**TWO_STAGE).run()
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_two_stage_high_phase1_growth_raises_value():
    # Higher phase-1 growth than the single-stage rate → higher enterprise value
    single = DCFModel(**BASE_INPUTS).run()
    two_stage = DCFModel(
        **{**BASE_INPUTS,
           "revenue_growth": (0.20, 0.28, 0.38),    # fast phase 1
           "revenue_growth_phase2": (0.05, 0.08, 0.12),  # slow phase 2
           "phase1_years": 5}
    ).run()
    assert two_stage["p50"] > single["p50"]


def test_capex_reduces_enterprise_value():
    no_capex = DCFModel(**BASE_INPUTS).run()
    with_capex = DCFModel(
        **{**BASE_INPUTS, "capex_pct_revenue": (0.04, 0.06, 0.10)}
    ).run()
    assert with_capex["p50"] < no_capex["p50"]


def test_net_debt_equity_bridge_present():
    result = DCFModel(**{**BASE_INPUTS, "net_debt": 200.0}).run()
    assert "equity_p50" in result
    assert "equity_p10" in result
    assert "equity_p90" in result
    assert result["net_debt"] == 200.0


def test_net_debt_equity_less_than_enterprise():
    result = DCFModel(**{**BASE_INPUTS, "net_debt": 200.0}).run()
    assert result["equity_p50"] == round(result["p50"] - 200.0, 1)


def test_net_cash_equity_greater_than_enterprise():
    # Negative net_debt means net cash — equity > enterprise value
    result = DCFModel(**{**BASE_INPUTS, "net_debt": -500.0}).run()
    assert result["equity_p50"] > result["p50"]


def test_net_debt_zero_no_equity_keys():
    # net_debt=0 (default) → no equity keys in output to keep output clean
    result = DCFModel(**BASE_INPUTS).run()
    assert "equity_p50" not in result


def test_two_stage_backward_compat_no_new_params():
    # Omitting all new params should produce same shape output as before
    result = DCFModel(**BASE_INPUTS).run()
    assert {"p10", "p50", "p90", "mean", "std", "ci_90", "unit", "values"}.issubset(result.keys())


def test_two_stage_capex_and_two_stage_combined():
    result = DCFModel(
        **{**BASE_INPUTS,
           "revenue_growth_phase2": (0.03, 0.05, 0.08),
           "phase1_years": 5,
           "capex_pct_revenue": (0.04, 0.06, 0.10)}
    ).run()
    assert result["p10"] > 0
    assert result["p10"] <= result["p50"] <= result["p90"]


# ── Validation guards ─────────────────────────────────────────────────────────

def test_only_phase1_years_raises():
    with pytest.raises(ValueError, match="both be set, or neither"):
        DCFModel(**{**BASE_INPUTS, "phase1_years": 5})


def test_only_growth_phase2_raises():
    with pytest.raises(ValueError, match="both be set, or neither"):
        DCFModel(**{**BASE_INPUTS, "revenue_growth_phase2": (0.03, 0.05, 0.08)})


def test_phase1_years_equal_to_years_raises():
    with pytest.raises(ValueError, match="phase1_years must be in"):
        DCFModel(
            **{**BASE_INPUTS,
               "phase1_years": 10,
               "revenue_growth_phase2": (0.03, 0.05, 0.08)}
        )


def test_phase1_years_zero_raises():
    with pytest.raises(ValueError, match="phase1_years must be in"):
        DCFModel(
            **{**BASE_INPUTS,
               "phase1_years": 0,
               "revenue_growth_phase2": (0.03, 0.05, 0.08)}
        )


# ── operating_margin_phase2 ───────────────────────────────────────────────────

def test_margin_phase2_runs():
    result = DCFModel(
        **{**BASE_INPUTS,
           "revenue_growth_phase2": (0.03, 0.05, 0.08),
           "phase1_years": 5,
           "operating_margin_phase2": (0.18, 0.24, 0.32)}
    ).run()
    assert result["p10"] > 0
    assert result["p10"] <= result["p50"] <= result["p90"]


def test_higher_phase2_margin_raises_value():
    low_margin2 = DCFModel(
        **{**BASE_INPUTS,
           "revenue_growth_phase2": (0.03, 0.05, 0.08),
           "phase1_years": 5,
           "operating_margin_phase2": (0.05, 0.08, 0.12)}
    ).run()
    high_margin2 = DCFModel(
        **{**BASE_INPUTS,
           "revenue_growth_phase2": (0.03, 0.05, 0.08),
           "phase1_years": 5,
           "operating_margin_phase2": (0.20, 0.28, 0.36)}
    ).run()
    assert high_margin2["p50"] > low_margin2["p50"]


def test_margin_phase2_without_two_stage_raises():
    with pytest.raises(ValueError, match="operating_margin_phase2 requires"):
        DCFModel(**{**BASE_INPUTS, "operating_margin_phase2": (0.18, 0.24, 0.32)})
