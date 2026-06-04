"""CLI integration tests — subprocess-based, covers all three subcommands."""
import json
import subprocess
import sys

import pytest


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mc", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(__file__).split("tests")[0],  # project root
    )


# ── breach ────────────────────────────────────────────────────────────────────

def test_breach_human_output_runs():
    r = run("breach", "--sector", "Healthcare", "--market-cap", "5000",
            "--beta", "1.1", "--growth", "0.10", "--n", "500", "--seed", "42")
    assert r.returncode == 0
    assert "p50" in r.stdout.lower() or "median" in r.stdout.lower()


def test_breach_json_keys():
    r = run("breach", "--sector", "Healthcare", "--market-cap", "5000",
            "--beta", "1.1", "--growth", "0.10", "--n", "500", "--seed", "42", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert {"p10", "p50", "p90", "ci_90", "tier", "prob_critical_pct"}.issubset(data)


def test_breach_json_percentile_ordering():
    r = run("breach", "--sector", "Healthcare", "--market-cap", "5000",
            "--beta", "1.1", "--growth", "0.10", "--n", "500", "--seed", "42", "--json")
    data = json.loads(r.stdout)
    assert 0 <= data["p10"] <= data["p50"] <= data["p90"] <= 100


def test_breach_tier_valid():
    r = run("breach", "--sector", "Healthcare", "--market-cap", "5000",
            "--beta", "1.1", "--growth", "0.10", "--n", "500", "--seed", "42", "--json")
    data = json.loads(r.stdout)
    assert data["tier"] in {"Low", "Moderate", "High", "Critical"}


def test_breach_missing_market_cap_exits_nonzero():
    r = run("breach", "--sector", "Healthcare", "--beta", "1.1", "--growth", "0.10")
    assert r.returncode != 0


def test_breach_missing_sector_exits_nonzero():
    r = run("breach", "--market-cap", "5000", "--beta", "1.1", "--growth", "0.10")
    assert r.returncode != 0


# ── cost ──────────────────────────────────────────────────────────────────────

def test_cost_human_output_runs():
    r = run("cost", "--market-cap", "5000", "--sector", "Healthcare",
            "--n", "500", "--seed", "42")
    assert r.returncode == 0


def test_cost_json_keys():
    r = run("cost", "--market-cap", "5000", "--sector", "Healthcare",
            "--n", "500", "--seed", "42", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert {"p10", "p50", "p90", "unit"}.issubset(data)
    assert data["unit"] == "$M"
    assert data["p50"] > 0


def test_cost_with_breach_score():
    r = run("cost", "--market-cap", "5000", "--sector", "Healthcare",
            "--breach-score", "75", "--n", "500", "--seed", "42", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["p50"] > 0


def test_cost_missing_market_cap_exits_nonzero():
    r = run("cost", "--sector", "Healthcare")
    assert r.returncode != 0


# ── dcf ───────────────────────────────────────────────────────────────────────

_DCF_BASE = [
    "--revenue", "500",
    "--growth", "0.08,0.14,0.20",
    "--margin", "0.10,0.15,0.22",
    "--discount", "0.10,0.02",
    "--terminal", "0.025,0.005",
    "--n", "500", "--seed", "42",
]


def test_dcf_human_output_runs():
    r = run("dcf", *_DCF_BASE)
    assert r.returncode == 0


def test_dcf_json_keys():
    r = run("dcf", *_DCF_BASE, "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert {"p10", "p50", "p90", "unit"}.issubset(data)
    assert data["unit"] == "$M"
    assert data["p50"] > 0


def test_dcf_two_stage_json():
    r = run("dcf", *_DCF_BASE,
            "--growth2", "0.04,0.06,0.10", "--phase1-years", "5", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["p50"] > 0


def test_dcf_capex_reduces_value():
    r_no_capex = run("dcf", *_DCF_BASE, "--json")
    r_capex    = run("dcf", *_DCF_BASE, "--capex", "0.05,0.07,0.12", "--json")
    assert r_no_capex.returncode == 0
    assert r_capex.returncode == 0
    assert json.loads(r_capex.stdout)["p50"] < json.loads(r_no_capex.stdout)["p50"]


def test_dcf_net_debt_equity_bridge():
    r = run("dcf", *_DCF_BASE, "--net-debt", "300", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert "equity_p50" in data
    assert data["equity_p50"] == round(data["p50"] - 300, 1)


def test_dcf_malformed_triple_exits_nonzero():
    args = list(_DCF_BASE)
    # Replace --growth value with a 2-element string
    args[args.index("0.08,0.14,0.20")] = "0.08,0.14"
    r = run("dcf", *args)
    assert r.returncode != 0
    assert "3" in r.stderr or "comma" in r.stderr.lower() or "error" in r.stderr.lower()


def test_dcf_missing_revenue_exits_nonzero():
    args = [a for a in _DCF_BASE if a != "500"]
    r = run("dcf", "--growth", "0.08,0.14,0.20",
            "--margin", "0.10,0.15,0.22",
            "--discount", "0.10,0.02",
            "--terminal", "0.025,0.005")
    assert r.returncode != 0
