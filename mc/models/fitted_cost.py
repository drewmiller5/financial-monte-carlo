"""Cyber breach cost fitted to public breach disclosures, with IBM sector costs as the baseline.

    ln(cost $M) = ibm_offset(sector) + a + b * ln(market_cap / $10B) + sigma * z

ibm_offset is IBM's sector log-mean from cyber_cost.py (_BASE_MU + SECTOR_LOG_ADJ). The data then
fits what IBM can't say: the level of material public-company breaches (a), how cost scales with
company size (b), and the spread (sigma). Fitted with use_ibm=False the offset is 0, so the model
still works if IBM's report stops being available.

Parameters are produced by validation/fit_cost.py (interval/censored regression on
validation/breach_cases.csv) and stored in fitted_cost_params.json. level_shift corrects the
median's high lean: it is the median leave-one-out error on complete-cost training cases
(lower-bound cases in training pull the fitted level up). Each simulation first draws
(a, b, sigma) from their estimated uncertainty, then a cost, so the interval covers both.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from mc.engine import MonteCarloEngine, SimulationResult
from mc.models.cyber_cost import _BASE_MU, DEFAULT_SECTOR_LOG_ADJ, SECTOR_LOG_ADJ

PARAMS_PATH = pathlib.Path(__file__).with_name("fitted_cost_params.json")
REF_MCAP_M = 10_000.0  # size term is centred on a $10B company


def ibm_offset(sector: str, use_ibm: bool = True) -> float:
    return _BASE_MU + SECTOR_LOG_ADJ.get(sector, DEFAULT_SECTOR_LOG_ADJ) if use_ibm else 0.0


class FittedCyberCostModel:
    """Monte Carlo breach cost ($M) for a public company, given market cap and sector."""

    def __init__(
        self,
        market_cap_m: float,
        sector: str = "Healthcare",
        params: dict | None = None,
        n: int = 10_000,
        seed: int | None = None,
    ):
        if market_cap_m <= 0:
            raise ValueError(f"market_cap_m must be positive, got {market_cap_m}")
        self.market_cap_m = market_cap_m
        self.sector = sector
        self.params = params if params is not None else json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
        self.engine = MonteCarloEngine(n=n, seed=seed)

    def run(self) -> dict:
        rng, n, p = self.engine.rng, self.engine.n, self.params
        theta = rng.multivariate_normal(p["theta"], p["cov"], n)  # columns: a, b, ln(sigma)
        mu = (ibm_offset(self.sector, p["use_ibm"]) + theta[:, 0] + p.get("level_shift", 0.0)
              + theta[:, 1] * math.log(self.market_cap_m / REF_MCAP_M))
        costs = np.exp(mu + np.exp(theta[:, 2]) * rng.standard_normal(n))
        result = SimulationResult(values=costs, n=n)
        lo, hi = result.confidence_interval(10, 90)
        return {
            "p10": round(result.p10, 1),
            "p50": round(result.p50, 1),
            "p90": round(result.p90, 1),
            "mean": round(result.mean, 1),
            "std": round(result.std, 1),
            "ci_90": (round(lo, 1), round(hi, 1)),
            "unit": "$M",
            "scores": result.values,
        }
