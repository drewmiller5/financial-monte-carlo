# financial-monte-carlo

Monte Carlo risk simulation engine for finance, cyber risk, and model validation.

Replaces deterministic point-estimate risk models with probability distributions.
Instead of a single score, you get a range: **p10 / p50 / p90 + confidence interval**.

## Why

Point estimates in risk models are false precision. A breach likelihood score of "63/100" implies
certainty that doesn't exist. A range of **44–81 (90% CI), 22% chance Critical** is honest.

This library applies that principle to three domains:

| Model | What it simulates | Replaces |
|---|---|---|
| `BreachLikelihoodModel` | Probability distribution of breach risk | Deterministic weighted heuristic |
| `DCFModel` | Distribution of DCF enterprise values | Single-point DCF output |
| `VaRModel` | Portfolio loss distribution via Cholesky | Parametric VaR |

## Install

```bash
pip install numpy
git clone https://github.com/drewmiller5/financial-monte-carlo
cd financial-monte-carlo
pip install -e .
```

## Quick Start

```python
from mc.models.breach import BreachLikelihoodModel

result = BreachLikelihoodModel(
    company_info={"name": "Acme Health", "sector": "Healthcare", "market_cap_m": 5000,
                  "beta": 1.1, "rev_growth": 0.10},
    industry_rows=[{"ticker": "PEER1"}, {"ticker": "PEER2"}],
    n=10_000,
    seed=42,
).run()

print(result["p50"])           # 68  (median score)
print(result["ci_90"])         # (48, 86)  (90% confidence interval)
print(result["prob_critical_pct"])  # 31.4  (% chance score ≥ 70)
print(result["tier"])          # "High"
```

## Models

### BreachLikelihoodModel
Forward-looking breach probability for companies with no SEC Item 1.05 history.
Each factor (sector, size, peers, growth, beta) is sampled from a distribution.

### DCFModel
Monte Carlo DCF — revenue growth, margins, discount rate, and terminal growth
are each Triangular or Normal distributions. Returns p10/p50/p90 of enterprise value.

**Note:** `operating_margin` is used as an FCF margin proxy (appropriate for asset-light businesses).

### VaRModel
Portfolio Value at Risk via correlated Monte Carlo simulation (Cholesky decomposition).
Returns 95%/99% VaR and CVaR (Expected Shortfall). Weights must sum to 1.0.

## Connection to SR 11-7

Federal Reserve SR 11-7 requires model validation: independent review, sensitivity analysis,
and documentation of model limitations. Replacing a point-estimate with a distribution
is the first step — it forces you to document what you don't know.

## Tests

```bash
pytest tests/ -v
```

49 tests, all passing, 96% coverage.
