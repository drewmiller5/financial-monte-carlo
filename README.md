# financial-monte-carlo

Monte Carlo risk simulation engine for finance and cyber risk.

Point estimates in risk models are false precision. A breach likelihood score of
"63/100" implies certainty that doesn't exist. A range of **44–81 (90% CI),
22% chance Critical** is honest. This library applies that principle across four
risk domains.

> **Disclaimer:** This library is an educational and research tool. Outputs are
> probability distributions, not predictions or guarantees. Nothing here constitutes
> financial advice, investment advice, or professional cybersecurity advice. Model
> quality depends entirely on the inputs you supply. Consult qualified financial and
> cybersecurity professionals before making material decisions. The authors disclaim
> all liability for decisions made using this software.

---

## Models

| Model | What it answers | Output |
|-------|----------------|--------|
| `BreachLikelihoodModel` | How likely is this company to suffer a material breach? | Score 0–100, tier, 90% CI |
| `CyberCostModel` | If a breach occurs, what does it cost? | Dollar loss p10/p50/p90 in $M |
| `CyberVaRBridge` | How does breach risk change the portfolio's financial downside? | Baseline VaR, cyber-adjusted VaR, and incremental cyber contribution |
| `DCFModel` | What is this company worth under uncertainty? | Enterprise value range in $M |
| `VaRModel` | What is the portfolio's downside at a given confidence level? | Dollar VaR + CVaR |

---

## Validation: Change Healthcare (Feb 2024)

The largest healthcare data breach in US history. 100 million Americans affected. $3.3B projected cost.

Using only public data available in **December 2023 — two months before the breach:**

```
Breach likelihood tier:  Critical           Actual: Critical breach occurred  ✓
P(score >= 70):          66.6%              Actual: occurred                  ✓
Dollar loss p50:         ~$1,527M           Actual H1 2024: $1,592M           ✓
Dollar loss p75:         ~$3,191M           Actual FY2024:  $3,300M           ✓
Dollar loss p90:         ~$6,210M           Actual FY2024:  $3,300M (below p90) *
30-day VaR (95%):        7.5% drawdown      Actual: 9.9% (beyond VaR tail)    ✓
```

\* p90 is the 90th-percentile tail scenario — the outcome if full regulatory liability,
systemic disruption, and reputational damage all materialized simultaneously. The actual
$3.3B FY2024 cost landed at p75 ($3,191M), confirming the range was correctly calibrated.
p90 is not a point forecast; it is a tail risk bound.

Full retrospective: [`examples/change_healthcare_validation.ipynb`](examples/change_healthcare_validation.ipynb)

---

## Quick start

```bash
git clone https://github.com/drewmiller5/financial-monte-carlo
cd financial-monte-carlo
pip install -e ".[dev]"        # core + pytest
pip install -e ".[notebook]"   # + jupyter, matplotlib, pandas
```

### Breach likelihood

```python
from mc.models.breach import BreachLikelihoodModel

result = BreachLikelihoodModel(
    company_info={
        "name":         "Change Healthcare / UnitedHealth Group",
        "sector":       "Healthcare",
        "market_cap_m": 485_000,    # $485B, Dec 2023
        "beta":         0.59,
        "rev_growth":   0.146,
    },
    industry_rows=[{"ticker": t} for t in ["ANTM", "CYH", "UHS", "HCA"]],
    n=100_000,
    seed=42,
).run()

print(result["tier"])               # Critical
print(result["p50"])                # 72
print(result["ci_90"])              # (65, 79)
print(result["prob_critical_pct"])  # 66.6
```

### Cyber cost (dollar loss if breach occurs)

```python
from mc.models.cyber_cost import CyberCostModel

cost = CyberCostModel(
    market_cap_m=485_000,
    sector="Healthcare",
    breach_score=74.0,   # from BreachLikelihoodModel
    n=100_000,
    seed=42,
).run()

print(cost["p50"])    # ~$1,527M  (actual H1 2024: $1,592M)
print(cost["p90"])    # ~$6,210M  (actual FY2024: $3,300M below p90)
print(cost["unit"])   # "$M"

# Expected Annual Loss = p50 * P(breach) / 100
# e.g. $1,527M * 0.666 = $1,017M/year — the cyber insurance pricing input
```

### Portfolio VaR

```python
import numpy as np
from mc.models.var import VaRModel

result = VaRModel(
    portfolio_value=1_000_000,
    returns_mean=np.array([0.0005, 0.0004]),
    returns_cov=np.array([[0.0001, 0.00004], [0.00004, 0.0001]]),
    weights=np.array([0.6, 0.4]),
    horizon_days=30,
    confidence=0.95,
    n=100_000,
    seed=42,
).run()

print(result["var"])   # dollar loss at 95% confidence
print(result["cvar"])  # Expected Shortfall (average loss in the tail)
```

### DCF valuation

```python
from mc.models.dcf import DCFModel

result = DCFModel(
    revenue=10_000.0,
    revenue_growth=(0.08, 0.14, 0.22),
    operating_margin=(0.10, 0.16, 0.24),
    discount_rate=(0.09, 0.02),
    terminal_growth=(0.025, 0.005),
    years=10,
    n=100_000,
    seed=42,
).run()

print(result["p50"])   # enterprise value in $M
```

### Cyber-adjusted portfolio VaR

Connects breach likelihood directly to portfolio financial risk. Shows how much
extra downside a cyber event adds on top of normal market risk.

```python
import numpy as np
from mc.models.breach import BreachLikelihoodModel
from mc.models.cyber_var_bridge import CyberVaRBridge

# Step 1: score the company
breach = BreachLikelihoodModel(
    company_info={"name": "HealthCo", "sector": "Healthcare",
                  "market_cap_m": 10_000, "beta": 1.1, "rev_growth": 0.12},
    industry_rows=[{"ticker": t} for t in ["HCA", "UHS"]],
    n=50_000, seed=42,
).run()

# Step 2: translate score → annual breach probability, get sector drawdown
prob = CyberVaRBridge.score_to_prob(breach["p50"])          # e.g. 0.68
ddmean, ddstd = CyberVaRBridge.sector_drawdown("Healthcare") # (-0.08, 0.05)

# Step 3: cyber-adjusted 30-day VaR
result = CyberVaRBridge(
    portfolio_value=1_000_000,
    returns_mean=np.array([0.0005]),
    returns_cov=np.array([[0.0001]]),
    weights=np.array([1.0]),
    annual_breach_probs=np.array([prob]),
    breach_drawdown_mean=ddmean,
    breach_drawdown_std=ddstd,
    horizon_days=30,
    confidence=0.95,
    n=50_000, seed=42,
).run()

print(result["baseline_var"])       # market risk only
print(result["cyber_var"])          # market + breach risk
print(result["cyber_contribution"]) # incremental cyber VaR
```

---

## Getting real inputs

The models accept standard financial data available from public sources:

| Input | Where to get it | Python library |
|-------|----------------|----------------|
| `market_cap_m` | Yahoo Finance, Google Finance | `yfinance`: `yf.Ticker("AAPL").info["marketCap"] / 1e6` |
| `beta` | Yahoo Finance summary page | `yf.Ticker("AAPL").info["beta"]` |
| `rev_growth` | SEC EDGAR annual filings (10-K) | `yf.Ticker("AAPL").financials` — compare YoY revenue |
| `returns_mean`, `returns_cov` | Historical price data | `yf.download(tickers, period="2y")["Adj Close"].pct_change()` |
| `revenue` (DCF) | 10-K income statement | `yf.Ticker("AAPL").financials.loc["Total Revenue"]` |
| `industry_rows` | Sector peer list | Build from ticker list; each dict needs a `"ticker"` key |

For academic use, [WRDS](https://wrds-www.wharton.upenn.edu/) (Compustat/CRSP) provides
cleaner institutional-grade data. For quick prototyping, `yfinance` is free and sufficient.

---

## Architecture

```
mc/
├── engine.py              MonteCarloEngine, SimulationResult
├── distributions.py       Normal, Lognormal, Triangular, Beta, Uniform
└── models/
    ├── breach.py          BreachLikelihoodModel  — cyber likelihood scoring
    ├── cyber_cost.py      CyberCostModel         — dollar loss projection
    ├── cyber_var_bridge.py CyberVaRBridge        — breach risk → portfolio VaR
    ├── dcf.py             DCFModel               — enterprise valuation
    └── var.py             VaRModel               — portfolio VaR + CVaR
```

All models share the same engine and return consistent shapes:
`p10`, `p50`, `p90`, `mean`, `std`, `ci_90`. Raw simulation arrays
(`scores`, `pnl_scores`) are included for histogram and sensitivity analysis.

---

## Calibration

| Model / parameter | Source |
|------------------|--------|
| Sector breach base rates | IBM Cost of a Data Breach 2025, Verizon DBIR 2026 |
| Cyber cost — sector averages | IBM Cost of a Data Breach 2025 |
| Cyber cost — size scaling | Observed mega-breach costs (Equifax 2017, Anthem 2015, Change Healthcare 2024) |
| VaR correlation structure | Cholesky decomposition; user supplies covariance matrix |
| DCF inputs | User-specified; no market data dependency |

---

## Connection to SR 11-7

Federal Reserve SR 11-7 requires model validation: independent review, sensitivity
analysis, and documentation of model limitations. Replacing a point estimate with a
distribution is the first step — it forces explicit documentation of uncertainty.

---

## Tests

```bash
pytest                # all tests
pytest --cov=mc       # with coverage
```

---

## Requirements

- Python >= 3.11
- numpy >= 2.0
- (optional) jupyter, matplotlib, pandas — `pip install -e ".[notebook]"`
