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
| `BreachLikelihoodModel` | How likely is this company to suffer a material breach? | Score 0–100 (uncalibrated index, not yet validated), tier, 90% CI |
| `FittedCyberCostModel` | If a material breach occurs, what does it cost? | Dollar loss p10/p50/p90 in $M (backtested; see Validation) |
| `CyberCostModel` | Original IBM-tier cost model, kept for comparison | Dollar loss p10/p50/p90 in $M (underestimates ~3–4×) |
| `CyberVaRBridge` | How does breach risk change the portfolio's financial downside? | Baseline VaR, cyber-adjusted VaR, and incremental cyber contribution |
| `DCFModel` | What is this company worth under uncertainty? | Enterprise value range in $M |
| `VaRModel` | What is the portfolio's downside at a given confidence level? | Dollar VaR + CVaR |

---

## Validation (in progress)

> **Status:** The cyber breach **cost** model has been rebuilt and backtested against real disclosed breach costs. Results so far are encouraging but the test set is still small and growing — treat the numbers below as preliminary. The breach **likelihood** model has **not** been validated yet (see Limits).

### What changed

The original cost model scaled IBM's average breach cost (Cost of a Data Breach 2025) by hand-picked company-size tiers. A backtest on real breaches showed it **underestimated material public-company breaches by about 3–4×**, so it was replaced.

The current model keeps IBM's sector costs as the starting point and fits the rest to data:

```
ln(cost) = IBM sector baseline + a + b·ln(market cap / $10B) + correction + noise
```

- **Data:** 51 public-company breaches (2007–2025) with costs disclosed in SEC filings, annual reports or reporting that cites them. Every case lists its source in `validation/breach_cases.csv`.
- **Inputs:** sector and market cap at the year-end *before* the breach was disclosed — only information available beforehand.
- **Partial costs:** 22 cases only disclose part of the cost (e.g. settlements). These are treated as lower bounds, not as the full cost.
- **Currencies:** non-USD costs are converted at Federal Reserve annual average rates (`validation/fx_rates.csv`).
- **Correction:** the fitted median ran high on cases with complete costs; a level correction learned from the training data brings it back in line.
- **Without IBM:** the model can be refit on the breach data alone (`--no-ibm`) and performs almost as well.

### Results so far

Every prediction below was made by a model that did not see that case.

| Test | Old model | Current model |
|---|---|---|
| Leave-one-out, 27 cases with complete costs | 48% inside the 80% range; median ~4× too low | 89% inside the 80% range; median within ~1.0× on average; typical miss ~2× |
| Time holdout: fit on breaches before 2024, predict 9 from 2024–25 | 5 of 9 inside the range | 9 of 9 inside the range; typical miss ~2× |
| Blind test set (separate file, never used for fitting) | — | 5 of 5 inside the range — **only 5 cases so far** |

How to read this:
- The **80% range (p10–p90)** is the main output. If the model is well calibrated, about 80% of actual costs should land inside it.
- **"Typical miss ~2×"** means the median estimate is usually within a factor of two of the actual cost — e.g. a $100M breach is typically estimated somewhere between $50M and $200M.
- The leave-one-out correction was learned on the same 51 cases it is scored on, so that row is mildly optimistic. The time holdout learns the correction from pre-2024 data only.
- Four of the five blind test cases were reviewed before the median correction was added; only one is a clean blind test of the current version. **The test set is being expanded** before its results are treated as conclusive.

### Limits

- **Small sample.** 51 training cases and 5 test cases. More cases will change the numbers.
- **Material breaches only.** Fitted on public companies' disclosed breaches; it does not describe small incidents.
- **Cost definitions vary.** Some disclosures include lost revenue, some are net of insurance, some are company estimates.
- **Likelihood model not validated.** The 0–100 breach likelihood score is a weighted index, not a calibrated probability. Validating it requires data on companies that were *not* breached.
- Not financial or insurance advice.

### Reproduce

```bash
pip install -e ".[validation]"
python validation/fit_cost.py        # refit + leave-one-out + time holdout
python validation/test_holdout.py    # score the blind test set
python validation/backtest_cost.py   # old model vs. simple baselines
```

---

## Quick start

```bash
git clone https://github.com/drewmiller5/financial-monte-carlo
cd financial-monte-carlo
pip install -e ".[dev]"        # core + pytest
pip install -e ".[notebook]"   # + jupyter, matplotlib, pandas
pip install -e ".[validation]" # + scipy, to refit and backtest the cost model
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
from mc.models.fitted_cost import FittedCyberCostModel

cost = FittedCyberCostModel(
    market_cap_m=10_000,   # $10B company
    sector="Healthcare",
    n=100_000,
    seed=42,
).run()

print(cost["p10"], cost["p50"], cost["p90"])   # ~42  ~139  ~467  ($M)
print(cost["unit"])                            # "$M"
```

Expected annual loss is intentionally not shown: the likelihood score is an uncalibrated index, not a probability.

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
