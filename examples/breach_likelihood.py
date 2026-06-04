"""
Example: Breach likelihood comparison — Healthcare vs Utilities company.

Demonstrates how Monte Carlo outputs a distribution (p10/p50/p90)
instead of a single point estimate.
"""
from mc.models.breach import BreachLikelihoodModel

healthcare = {
    "name": "MedCorp Health Systems",
    "sector": "Healthcare",
    "market_cap_m": 8500,
    "beta": 1.1,
    "rev_growth": 0.12,
}

utilities = {
    "name": "GridCo Power",
    "sector": "Utilities",
    "market_cap_m": 8500,
    "beta": 0.5,
    "rev_growth": 0.03,
}

industry_rows = [{"ticker": f"PEER{i}"} for i in range(4)]

for info in [healthcare, utilities]:
    result = BreachLikelihoodModel(info, industry_rows, n=10_000, seed=42).run()
    print(f"\n{info['name']}")
    print(f"  Tier:              {result['tier']}")
    print(f"  Score (p50):       {result['p50']}/100")
    print(f"  90% CI:            {result['ci_90'][0]}–{result['ci_90'][1]}")
    print(f"  Prob Critical:     {result['prob_critical_pct']}%")
    print(f"  Narrative:         {result['narrative']}")
