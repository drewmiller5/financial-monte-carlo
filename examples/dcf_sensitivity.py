"""
Example: Monte Carlo DCF — tech company under three growth scenarios.

Shows how replacing point estimates with distributions produces
a valuation range instead of false precision.
"""
from mc.models.dcf import DCFModel

scenarios = {
    "Bear":  dict(revenue_growth=(0.03, 0.06, 0.10), operating_margin=(0.10, 0.14, 0.18), discount_rate=(0.12, 0.02)),
    "Base":  dict(revenue_growth=(0.08, 0.14, 0.20), operating_margin=(0.15, 0.20, 0.26), discount_rate=(0.10, 0.02)),
    "Bull":  dict(revenue_growth=(0.18, 0.25, 0.35), operating_margin=(0.22, 0.28, 0.35), discount_rate=(0.09, 0.01)),
}

COMMON = dict(revenue=500.0, terminal_growth=(0.025, 0.005), years=10, n=10_000, seed=42)

print("DCF Valuation Scenarios — TechCo Inc. (Revenue: $500M)")
print(f"{'Scenario':<8} {'p10 ($M)':>10} {'p50 ($M)':>10} {'p90 ($M)':>10} {'Std Dev':>10}")
print("-" * 55)

for name, overrides in scenarios.items():
    result = DCFModel(**{**COMMON, **overrides}).run()
    print(f"{name:<8} {result['p10']:>10,.0f} {result['p50']:>10,.0f} {result['p90']:>10,.0f} {result['std']:>10,.0f}")
