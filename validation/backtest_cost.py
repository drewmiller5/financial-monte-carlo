"""Backtest CyberCostModel against breaches with publicly disclosed costs.

Run from the repo root:  py validation/backtest_cost.py

For each case the model gets only sector + pre-breach market cap (breach_score=None,
so no likelihood multiplier). A case "hits" if any part of its disclosed cost range
[cost_low, cost_high] overlaps the model's p10-p90 interval. A well-calibrated
80% interval should hit ~80% of cases.

The model is compared with two leave-one-out baselines fitted on the *other* cases:
  - constant: median and 10/90 quantiles of the other cases' log costs
  - size regression: log(cost) ~ a + b*log(mcap), residual spread for the interval
If the model can't beat these, its IBM/size calibration adds nothing.
"""
import csv
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mc.models.cyber_cost import CyberCostModel  # noqa: E402

N, SEED = 100_000, 42
Z90 = 1.2816  # standard normal 90th percentile


def load_fx(overrides):
    """Recommended rates from validation/fx_rates.csv (Fed G.5A annual averages), keyed (currency, year).
    overrides: {currency: usd_per_unit} from --fx, applied to every year of that currency."""
    with open(ROOT / "validation" / "fx_rates.csv", encoding="utf-8") as f:
        fx = {(r["currency"], r["year"]): float(r["usd_per_unit"]) for r in csv.DictReader(f)}
    used = {}

    def to_usd(amount, currency, year):
        if currency in ("", "USD"):
            return amount
        rate = overrides.get(currency, fx.get((currency, year)))
        if rate is None:
            raise SystemExit(f"No FX rate for {currency} {year}: add it to validation/fx_rates.csv or pass --fx {currency}=<usd per unit>")
        used[f"{currency} {year}" + (" (override)" if currency in overrides else "")] = rate
        return amount * rate

    return to_usd, used


def load(to_usd, filename="breach_cases.csv"):
    with open(ROOT / "validation" / filename, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["mcap_m"] = to_usd(float(r["mcap_mn"]), r["mcap_currency"], r["mcap_year"])
        r["lo"] = to_usd(float(r["cost_low_mn"]), r["currency"], r["cost_year"])
        r["hi"] = to_usd(float(r["cost_high_mn"]), r["currency"], r["cost_year"])
        assert 0 < r["lo"] <= r["hi"], r["company"]
        r["mid"] = math.sqrt(r["lo"] * r["hi"])  # geometric midpoint of the disclosed range
    return rows


def score(lo, hi, p10, p50, p90, lower_bound=False):
    """Overlap of disclosed range with [p10, p90], and log10 error of the midpoint vs p50.
    A lower bound (true cost >= lo) only misses if the model's p90 is below it."""
    hit = lo <= p90 if lower_bound else (hi >= p10 and lo <= p90)
    return hit, math.log10(math.sqrt(lo * hi) / p50)


def summarize(name, results):
    hits = [h for h, _ in results]
    errs = np.array([e for _, e in results])
    return (f"{name:<22} coverage {sum(hits):>2}/{len(hits)} ({100*np.mean(hits):3.0f}%)   "
            f"median log10 err {np.median(errs):+.2f} ({10**np.median(errs):.2f}x)   "
            f"mean |log10 err| {np.mean(np.abs(errs)):.2f} ({10**np.mean(np.abs(errs)):.1f}x typical miss)")


def parse_fx(args):
    """['GBP=1.25', 'EUR=1.1'] -> {'GBP': 1.25, 'EUR': 1.1}"""
    out = {}
    for a in args:
        cur, _, rate = a.partition("=")
        out[cur.strip().upper()] = float(rate)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fx", action="append", default=[], metavar="CUR=RATE",
                   help="override USD per unit for a currency, all years (e.g. --fx GBP=1.25). Default: validation/fx_rates.csv")
    to_usd, fx_used = load_fx(parse_fx(p.parse_args(argv).fx))
    rows = load(to_usd)
    print("FX (USD per unit): " + ", ".join(f"{k}={v:g}" for k, v in sorted(fx_used.items())) + "\n")
    model_res, const_res, reg_res = [], [], []

    print(f"{'Case':<26}{'mcap $B':>9}{'p10':>9}{'p50':>9}{'p90':>10}{'actual $M':>15}  {'pctile':>6}  in p10-p90")
    print("-" * 104)
    for i, r in enumerate(rows):
        res = CyberCostModel(market_cap_m=r["mcap_m"], sector=r["sector"], n=N, seed=SEED).run()
        sims = res["scores"]
        pct = 100 * float(np.mean(sims <= r["mid"]))
        lb = r["lower_bound"] == "yes"
        hit, err = score(r["lo"], r["hi"], res["p10"], res["p50"], res["p90"], lb)
        model_res.append((hit, err))

        # leave-one-out baselines on the other cases
        others = [o for j, o in enumerate(rows) if j != i]
        y = np.log([o["mid"] for o in others])
        q10, q50, q90 = np.exp(np.quantile(y, [0.1, 0.5, 0.9]))
        const_res.append(score(r["lo"], r["hi"], q10, q50, q90, lb))

        x = np.log([o["mcap_m"] for o in others])
        b, a = np.polyfit(x, y, 1)
        s = np.std(y - (a + b * x), ddof=2)
        m = a + b * math.log(r["mcap_m"])
        reg_res.append(score(r["lo"], r["hi"], math.exp(m - Z90 * s), math.exp(m), math.exp(m + Z90 * s), lb))

        actual = f"{r['lo']:,.0f}" if r["lo"] == r["hi"] else f"{r['lo']:,.0f}-{r['hi']:,.0f}"
        flags = ("*" if r["in_sample"] == "yes" else "") + ("+" if r["lower_bound"] == "yes" else "")
        print(f"{r['company'] + flags:<26}{r['mcap_m']/1000:>9.1f}{res['p10']:>9,.0f}{res['p50']:>9,.0f}"
              f"{res['p90']:>10,.0f}{actual:>15}  {pct:>5.0f}%  {'yes' if hit else 'NO'}")

    print("\n* in-sample (used to tune the size multiplier)   + lower bound (true cost likely higher)\n")
    oos = [k for k, r in enumerate(rows) if r["in_sample"] == "no"]
    clean = [k for k in oos if rows[k]["approx"] == "no"]
    complete = [k for k in clean if rows[k]["lower_bound"] == "no"]
    print(f"Underestimates (actual above model p90): {sum(r['lo'] > CyberCostModel(market_cap_m=r['mcap_m'], sector=r['sector'], n=N, seed=SEED).run()['p90'] for r in rows)}"
          f"   Overestimates (actual below model p10): {sum(r['hi'] < CyberCostModel(market_cap_m=r['mcap_m'], sector=r['sector'], n=N, seed=SEED).run()['p10'] for r in rows)}\n")
    for label, idx in (("ALL CASES", range(len(rows))), ("OUT-OF-SAMPLE ONLY", oos),
                       ("OUT-OF-SAMPLE, NO APPROXIMATED INPUTS", clean),
                       ("OUT-OF-SAMPLE, NO APPROX, FULL COSTS ONLY (no lower bounds)", complete)):
        print(f"{label} (n={len(idx)}), target coverage for an 80% interval = 80%")
        for name, res in (("Model", model_res), ("Baseline: constant", const_res), ("Baseline: size regr.", reg_res)):
            print("  " + summarize(name, [res[k] for k in idx]))
        print()

    # self-check: score() logic
    assert score(5, 5, 1, 5, 10) == (True, 0.0)
    assert score(20, 30, 1, 5, 10)[0] is False
    assert score(5, 5, 10, 20, 40, lower_bound=True)[0] is True  # prediction above a lower bound is fine
    assert score(50, 50, 10, 20, 40, lower_bound=True)[0] is False
    assert parse_fx(["gbp=1.25"]) == {"GBP": 1.25}


if __name__ == "__main__":
    main()
