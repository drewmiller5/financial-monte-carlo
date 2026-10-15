"""Fit FittedCyberCostModel on validation/breach_cases.csv and test it honestly.

Run from the repo root:
    py validation/fit_cost.py            # IBM sector baseline (default), writes mc/models/fitted_cost_params.json
    py validation/fit_cost.py --no-ibm   # fit on our data alone (if IBM is unavailable)
    py validation/fit_cost.py --fx GBP=1.25

Each case enters the likelihood by what we actually know about its cost:
  exact (low == high)           -> density at that value
  range (low < high)            -> probability the cost is inside the range
  lower bound (lower_bound=yes) -> probability the cost is at least `low`

Tests (every prediction is made by a model that never saw that case):
  1. leave-one-out over all 51 cases
  2. time holdout: fit on breaches disclosed before 2024, predict 2024-2025
Both are compared with the current IBM-staircase CyberCostModel.
"""
import argparse
import datetime
import json
import math
import pathlib
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest_cost import N, ROOT, SEED, load, load_fx, parse_fx, score, summarize  # noqa: E402

sys.path.insert(0, str(ROOT))
from mc.models.cyber_cost import CyberCostModel  # noqa: E402
from mc.models.fitted_cost import PARAMS_PATH, REF_MCAP_M, FittedCyberCostModel, ibm_offset  # noqa: E402

HOLDOUT_FROM = "2024-01"


def arrays(rows, use_ibm):
    x = np.array([math.log(r["mcap_m"] / REF_MCAP_M) for r in rows])
    off = np.array([ibm_offset(r["sector"], use_ibm) for r in rows])
    lo = np.log([r["lo"] for r in rows])
    hi = np.log([r["hi"] for r in rows])
    kind = np.array([2 if r["lower_bound"] == "yes" else (0 if r["lo"] == r["hi"] else 1) for r in rows])
    return x, off, lo, hi, kind


def nll(theta, x, off, lo, hi, kind):
    a, b, ln_s = theta
    s = math.exp(ln_s)
    mu = off + a + b * x
    zlo, zhi = (lo - mu) / s, (hi - mu) / s
    ll = np.where(kind == 0, stats.norm.logpdf(zlo) - ln_s,
         np.where(kind == 1, np.log(np.clip(stats.norm.cdf(zhi) - stats.norm.cdf(zlo), 1e-300, None)),
                  stats.norm.logsf(zlo)))
    return -ll.sum()


def hessian(f, t, h=1e-4):
    k = len(t)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(t + e_i + e_j) - f(t + e_i - e_j) - f(t - e_i + e_j) + f(t - e_i - e_j)) / (4 * h * h)
    return H


def fit(rows, use_ibm):
    data = arrays(rows, use_ibm)
    f = lambda t: nll(t, *data)  # noqa: E731
    start = np.array([np.mean((data[2] + data[3]) / 2 - data[1]), 0.3, 0.0])
    res = optimize.minimize(f, start, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20_000})
    cov = np.linalg.inv(hessian(f, res.x))
    assert res.success and np.all(np.linalg.eigvalsh(cov) > 0), "fit did not converge"
    return {"use_ibm": use_ibm, "theta": res.x.tolist(), "cov": cov.tolist()}


def predict(params, r):
    return FittedCyberCostModel(r["mcap_m"], r["sector"], params=params, n=N // 5, seed=SEED).run()


def level_shift(rows, use_ibm):
    """Median leave-one-out log error on complete-cost cases (no lower bound, no approximated inputs),
    using only `rows`. Negative = the model's medians run high and should come down."""
    complete = [i for i, r in enumerate(rows) if r["lower_bound"] == "no" and r["approx"] == "no"]
    errs = []
    for i in complete:
        q = predict(fit([r for j, r in enumerate(rows) if j != i], use_ibm), rows[i])
        errs.append(math.log(math.sqrt(rows[i]["lo"] * rows[i]["hi"]) / q["p50"]))
    return float(np.median(errs))


def old_model(r):
    return CyberCostModel(market_cap_m=r["mcap_m"], sector=r["sector"], n=N // 5, seed=SEED).run()


def evaluate(rows, train_idx_for, test_idx, use_ibm, shift=0.0):
    """Fit on train_idx_for(i) for each test case i; return {i: (hit, err)} for fitted and old models."""
    fitted, old = {}, {}
    for i in test_idx:
        params = dict(fit([rows[j] for j in train_idx_for(i)], use_ibm), level_shift=shift)
        r = rows[i]
        lb = r["lower_bound"] == "yes"
        p, o = predict(params, r), old_model(r)
        fitted[i] = score(r["lo"], r["hi"], p["p10"], p["p50"], p["p90"], lb)
        old[i] = score(r["lo"], r["hi"], o["p10"], o["p50"], o["p90"], lb)
    return fitted, old


def report(title, rows, results, subsets):
    print(f"\n{title}")
    for label, idx in subsets:
        idx = [k for k in idx if k in results[0][1]]
        if not idx:
            continue
        print(f"  {label} (n={len(idx)})")
        for name, res in results:
            print("    " + summarize(name, [res[k] for k in idx]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-ibm", action="store_true", help="fit without the IBM sector baseline")
    p.add_argument("--fx", action="append", default=[], metavar="CUR=RATE")
    args = p.parse_args(argv)
    to_usd, _ = load_fx(parse_fx(args.fx))
    rows = load(to_usd)
    n = len(rows)

    # 1. full fit, both variants, for the parameter report
    print(f"Fitted on {n} disclosed breaches (ln cost = IBM sector offset + a + b*ln(mcap/$10B) + sigma*z)\n")
    full = {}
    for use_ibm in (True, False):
        prm = fit(rows, use_ibm)
        full[use_ibm] = prm
        a, b, ln_s = prm["theta"]
        se = np.sqrt(np.diag(prm["cov"]))
        ex = FittedCyberCostModel(10_000, "Healthcare", params=prm, n=N, seed=SEED).run()
        print(f"  {'With IBM baseline' if use_ibm else 'No IBM (data only)':<20} a={a:+.2f}±{se[0]:.2f}  "
              f"b={b:.2f}±{se[1]:.2f} (10x bigger company -> {10**b:.1f}x cost)  sigma={math.exp(ln_s):.2f}  "
              f"| $10B healthcare: p10 ${ex['p10']:,.0f}M  p50 ${ex['p50']:,.0f}M  p90 ${ex['p90']:,.0f}M")

    use_ibm = not args.no_ibm
    shift = level_shift(rows, use_ibm)
    params = dict(full[use_ibm], level_shift=shift, n_cases=n, fitted_on="validation/breach_cases.csv",
                  fitted=datetime.date.today().isoformat(), ref_mcap_m=REF_MCAP_M)
    PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Median correction from training residuals: x{math.exp(shift):.2f}")
    print(f"  Saved {'IBM-baseline' if use_ibm else 'data-only'} parameters -> {PARAMS_PATH.relative_to(ROOT)}")

    clean = [k for k in range(n) if rows[k]["approx"] == "no"]
    complete = [k for k in clean if rows[k]["lower_bound"] == "no"]
    subsets = [("All cases", range(n)), ("No approximated inputs", clean), ("No approx, full costs only", complete)]

    # 2. leave-one-out (the correction is learned on all 51, so its row is mildly optimistic)
    others = lambda i: [j for j in range(n) if j != i]  # noqa: E731
    loo_ibm, old = evaluate(rows, others, range(n), True)
    loo_sft, _ = evaluate(rows, others, range(n), True, shift)
    loo_raw, _ = evaluate(rows, others, range(n), False)
    report("TEST 1 - LEAVE-ONE-OUT (target coverage 80%)", rows,
           [("Old IBM staircase", old), ("Fitted, IBM baseline", loo_ibm),
            ("Fitted, IBM + correction", loo_sft), ("Fitted, no IBM", loo_raw)], subsets)

    # 3. time holdout: the correction is learned from the pre-2024 cases only
    train = [j for j in range(n) if rows[j]["disclosed"] < HOLDOUT_FROM]
    test = [j for j in range(n) if rows[j]["disclosed"] >= HOLDOUT_FROM]
    t_shift = level_shift([rows[j] for j in train], True)
    t_ibm, t_old = evaluate(rows, lambda i: train, test, True)
    t_sft, _ = evaluate(rows, lambda i: train, test, True, t_shift)
    t_raw, _ = evaluate(rows, lambda i: train, test, False)
    report(f"TEST 2 - TIME HOLDOUT: fit on {len(train)} cases before {HOLDOUT_FROM}, predict {len(test)} later cases",
           rows, [("Old IBM staircase", t_old), ("Fitted, IBM baseline", t_ibm),
                  (f"Fitted, IBM + correction x{math.exp(t_shift):.2f}", t_sft), ("Fitted, no IBM", t_raw)],
           [("Held-out cases", test)])
    prm = dict(fit([rows[j] for j in train], True), level_shift=t_shift)
    print(f"\n  {'Held-out case':<34}{'p10':>8}{'p50':>8}{'p90':>9}{'actual $M':>14}  in range")
    for i in test:
        r, q = rows[i], predict(prm, rows[i])
        actual = f"{r['lo']:,.0f}" if r["lo"] == r["hi"] else f"{r['lo']:,.0f}-{r['hi']:,.0f}"
        flag = "+" if r["lower_bound"] == "yes" else ""
        print(f"  {r['company'] + flag:<34}{q['p10']:>8,.0f}{q['p50']:>8,.0f}{q['p90']:>9,.0f}{actual:>14}  "
              f"{'yes' if score(r['lo'], r['hi'], q['p10'], q['p50'], q['p90'], flag == '+')[0] else 'NO'}")
    print("  + lower bound")


if __name__ == "__main__":
    main()
