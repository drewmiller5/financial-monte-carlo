"""Score the shipped FittedCyberCostModel on the held-out TEST set.

    py validation/test_holdout.py

Training data: validation/breach_cases.csv  (read by fit_cost.py)
Test data:     validation/test_cases.csv    (read ONLY here; never used for fitting)

Add a new case to test_cases.csv before looking at what the model says about it.
A test case may only move into training after it has been scored here, and then it stops being a test.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backtest_cost import N, ROOT, SEED, load, load_fx, score, summarize  # noqa: E402

sys.path.insert(0, str(ROOT))
from mc.models.fitted_cost import FittedCyberCostModel  # noqa: E402


def main():
    to_usd, _ = load_fx({})
    test = load(to_usd, "test_cases.csv")
    train = {r["company"] for r in load(to_usd)}
    leaked = [r["company"] for r in test if r["company"] in train]
    assert not leaked, f"test cases also in training data: {leaked}"

    print(f"{'Test case':<28}{'p10':>8}{'p50':>8}{'p90':>9}{'actual $M':>14}  in range")
    results = []
    for r in test:
        lb = r["lower_bound"] == "yes"
        q = FittedCyberCostModel(r["mcap_m"], r["sector"], n=N, seed=SEED).run()
        hit, err = score(r["lo"], r["hi"], q["p10"], q["p50"], q["p90"], lb)
        results.append((hit, err))
        actual = f"{r['lo']:,.0f}" if r["lo"] == r["hi"] else f"{r['lo']:,.0f}-{r['hi']:,.0f}"
        print(f"{r['company'] + ('+' if lb else ''):<28}{q['p10']:>8,.0f}{q['p50']:>8,.0f}{q['p90']:>9,.0f}"
              f"{actual:>14}  {'yes' if hit else 'NO'}")
    print("+ lower bound\n")
    print(summarize(f"TEST SET (n={len(test)})", results))


if __name__ == "__main__":
    main()
