# Contributing

Bug reports, model improvements, and data source updates are welcome.

## Quick start

```bash
git clone https://github.com/drewmiller5/financial-monte-carlo
cd financial-monte-carlo
pip install -e ".[dev]"
pytest
```

## What's in scope

- Bug fixes in model math, edge-case handling, or distribution logic
- Updated calibration data (new IBM DBIR / Verizon DBIR reports)
- New Monte Carlo models that fit the engine's `SimulationResult` interface
- Streamlit UI improvements (app.py)
- Documentation and example notebooks

## What's out of scope

- Replacing the Monte Carlo approach with deterministic point estimates
- Adding financial advice, predictions, or guarantees to outputs
- External API integrations that introduce runtime data dependencies

## Adding a model

1. Create `mc/models/your_model.py` — subclass or mirror the `DCFModel` pattern
2. Return a dict with at minimum `p10`, `p50`, `p90`, `mean`, `std`, `ci_90`
3. Export from `mc/models/__init__.py`
4. Add tests in `tests/models/test_your_model.py`
5. Run `pytest --cov=mc --cov-fail-under=75`

## Updating calibration data

Sector breach rates live in `mc/models/breach.py` (`SECTOR_RISK`, `CALIBRATION_YEAR`).
Cost multipliers live in `mc/models/cyber_cost.py`. When the annual IBM Cost of a Data
Breach report publishes, update both dicts and bump `CALIBRATION_YEAR`.

## Tests

All PRs must pass CI (Python 3.11, 3.12, 3.13) and maintain ≥75% coverage.

```bash
pytest --tb=short -q
pytest --cov=mc --cov-fail-under=75 -q
```

## Code style

- Black-compatible formatting (no formatter enforced, just be consistent)
- Type hints on all public functions
- No comments explaining what the code does — only why when non-obvious

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
