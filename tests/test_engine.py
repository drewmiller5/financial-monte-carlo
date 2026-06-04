import numpy as np
import pytest
from mc.engine import MonteCarloEngine, SimulationResult


def test_simulation_result_percentiles():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = SimulationResult(values=values, n=10)
    assert result.p50 == pytest.approx(5.5, abs=0.1)
    assert result.p10 < result.p50
    assert result.p50 < result.p90


def test_simulation_result_probability_above():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    result = SimulationResult(values=values, n=10)
    assert result.probability_above(50.0) == pytest.approx(0.5, abs=0.01)


def test_simulation_result_probability_below():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    result = SimulationResult(values=values, n=10)
    assert result.probability_below(50.0) == pytest.approx(0.4, abs=0.01)


def test_simulation_result_confidence_interval():
    values = np.linspace(0, 100, 1000)
    result = SimulationResult(values=values, n=1000)
    lo, hi = result.confidence_interval(5, 95)
    assert lo == pytest.approx(5.0, abs=1.0)
    assert hi == pytest.approx(95.0, abs=1.0)


def test_simulation_result_summary_keys():
    values = np.random.default_rng(0).normal(50, 10, 1000)
    result = SimulationResult(values=values, n=1000)
    summary = result.summary()
    assert set(summary.keys()) == {"n", "mean", "std", "p10", "p50", "p90", "ci_90"}


def test_engine_run_callable():
    engine = MonteCarloEngine(n=1000, seed=42)
    counter = {"v": 0.0}

    def model():
        counter["v"] += 1.0
        return counter["v"]

    result = engine.run(model)
    assert result.n == 1000
    assert result.values.shape == (1000,)


def test_engine_run_vectorized():
    engine = MonteCarloEngine(n=1000, seed=42)

    def model_vec(n):
        return engine.rng.normal(50, 10, n)

    result = engine.run_vectorized(model_vec)
    assert result.n == 1000
    assert abs(result.mean - 50) < 2.0


def test_engine_seed_reproducibility():
    def constant_model():
        return 42.0

    r1 = MonteCarloEngine(n=100, seed=7).run(constant_model)
    r2 = MonteCarloEngine(n=100, seed=7).run(constant_model)
    np.testing.assert_array_equal(r1.values, r2.values)
