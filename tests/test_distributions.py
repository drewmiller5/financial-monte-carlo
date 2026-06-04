import numpy as np
import pytest
from mc.distributions import Normal, Lognormal, Triangular, Beta, Uniform


N = 50_000


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_normal_shape(rng):
    dist = Normal(mean=0.5, std=0.1)
    samples = dist.sample(rng, N)
    assert samples.shape == (N,)


def test_normal_mean_std_converge(rng):
    dist = Normal(mean=0.5, std=0.1)
    samples = dist.sample(rng, N)
    assert abs(samples.mean() - 0.5) < 0.01
    assert abs(samples.std() - 0.1) < 0.01


def test_lognormal_shape(rng):
    dist = Lognormal(mean=0.0, std=0.5)
    samples = dist.sample(rng, N)
    assert samples.shape == (N,)
    assert (samples > 0).all()  # lognormal is always positive


def test_triangular_bounds(rng):
    dist = Triangular(low=0.2, mode=0.5, high=0.8)
    samples = dist.sample(rng, N)
    assert samples.min() >= 0.2
    assert samples.max() <= 0.8


def test_triangular_mode_peak(rng):
    dist = Triangular(low=0.0, mode=0.5, high=1.0)
    samples = dist.sample(rng, N)
    # Most samples should cluster near mode
    assert abs(samples.mean() - 0.5) < 0.05


def test_beta_range(rng):
    dist = Beta(alpha=2.0, beta_param=5.0)
    samples = dist.sample(rng, N)
    assert (samples >= 0).all()
    assert (samples <= 1).all()


def test_uniform_bounds(rng):
    dist = Uniform(low=10.0, high=20.0)
    samples = dist.sample(rng, N)
    assert samples.min() >= 10.0
    assert samples.max() <= 20.0


def test_uniform_mean(rng):
    dist = Uniform(low=0.0, high=1.0)
    samples = dist.sample(rng, N)
    assert abs(samples.mean() - 0.5) < 0.01


def test_triangular_invalid_params():
    with pytest.raises(ValueError):
        Triangular(low=0.8, mode=0.5, high=0.2)


def test_distributions_default_n(rng):
    # Verify the default n=1 argument works for all classes
    assert Normal(0.5, 0.1).sample(rng).shape == (1,)
    assert Lognormal(0.0, 0.5).sample(rng).shape == (1,)
    assert Triangular(0.2, 0.5, 0.8).sample(rng).shape == (1,)
    assert Beta(2.0, 5.0).sample(rng).shape == (1,)
    assert Uniform(10.0, 20.0).sample(rng).shape == (1,)
