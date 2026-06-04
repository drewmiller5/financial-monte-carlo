import numpy as np
import pytest
from mc.distributions import Normal, Lognormal, Triangular, Beta, Uniform


RNG = np.random.default_rng(42)
N = 50_000


def test_normal_shape():
    dist = Normal(mean=0.5, std=0.1)
    samples = dist.sample(RNG, N)
    assert samples.shape == (N,)


def test_normal_mean_std_converge():
    dist = Normal(mean=0.5, std=0.1)
    samples = dist.sample(RNG, N)
    assert abs(samples.mean() - 0.5) < 0.01
    assert abs(samples.std() - 0.1) < 0.01


def test_lognormal_shape():
    dist = Lognormal(mean=0.0, std=0.5)
    samples = dist.sample(RNG, N)
    assert samples.shape == (N,)
    assert (samples > 0).all()  # lognormal is always positive


def test_triangular_bounds():
    dist = Triangular(low=0.2, mode=0.5, high=0.8)
    samples = dist.sample(RNG, N)
    assert samples.min() >= 0.2
    assert samples.max() <= 0.8


def test_triangular_mode_peak():
    dist = Triangular(low=0.0, mode=0.5, high=1.0)
    samples = dist.sample(RNG, N)
    # Most samples should cluster near mode
    assert abs(samples.mean() - 0.5) < 0.05


def test_beta_range():
    dist = Beta(alpha=2.0, beta_param=5.0)
    samples = dist.sample(RNG, N)
    assert (samples >= 0).all()
    assert (samples <= 1).all()


def test_uniform_bounds():
    dist = Uniform(low=10.0, high=20.0)
    samples = dist.sample(RNG, N)
    assert samples.min() >= 10.0
    assert samples.max() <= 20.0


def test_uniform_mean():
    dist = Uniform(low=0.0, high=1.0)
    samples = dist.sample(RNG, N)
    assert abs(samples.mean() - 0.5) < 0.01
