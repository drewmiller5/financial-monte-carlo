import numpy as np


class Normal:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.normal(self.mean, self.std, n)


class Lognormal:
    """mean and std are parameters of the underlying normal distribution."""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.lognormal(self.mean, self.std, n)


class Triangular:
    """Three-point estimate: low, most likely (mode), high."""
    def __init__(self, low: float, mode: float, high: float):
        if not (low <= mode <= high):
            raise ValueError(f"Triangular requires low <= mode <= high, got ({low}, {mode}, {high})")
        self.low = low
        self.mode = mode
        self.high = high

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, n)


class Beta:
    """Beta distribution — bounded [0, 1], useful for probabilities and rates."""
    def __init__(self, alpha: float, beta_param: float):
        self.alpha = alpha
        self.beta_param = beta_param

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.beta(self.alpha, self.beta_param, n)


class Uniform:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)
