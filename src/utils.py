# -*- coding: utf-8 -*-
"""
utils.py
========
Small numerical helpers shared across turbulence models: safe log/exp,
running (online) statistics for Monte Carlo, and confidence intervals.
"""
from __future__ import annotations

import numpy as np
from .constants import EPS


def safe_log(x):
    """log(x) with underflow protection. Returns -inf-safe value at x<=0."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.clip(x, EPS, None))


def clip01(x):
    """Clip an array to the physical transmittance range [0, 1]."""
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


class RunningStats:
    """Welford's online algorithm for mean/variance without storing samples.

    Used by the adaptive-stopping mode of the Monte Carlo engine, so that
    memory is O(1) in the sample count (mean, variance and standard error
    only).
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        for xi in x:
            self.n += 1
            delta = xi - self.mean
            self.mean += delta / self.n
            delta2 = xi - self.mean
            self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else float('nan')

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def sem(self):
        """Standard error of the mean."""
        return self.std / np.sqrt(self.n) if self.n > 0 else float('nan')

    def confidence_interval(self, z=1.96):
        """Normal-approximation CI on the mean (default: 95%)."""
        return self.mean - z * self.sem, self.mean + z * self.sem


def confidence_interval(samples, z=1.96):
    """95% (default) CI on the mean of an array of samples, normal approx."""
    samples = np.asarray(samples, dtype=np.float64)
    m = np.mean(samples)
    sem = np.std(samples, ddof=1) / np.sqrt(len(samples))
    return m - z * sem, m + z * sem


def relative_error(approx, exact):
    exact = np.asarray(exact, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)
    denom = np.where(np.abs(exact) > EPS, exact, EPS)
    return np.abs(approx - exact) / np.abs(denom)
