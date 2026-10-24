# -*- coding: utf-8 -*-
"""
monte_carlo.py
==============
Monte Carlo expectation of a function of the channel transmittance.
Supports fixed-sample-count runs and an adaptive mode that keeps drawing
batches until the standard error of the mean drops below a target
tolerance (or a sample cap is hit), plus a benchmark harness that sweeps
the sample-count grid from constants.MC_SAMPLE_GRID.
"""
from __future__ import annotations

import time
import numpy as np

from .utils import RunningStats
from .constants import DEFAULT_MC_SEED, MC_SAMPLE_GRID


def mc_expectation(channel, func, n, rng=None, seed=None):
    """E[func(eta)] from n i.i.d. samples of `channel`."""
    if rng is None:
        rng = np.random.default_rng(seed if seed is not None else DEFAULT_MC_SEED)
    eta = channel.sample(n, rng=rng)
    vals = np.asarray(func(eta), dtype=np.float64)
    return {'estimate': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
            'sem': float(np.std(vals, ddof=1) / np.sqrt(n)), 'n': int(n)}


def mc_expectation_adaptive(channel, func, tol=1e-3, batch=5_000, n_max=2_000_000,
                             seed=None, z=1.96):
    """Draw samples in batches, growing n, until the 95% CI half-width on
    the mean falls below `tol` (absolute) or n_max samples are used.
    Uses Welford's online algorithm so memory stays O(1) regardless of
    how many samples are ultimately drawn.
    """
    rng = np.random.default_rng(seed if seed is not None else DEFAULT_MC_SEED)
    stats = RunningStats()
    n_total = 0
    history = []
    while n_total < n_max:
        eta = channel.sample(batch, rng=rng)
        vals = np.asarray(func(eta), dtype=np.float64)
        stats.update(vals)
        n_total += batch
        half_width = z * stats.sem
        history.append({'n': n_total, 'mean': stats.mean, 'sem': stats.sem,
                         'half_width': half_width})
        if n_total >= 2 * batch and half_width < tol:
            break
    ci_lo, ci_hi = stats.confidence_interval(z=z)
    return {'estimate': stats.mean, 'std': stats.std, 'sem': stats.sem,
            'n': n_total, 'ci': (ci_lo, ci_hi), 'converged': stats.sem * z < tol,
            'history': history}


def benchmark_monte_carlo(channel, func, sample_grid=None, seed=None, n_repeats=3):
    """Runtime and estimate-stability sweep across constants.MC_SAMPLE_GRID
    (or a custom grid). Returns rows suitable for a markdown/CSV table:
    for each n, the mean runtime and the spread of the estimate across
    `n_repeats` independent seeds (a proxy for MC variance at that n).
    """
    sample_grid = sample_grid or MC_SAMPLE_GRID
    rows = []
    for n in sample_grid:
        estimates, times = [], []
        for rep in range(n_repeats):
            rng = np.random.default_rng((seed or 0) + rep)
            t0 = time.perf_counter()
            res = mc_expectation(channel, func, n, rng=rng)
            times.append(time.perf_counter() - t0)
            estimates.append(res['estimate'])
        rows.append({'n': n, 'mean_estimate': float(np.mean(estimates)),
                      'std_across_repeats': float(np.std(estimates)),
                      'mean_runtime_s': float(np.mean(times)),
                      'sem_single_run': res['sem']})
    return rows
