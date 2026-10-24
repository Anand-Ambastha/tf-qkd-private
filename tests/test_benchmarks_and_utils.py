# -*- coding: utf-8 -*-
"""Benchmark harnesses (Monte Carlo sample-count sweep, quadrature-order
sweep) execute cleanly and return finite, approximately monotonic outputs; RunningStats
online algorithm matches numpy's batch mean/variance."""
import numpy as np
from src.monte_carlo import benchmark_monte_carlo, mc_expectation_adaptive
from src.quadrature import benchmark_quadrature
from src.utils import RunningStats


def test_benchmark_monte_carlo_runs(gamma_gamma_channel, engine):
    func = lambda eta: engine.instantaneous_key_rate(0.20, eta)
    rows = benchmark_monte_carlo(gamma_gamma_channel, func, sample_grid=[200, 1000, 5000],
                                  seed=0, n_repeats=2)
    assert len(rows) == 3
    for r in rows:
        assert r['mean_runtime_s'] >= 0
        assert np.isfinite(r['mean_estimate'])


def test_benchmark_quadrature_runs(gamma_gamma_channel):
    rows = benchmark_quadrature(gamma_gamma_channel, lambda eta: eta,
                                 a=0.0, b=gamma_gamma_channel.eta_max,
                                 n_values=(8, 16, 32))
    assert rows[0]['method'].startswith('scipy_quad')
    gl = [r for r in rows if r['method'] == 'gauss_legendre']
    assert len(gl) == 3
    # accuracy should generally improve (or stay good) with more points
    assert gl[-1]['relative_error'] <= gl[0]['relative_error'] + 1e-6


def test_adaptive_mc_converges(lognormal_channel):
    result = mc_expectation_adaptive(lognormal_channel, lambda eta: eta, tol=1e-4,
                                      batch=2000, n_max=500_000, seed=0)
    assert result['n'] <= 500_000
    assert abs(result['estimate'] - lognormal_channel.eta_nominal) < 5e-3


def test_running_stats_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=10_000)
    rs = RunningStats()
    rs.update(data)
    assert np.isclose(rs.mean, np.mean(data), rtol=1e-9)
    assert np.isclose(rs.variance, np.var(data, ddof=1), rtol=1e-9)
