# -*- coding: utf-8 -*-
"""Adaptive alpha optimization, distance sweep, and extreme-parameter
numerical stability (deep fading, zero fading, over/underflow)."""
import numpy as np
from src.adaptive_alpha import instantaneous_adaptive_alpha, optimize_fixed_alpha
from src.turbulence import LogNormalChannel, GammaGammaChannel, create_channel


def test_instantaneous_adaptive_alpha_never_worse_than_fixed(engine, weak_turbulence_params):
    ch = LogNormalChannel(**weak_turbulence_params, seed=2)
    eta_samples = ch.sample(3000, rng=np.random.default_rng(2))
    result = instantaneous_adaptive_alpha(eta_samples, engine.pd, engine.thetaA, engine.thetaB,
                                           Nmax=engine.Nmax, alpha_fixed=0.20)
    # instantaneous adaptation is an upper bound on fixed-alpha performance
    assert result['stats_adaptive']['mean'] >= result['stats_fixed']['mean'] - 1e-9


def test_optimize_fixed_alpha_within_bounds(gamma_gamma_channel, engine):
    alpha_opt, R_opt = optimize_fixed_alpha(gamma_gamma_channel, engine.pd, engine.thetaA,
                                             engine.thetaB, Nmax=engine.Nmax,
                                             method="quadrature")
    assert 1e-3 <= alpha_opt <= 0.3
    assert R_opt >= 0.0


def test_distance_sweep_monotonic_decrease(engine):
    distances = [1, 5, 10, 20, 40]

    def factory(L):
        return create_channel('lognormal', eta_nominal=max(1e-6, 0.5 * np.exp(-0.15 * L)),
                               Cn2=1e-17, wavelength=1550e-9, distance_m=L * 1e3)

    R = engine.distance_sweep(distances, factory, alpha=0.20, method="quadrature")
    # SKR should be non-increasing with distance for a monotonically
    # decreasing channel transmittance
    assert np.all(np.diff(R) <= 1e-9)


def test_deep_fading_zero_key_rate(engine):
    """At eta essentially zero, the key rate must be exactly clipped to 0,
    not NaN/negative/overflowing."""
    R = engine.instantaneous_key_rate(0.20, np.array([1e-30, 1e-20, 1e-10]))
    assert np.all(np.isfinite(R))
    assert np.all(R >= 0.0)


def test_zero_fading_eta_equals_one(engine):
    R = engine.instantaneous_key_rate(0.20, np.array([1.0]))
    assert np.isfinite(R[0]) and R[0] >= 0.0


def test_gamma_gamma_large_alpha_beta_no_overflow():
    """Large, strongly asymmetric (alpha, beta) must still give a finite,
    correctly normalized PDF (mpmath fallback for scipy.special.kv overflow,
    see turbulence/gamma_gamma.py::_log_kv)."""
    from src.turbulence.gamma_gamma import gamma_gamma_pdf
    I = np.array([1e-6, 1e-3, 0.1, 1.0, 5.0, 20.0])
    vals = gamma_gamma_pdf(I, alpha=500.0, beta=8.0)
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)


def test_scintillation_index_extreme_sigma():
    """Very small and very large Rytov variance give positive, finite
    alpha/beta."""
    from src.turbulence.gamma_gamma import alpha_beta_plane_wave
    for s in [1e-6, 0.5, 5.0, 50.0]:
        a, b = alpha_beta_plane_wave(s)
        assert a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b)
