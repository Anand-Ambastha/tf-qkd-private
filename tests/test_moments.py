# -*- coding: utf-8 -*-
"""Sampling moments (mean, variance) vs. analytic/reference moments."""
import numpy as np
from src.validation import check_sampling_moments


def test_sampling_mean_matches_reference(all_stochastic_channels):
    for name, ch in all_stochastic_channels:
        result = check_sampling_moments(ch, n=200_000, rtol=0.05, seed=3)
        assert result['mean_ok'], (f"{name}: MC mean={result['mean_mc']:.6g} vs "
                                    f"reference={result['mean_ref']:.6g}")


def test_gamma_gamma_variance_formula():
    """Var(I) = 1/alpha + 1/beta + 1/(alpha beta) for normalized I (E[I]=1),
    cross-checked directly against a large Monte Carlo sample of I=Ix*Iy."""
    from src.turbulence.gamma_gamma import gamma_gamma_moments
    rng = np.random.default_rng(7)
    for alpha, beta in [(2.0, 3.0), (10.0, 1.5), (50.0, 50.0), (1.2, 0.8)]:
        Ix = rng.gamma(alpha, 1.0 / alpha, size=500_000)
        Iy = rng.gamma(beta, 1.0 / beta, size=500_000)
        I = Ix * Iy
        analytic = gamma_gamma_moments(alpha, beta)
        assert abs(np.mean(I) - 1.0) < 0.01
        assert abs(np.var(I) - analytic['variance']) / analytic['variance'] < 0.05


def test_lognormal_energy_neutral(lognormal_channel):
    """E[eta] should equal eta_nominal (turbulence redistributes energy,
    doesn't create or destroy it on average)."""
    s = lognormal_channel.sample(300_000)
    assert abs(np.mean(s) - lognormal_channel.eta_nominal) / lognormal_channel.eta_nominal < 0.02


def test_gamma_gamma_alpha_beta_match_andrews_phillips():
    """Regression for the sigma_R^(12/5) exponent (v2.2.0). Reference SI values
    are the well-known Gamma-Gamma behaviour: SI = Rytov variance-ish for weak
    turbulence, peaks ~1.2-1.3 near sigma_R^2 ~ 10, then saturates toward 1."""
    import numpy as np
    from src.turbulence.gamma_gamma import alpha_beta_plane_wave

    def si(s):
        a, b = alpha_beta_plane_wave(s)
        return 1 / a + 1 / b + 1 / (a * b)

    a1, b1 = alpha_beta_plane_wave(1.0)
    assert abs(a1 - 4.39) < 0.02 and abs(b1 - 2.56) < 0.02     # exponent-independent point
    assert abs(si(0.1) - 0.099) < 0.005
    assert 1.15 < si(10.0) < 1.30
    assert 1.15 < si(5.0) < 1.30
    assert 1.0 < si(1000.0) < 1.10      # saturation toward 1, never collapses to 0
    assert max(si(s) for s in np.logspace(-1, 3, 60)) < 1.35
