# -*- coding: utf-8 -*-
"""Two independent FSO links (Alice-Charlie, Bob-Charlie): the end-to-end
transmittance is eta = eta_A * eta_B with INDEPENDENT fading, and every
ergodic average is a 2-D average over (eta_A, eta_B)."""
import math
import numpy as np
import pytest

from src import legacy_baseline as CAL
from src.turbulence import (LogNormalChannel, GammaGammaChannel, TwoLinkChannel,
                            create_channel)

LAM = 1550e-9
PD = 1e-7
TH = float(np.arcsin(np.sqrt(0.02)))


def _kr(alpha=0.2):
    return lambda eta: CAL.key_rate(alpha, np.clip(eta, 1e-300, 1.0), PD, TH, -TH, 4)


# ---------------- log-normal "average term": mean-one is preserved ----------
@pytest.mark.parametrize("sA,sB", [(0.05, 0.05), (0.4, 0.4), (0.6, 0.1)])
def test_product_is_mean_one_unclipped(sA, sB):
    m = CAL.average_over_two_lognormal_links(lambda e: e, 0.01, 0.02, sA, sB, clip=False)
    assert m == pytest.approx(0.01 * 0.02, rel=1e-12)


def test_per_link_mu_is_minus_half_sigma2():
    p = CAL.two_link_lognormal_params(1e-15, LAM, 3000.0, 3000.0)
    for k in 'AB':
        assert p[f'mu_{k}'] == pytest.approx(-0.5 * p[f's2_{k}'], rel=1e-14)
    assert p['mu_total'] == pytest.approx(-0.5 * p['s2_total'], rel=1e-14)


def test_mc_mean_matches_nominal_when_clip_negligible():
    s = CAL.sample_lognormal_eta_two_link(0.01, 1e-15, LAM, 6000.0, 1_000_000,
                                           np.random.default_rng(0))
    assert s.mean() / 0.01 == pytest.approx(1.0, abs=3e-3)


# ---------------- separate Rytov variance per arm ---------------------------
def test_arm_variance_is_evaluated_at_half_length():
    L = 12_000.0
    p = CAL.two_link_lognormal_params(1e-15, LAM, L / 2, L / 2)
    full = float(CAL.rytov_variance(1e-15, LAM, L))
    assert p['sigma_R2_A'] == pytest.approx(full * 2 ** (-11 / 6), rel=1e-12)
    # total log-variance = 2^(-5/6) * single-factor value, i.e. the single-factor model overstates it ~1.78x
    assert p['s2_total'] == pytest.approx(full * 2 ** (-5 / 6), rel=1e-12)


# ---------------- independence ---------------------------------------------
def test_links_are_independent_even_with_identical_seeds():
    ch = create_channel('lognormal', eta_nominal=0.05, Cn2=1e-15, wavelength=LAM,
                        distance_m=6000, seed=7)
    eta, a, b = ch.sample_links(200_000, rng=np.random.default_rng(3))
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.01
    assert not np.allclose(a, b)
    assert np.allclose(eta, a * b)


# ---------------- 2-D rule == exact 1-D reduction ---------------------------
@pytest.mark.parametrize("sA", [0.05, 0.3])
def test_2d_average_equals_1d_with_summed_variance(sA):
    eta0 = 1e-3          # tiny nominal => clip at 1 irrelevant, reduction is exact
    two = CAL.average_over_two_lognormal_links(_kr(), math.sqrt(eta0), math.sqrt(eta0), sA, sA)
    one, _ = CAL.average_skr_lognormal(eta0, 2 * sA, 0.2, PD, TH, -TH, 4)
    assert two == pytest.approx(one, rel=1e-7)


def test_reduces_to_single_link_when_second_arm_is_ideal():
    two = CAL.average_skr_lognormal_two_link(0.01, 0.3, 0.0, 0.2, PD, TH, -TH, 4,
                                              eta_A_nominal=0.01, eta_B_nominal=1.0)
    one, _ = CAL.average_skr_lognormal(0.01, 0.3, 0.2, PD, TH, -TH, 4)
    assert two == pytest.approx(one, rel=1e-8)


# ---------------- quadrature vs Monte Carlo ----------------------------------
@pytest.mark.parametrize("eta0,Cn2,L", [(0.05, 1e-15, 6e3), (0.3, 5e-16, 10e3)])
def test_quadrature_matches_monte_carlo(eta0, Cn2, L):
    p = CAL.two_link_lognormal_params(Cn2, LAM, L / 2, L / 2)
    q = CAL.average_skr_lognormal_two_link(eta0, p['sigma_R2_A'], p['sigma_R2_B'],
                                            0.2, PD, TH, -TH, 4)
    s = CAL.sample_lognormal_eta_two_link(eta0, Cn2, LAM, L, 1_000_000, np.random.default_rng(5))
    R = CAL.key_rate(0.2, s, PD, TH, -TH, 4)
    assert abs(R.mean() - q) < 5 * R.std() / math.sqrt(len(R)) + 1e-4 * q


def test_two_link_differs_from_old_single_factor():
    """The two-link model differs from the single-factor model, which applies
    sigma_R^2 at full L and has ~1.78x the log-variance."""
    eta0, Cn2, L = 0.05, 1e-15, 6e3
    new = create_channel('lognormal', eta_nominal=eta0, Cn2=Cn2, wavelength=LAM, distance_m=L)
    old = create_channel('lognormal', eta_nominal=eta0, Cn2=Cn2, wavelength=LAM, distance_m=L,
                         two_link=False)
    f = _kr()
    r_new = new.average(f)
    r_old = old.average(f)
    assert r_new > r_old     # less log-variance => less deep-fade loss in the average SKR
    assert abs(r_new - r_old) / r_new > 1e-3


# ---------------- channel classes --------------------------------------------
def test_channel_average_matches_legacy_two_link_function():
    eta0, Cn2, L = 0.05, 1e-15, 6e3
    ch = create_channel('lognormal', eta_nominal=eta0, Cn2=Cn2, wavelength=LAM, distance_m=L)
    p = CAL.two_link_lognormal_params(Cn2, LAM, L / 2, L / 2)
    ref = CAL.average_skr_lognormal_two_link(eta0, p['sigma_R2_A'], p['sigma_R2_B'],
                                              0.2, PD, TH, -TH, 4)
    assert ch.average(_kr()) == pytest.approx(ref, rel=1e-9)


def test_channel_statistics_follow_independence():
    ch = create_channel('lognormal', eta_nominal=0.05, Cn2=1e-15, wavelength=LAM, distance_m=6000)
    st = ch.statistics()
    assert st['mean'] == pytest.approx(0.05, rel=1e-12)
    sA = st['link_A']['scintillation_index']
    assert st['scintillation_index'] == pytest.approx((1 + sA) ** 2 - 1, rel=1e-12)


def test_gamma_gamma_two_link_quadrature_vs_mc_and_mean():
    ch = create_channel('gamma_gamma', eta_nominal=0.02, Cn2=3e-15, wavelength=LAM,
                        distance_m=6000, seed=2)
    assert isinstance(ch, TwoLinkChannel)
    eA, wA = ch.link_A.quadrature_nodes()
    assert np.sum(wA) == pytest.approx(1.0, abs=1e-12)
    # independent-product mean is preserved (clip-free regime), to the O(1/n) tail truncation
    assert np.sum(wA * eA) == pytest.approx(ch.link_A.eta_nominal, rel=0.03)
    q = ch.average(_kr(), method="quadrature")
    mc = ch.average(_kr(), method="monte_carlo", n_mc=400_000, rng=np.random.default_rng(1))
    assert q == pytest.approx(mc, rel=0.03)


def test_pdf_cdf_of_product_are_consistent():
    ch = create_channel('lognormal', eta_nominal=0.05, Cn2=1e-15, wavelength=LAM, distance_m=6000)
    x = np.logspace(-3, np.log10(0.4), 400)
    pdf = ch.pdf(x)
    c = ch.cdf(x)
    assert np.all(np.diff(c) >= -1e-9)
    # d cdf / d eta ~ pdf (central differences on the interior)
    num = np.gradient(c, x)
    assert np.allclose(num[5:-5], pdf[5:-5], rtol=0.1, atol=1e-3 * pdf.max())


def test_factory_flags():
    kw = dict(eta_nominal=0.05, Cn2=1e-15, wavelength=LAM, distance_m=6000)
    assert isinstance(create_channel('lognormal', **kw), TwoLinkChannel)
    assert isinstance(create_channel('lognormal', two_link=False, **kw), LogNormalChannel)
    ch = create_channel('lognormal', distance_A_m=2000, distance_B_m=4000, **kw)
    assert ch.link_A.distance_m == 2000 and ch.link_B.distance_m == 4000
    assert ch.eta_nominal == pytest.approx(0.05)
    asym = create_channel('lognormal', eta_A_nominal=0.1, **kw)
    assert asym.link_B.eta_nominal == pytest.approx(0.5)
