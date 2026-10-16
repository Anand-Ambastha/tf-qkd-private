import numpy as np
from src import legacy_baseline as CAL
from src.turbulence import create_channel, beam_broadening_factor, long_term_beam_radius

LAM, W0, A = 1550e-9, 0.05, 0.15


def _eta(L):
    return float(CAL.deterministic_fso_channel(0.046, L, LAM, W0, A)['eta_fso'])


def test_factor_to_one_without_turbulence_and_monotone():
    assert beam_broadening_factor(0.0, LAM, 50e3, W0, A) == 1.0
    assert beam_broadening_factor(1e-25, LAM, 50e3, W0, A) > 0.999999
    f = [beam_broadening_factor(c, LAM, 50e3, W0, A) for c in (1e-17, 1e-15, 1e-13)]
    assert f[0] > f[1] > f[2] and all(0 < x <= 1 for x in f)


def test_matches_andrews_phillips_formula():
    z, cn2 = 30e3, 1e-15
    W, Wlt = long_term_beam_radius(cn2, LAM, z, W0)
    k = 2 * np.pi / LAM
    s2 = float(CAL.rytov_variance(cn2, LAM, z))
    assert np.isclose(Wlt, W * np.sqrt(1 + 1.63 * s2 ** 1.2 * 2 * z / (k * W ** 2)), rtol=1e-12)
    cap = lambda w: 1 - np.exp(-2 * A ** 2 / w ** 2)
    assert np.isclose(beam_broadening_factor(cn2, LAM, z, W0, A), cap(Wlt) / cap(W), rtol=1e-9)


def test_bb_channel_scales_each_arm_and_reduces_to_base():
    L = 80.0
    base = create_channel('gamma_gamma', eta_nominal=_eta(L), Cn2=1e-15, wavelength=LAM, distance_m=L * 1e3)
    bb = create_channel('gamma_gamma_bb', eta_nominal=_eta(L), Cn2=1e-15, wavelength=LAM,
                        distance_m=L * 1e3, w0=W0, receiver_radius=A)
    f = beam_broadening_factor(1e-15, LAM, L * 500.0, W0, A)
    assert np.isclose(bb.link_A.eta_nominal, base.link_A.eta_nominal * f, rtol=1e-12)
    assert np.isclose(bb.eta_nominal, base.eta_nominal * f * f, rtol=1e-12)
    assert bb.beam_broadening['factor_A'] == bb.beam_broadening['factor_B'] == f
    tiny = create_channel('gamma_gamma_bb', eta_nominal=_eta(L), Cn2=1e-30, wavelength=LAM,
                          distance_m=L * 1e3, w0=W0, receiver_radius=A)
    assert np.isclose(tiny.eta_nominal, base.eta_nominal, rtol=1e-6)


def test_bb_requires_geometry_and_two_link():
    import pytest
    with pytest.raises(ValueError):
        create_channel('gamma_gamma_bb', eta_nominal=1e-3, Cn2=1e-15, wavelength=LAM, distance_m=5e4)
    with pytest.raises(ValueError):
        create_channel('gamma_gamma_bb', eta_nominal=1e-3, Cn2=1e-15, wavelength=LAM, distance_m=5e4,
                       w0=W0, receiver_radius=A, two_link=False)
