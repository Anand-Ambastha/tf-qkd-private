# -*- coding: utf-8 -*-
"""With aperture averaging and pointing error both disabled, the corrected
Gamma-Gamma model must reduce exactly to the plain Gamma-Gamma model with
the same alpha, beta."""
import numpy as np
from src.turbulence import GammaGammaChannel, CorrectedGammaGammaChannel


def test_alpha_beta_match_when_disabled(strong_turbulence_params):
    gg = GammaGammaChannel(**strong_turbulence_params, seed=5)
    cgg = CorrectedGammaGammaChannel(**strong_turbulence_params, receiver_radius=0.15,
                                      enable_aperture_averaging=False,
                                      enable_pointing_error=False, seed=5)
    assert np.isclose(gg.alpha, cgg.alpha)
    assert np.isclose(gg.beta, cgg.beta)


def test_statistics_match_when_disabled(strong_turbulence_params):
    gg = GammaGammaChannel(**strong_turbulence_params, seed=5)
    cgg = CorrectedGammaGammaChannel(**strong_turbulence_params, receiver_radius=0.15,
                                      enable_aperture_averaging=False,
                                      enable_pointing_error=False, seed=5)
    s_gg = gg.statistics()
    s_cgg = cgg.statistics()
    assert abs(s_gg['mean'] - s_cgg['mean']) / s_gg['mean'] < 0.03
    assert abs(s_gg['variance'] - s_cgg['variance']) / s_gg['variance'] < 0.1


def test_aperture_averaging_reduces_scintillation(strong_turbulence_params):
    cgg_on = CorrectedGammaGammaChannel(**strong_turbulence_params, receiver_radius=0.15,
                                         enable_aperture_averaging=True,
                                         enable_pointing_error=False, seed=5)
    cgg_off = CorrectedGammaGammaChannel(**strong_turbulence_params, receiver_radius=0.15,
                                          enable_aperture_averaging=False,
                                          enable_pointing_error=False, seed=5)
    assert cgg_on.statistics()['scintillation_index'] < cgg_off.statistics()['scintillation_index']
