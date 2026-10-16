# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tfqkd_engine import TFQKDEngine
from src.turbulence import (DeterministicChannel, LogNormalChannel, GammaGammaChannel,
                             CorrectedGammaGammaChannel)

PD = 1e-7
THETA_A = np.arcsin(np.sqrt(0.02))
THETA_B = -THETA_A


@pytest.fixture
def engine():
    return TFQKDEngine(PD, THETA_A, THETA_B, Nmax=4)


@pytest.fixture
def weak_turbulence_params():
    """sigma_R^2 ~ 0.015, well inside the log-normal validity domain."""
    return dict(eta_nominal=0.1, Cn2=1e-16, wavelength=1550e-9, distance_m=3000)


@pytest.fixture
def moderate_turbulence_params():
    """sigma_R^2 ~ 2-4, Gamma-Gamma regime."""
    return dict(eta_nominal=0.05, Cn2=5e-15, wavelength=1550e-9, distance_m=6000)


@pytest.fixture
def strong_turbulence_params():
    """sigma_R^2 >~ 10, corrected Gamma-Gamma regime."""
    return dict(eta_nominal=0.01, Cn2=1e-14, wavelength=1550e-9, distance_m=10_000)


@pytest.fixture
def lognormal_channel(weak_turbulence_params):
    return LogNormalChannel(**weak_turbulence_params, seed=1)


@pytest.fixture
def gamma_gamma_channel(moderate_turbulence_params):
    return GammaGammaChannel(**moderate_turbulence_params, seed=1)


@pytest.fixture
def corrected_gamma_gamma_channel(strong_turbulence_params):
    return CorrectedGammaGammaChannel(**strong_turbulence_params, receiver_radius=0.15,
                                       jitter_std=1e-6, beam_wander_std=5e-7, seed=1)


@pytest.fixture
def deterministic_channel():
    return DeterministicChannel(alpha_atm=0.2, distance_km=5, wavelength=1550e-9,
                                 w0=0.05, receiver_radius=0.15)


@pytest.fixture
def all_stochastic_channels(lognormal_channel, gamma_gamma_channel, corrected_gamma_gamma_channel):
    return [('lognormal', lognormal_channel), ('gamma_gamma', gamma_gamma_channel),
            ('gamma_gamma_corrected', corrected_gamma_gamma_channel)]
