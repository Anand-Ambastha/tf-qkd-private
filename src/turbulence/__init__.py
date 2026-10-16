# -*- coding: utf-8 -*-
"""Turbulence model package: deterministic, log-normal, Gamma-Gamma, and
corrected Gamma-Gamma channels behind one polymorphic interface."""
from .base import TurbulenceChannel
from .deterministic import DeterministicChannel
from .lognormal import LogNormalChannel
from .gamma_gamma import GammaGammaChannel, alpha_beta_plane_wave, alpha_beta_spherical_wave
from .corrected_gamma_gamma import CorrectedGammaGammaChannel
from .two_link import TwoLinkChannel
from .beam_broadening import beam_broadening_factor, long_term_beam_radius
from .turbulence_factory import create_channel, classify_regime

__all__ = [
    "TurbulenceChannel", "DeterministicChannel", "LogNormalChannel",
    "GammaGammaChannel", "CorrectedGammaGammaChannel", "TwoLinkChannel", "create_channel",
    "classify_regime", "beam_broadening_factor", "long_term_beam_radius", "alpha_beta_plane_wave", "alpha_beta_spherical_wave",
]
