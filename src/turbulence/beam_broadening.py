# -*- coding: utf-8 -*-
"""
beam_broadening.py
==================
Turbulence-induced LONG-TERM BEAM BROADENING, applied per arm.

Free-space diffraction is already inside the deterministic channel
(`legacy_baseline.deterministic_fso_channel`: eta_geo(L/2)^2 with the Gaussian
beam radius w(z)). Optical turbulence adds extra spreading of the long-term
(time-averaged) beam. For a collimated Gaussian beam over a horizontal path
(Andrews & Phillips, Laser Beam Propagation through Random Media):

    W_LT^2 = W^2 * (1 + 1.63 * sigma_R^(12/5) * Lambda),     Lambda = 2 z / (k W^2)

with W = w(z) the free-space beam radius at the receiver, z the ARM length (L/2),
k = 2 pi / lambda, and sigma_R^2 the plane-wave Rytov variance of that arm
(`legacy_baseline.rytov_variance`, same quantity the fading models use);
sigma_R^(12/5) = (sigma_R^2)^(6/5).

The effect on the nominal arm transmittance is the capture-fraction ratio

    f_bb(z) = [1 - exp(-2 a^2 / W_LT^2)] / [1 - exp(-2 a^2 / W^2)]   (<= 1, -> 1 as Cn2 -> 0)

so   eta_arm_nominal(with BB) = eta_arm_nominal * f_bb(z).

The fading on top of it (log-normal / Gamma-Gamma, mean-one) is unchanged. Each arm
gets its own factor, so the end-to-end nominal is eta_nom * f_bb(L/2)^2.

Assumptions (state them when quoting results):
  * The long-term beam already contains beam wander, so do not ALSO add beam-wander
    jitter on top of this model (double counting).
  * Horizontal path, constant Cn2, collimated Gaussian beam at the transmitter.
  * Gamma-Gamma alpha/beta stay the plane-wave values (scintillation of the broadened
    beam is not re-derived), so this is a first-order correction.
"""
from __future__ import annotations

import numpy as np

from .. import legacy_baseline as CAL

BB_COEFF = 1.63


def free_space_beam_radius(wavelength, z_m, w0):
    return float(CAL.beam_radius(wavelength, z_m, w0))


def long_term_beam_radius(Cn2, wavelength, z_m, w0):
    """(W_free_space, W_long_term) at distance z_m [m]."""
    W = free_space_beam_radius(wavelength, z_m, w0)
    k = 2.0 * np.pi / wavelength
    Lam = 2.0 * z_m / (k * W ** 2)
    sig2 = float(CAL.rytov_variance(Cn2, wavelength, z_m))
    return W, W * float(np.sqrt(1.0 + BB_COEFF * sig2 ** (6.0 / 5.0) * Lam))


def _capture(a, W):
    return float(-np.expm1(-2.0 * a ** 2 / W ** 2))


def beam_broadening_factor(Cn2, wavelength, z_m, w0, receiver_radius):
    """Capture-fraction ratio f_bb in (0, 1] for ONE arm of length z_m."""
    if Cn2 is None or Cn2 <= 0.0 or z_m <= 0.0:
        return 1.0
    W, W_lt = long_term_beam_radius(Cn2, wavelength, z_m, w0)
    return float(np.clip(_capture(receiver_radius, W_lt) / _capture(receiver_radius, W), 0.0, 1.0))
