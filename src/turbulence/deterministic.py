# -*- coding: utf-8 -*-
"""
deterministic.py
=================
Model 1: deterministic FSO propagation, no turbulence. eta is a fixed
number set by diffraction (beam spreading), atmospheric (Beer-Lambert)
attenuation, and the geometric truncation loss at a finite receiver
aperture. This reuses the validated equations from the legacy baseline
verbatim (see src/legacy_baseline.py, Section 9) -- nothing here
re-derives or modifies that physics.

eta(L) = eta_atm(L) * eta_geo(L/2)^2

with

  eta_atm(L)  = exp(-alpha_atm * L)                       (Beer-Lambert)
  w(z)        = w0 * sqrt(1 + (lambda z / (pi w0^2))^2)   (Gaussian beam spreading)
  eta_geo(z)  = 1 - exp(-2 r_rx^2 / w(z)^2)                (truncation efficiency)

References
----------
[AP] Andrews & Phillips, "Laser Beam Propagation through Random Media",
     2nd ed., SPIE Press, 2005, Ch. 1 (Gaussian-beam propagation) and
     Ch. 12 (link budget / geometric loss).
"""
from __future__ import annotations

import numpy as np

from .base import TurbulenceChannel
from .. import legacy_baseline as CAL


class DeterministicChannel(TurbulenceChannel):
    """A degenerate 'random variable' that is a point mass at eta_nominal.

    Exists so the TF-QKD engine can treat "no turbulence" through exactly
    the same polymorphic interface as the stochastic models, at the cost
    of pdf/cdf being (mathematically honest) generalized/step functions
    rather than smooth densities.
    """

    name = "deterministic"

    def __init__(self, alpha_atm, distance_km, wavelength, w0, receiver_radius):
        fso = CAL.deterministic_fso_channel(alpha_atm, distance_km, wavelength,
                                             w0, receiver_radius)
        eta_nominal = float(fso['eta_fso'])
        super().__init__(max(eta_nominal, 1e-300))
        self.fso_breakdown = fso
        self.distance_km = distance_km

    def pdf(self, eta):
        """Dirac delta at eta_nominal; returned as a dense approximation
        (a narrow, area-1 top-hat of width 1e-9) so callers that expect a
        finite array (e.g. plotting) don't crash on a literal delta."""
        eta = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        width = max(1e-9, 1e-6 * self.eta_nominal)
        out = np.where(np.abs(eta - self.eta_nominal) <= width / 2, 1.0 / width, 0.0)
        return out if out.size > 1 else float(out[0])

    def cdf(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        return np.where(eta >= self.eta_nominal, 1.0, 0.0)

    def sample(self, n, rng=None):
        return np.full(int(n), self.eta_nominal, dtype=np.float64)

    def statistics(self, n_mc=None, rng=None):
        return {'mean': self.eta_nominal, 'variance': 0.0,
                'scintillation_index': 0.0, 'std': 0.0}

    def average(self, func, method="quadrature", n_mc=1, rng=None, quad_kwargs=None):
        # No fading to average over: E[f(eta)] = f(eta_nominal) exactly,
        # regardless of the requested method.
        return float(np.asarray(func(np.array([self.eta_nominal])))[0])

    def quadrature_nodes(self, n=None):
        return np.array([self.eta_nominal]), np.array([1.0])
