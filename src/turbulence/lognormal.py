# -*- coding: utf-8 -*-
"""
lognormal.py
============
Model 2: log-normal irradiance fluctuations (weak-turbulence / Rytov
regime, sigma_R^2 < 1). This module is a thin polymorphic wrapper: the
actual sampling, moment and Rytov-variance equations are the *unchanged*,
already-validated functions from src/legacy_baseline.py
(rytov_variance, lognormal_params, sample_lognormal_eta,
turbulence_statistics). Nothing physics-bearing is reimplemented here --
per the task constraint, the existing Log-Normal implementation is not
modified.

Validity
--------
The log-normal model is the weak-fluctuation limit of the Rytov
perturbation solution to the wave equation in a turbulent medium. It is
accurate for sigma_R^2 < 1 (see [AP], Ch. 9); beyond that point the
log-normal PDF underestimates the tails (deep-fading probability) because
multiple forward-scattering events are no longer negligible, and the
Gamma-Gamma model (turbulence/gamma_gamma.py) should be used instead. The
AUTO selector in turbulence_factory.py enforces this boundary.

References
----------
[AP] Andrews & Phillips, "Laser Beam Propagation through Random Media",
     2nd ed., SPIE Press, 2005, Ch. 9 (Rytov theory / log-normal PDF).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .base import TurbulenceChannel
from .. import legacy_baseline as CAL


class LogNormalChannel(TurbulenceChannel):
    """Log-normal fading of ONE free-space link: eta = eta_nominal * exp(X),
    X ~ N(mu_X, sigma_X^2), energy-neutral (E[exp(X)] = 1).

    A TF-QKD system has TWO independent FSO links (Alice->Charlie and
    Bob->Charlie). Use create_channel(..., two_link=True) (the default there)
    or turbulence.TwoLinkChannel(LogNormalChannel(link A), LogNormalChannel(link B))
    for the end-to-end channel; this class is the per-link building block.
    """

    name = "lognormal"

    def __init__(self, eta_nominal, Cn2, wavelength, distance_m, seed=None):
        super().__init__(eta_nominal)
        self.Cn2 = Cn2
        self.wavelength = wavelength
        self.distance_m = distance_m
        self.sigma_R2 = float(CAL.rytov_variance(Cn2, wavelength, distance_m))
        self.mu_X, self.sigma_X2 = (float(v) for v in CAL.lognormal_params(self.sigma_R2))
        self.sigma_X = float(np.sqrt(self.sigma_X2))
        self._rng_seed = seed

    # -- probabilistic primitives -----------------------------------
    def pdf(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        eta_safe = np.where(eta > 0, eta, np.nan)
        x = np.log(eta_safe / self.eta_nominal)
        dens = norm.pdf(x, loc=self.mu_X, scale=self.sigma_X) / eta_safe
        return np.nan_to_num(dens, nan=0.0)

    def cdf(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        out = np.zeros_like(eta)
        pos = eta > 0
        x = np.log(np.where(pos, eta, 1.0) / self.eta_nominal)
        out = np.where(pos, norm.cdf(x, loc=self.mu_X, scale=self.sigma_X), 0.0)
        return out

    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng(self._rng_seed)
        return CAL.sample_lognormal_eta(self.eta_nominal, self.Cn2, self.wavelength,
                                         self.distance_m, N_samples=int(n), rng=rng)

    # -- analytic moments (closed form, cross-validated against MC in
    #    validation.py) instead of the generic MC-only base-class version --
    def statistics(self, n_mc=20_000, rng=None):
        stats = CAL.turbulence_statistics(self.eta_nominal, self.Cn2, self.wavelength,
                                           self.distance_m, N_samples=n_mc, rng=rng)
        return {'mean': stats['mean_analytic'], 'variance': stats['var_analytic'],
                'scintillation_index': stats['SI_analytic'],
                'std': float(np.sqrt(stats['var_analytic'])),
                'mean_mc': stats['mean_mc'], 'var_mc': stats['var_mc'],
                'sigma_R2': self.sigma_R2}

    # -- exact Gauss-Hermite rule for E[g(eta)] (clip at eta = 1 included) --
    def quadrature_nodes(self, n=None):
        n = 128 if n is None else int(n)
        return CAL.lognormal_gh_nodes(self.eta_nominal, self.mu_X, self.sigma_X2, n, clip=True)
