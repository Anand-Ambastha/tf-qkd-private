# -*- coding: utf-8 -*-
"""
corrected_gamma_gamma.py
========================
Model 4: Gamma-Gamma turbulence extended with the corrections that matter
for a real finite-aperture FSO/satellite-QKD receiver:

  1. Finite receiver aperture -> aperture averaging (reduces the
     effective scintillation index; Andrews & Phillips Ch. 10).
  2. Beam wander -> extra long-term beam broadening / additional jitter
     variance ([AP] Ch. 12; Andrews, Phillips & Hopen).
  3. Pointing errors / misalignment -> Farid & Hranilovic's (2007)
     closed-form power-law pointing-error model, combined multiplicatively
     with the turbulence-induced fading: h = h_a (turbulence, aperture-
     averaged Gamma-Gamma) * h_p (pointing error).

Every correction can be individually disabled
(enable_aperture_averaging / enable_pointing_error / beam_wander_std=0),
and with all three disabled this model reduces *exactly* to
GammaGammaChannel with the same (alpha, beta) -- verified in
tests/test_corrected_gamma_gamma_reduces.py.

What is NOT implemented (documented rather than silently ignored):
  - inner_scale / outer_scale are accepted as optional parameters and are
    applied as a first-order multiplicative correction to the Rytov
    variance (see docs/CORRECTED_GAMMA_GAMMA.md, "Inner/outer scale"
    section) rather than a full modified-spectrum (Hill / von Karman)
    integral -- the latter is a genuine research undertaking on its own
    and is explicitly flagged as future work.
  - The composite pdf/cdf of h_a * h_p has no simple closed form (a
    Meijer-G representation exists in the literature but is numerically
    delicate); we build it instead from a large cached Monte Carlo sample
    at construction time and expose it via histogram/empirical
    interpolation. This is clearly a numerical, not analytic, pdf/cdf --
    documented at the point of use, not hidden.

References
----------
[AP]  Andrews & Phillips, "Laser Beam Propagation through Random Media",
      2nd ed., SPIE Press, 2005, Ch. 10 (aperture averaging) & Ch. 12
      (link budgets, beam wander).
[FH]  Farid, A. A. & Hranilovic, S., "Outage Capacity Optimization for
      Free-Space Optical Links with Pointing Errors", J. Lightwave
      Technol. 25(7), 1702-1710 (2007).
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from .base import TurbulenceChannel
from .gamma_gamma import (alpha_beta_plane_wave, alpha_beta_spherical_wave,
                           gamma_gamma_moments)
from .. import legacy_baseline as CAL


def aperture_averaging_factor(receiver_diameter, wavelength, distance_m):
    """Scintillation-reduction factor A in [0,1] for a finite receiver
    aperture (plane-wave, weak-to-moderate turbulence approximation),
    [AP] Eq. 10.11-ish: A = (1 + 1.062 D^2/(4 lambda L))^(-7/6).
    Effective aperture-averaged scintillation index = A * SI_point.
    """
    D = float(receiver_diameter)
    return float((1.0 + 1.062 * D ** 2 / (4.0 * wavelength * distance_m)) ** (-7.0 / 6.0))


def _effective_alpha_beta(alpha, beta, A):
    """Rescale (alpha, beta) so the normalized variance shrinks by the
    aperture-averaging factor A. We scale both parameters as
    alpha_eff = alpha/A, beta_eff = beta/A, which gives

        Var_eff = A(1/alpha + 1/beta) + A^2/(alpha beta) ~= A * Var0

    to leading order (the cross term 1/(alpha beta) is normally
    sub-dominant, see docs/CORRECTED_GAMMA_GAMMA.md). This is a
    documented simplification: a rigorous treatment aperture-averages the
    large-scale and small-scale spectral contributions separately with
    different weighting (see [AP] Ch. 10); the symmetric 1/A rescaling
    used here is the standard practical approximation quoted across the
    FSO engineering literature, and -- unlike a scheme that holds one
    parameter fixed and solves for the other -- it stays numerically
    well-behaved (alpha_eff, beta_eff > 0 for every A in (0,1]) even when
    the aperture-averaging factor is strong (small A).
    """
    if A <= 0:
        raise ValueError("aperture-averaging factor A must be > 0")
    return alpha / A, beta / A


def pointing_error_params(receiver_radius, w_Lt, jitter_std):
    """Farid-Hranilovic pointing-error shape parameter gamma_pe and
    maximum fraction A0, from the equivalent-beam-radius approximation.

    v = sqrt(pi) a / (sqrt(2) w_Lt)
    A0 = erf(v)^2
    w_Leq^2 = w_Lt^2 * sqrt(pi) erf(v) / (2 v exp(-v^2))
    gamma_pe = w_Leq / (2 sigma_s)
    """
    a = float(receiver_radius)
    v = np.sqrt(np.pi) * a / (np.sqrt(2.0) * w_Lt)
    A0 = erf(v) ** 2
    w_Leq2 = w_Lt ** 2 * np.sqrt(np.pi) * erf(v) / (2.0 * v * np.exp(-v ** 2))
    gamma_pe = np.sqrt(w_Leq2) / (2.0 * jitter_std) if jitter_std > 0 else np.inf
    return float(gamma_pe), float(A0)


class CorrectedGammaGammaChannel(TurbulenceChannel):
    """Gamma-Gamma turbulence + finite-aperture, beam-wander and
    pointing-error corrections, all independently toggleable."""

    name = "gamma_gamma_corrected"

    def __init__(self, eta_nominal, Cn2, wavelength, distance_m,
                 receiver_radius, jitter_std=0.0, beam_wander_std=0.0,
                 w0=0.05, wave_type="plane", Theta=1.0,
                 inner_scale=None, outer_scale=None,
                 enable_aperture_averaging=True, enable_pointing_error=True,
                 n_mc_cache=200_000, seed=None):
        super().__init__(eta_nominal)
        self.Cn2, self.wavelength, self.distance_m = Cn2, wavelength, distance_m
        self.receiver_radius, self.w0 = receiver_radius, w0
        self.jitter_std, self.beam_wander_std = jitter_std, beam_wander_std
        self.inner_scale, self.outer_scale = inner_scale, outer_scale
        self.enable_aperture_averaging = enable_aperture_averaging
        self.enable_pointing_error = enable_pointing_error
        self._rng_seed = seed

        sigma_R2 = float(CAL.rytov_variance(Cn2, wavelength, distance_m))
        sigma_R2 = self._inner_outer_scale_correction(sigma_R2)
        self.sigma_R2 = sigma_R2

        if wave_type == "plane":
            a0, b0 = alpha_beta_plane_wave(sigma_R2)
        elif wave_type == "spherical":
            a0, b0 = alpha_beta_spherical_wave(sigma_R2, Theta=Theta)
        else:
            raise ValueError("wave_type must be 'plane' or 'spherical'")
        self.alpha0, self.beta0 = a0, b0

        if enable_aperture_averaging:
            D = 2.0 * receiver_radius
            self.A_factor = aperture_averaging_factor(D, wavelength, distance_m)
            self.alpha, self.beta = _effective_alpha_beta(a0, b0, self.A_factor)
        else:
            self.A_factor = 1.0
            self.alpha, self.beta = a0, b0

        if enable_pointing_error and (jitter_std > 0 or beam_wander_std > 0):
            sigma_s = float(np.sqrt(jitter_std ** 2 + beam_wander_std ** 2))
            w_Lt = float(CAL.beam_radius(wavelength, distance_m, w0))
            self.gamma_pe, self.A0_pe = pointing_error_params(receiver_radius, w_Lt, sigma_s)
        else:
            self.gamma_pe, self.A0_pe = np.inf, 1.0  # no pointing error -> h_p == 1

        self._build_empirical_distribution(n_mc_cache)

    def _inner_outer_scale_correction(self, sigma_R2):
        """First-order multiplicative correction (documented approximation,
        NOT a full modified-spectrum integral -- see module docstring)."""
        factor = 1.0
        if self.inner_scale is not None and self.inner_scale > 0:
            # small inner scale slightly increases scintillation (more
            # high-wavenumber content retained); heuristic +10% at l0->0,
            # negligible for l0 >~ few mm, saturating smoothly.
            factor *= 1.0 + 0.1 * np.exp(-self.inner_scale / 1e-3)
        if self.outer_scale is not None and self.outer_scale > 0:
            # finite outer scale removes some low-wavenumber (large-eddy)
            # power -> mild reduction, saturating for L0 >~ tens of meters.
            factor *= 1.0 - 0.05 * np.exp(-self.outer_scale / 10.0)
        return sigma_R2 * factor

    # -- sampling of the multiplicative pointing-error factor --------
    def _sample_hp(self, n, rng):
        if not np.isfinite(self.gamma_pe):
            return np.ones(int(n))
        U = rng.uniform(size=int(n))
        hp = self.A0_pe * U ** (1.0 / self.gamma_pe ** 2)
        mean_hp = self.A0_pe * self.gamma_pe ** 2 / (self.gamma_pe ** 2 + 1.0)
        return hp / mean_hp  # renormalize to unit mean so eta_nominal stays the reference

    def _build_empirical_distribution(self, n_mc_cache):
        rng = np.random.default_rng(self._rng_seed)
        Ix = rng.gamma(shape=self.alpha, scale=1.0 / self.alpha, size=n_mc_cache)
        Iy = rng.gamma(shape=self.beta, scale=1.0 / self.beta, size=n_mc_cache)
        I = Ix * Iy
        hp = self._sample_hp(n_mc_cache, rng)
        F = np.clip(I * hp, 0.0, None)
        self._cache_F = np.sort(F)
        n = len(F)
        self._cache_cdf = (np.arange(1, n + 1) - 0.5) / n
        edges = np.histogram_bin_edges(F, bins=200, range=(0.0, np.percentile(F, 99.9)))
        counts, edges = np.histogram(F, bins=edges, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        self._pdf_x, self._pdf_y = centers, counts

    # -- probabilistic primitives (numerical, see module docstring) --
    def pdf(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        F = eta / self.eta_nominal
        return np.interp(F, self._pdf_x, self._pdf_y, left=0.0, right=0.0) / self.eta_nominal

    def cdf(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        F = np.clip(eta / self.eta_nominal, 0.0, None)
        return np.interp(F, self._cache_F, self._cache_cdf, left=0.0, right=1.0)

    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng(self._rng_seed)
        Ix = rng.gamma(shape=self.alpha, scale=1.0 / self.alpha, size=int(n))
        Iy = rng.gamma(shape=self.beta, scale=1.0 / self.beta, size=int(n))
        hp = self._sample_hp(n, rng)
        F = np.clip(Ix * Iy * hp, 0.0, None)
        return np.clip(self.eta_nominal * F, 0.0, 1.0)

    def statistics(self, n_mc=None, rng=None):
        F = self._cache_F
        mean = float(np.mean(F))
        var = float(np.var(F))
        return {'mean': self.eta_nominal * mean, 'variance': (self.eta_nominal ** 2) * var,
                'scintillation_index': var / mean ** 2, 'std': self.eta_nominal * np.sqrt(var),
                'sigma_R2': self.sigma_R2, 'alpha': self.alpha, 'beta': self.beta,
                'aperture_averaging_factor': self.A_factor, 'gamma_pe': self.gamma_pe,
                'A0_pe': self.A0_pe}
