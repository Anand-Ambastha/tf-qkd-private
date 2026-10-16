# -*- coding: utf-8 -*-
"""
gamma_gamma.py
==============
Model 3: Gamma-Gamma turbulence, implemented from first principles
(moderate-to-strong scintillation, sigma_R^2 >~ 1, up to the onset of
saturation). Derivation summary (full version in docs/GAMMA_GAMMA.md):

Physical picture
----------------
The Gamma-Gamma model (Al-Habash, Andrews & Phillips, 2001) treats the
received irradiance I as the product of two independent random processes
representing large-scale (refractive, eddy size > first Fresnel zone /
beam radius) and small-scale (diffractive) irradiance fluctuations:

    I = I_x * I_y ,   I_x ~ Gamma(alpha, 1/alpha),  I_y ~ Gamma(beta, 1/beta)

each normalized to unit mean, so E[I] = E[I_x] E[I_y] = 1 (energy is
conserved on average; only its higher moments are affected by turbulence).
alpha and beta are the *effective numbers* of large-scale and small-scale
turbulent cells contributing to the fluctuations -- larger alpha, beta
means more averaging and a narrower distribution.

Irradiance PDF (closed form, product of two Gammas)
----------------------------------------------------
    p_I(I) = 2 (alpha beta)^((alpha+beta)/2) / (Gamma(alpha) Gamma(beta))
             * I^((alpha+beta)/2 - 1) * K_{alpha-beta}(2 sqrt(alpha beta I)),   I > 0

where K_v is the modified Bessel function of the second kind of order v
(scipy.special.kv, which supports real, non-integer, negative order via
the reflection identity K_{-v} = K_v).

alpha, beta from Rytov theory (Andrews & Phillips, Eqs. 9.59-9.63)
-------------------------------------------------------------------
    alpha = [ exp( 0.49 sigma1^2 / (1 + 1.11 sigma1^(12/5))^(7/6) ) - 1 ]^-1
    beta  = [ exp( 0.51 sigma1^2 / (1 + 0.69 sigma1^(12/5))^(5/6) ) - 1 ]^-1

for a plane wave, where sigma1^2 is the Rytov variance (same quantity as
`legacy_baseline.rytov_variance`, reused unchanged) and sigma1^(12/5) =
(sigma1^2)^(6/5) is the Rytov STANDARD DEVIATION to the 12/5 power. (v2.2.0 fix:
earlier versions raised the variance itself to 12/5, which drove the
scintillation index to ~0 in strong turbulence instead of letting it saturate
near 1.) Spherical-wave
coefficients (also implemented) replace the 1.11 with 0.56(1+Theta) per
[AP] Table 9.2; Theta=1 (matched aperture) is used as the simplifying
default, documented explicitly in docs/GAMMA_GAMMA.md.

Moments (normalized, E[I]=1)
-----------------------------
    Var(I) = (1+1/alpha)(1+1/beta) - 1 = 1/alpha + 1/beta + 1/(alpha*beta)
    Scintillation index SI = Var(I) / E[I]^2 = Var(I)   (since E[I]=1)

Sampling
--------
Exact sampling by generating I_x ~ Gamma(alpha, scale=1/alpha) and
I_y ~ Gamma(beta, scale=1/beta) independently and multiplying -- this is
exact (not an approximation), because it uses the model's own generative
definition rather than inverse-CDF sampling of the closed-form PDF.

References
----------
[AP]  Andrews & Phillips, "Laser Beam Propagation through Random Media",
      2nd ed., SPIE Press, 2005, Ch. 9 & 10.
[AAP] Al-Habash, Andrews & Phillips, "Mathematical model for the
      irradiance probability density function of a laser beam propagating
      through turbulent media", Opt. Eng. 40(8), 1554-1562 (2001).
"""
from __future__ import annotations

import numpy as np
from scipy.special import kv, gammaln, gammainc
from scipy.integrate import quad

from .base import TurbulenceChannel
from .. import legacy_baseline as CAL


def alpha_beta_plane_wave(sigma_R2):
    """(alpha, beta) from the plane-wave Rytov variance ([AP] Eq. 9.59)."""
    s = float(sigma_R2)
    s125 = s ** (6.0 / 5.0)          # sigma_R^(12/5) = (sigma_R^2)^(6/5); sigma_R = sqrt(Rytov variance)
    alpha = 1.0 / (np.expm1(0.49 * s / (1.0 + 1.11 * s125) ** (7.0 / 6.0)))
    beta = 1.0 / (np.expm1(0.51 * s / (1.0 + 0.69 * s125) ** (5.0 / 6.0)))
    return float(alpha), float(beta)


def alpha_beta_spherical_wave(sigma_R2, Theta=1.0):
    """(alpha, beta) from the spherical-wave Rytov variance, [AP] Table 9.2.
    Theta is the beam curvature parameter at the receiver (Theta=1 for a
    matched/focused reference case -- documented simplification, see
    docs/GAMMA_GAMMA.md for the general Theta, Lambda beam-parameter form).
    """
    s = float(sigma_R2)
    s125 = s ** (6.0 / 5.0)          # sigma_R^(12/5) = (sigma_R^2)^(6/5)
    alpha = 1.0 / (np.expm1(0.49 * s / (1.0 + 0.56 * (1.0 + Theta) * s125) ** (7.0 / 6.0)))
    beta = 1.0 / (np.expm1(0.51 * s / (1.0 + 0.69 * s125) ** (5.0 / 6.0)))
    return float(alpha), float(beta)


def gamma_gamma_pdf(I, alpha, beta):
    """p_I(I) for normalized irradiance I (E[I]=1). Vectorized, log-space
    for numerical stability at large alpha*beta*I (Bessel-K argument)."""
    I_in = np.asarray(I, dtype=np.float64)
    scalar_input = (I_in.ndim == 0)
    Ivec = np.atleast_1d(I_in)
    out = np.zeros_like(Ivec, dtype=np.float64)
    pos = Ivec > 0
    if np.any(pos):
        Ip = Ivec[pos]
        arg = 2.0 * np.sqrt(alpha * beta * Ip)
        # log(2) + (a+b)/2 * log(ab) - lgamma(a) - lgamma(b) + ((a+b)/2-1) * log(I)
        log_pref = (np.log(2.0) + 0.5 * (alpha + beta) * np.log(alpha * beta)
                    - gammaln(alpha) - gammaln(beta))
        log_Ipow = (0.5 * (alpha + beta) - 1.0) * np.log(Ip)
        logK = _log_kv(alpha - beta, arg)
        out[pos] = np.exp(log_pref + log_Ipow + logK)
    return float(out[0]) if scalar_input else out


def _log_kv(v, x):
    """log K_v(x). scipy.special.kv(v, x) is used directly when it is
    finite and positive, which covers the vast majority of operating
    points (alpha, beta up to a few tens). For strongly asymmetric or
    very large (alpha, beta) -- which pushes the order v=|alpha-beta| high
    enough that K_v(x) itself overflows double precision (K_v ~ Gamma(v)
    for small x, and Gamma(~200+) already exceeds float64 range) -- we
    fall back to arbitrary-precision evaluation via mpmath.besselk for
    just the offending points, then take the log there where it is safe.
    This keeps the common case fast (pure NumPy/SciPy) while remaining
    numerically correct in the tail/high-order regime instead of silently
    returning inf/nan.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    xs = np.maximum(x, 1e-300)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        k = kv(v, xs)
    logk = np.full_like(xs, -np.inf)
    good = np.isfinite(k) & (k > 0)
    logk[good] = np.log(k[good])
    bad = np.where(~good)[0]
    if bad.size:
        import mpmath
        mpmath.mp.dps = 50
        v_mp = mpmath.mpf(float(abs(v)))
        for idx in bad:
            xv = float(xs[idx])
            try:
                val = mpmath.log(mpmath.besselk(v_mp, xv))
                logk[idx] = float(mpmath.re(val))
            except (ValueError, OverflowError, TypeError):
                logk[idx] = -np.inf
    return logk


def gamma_gamma_moments(alpha, beta):
    """Analytic normalized moments: mean=1, variance, scintillation index."""
    var = 1.0 / alpha + 1.0 / beta + 1.0 / (alpha * beta)
    return {'mean': 1.0, 'variance': var, 'scintillation_index': var,
            'std': np.sqrt(var)}


class GammaGammaChannel(TurbulenceChannel):
    """Gamma-Gamma turbulence channel for moderate-to-strong scintillation.

    Parameters
    ----------
    eta_nominal : float
        Deterministic (fading-free) link transmittance; eta = eta_nominal * I.
    Cn2, wavelength, distance_m : refractive-index structure parameter,
        optical wavelength [m], link distance [m] -- used only to compute
        the Rytov variance via the unchanged legacy formula.
    wave_type : {"plane", "spherical"}
    Theta : beam curvature parameter (spherical wave only), default 1.0.
    """

    name = "gamma_gamma"

    def __init__(self, eta_nominal, Cn2, wavelength, distance_m,
                 wave_type="plane", Theta=1.0, seed=None):
        super().__init__(eta_nominal)
        self.Cn2, self.wavelength, self.distance_m = Cn2, wavelength, distance_m
        self.sigma_R2 = float(CAL.rytov_variance(Cn2, wavelength, distance_m))
        if wave_type == "plane":
            self.alpha, self.beta = alpha_beta_plane_wave(self.sigma_R2)
        elif wave_type == "spherical":
            self.alpha, self.beta = alpha_beta_spherical_wave(self.sigma_R2, Theta=Theta)
        else:
            raise ValueError("wave_type must be 'plane' or 'spherical'")
        self.wave_type = wave_type
        self._rng_seed = seed
        self._cdf_grid_built = False

    def _build_cdf_grid(self, n_grid=2500, n_q=800, I_max=None):
        """Monotone CDF grid built from the model's own generative definition
        I = X * Y with X ~ Gamma(alpha, 1/alpha), Y ~ Gamma(beta, 1/beta):

            F_I(t) = E_Y[ F_X(t / Y) ] = Integral_0^1 P(alpha, alpha t / y(q)) dq

        where P is the regularized lower incomplete gamma function and y(q) the
        Gamma(beta, 1/beta) quantile function (midpoint rule in q). Unlike a
        cumulative integral of the Bessel-K density, this is numerically stable for
        ANY (alpha, beta) -- including the huge, nearly equal shape parameters of
        weak turbulence and the very asymmetric ones of strong turbulence, where
        K_v(x) with order v = alpha - beta under/overflows double precision and
        even arbitrary-precision evaluation can fail to converge. Built lazily on
        the first .cdf() call (Monte-Carlo-only use never needs it). The closed-form
        Bessel-K PDF (gamma_gamma_pdf) is unchanged and remains the reference.
        """
        from scipy.stats import gamma as _gamma_dist
        if I_max is None:
            var = gamma_gamma_moments(self.alpha, self.beta)['variance']
            I_max = max(20.0, 1.0 + 15.0 * np.sqrt(var))
        self._I_grid = np.concatenate([[0.0], np.logspace(-8, np.log10(I_max), n_grid)])
        q = (np.arange(n_q) + 0.5) / n_q
        y = _gamma_dist.ppf(q, a=self.beta, scale=1.0 / self.beta)
        y = np.maximum(y, 1e-300)
        cdf_vals = np.empty_like(self._I_grid)
        chunk = max(1, 400_000 // n_q)
        for k in range(0, len(self._I_grid), chunk):
            t = self._I_grid[k:k + chunk]
            cdf_vals[k:k + chunk] = gammainc(self.alpha, self.alpha * t[:, None] / y[None, :]).mean(axis=1)
        cdf_vals[0] = 0.0
        self._cdf_grid = np.clip(np.maximum.accumulate(cdf_vals), 0.0, 1.0)
        self._cdf_grid_built = True

    def _ensure_cdf_grid(self):
        if not self._cdf_grid_built:
            self._build_cdf_grid()

    def _cdf_exact(self, I_val):
        val, _ = quad(lambda t: gamma_gamma_pdf(t, self.alpha, self.beta), 0.0, I_val,
                       limit=200)
        return float(np.clip(val, 0.0, 1.0))

    # -- probabilistic primitives -----------------------------------
    def pdf(self, eta):
        eta = np.asarray(eta, dtype=np.float64)
        I = eta / self.eta_nominal
        return gamma_gamma_pdf(I, self.alpha, self.beta) / self.eta_nominal

    def cdf(self, eta):
        self._ensure_cdf_grid()
        eta = np.asarray(eta, dtype=np.float64)
        I = np.clip(eta / self.eta_nominal, 0.0, None)
        return np.interp(I, self._I_grid, self._cdf_grid, left=0.0, right=1.0)

    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng(self._rng_seed)
        Ix = rng.gamma(shape=self.alpha, scale=1.0 / self.alpha, size=int(n))
        Iy = rng.gamma(shape=self.beta, scale=1.0 / self.beta, size=int(n))
        I = Ix * Iy
        return np.clip(self.eta_nominal * I, 0.0, 1.0)

    def statistics(self, n_mc=None, rng=None):
        m = gamma_gamma_moments(self.alpha, self.beta)
        return {'mean': self.eta_nominal * m['mean'],
                'variance': (self.eta_nominal ** 2) * m['variance'],
                'scintillation_index': m['scintillation_index'],
                'std': self.eta_nominal * m['std'],
                'sigma_R2': self.sigma_R2, 'alpha': self.alpha, 'beta': self.beta}
