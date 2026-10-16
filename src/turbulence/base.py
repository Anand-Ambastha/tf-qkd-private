# -*- coding: utf-8 -*-
"""
base.py
=======
Abstract interface that every turbulence/channel model must implement.
The TF-QKD engine (tfqkd_engine.py) talks to channels *only* through this
interface, so it never needs to know whether it is looking at a
deterministic link, log-normal turbulence, Gamma-Gamma turbulence, or the
corrected Gamma-Gamma model with aperture averaging and pointing errors.

All models represent the instantaneous *transmittance* eta as the random
variable of interest, with the physical constraint 0 <= eta <= eta_max
(eta_max = deterministic/nominal transmittance in the absence of fading).
Internally, most turbulence literature works with normalized irradiance
I = eta / eta_nominal with E[I] = 1; the classes below expose both
representations where useful, but the public contract (pdf/cdf/sample/...)
is always in terms of eta so the TF-QKD engine has one thing to reason
about.
"""
from __future__ import annotations

import abc
import numpy as np


class TurbulenceChannel(abc.ABC):
    """Unified interface for all atmospheric turbulence / channel models."""

    name: str = "base"

    def __init__(self, eta_nominal: float):
        if not (0.0 < eta_nominal <= 1.0):
            raise ValueError(f"eta_nominal must lie in (0,1], got {eta_nominal}")
        self.eta_nominal = float(eta_nominal)

    @property
    def eta_max(self):
        """Physical upper bound of the support of eta. NOTE: this is *not*
        eta_nominal -- fading is multiplicative around a mean-1 normalized
        irradiance, so instantaneous eta can exceed eta_nominal (up to the
        hard physical ceiling of unit transmittance) just as it can fall
        far below it. Integrating expectations only up to eta_nominal would
        silently discard the upper half (or more) of the distribution's
        probability mass; every quadrature call in this codebase integrates
        up to eta_max, not eta_nominal.
        """
        return 1.0

    # ------------------------------------------------------------------
    # Required probabilistic primitives
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def pdf(self, eta):
        """Probability density of the instantaneous transmittance eta."""

    @abc.abstractmethod
    def cdf(self, eta):
        """Cumulative distribution function of eta."""

    @abc.abstractmethod
    def sample(self, n, rng=None):
        """Draw n i.i.d. samples of eta."""

    # ------------------------------------------------------------------
    # Derived quantities (shared implementation, may be overridden)
    # ------------------------------------------------------------------
    def statistics(self, n_mc=20_000, rng=None):
        """Return mean, variance, scintillation index (MC-estimated by
        default; analytic subclasses override with closed forms)."""
        s = self.sample(n_mc, rng=rng)
        mean = float(np.mean(s))
        var = float(np.var(s))
        si = var / mean ** 2 if mean > 0 else float('nan')
        return {'mean': mean, 'variance': var, 'scintillation_index': si,
                'std': float(np.sqrt(var))}

    def outage_probability(self, eta_th):
        """P(eta <= eta_th), i.e. the CDF at the outage threshold."""
        return float(self.cdf(eta_th))

    def average(self, func, method="quadrature", n_mc=20_000, rng=None,
                quad_kwargs=None):
        """E[func(eta)] via quadrature (default) or Monte Carlo.

        Parameters
        ----------
        func : callable
            Scalar-or-vectorized function of eta (e.g. the TF-QKD key-rate
            function at fixed alpha/pd/geometry).
        method : {"quadrature", "monte_carlo"}
        """
        if method == "monte_carlo":
            s = self.sample(n_mc, rng=rng)
            return float(np.mean(func(s)))
        elif method == "quadrature":
            from ..quadrature import quad_expectation
            kwargs = dict(quad_kwargs or {})
            kwargs.setdefault('points', [0.3 * self.eta_nominal, self.eta_nominal,
                                          3.0 * self.eta_nominal])
            return quad_expectation(self.pdf, func, 0.0, self.eta_max, **kwargs)
        raise ValueError(f"unknown method '{method}'")

    # ------------------------------------------------------------------
    # Discrete representation of the fading law (used for multi-link
    # products, where the end-to-end average is a multi-dimensional integral)
    # ------------------------------------------------------------------
    def quantile_nodes(self, n=400):
        """(eta_k, w_k): n quantile nodes with TAIL-REFINED probability cells.

        The probability axis [0, 1] is cut at u_j = (1 - cos(pi j / n)) / 2, which
        clusters cells at both ends; w_k = u_{k+1} - u_k is the cell probability and
        eta_k = F^-1((u_k + u_{k+1}) / 2) its probability-midpoint quantile, built from
        .cdf() only so it works for every model. Probability above eta_max (the clipped
        atom at eta = 1) lands on eta = 1, matching what .sample() does.

        Why not equal-probability cells (v2.1): the ergodic key rate near the outage
        threshold is dominated by RARE UPWARD fades, which equal-probability midpoint
        cells under-represent (a 400-node rule was ~6 % low there vs a 10^7-sample Monte
        Carlo); cosine spacing resolves both tails at the same node count.
        """
        cache = self.__dict__.setdefault('_node_cache', {})
        if ('q2', n) in cache:
            return cache[('q2', n)]
        lo = max(self.eta_nominal * 1e-8, 1e-300)
        hi = float(self.eta_max)
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), 4000))
        F = np.maximum.accumulate(np.clip(np.asarray(self.cdf(grid), dtype=np.float64), 0.0, 1.0))
        u_edges = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
        w = np.diff(u_edges)
        u = 0.5 * (u_edges[:-1] + u_edges[1:])
        eta = np.exp(np.interp(u, F, np.log(grid)))
        out = (eta, w)
        cache[('q2', n)] = out
        return out

    def quadrature_nodes(self, n=None):
        """(eta_k, w_k) with sum_k w_k g(eta_k) ~ E[g(eta)]. Default: quantile
        nodes; models with an exact rule (log-normal) override this."""
        return self.quantile_nodes(400 if n is None else int(n))

    def average_skr(self, skr_func, method="quadrature", n_mc=20_000, rng=None):
        """Ergodic average secret key rate E[skr_func(eta)]. Thin wrapper
        around .average() kept for interface symmetry with the spec /
        readability at call sites."""
        return self.average(skr_func, method=method, n_mc=n_mc, rng=rng)

    def monte_carlo(self, func, n=20_000, rng=None):
        """Convenience alias: E[func(eta)] via plain Monte Carlo."""
        return self.average(func, method="monte_carlo", n_mc=n, rng=rng)

    def quadrature(self, func, **kwargs):
        """Convenience alias: E[func(eta)] via numerical quadrature."""
        return self.average(func, method="quadrature", quad_kwargs=kwargs)

    def optimize_alpha(self, pd, thetaA, thetaB, Nmax=4, method="quadrature",
                        n_mc=20_000, rng=None, alpha_bounds=(1e-3, 0.3)):
        """Find the fixed reference intensity alpha that maximizes the
        *ergodic* (turbulence-averaged) TF-QKD key rate under this channel.
        Delegates the per-alpha expectation to .average(); the 1-D search
        itself is delegated to adaptive_alpha.optimize_fixed_alpha so the
        optimizer logic isn't duplicated per channel subclass.
        """
        from ..adaptive_alpha import optimize_fixed_alpha
        return optimize_fixed_alpha(self, pd, thetaA, thetaB, Nmax=Nmax,
                                     method=method, n_mc=n_mc, rng=rng,
                                     alpha_bounds=alpha_bounds)

    # ------------------------------------------------------------------
    # Plotting (thin wrappers; real work lives in plotting.py so figure
    # styling is centralized and consistent across all models)
    # ------------------------------------------------------------------
    def plot_pdf(self, ax=None, **kwargs):
        from ..plotting import plot_channel_pdf
        return plot_channel_pdf(self, ax=ax, **kwargs)

    def plot_cdf(self, ax=None, **kwargs):
        from ..plotting import plot_channel_cdf
        return plot_channel_cdf(self, ax=ax, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(eta_nominal={self.eta_nominal:.6g})"
