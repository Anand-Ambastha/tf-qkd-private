# -*- coding: utf-8 -*-
"""
tfqkd_engine.py
================
The TF-QKD engine consumes any TurbulenceChannel (deterministic,
log-normal, Gamma-Gamma, corrected Gamma-Gamma, or a user's own subclass)
through the polymorphic interface in turbulence/base.py and *only* that
interface: it never inspects which concrete class it was given, never
touches sigma_R2/alpha/beta/pointing-error internals, and never
re-derives or edits the Curty-Azuma-Lo equations in legacy_baseline.py.
The only thing turbulence changes, from the engine's point of view, is
that eta stops being a fixed number and becomes a random variable whose
expectation the channel object knows how to compute.
"""
from __future__ import annotations

import numpy as np
from . import legacy_baseline as CAL


class TFQKDEngine:
    """Twin-Field QKD secret-key-rate engine over an arbitrary channel.

    Parameters
    ----------
    pd, thetaA, thetaB, Nmax : Curty-Azuma-Lo security-proof parameters,
        passed through unchanged to legacy_baseline.key_rate /
        optimize_alpha (see src/legacy_baseline.py for the full equation
        references, Eqs. 12-35 of arXiv:1807.07667v2).
    """

    def __init__(self, pd, thetaA, thetaB, Nmax=4):
        self.pd, self.thetaA, self.thetaB, self.Nmax = pd, thetaA, thetaB, Nmax

    def instantaneous_key_rate(self, alpha, eta):
        """R(alpha, eta) at a fixed reference intensity and transmittance,
        i.e. the validated deterministic key-rate formula, unmodified."""
        eta = np.clip(np.asarray(eta, dtype=np.float64), 1e-300, 1.0)
        return np.maximum(CAL.key_rate(alpha, eta, self.pd, self.thetaA, self.thetaB, self.Nmax), 0.0)

    def optimize_alpha_at(self, eta):
        """alpha* maximizing R(alpha, eta) at a single fixed transmittance."""
        a_opt, _, R_max = CAL.optimize_alpha(float(eta), self.pd, self.thetaA, self.thetaB,
                                              delta=0.0, Nmax=self.Nmax)
        return a_opt, max(R_max, 0.0)

    def ergodic_key_rate(self, channel, alpha, method="quadrature", n_mc=20_000, rng=None):
        """E_eta[R(alpha, eta)] under `channel`'s fading statistics --
        the engine doesn't know or care whether `channel` is
        deterministic, log-normal, Gamma-Gamma, or corrected Gamma-Gamma;
        it just calls channel.average() (polymorphism, per the spec)."""
        func = lambda eta: self.instantaneous_key_rate(alpha, eta)
        return channel.average(func, method=method, n_mc=n_mc, rng=rng)

    def optimize_fixed_alpha(self, channel, method="quadrature", n_mc=20_000, rng=None,
                              alpha_bounds=(1e-3, 0.3)):
        """Best single alpha for this channel's fading statistics."""
        return channel.optimize_alpha(self.pd, self.thetaA, self.thetaB, Nmax=self.Nmax,
                                       method=method, n_mc=n_mc, rng=rng,
                                       alpha_bounds=alpha_bounds)

    def instantaneous_adaptive_analysis(self, channel, n_samples=20_000, alpha_fixed=0.20,
                                         rng=None):
        """Instantaneous adaptive-alpha analysis (per-realization alpha_i*)
        over samples drawn from `channel`. Thin pass-through to
        adaptive_alpha.instantaneous_adaptive_alpha; kept as a method here
        so callers only need the engine + a channel object."""
        from .adaptive_alpha import instantaneous_adaptive_alpha
        eta_samples = channel.sample(n_samples, rng=rng)
        return instantaneous_adaptive_alpha(eta_samples, self.pd, self.thetaA, self.thetaB,
                                             Nmax=self.Nmax, alpha_fixed=alpha_fixed)

    def distance_sweep(self, distances_km, channel_factory, alpha=0.20, method="quadrature",
                        n_mc=20_000, rng=None):
        """Ergodic SKR vs. distance for a family of channels, one per
        distance, built by channel_factory(distance_km) -> TurbulenceChannel.
        Kept generic (no turbulence-model-specific code) so it works for
        any of the four models.
        """
        distances_km = np.asarray(distances_km, dtype=np.float64)
        R = np.full_like(distances_km, np.nan)
        for i, L in enumerate(distances_km):
            ch = channel_factory(float(L))
            try:
                R[i] = self.ergodic_key_rate(ch, alpha, method=method, n_mc=n_mc, rng=rng)
            except Exception:
                R[i] = 0.0
        return R
