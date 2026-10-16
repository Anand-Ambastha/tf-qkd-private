# -*- coding: utf-8 -*-
"""
two_link.py
===========
End-to-end TF-QKD channel made of TWO INDEPENDENTLY FADING free-space links:
Alice->Charlie (A-C) and Bob->Charlie (B-C).

    eta = eta_A * eta_B,     eta_A ~ link_A fading,  eta_B ~ link_B fading,
                             eta_A  independent of  eta_B.

Why this is not just "one channel at distance L"
------------------------------------------------
The two arms cross different air columns, so their scintillation is
uncorrelated. Modelling the A-B path as a single fading factor evaluated at the
full length L (a) uses sigma_R^2(L) instead of sigma_R^2(L/2) per arm and
(b) treats the product of two independent random variables as one. For the
log-normal case sigma_R^2 ~ L^(11/6), so the two-arm log-variance is
2 * sigma_R^2(L/2) = 2^(-5/6) sigma_R^2(L) ~ 0.56 sigma_R^2(L): the one-factor
model overstates it by ~1.78x. For Gamma-Gamma the product of two independent
GG variables is not GG at all, so the one-factor shortcut has no exact analogue.

Every ergodic quantity is a 2-D average,

    E[g(eta)] = Integral Integral g(eta_A eta_B) p_A(eta_A) p_B(eta_B) d eta_A d eta_B,

evaluated here by a tensor-product rule built from each link's own
quadrature_nodes() (exact Gauss-Hermite for log-normal, equal-probability
quantile nodes for every other model), or by Monte Carlo with independent draws.

Modelling assumption: key_rate() is the symmetric-channel Curty-Azuma-Lo
formula evaluated at the effective transmittance eta = eta_A eta_B. An
amplitude imbalance between the two arms (eta_A != eta_B instantaneously)
would further reduce interference visibility; that extra penalty is not
modelled. Results are therefore an upper bound on a fully uncompensated
system.
"""
from __future__ import annotations

import numpy as np

from .base import TurbulenceChannel


class TwoLinkChannel(TurbulenceChannel):
    """eta = eta_A * eta_B with independent fading on each link.

    Parameters
    ----------
    link_A, link_B : TurbulenceChannel
        Single-link channels (e.g. two LogNormalChannel instances, each built
        with that link's own length and nominal transmittance).
    seed : optional int, seeds .sample() when no rng is passed. One rng stream
        feeds both links sequentially, so identical per-link seeds can never
        make the two links perfectly correlated.
    """

    name = "two_link"

    def __init__(self, link_A: TurbulenceChannel, link_B: TurbulenceChannel, seed=None):
        super().__init__(min(link_A.eta_nominal * link_B.eta_nominal, 1.0))
        self.link_A, self.link_B = link_A, link_B
        self._rng_seed = seed
        self.name = f"two_link_{link_A.name}"

    # -- convenience metadata ----------------------------------------
    @property
    def sigma_R2_A(self):
        return getattr(self.link_A, 'sigma_R2', float('nan'))

    @property
    def sigma_R2_B(self):
        return getattr(self.link_B, 'sigma_R2', float('nan'))

    @property
    def sigma_R2(self):
        """Per-link Rytov variance that decides the validity regime (the larger arm)."""
        return max(self.sigma_R2_A, self.sigma_R2_B)

    @property
    def distance_m(self):
        return getattr(self.link_A, 'distance_m', 0.0) + getattr(self.link_B, 'distance_m', 0.0)

    # -- sampling: independent draws ---------------------------------
    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng(self._rng_seed)
        eta_A = self.link_A.sample(n, rng=rng)
        eta_B = self.link_B.sample(n, rng=rng)
        return eta_A * eta_B

    def sample_links(self, n, rng=None):
        """(eta, eta_A, eta_B), e.g. to verify independence or to study imbalance."""
        if rng is None:
            rng = np.random.default_rng(self._rng_seed)
        a = self.link_A.sample(n, rng=rng)
        b = self.link_B.sample(n, rng=rng)
        return a * b, a, b

    # -- density of the product (Mellin convolution, numerical) -------
    #    p(eta) = Integral p_A(x) p_B(eta / x) / x dx, with link A replaced by its
    #    equal-probability quantile nodes (vectorized over nodes x eta points).
    def _mellin(self, eta, link_B_fn, n_nodes, with_jacobian):
        eta = np.asarray(eta, dtype=np.float64)
        nodes, w = self.link_A.quantile_nodes(n_nodes)
        flat = eta.reshape(-1)
        out = np.empty_like(flat)
        chunk = max(1, 400_000 // len(nodes))
        for k in range(0, len(flat), chunk):
            x = flat[None, k:k + chunk] / nodes[:, None]
            vals = np.asarray(link_B_fn(x), dtype=np.float64)
            if with_jacobian:
                vals = vals / nodes[:, None]
            out[k:k + chunk] = (w[:, None] * vals).sum(axis=0)
        return out.reshape(eta.shape)

    def pdf(self, eta, n_nodes=800):
        return self._mellin(eta, self.link_B.pdf, n_nodes, True)

    def cdf(self, eta, n_nodes=800):
        return self._mellin(eta, self.link_B.cdf, n_nodes, False)

    # -- moments from independence ------------------------------------
    def statistics(self, n_mc=20_000, rng=None):
        sA, sB = self.link_A.statistics(n_mc=n_mc, rng=rng), self.link_B.statistics(n_mc=n_mc, rng=rng)
        mean = sA['mean'] * sB['mean']
        second = (sA['variance'] + sA['mean'] ** 2) * (sB['variance'] + sB['mean'] ** 2)
        var = second - mean ** 2
        return {'mean': mean, 'variance': var, 'std': float(np.sqrt(max(var, 0.0))),
                'scintillation_index': var / mean ** 2 if mean > 0 else float('nan'),
                'link_A': sA, 'link_B': sB}

    # -- the 2-D average ---------------------------------------------
    def average(self, func, method="quadrature", n_mc=20_000, rng=None, quad_kwargs=None):
        """E[func(eta_A * eta_B)] over BOTH independent fading variables.

        method="quadrature": tensor-product rule over the two links' nodes
            (one vectorized func call on n_A * n_B points). quad_kwargs may
            hold n_nodes (applied to both links) or n_nodes_A / n_nodes_B.
        method="monte_carlo": mean of func over n_mc independent (eta_A, eta_B) draws.
        """
        if method == "monte_carlo":
            return float(np.mean(np.asarray(func(self.sample(n_mc, rng=rng)), dtype=np.float64)))
        if method != "quadrature":
            raise ValueError(f"unknown method '{method}'")
        kw = dict(quad_kwargs or {})
        n_all = kw.get('n_nodes')
        eA, wA = self.link_A.quadrature_nodes(kw.get('n_nodes_A', n_all))
        eB, wB = self.link_B.quadrature_nodes(kw.get('n_nodes_B', n_all))
        eta = np.outer(eA, eB).ravel()
        w = np.outer(wA, wB).ravel()
        return float(np.sum(w * np.asarray(func(eta), dtype=np.float64)))

    def quadrature_nodes(self, n=None):
        """Joint nodes of eta = eta_A * eta_B (n_A * n_B of them)."""
        eA, wA = self.link_A.quadrature_nodes(n)
        eB, wB = self.link_B.quadrature_nodes(n)
        return np.outer(eA, eB).ravel(), np.outer(wA, wB).ravel()

    def __repr__(self):
        return f"TwoLinkChannel(A={self.link_A!r}, B={self.link_B!r})"
