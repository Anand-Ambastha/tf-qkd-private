# -*- coding: utf-8 -*-
"""
adaptive_alpha.py
==================
Reference-intensity (alpha) optimization for TF-QKD over a fading
channel, at three levels documented in the project:

  fixed        : a single alpha, chosen once, used for every realization.
  optimized    : a single alpha chosen to maximize the *ergodic*
                 (turbulence-averaged) key rate E_eta[R(alpha,eta)].
  instantaneous: a different alpha_i* = argmax_alpha R(alpha, eta_i) for
                 *every* channel realization eta_i (requires the receiver
                 to know the instantaneous channel state -- an idealized
                 upper bound on adaptive-alpha gain).

All three call the unchanged, validated `optimize_alpha` /
`key_rate` from legacy_baseline.py; nothing about the TF-QKD security
proof is touched here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import legacy_baseline as CAL


def optimize_fixed_alpha(channel, pd, thetaA, thetaB, Nmax=4, method="quadrature",
                          n_mc=20_000, rng=None, alpha_bounds=(1e-3, 0.3)):
    """argmax_alpha E_eta[ key_rate(alpha, eta, pd, thetaA, thetaB, Nmax) ]
    over a turbulence channel, i.e. the single best *fixed* alpha for this
    channel's fading statistics.
    """
    theta = thetaA - thetaB

    def neg_ergodic_rate(alpha):
        func = lambda eta: np.maximum(
            CAL.key_rate(alpha, np.clip(eta, 1e-300, 1.0), pd, thetaA, thetaB, Nmax), 0.0)
        val = channel.average(func, method=method, n_mc=n_mc, rng=rng)
        return -val

    res = minimize_scalar(neg_ergodic_rate, bounds=alpha_bounds, method='bounded',
                           options={'xatol': 1e-5})
    alpha_opt = float(res.x)
    R_opt = -float(res.fun)
    return alpha_opt, R_opt


def instantaneous_adaptive_alpha(eta_samples, pd, thetaA, thetaB, Nmax=4,
                                  alpha_bounds=(1e-3, 0.3), alpha_fixed=0.20):
    """For every realization eta_i, solve alpha_i* = argmax_alpha R(alpha, eta_i).
    Also evaluates the fixed-alpha key rate at the same realizations for
    a like-for-like comparison. Returns a dict of arrays plus summary
    statistics and the alpha*-vs-eta correlation, matching the analysis
    already produced by instantaneous_adaptive_alpha.py / plot_adaptive_alpha.py
    in the original repository (kept here so the new framework can drive
    it for *any* turbulence model, not just log-normal).
    """
    eta_samples = np.asarray(eta_samples, dtype=np.float64)
    n = len(eta_samples)
    alpha_star = np.empty(n)
    R_adaptive = np.empty(n)
    R_fixed = np.empty(n)

    for i, eta in enumerate(eta_samples):
        eta_c = float(np.clip(eta, 1e-300, 1.0))
        a_opt, _, R_max = CAL.optimize_alpha(eta_c, pd, thetaA, thetaB, delta=0.0, Nmax=Nmax)
        alpha_star[i] = a_opt
        R_adaptive[i] = max(R_max, 0.0)
        R_fixed[i] = max(float(CAL.key_rate(alpha_fixed, eta_c, pd, thetaA, thetaB, Nmax)), 0.0)

    zero_fixed_mask = R_fixed <= 0
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_improve = np.where(zero_fixed_mask, np.nan,
                                100.0 * (R_adaptive - R_fixed) / np.where(R_fixed > 0, R_fixed, np.nan))

    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(eta_samples, alpha_star)
    sr, sp = spearmanr(eta_samples, alpha_star)
    slope, intercept = np.polyfit(eta_samples, alpha_star, 1)

    def _stats_block(R):
        return {'mean': float(np.mean(R)), 'median': float(np.median(R)),
                'std': float(np.std(R)), 'availability': float(np.mean(R > 0)),
                'outage_probability': float(np.mean(R <= 0))}

    return {
        'eta_samples': eta_samples, 'alpha_star': alpha_star,
        'R_adaptive': R_adaptive, 'R_fixed': R_fixed, 'alpha_fixed': alpha_fixed,
        'stats_adaptive': _stats_block(R_adaptive), 'stats_fixed': _stats_block(R_fixed),
        'pct_improve': pct_improve, 'zero_fixed_mask': zero_fixed_mask,
        'correlation': {'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr,
                         'spearman_p': sp, 'slope': slope, 'intercept': intercept},
    }
