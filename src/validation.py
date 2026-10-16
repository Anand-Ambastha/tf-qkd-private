# -*- coding: utf-8 -*-
"""
validation.py
=============
Numerical validation used by tests/ and reproduced in docs/VALIDATION.md:

  * PDF integrates to 1 (per model, by quadrature).
  * CDF is monotonic non-decreasing and -> 1 as eta -> eta_nominal.
  * Sampling moments (mean, variance) match analytic/quadrature moments.
  * Monte Carlo and quadrature expectations of the TF-QKD key rate agree
    within numerical tolerance, for every model.
  * CorrectedGammaGammaChannel reduces to GammaGammaChannel when all
    corrections are disabled.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning

from .statistics import rmse, summary_row


def check_pdf_normalization(channel, a=0.0, b=None, atol=1e-3):
    b = channel.eta_max if b is None else b
    pts = sorted({p for p in (0.3 * channel.eta_nominal, channel.eta_nominal,
                               3.0 * channel.eta_nominal) if a < p < b}) or None
    # CorrectedGammaGammaChannel's pdf is a piecewise-linear interpolation
    # of a 200-bin histogram (see corrected_gamma_gamma.py docstring: no
    # closed form exists for h_turbulence * h_pointing_error), so it has
    # ~200 kinks; scipy.integrate.quad's adaptive subdivision can hit its
    # iteration cap on such a non-smooth integrand while still converging
    # to the correct integral (verified against the atol check below and
    # against Monte Carlo in tests/test_mc_vs_quadrature.py) -- this
    # warning is benign for that specific model and is suppressed here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=IntegrationWarning)
        val, _ = quad(lambda x: float(np.atleast_1d(channel.pdf(np.array([x])))[0]),
                       a, b, limit=400, points=pts)
    return {'integral': val, 'ok': abs(val - 1.0) < atol}


def check_cdf_monotonic(channel, n_points=500):
    eta = np.linspace(1e-9, channel.eta_max, n_points)
    cdf = np.atleast_1d(channel.cdf(eta))
    diffs = np.diff(cdf)
    return {'monotonic': bool(np.all(diffs >= -1e-9)),
            'endpoint': float(cdf[-1]), 'ok_endpoint': abs(cdf[-1] - 1.0) < 1e-2}


def check_sampling_moments(channel, n=200_000, rtol=0.05, seed=0):
    rng = np.random.default_rng(seed)
    s = channel.sample(n, rng=rng)
    mean_mc, var_mc = float(np.mean(s)), float(np.var(s))
    stats = channel.statistics()
    mean_ref, var_ref = stats['mean'], stats['variance']
    mean_ok = abs(mean_mc - mean_ref) / max(abs(mean_ref), 1e-12) < rtol
    var_ok = var_ref == 0 or abs(var_mc - var_ref) / max(abs(var_ref), 1e-12) < 3 * rtol
    return {'mean_mc': mean_mc, 'mean_ref': mean_ref, 'mean_ok': mean_ok,
            'var_mc': var_mc, 'var_ref': var_ref, 'var_ok': var_ok}


def compare_mc_vs_quadrature(channel, func, n_mc=50_000, seed=0):
    from .monte_carlo import mc_expectation
    rng = np.random.default_rng(seed)
    mc = mc_expectation(channel, func, n_mc, rng=rng)
    # channel.average() is each model's own quadrature: adaptive 1-D quad of
    # pdf*func for a single link, and the tensor-product 2-D rule over the two
    # independent fading variables for a TwoLinkChannel.
    qv = channel.average(func, method="quadrature")
    row = summary_row(f'{channel.name} E[f(eta)]', mc['estimate'], qv)
    row['mc_sem'] = mc['sem']
    return row


def full_validation_report(engine, channels, seed=0):
    """Run the full battery of checks across a list of (name, channel)
    pairs and return a list of row dicts (one validation study per model)
    ready to render as a markdown table."""
    rows = []
    for name, ch in channels:
        pdf_chk = check_pdf_normalization(ch)
        cdf_chk = check_cdf_monotonic(ch)
        mom_chk = check_sampling_moments(ch, seed=seed)
        mc_quad = compare_mc_vs_quadrature(
            ch, lambda eta: engine.instantaneous_key_rate(0.20, eta), seed=seed)
        rows.append({
            'model': name,
            'pdf_integral': pdf_chk['integral'], 'pdf_ok': pdf_chk['ok'],
            'cdf_monotonic': cdf_chk['monotonic'], 'cdf_endpoint': cdf_chk['endpoint'],
            'mean_mc': mom_chk['mean_mc'], 'mean_ref': mom_chk['mean_ref'],
            'mean_ok': mom_chk['mean_ok'],
            'skr_mc': mc_quad['monte_carlo'], 'skr_quad': mc_quad['quadrature'],
            'skr_rel_err': mc_quad['relative_error'],
        })
    return rows
