# -*- coding: utf-8 -*-
"""
quadrature.py
=============
Three interchangeable quadrature backends for computing expectations
E[f(eta)] = integral f(eta) p(eta) d(eta) against a turbulence channel's
pdf, plus a benchmark harness comparing them on accuracy and runtime.

Methods
-------
scipy_quad       : scipy.integrate.quad, adaptive Clenshaw-Curtis/QUADPACK.
                    Most robust default; used by TurbulenceChannel.average().
gauss_legendre   : fixed-order Gauss-Legendre rule on [a,b] (fast, good for
                    smooth integrands away from singularities).
gauss_laguerre   : Gauss-Laguerre rule for semi-infinite integrals
                    (integral_0^inf f(x) e^-x dx); useful for the
                    Gamma-Gamma tail where the natural domain is [0, inf).
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss


def quad_expectation(pdf, func, a, b, epsabs=1e-10, epsrel=1e-8, limit=200, points=None):
    """E[func(eta)] via adaptive scipy quadrature, integrating pdf(eta)*func(eta).

    Fading-channel PDFs (Gamma-Gamma in particular) can be narrow relative
    to the physical support [0, eta_max=1], e.g. std/mean ~ 0.1-0.3 around
    a mean much smaller than 1 at large link loss. The initial coarse
    Gauss-Kronrod pass of an adaptive rule over the whole interval can miss
    such a peak and return a small error estimate for an incorrect value.
    Explicit interior `points` breakpoints are therefore passed to quad so
    that it subdivides around the region where the density is supported.
    See docs/VALIDATION.md for the comparison against Monte Carlo.
    """
    integrand = lambda x: float(np.asarray(pdf(x)).reshape(-1)[0] * np.asarray(func(np.array([x]))).reshape(-1)[0])
    pts = None
    if points is not None:
        pts = sorted({float(p) for p in points if a < float(p) < b})
        pts = pts or None
    # The subdivision-limit warning is expected for piecewise-linear
    # histogram-based pdfs (e.g. the corrected Gamma-Gamma model); see
    # validation.check_pdf_normalization. It is suppressed here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=IntegrationWarning)
        val, _ = quad(integrand, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit, points=pts)
    return val


def gauss_legendre_expectation(pdf, func, a, b, n=64):
    """E[func(eta)] via a fixed n-point Gauss-Legendre rule on [a,b]."""
    x, w = leggauss(n)
    # map from [-1,1] to [a,b]
    xm = 0.5 * (b - a) * x + 0.5 * (b + a)
    wm = 0.5 * (b - a) * w
    vals = pdf(xm) * func(xm)
    return float(np.sum(wm * vals))


def gauss_laguerre_expectation(density_over_exp, func, n=64):
    """E[func(x)] for x in [0, inf) written as integral f(x) g(x) e^-x dx,
    where density_over_exp(x) = p(x) * e^x (i.e. the pdf with its natural
    exponential tail factored out). Useful for the log-normal / Gamma-Gamma
    tails where p(x) ~ e^-x asymptotically.
    """
    x, w = laggauss(n)
    vals = density_over_exp(x) * func(x)
    return float(np.sum(w * vals))


def benchmark_quadrature(channel, func, a=None, b=None, n_values=(8, 16, 32, 64, 128),
                          reference_method="scipy_quad"):
    """Compare scipy_quad (as ground truth) against Gauss-Legendre at
    several orders n: accuracy (relative error) and wall-clock runtime.
    Returns a list of dict rows suitable for a markdown/CSV table.
    """
    a = 0.0 if a is None else a
    b = channel.eta_nominal if b is None else b

    t0 = time.perf_counter()
    ref = quad_expectation(channel.pdf, func, a, b)
    t_ref = time.perf_counter() - t0

    rows = [{'method': 'scipy_quad (reference)', 'n': None, 'value': ref,
             'relative_error': 0.0, 'runtime_s': t_ref}]
    for n in n_values:
        t0 = time.perf_counter()
        val = gauss_legendre_expectation(channel.pdf, func, a, b, n=n)
        dt = time.perf_counter() - t0
        rel_err = abs(val - ref) / abs(ref) if ref != 0 else abs(val)
        rows.append({'method': 'gauss_legendre', 'n': n, 'value': val,
                     'relative_error': rel_err, 'runtime_s': dt})
    return rows
