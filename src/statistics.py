# -*- coding: utf-8 -*-
"""
statistics.py
=============
Small shared statistics helpers used across validation, plotting and the
adaptive-alpha analysis: availability/outage from a sample of key rates,
and RMSE/relative-error table builders for the MC-vs-quadrature
validation study.
"""
from __future__ import annotations
import numpy as np


def availability(R_samples, threshold=0.0):
    """P(R > threshold), i.e. fraction of realizations with positive key rate."""
    R_samples = np.asarray(R_samples, dtype=np.float64)
    return float(np.mean(R_samples > threshold))


def outage_probability(R_samples, threshold=0.0):
    return 1.0 - availability(R_samples, threshold=threshold)


def rmse(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mean_absolute_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def summary_row(name, mc_value, quad_value):
    """One row of a MC-vs-quadrature comparison table."""
    abs_err = abs(mc_value - quad_value)
    rel_err = abs_err / abs(quad_value) if quad_value != 0 else float('nan')
    return {'quantity': name, 'monte_carlo': mc_value, 'quadrature': quad_value,
            'absolute_error': abs_err, 'relative_error': rel_err}
