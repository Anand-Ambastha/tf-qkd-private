# -*- coding: utf-8 -*-
"""Monte Carlo and quadrature must agree (within MC noise) on the ergodic
TF-QKD key rate, for every turbulence model. This is the central
cross-validation the task spec calls for."""
import numpy as np
from src.validation import compare_mc_vs_quadrature


def test_mc_quadrature_agreement(engine, all_stochastic_channels):
    func = lambda eta: engine.instantaneous_key_rate(0.20, eta)
    for name, ch in all_stochastic_channels:
        row = compare_mc_vs_quadrature(ch, func, n_mc=200_000, seed=11)
        # allow a few MC standard errors of slack
        tol = max(0.08, 4 * row['mc_sem'] / max(abs(row['quadrature']), 1e-12))
        assert row['relative_error'] < tol, (
            f"{name}: MC={row['monte_carlo']:.6g} quad={row['quadrature']:.6g} "
            f"rel_err={row['relative_error']:.3f} (tol={tol:.3f})")


def test_engine_ergodic_key_rate_matches_average(engine, gamma_gamma_channel):
    func = lambda eta: engine.instantaneous_key_rate(0.20, eta)
    direct = gamma_gamma_channel.average(func, method="quadrature")
    via_engine = engine.ergodic_key_rate(gamma_gamma_channel, 0.20, method="quadrature")
    assert np.isclose(direct, via_engine, rtol=1e-9)
