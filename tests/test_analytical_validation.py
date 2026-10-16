"""Guards for the analytical_validation package: derivative machinery and two printed manuscript values."""
import math
import numpy as np
from analytical_validation import common as C, cal_derivatives as D, mp_reference as M, averaging as A


def _eta_med(L):
    return C.eta0_eff(L) * math.exp(-C.rytov(L / 2) / 2)


def test_analytic_derivatives_match_high_precision_reference():
    eta = _eta_med(100)
    d = D.all_derivs(eta)
    for q in ('pXX', 'eX', 'eZ', 'R'):
        ref = M.derivs(q, eta)
        for n in range(5):
            assert abs(d[q][n] - ref[n]) / abs(ref[n]) < 1e-7


def test_printed_pxx_bracket_is_not_the_engine_value():
    eta = C.eta0_eff(80)
    assert abs(D.pxx_derivs(eta)[0] - C.pxx_ref(eta)) / C.pxx_ref(eta) < 1e-9
    assert D.pxx_derivs(eta, printed_form=True)[0] > 0.1          # Eq.(6) as printed is wrong by orders of magnitude


def test_lognormal_formula_reproduces_manuscript_table_vi_at_160km():
    L = 160
    eta0, s2 = C.eta0_eff(L), C.rytov(L / 2) / 2
    emed = eta0 * math.exp(-s2)
    d = D.all_derivs(emed)['R']
    ref = A.ln_reference('R', eta0, s2)
    assert abs(C.pct(A.ln_expansion(d, emed, s2, 2), ref) - (-1.341)) < 0.01
    assert abs(C.pct(A.ln_expansion(d, emed, s2, 4), ref) - (-0.072)) < 0.005


def test_gamma_gamma_exponent_convention():
    assert max(A.gg_si(*A.gg_params(s, 0.0)) for s in np.linspace(0.05, 10, 200)) > 1.2        # correct: saturates near 1.2-1.3
    assert A.gg_si(*A.gg_params(10.0, 0.0, 2.4)) < 0.1                                          # report convention: decays to ~0.08
