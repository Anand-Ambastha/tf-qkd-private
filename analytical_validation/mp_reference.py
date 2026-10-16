# -*- coding: utf-8 -*-
"""
High-precision (60 digit) re-implementation of the CAL quantities as functions of eta,
used ONLY as an independent reference for the derivatives. The eta-independent constants
(C_kl table, cat-state coefficients, residuals D_j) are taken from the validated engine;
everything that depends on eta is recomputed in mpmath, and the derivatives come from
mpmath's high-precision numerical differentiation (no formula from the Appendix is used).
"""
import math
import mpmath as mp
from . import common as C
from .common import ALPHA, PD, OMEGA, A_NODARK, THETA_A, THETA_B, NMAX
from src import legacy_baseline as CAL

mp.mp.dps = 60
_A, _PD, _OM, _AL = mp.mpf(A_NODARK), mp.mpf(PD), mp.mpf(OMEGA), mp.mpf(ALPHA)
_CTAB = {}


def _ctab(nA, nB):
    if (nA, nB) not in _CTAB:
        _CTAB[(nA, nB)] = CAL._qzz_coeff_table(nA, nB, THETA_A, THETA_B)
    return _CTAB[(nA, nB)]


def pxx(eta):
    g = _AL ** 2 * eta
    return _A * mp.e ** (-g) * mp.cosh(_OM * g) - _A ** 2 * mp.e ** (-2 * g)


def ex(eta):
    g = _AL ** 2 * eta
    num = mp.e ** (-_OM * g) - _A * mp.e ** (-g)
    den = mp.e ** (-_OM * g) + mp.e ** (_OM * g) - 2 * _A * mp.e ** (-g)
    return num / den


def _pzz(nA, nB, x):
    N = nA + nB
    ct = _ctab(nA, nB)
    s = (_PD - 1) * (1 - x) ** N
    for k in range(nA + 1):
        for l in range(nB + 1):
            if ct[k, l] != 0.0:
                s += mp.mpf(float(ct[k, l])) * x ** (k + l) * (1 - x) ** (N - k - l)
    return _A * s


def ez(eta):
    U = mp.mpf(0)
    for j in (0, 1):
        m_max = (NMAX - j) // 2
        c = CAL._cat_coeff_vec(j, ALPHA, m_max)
        S = mp.mpf(0)
        for mA in range(m_max + 1):
            for mB in range(m_max + 1):
                nA, nB = 2 * mA + j, 2 * mB + j
                if nA + nB > NMAX:
                    continue
                S += mp.mpf(float(c[mA])) * mp.mpf(float(c[mB])) * mp.sqrt(_pzz(nA, nB, eta))
        U += (S + mp.mpf(CAL.residual_j(j, ALPHA, NMAX))) ** 2
    return U / pxx(eta)


def _h(x):
    return -x * mp.log(x, 2) - (1 - x) * mp.log(1 - x, 2)


def R(eta):
    return 2 * pxx(eta) * (1 - _h(ex(eta)) - _h(ez(eta)))


FUNS = {'pXX': pxx, 'eX': ex, 'eZ': ez, 'R': R}


def derivs(name, eta, n_max=4):
    f = FUNS[name]
    x0 = mp.mpf(float(eta))
    return [float(mp.diff(f, x0, n)) if n else float(f(x0)) for n in range(n_max + 1)]
