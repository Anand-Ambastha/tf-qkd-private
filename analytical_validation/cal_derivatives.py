# -*- coding: utf-8 -*-
"""
Analytic eta-derivatives (orders 0..4) of the CAL quantities, implemented from the
manuscript Appendix (A-H) and the validated engine's definitions.

All derivatives are with respect to eta (gamma = alpha^2 eta, x = eta), returned as
arrays d[0..4].

Two points where the manuscript text and the validated CAL/engine code differ, and
which this module resolves in favour of CAL Eq.(30) / the engine (verified numerically
in 01_test_lognormal_formulas.py, test T1):
  * pXX: the A^2 term sits OUTSIDE the factor 1/2:
        pXX = 1/2 * A * [e^{-(1-O)g} + e^{-(1+O)g}] - A^2 e^{-2g}
    (manuscript Eq.(6) / Appendix A print it inside the bracket).
  * D_j (truncation residual) does not depend on eta (engine: residual_j), so its
    derivatives of order >= 1 vanish.
"""
import math
import numpy as np
from math import comb, factorial
from scipy.special import gammaln
from . import common as C
from .common import ALPHA, PD, OMEGA, A_NODARK, THETA_A, THETA_B, NMAX
from src import legacy_baseline as CAL

ORD = 4


def _falling(a, i):
    return math.factorial(a) // math.factorial(a - i) if 0 <= i <= a else 0


# ---------------------------------------------------------------- X basis
def pxx_derivs(eta, printed_form=False):
    """d^n pXX / d eta^n, n=0..4. printed_form=True uses the manuscript Eq.(6) bracket."""
    g = ALPHA ** 2 * eta
    A, Om = A_NODARK, OMEGA
    out = np.zeros(ORD + 1)
    c2 = 0.5 if printed_form else 1.0            # factor in front of the A^2 term
    for n in range(ORD + 1):
        t = 0.5 * (A * (-(1 - Om)) ** n * math.exp(-(1 - Om) * g)
                   + A * (-(1 + Om)) ** n * math.exp(-(1 + Om) * g))
        t -= (0.5 if printed_form else 1.0) * A ** 2 * (-2.0) ** n * math.exp(-2 * g) * (1.0 if not printed_form else 1.0) * (c2 / c2)
        if printed_form:
            t = 0.5 * (A * (-(1 - Om)) ** n * math.exp(-(1 - Om) * g)
                       + A * (-(1 + Om)) ** n * math.exp(-(1 + Om) * g)
                       - A ** 2 * (-2.0) ** n * math.exp(-2 * g))
        out[n] = t * ALPHA ** (2 * n)             # gamma-derivative -> eta-derivative
    return out


def ex_derivs(eta):
    """Appendix B recursion  e^(n) = [N^(n) - sum_k C(n,k) D^(k) e^(n-k)] / D."""
    g = ALPHA ** 2 * eta
    A, Om = A_NODARK, OMEGA
    N = [(-Om) ** n * math.exp(-Om * g) - A * (-1) ** n * math.exp(-g) for n in range(ORD + 1)]
    D = [(-Om) ** n * math.exp(-Om * g) + Om ** n * math.exp(Om * g) - 2 * A * (-1) ** n * math.exp(-g)
         for n in range(ORD + 1)]
    e = np.zeros(ORD + 1)
    e[0] = N[0] / D[0]
    for n in range(1, ORD + 1):
        e[n] = (N[n] - sum(comb(n, k) * D[k] * e[n - k] for k in range(1, n + 1))) / D[0]
    return e * ALPHA ** (2 * np.arange(ORD + 1))


# ---------------------------------------------------------------- Z basis
def _pzz_derivs(nA, nB, x):
    """d^n/dx^n of p_ZZ^(nA,nB)(x) = A[ pd(1-x)^N + sum_kl C_kl x^(k+l)(1-x)^(N-k-l) - (1-x)^N ]."""
    N = nA + nB
    Ctab = CAL._qzz_coeff_table(nA, nB, THETA_A, THETA_B)

    def d_xa_1mx_b(a, b, n):                      # d^n [x^a (1-x)^b]
        s = 0.0
        for i in range(n + 1):
            f1 = _falling(a, i) * x ** (a - i) if i <= a else 0.0
            m = n - i
            f2 = ((-1) ** m) * _falling(b, m) * (1 - x) ** (b - m) if m <= b else 0.0
            s += comb(n, i) * f1 * f2
        return s

    out = np.zeros(ORD + 1)
    for n in range(ORD + 1):
        v = PD * d_xa_1mx_b(0, N, n) - d_xa_1mx_b(0, N, n)
        for k in range(nA + 1):
            for l in range(nB + 1):
                c = Ctab[k, l]
                if c != 0.0:
                    v += c * d_xa_1mx_b(k + l, N - k - l, n)
        out[n] = A_NODARK * v
    return out


def _sqrt_derivs(p):
    """Appendix E: d^n sqrt(p) for n=0..4 from d^n p."""
    p0, p1, p2, p3, p4 = p
    s = math.sqrt(p0)
    return np.array([
        s,
        p1 / (2 * s),
        p2 / (2 * s) - p1 ** 2 / (4 * p0 ** 1.5),
        p3 / (2 * s) - 3 * p1 * p2 / (4 * p0 ** 1.5) + 3 * p1 ** 3 / (8 * p0 ** 2.5),
        p4 / (2 * s) - (4 * p1 * p3 + 3 * p2 ** 2) / (4 * p0 ** 1.5)
        + 9 * p1 ** 2 * p2 / (4 * p0 ** 2.5) - 15 * p1 ** 4 / (16 * p0 ** 3.5)])


def tj_derivs(j, eta):
    """T_j = S_j + D_j  (Appendix E); D_j is eta-independent."""
    m_max = (NMAX - j) // 2
    c = CAL._cat_coeff_vec(j, ALPHA, m_max)
    S = np.zeros(ORD + 1)
    for mA in range(m_max + 1):
        for mB in range(m_max + 1):
            nA, nB = 2 * mA + j, 2 * mB + j
            if nA + nB > NMAX:
                continue
            S += c[mA] * c[mB] * _sqrt_derivs(_pzz_derivs(nA, nB, eta))
    D = CAL.residual_j(j, ALPHA, NMAX)
    S[0] += D
    return S


def _leibniz(a, b):
    """derivatives of a product a*b."""
    return np.array([sum(comb(n, i) * a[i] * b[n - i] for i in range(n + 1)) for n in range(ORD + 1)])


def _quot_derivs(num, den):
    """derivatives of num/den by the Appendix-F recursion."""
    q = np.zeros(ORD + 1)
    q[0] = num[0] / den[0]
    for n in range(1, ORD + 1):
        q[n] = (num[n] - sum(comb(n, k) * den[k] * q[n - k] for k in range(1, n + 1))) / den[0]
    return q


def ez_derivs(eta, pxx=None):
    pxx = pxx_derivs(eta) if pxx is None else pxx
    U = np.zeros(ORD + 1)
    for j in (0, 1):
        T = tj_derivs(j, eta)
        U += _leibniz(T, T)
    return _quot_derivs(U, pxx)


# ---------------------------------------------------------------- entropy and R
def _h_derivs(x):
    """h(x) and h^(1..4) (Appendix G)."""
    l2 = math.log(2.0)
    return np.array([
        -x * math.log2(x) - (1 - x) * math.log2(1 - x),
        math.log2((1 - x) / x),
        -1.0 / (x * (1 - x) * l2),
        (1.0 / x ** 2 - 1.0 / (1 - x) ** 2) / l2,
        -2.0 / l2 * (1.0 / x ** 3 + 1.0 / (1 - x) ** 3)])


def _compose(f, g):
    """d^n f(g(eta)), f = derivs of f at g0 (0..4), g = derivs of g (0..4): Faa di Bruno, n<=4."""
    f1, f2, f3, f4 = f[1:]
    g1, g2, g3, g4 = g[1:]
    return np.array([f[0],
                     f1 * g1,
                     f1 * g2 + f2 * g1 ** 2,
                     f1 * g3 + 3 * f2 * g1 * g2 + f3 * g1 ** 3,
                     f1 * g4 + f2 * (4 * g1 * g3 + 3 * g2 ** 2) + 6 * f3 * g1 ** 2 * g2 + f4 * g1 ** 4])


def all_derivs(eta):
    """Dict of derivative arrays (order 0..4, wrt eta) for pXX, eX, eZ, K, R on the smooth branch."""
    px, ex = pxx_derivs(eta), ex_derivs(eta)
    ez = ez_derivs(eta, px)
    hx, hz = _compose(_h_derivs(ex[0]), ex), _compose(_h_derivs(ez[0]), ez)
    K = -hx - hz
    K[0] += 1.0
    R = 2.0 * _leibniz(px, K)
    return {'pXX': px, 'eX': ex, 'eZ': ez, 'K': K, 'R': R}


# ---------------------------------------------------------------- independent derivative (reference)
def chebyshev_derivs(fn, x0, rel_halfwidth=0.25, deg=24, var_log=False):
    """Numerical derivatives 0..4 of fn at x0 from a Chebyshev fit over [x0(1-w), x0(1+w)].
    Used only as an INDEPENDENT check of the analytic derivative formulas."""
    from numpy.polynomial import chebyshev as Cb
    lo, hi = x0 * (1 - rel_halfwidth), x0 * (1 + rel_halfwidth)
    k = np.arange(deg + 1)
    nodes = 0.5 * (hi + lo) + 0.5 * (hi - lo) * np.cos(np.pi * (2 * k + 1) / (2 * (deg + 1)))
    ys = np.array([fn(float(t)) for t in nodes])
    p = Cb.Chebyshev.fit(nodes, ys, deg, domain=[lo, hi])
    return np.array([p.deriv(n)(x0) if n else p(x0) for n in range(ORD + 1)])
