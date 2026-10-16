# -*- coding: utf-8 -*-
"""
The averaging formulas under test and their independent references.

Formulas (manuscript Sec. IV for log-normal, progress report Sec. 3 for Gamma-Gamma)
-------------------------------------------------------------------------------------
  Euler operator      Theta^n Q = sum_k S(n,k) eta^k Q^(k)            (Stirling numbers 2nd kind)
  Log-normal (LN)     E[Q] = Q(eta_med) + s2/2 Theta^2 Q + s4/8 Theta^4 Q      (Prop. 1)
  Gamma-Gamma (GG)    E[Q] = Q(eta_med) + k2/2 Th^2 Q + k3/6 Th^3 Q + (k4+3k2^2)/24 Th^4 Q
  naive delta         E[Q] = Q(m) + 1/2 Q''(m) Var(eta),  m = E[eta]

References (no use of the formulas above)
  * LN : Gauss-Hermite quadrature in X = ln(eta/eta0)         (+ scipy.quad spot check, + Monte Carlo)
  * GG : product Gauss-generalized-Laguerre quadrature over the two Gamma factors,
         Bessel-K density quadrature in ln I (as in the report), and Monte Carlo.
"""
import math
import numpy as np
from scipy import special as sp
from scipy.integrate import simpson
from . import common as C
from .common import ALPHA, PD, THETA, THETA_A, THETA_B, NMAX, PHI
from src import legacy_baseline as CAL

# ------------------------------------------------------------------ vectorised engine references
def Q_vec(name, eta):
    """CAL quantity as a function of eta (=sqrt(eta_A eta_B)), vectorised, from the validated engine."""
    eta = np.asarray(eta, dtype=float)
    if name == 'pXX':
        return CAL.p_xx_total(PHI, THETA, ALPHA ** 2 * eta, PD)
    if name == 'eX':
        return CAL.bit_error_rate(PHI, THETA, ALPHA ** 2 * eta, PD)
    if name == 'eZ':
        return CAL.phase_error_upper_bound(ALPHA, eta ** 2, PD, THETA_A, THETA_B, NMAX, PHI)
    if name == 'R':
        return CAL.key_rate(ALPHA, eta ** 2, PD, THETA_A, THETA_B, NMAX, PHI)
    raise KeyError(name)


# ------------------------------------------------------------------ Euler operator / Stirling
def stirling2(n, k):
    return sp.stirling2(n, k, exact=True)


def theta_ops(d, eta):
    """Theta^n Q for n=1..4 from eta-derivatives d[0..4]  (Theorem 1, Stirling weights)."""
    return np.array([sum(stirling2(n, k) * eta ** k * d[k] for k in range(1, n + 1)) for n in range(1, 5)])


# ------------------------------------------------------------------ log-normal expansion
def ln_expansion(d, eta_med, s2, order):
    """Proposition 1 (order 2: variance term only; order 4: + s2^2/8 term)."""
    th = theta_ops(d, eta_med)
    val = d[0] + 0.5 * s2 * th[1]
    if order >= 4:
        val += (s2 ** 2 / 8.0) * th[3]
    return val


def naive_delta(d, eta_bar, var_eta):
    """Eq.(24)/(57): second-order Taylor about the mean."""
    return d[0] + 0.5 * d[2] * var_eta


def ln_reference(name, eta0, s2, n=200):
    """E[Q(eta0 e^X)], X ~ N(-s2, s2)  (manuscript model eta = sqrt(eta_A eta_B))."""
    t, w = np.polynomial.hermite.hermgauss(n)
    X = -s2 + math.sqrt(2 * s2) * t
    return float(np.sum(w * Q_vec(name, eta0 * np.exp(X))) / math.sqrt(math.pi))


def ln_reference_quad(name, eta0, s2):
    from scipy.integrate import quad
    mu, sd = -s2, math.sqrt(s2)
    f = lambda X: float(Q_vec(name, eta0 * math.exp(X))) * math.exp(-0.5 * ((X - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    v, _ = quad(f, mu - 12 * sd, mu + 12 * sd, epsabs=0, epsrel=1e-11, limit=400)
    return v


def ln_mc(name, eta0, s2, n, seed=0):
    rng = np.random.default_rng(seed)
    X = -s2 + math.sqrt(s2) * rng.standard_normal(n)
    q = Q_vec(name, eta0 * np.exp(X))
    return float(q.mean()), float(q.std() / math.sqrt(n))


# ------------------------------------------------------------------ Gamma-Gamma parameters
def sigma_y2_aperture(s_R2, d2, expo=1.2):
    """Aperture-averaged small-scale log-variance (progress report Sec. 3.3).
    expo = 1.2 : sigma_R^(12/5) with sigma_R = sqrt(Rytov variance)  -> (s_R2)^1.2   (Andrews & Phillips, correct)
    expo = 2.4 : the Rytov VARIANCE raised to 12/5                                   (reproduces the report's figures)"""
    s125 = s_R2 ** expo
    return math.expm1(0.51 * s_R2 * (1 + 0.69 * s125) ** (-5 / 6) / (1 + 0.9 * d2 + 0.62 * d2 * s125) ** (5 / 6))


def gg_params(s_R2, d2=0.0, expo=1.2):
    """alpha (large scale, point-receiver form) and beta = 1/sigma_y^2(D) (aperture averaged). See sigma_y2_aperture for expo."""
    s125 = s_R2 ** expo
    alpha = 1.0 / math.expm1(0.49 * s_R2 / (1 + 1.11 * s125) ** (7 / 6))
    beta = 1.0 / sigma_y2_aperture(s_R2, d2, expo)
    return alpha, beta


def aperture_d2(L_km, D=2 * C.RX_RADIUS):
    return C.K_WAVE * D ** 2 / (4 * L_km * 1e3)


def gg_si(alpha, beta):
    return 1 / alpha + 1 / beta + 1 / (alpha * beta)


def log_gamma_cumulants(alpha, beta):
    """cumulants k1..k4 of Y = ln Xa + ln Xb, Xa~Gamma(alpha,rate alpha), Xb~Gamma(beta,rate beta)."""
    def one(k):
        return np.array([sp.digamma(k) - math.log(k), sp.polygamma(1, k), sp.polygamma(2, k), sp.polygamma(3, k)])
    return one(alpha) + one(beta)


# ------------------------------------------------------------------ Gamma-Gamma expansion (eta_tot = eta0_tot * I, eta_m = sqrt(eta_tot))
def gg_expansion(d, eta_pt_m, kap, y_pt_offset=0.0, order=4):
    """E[Q] about the point eta_pt (in the manuscript variable eta_m), Y = ln(eta_tot/eta0_tot).
    f^(n)(Y) = (1/2)^n Theta_m^n Q (since eta_m = sqrt(eta_tot)).  y_pt_offset = y0 - E[Y] (0 for the log-mean point).
    order = 2, 3, 4 (cumulant orders kept)."""
    th = theta_ops(d, eta_pt_m) * 0.5 ** np.arange(1, 5)           # f', f'', f''', f''''
    k1, k2, k3, k4 = kap
    dl = -y_pt_offset                                              # E[Y] - y0
    M = [dl, k2 + dl ** 2, k3 + 3 * k2 * dl + dl ** 3, k4 + 4 * k3 * dl + 3 * k2 ** 2 + 6 * k2 * dl ** 2 + dl ** 4]
    val = d[0] + th[0] * M[0] + th[1] * M[1] / 2.0
    if order >= 3:
        val += th[2] * M[2] / 6.0
    if order >= 4:
        val += th[3] * M[3] / 24.0
    return val


def ln_on_gg(d, eta_tot0_m, s_R2, order):
    """The (wrong) log-normal formula applied to the same Rytov variance: Y ~ N(-s/2, s), eta_med = eta0 e^{-s/4} (eta_m units)."""
    eta_med_m = eta_tot0_m * math.exp(-s_R2 / 4.0)
    th = theta_ops(d, eta_med_m) * 0.5 ** np.arange(1, 5)
    val = d[0] + 0.5 * s_R2 * th[1]
    if order >= 4:
        val += (s_R2 ** 2 / 8.0) * th[3]
    return val


def gg_reference_laguerre(name, eta0_m, alpha, beta, n=140):
    """E[Q(eta0_m sqrt(Xa Xb))] by tensor Gauss-generalized-Laguerre rules (robust for any alpha, beta)."""
    xa, wa = sp.roots_genlaguerre(n, alpha - 1.0); wa = wa / sp.gamma(alpha) if alpha < 150 else wa * np.exp(-sp.gammaln(alpha))
    xb, wb = sp.roots_genlaguerre(n, beta - 1.0); wb = wb / sp.gamma(beta) if beta < 150 else wb * np.exp(-sp.gammaln(beta))
    xa, xb = xa / alpha, xb / beta
    I = np.outer(xa, xb)
    q = Q_vec(name, eta0_m * np.sqrt(I.ravel())).reshape(I.shape)
    return float(wa @ q @ wb)


def gg_pdf_ln_i(y, alpha, beta):
    """log of the Gamma-Gamma density of I at ln I = y, times I (i.e. the density of Y), via log K_v."""
    I = np.exp(y)
    v, x = abs(alpha - beta), 2 * np.sqrt(alpha * beta * I)
    with np.errstate(divide='ignore'):
        logk = np.log(sp.kve(v, x)) - x
    logp = (math.log(2) + 0.5 * (alpha + beta) * math.log(alpha * beta) - sp.gammaln(alpha) - sp.gammaln(beta)
            + (0.5 * (alpha + beta) - 1) * y + logk)
    return np.exp(logp + y)


def gg_reference_bessel(name, eta0_m, alpha, beta, npts=6001):
    """Direct Bessel-K quadrature in y = ln I (the 'direct Bessel function quadrature' of the report)."""
    kap = log_gamma_cumulants(alpha, beta)
    mu, sd = kap[0], math.sqrt(kap[1])
    y = np.linspace(mu - 14 * sd, mu + 10 * sd, npts)
    p = gg_pdf_ln_i(y, alpha, beta)
    norm = simpson(p, x=y)
    val = simpson(p * Q_vec(name, eta0_m * np.exp(0.5 * y)), x=y)
    return float(val), float(norm)


def gg_mc(name, eta0_m, alpha, beta, n, seed=0):
    rng = np.random.default_rng(seed)
    I = rng.gamma(alpha, 1 / alpha, n) * rng.gamma(beta, 1 / beta, n)
    q = Q_vec(name, eta0_m * np.sqrt(I))
    return float(q.mean()), float(q.std() / math.sqrt(n))


def gg_mode_I(alpha, beta):
    from scipy.optimize import minimize_scalar
    f = lambda y: -float(gg_pdf_ln_i(np.array([y]), alpha, beta)[0] * math.exp(-y))     # density of I
    kap = log_gamma_cumulants(alpha, beta)
    r = minimize_scalar(f, bounds=(kap[0] - 4 * math.sqrt(kap[1]), kap[0] + 3 * math.sqrt(kap[1])), method='bounded',
                        options={'xatol': 1e-12})
    return math.exp(r.x)
