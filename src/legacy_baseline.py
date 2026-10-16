# -*- coding: utf-8 -*-
"""
TF-QKD over FSO channels with log-normal turbulence — optimized engine.

Security proof: Curty, Azuma & Lo, "Simple security proof of twin-field
type QKD protocol", arXiv:1807.07667v2, Protocol 3, Eqs. (12)-(35).

Implements: channel model (X/Z basis gains, Eqs. 27-33), photon-number
yields (Eq. 34-35), cat-state coefficients (Eqs. 12-13), phase-error
upper bound (Eqs. 20-22), asymptotic key rate (Eqs. 17-19) with alpha
optimization, PLOB bound, deterministic FSO channel (atmospheric +
diffraction + geometric loss), and ergodic SKR under log-normal
turbulence (Monte Carlo).

This is a performance rewrite of the original notebook. All physics,
equations and numerical outputs are unchanged (validated to float64
machine precision, see VALIDATION.md / inline asserts). Educational
narrative, per-cell prints and diagnostic plots have been removed;
see the optimization report for what changed and why.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.special import i0 as bessel_i0, gammaln
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

# =============================================================================
# 1. Global parameters
# =============================================================================

PARAMS = dict(
    pd=1e-7,
    alpha=0.2,
    thetaA=np.arcsin(np.sqrt(0.02)),
    thetaB=-np.arcsin(np.sqrt(0.02)),
    delta=0.0,
)
PARAMS['theta'] = PARAMS['thetaA'] - PARAMS['thetaB']
PARAMS['phi'] = PARAMS['delta'] * np.pi

NMAX_DEFAULT = 4          # Nmax=4 is near-optimal per paper Fig. 5
OPT_MAXITER = 100
ALPHA_BOUNDS = (1e-3, 0.3)


def dB_to_eta(loss_dB):
    """eta = 10^(-loss_dB/10); each half-link has transmittance sqrt(eta)."""
    return 10.0 ** (-np.asarray(loss_dB, dtype=np.float64) / 10.0)


def h(x):
    """Binary entropy, vectorized, safe at x=0,1."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    t1 = np.where(x > 0, -x * np.log2(np.where(x > 0, x, 1.0)), 0.0)
    t2 = np.where(1 - x > 0, -(1 - x) * np.log2(np.where(1 - x > 0, 1 - x, 1.0)), 0.0)
    return t1 + t2


# =============================================================================
# 2. X-basis channel model — Eqs. (27)-(31)
# =============================================================================

def omega(phi, theta):
    return np.cos(phi) * np.cos(theta)


def f_plus(phi, theta, gamma):
    Om = omega(phi, theta)
    gamma = np.asarray(gamma, dtype=np.float64)
    return np.exp(-gamma * (1.0 + Om)) - np.exp(-2.0 * gamma)


def f_minus(phi, theta, gamma):
    Om = omega(phi, theta)
    gamma = np.asarray(gamma, dtype=np.float64)
    return np.exp(-gamma * (1.0 - Om)) - np.exp(-2.0 * gamma)


def q_xx(kc, bA, bB, phi, theta, gamma):
    """Eq. (28). k_d = 1 - k_c implicitly."""
    xor_val = int(kc) ^ int(bA) ^ int(bB)
    return f_minus(phi, theta, gamma) if xor_val else f_plus(phi, theta, gamma)


def p_xx_conditional(kc, bA, bB, phi, theta, gamma, pd):
    """Eq. (27)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    qxx = q_xx(kc, bA, bB, phi, theta, gamma)
    return (1.0 - pd) * (pd * np.exp(-2.0 * gamma) + qxx)


def p_xx_total(phi, theta, gamma, pd):
    """Eq. (30) — marginal X-basis gain, equal for (1,0) and (0,1)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    Om = omega(phi, theta)
    term1 = 0.5 * (1.0 - pd) * (np.exp(-gamma * Om) + np.exp(gamma * Om)) * np.exp(-gamma)
    term2 = (1.0 - pd) ** 2 * np.exp(-2.0 * gamma)
    return term1 - term2


def bit_error_rate(phi, theta, gamma, pd):
    """Eq. (31)."""
    gamma = np.asarray(gamma, dtype=np.float64)
    Om = omega(phi, theta)
    num = np.exp(-gamma * Om) - (1.0 - pd) * np.exp(-gamma)
    den = np.exp(-gamma * Om) + np.exp(gamma * Om) - 2.0 * (1.0 - pd) * np.exp(-gamma)
    return np.where(np.abs(den) > 1e-300, num / den, 0.5)


# =============================================================================
# 3. Z-basis channel model — Eqs. (32)-(33)
# =============================================================================

def q_zz(betaA, betaB, theta, eta):
    """Eq. (33)."""
    betaA = np.asarray(betaA, dtype=np.float64)
    betaB = np.asarray(betaB, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    sqrt_eta = np.sqrt(eta)
    sum_b2 = betaA ** 2 + betaB ** 2
    arg = betaA * betaB * sqrt_eta * np.cos(theta)
    return np.exp(-0.5 * sum_b2 * sqrt_eta) * bessel_i0(arg) - np.exp(-sum_b2 * sqrt_eta)


def p_zz_beta(betaA, betaB, theta, eta, pd):
    """Eq. (32)."""
    eta = np.asarray(eta, dtype=np.float64)
    sum_b2 = betaA ** 2 + betaB ** 2
    qzz = q_zz(betaA, betaB, theta, eta)
    dark = pd * np.exp(-sum_b2 * np.sqrt(eta))
    return (1.0 - pd) * (dark + qzz)


# =============================================================================
# 4. Photon-number yields — Eq. (34)-(35)
#
#    KEY OPTIMIZATION: Eq.(35)'s nested k,l,m,p,q combinatorial sum is a
#    polynomial in sqrt(eta) whose coefficients C[k,l] do NOT depend on
#    eta at all -- only on (nA, nB, thetaA, thetaB). The original code
#    recomputed this nested loop (with exact-arithmetic comb()/factorial()
#    calls) on every single call, for every alpha trial, every loss point,
#    and every Monte-Carlo sample. Here the coefficient table is built
#    once per unique (nA,nB,thetaA,thetaB) and memoized; evaluation for
#    any (scalar or array) eta is then a few vectorized NumPy power ops.
# =============================================================================

@lru_cache(maxsize=None)
def _qzz_coeff_table(nA, nB, thetaA, thetaB):
    """C[k,l] = C(nA,k)C(nB,l) * BS(k,l) * inner_sum(k,l) -- eta-independent."""
    cA, sA = math.cos(thetaA), math.sin(thetaA)
    cB, sB = math.cos(thetaB), math.sin(thetaB)
    C = np.zeros((nA + 1, nB + 1), dtype=np.float64)
    for k in range(nA + 1):
        for l in range(nB + 1):
            BS = 1.0 / (2.0 ** (k + l) * math.factorial(k) * math.factorial(l))
            inner = 0.0
            for m in range(k + 1):
                for p in range(l + 1):
                    qmin, qmax = max(0, m + p - l), min(k, m + p)
                    for q in range(qmin, qmax + 1):
                        mp_q = m + p - q
                        if mp_q < 0 or mp_q > l:
                            continue
                        Ccomb = (math.comb(k, m) * math.comb(l, p) *
                                 math.comb(k, q) * math.comb(l, mp_q))
                        if Ccomb == 0:
                            continue
                        pA_cos, pB_cos = m + q, m + 2 * p - q
                        pA_sin, pB_sin = 2 * k - m - q, 2 * l - m - 2 * p + q
                        if min(pA_cos, pB_cos, pA_sin, pB_sin) < 0:
                            continue
                        Pfac = (cA ** pA_cos) * (cB ** pB_cos) * (sA ** pA_sin) * (sB ** pB_sin)
                        if Pfac == 0.0:
                            continue
                        upper, lower = m + p, k + l - m - p
                        if lower < 0:
                            continue
                        inner += Ccomb * Pfac * math.factorial(upper) * math.factorial(lower)
            C[k, l] = BS * inner
    binA = np.array([math.comb(nA, k) for k in range(nA + 1)], dtype=np.float64)
    binB = np.array([math.comb(nB, l) for l in range(nB + 1)], dtype=np.float64)
    return C * binA[:, None] * binB[None, :]


def qzz_photon_number(nA, nB, thetaA, thetaB, eta):
    """q_ZZ(k_c,k_d | n_A,n_B) -- Eq. (35), vectorized over eta (any shape)."""
    eta_arr = np.asarray(eta, dtype=np.float64)
    shape = eta_arr.shape
    flat = eta_arr.ravel()
    sqrt_eta = np.sqrt(flat)
    one_minus = 1.0 - sqrt_eta
    coeff = _qzz_coeff_table(int(nA), int(nB), float(thetaA), float(thetaB))

    k_idx, l_idx = np.arange(nA + 1), np.arange(nB + 1)
    kl_sum = (k_idx[:, None] + l_idx[None, :]).ravel()
    exps = (nA + nB - kl_sum)
    coeff_flat = coeff.ravel()
    nz = coeff_flat != 0.0
    vacuum = one_minus ** (nA + nB)

    if not np.any(nz):
        result = -vacuum
    else:
        se_pow = sqrt_eta[:, None] ** kl_sum[nz][None, :]
        om_pow = one_minus[:, None] ** exps[nz][None, :]
        result = (coeff_flat[nz][None, :] * se_pow * om_pow).sum(axis=1) - vacuum
    return result.reshape(shape)


def p_zz_photon(nA, nB, thetaA, thetaB, eta, pd):
    """Eq. (34)."""
    eta = np.asarray(eta, dtype=np.float64)
    sqrt_eta = np.sqrt(eta)
    dark = pd * (1.0 - sqrt_eta) ** (nA + nB)
    qzz_pn = qzz_photon_number(nA, nB, thetaA, thetaB, eta)
    return (1.0 - pd) * (dark + qzz_pn)


# =============================================================================
# 5. Cat-state coefficients — Eqs. (12)-(13)
#    Vectorized over the Fock sub-index (no Python-level per-n function calls).
# =============================================================================

def _cat_coeff_vec(j, alpha, m_max):
    if alpha <= 0.0 or m_max < 0:
        return np.zeros(max(m_max + 1, 0))
    m = np.arange(m_max + 1)
    n = 2 * m + j
    log_c = -0.5 * alpha ** 2 + n * np.log(alpha) - 0.5 * gammaln(n + 1)
    return np.exp(log_c)


def c_even(n, alpha):
    return float(_cat_coeff_vec(0, alpha, n)[n]) if n >= 0 and alpha > 0 else 0.0


def c_odd(n, alpha):
    return float(_cat_coeff_vec(1, alpha, n)[n]) if n >= 0 and alpha > 0 else 0.0


def marginal_coeff_sum(j, alpha, Nterms=80):
    """Sum_{n=0}^{Nterms-1} c^(j)_{2n+j}."""
    return float(np.sum(_cat_coeff_vec(j, alpha, Nterms - 1)))


def truncated_coeff_sum(j, alpha, Nmax):
    m_max = (Nmax - j) // 2
    return 0.0 if m_max < 0 else float(np.sum(_cat_coeff_vec(j, alpha, m_max)))


def residual_j(j, alpha, Nmax, Nterms=80):
    """Eq. (22)."""
    S_full = marginal_coeff_sum(j, alpha, Nterms)
    S_trunc = truncated_coeff_sum(j, alpha, Nmax)
    return max(S_full ** 2 - S_trunc ** 2, 0.0)


# =============================================================================
# 6. Phase-error upper bound — Eqs. (20)-(22)
#
#    inner_sum_j is now vectorized over eta: a single call evaluates the
#    bound for a scalar loss point OR for an entire array of (e.g.) MC
#    turbulence samples, reusing the cached Eq.(35) coefficient tables.
# =============================================================================

def inner_sum_j(j, alpha, eta, pd, thetaA, thetaB, Nmax):
    eta_arr = np.asarray(eta, dtype=np.float64)
    scalar_in = (eta_arr.ndim == 0)
    eta_flat = np.atleast_1d(eta_arr)
    m_max = (Nmax - j) // 2
    total = np.zeros_like(eta_flat)
    if m_max >= 0:
        c = _cat_coeff_vec(j, alpha, m_max)
        for mA in range(m_max + 1):
            cA = c[mA]
            if cA == 0.0:
                continue
            nA = 2 * mA + j
            for mB in range(m_max + 1):
                nB = 2 * mB + j
                if nA + nB > Nmax:
                    continue
                cB = c[mB]
                if cB == 0.0:
                    continue
                pzz = np.maximum(p_zz_photon(nA, nB, thetaA, thetaB, eta_flat, pd), 0.0)
                total += cA * cB * np.sqrt(pzz)
    return float(total[0]) if scalar_in else total.reshape(eta_arr.shape)


def phase_error_upper_bound(alpha, eta, pd, thetaA, thetaB, Nmax, phi=0.0):
    """Eqs. (20)-(21).

    NOTE (correctness/perf finding): in the original notebook this function
    accepted (kc, kd) parameters but never used them in the computation --
    the formula is identical for (kc,kd)=(1,0) and (0,1). Calling it twice
    therefore always returned numerically identical values; the duplicate
    evaluation has been removed (see key_rate() below), which is an exact
    2x reduction in cost with zero change to any numerical output.
    """
    theta = thetaA - thetaB
    eta_arr = np.asarray(eta, dtype=np.float64)
    scalar_in = (eta_arr.ndim == 0)
    eta_flat = np.atleast_1d(eta_arr)
    gamma = np.sqrt(eta_flat) * alpha ** 2
    pXX = p_xx_total(phi, theta, gamma, pd)

    eZ_num = np.zeros_like(eta_flat)
    for j in (0, 1):
        S_j = inner_sum_j(j, alpha, eta_flat, pd, thetaA, thetaB, Nmax)
        D_j = residual_j(j, alpha, Nmax)
        eZ_num += (S_j + D_j) ** 2

    eZ = np.where(pXX > 0.0, eZ_num / np.where(pXX > 0.0, pXX, 1.0), 0.5)
    eZ = np.clip(eZ, 0.0, 0.5)
    return float(eZ[0]) if scalar_in else eZ.reshape(eta_arr.shape)


# =============================================================================
# 7. Asymptotic key rate and alpha optimization — Eqs. (17)-(19)
# =============================================================================

def secret_key_rate_component(pxx, ex, ez):
    eX_c = np.clip(ex, 0.0, 0.5)
    eZ_c = np.clip(np.minimum(ez, 0.5), 0.0, 0.5)
    pxx = np.maximum(pxx, 0.0)
    return np.maximum(pxx * (1.0 - h(eX_c) - h(eZ_c)), 0.0)


def key_rate(alpha, eta, pd, thetaA, thetaB, Nmax, phi=0.0):
    """R_low_X = R10 + R01, exploiting R10 == R01 (see note above)."""
    theta = thetaA - thetaB
    eta_arr = np.asarray(eta, dtype=np.float64)
    scalar_in = (eta_arr.ndim == 0)
    eta_flat = np.atleast_1d(eta_arr)
    gamma = np.sqrt(eta_flat) * alpha ** 2

    pXX = p_xx_total(phi, theta, gamma, pd)
    eX = bit_error_rate(phi, theta, gamma, pd)
    eZ = phase_error_upper_bound(alpha, eta_flat, pd, thetaA, thetaB, Nmax, phi)

    R_single = secret_key_rate_component(pXX, eX, eZ)
    R_total = 2.0 * R_single
    return float(R_total[0]) if scalar_in else R_total.reshape(eta_arr.shape)


def total_secret_key_rate(alpha, eta, pd, thetaA, thetaB, delta, Nmax):
    """Eqs. (17)-(19). Returns (R_total, eX, eZ, pXX) -- phi passed explicitly,
    no global PARAMS mutation (thread-safe, avoids a dict read/write per call)."""
    phi = delta * np.pi
    theta = thetaA - thetaB
    gamma = np.sqrt(eta) * alpha ** 2

    pXX = max(float(p_xx_total(phi, theta, gamma, pd)), 0.0)
    eX = float(np.clip(bit_error_rate(phi, theta, gamma, pd), 0.0, 0.5))
    eZ = phase_error_upper_bound(alpha, eta, pd, thetaA, thetaB, Nmax, phi)

    R_total = 2.0 * float(secret_key_rate_component(pXX, eX, eZ))
    return R_total, eX, eZ, pXX


def optimize_alpha(eta, pd, thetaA, thetaB, delta=0.0, Nmax=NMAX_DEFAULT,
                    alpha_min=ALPHA_BOUNDS[0], alpha_max=ALPHA_BOUNDS[1]):
    """Maximise key_rate over alpha (Brent, bounded). Turbulence/eta-independent
    quantities (residual_j, Eq.35 coefficient tables) are cached across all
    trial alphas and across the whole outer loss/distance sweep."""
    phi = delta * np.pi

    def neg_rate(a):
        return -key_rate(a, eta, pd, thetaA, thetaB, Nmax, phi) if a > 0 else 0.0

    probes = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 10)
    probe_rates = np.array([key_rate(a, eta, pd, thetaA, thetaB, Nmax, phi) for a in probes])
    if np.all(probe_rates <= 0.0):
        return alpha_min, alpha_min**2, 0.0
    result = minimize_scalar(neg_rate, bounds=(alpha_min, alpha_max),
                              method='bounded',
                              options={'maxiter': OPT_MAXITER, 'xatol': 1e-4})
    alpha_opt = float(np.clip(result.x, alpha_min, alpha_max))
    return alpha_opt, alpha_opt ** 2, max(-float(result.fun), 0.0)


# =============================================================================
# 8. PLOB bound
# =============================================================================

def plob_bound(eta):
    eta = np.clip(np.asarray(eta, dtype=np.float64), 1e-300, 1.0 - 1e-15)
    return -np.log2(1.0 - eta)


# =============================================================================
# 9. Deterministic FSO channel
# =============================================================================

def atmospheric_transmittance(alpha_atm, distance_km):
    """Beer-Lambert: eta_atm = exp(-alpha_atm * L)."""
    distance_km = np.clip(np.asarray(distance_km, dtype=np.float64), 0.0, None)
    return np.clip(np.exp(-alpha_atm * distance_km), 0.0, 1.0)


def beam_radius(wavelength, distance_m, w0):
    """w(L) = w0 * sqrt(1 + (lambda L / (pi w0^2))^2)."""
    distance_m = np.clip(np.asarray(distance_m, dtype=np.float64), 0.0, None)
    z_R = np.pi * w0 ** 2 / wavelength
    return w0 * np.sqrt(1.0 + (distance_m / z_R) ** 2)


def geometric_efficiency(receiver_radius, w_at_L):
    """eta_geo = 1 - exp(-2 r_rx^2 / w^2)."""
    w_safe = np.where(np.asarray(w_at_L) > 0.0, w_at_L, 1e-30)
    return np.clip(1.0 - np.exp(-2.0 * receiver_radius ** 2 / w_safe ** 2), 0.0, 1.0)


def deterministic_fso_channel(alpha_atm, distance_km, wavelength, w0, receiver_radius):
    """eta_fso(L) = eta_atm(L) * eta_geo(L/2)^2 (both half-links)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    L_half_m = (distance_km / 2.0) * 1e3

    eta_atm = atmospheric_transmittance(alpha_atm, distance_km)
    w_half_m = beam_radius(wavelength, L_half_m, w0)
    eta_geo_half = geometric_efficiency(receiver_radius, w_half_m)
    eta_geo = eta_geo_half ** 2
    eta_fso = np.clip(eta_atm * eta_geo, 0.0, 1.0)

    eta_safe = np.where(eta_fso > 0.0, eta_fso, 1e-300)
    loss_dB = -10.0 * np.log10(eta_safe)
    return {'eta_atm': eta_atm, 'w_half_m': w_half_m, 'eta_geo_half': eta_geo_half,
            'eta_geo': eta_geo, 'eta_fso': eta_fso, 'loss_dB': loss_dB}


def tfqkd_skr_fso(distance_km, pd, thetaA, thetaB, fso_params, Nmax=4,
                   optimise=True, alpha_fixed=0.20):
    """SKR vs distance under deterministic FSO loss. optimise=False uses
    key_rate's array path directly (no Python per-point loop)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_fso_arr = fso['eta_fso']
    valid = eta_fso_arr >= 1e-50

    if not optimise:
        R_arr = np.zeros_like(eta_fso_arr)
        R_arr[valid] = key_rate(alpha_fixed, eta_fso_arr[valid], pd, thetaA, thetaB, Nmax)
        return R_arr, eta_fso_arr, np.full_like(eta_fso_arr, alpha_fixed)

    # alpha optimization is inherently per-point (1-D search); each call is
    # now ~20x cheaper due to the Eq.(35) coefficient cache (see report).
    R_arr = np.zeros_like(eta_fso_arr)
    alpha_arr = np.full_like(eta_fso_arr, alpha_fixed)
    for i in np.flatnonzero(valid):
        a_opt, _, R_max = optimize_alpha(float(eta_fso_arr[i]), pd, thetaA, thetaB,
                                          delta=0.0, Nmax=Nmax)
        R_arr[i], alpha_arr[i] = max(R_max, 0.0), a_opt
    return R_arr, eta_fso_arr, alpha_arr


# =============================================================================
# 10. Log-normal turbulence (Rytov / weak-fluctuation regime)
# =============================================================================

Cn2_classes = {'weak': 1e-17, 'moderate': 1e-15, 'strong': 1e-13}
N_MC = 50_000
MC_SEED = 42
turbulence_enabled = True


def rytov_variance(Cn2, wavelength, distance_m):
    """sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6) (plane-wave, weak regime)."""
    distance_m = np.clip(np.asarray(distance_m, dtype=np.float64), 0.0, None)
    k = 2.0 * np.pi / wavelength
    return np.clip(1.23 * Cn2 * (k ** (7.0 / 6.0)) * (distance_m ** (11.0 / 6.0)), 0.0, None)


def lognormal_params(sigma_R2):
    """(mu_X, sigma_X^2), energy-neutral (E[eta_LN] = 1), overflow-safe."""
    sigma_R2 = np.asarray(sigma_R2, dtype=np.float64)
    OVERFLOW_THRESH = 500.0
    safe_sR2 = np.minimum(sigma_R2, OVERFLOW_THRESH)
    sigma_X2 = np.where(sigma_R2 > OVERFLOW_THRESH, sigma_R2,
                         np.log1p(np.exp(safe_sR2) - 1.0))
    return -0.5 * sigma_X2, sigma_X2


def sample_lognormal_eta(eta_nominal, Cn2, wavelength, distance_m,
                          N_samples=50_000, rng=None):
    """eta_turb = eta_nominal * exp(X), X ~ N(mu_X, sigma_X^2), clipped to [0,1]."""
    if rng is None:
        rng = np.random.default_rng(MC_SEED)
    if not turbulence_enabled:
        return np.full(N_samples, float(eta_nominal), dtype=np.float64)

    sigma_R2 = float(rytov_variance(Cn2, wavelength, distance_m))
    mu_X, sigma_X2 = lognormal_params(sigma_R2)
    X = rng.normal(loc=float(mu_X), scale=float(np.sqrt(sigma_X2)), size=N_samples)
    eta_turb = float(eta_nominal) * np.exp(X)
    return np.clip(eta_turb, 0.0, 1.0)


def turbulence_statistics(eta_nominal, Cn2, wavelength, distance_m,
                           N_samples=50_000, rng=None):
    """Analytic + MC moments of eta_turb, for cross-validation."""
    if rng is None:
        rng = np.random.default_rng(MC_SEED)
    sigma_R2 = float(rytov_variance(Cn2, wavelength, distance_m))
    mu_X, sigma_X2 = (float(v) for v in lognormal_params(sigma_R2))

    mean_analytic = float(eta_nominal) * np.exp(mu_X + 0.5 * sigma_X2)
    var_analytic = (float(eta_nominal) ** 2) * (np.exp(sigma_X2) - 1.0)
    SI_analytic = np.exp(sigma_X2) - 1.0

    samples = sample_lognormal_eta(eta_nominal, Cn2, wavelength, distance_m, N_samples, rng)
    return {'sigma_R2': sigma_R2, 'sigma_X2': sigma_X2, 'mu_X': mu_X,
            'mean_analytic': mean_analytic, 'var_analytic': var_analytic,
            'SI_analytic': SI_analytic, 'mean_mc': float(np.mean(samples)),
            'var_mc': float(np.var(samples)), 'std_mc': float(np.std(samples)),
            'samples': samples}


def tfqkd_skr_turbulence(distance_km, pd, thetaA, thetaB, Cn2, fso_params,
                          N_samples=None, Nmax=4, alpha_fixed=0.20, rng=None):
    """Ergodic SKR under log-normal fading: R_turb(L) = E_eta[R_det(eta_turb)].

    KEY OPTIMIZATION: the original code called key_rate() once per individual
    MC sample inside a Python list comprehension (N_dist * N_samples scalar
    calls, each re-deriving the Eq.35 combinatorial sum from scratch). Because
    key_rate() now accepts a full eta array and reuses the cached Eq.(35)
    coefficient tables, every distance point's N_samples-strong ensemble is
    evaluated in a single vectorized call.
    """
    if N_samples is None:
        N_samples = N_MC
    if rng is None:
        rng = np.random.default_rng(MC_SEED)

    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    R_arr = np.zeros(len(distance_km))

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        eta_samples = sample_lognormal_eta(
            eta_nominal=float(eta_nom), Cn2=Cn2, wavelength=fso_params['wavelength'],
            distance_m=L_km * 1e3, N_samples=N_samples, rng=rng)
        R_samples = key_rate(alpha_fixed, eta_samples, pd, thetaA, thetaB, Nmax)
        R_arr[i] = float(np.mean(R_samples))

    return R_arr, eta_nominal_arr


# =============================================================================
# 10b. Analytical (quadrature-based) ensemble averaging under log-normal
#      fading -- preserves all Monte Carlo functions above unchanged.
#
#      Derivation summary (full derivation in accompanying writeup):
#
#      eta_turb = eta_nom * exp(X),  X ~ N(mu_X, sigma_X^2)
#
#      For the *unclipped* log-normal (the clip to [0,1] applied in
#      sample_lognormal_eta affects a negligible probability mass in the
#      sigma_R^2 < 1 validity regime; see _lognormal_clip_mass_above_one()
#      below for a diagnostic of this approximation), the standard
#      change-of-variables identity for a log-normal random variable gives,
#      for any observable g(eta):
#
#          <g>(eta_nom) = Integral_{-inf}^{inf} g(eta_nom * exp(X))
#                              * phi(X; mu_X, sigma_X^2) dX
#
#      where phi is the Gaussian PDF. This substitution is EXACT (not an
#      approximation) and removes the 1/eta singularity that the eta-space
#      log-normal density has at eta -> 0, which is why we integrate in
#      X-space rather than eta-space.
#
#      None of g = p_XX, g = e_X, g = R admit closed-form Gaussian integrals
#      (each involves exp(+/- gamma) with gamma ~ sqrt(eta) = sqrt(eta_nom)
#      exp(X/2), i.e. an exponential-of-exponential of the integration
#      variable -- and R additionally passes through the binary entropy
#      function and the Eq.(20-22) phase-error sum, which is transcendental
#      in eta). Per requirement 3, we do NOT attempt invalid symbolic
#      simplification; we evaluate the exact integral by quadrature
#      (scipy.integrate.quad) instead.
#
#      QBER requires special care: e_X(eta) = N(eta)/D(eta) is a RATIO, so
#      <e_X> != <N>/<D> in general (Jensen's inequality forbids exchanging
#      ratio and average). The physically meaningful average -- consistent
#      with "total expected wrong bits / total expected sifted bits" -- is
#      the gain-weighted average:
#
#          <e_X> = <p_XX * e_X> / <p_XX>
#
#      which is what average_qber_lognormal() computes.
# =============================================================================

QUAD_N_SIGMA = 8.0        # Gaussian truncation half-width (tail < 1e-15)
QUAD_EPSABS = 1e-12
QUAD_EPSREL = 1e-10
QUAD_LIMIT = 200


def _gaussian_pdf(X, mu_X, sigma_X):
    """phi(X; mu_X, sigma_X^2)."""
    return np.exp(-0.5 * ((X - mu_X) / sigma_X) ** 2) / (sigma_X * np.sqrt(2.0 * np.pi))


def _quad_bounds(mu_X, sigma_X, n_sigma=QUAD_N_SIGMA):
    """Truncated integration domain: [mu_X - n_sigma*sigma_X, mu_X + n_sigma*sigma_X].
    Tail mass beyond 8 sigma is ~1e-15, below double-precision noise floor,
    so this truncation introduces no numerically meaningful error while
    making quad's job well-posed (avoids relying on its +/-inf transform
    when sigma_X is tiny and the integrand is a sharp spike)."""
    return mu_X - n_sigma * sigma_X, mu_X + n_sigma * sigma_X


def _lognormal_clip_mass_above_one(eta_nominal, sigma_R2):
    """Diagnostic: P(eta_turb > 1) under the UNCLIPPED log-normal, i.e. the
    probability mass that sample_lognormal_eta() clips to exactly 1. This
    quantifies the approximation made by integrating the unclipped log-normal
    in average_*_lognormal() instead of the clipped one used by the MC
    sampler. Returns a probability in [0,1]; expected to be ~0 in the
    sigma_R^2 < 1 validity regime unless eta_nominal is itself close to 1."""
    from scipy.stats import norm
    mu_X, sigma_X2 = lognormal_params(sigma_R2)
    sigma_X = np.sqrt(sigma_X2)
    # P(eta_nom * exp(X) > 1) = P(X > -ln(eta_nom)) = P(X > ln(1/eta_nom))
    z = (np.log(1.0 / np.clip(eta_nominal, 1e-300, None)) - mu_X) / sigma_X
    return float(norm.sf(z))


def average_pxx_lognormal(eta_nominal, sigma_R2, phi, theta, alpha, pd,
                           n_sigma=QUAD_N_SIGMA):
    """<p_XX>(eta_nom) = Integral p_XX(eta_nom*exp(X)) * phi(X) dX  (Eq. 30
    pushed through the log-normal fading channel). Scalar eta_nominal/sigma_R2
    in, scalar out; see average_pxx_lognormal_array for the vectorized wrapper.
    """
    mu_X, sigma_X2 = (float(v) for v in lognormal_params(sigma_R2))
    sigma_X = math.sqrt(sigma_X2) if sigma_X2 > 0 else 1e-12
    lo, hi = _quad_bounds(mu_X, sigma_X, n_sigma)

    def integrand(X):
        eta = eta_nominal * math.exp(X)
        gamma = math.sqrt(eta) * alpha ** 2
        pxx = float(p_xx_total(phi, theta, gamma, pd))
        return pxx * _gaussian_pdf(X, mu_X, sigma_X)

    val, err = quad(integrand, lo, hi, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL,
                     limit=QUAD_LIMIT)
    return max(val, 0.0), err


# Public name requested by the analysis spec (Requirement 2). This is a thin
# alias, not a re-derivation: <p_XX> (the X-basis ensemble-averaged gain) is
# exactly what average_pxx_lognormal already computes via Eq. (30) pushed
# through the log-normal fading channel. Kept as a separate def (rather than
# `average_gain_lognormal = average_pxx_lognormal`) so its docstring reads
# naturally under the requested name.
def average_gain_lognormal(eta_nominal, sigma_R2, phi, theta, alpha, pd,
                            n_sigma=QUAD_N_SIGMA):
    """<p_XX>(eta_nom): ensemble-averaged X-basis gain under log-normal
    fading. Identical computation to average_pxx_lognormal(); see that
    function's docstring and the module-level derivation note for details.
    Returns (mean_pXX, quad_abserr)."""
    return average_pxx_lognormal(eta_nominal, sigma_R2, phi, theta, alpha, pd,
                                  n_sigma=n_sigma)


def average_qber_lognormal(eta_nominal, sigma_R2, phi, theta, alpha, pd,
                            n_sigma=QUAD_N_SIGMA):
    """<e_X>(eta_nom) = <p_XX * e_X> / <p_XX>  (gain-weighted average QBER;
    see derivation note above on why the naive unweighted average is wrong).
    Returns (mean_eX, (num_err, den_err))."""
    mu_X, sigma_X2 = (float(v) for v in lognormal_params(sigma_R2))
    sigma_X = math.sqrt(sigma_X2) if sigma_X2 > 0 else 1e-12
    lo, hi = _quad_bounds(mu_X, sigma_X, n_sigma)

    def integrand_num(X):
        eta = eta_nominal * math.exp(X)
        gamma = math.sqrt(eta) * alpha ** 2
        pxx = float(p_xx_total(phi, theta, gamma, pd))
        ex = float(np.clip(bit_error_rate(phi, theta, gamma, pd), 0.0, 0.5))
        return pxx * ex * _gaussian_pdf(X, mu_X, sigma_X)

    def integrand_den(X):
        eta = eta_nominal * math.exp(X)
        gamma = math.sqrt(eta) * alpha ** 2
        pxx = float(p_xx_total(phi, theta, gamma, pd))
        return pxx * _gaussian_pdf(X, mu_X, sigma_X)

    num, num_err = quad(integrand_num, lo, hi, epsabs=QUAD_EPSABS,
                         epsrel=QUAD_EPSREL, limit=QUAD_LIMIT)
    den, den_err = quad(integrand_den, lo, hi, epsabs=QUAD_EPSABS,
                         epsrel=QUAD_EPSREL, limit=QUAD_LIMIT)
    mean_eX = num / den if den > 1e-300 else 0.5
    return float(np.clip(mean_eX, 0.0, 0.5)), (num_err, den_err)


def average_phase_error_lognormal(eta_nominal, sigma_R2, alpha, pd, thetaA, thetaB,
                                   Nmax=NMAX_DEFAULT, phi=0.0, n_sigma=QUAD_N_SIGMA):
    """<e_Z>(eta_nom): ensemble-averaged phase-error upper bound under
    log-normal fading.

    Derivation (see module docstring for the general averaging framework):
    the deterministic bound (Eqs. 20-22) is the ratio

        e_Z(eta) = [ sum_{j=0,1} (S_j(eta) + D_j)^2 ] / p_XX(eta)

    where D_j (Eq. 22 truncation residual) is eta-INDEPENDENT (function of
    alpha, Nmax only) and S_j(eta) = sum_{mA,mB} c_mA^(j) c_mB^(j)
    sqrt(p_ZZ(nA,nB;eta)) (Eq. 20-21). As with e_X, e_Z is a ratio, so
    Jensen's inequality forbids <e_Z> = <numerator>/<denominator> taken
    naively -- but it does NOT forbid the gain-weighted construction

        <e_Z> := <p_XX * e_Z> / <p_XX>

    which is the physically meaningful average (expected phase errors per
    pulse, divided by expected X-clicks per pulse) and is exactly the
    quantity implicitly used whenever <R> = <p_XX*(1-h(e_X)-h(e_Z))> is
    averaged as a whole, consistent with average_qber_lognormal()'s
    treatment of e_X. Critically, p_XX * e_Z = sum_j (S_j+D_j)^2 exactly --
    the p_XX in e_Z's own denominator cancels -- so the numerator integrand
    is simply the un-normalized Eq. (20) phase-error numerator, with no
    division inside the integrand at all. This makes the integrand smoother
    and avoids amplifying quadrature error through a near-zero denominator
    at high loss, exactly mirroring how average_qber_lognormal avoids
    averaging the raw ratio e_X(eta).

    Returns (mean_eZ, (num_err, den_err))."""
    theta = thetaA - thetaB
    mu_X, sigma_X2 = (float(v) for v in lognormal_params(sigma_R2))
    sigma_X = math.sqrt(sigma_X2) if sigma_X2 > 0 else 1e-12
    lo, hi = _quad_bounds(mu_X, sigma_X, n_sigma)

    # D_j (Eq. 22) is eta-independent -- computed once, outside the integrand,
    # reusing the same cached residual_j() used by the deterministic pipeline.
    D = {j: residual_j(j, alpha, Nmax) for j in (0, 1)}

    def integrand_num(X):
        eta = eta_nominal * math.exp(X)
        eta = min(max(eta, 0.0), 1.0)
        total = 0.0
        for j in (0, 1):
            S_j = inner_sum_j(j, alpha, eta, pd, thetaA, thetaB, Nmax)
            total += (S_j + D[j]) ** 2
        return total * _gaussian_pdf(X, mu_X, sigma_X)

    def integrand_den(X):
        eta = eta_nominal * math.exp(X)
        eta = min(max(eta, 0.0), 1.0)
        gamma = math.sqrt(eta) * alpha ** 2
        pxx = float(p_xx_total(phi, theta, gamma, pd))
        return pxx * _gaussian_pdf(X, mu_X, sigma_X)

    num, num_err = quad(integrand_num, lo, hi, epsabs=QUAD_EPSABS,
                         epsrel=QUAD_EPSREL, limit=QUAD_LIMIT)
    den, den_err = quad(integrand_den, lo, hi, epsabs=QUAD_EPSABS,
                         epsrel=QUAD_EPSREL, limit=QUAD_LIMIT)
    mean_eZ = num / den if den > 1e-300 else 0.5
    return float(np.clip(mean_eZ, 0.0, 0.5)), (num_err, den_err)


def average_skr_lognormal(eta_nominal, sigma_R2, alpha, pd, thetaA, thetaB,
                           Nmax=NMAX_DEFAULT, phi=0.0, n_sigma=QUAD_N_SIGMA):
    """<R>(eta_nom) = Integral R(eta_nom*exp(X), alpha) * phi(X) dX.

    This is the direct analytical analogue of tfqkd_skr_turbulence's Monte
    Carlo mean(R(eta_samples)) -- same full key_rate() pipeline (Eqs. 17-22,
    27-35), but the expectation over fading is taken by 1-D Gauss-Kronrod
    quadrature (scipy.integrate.quad) instead of sample averaging. No
    closed form exists because R involves h(.) and the Eq.(20-22) phase
    error sum, both transcendental in eta -- see module-level derivation note.
    """
    theta = thetaA - thetaB
    mu_X, sigma_X2 = (float(v) for v in lognormal_params(sigma_R2))
    sigma_X = math.sqrt(sigma_X2) if sigma_X2 > 0 else 1e-12
    lo, hi = _quad_bounds(mu_X, sigma_X, n_sigma)

    def integrand(X):
        eta = eta_nominal * math.exp(X)
        eta = min(max(eta, 0.0), 1.0)  # match MC sampler's clip for consistency
        R = key_rate(alpha, eta, pd, thetaA, thetaB, Nmax, phi)
        return R * _gaussian_pdf(X, mu_X, sigma_X)

    val, err = quad(integrand, lo, hi, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL,
                     limit=QUAD_LIMIT)
    return max(val, 0.0), err


def average_skr_lognormal_array(distance_km, pd, thetaA, thetaB, Cn2,
                                 fso_params, Nmax=NMAX_DEFAULT,
                                 alpha_fixed=0.20, n_sigma=QUAD_N_SIGMA):
    """Vectorized-over-distance wrapper around average_skr_lognormal(), mirroring
    tfqkd_skr_turbulence()'s signature/return so the two can be swapped for
    direct comparison. quad() itself is inherently scalar per evaluation
    point, so this loops over distance (same structural shape as the MC
    function's loop over distance, but with O(1) quadrature evaluations per
    point instead of O(N_samples) sample draws)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    R_arr = np.zeros(len(distance_km))
    err_arr = np.zeros(len(distance_km))

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        sigma_R2 = float(rytov_variance(Cn2, fso_params['wavelength'], L_km * 1e3))
        R_arr[i], err_arr[i] = average_skr_lognormal(
            float(eta_nom), sigma_R2, alpha_fixed, pd, thetaA, thetaB, Nmax,
            n_sigma=n_sigma)
    return R_arr, eta_nominal_arr, err_arr


def average_gain_lognormal_array(distance_km, pd, thetaA, thetaB, Cn2,
                                  fso_params, alpha_fixed=0.20, delta=0.0,
                                  n_sigma=QUAD_N_SIGMA):
    """Vectorized-over-distance wrapper around average_gain_lognormal()
    (<p_XX> vs L), mirroring average_skr_lognormal_array()'s structure."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    theta = thetaA - thetaB
    phi = delta * np.pi
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    pXX_arr = np.zeros(len(distance_km))
    err_arr = np.zeros(len(distance_km))

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        sigma_R2 = float(rytov_variance(Cn2, fso_params['wavelength'], L_km * 1e3))
        pXX_arr[i], err_arr[i] = average_gain_lognormal(
            float(eta_nom), sigma_R2, phi, theta, alpha_fixed, pd, n_sigma=n_sigma)
    return pXX_arr, eta_nominal_arr, err_arr


def average_qber_lognormal_array(distance_km, pd, thetaA, thetaB, Cn2,
                                  fso_params, alpha_fixed=0.20, delta=0.0,
                                  n_sigma=QUAD_N_SIGMA):
    """Vectorized-over-distance wrapper around average_qber_lognormal()
    (<e_X> vs L), mirroring average_skr_lognormal_array()'s structure."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    theta = thetaA - thetaB
    phi = delta * np.pi
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    eX_arr = np.zeros(len(distance_km))
    err_arr = np.zeros((len(distance_km), 2))

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        sigma_R2 = float(rytov_variance(Cn2, fso_params['wavelength'], L_km * 1e3))
        eX_arr[i], (err_arr[i, 0], err_arr[i, 1]) = average_qber_lognormal(
            float(eta_nom), sigma_R2, phi, theta, alpha_fixed, pd, n_sigma=n_sigma)
    return eX_arr, eta_nominal_arr, err_arr


def average_phase_error_lognormal_array(distance_km, pd, thetaA, thetaB, Cn2,
                                         fso_params, Nmax=NMAX_DEFAULT,
                                         alpha_fixed=0.20, delta=0.0,
                                         n_sigma=QUAD_N_SIGMA):
    """Vectorized-over-distance wrapper around average_phase_error_lognormal()
    (<e_Z> vs L), mirroring average_skr_lognormal_array()'s structure."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    phi = delta * np.pi
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    eZ_arr = np.zeros(len(distance_km))
    err_arr = np.zeros((len(distance_km), 2))

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        sigma_R2 = float(rytov_variance(Cn2, fso_params['wavelength'], L_km * 1e3))
        eZ_arr[i], (err_arr[i, 0], err_arr[i, 1]) = average_phase_error_lognormal(
            float(eta_nom), sigma_R2, alpha_fixed, pd, thetaA, thetaB, Nmax,
            phi=phi, n_sigma=n_sigma)
    return eZ_arr, eta_nominal_arr, err_arr


def ensemble_statistics_vs_distance(distance_km, pd, thetaA, thetaB, Cn2,
                                     fso_params, Nmax=NMAX_DEFAULT,
                                     alpha_fixed=0.20, delta=0.0,
                                     n_sigma=QUAD_N_SIGMA):
    """Single-pass convenience wrapper: computes <p_XX>, <e_X>, <e_Z>, <R>,
    eta_nominal and sigma_R^2 vs distance_km in one call. Used by the
    diagnostic table (Requirement 7) and Figures D/E/F so each quantity is
    evaluated under IDENTICAL conditions (same eta_nominal array, same
    sigma_R2 array) rather than being recomputed separately with potential
    drift between calls. Does not replace the individual *_array functions
    above (those remain independently usable/testable); this just composes
    them and adds sigma_R^2(L) for convenience."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    pXX, eta_nom, _ = average_gain_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params,
        alpha_fixed=alpha_fixed, delta=delta, n_sigma=n_sigma)
    eX, _, _ = average_qber_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params,
        alpha_fixed=alpha_fixed, delta=delta, n_sigma=n_sigma)
    eZ, _, _ = average_phase_error_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params, Nmax=Nmax,
        alpha_fixed=alpha_fixed, delta=delta, n_sigma=n_sigma)
    R, _, _ = average_skr_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params, Nmax=Nmax,
        alpha_fixed=alpha_fixed, n_sigma=n_sigma)
    sigma_R2 = rytov_variance(Cn2, fso_params['wavelength'], distance_km * 1e3)
    return {'distance_km': distance_km, 'eta_nominal': eta_nom, 'sigma_R2': sigma_R2,
            'pXX': pXX, 'eX': eX, 'eZ': eZ, 'R': R}


def optimize_alpha_lognormal(eta_nominal, sigma_R2, pd, thetaA, thetaB,
                              delta=0.0, Nmax=NMAX_DEFAULT,
                              alpha_min=ALPHA_BOUNDS[0], alpha_max=ALPHA_BOUNDS[1],
                              n_sigma=QUAD_N_SIGMA):
    """alpha* = argmax_alpha <R(alpha)>, where <R> is the QUADRATURE-averaged
    SKR under log-normal fading (average_skr_lognormal), at a single distance
    point (fixed eta_nominal, sigma_R2). Direct analogue of optimize_alpha()
    (Eqs. 17-19) but optimizing the ensemble-averaged rate rather than the
    deterministic rate -- this is requirement 10."""
    phi = delta * np.pi

    def neg_avg_rate(a):
        if a <= 0:
            return 0.0
        val, _ = average_skr_lognormal(eta_nominal, sigma_R2, a, pd, thetaA,
                                        thetaB, Nmax, phi, n_sigma)
        return -val

    probes = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 5)
    probe_rates = np.array([-neg_avg_rate(a) for a in probes])
    if np.all(probe_rates <= 0.0):
        return alpha_min, 0.0

    result = minimize_scalar(neg_avg_rate, bounds=(alpha_min, alpha_max),
                              method='bounded',
                              options={'maxiter': OPT_MAXITER, 'xatol': 1e-4})
    alpha_opt = float(np.clip(result.x, alpha_min, alpha_max))
    return alpha_opt, max(-float(result.fun), 0.0)


def optimize_alpha_lognormal_vs_distance(distance_km, pd, thetaA, thetaB, Cn2,
                                          fso_params, Nmax=NMAX_DEFAULT,
                                          alpha_fixed=0.20, n_sigma=QUAD_N_SIGMA):
    """Sweep optimize_alpha_lognormal() over distance: returns alpha*(L),
    <R(alpha*)>(L), and <R(alpha_fixed)>(L) for the 'optimized vs fixed'
    comparison plot (requirement 11)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    n = len(distance_km)
    alpha_star = np.full(n, alpha_fixed)
    R_star = np.zeros(n)
    R_fixed = np.zeros(n)

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        sigma_R2 = float(rytov_variance(Cn2, fso_params['wavelength'], L_km * 1e3))
        a_opt, R_opt = optimize_alpha_lognormal(
            float(eta_nom), sigma_R2, pd, thetaA, thetaB, Nmax=Nmax,
            n_sigma=n_sigma)
        R_fix, _ = average_skr_lognormal(
            float(eta_nom), sigma_R2, alpha_fixed, pd, thetaA, thetaB, Nmax,
            n_sigma=n_sigma)
        alpha_star[i], R_star[i] = a_opt, max(R_opt, 0.0)
        R_fixed[i] = max(R_fix, 0.0)

    return alpha_star, R_star, R_fixed, eta_nominal_arr


def validate_mc_vs_quadrature(distance_km, pd, thetaA, thetaB, Cn2, fso_params,
                               N_samples=None, Nmax=NMAX_DEFAULT,
                               alpha_fixed=0.20, rng=None, n_sigma=QUAD_N_SIGMA):
    """Requirement 7-8: compare Monte Carlo (tfqkd_skr_turbulence, UNCHANGED)
    against quadrature (average_skr_lognormal_array, new) over distance, and
    compute the relative error abs(MC - Quad)/Quad point-by-point. Returns a
    dict with both SKR curves, sigma_R^2(L), and the relative-error array.

    This routine calls -- but never modifies -- the existing MC function,
    satisfying requirement 13 (MC remains available as the validation
    benchmark, not replaced)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)

    R_mc, eta_nom_mc = tfqkd_skr_turbulence(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params,
        N_samples=N_samples, Nmax=Nmax, alpha_fixed=alpha_fixed, rng=rng)

    R_quad, eta_nom_quad, quad_err = average_skr_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params, Nmax=Nmax,
        alpha_fixed=alpha_fixed, n_sigma=n_sigma)

    sigma_R2_arr = rytov_variance(Cn2, fso_params['wavelength'], distance_km * 1e3)

    quad_safe = np.where(R_quad > 1e-300, R_quad, np.nan)
    rel_error = np.abs(R_mc - R_quad) / quad_safe

    return {'distance_km': distance_km, 'R_mc': R_mc, 'R_quad': R_quad,
            'quad_err': quad_err, 'eta_nominal': eta_nom_mc,
            'sigma_R2': sigma_R2_arr, 'rel_error': rel_error,
            'sigma_R2_lt_1': sigma_R2_arr < 1.0}


def _mc_ensemble_observables(eta_samples, alpha, pd, thetaA, thetaB, theta, Nmax):
    """Helper: gain-weighted MC estimators for <p_XX>, <e_X>, <e_Z> from a
    fixed batch of eta samples. Gain-weighting matches the quadrature
    definitions exactly (<e_X> = <p_XX*e_X>/<p_XX>, <e_Z> = <p_XX*e_Z>/<p_XX>)
    so MC and quadrature are compared on the SAME estimator, not an
    unweighted-vs-weighted mismatch."""
    gamma_s = np.sqrt(eta_samples) * alpha ** 2
    pXX_s = p_xx_total(0.0, theta, gamma_s, pd)
    eX_s = np.clip(bit_error_rate(0.0, theta, gamma_s, pd), 0.0, 0.5)
    eZ_s = phase_error_upper_bound(alpha, eta_samples, pd, thetaA, thetaB, Nmax)

    pXX_mc = float(np.mean(pXX_s))
    denom = np.sum(pXX_s)
    eX_mc = float(np.sum(pXX_s * eX_s) / denom) if denom > 0 else 0.5
    eZ_mc = float(np.sum(pXX_s * eZ_s) / denom) if denom > 0 else 0.5
    return pXX_mc, eX_mc, eZ_mc


def validate_mc_vs_quadrature_observables(distance_km, pd, thetaA, thetaB, Cn2,
                                           fso_params, N_samples=None,
                                           Nmax=NMAX_DEFAULT, alpha_fixed=0.20,
                                           rng=None, n_sigma=QUAD_N_SIGMA):
    """Requirement 4: MC-vs-quadrature validation for the THREE ensemble
    observables <p_XX>, <e_X>, <e_Z> individually (validate_mc_vs_quadrature
    above only covers <R>). Returns relative-error arrays
    abs(MC - Quad)/Quad for gain, QBER, and phase error vs distance, plus
    sigma_R^2(L) for the validity mask. Uses the same eta_samples batch per
    distance point for all three observables (paired comparison, lower
    variance than redrawing samples per quantity)."""
    if N_samples is None:
        N_samples = N_MC
    if rng is None:
        rng = np.random.default_rng(MC_SEED)

    distance_km = np.asarray(distance_km, dtype=np.float64)
    theta = thetaA - thetaB
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    n = len(distance_km)

    pXX_mc = np.full(n, np.nan)
    eX_mc = np.full(n, np.nan)
    eZ_mc = np.full(n, np.nan)

    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        eta_samples = sample_lognormal_eta(
            eta_nominal=float(eta_nom), Cn2=Cn2, wavelength=fso_params['wavelength'],
            distance_m=L_km * 1e3, N_samples=N_samples, rng=rng)
        pXX_mc[i], eX_mc[i], eZ_mc[i] = _mc_ensemble_observables(
            eta_samples, alpha_fixed, pd, thetaA, thetaB, theta, Nmax)

    pXX_quad, _, _ = average_gain_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params, alpha_fixed=alpha_fixed,
        n_sigma=n_sigma)
    eX_quad, _, _ = average_qber_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params, alpha_fixed=alpha_fixed,
        n_sigma=n_sigma)
    eZ_quad, _, _ = average_phase_error_lognormal_array(
        distance_km, pd, thetaA, thetaB, Cn2, fso_params, Nmax=Nmax,
        alpha_fixed=alpha_fixed, n_sigma=n_sigma)

    sigma_R2_arr = rytov_variance(Cn2, fso_params['wavelength'], distance_km * 1e3)

    def rel_err(mc, quad):
        safe = np.where(quad > 1e-300, quad, np.nan)
        return np.abs(mc - quad) / safe

    return {'distance_km': distance_km, 'eta_nominal': eta_nominal_arr,
            'sigma_R2': sigma_R2_arr, 'sigma_R2_lt_1': sigma_R2_arr < 1.0,
            'pXX_mc': pXX_mc, 'pXX_quad': pXX_quad, 'pXX_rel_err': rel_err(pXX_mc, pXX_quad),
            'eX_mc': eX_mc, 'eX_quad': eX_quad, 'eX_rel_err': rel_err(eX_mc, eX_quad),
            'eZ_mc': eZ_mc, 'eZ_quad': eZ_quad, 'eZ_rel_err': rel_err(eZ_mc, eZ_quad)}


# =============================================================================
# 12. Physical interpretation: attributing SKR degradation to gain / QBER /
#     phase-error channels (Requirements 6, 8)
#
#     R = p_XX * [1 - h(e_X) - h(e_Z)] * 2   (Eqs. 17-19)
#
#     ln(R_turb/R_det) = ln(p_XX_turb/p_XX_det)
#                        + ln([1-h(eX_turb)-h(eZ_turb)] / [1-h(eX_det)-h(eZ_det)])
#
#     The bracketed factor does not separate additively between e_X and e_Z
#     in closed form (h is nonlinear), so the clean log-additive split is
#     only exact between "gain" and "the information-reconciliation factor
#     as a whole." To attribute degradation specifically between e_X and e_Z
#     we use a controlled counterfactual substitution: holding p_XX and one
#     of {e_X, e_Z} fixed at its turbulence-averaged value while restoring
#     the OTHER to its deterministic value, and measuring how much of the
#     total R degradation is recovered. This is the standard "one-factor-at-
#     a-time" sensitivity decomposition and is exact at the level of R
#     itself (no linearization), even though it is not a unique additive
#     decomposition (cross-terms are attributed by the ordering chosen;
#     we report both orderings so the reader can see the cross-term size).
# =============================================================================

def degradation_breakdown(eta_nominal, sigma_R2, alpha, pd, thetaA, thetaB,
                           Nmax=NMAX_DEFAULT, phi=0.0, n_sigma=QUAD_N_SIGMA):
    """Requirement 6/8: decompose the ensemble-averaged SKR degradation at a
    single (eta_nominal, sigma_R2) point into contributions from gain
    reduction, QBER increase, and phase-error increase.

    Returns a dict with:
      - deterministic and turbulence-averaged pXX, eX, eZ, R
      - pct_change_{pXX,eX,eZ,R}: percentage change (turbulence vs det.)
      - R_counterfactual_*: R evaluated with ONE quantity swapped to its
        turbulence-averaged value and the other two held at deterministic
        values (isolates each channel's individual effect on R)
      - attribution_*: fraction of total ln(R_turb/R_det) explained by
        swapping gain only / eX only / eZ only (one-at-a-time, both single-
        factor counterfactuals -- not required to sum to 1 because of
        nonlinear cross-terms; the residual cross-term is reported too)
    """
    theta = thetaA - thetaB
    gamma_det = math.sqrt(eta_nominal) * alpha ** 2

    pXX_det = max(float(p_xx_total(phi, theta, gamma_det, pd)), 0.0)
    eX_det = float(np.clip(bit_error_rate(phi, theta, gamma_det, pd), 0.0, 0.5))
    eZ_det = float(phase_error_upper_bound(alpha, eta_nominal, pd, thetaA, thetaB, Nmax, phi))
    R_det = 2.0 * float(secret_key_rate_component(pXX_det, eX_det, eZ_det))

    pXX_t, _ = average_gain_lognormal(eta_nominal, sigma_R2, phi, theta, alpha, pd, n_sigma)
    eX_t, _ = average_qber_lognormal(eta_nominal, sigma_R2, phi, theta, alpha, pd, n_sigma)
    eZ_t, _ = average_phase_error_lognormal(eta_nominal, sigma_R2, alpha, pd, thetaA,
                                             thetaB, Nmax, phi, n_sigma)
    R_t, _ = average_skr_lognormal(eta_nominal, sigma_R2, alpha, pd, thetaA, thetaB,
                                    Nmax, phi, n_sigma)

    def pct(turb, det):
        return 100.0 * (turb - det) / det if abs(det) > 1e-300 else np.nan

    # One-at-a-time counterfactuals: swap exactly one ensemble-averaged
    # quantity into the otherwise-deterministic R formula.
    R_gain_only = 2.0 * secret_key_rate_component(pXX_t, eX_det, eZ_det)
    R_eX_only = 2.0 * secret_key_rate_component(pXX_det, eX_t, eZ_det)
    R_eZ_only = 2.0 * secret_key_rate_component(pXX_det, eX_det, eZ_t)

    # log-ratio attribution (additive in log-space for the gain factor,
    # which truly factorizes; eX/eZ split via one-at-a-time as noted above)
    logR_det = math.log(max(R_det, 1e-300))
    logR_t = math.log(max(R_t, 1e-300))
    total_logdrop = logR_det - logR_t  # >=0 when turbulence degrades R

    def logdrop_from(R_cf):
        """log(R_det/R_counterfactual): how much of the total log-drop is
        reproduced by changing ONLY this one quantity to its turbulence
        value."""
        return logR_det - math.log(max(R_cf, 1e-300))

    drop_gain = logdrop_from(R_gain_only)
    drop_eX = logdrop_from(R_eX_only)
    drop_eZ = logdrop_from(R_eZ_only)
    drop_sum = drop_gain + drop_eX + drop_eZ

    frac_gain = drop_gain / total_logdrop if total_logdrop > 1e-300 else np.nan
    frac_eX = drop_eX / total_logdrop if total_logdrop > 1e-300 else np.nan
    frac_eZ = drop_eZ / total_logdrop if total_logdrop > 1e-300 else np.nan
    cross_term_frac = 1.0 - (frac_gain + frac_eX + frac_eZ) if total_logdrop > 1e-300 else np.nan

    return {
        'pXX_det': pXX_det, 'eX_det': eX_det, 'eZ_det': eZ_det, 'R_det': R_det,
        'pXX_turb': pXX_t, 'eX_turb': eX_t, 'eZ_turb': eZ_t, 'R_turb': max(R_t, 0.0),
        'pct_change_pXX': pct(pXX_t, pXX_det), 'pct_change_eX': pct(eX_t, eX_det),
        'pct_change_eZ': pct(eZ_t, eZ_det), 'pct_change_R': pct(R_t, R_det),
        'R_counterfactual_gain_only': max(R_gain_only, 0.0),
        'R_counterfactual_eX_only': max(R_eX_only, 0.0),
        'R_counterfactual_eZ_only': max(R_eZ_only, 0.0),
        'total_log_drop': total_logdrop,
        'attribution_gain': frac_gain, 'attribution_eX': frac_eX,
        'attribution_eZ': frac_eZ, 'attribution_cross_term': cross_term_frac,
    }


def degradation_breakdown_vs_distance(distance_km, pd, thetaA, thetaB, Cn2,
                                       fso_params, Nmax=NMAX_DEFAULT,
                                       alpha_fixed=0.20, n_sigma=QUAD_N_SIGMA):
    """Sweep degradation_breakdown() over distance. Returns a dict of arrays
    (same keys as degradation_breakdown, vectorized over distance_km) plus
    eta_nominal and sigma_R2, for the diagnostic table (Requirement 7) and
    the dominant-mechanism conclusion (Requirement 8)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    sigma_R2_arr = rytov_variance(Cn2, fso_params['wavelength'], distance_km * 1e3)

    keys = ['pXX_det', 'eX_det', 'eZ_det', 'R_det', 'pXX_turb', 'eX_turb', 'eZ_turb',
            'R_turb', 'pct_change_pXX', 'pct_change_eX', 'pct_change_eZ', 'pct_change_R',
            'R_counterfactual_gain_only', 'R_counterfactual_eX_only',
            'R_counterfactual_eZ_only', 'total_log_drop', 'attribution_gain',
            'attribution_eX', 'attribution_eZ', 'attribution_cross_term']
    out = {k: np.full(len(distance_km), np.nan) for k in keys}

    for i, (L_km, eta_nom, sR2) in enumerate(zip(distance_km, eta_nominal_arr, sigma_R2_arr)):
        if eta_nom < 1e-50:
            continue
        res = degradation_breakdown(float(eta_nom), float(sR2), alpha_fixed, pd,
                                     thetaA, thetaB, Nmax, n_sigma=n_sigma)
        for k in keys:
            out[k][i] = res[k]

    out['distance_km'] = distance_km
    out['eta_nominal'] = eta_nominal_arr
    out['sigma_R2'] = sigma_R2_arr
    out['sigma_R2_lt_1'] = sigma_R2_arr < 1.0
    return out


def diagnostic_table(distances_km, pd, thetaA, thetaB, Cn2, fso_params,
                      Nmax=NMAX_DEFAULT, alpha_fixed=0.20, n_sigma=QUAD_N_SIGMA):
    """Requirement 7: build the representative-distance diagnostic table
    (default 20/50/80/100 km) reporting <p_XX>, <e_X>, <e_Z>, <R> for
    deterministic vs weak-turbulence cases plus percentage changes, restricted
    to (and flagged against) the sigma_R^2 < 1 validity regime (Requirement 9).
    Returns a list of per-distance dicts; see print_diagnostic_table() for a
    formatted rendering."""
    distances_km = np.asarray(distances_km, dtype=np.float64)
    rows = []
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distances_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nom_arr = fso['eta_fso']
    sigma_R2_arr = rytov_variance(Cn2, fso_params['wavelength'], distances_km * 1e3)

    for L_km, eta_nom, sR2 in zip(distances_km, eta_nom_arr, sigma_R2_arr):
        row = {'distance_km': float(L_km), 'eta_nominal': float(eta_nom),
               'sigma_R2': float(sR2), 'valid_weak_regime': bool(sR2 < 1.0)}
        if eta_nom < 1e-50:
            rows.append(row)
            continue
        res = degradation_breakdown(float(eta_nom), float(sR2), alpha_fixed, pd,
                                     thetaA, thetaB, Nmax, n_sigma=n_sigma)
        row.update(res)
        rows.append(row)
    return rows


def print_diagnostic_table(rows):
    """Pretty-print the diagnostic_table() output (Requirement 7)."""
    hdr = (f"{'L[km]':>7} {'sigma_R2':>9} {'valid':>6} | "
           f"{'pXX_det':>11} {'pXX_turb':>11} {'d%':>7} | "
           f"{'eX_det':>9} {'eX_turb':>9} {'d%':>7} | "
           f"{'eZ_det':>9} {'eZ_turb':>9} {'d%':>7} | "
           f"{'R_det':>11} {'R_turb':>11} {'d%':>7}")
    print(hdr)
    print('-' * len(hdr))
    for row in rows:
        if 'pXX_det' not in row:
            print(f"{row['distance_km']:7.1f} {row['sigma_R2']:9.4f} "
                  f"{'Y' if row['valid_weak_regime'] else 'N':>6} | (eta_nominal underflow -- no positive-rate channel)")
            continue
        v = 'Y' if row['valid_weak_regime'] else 'N'
        print(f"{row['distance_km']:7.1f} {row['sigma_R2']:9.4f} {v:>6} | "
              f"{row['pXX_det']:11.4e} {row['pXX_turb']:11.4e} {row['pct_change_pXX']:7.2f} | "
              f"{row['eX_det']:9.4f} {row['eX_turb']:9.4f} {row['pct_change_eX']:7.2f} | "
              f"{row['eZ_det']:9.4f} {row['eZ_turb']:9.4f} {row['pct_change_eZ']:7.2f} | "
              f"{row['R_det']:11.4e} {row['R_turb']:11.4e} {row['pct_change_R']:7.2f}")


def dominant_degradation_mechanism(degradation_rows, sigma_R2_max=1.0):
    """Requirement 8: aggregate degradation_breakdown results restricted to
    the sigma_R^2 < sigma_R2_max validity regime (Requirement 9) and answer
    "is SKR degradation primarily driven by (a) gain, (b) QBER, or
    (c) phase error?" using the mean one-at-a-time log-attribution fractions.
    Returns a dict with the per-mechanism mean attribution and the winning
    mechanism label."""
    valid_rows = [r for r in degradation_rows
                  if r.get('valid_weak_regime', False) and 'attribution_gain' in r
                  and np.isfinite(r.get('attribution_gain', np.nan))]
    if not valid_rows:
        return {'n_valid_points': 0, 'mechanism': None}

    mean_gain = float(np.mean([r['attribution_gain'] for r in valid_rows]))
    mean_eX = float(np.mean([r['attribution_eX'] for r in valid_rows]))
    mean_eZ = float(np.mean([r['attribution_eZ'] for r in valid_rows]))
    mean_cross = float(np.mean([r['attribution_cross_term'] for r in valid_rows]))

    mechanism_map = {'gain': 'reduced gain (a)', 'eX': 'increased QBER (b)',
                      'eZ': 'increased phase error (c)'}
    scores = {'gain': mean_gain, 'eX': mean_eX, 'eZ': mean_eZ}
    winner = max(scores, key=scores.get)

    return {'n_valid_points': len(valid_rows), 'mean_attribution_gain': mean_gain,
            'mean_attribution_eX': mean_eX, 'mean_attribution_eZ': mean_eZ,
            'mean_attribution_cross_term': mean_cross,
            'mechanism': mechanism_map[winner], 'mechanism_key': winner}



# =============================================================================
# 10e. TWO INDEPENDENT FSO LINKS (Alice->Charlie and Bob->Charlie)
#
#      WHY THIS SECTION EXISTS
#      -----------------------
#      TF-QKD has TWO optical paths: Alice->Charlie (A-C) and Bob->Charlie
#      (B-C). They traverse different air columns, so their turbulence-induced
#      fading is statistically INDEPENDENT. Sections 10/10b above model the
#      whole A-B link as ONE log-normal factor evaluated at the full distance
#      L, i.e. eta = eta_nom * exp(X) with sigma_R^2(L). That is not the right
#      model for a twin-field geometry. The correct one is
#
#          eta_A = eta_A_nom * exp(X_A),   X_A ~ N(-s2_A/2, s2_A),  s2_A from sigma_R^2(L_A)
#          eta_B = eta_B_nom * exp(X_B),   X_B ~ N(-s2_B/2, s2_B),  s2_B from sigma_R^2(L_B)
#          X_A, X_B independent,           eta = eta_A * eta_B       (end-to-end)
#
#      and every ergodic quantity is a 2-D average over (X_A, X_B):
#
#          <g> = Integral Integral g(eta_A * eta_B) phi(X_A) phi(X_B) dX_A dX_B
#
#      Defaults match deterministic_fso_channel(): Charlie sits midway, so
#      L_A = L_B = L/2 and eta_A_nom = eta_B_nom = sqrt(eta_nom) (because
#      eta_nom = eta_atm(L) * eta_geo(L/2)^2 = [eta_atm(L/2) eta_geo(L/2)]^2).
#
#      Consistency facts used by the tests (see tests/test_two_link.py):
#        * E[eta_A eta_B] = eta_A_nom eta_B_nom  (mean-one fading on each link
#          => product of independent mean-one factors is mean-one).
#        * Without the [0,1] clip, X_A + X_B ~ N(-(s2_A+s2_B)/2, s2_A+s2_B),
#          so the 2-D average of any g(eta) must equal the 1-D average with
#          s2_tot = s2_A + s2_B. This is an exact cross-check of the 2-D rule.
#        * For sigma_R^2 ~ L^(11/6): sigma_R^2(L/2) = 2^(-11/6) sigma_R^2(L), so
#          s2_A + s2_B = 2^(-5/6) sigma_R^2(L) ~ 0.561 sigma_R^2(L). The single
#          factor model at full L overstates the log-variance by ~1.78x.
#
#      Modelling assumption (flag for the write-up): key_rate() is the
#      Curty-Azuma-Lo formula for a symmetric channel with total transmittance
#      eta. Feeding it eta = eta_A*eta_B assumes the two amplitudes arriving at
#      Charlie are balanced (eta_A = eta_B) or compensated, i.e. it uses the
#      effective (geometric-mean) transmittance. A fading-induced amplitude
#      IMBALANCE between the two arms would additionally reduce interference
#      visibility; that penalty is NOT modelled here.
# =============================================================================

TWO_LINK_GH_NODES = 128   # 64 suffices for eta_link << 1; 128 resolves the clip kink at eta_link ~ 0.5


def two_link_geometry(distance_m, distance_A_m=None, distance_B_m=None):
    """(L_A, L_B) in metres. Default: Charlie midway, L_A = L_B = L/2."""
    if distance_A_m is None and distance_B_m is None:
        return 0.5 * float(distance_m), 0.5 * float(distance_m)
    if distance_A_m is None:
        return float(distance_m) - float(distance_B_m), float(distance_B_m)
    if distance_B_m is None:
        return float(distance_A_m), float(distance_m) - float(distance_A_m)
    return float(distance_A_m), float(distance_B_m)


def two_link_sigma_R2(Cn2, wavelength, distance_m):
    """Per-ARM Rytov variance, sigma_R^2(L/2) (Charlie midway). This -- not
    sigma_R^2(L) -- is the quantity that decides each arm's validity regime."""
    return rytov_variance(Cn2, wavelength, 0.5 * np.asarray(distance_m, dtype=np.float64))


def two_link_nominal(eta_nominal, eta_A_nominal=None, eta_B_nominal=None):
    """(eta_A_nom, eta_B_nom) with eta_A_nom * eta_B_nom = eta_nominal.
    Default symmetric split sqrt(eta_nominal) each."""
    eta_nominal = float(eta_nominal)
    if eta_A_nominal is None and eta_B_nominal is None:
        r = math.sqrt(eta_nominal)
        return r, r
    if eta_A_nominal is None:
        return eta_nominal / float(eta_B_nominal), float(eta_B_nominal)
    if eta_B_nominal is None:
        return float(eta_A_nominal), eta_nominal / float(eta_A_nominal)
    return float(eta_A_nominal), float(eta_B_nominal)


def two_link_lognormal_params(Cn2, wavelength, distance_A_m, distance_B_m):
    """Per-link Rytov variances and log-normal parameters (each link uses the
    SAME lognormal_params() mapping as the single-link model, evaluated at its
    OWN length), plus the parameters of the sum X_A + X_B (unclipped)."""
    sA = float(rytov_variance(Cn2, wavelength, distance_A_m))
    sB = float(rytov_variance(Cn2, wavelength, distance_B_m))
    muA, s2A = (float(v) for v in lognormal_params(sA))
    muB, s2B = (float(v) for v in lognormal_params(sB))
    return {'sigma_R2_A': sA, 'sigma_R2_B': sB,
            'mu_A': muA, 's2_A': s2A, 'mu_B': muB, 's2_B': s2B,
            'mu_total': muA + muB, 's2_total': s2A + s2B}


def sample_lognormal_eta_two_link(eta_nominal, Cn2, wavelength, distance_m,
                                   N_samples=50_000, rng=None,
                                   distance_A_m=None, distance_B_m=None,
                                   eta_A_nominal=None, eta_B_nominal=None,
                                   return_links=False):
    """Monte Carlo samples of eta = eta_A * eta_B with INDEPENDENT log-normal
    fading on each link. Each link is clipped to [0,1] separately (a single
    link cannot exceed unit transmittance). Draws X_A then X_B from one rng,
    so the two streams are independent by construction."""
    if rng is None:
        rng = np.random.default_rng(MC_SEED)
    LA, LB = two_link_geometry(distance_m, distance_A_m, distance_B_m)
    eA0, eB0 = two_link_nominal(eta_nominal, eta_A_nominal, eta_B_nominal)
    if not turbulence_enabled:
        out = np.full(N_samples, eA0 * eB0, dtype=np.float64)
        return (out, np.full(N_samples, eA0), np.full(N_samples, eB0)) if return_links else out
    p = two_link_lognormal_params(Cn2, wavelength, LA, LB)
    XA = rng.normal(p['mu_A'], math.sqrt(p['s2_A']), size=N_samples)
    XB = rng.normal(p['mu_B'], math.sqrt(p['s2_B']), size=N_samples)
    etaA = np.clip(eA0 * np.exp(XA), 0.0, 1.0)
    etaB = np.clip(eB0 * np.exp(XB), 0.0, 1.0)
    eta = etaA * etaB
    return (eta, etaA, etaB) if return_links else eta


def turbulence_statistics_two_link(eta_nominal, Cn2, wavelength, distance_m,
                                    N_samples=50_000, rng=None, **link_kwargs):
    """Analytic (unclipped) + MC moments of eta = eta_A eta_B.
    Independence gives E[eta] = E[eta_A]E[eta_B] and
    1 + SI = (1 + SI_A)(1 + SI_B), with 1 + SI_i = exp(s2_i)."""
    LA, LB = two_link_geometry(distance_m, link_kwargs.get('distance_A_m'),
                                link_kwargs.get('distance_B_m'))
    eA0, eB0 = two_link_nominal(eta_nominal, link_kwargs.get('eta_A_nominal'),
                                 link_kwargs.get('eta_B_nominal'))
    p = two_link_lognormal_params(Cn2, wavelength, LA, LB)
    mean_analytic = eA0 * eB0 * math.exp(p['mu_total'] + 0.5 * p['s2_total'])
    SI_analytic = math.exp(p['s2_total']) - 1.0
    var_analytic = (eA0 * eB0) ** 2 * SI_analytic
    s = sample_lognormal_eta_two_link(eta_nominal, Cn2, wavelength, distance_m,
                                       N_samples, rng, **link_kwargs)
    return {**p, 'mean_analytic': mean_analytic, 'var_analytic': var_analytic,
            'SI_analytic': SI_analytic, 'mean_mc': float(np.mean(s)),
            'var_mc': float(np.var(s)), 'std_mc': float(np.std(s)), 'samples': s}


def lognormal_gh_nodes(eta_link_nominal, mu_X, sigma_X2, n_nodes=TWO_LINK_GH_NODES,
                        clip=True):
    """Gauss-Hermite nodes/weights for ONE link: sum_i w_i g(eta_i) ~ E[g(eta)],
    eta = eta_link_nominal * exp(X), X ~ N(mu_X, sigma_X2). Weights sum to 1.
    If clip, eta_i is clipped to <= 1 (the clipped mass sits at eta = 1, which
    is exactly what the Monte Carlo sampler does)."""
    z, w = np.polynomial.hermite.hermgauss(int(n_nodes))
    s = math.sqrt(sigma_X2) if sigma_X2 > 0 else 0.0
    X = mu_X + math.sqrt(2.0) * s * z
    eta = float(eta_link_nominal) * np.exp(X)
    if clip:
        eta = np.minimum(eta, 1.0)
    return eta, w / math.sqrt(math.pi)


def average_over_two_lognormal_links(func, eta_A_nominal, eta_B_nominal,
                                      sigma_R2_A, sigma_R2_B,
                                      n_nodes=TWO_LINK_GH_NODES, clip=True):
    """E[func(eta_A * eta_B)] over INDEPENDENT log-normal fading on both links,
    by a tensor-product Gauss-Hermite rule (n_nodes^2 points, one vectorized
    call of func). func must accept a 1-D eta array and return an array."""
    muA, s2A = (float(v) for v in lognormal_params(sigma_R2_A))
    muB, s2B = (float(v) for v in lognormal_params(sigma_R2_B))
    eA, wA = lognormal_gh_nodes(eta_A_nominal, muA, s2A, n_nodes, clip)
    eB, wB = lognormal_gh_nodes(eta_B_nominal, muB, s2B, n_nodes, clip)
    eta = np.outer(eA, eB).ravel()
    w = np.outer(wA, wB).ravel()
    return float(np.sum(w * np.asarray(func(eta), dtype=np.float64)))


def average_skr_lognormal_two_link(eta_nominal, sigma_R2_A, sigma_R2_B, alpha, pd,
                                    thetaA, thetaB, Nmax=NMAX_DEFAULT, phi=0.0,
                                    n_nodes=TWO_LINK_GH_NODES,
                                    eta_A_nominal=None, eta_B_nominal=None):
    """<R> = E_{X_A,X_B}[ R(alpha, eta_A eta_B) ]  (two independent links)."""
    eA0, eB0 = two_link_nominal(eta_nominal, eta_A_nominal, eta_B_nominal)
    f = lambda eta: key_rate(alpha, np.clip(eta, 1e-300, 1.0), pd, thetaA, thetaB, Nmax, phi)
    return max(average_over_two_lognormal_links(f, eA0, eB0, sigma_R2_A, sigma_R2_B,
                                                 n_nodes), 0.0)


def average_observables_two_link(eta_nominal, sigma_R2_A, sigma_R2_B, alpha, pd,
                                  thetaA, thetaB, Nmax=NMAX_DEFAULT, phi=0.0,
                                  n_nodes=TWO_LINK_GH_NODES,
                                  eta_A_nominal=None, eta_B_nominal=None):
    """Two-link ensemble <p_XX>, <e_X>, <e_Z>, <R>. e_X and e_Z are the
    gain-weighted averages <p_XX e>/<p_XX>, same definition as the single-link
    average_qber_lognormal / average_phase_error_lognormal."""
    theta = thetaA - thetaB
    eA0, eB0 = two_link_nominal(eta_nominal, eta_A_nominal, eta_B_nominal)
    muA, s2A = (float(v) for v in lognormal_params(sigma_R2_A))
    muB, s2B = (float(v) for v in lognormal_params(sigma_R2_B))
    eA, wA = lognormal_gh_nodes(eA0, muA, s2A, n_nodes)
    eB, wB = lognormal_gh_nodes(eB0, muB, s2B, n_nodes)
    eta = np.clip(np.outer(eA, eB).ravel(), 1e-300, 1.0)
    w = np.outer(wA, wB).ravel()
    gamma = np.sqrt(eta) * alpha ** 2
    pxx = np.maximum(p_xx_total(phi, theta, gamma, pd), 0.0)
    ex = np.clip(bit_error_rate(phi, theta, gamma, pd), 0.0, 0.5)
    ez = phase_error_upper_bound(alpha, eta, pd, thetaA, thetaB, Nmax, phi)
    R = np.maximum(key_rate(alpha, eta, pd, thetaA, thetaB, Nmax, phi), 0.0)
    pxx_avg = float(np.sum(w * pxx))
    den = pxx_avg if pxx_avg > 1e-300 else np.nan
    return {'pXX': pxx_avg,
            'eX': float(np.clip(np.sum(w * pxx * ex) / den, 0.0, 0.5)) if den == den else 0.5,
            'eZ': float(np.clip(np.sum(w * pxx * ez) / den, 0.0, 0.5)) if den == den else 0.5,
            'R': float(np.sum(w * R))}


def tfqkd_skr_turbulence_two_link(distance_km, pd, thetaA, thetaB, Cn2, fso_params,
                                   N_samples=None, Nmax=4, alpha_fixed=0.20, rng=None):
    """Monte Carlo ergodic SKR, independent fading on A-C and B-C (each of
    length L/2, Rytov variance evaluated at L/2). Two-link counterpart of
    tfqkd_skr_turbulence()."""
    if N_samples is None:
        N_samples = N_MC
    if rng is None:
        rng = np.random.default_rng(MC_SEED)
    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    R_arr = np.zeros(len(distance_km))
    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        eta_s = sample_lognormal_eta_two_link(float(eta_nom), Cn2, fso_params['wavelength'],
                                               L_km * 1e3, N_samples, rng)
        R_arr[i] = float(np.mean(key_rate(alpha_fixed, eta_s, pd, thetaA, thetaB, Nmax)))
    return R_arr, eta_nominal_arr


def average_skr_lognormal_two_link_array(distance_km, pd, thetaA, thetaB, Cn2,
                                          fso_params, Nmax=NMAX_DEFAULT,
                                          alpha_fixed=0.20, n_nodes=TWO_LINK_GH_NODES):
    """Quadrature ergodic SKR vs distance, two independent links of length L/2.
    Returns (R, eta_nominal, sigma_R2_per_link)."""
    distance_km = np.asarray(distance_km, dtype=np.float64)
    fso = deterministic_fso_channel(fso_params['alpha_atm'], distance_km,
                                     fso_params['wavelength'], fso_params['w0'],
                                     fso_params['receiver_radius'])
    eta_nominal_arr = fso['eta_fso']
    R_arr = np.zeros(len(distance_km))
    sR2_link = rytov_variance(Cn2, fso_params['wavelength'], 0.5 * distance_km * 1e3)
    for i, (L_km, eta_nom) in enumerate(zip(distance_km, eta_nominal_arr)):
        if eta_nom < 1e-50:
            continue
        R_arr[i] = average_skr_lognormal_two_link(
            float(eta_nom), float(sR2_link[i]), float(sR2_link[i]), alpha_fixed, pd,
            thetaA, thetaB, Nmax, n_nodes=n_nodes)
    return R_arr, eta_nominal_arr, sR2_link


def _self_test():
    thetaA, thetaB = PARAMS['thetaA'], PARAMS['thetaB']
    pd = PARAMS['pd']
    loss_dB = np.array([0, 10, 20, 40, 60, 80, 100])
    eta = dB_to_eta(loss_dB)

    pZZ_pn = p_zz_photon(1, 1, thetaA, thetaB, eta, pd)
    assert np.all(pZZ_pn >= -1e-12) and np.all(pZZ_pn <= 1 + 1e-12)

    R = key_rate(PARAMS['alpha'], eta, pd, thetaA, thetaB, NMAX_DEFAULT)
    assert np.all(R >= 0) and np.all(np.diff(R) <= 1e-15)

    h_vals = h(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(h_vals, [0.0, 1.0, 0.0], atol=1e-12)

    a_opt, _, R_max = optimize_alpha(float(eta[2]), pd, thetaA, thetaB, Nmax=NMAX_DEFAULT)
    assert R_max >= 0 and ALPHA_BOUNDS[0] <= a_opt <= ALPHA_BOUNDS[1]

    # --- New: quadrature-based ensemble averaging sanity checks ---
    theta_test = thetaA - thetaB
    sigma_R2_test = 0.3  # well inside sigma_R^2 < 1 validity regime
    eta_nom_test = float(eta[3])  # 40 dB point

    pxx_q, _ = average_pxx_lognormal(eta_nom_test, sigma_R2_test, 0.0, theta_test,
                                      PARAMS['alpha'], pd)
    assert 0.0 <= pxx_q <= 1.0

    eX_q, _ = average_qber_lognormal(eta_nom_test, sigma_R2_test, 0.0, theta_test,
                                      PARAMS['alpha'], pd)
    assert 0.0 <= eX_q <= 0.5

    R_q, _ = average_skr_lognormal(eta_nom_test, sigma_R2_test, PARAMS['alpha'], pd,
                                    thetaA, thetaB, NMAX_DEFAULT)
    assert R_q >= 0.0

    # Sigma_R2 -> 0 limit: <R> must reduce to deterministic key_rate(eta_nominal)
    R_q_zero, _ = average_skr_lognormal(eta_nom_test, 1e-10, PARAMS['alpha'], pd,
                                         thetaA, thetaB, NMAX_DEFAULT)
    R_det_zero = key_rate(PARAMS['alpha'], eta_nom_test, pd, thetaA, thetaB, NMAX_DEFAULT)
    assert abs(R_q_zero - R_det_zero) < 1e-6 * max(R_det_zero, 1e-12) + 1e-8

    print('Self-test OK.')

    # --- New: average_gain_lognormal / average_phase_error_lognormal checks ---
    gain_q, _ = average_gain_lognormal(eta_nom_test, sigma_R2_test, 0.0, theta_test,
                                        PARAMS['alpha'], pd)
    assert abs(gain_q - pxx_q) < 1e-12  # must be an exact alias of average_pxx_lognormal
    assert 0.0 <= gain_q <= 1.0

    eZ_q, _ = average_phase_error_lognormal(eta_nom_test, sigma_R2_test, PARAMS['alpha'],
                                             pd, thetaA, thetaB, NMAX_DEFAULT)
    assert 0.0 <= eZ_q <= 0.5

    # Sigma_R2 -> 0 limit: <p_XX>, <e_X>, <e_Z> must reduce to deterministic values
    gamma_zero = math.sqrt(eta_nom_test) * PARAMS['alpha'] ** 2
    pXX_det_zero = float(p_xx_total(0.0, theta_test, gamma_zero, pd))
    eX_det_zero = float(np.clip(bit_error_rate(0.0, theta_test, gamma_zero, pd), 0.0, 0.5))
    eZ_det_zero = float(phase_error_upper_bound(PARAMS['alpha'], eta_nom_test, pd,
                                                 thetaA, thetaB, NMAX_DEFAULT))

    gain_q_zero, _ = average_gain_lognormal(eta_nom_test, 1e-10, 0.0, theta_test,
                                             PARAMS['alpha'], pd)
    eX_q_zero, _ = average_qber_lognormal(eta_nom_test, 1e-10, 0.0, theta_test,
                                           PARAMS['alpha'], pd)
    eZ_q_zero, _ = average_phase_error_lognormal(eta_nom_test, 1e-10, PARAMS['alpha'],
                                                  pd, thetaA, thetaB, NMAX_DEFAULT)

    assert abs(gain_q_zero - pXX_det_zero) < 1e-6 * max(pXX_det_zero, 1e-12) + 1e-9
    assert abs(eX_q_zero - eX_det_zero) < 1e-5 * max(eX_det_zero, 1e-12) + 1e-7
    assert abs(eZ_q_zero - eZ_det_zero) < 1e-4 * max(eZ_det_zero, 1e-12) + 1e-6

    # degradation_breakdown internal consistency: log-attribution fractions
    # for gain/eX/eZ plus cross-term must sum to 1 (by construction)
    brk = degradation_breakdown(eta_nom_test, sigma_R2_test, PARAMS['alpha'], pd,
                                 thetaA, thetaB, NMAX_DEFAULT)
    frac_sum = (brk['attribution_gain'] + brk['attribution_eX'] +
                brk['attribution_eZ'] + brk['attribution_cross_term'])
    assert abs(frac_sum - 1.0) < 1e-9
    assert brk['R_turb'] <= brk['R_det'] + 1e-12  # turbulence cannot increase ensemble R here
    # (pXX_turb/eX/eZ all worse-or-equal at this single-eta point pushes R down;
    #  see module-level note for why this needn't hold pointwise in eta but does
    #  hold for the AVERAGE quantities feeding R_det vs R_turb via the
    #  counterfactual construction, since each one-factor swap is individually
    #  unfavourable in the weak-turbulence regime used here)

    print('Extended self-test (ensemble gain/QBER/phase-error + attribution) OK.')

    # --- Two independent links (Alice-Charlie, Bob-Charlie) ---
    # mean-one fading survives the product of two independent factors
    m = average_over_two_lognormal_links(lambda e: e, 0.01, 0.02, 0.3, 0.2, clip=False)
    assert abs(m / (0.01 * 0.02) - 1.0) < 1e-12
    # 2-D rule == exact 1-D reduction with s2_A + s2_B (tiny eta => clip irrelevant)
    f_kr = lambda e: key_rate(PARAMS['alpha'], np.clip(e, 1e-300, 1.0), pd, thetaA, thetaB, NMAX_DEFAULT)
    two = average_over_two_lognormal_links(f_kr, math.sqrt(1e-3), math.sqrt(1e-3), 0.15, 0.15)
    one, _ = average_skr_lognormal(1e-3, 0.30, PARAMS['alpha'], pd, thetaA, thetaB, NMAX_DEFAULT)
    assert abs(two - one) < 1e-6 * one
    print('Two-link (independent A-C / B-C fading) self-test OK.')


if __name__ == '__main__':
    _self_test()

    FSO_PARAMS = dict(alpha_atm=0.046, wavelength=1550e-9, w0=0.05,
                       receiver_radius=0.15, dist_min_km=0.5, dist_max_km=500.0)
    dist_km = np.linspace(FSO_PARAMS['dist_min_km'], FSO_PARAMS['dist_max_km'], 60)

    R_fso, eta_fso, alpha_fso = tfqkd_skr_fso(
        dist_km, PARAMS['pd'], PARAMS['thetaA'], PARAMS['thetaB'],
        FSO_PARAMS, Nmax=NMAX_DEFAULT, optimise=True)
    print(f'Deterministic FSO: {int(np.sum(R_fso > 1e-15))}/{len(dist_km)} '
          f'positive-rate points; max range '
          f'~{dist_km[R_fso > 1e-15][-1]:.0f} km' if np.any(R_fso > 1e-15) else 'no positive-rate points')

    rng = np.random.default_rng(MC_SEED)
    R_turb, eta_nom = tfqkd_skr_turbulence(
        dist_km, PARAMS['pd'], PARAMS['thetaA'], PARAMS['thetaB'],
        Cn2_classes['moderate'], FSO_PARAMS, N_samples=N_MC,
        Nmax=NMAX_DEFAULT, alpha_fixed=0.20, rng=rng)
    print(f'Turbulent (moderate) ergodic SKR computed for {len(dist_km)} distance points, '
          f'N_MC={N_MC:,} samples/point.')