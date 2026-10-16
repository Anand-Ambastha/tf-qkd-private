# -*- coding: utf-8 -*-
"""
02_test_gammagamma_formulas.py
==============================
Tests the GAMMA-GAMMA extension of the analytical averaging framework (progress report,
Sections 3-4): cumulant-based expansion with aperture averaging, single cumulative channel
eta_tot = eta_FSO(L) * I,  I = Xa Xb,  Xa ~ Gamma(alpha), Xb ~ Gamma(beta).

  G0  cumulants      log-Gamma cumulants kappa_1..4 (polygamma) vs direct numerical integration
  G1  reduction      kappa_3 = kappa_4 = 0 recovers the log-normal formula exactly (Sec. 3.2)
  G2  scintillation  alpha, beta, SI(sigma_R^2) with/without aperture (Fig. 2); Var(I) vs Monte Carlo
  G3  Table 1        errors of delta method, 2nd/3rd/4th-order GG formula vs quadrature, aperture gain
  G4  Table 2        cost of applying the log-normal formula to a Gamma-Gamma channel
  G5  Fig. 3         error vs Rytov variance (L = 80 km, aperture averaged)
  G6  Sec. 4.3       choice of expansion point: median (log-mean) vs mean vs mode
  G7  references     Laguerre quadrature vs Bessel-K quadrature vs Monte Carlo (reference quality)
  G8  claims         "naive log-normal delta gives a negative key rate near Rytov variance two"

Outputs: results/tables/gammagamma_*.csv|json, results/figures/gg_*.png
Run:  python3 analytical_validation/02_test_gammagamma_formulas.py       (QUICK=1 for a short run)
"""
import os, sys, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy import special as sp
from scipy.integrate import quad
from analytical_validation import common as C, cal_derivatives as D, averaging as A
from analytical_validation import paper_values as P, reporting as R
from analytical_validation.reporting import plt

QUICK = bool(os.environ.get('QUICK'))
chk = R.Checks('gammagamma')


# --------------------------------------------------------------- one row of Table 1/2 (a link configuration)
def config(row_L, s_R2=None):
    """row_L: distance in km (the '80*' rows use L=80 for eta0 and d^2 but a prescribed Rytov variance)."""
    L = 80 if row_L == '80*' else row_L
    s = C.rytov(L) if s_R2 is None else s_R2
    d2 = A.aperture_d2(L)
    return dict(L=L, s=s, d2=d2, eta_tot0=C.eta_fso(L), eta0m=math.sqrt(C.eta_fso(L)))


def gg_numbers(cf, q='R', d2=None, n_lag=140, expo=1.2):
    """All quantities for one configuration: reference + every formula (relative errors in %)."""
    d2 = cf['d2'] if d2 is None else d2
    al, be = A.gg_params(cf['s'], d2, expo)
    kap = A.log_gamma_cumulants(al, be)
    e0 = cf['eta0m']
    ref = A.gg_reference_laguerre(q, e0, al, be, n=n_lag)
    out = dict(alpha=al, beta=be, SI=A.gg_si(al, be), ref=ref, kap=kap)
    # --- expansion about the log-mean point eta0 e^{k1/2}  (the report's "median")
    eta_pt = e0 * math.exp(0.5 * kap[0]); d_pt = D.all_derivs(eta_pt)[q]
    for o in (2, 3, 4):
        out[f'o{o}'] = C.pct(A.gg_expansion(d_pt, eta_pt, kap, 0.0, o), ref)
    # --- naive delta method about the mean (I = 1):  Q(eta0) + 1/2 Q''(eta0) Var(eta_tot)
    d0 = D.all_derivs(e0)[q]
    th = A.theta_ops(d0, e0) * 0.5 ** np.arange(1, 5)
    q2 = (th[1] - th[0]) / cf['eta_tot0'] ** 2
    out['naive'] = C.pct(d0[0] + 0.5 * q2 * cf['eta_tot0'] ** 2 * out['SI'], ref)
    # --- log-normal formula with the same Rytov variance (wrong model)
    out['ln2'] = C.pct(A.ln_on_gg(D.all_derivs(e0 * math.exp(-cf['s'] / 4))[q], e0, cf['s'], 2), ref)
    out['ln4'] = C.pct(A.ln_on_gg(D.all_derivs(e0 * math.exp(-cf['s'] / 4))[q], e0, cf['s'], 4), ref)
    # --- log-normal naive delta (mean-preserving, eta_tot): Var = eta0^2 (e^s - 1)
    out['ln_naive'] = C.pct(d0[0] + 0.5 * q2 * cf['eta_tot0'] ** 2 * math.expm1(cf['s']), ref)
    # aperture gain
    al0, be0 = A.gg_params(cf['s'], 0.0, expo)
    out['gain'] = C.pct(ref, A.gg_reference_laguerre(q, e0, al0, be0, n=n_lag))
    return out


# =============================================================== G0 cumulants
print('G0 cumulants ...')
def cum_numeric(k):
    """cumulants 1..4 of ln X, X~Gamma(k, rate k), by direct numerical integration of the density."""
    lg = lambda x: math.log(x)
    pdf = lambda x: math.exp(k * math.log(k) - sp.gammaln(k) + (k - 1) * math.log(x) - k * x)
    m = [quad(lambda x, n=n: lg(x) ** n * pdf(x), 0, np.inf, epsabs=0, epsrel=1e-12, limit=500)[0] for n in range(1, 5)]
    mu = m[0]
    c2 = m[1] - mu ** 2
    c3 = m[2] - 3 * mu * m[1] + 2 * mu ** 3
    mu4 = m[3] - 4 * mu * m[2] + 6 * mu ** 2 * m[1] - 3 * mu ** 4
    return np.array([mu, c2, c3, mu4 - 3 * c2 ** 2])
worst = 0.0
rows = []
for k in (2.0, 5.3, 11.4, 30.0):
    num = cum_numeric(k)
    form = np.array([sp.digamma(k) - math.log(k), sp.polygamma(1, k), sp.polygamma(2, k), sp.polygamma(3, k)])
    rel = np.abs(num - form) / np.abs(form)
    worst = max(worst, rel.max())
    rows.append([k] + [f'{x:.3e}' for x in rel])
C.save_csv('gammagamma_G0_cumulants.csv', ['k', 'kappa1_relerr', 'kappa2_relerr', 'kappa3_relerr', 'kappa4_relerr'], rows)
chk.add('G0-cumulants', 'Polygamma cumulants of ln Gamma(k,k) vs direct numerical integration (k = 2..30, orders 1-4)', worst, 1e-5, unit='max rel. error',
        note='limited by 4th-moment cancellation in the numerical reference at large k')

# =============================================================== G1 reduction to log-normal
print('G1 reduction ...')
red = 0.0
for L in (20, 80, 160):
    cf = config(L); s = cf['s']
    kap_ln = np.array([-s / 2.0, s, 0.0, 0.0])
    eta_pt = cf['eta0m'] * math.exp(0.5 * kap_ln[0]); d = D.all_derivs(eta_pt)['R']
    for o in (2, 4):
        a = A.gg_expansion(d, eta_pt, kap_ln, 0.0, o)
        b = A.ln_on_gg(d, cf['eta0m'], s, o)
        red = max(red, abs(a - b) / abs(b))
chk.add('G1-reduction', 'Cumulant formula with kappa_3 = kappa_4 = 0 equals the log-normal formula (2nd and 4th order)', red, 1e-12, unit='max rel. diff')

# =============================================================== G2 scintillation, moments
print('G2 scintillation ...')
L80 = 80; d2_80 = A.aperture_d2(L80)
chk.add('G2-d2', 'Aperture parameter d^2 = k D^2 / 4L at L=80 km, D=0.30 m (printed 1.14)', abs(d2_80 - 1.14), 0.01, unit='abs. diff')
s_ax = np.linspace(0.01, 10, 400)
si_pt = np.array([A.gg_si(*A.gg_params(s, 0.0)) for s in s_ax]); si_ap = np.array([A.gg_si(*A.gg_params(s, d2_80)) for s in s_ax])
si_pt_r = np.array([A.gg_si(*A.gg_params(s, 0.0, 2.4)) for s in s_ax]); si_ap_r = np.array([A.gg_si(*A.gg_params(s, d2_80, 2.4)) for s in s_ax])
# (i) the report's Fig. 2 is reproduced only with the Rytov VARIANCE raised to 12/5
chk.add('G2-fig2-point-report-convention', 'Report Fig. 2: point-receiver SI peak 0.71 (reproduced with variance^(12/5) convention)', abs(si_pt_r.max() - 0.71), 0.01, unit='abs. diff')
chk.add('G2-fig2-aperture-report-convention', 'Report Fig. 2: aperture-averaged SI peak 0.44 (same convention)', abs(si_ap_r.max() - 0.44), 0.01, unit='abs. diff')
chk.add('G2-fig2-peak-position', 'Report Fig. 2: SI peak at Rytov variance ~1 (same convention)', abs(s_ax[np.argmax(si_pt_r)] - 1.0), 0.25, unit='abs. diff')
chk.add('G2-fig2-decay-report', 'Report convention: SI at Rytov variance 10 falls to ~0.08 (point receiver)', abs(si_pt_r[-1] - 0.08), 0.02, unit='abs. diff')
# (ii) the correct Andrews-Phillips exponent: SI rises to ~1.2-1.3 and then saturates toward 1, never decays to ~0
chk.add('G2-correct-saturation', 'Correct exponent sigma_R^(12/5): SI(10) lies in [1.1, 1.35] (saturation toward 1)', float(1.1 <= si_pt[-1] <= 1.35), 1.0, mode='>=', unit='bool')
chk.add('G2-correct-vs-report-at-10', 'SI at Rytov variance 10: correct / report convention (point receiver)', si_pt[-1] / si_pt_r[-1], 5.0, mode='>=', unit='ratio',
        note=f'correct {si_pt[-1]:.3f} vs report convention {si_pt_r[-1]:.3f}')
chk.add('G2-aperture-lowers', 'Aperture lowers SI at every turbulence strength (correct convention)', float(np.all(si_ap < si_pt)), 1.0, mode='>=', unit='bool')
chk.info('G2-exponent-finding', 'Gamma-Gamma exponent in the progress report',
         'The report\'s Fig. 2 (peak 0.71 then decay to ~0.08 at Rytov variance 10) is reproduced only when sigma_R^(12/5) is evaluated as (Rytov variance)^(12/5). '
         'In the Andrews-Phillips formulas sigma_R is the Rytov STANDARD DEVIATION, i.e. the exponent is 6/5 on the variance; with it the scintillation index peaks at ~1.2-1.3 near '
         'variance 5-10 and saturates toward 1 (aperture-averaged curve scales down accordingly). The same slip was present in the repository\'s Gamma-Gamma code until v2.2.0. '
         'The cumulant formulas themselves are unaffected and are tested below under BOTH conventions.')
al, be = A.gg_params(0.6, d2_80)
rng = np.random.default_rng(1); n_mc = 600_000 if not QUICK else 100_000
I = rng.gamma(al, 1 / al, n_mc) * rng.gamma(be, 1 / be, n_mc)
si_mc = I.var() / I.mean() ** 2
chk.add('G2-SI-vs-MC', 'SI = 1/a + 1/b + 1/(ab) vs Monte Carlo variance of I (sigma_R^2 = 0.6)', abs(si_mc - A.gg_si(al, be)) / A.gg_si(al, be), 0.01, unit='rel. diff')
# Fig. 1: log-domain skewness at sigma_R^2 = 0.6 (point receiver and aperture)
al_p, be_p = A.gg_params(0.6, 0.0); k_p = A.log_gamma_cumulants(al_p, be_p); k_a = A.log_gamma_cumulants(al, be)
k_pr = A.log_gamma_cumulants(*A.gg_params(0.6, 0.0, 2.4)); k_ar = A.log_gamma_cumulants(*A.gg_params(0.6, d2_80, 2.4))
chk.info('G2-kappa3', 'Fig. 1 label "kappa_3 = -0.19" at sigma_R^2 = 0.6',
         f'kappa_3 = {k_pr[2]:+.3f} (point) / {k_ar[2]:+.3f} (aperture) with the report convention; {k_p[2]:+.3f} / {k_a[2]:+.3f} with the correct exponent. Either way kappa_3 < 0: left-skewed log-domain distribution.')
Y = np.log(I)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].hist(Y, bins=160, density=True, alpha=.7, color='C0', label='sampled Y = ln Xa + ln Xb (aperture)')
yy = np.linspace(Y.min(), Y.max(), 400); ax[0].plot(yy, np.exp(-(yy - Y.mean()) ** 2 / (2 * Y.var())) / math.sqrt(2 * math.pi * Y.var()), 'r', label='Gaussian, same mean/variance')
ax[0].set_xlabel('Y = ln(eta/eta_nominal)'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=7); ax[0].set_title(f'Log-domain skewness (sigma_R^2=0.6, kappa_3={k_a[2]:+.2f})')
ax[1].plot(s_ax, si_pt, 'C0', label='point receiver'); ax[1].plot(s_ax, si_ap, 'C2', label=f'aperture-averaged (D=0.30 m, L=80 km, d^2={d2_80:.2f})')
ax[1].axvline(1, color='gray', ls=':'); ax[1].set_xlabel('Rytov variance'); ax[1].set_ylabel('scintillation index'); ax[1].legend(fontsize=8); ax[1].set_title('Scintillation saturates; aperture lowers it')
sg = np.logspace(-2, 3, 200); ap_ = np.array([A.gg_params(s, d2_80) for s in sg]); pp_ = np.array([A.gg_params(s, 0.0) for s in sg])
ax[2].loglog(sg, pp_[:, 0], 'C3', label='alpha'); ax[2].loglog(sg, pp_[:, 1], 'C1', label='beta (point)'); ax[2].loglog(sg, ap_[:, 1], 'C2--', label='beta (aperture)')
ax[2].set_xlabel('Rytov variance'); ax[2].set_ylabel('shape parameter'); ax[2].legend(fontsize=8); ax[2].set_title('Gamma-Gamma parameters')
R.savefig(fig, 'gg_F1_log_skewness_and_scintillation')

# =============================================================== G3 / G4 Tables 1, 2
print('G3/G4 Tables 1 and 2 ...')
rows_full = [(20, None), (50, None), (80, None), (100, None), (120, None), (160, None), ('80*', 3.0), ('80*', 5.0)]
sel = list(range(8)) if not QUICK else [0, 2, 5, 7]
rows_cfg = [rows_full[i] for i in sel]
res, res_rep = [], []
for (rl, sprescribed) in rows_cfg:
    cf = config(rl, sprescribed)
    res.append((rl, cf, gg_numbers(cf)))                       # correct exponent  sigma_R^(12/5) = (s)^(6/5)
    res_rep.append((rl, cf, gg_numbers(cf, expo=2.4)))         # report convention (s)^(12/5)
TOL = 0.25                                                      # percentage points
t1, t2, cmp1, cmp2 = [], [], [], []
for j, i in enumerate(sel):
    rl, cf, g = res[j]; _, _, gr = res_rep[j]
    row = [rl, f"{cf['s']:.2f}", f"{g['alpha']:.2f}/{g['beta']:.2f}", f"{gr['alpha']:.2f}/{gr['beta']:.2f}"]
    for key in ('gain', 'naive', 'o2', 'o3', 'o4'):
        pv = P.GG_T1[key][i]; ok = abs(gr[key] - pv) <= TOL
        row += [f'{g[key]:+.3f}', f'{gr[key]:+.3f}', f'{pv:+.2f}', 'MATCH' if ok else 'DIFFERS']
        cmp1.append((rl, key, gr[key], pv, ok))
    t1.append(row)
    row2 = [rl, f"{cf['s']:.2f}"]
    for key, pk in (('o4', 'gg'), ('ln2', 'ln2'), ('ln4', 'ln4')):
        pv = P.GG_T2[pk][i]; tol2 = max(1.0, 0.10 * abs(pv)) if key != 'o4' else TOL; ok = abs(gr[key] - pv) <= tol2
        row2 += [f'{g[key]:+.3f}', f'{gr[key]:+.3f}', f'{pv:+.2f}', 'MATCH' if ok else 'DIFFERS']
        cmp2.append((rl, key, gr[key], pv, ok))
    t2.append(row2)
h1 = ['L_km', 'sigma_R2', 'alpha/beta (correct)', 'alpha/beta (report conv.)'] + [f'{k}_{s}' for k in ('gain', 'naive', 'o2', 'o3', 'o4') for s in ('correct_%', 'report_conv_%', 'paper_%', 'status')]
h2 = ['L_km', 'sigma_R2'] + [f'{k}_{s}' for k in ('GG_4th', 'LN_2nd', 'LN_4th') for s in ('correct_%', 'report_conv_%', 'paper_%', 'status')]
C.save_csv('gammagamma_G3_table1.csv', h1, t1); C.save_csv('gammagamma_G4_table2.csv', h2, t2)
arr = lambda rs, k: np.array([g[k] for _, _, g in rs])
for tag, rs in (('correct', res), ('report-conv', res_rep)):
    o2, o3, o4 = arr(rs, 'o2'), arr(rs, 'o3'), arr(rs, 'o4')
    chk.add(f'G3-4th-order-{tag}', f'Claim: 4th-order GG formula within ~0.02 % of quadrature in all Table-1 rows ({tag} exponent)', np.abs(o4).max(), 0.03, unit='max |error| %')
    chk.add(f'G3-ordering-{tag}', f'Error shrinks with order, max|4th| < max|3rd| < max|delta| ({tag})', float(np.abs(o4).max() < np.abs(o3).max() < np.abs(arr(rs, "naive")).max() + 1e-9), 1.0, mode='>=', unit='bool')
    chk.add(f'G3-naive-modest-{tag}', f'Claim: delta method degrades gracefully for Gamma-Gamma, |error| < 2 % in all rows ({tag})', np.abs(arr(rs, 'naive')).max(), 2.0, unit='max |error| %')
    chk.add(f'G3-aperture-positive-{tag}', f'Aperture averaging raises the averaged key rate in every row ({tag})', float(np.all(arr(rs, 'gain') > 0)), 1.0, mode='>=', unit='bool')
n1 = sum(1 for r in cmp1 if r[-1])
chk.add('G3-reproduce-table1', f'Printed Table 1 entries reproduced with the report\'s exponent convention, within 0.25 pp ({n1} of {len(cmp1)})', n1 / len(cmp1), 0.7, mode='>=', unit='fraction',
        note='rows L>=50 reproduce; the L=20 row does not (see G4-L20-artifact)')
n1c = sum(1 for (rl, cf, g), i in zip(res, sel) for key in ('gain', 'naive', 'o2', 'o3', 'o4') if abs(g[key] - P.GG_T1[key][i]) <= TOL)
chk.info('G3-correct-exponent', 'Table 1 with the correct exponent', f'{n1c} of {len(cmp1)} printed entries are within 0.25 pp when the correct sigma_R^(12/5) is used; the differences are concentrated in the aperture-gain '
         'and delta-method columns at Rytov variance > 1 (the correct SI is larger there). The 4th-order formula stays within 0.03 % either way.')
ln2, ln4 = arr(res, 'ln2'), arr(res, 'ln4'); ln2r, ln4r = arr(res_rep, 'ln2'), arr(res_rep, 'ln4')
chk.add('G4-ln-model-error-report', 'Report convention: log-normal formula loses > 50 % of the key rate at Rytov variance 5 (2nd order)', abs(ln2r[-1]), 50.0, mode='>=', unit='|error| %')
chk.add('G4-ln-model-error-correct', 'Correct exponent: error of the log-normal formula at Rytov variance 5 (2nd order); still large', abs(ln2[-1]), 10.0, mode='>=', unit='|error| %',
        note=f'{ln2[-1]:+.1f} % (correct exponent) vs {ln2r[-1]:+.1f} % (report convention)')
chk.add('G4-order-no-fix', 'Raising the log-normal order does not repair the model error (|4th| > 0.5 |2nd| in the last row, correct exponent)', float(abs(ln4[-1]) > 0.5 * abs(ln2[-1])), 1.0, mode='>=', unit='bool')
n2 = sum(1 for r in cmp2 if r[-1]); chk.add('G4-reproduce-table2', f'Printed Table 2 entries reproduced with the report convention, tolerance max(1 pp, 10 %) ({n2} of {len(cmp2)})', n2 / len(cmp2), 0.6, mode='>=', unit='fraction')
ln_naive = arr(res, 'ln_naive')
chk.info('G4-L20-artifact', 'Table 1/2, L = 20 km row',
         f'the printed +0.36 % appears identically in the delta, 2nd, 3rd and 4th-order columns, so it does not depend on the formula. With the Laguerre, Bessel-K and Monte Carlo references used here the four '
         f'errors are {res[0][2]["naive"]:+.3f}, {res[0][2]["o2"]:+.3f}, {res[0][2]["o3"]:+.3f}, {res[0][2]["o4"]:+.3f} % (alpha, beta = {res[0][2]["alpha"]:.0f}, {res[0][2]["beta"]:.0f}). '
         'A formula-independent offset of this kind points at the reference integral for very large shape parameters (or its normalisation), not at the expansion.')

# =============================================================== G5 Fig. 3  error vs Rytov variance at L = 80 km
print('G5 error vs Rytov variance ...')
sv = np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]) if not QUICK else np.array([0.3, 1.0, 3.0, 5.0])
sw = []
for s in sv:
    cf = config('80*', float(s)); g = gg_numbers(cf)
    gr_ = gg_numbers(cf, expo=2.4)
    sw.append([s, g['o4'], g['o3'], g['o2'], g['naive'], g['ln2'], g['ln4'], g['ln_naive'], gr_['ln2'], gr_['o4']])
sw = np.array(sw)
C.save_csv('gammagamma_G5_sweep.csv', ['sigma_R2', 'GG_4th_%', 'GG_3rd_%', 'GG_2nd_%', 'GG_naive_%', 'LN_2nd_%', 'LN_4th_%', 'LN_naive_%', 'LN_2nd_%_report_conv', 'GG_4th_%_report_conv'], sw.tolist())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
ax[0].plot(sw[:, 0], sw[:, 1], 'g-d', label='correct model: Gamma-Gamma, 4th order'); ax[0].plot(sw[:, 0], sw[:, 2], 'g:s', ms=3, label='Gamma-Gamma, 3rd order')
ax[0].plot(sw[:, 0], sw[:, 5], 'r--s', label='WRONG model: log-normal formula, 2nd order'); ax[0].plot(sw[:, 0], sw[:, 8], color='gray', ls=':', label='same, with the report\'s exponent convention'); ax[0].plot(sw[:, 0], sw[:, 6], 'C1--o', label='WRONG model: log-normal formula, 4th order')
ax[0].set_xlabel('Rytov variance'); ax[0].set_ylabel('relative error in E[R] [%]'); ax[0].legend(fontsize=7.5); ax[0].set_title('Cost of the wrong turbulence model (L = 80 km, aperture-averaged)')
ax[1].semilogy(sw[:, 0], np.maximum(np.abs(sw[:, 1]), 1e-6), 'g-d', label='4th'); ax[1].semilogy(sw[:, 0], np.maximum(np.abs(sw[:, 2]), 1e-6), 'b-s', ms=3, label='3rd')
ax[1].semilogy(sw[:, 0], np.maximum(np.abs(sw[:, 3]), 1e-6), 'C1-o', ms=3, label='2nd'); ax[1].semilogy(sw[:, 0], np.maximum(np.abs(sw[:, 4]), 1e-6), 'k--', label='delta method')
ax[1].set_xlabel('Rytov variance'); ax[1].set_ylabel('|error| of E[R] [%] (floor 1e-6)'); ax[1].legend(fontsize=8); ax[1].set_title('Gamma-Gamma formulas: accuracy by order')
R.savefig(fig, 'gg_F2_error_vs_rytov')
chk.add('G5-4th-all-sigma', '4th-order GG error < 0.05 % for Rytov variance 0.1-5 (L = 80 km)', np.abs(sw[:, 1]).max(), 0.05, unit='max |error| %')
chk.add('G5-4th-all-sigma-report', '4th-order GG error < 0.05 % for Rytov variance 0.1-5 (report convention)', np.abs(sw[:, 9]).max(), 0.05, unit='max |error| %')

# G8 claim: naive LN delta negative near Rytov variance two (single cumulative log-normal channel)
sv2 = np.linspace(0.3, 3.5, 17); neg_at = None
for s in sv2:
    cf = config('80*', float(s)); g = gg_numbers(cf)
    if g['ln_naive'] <= -100.0 and neg_at is None:
        neg_at = s
chk.info('G8-ln-naive-negative', 'Report: naive delta applied to a log-normal channel gives a negative key rate near Rytov variance 2',
         ('this test: the delta-method key rate first becomes <= 0 at Rytov variance ~%.2f (single cumulative channel, sigma_X^2 = sigma_R^2)' % neg_at) if neg_at else
         'not reproduced in 0.3-3.5 (single cumulative channel).')

# =============================================================== G6 expansion point
print('G6 expansion point ...')
ep_rows, ep_err = [], {'logmean': [], 'mean': [], 'mode': []}
for (rl, cf, g) in res:
    al, be = A.gg_params(cf['s'], cf['d2']); kap = g['kap']; e0 = cf['eta0m']; ref = g['ref']
    row = [rl, f"{cf['s']:.2f}"]
    for name, y0 in (('logmean', kap[0]), ('mean', 0.0), ('mode', math.log(A.gg_mode_I(al, be)))):
        eta_pt = e0 * math.exp(0.5 * y0); dpt = D.all_derivs(eta_pt)['R']
        for o in (2, 4):
            e = C.pct(A.gg_expansion(dpt, eta_pt, kap, y0 - kap[0], o), ref)
            row.append(f'{e:+.4f}'); ep_err[name].append(abs(e))
    ep_rows.append(row)
C.save_csv('gammagamma_G6_expansion_point.csv', ['L_km', 'sigma_R2', 'logmean_2nd', 'logmean_4th', 'mean_2nd', 'mean_4th', 'mode_2nd', 'mode_4th'], ep_rows)
best = {k: np.array(v) for k, v in ep_err.items()}
chk.add('G6-logmean-best', 'Report: expanding at the median (log-mean point) is better than at the mean or the mode (sum of |errors|, all rows)',
        float(best['logmean'].sum() < min(best['mean'].sum(), best['mode'].sum())), 1.0, mode='>=', unit='bool')
chk.info('G6-terminology', 'Expansion point name', 'the report calls exp(E[ln eta]) the "median"; for Gamma-Gamma it is the geometric mean (log-mean), which differs slightly from the true median of eta. '
         'For the log-normal model the two coincide.')
fig, ax = plt.subplots(figsize=(9, 4.2))
labels = [f"{r[0]}/{r[1]}" for r in ep_rows]; x = np.arange(len(labels)); w = 0.13
for k, (name, col) in enumerate((('logmean', 'C3'), ('mean', 'C0'), ('mode', 'C2'))):
    for j, o in enumerate((2, 4)):
        vals = [abs(float(r[2 + 2 * k + j])) for r in ep_rows]
        ax.bar(x + (2 * k + j - 2.5) * w, np.maximum(vals, 1e-5), w, color=col, alpha=0.5 + 0.5 * j, label=f'{name}, {o}th order' if True else None)
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, fontsize=8); ax.set_xlabel('row: L [km] / Rytov variance'); ax.set_ylabel('|error| of E[R] [%]')
ax.legend(fontsize=7, ncol=3); ax.set_title('Choice of expansion point (Sec. 4.3)')
R.savefig(fig, 'gg_F3_expansion_point')

# =============================================================== G7 references
print('G7 reference quality ...')
rr = []
for (rl, cf, g) in res:
    al, be = A.gg_params(cf['s'], cf['d2']); e0 = cf['eta0m']
    b, norm = A.gg_reference_bessel('R', e0, al, be)
    m, se = A.gg_mc('R', e0, al, be, 300_000 if not QUICK else 60_000, seed=7)
    rr.append([rl, f"{cf['s']:.2f}", f'{al:.1f}', f'{be:.1f}', g['ref'], b, 100 * (b - g['ref']) / g['ref'], norm, m, 100 * (m - g['ref']) / g['ref'], 100 * se / g['ref'], abs(m - g['ref']) / se])
C.save_csv('gammagamma_G7_references.csv', ['L_km', 'sigma_R2', 'alpha', 'beta', 'R_Laguerre', 'R_BesselK', 'Bessel_vs_Laguerre_%', 'Bessel_pdf_norm', 'R_MC', 'MC_vs_Laguerre_%', 'MC_s.e._%', 'MC_z'], rr)
chk.add('G7-laguerre-vs-bessel', 'Laguerre tensor quadrature vs Bessel-K density quadrature (max over rows)', max(abs(r[6]) for r in rr) / 100, 1e-4, unit='max rel. diff')
chk.add('G7-mc', 'Monte Carlo vs Laguerre quadrature: max |z|', max(r[-1] for r in rr), 4.0, unit='max |z|')
chk.add('G7-pdf-norm', 'Bessel-K density integrates to 1 (max |norm-1|)', max(abs(r[7] - 1) for r in rr), 1e-6, unit='max abs. diff')
fig, ax = plt.subplots(figsize=(7.5, 4))
xs = np.arange(len(rr)); ax.bar(xs - 0.2, [abs(r[6]) for r in rr], 0.4, label='|Bessel-K - Laguerre| [%]'); ax.bar(xs + 0.2, [abs(r[9]) for r in rr], 0.4, label='|MC - Laguerre| [%]')
ax.set_yscale('log'); ax.set_xticks(xs); ax.set_xticklabels([f'{r[0]}/{r[1]}' for r in rr], rotation=30, fontsize=8); ax.legend(fontsize=8)
ax.set_ylabel('relative difference of E[R] [%]'); ax.set_title('Agreement of the three independent references')
R.savefig(fig, 'gg_F4_references')

# =============================================================== extra figure: error vs L (Table 1 columns) and aperture gain
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
Lrows = [r for r in res if r[0] != '80*']; Lx = [r[0] for r in Lrows]
for key, lab, st, col in (('naive', 'delta method', 'k--', 'k'), ('o2', '2nd order', 'C1-o', 'C1'), ('o3', '3rd order', 'b-s', 'b'), ('o4', '4th order', 'g-d', 'g')):
    ax[0].plot(Lx, [r[2][key] for r in Lrows], st, ms=4, label=lab)
ax[0].axhline(0, color='k', lw=.5); ax[0].set_xlabel('L [km]'); ax[0].set_ylabel('relative error of E[R] [%]'); ax[0].legend(fontsize=8); ax[0].set_title('Gamma-Gamma formulas vs quadrature (aperture-averaged)')
pg = [P.GG_T1['gain'][i] for i, (a, b) in enumerate(P.GG_ROWS) if a != '80*' and a in Lx]
ax[1].plot(Lx, [r[2]['gain'] for r in Lrows], 'C2-o', label='this test'); ax[1].plot([a for a, b in P.GG_ROWS if a != '80*' and a in Lx], pg, 'k^--', label='progress report, Table 1')
ax[1].set_xlabel('L [km]'); ax[1].set_ylabel('aperture gain in E[R] [%]'); ax[1].legend(); ax[1].set_title('Gain from aperture averaging (D = 0.30 m)')
R.savefig(fig, 'gg_F5_error_vs_distance_and_gain')

chk.save()
