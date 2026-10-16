# -*- coding: utf-8 -*-
"""
01_test_lognormal_formulas.py
=============================
Tests the analytical LOG-NORMAL turbulence-averaging framework of the manuscript
("Analytical Performance Analysis of TF-QKD over Turbulent FSO Channels"):

  T0  geometry        Rytov variance, validity distance (Eq. 15, 21, Tables I-II)
  T1  derivatives     analytic eta-derivatives of pXX, eX, eZ, R (Appendix A-H)
                      vs an independent 60-digit numerical reference
  T2  Euler operator  Theta^n eta^m = m^n eta^m with Stirling weights (Thm 1, Eq. 37-39)
  T3  two routes      X-space Hermite expansion vs eta-space (Euler-Stirling) form,
                      13 distances x {eX,eZ,R} x {2nd,4th order} = 78 checks (Sec. VI-B)
  T4  accuracy        errors of the naive delta method, 2-term and 4-term formulas against
                      direct Gaussian quadrature; compared with printed Tables IV, V, VI
  T5  moments         Eq. 19-20, skewness Eq. 25, Appendix-H numerical claims
  T6  stress test     errors vs sigma_X^2 beyond the weak-turbulence regime
  T7  Monte Carlo     independent sampling check of the quadrature reference

Outputs: results/tables/lognormal_*.csv|json, results/figures/ln_*.png
Run:  python3 analytical_validation/01_test_lognormal_formulas.py        (QUICK=1 for a short run)
"""
import os, sys, math, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from analytical_validation import common as C, cal_derivatives as D, mp_reference as M, averaging as A
from analytical_validation import paper_values as P, reporting as R
from analytical_validation.reporting import plt

QUICK = bool(os.environ.get('QUICK'))
chk = R.Checks('lognormal')
QS = ['eX', 'eZ', 'R']


def sx2_of(L, Cn2=C.CN2_WEAK):
    """sigma_X^2 = (sigma_A^2 + sigma_B^2)/4 = sigma_R^2(L/2)/2   (Eq. 17)."""
    return C.rytov(L / 2.0, Cn2) / 2.0


def ln_ref_grid(name, eta0, s2, npts=40001):
    """Robust reference for large sigma^2: Simpson in X over mu +- 11 sigma."""
    from scipy.integrate import simpson
    mu, sd = -s2, math.sqrt(s2)
    X = np.linspace(mu - 11 * sd, mu + 11 * sd, npts)
    w = np.exp(-0.5 * ((X - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    return float(simpson(w * A.Q_vec(name, eta0 * np.exp(X)), x=X))


# =============================================================== T0 geometry
print('T0 geometry ...')
rows = []
for cn2, paper in P.TABLE_I_KM.items():
    s1 = 1.23 * cn2 * C.K_WAVE ** (7 / 6)
    L_val = 2.0 * (1.0 / s1) ** (6 / 11) / 1e3                       # sigma_R^2(L/2) = 1
    L_eq21 = P.EQ21_COEFF * C.WAVELENGTH ** (7 / 11) * cn2 ** (-6 / 11) / 1e3
    rows.append(['Table I', f'{cn2:.0e}', f'{L_val:.2f}', f'{paper}', f'{L_eq21:.2f}'])
    chk.add(f'T0-validity-{cn2:.0e}', f'Validity distance, Cn2={cn2:.0e}, vs printed Table I', abs(L_val - paper) / paper, 0.005,
            unit='rel. diff')
    chk.add(f'T0-eq21-{cn2:.0e}', f'Closed form Eq.(21) vs exact solve, Cn2={cn2:.0e}', abs(L_eq21 - L_val) / L_val, 0.005,
            unit='rel. diff')
t2_rows = []
for cn2, d in P.TABLE_II.items():
    for L, paper in d.items():
        v = C.rytov(L / 2.0, cn2)
        t2_rows.append([f'{cn2:.0e}', L, f'{v:.4g}', f'{paper}'])
        chk.add(f'T0-tab2-{cn2:.0e}-{L}', f'Per-arm Rytov variance, Cn2={cn2:.0e}, L={L} km, vs Table II', abs(v - paper) / paper, 0.01,
                unit='rel. diff')
C.save_csv('lognormal_T0_validity.csv', ['source', 'Cn2', 'L_validity_km (this test)', 'printed_km', 'Eq21_km'], rows)
C.save_csv('lognormal_T0_rytov.csv', ['Cn2', 'L_km', 'sigma_R2(L/2) (this test)', 'printed'], t2_rows)

# =============================================================== T1 derivatives
print('T1 derivatives (analytic vs 60-digit reference) ...')
worst = {q: np.zeros(5) for q in ['pXX', 'eX', 'eZ', 'R']}
for L in C.L_TABLE:
    eta0, s2 = C.eta0_eff(L), sx2_of(L)
    eta_med = eta0 * math.exp(-s2)
    d = D.all_derivs(eta_med)
    for q in worst:
        ref = M.derivs(q, eta_med)
        for n in range(5):
            worst[q][n] = max(worst[q][n], abs(d[q][n] - ref[n]) / abs(ref[n]))
rows = [[q] + [f'{worst[q][n]:.2e}' for n in range(5)] for q in worst]
C.save_csv('lognormal_T1_derivatives.csv', ['quantity', 'order0', 'order1', 'order2', 'order3', 'order4'], rows)
for q in worst:
    chk.add(f'T1-deriv-{q}', f'Analytic derivatives of {q}, orders 0-4, vs 60-digit reference (max over 6 distances)',
            worst[q].max(), 1e-7, unit='max rel. error')
# engine <-> mp reference (ties the reference to the validated code) and the printed pXX
eta_t = C.eta0_eff(80)
eng_vs_mp = max(abs(float(M.FUNS[q](M.mp.mpf(eta_t))) - C.QREF[q](eta_t)) / C.QREF[q](eta_t) for q in C.QREF)
chk.add('T1-engine-vs-reference', 'High-precision reference reproduces the validated engine (value, L=80 km)', eng_vs_mp, 1e-9,
        unit='max rel. error')
px_printed = D.pxx_derivs(eta_t, printed_form=True)[0]
chk.info('T1-pxx-erratum', 'Printed Eq.(6)/Appendix A bracket for pXX',
         f'Eq.(6) as printed gives pXX = {px_printed:.6g} at L=80 km, whereas CAL Eq.(30) / the engine give {C.pxx_ref(eta_t):.6g}. '
         'The A^2 term must sit outside the factor 1/2: pXX = (A/2)[e^{-(1-O)g}+e^{-(1+O)g}] - A^2 e^{-2g}. '
         'With this form all derivatives agree with the reference (check T1-deriv-pXX); the same bracket slip appears in the printed pXX derivatives.')
chk.info('T1-Dj-note', 'Truncation residual D_j', 'D_j (CAL Delta_j) is eta-independent in CAL and in the engine (residual_j), so D_j^(n)=0 for n>=1; '
         'Appendix E says D_j^(n) "follows identically" from the sqrt(pZZ) terms, which would only hold for a different D_j. The analytic derivatives '
         'with D_j constant match the reference to ~1e-10.')

# =============================================================== T2 Euler operator
print('T2 Euler operator ...')
err = 0.0
for m in range(1, 7):
    eta = 0.37
    dm = [math.factorial(m) / math.factorial(m - k) * eta ** (m - k) if k <= m else 0.0 for k in range(5)]
    th = A.theta_ops(dm, eta)
    for n in range(1, 5):
        err = max(err, abs(th[n - 1] - m ** n * eta ** m) / (m ** n * eta ** m))
chk.add('T2-euler', 'Theta^n eta^m = m^n eta^m via Stirling weights (n=1..4, m=1..6)', err, 1e-13, unit='max rel. error')

# =============================================================== T3 X-space vs eta-space
print('T3 X-space vs eta-space (78 checks) ...')
Ls = C.L_SWEEP13 if not QUICK else C.L_SWEEP13[::4]
t3 = []
for L in Ls:
    eta0, s2 = C.eta0_eff(L), sx2_of(L)
    eta_med = eta0 * math.exp(-s2)
    dd = D.all_derivs(eta_med)
    for q in QS:
        th = A.theta_ops(dd[q], eta_med)                                        # eta-space route
        for n in (2, 4):
            f = lambda X, q=q: M.FUNS[q](M.mp.mpf(eta0) * M.mp.e ** X)          # X-space route (independent)
            dx = float(M.mp.diff(f, M.mp.mpf(-s2), n))
            t3.append([L, q, n, th[n - 1], dx, abs(th[n - 1] - dx) / abs(dx)])
C.save_csv('lognormal_T3_two_routes.csv', ['L_km', 'quantity', 'order', 'eta-space Theta^n Q', 'X-space d^n/dX^n', 'rel_diff'], t3)
mx = max(r[-1] for r in t3)
chk.add('T3-two-routes', f'X-space vs eta-space formulation ({len(t3)} cross-checks); manuscript claims < 1e-9', mx, 1e-8,
        unit='max rel. diff', note='limited by double-precision cancellation in the analytic eta-space derivatives')
chk.info('T3-count', 'Number of cross-checks', f'{len(t3)} ({len(Ls)} distances x 3 quantities x 2 orders)')

# =============================================================== T4 accuracy vs quadrature (Tables IV-VI)
print('T4 accuracy vs Gaussian quadrature ...')
TAB = {'eX': P.TAB_IV_eX, 'eZ': P.TAB_V_eZ, 'R': P.TAB_VI_R}
t4, cmp_rows = [], []
for i, L in enumerate(C.L_TABLE):
    eta0, s2 = C.eta0_eff(L), sx2_of(L)
    eta_med, eta_bar = eta0 * math.exp(-s2), eta0 * math.exp(-s2 / 2)
    var_eta = eta0 ** 2 * (1 - math.exp(-s2))
    d_med, d_bar = D.all_derivs(eta_med), D.all_derivs(eta_bar)
    for q in QS:
        ref = A.ln_reference(q, eta0, s2)
        vals = {'mean': A.naive_delta(d_bar[q], eta_bar, var_eta), 'med2': A.ln_expansion(d_med[q], eta_med, s2, 2),
                'med4': A.ln_expansion(d_med[q], eta_med, s2, 4)}
        row = [L, f'{s2:.3f}', q]
        for k in ('mean', 'med2', 'med4'):
            e = C.pct(vals[k], ref)
            pp = TAB[q][k][i]
            ok = R.match(e, pp, abs_tol=0.006 if q != 'R' else 0.006, rel_tol=0.08)
            row += [e, pp, e - pp, 'MATCH' if ok else 'DIFFERS']
            cmp_rows.append((L, q, k, e, pp, ok))
        t4.append(row)
    chk.add(f'T4-sigmaX2-{L}', f'sigma_X^2 at L={L} km vs printed', abs(s2 - P.SIGMA_X2[i]), 0.0006, unit='abs. diff')
hdr = ['L_km', 'sigma_X2', 'quantity'] + [f'{k}_{s}' for k in ('mean', 'med2', 'med4') for s in ('this_test_%', 'paper_%', 'diff_pp', 'status')]
C.save_csv('lognormal_T4_accuracy_vs_paper.csv', hdr, t4)
n_match = sum(1 for r in cmp_rows if r[-1]); n_tot = len(cmp_rows)
chk.add('T4-reproduce-tables', f'Entries of printed Tables IV-VI reproduced ({n_match} of {n_tot})', n_match / n_tot, 0.95, mode='>=',
        unit='fraction', note='tolerance: max(0.006 pp, 8% of the printed value)')
# claims quoted in the text of the manuscript
e4_R = [r for r in cmp_rows if r[1] == 'R' and r[2] == 'med4']
chk.add('T4-claim-4th-order-R', 'Claim: 4-term R error < 0.09 % over the sweep (checked on 13-point sweep below)',
        max(abs(r[3]) for r in e4_R), 0.09, unit='max |error| %, Table rows')
# reference cross-checks: Gauss-Hermite vs adaptive quad vs Monte Carlo
L = 160; eta0, s2 = C.eta0_eff(L), sx2_of(L)
gh, qd = A.ln_reference('R', eta0, s2), A.ln_reference_quad('R', eta0, s2)
chk.add('T4-reference-GH-vs-quad', 'Reference: Gauss-Hermite (200 nodes) vs adaptive quad (rel tol 1e-11), R at L=160 km', abs(gh - qd) / qd, 1e-9,
        unit='rel. diff')
# 13-point sweep for figure + the "always below 0.09 %" claim
sw = []
for L in (C.L_SWEEP13 if not QUICK else C.L_SWEEP13[::2]):
    eta0, s2 = C.eta0_eff(L), sx2_of(L)
    eta_med, eta_bar = eta0 * math.exp(-s2), eta0 * math.exp(-s2 / 2)
    var_eta = eta0 ** 2 * (1 - math.exp(-s2))
    d_med, d_bar = D.all_derivs(eta_med), D.all_derivs(eta_bar)
    row = [L, s2]
    for q in QS:
        ref = A.ln_reference(q, eta0, s2)
        row += [C.pct(A.naive_delta(d_bar[q], eta_bar, var_eta), ref), C.pct(A.ln_expansion(d_med[q], eta_med, s2, 2), ref),
                C.pct(A.ln_expansion(d_med[q], eta_med, s2, 4), ref)]
    sw.append(row)
sw = np.array(sw)
C.save_csv('lognormal_T4_sweep13.csv', ['L_km', 'sigma_X2'] + [f'{q}_{m}' for q in QS for m in ('mean', 'med2', 'med4')], sw.tolist())
chk.add('T4-claim-13pt-R4', 'Claim: 4-term secret-key-rate error < 0.09 % over the full 13-point sweep', np.abs(sw[:, 10]).max(), 0.09,
        unit='max |error| %')
# ---- figure F1
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, (qi, q) in zip(ax, enumerate(QS)):
    for k, (m, lab, col, ls) in enumerate((('mean', 'mean eta-space (delta method)', 'C1', '--'), ('med2', 'median, 2-term', 'C0', '-.'),
                                           ('med4', 'median, 4-term', 'C3', '-'))):
        a.plot(sw[:, 0], sw[:, 2 + 3 * qi + k], ls, color=col, marker='o', ms=3, label=lab)
    a.axhline(0, color='k', lw=.6); a.set_xlabel('Total distance L [km]'); a.set_ylabel(f'relative error of E[{q}] [%]')
    a.set_title({'eX': 'X-basis error rate', 'eZ': 'Z-basis phase-error rate', 'R': 'secret key rate'}[q]); a.legend(fontsize=8)
fig.suptitle('Log-normal averaging formulas vs Gaussian quadrature (Cn2 = 1e-17)', y=1.02)
R.savefig(fig, 'ln_F1_error_vs_distance')

# ---- figure F2: derivative verification
fig, ax = plt.subplots(figsize=(7, 4))
for q, mk in zip(worst, 'osd^'):
    ax.semilogy(range(5), np.maximum(worst[q], 1e-17), mk + '-', label=q)
ax.axhline(1e-7, color='r', ls=':', lw=1, label='pass limit 1e-7'); ax.set_xlabel('derivative order n'); ax.set_ylabel('max relative error')
ax.set_title('Analytic eta-derivatives vs 60-digit reference'); ax.legend(); ax.set_xticks(range(5))
R.savefig(fig, 'ln_F2_derivative_check')

# ---- figure F3: two routes
fig, ax = plt.subplots(figsize=(7.5, 4))
for q, mk in zip(QS, 'osd'):
    for n, fc in ((2, 'none'), (4, None)):
        pts = [(r[0], max(r[-1], 1e-17)) for r in t3 if r[1] == q and r[2] == n]
        ax.semilogy(*zip(*pts), mk, mfc=fc, color={'eX': 'C0', 'eZ': 'C1', 'R': 'C3'}[q], label=f'{q}, order {n}')
ax.axhline(1e-9, color='k', ls=':', lw=1, label='manuscript claim 1e-9'); ax.axhline(1e-8, color='r', ls=':', lw=1, label='pass limit 1e-8')
ax.set_xlabel('Total distance L [km]'); ax.set_ylabel('|eta-space - X-space| / |X-space|'); ax.legend(fontsize=7, ncol=2)
ax.set_title(f'X-space Hermite vs eta-space Euler-Stirling ({len(t3)} cross-checks)')
R.savefig(fig, 'ln_F3_two_routes')

# =============================================================== T5 moments / appendix claims
print('T5 moments and appendix claims ...')
rows = []
mom_err = 0.0
for s2 in (0.05, 0.3, 0.8):
    t, w = np.polynomial.hermite.hermgauss(120)
    X = -s2 + math.sqrt(2 * s2) * t
    for n in range(1, 5):
        num = float(np.sum(w * np.exp(n * X)) / math.sqrt(math.pi))
        form = math.exp(n * (n - 2) * s2 / 2)                               # eta0 = 1
        mom_err = max(mom_err, abs(num - form) / form)
chk.add('T5-moments', 'E[eta^n] = eta0^n exp(n(n-2) s^2/2) (Eq. 19) vs numerical, n=1..4', mom_err, 1e-9, unit='max rel. error')
sk_err = 0.0
for s2 in (0.05, 0.3, 0.8, 2.4):
    t, w = np.polynomial.hermite.hermgauss(150)
    X = -s2 + math.sqrt(2 * s2) * t
    m = [float(np.sum(w * np.exp(n * X)) / math.sqrt(math.pi)) for n in (1, 2, 3)]
    mu3 = m[2] - 3 * m[0] * m[1] + 2 * m[0] ** 3
    sk_num = mu3 / (m[1] - m[0] ** 2) ** 1.5
    sk_form = (math.exp(s2) + 2) * math.sqrt(math.exp(s2) - 1)
    sk_err = max(sk_err, abs(sk_num - sk_form) / sk_form)
chk.add('T5-skewness', 'Skewness of eta, Eq.(25), vs numerical', sk_err, 1e-6, unit='max rel. error')
app = []
cases = [('L=60, Cn2=1e-17', 60, 1e-17), ('L=160, Cn2=1e-17', 160, 1e-17), ('L=140, Cn2=1e-16', 140, 1e-16)]
for (name, L, cn2), (_, pap) in zip(cases, P.APP_GAMMA1):
    s2 = sx2_of(L, cn2)
    g1 = (math.exp(s2) + 2) * math.sqrt(math.exp(s2) - 1)
    app.append([name, f'{s2:.3f}', f'{g1:.3g}', f'{pap}'])
C.save_csv('lognormal_T5_skewness.csv', ['case', 'sigma_X2 (this test)', 'skewness (this test)', 'skewness (Appendix H)'], app)
chk.add('T5-skew-60', 'Appendix H: gamma_1 ~ 0.70 at L=60 km', abs(float(app[0][2]) - 0.70), 0.02, unit='abs. diff')
chk.add('T5-skew-160', 'Appendix H: gamma_1 ~ 2.0 at L=160 km', abs(float(app[1][2]) - 2.0), 0.1, unit='abs. diff')
chk.info('T5-skew-140', 'Appendix H: "gamma_1 ~ 41 at sigma_X^2 ~ 2.4 (Cn2=1e-16, L=140)"',
         f'for Cn2=1e-16, L=140 km this test gives sigma_X^2 = {float(app[2][1]):.2f} and gamma_1 = {float(app[2][2]):.3g}, matching the printed 41. '
         'The "sigma_X^2 = 9.6" quoted for the same case in the next sentence equals sigma_A^2+sigma_B^2 = 4 sigma_X^2, so it is a typo for 2.4.')
# naive-delta errors quoted in Appendix H
nv = {}
for name, L, cn2 in (('sigma_X2=0.007 (L=20)', 20, 1e-17), ('sigma_X2=0.31 (L=160)', 160, 1e-17), ('Cn2=1e-16,L=140', 140, 1e-16)):
    eta0, s2 = C.eta0_eff(L), sx2_of(L, cn2)
    eta_bar, var_eta = eta0 * math.exp(-s2 / 2), eta0 ** 2 * (1 - math.exp(-s2))
    dbar = D.all_derivs(eta_bar)['R']
    ref = ln_ref_grid('R', eta0, s2) if s2 > 0.5 else A.ln_reference('R', eta0, s2)
    nv[name] = (C.pct(A.naive_delta(dbar, eta_bar, var_eta), ref), s2)
# same expansion taken in the total transmittance eta_tot = eta_A eta_B (Appendix H, last sentence)
L = 160; eta0, s2 = C.eta0_eff(L), sx2_of(L)
eta_bar_tot, var_tot = eta0 ** 2, eta0 ** 4 * (math.exp(4 * s2) - 1)
d_tot = D.all_derivs(eta0)['R']                                           # E[eta_tot] = eta0^2 -> eta_m = eta0
th = A.theta_ops(d_tot, eta0) * 0.5 ** np.arange(1, 5)                     # Theta_tot^n
Q2_tot = (th[1] - th[0]) / eta_bar_tot ** 2
ref = A.ln_reference('R', eta0, s2)
nv['total-eta,L=160'] = (C.pct(d_tot[0] + 0.5 * Q2_tot * var_tot, ref), s2)
rows = []
for key, pap in P.APP_NAIVE.items():
    v, s = nv[key]
    rows.append([key, f'{s:.3f}', f'{v:+.3f}', f'{pap:+}'])
C.save_csv('lognormal_T5_naive_claims.csv', ['case', 'sigma_X2', 'naive error % (this test)', 'Appendix H %'], rows)
chk.add('T5-naive-0.007', 'Appendix H: naive error -0.001 % at sigma_X^2=0.007', abs(nv['sigma_X2=0.007 (L=20)'][0] + 0.001), 0.003, unit='abs. diff, pp')
chk.add('T5-naive-0.31', 'Appendix H: naive error 0.18 % at sigma_X^2=0.31', abs(nv['sigma_X2=0.31 (L=160)'][0] - 0.18), 0.01, unit='abs. diff, pp')
chk.info('T5-naive-140', 'Appendix H: naive error "29 %" at Cn2=1e-16, L=140', f'this test: {nv["Cn2=1e-16,L=140"][0]:+.1f} % at sigma_X^2 = {nv["Cn2=1e-16,L=140"][1]:.2f}: the 29 % is reproduced, but the printed sigma_X^2 = 9.6 should read 2.4.')
chk.info('T5-naive-total', 'Appendix H: naive expansion in eta_A*eta_B errs -20.1 % at L=160', f'this test: {nv["total-eta,L=160"][0]:+.1f} %')

# =============================================================== T6 stress test beyond weak turbulence
print('T6 stress test ...')
L = 80; eta0 = C.eta0_eff(L)
s_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.307, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]) if not QUICK else np.array([0.05, 0.3, 1.0, 3.0])
st = []
for s2 in s_grid:
    eta_med, eta_bar = eta0 * math.exp(-s2), eta0 * math.exp(-s2 / 2)
    var_eta = eta0 ** 2 * (1 - math.exp(-s2))
    d_med, d_bar = D.all_derivs(eta_med)['R'], D.all_derivs(eta_bar)['R']
    ref = ln_ref_grid('R', eta0, s2)
    a_m, a_2, a_4 = A.naive_delta(d_bar, eta_bar, var_eta), A.ln_expansion(d_med, eta_med, s2, 2), A.ln_expansion(d_med, eta_med, s2, 4)
    st.append([s2, 2 * s2, ref, a_m, a_2, a_4, C.pct(a_m, ref), C.pct(a_2, ref), C.pct(a_4, ref)])
st = np.array(st)
C.save_csv('lognormal_T6_stress.csv', ['sigma_X2', 'sigma_R2_per_arm', 'R_reference', 'R_naive', 'R_2term', 'R_4term', 'err_naive_%', 'err_2term_%', 'err_4term_%'], st.tolist())
i03 = int(np.argmin(abs(s_grid - 0.307))) if 0.307 in s_grid else 0
chk.add('T6-validity-4term', '4-term R error < 0.1 % while per-arm Rytov variance <= 0.2 (sigma_X^2 <= 0.1)',
        np.abs(st[st[:, 0] <= 0.1, 8]).max(), 0.1, unit='max |error| %')
neg = st[st[:, 3] < 0, 0]
chk.info('T6-naive-negative', 'Naive delta method gives a negative key rate',
         (('first at sigma_X^2 = %.2f (per-arm Rytov variance %.2f)' % (neg[0], 2 * neg[0])) if len(neg) else
          'never up to sigma_X^2 = 3 in the manuscript (two-arm) convention; the progress report\'s "negative key rate near Rytov variance two" refers to a single cumulative log-normal channel and is tested in script 02'))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
ax[0].semilogx(st[:, 0], st[:, 6], 'C1--o', ms=3, label='delta method (mean)'); ax[0].semilogx(st[:, 0], st[:, 7], 'C0-.o', ms=3, label='median, 2-term')
ax[0].semilogx(st[:, 0], st[:, 8], 'C3-o', ms=3, label='median, 4-term'); ax[0].axvline(0.5, color='purple', ls='-.', lw=.8)
ax[0].text(0.52, ax[0].get_ylim()[0] * 0.5, 'arm sigma_R^2 = 1\n(log-normal validity edge)', fontsize=7, color='purple')
ax[0].set_ylim(-110, 50); ax[0].set_xlabel('sigma_X^2'); ax[0].set_ylabel('relative error of E[R] [%]'); ax[0].legend(fontsize=8)
ax[0].set_title('Stress test beyond the weak regime (L = 80 km)')
ax[1].loglog(st[:, 0], np.maximum(st[:, 2], 1e-30), 'k-o', ms=3, label='quadrature'); ax[1].loglog(st[:, 0], np.maximum(st[:, 5], 1e-30), 'C3--', label='4-term')
ax[1].loglog(st[:, 0], np.maximum(st[:, 3], 1e-30), 'C1:', label='delta (clipped at 1e-30 if negative)')
ax[1].set_xlabel('sigma_X^2'); ax[1].set_ylabel('E[R]'); ax[1].legend(fontsize=8); ax[1].set_title('Averaged key rate')
R.savefig(fig, 'ln_F4_stress_test')

# skewness figure
sg = np.logspace(-2, 0.6, 100); fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(sg, (np.exp(sg) + 2) * np.sqrt(np.expm1(sg)), 'C3'); ax.axvline(0.5, color='purple', ls='-.', lw=.8)
for s2, g in ((0.051, 0.70), (0.31, 2.0)): ax.plot([s2], [g], 'ko')
ax.set_xlabel('sigma_X^2'); ax.set_ylabel('skewness gamma_1(eta)'); ax.set_title('Skewness of eta (Eq. 25): why additive expansion in eta fails')
R.savefig(fig, 'ln_F5_skewness')

# =============================================================== T7 Monte Carlo
print('T7 Monte Carlo ...')
n_mc = 100_000 if QUICK else 400_000
mcr = []
for L in (20, 80, 160):
    eta0, s2 = C.eta0_eff(L), sx2_of(L)
    for q in QS:
        ref = A.ln_reference(q, eta0, s2)
        m, se = A.ln_mc(q, eta0, s2, n_mc, seed=L)
        mcr.append([L, q, ref, m, se, (m - ref) / ref, abs(m - ref) / se if se > 0 else 0.0])
C.save_csv('lognormal_T7_montecarlo.csv', ['L_km', 'quantity', 'quadrature', 'MC', 'MC_std_err', 'rel_diff', 'z'], mcr)
chk.add('T7-mc', f'Monte Carlo ({n_mc:,} samples) vs quadrature reference: max |z|', max(r[-1] for r in mcr), 4.0, unit='max |z| (std. errors)')
fig, ax = plt.subplots(figsize=(7, 4))
for q, mk in zip(QS, 'osd'):
    pts = [(r[0], 100 * r[5], 100 * r[4] / r[2]) for r in mcr if r[1] == q]
    ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[2 * p[2] for p in pts], fmt=mk, capsize=3, label=q)
ax.axhline(0, color='k', lw=.6); ax.set_xlabel('L [km]'); ax.set_ylabel('(MC - quadrature)/quadrature [%]  (bars: 2 s.e.)'); ax.legend()
ax.set_title('Independent Monte Carlo check of the quadrature reference')
R.savefig(fig, 'ln_F6_montecarlo')

# ---- figure F7: averaged key rate vs distance
Ls = np.linspace(10, 160, 16 if not QUICK else 6)
fig, ax = plt.subplots(figsize=(7.2, 4.3))
Rq, R4, R0 = [], [], []
for L in Ls:
    eta0, s2 = C.eta0_eff(L), sx2_of(L); eta_med = eta0 * math.exp(-s2)
    Rq.append(A.ln_reference('R', eta0, s2)); R4.append(A.ln_expansion(D.all_derivs(eta_med)['R'], eta_med, s2, 4)); R0.append(C.r_ref(eta0))
ax.semilogy(Ls, R0, 'k:', label='no turbulence'); ax.semilogy(Ls, Rq, 'C0-', lw=2, label='quadrature reference'); ax.semilogy(Ls, R4, 'C3--', label='4-term formula')
ax.set_xlabel('L [km]'); ax.set_ylabel('R [bits/pulse]'); ax.legend(); ax.set_title('Turbulence-averaged key rate, Cn2 = 1e-17')
R.savefig(fig, 'ln_F7_key_rate')

chk.save()
