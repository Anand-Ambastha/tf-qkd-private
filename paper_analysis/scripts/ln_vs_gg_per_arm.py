"""Per-arm Gamma-Gamma: analytic cumulant expansion vs exact (convolution) reference, with/without beam broadening.
Uses repo code (cal_derivatives, averaging, beam_broadening) unchanged; only the arm->effective-channel bookkeeping is new."""
import sys, os, math, csv, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEX = os.path.join(ROOT, 'paper_analysis', 'latex'); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True); sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')
import numpy as np
from analytical_validation import common as C, cal_derivatives as D, averaging as A
from src.turbulence.beam_broadening import beam_broadening_factor
from src.turbulence.gamma_gamma import alpha_beta_plane_wave

def arm(cn2, L):
    s2 = C.rytov(L / 2.0, cn2)
    al, be = alpha_beta_plane_wave(s2)
    return s2, al, be

def exact_R(eta0, al, be, q='R', npts=6001):
    """E[Q(eta0 * exp(S/2))], S = Y_A + Y_B, Y = ln(Xa Xb); S-density by trapezoid convolution of the Bessel-K log-density."""
    kap = A.log_gamma_cumulants(al, be); mu, sd = kap[0], math.sqrt(kap[1])
    y = np.linspace(mu - 16 * sd, mu + 12 * sd, npts); h = y[1] - y[0]
    p = A.gg_pdf_ln_i(y, al, be)
    pS = np.convolve(p, p) * h
    s = 2 * y[0] + h * np.arange(len(pS))
    w = pS * h
    eta = np.minimum(eta0 * np.exp(0.5 * s), 1.0)
    return float(np.sum(w * A.Q_vec(q, eta))), float(w.sum()), float(np.sum(w * np.exp(s)))

def analytic(eta0, al, be, q='R'):
    """X = (Y_A+Y_B)/2 ; cumulants c_n = 2^(1-n) kappa_n(arm); expansion about eta_m = eta0 exp(c1)."""
    kap = A.log_gamma_cumulants(al, be)
    c = np.array([2.0 ** (1 - n) * kap[n - 1] for n in (1, 2, 3, 4)])
    eta_m = eta0 * math.exp(c[0])
    d = D.all_derivs(eta_m)[q]; th = A.theta_ops(d, eta_m)
    o2 = d[0] + 0.5 * c[1] * th[1]
    o3 = o2 + c[2] / 6 * th[2]
    o4 = o3 + (c[3] + 3 * c[1] ** 2) / 24 * th[3]
    # delta method in eta_tot about the mean (as in 02_test_gammagamma): eta_tot0 = eta0^2, Var = eta_tot0^2 SI_tot
    d0 = D.all_derivs(eta0)[q]; thT = A.theta_ops(d0, eta0) * 0.5 ** np.arange(1, 5)
    SI = A.gg_si(al, be); SItot = (1 + SI) ** 2 - 1
    naive = d0[0] + 0.5 * (thT[1] - thT[0]) * SItot
    return dict(naive=naive, o2=o2, o3=o3, o4=o4, c=c, eta_m=eta_m, SI=SI)

# ---------- self-consistency: X-variable formula == repo's Y-variable gg_expansion (single cumulative channel)
chk = 0.0
for L in (50, 100, 160):
    s = C.rytov(L); al, be = A.gg_params(s, A.aperture_d2(L)); kap = A.log_gamma_cumulants(al, be)
    e0 = math.sqrt(C.eta_fso(L)); eta_pt = e0 * math.exp(0.5 * kap[0]); d = D.all_derivs(eta_pt)['R']
    ref = A.gg_expansion(d, eta_pt, kap, 0.0, 4)
    c = np.array([2.0 ** -n * kap[n - 1] for n in (1, 2, 3, 4)]); th = A.theta_ops(d, eta_pt)
    mine = d[0] + 0.5 * c[1] * th[1] + c[2] / 6 * th[2] + (c[3] + 3 * c[1] ** 2) / 24 * th[3]
    chk = max(chk, abs(mine - ref) / abs(ref))
print('X-variable vs repo Y-variable 4th order, max rel diff:', chk)

rows = []
for cname, cn2 in (('weak', 1e-17), ('moderate', 1e-15)):
    for L in (20, 50, 80, 100, 120, 160):
        s2, al, be = arm(cn2, L)
        fb = beam_broadening_factor(cn2, C.WAVELENGTH, 0.5 * L * 1e3, C.W0, C.RX_RADIUS)
        for bb in (False, True):
            eta0 = math.sqrt(C.eta_fso(L)) * (fb if bb else 1.0)
            try:
                ex, nrm, en = exact_R(eta0, al, be)
                a = analytic(eta0, al, be)
                err = {k: 100 * (a[k] - ex) / ex for k in ('naive', 'o2', 'o3', 'o4')}
            except Exception as e:
                ex = nrm = en = float('nan'); err = {k: float('nan') for k in ('naive', 'o2', 'o3', 'o4')}; a = dict(SI=float('nan'))
                print('fail', cname, L, bb, e)
            rows.append([cname, cn2, L, int(bb), s2, al, be, a['SI'], fb ** 2 if bb else 1.0, eta0, ex, nrm, en,
                         err['naive'], err['o2'], err['o3'], err['o4']])
            print(cname, L, 'BB' if bb else '--', f's2={s2:.3g} a={al:.3g} b={be:.3g} fbb2={fb**2:.3g} R={ex:.4e} norm={nrm:.6f} E[I]={en:.6f}',
                  ' '.join(f'{k}={v:+.4f}' for k, v in err.items()))
hdr = ['class', 'Cn2', 'L_km', 'BB', 'sigmaR2_arm', 'alpha', 'beta', 'SI_arm', 'fbb2', 'eta_eff0', 'R_exact', 'pdf_norm', 'E_I_tot', 'err_naive_%', 'err_o2_%', 'err_o3_%', 'err_o4_%']
with open(os.path.join(TAB, 'perarm_analytic.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

# ---------- LN vs GG per arm at fixed per-arm Rytov variance
rows2 = []
for cname, cn2 in (('weak', 1e-17), ('moderate', 1e-15), ('strong', 1e-13)):
    s1 = 1.23 * cn2 * C.K_WAVE ** (7 / 6)
    for s in (0.25, 0.5, 1.0, 2.0, 4.0):
        L = 2.0 * (s / s1) ** (6 / 11) / 1e3
        eta0 = math.sqrt(C.eta_fso(L)); _, al, be = arm(cn2, L)
        gg, nrm, en = exact_R(eta0, al, be)
        sx2 = s / 2.0
        mu, sd = -sx2, math.sqrt(sx2)
        X = np.linspace(mu - 12 * sd, mu + 12 * sd, 40001)
        wX = np.exp(-0.5 * ((X - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        from scipy.integrate import simpson
        ln = float(simpson(wX * A.Q_vec('R', np.minimum(eta0 * np.exp(X), 1.0)), x=X))
        det = float(C.r_ref(eta0))
        rows2.append([cname, cn2, s, L, SI := A.gg_si(al, be), math.expm1(s), det, gg, ln, 100 * (ln - gg) / gg])
        print(cname, f's={s} L={L:.2f} SI_GG={rows2[-1][4]:.3f} SI_LN={math.expm1(s):.3f} Rdet={det:.3e} R_GG={gg:.4e} R_LN={ln:.4e} err={rows2[-1][-1]:+.1f}%')
with open(os.path.join(TAB, 'ln_vs_gg.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['class', 'Cn2', 'sigmaR2_arm', 'L_km', 'SI_GG', 'SI_LN', 'R_det', 'R_GG', 'R_LN', 'LN_vs_GG_%']); w.writerows(rows2)
