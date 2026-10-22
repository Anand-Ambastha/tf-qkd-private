"""Per-arm Gamma-Gamma: analytic cumulant expansion vs exact reference for eX, eZ, R; with/without beam broadening."""
import sys, os, math, csv, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEX = os.path.join(ROOT, 'paper_analysis', 'latex'); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True); sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')
import numpy as np
from scipy import special as sp
from analytical_validation import common as C, cal_derivatives as D, averaging as A
from src.turbulence.beam_broadening import beam_broadening_factor
from src.turbulence.gamma_gamma import alpha_beta_plane_wave
QS = ('eX', 'eZ', 'R')
OUT = TAB

def exact_all(eta0, al, be, npts=6001):
    kap = A.log_gamma_cumulants(al, be); mu, sd = kap[0], math.sqrt(kap[1])
    y = np.linspace(mu - 16 * sd, mu + 12 * sd, npts); h = y[1] - y[0]
    p = A.gg_pdf_ln_i(y, al, be)
    pS = np.convolve(p, p) * h
    s = 2 * y[0] + h * np.arange(len(pS)); w = pS * h
    eta = np.minimum(eta0 * np.exp(0.5 * s), 1.0)
    return {q: float(np.sum(w * A.Q_vec(q, eta))) for q in QS}, float(w.sum()), float(np.sum(w * np.exp(s)))

def approx(eta0, al, be, q):
    kap = A.log_gamma_cumulants(al, be)
    c = np.array([2.0 ** (1 - n) * kap[n - 1] for n in (1, 2, 3, 4)])      # cumulants of X=(Y_A+Y_B)/2
    eta_m = eta0 * math.exp(c[0])
    d = D.all_derivs(eta_m)[q]; th = A.theta_ops(d, eta_m)
    o2 = d[0] + 0.5 * c[1] * th[1]; o3 = o2 + c[2] / 6 * th[2]; o4 = o3 + (c[3] + 3 * c[1] ** 2) / 24 * th[3]
    # mean, eta-space delta method (as manuscript Tables IV-VI): E[eta]=eta0*m, Var=eta0^2 (1-m^2), m=E[e^X]=(E sqrt I)^2
    m1 = math.exp(2 * (sp.gammaln(al + .5) + sp.gammaln(be + .5) - sp.gammaln(al) - sp.gammaln(be) - 0.5 * math.log(al * be)))
    eb = eta0 * m1; dbar = D.all_derivs(eb)[q]
    naive = A.naive_delta(dbar, eb, eta0 ** 2 * (1 - m1 ** 2))
    # mean, total-eta-space delta method (progress report, Table 1 convention)
    d0 = D.all_derivs(eta0)[q]; thT = A.theta_ops(d0, eta0) * 0.5 ** np.arange(1, 5)
    SI = A.gg_si(al, be); naive_tot = d0[0] + 0.5 * (thT[1] - thT[0]) * ((1 + SI) ** 2 - 1)
    return dict(naive=naive, naive_tot=naive_tot, o2=o2, o3=o3, o4=o4, c=c)

rows = []
for cname, cn2 in (('weak', 1e-17), ('moderate', 1e-15)):
    for L in (20, 50, 80, 100, 120, 160):
        s2 = C.rytov(L / 2.0, cn2); al, be = alpha_beta_plane_wave(s2)
        fb = beam_broadening_factor(cn2, C.WAVELENGTH, 0.5 * L * 1e3, C.W0, C.RX_RADIUS)
        for bb in (0, 1):
            eta0 = math.sqrt(C.eta_fso(L)) * (fb if bb else 1.0)
            ex, nrm, en = exact_all(eta0, al, be)
            for q in QS:
                try:
                    a = approx(eta0, al, be, q); err = {k: 100 * (a[k] - ex[q]) / ex[q] for k in ('naive', 'naive_tot', 'o2', 'o3', 'o4')}
                except Exception as e:
                    print('fail', cname, L, bb, q, e); err = {k: float('nan') for k in ('naive', 'naive_tot', 'o2', 'o3', 'o4')}
                kap = A.log_gamma_cumulants(al, be); c = [2.0 ** (1 - n) * kap[n - 1] for n in (1, 2, 3, 4)]
                rows.append([cname, cn2, L, bb, q, s2, al, be, fb ** 2 if bb else 1.0, eta0, ex[q], nrm, en, *c,
                             err['naive'], err['naive_tot'], err['o2'], err['o3'], err['o4']])
            print(cname, L, bb, f's2={s2:.3g} R={ex["R"]:.4e}', ' | '.join(f'{r[4]}: n={r[-5]:+.4f} 2={r[-3]:+.4f} 4={r[-1]:+.4f}' for r in rows[-3:]))
hdr = ['class', 'Cn2', 'L_km', 'BB', 'quantity', 'sigmaR2_arm', 'alpha', 'beta', 'fbb2', 'eta_eff0', 'exact', 'pdf_norm', 'E_I_tot', 'c1', 'c2', 'c3', 'c4',
       'err_mean_%', 'err_mean_tot_%', 'err_med2_%', 'err_med3_%', 'err_med4_%']
with open(os.path.join(OUT, 'perarm_analytic2.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
