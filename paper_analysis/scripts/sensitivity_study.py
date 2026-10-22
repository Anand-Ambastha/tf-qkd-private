"""Controlled sensitivity study: approximation error vs log-transmittance variance sigma_X^2 at fixed L.
Models: LN (X~N(-s,s)), GG per arm, GG per arm + beam broadening (eta0 -> eta0*f_bb).
sigma_X^2 is imposed; for GG the per-arm Rytov variance s is solved so that Var(X)=c2(s) equals the target (c2 <= 0.858)."""
import sys, os, math, csv, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEX = os.path.join(ROOT, 'paper_analysis', 'latex'); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True); sys.path.insert(0, ROOT); warnings.filterwarnings('ignore')
import numpy as np
from scipy import special as sp
from scipy.optimize import brentq
from scipy.integrate import simpson
from analytical_validation import common as C, cal_derivatives as D, averaging as A
from src.turbulence.gamma_gamma import alpha_beta_plane_wave
from src.turbulence import beam_broadening_factor
from src import legacy_baseline as CAL
OUT = TAB
QS = ('eX', 'eZ', 'R'); LS = (80, 120, 160)
GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SMAX = 51.79

def c2_of_s(s):
    a, b = alpha_beta_plane_wave(s); return 0.5 * (sp.polygamma(1, a) + sp.polygamma(1, b))
def s_of_c2(t): return brentq(lambda s: c2_of_s(s) - t, 1e-8, SMAX, xtol=1e-14, rtol=1e-13)

def fbb(s, L):
    z = 0.5 * L * 1e3; W = float(CAL.beam_radius(C.WAVELENGTH, z, C.W0)); Lam = 2 * z / (C.K_WAVE * W ** 2)
    Wl = W * math.sqrt(1 + 1.63 * s ** 1.2 * Lam); cap = lambda w: -math.expm1(-2 * C.RX_RADIUS ** 2 / w ** 2)
    return cap(Wl) / cap(W), W, Wl

def ref_ln(eta0, s2, q):
    mu, sd = -s2, math.sqrt(s2); X = np.linspace(mu - 12 * sd, mu + 12 * sd, 40001)
    w = np.exp(-0.5 * ((X - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)); eta = eta0 * np.exp(X)
    tail = float(simpson(w * (eta > 1), x=X))
    return float(simpson(w * A.Q_vec(q, np.minimum(eta, 1.0)), x=X)), tail

def ref_gg(eta0, al, be, q, npts=6001):
    kap = A.log_gamma_cumulants(al, be); mu, sd = kap[0], math.sqrt(kap[1])
    y = np.linspace(mu - 16 * sd, mu + 12 * sd, npts); h = y[1] - y[0]
    p = A.gg_pdf_ln_i(y, al, be); pS = np.convolve(p, p) * h; s = 2 * y[0] + h * np.arange(len(pS)); w = pS * h
    eta = eta0 * np.exp(0.5 * s)
    return float(np.sum(w * A.Q_vec(q, np.minimum(eta, 1.0)))), float(np.sum(w * (eta > 1))), float(w.sum())

def approx(eta0, c, m1, q):
    """c = cumulants c1..c4 of X; m1 = E[e^X]; E[e^{2X}] = 1 (energy neutrality). Returns (mean, 2nd, 4th), expansion-point R."""
    eta_m = eta0 * math.exp(c[0]); d = D.all_derivs(eta_m)[q]; th = A.theta_ops(d, eta_m)
    o2 = d[0] + 0.5 * c[1] * th[1]; o4 = o2 + c[2] / 6 * th[2] + (c[3] + 3 * c[1] ** 2) / 24 * th[3]
    eb = eta0 * m1; db = D.all_derivs(eb)[q]
    return A.naive_delta(db, eb, eta0 ** 2 * (1 - m1 ** 2)), o2, o4

rows = []
for L in LS:
    e0 = math.sqrt(C.eta_fso(L))
    for t in GRID:
        s = s_of_c2(t); al, be = alpha_beta_plane_wave(s); kap = A.log_gamma_cumulants(al, be)
        cG = np.array([2.0 ** (1 - n) * kap[n - 1] for n in (1, 2, 3, 4)])
        m1G = math.exp(2 * (sp.gammaln(al + .5) + sp.gammaln(be + .5) - sp.gammaln(al) - sp.gammaln(be) - 0.5 * math.log(al * be)))
        f, W, Wl = fbb(s, L)
        models = {'LN': (e0, np.array([-t, t, 0.0, 0.0]), math.exp(-t / 2), None),
                  'GG': (e0, cG, m1G, (al, be)), 'GGBB': (e0 * f, cG, m1G, (al, be))}
        for mname, (eta0, c, m1, ab) in models.items():
            for q in QS:
                try:
                    if mname == 'LN': ref, tail = ref_ln(eta0, t, q); nrm = 1.0
                    else: ref, tail, nrm = ref_gg(eta0, *ab, q)
                    on = float(A.Q_vec('R', np.array([eta0 * math.exp(c[0])]))[0])      # key rate at expansion point (branch check)
                    if not ref > 0: raise ValueError('reference <= 0')
                    ap = approx(eta0, c, m1, q); err = [100 * (v - ref) / ref for v in ap]
                except Exception as ex:
                    ref, tail, nrm, on, err = float('nan'), float('nan'), float('nan'), float('nan'), [float('nan')] * 3
                rows.append([mname, L, t, q, s if mname != 'LN' else t * 2, al if ab_ok else 0 if False else 0, 0, 0, 0] if False else
                            [mname, L, t, q, (2 * t if mname == 'LN' else s), (f ** 2 if mname == 'GGBB' else 1.0), eta0, ref, tail, nrm, on, *err])
        print(L, t, f's={s:.4g} fbb2={f**2:.3g}', flush=True)
hdr = ['model', 'L_km', 'sigmaX2', 'quantity', 'sigmaR2_arm', 'fbb2', 'eta0', 'reference', 'P_eta_gt_1', 'pdf_norm', 'R_at_expansion_pt', 'err_mean_%', 'err_2nd_%', 'err_4th_%']
with open(os.path.join(OUT, 'sensitivity.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
