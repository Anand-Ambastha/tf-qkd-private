import sys, os, warnings, csv
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEX = os.path.join(ROOT, 'paper_analysis', 'latex'); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True); sys.path.insert(0, ROOT)
warnings.filterwarnings('ignore')
import numpy as np
from src import legacy_baseline as CAL
from src.turbulence import create_channel, beam_broadening_factor, long_term_beam_radius
from src.tfqkd_engine import TFQKDEngine
PD, NMAX, AREF = 1e-7, 4, 0.20
TH = np.arcsin(np.sqrt(0.02))
LAM, W0, RX, AATM = 1550e-9, 0.05, 0.15, 0.046
CN2 = dict(CAL.Cn2_classes)
print(CN2)
eng = TFQKDEngine(PD, TH, -TH, Nmax=NMAX)
def eta_nom(L): return float(np.clip(CAL.deterministic_fso_channel(AATM, L, LAM, W0, RX)['eta_fso'], 1e-300, 1.0))
def build(model, cn2, L): return create_channel(model, eta_nominal=eta_nom(L), Cn2=cn2, wavelength=LAM, distance_m=L*1e3, w0=W0, receiver_radius=RX)
def erg(ch, n=300): return float(ch.average(lambda e: eng.instantaneous_key_rate(AREF, e), method='quadrature', quad_kwargs=dict(n_nodes=n)))
rows = []
for c, cn2 in CN2.items():
    for L in (20, 50, 80, 100, 120, 160):
        s2 = float(CAL.rytov_variance(cn2, LAM, 0.5*L*1e3))
        fb = beam_broadening_factor(cn2, LAM, 0.5*L*1e3, W0, RX)
        W, Wlt = long_term_beam_radius(cn2, LAM, 0.5*L*1e3, W0)
        det = float(eng.instantaneous_key_rate(AREF, np.array([eta_nom(L)]))[0])
        gg = erg(build('gamma_gamma', cn2, L)); bb = erg(build('gamma_gamma_bb', cn2, L))
        rows.append([c, cn2, L, s2, W, Wlt, fb**2, -10*np.log10(max(fb**2,1e-300)), det, gg, bb])
        print(rows[-1], flush=True)
with open(os.path.join(TAB, 'gg_beamwander_table.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['class','Cn2','L_km','sigmaR2_arm','W_m','W_LT_m','fbb2','bb_loss_dB','R_det','R_GG','R_GG_BB']); w.writerows(rows)
