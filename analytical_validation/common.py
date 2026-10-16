# -*- coding: utf-8 -*-
"""
Shared setup for the analytical-formula validation scripts.

Conventions (identical to the manuscript, Sec. II-III):
  eta   = sqrt(eta_A * eta_B)   ("effective" transmittance; gamma = alpha^2 * eta)
  The validated engine in src/legacy_baseline.py takes the TOTAL transmittance
  eta_tot = eta_A*eta_B, so   eta_tot = eta**2.
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import legacy_baseline as CAL

HERE = os.path.dirname(os.path.abspath(__file__))
TAB_DIR = os.path.join(HERE, 'results', 'tables')
FIG_DIR = os.path.join(HERE, 'results', 'figures')
os.makedirs(TAB_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

# ---- Table III operating point (manuscript) ---------------------------------
ALPHA = 0.2
PD = 1e-7
THETA_A = math.asin(math.sqrt(0.02)); THETA_B = -THETA_A
THETA = THETA_A - THETA_B
NMAX = 4
PHI = 0.0
OMEGA = math.cos(PHI) * math.cos(THETA)          # = 0.96
A_NODARK = 1.0 - PD
CN2_WEAK = 1e-17
WAVELENGTH, W0, RX_RADIUS, ALPHA_ATM = 1550e-9, 0.05, 0.15, 0.046
K_WAVE = 2 * math.pi / WAVELENGTH
L_TABLE = [20, 50, 80, 100, 120, 160]            # km, rows of the printed tables
L_SWEEP13 = [float(x) for x in np.linspace(20, 160, 13)]   # 13-point sweep


def eta_fso(L_km):
    """Deterministic total Alice-Bob transmittance eta_FSO(L) (manuscript Eq. 14)."""
    return float(CAL.deterministic_fso_channel(ALPHA_ATM, L_km, WAVELENGTH, W0, RX_RADIUS)['eta_fso'])


def eta0_eff(L_km):
    """eta_0 = sqrt(eta_FSO): the effective operating point of the manuscript."""
    return math.sqrt(eta_fso(L_km))


def rytov(L_km, Cn2=CN2_WEAK):
    """Plane-wave Rytov variance sigma_R^2(L) = 1.23 Cn2 k^(7/6) L^(11/6)  (Eq. 15)."""
    return float(CAL.rytov_variance(Cn2, WAVELENGTH, L_km * 1e3))


# ---- CAL quantities from the validated engine (reference values) -------------
def pxx_ref(eta):
    return float(CAL.p_xx_total(PHI, THETA, ALPHA ** 2 * eta, PD))


def ex_ref(eta):
    return float(CAL.bit_error_rate(PHI, THETA, ALPHA ** 2 * eta, PD))


def ez_ref(eta):
    return float(CAL.phase_error_upper_bound(ALPHA, eta ** 2, PD, THETA_A, THETA_B, NMAX, PHI))


def r_ref(eta):
    return float(CAL.key_rate(ALPHA, eta ** 2, PD, THETA_A, THETA_B, NMAX, PHI))


def vec(fn):
    """Vectorise a scalar reference function over an array of eta."""
    return lambda e: np.array([fn(float(x)) for x in np.atleast_1d(e)])


QREF = {'pXX': pxx_ref, 'eX': ex_ref, 'eZ': ez_ref, 'R': r_ref}


def pct(a, ref):
    return 100.0 * (a - ref) / ref


def save_csv(name, header, rows):
    import csv
    p = os.path.join(TAB_DIR, name)
    with open(p, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    return p
