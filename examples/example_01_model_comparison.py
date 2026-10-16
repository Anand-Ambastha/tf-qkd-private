# -*- coding: utf-8 -*-
"""
example_01_model_comparison.py
================================
Build all four turbulence models at matched operating points spanning
weak -> moderate -> strong scintillation, print their statistics, and
save a PDF comparison figure. Run: python3 examples/example_01_model_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.turbulence import (DeterministicChannel, LogNormalChannel, GammaGammaChannel,
                             CorrectedGammaGammaChannel, create_channel, classify_regime)
from src.tfqkd_engine import TFQKDEngine
from src.plotting import plot_model_comparison_pdf
from src import legacy_baseline as CAL

PD = 1e-7
THETA_A = np.arcsin(np.sqrt(0.02))
THETA_B = -THETA_A
WAVELENGTH = 1550e-9

engine = TFQKDEngine(PD, THETA_A, THETA_B, Nmax=4)

print("=" * 70)
print("Model 1: Deterministic FSO channel (no turbulence)")
print("=" * 70)
det = DeterministicChannel(alpha_atm=0.2, distance_km=5, wavelength=WAVELENGTH,
                            w0=0.05, receiver_radius=0.15)
print(f"eta_nominal = {det.eta_nominal:.4e}")
print(f"R(alpha=0.20) = {engine.ergodic_key_rate(det, 0.20):.6e} bits/pulse\n")

print("=" * 70)
print("Model 2/3/4 comparison at matched eta_nominal, increasing turbulence")
print("=" * 70)
eta_nom = 0.05
scenarios = [
    ("weak (Cn2=1e-17, L=3 km)", 1e-17, 3000),
    ("moderate (Cn2=3e-15, L=6 km)", 3e-15, 6000),
    ("strong (Cn2=1e-14, L=10 km)", 1e-14, 10_000),
]
channels_for_plot, labels_for_plot = [], []
for label, Cn2, L in scenarios:
    sigma_R2 = float(CAL.two_link_sigma_R2(Cn2, WAVELENGTH, L))   # per-arm (L/2)
    regime = classify_regime(sigma_R2)
    ch = create_channel('auto', eta_nominal=eta_nom, Cn2=Cn2, wavelength=WAVELENGTH,
                         distance_m=L, receiver_radius=0.15, jitter_std=1e-6,
                         beam_wander_std=5e-7)
    stats = ch.statistics()
    R = engine.ergodic_key_rate(ch, 0.20, method="quadrature")
    print(f"\n{label}: per-arm sigma_R^2={sigma_R2:.3f} -> regime='{regime}' -> AUTO picked '{ch.name}'")
    print(f"  mean(eta)={stats['mean']:.4e}  scintillation_index={stats['scintillation_index']:.4f}")
    print(f"  ergodic SKR = {R:.6e} bits/pulse")
    channels_for_plot.append(ch)
    labels_for_plot.append(f"{ch.name} ({label.split('(')[0].strip()})")

fig = plot_model_comparison_pdf(channels_for_plot, labels=labels_for_plot,
                                 out_dir=os.path.join(os.path.dirname(__file__), '..', 'figures'),
                                 name='example01_model_comparison_pdf')
print("\nFigure saved to figures/example01_model_comparison_pdf.{png,pdf}")
