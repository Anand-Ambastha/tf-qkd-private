# -*- coding: utf-8 -*-
"""
example_03_distance_sweep_and_validation.py
==============================================
(a) Ergodic SKR vs. distance for log-normal, Gamma-Gamma, and corrected
    Gamma-Gamma turbulence at a fixed Cn2, saved as a figure.
(b) Full validation report (PDF normalization, CDF checks, MC-vs-
    quadrature agreement) printed as a markdown table -- the same checks
    used in tests/ and documented in docs/VALIDATION.md.
Run: python3 examples/example_03_distance_sweep_and_validation.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.turbulence import create_channel
from src.tfqkd_engine import TFQKDEngine
from src.plotting import plot_distance_sweep
from src.validation import full_validation_report

PD = 1e-7
THETA_A = np.arcsin(np.sqrt(0.02))
THETA_B = -THETA_A
WAVELENGTH = 1550e-9
CN2 = 2e-15

engine = TFQKDEngine(PD, THETA_A, THETA_B, Nmax=4)
distances_km = np.arange(1, 21, 1)


def eta_nominal_at(L_km, alpha_atm=0.15):
    return float(np.clip(np.exp(-alpha_atm * L_km) * 0.3, 1e-6, 1.0))


def lognormal_factory(L_km):
    return create_channel('lognormal', eta_nominal=eta_nominal_at(L_km), Cn2=CN2,
                          wavelength=WAVELENGTH, distance_m=L_km * 1e3)   # two independent arms


def gg_factory(L_km):
    return create_channel('gamma_gamma', eta_nominal=eta_nominal_at(L_km), Cn2=CN2,
                          wavelength=WAVELENGTH, distance_m=L_km * 1e3)


def cgg_factory(L_km):
    return create_channel('gamma_gamma_corrected', eta_nominal=eta_nominal_at(L_km), Cn2=CN2,
                          wavelength=WAVELENGTH, distance_m=L_km * 1e3,
                          receiver_radius=0.15, jitter_std=1e-6, beam_wander_std=5e-7)


print("Running distance sweep (this evaluates the validated key-rate formula "
      "at every distance x model combination)...")
results = {}
for name, factory in [("Log-Normal", lognormal_factory), ("Gamma-Gamma", gg_factory),
                       ("Corrected Gamma-Gamma", cgg_factory)]:
    results[name] = engine.distance_sweep(distances_km, factory, alpha=0.20, method="quadrature")

fig = plot_distance_sweep(distances_km, results,
                           out_dir=os.path.join(os.path.dirname(__file__), '..', 'figures'),
                           name='example03_distance_sweep')
print("Figure saved to figures/example03_distance_sweep.{png,pdf}\n")

print("=" * 78)
print("Validation report (see docs/VALIDATION.md for the full methodology)")
print("=" * 78)
channels = [
    ("lognormal", lognormal_factory(6.0)),
    ("gamma_gamma", gg_factory(6.0)),
    ("gamma_gamma_corrected", cgg_factory(6.0)),
]
rows = full_validation_report(engine, channels, seed=0)
header = ("model", "pdf_integral", "cdf_endpoint", "mean_ok", "skr_rel_err(mc,quad)")
print(f"{header[0]:<24}{header[1]:<14}{header[2]:<14}{header[3]:<10}{header[4]:<10}")
for r in rows:
    print(f"{r['model']:<24}{r['pdf_integral']:<14.5f}{r['cdf_endpoint']:<14.5f}"
          f"{str(r['mean_ok']):<10}{r['skr_rel_err']:<10.4f}")
