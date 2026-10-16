# -*- coding: utf-8 -*-
"""
example_02_adaptive_alpha.py
==============================
Instantaneous per-realization adaptive alpha vs. a single ergodically-
optimized fixed alpha vs. the legacy fixed alpha=0.20, under Gamma-Gamma
turbulence. Run: python3 examples/example_02_adaptive_alpha.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.turbulence import GammaGammaChannel
from src.tfqkd_engine import TFQKDEngine

PD = 1e-7
THETA_A = np.arcsin(np.sqrt(0.02))
THETA_B = -THETA_A

engine = TFQKDEngine(PD, THETA_A, THETA_B, Nmax=4)
ch = GammaGammaChannel(eta_nominal=0.05, Cn2=3e-15, wavelength=1550e-9, distance_m=6000, seed=0)

print(f"Channel: {ch.name}, sigma_R^2={ch.sigma_R2:.3f}, alpha={ch.alpha:.3f}, beta={ch.beta:.3f}")

# 1) legacy fixed alpha = 0.20
R_legacy = engine.ergodic_key_rate(ch, 0.20, method="quadrature")
print(f"\n1) Fixed alpha=0.20 (legacy default):        ergodic SKR = {R_legacy:.6e}")

# 2) single ergodically-optimized fixed alpha
alpha_opt, R_opt = engine.optimize_fixed_alpha(ch, method="quadrature")
print(f"2) Ergodically-optimized fixed alpha={alpha_opt:.4f}: ergodic SKR = {R_opt:.6e} "
      f"({100 * (R_opt / R_legacy - 1):+.2f}% vs legacy fixed)")

# 3) instantaneous adaptive alpha (per-realization, idealized upper bound)
result = engine.instantaneous_adaptive_analysis(ch, n_samples=20_000, alpha_fixed=0.20,
                                                  rng=np.random.default_rng(1))
print(f"3) Instantaneous adaptive alpha:               "
      f"mean SKR = {result['stats_adaptive']['mean']:.6e} "
      f"({100 * (result['stats_adaptive']['mean'] / R_legacy - 1):+.2f}% vs legacy fixed)")

corr = result['correlation']
print(f"\nCorr(alpha*, eta): Pearson r={corr['pearson_r']:.3f}, Spearman rho={corr['spearman_r']:.3f}")
print(f"Availability: adaptive={result['stats_adaptive']['availability']*100:.3f}%  "
      f"fixed={result['stats_fixed']['availability']*100:.3f}%")
