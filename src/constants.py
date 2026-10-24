# -*- coding: utf-8 -*-
"""
constants.py
============
Shared physical constants and default operating parameters.

References
----------
[AP] Andrews, L. C. & Phillips, R. L., "Laser Beam Propagation through
     Random Media", 2nd ed., SPIE Press, 2005.
[CAL] Curty, Azuma & Lo, "Simple security proof of twin-field type QKD
      protocol", npj Quantum Information 5, 64 (2019); arXiv:1807.07667v2.
"""

# Rytov-variance regime boundaries used by the AUTO model selector
# (Andrews & Phillips, Ch. 9): conventional weak / moderate / strong
# boundaries. Near sigma_R^2 ~ 25 the channel approaches saturation, where
# the corrected (aperture-averaged) Gamma-Gamma model is preferred for a
# finite receiver.
SIGMA_R2_WEAK_MODERATE = 1.0
SIGMA_R2_MODERATE_STRONG = 10.0

# Default TF-QKD security-proof parameters (Curty-Azuma-Lo), identical to
# those used in src/legacy_baseline.py.
import numpy as np

DEFAULT_TFQKD_PARAMS = dict(
    pd=1e-7,
    alpha=0.2,
    thetaA=np.arcsin(np.sqrt(0.02)),
    thetaB=-np.arcsin(np.sqrt(0.02)),
    Nmax=4,
)

# Default FSO link geometry (used by deterministic + turbulence channels)
DEFAULT_FSO_PARAMS = dict(
    alpha_atm=0.046,        # atmospheric attenuation coefficient [1/km], clear-air
    wavelength=1550e-9,   # m
    w0=0.05,              # transmit beam waist, m
    receiver_radius=0.15, # receiver aperture radius, m
)

# Default Monte Carlo settings
DEFAULT_MC_SEED = 42
DEFAULT_MC_SAMPLES = 50_000
MC_SAMPLE_GRID = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

# Numerical safety floors
EPS = 1e-300
