# -*- coding: utf-8 -*-
"""
TFQKD_MultiTurbulence.src
==========================
Twin-Field QKD over free-space optical channels with multiple
atmospheric turbulence models (deterministic, log-normal, Gamma-Gamma,
corrected Gamma-Gamma). The validated Curty-Azuma-Lo TF-QKD physics
lives unmodified in `legacy_baseline`; everything else in this package
is new, modular, and talks to that baseline only through
`tfqkd_engine.TFQKDEngine` and the `turbulence.TurbulenceChannel`
interface.
"""
__version__ = "2.4.0"
