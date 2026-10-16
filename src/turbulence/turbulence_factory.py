# -*- coding: utf-8 -*-
"""
turbulence_factory.py
======================
Single entry point for constructing any turbulence channel by name, plus
the AUTO regime selector:

    weak (sigma_R^2 < 1)                -> Log-Normal
    moderate (1 <= sigma_R^2 < 10)      -> Gamma-Gamma
    strong (sigma_R^2 >= 10)            -> Corrected Gamma-Gamma

The boundaries are the conventional ones used throughout the FSO
literature (Andrews & Phillips Ch. 9-10): sigma_R^2=1 is the accepted
weak/moderate boundary where the log-normal model's tail starts to
under-predict deep fades; sigma_R^2~10-25 is where finite-aperture
averaging and pointing-error effects become non-negligible relative to
the bare Gamma-Gamma prediction for a realistic (non-point) receiver, so
AUTO reaches for the corrected model there. Users can always override by
passing model= explicitly.

Two independent links (default)
-------------------------------
TF-QKD has two free-space arms, Alice->Charlie and Bob->Charlie, whose
turbulence is statistically independent. For every stochastic model,
create_channel(..., two_link=True) therefore returns a TwoLinkChannel whose
transmittance is eta = eta_A * eta_B with SEPARATE fading on each arm (each
arm: length L/2 and nominal sqrt(eta_nominal) unless distance_A_m /
distance_B_m / eta_A_nominal / eta_B_nominal say otherwise), and all averages
run over the two independent random variables. The AUTO regime selector is
applied per arm (Rytov variance at the arm length). Pass two_link=False to get
the old single-fading-factor-at-full-length channel (kept only for comparison).
"""
from __future__ import annotations

from .deterministic import DeterministicChannel
from .lognormal import LogNormalChannel
from .gamma_gamma import GammaGammaChannel
from .corrected_gamma_gamma import CorrectedGammaGammaChannel
from .two_link import TwoLinkChannel
from .beam_broadening import beam_broadening_factor, long_term_beam_radius
from .. import legacy_baseline as CAL
from ..constants import SIGMA_R2_WEAK_MODERATE, SIGMA_R2_MODERATE_STRONG

_MODELS = {
    'deterministic': DeterministicChannel,
    'lognormal': LogNormalChannel,
    'gamma_gamma': GammaGammaChannel,
    'gamma_gamma_corrected': CorrectedGammaGammaChannel,
}


def classify_regime(sigma_R2):
    """Return 'weak', 'moderate', or 'strong' given a Rytov variance."""
    if sigma_R2 < SIGMA_R2_WEAK_MODERATE:
        return 'weak'
    elif sigma_R2 < SIGMA_R2_MODERATE_STRONG:
        return 'moderate'
    return 'strong'


def _build_single(model, eta_nominal, Cn2, wavelength, distance_m, receiver_radius,
                  jitter_std, beam_wander_std, w0, wave_type, kwargs):
    """One stochastic FSO link of the given model."""
    if model == "lognormal":
        return LogNormalChannel(eta_nominal, Cn2, wavelength, distance_m, **kwargs)
    if model == "gamma_gamma":
        return GammaGammaChannel(eta_nominal, Cn2, wavelength, distance_m,
                                  wave_type=wave_type, **kwargs)
    if model == "gamma_gamma_corrected":
        if receiver_radius is None:
            raise ValueError("model='gamma_gamma_corrected' requires receiver_radius.")
        return CorrectedGammaGammaChannel(eta_nominal, Cn2, wavelength, distance_m,
                                           receiver_radius, jitter_std=jitter_std,
                                           beam_wander_std=beam_wander_std,
                                           w0=(w0 or 0.05), wave_type=wave_type, **kwargs)
    raise ValueError(f"unknown model '{model}'")


def create_channel(model, eta_nominal, Cn2=None, wavelength=None, distance_m=None,
                    distance_km=None, alpha_atm=None, w0=None, receiver_radius=None,
                    jitter_std=0.0, beam_wander_std=0.0, wave_type="plane",
                    two_link=True, distance_A_m=None, distance_B_m=None,
                    eta_A_nominal=None, eta_B_nominal=None, **kwargs):
    """Construct a turbulence channel by name.

    Parameters
    ----------
    model : {"deterministic","lognormal","gamma_gamma","gamma_gamma_corrected","auto"},
        or any stochastic model + "_bb" (e.g. "gamma_gamma_bb") = that fading model PLUS
        per-arm turbulence-induced long-term beam broadening (needs w0 and receiver_radius).
    eta_nominal : end-to-end deterministic (fading-free) transmittance --
        required for every stochastic model; ignored for "deterministic"
        (which derives its own eta from alpha_atm/distance_km/w0/receiver_radius).
    distance_m : TOTAL Alice-Bob path length [m]. With two_link=True each arm
        defaults to distance_m / 2 (Charlie midway).
    two_link : bool, default True. True -> TwoLinkChannel with independent
        fading on Alice-Charlie and Bob-Charlie; False -> legacy single fading
        factor at the full length (not physically appropriate for TF-QKD).
    distance_A_m, distance_B_m : optional arm lengths (override the L/2 split).
    eta_A_nominal, eta_B_nominal : optional arm transmittances; their product
        must equal eta_nominal (if only one is given the other is derived).
    """
    seed = kwargs.pop('seed', None)

    # ---- '<model>_bb': same fading model + turbulence-induced beam broadening per arm
    bb = isinstance(model, str) and model.endswith('_bb')
    if bb:
        model = model[:-3]
        if model not in ("lognormal", "gamma_gamma", "gamma_gamma_corrected"):
            raise ValueError("beam broadening is available for 'lognormal', 'gamma_gamma' and "
                             "'gamma_gamma_corrected' (use e.g. 'gamma_gamma_bb').")
        if not two_link:
            raise ValueError("'*_bb' models are per-arm; two_link=False is not supported.")
        if None in (Cn2, wavelength, distance_m, w0, receiver_radius):
            raise ValueError("'*_bb' models require Cn2, wavelength, distance_m, w0 and receiver_radius.")

    if model == "auto":
        if Cn2 is None or wavelength is None or distance_m is None:
            raise ValueError("model='auto' requires Cn2, wavelength, distance_m "
                              "to evaluate the Rytov variance.")
        if two_link:
            LA, LB = CAL.two_link_geometry(distance_m, distance_A_m, distance_B_m)
            sigma_R2 = max(float(CAL.rytov_variance(Cn2, wavelength, LA)),
                           float(CAL.rytov_variance(Cn2, wavelength, LB)))
        else:
            sigma_R2 = float(CAL.rytov_variance(Cn2, wavelength, distance_m))
        regime = classify_regime(sigma_R2)
        model = {'weak': 'lognormal', 'moderate': 'gamma_gamma',
                 'strong': 'gamma_gamma_corrected'}[regime]

    if model == "deterministic":
        if None in (alpha_atm, distance_km, wavelength, w0, receiver_radius):
            raise ValueError("model='deterministic' requires alpha_atm, distance_km, "
                              "wavelength, w0, receiver_radius.")
        return DeterministicChannel(alpha_atm, distance_km, wavelength, w0, receiver_radius)

    if model not in ("lognormal", "gamma_gamma", "gamma_gamma_corrected"):
        raise ValueError(f"unknown model '{model}'; choose from "
                          f"{list(_MODELS) + ['auto']}")

    if not two_link:
        return _build_single(model, eta_nominal, Cn2, wavelength, distance_m, receiver_radius,
                              jitter_std, beam_wander_std, w0, wave_type,
                              dict(kwargs, seed=seed))

    LA, LB = CAL.two_link_geometry(distance_m, distance_A_m, distance_B_m)
    eA, eB = CAL.two_link_nominal(eta_nominal, eta_A_nominal, eta_B_nominal)
    if bb:
        fA = beam_broadening_factor(Cn2, wavelength, LA, w0, receiver_radius)
        fB = beam_broadening_factor(Cn2, wavelength, LB, w0, receiver_radius)
        eA, eB = eA * fA, eB * fB
    kA = dict(kwargs, seed=seed)
    kB = dict(kwargs, seed=None if seed is None else seed + 1)
    link_A = _build_single(model, eA, Cn2, wavelength, LA, receiver_radius,
                           jitter_std, beam_wander_std, w0, wave_type, kA)
    link_B = _build_single(model, eB, Cn2, wavelength, LB, receiver_radius,
                           jitter_std, beam_wander_std, w0, wave_type, kB)
    ch = TwoLinkChannel(link_A, link_B, seed=seed)
    if bb:
        WA, WLA = long_term_beam_radius(Cn2, wavelength, LA, w0)
        WB, WLB = long_term_beam_radius(Cn2, wavelength, LB, w0)
        ch.beam_broadening = {'factor_A': fA, 'factor_B': fB, 'W_A': WA, 'W_LT_A': WLA,
                              'W_B': WB, 'W_LT_B': WLB}
        ch.name = f"two_link_{model}_bb"
    return ch
