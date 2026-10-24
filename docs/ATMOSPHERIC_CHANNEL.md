# ATMOSPHERIC_CHANNEL.md — Deterministic FSO Link Budget

Implementation: `src/turbulence/deterministic.py`, reusing
`legacy_baseline.deterministic_fso_channel`.

## Components

    eta(L) = eta_atm(L) * eta_geo(L)^2

1. **Atmospheric (Beer-Lambert) attenuation:**
   eta_atm(L) = exp(-alpha_atm * L), where alpha_atm [1/km] aggregates
   molecular absorption and aerosol scattering (clear-air default 0.2/km
   in `constants.DEFAULT_FSO_PARAMS`; overridden per link budget as
   needed; the coefficient is a user-supplied input and is not derived
   here).

2. **Diffraction / Gaussian-beam spreading:**
   w(z) = w0 sqrt(1 + (lambda z / (pi w0^2))^2), the standard
   free-space Gaussian-beam radius at propagation distance z, with w0
   the transmit beam waist.

3. **Geometric (truncation) efficiency** at a circular receiver of
   radius r_rx:
   eta_geo(z) = 1 - exp(-2 r_rx^2 / w(z)^2)
   (fraction of a Gaussian beam's power falling within radius r_rx).

The squared eta_geo^2 in the top formula corresponds to the truncation
efficiency being applied once per arm of the two-arm TF-QKD geometry, as
implemented in `deterministic_fso_channel`. See `legacy_baseline.py` for the
exact expression; this document describes only what each factor represents.

## Role in the polymorphic framework

`DeterministicChannel` is the "no turbulence" member of the
`TurbulenceChannel` family (Model 1). Its pdf and cdf are degenerate: a Dirac delta at eta_nominal, represented
as a narrow top-hat of unit area for callers that need an array. `.average(func)`
returns `func(eta_nominal)` for any method argument, since there is nothing to
average over. This allows `TFQKDEngine` and the distance-sweep and optimization
routines to treat the no-turbulence case as another channel.
