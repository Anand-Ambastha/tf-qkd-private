# ATMOSPHERIC_CHANNEL.md — Deterministic FSO Link Budget

Implementation: `src/turbulence/deterministic.py`, reusing
`legacy_baseline.deterministic_fso_channel` unmodified.

## Components

    eta(L) = eta_atm(L) * eta_geo(L)^2

1. **Atmospheric (Beer-Lambert) attenuation:**
   eta_atm(L) = exp(-alpha_atm * L), where alpha_atm [1/km] aggregates
   molecular absorption and aerosol scattering (clear-air default 0.2/km
   in `constants.DEFAULT_FSO_PARAMS`; overridden per link budget as
   needed — e.g. fog/haze conditions push this an order of magnitude
   higher, which is a user-supplied input, not derived here).

2. **Diffraction / Gaussian-beam spreading:**
   w(z) = w0 sqrt(1 + (lambda z / (pi w0^2))^2), the standard
   free-space Gaussian-beam radius at propagation distance z, with w0
   the transmit beam waist.

3. **Geometric (truncation) efficiency** at a circular receiver of
   radius r_rx:
   eta_geo(z) = 1 - exp(-2 r_rx^2 / w(z)^2)
   (fraction of a Gaussian beam's power falling within radius r_rx).

The squared eta_geo^2 in the top formula reflects that the legacy
baseline's `deterministic_fso_channel` applies the truncation efficiency
once per direction of a two-way/dual-link geometry consistent with the
TF-QKD setup it was built for — see `legacy_baseline.py` for the
authoritative, unmodified expression; this document only explains what
each factor physically represents, it does not redefine them.

## Role in the polymorphic framework

`DeterministicChannel` is the "no turbulence" member of the
`TurbulenceChannel` family (Model 1). Its pdf/cdf are (honestly)
degenerate — a Dirac delta at eta_nominal, represented as a narrow area-1
top-hat for the rare caller that needs a literal array rather than
recognizing the point-mass special case — and `.average(func)` returns
`func(eta_nominal)` exactly for any method argument, since there is
nothing to average over. This lets `TFQKDEngine` and every
distance-sweep/optimization routine treat "no turbulence" as just another
channel without special-casing it at the call site.
