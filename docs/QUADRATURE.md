# QUADRATURE.md — Numerical Quadrature Engine

Implementation: `src/quadrature.py`.

## Methods

1. **`quad_expectation`** — adaptive `scipy.integrate.quad`
   (QUADPACK/Clenshaw-Curtis-family), the default and most robust method,
   used internally by `TurbulenceChannel.average(..., method="quadrature")`.
2. **`gauss_legendre_expectation`** — fixed-order n-point Gauss-Legendre
   rule on [a,b], fast for smooth integrands, used for the accuracy/speed
   benchmark below.
3. **`gauss_laguerre_expectation`** — Gauss-Laguerre rule for semi-infinite
   integrals written with the exponential tail factored out
   (integral_0^inf f(x) g(x) e^-x dx); useful when working directly in
   the natural [0, infinity) support of the Gamma-Gamma / log-normal
   tails rather than the truncated-to-eta_max form used elsewhere in this
   codebase.

## A pitfall this module specifically guards against

Fading-channel PDFs (Gamma-Gamma especially) can be a **very narrow
spike** relative to the full physical support [0, eta_max=1] — e.g. a
mean transmittance of 1% with std/mean ~ 0.1-0.3. Plain adaptive `quad`'s
initial coarse pass over the *whole* interval can step right over that
spike and return a confidently-wrong answer with a deceptively small
reported error estimate (discovered during development of this
repository — see docs/VALIDATION.md for the before/after numbers: this
silently produced values off by a factor of ~2 before being caught by
the Monte Carlo cross-check). The fix implemented here,
`quad_expectation(..., points=[...])`, hands `scipy.integrate.quad`
explicit interior breakpoints — by default 0.3, 1.0, and 3.0 times
`channel.eta_nominal`, set by `TurbulenceChannel.average()` — forcing it
to subdivide around the region where the density actually lives.
**Anyone adding a new turbulence model or a new use of `quad_expectation`
directly should pass sensible `points` hints, not rely on default
unaided adaptivity, especially for narrow or multi-modal densities.**

## Also: integration bound is eta_max, not eta_nominal

A second, related pitfall (also documented in `turbulence/base.py`):
fading is multiplicative around a mean-1 normalized irradiance, so
instantaneous eta can exceed eta_nominal (up to the physical ceiling of
1) just as it can fall below it. Every quadrature call in this codebase
integrates up to `channel.eta_max` (=1 by default), not `eta_nominal` —
integrating only to eta_nominal was an earlier bug here that silently
discarded roughly half the probability mass for symmetric-ish
distributions.

## Benchmark: `benchmark_quadrature`

Compares Gauss-Legendre at several fixed orders n against `scipy.quad`
as ground truth: relative error and wall-clock runtime per order. Table
reproduced in docs/PERFORMANCE.md; visualized by
`plotting.plot_quadrature_benchmark`.
