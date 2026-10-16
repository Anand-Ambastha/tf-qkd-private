# LOGNORMAL.md — Log-Normal Turbulence Model

Implementation: `src/turbulence/lognormal.py`. Per the task constraint,
this module does **not** modify the existing validated log-normal
physics — it wraps `legacy_baseline.rytov_variance`,
`legacy_baseline.lognormal_params`, `legacy_baseline.sample_lognormal_eta`,
and `legacy_baseline.turbulence_statistics` verbatim behind the
`TurbulenceChannel` interface so the rest of the framework (TF-QKD
engine, Monte Carlo, quadrature, adaptive alpha, plotting) can treat it
polymorphically alongside the new models.

> **Note (v2.1.0):** `LogNormalChannel` is now the *single-link* building block.
> The end-to-end TF-QKD channel uses two independent links, see
> [TWO_LINK_FADING.md](TWO_LINK_FADING.md) (`create_channel(..., two_link=True)`, the default).

## Model

eta = eta_nominal * exp(X), X ~ N(mu_X, sigma_X^2), with
mu_X = -sigma_X^2/2 so that E[exp(X)] = 1 (energy-neutral). sigma_X^2 is
derived from the Rytov variance via the unchanged `lognormal_params`
function (see docs/THEORY.md Section 3 for the physical derivation).

## Validity: sigma_R^2 < 1

The log-normal model is the weak-perturbation (single-scatter Rytov)
limit. Its accuracy degrades once sigma_R^2 approaches and exceeds 1:

- The Rytov series itself is a perturbative expansion in Cn^2 L^(11/6);
  at sigma_R^2 ~ 1 higher-order (multiple-scattering) terms are no longer
  negligible, and the series is not guaranteed to converge beyond this
  point.
- Empirically, the log-normal PDF systematically **underestimates deep
  fades** (the probability of very low instantaneous transmittance) once
  turbulence strengthens past weak, because it lacks the heavier tail
  that measured strong-turbulence irradiance statistics exhibit. For a
  QKD link, deep fades are exactly the events that drive outage
  probability and dominate the ergodic key-rate penalty, so
  underestimating their probability is a non-conservative error in
  exactly the metric that matters most.
- Beyond sigma_R^2 ~ 1, use the Gamma-Gamma model instead
  (`turbulence/gamma_gamma.py`, docs/GAMMA_GAMMA.md), which is calibrated
  against strong-turbulence data through saturation.

`turbulence_factory.classify_regime()` and the `model="auto"` selector
enforce this boundary (`constants.SIGMA_R2_WEAK_MODERATE = 1.0`) so a
caller who doesn't specify a model explicitly won't silently get a
log-normal channel outside its validity domain.

## What's exposed through the polymorphic interface

- `pdf`, `cdf`: closed-form log-normal density/CDF in eta (via
  `scipy.stats.norm` applied to ln(eta/eta_nominal)).
- `sample`: delegates to `legacy_baseline.sample_lognormal_eta`.
- `statistics`: returns the *analytic* mean/variance/scintillation index
  from `legacy_baseline.turbulence_statistics` (which itself
  cross-checks against a Monte Carlo estimate internally — both are
  surfaced in the returned dict as `mean`/`variance` (analytic) and
  `mean_mc`/`var_mc`).
