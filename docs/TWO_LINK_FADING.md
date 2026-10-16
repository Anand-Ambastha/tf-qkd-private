# TWO_LINK_FADING.md — Independent fading on Alice–Charlie and Bob–Charlie

## The issue

TF-QKD has **two** free-space arms, Alice→Charlie (A–C) and Bob→Charlie
(B–C). They cross different air columns, so their turbulence-induced fading
is statistically **independent**. Earlier versions of this framework modelled
the whole A–B path as **one** fading factor at the full distance L:

    eta = eta_nom * exp(X),   X ~ N(-s2/2, s2),   s2 = s2(sigma_R^2(L))

## The model now used (default)

    eta_A = eta_A_nom * exp(X_A),   X_A ~ N(-s2_A/2, s2_A),   s2_A from sigma_R^2(L_A)
    eta_B = eta_B_nom * exp(X_B),   X_B ~ N(-s2_B/2, s2_B),   s2_B from sigma_R^2(L_B)
    X_A independent of X_B,         eta = eta_A * eta_B

with Charlie midway by default (L_A = L_B = L/2, eta_A_nom = eta_B_nom =
sqrt(eta_nom); this is consistent with `deterministic_fso_channel`, where
eta_nom = eta_atm(L) eta_geo(L/2)^2). Every ergodic quantity is a 2-D average:

    <g> = ∬ g(eta_A eta_B) p_A(eta_A) p_B(eta_B) d eta_A d eta_B

The same structure is applied to **all** stochastic models (log-normal,
Gamma-Gamma, corrected Gamma-Gamma): two independent single-link channels
wrapped in `turbulence.TwoLinkChannel`.

## Log-normal "average term" check

* Each arm keeps the mean-one (energy-neutral) normalisation
  mu = -s2/2, hence E[eta_A eta_B] = eta_A_nom eta_B_nom exactly (independence).
  Verified to 1e-12 (`tests/test_two_link.py::test_product_is_mean_one_unclipped`).
* Without the [0,1] clip, X_A + X_B ~ N(-(s2_A+s2_B)/2, s2_A+s2_B), so the
  2-D average of any g must equal the 1-D average with s2_A + s2_B. Verified
  for the key rate to ~1e-7 relative (a hard cross-check of the 2-D rule).
* Per-arm variance scales as L^(11/6): sigma_R^2(L/2) = 2^(-11/6) sigma_R^2(L),
  so s2_A + s2_B = 2^(-5/6) sigma_R^2(L) ≈ 0.561 sigma_R^2(L). The old
  one-factor model carried ≈1.78× too much log-variance.
* **Caveat:** clipping each arm at eta ≤ 1 (needed physically and done by the
  sampler) makes the *sampled* mean fall below nominal when the arm's nominal
  transmittance is large (e.g. a single factor with eta_nom = 0.5 lost ≈26 % of
  its mean). Quadrature and Monte Carlo treat the clip identically.

## API

```python
from src.turbulence import create_channel, TwoLinkChannel
ch = create_channel('lognormal', eta_nominal=0.05, Cn2=1e-15,
                    wavelength=1550e-9, distance_m=6000)      # two_link=True (default)
ch.average(func)                                  # 2-D tensor-product quadrature
ch.average(func, method='monte_carlo', n_mc=...)  # independent (eta_A, eta_B) draws
create_channel(..., two_link=False)               # legacy single factor (comparison only)
create_channel(..., distance_A_m=2000, distance_B_m=4000, eta_A_nominal=0.1)  # asymmetric
```

`model="auto"` classifies the regime **per arm** (sigma_R^2 at the arm length).
Legacy-module counterparts (`legacy_baseline.py` and
`notebooks/tfqkd_fso_turbulence_optimized.py`): `sample_lognormal_eta_two_link`,
`two_link_lognormal_params`, `two_link_sigma_R2`, `average_over_two_lognormal_links`,
`average_skr_lognormal_two_link(_array)`, `average_observables_two_link`,
`tfqkd_skr_turbulence_two_link`, `turbulence_statistics_two_link`.

## Numerical methods

* log-normal: tensor-product Gauss–Hermite (128 × 128 nodes by default),
  clip at eta = 1 included; agrees with Monte Carlo within MC error.
* other models: equal-probability quantile nodes per arm (O(1/n)); validated
  against Monte Carlo to ≲0.3 %.

## Modelling assumption to state in the write-up

`key_rate()` is the symmetric-channel Curty–Azuma–Lo formula evaluated at the
effective transmittance eta = eta_A eta_B. It assumes the amplitudes reaching
Charlie are balanced or compensated. An *instantaneous* amplitude imbalance
between the arms (which independent fading produces) would additionally reduce
interference visibility; that penalty is **not** modelled, so results are an
upper bound for a fully uncompensated system.

Also unchanged from before: the log-normal parameterisation sets
sigma_X^2 = sigma_R^2 (so SI = e^{sigma_R^2} - 1, slightly above sigma_R^2).
Worth confirming this is the intended convention.
