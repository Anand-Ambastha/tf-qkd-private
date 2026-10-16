# GAMMA_GAMMA.md — Gamma-Gamma Turbulence Model, Full Derivation

Implementation: `src/turbulence/gamma_gamma.py`. This document derives
every equation used there, step by step.

## 1. Physical model: product of two Gamma processes

Following Al-Habash, Andrews & Phillips (2001) [AAP], the received
normalized irradiance is modeled as

    I = I_x * I_y

where I_x represents large-scale (refractive) fluctuations from eddies
larger than the first Fresnel zone / beam radius, and I_y represents
small-scale (diffractive) fluctuations from eddies smaller than that
scale. Conditioned on I_y, I_x is modeled as Gamma-distributed (this is
itself a modulation argument: large-scale eddies modulate the mean
small-scale irradiance), and by symmetry of the derivation I_y is also
Gamma-distributed:

    I_x ~ Gamma(alpha, 1/alpha),   E[I_x] = 1
    I_y ~ Gamma(beta,  1/beta),    E[I_y] = 1

so that E[I] = E[I_x] E[I_y] = 1 (energy conservation: turbulence
redistributes irradiance around a fixed mean, it doesn't add or remove
average power). alpha and beta are interpreted as the *effective number*
of discrete large-scale and small-scale scattering cells contributing to
the fluctuations.

## 2. PDF: product of two Gamma random variables

For independent I_x ~ Gamma(alpha, 1/alpha), I_y ~ Gamma(beta, 1/beta),
the PDF of the product I = I_x I_y follows from the standard
product-of-two-Gammas result (a Mellin-convolution calculation, see
[AAP] Eq. 6-9, or any table of product distributions):

    p_I(I) = 2 (alpha beta)^((alpha+beta)/2) / [Gamma(alpha) Gamma(beta)]
             * I^((alpha+beta)/2 - 1) * K_{alpha-beta}( 2 sqrt(alpha beta I) ),
             I > 0

where K_v is the modified Bessel function of the second kind, order v
(implemented via `scipy.special.kv`, with an arbitrary-precision `mpmath`
fallback for the large-order regime — see implementation note below).
This is implemented exactly as `gamma_gamma_pdf(I, alpha, beta)`.

**Derivation sketch.** Write I_y = I/I_x and use the Jacobian
dI_y = dI/I_x:

    p_I(I) = integral_0^inf p_{I_x}(x) p_{I_y}(I/x) (1/x) dx

Substituting the two Gamma densities and using the integral
representation of K_v,

    K_v(z) = (1/2) (z/2)^v integral_0^inf t^(-v-1) exp(-t - z^2/(4t)) dt,

after algebra (see [AAP] Appendix, or [AP] Sec. 9.5) yields the closed
form above.

## 3. alpha, beta from Rytov theory

The task is then to relate alpha and beta to the physical turbulence
strength (Cn^2, wavelength, path length) through the Rytov variance
sigma_R^2 (`legacy_baseline.rytov_variance`, unmodified). [AP] Eq. 9.59
(plane wave):

    alpha = { exp[ 0.49 sigma1^2 / (1 + 1.11 sigma1^(12/5))^(7/6) ] - 1 }^-1
    beta  = { exp[ 0.51 sigma1^2 / (1 + 0.69 sigma1^(12/5))^(5/6) ] - 1 }^-1

**Notation (important, corrected in v2.2.0):** sigma1^2 is the Rytov *variance*, so
sigma1^(12/5) = (sigma1^2)^(6/5). Versions <= 2.1.0 evaluated `(sigma1^2)^(12/5)` by
mistake, which collapsed the scintillation index to ~0 for sigma1^2 >> 1 (e.g. SI = 0.08
at sigma1^2 = 10 instead of 1.24) and made strong turbulence look benign. The corrected
model has SI = 0.71 at sigma1^2 = 1, peaks near 1.2-1.3 around sigma1^2 ~ 5-10 and
saturates toward 1 (alpha -> inf, beta -> 1). Regression test:
`tests/test_moments.py::test_gamma_gamma_alpha_beta_match_andrews_phillips`.

These come from matching the Gamma-Gamma model's large-scale and
small-scale log-irradiance variances to the weak-fluctuation Rytov
theory's known asymptotic large-scale/small-scale variance components
(themselves derived from the Rytov spatial-filtering functions), then
using a semi-empirical bridging function (the denominators above) so the
result also matches the observed saturation behavior (alpha->inf,
beta->1) at sigma_R^2 -> infinity. This bridging step is *not* rigorously
derived from first principles — it is Andrews & Phillips's calibrated fit
to a large body of experimental and simulation data, which is why the
Gamma-Gamma model is described as "semi-empirical" in the FSO literature,
in contrast to the fully first-principles Rytov/log-normal weak-turbulence
result.

Spherical-wave coefficients (`alpha_beta_spherical_wave`, used when
`wave_type="spherical"`), [AP] Table 9.2:

    alpha = { exp[ 0.49 sigma1^2 / (1 + 0.56(1+Theta) sigma1^(12/5))^(7/6) ] - 1 }^-1
    beta  = { exp[ 0.51 sigma1^2 / (1 + 0.69 sigma1^(12/5))^(5/6) ] - 1 }^-1

(same convention: sigma1^(12/5) = (sigma1^2)^(6/5)).

Theta is the beam curvature parameter at the receiver plane (Theta=1 for
a "matched"/focused reference case, the default used here); the fully
general spherical/Gaussian-beam treatment introduces two further beam
parameters (Theta, Lambda) that this repository does not separately
expose — documented simplification, consistent with the module docstring
in `gamma_gamma.py`.

## 4. Moments

Since I_x, I_y are independent with unit mean,

    E[I] = 1
    E[I^2] = E[I_x^2] E[I_y^2] = (1 + 1/alpha)(1 + 1/beta)
    Var(I) = E[I^2] - 1 = 1/alpha + 1/beta + 1/(alpha beta)
    Scintillation index SI = Var(I)/E[I]^2 = Var(I)   (since E[I]=1)

(`gamma_gamma_moments`). As alpha, beta -> infinity (turbulence-free
limit), Var(I) -> 0 as expected. As sigma_R^2 -> infinity, alpha -> infinity
and beta -> 1, so SI -> 1 — the well-known irradiance-saturation limit
where I becomes exponentially distributed (a negative-exponential speckle
pattern), consistent with strong-turbulence experimental observations.

## 5. Sampling

Exact (not approximate) sampling: draw I_x ~ Gamma(alpha, scale=1/alpha)
and I_y ~ Gamma(beta, scale=1/beta) independently via
`numpy.random.Generator.gamma`, and multiply. This is exact because it
uses the model's own generative definition, not inverse-CDF sampling of
the (more awkward, Bessel-K-based) marginal PDF.

## 6. CDF

No elementary closed form exists for the CDF (it involves a Meijer-G
function in general). `GammaGammaChannel` builds a monotone CDF via
cumulative trapezoidal quadrature of the PDF over a fine log-spaced grid
at construction time (`_build_cdf_grid`), interpolated at query time;
an exact single-point CDF via `scipy.integrate.quad` is also available
(`_cdf_exact`) for spot-checking (used in tests/test_distributions.py).

## 7. From I to eta to SKR

eta = eta_nominal * I. `TFQKDEngine` evaluates the unmodified
Curty-Azuma-Lo key-rate formula R(alpha, eta) at each realization (or its
ergodic average E_I[R(alpha, eta_nominal I)] via quadrature/Monte Carlo);
see THEORY.md Section 6 and tfqkd_engine.py.

## Implementation note: numerical stability for large/asymmetric (alpha, beta)

K_v(x) for small x and large order v behaves like
K_v(x) ~ (1/2) Gamma(v) (2/x)^v, which overflows IEEE double precision
already for v > ~170 (Gamma(170) ~ 10^305). Strongly asymmetric
(alpha, beta) — common at moderate-to-strong turbulence with a narrow
beam — push |alpha - beta| well past that. `_log_kv` in `gamma_gamma.py`
therefore evaluates `scipy.special.kv` directly wherever it is finite
(the common case, fast), and falls back to `mpmath.besselk` at
arbitrary precision (then takes the log there, where it's safe) for the
handful of grid points where it overflows. This was caught during
development by the PDF-normalization test failing silently (see
docs/VALIDATION.md) and is now covered explicitly by
`tests/test_adaptive_alpha_and_stability.py::test_gamma_gamma_large_alpha_beta_no_overflow`.

## References

See docs/THEORY.md References section ([AP], [AAP]).
