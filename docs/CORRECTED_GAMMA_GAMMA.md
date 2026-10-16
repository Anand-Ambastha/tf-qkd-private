# CORRECTED_GAMMA_GAMMA.md — Aperture Averaging, Beam Wander, Pointing Errors

Implementation: `src/turbulence/corrected_gamma_gamma.py`.

## 1. Aperture averaging

A receiver of finite diameter D integrates irradiance over its aperture;
scintillation cells smaller than D partially cancel, reducing the
*effective* scintillation index below the point-receiver value. The
plane-wave aperture-averaging factor implemented here ([AP] Ch. 10,
weak-to-moderate turbulence approximation):

    A(D) = [ 1 + 1.062 D^2 / (4 lambda L) ]^(-7/6)

`aperture_averaging_factor()`. A in (0, 1]; A=1 recovers the point-
receiver (no averaging) case.

**Applying A to (alpha, beta).** A rigorous treatment aperture-averages
the large-scale and small-scale spectral contributions to scintillation
*separately*, since they have different spatial correlation lengths
relative to the aperture. This repository uses the standard practical
simplification of scaling both Gamma-Gamma shape parameters by the same
factor,

    alpha_eff = alpha / A,   beta_eff = beta / A

which gives Var_eff = A(1/alpha + 1/beta) + A^2/(alpha beta) ~= A * Var0
to leading order (the cross term is normally sub-dominant). An earlier
implementation attempt held beta fixed and solved for alpha_eff from the
exact variance identity; this is numerically fragile — for aggressive
aperture averaging (small A), the target variance can fall below beta's
own contribution alone, forcing alpha_eff negative and requiring an ad
hoc clip. The symmetric 1/A rescaling used in the shipped code has no
such failure mode (alpha_eff, beta_eff stay positive for every A in
(0,1]) and was adopted for that reason during development and validation
(see docs/VALIDATION.md).

## 2. Beam wander

Large-scale eddies (larger than the beam diameter) displace the
instantaneous beam centroid randomly about its mean position — "beam
wander" — which is a *long-term* beam-broadening / additional-jitter
effect distinct from (and in addition to) scintillation. It's modeled
here as an additional contribution to the effective pointing-jitter
standard deviation:

    sigma_s_total = sqrt(jitter_std^2 + beam_wander_std^2)

fed into the pointing-error model of Section 3. This combines mechanical
pointing jitter and beam-wander-induced centroid displacement into one
effective misalignment budget, consistent with how both are commonly
treated together in the link-budget literature ([AP] Ch. 12).

## 3. Pointing errors: Farid & Hranilovic (2007) model

For a Gaussian beam of long-term radius w_Lt at the receiver plane
(`legacy_baseline.beam_radius`, unmodified) and a receiver of radius a,
define

    v = sqrt(pi) a / (sqrt(2) w_Lt)
    A0 = erf(v)^2                              (max fraction of power collected, zero-jitter limit)
    w_Leq^2 = w_Lt^2 sqrt(pi) erf(v) / (2 v exp(-v^2))   (equivalent beam radius)
    gamma_pe = w_Leq / (2 sigma_s)              (pointing-error shape parameter)

Farid & Hranilovic show the fraction of power collected under Gaussian
pointing-jitter h_p (radial displacement ~ Rayleigh with scale sigma_s)
has the closed-form power-law PDF

    f(h_p) = (gamma_pe^2 / A0^(gamma_pe^2)) * h_p^(gamma_pe^2 - 1),   0 <= h_p <= A0

`pointing_error_params()` computes (gamma_pe, A0); sampling is by exact
inverse-CDF transform, h_p = A0 * U^(1/gamma_pe^2) for U ~ Uniform(0,1)
(`_sample_hp`), renormalized to unit mean so eta_nominal remains the
reference "no impairment" transmittance.

## 4. Composite channel and its (numerical) pdf/cdf

The total normalized fading factor is F = I_turbulence * (h_p / E[h_p]),
the product of an aperture-averaged Gamma-Gamma random variable and an
independent power-law random variable. No simple closed form exists for
the PDF of this product in general (a Meijer-G representation exists in
the specialized FSO literature but is numerically delicate to evaluate
robustly across parameter regimes). Rather than adopt an approximation
that would need its own separate validation, `CorrectedGammaGammaChannel`
builds F's distribution *empirically*, from a large (default 200,000)
cached Monte Carlo sample generated once at construction:

  - `pdf(eta)`: linear interpolation of a 200-bin histogram density.
  - `cdf(eta)`: linear interpolation of the empirical (sorted-sample)
    CDF.
  - `sample(n)`: still exact, independent generative sampling (not drawn
    from the cache) — only pdf/cdf are cache-based.

This is explicitly a numerical, not analytic, representation, and is
documented at the point of use in the source rather than silently
presented as a closed form. It is validated the same way as every other
model: PDF integrates to 1 (`tests/test_distributions.py`), and
Monte-Carlo/quadrature expectations of the TF-QKD key rate agree to
<0.1% at the operating points checked in `tests/test_mc_vs_quadrature.py`.

## 5. Inner scale / outer scale (documented approximation, not full spectrum)

`inner_scale` (l0) and `outer_scale` (L0) are accepted as optional
constructor parameters. A rigorous treatment replaces the Kolmogorov
spectrum with the Tatarskii (inner-scale) or von Karman (outer-scale)
spectrum inside the Rytov-variance integral (see docs/THEORY.md, Section
1) — a substantial standalone derivation. Here, instead, they apply a
first-order multiplicative heuristic correction directly to sigma_R^2
(`_inner_outer_scale_correction`): finite inner scale slightly increases
scintillation (more high-wavenumber power retained, +10% cap as l0->0,
negligible for l0 above a few mm), and finite outer scale slightly
decreases it (removes some large-eddy/low-wavenumber power, -5% cap,
negligible for L0 above tens of meters). This is flagged in the source
as a simplification and is not claimed to reproduce the literature's
modified-spectrum results quantitatively — it exists so the parameter
plumbing is in place for a future, more rigorous implementation, rather
than being silently ignored.

## 6. Reduction to plain Gamma-Gamma

With `enable_aperture_averaging=False`, `enable_pointing_error=False`,
`inner_scale=None`, `outer_scale=None` (the defaults for the latter two),
`CorrectedGammaGammaChannel` computes exactly the same (alpha, beta) as
`GammaGammaChannel` at the same (Cn2, wavelength, distance_m, wave_type),
and h_p degenerates to the constant 1 — verified in
`tests/test_corrected_gamma_gamma_reduces.py`.

## References

- [AP] Andrews & Phillips, *Laser Beam Propagation through Random Media*,
  2nd ed., SPIE Press, 2005, Ch. 10 & 12.
- [FH] Farid, A. A. & Hranilovic, S., "Outage Capacity Optimization for
  Free-Space Optical Links with Pointing Errors," J. Lightwave Technol.
  25(7), 1702-1710 (2007).
