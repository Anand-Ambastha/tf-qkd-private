# THEORY.md — Atmospheric Turbulence Theory for Free-Space Optical TF-QKD

This document derives the physics that underlies every turbulence model
in `src/turbulence/`. It is deliberately a *working derivation*, not an
exhaustive textbook chapter — for full rigor at every step, see the cited
sources, principally Andrews & Phillips [AP].

## 1. Refractive-index fluctuations and the Kolmogorov spectrum

Atmospheric turbulence arises from random temperature (and, to a lesser
extent, humidity and pressure) fluctuations that perturb the air's
refractive index n = 1 + δn(**r**, t). Under Kolmogorov's local isotropy
hypothesis, kinetic energy cascades from large eddies (outer scale L0,
meters to tens of meters) down to small eddies (inner scale l0,
millimeters) where it dissipates as heat. The refractive-index structure
function

    D_n(r) = <[n(r1) - n(r1+r)]^2> = Cn^2 r^(2/3),     l0 << r << L0

defines the **refractive-index structure parameter** Cn^2 [m^(-2/3)],
the single scalar that (in the Kolmogorov model) characterizes turbulence
strength at a point along the path. Typical values: 1e-17 (weak,
night-time, high altitude) to 1e-13 (strong, daytime, near ground)
m^(-2/3); see `constants.Cn2_classes` inherited from the legacy baseline.

The corresponding 3-D power spectral density of refractive-index
fluctuations (Kolmogorov spectrum) is

    Phi_n(kappa) = 0.033 Cn^2 kappa^(-11/3),    1/L0 << kappa << 1/l0

The **Tatarskii spectrum** extends this with a Gaussian rolloff at high
wavenumber to model the inner-scale dissipation range:

    Phi_n(kappa) = 0.033 Cn^2 kappa^(-11/3) exp(-kappa^2/kappa_m^2),
    kappa_m = 5.92/l0

`CorrectedGammaGammaChannel`'s optional `inner_scale`/`outer_scale`
parameters are a first-order stand-in for the effect this spectral
modification has on the Rytov variance (see docs/CORRECTED_GAMMA_GAMMA.md)
rather than a full re-derivation of the Rytov integral against the
Tatarskii/von Karman spectrum — that integral is a substantial undertaking
on its own and is flagged as future work.

## 2. The Rytov approximation and scintillation

For weak fluctuations, the wave equation is solved perturbatively (Rytov
method): the complex phase psi = chi + i S is expanded to first order,
where chi is the log-amplitude fluctuation. The **Rytov variance** for a
plane wave over path length L is

    sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6),     k = 2 pi / lambda

(implemented, unchanged, as `legacy_baseline.rytov_variance`). This is
*not* the variance of the irradiance directly — it's the variance of the
first-order Rytov log-amplitude solution, and by convention is used as
the turbulence-strength parameter that indexes every downstream model in
this repository (log-normal, Gamma-Gamma alpha/beta, and the AUTO
regime selector in `turbulence_factory.py`).

Weak-fluctuation theory predicts the log-irradiance variance is
sigma_ln I^2 = 4 sigma_R^2 (Rytov's original result for chi), and the
**scintillation index** SI = Var(I)/E[I]^2 -> 4 sigma_R^2 as sigma_R^2 -> 0.
This is the basis for the log-normal irradiance model (Section 3).

## 3. Log-normal irradiance model (weak turbulence, sigma_R^2 < 1)

If ln(I) is Gaussian (Rytov's weak-perturbation result) and the model is
constrained to be energy-conserving (E[I]=1), then

    I = exp(2 chi),   chi ~ N(mu_chi, sigma_chi^2),  mu_chi = -sigma_chi^2

so ln I ~ N(mu_X, sigma_X^2) with mu_X = -sigma_X^2/2 (this is exactly
`legacy_baseline.lognormal_params`, kept unmodified — see
docs/LOGNORMAL.md). This model is accurate while multiple-scattering and
saturation effects are negligible, i.e. sigma_R^2 < 1; beyond that the
predicted tail underestimates the true deep-fade probability because the
single-scatter Rytov perturbation series itself stops converging.

## 4. Strong/moderate turbulence: the Gamma-Gamma model

Beyond sigma_R^2 ~ 1, the Rytov series breaks down, but a *phenomenological*
model — irradiance as the product of independent large-scale (refractive)
and small-scale (diffractive) Gamma-distributed processes — reproduces
measured PDFs accurately from moderate turbulence through saturation
(Al-Habash, Andrews & Phillips 2001). Full derivation, closed-form PDF,
alpha/beta parameter equations, and moments are in docs/GAMMA_GAMMA.md;
this section only states the physical motivation: I = I_x I_y, where I_x
captures large-eddy focusing/defocusing (effective number of large-scale
cells alpha) and I_y captures small-eddy diffraction (effective number of
small-scale cells beta). As turbulence strengthens, alpha grows without
bound (the large-scale contribution to normalized variance shrinks) while
beta approaches 1, so that SI = 1/alpha + 1/beta + 1/(alpha beta) -> 1,
the well-known saturation value.

## 5. Aperture averaging, beam wander, pointing errors

A finite receiver averages irradiance over its aperture, smoothing out
scintillation whose correlation length is smaller than the aperture
(Andrews & Phillips Ch. 10). Beam wander (random displacement of the
instantaneous beam centroid by large-scale eddies) and pointing/jitter
errors both act as *additional* multiplicative fading on top of
turbulence-induced scintillation, and are the dominant impairments for a
narrow, long-range beam (e.g. a ground-to-satellite QKD uplink) with a
receiver much smaller than the beam footprint. `CorrectedGammaGammaChannel`
implements all three; full equations and their assumptions/references are
in docs/CORRECTED_GAMMA_GAMMA.md.

### 5b. Independent arms and beam broadening (v2.2 and v2.3)

In TF-QKD both Alice and Bob send to Charlie. Each arm, of length L/2, fades independently, so
eta = eta_A * eta_B, and each arm has its own Rytov variance and its own alpha and beta (TWO_LINK_FADING.md).

With the correct sigma_R^(12/5) exponent, the Gamma-Gamma scintillation index peaks near 1.2 to 1.3 for a Rytov
variance around 5 to 10 and then settles toward 1. A Log-Normal model forced into that region keeps growing as
exp(sigma_R^2) - 1, which is why Log-Normal is trusted only for sigma_R^2 up to 1.

Fading models keep the mean power fixed. Turbulence also spreads the long-term beam, which lowers the mean.
That loss is treated in BEAM_BROADENING.md.

## 6. From irradiance to transmittance to secret key rate

Every model in this repository ultimately produces a random *normalized
irradiance* I with E[I]=1 (or, for the deterministic model, the
degenerate case I=1 always), and reports the physical transmittance as

    eta(t) = eta_nominal * I(t)

where eta_nominal is the deterministic (diffraction + atmospheric +
geometric) link transmittance from Model 1. `TFQKDEngine` (see
tfqkd_engine.py) then evaluates the *unmodified* Curty-Azuma-Lo secret
key rate formula R(alpha, eta) at each realization, and computes the
ergodic (turbulence-averaged) rate E_eta[R(alpha, eta)] either by
numerical quadrature against the channel's pdf or by Monte Carlo
sampling — see docs/QUADRATURE.md and docs/MONTE_CARLO.md.

## References

- [AP] Andrews, L. C. & Phillips, R. L., *Laser Beam Propagation through
  Random Media*, 2nd ed., SPIE Press, 2005.
- [AAP] Al-Habash, M. A., Andrews, L. C. & Phillips, R. L., "Mathematical
  model for the irradiance probability density function of a laser beam
  propagating through turbulent media," Opt. Eng. 40(8), 1554-1562 (2001).
- [FH] Farid, A. A. & Hranilovic, S., "Outage Capacity Optimization for
  Free-Space Optical Links with Pointing Errors," J. Lightwave Technol.
  25(7), 1702-1710 (2007).
- [Goodman] Goodman, J. W., *Statistical Optics*, Wiley, 1985 (2nd ed. 2015).
- [Korotkova] Korotkova, O., *Random Light Beams: Theory and Applications*,
  CRC Press, 2013.
- [Curty] Curty, M., Azuma, K. & Lo, H.-K., "Simple security proof of
  twin-field type QKD protocol," npj Quantum Information 5, 64 (2019);
  arXiv:1807.07667v2 — the TF-QKD security proof implemented, unmodified,
  in `src/legacy_baseline.py`.
