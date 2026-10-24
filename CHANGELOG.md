# CHANGELOG

## Version History

### v2.4.1: truncation residual Delta_j (CAL Eq. 22)
- `residual_j` subtracted a square set of photon numbers (m_A, m_B <= (Nmax - j)//2) from the full double sum, while
  `inner_sum_j` sums over the triangular set n_A + n_B <= Nmax. Pairs such as (n_A, n_B) = (2, 4) were in neither term, so
  Delta_j was too small (at alpha = 0.2 and Nmax = 4: 4.7e-6 against 2.3e-5 for j = 0). `residual_j` now subtracts the
  triangular set. Added a brute-force regression test.
- Effect on the key rate at alpha = 0.2 (deterministic): -0.1% at 10 dB, -2.6% at 60 dB, -13% at 75 dB, -70% at 80 dB; the
  cutoff loss moves by about 0.26 dB. Maximum range (R > 1e-9): 252.0 -> 248.8 km deterministic, 291.7 -> 288.7 km
  Gamma-Gamma weak, 317.6 -> 314.5 km log-normal weak.
- Pinned manuscript Tables IV to VI and one Appendix H value in `analytical_validation/paper_values.py` and the test
  suite were regenerated; they move by at most 0.025 percentage points (old values are noted in the file).
  The result tables, figures and logs in `results_v2.3.0/`, `paper_analysis/results/` and the reports predate this fix and
  have not been regenerated.

### v2.4.0: analytical_validation/ (tests of the log-normal and Gamma-Gamma analytical formulas)
- New folder `analytical_validation/` with three scripts and a PDF report: 01 tests the log-normal averaging framework,
  02 tests the Gamma-Gamma cumulant expansion with aperture averaging, 03 builds
  `results/Analytical_Formula_Validation_Report.pdf` (tables and graphs). 70 numeric checks, all passing; `run_all.py`
  reruns everything in about 30 s.
- The analytic derivatives (orders 0 to 4) of pXX, eX, eZ and R are implemented from the manuscript appendix and checked
  against a separate 60-digit reference. All 54 entries of the manuscript's Tables IV to VI are reproduced.
- Findings recorded in the report: the progress report's Gamma-Gamma figures use (Rytov variance)^(12/5) instead of
  sigma_R^(12/5) (the exponent slip fixed in v2.2.0); the pXX bracket in manuscript Eq. (6) and Appendix A is misplaced;
  "sigma_X^2 = 9.6" in Appendix H should read 2.4; the L = 20 km row of report Tables 1 and 2 shows a formula-independent offset.
- New `tests/test_analytical_validation.py` (4 tests).

### v2.3.1: examples 01-05 re-run, old results archived
- Examples 01 to 05 were re-run after the Gamma-Gamma fix. New figures, reports and logs are in
  `results_v2.3.0/`. The earlier outputs are untouched in `old_results_v2.1.0_pre_gg_fix/` and are there for
  comparison only.
- Examples 04 and 05 had a hard cutoff that set the plain Gamma-Gamma and Log-Normal rates to zero once the
  arm Rytov variance passed 300. For strong Cn2 that happened at about 30 km, which is why plain
  Gamma-Gamma showed a 30.6 km range there. The cutoff is gone and the range is 291.6 km, in line with
  example 06.
- The interpretation text in example 04's report used to quote numbers from an older setup that did not match
  its own tables. It is now built from the results, and it marks any range that simply hit the sweep limit.

### v2.3.0: turbulence-induced beam broadening
- New models `gamma_gamma_bb`, `lognormal_bb` and `gamma_gamma_corrected_bb` (`src/turbulence/beam_broadening.py`).
  Each arm gets the usual fading plus long-term beam broadening from Andrews and Phillips:
  W_LT^2 = W^2 (1 + 1.63 sigma_R^(12/5) Lambda), with Lambda = 2z/(k W^2) and z = L/2. The arm's nominal
  transmittance is multiplied by the ratio of captured power with and without broadening,
  f_bb = [1 - exp(-2a^2/W_LT^2)] / [1 - exp(-2a^2/W^2)]. The fading is left as it was, and at Cn2 = 0 the model
  reduces to its base model. It needs `w0` and `receiver_radius` and only works per arm.
- Added `examples/example_07_beam_broadening_gamma_gamma.py`, `docs/BEAM_BROADENING.md`,
  `docs/BEAM_BROADENING_REPORT.txt` and `tests/test_beam_broadening.py`.
- Because broadening lowers the mean transmittance, the maximum range now falls as Cn2 rises. Plain
  Gamma-Gamma does not do that: its range saturates and barely depends on Cn2.

### v2.2.0: Gamma-Gamma fix, per-arm validation against Log-Normal
- Fixed a physics error. `alpha_beta_plane_wave` and `alpha_beta_spherical_wave` raised the Rytov variance to the
  12/5 power, but the formula needs the standard deviation, i.e. the variance to the 6/5 power. The scintillation
  index dropped to nearly zero for large Rytov variance (0.08 at a variance of 10, against the correct 1.24).
  Every moderate and strong Gamma-Gamma and corrected Gamma-Gamma result from v2.0 and v2.1 was therefore too
  optimistic. The weak regime and a Rytov variance of 1 are unaffected. A regression test now guards the values.
- Gamma-Gamma stays independent on each arm by default (`TwoLinkChannel`). Each arm of length L/2 has its own
  Rytov variance and its own alpha and beta, and the two draws are independent (measured correlation about 0).
  The single-factor channel is still available with `two_link=False`, for comparison.
- The Gamma-Gamma CDF is now built from I = X*Y, a product of two Gamma variables, using the regularized
  incomplete gamma function. Integrating the Bessel-K density failed for very weak turbulence, where alpha and
  beta are around 1e5.
- `quantile_nodes` uses cosine-spaced probability cells in place of equal-probability cells. The old rule
  under-weighted rare upward fades and came out about 6 % low at the outage threshold with 400 nodes. The new
  one changes by less than 0.5 % between 200 and 1600 nodes.
- Added `examples/example_06_gamma_gamma_per_arm_validation.py` (maximum range for weak, moderate and strong
  Cn2, Gamma-Gamma against Log-Normal, a Monte Carlo cross-check, ten figures) and its report,
  `docs/GAMMA_GAMMA_PER_ARM_REPORT.txt`.

### v2.1.0 — Independent fading on Alice–Charlie and Bob–Charlie
- **Fix (physics):** the turbulence channel used ONE fading factor at the full
  distance L. TF-QKD has two independent FSO arms, so the end-to-end
  transmittance is now eta = eta_A * eta_B with separate log-normal / Gamma-Gamma
  fading per arm (length L/2, Rytov variance per arm) and all ergodic averages
  taken over the two independent random variables.
- New `turbulence/two_link.py::TwoLinkChannel`; `create_channel(..., two_link=True)`
  is the new default (`two_link=False` restores the old behaviour for comparison).
  `model="auto"` classifies the regime per arm.
- New `quadrature_nodes()` / `quantile_nodes()` on every channel (exact
  Gauss–Hermite for log-normal); `validation.compare_mc_vs_quadrature` now uses
  `channel.average()`.
- `legacy_baseline.py` (and the identical `notebooks/tfqkd_fso_turbulence_optimized.py`):
  additive section 10e with two-link sampling/quadrature; existing single-link
  functions untouched.
- Examples 01, 03, 04, 05 and `notebooks/instantaneous_adaptive_alpha.py` ported.
- Verified: mean-one fading preserved (1e-12), 2-D rule == 1-D reduction with
  s2_A+s2_B (~1e-7), single-link limit recovered, quadrature == Monte Carlo.
  New tests: `tests/test_two_link.py`. See docs/TWO_LINK_FADING.md.

### v2.0.0 — Multi-Turbulence Framework (this repository)
**Status:** Completed

- New modular `src/turbulence/` package with a polymorphic
  `TurbulenceChannel` interface, implementing:
  - `DeterministicChannel` (Model 1) — reuses validated FSO link budget
    from v1.0.0 unchanged.
  - `LogNormalChannel` (Model 2) — reuses validated log-normal turbulence
    from v1.1.0 unchanged.
  - `GammaGammaChannel` (Model 3) — new, first-principles implementation
    (PDF, alpha/beta from Rytov theory, moments, sampling, CDF).
  - `CorrectedGammaGammaChannel` (Model 4) — new: aperture averaging,
    beam wander, Farid-Hranilovic pointing-error model, all individually
    toggleable and verified to reduce to plain Gamma-Gamma when disabled.
  - `turbulence_factory.create_channel(model=...)` including `"auto"`
    regime selection from the Rytov variance.
- New `TFQKDEngine` (`tfqkd_engine.py`): consumes any `TurbulenceChannel`
  polymorphically; the validated Curty-Azuma-Lo key-rate formula
  (`legacy_baseline.key_rate`, from v0.1.0) is never modified.
- New `monte_carlo.py` (fixed, adaptive-stopping, and benchmark modes)
  and `quadrature.py` (scipy adaptive, Gauss-Legendre, Gauss-Laguerre,
  benchmark) — generalizes and supersedes the log-normal-only quadrature
  work begun in v1.1.1, extending ergodic averaging to all four
  turbulence models. Cross-validated Monte Carlo against quadrature to
  <0.1% agreement on the ergodic TF-QKD key rate across all three
  stochastic models (see docs/VALIDATION.md) — the same MC-vs-quadrature
  relative-error analysis v1.1.1 planned, now completed and extended
  beyond log-normal.
- New `adaptive_alpha.py`: ergodic fixed-alpha optimization
  (`optimize_fixed_alpha`, generalizing v1.1.1's log-normal alpha
  optimization) and instantaneous per-realization adaptive alpha
  (`instantaneous_adaptive_alpha`), generalized to work with any
  `TurbulenceChannel`, not just log-normal.
- New `plotting.py` (consistent journal-style PNG+PDF figures),
  `validation.py` (PDF/CDF/moment/MC-vs-quad checks), `statistics.py`,
  `utils.py`, `constants.py`.
- 23 passing pytest tests (`tests/`) covering distribution sanity,
  moments, MC-vs-quadrature agreement, corrected-Gamma-Gamma reduction,
  adaptive alpha, benchmarking utilities, and numerical-stability edge
  cases (deep fading, large/asymmetric alpha-beta, extreme Rytov
  variance).
- 3 runnable example scripts producing real figures and validation
  tables (`examples/`).
- Full documentation set in `docs/` (THEORY, ATMOSPHERIC_CHANNEL,
  LOGNORMAL, GAMMA_GAMMA, CORRECTED_GAMMA_GAMMA, MONTE_CARLO, QUADRATURE,
  VALIDATION, API_REFERENCE, PERFORMANCE, FIGURES), each with equations,
  assumptions, and references.

**Output**
- PDF/CDF and statistics for all four turbulence models, individually
  and overlaid for comparison (weak/moderate/strong regimes).
- Ergodic TF-QKD SKR vs. distance across all models.
- Fixed vs. ergodically-optimized vs. instantaneous-adaptive alpha
  comparison, with correlation/regression of alpha* against eta.
- Monte Carlo vs. quadrature validation tables (relative error, runtime)
  and sample-count / quadrature-order benchmark tables.

#### Bugs found and fixed during this release's own development
(kept here rather than only in docs/VALIDATION.md, since they're exactly
the kind of regression a future change to `quadrature.py` or a new
turbulence model could reintroduce)

- `scipy.special.kv` overflow for large/asymmetric Gamma-Gamma
  (alpha, beta) — fixed with an `mpmath.besselk` fallback in
  `gamma_gamma._log_kv`.
- Aperture-averaging rescaling of (alpha, beta) could go negative for
  strong averaging under a hold-beta-fixed scheme — replaced with a
  numerically stable symmetric `alpha/A, beta/A` rescaling.
- Quadrature expectations were bounded by `eta_nominal` instead of the
  physical `eta_max=1`, silently discarding roughly half the probability
  mass — fixed via `TurbulenceChannel.eta_max`.
- Adaptive `scipy.integrate.quad` without breakpoint hints could step
  over a narrow Gamma-Gamma density spike and return a confidently-wrong
  value ~2x off from Monte Carlo — fixed by passing explicit `points`
  breakpoints in `quad_expectation`.

---

### v1.1.1 — Analytical Log-Normal Averaging Framework
**Status:** In Progress (superseded/completed by v2.0.0's `quadrature.py`
and `adaptive_alpha.py`, generalized beyond log-normal to all four models)

**Features**
- Derivation of log-normal fading probability density function.
- Numerical quadrature evaluation of ensemble averages.
- Average X-basis gain.
- Average X-basis QBER.
- Average TF-QKD secret key rate.
- Quadrature-based ergodic SKR computation.
- Monte Carlo validation framework.
- Relative error analysis between Monte Carlo and quadrature methods.
- Optimization of signal intensity alpha under log-normal fading.

**Planned Output**
- (carried forward and completed in v2.0.0, see above)

---

### v1.1.0 — Log-Normal Turbulence Integration
**Status:** Completed

**Features**
- Rytov variance computation.
- Log-normal turbulence model.
- Energy-neutral fading implementation.
- Monte Carlo channel realization generation.
- Ergodic TF-QKD secret key rate estimation.
- Weak, moderate, and strong turbulence studies.
- Turbulence statistics validation.
- sigma_R^2 validity analysis for weak-fluctuation regime.

**Output**
- Irradiance probability distributions.
- Turbulence-induced channel statistics.
- Ergodic SKR under atmospheric turbulence.
- Deterministic vs. turbulent channel comparison.

---

### v1.0.0 — Deterministic Free-Space Optical Channel
**Status:** Completed

**Features**
- Atmospheric attenuation model (Beer-Lambert law).
- Gaussian beam propagation.
- Receiver aperture coupling efficiency.
- Geometric diffraction losses.
- End-to-end deterministic FSO transmittance model.

**Output**
- Atmospheric transmittance vs. distance.
- Geometric efficiency vs. distance.
- Deterministic TF-QKD SKR vs. FSO link distance.
- PLOB comparison under deterministic FSO loss.

---

### v0.1.0 — TF-QKD Security Protocol Foundation
**Status:** Completed

**Features**
- Implemented Curty-Azuma-Lo Protocol 3 security proof.
- X-basis gain and QBER calculations (Eqs. 27-31).
- Z-basis gain calculations (Eqs. 32-35).
- Photon-number yield evaluation.
- Cat-state coefficient generation.
- Phase-error upper bound estimation.
- Asymptotic secret key rate (SKR) computation.
- Signal intensity (alpha) optimization.
- PLOB bound comparison.

**Output**
- SKR vs. channel loss.
- Optimized alpha vs. channel loss.
- Security proof validation.
