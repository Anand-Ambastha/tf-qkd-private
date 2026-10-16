# VALIDATION.md — Numerical Validation

All numbers below are reproduced by running:

```
python3 examples/example_03_distance_sweep_and_validation.py
```

or directly via `src/validation.full_validation_report`, and are backed
by the automated tests in `tests/` (23 tests, all passing — run
`pytest tests/ -q`).

## 1. PDF normalization, CDF monotonicity/endpoint

One representative operating point per model, chosen inside that model's
validity regime (log-normal at sigma_R^2 << 1, Gamma-Gamma at moderate
sigma_R^2, corrected Gamma-Gamma at strong sigma_R^2):

| Model | sigma_R^2 | integral of pdf | CDF monotonic | CDF(eta_max) |
|---|---|---|---|---|
| Log-Normal | 0.015 | 1.0000000000 | yes | 1.0000 |
| Gamma-Gamma | 1.595 | 0.9999999984 | yes | 1.0000 |
| Corrected Gamma-Gamma | 13.564 | 0.9999075681 | yes | 1.0000 |

(Corrected Gamma-Gamma's pdf integral is the only one not accurate to
10 decimal places because it is built from a finite, 200,000-sample
empirical histogram rather than a closed-form density — see
docs/CORRECTED_GAMMA_GAMMA.md Section 4. 0.9999 is within the
`atol=5e-3` tolerance used in `tests/test_distributions.py`.)

## 2. Sampling moments vs. reference

| Model | mean (MC, n=2e5) | mean (reference) | rel. error |
|---|---|---|---|
| Log-Normal | 0.100003 | 0.100000 | 0.003% |
| Gamma-Gamma | 0.050144 | 0.050000 | 0.29% |
| Corrected Gamma-Gamma | 0.010005 | 0.009997 | 0.08% |

## 3. Monte Carlo vs. quadrature agreement on the ergodic TF-QKD key rate

E_eta[ R(alpha=0.20, eta) ] under the unmodified Curty-Azuma-Lo key-rate
formula, MC with n=2e5 samples vs. adaptive quadrature:

| Model | MC estimate | Quadrature estimate | Relative error |
|---|---|---|---|
| Log-Normal | 1.01056e-02 | 1.01049e-02 | 0.007% |
| Gamma-Gamma | 6.38089e-03 | 6.37717e-03 | 0.058% |
| Corrected Gamma-Gamma | 2.75145e-03 | 2.74949e-03 | 0.071% |

This agreement is the headline validation result: two independent
numerical methods, implemented independently (`monte_carlo.py`,
`quadrature.py`), converge on the same answer for all four models to
well under 0.1%, which is inside Monte Carlo's own sampling noise at
n=2e5.

**This was not true on the first implementation.** Two real bugs were
caught and fixed by this exact cross-check during development, kept here
as a record for anyone modifying `quadrature.py` or adding a new
turbulence model:

1. Quadrature originally integrated E[f(eta)] only over [0, eta_nominal]
   instead of [0, eta_max=1]. Since fading pushes eta above eta_nominal
   roughly as often as below it, this discarded close to half the
   probability mass. Symptom: quadrature systematically ~2x lower than
   Monte Carlo, consistently across all three stochastic models.
2. Even after fixing the integration bound, plain adaptive
   `scipy.integrate.quad` stepped over the Gamma-Gamma density's narrow
   spike (the density's characteristic width can be a small fraction of
   the full [0,1] domain when eta_nominal is small) and returned a
   confidently-wrong value with a deceptively tight reported error
   bound. Fixed by supplying explicit interior breakpoints near the
   density's mass (`quad_expectation(..., points=...)`,
   `TurbulenceChannel.average()`); see docs/QUADRATURE.md.

Both are exactly the kind of error that would have silently produced a
wrong "Gamma-Gamma gives you a 2x pessimistic SKR estimate vs Monte
Carlo" conclusion in a paper if only one method had been used, or if the
two methods had been compared less carefully (e.g., at a single
operating point that happened not to expose the bug). Cross-validating
independent numerical methods against each other, at multiple operating
points spanning the three turbulence regimes, is what surfaced both.

## 4. Corrected Gamma-Gamma reduces to plain Gamma-Gamma

`tests/test_corrected_gamma_gamma_reduces.py`: with
`enable_aperture_averaging=False, enable_pointing_error=False`, alpha and
beta match `GammaGammaChannel` at the same (Cn2, wavelength, distance_m)
to floating-point precision (`np.isclose`, default tolerance), and the
resulting mean/variance agree to <3%/<10% (the latter looser because the
corrected channel's statistics are computed from a finite Monte Carlo
cache rather than the closed form).

## 5. Numerical stability / extreme parameters

- **Deep fading** (eta down to 1e-30): key rate stays finite and
  clips to exactly 0 below threshold (`test_deep_fading_zero_key_rate`).
- **eta = 1** (no loss): finite, non-negative
  (`test_zero_fading_eta_equals_one`).
- **Large, strongly asymmetric (alpha, beta)** (up to 500, 8): the
  Gamma-Gamma PDF stays finite via the `mpmath` fallback in `_log_kv`
  (`test_gamma_gamma_large_alpha_beta_no_overflow`) — plain
  `scipy.special.kv` overflows in this regime (see docs/GAMMA_GAMMA.md,
  Implementation Note).
- **Rytov variance from 1e-6 to 50**: alpha, beta stay positive and
  finite across the whole weak-to-saturation range
  (`test_scintillation_index_extreme_sigma`).

## 6. Known asymptotic limits checked

- Gamma-Gamma variance formula Var(I) = 1/alpha + 1/beta + 1/(alpha beta)
  verified directly against a 500,000-sample Monte Carlo product of two
  Gamma variates, for four (alpha, beta) pairs spanning symmetric,
  asymmetric, large, and near-unity shape parameters
  (`test_gamma_gamma_variance_formula`).
- Log-normal channel is energy-neutral: E[eta] = eta_nominal to <2%
  at n=3e5 samples (`test_lognormal_energy_neutral`).
- Distance sweep: ergodic SKR is non-increasing with distance for a
  monotonically decreasing nominal transmittance
  (`test_distance_sweep_monotonic_decrease`).

## Running the validation suite yourself

```bash
pip install -r requirements.txt
pytest tests/ -q                                    # 23 tests, ~50s
python3 examples/example_03_distance_sweep_and_validation.py
```


## Additions in v2.2.0 and v2.3.0

- Gamma-Gamma exponent regression (`tests/test_moments.py`). SI(0.1) is about 0.099, SI between sigma_R^2 = 5 and
  10 lies in 1.15 to 1.30, SI(1000) lies in 1.0 to 1.1, and alpha(1) = 4.39, beta(1) = 2.56.
- Quadrature convergence with the cosine-spaced nodes. At the outage threshold the ergodic key rate moves by
  less than 0.5 % between n = 200 and n = 1600, and it agrees with 10^6 to 10^7 Monte Carlo samples to about
  1 to 3 %.
- Per-arm maximum range, Gamma-Gamma against Log-Normal: `GAMMA_GAMMA_PER_ARM_REPORT.txt`. It includes the Monte
  Carlo cross-check and the agreement between the two models inside the Log-Normal validity range.
- Beam broadening: `BEAM_BROADENING_REPORT.txt` (range with and without broadening, convergence for
  n = 200, 400 and 800, and a Monte Carlo check at each maximum-range point) and `tests/test_beam_broadening.py`.
