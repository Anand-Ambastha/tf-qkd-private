# API_REFERENCE.md

Full parameter-level docstrings live in the source (every public
function/class has a docstring with a physical/numerical explanation);
this document is a navigable index by module, not a duplicate of that
text. See docs/THEORY.md, GAMMA_GAMMA.md, CORRECTED_GAMMA_GAMMA.md for
the underlying derivations.

## `src.turbulence` (package)

`from src.turbulence import create_channel, classify_regime, TurbulenceChannel, DeterministicChannel, LogNormalChannel, GammaGammaChannel, CorrectedGammaGammaChannel`

### `TurbulenceChannel` (abstract base, `turbulence/base.py`)
| Member | Signature | Description |
|---|---|---|
| `eta_nominal` | attribute | deterministic (fading-free) transmittance |
| `eta_max` | property | physical upper bound of eta's support (1.0) |
| `pdf(eta)` | abstract | probability density of eta |
| `cdf(eta)` | abstract | CDF of eta |
| `sample(n, rng=None)` | abstract | draw n i.i.d. samples |
| `statistics(n_mc=20000, rng=None)` | dict | mean, variance, scintillation_index, std |
| `outage_probability(eta_th)` | float | P(eta <= eta_th) |
| `average(func, method="quadrature"\|"monte_carlo", n_mc=20000, rng=None, quad_kwargs=None)` | float | E[func(eta)] |
| `average_skr(skr_func, ...)` | float | alias of `average` |
| `monte_carlo(func, n=20000, rng=None)` | float | alias, MC only |
| `quadrature(func, **kwargs)` | float | alias, quadrature only |
| `optimize_alpha(pd, thetaA, thetaB, Nmax=4, ...)` | (alpha*, R*) | best fixed alpha for this channel |
| `plot_pdf(ax=None, **kwargs)` / `plot_cdf(...)` | matplotlib Axes | thin wrappers around `plotting.py` |

### `DeterministicChannel(alpha_atm, distance_km, wavelength, w0, receiver_radius)`
Model 1. `.fso_breakdown` exposes the legacy per-term loss breakdown dict.

### `LogNormalChannel(eta_nominal, Cn2, wavelength, distance_m, seed=None)`
Model 2. `.sigma_R2`, `.mu_X`, `.sigma_X2`, `.sigma_X` attributes.

### `GammaGammaChannel(eta_nominal, Cn2, wavelength, distance_m, wave_type="plane"|"spherical", Theta=1.0, seed=None)`
Model 3. `.alpha`, `.beta`, `.sigma_R2` attributes.
Module-level helpers: `alpha_beta_plane_wave(sigma_R2)`,
`alpha_beta_spherical_wave(sigma_R2, Theta=1.0)`, `gamma_gamma_pdf(I, alpha, beta)`,
`gamma_gamma_moments(alpha, beta)`.

### `CorrectedGammaGammaChannel(eta_nominal, Cn2, wavelength, distance_m, receiver_radius, jitter_std=0, beam_wander_std=0, w0=0.05, wave_type="plane", Theta=1.0, inner_scale=None, outer_scale=None, enable_aperture_averaging=True, enable_pointing_error=True, n_mc_cache=200000, seed=None)`
Model 4. `.alpha`, `.beta` (aperture-averaged), `.alpha0`, `.beta0` (bare
Gamma-Gamma, pre-correction), `.A_factor`, `.gamma_pe`, `.A0_pe`.
Module-level helpers: `aperture_averaging_factor(D, wavelength, L)`,
`pointing_error_params(receiver_radius, w_Lt, jitter_std)`.

### `create_channel(model, eta_nominal, Cn2=None, wavelength=None, distance_m=None, distance_km=None, alpha_atm=None, w0=None, receiver_radius=None, jitter_std=0.0, beam_wander_std=0.0, wave_type="plane", **kwargs)`
`model in {"deterministic","lognormal","gamma_gamma","gamma_gamma_corrected","auto"}`.

### `classify_regime(sigma_R2) -> "weak"|"moderate"|"strong"`

## `src.tfqkd_engine.TFQKDEngine(pd, thetaA, thetaB, Nmax=4)`
| Method | Description |
|---|---|
| `instantaneous_key_rate(alpha, eta)` | R(alpha, eta), validated CAL formula, clipped >= 0 |
| `optimize_alpha_at(eta)` | (alpha*, R*) at one fixed eta |
| `ergodic_key_rate(channel, alpha, method=..., n_mc=..., rng=...)` | E_eta[R(alpha,eta)] |
| `optimize_fixed_alpha(channel, ...)` | best single alpha for a channel |
| `instantaneous_adaptive_analysis(channel, n_samples=20000, alpha_fixed=0.20, rng=None)` | full per-realization adaptive-alpha study |
| `distance_sweep(distances_km, channel_factory, alpha=0.20, method=..., n_mc=..., rng=...)` | R vs distance for any channel family |

## `src.adaptive_alpha`
- `optimize_fixed_alpha(channel, pd, thetaA, thetaB, Nmax=4, method=..., n_mc=..., rng=None, alpha_bounds=(1e-3,0.3))`
- `instantaneous_adaptive_alpha(eta_samples, pd, thetaA, thetaB, Nmax=4, alpha_bounds=..., alpha_fixed=0.20)`
  returns a dict with `eta_samples, alpha_star, R_adaptive, R_fixed, alpha_fixed,
  stats_adaptive, stats_fixed, pct_improve, zero_fixed_mask, correlation`.

## `src.monte_carlo`
- `mc_expectation(channel, func, n, rng=None, seed=None)`
- `mc_expectation_adaptive(channel, func, tol=1e-3, batch=5000, n_max=2e6, seed=None, z=1.96)`
- `benchmark_monte_carlo(channel, func, sample_grid=None, seed=None, n_repeats=3)`

## `src.quadrature`
- `quad_expectation(pdf, func, a, b, epsabs=1e-10, epsrel=1e-8, limit=200, points=None)`
- `gauss_legendre_expectation(pdf, func, a, b, n=64)`
- `gauss_laguerre_expectation(density_over_exp, func, n=64)`
- `benchmark_quadrature(channel, func, a=None, b=None, n_values=(8,16,32,64,128), reference_method="scipy_quad")`

## `src.statistics`
`availability`, `outage_probability`, `rmse`, `mean_absolute_error`, `summary_row`.

## `src.validation`
`check_pdf_normalization`, `check_cdf_monotonic`, `check_sampling_moments`,
`compare_mc_vs_quadrature`, `full_validation_report(engine, channels, seed=0)`.

## `src.plotting`
`plot_channel_pdf`, `plot_channel_cdf`, `plot_model_comparison_pdf`,
`plot_distance_sweep`, `plot_mc_convergence`, `plot_quadrature_benchmark`
— every figure function saves both `.png` and `.pdf`.

## `src.utils`
`safe_log`, `clip01`, `RunningStats` (Welford's online mean/variance),
`confidence_interval`, `relative_error`.

## `src.constants`
`SIGMA_R2_WEAK_MODERATE=1.0`, `SIGMA_R2_MODERATE_STRONG=10.0`,
`DEFAULT_TFQKD_PARAMS`, `DEFAULT_FSO_PARAMS`, `DEFAULT_MC_SEED=42`,
`DEFAULT_MC_SAMPLES=50000`, `MC_SAMPLE_GRID`, `EPS`.

## `src.legacy_baseline`
The unmodified, validated Curty-Azuma-Lo TF-QKD + deterministic FSO +
log-normal turbulence implementation this whole framework builds on top
of. Not part of the new API surface — consult it directly only if you
need to verify what a wrapped function does internally.
