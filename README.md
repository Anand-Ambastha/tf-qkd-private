# TFQKD_MultiTurbulence

A Python implementation of twin-field quantum key distribution (TF-QKD) over free-space optical (FSO) links, supporting multiple atmospheric turbulence models through a unified interface.

The framework includes deterministic, Log-Normal, Gamma-Gamma, corrected Gamma-Gamma, and turbulence models incorporating beam broadening. In a twin-field configuration, Alice and Bob transmit independently to Charlie, so atmospheric fading is modeled separately for each link. Numerical key-rate calculations are validated against Monte Carlo simulations.

The Curty–Azuma–Lo (2019) security proof and the original Log-Normal implementation are retained in `src/legacy_baseline.py`. Other components interact with this baseline through the `TFQKDEngine` and `TurbulenceChannel` interfaces.

## Main Results

The following table reports the maximum total Alice–Bob distance at which the ergodic secret key rate remains above \(10^{-9}\) bits per pulse.

Simulation parameters: wavelength of 1550 nm, beam waist of 5 cm, receiver radius of 15 cm, and attenuation of 0.2 dB/km. The turbulence strengths are represented by mean \(C_n^2\) values of \(10^{-17}\), \(10^{-15}\), and \(10^{-13}\,\mathrm{m}^{-2/3}\) for weak, moderate, and strong turbulence, respectively.

| Model | Weak | Moderate | Strong |
|---|---:|---:|---:|
| No turbulence | 252.0 km | 252.0 km | 252.0 km |
| Gamma-Gamma, per arm | 291.7 km | 293.2 km | 291.7 km |
| Log-Normal, per arm | 317.6 km | 132.7 km | 13.1 km |
| Gamma-Gamma with beam broadening, per arm | 287.9 km | 198.6 km | 81.9 km |

These results should be interpreted within the validity limits of each channel model. The Log-Normal approximation is intended for cases where the Rytov variance of an individual link does not exceed one. Within that regime, its predictions are close to those of the Gamma-Gamma model, but the two can diverge outside it.

The standard Gamma-Gamma model exhibits limited sensitivity to the specified turbulence strength in these simulations. Its predicted range can exceed the deterministic baseline because occasional high-transmission fading events contribute to the average key rate near the cutoff. Beam broadening introduces additional mean-power loss, reducing the predicted range as turbulence increases. The two Gamma-Gamma results, with and without beam broadening, provide an upper and lower estimate under strong turbulence, subject to the assumptions described in `docs/BEAM_BROADENING.md`.

## Supported Channel Models

| Model | Application |
|---|---|
| Deterministic | Baseline without atmospheric turbulence |
| Log-Normal | Weak turbulence, with Rytov variance below one |
| Gamma-Gamma | Moderate-to-strong turbulence |
| Corrected Gamma-Gamma | Finite receiver aperture, aperture averaging, beam wander, and pointing error |
| Beam-broadening variants | Models incorporating long-term beam broadening and associated mean-power loss |

The `auto` model-selection option chooses a channel model based on the Rytov variance of one link. Users can also specify a model directly.

The theoretical assumptions and derivations are documented in `docs/THEORY.md`, `docs/GAMMA_GAMMA.md`, `docs/CORRECTED_GAMMA_GAMMA.md`, `docs/TWO_LINK_FADING.md`, and `docs/BEAM_BROADENING.md`.

## Analytical Formula Validation

The `analytical_validation/` module evaluates closed-form turbulence-averaging formulas used in the associated manuscript and progress report.

It compares the Log-Normal formulation, including Hermite and Euler–Stirling expansions, and the Gamma-Gamma formulation, including a cumulant expansion with aperture averaging, against independent numerical quadrature, Monte Carlo simulations, and a 60-digit derivative-based reference calculation. The module generates validation tables, figures, and a PDF report.

To run the validation workflow:

```bash
python3 analytical_validation/run_all.py
```

Additional details are available in `analytical_validation/README.md`.

## Validation and Numerical Accuracy

The project includes tests for probability density and cumulative distribution functions, statistical moments, quadrature, Monte Carlo agreement, independent link fading, and beam broadening.

According to the project's validation documentation, Monte Carlo and quadrature estimates of the ergodic key rate agree to within 0.1% at typical operating points. Near the outage threshold, where rare high-transmission events can dominate the estimated rate, discrepancies are larger, approximately 1–3% for simulations using \(10^6\) to \(10^7\) samples.

The validation process also identified numerical integration issues involving truncated integration bounds and adaptive quadrature missing a narrow probability-density peak. These checks should be repeated after modifications to the quadrature implementation or channel probability distributions.

## Installation

Requirements: Python 3.9 or later, NumPy, SciPy, Matplotlib, mpmath, and pytest.

```bash
git clone https://github.com/<your-username>/TFQKD_MultiTurbulence.git
cd TFQKD_MultiTurbulence
pip install -r requirements.txt
python -m pytest tests
```

Installing the package in editable mode is optional:

```bash
pip install -e .
```

## Repository Structure

- `src/`: TF-QKD engine, channel models, numerical methods, statistics, and validation utilities.
- `docs/`: Theory, derivations, validation notes, performance benchmarks, and API documentation.
- `examples/`: Seven example scripts covering model comparisons, parameter optimization, distance sweeps, validity limits, and beam broadening.
- `tests/`: Automated tests for the numerical and physical models.
- `results_v2.3.0/`: Current simulation figures, reports, and logs.
- `old_results_v2.1.0_pre_gg_fix/`: Earlier outputs retained for comparison.
- `analytical_validation/`: Analytical formula validation scripts and generated results.

## Limitations

- Figures are exported as PNG and PDF.
- Inner and outer turbulence scales are included as a first-order correction to the Rytov variance rather than through a full von Kármán integral.
- The spherical-wave Gamma-Gamma formulation uses \(\Theta=1\), rather than a general Gaussian-beam treatment.
- The beam-broadening model is based on weak-to-moderate turbulence theory and does not saturate at high \(C_n^2\). It may therefore overestimate losses under strong turbulence, making the corresponding predicted range a conservative lower estimate under the stated assumptions.

These limitations should be considered when interpreting results beyond the regimes for which the models were developed.

## References

1. Curty, M., Azuma, K., and Lo, H.-K. “Simple security proof of twin-field type QKD protocol.” *npj Quantum Information* 5, 64 (2019). arXiv:1807.07667v2.
2. Andrews, L. C., and Phillips, R. L. *Laser Beam Propagation through Random Media*. 2nd ed., SPIE Press, 2005.
3. Al-Habash, M. A., Andrews, L. C., and Phillips, R. L. “Mathematical model for the irradiance probability density function of a laser beam propagating through turbulent media.” *Optical Engineering* 40(8), 1554–1562 (2001).
4. Farid, A. A., and Hranilovic, S. “Outage Capacity Optimization for Free-Space Optical Links with Pointing Errors.” *Journal of Lightwave Technology* 25(7), 1702–1710 (2007).
5. Goodman, J. W. *Statistical Optics*. Wiley, 1985/2015.
6. Korotkova, O. *Random Light Beams: Theory and Applications*. CRC Press, 2013.

## Authors and Acknowledgements

**Author:** Anand Kumar

**Research affiliation:** Centre for Applied Research in Electronics (CARE), IIT Delhi.

**Supervision:** Dr. Neelkanth Kundu and Dr. Ramniwas Meena.

This work was carried out during a research internship at CARE, IIT Delhi, from June to December 2026. The author acknowledges the guidance of the supervisors and the support of the National Quantum Mission and the Samngya Foundation.

## License

MIT License. Copyright © 2026 Anand Kumar.

See `LICENSE` for details.