# TFQKD_MultiTurbulence

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source Python research framework for studying twin-field quantum key distribution (TF-QKD) over free-space optical (FSO) links under atmospheric turbulence.

The project provides a unified interface for deterministic and stochastic channel models, numerical ergodic key-rate calculations, validation tools, and reproducible examples. In the twin-field configuration, Alice and Bob transmit over separate atmospheric links to Charlie; fading is modeled independently on each arm.

> **Research software:** This repository is intended for simulation and research. Results depend on the assumptions and validity ranges of the selected physical models and are not a guarantee of real-world system performance.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Examples and validation](#examples-and-validation)
- [Channel models](#channel-models)
- [Example results](#example-results)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Features

- Deterministic, Log-Normal, Gamma-Gamma, corrected Gamma-Gamma, and beam-broadening channel models.
- A common `TurbulenceChannel` interface and `auto` model selection based on per-arm Rytov variance.
- Ergodic secret-key-rate calculations through `TFQKDEngine`.
- Numerical quadrature, Monte Carlo comparisons, analytical-formula validation, and plotting utilities.
- Documentation of theory, model assumptions, validation, and the public API.

The Curty–Azuma–Lo (2019) security-proof implementation and original Log-Normal baseline are retained in `src/legacy_baseline.py`. The modular engine and channel models interact with that baseline through `TFQKDEngine` and `TurbulenceChannel`.

## Quick start

The following example constructs a channel using automatic model selection and evaluates its ergodic key rate:

```python
import numpy as np

from src.tfqkd_engine import TFQKDEngine
from src.turbulence import create_channel

engine = TFQKDEngine(
    pd=1e-7,
    thetaA=np.arcsin(np.sqrt(0.02)),
    thetaB=-np.arcsin(np.sqrt(0.02)),
    Nmax=4,
)

channel = create_channel(
    "auto",
    eta_nominal=0.05,
    Cn2=1e-15,
    wavelength=1550e-9,
    distance_m=6000,
    receiver_radius=0.15,
    jitter_std=1e-6,
    beam_wander_std=5e-7,
)

rate = engine.ergodic_key_rate(channel, alpha=0.20, method="quadrature")
print(f"Ergodic key rate: {rate:.6e} bits/pulse")
```

For a complete model comparison, including channel statistics and a generated figure, see [`examples/example_01_model_comparison.py`](examples/example_01_model_comparison.py).

## Installation

Requirements: Python 3.9 or later.

Clone the repository and install the project with its runtime dependencies:

```bash
git clone https://github.com/Anand-Ambastha/tf-qkd-private.git
cd tf-qkd-private
python -m pip install -e .
```

To install the test dependencies as well:

```bash
python -m pip install -e ".[dev]"
```

Alternatively, install the dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Examples and validation

Run any example from the repository root. For instance:

```bash
python examples/example_01_model_comparison.py
```

The seven scripts in `examples/` cover model comparison, adaptive parameters, distance sweeps, model validity, and beam broadening. See [`examples/`](examples/) for the full list.

Run the automated test suite with:

```bash
python -m pytest
```

To validate the closed-form turbulence-averaging formulas used in the associated manuscript and progress report:

```bash
python analytical_validation/run_all.py
```

This workflow compares Log-Normal and Gamma-Gamma formulations with numerical quadrature, Monte Carlo simulations, and a high-precision derivative-based reference. Additional information is in [`analytical_validation/README.md`](analytical_validation/README.md).

## Channel models

| Model | Intended use |
|---|---|
| Deterministic | Reference channel without atmospheric turbulence |
| Log-Normal | Weak turbulence; use when the per-link Rytov variance is below one |
| Gamma-Gamma | Moderate-to-strong turbulence |
| Corrected Gamma-Gamma | Finite receiver aperture, aperture averaging, beam wander, and pointing error |
| Beam-broadening variants | Models that include long-term beam broadening and associated mean-power loss |

The `auto` option selects a model based on the Rytov variance of an individual arm. A model can also be selected explicitly. See the theory and model documentation in [`docs/`](docs/), including [THEORY.md](docs/THEORY.md), [GAMMA_GAMMA.md](docs/GAMMA_GAMMA.md), [CORRECTED_GAMMA_GAMMA.md](docs/CORRECTED_GAMMA_GAMMA.md), [TWO_LINK_FADING.md](docs/TWO_LINK_FADING.md), and [BEAM_BROADENING.md](docs/BEAM_BROADENING.md).

## Example results

The table shows the simulated maximum total Alice–Bob distance at which the ergodic secret key rate remains above \(10^{-9}\) bits per pulse. These are outputs for the parameter set below, not universal performance figures.

Parameters: wavelength 1550 nm, beam waist 5 cm, receiver radius 15 cm, and attenuation 0.2 dB/km. The weak, moderate, and strong cases use mean \(C_n^2\) values of \(10^{-17}\), \(10^{-15}\), and \(10^{-13}\,\mathrm{m}^{-2/3}\), respectively.

| Model | Weak | Moderate | Strong |
|---|---:|---:|---:|
| No turbulence | 252.0 km | 252.0 km | 252.0 km |
| Gamma-Gamma, per arm | 291.7 km | 293.2 km | 291.7 km |
| Log-Normal, per arm | 317.6 km | 132.7 km | 13.1 km |
| Gamma-Gamma with beam broadening, per arm | 287.9 km | 198.6 km | 81.9 km |

Interpret these results within each model's validity range. The Log-Normal approximation is intended for per-link Rytov variance no greater than one; its predictions can diverge from Gamma-Gamma outside that regime. In these simulations, standard Gamma-Gamma range changes little with the specified turbulence strength and can exceed the deterministic baseline near the rate cutoff because high-transmission fading events contribute to the average. Beam broadening adds mean-power loss and reduces the predicted range as turbulence increases. Model assumptions and limitations are described in [BEAM_BROADENING.md](docs/BEAM_BROADENING.md).

## Project layout

- `src/` — TF-QKD engine, channel models, numerical methods, statistics, and plotting utilities.
- `tests/` — automated tests for numerical and physical model behavior.
- `examples/` — runnable examples and figure-generation scripts.
- `analytical_validation/` — independent checks of analytical averaging formulas.
- `docs/` — theory, derivations, API reference, validation notes, and performance documentation.
- `results_v2.3.0/` — simulation outputs, reports, and logs associated with this version of the work.
- `old_results_v2.1.0_pre_gg_fix/` — earlier outputs retained for comparison.

## Limitations

- Inner and outer turbulence scales are included as a first-order correction to Rytov variance, rather than through a full von Kármán integral.
- The spherical-wave Gamma-Gamma formulation uses \(\Theta=1\), rather than a general Gaussian-beam treatment.
- The beam-broadening model is based on weak-to-moderate turbulence theory and does not saturate at high \(C_n^2\); it may overestimate losses under strong turbulence.
- Near the outage threshold, rare high-transmission events can make Monte Carlo estimates noisy. Typical validation points show agreement between Monte Carlo and quadrature within 0.1%; near-threshold discrepancies of approximately 1–3% have been observed with \(10^6\) to \(10^7\) samples.

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the validation methodology. Recheck numerical integration and distribution changes against Monte Carlo and independent quadrature.

## Contributing

Contributions, bug reports, and scientific corrections are welcome. Before opening a pull request:

1. Install development dependencies and run `python -m pytest`.
2. For a new turbulence model, include its channel implementation, factory entry, tests (including PDF/CDF, moments, and numerical agreement), and a derivation in `docs/`.
3. Cross-check changes to quadrature or a model's PDF/CDF against Monte Carlo as described in [`docs/VALIDATION.md`](docs/VALIDATION.md).
4. Add a relevant entry to `CHANGELOG.md`.
5. Write generated outputs to a new `results_vX.Y.Z/` directory; do not overwrite archived result folders.

The validated baseline in `src/legacy_baseline.py` is intentionally retained. Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for project-specific development guidance.

## Citation

If you use this software, please cite the project using [`CITATION.cff`](CITATION.cff), and cite the underlying methods relevant to your work. Key references include:

1. Curty, M., Azuma, K., and Lo, H.-K. “Simple security proof of twin-field type QKD protocol.” *npj Quantum Information* 5, 64 (2019). [arXiv:1807.07667](https://arxiv.org/abs/1807.07667).
2. Andrews, L. C., and Phillips, R. L. *Laser Beam Propagation through Random Media*. 2nd ed., SPIE Press, 2005.
3. Al-Habash, M. A., Andrews, L. C., and Phillips, R. L. “Mathematical model for the irradiance probability density function of a laser beam propagating through turbulent media.” *Optical Engineering* 40(8), 1554–1562 (2001).
4. Farid, A. A., and Hranilovic, S. “Outage Capacity Optimization for Free-Space Optical Links with Pointing Errors.” *Journal of Lightwave Technology* 25(7), 1702–1710 (2007).
5. Goodman, J. W. *Statistical Optics*. Wiley, 1985/2015.
6. Korotkova, O. *Random Light Beams: Theory and Applications*. CRC Press, 2013.

## Acknowledgements

This work was carried out by Anand Kumar during a research internship at the Centre for Applied Research in Electronics (CARE), IIT Delhi, from June to December 2026, under the supervision of Dr. Neelkanth Kundu and Dr. Ramniwas Meena. The author acknowledges the support of the National Quantum Mission and the Samngya Foundation.

## License

This project is distributed under the [MIT License](LICENSE). Copyright © 2026 Anand Kumar.