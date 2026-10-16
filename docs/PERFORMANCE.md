# PERFORMANCE.md — Benchmarks

Measured on the development container (single CPU core, no GPU), Python
3.x + NumPy/SciPy. Reproduce with:

```python
from src.turbulence import GammaGammaChannel
from src.tfqkd_engine import TFQKDEngine
from src.monte_carlo import benchmark_monte_carlo
from src.quadrature import benchmark_quadrature

engine = TFQKDEngine(1e-7, ..., ..., Nmax=4)
ch = GammaGammaChannel(0.05, 3e-15, 1550e-9, 6000)
func = lambda eta: engine.instantaneous_key_rate(0.20, eta)
benchmark_monte_carlo(ch, func, sample_grid=[...])
benchmark_quadrature(ch, func)
```

## Monte Carlo: sample count vs. accuracy vs. runtime

Estimating E[R(alpha=0.20, eta)] for a Gamma-Gamma channel
(sigma_R^2=1.60), 3 repeats per N:

| N | estimate | spread across repeats | runtime |
|---|---|---|---|
| 100 | 6.257e-03 | 6.22e-05 | 3.7 ms |
| 500 | 6.272e-03 | 1.86e-04 | 3.5 ms |
| 1,000 | 6.369e-03 | 4.69e-05 | 1.8 ms |
| 5,000 | 6.365e-03 | 3.66e-05 | 15.7 ms |
| 10,000 | 6.368e-03 | 1.36e-05 | 14.8 ms |
| 50,000 | 6.370e-03 | 1.02e-05 | 82.5 ms |
| 100,000 | 6.376e-03 | 1.19e-05 | 204.1 ms |
| 500,000 | 6.379e-03 | 2.18e-06 | 915.5 ms |

The reference (quadrature) value is 6.377166e-03 — Monte Carlo is within
~0.1% of it by N=10,000 and runtime scales linearly with N as expected
(dominated by evaluating the CAL key-rate formula elementwise over the
sample array, not by the sampling itself).

## Quadrature: Gauss-Legendre order vs. accuracy vs. runtime

Same channel/function, `scipy.integrate.quad` as ground truth
(6.377165616e-03, 173.9 ms — most of that is Python-level overhead per
integrand evaluation from the elementwise wrapping in `quad_expectation`,
not the quadrature rule itself):

| Order n | relative error | runtime |
|---|---|---|
| 8 | 1.70e-02 | 1.9 ms |
| 16 | 9.08e-04 | 1.0 ms |
| 32 | 6.90e-06 | 1.3 ms |
| 64 | 6.71e-08 | 2.7 ms |
| 128 | 7.60e-10 | 3.6 ms |

**Takeaway**: for a smooth integrand like this (key rate against a
well-resolved Gamma-Gamma density, with the narrow-spike pitfall already
handled by breakpoints — see docs/QUADRATURE.md), a 32-64 point
Gauss-Legendre rule matches adaptive `scipy.quad` to 6-8 significant
figures at roughly 100x lower wall-clock cost, and Monte Carlo needs on
the order of 10^4-10^5 samples to reach comparable (but never
better-than-1/sqrt(N)) accuracy. For single ergodic-rate evaluations,
quadrature is the clear choice; Monte Carlo remains necessary wherever a
closed-form or tractable pdf isn't available (e.g. the corrected
Gamma-Gamma model's composite pdf, docs/CORRECTED_GAMMA_GAMMA.md) or
where the sampling process itself is the object of study (e.g. the
instantaneous adaptive-alpha analysis, which needs actual per-realization
eta draws).

## Where the time actually goes

For both methods, the dominant cost at realistic operating points is
evaluating `legacy_baseline.key_rate` (the CAL security-proof formula,
which itself involves a truncated cat-state sum up to `Nmax`), not the
turbulence sampling/quadrature machinery around it. Increasing `Nmax`
(security-proof truncation order) affects both methods' runtime
proportionally; it is a security-proof accuracy/runtime knob, not a
turbulence-model one, and is left untouched by this framework (passed
through unchanged from `TFQKDEngine.__init__`).
