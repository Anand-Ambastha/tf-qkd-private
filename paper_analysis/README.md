# gg_extension

Gamma-Gamma analysis of the TF-QKD key rate with independent fading on each arm. It covers the CSI and no-CSI averaging conventions, amplitude optimisation, finite decoy intensities, beam broadening, and the analytic cumulant expansion against quadrature. It imports `src/` unchanged and needs numpy, scipy, matplotlib and pandas. The finite-decoy linear program uses `scipy.optimize.linprog` (HiGHS), so cvxpy is not required.

Run from the repo root with `sh paper_analysis/gg_extension/run_all.sh`. Outputs go to `results/tables` and `results/figures`.

| script | content |
|---|---|
| `e1_sweeps.py` | key rate vs distance (deterministic, Gamma-Gamma, log-normal; CSI and no CSI) and maximum ranges |
| `e23_decomp_opt.py` | decomposition of the fading gain, alpha*(eta), fixed / ergodic / instantaneous alpha |
| `e45_decoy.py` | finite-decoy bounds, decoy-set comparison, finite decoy under fading |
| `analytic_vs_quad.py` | analytic expansion vs quadrature (weak and strong, distance table) |
| `gg_master.py` | master table: plain vs beam broadening, infinite vs finite decoy, fixed / optimised alpha, quadrature vs analytic, plus ranges |
| `plot_all.py` | figures from the tables |

## Conventions

- CSI: the key rate is evaluated per realisation and averaged, E[R(eta)]. This is what `src/` computes.
- No CSI: the key rate is built from fading-averaged gains and yields only.
- Residual: before v2.4.1, `src/legacy_baseline.residual_j` subtracted a square set of photon numbers while the sum over S_j is triangular; it is fixed in `src`. Columns and keys named `repo` use the old residual (`legacy_residual` in `tfqkd_ext.py`) and reproduce the previously published numbers.
- Beam broadening is the long-term beam of `src/turbulence/beam_broadening.py`, which already includes wander. Pointing jitter is not modelled.
- Analytic derivatives are 5-point finite differences in ln(eta), step 0.1.
- Range is the largest distance with R > 1e-9 bit/pulse. Range searches use 150 quadrature nodes per arm, accurate to about 1% near cutoff.

## Not covered

Finite-key statistics, amplitude imbalance between the arms, pointing jitter, aperture averaging in the arm model.
