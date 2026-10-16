# MONTE_CARLO.md — Monte Carlo Expectation Engine

Implementation: `src/monte_carlo.py`.

## Fixed-sample estimation: `mc_expectation`

Draws n i.i.d. samples eta_1..eta_n from a channel and returns the sample
mean, sample standard deviation, and standard error of the mean (SEM)
for E[func(eta)]. Used directly by `TurbulenceChannel.average(...,
method="monte_carlo")` and throughout the test suite as the independent
cross-check against quadrature (`tests/test_mc_vs_quadrature.py`).

## Adaptive stopping: `mc_expectation_adaptive`

Draws samples in batches (default 5,000), updating a running mean and
variance online via Welford's algorithm (`utils.RunningStats` — O(1)
memory regardless of how many samples are ultimately drawn), and stops
once the 95% confidence half-width z*SEM falls below a target absolute
tolerance `tol`, or a sample cap `n_max` is reached. This is the mode to
reach for when you want "just tell me the answer to 3 significant
figures" rather than committing to a specific N in advance.

## Sample-count benchmark: `benchmark_monte_carlo`

Sweeps `constants.MC_SAMPLE_GRID = [100, 500, 1e3, 5e3, 1e4, 5e4, 1e5,
5e5]` (or a custom grid), running `n_repeats` independent-seed trials at
each N and reporting: mean estimate, spread of the estimate across
repeats (an empirical proxy for MC standard error at that N — cross-check
against the theoretical SEM from a single run), and mean wall-clock
runtime. This is what `plotting.plot_mc_convergence` visualizes and what
docs/PERFORMANCE.md's Monte Carlo table is built from.

## Design notes

- **Seeding**: every entry point accepts an explicit `rng` (a
  `numpy.random.Generator`) or `seed`; `constants.DEFAULT_MC_SEED = 42`
  is used when neither is given, so repeated runs are reproducible unless
  the caller deliberately re-seeds.
- **Where the randomness actually lives**: `channel.sample(n, rng=rng)`
  on the concrete `TurbulenceChannel` subclass — `monte_carlo.py` itself
  contains no turbulence-model-specific code, consistent with the
  polymorphism requirement (`TFQKDEngine`/Monte Carlo/quadrature never
  branch on which model they were handed).
