# Contributing

1. `pip install -r requirements.txt` (Python 3.9+), then `python -m pytest tests`; every test should pass (about 45 s).
2. Please leave `src/legacy_baseline.py` alone. It is the validated TF-QKD security-proof / log-normal baseline,
   reused as it was. New physics goes in `src/turbulence/` behind the `TurbulenceChannel` interface.
3. A new turbulence model needs: a channel class, a factory entry in `src/turbulence/turbulence_factory.py`,
   tests (PDF/CDF sanity, moments, Monte-Carlo vs quadrature agreement, reduction to the base model), and a
   derivation in `docs/`.
4. Any change to `quadrature.py` or a model's `pdf()`/`cdf()` must be cross-checked against Monte Carlo
   (see `docs/VALIDATION.md`); numbers produced by the examples are only trustworthy after that check.
5. Add a CHANGELOG entry. Generated outputs go in `results_vX.Y.Z/`; never overwrite an archived results folder.
