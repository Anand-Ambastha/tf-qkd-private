# notebooks/

No Jupyter notebooks are bundled in this delivery (see README.md,
"Scope and honest limitations") — every workflow in this repository is
also available as a plain runnable script under `examples/`, which is
easier to diff, test, and run in CI than a notebook. To turn any example
into a notebook: `pip install jupytext` then
`jupytext --to notebook examples/example_01_model_comparison.py`.
