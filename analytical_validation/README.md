# analytical_validation

Scripts that test the analytical turbulence-averaging formulas of the manuscript (log-normal) and the progress
report (Gamma-Gamma), and write the results as tables and graphs into a PDF.

```
python3 analytical_validation/run_all.py          # about 30 s; QUICK=1 for a coarse run
```

| File | Purpose |
|---|---|
| `01_test_lognormal_formulas.py` | Log-normal framework: validity distance and Rytov variance, analytic derivatives against a 60-digit reference, Euler operator, X-space against eta-space (78 cross-checks), error tables IV to VI, moments and Appendix-H claims, stress test, Monte Carlo |
| `02_test_gammagamma_formulas.py` | Gamma-Gamma cumulant expansion with aperture averaging: cumulants, reduction to log-normal, scintillation index, Tables 1 and 2, error against Rytov variance, choice of expansion point, three independent references |
| `03_make_pdf_report.py` | Builds `results/Analytical_Formula_Validation_Report.pdf` from the CSV and PNG files written by 01 and 02 |
| `cal_derivatives.py` | Analytic eta-derivatives (orders 0 to 4) of pXX, eX, eZ, R, written from the manuscript appendix |
| `mp_reference.py` | 60-digit mpmath re-implementation used only as an independent derivative reference |
| `averaging.py` | The averaging formulas under test, and the quadrature and Monte Carlo references |
| `common.py`, `paper_values.py`, `reporting.py` | Operating point and engine wrappers, the numbers printed in the papers, pass/fail bookkeeping |

Outputs: `results/tables/*.csv|json` (every table and the full pass/fail list), `results/figures/*.png`, and the PDF.

## How the tests are built

Each formula is checked against a reference that shares no code with it. Quadrature (Gauss-Hermite, tensor
Gauss-Laguerre, Bessel-K density) and Monte Carlo are used for the averages, and a separate high-precision
implementation for the derivatives. Where the papers print numbers, the scripts reproduce them and mark each
entry MATCH or DIFFERS, so disagreements are visible instead of being averaged away.

Gamma-Gamma results are computed under two parameter conventions: the correct Andrews-Phillips exponent
(sigma_R^(12/5) = variance^(6/5)) and the report's (variance^(12/5)), which is what its figures and tables use.

## Results at a glance

* 70 numeric checks pass, 0 fail (see the PDF, section 4).
* The progress report's Gamma-Gamma parameters use variance^(12/5); this is the same exponent slip that was fixed in
  the repository code in v2.2.0. The cumulant formulas themselves are unaffected.
* Typos found in the manuscript: the pXX bracket in Eq. (6) and Appendix A, and "sigma_X^2 = 9.6" in Appendix H (should be 2.4).
