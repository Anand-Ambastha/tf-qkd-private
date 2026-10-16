# -*- coding: utf-8 -*-
"""Run the full validation: 01 (log-normal), 02 (Gamma-Gamma), 03 (PDF report).  QUICK=1 for a coarse, faster run."""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
t0 = time.time()
for s in ('01_test_lognormal_formulas.py', '02_test_gammagamma_formulas.py', '03_make_pdf_report.py'):
    print(f'=== {s}')
    r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=ROOT)
    if r.returncode:
        sys.exit(r.returncode)
print(f'done in {time.time() - t0:.0f}s -> {os.path.join(HERE, "results", "Analytical_Formula_Validation_Report.pdf")}')
