#!/bin/sh
# Run from the repo root: sh paper_analysis/gg_extension/run_all.sh
# Roughly 35 minutes on one CPU. The scripts are independent except plot_all.py, which needs all tables.
set -e
cd "$(dirname "$0")"
python3 e1_sweeps.py   > results/e1.log  2>&1
python3 e23_decomp_opt.py > results/e23.log 2>&1
python3 e45_decoy.py   > results/e45.log 2>&1
python3 analytic_vs_quad.py > results/e6.log 2>&1
python3 gg_master.py points > results/e7p.log 2>&1
python3 gg_master.py ranges > results/e7r.log 2>&1
python3 plot_all.py
