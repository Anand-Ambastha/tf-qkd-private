# -*- coding: utf-8 -*-
"""Check registry + plotting style shared by the validation scripts."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 220, 'axes.grid': True, 'grid.alpha': 0.3,
                     'font.size': 9.5, 'font.family': 'serif'})


class Checks:
    """Collects pass/fail checks; written to results/tables/<name>_checks.json for the PDF report."""
    def __init__(self, name):
        self.name, self.rows = name, []

    def add(self, cid, description, value, limit, mode='<=', unit='', note=''):
        ok = (value <= limit) if mode == '<=' else (value >= limit)
        self.rows.append(dict(id=cid, description=description, value=float(value), limit=float(limit), mode=mode,
                              unit=unit, status='PASS' if ok else 'FAIL', note=note))
        return ok

    def info(self, cid, description, text, note=''):
        self.rows.append(dict(id=cid, description=description, value=None, limit=None, mode='info', unit='',
                              status='INFO', note=text + (' ' + note if note else '')))

    def save(self):
        p = os.path.join(C.TAB_DIR, f'{self.name}_checks.json')
        json.dump(self.rows, open(p, 'w'), indent=1)
        n_pass = sum(r['status'] == 'PASS' for r in self.rows); n_fail = sum(r['status'] == 'FAIL' for r in self.rows)
        print(f'[{self.name}] checks: {n_pass} PASS, {n_fail} FAIL, {sum(r["status"]=="INFO" for r in self.rows)} INFO')
        return p


def savefig(fig, name):
    p = os.path.join(C.FIG_DIR, name + '.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('  figure', os.path.basename(p))
    return p


def match(value, paper, abs_tol=0.01, rel_tol=0.10):
    """Compare a reproduced percentage with the printed one; True if within max(abs_tol, rel_tol*|paper|) points."""
    return abs(value - paper) <= max(abs_tol, rel_tol * abs(paper))
