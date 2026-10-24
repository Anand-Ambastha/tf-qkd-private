# -*- coding: utf-8 -*-
"""
plotting.py
===========
Plotting routines for turbulence models and TF-QKD results. Every figure
is saved as both .png (raster) and .pdf (vector). Style is set once in
`_style()` so all figures share fonts, DPI, and grid conventions.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _style():
    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300,
        'axes.grid': True, 'grid.alpha': 0.3,
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 9.5, 'lines.linewidth': 1.8,
        'font.family': 'serif',
    })


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), bbox_inches='tight')


def _display_range(channel, n_points):
    # Fading can push eta above eta_nominal; the plotted window covers the
    # bulk of the distribution and omits the thin tail out to eta_max = 1.
    hi = min(channel.eta_max, 8.0 * channel.eta_nominal)
    return np.linspace(1e-9, hi, n_points)


def plot_channel_pdf(channel, ax=None, n_points=500, out_dir=None, name=None):
    _style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    eta = _display_range(channel, n_points)
    ax.plot(eta, channel.pdf(eta), color='C0')
    ax.set_xlabel(r'Instantaneous transmittance $\eta$')
    ax.set_ylabel(r'$p(\eta)$')
    ax.set_title(f'{channel.name} channel: PDF of $\\eta$')
    if fig is not None and out_dir is not None:
        _save(fig, out_dir, name or f'{channel.name}_pdf')
    return ax


def plot_channel_cdf(channel, ax=None, n_points=500, out_dir=None, name=None):
    _style()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    eta = _display_range(channel, n_points)
    ax.plot(eta, channel.cdf(eta), color='C1')
    ax.axhline(1.0, color='k', lw=0.7, ls=':')
    ax.set_xlabel(r'Instantaneous transmittance $\eta$')
    ax.set_ylabel(r'$P(\eta_i \leq \eta)$')
    ax.set_title(f'{channel.name} channel: CDF of $\\eta$')
    if fig is not None and out_dir is not None:
        _save(fig, out_dir, name or f'{channel.name}_cdf')
    return ax


def plot_model_comparison_pdf(channels, labels=None, out_dir='figures', name='model_comparison_pdf'):
    """Overlay PDFs of several channels (e.g. Log-Normal vs Gamma-Gamma vs
    Corrected Gamma-Gamma at the same operating point) on one axis."""
    _style()
    labels = labels or [c.name for c in channels]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    hi = max(min(c.eta_max, 8.0 * c.eta_nominal) for c in channels)
    eta = np.linspace(1e-9, hi, 600)
    for c, lab, color in zip(channels, labels, ['C0', 'C1', 'C2', 'C3']):
        ax.plot(eta, c.pdf(eta), label=lab, color=color)
    ax.set_xlabel(r'Instantaneous transmittance $\eta$')
    ax.set_ylabel(r'$p(\eta)$')
    ax.set_title('Turbulence model comparison: PDF of $\\eta$')
    ax.legend()
    _save(fig, out_dir, name)
    plt.close(fig)
    return fig


def plot_distance_sweep(distances_km, results_dict, ylabel='Secret key rate [bits/pulse]',
                         out_dir='figures', name='distance_sweep', logy=True):
    """results_dict: {label: R_array} for one or more models/settings."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, R in results_dict.items():
        R_plot = np.clip(R, 1e-300, None) if logy else R
        ax.plot(distances_km, R_plot, marker='o', ms=3, label=label)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel(ylabel)
    ax.set_title('TF-QKD performance vs distance')
    ax.legend()
    _save(fig, out_dir, name)
    plt.close(fig)
    return fig


def plot_mc_convergence(rows, out_dir='figures', name='mc_convergence'):
    """rows: output of monte_carlo.benchmark_monte_carlo."""
    _style()
    n = [r['n'] for r in rows]
    est = [r['mean_estimate'] for r in rows]
    spread = [r['std_across_repeats'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].errorbar(n, est, yerr=spread, marker='o', color='C0', capsize=3)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Sample count N')
    ax[0].set_ylabel('MC estimate')
    ax[0].set_title('Monte Carlo convergence')
    ax[1].loglog(n, [r['mean_runtime_s'] for r in rows], marker='s', color='C2')
    ax[1].set_xlabel('Sample count N')
    ax[1].set_ylabel('Runtime [s]')
    ax[1].set_title('Monte Carlo runtime scaling')
    _save(fig, out_dir, name)
    plt.close(fig)
    return fig


def plot_quadrature_benchmark(rows, out_dir='figures', name='quadrature_benchmark'):
    """rows: output of quadrature.benchmark_quadrature."""
    _style()
    gl_rows = [r for r in rows if r['method'] == 'gauss_legendre']
    n = [r['n'] for r in gl_rows]
    rel_err = [max(r['relative_error'], 1e-16) for r in gl_rows]
    runtime = [r['runtime_s'] for r in gl_rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].loglog(n, rel_err, marker='o', color='C3')
    ax[0].set_xlabel('Quadrature order n')
    ax[0].set_ylabel('Relative error vs scipy.quad')
    ax[0].set_title('Gauss-Legendre accuracy')
    ax[1].loglog(n, runtime, marker='s', color='C4')
    ax[1].set_xlabel('Quadrature order n')
    ax[1].set_ylabel('Runtime [s]')
    ax[1].set_title('Gauss-Legendre runtime scaling')
    _save(fig, out_dir, name)
    plt.close(fig)
    return fig
