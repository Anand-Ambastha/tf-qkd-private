# -*- coding: utf-8 -*-
"""
example_05_instantaneous_adaptive_all_graphs.py
==================================================
Five graphs, each with two parts (a: SKR vs Distance, b: SKR vs Loss in
dB), all computed under instantaneous adaptive reference-intensity alpha:
alpha is re-optimized for every channel realization,
alpha_i* = argmax_alpha R(alpha, eta_i), rather than fixed at alpha=0.20.

  Graph 1: Log-Normal, forced "throughout" at weak / moderate / strong
           Cn2 (three curves), with each curve's own validity segment
           (sigma_R^2 < 1) drawn solid and the rest dashed.
  Graph 2: Gamma-Gamma, same three-Cn2-throughout treatment, valid
           segment (sigma_R^2 >= 1) solid.
  Graph 3: Corrected Gamma-Gamma ("Gamma-Gamma Extended"), same
           treatment, valid segment (sigma_R^2 >= 10) solid.
  Graph 4: AUTO mode, one curve per Cn2 class (weak/moderate/strong),
           each stitched from the valid segments of Graphs 1-3 at
           the same Cn2 (the valid model at each distance; no
           additional computation).
  Graph 5: Deterministic (no turbulence, Cn2-independent), single curve.

Numerical method for instantaneous adaptive alpha
-------------------------------------------------
legacy_baseline.key_rate (and optimize_alpha) requires a scalar alpha, since
its cat-state coefficient tables are not vectorized over alpha. Per-sample
optimization would need one scipy.optimize call per Monte Carlo draw, which
is impractical for this sweep.

Instead, alpha*(eta) is precomputed once on a log-spaced grid of eta (200
bins spanning 1e-16 to 1.0) with TFQKDEngine.optimize_alpha_at, and each
Monte Carlo sample is assigned the optimal alpha of its nearest bin.
alpha*(eta) is smooth and slowly varying (see the lookup-table check printed
at the start of the run), so the piecewise-constant approximation is close to
the continuous optimum and about 1000x cheaper. key_rate is evaluated once
per populated bin, vectorized over the samples in that bin.

Run: python3 examples/example_05_instantaneous_adaptive_all_graphs.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src import legacy_baseline as CAL
from src.turbulence import create_channel
from src.tfqkd_engine import TFQKDEngine


def rytov_arm(Cn2, wavelength, distance_m):
    """Per-ARM Rytov variance sigma_R^2(L/2): each of the two independent FSO
    links (Alice-Charlie, Bob-Charlie) has length L/2, so regime/validity
    decisions are made per arm, not at the full Alice-Bob distance."""
    return CAL.rytov_variance(Cn2, wavelength, 0.5 * distance_m)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PD = 1e-7
THETA_A = np.arcsin(np.sqrt(0.02))
THETA_B = -THETA_A
NMAX = 4
WAVELENGTH = 1550e-9
W0 = 0.05
RECEIVER_RADIUS = 0.15
ALPHA_ATM = 0.046
OUTAGE_SKR_THRESHOLD = 1e-9
MC_N = 20_000
MC_SEED = 0
SIGMA_R2_CAP = float('inf')   # v2.3: legacy hard cutoff removed (it zeroed plain GG/LN rates at strong Cn2)

CN2_CLASSES = CAL.Cn2_classes   # {'weak': 1e-17, 'moderate': 1e-15, 'strong': 1e-13}
engine = TFQKDEngine(PD, THETA_A, THETA_B, Nmax=NMAX)

# ---------------------------------------------------------------------
# Global alpha*(eta) lookup table (built once; reused for every model,
# Cn2 class, and distance below; see module docstring)
# ---------------------------------------------------------------------
print("Building global alpha*(eta) lookup table (200 bins, one-time cost)...")
N_BINS = 200
bin_edges = np.logspace(-16, 0, N_BINS + 1)
bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
alpha_per_bin = np.empty(N_BINS)
for i, ec in enumerate(bin_centers):
    a_opt, _ = engine.optimize_alpha_at(ec)
    alpha_per_bin[i] = a_opt
print(f"  done. alpha* ranges from {alpha_per_bin.min():.4f} to {alpha_per_bin.max():.4f} "
      f"across eta in [{bin_edges[0]:.0e}, {bin_edges[-1]:.0e}]")


def instantaneous_adaptive_rate(eta_samples):
    """Mean_i[ R(alpha_i*, eta_i) ] via the binned alpha*(eta) lookup."""
    eta_samples = np.clip(eta_samples, bin_edges[0], bin_edges[-1] * 0.999999)
    bin_idx = np.clip(np.searchsorted(bin_edges, eta_samples) - 1, 0, N_BINS - 1)
    total = 0.0
    n = len(eta_samples)
    for b in np.unique(bin_idx):
        mask = bin_idx == b
        eta_sub = eta_samples[mask]
        R_sub = CAL.key_rate(float(alpha_per_bin[b]), eta_sub, PD, THETA_A, THETA_B, NMAX)
        total += float(np.sum(np.maximum(R_sub, 0.0)))
    return total / n


def instantaneous_adaptive_rate_deterministic(eta0):
    """Single-point version: uses the optimizer directly rather than the
    binned approximation, since there is a single eta."""
    _, R = engine.optimize_alpha_at(eta0)
    return max(R, 0.0)


# ---------------------------------------------------------------------
# Shared distance / loss grid
# ---------------------------------------------------------------------
distances_km = np.linspace(0.5, 400.0, 120)


def eta_nominal_at(L_km):
    fso = CAL.deterministic_fso_channel(ALPHA_ATM, L_km, WAVELENGTH, W0, RECEIVER_RADIUS)
    return float(np.clip(fso['eta_fso'], 1e-300, 1.0))


def loss_dB(eta):
    return -10.0 * np.log10(max(eta, 1e-300))


eta_nom_arr = np.array([eta_nominal_at(L) for L in distances_km])
loss_arr = np.array([loss_dB(e) for e in eta_nom_arr])

print("\nComputing deterministic curve (Graph 5, instantaneous adaptive alpha)...")
det_R_arr = np.array([instantaneous_adaptive_rate_deterministic(e) for e in eta_nom_arr])


def find_regime_crossing(sigma_arr, threshold):
    idx = np.where(sigma_arr >= threshold)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return distances_km[0]
    L0, L1 = distances_km[i - 1], distances_km[i]
    s0, s1 = sigma_arr[i - 1], sigma_arr[i]
    frac = (threshold - s0) / (s1 - s0) if s1 != s0 else 0.0
    return L0 + frac * (L1 - L0)


def find_max_range(R_arr):
    above = R_arr > OUTAGE_SKR_THRESHOLD
    if not np.any(above):
        return None
    last_good = None
    for i in range(len(above)):
        if above[i]:
            last_good = i
        elif last_good is not None and i > last_good + 2:
            break
    return last_good


# ---------------------------------------------------------------------
# Per-(model, Cn2 class) sweep, instantaneous adaptive alpha throughout
# ---------------------------------------------------------------------
def ctor_lognormal(eta0, cn2, L_km):
    return create_channel('lognormal', eta_nominal=eta0, Cn2=cn2, wavelength=WAVELENGTH,
                          distance_m=L_km * 1e3)   # two independent arms


def ctor_gamma_gamma(eta0, cn2, L_km):
    return create_channel('gamma_gamma', eta_nominal=eta0, Cn2=cn2, wavelength=WAVELENGTH,
                          distance_m=L_km * 1e3)


def ctor_corrected(eta0, cn2, L_km):
    return create_channel('gamma_gamma_corrected', eta_nominal=eta0, Cn2=cn2,
                          wavelength=WAVELENGTH, distance_m=L_km * 1e3,
                          receiver_radius=RECEIVER_RADIUS, jitter_std=1e-6,
                          beam_wander_std=5e-7, n_mc_cache=20_000)


def sweep_model(channel_ctor, cn2_val, is_corrected):
    R_arr = np.empty_like(distances_km)
    sigma_arr = np.empty_like(distances_km)
    for i, L in enumerate(distances_km):
        eta0 = eta_nom_arr[i]
        s2 = float(rytov_arm(cn2_val, WAVELENGTH, L * 1e3))
        sigma_arr[i] = s2
        if (not is_corrected) and s2 > SIGMA_R2_CAP:
            R_arr[i] = 0.0
            continue
        ch = channel_ctor(eta0, cn2_val, L)
        s = ch.sample(MC_N, rng=np.random.default_rng(MC_SEED))
        R_arr[i] = instantaneous_adaptive_rate(s)
    return R_arr, sigma_arr


results = {}   # results[model_name][cn2_name] = {'R': arr, 'sigma2': arr}
model_ctors = {'lognormal': (ctor_lognormal, False), 'gamma_gamma': (ctor_gamma_gamma, False),
               'corrected': (ctor_corrected, True)}

for model_name, (ctor, is_corr) in model_ctors.items():
    results[model_name] = {}
    for cn2_name, cn2_val in CN2_CLASSES.items():
        print(f"Sweeping model='{model_name}' at Cn2 class '{cn2_name}' "
              f"({len(distances_km)} distances, instantaneous adaptive alpha)...")
        R_arr, sigma_arr = sweep_model(ctor, cn2_val, is_corr)
        results[model_name][cn2_name] = {'R': R_arr, 'sigma2': sigma_arr}

print("\nAll sweeps complete.")

# ---------------------------------------------------------------------
# Plot helper: solid within validity, dashed outside, single continuous line
# ---------------------------------------------------------------------
def plot_with_validity(ax, x, y, valid_mask, color, label):
    y = np.clip(y, 1e-12, None)
    change_points = np.where(np.diff(valid_mask.astype(int)) != 0)[0] + 1
    segments = np.split(np.arange(len(x)), change_points)
    first = True
    for seg in segments:
        if len(seg) == 0:
            continue
        lo = max(seg[0] - 1, 0)
        hi = min(seg[-1] + 1, len(x) - 1)
        idx = np.arange(lo, hi + 1)
        style = '-' if valid_mask[seg[0]] else '--'
        ax.plot(x[idx], y[idx], style, color=color, lw=1.8,
                 label=label if first else None, alpha=1.0 if valid_mask[seg[0]] else 0.55)
        first = False


plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 300, 'axes.grid': True,
                      'grid.alpha': 0.3, 'font.size': 10, 'font.family': 'serif'})
figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(figures_dir, exist_ok=True)
class_colors = {'weak': 'C0', 'moderate': 'C1', 'strong': 'C2'}


def make_graph(model_name, title, validity_fn, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for cn2_name, cn2_val in CN2_CLASSES.items():
        data = results[model_name][cn2_name]
        valid_mask = validity_fn(data['sigma2'])
        plot_with_validity(axes[0], distances_km, data['R'], valid_mask,
                            class_colors[cn2_name], f"{cn2_name} (Cn2={cn2_val:.0e})")
        plot_with_validity(axes[1], loss_arr, data['R'], valid_mask,
                            class_colors[cn2_name], f"{cn2_name} (Cn2={cn2_val:.0e})")
    for ax, xlabel, part in [(axes[0], 'Total Alice-Bob distance [km]', 'a'),
                              (axes[1], 'Total channel loss [dB]', 'b')]:
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Ergodic SKR [bits/pulse]\n(instantaneous adaptive alpha)')
        ax.set_ylim(bottom=1e-9)
        ax.set_title(f"({part}) {title} vs {'Distance' if part=='a' else 'Loss'}")
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f"{title} -- weak/moderate/strong Cn2 throughout, own validity regime "
                 f"solid, extrapolated dashed", y=1.02)
    fig.tight_layout()
    path = os.path.join(figures_dir, fname)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path} (+ .pdf)")


make_graph('lognormal', 'Graph 1: Log-Normal', lambda s2: s2 < 1.0,
           'example05_graph1_lognormal.png')
make_graph('gamma_gamma', 'Graph 2: Gamma-Gamma', lambda s2: s2 >= 1.0,
           'example05_graph2_gamma_gamma.png')
make_graph('corrected', 'Graph 3: Gamma-Gamma Extended (Corrected)', lambda s2: s2 >= 10.0,
           'example05_graph3_corrected_gamma_gamma.png')

# ---------------------------------------------------------------------
# Graph 4: AUTO mode, one curve per Cn2 class, stitched from Graphs 1-3
# at that same Cn2 according to the regime boundaries (no recomputation)
# ---------------------------------------------------------------------
print("\nBuilding Graph 4 (AUTO, stitched from Graphs 1-3 at matching Cn2)...")
auto_results = {}
for cn2_name, cn2_val in CN2_CLASSES.items():
    sigma_arr = results['lognormal'][cn2_name]['sigma2']
    R_ln = results['lognormal'][cn2_name]['R']
    R_gg = results['gamma_gamma'][cn2_name]['R']
    R_cgg = results['corrected'][cn2_name]['R']
    R_auto = np.where(sigma_arr < 1.0, R_ln, np.where(sigma_arr < 10.0, R_gg, R_cgg))
    auto_results[cn2_name] = {'R': R_auto, 'sigma2': sigma_arr}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for cn2_name, cn2_val in CN2_CLASSES.items():
    data = auto_results[cn2_name]
    Lwm = find_regime_crossing(data['sigma2'], 1.0)
    Lms = find_regime_crossing(data['sigma2'], 10.0)
    for ax, x_arr in [(axes[0], distances_km), (axes[1], loss_arr)]:
        ax.plot(x_arr, np.clip(data['R'], 1e-12, None), '-', color=class_colors[cn2_name],
                lw=2.0, label=f"{cn2_name} (Cn2={cn2_val:.0e})")
    if Lwm:
        axes[0].axvline(Lwm, color=class_colors[cn2_name], ls=':', lw=1.0, alpha=0.6)
    if Lms:
        axes[0].axvline(Lms, color=class_colors[cn2_name], ls='-.', lw=1.0, alpha=0.6)
for ax, xlabel, part in [(axes[0], 'Total Alice-Bob distance [km]', 'a'),
                          (axes[1], 'Total channel loss [dB]', 'b')]:
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ergodic SKR [bits/pulse]\n(instantaneous adaptive alpha)')
    ax.set_ylim(bottom=1e-9)
    ax.set_title(f"({part}) Graph 4: AUTO mode vs {'Distance' if part=='a' else 'Loss'}")
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle("Graph 4: AUTO mode, respective Cn2 per curve "
             "(dotted=sigma_R^2=1, dash-dot=sigma_R^2=10 crossing)", y=1.02)
fig.tight_layout()
path = os.path.join(figures_dir, 'example05_graph4_auto.png')
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"Saved {path} (+ .pdf)")

# ---------------------------------------------------------------------
# Graph 5: Deterministic (single curve, Cn2-independent)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
axes[0].plot(distances_km, np.clip(det_R_arr, 1e-12, None), '-', color='gray', lw=2.2,
             label='Deterministic (Cn2-independent)')
axes[1].plot(loss_arr, np.clip(det_R_arr, 1e-12, None), '-', color='gray', lw=2.2,
             label='Deterministic (Cn2-independent)')
for ax, xlabel, part in [(axes[0], 'Total Alice-Bob distance [km]', 'a'),
                          (axes[1], 'Total channel loss [dB]', 'b')]:
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ergodic SKR [bits/pulse]\n(instantaneous adaptive alpha)')
    ax.set_ylim(bottom=1e-9)
    ax.set_title(f"({part}) Graph 5: Deterministic vs {'Distance' if part=='a' else 'Loss'}")
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle("Graph 5: Deterministic channel (no turbulence)", y=1.02)
fig.tight_layout()
path = os.path.join(figures_dir, 'example05_graph5_deterministic.png')
fig.savefig(path, bbox_inches='tight')
fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"Saved {path} (+ .pdf)")

# ---------------------------------------------------------------------
# Graph 6: Adaptive model selection according to validity regime
# Log-Normal (sigma_R^2 < 1)
# Gamma-Gamma (1 <= sigma_R^2 < 10)
# Corrected Gamma-Gamma (sigma_R^2 >= 10)
# ---------------------------------------------------------------------

print("\nBuilding Graph 6 (Adaptive validity-regime switching)...")

adaptive_results = {}

for cn2_name in CN2_CLASSES:

    sigma = results['lognormal'][cn2_name]['sigma2']

    R_ln = results['lognormal'][cn2_name]['R']
    R_gg = results['gamma_gamma'][cn2_name]['R']
    R_cgg = results['corrected'][cn2_name]['R']

    R_adaptive = np.zeros_like(R_ln)

    # Weak turbulence
    idx = sigma < 1
    R_adaptive[idx] = R_ln[idx]

    # Moderate turbulence
    idx = (sigma >= 1) & (sigma < 10)
    R_adaptive[idx] = R_gg[idx]

    # Strong turbulence
    idx = sigma >= 10
    R_adaptive[idx] = R_cgg[idx]

    adaptive_results[cn2_name] = {
        'R': R_adaptive,
        'sigma2': sigma
    }

fig, axes = plt.subplots(1,2,figsize=(13,5.2))

for cn2_name, cn2_val in CN2_CLASSES.items():

    data = adaptive_results[cn2_name]

    axes[0].plot(
        distances_km,
        np.clip(data['R'],1e-12,None),
        lw=2.3,
        color=class_colors[cn2_name],
        label=f"{cn2_name} (Cn²={cn2_val:.0e})"
    )

    axes[1].plot(
        loss_arr,
        np.clip(data['R'],1e-12,None),
        lw=2.3,
        color=class_colors[cn2_name],
        label=f"{cn2_name} (Cn²={cn2_val:.0e})"
    )

    # Regime boundaries
    L1 = find_regime_crossing(data['sigma2'],1.0)
    L2 = find_regime_crossing(data['sigma2'],10.0)

    if L1 is not None:
        axes[0].axvline(
            L1,
            color=class_colors[cn2_name],
            linestyle=':',
            alpha=0.6
        )

    if L2 is not None:
        axes[0].axvline(
            L2,
            color=class_colors[cn2_name],
            linestyle='--',
            alpha=0.6
        )

for ax, xlabel, part in [
    (axes[0],'Total Alice-Bob distance [km]','a'),
    (axes[1],'Total channel loss [dB]','b')
]:

    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-9)
    ax.grid(True,alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Ergodic SKR [bits/pulse]\n(Adaptive turbulence model)')
    ax.set_title(f'({part}) Adaptive Model Selection')
    ax.legend(fontsize=8)

fig.suptitle(
    'Graph 6: Adaptive Turbulence Model Selection\n'
    'Log-Normal → Gamma-Gamma → Gamma-Gamma Extended',
    y=1.02
)

fig.tight_layout()

path = os.path.join(
    figures_dir,
    'example05_graph6_adaptive_validity_switch.png'
)

fig.savefig(path,bbox_inches='tight')
fig.savefig(path.replace('.png','.pdf'),bbox_inches='tight')
plt.close(fig)

print(f"Saved {path} (+ .pdf)")

# ---------------------------------------------------------------------
# Graph 7: Adaptive Cn² + Adaptive Turbulence Model
#
# sigma_R² < 1      : Weak Cn² + Log-Normal
# 1 <= sigma_R² <10 : Moderate Cn² + Gamma-Gamma
# sigma_R² >=10     : Strong Cn² + Corrected Gamma-Gamma
# ---------------------------------------------------------------------

print("\nBuilding Graph 6 (Adaptive Cn² + Adaptive Model)...")

R_adaptive = np.zeros_like(distances_km)
sigma_used = np.zeros_like(distances_km)

for i, L in enumerate(distances_km):

    eta0 = eta_nom_arr[i]

    # ---- First compute sigma using weak Cn²
    sigma = rytov_arm(
        CN2_CLASSES['weak'],
        WAVELENGTH,
        L*1e3
    )

    if sigma < 1:

        sigma_used[i] = sigma

        R_adaptive[i] = results['lognormal']['weak']['R'][i]

    else:

        sigma = rytov_arm(
            CN2_CLASSES['moderate'],
            WAVELENGTH,
            L*1e3
        )

        if sigma < 10:

            sigma_used[i] = sigma

            R_adaptive[i] = results['gamma_gamma']['moderate']['R'][i]

        else:

            sigma = rytov_arm(
                CN2_CLASSES['strong'],
                WAVELENGTH,
                L*1e3
            )

            sigma_used[i] = sigma

            R_adaptive[i] = results['corrected']['strong']['R'][i]

plt.figure(figsize=(7,5))

plt.semilogy(
    distances_km,
    np.clip(R_adaptive,1e-12,None),
    'k',
    lw=2.5,
    label='Adaptive Cn²'
)

plt.xlabel("Distance [km]")
plt.ylabel("Ergodic SKR [bits/pulse]")
plt.grid(True,alpha=0.3)
plt.legend()

plt.tight_layout()

path = os.path.join(
    figures_dir,
    "example05_graph7_adaptive_cn2.png"
)

plt.savefig(path,bbox_inches="tight")
plt.savefig(path.replace(".png",".pdf"),bbox_inches="tight")
plt.close()

print(f"Saved {path}")

plt.figure(figsize=(8,5))

# ---------------------------------------------------------
# Find validity boundaries
# ---------------------------------------------------------
sigma_weak = np.array([
    rytov_arm(CN2_CLASSES['weak'], WAVELENGTH, L*1e3)
    for L in distances_km
])

sigma_mod = np.array([
    rytov_arm(CN2_CLASSES['moderate'], WAVELENGTH, L*1e3)
    for L in distances_km
])

L_weak = None
L_strong = None

idx = np.where(sigma_weak >= 1)[0]
if len(idx):
    L_weak = distances_km[idx[0]]

idx = np.where(sigma_mod >= 10)[0]
if len(idx):
    L_strong = distances_km[idx[0]]

# ---------------------------------------------------------
# Background validity regions
# ---------------------------------------------------------
if L_weak is not None:
    plt.axvspan(0, L_weak,
                color='green',
                alpha=0.12,
                label='Weak regime\n(Log-Normal)')

if (L_weak is not None) and (L_strong is not None):
    plt.axvspan(L_weak, L_strong,
                color='gold',
                alpha=0.12,
                label='Moderate regime\n(Gamma-Gamma)')

if L_strong is not None:
    plt.axvspan(L_strong, distances_km[-1],
                color='red',
                alpha=0.10,
                label='Strong regime\n(Extended Gamma-Gamma)')

# ---------------------------------------------------------
# Vertical boundary lines
# ---------------------------------------------------------
if L_weak is not None:
    plt.axvline(L_weak,
                color='green',
                linestyle='--',
                linewidth=2)

if L_strong is not None:
    plt.axvline(L_strong,
                color='red',
                linestyle='--',
                linewidth=2)

# ---------------------------------------------------------
# Adaptive curve
# ---------------------------------------------------------
# ---------------------------------------------------------
# Adaptive curve with regime-dependent colors
# ---------------------------------------------------------

# Weak regime (Log-Normal)
mask = sigma_used < 1
plt.semilogy(
    distances_km[mask],
    np.clip(R_adaptive[mask], 1e-12, None),
    color='forestgreen',
    lw=2.8,
    label=r'Weak: Log-Normal ($\sigma_R^2<1$)'
)

# Moderate regime (Gamma-Gamma)
mask = (sigma_used >= 1) & (sigma_used < 10)
plt.semilogy(
    distances_km[mask],
    np.clip(R_adaptive[mask], 1e-12, None),
    color='royalblue',
    lw=2.8,
    label=r'Moderate: Gamma-Gamma ($1\leq\sigma_R^2<10$)'
)

# Strong regime (Extended Gamma-Gamma)
mask = sigma_used >= 10
plt.semilogy(
    distances_km[mask],
    np.clip(R_adaptive[mask], 1e-12, None),
    color='crimson',
    lw=2.8,
    label=r'Strong: Extended GG ($\sigma_R^2\geq10$)'
)

plt.xlabel("Distance (km)")
plt.ylabel("Ergodic SKR (bits/pulse)")
plt.grid(True,alpha=0.3)
plt.legend(loc='best',fontsize=8)

plt.tight_layout()

path = os.path.join(
    figures_dir,
    "example05_graph8_adaptive_cn2.png"
)

plt.savefig(path,bbox_inches="tight")
plt.savefig(path.replace(".png",".pdf"),bbox_inches="tight")
plt.close()

print(f"Saved {path}")

plt.figure(figsize=(8,5))

# ---------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------
lines = []
lines.append("=" * 96)
lines.append("INSTANTANEOUS-ADAPTIVE-ALPHA MAX-RANGE SUMMARY (all 5 graphs)")
lines.append("=" * 96)
lines.append(f"alpha optimized per-realization via a {N_BINS}-bin log-eta lookup table "
             f"(see module docstring for why key_rate cannot be vectorized over alpha directly).")
lines.append(f"Distance convention: TOTAL Alice-Bob distance. Outage threshold: "
             f"{OUTAGE_SKR_THRESHOLD:.0e} bits/pulse.")
lines.append("")

for graph_name, model_name in [('Graph 1: Log-Normal', 'lognormal'),
                                 ('Graph 2: Gamma-Gamma', 'gamma_gamma'),
                                 ('Graph 3: Gamma-Gamma Extended (Corrected)', 'corrected')]:
    lines.append(f"{graph_name}")
    for cn2_name in CN2_CLASSES:
        data = results[model_name][cn2_name]
        idx = find_max_range(data['R'])
        if idx is None:
            lines.append(f"  {cn2_name:<10} never in range")
        else:
            lines.append(f"  {cn2_name:<10} max range = {distances_km[idx]:6.2f} km, "
                         f"loss = {loss_arr[idx]:6.2f} dB")
    lines.append("")

lines.append("Graph 4: AUTO mode")
for cn2_name in CN2_CLASSES:
    data = auto_results[cn2_name]
    idx = find_max_range(data['R'])
    if idx is None:
        lines.append(f"  {cn2_name:<10} never in range")
    else:
        lines.append(f"  {cn2_name:<10} max range = {distances_km[idx]:6.2f} km, "
                     f"loss = {loss_arr[idx]:6.2f} dB")
lines.append("")

idx = find_max_range(det_R_arr)
lines.append("Graph 5: Deterministic")
if idx is not None:
    lines.append(f"  max range = {distances_km[idx]:6.2f} km, loss = {loss_arr[idx]:6.2f} dB")
lines.append("=" * 96)

report_text = "\n".join(lines)
print("\n" + report_text)
docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(docs_dir, exist_ok=True)
report_path = os.path.join(docs_dir, 'INSTANTANEOUS_ADAPTIVE_ALL_GRAPHS_REPORT.txt')
with open(report_path, 'w') as f:
    f.write(report_text + "\n")
print(f"\nReport saved to {report_path}")
