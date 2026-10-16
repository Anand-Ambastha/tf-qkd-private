# -*- coding: utf-8 -*-
"""
example_04_max_range_and_validity.py
=======================================
Full "how far can this link go" analysis, run separately for the three
standard atmospheric turbulence-strength classes (legacy_baseline.Cn2_classes:
weak=1e-17, moderate=1e-15, strong=1e-13 m^(-2/3)) rather than one
arbitrary fixed Cn2 -- turbulence strength at a real site is a physical
site condition, not a free parameter, so "max range" only means something
once you fix which site condition you're asking about.

For EACH Cn2 class, this script:
  (a) Sweeps total Alice-Bob distance L for every turbulence model
      individually (deterministic, log-normal, Gamma-Gamma, corrected
      Gamma-Gamma) -- including distances where a given model is no
      longer physically valid, so you can see what "using the wrong
      model" would have told you at that site condition.
  (b) Runs the same sweep in model="auto" mode, which switches model at
      each distance according to the Rytov-variance regime boundaries
      *for that Cn2*.
  (c) Finds and marks the MAXIMUM RANGE (distance at which the ergodic
      secret-key rate drops below a numerical-outage threshold) and the
      corresponding total channel LOSS in dB, for every curve.
  (d) Marks the two regime-validity boundaries (sigma_R^2=1 weak/moderate,
      sigma_R^2=10 moderate/strong) for that Cn2 as vertical reference
      lines.
  (e) Prints a full text report (one section per Cn2 class + a
      cross-class summary table) and saves a 3-panel figure.

Run: python3 examples/example_04_max_range_and_validity.py

NOTE ON A BUG FOUND DURING DEVELOPMENT OF THIS SCRIPT: an earlier version
of the four skr_* evaluator functions below included a shortcut
`if eta0 < 1e-8: return 0.0`, intended as a cheap numerical-safety floor.
Because eta_nominal(L) is Cn2-independent, this forced EVERY
turbulence-dependent curve to zero at the exact same distance
(~79.45 km), producing an identical, artificial cutoff across all models
and all three Cn2 classes that had nothing to do with the actual physics
-- and, worse, it silently discarded exactly the tail contribution that
makes modeling turbulence worth doing in the first place (occasional
upward fades can push the instantaneous transmittance back above the
key-rate threshold even when the mean transmittance has already fallen
below it, extending the true ergodic-rate max range slightly beyond the
deterministic-only prediction). The guard has been removed; only a
sigma_R^2 cap remains, applied solely to the *forced, uncorrected*
log-normal/Gamma-Gamma curves as a genuine (not premature) numerical
safety measure for pathologically large shape parameters.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src import legacy_baseline as CAL
from src.turbulence import (LogNormalChannel, GammaGammaChannel,
                             CorrectedGammaGammaChannel, create_channel, classify_regime)
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
ALPHA_REF = 0.20
WAVELENGTH = 1550e-9
W0 = 0.05
RECEIVER_RADIUS = 0.15
ALPHA_ATM = 0.046
OUTAGE_SKR_THRESHOLD = 1e-9
MC_N = 20_000
MC_SEED = 0
SIGMA_R2_CAP = float('inf')   # v2.3: legacy hard cutoff (300) removed; it zeroed the rate for strong Cn2 beyond ~30 km
                              # and made 'Gamma-Gamma (forced)' look like a 30 km model. Gamma-Gamma now saturates correctly.

CN2_CLASSES = CAL.Cn2_classes

engine = TFQKDEngine(PD, THETA_A, THETA_B, Nmax=NMAX)

distances_km = np.linspace(0.5, 300.0, 180)


def eta_nominal_at(L_km):
    fso = CAL.deterministic_fso_channel(ALPHA_ATM, L_km, WAVELENGTH, W0, RECEIVER_RADIUS)
    return float(np.clip(fso['eta_fso'], 1e-300, 1.0))


def loss_dB(eta):
    return -10.0 * np.log10(max(eta, 1e-300))


eta_nom_arr = np.array([eta_nominal_at(L) for L in distances_km])
det_R_arr = np.array([float(engine.instantaneous_key_rate(ALPHA_REF, np.array([e]))[0])
                       for e in eta_nom_arr])


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


def skr_lognormal(L_km, eta0, cn2, sigma2):
    if sigma2 > SIGMA_R2_CAP:
        return 0.0
    ch = create_channel('lognormal', eta_nominal=eta0, Cn2=cn2, wavelength=WAVELENGTH,
                         distance_m=L_km * 1e3)   # two independent arms
    return engine.ergodic_key_rate(ch, ALPHA_REF, method="monte_carlo", n_mc=MC_N,
                                    rng=np.random.default_rng(MC_SEED))


def skr_gamma_gamma(L_km, eta0, cn2, sigma2):
    if sigma2 > SIGMA_R2_CAP:
        return 0.0
    ch = create_channel('gamma_gamma', eta_nominal=eta0, Cn2=cn2, wavelength=WAVELENGTH,
                         distance_m=L_km * 1e3)
    return engine.ergodic_key_rate(ch, ALPHA_REF, method="monte_carlo", n_mc=MC_N,
                                    rng=np.random.default_rng(MC_SEED))


def skr_corrected_gg(L_km, eta0, cn2, sigma2):
    ch = create_channel('gamma_gamma_corrected', eta_nominal=eta0, Cn2=cn2,
                         wavelength=WAVELENGTH, distance_m=L_km * 1e3,
                         receiver_radius=RECEIVER_RADIUS, jitter_std=1e-6,
                         beam_wander_std=5e-7, n_mc_cache=20_000)
    return engine.ergodic_key_rate(ch, ALPHA_REF, method="monte_carlo", n_mc=MC_N,
                                    rng=np.random.default_rng(MC_SEED))


def skr_auto(L_km, eta0, cn2, sigma2):
    regime = classify_regime(sigma2)
    picked_default = {'weak': 'lognormal', 'moderate': 'gamma_gamma',
                       'strong': 'gamma_gamma_corrected'}[regime]
    ch = create_channel('auto', eta_nominal=eta0, Cn2=cn2, wavelength=WAVELENGTH,
                         distance_m=L_km * 1e3, receiver_radius=RECEIVER_RADIUS,
                         jitter_std=1e-6, beam_wander_std=5e-7)
    R = engine.ergodic_key_rate(ch, ALPHA_REF, method="monte_carlo", n_mc=MC_N,
                                 rng=np.random.default_rng(MC_SEED))
    return R, ch.name


class_results = {}

for cn2_name, cn2_val in CN2_CLASSES.items():
    print(f"\n--- Sweeping Cn2 class '{cn2_name}' = {cn2_val:.1e} m^(-2/3) "
          f"({len(distances_km)} distances) ---")
    sigma2_arr = np.array([float(rytov_arm(cn2_val, WAVELENGTH, L * 1e3))
                            for L in distances_km])
    Lwm = find_regime_crossing(sigma2_arr, 1.0)
    Lms = find_regime_crossing(sigma2_arr, 10.0)
    print(f"  sigma_R^2=1  (weak/moderate boundary) at L = "
          f"{Lwm:.2f} km" if Lwm else "  never reaches sigma_R^2=1 in range")
    print(f"  sigma_R^2=10 (moderate/strong boundary) at L = "
          f"{Lms:.2f} km" if Lms else "  never reaches sigma_R^2=10 in range")

    res = {
        'Deterministic': det_R_arr.copy(),
        'Log-Normal (forced)': np.zeros_like(distances_km),
        'Gamma-Gamma (forced)': np.zeros_like(distances_km),
        'Corrected Gamma-Gamma (forced)': np.zeros_like(distances_km),
        'AUTO': np.zeros_like(distances_km),
    }
    auto_model_at = []
    for i, L in enumerate(distances_km):
        eta0, s2 = eta_nom_arr[i], sigma2_arr[i]
        res['Log-Normal (forced)'][i] = max(skr_lognormal(L, eta0, cn2_val, s2), 0.0)
        res['Gamma-Gamma (forced)'][i] = max(skr_gamma_gamma(L, eta0, cn2_val, s2), 0.0)
        res['Corrected Gamma-Gamma (forced)'][i] = max(skr_corrected_gg(L, eta0, cn2_val, s2), 0.0)
        R, picked = skr_auto(L, eta0, cn2_val, s2)
        res['AUTO'][i] = max(R, 0.0)
        auto_model_at.append(picked)

    class_results[cn2_name] = {'sigma2': sigma2_arr, 'results': res,
                                'auto_model_at': auto_model_at,
                                'crossings': (Lwm, Lms), 'cn2': cn2_val}
    print(f"  done.")

lines = []
lines.append("=" * 96)
lines.append("MAX-RANGE / REGIME-VALIDITY ANALYSIS -- weak / moderate / strong Cn2 classes")
lines.append("=" * 96)
lines.append(f"Fixed: wavelength={WAVELENGTH*1e9:.0f} nm, alpha_atm={ALPHA_ATM}/km "
             f"(~{ALPHA_ATM*4.343:.2f} dB/km), w0={W0} m, receiver radius={RECEIVER_RADIUS} m, "
             f"alpha_ref={ALPHA_REF}, outage threshold={OUTAGE_SKR_THRESHOLD:.0e} bits/pulse")
lines.append("Distance convention: TOTAL Alice-Bob distance (arm-aware internally).")
lines.append(f"Cn2 classes (legacy_baseline.Cn2_classes): "
             + ", ".join(f"{k}={v:.1e}" for k, v in CN2_CLASSES.items()))
lines.append("")

summary_rows = []
for cn2_name, data in class_results.items():
    lines.append("-" * 96)
    lines.append(f"Cn2 class: {cn2_name.upper()}  (Cn2 = {data['cn2']:.1e} m^(-2/3))")
    lines.append("-" * 96)
    Lwm, Lms = data['crossings']
    if Lwm:
        lines.append(f"  sigma_R^2=1  (log-normal validity limit) at L={Lwm:.2f} km "
                     f"(loss={loss_dB(eta_nominal_at(Lwm)):.2f} dB)")
    if Lms:
        lines.append(f"  sigma_R^2=10 (moderate/strong boundary)   at L={Lms:.2f} km "
                     f"(loss={loss_dB(eta_nominal_at(Lms)):.2f} dB)")
    lines.append("")
    lines.append(f"  {'Model':<34}{'Max range (km)':<18}{'Loss @ max (dB)':<18}{'eta_nom @ max':<16}")
    for name, R_arr in data['results'].items():
        idx = find_max_range(R_arr)
        if idx is None:
            lines.append(f"  {name:<34}{'never in range':<18}{'--':<18}{'--':<16}")
            summary_rows.append((cn2_name, name, None, None))
        else:
            Lmax = distances_km[idx]
            eta_v = eta_nom_arr[idx]
            note = '  (hit sweep limit)' if idx >= len(distances_km) - 1 else ''
            lines.append(f"  {name:<34}{Lmax:<18.2f}{loss_dB(eta_v):<18.2f}{eta_v:<16.3e}{note}")
            summary_rows.append((cn2_name, name, Lmax, loss_dB(eta_v)))
    lines.append("")
    lines.append("  AUTO-mode model selection along this sweep:")
    prev = None
    for L, picked in zip(distances_km, data['auto_model_at']):
        if picked != prev:
            lines.append(f"    L = {L:6.2f} km  ->  AUTO selects '{picked}'")
            prev = picked
    lines.append("")

lines.append("=" * 96)
lines.append("CROSS-CLASS SUMMARY: AUTO-mode max range vs. site turbulence condition")
lines.append("=" * 96)
lines.append(f"{'Cn2 class':<14}{'Max range (km)':<18}{'Loss @ max (dB)':<18}")
for cn2_name, data in class_results.items():
    idx = find_max_range(data['results']['AUTO'])
    if idx is None:
        lines.append(f"{cn2_name:<14}{'never in range':<18}{'--':<18}")
    else:
        Lmax = distances_km[idx]
        lines.append(f"{cn2_name:<14}{Lmax:<18.2f}{loss_dB(eta_nom_arr[idx]):<18.2f}")
lines.append("")
lines.append("Interpretation:")
lines.append("- 'forced' rows show what each single model reports even past its own validity")
lines.append("  boundary for that Cn2 -- included to show how much a misapplied model over/")
lines.append("  understates range, not as a recommended operating mode.")
lines.append("- AUTO is the physically appropriate estimate at every distance for a given site's")
lines.append("  Cn2, switching models at the regime boundaries listed above.")
def _mr(cls, model):
    i = find_max_range(class_results[cls]['results'][model])
    return None if i is None else float(distances_km[i])


_det = _mr('weak', 'Deterministic')
_f = lambda v: 'n/a' if v is None else f"{v:.1f} km"
lines.append(f"- Fading can LENGTHEN the max range beyond the deterministic cutoff ({_f(_det)}): R(eta) is convex near the")
lines.append(f"  cutoff, so rare upward fades lift the ergodic average. Here Gamma-Gamma (forced) gives "
             f"{_f(_mr('weak','Gamma-Gamma (forced)'))} / {_f(_mr('moderate','Gamma-Gamma (forced)'))} / "
             f"{_f(_mr('strong','Gamma-Gamma (forced)'))} for weak / moderate / strong Cn2.")
lines.append(f"- Log-Normal forced outside its validity (arm sigma_R^2 > 1) is NOT a prediction: moderate "
             f"{_f(_mr('moderate','Log-Normal (forced)'))}, strong {_f(_mr('strong','Log-Normal (forced)'))} "
             f"(weak value {_f(_mr('weak','Log-Normal (forced)'))} is the sweep limit, i.e. not a true maximum).")
lines.append("- Gamma-Gamma saturates (scintillation index -> ~1), so with plain Gamma-Gamma the max range is nearly")
lines.append("  Cn2-independent. Turbulence beam broadening (v2.3, example 07) is what makes range fall with Cn2.")
lines.append("- 'Corrected' / AUTO strong-class rows include aperture averaging with small pointing jitter and beam wander.")
lines.append("- Numerical note: this example uses 20,000-sample Monte Carlo per point on a 180-point grid (coarser than")
lines.append("  example 06's quadrature + bisection), so max ranges agree with example 06 only to ~0.1-1 km.")
lines.append("=" * 96)

report_text = "\n".join(lines)
print("\n" + report_text)

docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(docs_dir, exist_ok=True)
report_path = os.path.join(docs_dir, 'MAX_RANGE_REPORT.txt')
with open(report_path, 'w') as f:
    f.write(report_text + "\n")
print(f"\nReport saved to {report_path}")

plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 300, 'axes.grid': True,
                      'grid.alpha': 0.3, 'font.size': 9.5, 'font.family': 'serif'})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

colors = {'Deterministic': 'gray', 'Log-Normal (forced)': 'C0',
          'Gamma-Gamma (forced)': 'C1', 'Corrected Gamma-Gamma (forced)': 'C2', 'AUTO': 'k'}
styles = {'Deterministic': ':', 'Log-Normal (forced)': '--', 'Gamma-Gamma (forced)': '--',
          'Corrected Gamma-Gamma (forced)': '--', 'AUTO': '-'}

for ax, (cn2_name, data) in zip(axes, class_results.items()):
    for name, R_arr in data['results'].items():
        lw = 2.4 if name == 'AUTO' else 1.2
        ax.plot(distances_km, np.clip(R_arr, 1e-12, None), styles[name], color=colors[name],
                 lw=lw, label=name)
    Lwm, Lms = data['crossings']
    if Lwm:
        ax.axvline(Lwm, color='purple', ls='-.', lw=1.0, alpha=0.7)
    if Lms:
        ax.axvline(Lms, color='darkred', ls='-.', lw=1.0, alpha=0.7)
    idx = find_max_range(data['results']['AUTO'])
    if idx is not None:
        Lmax = distances_km[idx]
        ax.axvline(Lmax, color='k', alpha=0.3, lw=1.0)
        ax.annotate(f"{Lmax:.1f} km\n{loss_dB(eta_nom_arr[idx]):.1f} dB",
                     xy=(Lmax, max(data['results']['AUTO'][idx], 1e-11)), xytext=(4, 14),
                     textcoords='offset points', fontsize=7.5,
                     arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.set_yscale('log')
    ax.set_xlabel('Total Alice-Bob distance [km]')
    ax.set_title(f"Cn2 = {data['cn2']:.0e} ({cn2_name})")
    ax.set_ylim(bottom=1e-9)

axes[0].set_ylabel('Ergodic secret key rate [bits/pulse]')
axes[-1].legend(fontsize=7.5, loc='upper right')
fig.suptitle('Max range and regime-validity analysis across turbulence-strength classes', y=1.03)
fig.tight_layout()

figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(figures_dir, exist_ok=True)
fig_path = os.path.join(figures_dir, 'example04_max_range_validity.png')
fig.savefig(fig_path, bbox_inches='tight')
fig.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"Figure saved to {fig_path} (+ .pdf)")

# ---------------------------------------------------------------------
# Second figure: SKR vs Distance AND SKR vs Loss(dB), one row per model
# (the 4 cases: Deterministic, Log-Normal, Gamma-Gamma, Corrected
# Gamma-Gamma), with the three Cn2 site conditions overlaid in each row
# (Deterministic is Cn2-independent, so its row shows one curve; the
# other three rows show weak/moderate/strong as separate lines so you
# can see how the SAME model's prediction shifts with turbulence
# strength, in both distance-domain and loss-domain).
# ---------------------------------------------------------------------
loss_arr = np.array([loss_dB(e) for e in eta_nom_arr])   # shared x-mapping, Cn2-independent

row_keys = ['Deterministic', 'Log-Normal (forced)', 'Gamma-Gamma (forced)',
            'Corrected Gamma-Gamma (forced)']
class_colors = {'weak': 'C0', 'moderate': 'C1', 'strong': 'C2'}

fig2, axes2 = plt.subplots(4, 2, figsize=(12, 15))

for row, model_key in enumerate(row_keys):
    ax_dist, ax_loss = axes2[row, 0], axes2[row, 1]

    if model_key == 'Deterministic':
        R_arr = det_R_arr
        idx = find_max_range(R_arr)
        for ax, x_arr, xlabel in [(ax_dist, distances_km, 'Total Alice-Bob distance [km]'),
                                   (ax_loss, loss_arr, 'Total channel loss [dB]')]:
            ax.plot(x_arr, np.clip(R_arr, 1e-12, None), '-', color='gray', lw=2.0,
                    label='Deterministic (Cn2-independent)')
            if idx is not None:
                ax.axvline(x_arr[idx], color='gray', alpha=0.4, lw=1.0, ls=':')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylim(bottom=1e-9)
    else:
        for cn2_name, data in class_results.items():
            R_arr = data['results'][model_key]
            idx = find_max_range(R_arr)
            color = class_colors[cn2_name]
            for ax, x_arr in [(ax_dist, distances_km), (ax_loss, loss_arr)]:
                ax.plot(x_arr, np.clip(R_arr, 1e-12, None), '-', color=color, lw=1.6,
                        label=f"{cn2_name} (Cn2={data['cn2']:.0e})")
                if idx is not None:
                    ax.plot([x_arr[idx]], [max(R_arr[idx], 1e-11)], 'o', color=color, ms=4)
            ax_dist.set_xlabel('Total Alice-Bob distance [km]')
            ax_loss.set_xlabel('Total channel loss [dB]')
        ax_dist.set_yscale('log')
        ax_loss.set_yscale('log')
        ax_dist.set_ylim(bottom=1e-9)
        ax_loss.set_ylim(bottom=1e-9)

    ax_dist.set_ylabel(f"{model_key}\nSKR [bits/pulse]")
    ax_dist.legend(fontsize=7, loc='upper right')
    ax_loss.legend(fontsize=7, loc='upper right')

axes2[0, 0].set_title('SKR vs Distance')
axes2[0, 1].set_title('SKR vs Loss (dB)')
fig2.suptitle('SKR vs. Distance and SKR vs. Loss for all four channel models\n'
              '(weak / moderate / strong Cn2 overlaid where turbulence-dependent)', y=1.005)
fig2.tight_layout()

fig2_path = os.path.join(figures_dir, 'example04_skr_vs_distance_and_loss_by_model.png')
fig2.savefig(fig2_path, bbox_inches='tight')
fig2.savefig(fig2_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig2)
print(f"Figure saved to {fig2_path} (+ .pdf)")