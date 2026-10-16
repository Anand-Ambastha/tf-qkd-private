# -*- coding: utf-8 -*-
"""
example_06_gamma_gamma_per_arm_validation.py
============================================
Gamma-Gamma turbulence applied INDEPENDENTLY ON EACH ARM (Alice->Charlie and
Bob->Charlie, each of length L/2, each with its own Rytov variance and its own
(alpha, beta)), validated against the per-arm Log-Normal model.

Validation question answered here
---------------------------------
For the WEAK / MODERATE / STRONG Cn2 classes (1e-17 / 1e-15 / 1e-13 m^(-2/3)):
what is the maximum total Alice-Bob distance (ergodic SKR > outage threshold) under
    * Gamma-Gamma, applied per arm            (the physically appropriate model)
    * Log-Normal, applied per arm             (valid only while arm sigma_R^2 < 1)
    * Deterministic (no turbulence)           (reference)
    * Gamma-Gamma / Log-Normal as ONE factor at full length (legacy, comparison only)
and how do the Gamma-Gamma and Log-Normal answers compare?

Method
------
* Ergodic SKR = 2-D tensor-product quadrature over (eta_A, eta_B) (TwoLinkChannel),
  noise-free, so the max-range search is deterministic. Max range is located on a
  coarse grid and refined by bisection.
* Independent Monte Carlo (own RNG stream, independent (eta_A, eta_B) draws) is run
  at the reported max-range points and across the distance grid as a cross-check.

Figures (figures/example06_*.png/.pdf) and report (docs/GAMMA_GAMMA_PER_ARM_REPORT.txt).

Run:  python3 examples/example_06_gamma_gamma_per_arm_validation.py
      QUICK=1 python3 examples/...   (coarse grids, for smoke-testing)
"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import legacy_baseline as CAL
from src.turbulence import create_channel
from src.turbulence.gamma_gamma import gamma_gamma_pdf
from src.tfqkd_engine import TFQKDEngine

QUICK = bool(os.environ.get('QUICK'))

# ---------------------------------------------------------------- parameters
PD, NMAX, ALPHA_REF = 1e-7, 4, 0.20
THETA_A = np.arcsin(np.sqrt(0.02)); THETA_B = -THETA_A
WAVELENGTH, W0, RX_RADIUS, ALPHA_ATM = 1550e-9, 0.05, 0.15, 0.046
SKR_THRESHOLD = 1e-9                       # outage threshold [bits/pulse]
CN2 = dict(CAL.Cn2_classes)                # weak 1e-17, moderate 1e-15, strong 1e-13
COLORS = {'Deterministic': 'gray', 'Gamma-Gamma (per arm)': 'C3', 'Log-Normal (per arm)': 'C0',
          'Gamma-Gamma (1 factor, legacy)': 'C3', 'Log-Normal (1 factor, legacy)': 'C0'}
STYLES = {'Deterministic': ':', 'Gamma-Gamma (per arm)': '-', 'Log-Normal (per arm)': '-',
          'Gamma-Gamma (1 factor, legacy)': '--', 'Log-Normal (1 factor, legacy)': '--'}
MODEL_KEYS = ['Deterministic', 'Gamma-Gamma (per arm)', 'Log-Normal (per arm)',
              'Gamma-Gamma (1 factor, legacy)', 'Log-Normal (1 factor, legacy)']

L_MAX_KM = 450.0
GRID_KM = np.linspace(1.0, L_MAX_KM, 40 if QUICK else 150)
N_BISECT = 8 if QUICK else 14
MC_N = 200_000 if QUICK else 2_000_000

engine = TFQKDEngine(PD, THETA_A, THETA_B, Nmax=NMAX)
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
DOC_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(DOC_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'axes.grid': True,
                     'grid.alpha': 0.3, 'font.size': 9.5, 'font.family': 'serif'})


def save(fig, name):
    p = os.path.join(FIG_DIR, f'example06_{name}.png')
    fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig); print('  saved', os.path.basename(p))


# ------------------------------------------------------------------ helpers
def eta_nom(L_km):
    return float(np.clip(CAL.deterministic_fso_channel(ALPHA_ATM, L_km, WAVELENGTH, W0,
                                                      RX_RADIUS)['eta_fso'], 1e-300, 1.0))


def loss_dB(eta):
    return -10.0 * np.log10(max(eta, 1e-300))


def sigma2_arm(cn2, L_km):
    return float(CAL.rytov_variance(cn2, WAVELENGTH, 0.5 * L_km * 1e3))


def build(model, cn2, L_km, two_link=True, seed=None):
    return create_channel(model, eta_nominal=eta_nom(L_km), Cn2=cn2, wavelength=WAVELENGTH,
                          distance_m=L_km * 1e3, two_link=two_link, seed=seed)


def ergodic_quad(ch):
    """Noise-free ergodic SKR. Two-link: tensor-product rule. Single link: 1-D nodes."""
    func = lambda e: engine.instantaneous_key_rate(ALPHA_REF, e)
    if hasattr(ch, 'link_A'):
        return float(ch.average(func, method='quadrature'))
    eta, w = ch.quadrature_nodes(2000) if ch.name == 'gamma_gamma' else ch.quadrature_nodes()
    return float(np.sum(w * func(eta)))


def ergodic_mc(ch, n, seed):
    s = ch.sample(n, rng=np.random.default_rng(seed))
    r = engine.instantaneous_key_rate(ALPHA_REF, s)
    return float(r.mean()), float(r.std() / np.sqrt(n))


def rate(key, cn2, L_km):
    if key == 'Deterministic':
        return float(engine.instantaneous_key_rate(ALPHA_REF, np.array([eta_nom(L_km)]))[0])
    model = 'gamma_gamma' if key.startswith('Gamma') else 'lognormal'
    return ergodic_quad(build(model, cn2, L_km, two_link='per arm' in key))


def max_range(R_grid, key, cn2):
    """Last grid point with R > threshold (contiguous from the short end), refined by bisection."""
    above = R_grid > SKR_THRESHOLD
    if not above[0]:
        return None
    i = 0
    while i + 1 < len(above) and above[i + 1]:
        i += 1
    if i + 1 >= len(above):
        return float(GRID_KM[-1])          # still above threshold at the sweep limit
    lo, hi = GRID_KM[i], GRID_KM[i + 1]
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if rate(key, cn2, mid) > SKR_THRESHOLD:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def L_at_sigma(cn2, target):
    """Total distance at which the per-arm Rytov variance equals `target` (closed form, ~L^(11/6))."""
    s1 = sigma2_arm(cn2, 1.0)
    return float((target / s1) ** (6.0 / 11.0))


# ---------------------------------------------------------------- sweeps
print(f"Sweeping {len(GRID_KM)} distances x 3 Cn2 classes x {len(MODEL_KEYS)} models "
      f"({'QUICK' if QUICK else 'full'}) ...")
det_R = np.array([rate('Deterministic', None, L) for L in GRID_KM])
eta_nom_grid = np.array([eta_nom(L) for L in GRID_KM])
loss_grid = np.array([loss_dB(e) for e in eta_nom_grid])

R = {c: {} for c in CN2}
MAXR = {c: {} for c in CN2}
for c, cn2 in CN2.items():
    for key in MODEL_KEYS:
        if key == 'Deterministic':
            R[c][key] = det_R
        else:
            R[c][key] = np.array([rate(key, cn2, L) for L in GRID_KM])
        MAXR[c][key] = max_range(R[c][key], key, cn2)
    print(f"  {c:9s} done: " + ", ".join(
        f"{k.split(' (')[0][:5]}{'*' if 'legacy' in k else ''}={MAXR[c][k]:.1f}" if MAXR[c][k] else
        f"{k.split(' (')[0][:5]}=none" for k in MODEL_KEYS))

# ------------------------------------------------------- MC cross-check
print("Monte Carlo cross-check at max-range points ...")
MC_ROWS = []
for c, cn2 in CN2.items():
    for key in ('Gamma-Gamma (per arm)', 'Log-Normal (per arm)'):
        Lm = MAXR[c][key]
        if Lm is None:
            continue
        ch = build('gamma_gamma' if key.startswith('Gamma') else 'lognormal', cn2, Lm, True, seed=11)
        rq = ergodic_quad(ch)
        rm, se = ergodic_mc(ch, MC_N, seed=2024)
        MC_ROWS.append((c, key, Lm, rq, rm, se))

# --------------------------------------------- MC-vs-quadrature across distance (GG per arm)
print("MC-vs-quadrature scan across distance (Gamma-Gamma per arm) ...")
SCAN = {}
for c, cn2 in CN2.items():
    Ls = np.linspace(5.0, MAXR[c]['Gamma-Gamma (per arm)'] * 0.98, 6 if QUICK else 12)
    rel, band = [], []
    for L in Ls:
        ch = build('gamma_gamma', cn2, L, True, seed=5)
        rq = ergodic_quad(ch); rm, se = ergodic_mc(ch, MC_N, seed=99)
        rel.append((rm - rq) / rq if rq > 0 else np.nan); band.append(2 * se / rq if rq > 0 else np.nan)
    SCAN[c] = (Ls, np.array(rel), np.array(band))

# --------------------------------------------- statistics & outage vs distance
print("Per-arm statistics and outage probability vs distance ...")
D_STAT = np.linspace(2.0, 300.0, 40 if QUICK else 100)
eta_th = None
lo, hi = 1e-14, 1.0                              # instantaneous-eta threshold with R(alpha_ref, eta) = SKR_THRESHOLD
for _ in range(80):
    mid = np.sqrt(lo * hi)
    if engine.instantaneous_key_rate(ALPHA_REF, np.array([mid]))[0] > SKR_THRESHOLD:
        hi = mid
    else:
        lo = mid
eta_th = hi
SI_TOT, P_OUT = {c: {} for c in CN2}, {c: {} for c in CN2}
for c, cn2 in CN2.items():
    for tag, model in (('GG', 'gamma_gamma'), ('LN', 'lognormal')):
        si, po = [], []
        for L in D_STAT:
            ch = build(model, cn2, L, True, seed=3)
            sA = ch.link_A.statistics()['scintillation_index']; sB = ch.link_B.statistics()['scintillation_index']
            si.append((1 + sA) * (1 + sB) - 1)
            po.append(float(np.mean(ch.sample(20_000 if QUICK else 60_000, rng=np.random.default_rng(7)) < eta_th)))
        SI_TOT[c][tag], P_OUT[c][tag] = np.array(si), np.array(po)

# --------------------------- GG vs LN agreement INSIDE the log-normal validity range
AGREE = {}
for c, cn2 in CN2.items():
    L1 = L_at_sigma(cn2, 1.0)
    m = (GRID_KM <= L1) & (R[c]['Gamma-Gamma (per arm)'] > 1e-6)
    if m.sum() >= 1:
        d = np.abs(R[c]['Log-Normal (per arm)'][m] - R[c]['Gamma-Gamma (per arm)'][m]) / R[c]['Gamma-Gamma (per arm)'][m]
        AGREE[c] = (L1, int(m.sum()), float(d.max()), float(np.median(d)))
    else:
        AGREE[c] = (L1, 0, float('nan'), float('nan'))
# very-weak reference point (both models valid everywhere): Cn2 = 1e-18
VW = []
for L in (20.0, 60.0, 120.0):
    g, l = rate('Gamma-Gamma (per arm)', 1e-18, L), rate('Log-Normal (per arm)', 1e-18, L)
    VW.append((L, sigma2_arm(1e-18, L), g, l, abs(l - g) / g))

# =========================================================== REPORT
lines = ["=" * 100,
         "GAMMA-GAMMA APPLIED INDEPENDENTLY ON EACH ARM -- MAX-RANGE VALIDATION vs LOG-NORMAL",
         "=" * 100,
         f"lambda={WAVELENGTH*1e9:.0f} nm  alpha_atm={ALPHA_ATM}/km  w0={W0} m  Rx radius={RX_RADIUS} m  "
         f"alpha_ref={ALPHA_REF}  pd={PD:.0e}  Nmax={NMAX}",
         f"Max range = largest TOTAL Alice-Bob distance with ergodic SKR > {SKR_THRESHOLD:.0e} bits/pulse "
         f"(sweep limit {L_MAX_KM:.0f} km, grid {len(GRID_KM)} pts + {N_BISECT} bisections).",
         "Per-arm: arms of length L/2, eta_A_nom = eta_B_nom = sqrt(eta_nom(L)), independent fading, eta = eta_A*eta_B.",
         f"Instantaneous-eta outage threshold at alpha_ref: eta_th = {eta_th:.3e} ({loss_dB(eta_th):.2f} dB).", ""]
for c, cn2 in CN2.items():
    L1, L10 = L_at_sigma(cn2, 1.0), L_at_sigma(cn2, 10.0)
    lines += ["-" * 100, f"{c.upper()}  Cn2 = {cn2:.0e} m^(-2/3)", "-" * 100,
              f"  per-arm sigma_R^2 = 1  at total L = {L1:8.2f} km   (Log-Normal validity limit)",
              f"  per-arm sigma_R^2 = 10 at total L = {L10:8.2f} km   (moderate -> strong boundary)", "",
              f"  {'Model':<34}{'Max range [km]':>15}{'Loss@max [dB]':>15}{'sigma_R2/arm @max':>19}   note"]
    for key in MODEL_KEYS:
        Lm = MAXR[c][key]
        if Lm is None:
            lines.append(f"  {key:<34}{'none':>15}")
            continue
        s2 = sigma2_arm(cn2, Lm) if key != 'Deterministic' else float('nan')
        note = ''
        if key.startswith('Log') and s2 > 1.0:
            note = 'LN outside validity (sigma_R^2>1): indicative only'
        if Lm >= L_MAX_KM - 1e-6:
            note = (note + '; ' if note else '') + 'hit sweep limit'
        lines.append(f"  {key:<34}{Lm:>15.2f}{loss_dB(eta_nom(Lm)):>15.2f}{s2:>19.3g}   {note}")
    gg, ln, det = (MAXR[c]['Gamma-Gamma (per arm)'], MAXR[c]['Log-Normal (per arm)'],
                   MAXR[c]['Deterministic'])
    if gg and ln:
        lines.append(f"\n  Gamma-Gamma - Log-Normal (per arm): {gg - ln:+.2f} km ({100*(gg-ln)/ln:+.1f} % of LN range)")
    if gg and det:
        lines.append(f"  Gamma-Gamma - Deterministic       : {gg - det:+.2f} km")
    lines.append("")

lines += ["=" * 100, "SUMMARY (max range, km)", "=" * 100,
          f"{'Cn2 class':<10}{'Deterministic':>15}{'GG per arm':>13}{'LN per arm':>13}{'GG-LN [km]':>13}"
          f"{'GG 1-factor':>13}{'LN 1-factor':>13}"]
for c in CN2:
    m = MAXR[c]
    f = lambda k: f"{m[k]:.1f}" if m[k] else 'none'
    d = f"{m['Gamma-Gamma (per arm)'] - m['Log-Normal (per arm)']:+.1f}" if (
        m['Gamma-Gamma (per arm)'] and m['Log-Normal (per arm)']) else 'n/a'
    lines.append(f"{c:<10}{f('Deterministic'):>15}{f('Gamma-Gamma (per arm)'):>13}{f('Log-Normal (per arm)'):>13}"
                 f"{d:>13}{f('Gamma-Gamma (1 factor, legacy)'):>13}{f('Log-Normal (1 factor, legacy)'):>13}")
lines += ["", "=" * 100, "GAMMA-GAMMA vs LOG-NORMAL AGREEMENT INSIDE THE LOG-NORMAL VALIDITY RANGE (arm sigma_R^2 <= 1)", "=" * 100,
          f"{'Cn2 class':<10}{'valid up to L [km]':>20}{'grid pts':>10}{'max |R_LN-R_GG|/R_GG':>24}{'median':>10}"]
for c, (L1, npts, dmax, dmed) in AGREE.items():
    lines.append(f"{c:<10}{L1:>20.2f}{npts:>10d}{dmax:>24.2%}{dmed:>10.2%}" if npts else
                 f"{c:<10}{L1:>20.2f}{'(no grid points below this distance)':>44}")
lines += ["", "Very weak reference (Cn2 = 1e-18, both models valid at every distance shown):",
          f"  {'L [km]':>8}{'sigma_R2/arm':>14}{'R_GG':>13}{'R_LN':>13}{'|diff|/R_GG':>13}"]
for L, s2v, g, l, dd in VW:
    lines.append(f"  {L:>8.1f}{s2v:>14.3g}{g:>13.4e}{l:>13.4e}{dd:>13.2%}")
lines += ["", "=" * 100, "MONTE CARLO CROSS-CHECK AT THE REPORTED MAX-RANGE POINTS", "=" * 100,
          f"{'Cn2':<10}{'Model':<24}{'L_max [km]':>11}{'R quad':>13}{'R MC':>13}{'MC s.e.':>11}{'(MC-q)/q':>11}{'|z|':>7}"]
for c, key, Lm, rq, rm, se in MC_ROWS:
    z = abs(rm - rq) / se if se > 0 else float('nan')
    lines.append(f"{c:<10}{key:<24}{Lm:>11.2f}{rq:>13.3e}{rm:>13.3e}{se:>11.2e}"
                 f"{(rm - rq) / rq if rq else float('nan'):>+11.2%}{z:>7.2f}")
lines += ["", f"(MC uses {MC_N:,} independent (eta_A, eta_B) draws. At the outage threshold the rate is carried by",
          " rare upward fades, so MC s.e. is a few % there. Log-Normal rows with arm sigma_R^2 >> 1 can show",
          " large quad-vs-MC gaps: with sigma_X^2 = sigma_R^2 >> 1 the mean-one log-normal puts almost all",
          " probability in deep fades and the mean is carried by astronomically rare events that NEITHER",
          " 128-node Gauss-Hermite NOR 10^6 samples resolve -> those LN numbers are unconverged, and the",
          " model is outside its validity anyway.)", ""]
lines += ["=" * 100, "INTERPRETATION", "=" * 100,
          "* Inside its validity range (arm sigma_R^2 <= 1) Log-Normal and Gamma-Gamma agree well (median 1-7 %, worst 7-15 % near the sigma_R^2=1 edge; table",
          "  above); they separate only once the arm Rytov variance passes ~1, which for the WEAK class is",
          "  beyond ~200 km total distance, for MODERATE ~17 km, for STRONG ~1.4 km.",
          "* Every class's max range lies beyond its own Log-Normal validity limit, so the Log-Normal",
          "  max-range number is a comparison/diagnostic, not a recommendation.",
          "* Gamma-Gamma saturates (SI -> ~1, independent of Cn2) while a forced Log-Normal keeps",
          "  SI = exp(sigma_R^2)-1 -> huge. That is why per-arm Gamma-Gamma gives nearly the SAME max range",
          "  for all three classes (once arms are saturated the fading statistics stop depending on Cn2),",
          "  while Log-Normal collapses for moderate/strong.",
          "* Fading lengthens the max range by ~40 km relative to the deterministic cutoff: R(eta) is",
          "  convex near the cutoff, so occasional upward fades lift the ergodic average (Jensen).",
          "* Not modelled (document in the write-up): turbulence-induced beam broadening / beam wander at",
          "  large Cn2 and arm amplitude imbalance (visibility penalty); Gamma-Gamma here uses the",
          "  plane-wave alpha/beta of Andrews & Phillips. Results are an upper bound for strong Cn2.", "=" * 100]
report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(DOC_DIR, 'GAMMA_GAMMA_PER_ARM_REPORT.txt'), 'w') as f:
    f.write(report + "\n")

# =========================================================== FIGURES
print("Figures ...")
cls_names = list(CN2)

# ---- G1: GG parameters / scintillation per arm vs Rytov variance
from src.turbulence.gamma_gamma import alpha_beta_plane_wave
s_axis = np.logspace(-3, 4, 300)
ab = np.array([alpha_beta_plane_wave(s) for s in s_axis])
si_gg = 1 / ab[:, 0] + 1 / ab[:, 1] + 1 / (ab[:, 0] * ab[:, 1])
si_ln = np.expm1(s_axis)                                     # LN with sigma_X^2 = sigma_R^2, mean-one
e_bug = s_axis ** (12 / 5)                                   # pre-fix (v2.1) exponent, for reference only
a_b = 1 / np.expm1(0.49 * s_axis / (1 + 1.11 * e_bug) ** (7 / 6)); b_b = 1 / np.expm1(0.51 * s_axis / (1 + 0.69 * e_bug) ** (5 / 6))
si_bug = 1 / a_b + 1 / b_b + 1 / (a_b * b_b)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ax[0].loglog(s_axis, ab[:, 0], 'C3', label=r'$\alpha$ (large scale)'); ax[0].loglog(s_axis, ab[:, 1], 'C1', label=r'$\beta$ (small scale)')
ax[0].set_xlabel(r'Rytov variance per arm $\sigma_R^2$'); ax[0].set_ylabel('shape parameter'); ax[0].legend(); ax[0].set_title('Gamma-Gamma parameters per arm')
ax[1].loglog(s_axis, si_gg, 'C3', lw=2, label='Gamma-Gamma (corrected exponent)')
ax[1].loglog(s_axis, si_ln, 'C0', label=r'Log-Normal, $e^{\sigma_R^2}-1$')
ax[1].loglog(s_axis, si_bug, 'gray', ls=':', label=r'pre-fix $\sigma_R^{4.8}$ (wrong)')
ax[1].axvline(1, color='purple', ls='-.', lw=.8); ax[1].axvline(10, color='darkred', ls='-.', lw=.8)
ax[1].set_ylim(1e-3, 1e3); ax[1].set_xlabel(r'$\sigma_R^2$ per arm'); ax[1].set_ylabel('scintillation index (one arm)')
ax[1].legend(fontsize=8); ax[1].set_title('Saturation: GG peaks ~1.2 then -> 1, LN diverges')
Lk = np.logspace(-0.5, 2.6, 200)
for c, col in zip(cls_names, ('C0', 'C1', 'C2')):
    ax[2].loglog(Lk, [sigma2_arm(CN2[c], L) for L in Lk], color=col, label=f'{c} (Cn2={CN2[c]:.0e})')
ax[2].axhline(1, color='purple', ls='-.', lw=.8); ax[2].axhline(10, color='darkred', ls='-.', lw=.8)
ax[2].set_xlabel('Total Alice-Bob distance [km] (arm = L/2)'); ax[2].set_ylabel(r'$\sigma_R^2$ per arm'); ax[2].legend(fontsize=8)
ax[2].set_title('Regime map: which arm turbulence regime each class is in')
fig.tight_layout(); save(fig, 'g1_parameters_and_scintillation')

# ---- G2: arm PDF and end-to-end PDF of normalized irradiance, L = 60 km total
L_REP = 60.0
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
Ix = np.logspace(-4, 1.6, 600)
for j, c in enumerate(cls_names):
    cn2 = CN2[c]
    gg_arm = build('gamma_gamma', cn2, L_REP, True).link_A
    ln_arm = build('lognormal', cn2, L_REP, True).link_A
    a = axes[0, j]
    p_gg = gamma_gamma_pdf(Ix, gg_arm.alpha, gg_arm.beta)
    p_ln = ln_arm.pdf(Ix * ln_arm.eta_nominal) * ln_arm.eta_nominal
    a.loglog(Ix, Ix * p_gg, 'C3', lw=2, label=f'GG  (α={gg_arm.alpha:.2f}, β={gg_arm.beta:.2f})')
    a.loglog(Ix, Ix * p_ln, 'C0', lw=1.6, label=f'LN  (σ_X²={ln_arm.sigma_X2:.2f})')
    a.set_ylim(1e-4, 5); a.set_xlabel(r'normalized irradiance $I_A=\eta_A/\eta_{A,nom}$'); a.set_ylabel(r'$I\,p(I)$ (density in $\ln I$)')
    a.set_title(f'{c}: ONE arm (L/2={L_REP/2:.0f} km, σ_R²={gg_arm.sigma_R2:.3g})'); a.legend(fontsize=8)
    b = axes[1, j]
    rng = np.random.default_rng(1)
    bins = np.linspace(-9, 3.5, 140)
    for key, model, tl, col, ls in (('GG per arm', 'gamma_gamma', True, 'C3', '-'),
                                    ('LN per arm', 'lognormal', True, 'C0', '-'),
                                    ('GG 1 factor (legacy)', 'gamma_gamma', False, 'C3', '--')):
        ch = build(model, cn2, L_REP, tl, seed=1)
        s = ch.sample(MC_N, rng=rng) / ch.eta_nominal
        h, e = np.histogram(np.log(np.clip(s, 1e-30, None)), bins=bins, density=True)
        b.semilogy(0.5 * (e[1:] + e[:-1]), np.clip(h, 1e-6, None), color=col, ls=ls, lw=1.8 if 'per' in key else 1.2, label=key)
    b.set_xlabel(r'$\ln(\eta/\eta_{nom})$, end-to-end'); b.set_ylabel(r'density of $\ln\eta$'); b.set_ylim(1e-4, 5)
    b.set_title(f'{c}: END-TO-END  η = η_A·η_B (L={L_REP:.0f} km)'); b.legend(fontsize=8)
fig.suptitle('Per-arm and end-to-end irradiance statistics: Gamma-Gamma vs Log-Normal', y=1.0)
fig.tight_layout(); save(fig, 'g2_pdfs_per_arm_and_end_to_end')

# ---- G3 / G4: SKR vs distance / vs loss
for gname, xs, xl in (('g3_skr_vs_distance', GRID_KM, 'Total Alice-Bob distance [km]'),
                      ('g4_skr_vs_loss', loss_grid, 'Total nominal channel loss [dB]')):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, c in zip(axes, cls_names):
        for key in MODEL_KEYS:
            ax.plot(xs, np.clip(R[c][key], 1e-12, None), STYLES[key], color=COLORS[key],
                    lw=2.2 if 'per arm' in key else 1.1, label=key)
            Lm = MAXR[c][key]
            if Lm is not None and 'per arm' in key:
                xm = Lm if xs is GRID_KM else loss_dB(eta_nom(Lm))
                ax.plot([xm], [SKR_THRESHOLD], 'o', color=COLORS[key], ms=6)
        ax.axhline(SKR_THRESHOLD, color='k', lw=.6, ls=':')
        if xs is GRID_KM:
            ax.axvspan(L_at_sigma(CN2[c], 1.0), xs[-1], color='C0', alpha=.05, lw=0)
            for t, col in ((1.0, 'purple'), (10.0, 'darkred')):
                Lb = L_at_sigma(CN2[c], t)
                if Lb < xs[-1]:
                    ax.axvline(Lb, color=col, ls='-.', lw=.8, alpha=.7)
        ax.set_yscale('log'); ax.set_ylim(1e-10, 1); ax.set_xlabel(xl)
        ax.set_title(f'{c} (Cn2={CN2[c]:.0e})')
    axes[0].set_ylabel('Ergodic SKR [bits/pulse]'); axes[-1].legend(fontsize=7.5, loc='upper right')
    fig.suptitle('Ergodic SKR, Gamma-Gamma applied per arm vs Log-Normal vs deterministic '
                 '(dots: max range; dash-dot: per-arm σ_R²=1 / 10; blue shade: Log-Normal invalid)', y=1.02)
    fig.tight_layout(); save(fig, gname)

# ---- G5: ergodic penalty R/R_det
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
with np.errstate(divide='ignore', invalid='ignore'):
    for ax, c in zip(axes, cls_names):
        for key in MODEL_KEYS[1:]:
            ratio = np.where(det_R > 1e-12, R[c][key] / det_R, np.nan)
            ax.semilogy(GRID_KM, np.clip(ratio, 1e-6, None), STYLES[key], color=COLORS[key],
                        lw=2.2 if 'per arm' in key else 1.1, label=key)
        ax.axhline(1, color='gray', lw=.6); ax.set_xlim(0, 260)
        ax.set_xlabel('Total distance [km]'); ax.set_title(f'{c}')
axes[0].set_ylabel(r'$R_{ergodic}/R_{deterministic}$'); axes[-1].legend(fontsize=7.5, loc='lower left')
fig.suptitle('Turbulence penalty on the ergodic key rate (ratio to the no-turbulence rate)', y=1.02)
fig.tight_layout(); save(fig, 'g5_turbulence_penalty_ratio')

# ---- G6: max-range bars (validation headline)
fig, ax = plt.subplots(figsize=(9.5, 5))
bar_keys = ['Deterministic', 'Gamma-Gamma (per arm)', 'Log-Normal (per arm)']
w = 0.26
for k, key in enumerate(bar_keys):
    vals = [MAXR[c][key] if MAXR[c][key] else 0.0 for c in cls_names]
    bars = ax.bar(np.arange(3) + (k - 1) * w, vals, w, color=COLORS[key], alpha=.85, label=key,
                  hatch='//' if key.startswith('Log') else None)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 4, f'{v:.1f}', ha='center', fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([f'{c}\nCn2={CN2[c]:.0e}' for c in cls_names])
ax.set_ylabel('Max total distance [km]'); ax.set_title(f'Validation: max range (ergodic SKR > {SKR_THRESHOLD:.0e}), per-arm Gamma-Gamma vs Log-Normal')
ax.legend(); fig.tight_layout(); save(fig, 'g6_max_range_gg_vs_ln')

# ---- G7: end-to-end scintillation index vs distance (analytic, per-arm product)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, c in zip(axes, cls_names):
    ax.semilogy(D_STAT, SI_TOT[c]['GG'], 'C3', lw=2, label='Gamma-Gamma  (1+SI_A)(1+SI_B)-1')
    ax.semilogy(D_STAT, SI_TOT[c]['LN'], 'C0', lw=2, label='Log-Normal   (1+SI_A)(1+SI_B)-1')
    for t, col in ((1.0, 'purple'), (10.0, 'darkred')):
        Lb = L_at_sigma(CN2[c], t)
        if Lb < D_STAT[-1]:
            ax.axvline(Lb, color=col, ls='-.', lw=.8, label=f'arm σ_R²={t:g}')
    ax.set_xlabel('Total distance [km]'); ax.set_title(c); ax.set_ylim(1e-4, 1e6)
axes[0].set_ylabel('end-to-end scintillation index'); axes[-1].legend(fontsize=7.5)
fig.suptitle('End-to-end scintillation index of η = η_A η_B', y=1.02)
fig.tight_layout(); save(fig, 'g7_scintillation_index_vs_distance')

# ---- G8: outage probability
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, c in zip(axes, cls_names):
    ax.plot(D_STAT, P_OUT[c]['GG'], 'C3', lw=2, label='Gamma-Gamma (per arm)')
    ax.plot(D_STAT, P_OUT[c]['LN'], 'C0', lw=2, label='Log-Normal (per arm)')
    ax.plot(D_STAT, (np.array([eta_nom(L) for L in D_STAT]) < eta_th).astype(float), 'k:', label='Deterministic')
    ax.set_xlabel('Total distance [km]'); ax.set_title(c)
axes[0].set_ylabel(r'P($\eta < \eta_{th}$)  (SKR below threshold)'); axes[-1].legend(fontsize=8)
fig.suptitle(f'Outage probability: P(instantaneous SKR(α={ALPHA_REF}) < {SKR_THRESHOLD:.0e})', y=1.02)
fig.tight_layout(); save(fig, 'g8_outage_probability')

# ---- G9: MC vs quadrature validation
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
for ax, c in zip(axes, cls_names):
    Ls, rel, band = SCAN[c]
    ax.fill_between(Ls, -band * 100, band * 100, color='C3', alpha=.18, label='±2 MC s.e.')
    ax.plot(Ls, rel * 100, 'o-', color='C3', label='(MC − quad)/quad')
    ax.axhline(0, color='k', lw=.6); ax.set_xlabel('Total distance [km]'); ax.set_title(f'{c}: Gamma-Gamma per arm')
axes[0].set_ylabel('relative difference [%]'); axes[0].legend(fontsize=8)
fig.suptitle('Validation: independent Monte Carlo vs tensor-product quadrature (Gamma-Gamma, per arm)', y=1.02)
fig.tight_layout(); save(fig, 'g9_mc_vs_quadrature')

# ---- G10: max range vs Cn2 (continuous)
print("Max range vs Cn2 scan ...")
cn2_scan = np.logspace(-18, -12, 7 if QUICK else 13)
coarse = np.linspace(1.0, L_MAX_KM, 18 if QUICK else 30)
scan_out = {k: [] for k in ('Gamma-Gamma (per arm)', 'Log-Normal (per arm)')}
det_max = MAXR['weak']['Deterministic']
for cn2 in cn2_scan:
    for key in scan_out:
        Rg = np.array([rate(key, cn2, L) for L in coarse])
        above = Rg > SKR_THRESHOLD
        if not above[0]:
            scan_out[key].append(np.nan); continue
        i = 0
        while i + 1 < len(above) and above[i + 1]:
            i += 1
        if i + 1 >= len(coarse):
            scan_out[key].append(L_MAX_KM); continue
        lo, hi = coarse[i], coarse[i + 1]
        for _ in range(8 if QUICK else 11):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if rate(key, cn2, mid) > SKR_THRESHOLD else (lo, mid)
        scan_out[key].append(0.5 * (lo + hi))
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(cn2_scan, scan_out['Gamma-Gamma (per arm)'], 'o-', color='C3', lw=2, label='Gamma-Gamma (per arm)')
ax.semilogx(cn2_scan, scan_out['Log-Normal (per arm)'], 's-', color='C0', lw=2, label='Log-Normal (per arm)')
ax.axhline(det_max, color='gray', ls=':', label=f'Deterministic ({det_max:.1f} km)')
for c, col in zip(cls_names, ('C0', 'C1', 'C2')):
    ax.axvline(CN2[c], color=col, ls='-.', lw=.9); ax.text(CN2[c], ax.get_ylim()[1] * .02 + 5, c, rotation=90, fontsize=8, va='bottom', ha='right')
ax.set_xlabel(r'$C_n^2$ [m$^{-2/3}$]'); ax.set_ylabel('Max total distance [km]')
ax.set_title('Max range vs turbulence strength (per-arm Gamma-Gamma vs Log-Normal)'); ax.legend()
fig.tight_layout(); save(fig, 'g10_max_range_vs_cn2')

print("Done.")
