# -*- coding: utf-8 -*-
"""
example_07_beam_broadening_gamma_gamma.py
=========================================
New model 'gamma_gamma_bb': per-arm Gamma-Gamma fading PLUS per-arm turbulence-induced
long-term beam broadening (src/turbulence/beam_broadening.py).

Compared for weak / moderate / strong Cn2:
    Deterministic                      (free-space diffraction + atmosphere only)
    Deterministic + beam broadening    (broadening only, no fading)   -> isolates the BB effect
    Gamma-Gamma (per arm)              (fading only, v2.2.0 result)
    Gamma-Gamma + BB (per arm)         (NEW: both effects)
Max range = largest total Alice-Bob distance with ergodic SKR > 1e-9 bits/pulse.

Run: python3 examples/example_07_beam_broadening_gamma_gamma.py   (QUICK=1 for a smoke run)
"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import legacy_baseline as CAL
from src.turbulence import create_channel, beam_broadening_factor, long_term_beam_radius
from src.tfqkd_engine import TFQKDEngine

QUICK = bool(os.environ.get('QUICK'))
PD, NMAX, ALPHA_REF = 1e-7, 4, 0.20
TH = np.arcsin(np.sqrt(0.02))
LAM, W0, RX, ALPHA_ATM = 1550e-9, 0.05, 0.15, 0.046
THR = 1e-9
CN2 = dict(CAL.Cn2_classes)
KEYS = ['Deterministic', 'Deterministic + BB', 'Gamma-Gamma (per arm)', 'Gamma-Gamma + BB (per arm)']
COL = {'Deterministic': 'gray', 'Deterministic + BB': 'C2', 'Gamma-Gamma (per arm)': 'C3',
       'Gamma-Gamma + BB (per arm)': 'C4'}
LS = {'Deterministic': ':', 'Deterministic + BB': '--', 'Gamma-Gamma (per arm)': '-', 'Gamma-Gamma + BB (per arm)': '-'}
L_MAX = 450.0
GRID = np.logspace(0.0, np.log10(L_MAX), 40 if QUICK else 110)
NB = 8 if QUICK else 14
MC_N = 200_000 if QUICK else 2_000_000
engine = TFQKDEngine(PD, TH, -TH, Nmax=NMAX)
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures'); DOC = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(FIG, exist_ok=True); os.makedirs(DOC, exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': .3,
                     'font.size': 9.5, 'font.family': 'serif'})


def save(fig, name):
    p = os.path.join(FIG, f'example07_{name}.png')
    fig.savefig(p, bbox_inches='tight'); fig.savefig(p.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig); print('  saved', os.path.basename(p))


def eta_nom(L):
    return float(np.clip(CAL.deterministic_fso_channel(ALPHA_ATM, L, LAM, W0, RX)['eta_fso'], 1e-300, 1.0))


def f_bb(cn2, L):
    return beam_broadening_factor(cn2, LAM, 0.5 * L * 1e3, W0, RX)


def loss_dB(e):
    return -10 * np.log10(max(e, 1e-300))


def sig2(cn2, L):
    return float(CAL.rytov_variance(cn2, LAM, 0.5 * L * 1e3))


def build(model, cn2, L, seed=None):
    return create_channel(model, eta_nominal=eta_nom(L), Cn2=cn2, wavelength=LAM, distance_m=L * 1e3,
                          w0=W0, receiver_radius=RX, seed=seed)


def ergodic_quad(ch, n=400):
    return float(ch.average(lambda e: engine.instantaneous_key_rate(ALPHA_REF, e), method='quadrature',
                            quad_kwargs=dict(n_nodes=n)))


def rate(key, cn2, L):
    if key == 'Deterministic':
        e = eta_nom(L)
    elif key == 'Deterministic + BB':
        e = eta_nom(L) * f_bb(cn2, L) ** 2
    else:
        return ergodic_quad(build('gamma_gamma_bb' if '+ BB' in key else 'gamma_gamma', cn2, L))
    return float(engine.instantaneous_key_rate(ALPHA_REF, np.array([max(e, 1e-300)]))[0])


def max_range(Rg, key, cn2, grid=GRID, nb=NB):
    ab = Rg > THR
    if not ab[0]:
        return None
    i = 0
    while i + 1 < len(ab) and ab[i + 1]:
        i += 1
    if i + 1 >= len(ab):
        return float(grid[-1])
    lo, hi = grid[i], grid[i + 1]
    for _ in range(nb):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rate(key, cn2, mid) > THR else (lo, mid)
    return 0.5 * (lo + hi)


print(f"Sweeping {len(GRID)} distances x 3 classes x {len(KEYS)} models ...")
R, MAXR = {c: {} for c in CN2}, {c: {} for c in CN2}
for c, cn2 in CN2.items():
    for k in KEYS:
        R[c][k] = np.array([rate(k, cn2, L) for L in GRID])
        MAXR[c][k] = max_range(R[c][k], k, cn2)
    print(f"  {c}: " + ", ".join(f"{k}={MAXR[c][k]:.1f}" if MAXR[c][k] else f"{k}=none" for k in KEYS))

# ---- validation of the new model at its max range: MC + quadrature convergence
print("Validation at GG+BB max range ...")
VAL = []
for c, cn2 in CN2.items():
    Lm = MAXR[c]['Gamma-Gamma + BB (per arm)']
    if Lm is None:
        continue
    ch = build('gamma_gamma_bb', cn2, Lm, seed=5)
    qs = [ergodic_quad(ch, n) for n in (200, 400, 800)]
    s = ch.sample(MC_N, rng=np.random.default_rng(77))
    r = engine.instantaneous_key_rate(ALPHA_REF, s)
    VAL.append((c, Lm, qs, float(r.mean()), float(r.std() / np.sqrt(MC_N)), ch.beam_broadening))

# ---- report
ln = ["=" * 100, "GAMMA-GAMMA + TURBULENCE-INDUCED BEAM BROADENING (per arm) -- MAX RANGE", "=" * 100,
      f"lambda={LAM*1e9:.0f} nm  alpha_atm={ALPHA_ATM}/km  w0={W0} m  Rx radius={RX} m  alpha_ref={ALPHA_REF}  pd={PD:.0e}",
      f"Max range = largest TOTAL distance with ergodic SKR > {THR:.0e} bits/pulse (log grid {len(GRID)} pts, {NB} bisections, limit {L_MAX:.0f} km).",
      "BB model: W_LT^2 = W^2 (1 + 1.63 sigma_R^(12/5) Lambda), Lambda = 2z/(k W^2), z = L/2 per arm;",
      "          eta_arm_nominal *= [1-exp(-2a^2/W_LT^2)] / [1-exp(-2a^2/W^2)]; Gamma-Gamma fading unchanged.", ""]
for c, cn2 in CN2.items():
    ln += ["-" * 100, f"{c.upper()}  Cn2 = {cn2:.0e}", "-" * 100,
           f"  {'Model':<30}{'Max range [km]':>16}{'Nominal loss @max [dB]':>25}{'BB loss @max [dB]':>20}"]
    for k in KEYS:
        Lm = MAXR[c][k]
        if Lm is None:
            ln.append(f"  {k:<30}{'none':>16}"); continue
        bbl = -20 * np.log10(f_bb(cn2, Lm)) if '+ BB' in k else 0.0
        ln.append(f"  {k:<30}{Lm:>16.2f}{loss_dB(eta_nom(Lm)):>25.2f}{bbl:>20.2f}")
    g, gb, db = MAXR[c]['Gamma-Gamma (per arm)'], MAXR[c]['Gamma-Gamma + BB (per arm)'], MAXR[c]['Deterministic + BB']
    if g and gb:
        ln.append(f"\n  Beam-broadening effect on Gamma-Gamma range: {gb - g:+.2f} km ({100*(gb-g)/g:+.1f} %)")
    ln.append("")
ln += ["=" * 100, "SUMMARY (max range, km)", "=" * 100,
       f"{'Cn2 class':<10}{'Deterministic':>15}{'Det + BB':>11}{'GG per arm':>13}{'GG + BB':>11}{'BB effect on GG':>18}"]
for c in CN2:
    m = MAXR[c]; f = lambda k: f"{m[k]:.1f}" if m[k] else 'none'
    d = (f"{m['Gamma-Gamma + BB (per arm)'] - m['Gamma-Gamma (per arm)']:+.1f} km"
         if m['Gamma-Gamma + BB (per arm)'] and m['Gamma-Gamma (per arm)'] else 'n/a')
    ln.append(f"{c:<10}{f('Deterministic'):>15}{f('Deterministic + BB'):>11}{f('Gamma-Gamma (per arm)'):>13}"
              f"{f('Gamma-Gamma + BB (per arm)'):>11}{d:>18}")
ln += ["", "=" * 100, "VALIDATION OF 'gamma_gamma_bb' AT ITS MAX-RANGE POINT", "=" * 100,
       f"{'Cn2':<10}{'L_max':>8}{'quad n=200':>13}{'n=400':>12}{'n=800':>12}{'MC':>12}{'MC s.e.':>10}{'(MC-q800)/q800':>16}"
       f"{'W [m]':>8}{'W_LT [m]':>10}{'f_bb/arm':>10}"]
for c, Lm, qs, rm, se, bb in VAL:
    ln.append(f"{c:<10}{Lm:>8.2f}{qs[0]:>13.3e}{qs[1]:>12.3e}{qs[2]:>12.3e}{rm:>12.3e}{se:>10.1e}"
              f"{(rm-qs[2])/qs[2]:>+16.2%}{bb['W_A']:>8.2f}{bb['W_LT_A']:>10.2f}{bb['factor_A']:>10.3g}")
ln += ["", "Notes:",
       "* BB depends only on the arm Rytov variance; it is a nominal-loss term, so it is NOT bounded by the",
       "  fading saturation that made plain Gamma-Gamma range nearly Cn2-independent. With BB, range now",
       "  falls with Cn2.",
       "* Long-term beam includes beam wander: do not add extra wander jitter on top (double counting).",
       "* Assumes horizontal path, constant Cn2, collimated Gaussian beam; Gamma-Gamma alpha/beta are the",
       "  plane-wave values (first-order correction). Arm amplitude-imbalance visibility penalty not modelled.",
       "=" * 100]
rep = "\n".join(ln); print("\n" + rep)
open(os.path.join(DOC, 'BEAM_BROADENING_REPORT.txt'), 'w').write(rep + "\n")

# ---- figures
print("Figures ...")
names = list(CN2)
Dd = np.logspace(0, np.log10(300), 160)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for a, c in zip(ax, names):
    Wf = [long_term_beam_radius(CN2[c], LAM, 0.5 * L * 1e3, W0)[0] for L in Dd]
    Wl = [long_term_beam_radius(CN2[c], LAM, 0.5 * L * 1e3, W0)[1] for L in Dd]
    a.loglog(Dd, Wf, 'k:', label='free-space W(z)'); a.loglog(Dd, Wl, 'C4', lw=2, label='long-term $W_{LT}$(z)')
    a.axhline(RX, color='C1', ls='--', label='receiver radius'); a.set_xlabel('Total distance [km] (arm z = L/2)'); a.set_title(f'{c} (Cn2={CN2[c]:.0e})')
ax[0].set_ylabel('beam radius at receiver [m]'); ax[0].legend(fontsize=8)
fig.suptitle('Beam radius per arm: free space vs turbulence-broadened', y=1.02); fig.tight_layout(); save(fig, 'b1_beam_radius')

fig, a = plt.subplots(figsize=(8, 5))
for c, col in zip(names, ('C0', 'C1', 'C2')):
    a.semilogy(Dd, [max(-20 * np.log10(f_bb(CN2[c], L)), 1e-4) for L in Dd], color=col, lw=2, label=f'{c} ({CN2[c]:.0e})')
a.set_xlabel('Total distance [km]'); a.set_ylabel('beam-broadening loss, both arms [dB]'); a.legend()
a.set_title('Extra nominal loss caused by turbulence-induced beam broadening'); fig.tight_layout(); save(fig, 'b2_bb_loss_vs_distance')

fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for a, c in zip(ax, names):
    for k in KEYS:
        a.semilogy(GRID, np.clip(R[c][k], 1e-12, None), LS[k], color=COL[k], lw=2.2 if '+ BB' in k and 'Gamma' in k else 1.3, label=k)
        if MAXR[c][k] and 'Gamma' in k:
            a.plot([MAXR[c][k]], [THR], 'o', color=COL[k])
    a.axhline(THR, color='k', lw=.6, ls=':'); a.set_ylim(1e-10, 1); a.set_xlim(0, 330); a.set_xlabel('Total distance [km]'); a.set_title(c)
ax[0].set_ylabel('Ergodic SKR [bits/pulse]'); ax[-1].legend(fontsize=7.5)
fig.suptitle('SKR vs distance: Gamma-Gamma with and without beam broadening', y=1.02); fig.tight_layout(); save(fig, 'b3_skr_vs_distance')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
with np.errstate(divide='ignore', invalid='ignore'):
    for a, c in zip(ax, names):
        base = R[c]['Deterministic']
        for k in KEYS[1:]:
            a.semilogy(GRID, np.clip(np.where(base > 1e-12, R[c][k] / base, np.nan), 1e-8, None), LS[k], color=COL[k], lw=2, label=k)
        a.axhline(1, color='gray', lw=.6); a.set_xlim(0, 300); a.set_xlabel('Total distance [km]'); a.set_title(c)
ax[0].set_ylabel('SKR / deterministic SKR'); ax[-1].legend(fontsize=8)
fig.suptitle('Penalty relative to the no-turbulence channel', y=1.02); fig.tight_layout(); save(fig, 'b4_penalty_ratio')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for a, c in zip(ax, names):
    a.plot(Dd, [loss_dB(eta_nom(L)) for L in Dd], 'k:', label='free-space + atmosphere')
    a.plot(Dd, [loss_dB(eta_nom(L)) - 20 * np.log10(f_bb(CN2[c], L)) for L in Dd], 'C4', lw=2, label='+ beam broadening')
    a.set_xlabel('Total distance [km]'); a.set_title(c)
ax[0].set_ylabel('nominal channel loss [dB]'); ax[0].legend(fontsize=8)
fig.suptitle('Nominal (fading-free) loss budget', y=1.02); fig.tight_layout(); save(fig, 'b5_nominal_loss')

fig, a = plt.subplots(figsize=(10, 5)); w = .2
for j, k in enumerate(KEYS):
    vals = [MAXR[c][k] or 0 for c in names]
    bars = a.bar(np.arange(3) + (j - 1.5) * w, vals, w, color=COL[k], label=k, alpha=.9)
    for b_, v in zip(bars, vals):
        a.text(b_.get_x() + b_.get_width() / 2, v + 4, f'{v:.0f}', ha='center', fontsize=8)
a.set_xticks(range(3)); a.set_xticklabels([f'{c}\nCn2={CN2[c]:.0e}' for c in names]); a.set_ylabel('Max total distance [km]')
a.set_title('Max range: effect of beam broadening on per-arm Gamma-Gamma'); a.legend(fontsize=8); fig.tight_layout(); save(fig, 'b6_max_range_bars')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for a, c in zip(ax, names):
    L0 = 40.0; rng = np.random.default_rng(1)
    for key, col in (('gamma_gamma', 'C3'), ('gamma_gamma_bb', 'C4')):
        s = build(key, CN2[c], L0, seed=2).sample(MC_N, rng=rng)
        h, e = np.histogram(10 * np.log10(np.clip(s, 1e-30, None)), bins=np.linspace(-130, 0, 200), density=True)
        a.semilogy(0.5 * (e[1:] + e[:-1]), np.clip(h, 1e-6, None), color=col, label=key)
    a.axvline(-loss_dB(eta_nom(L0)), color='gray', ls=':', label='deterministic'); a.set_ylim(1e-5, 1)
    a.set_xlabel('end-to-end transmittance [dB]'); a.set_title(f'{c}, L={L0:.0f} km')
ax[0].set_ylabel('density'); ax[0].legend(fontsize=8)
fig.suptitle('End-to-end transmittance distribution with / without beam broadening', y=1.02); fig.tight_layout(); save(fig, 'b7_transmittance_distribution')

print("Max range vs Cn2 ...")
cs = np.logspace(-18, -12, 7 if QUICK else 13); cg = np.logspace(0, np.log10(L_MAX), 14 if QUICK else 24)
out = {k: [] for k in ('Gamma-Gamma (per arm)', 'Gamma-Gamma + BB (per arm)', 'Deterministic + BB')}
for cn2 in cs:
    for k in out:
        Rg = np.array([rate(k, cn2, L) for L in cg])
        out[k].append(max_range(Rg, k, cn2, grid=cg, nb=7 if QUICK else 9) or np.nan)
fig, a = plt.subplots(figsize=(9, 5))
for k, mk in zip(out, ('o-', 's-', '^--')):
    a.semilogx(cs, out[k], mk, color=COL[k], lw=2, label=k)
a.axhline(MAXR['weak']['Deterministic'], color='gray', ls=':', label='Deterministic')
a.set_xlabel(r'$C_n^2$ [m$^{-2/3}$]'); a.set_ylabel('Max total distance [km]'); a.legend(fontsize=8)
a.set_title('Max range vs turbulence strength'); fig.tight_layout(); save(fig, 'b8_max_range_vs_cn2')
print("Done.")
