import csv, math, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
REPO = os.path.join(ROOT, 'analytical_validation', 'results', 'tables') + '/'
TEXD = os.path.join(ROOT, 'paper_analysis', 'latex', 'paper_content') + '/'
SCR = os.path.join(ROOT, 'paper_analysis', 'results', 'tables') + '/'
def rd(p): return list(csv.DictReader(open(p)))
def m(x): return f'${x}$'
def pm(x, nd=3):
    x = float(x)
    if math.isnan(x): return '--'
    if abs(x) < 0.5 * 10 ** (-nd): return f'${0:.{nd}f}$'
    return f'${x:+.{nd}f}$'.replace('+', '+')
def ad(x):
    x = float(x)
    if math.isnan(x): return '--'
    ax = abs(x)
    nd = 4 if ax < 0.01 else (3 if ax < 1 else (2 if ax < 100 else 1))
    return pm(x, nd)
def sci(x, nd=2):
    x = float(x)
    if x == 0: return '$0$'
    mant, ex = f'{x:.{nd}e}'.split('e')
    return f'${mant}\\times10^{{{int(ex)}}}$'
def num(x, nd=2): return f'${float(x):.{nd}f}$'
def g(x, nd=3): return f'${float(x):.{nd}g}$'

T = []
def table(label, caption, cols, head, body, size='\\small', note=None, pos='htbp'):
    s = f'\\begin{{table}}[{pos}]\n\\centering{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\\\\\n\\midrule\n'
    s += body + '\n\\bottomrule\n\\end{tabular}\n'
    if note: s += f'\\par\\smallskip\n\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}\\end{{minipage}}\n'
    s += '\\end{table}\n'
    return s

# ------------------------------------------------ Table params
params = r'''\begin{table}[htbp]
\centering\small
\caption{Operating point used in all tables (identical to \texttt{analytical\_validation/common.py} and \texttt{examples/}).}
\label{tab:params}
\begin{tabular}{ll@{\qquad}ll}
\toprule
wavelength & $\lambda=1550$~nm & signal amplitude & $\alpha=0.2$ \\
atmospheric attenuation & $0.046$~km$^{-1}$ & dark-count probability & $p_d=10^{-7}$ \\
transmitter waist & $w_0=0.05$~m & phase offsets & $\theta_A=-\theta_B=\arcsin\sqrt{0.02}$ \\
receiver radius & $a=0.15$~m ($D=0.30$~m) & photon-number cut-off & $N_{\max}=4$,\; $\varphi=0$ \\
$C_n^2$ classes & $10^{-17},\,10^{-15},\,10^{-13}\ \mathrm{m^{-2/3}}$ (weak / moderate / strong) & rate threshold (max.\ range) & $R>10^{-9}$ bit/pulse \\
\bottomrule
\end{tabular}
\end{table}
'''
T.append(params)

# ------------------------------------------------ Set 1: why LN / approximations fail
val = {r['Cn2']: r for r in rd(REPO + 'lognormal_T0_validity.csv')}
ryt = rd(REPO + 'lognormal_T0_rytov.csv')
body = ''
for cn2 in ('1e-17', '1e-15', '1e-13'):
    rr = {int(r['L_km']): r['sigma_R2(L/2) (this test)'] for r in ryt if float(r['Cn2']) == float(cn2)}
    L = val[cn2]
    base, ex = cn2.split('e'); cn = f'$10^{{{int(ex)}}}$'
    body += f"{cn} & {num(L['L_validity_km (this test)'],2)} & {num(L['Eq21_km'],2)} & {g(rr[10],4)} & {g(rr[50],4)} & {g(rr[100],4)} \\\\\n"
T.append(table('tab:val', 'Validity of the log-normal model. $L_{\\rm val}$ is the total distance at which the per-arm plane-wave Rytov variance $\\sigma_R^2(L/2)=1.23C_n^2k^{7/6}(L/2)^{11/6}$ reaches $1$; $L_{\\rm val}\\simeq0.5547\\,\\lambda^{7/11}(C_n^2)^{-6/11}$ is the closed form (Eq.~21 of the manuscript). Beyond $L_{\\rm val}$ the Rytov perturbation series that underlies the log-normal law no longer converges. Values reproduce the printed Tables I--II of the manuscript.',
    'lccccc', '$C_n^2$ [m$^{-2/3}$] & $L_{\\rm val}$ [km] & closed form [km] & $\\sigma_R^2(L/2)$, $L{=}10$ & $L{=}50$ & $L{=}100$ km', body))

# LN vs GG per arm (new)
lg = rd(SCR + 'ln_vs_gg.csv'); body = ''
names = {'weak': '$10^{-17}$', 'moderate': '$10^{-15}$', 'strong': '$10^{-13}$'}
first = True
for cl in ('weak', 'moderate', 'strong'):
    rows = [r for r in lg if r['class'] == cl and not (cl == 'weak' and float(r['sigmaR2_arm']) > 1.0)]
    if not first: body += '\\midrule\n'
    first = False
    for r in rows:
        body += f"{names[cl]} & {num(r['sigmaR2_arm'],2)} & {num(r['L_km'],2)} & {num(r['SI_LN'],2)} & {num(r['SI_GG'],2)} & {sci(r['R_GG'],3)} & {sci(r['R_LN'],3)} & {pm(r['LN_vs_GG_%'],1)} \\\\\n"
T.append(table('tab:lnvsgg', 'Log-normal (LN) versus Gamma--Gamma (GG) per arm at equal per-arm Rytov variance. $\\mathrm{SI}_{\\rm LN}=e^{\\sigma_R^2}-1$ grows without bound while $\\mathrm{SI}_{\\rm GG}$ saturates near $1$. $R_{\\rm GG}$, $R_{\\rm LN}$: turbulence-averaged key rate [bit/pulse]; error $=100(R_{\\rm LN}-R_{\\rm GG})/R_{\\rm GG}$. Both references are numerical (Bessel-$K$ log-density convolution for GG, Simpson quadrature in $X$ for LN, both with the combined transmittance capped at $1$). Rows with $\\sigma_R^2>1$ of the weak class lie beyond the $252$~km deterministic cut-off ($R<10^{-9}$) and are omitted.',
    'lccccccc', '$C_n^2$ & $\\sigma_R^2(L/2)$ & $L$ [km] & $\\mathrm{SI}_{\\rm LN}$ & $\\mathrm{SI}_{\\rm GG}$ & $R_{\\rm GG}$ & $R_{\\rm LN}$ & LN error [\\%]', body, size='\\small'))

# stress test
st = rd(REPO + 'lognormal_T6_stress.csv'); body = ''
for r in st:
    s2 = float(r['sigma_X2'])
    if s2 not in (0.01, 0.05, 0.1, 0.307, 0.5, 1.0, 1.5, 2.0, 3.0): continue
    g1 = (math.exp(s2) + 2) * math.sqrt(math.exp(s2) - 1)
    if r['err_naive_%'] == 'nan':
        body += f"{s2:g} & {num(r['sigma_R2_per_arm'],2)} & {g(g1,3)} & -- & -- & -- & -- \\\\\n"
    else:
        body += f"{s2:g} & {num(r['sigma_R2_per_arm'],2)} & {g(g1,3)} & {sci(r['R_reference'],3)} & {pm(r['err_naive_%'],3)} & {pm(r['err_2term_%'],3)} & {pm(r['err_4term_%'],3)} \\\\\n"
T.append(table('tab:stress', 'Failure of the log-normal analytic formulas as the log-variance grows (single operating point $L=80$~km, key rate $R$; errors against Gauss--Hermite/Simpson quadrature, $100(\\text{formula}-\\text{ref})/\\text{ref}$). ``mean'' is the delta method about $\\bar\\eta$, ``2-term'' and ``4-term'' are Eq.~\\eqref{eq:ln} truncated after $\\Theta^2$ and $\\Theta^4$. $\\gamma_1$ is the skewness of $\\eta$. For $\\sigma_X^2\\ge2$ the unclipped reference overflows; the three formulas then give $R=4.0\\times10^{-4}$, $2.6\\times10^{-4}$, $3.4\\times10^{-4}$ ($\\sigma_X^2=2$) and $2.9\\times10^{-4}$, $1.2\\times10^{-4}$, $1.8\\times10^{-4}$ ($\\sigma_X^2=3$), i.e.\\ their largest/smallest ratio is $1.5$ ($\\sigma_X^2=2$) and $2.4$ ($\\sigma_X^2=3$).',
    'cccccrr', '$\\sigma_X^2$ & $\\sigma_R^2(L/2)$ & $\\gamma_1$ & $R$ (ref.) & mean [\\%] & 2-term [\\%] & 4-term [\\%]', body, size='\\small'))

# delta method
t5 = rd(REPO + 'lognormal_T5_naive_claims.csv')
cases = [('$20$', '$10^{-17}$', t5[0]), ('$160$', '$10^{-17}$', t5[1]), ('$140$', '$10^{-16}$', t5[2])]
body = ''
for L, cn, r in cases:
    s2 = float(r['sigma_X2']); g1 = (math.exp(s2) + 2) * math.sqrt(math.exp(s2) - 1)
    body += f"{L} & {cn} & {num(s2,3)} & {g(g1,3)} & {pm(r['naive error % (this test)'],3)} & {pm(r['Appendix H %'],3)} \\\\\n"
r = t5[3]
body += f"$160$ (in $\\eta_{{\\rm tot}}$) & $10^{{-17}}$ & {num(r['sigma_X2'],3)} & {g((math.exp(float(r['sigma_X2']))+2)*math.sqrt(math.exp(float(r['sigma_X2']))-1),3)} & {pm(r['naive error % (this test)'],3)} & {pm(r['Appendix H %'],3)} \\\\\n"
T.append(table('tab:delta', 'Breakdown of the delta method (mean-based expansion) for the key rate. $\\gamma_1=(e^{\\sigma_X^2}+2)\\sqrt{e^{\\sigma_X^2}-1}$ is the skewness of $\\eta$; errors are against quadrature. The last row repeats the expansion in the total transmittance $\\eta_{\\rm tot}=\\eta^2$. ``Appendix~H'' lists the values printed in the manuscript; they are reproduced (the printed ``$\\sigma_X^2=9.6$'' for the $140$~km case should read $2.4$).',
    'cccccc', '$L$ [km] & $C_n^2$ & $\\sigma_X^2$ & $\\gamma_1$ & error [\\%] (this work) & printed [\\%]', body, size='\\small'))

# max range LN vs GG
rng = r'''\begin{table}[htbp]
\centering\small
\caption{Maximum range (largest total distance with ergodic key rate $>10^{-9}$ bit/pulse), per-arm models, from \texttt{results\_v2.3.0/reports/GAMMA\_GAMMA\_PER\_ARM\_REPORT.txt}. Every class has its maximum range beyond its own log-normal validity limit, so the log-normal numbers are diagnostic only; Monte Carlo disagrees with the log-normal quadrature at its own maximum-range point by $-28\%$ (moderate) and $-46\%$ (strong).}
\label{tab:range-ln}
\begin{tabular}{lccccc}
\toprule
$C_n^2$ & $L_{\rm val}$ [km] & deterministic [km] & GG [km] & LN [km] & GG$-$LN [km] \\
\midrule
$10^{-17}$ & $208.80$ & $252.04$ & $291.72$ & $317.60$ & $-25.88$ \\
$10^{-15}$ & $16.94$ & $252.04$ & $293.23$ & $132.73$ & $+160.50$ \\
$10^{-13}$ & $1.37$ & $252.04$ & $291.72$ & $13.08$ & $+278.64$ \\
\bottomrule
\end{tabular}
\end{table}
'''
T.append(rng)
set1 = '\n'.join(T[1:]); params_tex = T[0]; T = []

# ------------------------------------------------ Set 2: GG with / without beam wander
bw = rd(SCR + 'gg_beamwander_table.csv'); body = ''; first = True
for cl in ('weak', 'moderate', 'strong'):
    if not first: body += '\\midrule\n'
    first = False
    for r in [x for x in bw if x['class'] == cl]:
        rg, rb = float(r['R_GG']), float(r['R_GG_BB'])
        dag = '$^{\\dagger}$' if rb < 1e-9 else ''
        ratio = rb / rg
        body += (f"{names[cl]} & {int(r['L_km'])} & {g(r['sigmaR2_arm'],3)} & {num(r['W_m'],2)} & {num(r['W_LT_m'],2)} & {num(r['bb_loss_dB'],2)} & "
                 f"{sci(r['R_det'],2)} & {sci(rg,2)} & {sci(rb,2)}{dag} & {(sci(ratio,1) if ratio < 0.01 else '$'+format(ratio,'.3f')+'$')} \\\\\n".replace('& $', '& $'))
T.append(table('tab:bw-skr', 'Gamma--Gamma (per arm) key rate without (GG) and with (GG+BB) turbulence-induced long-term beam broadening, which contains beam wander ($\\alpha=0.2$ fixed, bit/pulse). $W$, $W_{\\rm LT}$: free-space and long-term beam radius at one receiver [m]; BB loss $=-10\\log_{10}f_{\\rm bb}^2$ [dB]; $R_{\\rm det}$: deterministic channel without turbulence. Ergodic rates by the repository\'s 2-D quadrature (300 nodes per arm). $^{\\dagger}$: below the $10^{-9}$ threshold. Weak-class values agree with an independent exact reference (Table~\\ref{tab:perarm-weak} method) to $\\le0.006\\%$.',
    'lcccccccrr', '$C_n^2$ & $L$ [km] & $\\sigma_R^2(L/2)$ & $W$ & $W_{\\rm LT}$ & BB loss [dB] & $R_{\\rm det}$ & $R_{\\rm GG}$ & $R_{\\rm GG+BB}$ & ratio', body, size='\\scriptsize'))
bwr = r'''\begin{table}[htbp]
\centering\small
\caption{Maximum range with and without beam broadening (beam wander), per arm, from \texttt{results\_v2.3.0/reports/BEAM\_BROADENING\_REPORT.txt}. Without broadening Gamma--Gamma saturates and the range is nearly independent of $C_n^2$; with broadening it falls steeply with $C_n^2$. For $\sigma_R^2\gg1$ the broadening law $1.63\,\sigma_R^{12/5}\Lambda$ keeps growing, so the strong-class GG+BB range is a lower bound and plain GG the matching upper bound.}
\label{tab:bw-range}
\begin{tabular}{lccccc}
\toprule
$C_n^2$ & deterministic [km] & det.+BB [km] & GG [km] & GG+BB [km] & BB effect on GG \\
\midrule
$10^{-17}$ & $252.04$ & $249.06$ & $291.72$ & $287.94$ & $-3.78$ km ($-1.3\%$) \\
$10^{-15}$ & $252.04$ & $167.19$ & $293.23$ & $198.61$ & $-94.62$ km ($-32.3\%$) \\
$10^{-13}$ & $252.04$ & $63.79$ & $291.72$ & $81.89$ & $-209.84$ km ($-71.9\%$) \\
\bottomrule
\end{tabular}
\end{table}
'''
T.append(bwr)
set2 = '\n'.join(T); T = []

# ------------------------------------------------ Set 3: analytic approximations and errors
g3 = rd(REPO + 'gammagamma_G3_table1.csv'); g4 = rd(REPO + 'gammagamma_G4_table2.csv'); body = ''
for a, b in zip(g3, g4):
    Lb = a['L_km'].replace('*', '$^{*}$')
    al, be = a['alpha/beta (correct)'].split('/')
    body += (f"{Lb} & {num(a['sigma_R2'],2)} & {num(al,2)}/{num(be,2)} & {pm(a['gain_correct_%'])} & {pm(a['naive_correct_%'])} & {pm(a['o2_correct_%'])} & "
             f"{pm(a['o3_correct_%'])} & {pm(a['o4_correct_%'])} & {pm(b['LN_2nd_correct_%'],2)} & {pm(b['LN_4th_correct_%'],2)} \\\\\n").replace('$/$', '/')
T.append(table('tab:gg-single', 'Gamma--Gamma analytic approximations of $\\mathbb E[R]$, single cumulative channel $\\eta_{\\rm tot}=\\eta_{\\rm FSO}(L)\\,I$, aperture-averaged ($D=0.30$~m, $\\sigma_R^2=\\sigma_R^2(L)$; rows marked $^{*}$ use $L=80$~km with the prescribed $\\sigma_R^2$). Errors [\\%] against tensor Gauss--Laguerre quadrature (confirmed by Bessel-$K$ quadrature to $10^{-8}\\%$ and by Monte Carlo): \\emph{mean} = delta method about $\\mathbb E[\\eta_{\\rm tot}]$; 2nd, 3rd, 4th = Eq.~\\eqref{eq:main} truncated after $\\Theta^2$, $\\Theta^3$, $\\Theta^4$, expanded about the log-mean (``median'') point; ``gain'' is the increase of $\\mathbb E[R]$ from aperture averaging. The last two columns apply the \\emph{log-normal} formula Eq.~\\eqref{eq:ln} with the same Rytov variance to the Gamma--Gamma channel. Shape parameters use $(\\sigma_R^2)^{6/5}$ (code since v2.2.0); with the report\'s $(\\sigma_R^2)^{12/5}$ the 4th-order error stays $\\le0.012\\%$ (here $\\le0.023\\%$).',
    'lccccccccc', 'L [km] & $\\sigma_R^2$ & $\\alpha/\\beta$ & gain & mean & 2nd & 3rd & 4th & LN 2nd & LN 4th', body, size='\\scriptsize',
    note='At $L=20$~km the progress report prints $+0.36\\%$ in all formula columns; with three independent references the errors here are $|\\cdot|\\le0.006\\%$ (formula-independent offset in the report, see Sec.~\\ref{sec:notes}).'))

ep = rd(REPO + 'gammagamma_G6_expansion_point.csv'); body = ''
for r in ep:
    body += f"{r['L_km'].replace('*','$^{*}$')} & {num(r['sigma_R2'],2)} & {pm(r['logmean_2nd'],4)} & {pm(r['logmean_4th'],4)} & {pm(r['mean_2nd'],4)} & {pm(r['mean_4th'],4)} & {pm(r['mode_2nd'],4)} & {pm(r['mode_4th'],4)} \\\\\n"
T.append(table('tab:expoint', 'Choice of expansion point for the Gamma--Gamma formula (single channel as in Table~\\ref{tab:gg-single}): error [\\%] of $\\mathbb E[R]$ when Eq.~\\eqref{eq:main} (2nd and 4th order) is centred at the log-mean (``median''), at the mean, or at the mode of $\\eta$. The log-mean point is best in every row.',
    'lcccccccc'.replace('lcccccccc', 'lccccccc'), 'L [km] & $\\sigma_R^2$ & log-mean 2nd & log-mean 4th & mean 2nd & mean 4th & mode 2nd & mode 4th', body, size='\\small'))

pa = rd(SCR + 'perarm_analytic2.csv')
def perarm_table(cl, label, cn2tex, caption, note=None):
    body = ''
    for bb in ('0', '1'):
        body += ('\\multicolumn{14}{l}{\\emph{' + ('without beam wander (GG)' if bb == '0' else 'with beam wander (GG+BB, $\\eta_0\\to\\eta_0f_{\\rm bb}$)') + '}} \\\\\n')
        for L in (20, 50, 80, 100, 120, 160):
            rows = {r['quantity']: r for r in pa if r['class'] == cl and int(r['L_km']) == L and r['BB'] == bb}
            R0 = rows['R']
            body += f"{L} & {g(R0['sigmaR2_arm'],3)} & {num(R0['c2'],3)} & {num(R0['fbb2'],3) if bb=='1' else '--'}"
            for q in ('eX', 'eZ', 'R'):
                r = rows[q]
                body += f" & {ad(r['err_mean_%'])} & {ad(r['err_med2_%'])} & {ad(r['err_med4_%'])}"
            body += ' \\\\\n'
        if bb == '0': body += '\\midrule\n'
    head = ('$L$ & $\\sigma_R^2(L/2)$ & $c_2$ & $f_{\\rm bb}^2$ & \\multicolumn{3}{c}{$e_X$} & \\multicolumn{3}{c}{$e_Z$} & \\multicolumn{3}{c}{$R$}\\\\\n'
            ' & & & & mean & 2nd & 4th & mean & 2nd & 4th & mean & 2nd & 4th')
    return table(label, caption, 'rcccccccccccc'.replace('rcccccccccccc', 'rccc' + 'ccc' * 3), head.replace('\\\\\n', '\\\\\n\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n', 1), body.replace('{14}', '{13}'), size='\\scriptsize', note=note)
T.append(perarm_table('weak', 'tab:perarm-weak', '$10^{-17}$',
    'Analytic approximations and their errors for the per-arm Gamma--Gamma channel, \\textbf{weak class} $C_n^2=10^{-17}$, with and without beam wander. Error [\\%] of $\\mathbb E[e_X]$, $\\mathbb E[e_Z]$, $\\mathbb E[R]$ relative to an exact reference (convolution of the two arms\' Bessel-$K$ log-densities, $6001$ points; density norm and $\\mathbb E[I_{\\rm tot}]$ equal $1$ to $10^{-11}$). \\emph{mean}: delta method about $\\bar\\eta=\\mathbb E[\\eta]$ in $\\eta$-space (as manuscript Tables IV--VI); \\emph{2nd}, \\emph{4th}: Eq.~\\eqref{eq:main} about $\\eta_m=\\eta_0f_{\\rm bb}e^{c_1}$ truncated after $\\Theta^2$ and $\\Theta^4$, with $c_n=2^{1-n}\\kappa_n(\\alpha,\\beta)$ at $\\sigma_R^2(L/2)$ (plane wave, no aperture). $c_2=\\mathrm{Var}(X)$. Beam wander only shifts $\\eta_0$ ($f_{\\rm bb}^2$ is the end-to-end factor); the repository does not contain this analytic-with-BB comparison, it is computed here from the repository functions.'))
T.append(perarm_table('moderate', 'tab:perarm-mod', '$10^{-15}$',
    'As Table~\\ref{tab:perarm-weak} for the \\textbf{moderate class} $C_n^2=10^{-15}$ ($\\sigma_R^2(L/2)=1.4$--$61$, $\\beta\\to1$, $c_2=O(1)$). There is no small parameter: the 4th-order formula is no longer better than the 2nd, and with beam wander at $L=160$~km ($f_{\\rm bb}^2=4.5\\times10^{-3}$, operating point below the key-rate cut-off) every formula fails.',
    note='The reference is the unclipped Gamma--Gamma product; the repository engine caps each arm at $\\eta\\le1$, which lowers $\\mathbb E[R]$ by $9.4\\%$ at $L=20$~km (arm nominal $0.56$), $\\le0.53\\%$ at $L=50$ and $\\le0.07\\%$ for $L\\ge80$~km; the large $L=20$~km errors of the 4th order are therefore a property of the expansion, not of this cap.'))
set3 = '\n'.join(T)
open(TEXD + 'tables_params.tex', 'w').write(params_tex)
open(TEXD + 'tables_set1.tex', 'w').write(set1)
open(TEXD + 'tables_set2.tex', 'w').write(set2)
open(TEXD + 'tables_set3.tex', 'w').write(set3)
print(len(set1), len(set2), len(set3))
