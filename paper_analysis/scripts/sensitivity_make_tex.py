import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEXD = os.path.join(ROOT, 'paper_analysis', 'latex', 'sensitivity_study')
import csv, math, json
R=list(csv.DictReader(open(os.path.join(TAB,'sensitivity.csv')))); thr=json.load(open(os.path.join(TAB,'sensitivity_thresholds.json')))
def ad(x):
    x=float(x)
    if math.isnan(x): return '--'
    a=abs(x)
    if a<5e-5: return '$0$'
    nd=4 if a<0.01 else 3 if a<1 else 2 if a<10 else 1
    return f'${x:+.{nd}f}$'
MN={'LN':'Log-normal','GG':'Gamma--Gamma (per arm)','GGBB':'Gamma--Gamma + beam wander (per arm)'}
def full(m):
    body=''
    for L in (80,120,160):
        for t in sorted({float(r['sigmaX2']) for r in R}):
            c=[]
            for q in ('eX','eZ','R'):
                r=[x for x in R if x['model']==m and int(x['L_km'])==L and x['quantity']==q and float(x['sigmaX2'])==t][0]
                c+= [ad(r['err_mean_%']),ad(r['err_2nd_%']),ad(r['err_4th_%'])]
            r0=[x for x in R if x['model']==m and int(x['L_km'])==L and x['quantity']=='R' and float(x['sigmaX2'])==t][0]
            extra = f"{float(r0['sigmaR2_arm']):.3g} & {float(r0['fbb2']):.3f}" if m!='LN' else f"{float(r0['sigmaR2_arm']):.3g} & --"
            body+=f"{L if t==0.005 else ''} & {t:g} & {extra} & "+' & '.join(c)+' \\\\\n'
        if L!=160: body+='\\midrule\n'
    return (f"\\begin{{table}}[htbp]\n\\centering\\scriptsize\n\\caption{{Turbulence-sensitivity table, {MN[m]}. Relative error [\\%] of the turbulence-averaged $e_X$, $e_Z$, $R$ for the conventional mean-transmittance (delta) approximation, the 2nd-order and the 4th-order median formula, against the independent quadrature reference, at fixed $L$ and imposed $\\sigma_X^2$. "
            + ("$\\sigma_R^2$: per-arm Rytov variance implied by $\\sigma_X^2=\\sigma_R^2/2$ (manuscript convention), not an independent input." if m=='LN' else "$\\sigma_R^2$: per-arm Rytov variance solved so that $\\mathrm{Var}(X)=\\sigma_X^2$; $f_{\\rm bb}^2$: end-to-end broadening factor for an arm of length $L/2$ with that $\\sigma_R^2$"+(" (applied in this model; $=1$ and unused in the plain GG model)." if m=='GGBB' else " (shown for reference only, not applied in this table)."))
            + f"}}\n\\label{{tab:sens-{m}}}\n\\begin{{tabular}}{{rrrrrrrrrrrrr}}\n\\toprule\n & & & & \\multicolumn{{3}}{{c}}{{$e_X$}} & \\multicolumn{{3}}{{c}}{{$e_Z$}} & \\multicolumn{{3}}{{c}}{{$R$}}\\\\\n\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}\\cmidrule(lr){{11-13}}\n$L$ [km] & $\\sigma_X^2$ & $\\sigma_R^2$ & $f_{{\\rm bb}}^2$ & mean & 2nd & 4th & mean & 2nd & 4th & mean & 2nd & 4th\\\\\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
def fmt(x): return '--' if x!=x else (f'{x:.2f}' if x>=0.01 else f'{x:.3f}')
# threshold table for R
tb=''
for m in MN:
    for L in (80,120,160):
        row=[r for r in thr if r[0]==m and r[1]==L and r[2]=='R'][0]
        cell=lambda i: f"{fmt(row[4+4*i])} / {fmt(row[5+4*i]) if row[5+4*i]==row[5+4*i] else '$>0.8$'} / {row[3+4*i]:.2f}"
        cell=lambda i,row=row: ( ('$<0.005$' if False else fmt(row[4+4*i])) if row[4+4*i]==row[4+4*i] else '$>0.8$') + ' / ' + (fmt(row[5+4*i]) if row[5+4*i]==row[5+4*i] else '$>0.8$') + f' / {row[3+4*i]:.2f}'
        tb+=f"{m.replace('GGBB','GG+BB')} & {L} & {cell(0)} & {cell(1)} & {cell(2)} \\\\\n"
    if m!='GGBB': tb+='\\midrule\n'
tb2=''
for m in MN:
    for L in (80,120,160):
        row=[]
        for q in ('eX','eZ'):
            r=[x for x in thr if x[0]==m and x[1]==L and x[2]==q][0]
            row+= [f'{r[3]:.3f}',f'{r[7]:.3f}',f'{r[11]:.3f}']
        tb2+=f"{m.replace('GGBB','GG+BB')} & {L} & "+' & '.join(row)+' \\\\\n'
    if m!='GGBB': tb2+='\\midrule\n'
doc=open(os.path.join(TEXD,'doc_template.tex')).read()
doc=doc.replace('%%THR%%',tb).replace('%%THR2%%',tb2).replace('%%LN%%',full('LN')).replace('%%GG%%',full('GG')).replace('%%GGBB%%',full('GGBB'))
open(os.path.join(TEXD,'tfqkd_sensitivity_study.tex'),'w').write(doc)
