import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEXD = os.path.join(ROOT, 'paper_analysis', 'latex', 'sensitivity_study')
import csv, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=list(csv.DictReader(open(os.path.join(TAB,'sensitivity.csv'))))
plt.rcParams.update({'font.size':9,'font.family':'serif','axes.grid':True,'grid.alpha':.25,'axes.spines.top':False,'axes.spines.right':False})
COL={'err_mean_%':'#0072B2','err_2nd_%':'#D55E00','err_4th_%':'#009E73'}
LAB={'err_mean_%':'conventional (mean, delta method)','err_2nd_%':'2nd-order median','err_4th_%':'4th-order median'}
MK={'err_mean_%':'o','err_2nd_%':'s','err_4th_%':'^'}
LSD={80:'-',120:'--',160:':'}
TIT={'LN':'Log-normal','GG':'Gamma--Gamma (per arm)','GGBB':'Gamma--Gamma + beam wander'}
FLOOR=1e-5
for m in TIT:
    fig,ax=plt.subplots(1,3,figsize=(13,3.9),sharex=True)
    for a,q,t in zip(ax,('eX','eZ','R'),('(a) $e_X$','(b) $e_Z$','(c) $R$')):
        for k in COL:
            for L in (80,120,160):
                rr=[r for r in R if r['model']==m and int(r['L_km'])==L and r['quantity']==q]
                x=np.array([float(r['sigmaX2']) for r in rr]); e=np.maximum(np.abs([float(r[k]) for r in rr]),FLOOR)
                a.semilogy(x,e,LSD[L],color=COL[k],marker=MK[k],ms=3.5,lw=1.3)
        for y in (0.1,1.0): a.axhline(y,color='gray',lw=.7,ls='-.')
        a.set_title(t); a.set_xlabel(r'$\sigma_X^2$ (imposed log-transmittance variance)'); a.set_ylim(FLOOR*0.8,30)
    ax[0].set_ylabel('|relative error| [%]  (floor $10^{-5}$)')
    h=[Line2D([],[],color=COL[k],marker=MK[k],ms=4,label=LAB[k]) for k in COL]+[Line2D([],[],color='k',ls=LSD[L],label=f'L = {L} km') for L in LSD]
    fig.legend(handles=h,loc='lower center',ncol=6,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.07))
    fig.suptitle(TIT[m].replace('--','–')+': error of the analytic averages vs quadrature reference (grey lines: 0.1 % and 1 %)',y=1.0,fontsize=10)
    fig.savefig(os.path.join(FIG,f'sens_fig_{m}.pdf'),bbox_inches='tight'); fig.savefig(os.path.join(FIG,f'sens_fig_{m}.png'),dpi=150,bbox_inches='tight'); plt.close(fig)
print('ok')
