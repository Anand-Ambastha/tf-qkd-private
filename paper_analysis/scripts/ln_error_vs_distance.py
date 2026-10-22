import sys, os, math, warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEX = os.path.join(ROOT, 'paper_analysis', 'latex'); os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True); sys.path.insert(0, ROOT); warnings.filterwarnings('ignore')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from analytical_validation import common as C, cal_derivatives as D, averaging as A
OUT=TAB
def ref(q,eta0,s2,n=40001):
    mu,sd=-s2,math.sqrt(s2); X=np.linspace(mu-12*sd,mu+12*sd,n)
    w=np.exp(-0.5*((X-mu)/sd)**2)/(sd*math.sqrt(2*math.pi))
    return float(simpson(w*A.Q_vec(q,np.minimum(eta0*np.exp(X),1.0)),x=X))
s2f=lambda L: C.rytov(L/2)/2
Rmed=lambda L: C.r_ref(C.eta0_eff(L)*math.exp(-s2f(L)))
Rdet=lambda L: C.r_ref(C.eta0_eff(L))
def bis(f,lo,hi):
    for _ in range(40):
        mid=.5*(lo+hi); (lo,hi)=(mid,hi) if f(mid)>0 else (lo,mid)
    return lo
LA=bis(Rmed,200,260); LD=bis(Rdet,200,300)
print('SKR at expansion point (eta_med) -> 0 at %.2f km ; deterministic SKR -> 0 at %.2f km'%(LA,LD))
Ls=np.unique(np.concatenate([np.arange(2,LA,2.0),[LA]]))
E={q:[] for q in ('eX','eZ','R')}
for L in Ls:
    eta0=C.eta0_eff(L); s2=s2f(L); em,eb=eta0*math.exp(-s2),eta0*math.exp(-s2/2); var=eta0**2*(1-math.exp(-s2))
    for q in E:
        r=ref(q,eta0,s2); dm,db=D.all_derivs(em)[q],D.all_derivs(eb)[q]
        E[q].append([C.pct(A.naive_delta(db,eb,var),r),C.pct(A.ln_expansion(dm,em,s2,2),r),C.pct(A.ln_expansion(dm,em,s2,4),r)])
E={q:np.array(v) for q,v in E.items()}
np.savetxt(os.path.join(OUT,'ln_errors_cut.csv'),np.column_stack([Ls]+[E[q] for q in E]),delimiter=',',header='L_km,'+','.join(f'{q}_{k}' for q in E for k in ('mean','2nd','4th')),comments='')
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.grid':True,'grid.alpha':.3})
sty=[('Mean $\\eta$-space','#ff7f0e','--'),('Median (2-term)','#1f77b4','--'),('Median (4-term)','#d62728',':')]
for q in E:
    fig,ax=plt.subplots(figsize=(4.6,2.7))
    for j,(lab,c,ls) in enumerate(sty): ax.plot(Ls,E[q][:,j],ls,color=c,lw=1.8 if ls==':' else 1.4,label=lab)
    ax.axhline(0,color='k',lw=.6); ax.set_xlabel('$L$ (km)'); ax.set_ylabel('Relative error (%)'); ax.set_xlim(0,LA+3)
    ax.legend(fontsize=7,loc='lower left' if q=='R' else 'best')
    for ext in ('png','pdf'): fig.savefig(os.path.join(FIG,f'ln_error_{q}_cut_800dpi.{ext}'),dpi=800,bbox_inches='tight')
    plt.close(fig); print(q,'at cut: mean %.3f 2nd %.3f 4th %.3f'%tuple(E[q][-1]),'| max|err| mean %.3f 2nd %.3f 4th %.3f'%tuple(np.abs(E[q]).max(0)))
