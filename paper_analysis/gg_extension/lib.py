"""Key-rate evaluation under per-arm fading: CSI and no-CSI averaging, finite-decoy yield tables, alpha optimiser, range search."""
import numpy as np
from scipy.optimize import minimize_scalar
from chan import *
from tfqkd_ext import *

DEC3=[0.0,0.001,0.1]
SETS_TRI={j:S_TRI(j,4) for j in (0,1)}
SETS_7=S_CAL7
THR=1e-9

class YieldTable:
    """LP upper bounds on the 7 CAL yields on a log-eta grid (eta-independent of alpha)."""
    def __init__(self,decoys,n=170,lo=-13,hi=0.0):
        self.decoys=decoys; self.lg=np.linspace(lo,hi,n); self.tab={t:np.zeros(n) for t in ALL_TARGETS}
        for i,l in enumerate(self.lg):
            ub=lp_upper_bounds(observed_gains(10**l,decoys),decoys,ALL_TARGETS)
            for t in ALL_TARGETS: self.tab[t][i]=ub[t]
    def __call__(self,nA,nB,eta):
        t=(nA,nB)
        if t not in self.tab: return np.ones_like(np.asarray(eta,float))
        return np.exp(np.interp(np.log10(np.asarray(eta,float)),self.lg,np.log(np.maximum(self.tab[t],1e-300))))

def get_nodes(L,cn2,model='gamma_gamma',n=None):
    if cn2 is None: return np.array([eta_nom(L)]),np.array([1.0])
    ch=build(model,cn2,L); return nodes(ch,n)

def prep_nocsi(e,w,decoy):
    e=np.asarray(e,float); w=np.asarray(w,float)
    if decoy is None:
        return {t:float(np.sum(w*yield_exact(t[0],t[1],e))) for s_ in SETS_TRI.values() for t in s_}
    gains={(a,b):float(np.sum(w*CAL.p_zz_beta(np.sqrt(a),np.sqrt(b),THETA,e,PD))) for a in decoy for b in decoy}
    return lp_upper_bounds(gains,decoy,ALL_TARGETS)

def R_point(alpha,e,w,conv='csi',decoy=None,sets=SETS_TRI,mode='tri',table=None,ybar=None):
    """ergodic / averaged key rate. conv='csi': E[R(eta)] ; conv='nocsi': rate from averaged statistics"""
    e=np.asarray(e,float); w=np.asarray(w,float)
    if decoy is None: yf=yield_exact
    elif conv=='csi': yf=table
    if conv=='csi':
        return float(np.sum(w*R_exact(alpha,e,sets=sets,mode=mode,yfun=yf)))
    # ---- no-CSI : only averaged observables available ----
    g=np.sqrt(e)*alpha**2
    pxx=CAL.p_xx_total(0.0,THETA,g,PD); eX=np.clip(CAL.bit_error_rate(0.0,THETA,g,PD),0,0.5)
    P=float(np.sum(w*pxx)); EX=float(np.sum(w*pxx*eX)/P) if P>0 else 0.5
    Ybar=ybar if ybar is not None else prep_nocsi(e,w,decoy)
    yf2=lambda nA,nB,eta: np.full(np.shape(np.atleast_1d(eta)),Ybar.get((nA,nB),1.0))
    num=float(eZ_num(alpha,np.array([1e-3]),sets,yf2,mode)[0])
    EZ=min(num/P,0.5) if P>0 else 0.5
    return float(2*P*CAL.secret_key_rate_component(1.0,EX,EZ))

def opt_alpha(fun,lo=0.02,hi=0.35):
    grid=np.linspace(lo,hi,12); vals=[fun(a) for a in grid]; k=int(np.argmax(vals))
    if vals[k]<=0: return float(grid[k]),0.0
    a0,a1=grid[max(k-1,0)],grid[min(k+1,len(grid)-1)]
    r=minimize_scalar(lambda a:-fun(a),bounds=(a0,a1),method='bounded',options={'xatol':1e-4})
    return float(r.x),float(-r.fun)

def max_range(Rfun,Lgrid):
    vals=np.array([Rfun(L) for L in Lgrid]); ab=vals>THR
    if not ab[0]: return None
    i=0
    while i+1<len(ab) and ab[i+1]: i+=1
    if i+1>=len(ab): return float(Lgrid[-1])
    lo,hi=Lgrid[i],Lgrid[i+1]
    for _ in range(14):
        mid=0.5*(lo+hi)
        if Rfun(mid)>THR: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)
