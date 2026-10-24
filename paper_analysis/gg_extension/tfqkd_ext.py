"""Extension utilities on top of tf-qkd-private (repo code imported unchanged)."""
import sys, numpy as np
import os; sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from math import factorial
from scipy.optimize import linprog
from scipy.special import gammaln
from src import legacy_baseline as CAL

PD=1e-7; TA=np.arcsin(np.sqrt(0.02)); TB=-TA; THETA=TA-TB
h=CAL.h

# ---------------- cat-state coefficients and residuals ----------------
def cvec(j, alpha, m_max): return CAL._cat_coeff_vec(j, alpha, m_max)
def full_sum(j, alpha, N=80): return float(np.sum(cvec(j,alpha,N-1)))
def delta_for_set(j, alpha, pairs):
    """Delta_j = sum over (mA,mB) NOT in pairs of c c  (Eq.22), pairs = list of (nA,nB) in that parity class"""
    c=cvec(j,alpha,79); inS=sum(c[(nA-j)//2]*c[(nB-j)//2] for nA,nB in pairs)
    return max(full_sum(j,alpha)**2 - inS, 0.0)

S_TRI = lambda j,Nmax: [(2*a+j,2*b+j) for a in range(Nmax) for b in range(Nmax) if (2*a+j)+(2*b+j)<=Nmax]
S_CAL7 = {0:[(0,0),(0,2),(2,0),(2,2)], 1:[(1,1),(1,3),(3,1)]}

def yield_exact(nA,nB,eta):
    return np.maximum(CAL.p_zz_photon(nA,nB,TA,TB,eta,PD),0.0)

def legacy_residual(j, alpha, Nmax):
    """Delta_j as computed before v2.4.1 (square set); kept so the *_repo columns stay reproducible."""
    c = cvec(j, alpha, 79); m = (Nmax - j) // 2
    inner = float(np.sum(c[:m + 1])) if m >= 0 else 0.0
    return max(float(np.sum(c)) ** 2 - inner ** 2, 0.0)

def eZ_num(alpha, eta, sets, yfun, mode):
    """sum_j (sum_{S_j} c c sqrt(yield) + Delta_j)^2 ; yfun(nA,nB,eta)->array"""
    eta=np.atleast_1d(eta).astype(float); tot=np.zeros_like(eta)
    for j in (0,1):
        c=cvec(j,alpha,79); S=np.zeros_like(eta)
        for nA,nB in sets[j]:
            S+=c[(nA-j)//2]*c[(nB-j)//2]*np.sqrt(np.maximum(yfun(nA,nB,eta),0.0))
        if mode=='repo': D=legacy_residual(j,alpha,4)
        else: D=delta_for_set(j,alpha,sets[j])
        tot+=(S+D)**2
    return tot

def Rcore(alpha, pxx, eX, eZ):
    return 2.0*CAL.secret_key_rate_component(pxx,eX,np.minimum(eZ,0.5))

def R_exact(alpha, eta, sets=None, mode='tri', yfun=None):
    eta=np.atleast_1d(np.asarray(eta,float))
    if sets is None: sets={j:S_TRI(j,4) for j in (0,1)}
    if yfun is None: yfun=yield_exact
    g=np.sqrt(eta)*alpha**2
    pxx=CAL.p_xx_total(0.0,THETA,g,PD); eX=np.clip(CAL.bit_error_rate(0.0,THETA,g,PD),0,0.5)
    num=eZ_num(alpha,eta,sets,yfun,mode)
    eZ=np.where(pxx>0, num/np.where(pxx>0,pxx,1), 0.5)
    return Rcore(alpha,pxx,eX,np.clip(eZ,0,0.5))

# ---------------- finite-decoy LP (CAL Eq. 26), HiGHS ----------------
def poisson_row(lA,lB,M):
    n=np.arange(M+1)
    pA=np.exp(-lA + n*np.log(max(lA,1e-300)) - gammaln(n+1)) if lA>0 else (n==0).astype(float)
    pB=np.exp(-lB + n*np.log(max(lB,1e-300)) - gammaln(n+1)) if lB>0 else (n==0).astype(float)
    return np.outer(pA,pB)   # [nA,nB]

def lp_upper_bounds(gains, decoys, targets, M=10):
    """gains: dict (lA,lB)->observed gain p_ZZ(beta). returns dict target->upper bound"""
    rows=[];ub=[];lb=[]
    for lA in decoys:
        for lB in decoys:
            a=poisson_row(lA,lB,M).ravel(); g=gains[(lA,lB)]; tau=1.0-a.sum()
            s=1.0/max(g,1e-12)
            rows.append(a*s); ub.append(g*s); lb.append((g-tau)*s)
    A=np.array(rows); ubv=np.array(ub); lbv=np.array(lb)
    A_ub=np.vstack([A,-A]); b_ub=np.concatenate([ubv,-lbv])
    out={}
    for (nA,nB) in targets:
        c=np.zeros((M+1)**2); c[nA*(M+1)+nB]=-1.0
        r=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=(0,1),method='highs')
        out[(nA,nB)]= -r.fun if r.status==0 else 1.0
    return out

def observed_gains(eta, decoys):
    return {(lA,lB): float(CAL.p_zz_beta(np.sqrt(lA),np.sqrt(lB),THETA,eta,PD)) for lA in decoys for lB in decoys}

ALL_TARGETS=[t for j in (0,1) for t in S_CAL7[j]]+[(0,4),(4,0)]
