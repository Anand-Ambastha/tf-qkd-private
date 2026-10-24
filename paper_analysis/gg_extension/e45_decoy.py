import sys, time, json, csv
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','tables'); os.makedirs(OUT,exist_ok=True)
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]",*a,flush=True)
DECOYS={'D3 {0,.001,.1}':[0.0,0.001,0.1],'D2 {0,.1}':[0.0,0.1],'D4 {0,.001,.1,.3}':[0.0,0.001,0.1,0.3],'D5 {0,.001,.01,.1,.5}':[0.0,0.001,0.01,0.1,0.5]}
TAB={k:YieldTable(v) for k,v in DECOYS.items()}; log("tables built")
# ---------- yield-tightness table at 20/40/60/80 dB ----------
tight=[]
for Ldb in (20,40,60,80):
    eta=10**(-Ldb/10)
    for k,dv in DECOYS.items():
        ub=lp_upper_bounds(observed_gains(eta,dv),dv,ALL_TARGETS)
        for t in ALL_TARGETS:
            ex=float(yield_exact(t[0],t[1],eta)); tight.append(dict(loss_dB=Ldb,decoys=k,nA=t[0],nB=t[1],exact=ex,LP_upper=ub[t],ratio=ub[t]/ex))
with open(os.path.join(OUT,'e4_yield_tightness.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(tight[0].keys())); wr.writeheader(); wr.writerows(tight)
log("tightness done")
# ---------- deterministic finite-decoy vs loss ----------
def Rdet_opt(Ldb,kind):
    eta=np.array([10**(-Ldb/10)]); w=np.array([1.0])
    if kind=='inf_tri': f=lambda a:R_point(a,eta,w,'csi',None,SETS_TRI,'tri')
    elif kind=='inf_7': f=lambda a:R_point(a,eta,w,'csi',None,SETS_7,'tri')
    elif kind.startswith('fin7:'): f=lambda a:R_point(a,eta,w,'csi',DECOYS[kind[5:]],SETS_7,'tri',table=TAB[kind[5:]])
    elif kind.startswith('finT:'): f=lambda a:R_point(a,eta,w,'csi',DECOYS[kind[5:]],SETS_TRI,'tri',table=TAB[kind[5:]])
    a,R=opt_alpha(f); return a,R
kinds=['inf_tri','inf_7']+[f'fin7:{k}' for k in DECOYS]+[f'finT:{k}' for k in DECOYS]
LD=np.arange(2,86,2.0); out=[]
for Ldb in LD:
    r={'loss_dB':Ldb}
    for kd in kinds:
        a,R=Rdet_opt(Ldb,kd); r['R_'+kd]=R; r['a_'+kd]=a
    out.append(r)
with open(os.path.join(OUT,'e4_det_vs_loss.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
log("det vs loss done")
# cutoff distance (R>1e-9) with alpha optimised, deterministic
Lg=np.arange(100,330,10.0); cut={}
for kd in kinds:
    f=lambda L,kd=kd: Rdet_opt(-10*np.log10(eta_nom(L)),kd)[1]
    cut[kd]=max_range(f,Lg); log("det range",kd,cut[kd])
json.dump(cut,open(os.path.join(OUT,'e4_det_ranges.json'),'w'),indent=1)
# ---------- GG per-arm finite decoy (CSI and no-CSI) ----------
NN=150; rows=[]
LG2=np.arange(20,321,20.0)
for cl,cn2 in (('weak',1e-17),('strong',1e-13)):
    for L in LG2:
        e,w=get_nodes(L,cn2,'gamma_gamma',NN); r=dict(cls=cl,L_km=L)
        def opt(conv,decoy,sets,table=None):
            yb=prep_nocsi(e,w,decoy) if conv=='nocsi' else None
            f=lambda a:R_point(a,e,w,conv,decoy,sets,'tri',table=table,ybar=yb); return opt_alpha(f)
        for nm,(conv,decoy,sets,tb) in {'inf_tri_csi':('csi',None,SETS_TRI,None),'inf7_csi':('csi',None,SETS_7,None),'fin3_csi':('csi',DEC3,SETS_7,TAB['D3 {0,.001,.1}']),
                                        'inf_tri_nocsi':('nocsi',None,SETS_TRI,None),'inf7_nocsi':('nocsi',None,SETS_7,None),'fin3_nocsi':('nocsi',DEC3,SETS_7,None)}.items():
            a,R=opt(conv,decoy,sets,tb); r['R_'+nm]=R; r['a_'+nm]=a
        r['R_det_inf_tri']=Rdet_opt(-10*np.log10(eta_nom(L)),'inf_tri')[1]
        r['R_det_fin3']=Rdet_opt(-10*np.log10(eta_nom(L)),'fin7:D3 {0,.001,.1}')[1]
        rows.append(r); log("E5",cl,L,r['R_fin3_csi'],r['R_fin3_nocsi'])
with open(os.path.join(OUT,'e5_gg_finite_decoy.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
log("DONE")
