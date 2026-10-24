import sys, time, csv, json
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','tables'); os.makedirs(OUT,exist_ok=True)
t0=time.time()
LG=np.arange(5,335,5.0)
rows=[]; 
def log(*a): print(f"[{time.time()-t0:6.0f}s]",*a,flush=True)
for L in LG:
    r={'L_km':L,'loss_dB':-10*np.log10(eta_nom(L))}
    e0=np.array([eta_nom(L)]); w0=np.array([1.0])
    r['R_det']=R_point(0.2,e0,w0,'csi',mode='tri'); r['R_det_repo']=R_point(0.2,e0,w0,'csi',mode='repo')
    for c,cn2 in CN2.items():
        for m,mn in (('gamma_gamma','GG'),('lognormal','LN')):
            e,w=get_nodes(L,cn2,m,250)
            r[f'R_{mn}_csi_{c}']=R_point(0.2,e,w,'csi',mode='tri')
            r[f'R_{mn}_csi_repo_{c}']=R_point(0.2,e,w,'csi',mode='repo')
            r[f'R_{mn}_nocsi_{c}']=R_point(0.2,e,w,'nocsi',mode='tri')
    rows.append(r)
    if int(L)%25==0: log("L",L)
keys=list(rows[0].keys())
with open(os.path.join(OUT,'e1_sweeps.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader(); wr.writerows(rows)
log("sweeps done")
# ---- max ranges ----
def rng_from(col_fun):
    return max_range(col_fun,LG)
res={}
def det(mode): return lambda L: R_point(0.2,np.array([eta_nom(L)]),np.array([1.0]),'csi',mode=mode)
res['det_tri']=max_range(det('tri'),LG); res['det_repo']=max_range(det('repo'),LG)
for c,cn2 in CN2.items():
    for m,mn in (('gamma_gamma','GG'),('lognormal','LN')):
        for conv,mode,tag in (('csi','repo','csi_repo'),('csi','tri','csi_tri'),('nocsi','tri','nocsi_tri')):
            if mn=='LN' and conv=='nocsi': continue
            def f(L,m=m,cn2=cn2,conv=conv,mode=mode):
                e,w=get_nodes(L,cn2,m,None); return R_point(0.2,e,w,conv,mode=mode)
            res[f'{mn}_{tag}_{c}']=max_range(f,LG)
            log("range",mn,tag,c,res[f'{mn}_{tag}_{c}'])
json.dump(res,open(os.path.join(OUT,'e1_ranges.json'),'w'),indent=1)
log("DONE",res)
