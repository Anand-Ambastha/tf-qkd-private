import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
TAB = os.path.join(ROOT, 'paper_analysis', 'results', 'tables'); FIG = os.path.join(ROOT, 'paper_analysis', 'results', 'figures'); TEXD = os.path.join(ROOT, 'paper_analysis', 'latex', 'sensitivity_study')
import csv, math, numpy as np
R=list(csv.DictReader(open(os.path.join(TAB,'sensitivity.csv'))))
def series(m,L,q,k):
    rr=[r for r in R if r['model']==m and int(r['L_km'])==L and r['quantity']==q]
    return np.array([float(r['sigmaX2']) for r in rr]), np.array([float(r[k]) for r in rr])
def cross(x,e,thr):
    a=np.abs(e)
    for i in range(1,len(x)):
        if a[i]>thr>=a[i-1]:
            f=(math.log(thr)-math.log(a[i-1]))/(math.log(a[i])-math.log(a[i-1])); return x[i-1]+f*(x[i]-x[i-1])
    return float('nan') if a[0]<=thr else x[0]
out=[]
for m in ('LN','GG','GGBB'):
  for L in (80,120,160):
    for q in ('eX','eZ','R'):
      row=[m,L,q]
      for k in ('err_mean_%','err_2nd_%','err_4th_%'):
        x,e=series(m,L,q,k); row+= [np.abs(e).max(), cross(x,e,0.1), cross(x,e,1.0), bool(np.all(np.diff(np.abs(e))>=-1e-9))]
      out.append(row)
      print(m,L,q,' | '.join(f"max={row[3+4*i]:.3g} t0.1={row[4+4*i]:.3g} t1={row[5+4*i]:.3g} mono={row[6+4*i]}" for i in range(3)))
import json; json.dump([[x if not isinstance(x,(np.floating,np.bool_)) else (bool(x) if isinstance(x,np.bool_) else float(x)) for x in r] for r in out], open(os.path.join(TAB,'sensitivity_thresholds.json'),'w'))
