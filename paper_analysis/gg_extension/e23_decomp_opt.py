import sys, time, json, csv
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','tables'); os.makedirs(OUT,exist_ok=True)
t0=time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]",*a,flush=True)
NN=250
# ---------- Rmax(eta) table (alpha optimised per eta, exact yields, tri set, corrected residual) ----------
lg=np.linspace(-10.5,0,260); Rm=np.zeros_like(lg); Am=np.zeros_like(lg)
for i,l in enumerate(lg):
    eta=np.array([10**l]); f=lambda a:float(R_exact(a,eta,sets=SETS_TRI,mode='tri')[0])
    Am[i],Rm[i]=opt_alpha(f)
log("Rmax table built")
np.savetxt(os.path.join(OUT,'alpha_star_vs_eta.csv'),np.c_[lg,Am,Rm],delimiter=',',header='log10_eta,alpha_star,Rmax',comments='')
Rmax_interp=lambda e: np.interp(np.log10(e),lg,Rm,left=0.0)
# ---------- E2: decomposition at selected distances (GG weak) ----------
dec=[]
for L in [100,150,200,250,270,290]:
    e,w=get_nodes(L,1e-17,'gamma_gamma',NN)
    e0=eta_nom(L); a=0.2
    Rdet=float(R_exact(a,np.array([e0]),sets=SETS_TRI,mode='tri')[0])
    Rv=R_exact(a,e,sets=SETS_TRI,mode='tri'); Rcsi=float(np.sum(w*Rv))
    Rno=R_point(a,e,w,'nocsi',mode='tri')
    g=np.sqrt(e)*a**2; pxx=CAL.p_xx_total(0.0,THETA,g,PD); eX=np.clip(CAL.bit_error_rate(0.0,THETA,g,PD),0,0.5)
    num=eZ_num(a,e,SETS_TRI,yield_exact,'tri'); eZ=np.clip(num/np.maximum(pxx,1e-300),0,0.5)
    P=float(np.sum(w*pxx)); exa=float(np.sum(w*pxx*eX)/P); eza=float(np.sum(w*num)/P)      # gain-weighted averages (CSI)
    # no-CSI eZ
    Yb={t:float(np.sum(w*yield_exact(t[0],t[1],e))) for s in SETS_TRI.values() for t in s}
    yf=lambda nA,nB,eta: np.full(np.shape(np.atleast_1d(eta)),Yb.get((nA,nB),1.0))
    ezn=float(eZ_num(a,np.array([1e-3]),SETS_TRI,yf,'tri')[0]/P)
    # contribution share of fades above nominal / above 2x nominal, and top-fade share
    order=np.argsort(e); cw=np.cumsum((w*Rv)[order])/Rcsi; es=e[order]
    share_above=1-float(np.interp(e0,es,cw)); share_above2=1-float(np.interp(2*e0,es,cw))
    share_above5=1-float(np.interp(5*e0,es,cw))
    prob_above=float(np.sum(w[e>e0]))
    # local power-law exponent of R(eta) at nominal
    d=1e-3; p=float((np.log(R_exact(a,np.array([e0*(1+d)]),sets=SETS_TRI,mode='tri')[0])-np.log(R_exact(a,np.array([e0*(1-d)]),sets=SETS_TRI,mode='tri')[0]))/(np.log(1+d)-np.log(1-d))) if Rdet>0 else float('nan')
    dec.append(dict(L_km=L,loss_dB=-10*np.log10(e0),R_det=Rdet,R_GG_csi=Rcsi,R_GG_nocsi=Rno,ratio_csi=Rcsi/Rdet if Rdet>0 else float('nan'),ratio_nocsi=Rno/Rdet if Rdet>0 else float('nan'),
                    eX_det=float(np.clip(CAL.bit_error_rate(0.0,THETA,np.sqrt(e0)*a**2,PD),0,.5)),eX_avg=exa,
                    eZ_det=float(np.clip(eZ_num(a,np.array([e0]),SETS_TRI,yield_exact,'tri')[0]/CAL.p_xx_total(0.0,THETA,np.sqrt(e0)*a**2,PD),0,.5)),
                    eZ_avg_csi=eza,eZ_avg_nocsi=ezn,pXX_det=float(CAL.p_xx_total(0.0,THETA,np.sqrt(e0)*a**2,PD)),pXX_avg=P,
                    P_eta_gt_nom=prob_above,Rshare_eta_gt_nom=share_above,Rshare_eta_gt_2nom=share_above2,Rshare_eta_gt_5nom=share_above5,local_exponent_p=p))
    log("E2",L,dec[-1]['ratio_csi'],dec[-1]['ratio_nocsi'])
with open(os.path.join(OUT,'e2_decomposition.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(dec[0].keys())); wr.writeheader(); wr.writerows(dec)
# ---------- E3: optimisation ----------
rows=[]
cases=[('GG','weak',1e-17,'gamma_gamma'),('GG','strong',1e-13,'gamma_gamma'),('LN','weak',1e-17,'lognormal')]
for L in [20,50,100,150,200,250,280]:
    for mn,cl,cn2,m in cases:
        e,w=get_nodes(L,cn2,m,NN)
        r=dict(model=mn,cls=cl,L_km=L)
        f02=lambda a:R_point(a,e,w,'csi',mode='tri'); r['R_a0.2_csi']=f02(0.2)
        a1,R1=opt_alpha(f02,0.02,0.35); r['alpha_star_csi']=a1; r['R_opt_csi']=R1
        r['R_inst_csi']=float(np.sum(w*Rmax_interp(e)))
        yb=prep_nocsi(e,w,None)
        f02n=lambda a:R_point(a,e,w,'nocsi',mode='tri',ybar=yb); r['R_a0.2_nocsi']=f02n(0.2)
        a2,R2=opt_alpha(f02n,0.02,0.35); r['alpha_star_nocsi']=a2; r['R_opt_nocsi']=R2
        e0=eta_nom(L); r['alpha_star_det']=float(np.interp(np.log10(e0),lg,Am)); r['R_opt_det']=float(np.interp(np.log10(e0),lg,Rm))
        r['R_a0.2_det']=float(R_exact(0.2,np.array([e0]),sets=SETS_TRI,mode='tri')[0])
        rows.append(r); log("E3",mn,cl,L,r['alpha_star_csi'],r['R_opt_csi']/max(r['R_a0.2_csi'],1e-300))
with open(os.path.join(OUT,'e3_optimisation.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
log("DONE")
