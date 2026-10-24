"""Master Gamma-Gamma table: {plain, +beam broadening (long-term, includes wander)} x {weak, moderate, strong}
x {infinite decoy, finite decoy D3} x {alpha=0.2, ergodic alpha*, instantaneous alpha} ; quadrature vs analytic (naive, o2, o4).
Also maximum ranges (quadrature and analytic, CSI; quadrature no-CSI). Stage selectable: python gg_master.py points|ranges"""
import os, sys, time, csv, math, json
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from analytical_validation import averaging as A
from src.turbulence.gamma_gamma import alpha_beta_plane_wave
from src.turbulence.beam_broadening import beam_broadening_factor
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','tables'); t0=time.time()
def log(*a): print(f"[{time.time()-t0:5.0f}s]",*a,flush=True)
H=0.1; DEC=DEC3; STAGE=sys.argv[1] if len(sys.argv)>1 else 'points'
def fb2(cn2,L): return float(beam_broadening_factor(cn2,LAM,0.5*L*1e3,W0,RX))**2
def derivs(f,x):
    k=np.arange(-2,3); fm2,fm1,f0,fp1,fp2=np.asarray(f(x+k*H),float)
    return f0,(-fp2+8*fp1-8*fm1+fm2)/(12*H),(-fp2+16*fp1-30*f0+16*fm1-fm2)/(12*H**2),(fp2-2*fp1+2*fm1-fm2)/(2*H**3),(fp2-4*fp1+6*f0-4*fm1+fm2)/H**4
def expansions(f,eta0,al,be):
    kS=2*A.log_gamma_cumulants(al,be); xm=math.log(eta0)+kS[0]
    f0,_,f2,f3,f4=derivs(f,xm); m2=kS[1]; m3=kS[2]; m4=kS[3]+3*kS[1]**2
    o2=f0+m2*f2/2; o4=o2+m3*f3/6+m4*f4/24
    g0,g1,g2,_,_=derivs(f,math.log(eta0)); naive=g0+0.5*(g2-g1)*((1+A.gg_si(al,be))**2-1)
    return dict(naive=naive,o2=o2,o4=o4)
def setup(cl,cn2,L,bb,n=None):
    ch=build('gamma_gamma',cn2,L); e,w=nodes(ch,n); s=fb2(cn2,L) if bb else 1.0
    return e*s,w,eta_nom(L)*s,alpha_beta_plane_wave(ch.sigma_R2_A)
TAB3=YieldTable(DEC,n=300)
lg=np.linspace(-10.5,0,500); Rm=np.zeros_like(lg); Rmf=np.zeros_like(lg)
for i,l in enumerate(lg):
    eta=np.array([10**l]); Rm[i]=opt_alpha(lambda a:float(R_exact(a,eta,sets=SETS_TRI,mode='tri')[0]))[1]
    Rmf[i]=opt_alpha(lambda a:float(R_exact(a,eta,sets=SETS_7,mode='tri',yfun=TAB3)[0]))[1]
log("tables")
def direct_ub(etas): return [lp_upper_bounds(observed_gains(float(e),DEC),DEC,ALL_TARGETS) for e in etas]
def yfun_points(etas,ubs):
    etas=np.asarray(etas,float)
    def y(nA,nB,eta):
        eta=np.asarray(eta,float); out=np.ones_like(eta)
        for i,e in enumerate(etas): out[np.isclose(np.log(eta),np.log(e),atol=1e-9)]=ubs[i].get((nA,nB),1.0)
        return out
    return y
CLS={'weak':1e-17,'moderate':1e-15,'strong':1e-13}
if STAGE=='points':
    rows=[]
    for bb in (0,1):
        for cl,cn2 in CLS.items():
            for L in [20,50,100,150,200,250,280]:
                e,w,eta0,(al,be)=setup(cl,cn2,L,bb); kS=2*A.log_gamma_cumulants(al,be)
                xs=np.concatenate([math.log(eta0)+kS[0]+np.arange(-2,3)*H,math.log(eta0)+np.arange(-2,3)*H]); etas=np.exp(xs); yf=yfun_points(etas,direct_ub(etas))
                for fin in (0,1):
                    sets=SETS_7 if fin else SETS_TRI
                    Rf=lambda a,x: R_exact(a,np.exp(x),sets=sets,mode='tri',yfun=yf) if fin else R_exact(a,np.exp(x),sets=sets,mode='tri')
                    quad=lambda a: float(np.sum(w*(R_exact(a,e,sets=sets,mode='tri',yfun=TAB3) if fin else R_exact(a,e,sets=sets,mode='tri'))))
                    det=float(R_exact(0.2,np.array([eta0]),sets=sets,mode='tri',yfun=TAB3 if fin else None)[0])
                    def rec(case,alpha,q,ex): 
                        r=dict(bb=bb,cls=cl,L_km=L,finite=fin,case=case,alpha=alpha,R_det=det,R_quad=q)
                        for k,v in ex.items(): r['R_'+k]=v; r['err_'+k+'_%']=100*(v-q)/q if q>1e-15 else float('nan')
                        rows.append(r)
                    rec('alpha=0.2',0.2,quad(0.2),expansions(lambda x:Rf(0.2,x),eta0,al,be))
                    aq,Rq=opt_alpha(quad,0.02,0.35); rec('alpha*_ergodic',aq,Rq,expansions(lambda x:Rf(aq,x),eta0,al,be))
                    aa,_=opt_alpha(lambda a:expansions(lambda x:Rf(a,x),eta0,al,be)['o4'],0.02,0.35)
                    rows[-1]['alpha_analytic_opt']=aa; rows[-1]['R_true_at_analytic_alpha']=quad(aa)
                    Rtab=Rmf if fin else Rm; qi=float(np.sum(w*np.interp(np.log10(e),lg,Rtab,left=0.0)))
                    def fi(x):
                        out=[]
                        for xi in np.atleast_1d(x):
                            eta=np.array([math.exp(xi)])
                            f=(lambda a:float(R_exact(a,eta,sets=sets,mode='tri',yfun=yf)[0])) if fin else (lambda a:float(R_exact(a,eta,sets=sets,mode='tri')[0]))
                            out.append(opt_alpha(f)[1])
                        return np.array(out)
                    rec('alpha_instantaneous',float('nan'),qi,expansions(fi,eta0,al,be))
                log(bb,cl,L)
    keys=sorted({k for r in rows for k in r},key=lambda k:(k not in ('bb','cls','L_km','finite','case','alpha','R_det','R_quad'),k))
    with open(os.path.join(OUT,'e7_master_points.csv'),'w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader(); wr.writerows(rows)
else:
    Lg=np.arange(100,340,10.0); res={}
    for bb in (0,1):
        for cl,cn2 in CLS.items():
            for fin in (0,1):
                sets=SETS_7 if fin else SETS_TRI
                def Rquad(L,mode,conv='csi'):
                    e,w,eta0,_=setup(cl,cn2,L,bb,150)
                    if mode=='fixed':
                        return R_point(0.2,e,w,conv,DEC if fin else None,sets,'tri',table=TAB3 if fin else None)
                    yb=prep_nocsi(e,w,DEC if fin else None) if conv=='nocsi' else None
                    return opt_alpha(lambda a:R_point(a,e,w,conv,DEC if fin else None,sets,'tri',table=TAB3 if fin else None,ybar=yb))[1]
                def Ran(L,mode):
                    e,w,eta0,(al,be)=setup(cl,cn2,L,bb,150); kS=2*A.log_gamma_cumulants(al,be)
                    xs=np.concatenate([math.log(eta0)+kS[0]+np.arange(-2,3)*H,math.log(eta0)+np.arange(-2,3)*H]); etas=np.exp(xs); yf=yfun_points(etas,direct_ub(etas)) if fin else None
                    Rf=lambda a,x: R_exact(a,np.exp(x),sets=sets,mode='tri',yfun=yf) if fin else R_exact(a,np.exp(x),sets=sets,mode='tri')
                    if mode=='fixed': return max(expansions(lambda x:Rf(0.2,x),eta0,al,be)['o4'],0.0)
                    return max(opt_alpha(lambda a:expansions(lambda x:Rf(a,x),eta0,al,be)['o4'],0.02,0.35)[1],0.0)
                # deterministic range with same BB/decoy
                def Rdet(L,mode):
                    eta0=eta_nom(L)*(fb2(cn2,L) if bb else 1.0); e=np.array([eta0]); w=np.array([1.0])
                    if mode=='fixed': return R_point(0.2,e,w,'csi',DEC if fin else None,sets,'tri',table=TAB3 if fin else None)
                    return opt_alpha(lambda a:R_point(a,e,w,'csi',DEC if fin else None,sets,'tri',table=TAB3 if fin else None))[1]
                for mode in ('fixed','optimised'):
                    key=f'bb{bb}_{cl}_fin{fin}_{mode}'
                    res[key]=dict(det=max_range(lambda L:Rdet(L,mode),Lg),quad_csi=max_range(lambda L:Rquad(L,mode),Lg),
                                  quad_nocsi=max_range(lambda L:Rquad(L,mode,'nocsi'),Lg),analytic_o4=max_range(lambda L:Ran(L,mode),Lg))
                    log(key,res[key])
    json.dump(res,open(os.path.join(OUT,'e7_master_ranges.json'),'w'),indent=1)
log("DONE",STAGE)
