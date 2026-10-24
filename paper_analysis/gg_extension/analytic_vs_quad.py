"""Analytic Gamma-Gamma cumulant (Euler-operator) expansion vs quadrature, for fixed alpha, ergodic-optimised alpha,
instantaneous-optimised alpha and finite-decoy (LP) key rates. Same formulas as analytical_validation/averaging.py
(naive delta, order 2/3/4 about the log-mean point); derivatives by 5-point finite differences in x = ln(eta_tot)."""
import os, sys, time, csv, math
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
from analytical_validation import averaging as A
from src.turbulence.gamma_gamma import alpha_beta_plane_wave
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','tables'); t0=time.time()
def log(*a): print(f"[{time.time()-t0:5.0f}s]",*a,flush=True)
H=0.1; CLS={'weak':1e-17,'strong':1e-13}; DEC=DEC3
def derivs(f,x):                       # f(x) callable on arrays of x -> values; returns f0,f1,f2,f3,f4 at x
    k=np.arange(-2,3); v=np.asarray(f(x+k*H),float); fm2,fm1,f0,fp1,fp2=v
    return f0,(-fp2+8*fp1-8*fm1+fm2)/(12*H),(-fp2+16*fp1-30*f0+16*fm1-fm2)/(12*H**2),(fp2-2*fp1+2*fm1-fm2)/(2*H**3),(fp2-4*fp1+6*f0-4*fm1+fm2)/H**4
def expansions(f,L,al,be):
    kS=2*A.log_gamma_cumulants(al,be); eta0=eta_nom(L); xm=math.log(eta0)+kS[0]       # log-mean point of S=Y_A+Y_B
    f0,_,f2,f3,f4=derivs(f,xm); m2=kS[1]; m3=kS[2]; m4=kS[3]+3*kS[1]**2
    o2=f0+m2*f2/2; o3=o2+m3*f3/6; o4=o3+m4*f4/24
    g0,g1,g2,_,_=derivs(f,math.log(eta0)); SI=A.gg_si(al,be); naive=g0+0.5*(g2-g1)*((1+SI)**2-1)
    return dict(naive=naive,o2=o2,o3=o3,o4=o4,f_median=f0)
def arm_params(ch): return alpha_beta_plane_wave(ch.sigma_R2_A)
# ---- fine reference tables (quadrature side) ----
log("building tables")
TAB3=YieldTable(DEC,n=300)
lg=np.linspace(-10.5,0,500); Rm=np.zeros_like(lg); Rmf=np.zeros_like(lg)
for i,l in enumerate(lg):
    eta=np.array([10**l]); f=lambda a:float(R_exact(a,eta,sets=SETS_TRI,mode='tri')[0]); Rm[i]=opt_alpha(f)[1]
    ff=lambda a:float(R_exact(a,eta,sets=SETS_7,mode='tri',yfun=TAB3)[0]); Rmf[i]=opt_alpha(ff)[1]
log("tables done")
def direct_ub(etas): return [lp_upper_bounds(observed_gains(float(e),DEC),DEC,ALL_TARGETS) for e in etas]
def yfun_points(etas,ubs):
    etas=np.asarray(etas,float)
    def y(nA,nB,eta):
        eta=np.asarray(eta,float); out=np.ones_like(eta)
        for i,e in enumerate(etas): out[np.isclose(np.log(eta),np.log(e),atol=1e-9)]=ubs[i].get((nA,nB),1.0)
        return out
    return y
rows=[]
for cl,cn2 in CLS.items():
    for L in [20,50,100,150,200,230,250,280]:
        ch=build('gamma_gamma',cn2,L); e,w=nodes(ch,None); al,be=arm_params(ch); eta0=eta_nom(L)
        # stencil etas (both expansion centres)
        kS=2*A.log_gamma_cumulants(al,be); xs=np.concatenate([math.log(eta0)+kS[0]+np.arange(-2,3)*H, math.log(eta0)+np.arange(-2,3)*H])
        etas=np.exp(xs); ubs=direct_ub(etas); yf=yfun_points(etas,ubs)
        def mk(alpha,fin): 
            if fin: return lambda x: R_exact(alpha,np.exp(x),sets=SETS_7,mode='tri',yfun=yf)
            return lambda x: R_exact(alpha,np.exp(x),sets=SETS_TRI,mode='tri')
        def quad(alpha,fin): return float(np.sum(w*(R_exact(alpha,e,sets=SETS_7,mode='tri',yfun=TAB3) if fin else R_exact(alpha,e,sets=SETS_TRI,mode='tri'))))
        for fin in (False,True):
            # (a) fixed alpha = 0.2
            for tag,alpha in (('a0.2',0.2),):
                q=quad(alpha,fin); ex=expansions(mk(alpha,fin),L,al,be)
                rows.append(dict(cls=cl,L_km=L,finite=int(fin),case=tag,alpha=alpha,R_quad=q,**{f'err_{k}_%':100*(v-q)/q if q>0 else float('nan') for k,v in ex.items() if k!='f_median'},R_at_median=ex['f_median']))
            # (b) ergodic-optimised alpha (quadrature optimum) and analytic optimiser
            aq,Rq=opt_alpha(lambda a:quad(a,fin),0.02,0.35); ex=expansions(mk(aq,fin),L,al,be)
            rows.append(dict(cls=cl,L_km=L,finite=int(fin),case='alpha*_quad',alpha=aq,R_quad=Rq,**{f'err_{k}_%':100*(v-Rq)/Rq if Rq>0 else float('nan') for k,v in ex.items() if k!='f_median'},R_at_median=ex['f_median']))
            an=lambda a:expansions(mk(a,fin),L,al,be)['o4']
            aa,Ra=opt_alpha(an,0.02,0.35); Rtrue=quad(aa,fin)
            rows.append(dict(cls=cl,L_km=L,finite=int(fin),case='alpha*_analytic',alpha=aa,R_quad=Rtrue,**{'err_o4_%':100*(Ra-Rtrue)/Rtrue if Rtrue>0 else float('nan'),'regret_vs_quad_opt_%':100*(Rtrue-Rq)/Rq if Rq>0 else float('nan')}))
            # (c) instantaneous-optimised
            Rtab=Rmf if fin else Rm; q=float(np.sum(w*np.interp(np.log10(e),lg,Rtab,left=0.0)))
            def fi(x,fin=fin):
                x=np.atleast_1d(x); out=[]
                for xi in x:
                    eta=np.array([math.exp(xi)])
                    f=(lambda a:float(R_exact(a,eta,sets=SETS_7,mode='tri',yfun=yf)[0])) if fin else (lambda a:float(R_exact(a,eta,sets=SETS_TRI,mode='tri')[0]))
                    out.append(opt_alpha(f)[1])
                return np.array(out)
            ex=expansions(fi,L,al,be)
            rows.append(dict(cls=cl,L_km=L,finite=int(fin),case='alpha_instantaneous',alpha=float('nan'),R_quad=q,**{f'err_{k}_%':100*(v-q)/q if q>0 else float('nan') for k,v in ex.items() if k!='f_median'},R_at_median=ex['f_median']))
        log(cl,L,"done")
keys=sorted({k for r in rows for k in r},key=lambda k:(k not in ('cls','L_km','finite','case','alpha','R_quad'),k))
with open(os.path.join(OUT,'e6_analytic_vs_quadrature.csv'),'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader(); wr.writerows(rows)
log("DONE")
