"""Figures for the Gamma-Gamma paper, from results/tables/*.csv|json (run e1, e23, e45 first)."""
import os, json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
H=os.path.dirname(os.path.abspath(__file__)); T=os.path.join(H,'results','tables'); F=os.path.join(H,'results','figures'); os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':9.5,'axes.grid':True,'grid.alpha':.3,'savefig.dpi':300})
def save(fig,n): fig.savefig(os.path.join(F,n+'.png'),bbox_inches='tight'); fig.savefig(os.path.join(F,n+'.pdf'),bbox_inches='tight'); plt.close(fig); print('saved',n)
s=pd.read_csv(os.path.join(T,'e1_sweeps.csv')); cls=['weak','moderate','strong']
# F1 SKR vs distance
fig,ax=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
for a,c in zip(ax,cls):
    a.semilogy(s.L_km,s.R_det.clip(lower=1e-30),'k:',label='deterministic')
    a.semilogy(s.L_km,s[f'R_LN_csi_{c}'].clip(lower=1e-30),'C0',label='log-normal per arm')
    a.semilogy(s[ 'L_km'],s[f'R_GG_csi_{c}'].clip(lower=1e-30),'C3',lw=2,label='Gamma-Gamma per arm (CSI)')
    a.semilogy(s.L_km,s[f'R_GG_nocsi_{c}'].clip(lower=1e-30),'C3--',label='Gamma-Gamma per arm (no CSI)')
    a.axhline(1e-9,c='gray',lw=.6); a.set_ylim(1e-10,1); a.set_title(c); a.set_xlabel('total distance [km]')
ax[0].set_ylabel('key rate [bit/pulse]'); ax[0].legend(fontsize=7); save(fig,'gg_F1_skr_vs_distance')
# F2 CSI vs no-CSI ratio
fig,ax=plt.subplots(figsize=(5,3.6))
for c,col in zip(['weak','strong'],['C0','C3']):
    m=s[f'R_GG_csi_{c}']>1e-9; ax.plot(s.L_km[m],(s[f'R_GG_nocsi_{c}']/s[f'R_GG_csi_{c}'])[m],col,label=c)
ax.set_xlabel('total distance [km]'); ax.set_ylabel('R(no CSI) / R(CSI)'); ax.legend(); save(fig,'gg_F2_csi_vs_nocsi')
# F3 max ranges
r=json.load(open(os.path.join(T,'e1_ranges.json')))
fig,ax=plt.subplots(figsize=(6.5,3.6)); x=np.arange(3); wd=.16
series=[('det_tri',None,'deterministic'),('LN_csi_tri',1,'LN'),('GG_csi_tri',1,'GG (CSI)'),('GG_nocsi_tri',1,'GG (no CSI)')]
for i,(k,_,lab) in enumerate(series):
    v=[r[k] if k=='det_tri' else r[f'{k}_{c}'] for c in cls]; ax.bar(x+(i-1.5)*wd,[0 if q is None else q for q in v],wd,label=lab)
ax.set_xticks(x); ax.set_xticklabels(cls); ax.set_ylabel('max range [km] (R>1e-9)'); ax.legend(fontsize=7); save(fig,'gg_F3_max_range')
# F4 decomposition
d=pd.read_csv(os.path.join(T,'e2_decomposition.csv'))
fig,ax=plt.subplots(1,3,figsize=(12,3.4))
ax[0].plot(d.L_km,d.R_GG_csi/d.R_det,'o-',label='CSI'); ax[0].plot(d.L_km,d.R_GG_nocsi/d.R_det,'s--',label='no CSI'); ax[0].set_title('E[R] / R_det (weak, GG)'); ax[0].legend()
ax[1].plot(d.L_km,d.local_exponent_p,'o-'); ax[1].set_title('local exponent d lnR/d ln eta at nominal')
ax[2].plot(d.L_km,d.Rshare_eta_gt_nom,'o-',label='eta>nom'); ax[2].plot(d.L_km,d.Rshare_eta_gt_2nom,'s-',label='eta>2 nom'); ax[2].set_title('share of E[R] from upward fades'); ax[2].legend()
for a in ax: a.set_xlabel('total distance [km]')
save(fig,'gg_F4_decomposition')
# F5 optimisation
o=pd.read_csv(os.path.join(T,'e3_optimisation.csv')); a_=pd.read_csv(os.path.join(T,'alpha_star_vs_eta.csv'))
fig,ax=plt.subplots(1,3,figsize=(12,3.4))
ax[0].plot(a_.log10_eta,a_.alpha_star); ax[0].set_xlabel('log10 eta'); ax[0].set_ylabel('alpha*'); ax[0].set_title('instantaneous optimum')
for (m,c),g in o.groupby(['model','cls']):
    ax[1].plot(g.L_km,g.alpha_star_csi,'o-',label=f'{m} {c}'); ax[2].plot(g.L_km,100*(g.R_opt_csi/g.R_a02_csi-1) if 'R_a02_csi' in g else 100*(g['R_opt_csi']/g['R_a0.2_csi']-1),'o-',label=f'{m} {c}')
ax[1].set_title('ergodic fixed alpha*'); ax[2].set_title('gain vs alpha=0.2 [%]'); ax[1].legend(fontsize=7)
for a in ax[1:]: a.set_xlabel('total distance [km]')
save(fig,'gg_F5_optimisation')
# F6 finite decoy deterministic
e=pd.read_csv(os.path.join(T,'e4_det_vs_loss.csv'))
fig,ax=plt.subplots(1,2,figsize=(11,3.6)); cols=[c for c in e.columns if c.startswith('R_')]
for c in cols: ax[0].semilogy(e.loss_dB,e[c].clip(lower=1e-30),label=c[2:])
ax[0].set_ylim(1e-10,1); ax[0].set_xlabel('loss [dB]'); ax[0].legend(fontsize=5)
for c in cols:
    if c!='R_inf_7': ax[1].plot(e.loss_dB,e[c]/e['R_inf_7'].where(e['R_inf_7']>0),label=c[2:])
ax[1].set_ylim(0,1.6); ax[1].set_xlabel('loss [dB]'); ax[1].set_title('ratio to infinite-decoy (7 yields)'); save(fig,'gg_F6_finite_decoy_det')
# F7 finite decoy under GG
g=pd.read_csv(os.path.join(T,'e5_gg_finite_decoy.csv')); fig,ax=plt.subplots(1,2,figsize=(11,3.6),sharey=True)
for a,c in zip(ax,['weak','strong']):
    q=g[g.cls==c]
    for k,st in (('inf7_csi','C0-'),('fin3_csi','C0--'),('inf7_nocsi','C3-'),('fin3_nocsi','C3--')): a.semilogy(q.L_km,q['R_'+k].clip(lower=1e-30),st,label=k)
    a.semilogy(q.L_km,q.R_det_fin3.clip(lower=1e-30),'k:',label='det fin3'); a.set_ylim(1e-10,1); a.set_title(c); a.set_xlabel('total distance [km]')
ax[0].legend(fontsize=7); save(fig,'gg_F7_finite_decoy_gg')
