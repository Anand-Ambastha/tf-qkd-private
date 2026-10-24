"""Operating point and channel construction shared by the gg_extension scripts."""
import os, sys
import numpy as np
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from src import legacy_baseline as CAL
from src.turbulence import create_channel
from src.tfqkd_engine import TFQKDEngine
LAM,W0,RX,AATM=1550e-9,0.05,0.15,0.046
CN2={'weak':1e-17,'moderate':1e-15,'strong':1e-13}
def eta_nom(L): return float(np.clip(CAL.deterministic_fso_channel(AATM,L,LAM,W0,RX)['eta_fso'],1e-300,1.0))
def build(model,cn2,L,**kw): return create_channel(model,eta_nominal=eta_nom(L),Cn2=cn2,wavelength=LAM,distance_m=L*1e3,**kw)
def nodes(ch,n=None):
    e,w=ch.quadrature_nodes(n); e=np.clip(e,1e-300,1.0); return e,w
