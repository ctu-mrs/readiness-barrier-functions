"""robust_v2.py -- ablation of the robust filter (lower block of Table III).

Compares the robust metric with nominal drift ("metric") against the full
robust filter with worst-case drift ("full"). The floor is tight (kappa = 0.9)
so that the drift term is not hidden by a conservative floor. roll() is used by
exp_robust.py; running this file gives the small 8-plant version.
"""
import numpy as np
from platforms import hexarotor
from daam import h_value
from harden import traj
from final_exp import floor_rule
from robust import perturb_params
import allocators as AL
phi=lambda v: v*np.abs(v)

def roll(Ptrue,Pnom,Pw,ellbar,p,mode,amp=1.5,thrust=0.75,T=1.5,dt=1e-3):
    """Robust filter on plant Ptrue; mode "metric" or "full". Returns (min robust h, RMS wrench error)."""
    w,wd=traj(Ptrue,amp,thrust)
    v=AL.fiber_argmax_h(w(0.0),Pnom,np.full(Pnom["n"],Pnom["vstar"]))
    hs=[];we=[]
    for k in range(int(T/dt)):
        t=k*dt; wd_,wdd_=w(t),wd(t)
        tr_,_=AL.energy_qp_tau(v,wd_,wdd_,Pnom)
        tau, _, _ = AL.cbf_qp_robust(
            v, wd_, wdd_, Pnom, Pw, ellbar, p, tau_ref=tr_,
            worst_case_drift=(mode != "metric"))
        d=-(Ptrue["b"]*phi(v))/Ptrue["mass"]
        v=np.clip(v+dt*(d+tau/Ptrue["mass"]),-0.999*Ptrue["vsat"],0.999*Ptrue["vsat"])
        hs.append(h_value(v,Pw,ellbar))   # robust barrier value (what we certify)
        we.append(np.linalg.norm(Ptrue["A"]@phi(v)-wd_))
    return float(np.min(hs)),float(np.sqrt(np.mean(np.array(we)**2)))

if __name__=="__main__":
    P=hexarotor(); NR=8
    print("Floor set TIGHT (kappa=0.9) so the drift mismatch is not masked by conservatism.\n")
    print("%5s | %-22s %10s %10s %9s %9s"%("p","controller","minh_mean","minh_worst","viol","wErr"))
    for p in [0.10,0.20,0.30]:
        Pw=perturb_params(P,p,worst=True)
        eb,ld,hop=floor_rule(Pw,1.5,0.75,kappa=0.9)   # aggressive floor
        for mode,lab in [("metric","robust metric (nom drift)"),("full","fully robust (wc drift)")]:
            rng=np.random.default_rng(5); hm=[];wr=[]
            for _ in range(NR):
                Pt=perturb_params(P,p,worst=False,rng=rng)
                a,b=roll(Pt,P,Pw,eb,p,mode); hm.append(a); wr.append(b)
            hm=np.array(hm)
            print("%4.0f%% | %-22s %+10.4f %+10.4f %9.2f %9.5f"
                  %(p*100,lab,hm.mean(),hm.min(),float(np.mean(hm<0)),float(np.mean(wr))))
        print()
