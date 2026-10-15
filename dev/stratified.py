"""
stratified.py — (a) stratified MC in the authority-scarce corner;
(b) robustness to parametric error (controller uses nominal, plant is perturbed).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np
from project_paths import result_path
from platforms import hexarotor
from daam import h_value
from gate import ell_dagger
from harden import traj, run
from final_exp import floor_rule
import allocators as AL
from mc_study import ci95
phi=lambda v: v*np.abs(v)

def run_mismatch(Pt, Pc, ellbar, method, T=1.5, dt=1e-3, amp=1.5, thrust=0.75):
    """Pt = true plant params (perturbed); Pc = controller's nominal model."""
    w,wd=traj(Pt,amp,thrust)
    v=AL.fiber_argmax_h(w(0.0),Pc,np.full(Pc["n"],Pc["vstar"]))
    N=int(T/dt); hs=[]; werr=[]
    for k in range(N):
        t=k*dt; wd_,wdd_=w(t),wd(t)
        if method=="energy": tau,_=AL.energy_qp_tau(v,wd_,wdd_,Pc)
        else:
            tr_,_=AL.energy_qp_tau(v,wd_,wdd_,Pc)
            tau,_,_=AL.cbf_qp(v,wd_,wdd_,Pc,ellbar,tau_ref=tr_)
        d=-(Pt["b"]*phi(v))/Pt["mass"]          # TRUE dynamics
        v=np.clip(v+dt*(d+tau/Pt["mass"]),-0.999*Pt["vsat"],0.999*Pt["vsat"])
        hs.append(h_value(v,Pt,ellbar)); werr.append(np.linalg.norm(Pt["A"]@phi(v)-wd_))
    return dict(hmin=float(np.min(hs)), werr=float(np.sqrt(np.mean(np.array(werr)**2))))

if __name__=="__main__":
    P=hexarotor()
    # ---------- (a) stratified MC: authority-scarce corner ----------
    print("=== Stratified MC: authority-scarce corner (thrust U[0.68,0.82], amp U[1.4,2.0]), n=10 ===")
    rng=np.random.default_rng(7)
    acc={m:{"h":[],"TV":[],"e":[]} for m in ["energy","greedy","cbf"]}
    used=0; rej=0
    while used<10:
        th=rng.uniform(0.68,0.82); amp=rng.uniform(1.4,2.0)
        eb,ld,hop=floor_rule(P,amp,th,kappa=0.5)
        if eb is None: rej+=1; continue
        for m in ["energy","greedy","cbf"]:
            r=run(P,m,eb,dt=1e-3,T=1.5,amp=amp,thrust=th,seed=used)
            acc[m]["h"].append(r["hmin"]); acc[m]["TV"].append(r["TV"]); acc[m]["e"].append(r["werr"])
        used+=1
    print("  (%d draws rejected as non-certifiable)"%rej)
    print("%-8s %20s %18s %20s %10s"%("method","h_min mean+/-CI","TV mean+/-CI","wErr mean+/-CI","viol.rate"))
    for m in ["energy","greedy","cbf"]:
        hm,hc=ci95(acc[m]["h"]); tm,tc=ci95(acc[m]["TV"]); em,ec=ci95(acc[m]["e"])
        vr=float(np.mean(np.array(acc[m]["h"])<0))
        print("%-8s %8.3f +/- %-8.3f %7.2f +/- %-7.2f %9.4f +/- %-8.4f %8.2f"%(m,hm,hc,tm,tc,em,ec,vr))
    np.save(result_path("strat.npy"),acc,allow_pickle=True)

    # ---------- (b) parametric robustness ----------
    print("\n=== Parametric robustness: plant perturbed, controller nominal (thrust .75, amp 1.5) ===")
    eb,_,_=floor_rule(P,1.5,0.75,kappa=0.5)
    print("%8s | %-8s %10s %10s %10s"%("pert","method","h_min","wErr","floor"))
    rng=np.random.default_rng(3)
    for pct in [0.0,0.10,0.20,0.30]:
        for m in ["energy","cbf"]:
            hmins=[]; werrs=[]
            for rep in range(4 if pct>0 else 1):
                Pt=hexarotor()
                if pct>0:
                    Pt["b"]=Pt["b"]*(1+rng.uniform(-pct,pct,size=P["n"]))
                    Pt["tau_bar"]=Pt["tau_bar"]*(1+rng.uniform(-pct,pct,size=P["n"]))
                    Pt["vsat"]=float(np.sqrt(np.min(Pt["tau_bar"]/Pt["b"])))
                r=run_mismatch(Pt,P,eb,m,amp=1.5,thrust=0.75)
                hmins.append(r["hmin"]); werrs.append(r["werr"])
            print("%7.0f%% | %-8s %+10.4f %10.5f %10s"%(pct*100,m,np.mean(hmins),np.mean(werrs),
                  "held" if np.min(hmins)>=0 else "VIOLATED"))
