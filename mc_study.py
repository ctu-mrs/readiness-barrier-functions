"""
mc_study.py — Monte-Carlo over randomized certifiable missions + ablations.
Reports mean +/- 95% CI (t-based) for h_min, TV, wrench error.
"""
import numpy as np
from project_paths import result_path
from platforms import hexarotor
from gate import ell_dagger
from harden import run
from final_exp import floor_rule

def ci95(x):
    x=np.asarray(x,float); n=len(x)
    if n<2: return float(x.mean()),0.0
    return float(x.mean()), float(1.96*x.std(ddof=1)/np.sqrt(n))

def mc(P, nseed=12, kappa=0.5, alpha=5.0, seed0=0, T=1.5):
    rng=np.random.default_rng(seed0)
    acc={m:{"h":[],"TV":[],"e":[]} for m in ["energy","greedy","cbf"]}
    nfail=0; used=0
    while used<nseed:
        th=rng.uniform(0.7,1.2); amp=rng.uniform(1.0,2.0)
        eb,ld,hop=floor_rule(P,amp,th,kappa=kappa)
        if eb is None: nfail+=1; continue
        for m in ["energy","greedy","cbf"]:
            r=run(P,m,eb,dt=1e-3,T=T,amp=amp,thrust=th,seed=used)
            acc[m]["h"].append(r["hmin"]); acc[m]["TV"].append(r["TV"]); acc[m]["e"].append(r["werr"])
        used+=1
    return acc,nfail

if __name__=="__main__":
    P=hexarotor()
    print("=== Monte Carlo: 12 randomized certifiable missions (thrust U[0.7,1.2], amp U[1.0,2.0]) ===")
    acc,nf=mc(P,nseed=12)
    print("  (%d sampled missions rejected as non-certifiable)"%nf)
    print("%-8s %22s %20s %22s %10s"%("method","h_min mean+/-CI","TV mean+/-CI","wErr mean+/-CI","viol.rate"))
    for m in ["energy","greedy","cbf"]:
        hm,hc=ci95(acc[m]["h"]); tm,tc=ci95(acc[m]["TV"]); em,ec=ci95(acc[m]["e"])
        vr=float(np.mean(np.array(acc[m]["h"])<0))
        print("%-8s %10.3f +/- %-8.3f %8.2f +/- %-8.2f %10.4f +/- %-8.4f %10.2f"%(m,hm,hc,tm,tc,em,ec,vr))
    np.save(result_path("mc.npy"),acc,allow_pickle=True)

    print("\n=== Ablation: floor aggressiveness kappa (thrust 0.75, amp 1.5) ===")
    print("%6s %9s | %-8s %8s %9s %9s"%("kappa","ellbar","method","h_min","TV","wErr"))
    for k in [0.25,0.5,0.75]:
        eb,ld,hop=floor_rule(P,1.5,0.75,kappa=k)
        for m in ["energy","cbf"]:
            r=run(P,m,eb,dt=1e-3,T=1.5,amp=1.5,thrust=0.75)
            tag="%6.2f %9.3f"%(k,eb) if m=="energy" else "%6s %9s"%("","")
            print("%s | %-8s %+8.4f %8.2f %9.5f"%(tag,m,r["hmin"],r["TV"],r["werr"]))
