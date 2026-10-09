"""
final_exp.py — corrected closed-loop evaluation.
Floor rule:  ellbar = ell_drop + kappa*(h_op_min - ell_drop),  kappa in (0,1),
requires h_op_min > ell_drop (task-achievable authority above the dropout level).
"""
import numpy as np, os
from project_paths import result_path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from platforms import hexarotor, tilted_octorotor
from daam import h_value
from gate import ell_dagger, h_max_point
import allocators as AL
from harden import traj, run

def achievable(P, amp, thrust, T=2.0, npts=20):
    w,_=traj(P,amp,thrust); v0=np.full(P["n"],P["vstar"]); b=[]
    for t in np.linspace(0,T,npts):
        v=AL.fiber_argmax_h(w(t),P,v0); v0=v; b.append(h_value(v,P,0.0))
    return float(np.min(b))

def floor_rule(P, amp, thrust, kappa=0.5):
    ld,_=ell_dagger(P); hop=achievable(P,amp,thrust)
    if hop<=ld: return None, ld, hop
    return ld+kappa*(hop-ld), ld, hop

if __name__=="__main__":
    P=hexarotor(); _,hmx=h_max_point(P)
    print("=== corrected gate (hexarotor) ===  h_max=%.3f"%hmx)
    amps=[1.0,1.5,2.0]; thrust=0.9
    print("\n=== closed loop, thrust=%.1f, certifiable floors ==="%thrust)
    print("%5s %9s %9s %9s | %-8s %7s %8s %9s"%("amp","ell_drop","h_op_min","ellbar","method","TV","wErr","h_min"))
    rows={}
    for amp in amps:
        eb,ld,hop=floor_rule(P,amp,thrust)
        if eb is None:
            print("%5.1f  NOT CERTIFIABLE (h_op_min=%.2f <= ell_drop=%.2f)"%(amp,hop,ld)); continue
        for mth in ["energy","greedy","cbf"]:
            r=run(P,mth,eb,eta=0.0,dt=1e-3,T=2.0,amp=amp,thrust=thrust)
            rows[(amp,mth)]=r
            tag="%5.1f %9.3f %9.3f %9.3f"%(amp,ld,hop,eb) if mth=="energy" else "%5s %9s %9s %9s"%("","","","")
            print("%s | %-8s %7.2f %8.4f %+9.4f"%(tag,mth,r["TV"],r["werr"],r["hmin"]))
    np.save(result_path("final_rows.npy"),rows,allow_pickle=True)
