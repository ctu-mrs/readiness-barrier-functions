"""octo.py — three-way comparison on the fully-actuated tilted octorotor (n=8,m=6)."""
import numpy as np
from platforms import tilted_octorotor
from gate import ell_dagger
from harden import run
from final_exp import floor_rule
Q=tilted_octorotor(); ld,_=ell_dagger(Q)
print("tilted octorotor n=8 m=6 : ell_drop=%.3f"%ld)
print("%6s %6s %9s | %-8s %8s %8s %10s"%("thr","amp","ellbar","method","h_min","TV","wErr"))
for th,amp in [(0.8,1.5),(0.9,1.5)]:
    eb,_,hop=floor_rule(Q,amp,th,kappa=0.5)
    if eb is None:
        print("%6.2f %6.1f  NOT CERTIFIABLE (h_op=%.3f)"%(th,amp,hop)); continue
    for m in ["energy","greedy","cbf"]:
        r=run(Q,m,eb,dt=1e-3,T=1.0,amp=amp,thrust=th)
        tag="%6.2f %6.1f %9.3f"%(th,amp,eb) if m=="energy" else "%6s %6s %9s"%("","","")
        print("%s | %-8s %+8.4f %8.2f %10.5f"%(tag,m,r["hmin"],r["TV"],r["werr"]))
