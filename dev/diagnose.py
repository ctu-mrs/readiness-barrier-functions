"""diagnose.py — why h_min<0: is the floor above the task-achievable authority?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np
from platforms import hexarotor, tilted_octorotor
from daam import h_value
from gate import ell_dagger, h_max_point
import allocators as AL
from harden import traj

def achievable(P, amp, thrust, T=2.0, npts=25):
    """min over the task trajectory of the fiberwise-MAX authority (best possible)."""
    w,_ = traj(P, amp, thrust)
    v0 = np.full(P["n"], P["vstar"]); best=[]
    for t in np.linspace(0,T,npts):
        v = AL.fiber_argmax_h(w(t), P, v0); v0 = v
        best.append(h_value(v,P,0.0))
    return float(np.min(best)), float(np.max(best))

P = hexarotor(); ld,_ = ell_dagger(P); _,hmx = h_max_point(P)
print("hexarotor: h_max=%.3f  ell_drop=%.3f\n" % (hmx, ld))
print("%8s %8s | %10s %10s | %s" % ("thrust","amp","h_op_min","window","verdict"))
for thrust,amp in [(0.5,2.5),(0.7,2.0),(0.8,1.5),(1.0,1.0),(1.0,2.0)]:
    lo,hi = achievable(P, amp, thrust)
    win = lo - ld
    print("%8.2f %8.1f | %10.3f %10.3f | %s" % (thrust,amp,lo,win,
          "CERTIFIABLE (floor fits)" if win>0.15 else "NO ROOM: task authority below dropout+"))
