"""exp_fig5.py -- time histories of the moment-reversal mission (Fig. 4 of the revision).

The file name follows the numbering of the submitted manuscript, where this was Fig. 5.
Hexarotor at collective 0.75 and moment amplitude 1.5, all four allocators, 2 T0.

The commanded rate is exported as its maximum over 0.02 T0 windows (the greedy
update period), so no setpoint jump is lost by decimation; h is sampled every
4e-3 T0. Both are clipped from below for the log axis.

Output: results/generated/rate_all.dat and h_all.dat (plotted as they are).
"""
import numpy as np
from project_paths import result_path
from platforms import hexarotor
from final_exp import floor_rule
from sim_core import run, METHODS
P = hexarotor(); eb, ld, hop = floor_rule(P, 1.5, 0.75, kappa=0.5)
R = {m: run(P, m, eb, amp=1.5, thrust=0.75, log=True) for m in METHODS}
t = R["cbf"]["t"]; B = 4                         # h: keep every 4th sample
nb = len(t) // B
tb = t[:nb*B].reshape(nb, B)[:, -1]
BR = 20; nr = len(t) // BR                       # rate: max over 20-sample windows
tr = t[:nr*BR].reshape(nr, BR)[:, -1]
rr = {m: R[m]["rate"].copy() for m in METHODS}
for m in METHODS: rr[m][0] = 0.0                 # k=0 has no previous command
rate = {m: np.maximum(rr[m][:nr*BR].reshape(nr, BR).max(1), 1e-2) for m in METHODS}   # floor at 1e-2 for the log plot
hh = {m: R[m]["h"][:nb*B].reshape(nb, B)[:, -1] for m in METHODS}
np.savetxt(result_path("rate_all.dat"), np.column_stack([tr] + [rate[m] for m in METHODS]),
           header="t rate_energy rate_greedy rate_lowpass rate_cbf", comments="", fmt="%.6g")
np.savetxt(result_path("h_all.dat"), np.column_stack([tb] + [hh[m] for m in METHODS]),
           header="t energy greedy lowpass cbf", comments="", fmt="%.6g")
for m in METHODS:
    r = R[m]
    print("%-8s hmin %+.4f hmean %+.4f tbelow %.3f peak_rate %.2f werr %.5f minabsv %.3f" %
          (m, r["hmin"], r["hmean"], r["tbelow"], r["rate_peak"], r["werr"], r["minabsv"]))
print("ellbar", eb, "ldrop", ld, "Lop", hop)
