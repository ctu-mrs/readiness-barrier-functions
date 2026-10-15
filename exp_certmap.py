"""exp_certmap.py -- certifiability map (Fig. 5, left).

Window L_op - ell_drop of the hexarotor over a grid of collective (0.55 ... 1.30)
and moment amplitude (0.5 ... 3.0). Where it is negative no allocator can keep
any floor above the dropout level for that mission.

Output: results/generated/certmap.dat (columns thrust, amp, window; one blank
line after each collective, as pgfplots' matrix plot expects).
"""
import numpy as np, json
import multiprocessing as mp
from project_paths import result_path
from platforms import hexarotor
from final_exp import achievable
from gate import ell_dagger
P = hexarotor(); ld, _ = ell_dagger(P)
TH = np.round(np.arange(0.55, 1.301, 0.05), 3); AM = np.round(np.arange(0.5, 3.01, 0.25), 3)
def job(x):
    """Window at one grid point; NaN if the fibre optimisation fails."""
    th, a = x
    try: return th, a, achievable(P, a, th) - ld
    except Exception: return th, a, np.nan
if __name__ == "__main__":
    with mp.Pool(2) as pool: res = pool.map(job, [(t, a) for t in TH for a in AM])
    with open(result_path("certmap.dat"), "w") as f:
        f.write("thrust amp window\n")
        for t in TH:
            for th, a, wv in res:
                if th == t: f.write("%.3f %.3f %.5f\n" % (th, a, wv))
            f.write("\n")
    print("done", min(r[2] for r in res), max(r[2] for r in res))
