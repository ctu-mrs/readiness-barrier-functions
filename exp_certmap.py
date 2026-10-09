"""exp_certmap.py -- which missions are certifiable? Window width L_op - ell_drop over
(collective, moment amplitude); negative = no allocator can hold any floor above ell_drop."""
import numpy as np, json
import multiprocessing as mp
from project_paths import result_path
from platforms import hexarotor
from final_exp import achievable
from gate import ell_dagger
P = hexarotor(); ld, _ = ell_dagger(P)
TH = np.round(np.arange(0.55, 1.301, 0.05), 3); AM = np.round(np.arange(0.5, 3.01, 0.25), 3)
def job(x):
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
