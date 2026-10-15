"""exp_sweep.py -- collective sweep (Table I and Fig. 3).

Hexarotor, moment amplitude 1.5, collective 0.6 ... 1.2 times the sweet-spot
value. For each certifiable collective the floor comes from floor_rule
(kappa = 0.5) and the four allocators are simulated for 2 T0. Records min h,
mean h, time below the floor, setpoint total variation, RMS wrench error, peak
commanded rate, barrier activity and largest slack.

Output: results/generated/sweep.json (make_paper_data.py turns it into data/sweep.dat).
"""
import numpy as np, json, sys
from multiprocessing import Pool
from project_paths import result_path
from platforms import hexarotor
from final_exp import floor_rule
from sim_core import run, METHODS
P = hexarotor(); AMP = 1.5
def job(args):
    """One (collective, allocator, floor) run; executed in a worker process."""
    th, m, eb = args
    r = run(P, m, eb, amp=AMP, thrust=th)
    return th, m, {k: r[k] for k in ["hmin","hmean","tbelow","TV","werr","rate_peak","active","delta_max"]}
if __name__ == "__main__":
    ths = [0.6,0.7,0.8,0.9,1.0,1.1,1.2]; floors = {}; jobs = []
    for th in ths:
        eb, ld, hop = floor_rule(P, AMP, th, kappa=0.5); floors[th] = (eb, ld, hop)
        if eb is not None: jobs += [(th, m, eb) for m in METHODS]
    with Pool(2) as pool: res = pool.map(job, jobs)
    # keys of "runs" are "<collective>_<method>", e.g. "0.7_greedy"
    out = {"floors": {str(k): v for k, v in floors.items()}, "runs": {}}
    for th, m, r in res: out["runs"]["%.1f_%s" % (th, m)] = r
    json.dump(out, open(result_path("sweep.json"), "w"), indent=1)
    for th in ths:
        eb = floors[th][0]
        if eb is None: print(th, "not certifiable", floors[th]); continue
        print("%.1f eb=%.3f" % (th, eb), " | ".join("%s h%+.3f hm%+.3f TV%.2f e%.4f tb%.3f" % (m[:3], *(out["runs"]["%.1f_%s"%(th,m)][k] for k in ["hmin","hmean","TV","werr","tbelow"])) for m in METHODS))
