"""exp_mc.py -- randomised missions (Table II).

Collective and moment amplitude are drawn uniformly in four boxes: the normal
envelope and an authority-scarce corner, for the hexarotor and the octorotor.
Draws that are not certifiable are rejected and counted. Each accepted mission
is run with the four allocators for 1.5 T0.

Usage: python exp_mc.py [N]   (N hexarotor missions per box, N/2 for the octorotor; default 100)
Output: results/generated/mc.json; the summary (violation rate, mean h_min and
RMS wrench error with 95 % normal intervals) is printed.
"""
import numpy as np, json, sys
from multiprocessing import Pool
from project_paths import result_path
from platforms import hexarotor, tilted_octorotor
from final_exp import floor_rule
from sim_core import run, METHODS
def draw(name, N, thr, amp, seed0):
    """Draw N certifiable missions (collective in thr, amplitude in amp); return them and the rejections."""
    P = hexarotor() if name == "hex" else tilted_octorotor()
    rng = np.random.default_rng(seed0); out = []; rej = 0
    while len(out) < N:
        th = rng.uniform(*thr); a = rng.uniform(*amp)
        eb, ld, hop = floor_rule(P, a, th, kappa=0.5)
        if eb is None: rej += 1; continue
        out.append((name, len(out), th, a, eb))
    return out, rej
def job(x):
    """Run the four allocators on one mission; the mission index is also the greedy seed."""
    name, i, th, a, eb = x
    P = hexarotor() if name == "hex" else tilted_octorotor()
    res = {}
    for m in METHODS:
        r = run(P, m, eb, T=1.5, amp=a, thrust=th, seed=i)
        res[m] = {k: r[k] for k in ["hmin", "hmean", "tbelow", "TV", "werr", "rate_peak", "active"]}
    return name, i, th, a, eb, res
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    # key: (platform, number of missions, collective range, amplitude range, seed)
    sets = {"hex_env": ("hex", N, (0.7, 1.2), (1.0, 2.0), 1),
            "hex_scarce": ("hex", N, (0.68, 0.82), (1.0, 2.0), 2),
            "oct_env": ("oct", N//2, (0.7, 1.2), (1.0, 2.0), 3),
            "oct_scarce": ("oct", N//2, (0.4, 0.6), (1.0, 3.0), 4)}
    allres = {}
    for key, (name, n, thr, amp, s0) in sets.items():
        missions, rej = draw(name, n, thr, amp, s0)
        with Pool(2) as pool: rr = pool.map(job, missions)
        allres[key] = {"rejected": rej, "missions": [dict(th=r[2], amp=r[3], eb=r[4], res=r[5]) for r in rr]}
        json.dump(allres, open(result_path("mc.json"), "w"))   # saved after each box
        print(key, "rejected", rej, flush=True)
        for m in METHODS:
            h = np.array([x[5][m]["hmin"] for x in rr]); e = np.array([x[5][m]["werr"] for x in rr])
            ci = lambda z: 1.96 * z.std(ddof=1) / np.sqrt(len(z))
            print("  %-8s viol %.2f  hmin %+.3f+-%.3f  werr %.4f+-%.4f  (median %.4f)" %
                  (m, np.mean(h < 0), h.mean(), ci(h), e.mean(), ci(e), np.median(e)), flush=True)
