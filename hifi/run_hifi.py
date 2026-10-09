"""run_hifi.py -- high-fidelity campaign (measured APC 10x4.7SF + X2212-980KV data).
Writes results/generated/hifi.json."""
import json, sys, numpy as np
from multiprocessing import Pool
from hifi_sim import simulate
from project_paths import result_path
METH = ["energy", "greedy", "lowpass", "cbf", "robust"]
CONF = [("rock", 1.8, dict(Phi=30, P=1.0)), ("rock", 1.8, dict(Phi=45, P=0.8)),
        ("rock", 1.35, dict(Phi=45, P=1.0)), ("rock", 1.35, dict(Phi=60, P=1.0)),
        ("load", 1.8, dict(Mamp=0.6, P=1.0)), ("load", 1.8, dict(Mamp=1.0, P=1.0)),
        ("load", 1.8, dict(Mamp=1.3, P=1.0)), ("load", 1.27, dict(Mamp=0.6, P=1.0)),
        ("load", 1.27, dict(Mamp=1.0, P=1.0)),
        ("cruise", 1.8, dict(V=0.0)), ("cruise", 1.8, dict(V=5.0)), ("cruise", 1.8, dict(V=10.0)),
        ("cruise", 1.8, dict(V=15.0))]
SEEDS = [0, 1, 2, 3, 4]
DELAYS = [0.0, 0.005, 0.010, 0.020, 0.040]
KEYS = ["certifiable", "eb", "ld", "hop", "hmin", "htrue_min", "hmean", "tbelow_true", "werr", "eR", "ez",
        "sat", "minabsw", "crossings", "rate_peak", "active", "completed"]
def job(a):
    tag, kind, mass, kw, m, seed, dl = a
    r = simulate(m, kind, mass=mass, T=(6.0 if kind == "cruise" else 4.0), mkw=kw, seed=seed, extra_delay=dl)
    return tag, kind, mass, kw, m, seed, dl, {k: (float(r[k]) if isinstance(r.get(k), (float, np.floating)) else r.get(k)) for k in KEYS}
if __name__ == "__main__":
    jobs = [("main", k, ms, kw, m, s, 0.0) for (k, ms, kw) in CONF for m in METH for s in SEEDS]
    jobs += [("delay", "load", 1.27, dict(Mamp=1.0, P=1.0), m, s, d) for m in ["energy", "cbf"] for s in SEEDS[:3] for d in DELAYS[1:]]
    out = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            out.append(dict(zip(["tag", "kind", "mass", "kw", "method", "seed", "delay", "res"], r)))
            if i % 10 == 0:
                print(i, len(jobs), flush=True)
                json.dump(out, open(result_path("hifi.json"), "w"))
    json.dump(out, open(result_path("hifi.json"), "w"), indent=0)
    print("done")
