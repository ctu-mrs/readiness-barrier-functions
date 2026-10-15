"""exp_kappa.py -- choice of the floor parameter kappa.

For kappa = 0.1 ... 0.9 (hexarotor, amplitude 1.5, collective 0.75 and 0.70):
  * floor ell_bar, authority above dropout ell_bar - ell_drop, and the slack
    L_op - ell_bar = (1 - kappa)(L_op - ell_drop) left for unmodelled effects;
  * the certified delay ceiling td*(kappa), from eta(td*) = slack (see delay_study.py);
  * nominal closed loop (filter and effort QP), delays of 0.02/0.05/0.1 T0, and
    20 plants with +-10 % random motor parameters.

Output: results/generated/kappa.json.
"""
import numpy as np, json
import multiprocessing as mp
from project_paths import result_path
from scipy.optimize import brentq
from platforms import hexarotor
from final_exp import floor_rule
from robust import perturb_params
from robust_campaign import rollout
from delay_study import constants
from sim_core import run
P = hexarotor(); ALPHA = 5.0; LMAX = -10.128   # L_max of the hexarotor, rounded (gate.h_max_point gives -10.12776)
KAP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def job(t):
    """kind "nom": nominal runs at collective arg; "delay": input delay arg; "par": perturbed plant arg."""
    kind, kap, eb, arg = t
    if kind == "nom":
        th = arg; r = run(P, "cbf", eb, amp=1.5, thrust=th); re = run(P, "energy", eb, amp=1.5, thrust=th)
        return t[:3] + (th,), dict(hmin=r["hmin"], werr=r["werr"], active=r["active"], dmax=r["delta_max"], hmin_energy=re["hmin"])
    if kind == "delay":
        r = run(P, "cbf", eb, amp=1.5, thrust=0.75, delay=arg)
        return t[:3] + (arg,), dict(hmin=r["hmin"])
    if kind == "par":
        hm, we = rollout(arg, P, eb, "cbf"); return t[:3] + (id(arg),), dict(hmin=hm, werr=we)
if __name__ == "__main__":
    K1, K2, V = constants(P); K = K1 + K2 * float(np.max(P["tau_bar"] / P["mass"]))
    out = {"K1": K1, "K2": K2, "V": V, "rows": {}}
    tasks = []; rng = np.random.default_rng(21); plants = [perturb_params(P, 0.10, worst=False, rng=rng) for _ in range(20)]
    for kap in KAP:
        eb, ld, hop = floor_rule(P, 1.5, 0.75, kappa=kap)
        eb7, _, hop7 = floor_rule(P, 1.5, 0.70, kappa=kap)
        Hbar = LMAX - eb; avail = hop - eb
        eta = lambda td: (ALPHA * Hbar + K * V * td) * td + K * V * td / ALPHA   # delay undershoot bound
        tds = brentq(lambda t: eta(t) - avail, 1e-9, 0.1)
        out["rows"][str(kap)] = dict(eb=eb, ldrop=ld, Lop=hop, auth=eb - ld, slack=avail, tdstar=tds, eb70=eb7, Lop70=hop7)
        tasks += [("nom", kap, eb, 0.75), ("nom", kap, eb7, 0.70)]
        tasks += [("delay", kap, eb, d) for d in [0.02, 0.05, 0.1]]
        tasks += [("par", kap, eb, Pt) for Pt in plants]
    with mp.Pool(2) as pool: res = pool.map(job, tasks)
    for key, r in res:
        kind, kap, eb, arg = key; row = out["rows"][str(kap)]
        if kind == "nom": row["cl_%.2f" % arg] = r
        elif kind == "delay": row["delay_%g" % arg] = r["hmin"]
        else: row.setdefault("par_h", []).append(r["hmin"])
    for kap in KAP:
        row = out["rows"][str(kap)]; ph = np.array(row.pop("par_h"))
        row["par_viol"] = float(np.mean(ph < 0)); row["par_worst"] = float(ph.min())
        print(kap, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    json.dump(out, open(result_path("kappa.json"), "w"), indent=1)
