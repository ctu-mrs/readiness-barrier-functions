"""exp_delay.py -- input-delay sweep (Fig. 6).

Hexarotor, collective 0.75, amplitude 1.5, filter only. Two families of runs:
  * plain floor ell_bar, delays from 0 to 0.2 T0, with the delay buffer
    initialised either with zeros (as in the submitted version) or with the
    torques that hold v(0) ("hold", the one plotted);
  * margined floor ell_bar + eta(td), only for td below the certified ceiling
    td*, where the raised floor still lies below L_op.
min h is always measured against the original floor ell_bar.

Output: results/generated/delay.json, with the delay constants K1, K2, V and td*.
"""
import numpy as np, json
import multiprocessing as mp
from project_paths import result_path
from scipy.optimize import brentq
from platforms import hexarotor
from final_exp import floor_rule
from delay_study import constants
from sim_core import run
P = hexarotor(); ALPHA = 5.0; LMAX = -10.128   # L_max of the hexarotor, rounded (gate.h_max_point gives -10.12776)
def job(t):
    """Run the filter with floor eb_enf and delay td; report min h relative to eb."""
    kind, td, eb_enf, eb, init = t
    r = run(P, "cbf", eb_enf, amp=1.5, thrust=0.75, delay=td, delay_init=init, log=True)
    hm = float(np.min(r["h"] + eb_enf - eb))  # h w.r.t. the ORIGINAL floor
    return t, hm
if __name__ == "__main__":
    eb, ld, hop = floor_rule(P, 1.5, 0.75, kappa=0.5)
    K1, K2, V = constants(P); K = K1 + K2 * float(np.max(P["tau_bar"] / P["mass"]))
    Hbar = LMAX - eb; avail = hop - eb
    # undershoot bound of the delay proposition; td* is where it uses up the slack L_op - ell_bar
    eta = lambda td: (ALPHA * Hbar + K * V * td) * td + K * V * td / ALPHA
    tds = brentq(lambda t: eta(t) - avail, 1e-9, 0.1)
    D = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    tasks = [("plain", td, eb, eb, init) for td in D for init in ["zeros", "hold"]]
    M = [0.00025, 0.0005, 0.001, 0.0015, tds * 0.999]
    tasks += [("marg", td, eb + eta(td), eb, "hold") for td in M]
    with mp.Pool(2) as pool: res = pool.map(job, tasks)
    out = dict(K1=K1, K2=K2, K=K, V=V, KV=K1 * V, Hbar=Hbar, avail=avail, tdstar=tds, eb=eb, rows=[])
    for (kind, td, eb_enf, _, init), hm in res:
        out["rows"].append(dict(kind=kind, td=td, init=init, hmin=hm, eta=eta(td), eb_enf=eb_enf))
        print(kind, init, "%.4f" % (td * 1e3), "hmin %+.4f eta %.4f" % (hm, eta(td)), flush=True)
    print("tdstar", tds, "K", K, "V", V, "K1*V", K1 * V, "K*V", K * V)
    json.dump(out, open(result_path("delay.json"), "w"), indent=1)
