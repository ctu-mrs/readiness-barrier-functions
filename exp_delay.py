"""exp_delay.py -- delay sweep (Fig. 6) on a denser grid, including delays below the
certified ceiling, plain vs margined floor, and two buffer initialisations:
'zeros' (submitted: zero torque for the first tau_d) and 'hold' (torque that holds v(0))."""
import numpy as np, json
import multiprocessing as mp
from project_paths import result_path
from scipy.optimize import brentq
from platforms import hexarotor
from final_exp import floor_rule
from delay_study import constants
from sim_core import run
P = hexarotor(); ALPHA = 5.0; LMAX = -10.128
def job(t):
    kind, td, eb_enf, eb, init = t
    r = run(P, "cbf", eb_enf, amp=1.5, thrust=0.75, delay=td, delay_init=init, log=True)
    hm = float(np.min(r["h"] + eb_enf - eb))  # h w.r.t. the ORIGINAL floor
    return t, hm
if __name__ == "__main__":
    eb, ld, hop = floor_rule(P, 1.5, 0.75, kappa=0.5)
    K1, K2, V = constants(P); K = K1 + K2 * float(np.max(P["tau_bar"] / P["mass"]))
    Hbar = LMAX - eb; avail = hop - eb
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
