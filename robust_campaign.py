"""
robust_campaign.py -- closed loop with a perturbed plant (used by exp_robust.py).

rollout() simulates one controller on a true plant Ptrue while the controller
uses Pctrl (and Pworst for the robust filter). Running this file gives the
small original version of the campaign (8 plants per level); the paper's
Table III comes from exp_robust.py.
"""
import numpy as np
from platforms import hexarotor
from daam import h_value
from gate import ell_dagger
from harden import traj
from final_exp import floor_rule
from robust import perturb_params
from project_paths import result_path
import allocators as AL
phi = lambda v: v*np.abs(v)

def rollout(Ptrue, Pctrl, ellbar, method, amp=1.5, thrust=0.75, T=1.5, dt=1e-3,
            p=0.0, Pworst=None):
    """method: "energy", "cbf" (nominal filter) or "cbf-robust". Returns (min h, RMS wrench error).

    Note that exp_robust.py calls this with Pctrl = Pworst and method "cbf" for
    the robust row, i.e. the nominal filter run on the worst-case model.
    """
    w, wd = traj(Ptrue, amp, thrust)
    v = AL.fiber_argmax_h(w(0.0), Pctrl, np.full(Pctrl["n"], Pctrl["vstar"]))
    hs = []; we = []
    for k in range(int(T/dt)):
        t = k*dt; wd_, wdd_ = w(t), wd(t)
        if method == "energy":
            tau, _ = AL.energy_qp_tau(v, wd_, wdd_, Pctrl)
        else:
            tr_, _ = AL.energy_qp_tau(v, wd_, wdd_, Pctrl)
            if method == "cbf-robust":
                tau, _, _ = AL.cbf_qp_robust(
                    v, wd_, wdd_, Pctrl, Pworst, ellbar, p, tau_ref=tr_)
            else:
                tau, _, _ = AL.cbf_qp(v, wd_, wdd_, Pctrl, ellbar, tau_ref=tr_)
        d = -(Ptrue["b"]*phi(v))/Ptrue["mass"]          # the plant uses the true parameters
        v = np.clip(v + dt*(d + tau/Ptrue["mass"]), -0.999*Ptrue["vsat"], 0.999*Ptrue["vsat"])
        # Score robust runs against the metric whose floor the controller certifies.
        barrier_model = Pworst if method == "cbf-robust" else Ptrue
        hs.append(h_value(v, barrier_model, ellbar))
        we.append(np.linalg.norm(Ptrue["A"]@phi(v) - wd_))
    return float(np.min(hs)), float(np.sqrt(np.mean(np.array(we)**2)))

if __name__ == "__main__":
    P = hexarotor(); NR = 8
    eb_nom, ld, hop = floor_rule(P, 1.5, 0.75, kappa=0.5)
    print("nominal floor ellbar=%.4f (ell_drop=%.3f, h_op=%.3f)\n" % (eb_nom, ld, hop))
    print("%5s | %-14s %9s %9s %9s %9s" % ("p", "controller", "minh_mean", "minh_worst", "viol.rate", "wErr"))
    rows = {}
    for p in [0.0, 0.10, 0.20, 0.30]:
        Pw = perturb_params(P, p, worst=True)
        # robust floor must be achievable under the worst-case model
        eb_rob, ldw, hopw = floor_rule(Pw, 1.5, 0.75, kappa=0.5)
        rng = np.random.default_rng(11)
        for label in ["energy", "cbf-nominal", "cbf-robust"]:
            if label == "cbf-robust" and eb_rob is None:
                print("%4.0f%% | %-14s   NOT CERTIFIABLE under worst case (h_op=%.2f<=%.2f)"
                      % (p*100, label, hopw, ldw)); continue
            hmins = []; werrs = []
            for r in range(NR if p > 0 else 1):
                Pt = perturb_params(P, p, worst=False, rng=rng) if p > 0 else P
                Pc = P
                eb = eb_rob if label == "cbf-robust" else eb_nom
                mth = label if label == "cbf-robust" else label.removeprefix("cbf-")
                hm, we = rollout(Pt, Pc, eb, mth, p=p, Pworst=Pw)
                hmins.append(hm); werrs.append(we)
            hmins = np.array(hmins)
            print("%4.0f%% | %-14s %+9.4f %+10.4f %9.2f %9.5f"
                  % (p*100, label, hmins.mean(), hmins.min(),
                     float(np.mean(hmins < 0)), float(np.mean(werrs))))
            rows[(p, label)] = (hmins.mean(), hmins.min(), float(np.mean(hmins < 0)), float(np.mean(werrs)))
        print()
    np.save(result_path("robust_rows.npy"), rows, allow_pickle=True)
