"""
make_sim_fig.py — mechanism figure: commanded allocation + authority history,
at a CERTIFIABLE authority-scarce operating point (thrust 0.75, amp 1.5),
i.e. the regime where greedy DAAM violates the floor.
Writes figures/generated/fig_timehist.pdf and the corresponding CSV data.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from platforms import hexarotor
from daam import h_value
import allocators as AL
from harden import traj
from final_exp import floor_rule
from project_paths import figure_data_path, figure_path

phi = lambda v: v*np.abs(v)

def rollout(P, method, ellbar, amp=1.5, thrust=0.75, T=2.0, dt=1e-3, seed=0):
    w, wd = traj(P, amp, thrust)
    v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"]))
    N = int(T/dt); UPD = max(1, int(0.02/dt)); hold = v.copy()
    ts, cmd, hs = [], [], []
    for k in range(N):
        t = k*dt; wd_, wdd_ = w(t), wd(t)
        c = v.copy()
        if method == "energy":
            tau, _ = AL.energy_qp_tau(v, wd_, wdd_, P)
        elif method == "greedy":
            if k % UPD == 0:
                hold = AL.fiber_argmax_h_global(wd_, P, seed=seed, restarts=3)
            c = hold; tau = AL.track_setpoint(v, c, P)
        else:
            tr_, _ = AL.energy_qp_tau(v, wd_, wdd_, P)
            tau, _, _ = AL.cbf_qp(v, wd_, wdd_, P, ellbar, tau_ref=tr_)
        d = -(P["b"]*phi(v))/P["mass"]
        v = np.clip(v + dt*(d + tau/P["mass"]), -0.999*P["vsat"], 0.999*P["vsat"])
        ts.append(t); cmd.append(c.copy()); hs.append(h_value(v, P, ellbar))
    return np.array(ts), np.array(cmd), np.array(hs)

def main():
    figure_path(".")
    figure_data_path(".")
    P = hexarotor()
    eb, ld, hop = floor_rule(P, 1.5, 0.75, kappa=0.5)
    print("certifiable point: ell_drop=%.3f h_op=%.3f -> ellbar=%.3f" % (ld, hop, eb))
    tg, cg, hg = rollout(P, "greedy", eb)
    tc, cc, hc = rollout(P, "cbf", eb)
    te, ce, he = rollout(P, "energy", eb)
    # ---- CSV export (so figures rebuild without re-running) ----
    np.savetxt(figure_data_path("timehist_greedy_cmd.csv"),
               np.column_stack([tg, cg]), delimiter=",",
               header="t," + ",".join("v%d" % i for i in range(P["n"])), comments="")
    np.savetxt(figure_data_path("timehist_cbf_cmd.csv"),
               np.column_stack([tc, cc]), delimiter=",",
               header="t," + ",".join("v%d" % i for i in range(P["n"])), comments="")
    np.savetxt(figure_data_path("timehist_h.csv"),
               np.column_stack([te, he, hg, hc]), delimiter=",",
               header="t,h_energy,h_greedy,h_cbf", comments="")
    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for i in range(P["n"]): ax[0].plot(tg, cg[:, i], lw=0.7)
    ax[0].set_title(r"greedy DAAM: commanded $v^\star(t)$", fontsize=8)
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"$v_i$")
    for i in range(P["n"]): ax[1].plot(tc, cc[:, i], lw=0.7)
    ax[1].set_title(r"proposed: commanded $v(t)$", fontsize=8); ax[1].set_xlabel("t [s]")
    ax[1].set_ylim(ax[0].get_ylim())
    ax[2].plot(te, he, color="gray", lw=0.9, label="energy-QP")
    ax[2].plot(tg, hg, color="tab:red", lw=0.9, label="greedy")
    ax[2].plot(tc, hc, color="tab:blue", lw=1.2, label="proposed")
    ax[2].axhline(0, color="k", ls="--", lw=0.8)
    ax[2].set_title(r"authority $h(t)$", fontsize=8); ax[2].set_xlabel("t [s]")
    ax[2].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(figure_path("fig_timehist.pdf")); plt.close()
    print("h_min: energy %+.4f | greedy %+.4f | cbf %+.4f" % (he.min(), hg.min(), hc.min()))
    print("TV cmd: greedy %.2f | cbf %.2f" % (np.abs(np.diff(cg,axis=0)).sum(),
                                              np.abs(np.diff(cc,axis=0)).sum()))
    print("wrote fig_timehist.pdf + figdata/timehist_*.csv")

if __name__ == "__main__":
    main()
