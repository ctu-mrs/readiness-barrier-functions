"""
sim.py — closed-loop allocation-layer comparison on a wrench trajectory that
enters the component-crossing regime (low thrust + large torque reversal).
Integrates vdot=d(v)+M^{-1}tau at 1 kHz; records continuity, tracking, authority.
"""
import numpy as np
from project_paths import result_path
from platforms import hexarotor
from daam import h_value
import allocators as AL

phi = lambda v: v * np.abs(v)


def wrench_traj(P):
    """low collective (0.4 nominal) + torque_x reversal through the JMP regime."""
    phistar = P["vstar"] * abs(P["vstar"])
    fz_nom = P["n"] * phistar
    tau_ref = 0.25 * phistar
    def w(t):
        fz = 0.70 * fz_nom
        tx = 2.5 * tau_ref * np.sin(2 * np.pi * t / 2.0)   # period 2 s
        return np.array([fz, tx, 0.0, 0.0])
    def wdot(t):
        tx = 2.5 * tau_ref * (2 * np.pi / 2.0) * np.cos(2 * np.pi * t / 2.0)
        return np.array([0.0, tx, 0.0, 0.0])
    return w, wdot, tau_ref


def run(method, P, ellbar, T=3.0, dt=1e-3, seed=0):
    w, wdot, _ = wrench_traj(P)
    N = int(T / dt)
    # init at fiberwise-optimal for the initial wrench, in the positive component
    v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"]))
    log = dict(t=[], v=[], vsp=[], werr=[], h=[], tau=[], rt=[], delta=[])
    vsp_filt = v.copy()
    for k in range(N):
        t = k * dt
        wd, wdd = w(t), wdot(t)
        rt = 0.0; delta = np.zeros(P["m"]); vsp = v.copy()
        if method == "energy":
            tau, rt = AL.energy_qp_tau(v, wd, wdd, P)
        elif method == "greedy":
            vsp = AL.fiber_argmax_h(wd, P, v)
            tau = AL.track_setpoint(v, vsp, P)
        elif method == "lowpass":
            raw = AL.fiber_argmax_h(wd, P, vsp_filt)
            a = dt / (0.05 + dt)                     # 50 ms low-pass
            vsp_filt = (1 - a) * vsp_filt + a * raw
            vsp = vsp_filt
            tau = AL.track_setpoint(v, vsp, P)
        elif method == "cbf":
            tau_ref, _ = AL.energy_qp_tau(v, wd, wdd, P)
            tau, delta, rt = AL.cbf_qp(v, wd, wdd, P, ellbar, tau_ref=tau_ref)
        # integrate actuator dynamics
        d = -(P["b"] * phi(v)) / P["mass"]
        v = v + dt * (d + tau / P["mass"])
        v = np.clip(v, -0.999 * P["vsat"], 0.999 * P["vsat"])
        log["t"].append(t); log["v"].append(v.copy()); log["vsp"].append(vsp.copy())
        log["werr"].append(np.linalg.norm(P["A"] @ phi(v) - wd))
        log["h"].append(h_value(v, P, ellbar)); log["tau"].append(tau.copy())
        log["rt"].append(rt); log["delta"].append(np.linalg.norm(delta))
    for kk in log: log[kk] = np.array(log[kk])
    return log


def metrics(log, dt=1e-3):
    v = log["v"]; vsp = log["vsp"]
    tv_cmd = np.sum(np.abs(np.diff(vsp, axis=0)))           # total variation of commanded setpoint
    max_rate = np.max(np.abs(np.diff(v, axis=0)) / dt)      # peak realized |vdot|
    # component crossings = sign flips of any rotor along the commanded setpoint
    sgn = np.sign(vsp); sgn[sgn == 0] = 1
    flips = int(np.sum(np.abs(np.diff(sgn, axis=0)) > 0))
    return dict(
        flips=flips,
        TV_cmd=tv_cmd,
        max_vdot=max_rate,
        werr_rms=np.sqrt(np.mean(log["werr"]**2)),
        werr_max=np.max(log["werr"]),
        h_min=np.min(log["h"]),
        below_floor_frac=np.mean(log["h"] < 0),
        rt_ms=(np.mean(log["rt"]) * 1e3 if np.any(log["rt"] > 0) else np.nan),
    )


if __name__ == "__main__":
    P = hexarotor()
    from gate import ell_dagger
    ldag, _ = ell_dagger(P)
    probe = run("greedy", P, 0.0)          # ellbar=0 -> log["h"] is absolute logdet D
    op_min = float(np.min(probe["h"]))
    ellbar = ldag + 0.25 * (op_min - ldag) if op_min > ldag else ldag + 0.1
    print("ell_dagger=%.3f  op_min(logdetD)=%.3f  floor ellbar=%.3f  (above saddle=%.3f)"
          % (ldag, op_min, ellbar, ellbar - ldag))
    print()
    hdr = ("method", "flips", "TV_cmd", "max_vdot", "werr_rms", "werr_max",
           "h_min", "belowFlr", "rt[ms]")
    print("%-9s %6s %10s %10s %10s %10s %8s %9s %8s" % hdr)
    results = {}
    for mth in ["energy", "greedy", "lowpass", "cbf"]:
        log = run(mth, P, ellbar)
        m = metrics(log)
        results[mth] = (log, m)
        rt = ("%.3f" % m["rt_ms"]) if not np.isnan(m["rt_ms"]) else "-"
        print("%-9s %6d %10.3f %10.1f %10.4f %10.4f %8.3f %9.3f %8s"
              % (mth, m["flips"], m["TV_cmd"], m["max_vdot"], m["werr_rms"],
                 m["werr_max"], m["h_min"], m["below_floor_frac"], rt))
    np.save(result_path("sim_results.npy"), {k: v[0] for k, v in results.items()}, allow_pickle=True)
    np.save(result_path("sim_meta.npy"), {"ellbar": ellbar, "ldag": ldag}, allow_pickle=True)
