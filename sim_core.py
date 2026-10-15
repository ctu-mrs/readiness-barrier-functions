"""
sim_core.py -- closed-loop simulator used by the revision experiments (exp_*.py).

Same integration scheme, setpoint update rate and allocators as harden.run, so
the energy, greedy and cbf results agree with it. In addition it provides
  * the low-pass baseline with time constant tau_f = 0.05 T0 (a = dt/(tau_f + dt)
    at every step); harden.run has since been corrected in the same way;
  * more logging: time-averaged readiness, time below the floor, barrier
    activity, slack, commanded rate and solver time;
  * hooks for the robustness, delay and forward-flight studies: a different
    plant (Pplant), a custom wrench reference (wfun), the true allocation matrix
    A_eff(v, t) of the plant, an input delay, speed-measurement noise and an
    integral loop on the true wrench (ki).

Time is nondimensional, t in units of T0 = m / sqrt(tau_bar b).
"""
import numpy as np
from daam import h_value, grad_h
import allocators as AL
from harden import traj

phi = lambda v: v * np.abs(v)
METHODS = ["energy", "greedy", "lowpass", "cbf"]


def run(P, method, ellbar, T=2.0, dt=1e-3, amp=1.5, thrust=0.9, seed=0,
        wfun=None, Pplant=None, Aeff=None, delay=0.0, delay_init="hold",
        tau_f=0.05, restarts=2, upd=0.02, alpha=5.0, rho=1e4, ki=0.0,
        noise=0.0, log=False, Aslope=None):
    """Simulate one allocator ("energy", "greedy", "lowpass" or "cbf") for T time units.

    P is the model the allocator uses; Pplant (default P) is the plant that is
    integrated. delay is an input delay in T0; the buffer starts either with the
    torques that hold the initial speeds ("hold") or with zeros ("zeros").
    Aeff(v, t) gives the true wrench map (forward flight) and Aslope(v, t) the
    true slope matrix used for the true readiness htrue. With log=True the time
    histories are returned as well.
    """
    w, wd = traj(P, amp, thrust) if wfun is None else wfun
    PL = P if Pplant is None else Pplant
    rng = np.random.default_rng(1000 + seed)
    # start from the readiness-optimal allocation of the initial wrench
    v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"]))
    N = int(T / dt); UPD = max(1, int(upd / dt)); hold = v.copy(); vlp = v.copy()
    prev = v.copy(); TV = 0.0
    kd = int(round(delay / dt))
    if kd > 0:
        tau_hold = PL["b"] * phi(v)  # torque that holds the initial speeds
        buf = [tau_hold.copy() if delay_init == "hold" else np.zeros(P["n"]) for _ in range(kd)]
    z = np.zeros(P["m"])  # integral outer-loop state (true-wrench correction)
    H, Ltrue, WE, RATE, ACT, DEL, RT, TT, V = [], [], [], [], [], [], [], [], []
    for k in range(N):
        t = k * dt
        wref, wdref = w(t), wd(t)
        wc = wref + z                       # commanded wrench, with integral correction
        vm = v + (noise * rng.standard_normal(P["n"]) if noise > 0 else 0.0)
        sp = vm.copy(); act = 0; dl = np.zeros(P["m"]); rt = 0.0
        if method == "energy":
            tau, rt = AL.energy_qp_tau(vm, wc, wdref, P)
        elif method in ("greedy", "lowpass"):
            if k % UPD == 0:
                hold = AL.fiber_argmax_h_global(wc, P, seed=seed, restarts=restarts)
            if method == "greedy":
                sp = hold
            else:
                a = dt / (tau_f + dt); vlp = (1 - a) * vlp + a * hold; sp = vlp
            tau = AL.track_setpoint(vm, sp, P)
        elif method == "cbf":
            tr_, _ = AL.energy_qp_tau(vm, wc, wdref, P)
            tau, dl, rt = AL.cbf_qp(vm, wc, wdref, P, ellbar, tau_ref=tr_, alpha=alpha, rho=rho)
            act = int(np.max(np.abs(tau - tr_)) > 1e-5)   # 1 when the filter changed the nominal torque
        else:
            raise ValueError(method)
        if kd > 0:                          # input delay: FIFO of kd samples
            buf.append(tau); tau_app = buf.pop(0)
        else:
            tau_app = tau
        RATE.append(float(np.max(np.abs(sp - prev)) / dt))   # fastest commanded speed change
        TV += float(np.sum(np.abs(sp - prev))); prev = sp
        d = -(PL["b"] * phi(v)) / PL["mass"]
        tau_app = np.clip(tau_app, -PL["tau_bar"], PL["tau_bar"])
        v = np.clip(v + dt * (d + tau_app / PL["mass"]), -0.999 * PL["vsat"], 0.999 * PL["vsat"])
        At = P["A"] if Aeff is None else Aeff(v, t)
        wtrue = At @ phi(v)
        if ki > 0:
            z = z + dt * ki * (wref - wtrue)
        WE.append(float(np.linalg.norm(wtrue - wref)))
        H.append(h_value(v, P, ellbar))
        if Aeff is not None or Pplant is not None:
            # readiness of the true plant: true slopes / allocation map and true motor parameters
            from daam import psi
            As = At if Aslope is None else Aslope(v, t)
            Dt = 4.0 * (As * psi(v, PL)[None, :]) @ As.T
            s, ld = np.linalg.slogdet(Dt); Ltrue.append((ld if s > 0 else -np.inf) - ellbar)
        ACT.append(act); DEL.append(float(np.linalg.norm(dl))); RT.append(rt)
        if log:
            TT.append(t); V.append(v.copy())
    H = np.array(H)
    out = dict(hmin=float(H.min()), hmean=float(H.mean()), tbelow=float(np.sum(H < 0) * dt),
               TV=TV, werr=float(np.sqrt(np.mean(np.square(WE)))), werr_max=float(np.max(WE)),
               rate_peak=float(np.max(RATE[1:])), active=float(np.mean(ACT)),
               delta_max=float(np.max(DEL)), rt_mean=float(np.mean(RT)),
               rt_p99=float(np.percentile(RT, 99)) if any(RT) else 0.0,
               minabsv=None)
    if Ltrue:
        Lt = np.array(Ltrue); out.update(htrue_min=float(Lt.min()), htrue_mean=float(Lt.mean()))
    if log:
        out.update(t=np.array(TT), v=np.array(V), h=H, rate=np.array(RATE), werr_t=np.array(WE),
                   delta=np.array(DEL), act=np.array(ACT))
        out["minabsv"] = float(np.min(np.abs(out["v"])))
        if Ltrue:
            out["htrue"] = np.array(Ltrue)
    return out
