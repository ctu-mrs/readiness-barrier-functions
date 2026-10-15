"""
harden.py -- reference wrench trajectory and the original closed-loop simulator.

traj() is the mission used by every nominal experiment: constant collective
plus a sinusoidal moment about one axis, whose reversals force a redundant
allocation to change shape. run() is the simulator of the submitted
manuscript; sim_core.run() extends it and is what the revision experiments use.

Running the file performs three checks:
  H1  a discrete-time margin eta on the barrier removes the small undershoot of
      the sampled-data loop;
  H2  the same comparison on the tilted octorotor (n = 8, m = 6);
  H3  the null-space alignment ratio theta along the filter trajectory, against
      the condition of the feasibility theorem.
"""
import numpy as np
from platforms import hexarotor, tilted_octorotor
from daam import h_value
from gate import ell_dagger
import allocators as AL
phi = lambda v: v*np.abs(v)


def traj(P, amp=2.5, thrust=0.5, period=2.0):
    """Wrench reference w(t) and its rate wd(t).

    Collective = thrust x (collective with every rotor at the sweet spot);
    moment = amp x 0.25 phi(v*) x sin(2 pi t / period) about the first moment axis.
    """
    ps = P["vstar"]*abs(P["vstar"]); fz = P["n"]*ps; tr = 0.25*ps
    m = P["m"]
    # wrench layout is platform dependent:
    #   m=4 (planar):  [f_z, tau_x, tau_y, tau_z]      -> collective idx 0, moment idx 1
    #   m=6 (full):    [f_x,f_y,f_z, m_x,m_y,m_z]      -> collective idx 2, moment idx 3
    iC, iM = (0, 1) if m == 4 else (2, 3)
    def w(t):
        x = np.zeros(m); x[iC] = thrust*fz
        x[iM] = amp*tr*np.sin(2*np.pi*t/period)
        return x
    def wd(t):
        x = np.zeros(m)
        x[iM] = amp*tr*(2*np.pi/period)*np.cos(2*np.pi*t/period)
        return x
    return w, wd


def run(P, method, ellbar, eta=0.0, T=2.0, dt=1e-3, amp=2.5, thrust=0.5,
        track_theta=False, seed=0):
    """Explicit-Euler closed loop of one allocator; returns TV, RMS wrench error and min h.

    TV is the total variation of the speed setpoint (for energy and cbf the
    setpoint is the current speed). The greedy setpoint is recomputed every 0.02 T0.
    """
    w, wd = traj(P, amp, thrust)
    v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"]))
    N = int(T/dt); UPD = max(1, int(0.02/dt)); hold = v.copy()
    TV = 0.0; prev = v.copy(); werr = []; hs = []; ths = []; vlp = v.copy()
    for k in range(N):
        t = k*dt; wd_, wdd_ = w(t), wd(t)
        sp = v.copy()
        if method == "energy":
            tau, _ = AL.energy_qp_tau(v, wd_, wdd_, P)
        elif method == "greedy":
            if k % UPD == 0: hold = AL.fiber_argmax_h_global(wd_, P, seed=seed, restarts=2)
            sp = hold; tau = AL.track_setpoint(v, sp, P)
        elif method == "lowpass":
            if k % UPD == 0: hold = AL.fiber_argmax_h_global(wd_, P, seed=seed, restarts=2)
            # first-order filter with time constant 0.05 T0, applied every step.
            # The submitted version used aa = (UPD dt)/(0.05 + UPD dt) at every step,
            # an effective time constant of about 3e-3 T0. Fixed here.
            aa = dt/(0.05 + dt); vlp = (1-aa)*vlp + aa*hold
            sp = vlp; tau = AL.track_setpoint(v, sp, P)
        else:  # cbf: the effort-QP torque is the reference the filter stays close to
            tr_, _ = AL.energy_qp_tau(v, wd_, wdd_, P)
            tau, _, _ = AL.cbf_qp(v, wd_, wdd_, P, ellbar, tau_ref=tr_, eta=eta)
        TV += np.sum(np.abs(sp-prev)); prev = sp
        # motor dynamics, explicit Euler; speeds kept just inside saturation
        d = -(P["b"]*phi(v))/P["mass"]
        v = np.clip(v+dt*(d+tau/P["mass"]), -0.999*P["vsat"], 0.999*P["vsat"])
        werr.append(np.linalg.norm(P["A"]@phi(v)-wd_)); hs.append(h_value(v, P, ellbar))
        if track_theta and k % 25 == 0:
            ths.append(AL.gamma_theta(v, P)[1])
    out = dict(TV=TV, werr=float(np.sqrt(np.mean(np.array(werr)**2))), hmin=float(np.min(hs)))
    if track_theta: out["theta_min"] = float(np.min(ths)); out["theta_med"] = float(np.median(ths))
    return out


if __name__ == "__main__":
    # ---- H1: discrete-time margin ----
    P = hexarotor(); ld, _ = ell_dagger(P); eb = ld + 0.4
    L = AL.hessian_bound(P, nsamp=80)
    # margin 0.5 L dt n max|vdot|, with max|vdot| taken as 1 in these units
    dt = 1e-3; eta = 0.5*L*dt*P["n"]*1.0
    print("H1  discrete-time margin (hexarotor, dt=%.0e, L=%.1f, eta=%.3f)" % (dt, L, eta))
    for name, e in [("eta=0 (plain)", 0.0), ("eta=margin", eta)]:
        r = run(P, "cbf", eb, eta=e, dt=dt)
        print("    %-14s h_min=%+.4f   TV=%6.2f  wErr=%.4f   %s"
              % (name, r["hmin"], r["TV"], r["werr"],
                 "FLOOR HELD" if r["hmin"] >= -1e-6 else "undershoot"))

    # ---- H3: alignment ratio ----
    r = run(P, "cbf", eb, eta=eta, dt=dt, track_theta=True)
    print("\nH3  alignment ratio along CBF trajectory: theta_min=%.3f  theta_median=%.3f"
          % (r["theta_min"], r["theta_med"]))
    for eps in [0.5, 0.6, 0.7]:
        print("      eps=%.1f -> Thm condition needs theta >= %.2f : %s"
              % (eps, (1-eps)/eps, "SATISFIED" if r["theta_min"] >= (1-eps)/eps else "not satisfied"))

    # ---- H2: octorotor ----
    Q = tilted_octorotor(); ldq, _ = ell_dagger(Q); ebq = ldq + 0.4
    Lq = AL.hessian_bound(Q, nsamp=60); etaq = 0.5*Lq*dt*Q["n"]*1.0
    print("\nH2  tilted octorotor (n=8,m=6), floor=ldrop+0.4, eta=%.3f" % etaq)
    print("    %-8s %8s %10s %10s" % ("method", "TV", "wErr", "h_min"))
    for mth in ["energy", "cbf"]:
        r = run(Q, mth, ebq, eta=(etaq if mth == "cbf" else 0.0), dt=dt, T=1.5)
        print("    %-8s %8.2f %10.4f %+10.4f" % (mth, r["TV"], r["werr"], r["hmin"]))
