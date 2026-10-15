"""
allocators.py -- the allocation policies compared in the paper.

Every policy outputs motor torques tau for the actuator model

    dv/dt = d(v) + M^-1 tau,   d(v) = -M^-1 b v|v|,   |tau_i| <= tau_bar_i,

and is asked to track a desired wrench w_des(t) with rate wdot_des(t).

  energy_qp_tau      effort-minimising QP: min ||tau||^2 s.t. wrench servo and torque box.
                     Continuous, but blind to actuator authority.
  greedy             PD tracking of the readiness-maximising speeds on the current wrench
                     fibre (fiber_argmax_h_global). The optimum can jump between sign
                     patterns, so the setpoint is discontinuous.
  low-pass           the same setpoint passed through a first-order filter; smooths the
                     jumps at the cost of wrench error while the filter lags.
  cbf_qp             proposed filter: stay close to a reference torque (the effort QP)
                     subject to the readiness barrier, the wrench servo (with slack) and
                     the torque box.
  cbf_qp_robust      the same filter evaluated on the worst-case motor parameters of an
                     uncertainty box, with worst-case drift and a torque reserve.

The greedy and low-pass loops live in the simulators (sim_core.py, harden.py);
this module provides their setpoint and tracking law.
"""
import numpy as np
import scipy.sparse as sp
import osqp
from daam import D_matrix, grad_h, h_value, a_bar
from scipy.optimize import minimize

phi = lambda v: v * np.abs(v)


def _osqp_solve(Pm, q, Am, l, u, warm=None):
    """Solve min 0.5 x'Pm x + q'x s.t. l <= Am x <= u. Returns (x or None, status, time)."""
    m = osqp.OSQP()
    Pm = sp.csc_matrix(Pm); Am = sp.csc_matrix(Am)
    m.setup(P=Pm, q=q, A=Am, l=l, u=u, verbose=False,
            eps_abs=1e-7, eps_rel=1e-7, max_iter=8000, polish=True)
    if warm is not None:
        m.warm_start(x=warm)
    r = m.solve()
    # OSQP >= 1.0 still returns an iterate when the problem is infeasible or the
    # solver stops early. Accept only "solved" (1) and "solved inaccurate" (2) so
    # that the callers' fallbacks run otherwise.
    if r.info.status_val not in (1, 2):
        return None, r.info.status_val, r.info.run_time
    return r.x, r.info.status_val, r.info.run_time


def _barrier_fallback(gh, tau_limit, J, d, Minv, mu):
    """Used when the filter QP fails: push every motor along the barrier gradient.

    Readiness has priority, so only the wrench task is given up; delta is the
    resulting wrench-servo residual.
    """
    tau = np.sign(gh) * tau_limit
    delta = J @ (d + Minv @ tau) - mu
    return tau, delta


# ---------- readiness-maximising setpoints on the wrench fibre (greedy / low-pass) ----------
def fiber_argmax_h(w, P, v0):
    """Local solution of max h(v) s.t. A phi(v) = w, started from v0 (SLSQP)."""
    A = P["A"]; lim = 0.999 * P["vsat"]
    cons = {"type": "eq", "fun": lambda v: A @ phi(v) - w}
    res = minimize(lambda v: -h_value(v, P), v0,
                   constraints=[cons], bounds=[(-lim, lim)] * P["n"],
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
    return (res.x if res.success else v0)


def fiber_min_energy(w, P, v0):
    """Minimum sum of squared thrusts on the fibre (not used by the experiments)."""
    A = P["A"]; lim = 0.999 * P["vsat"]
    cons = {"type": "eq", "fun": lambda v: A @ phi(v) - w}
    res = minimize(lambda v: np.sum(phi(v)**2), v0,
                   constraints=[cons], bounds=[(-lim, lim)] * P["n"],
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-12})
    return (res.x if res.success else v0)


# ---------- torque law that tracks a speed setpoint (greedy and low-pass) ----------
def track_setpoint(v, vstar_sp, P, Kp=60.0):
    """Feedback-linearising P loop on rotor speed, clipped to the torque box."""
    d = -(np.diag(1 / P["mass"]) @ (P["b"] * phi(v)))
    tau = P["mass"] * (Kp * (vstar_sp - v) - d)
    return np.clip(tau, -P["tau_bar"], P["tau_bar"])


# ---------- proposed readiness filter (Algorithm 1) ----------
def cbf_qp(v, w_des, wdot_des, P, ellbar, alpha=5.0, Kw=8.0, rho=1e4, Rreg=1.0,
           tau_ref=None, eta=0.0):
    """Readiness-barrier QP at torque level.

        min_{tau, delta}  ||tau - tau_ref||^2_R + rho ||delta||^2
        s.t.  grad_h' (d + M^-1 tau) >= -alpha h + eta          (barrier)
              J (d + M^-1 tau) - delta = wdot_des + Kw (w_des - A phi(v))   (wrench servo)
              |tau_i| <= tau_bar_i                              (torque box)

    J = 2 A diag|v| is the Jacobian of the wrench map. The slack delta keeps the
    QP feasible: when the barrier and the servo conflict, tracking gives way.
    eta >= 0 tightens the barrier (discrete-time or delay margin).
    Returns (tau, delta, solver time).
    """
    n, m = P["n"], P["m"]
    A = P["A"]
    d = -(P["b"] * phi(v)) / P["mass"]
    Minv = np.diag(1 / P["mass"])
    J = 2 * A * np.abs(v)[None, :]
    gh = grad_h(v, P)
    h = h_value(v, P, ellbar)
    wtil = w_des - A @ phi(v)
    mu = wdot_des + Kw * wtil
    if tau_ref is None:
        tau_ref = np.zeros(n)
    # decision variables x = [tau (n); delta (m)]
    Pm = np.zeros((n + m, n + m))
    Pm[:n, :n] = 2 * Rreg * np.eye(n)
    Pm[n:, n:] = 2 * rho * np.eye(m)
    q = np.concatenate([-2 * Rreg * tau_ref, np.zeros(m)])
    rows_l, rows_u, Arows = [], [], []
    # barrier: grad_h' M^-1 tau >= -alpha h + eta - grad_h' d
    Arows.append(np.concatenate([gh @ Minv, np.zeros(m)]))
    rows_l.append(-alpha * h + eta - gh @ d); rows_u.append(np.inf)
    # wrench servo as an equality: J M^-1 tau - delta = mu - J d
    Aeq = np.hstack([J @ Minv, -np.eye(m)])
    beq = mu - J @ d
    for r in range(m):
        Arows.append(Aeq[r]); rows_l.append(beq[r]); rows_u.append(beq[r])
    # torque box
    for i in range(n):
        e = np.zeros(n + m); e[i] = 1.0
        Arows.append(e); rows_l.append(-P["tau_bar"][i]); rows_u.append(P["tau_bar"][i])
    Am = np.array(Arows); l = np.array(rows_l); u = np.array(rows_u)
    x, st, rt = _osqp_solve(Pm, q, Am, l, u)
    if x is None:
        return (*_barrier_fallback(gh, P["tau_bar"], J, d, Minv, mu), rt)
    tau = np.clip(x[:n], -P["tau_bar"], P["tau_bar"])
    delta = x[n:]
    return tau, delta, rt


def energy_qp_tau(v, w_des, wdot_des, P, Kw=8.0):
    """Effort-minimising baseline: min ||tau||^2 s.t. exact wrench servo and torque box.

    If the requested wrench rate cannot be produced inside the torque box, the
    servo is relaxed with a heavily weighted slack (least-squares tracking).
    Also used as the reference torque tau_ref of the readiness filter.
    Returns (tau, solver time).
    """
    n, m = P["n"], P["m"]; A = P["A"]
    d = -(P["b"] * phi(v)) / P["mass"]; Minv = np.diag(1 / P["mass"])
    J = 2 * A * np.abs(v)[None, :]
    mu = wdot_des + Kw * (w_des - A @ phi(v))
    Pm = 2 * np.eye(n); q = np.zeros(n)
    Aeq = J @ Minv; beq = mu - J @ d
    Arows = list(Aeq); l = list(beq); u = list(beq)
    for i in range(n):
        e = np.zeros(n); e[i] = 1; Arows.append(e); l.append(-P["tau_bar"][i]); u.append(P["tau_bar"][i])
    x, st, rt = _osqp_solve(Pm, q, np.array(Arows), np.array(l), np.array(u))
    if x is None:
        # infeasible: min ||tau||^2 + rho_e ||delta||^2  s.t. J M^-1 tau - delta = mu - J d, box
        rho_e = 1e4
        Pm2 = np.zeros((n + m, n + m)); Pm2[:n, :n] = 2 * np.eye(n); Pm2[n:, n:] = 2 * rho_e * np.eye(m)
        A2 = np.vstack([np.hstack([Aeq, -np.eye(m)]), np.hstack([np.eye(n), np.zeros((n, m))])])
        l2 = np.concatenate([beq, -P["tau_bar"]]); u2 = np.concatenate([beq, P["tau_bar"]])
        x2, st2, rt2 = _osqp_solve(Pm2, np.zeros(n + m), A2, l2, u2)
        if x2 is None:
            return np.zeros(n), rt + rt2
        return np.clip(x2[:n], -P["tau_bar"], P["tau_bar"]), rt + rt2
    return np.clip(x, -P["tau_bar"], P["tau_bar"]), rt


def fiber_argmax_h_global(w, P, seed=0, restarts=6):
    """Multi-start version of fiber_argmax_h: the setpoint of greedy DAAM.

    Starts from +v*, -v* and `restarts` random points and keeps the best local
    optimum. Because it picks the global optimum it can switch sign pattern from
    one call to the next, which is the jump the paper discusses.
    """
    A = P["A"]; lim = 0.999 * P["vsat"]
    rng = np.random.default_rng(seed)
    best = None; bestval = -np.inf
    starts = [np.full(P["n"], P["vstar"]), -np.full(P["n"], P["vstar"])]
    for _ in range(restarts):
        starts.append(rng.uniform(-lim, lim, size=P["n"]))
    for v0 in starts:
        cons = {"type": "eq", "fun": lambda v: A @ phi(v) - w}
        res = minimize(lambda v: -h_value(v, P), v0, constraints=[cons],
                       bounds=[(-lim, lim)] * P["n"], method="SLSQP",
                       options={"maxiter": 200, "ftol": 1e-10})
        if res.success and -res.fun > bestval:
            bestval = -res.fun; best = res.x
    return best if best is not None else np.full(P["n"], P["vstar"])


# ---------- diagnostics for the feasibility theorem ----------
def kernel_basis(v, P):
    """Orthonormal basis of ker J(v), the speed directions that leave the wrench unchanged."""
    J = 2 * P["A"] * np.abs(v)[None, :]
    u, s, vt = np.linalg.svd(J)
    r = np.sum(s > 1e-10)
    return vt[r:].T            # n x (n - r)


def gamma_theta(v, P):
    """Null-space alignment: gamma = max grad_h' xi over xi in ker J with |xi_i| <= a_bar_i.

    theta = gamma / ||W^(1/2) grad_h|| measures how much readiness can be gained
    without disturbing the wrench (a small LP, solved with HiGHS).
    """
    from daam import grad_h as _g, a_bar as _ab
    N = kernel_basis(v, P)
    if N.shape[1] == 0:
        return 0.0, 0.0
    gh = _g(v, P); ab = _ab(v, P)
    # LP in kernel coordinates z: max (N' gh)' z  s.t. |(N z)_i| <= a_bar_i
    from scipy.optimize import linprog
    c = -(N.T @ gh)
    Aub = np.vstack([N, -N]); bub = np.concatenate([ab, ab])
    r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * N.shape[1], method="highs")
    gam = float(-r.fun) if r.success else 0.0
    denom = float(np.linalg.norm(np.sqrt(np.maximum(ab, 0)) * gh))
    return gam, (gam / denom if denom > 1e-12 else 0.0)


def hessian_bound(P, nsamp=400, seed=0):
    """Sampled bound on ||Hess h||_2 for speeds in [0.3, 0.9] v_sat (sizes the discrete-time margin)."""
    from daam import grad_h as _g
    rng = np.random.default_rng(seed); L = 0.0
    for _ in range(nsamp):
        v = rng.uniform(0.3, 0.9, size=P["n"]) * P["vsat"]
        H = np.zeros((P["n"], P["n"])); e = 1e-5
        for i in range(P["n"]):
            vp = v.copy(); vp[i] += e; vm = v.copy(); vm[i] -= e
            H[:, i] = (_g(vp, P) - _g(vm, P)) / (2 * e)
        L = max(L, np.linalg.norm(H, 2))
    return L


# ---------- robust filter ----------
def worst_case_drift_term(v, gh, P, p):
    """min over the uncertainty box of grad_h' d(v; theta).

    d_i = -b_i v_i|v_i| / m_i is linear in b_i, so the minimum is reached at an
    end of [b (1 - p), b (1 + p)], independently for each rotor.
    """
    ph = v * np.abs(v)
    tot = 0.0
    for i in range(len(v)):
        lo = -gh[i] * P["b"][i] * (1 - p) * ph[i] / P["mass"][i]
        hi = -gh[i] * P["b"][i] * (1 + p) * ph[i] / P["mass"][i]
        tot += min(lo, hi)
    return tot


def cbf_qp_robust(v, w_des, wdot_des, Pnom, Pworst, ellbar, p,
                  alpha=5.0, Kw=8.0, rho=1e4, Rreg=1.0, tau_ref=None,
                  worst_case_drift=True):
    """Readiness filter that holds for every plant in the +-p parameter box.

    Three changes with respect to cbf_qp:
      * h and grad h are evaluated on the worst-case corner Pworst
        (tau_bar (1 - p), b (1 + p)), which lower-bounds ln det D over the box;
      * the drift term in the barrier row is the worst case over the box;
      * the torque box is shrunk to tau_bar (1 - p), so the commanded torque is
        available whatever the true limit.

    worst_case_drift=False keeps the nominal drift (and the torque reserve); it is
    only used for the metric-only ablation.
    """
    from daam import grad_h as _g, h_value as _h
    n, m = Pnom["n"], Pnom["m"]
    A = Pnom["A"]
    ph = v * np.abs(v)
    Minv = np.diag(1 / Pnom["mass"])
    J = 2 * A * np.abs(v)[None, :]
    gh = _g(v, Pworst)                      # gradient of the robust barrier
    h = _h(v, Pworst, ellbar)
    d_nom = -(Pnom["b"] * ph) / Pnom["mass"]
    drift = (worst_case_drift_term(v, gh, Pnom, p) if worst_case_drift
             else float(gh @ d_nom))
    tau_limit = Pnom["tau_bar"] * (1 - p)
    mu = wdot_des + Kw * (w_des - A @ ph)
    if tau_ref is None:
        tau_ref = np.zeros(n)
    Pm = np.zeros((n + m, n + m))
    Pm[:n, :n] = 2 * Rreg * np.eye(n); Pm[n:, n:] = 2 * rho * np.eye(m)
    q = np.concatenate([-2 * Rreg * tau_ref, np.zeros(m)])
    Arows, l, u = [], [], []
    # barrier row with the chosen drift bound
    Arows.append(np.concatenate([gh @ Minv, np.zeros(m)]))
    l.append(-alpha * h - drift); u.append(np.inf)
    # wrench servo (nominal model) with slack
    Aeq = np.hstack([J @ Minv, -np.eye(m)]); beq = mu - J @ d_nom
    for r in range(m):
        Arows.append(Aeq[r]); l.append(beq[r]); u.append(beq[r])
    # reduced torque box
    for i in range(n):
        e = np.zeros(n + m); e[i] = 1.0
        Arows.append(e); l.append(-tau_limit[i]); u.append(tau_limit[i])
    x, st, rt = _osqp_solve(Pm, q, np.array(Arows), np.array(l), np.array(u))
    if x is None:
        return (*_barrier_fallback(gh, tau_limit, J, d_nom, Minv, mu), rt)
    return np.clip(x[:n], -tau_limit, tau_limit), x[n:], rt
