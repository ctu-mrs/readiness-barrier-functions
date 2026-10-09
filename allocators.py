"""
allocators.py — four torque-level allocation policies feeding a common
actuator integrator  vdot = d(v) + M^{-1} tau,  |tau| <= tau_bar.

Task = wrench tracking of w_des(t). All policies output tau.

  energy_qp   : min ||tau||^2 s.t. servo(Ja=wdot+Kw*wtil); continuous, authority-blind
  greedy_daam : PD-track the fiberwise argmax_h setpoint v*(w); JUMPS at strata
  lowpass_daam: PD-track a low-pass-filtered v*(w); smooths jumps but lags -> wrench error
  cbf         : proposed readiness-CBF QP (barrier + servo + box)
"""
import numpy as np
import scipy.sparse as sp
import osqp
from daam import D_matrix, grad_h, h_value, a_bar
from scipy.optimize import minimize

phi = lambda v: v * np.abs(v)


def _osqp_solve(Pm, q, Am, l, u, warm=None):
    m = osqp.OSQP()
    Pm = sp.csc_matrix(Pm); Am = sp.csc_matrix(Am)
    m.setup(P=Pm, q=q, A=Am, l=l, u=u, verbose=False,
            eps_abs=1e-7, eps_rel=1e-7, max_iter=8000, polish=True)
    if warm is not None:
        m.warm_start(x=warm)
    r = m.solve()
    # OSQP >= 1.0 returns a (meaningless) iterate when the problem is infeasible;
    # report it as None so that callers' explicit fallbacks are triggered.
    if r.info.status_val not in (1, 2):
        return None, r.info.status_val, r.info.run_time
    return r.x, r.info.status_val, r.info.run_time


def _barrier_fallback(gh, tau_limit, J, d, Minv, mu):
    # Keep the barrier row when the QP solver fails; relax only the wrench task.
    tau = np.sign(gh) * tau_limit
    delta = J @ (d + Minv @ tau) - mu
    return tau, delta


# ---------- fiberwise argmax h (setpoint used by greedy / low-pass) ----------
def fiber_argmax_h(w, P, v0):
    A = P["A"]; lim = 0.999 * P["vsat"]
    cons = {"type": "eq", "fun": lambda v: A @ phi(v) - w}
    res = minimize(lambda v: -h_value(v, P), v0,
                   constraints=[cons], bounds=[(-lim, lim)] * P["n"],
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
    return (res.x if res.success else v0)


def fiber_min_energy(w, P, v0):
    A = P["A"]; lim = 0.999 * P["vsat"]
    cons = {"type": "eq", "fun": lambda v: A @ phi(v) - w}
    res = minimize(lambda v: np.sum(phi(v)**2), v0,
                   constraints=[cons], bounds=[(-lim, lim)] * P["n"],
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-12})
    return (res.x if res.success else v0)


# ---------- torque law that PD-tracks a spin setpoint (shared by 3 baselines) ----------
def track_setpoint(v, vstar_sp, P, Kp=60.0):
    d = -(np.diag(1 / P["mass"]) @ (P["b"] * phi(v)))
    tau = P["mass"] * (Kp * (vstar_sp - v) - d)     # feedback-linearizing PD
    return np.clip(tau, -P["tau_bar"], P["tau_bar"])


# ---------- proposed CBF-QP (torque level) ----------
def cbf_qp(v, w_des, wdot_des, P, ellbar, alpha=5.0, Kw=8.0, rho=1e4, Rreg=1.0,
           tau_ref=None, eta=0.0):
    """min ||tau-tau_ref||_R^2 + rho||delta||^2  s.t. barrier, servo, box."""
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
    # vars x = [tau (n); delta (m)]
    Pm = np.zeros((n + m, n + m))
    Pm[:n, :n] = 2 * Rreg * np.eye(n)
    Pm[n:, n:] = 2 * rho * np.eye(m)
    q = np.concatenate([-2 * Rreg * tau_ref, np.zeros(m)])
    # constraints
    rows_l, rows_u, Arows = [], [], []
    # barrier: gh^T (d + Minv tau) >= -alpha*h  ->  gh^T Minv tau >= -alpha*h - gh^T d
    Arows.append(np.concatenate([gh @ Minv, np.zeros(m)]))
    rows_l.append(-alpha * h + eta - gh @ d); rows_u.append(np.inf)
    # servo (equality as double inequality): J(d+Minv tau) - delta = mu
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
    """continuous, authority-blind baseline: min ||tau||^2 s.t. servo + box."""
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
        # wrench not producible within the torque box: least-squares relaxation
        # min ||tau||^2 + rho_e ||delta||^2  s.t. J M^-1 tau - delta = mu - J d, box
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
    """Global multi-start fiberwise argmax h (matches corpus multi-start SQP).
    Represents 'true greedy DAAM': the global optimum, which jumps across strata."""
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


# ---------- Theorem diagnostics: alignment ratio theta and Slater margin ----------
def kernel_basis(v, P):
    """orthonormal basis of ker J(v)."""
    J = 2 * P["A"] * np.abs(v)[None, :]
    u, s, vt = np.linalg.svd(J)
    r = np.sum(s > 1e-10)
    return vt[r:].T            # n x (n-r)


def gamma_theta(v, P):
    """gamma = max{grad_h^T xi : xi in ker J, |xi_i|<=abar_i}; theta = gamma/||W^{1/2} grad_h||."""
    from daam import grad_h as _g, a_bar as _ab
    N = kernel_basis(v, P)
    if N.shape[1] == 0:
        return 0.0, 0.0
    gh = _g(v, P); ab = _ab(v, P)
    # LP: max (N^T gh)^T z  s.t. |(N z)_i| <= ab_i   -> small LP, solve by scipy linprog
    from scipy.optimize import linprog
    c = -(N.T @ gh)
    Aub = np.vstack([N, -N]); bub = np.concatenate([ab, ab])
    r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * N.shape[1], method="highs")
    gam = float(-r.fun) if r.success else 0.0
    denom = float(np.linalg.norm(np.sqrt(np.maximum(ab, 0)) * gh))
    return gam, (gam / denom if denom > 1e-12 else 0.0)


def hessian_bound(P, nsamp=400, seed=0):
    """numerical bound on ||Hess h|| over a sweet-spot neighbourhood (for the dt margin)."""
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


def worst_case_drift_term(v, gh, P, p):
    """min_{theta in Theta_p} grad_h^T d(v;theta).  d_i = -b_i v_i|v_i|/m_i is
    linear in b_i, so the min is attained at an endpoint of [b(1-p), b(1+p)]."""
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
    """Use the worst-case metric, torque reserve, and optionally drift bound.

    Set ``worst_case_drift=False`` only for the metric-only ablation.
    """
    from daam import grad_h as _g, h_value as _h
    n, m = Pnom["n"], Pnom["m"]
    A = Pnom["A"]
    ph = v * np.abs(v)
    Minv = np.diag(1 / Pnom["mass"])
    J = 2 * A * np.abs(v)[None, :]
    gh = _g(v, Pworst)                      # gradient of the ROBUST barrier
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
    # The metric-only ablation uses nominal drift but keeps the torque reserve.
    Arows.append(np.concatenate([gh @ Minv, np.zeros(m)]))
    l.append(-alpha * h - drift); u.append(np.inf)
    Aeq = np.hstack([J @ Minv, -np.eye(m)]); beq = mu - J @ d_nom
    for r in range(m):
        Arows.append(Aeq[r]); l.append(beq[r]); u.append(beq[r])
    for i in range(n):
        e = np.zeros(n + m); e[i] = 1.0
        Arows.append(e); l.append(-tau_limit[i]); u.append(tau_limit[i])
    x, st, rt = _osqp_solve(Pm, q, np.array(Arows), np.array(l), np.array(u))
    if x is None:
        return (*_barrier_fallback(gh, tau_limit, J, d_nom, Minv, mu), rt)
    return np.clip(x[:n], -tau_limit, tau_limit), x[n:], rt
