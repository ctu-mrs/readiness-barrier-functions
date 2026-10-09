"""
gate.py — Phase-1 non-vacuity gate.
Computes h_max, the binding saddle level ell_dagger (single rotor to zero),
the usable authority window, and the operational-envelope single-component test.
"""
import numpy as np
from daam import h_value, D_matrix, psi
from scipy.optimize import minimize


def h_max_point(P):
    """Global max of h at all-sweet-spot (psi even in v_i)."""
    v = np.full(P["n"], P["vstar"])
    return v, h_value(v, P)


def verify_hmax_random(P, N=200000, seed=0):
    rng = np.random.default_rng(seed)
    best = -np.inf
    lim = P["vsat"]
    for _ in range(N):
        v = rng.uniform(-lim, lim, size=P["n"])
        h = h_value(v, P)
        if h > best:
            best = h
    return best


def ell_dagger(P):
    """Binding saddle = highest single-rotor-to-zero level (others at sweet-spot)."""
    pstar = psi(np.full(P["n"], P["vstar"]), P)[0]
    levels = []
    for k in range(P["n"]):
        pv = np.full(P["n"], pstar); pv[k] = 0.0
        Dk = 4.0 * (P["A"] * pv[None, :]) @ P["A"].T
        s, ld = np.linalg.slogdet(Dk)
        levels.append(ld if s > 0 else -np.inf)
    return float(np.max(levels)), np.array(levels)


def alloc_on_fiber(w, P, v0):
    """DAAM-optimal allocation on fiber f(v)=w (max h s.t. A phi(v)=w)."""
    A = P["A"]
    phi = lambda v: v * np.abs(v)
    cons = {"type": "eq", "fun": lambda v: A @ phi(v) - w}
    lim = 0.999 * P["vsat"]
    res = minimize(lambda v: -h_value(v, P), v0, constraints=[cons],
                   bounds=[(-lim, lim)] * P["n"], method="SLSQP",
                   options={"maxiter": 400, "ftol": 1e-11})
    return res.x, h_value(res.x, P), res.success


def report(P):
    print("=" * 64)
    print("PLATFORM:", P["name"], " n=%d m=%d  redundancy=%d"
          % (P["n"], P["m"], P["n"] - P["m"]))
    print("rank(A) =", np.linalg.matrix_rank(P["A"]),
          "  vstar=%.4f vsat=%.4f" % (P["vstar"], P["vsat"]))
    vmx, hmx = h_max_point(P)
    hrand = verify_hmax_random(P)
    ldag, levels = ell_dagger(P)
    print("h_max (analytic)   = %.6f" % hmx)
    print("h_max (2e5 random) = %.6f   (<= analytic ok? %s)"
          % (hrand, hrand <= hmx + 1e-6))
    print("ell_dagger         = %.6f" % ldag)
    gap = hmx - ldag
    print("GAP h_max-ell_dag  = %.6f   (=ln3? %s)"
          % (gap, abs(gap - np.log(3)) < 1e-6))
    print("usable floor window (ell_dagger, h_max) width = %.6f  -> %s"
          % (gap, "NON-VACUOUS" if gap > 0.1 else "TOO THIN"))
    return dict(h_max=hmx, ell_dagger=ldag, gap=gap)


if __name__ == "__main__":
    from platforms import hexarotor, tilted_octorotor
    for P in (hexarotor(), tilted_octorotor()):
        report(P)
        print()


def weighted_leverage(P):
    """Exact dropout gaps for HETEROGENEOUS motors:
       Delta_k = -ln(1 - lam_k),  lam_k = psi*_k A_k^T (A diag(psi*) A^T)^{-1} A_k.
    Reduces to the unweighted leverage iff all psi*_i coincide."""
    from daam import psi as _psi
    A = P["A"]
    vst = np.sqrt(P["tau_bar"] / (3 * P["b"]))
    ps = _psi(vst, P)
    S = (A * ps[None, :]) @ A.T
    lam = np.array([ps[k] * (A[:, k] @ np.linalg.solve(S, A[:, k])) for k in range(P["n"])])
    with np.errstate(divide="ignore"):
        Delta = -np.log(np.maximum(1e-300, 1.0 - lam))
    return lam, Delta, ps


def ell_dagger_exact(P):
    """ell_dagger = L_max - min_k Delta_k, using the exact (weighted) gaps."""
    from daam import psi as _psi
    A = P["A"]
    vst = np.sqrt(P["tau_bar"] / (3 * P["b"]))
    ps = _psi(vst, P)
    Lmax = np.linalg.slogdet(4 * (A * ps[None, :]) @ A.T)[1]
    lam, Delta, _ = weighted_leverage(P)
    return Lmax - np.min(Delta), Lmax, Delta
