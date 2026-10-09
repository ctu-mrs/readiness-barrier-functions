"""
robust.py — ROBUST readiness barrier.

Key fact: D(v;theta) = 4 A diag(psi) A^T with psi_i = v_i^2 abar_i^2 and
abar_i = (taubar_i - b_i v_i^2)/m_i is Loewner-MONOTONE in each psi_i, and
psi_i is monotone in (taubar_i up, b_i down). Hence over the box
  taubar_i in [taubar(1-p), taubar(1+p)],  b_i in [b(1-p), b(1+p)]
the WORST CASE is attained at taubar(1-p), b(1+p), in closed form:
  abar_i^-(v) = (taubar_i(1-p) - b_i(1+p) v_i^2)/m_i.
So the robust barrier  h_p(v) = ln det D^-(v) - ellbar  is exact (not conservative
beyond the box) and costs nothing extra to evaluate.
Enforcing h_p >= 0 certifies h(v;theta) >= 0 for EVERY theta in the box.
"""
import numpy as np
from platforms import hexarotor


def perturb_params(P, p, worst=True, rng=None):
    """worst=True -> worst-case corner (taubar low, b high). Else random draw."""
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    if worst:
        Q["tau_bar"] = P["tau_bar"] * (1 - p)
        Q["b"] = P["b"] * (1 + p)
    else:
        Q["tau_bar"] = P["tau_bar"] * (1 + rng.uniform(-p, p, size=P["n"]))
        Q["b"] = P["b"] * (1 + rng.uniform(-p, p, size=P["n"]))
    Q["vsat"] = float(np.sqrt(np.min(Q["tau_bar"] / Q["b"])))
    Q["vstar"] = float(np.sqrt(np.min(Q["tau_bar"] / (3 * Q["b"]))))
    return Q


def verify_worst_case(P, p=0.2, ntrial=4000, seed=0):
    """Empirically confirm the corner is the minimizer of ln det D over the box."""
    from daam import h_value
    rng = np.random.default_rng(seed)
    Pw = perturb_params(P, p, worst=True)
    bad = 0; margin = np.inf
    for _ in range(ntrial):
        v = rng.uniform(0.3, 0.85, size=P["n"]) * P["vsat"]
        hw = h_value(v, Pw, 0.0)
        Pr = perturb_params(P, p, worst=False, rng=rng)
        hr = h_value(v, Pr, 0.0)
        margin = min(margin, hr - hw)
        if hr < hw - 1e-9: bad += 1
    return bad, margin


if __name__ == "__main__":
    P = hexarotor()
    print("=== worst-case corner check (is the closed-form corner really the minimum?) ===")
    for p in [0.10, 0.20, 0.30]:
        bad, marg = verify_worst_case(P, p)
        print("  p=%3.0f%%: random draws below the corner: %d/4000 ; min margin %+.4f  -> %s"
              % (p*100, bad, marg, "CORNER IS WORST CASE" if bad == 0 else "FAILED"))
