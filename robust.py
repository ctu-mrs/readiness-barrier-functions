"""
robust.py -- parameter boxes for the robust readiness barrier.

D(v; theta) = 4 A diag(psi) A^T is monotone (in the Loewner order) in every
psi_i = v_i^2 a_bar_i^2, and psi_i grows with tau_bar_i and shrinks with b_i
(while a_bar_i >= 0). Over the box
    tau_bar_i in [tau_bar (1 - p), tau_bar (1 + p)],   b_i in [b (1 - p), b (1 + p)]
the smallest D is therefore reached at the corner tau_bar (1 - p), b (1 + p):
    a_bar_i^-(v) = (tau_bar_i (1 - p) - b_i (1 + p) v_i^2) / m_i.
The robust barrier h_p(v) = ln det D^-(v) - ell_bar costs the same as the
nominal one, and h_p >= 0 implies h(v; theta) >= 0 for every theta in the box.

Running the file checks the corner claim against random plants.
"""
import numpy as np
from platforms import hexarotor


def perturb_params(P, p, worst=True, rng=None):
    """Copy of P with perturbed tau_bar and b.

    worst=True gives the worst-case corner (tau_bar low, b high); otherwise each
    motor's parameters are drawn uniformly in +-p using rng.
    """
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
    """Count random plants whose ln det D falls below the corner's at random speeds (should be 0)."""
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
