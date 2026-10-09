"""verify.py — Phase-1 due-diligence checks (gradient + exact gap law)."""
import numpy as np
from platforms import hexarotor, tilted_octorotor
from daam import grad_h, grad_h_numeric, D_matrix, psi

# ---- 1) analytic vs numerical gradient at random regular points ----
print("[1] Gradient check (analytic vs central-difference):")
for P in (hexarotor(), tilted_octorotor()):
    rng = np.random.default_rng(1)
    maxerr = 0.0
    for _ in range(50):
        v = rng.uniform(-0.9 * P["vsat"], 0.9 * P["vsat"], size=P["n"])
        ga, gn = grad_h(v, P), grad_h_numeric(v, P)
        maxerr = max(maxerr, np.max(np.abs(ga - gn)))
    print("   %-16s max|analytic-numeric| = %.3e  (PASS<1e-4: %s)"
          % (P["name"], maxerr, maxerr < 1e-4))

# ---- 2) exact gap law:  gap = -ln(1 - lambda_k),  lambda_k = leverage score ----
print("\n[2] Exact single-dropout gap law  gap = ln(n/(n-m)) for symmetric platforms:")
for P in (hexarotor(), tilted_octorotor()):
    A = P["A"]
    G = A @ A.T
    lev = np.array([A[:, k] @ np.linalg.solve(G, A[:, k]) for k in range(P["n"])])
    gap_pred_perrotor = -np.log(1 - lev)
    gap_symmetric = np.log(P["n"] / (P["n"] - P["m"]))
    print("   %-16s leverage scores in [%.4f,%.4f] (mean m/n=%.4f)"
          % (P["name"], lev.min(), lev.max(), P["m"] / P["n"]))
    print("       gap per-rotor in [%.4f,%.4f];  ln(n/(n-m))=%.6f"
          % (gap_pred_perrotor.min(), gap_pred_perrotor.max(), gap_symmetric))

# ---- 3) reconcile absolute logdet scale (self-consistency of h_max) ----
print("\n[3] h_max sanity (all-sweet-spot):")
for P in (hexarotor(), tilted_octorotor()):
    v = np.full(P["n"], P["vstar"])
    D = D_matrix(v, P)
    s, ld = np.linalg.slogdet(D)
    # decompose: logdet(4 psi* A A^T) = m ln(4 psi*) + logdet(A A^T)
    ps = psi(v, P)[0]
    check = P["m"] * np.log(4 * ps) + np.linalg.slogdet(P["A"] @ P["A"].T)[1]
    print("   %-16s logdet D=%.6f  (decomp check=%.6f, match:%s)"
          % (P["name"], ld, check, abs(ld - check) < 1e-9))
