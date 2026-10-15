"""audit2.py -- numerical sanity checks written while reviewing the theory (not needed for the paper).

A: a zero column of A empties the floor window;  B: weighted vs plain leverage
for different motors;  C: the null-space projector;  D: norms in theta;
E: the wrench-rate budget identity.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np
from platforms import hexarotor
from daam import D_matrix, grad_h, h_value, psi, a_bar
np.set_printoptions(precision=5, suppress=True)
P = hexarotor()

print("AUDIT A: does rank(A)=m imply every column nonzero? and what breaks?")
A0 = P["A"].copy(); A0[:,5] = 0.0
print("  rank(A with zero col 5) =", np.linalg.matrix_rank(A0), " (=m, still full row rank)")
G = np.linalg.inv(A0@A0.T)
lev = np.array([A0[:,k]@G@A0[:,k] for k in range(6)])
print("  leverages:", lev, " -> lambda_5=0 => Delta_5=-ln(1-0)=0")
print("  ell_dagger = L_max - min_k Delta_k = L_max - 0 = L_max")
print("  => admissible window (ell_dagger, L_max] is EMPTY. FLOOR CONDITION UNSATISFIABLE.")
print("  CONCLUSION: A_i != 0 for all i must be an explicit assumption. FATAL if omitted.\n")

print("AUDIT B: heterogeneous motors -- is the plain leverage formula still valid?")
Ph = hexarotor(); rng=np.random.default_rng(0)
Ph["tau_bar"] = np.array([1.0,1.3,0.8,1.1,0.9,1.2]); Ph["b"]=np.array([1.0,0.9,1.2,1.0,1.1,0.95])
Ph["vstar"] = np.sqrt(Ph["tau_bar"]/(3*Ph["b"])); Ph["vsat"]=np.sqrt(Ph["tau_bar"]/Ph["b"])
vst = np.sqrt(Ph["tau_bar"]/(3*Ph["b"]))
ps  = psi(vst, Ph)                       # per-rotor psi*, now DIFFERENT
A = Ph["A"]
Dmax = 4*(A*ps[None,:])@A.T
Lmax = np.linalg.slogdet(Dmax)[1]
print("  psi* per rotor (heterogeneous):", ps)
for k in [0,1,2]:
    pv = ps.copy(); pv[k]=0.0
    Lk = np.linalg.slogdet(4*(A*pv[None,:])@A.T)[1]
    gap_true = Lmax - Lk
    lam_plain = A[:,k]@np.linalg.solve(A@A.T, A[:,k])          # UNWEIGHTED leverage
    lam_wt    = ps[k]*(A[:,k]@np.linalg.solve(Dmax/4, A[:,k])) # WEIGHTED leverage
    print("   k=%d: true gap=%.6f | -ln(1-lam_plain)=%.6f  | -ln(1-lam_weighted)=%.6f"
          %(k,gap_true,-np.log(1-lam_plain),-np.log(1-lam_wt)))
print("  => plain leverage is WRONG for heterogeneous motors; the WEIGHTED one is exact.\n")

print("AUDIT C: projector Pi = I - W J^T D^-1 J")
v = np.full(6,P["vstar"])*0.9
J = 2*P["A"]*np.abs(v)[None,:]; W=np.diag(a_bar(v,P)**2); D=J@W@J.T
Pi = np.eye(6) - W@J.T@np.linalg.solve(D,J)
print("  ||Pi^2-Pi|| = %.2e (idempotent)"%np.linalg.norm(Pi@Pi-Pi))
print("  ||J Pi||    = %.2e (range = ker J)"%np.linalg.norm(J@Pi))
Winv=np.linalg.inv(W); x=np.random.default_rng(1).normal(size=4); xi=Pi@np.random.default_rng(2).normal(size=6)
print("  <W J^T x, xi>_{W^-1} = %.2e (W^-1-orthogonal complement)"%((W@J.T@x)@Winv@xi))

print("\nAUDIT D: theta norms -- is gamma bounded by the 1-norm or 2-norm?")
gh = grad_h(v,P); ab=a_bar(v,P)
print("  ||W^1/2 grad h||_1 = %.4f   ||W^1/2 grad h||_2 = %.4f  ratio=%.3f (<=sqrt(n)=%.3f)"
      %(np.sum(ab*np.abs(gh)), np.linalg.norm(ab*gh), np.sum(ab*np.abs(gh))/np.linalg.norm(ab*gh), np.sqrt(6)))
print("  => gamma <= ||.||_1, so theta:=gamma/||.||_2 in [0,sqrt(n)] is CORRECT but mixes norms.")

print("\nAUDIT E: demand budget dimensional check  ||W^-1/2 a_0||^2 = mu^T D^-1 mu ?")
mu=np.array([0.3,0.1,-0.2,0.05]); a0=W@J.T@np.linalg.solve(D,mu)
print("  ||W^-1/2 a0||^2 = %.8f ; mu^T D^-1 mu = %.8f  -> identity %s"
      %(a0@np.linalg.inv(W)@a0, mu@np.linalg.solve(D,mu),
        "HOLDS" if abs(a0@np.linalg.inv(W)@a0 - mu@np.linalg.solve(D,mu))<1e-10 else "FAILS"))
print("  => budget ||mu|| <= (1-eps) sqrt(lambda_min) is dimensionally correct (sqrt IS needed).")
