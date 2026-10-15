"""audit3.py -- checks behind the robust barrier (not needed for the paper).

F: enforcing the robust metric with nominal drift leaves an uncontrolled
drift mismatch, hence the worst-case drift term;  G: a_bar^- >= 0 only on a
smaller speed box.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np
from platforms import hexarotor
from daam import grad_h, h_value, a_bar
from robust import perturb_params
P=hexarotor()

print("AUDIT F: robust barrier -- does it certify INVARIANCE or only the METRIC?")
print("  CBF condition enforced by controller (nominal drift d(v;theta_hat)):")
print("      grad h_p^T ( d(v;theta_hat) + M^-1 tau ) >= -alpha(h_p)")
print("  TRUE closed loop evolves with d(v;theta):")
print("      hdot_p = grad h_p^T ( d(v;theta) + M^-1 tau )")
print("  mismatch term = grad h_p^T ( d(v;theta) - d(v;theta_hat) )   <-- NOT controlled\n")
v=np.full(6,0.55); p=0.30
Pw=perturb_params(P,p,worst=True)
gh=grad_h(v,Pw)
phi=lambda x: x*np.abs(x)
d_nom=-(P["b"]*phi(v))/P["mass"]
worst=0.0
rng=np.random.default_rng(0)
for _ in range(20000):
    Pt=perturb_params(P,p,worst=False,rng=rng)
    d_true=-(Pt["b"]*phi(v))/Pt["mass"]
    worst=min(worst, gh@(d_true-d_nom))
print("  empirical min of mismatch over 20000 draws: %.5f"%worst)
# closed form: linear in each b_i -> extremum at an endpoint
mism=0.0
for i in range(6):
    lo=-gh[i]*P["b"][i]*(1-p)*phi(v)[i]/P["mass"][i]
    hi=-gh[i]*P["b"][i]*(1+p)*phi(v)[i]/P["mass"][i]
    nom=-gh[i]*P["b"][i]*phi(v)[i]/P["mass"][i]
    mism+=min(lo,hi)-nom
print("  closed-form worst-case mismatch      : %.5f"%mism)
print("  => NONZERO. Enforcing with nominal drift does NOT certify robust invariance.")
print("     FIX: tighten the CBF row by the closed-form worst-case drift:")
print("       grad h_p^T M^-1 tau >= -alpha(h_p) - min_{theta in Theta} grad h_p^T d(v;theta)")
print("     which is exact (linear in each b_i => endpoint) and costs nothing.\n")

print("AUDIT G: is a_bar^-(v) >= 0 on the robust set? (psi=v^2 abar^2 monotone only for abar>=0)")
for pp in [0.1,0.2,0.3]:
    Pw=perturb_params(P,pp,worst=True)
    vlim=np.sqrt(P["tau_bar"][0]*(1-pp)/(P["b"][0]*(1+pp)))
    print("   p=%.0f%%: abar^- >= 0 only for |v_i| <= %.4f  (nominal v_sat=%.3f) -> robust box SHRINKS"
          %(pp*100,vlim,P["vsat"]))
print("   => monotonicity domain must be stated: robust set lives in the SHRUNKEN box E^-.")
