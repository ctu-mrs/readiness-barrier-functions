"""Generate readiness-profile and applicability diagnostic plots."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from platforms import hexarotor
from daam import h_value, psi
from gate import h_max_point, ell_dagger

from project_paths import figure_data_path, figure_path
P = hexarotor()
vstar = P["vstar"]
v_hi, hmx = h_max_point(P)
ldag, _ = ell_dagger(P)

# ---- Fig 1: h along a single-rotor sign flip (the jump mechanism + gap) ----
ts = np.linspace(-1, 1, 401)
Hs = []
for t in ts:
    v = np.full(P["n"], vstar); v[0] = t * vstar    # flip rotor 1 through 0
    Hs.append(h_value(v, P))
Hs = np.array(Hs)
plt.figure(figsize=(3.4, 2.5))
plt.plot(ts, Hs, "b-", lw=1.6)
plt.axhline(hmx, color="g", ls="--", lw=0.9, label=r"$h_{\max}$")
plt.axhline(ldag, color="r", ls="--", lw=0.9, label=r"$\ell^\dagger$")
plt.annotate("", xy=(0, hmx), xytext=(0, ldag),
             arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
plt.text(0.05, 0.5 * (hmx + ldag), r"$\ln\frac{n}{n-m}$", fontsize=9)
plt.xlabel(r"$v_1/v^\star$  (sign flip through $0$)")
plt.ylabel(r"$\ln\det D$")
plt.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.tight_layout()
np.savetxt(figure_data_path("flip_curve.csv"),np.column_stack([ts,Hs]),delimiter=",",header="v1_over_vstar,logdetD",comments="")
plt.savefig(figure_path("fig_flip.pdf"))
plt.close()

# ---- Fig 2: applicability map (thrust bias x torque authority) ----
from gate import alloc_on_fiber
phistar = vstar * abs(vstar); fz_nom = P["n"] * phistar; tau_ref = 0.25 * phistar
tf = [0.25, 0.5, 0.75, 1.0, 1.25]
tm = [0.5, 1.0, 2.0, 3.0]
M = np.zeros((len(tf), len(tm)))
for i, a in enumerate(tf):
    for j, b in enumerate(tm):
        allpos = True; feas = True
        for gx in [-1, -0.5, 0.5, 1]:
            for gy in [-1, -0.5, 0.5, 1]:
                w = np.array([a * fz_nom, gx * b * tau_ref, gy * b * tau_ref, 0.0])
                v, _, ok = alloc_on_fiber(w, P, v_hi.copy())
                if not ok: feas = False
                elif np.any(v <= 0): allpos = False
        M[i, j] = 2 if (allpos and feas) else (1 if feas else 0)
plt.figure(figsize=(3.2, 2.5))
plt.imshow(M, aspect="auto", origin="lower", cmap="RdYlGn", vmin=0, vmax=2)
plt.xticks(range(len(tm)), [r"$%.1f$" % x for x in tm])
plt.yticks(range(len(tf)), [r"$%.2f$" % x for x in tf])
plt.xlabel("torque authority (× ref)")
plt.ylabel("thrust bias (× nominal)")
for i in range(len(tf)):
    for j in range(len(tm)):
        plt.text(j, i, {2: "OK", 1: "JMP", 0: "INF"}[M[i, j]],
                 ha="center", va="center", fontsize=7)
plt.tight_layout()
np.savetxt(figure_data_path("applicability.csv"),M,delimiter=",",
    header="cols=torque x0.5,1,2,3 ; rows=thrust 0.25,0.5,0.75,1.0,1.25 ; 2=OK 1=JMP 0=INF",comments="")
plt.savefig(figure_path("fig_applicability.pdf"))
plt.close()
print("figures written to", figure_path("."))
