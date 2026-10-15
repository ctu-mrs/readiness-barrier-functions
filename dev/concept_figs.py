"""
concept_figs.py — data for the two conceptual figures.

(a) fig_ellipsoid: the achievable wrench-rate set {x : x^T D^-1 x <= 1}, projected
    onto the (tau_x,tau_y) plane, at three configurations. Semi-axes are
    sqrt(eig(.)) of the corresponding 2x2 block of D, so the drawn ellipses are
    REAL, not schematic. Volume of the m-dim set is omega_m*sqrt(det D), hence
    ln det D = 2 ln(vol/omega_m): the barrier floors a log-volume.

(b) fig_jump: the readiness level along a straight path between two sign-image
    strata (v1: +v* -> -v*), showing that the strata are EQUAL in authority and
    the collapse of depth Delta = ln(n/(n-m)) lies in the valley between them.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import numpy as np, os
from platforms import hexarotor
from daam import D_matrix
from gate import weighted_leverage
from project_paths import figure_data_path, figure_path

HERE = os.path.dirname(os.path.abspath(__file__))
IDX  = (1, 2)          # (tau_x, tau_y) block of the hexarotor wrench


def ellipse_params(P, v, idx=IDX):
    """Semi-axes (a,b) and rotation angle [deg] of the projected ellipse."""
    D = D_matrix(v, P)
    B = D[np.ix_(idx, idx)]                     # projection == block of D
    lam, U = np.linalg.eigh(B)
    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]
    ang = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))
    return float(np.sqrt(lam[0])), float(np.sqrt(lam[1])), ang


def main():
    figure_data_path(".")
    P = hexarotor()
    vs   = P["vstar"]
    cfgs = [("sweet",   np.full(P["n"], vs),                       "all rotors at $v^\\star$"),
            ("stopped", np.concatenate([[0.0], np.full(P["n"]-1, vs)]), "rotor 1 stopped"),
            ("sat",     np.full(P["n"], 0.97*P["vsat"]),           "all rotors near saturation")]
    rows = []
    print("%-9s %9s %9s %8s   %10s" % ("config", "semi-a", "semi-b", "ang[deg]", "ln det D"))
    for tag, v, _ in cfgs:
        a, b, ang = ellipse_params(P, v)
        L = float(np.linalg.slogdet(D_matrix(v, P))[1])
        rows.append((tag, a, b, ang, L))
        print("%-9s %9.4f %9.4f %8.2f   %10.4f" % (tag, a, b, ang, L))
    # normalize so the sweet-spot major axis is 1 (drawing units)
    s = 1.0 / rows[0][1]
    with open(figure_data_path("ellipse.csv"), "w") as f:
        f.write("config,semi_a,semi_b,angle_deg,logdetD,semi_a_scaled,semi_b_scaled\n")
        for tag, a, b, ang, L in rows:
            f.write("%s,%.6f,%.6f,%.3f,%.6f,%.6f,%.6f\n" % (tag, a, b, ang, L, a*s, b*s))
    print("wrote ellipse.csv (scale factor %.4f -> sweet-spot major axis = 1)" % s)

    # ---- (b) readiness along the sign flip of rotor 1 ----
    ts = np.linspace(-1.0, 1.0, 401)
    Ls = []
    for t in ts:
        v = np.full(P["n"], vs); v[0] = t*vs
        Dm = D_matrix(v, P)
        Ls.append(np.linalg.slogdet(Dm)[1] if np.linalg.det(Dm) > 0 else np.nan)
    Ls = np.array(Ls)
    _, Delta, _ = weighted_leverage(P)
    Lmax, Lmid = np.nanmax(Ls), Ls[len(ts)//2]
    print("\nstrata equal: L(+v*)=%.6f  L(-v*)=%.6f  diff=%.2e" % (Ls[-1], Ls[0], abs(Ls[-1]-Ls[0])))
    print("valley depth = %.6f  == Delta = ln(n/(n-m)) = %.6f" % (Lmax-Lmid, Delta[0]))
    np.savetxt(figure_data_path("jump_profile.csv"),
               np.column_stack([ts, Ls]), delimiter=",",
               header="v1_over_vstar,logdetD", comments="")
    print("wrote jump_profile.csv")


if __name__ == "__main__":
    main()


def render_pdfs():
    """matplotlib fallbacks for the two conceptual figures (PDF backend)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    P = hexarotor(); vs = P["vstar"]
    figure_path(".")
    # ---- ellipsoid ----
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    specs = [(np.full(P["n"], vs), "tab:blue", "sweet spot"),
             (np.concatenate([[0.0], np.full(P["n"]-1, vs)]), "tab:orange", "rotor 1 stopped"),
             (np.full(P["n"], 0.97*P["vsat"]), "tab:red", "near saturation")]
    s = None
    for v, c, lab in specs:
        a, b, ang = ellipse_params(P, v)
        if s is None: s = 1.0/a
        ax.add_patch(Ellipse((0, 0), 2*a*s, 2*b*s, angle=ang, facecolor=c,
                             alpha=0.30, edgecolor=c, lw=1.4, label=lab))
    ax.arrow(0, 0, 0.36, 0.62, width=0.012, color="k", length_includes_head=True)
    ax.text(0.40, 0.66, r"$\dot w_{\rm des}$", fontsize=8)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.1, 1.25); ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\dot\tau_x$", fontsize=8); ax.set_ylabel(r"$\dot\tau_y$", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="lower left")
    plt.tight_layout(); plt.savefig(figure_path("fig_ellipsoid.pdf")); plt.close()
    # ---- concept (right panel only in the fallback) ----
    d = np.genfromtxt(figure_data_path("jump_profile.csv"), delimiter=",", skip_header=1)
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(d[:, 0], d[:, 1], color="tab:blue", lw=1.6)
    ax.annotate("", xy=(0, -10.1278), xytext=(0, -11.2264),
                arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(0.05, -10.68, r"$\ln\frac{n}{n-m}$", fontsize=8)
    ax.axhline(-11.06, color="tab:blue", ls="--", lw=1.0)
    ax.text(-1.0, -10.99, r"$\bar\ell$", color="tab:blue", fontsize=8)
    ax.set_yticks([-11.2264, -10.1278]); ax.set_yticklabels([r"$\ell^\dagger$", r"$L^{\max}$"], fontsize=8)
    ax.set_xlabel(r"$v_1/v^\star$", fontsize=8); ax.set_ylabel(r"$L(v)$", fontsize=8)
    ax.set_ylim(-11.55, -9.85); ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(figure_path("fig_concept.pdf")); plt.close()
    print("wrote fig_ellipsoid.pdf, fig_concept.pdf")
