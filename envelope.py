"""
envelope.py — operational-envelope single-component test and applicability map.
Shows that hovering flight stays in one connected component of {h>=ell_bar}
(no stratum jump), and maps where component-crossing is forced.
"""
import numpy as np
from platforms import hexarotor
from gate import alloc_on_fiber, ell_dagger, h_max_point


def run():
    P = hexarotor()
    phistar = P["vstar"] * abs(P["vstar"])
    fz_nom = P["n"] * phistar                 # collective at all-sweet-spot
    tau_ref = 0.25 * phistar                  # ~ one-rotor arm torque at sweet-spot
    v_hi, hmx = h_max_point(P)
    ldag, _ = ell_dagger(P)

    # (a) operational torque sweep around hover (thrust = nominal)
    grid = np.linspace(-1, 1, 9)
    npos = ntot = 0
    worstH = np.inf; minspin = 9.0
    for gx in grid:
        for gy in grid:
            w = np.array([fz_nom, gx * tau_ref, gy * tau_ref, 0.0])
            v, Hv, ok = alloc_on_fiber(w, P, v_hi.copy())
            if not ok:
                continue
            ntot += 1
            npos += int(np.all(v > 0))
            minspin = min(minspin, v.min())
            worstH = min(worstH, Hv)
    print("Operational sweep (%d pts):" % ntot)
    print("  fraction in all-positive component : %.3f" % (npos / ntot))
    print("  min rotor spin over sweep          : %.4f (>0 => no zero-crossing)" % minspin)
    print("  worst-case h over envelope         : %.4f" % worstH)
    print("  margin of worst-case h above saddle: %.4f" % (worstH - ldag))
    print("  => valid floor window (ell_dag, worstH) width = %.4f" % (worstH - ldag))

    # (b) applicability map: thrust bias x torque authority
    print("\nApplicability map (OK=single component / JMP=crossing forced / INF=infeasible):")
    print("            torque x0.5   x1    x2    x3")
    for tf in [0.25, 0.5, 0.75, 1.0, 1.25]:
        row = []
        for tm in [0.5, 1.0, 2.0, 3.0]:
            allpos = True; feas = True
            for gx in [-1, -0.5, 0.5, 1]:
                for gy in [-1, -0.5, 0.5, 1]:
                    w = np.array([tf * fz_nom, gx * tm * tau_ref, gy * tm * tau_ref, 0.0])
                    v, _, ok = alloc_on_fiber(w, P, v_hi.copy())
                    if not ok:
                        feas = False; continue
                    if np.any(v <= 0):
                        allpos = False
            row.append("OK " if (allpos and feas) else ("JMP" if feas else "INF"))
        print("  thrust %.2f:  " % tf, row)


if __name__ == "__main__":
    run()
