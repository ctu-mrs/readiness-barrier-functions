"""
make_paper_data.py -- rebuild every plotting table read by the manuscript figures.

The TikZ sources in figures/tikz/ read whitespace-separated tables from data/.
This script regenerates all of them, either from the experiment outputs
(sweep.json, delay.json, ff.json, certmap.dat, h_all.dat, rate_all.dat) or,
for the two conceptual figures, directly from the model.

    Fig. 1  fig_ellipse.tex       ell_sweet.dat, ell_stopped.dat, ell_sat.dat   (model)
    Fig. 2  fig_nested.tex        n2_C.dat, n2_Keps.dat                         (model)
    Fig. 3  fig_sweep.tex         sweep.dat                                     (exp_sweep.py)
    Fig. 4  fig_lpf2.tex          rate_all.dat, h_all.dat                       (exp_fig5.py)
    Fig. 5  fig_applic.tex        certmap.dat, certfront.dat                    (exp_certmap.py)
                                  ff.dat                                        (exp_ff.py)
    Fig. 6  fig_delayregimes.tex  delay.dat, delay_marg.dat, delay_eta.dat      (exp_delay.py)

A few numbers are typed directly into the TikZ files (the certified delay
ceiling, the delay at which the plain floor fails, the delay-free margin and
the forward-flight certifiability limit). They are printed at the end so they
can be checked against the figure sources.

Usage
    python make_paper_data.py                      # read results/generated, write results/generated/paper_data
    python make_paper_data.py --results reference  # build them from the supplied reference results instead
    python make_paper_data.py --compare data       # also report the difference to the tables in data/
    python make_paper_data.py --out data           # overwrite the tables used by the figures
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from contourpy import contour_generator

from daam import D_matrix
from platforms import hexarotor
from project_paths import ROOT

METHODS = ["energy", "greedy", "lowpass", "cbf"]
FF_SPEEDS = [0, 3, 6, 9, 12, 15]


def write_table(path, header, rows, fmt):
    """Whitespace-separated table with a one-line header, as pgfplots expects."""
    with open(path, "w", newline="\n") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(fmt % tuple(r) + "\n")


def write_polygons(path, polygons):
    """Closed polygons separated by blank lines (one pgfplots fill per component)."""
    with open(path, "w", newline="\n") as f:
        for poly in polygons:
            for x, y in poly:
                f.write("%.4f %.4f\n" % (x, y))
            f.write("\n")


def zero_crossing(x, y):
    """Linear interpolation of the last sign change of y from >= 0 to < 0."""
    for k in range(len(y) - 1, 0, -1):
        if y[k - 1] >= 0 > y[k]:
            return x[k - 1] + (x[k] - x[k - 1]) * y[k - 1] / (y[k - 1] - y[k])
    return np.nan


# ---------------------------------------------------------------- Fig. 1
def ellipses(out):
    """Achievable moment-rate ellipse {x : x^T B^-1 x <= 1}, B = (tau_x, tau_y) block of D.

    Three hexarotor configurations: every rotor at the sweet spot, rotor 1
    stopped, and every rotor at 97 % of saturation.  Points are
    U diag(sqrt(lambda)) [cos t, sin t] with eigenvalues in descending order.
    """
    P = hexarotor()
    vs = P["vstar"]
    configs = {"sweet": np.full(P["n"], vs),
               "stopped": np.concatenate([[0.0], np.full(P["n"] - 1, vs)]),
               "sat": np.full(P["n"], 0.97 * P["vsat"])}
    t = np.linspace(0.0, 2 * np.pi, 241)
    for name, v in configs.items():
        B = D_matrix(v, P)[1:3, 1:3]
        lam, U = np.linalg.eigh(B)
        order = np.argsort(lam)[::-1]
        pts = (U[:, order] * np.sqrt(lam[order])) @ np.vstack([np.cos(t), np.sin(t)])
        write_table(out / ("ell_%s.dat" % name), "x y", pts.T, "%.5f %.5f")


# ---------------------------------------------------------------- Fig. 2
# Two-rotor illustration with a square allocation map (A = I, so D = 4 diag(psi_1, psi_2)),
# in units where tau_bar = b = m = 1, i.e. psi(v) = v^2 (1 - v^2)^2 with v = spin / v_sat.
N2_FLOOR_DEPTH = 1.3   # floor sits 1.3 nats below L_max
N2_EPS = 0.02          # K_eps = {psi_i >= eps for every rotor}


def _psi_unit(v):
    return v ** 2 * (1.0 - v ** 2) ** 2


def certified_set_2d(out, step=0.0025):
    vstar = 1.0 / np.sqrt(3.0)
    L_max = 2.0 * np.log(_psi_unit(vstar))
    floor = L_max - N2_FLOOR_DEPTH

    # C = {ln psi_1 + ln psi_2 >= floor}: one closed curve in each quadrant.
    # Contour the first quadrant and mirror it; the curve does not touch the axes.
    g = np.arange(step, 1.0, step)
    X, Y = np.meshgrid(g, g)
    Z = np.log(_psi_unit(X)) + np.log(_psi_unit(Y))
    lines = contour_generator(X, Y, Z).lines(floor)
    curve = max(lines, key=len)

    # K_eps is a box in each quadrant: psi(v) >= eps between the two roots of psi(v) = eps.
    vv = np.linspace(0.0, 1.0, 200001)
    inside = vv[_psi_unit(vv) >= N2_EPS]
    lo, hi = inside.min(), inside.max()
    box = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi], [lo, lo]])

    signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    write_polygons(out / "n2_C.dat", [curve * s for s in signs])
    write_polygons(out / "n2_Keps.dat", [box * s for s in signs])


# ---------------------------------------------------------------- Fig. 3
def sweep_table(res, out):
    sw = json.load(open(res / "sweep.json"))
    keys = ["hmin", "TV", "werr", "hmean", "tbelow"]
    header = "thrust ellbar " + " ".join("%s_%s" % (m, k) for m in METHODS for k in keys)
    rows = []
    for th, (eb, _, _) in sorted(sw["floors"].items(), key=lambda kv: float(kv[0])):
        if eb is None:                       # collective not certifiable: no runs, grey band in the plot
            continue
        runs = [sw["runs"]["%.1f_%s" % (float(th), m)] for m in METHODS]
        rows.append([float(th), eb] + [r[k] for r in runs for k in keys])
    write_table(out / "sweep.dat", header, rows, " ".join(["%.6g"] * len(rows[0])))


# ---------------------------------------------------------------- Fig. 4
def time_histories(res, out):
    # exp_fig5.py already writes these in the plotted format.
    for name in ["rate_all.dat", "h_all.dat"]:
        shutil.copyfile(res / name, out / name)


# ---------------------------------------------------------------- Fig. 5
def certifiability(res, out):
    shutil.copyfile(res / "certmap.dat", out / "certmap.dat")
    cm = np.genfromtxt(res / "certmap.dat", skip_header=1)
    rows = []
    for th in np.unique(cm[:, 0]):
        sel = cm[cm[:, 0] == th]
        sel = sel[np.argsort(sel[:, 1])]
        rows.append([th, zero_crossing(sel[:, 1], sel[:, 2])])   # largest certifiable moment amplitude
    write_table(out / "certfront.dat", "thrust amp", rows, "%.4f %.4f")


def forward_flight(res, out):
    """Hexarotor true-readiness margin vs airspeed, and where L^op drops below the floor."""
    hx = json.load(open(res / "ff.json"))["hex"]
    rows = []
    for V in FF_SPEEDS:
        s = hx["speeds"][str(V)]
        rows.append([V] + [s[m]["htrue_min"] for m in ["energy", "greedy", "cbf", "cbf_aware"]])
    write_table(out / "ff.dat", "V energy greedy cbf aware", rows, "%d %.4f %.4f %.4f %.4f")
    window = [hx["speeds"][str(V)]["Lop_s"] - hx["eb"] for V in FF_SPEEDS]
    return zero_crossing(FF_SPEEDS, window)


# ---------------------------------------------------------------- Fig. 6
def delay_tables(res, out):
    d = json.load(open(res / "delay.json"))
    alpha = 5.0                                           # as in exp_delay.py
    K, V, Hbar = d["K"], d["V"], d["Hbar"]
    eta = lambda td: (alpha * Hbar + K * V * td) * td + K * V * td / alpha

    plain = sorted((r for r in d["rows"] if r["kind"] == "plain" and r["init"] == "hold" and r["td"] > 0),
                   key=lambda r: r["td"])
    marg = sorted((r for r in d["rows"] if r["kind"] == "marg"), key=lambda r: r["td"])
    # the plot's x axis is tau_d in units of 1e-3 T0 (labelled "ms" in the column names)
    write_table(out / "delay.dat", "tau_d_ms minh_plain eta_bound",
                [[r["td"] * 1e3, r["hmin"], -r["eta"]] for r in plain], "%g %.6f %.6f")
    write_table(out / "delay_marg.dat", "tau_d_ms minh_margined",
                [[r["td"] * 1e3, r["hmin"]] for r in marg], "%.5g %.6f")
    td = np.geomspace(0.2, 40.0, 140)
    write_table(out / "delay_eta.dat", "tau_d_ms eta",
                [[x, -eta(x * 1e-3)] for x in td], "%.5g %.6f")

    delay_free = next(r["hmin"] for r in d["rows"] if r["kind"] == "plain" and r["td"] == 0)
    fail = zero_crossing([r["td"] * 1e3 for r in plain], [r["hmin"] for r in plain])
    return dict(tdstar=d["tdstar"] * 1e3, fail=fail, delay_free=delay_free)


# ---------------------------------------------------------------- comparison
def _read_numeric(path):
    with open(path) as f:
        first = f.readline().split()
    skip = 0 if all(_is_number(x) for x in first) else 1
    return np.genfromtxt(path, skip_header=skip)


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _curve_distance(a, b):
    """Largest distance from a point of a to the polygon edges of b, and vice versa (Hausdorff)."""
    def one_way(p, q):
        seg_a, seg_b = q[:-1], q[1:]
        d = seg_b - seg_a
        L2 = np.maximum((d ** 2).sum(1), 1e-30)
        worst = 0.0
        for x in p:
            s = np.clip(((x - seg_a) * d).sum(1) / L2, 0.0, 1.0)
            worst = max(worst, np.min(np.linalg.norm(seg_a + s[:, None] * d - x, axis=1)))
        return worst
    return max(one_way(a, b), one_way(b, a))


def _split_polygons(path):
    polys, cur = [], []
    for line in open(path):
        if line.strip():
            cur.append([float(x) for x in line.split()])
        elif cur:
            polys.append(np.array(cur)); cur = []
    if cur:
        polys.append(np.array(cur))
    return polys


def compare(new_dir, old_dir):
    print("\nComparison with %s (largest absolute difference per file):" % old_dir)
    for new in sorted(new_dir.glob("*.dat")):
        old = old_dir / new.name
        if not old.exists():
            print("  %-16s missing in %s" % (new.name, old_dir))
            continue
        if new.name.startswith("n2_"):
            pa, pb = _split_polygons(new), _split_polygons(old)
            # match components by quadrant (sign of the centroid)
            q = lambda p: tuple(np.sign(p.mean(0)))
            pb = {q(p): p for p in pb}
            dist = max(_curve_distance(p, pb[q(p)]) for p in pa)
            print("  %-16s curve distance %.1e (axis units)" % (new.name, dist))
            continue
        a, b = _read_numeric(new), _read_numeric(old)
        if a.shape != b.shape:
            print("  %-16s shape %s vs %s" % (new.name, a.shape, b.shape))
            continue
        diff = np.abs(a - b)
        i = np.unravel_index(np.nanargmax(diff), diff.shape)
        print("  %-16s %.2e   (row %d, column %d)" % (new.name, diff[i], i[0] + 1, i[1] + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="generated",
                    help="'generated', 'reference', or a directory holding the experiment outputs")
    ap.add_argument("--out", default=None, help="output directory (default: results/generated/paper_data)")
    ap.add_argument("--compare", default=None, help="directory of tables to compare against, e.g. data")
    args = ap.parse_args()

    res = Path(args.results)
    if args.results in ("generated", "reference"):
        res = ROOT / "results" / args.results
    # never write next to the reference results, they are kept read-only
    out = Path(args.out) if args.out else ROOT / "results" / "generated" / "paper_data"
    out.mkdir(parents=True, exist_ok=True)

    ellipses(out)
    certified_set_2d(out)
    sweep_table(res, out)
    time_histories(res, out)
    certifiability(res, out)
    v_ff = forward_flight(res, out)
    dl = delay_tables(res, out)
    print("Read experiment outputs from", res)
    print("Wrote the figure tables to  ", out)

    print("\nValues typed into the TikZ sources (check that they still match):")
    print("  fig_delayregimes.tex  tau_d^*            = %.4f   (1e-3 T0)" % dl["tdstar"])
    print("  fig_delayregimes.tex  plain floor fails  = %.1f     (1e-3 T0, linear interpolation)" % dl["fail"])
    print("  fig_delayregimes.tex  delay-free min h   = %.4f" % dl["delay_free"])
    print("  fig_applic.tex        L^op < floor from  = %.2f    m/s" % v_ff)

    if args.compare:
        compare(out, Path(args.compare))


if __name__ == "__main__":
    main()
