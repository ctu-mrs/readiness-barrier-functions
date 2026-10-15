"""fill_paper.py -- write the experiment numbers into the manuscript.

Replaces the @KEY@ placeholders of main.tex (table rows and a few sentences)
with values computed from results/generated/*.json, writes data/ff.dat next to
the manuscript, and fills the placeholders of figures/tikz/fig_applic.tex.
Numbers that were already in the submitted manuscript and re-checked (gap
levels, the 731 / 14.6 rates and the 0.629 / 0.756 s times) are not touched.

This script MODIFIES the manuscript given on the command line; keep a copy.
Once the placeholders have been replaced, running it again changes nothing.

Usage: python fill_paper.py PATH_TO_MAIN_TEX
"""
import json, sys, numpy as np
from pathlib import Path
from project_paths import RESULTS_DIR

if len(sys.argv) != 2:
    raise SystemExit("Usage: python fill_paper.py PATH_TO_MAIN_TEX")

PAPER_TEX = Path(sys.argv[1]).resolve()
PAPER_DIR = PAPER_TEX.parent
APPLIC_FIGURE = PAPER_DIR / "figures" / "tikz" / "fig_applic.tex"
PAPER_DATA_DIR = PAPER_DIR / "data"
if not PAPER_TEX.is_file() or not APPLIC_FIGURE.is_file() or not PAPER_DATA_DIR.is_dir():
    raise SystemExit("The manuscript, applicability TikZ source, and paper data directory must all exist.")
sw = json.load(open(RESULTS_DIR / "sweep.json")); rb = json.load(open(RESULTS_DIR / "robust.json"))
mc = json.load(open(RESULTS_DIR / "mc.json")); ff = json.load(open(RESULTS_DIR / "ff.json")); tm = json.load(open(RESULTS_DIR / "timing.json"))
M = ["energy", "greedy", "lowpass", "cbf"]
tex = PAPER_TEX.read_text()
T = {}
f3 = lambda x: "%+.2f" % x
# ---------------- Table I
rows = []
for th in ["0.7", "0.8", "0.9", "1.0", "1.1", "1.2"]:
    eb = sw["floors"][th][0]; cells = []
    for m in M:
        r = sw["runs"][th + "_" + m]
        h = "$\\mathbf{%+.2f}$" % r["hmin"] if r["hmin"] < 0 else "$%+.2f$" % r["hmin"]
        cells += [h, "%.1f" % r["TV"], "%.4f" % r["werr"]]
    rows.append("    %s & $%.2f$ & %s\\\\" % (th, eb, " & ".join(cells)))
T["SWEEP_ROWS"] = "\n".join(rows)
g = lambda th, m, k: sw["runs"][th + "_" + m][k]
T["G07"] = "%+.2f" % g("0.7", "greedy", "hmin"); T["G08"] = "%+.2f" % g("0.8", "greedy", "hmin")
T["G07TV"] = "%.1f" % g("0.7", "greedy", "TV"); T["C07TV"] = "%.1f" % g("0.7", "cbf", "TV")
T["G07E"] = "%.3f" % g("0.7", "greedy", "werr"); T["C07E"] = "%.4f" % g("0.7", "cbf", "werr")
T["RATIO07"] = "%d" % round(g("0.7", "greedy", "werr") / g("0.7", "cbf", "werr"))
T["C07"] = "%+.2f" % g("0.7", "cbf", "hmin"); T["E07"] = "%+.2f" % g("0.7", "energy", "hmin")
T["L07"] = "%+.2f" % g("0.7", "lowpass", "hmin")
adv = lambda m: max(g(th, m, "hmean") - g(th, "cbf", "hmean") for th in ["0.9", "1.0", "1.1", "1.2"])
T["GADV"] = "%.3f" % adv("greedy"); T["LADV"] = "%.3f" % adv("lowpass")
T["GMEAN07"] = "%+.2f" % g("0.7", "greedy", "hmean"); T["LMEAN07"] = "%+.2f" % g("0.7", "lowpass", "hmean")
T["CMEAN07"] = "%+.2f" % g("0.7", "cbf", "hmean")
# ---------------- Monte Carlo
def stats(key, m):
    L = mc[key]["missions"]; h = np.array([x["res"][m]["hmin"] for x in L]); e = np.array([x["res"][m]["werr"] for x in L]) * 1e3
    ci = lambda z: 1.96 * z.std(ddof=1) / np.sqrt(len(z))
    return np.mean(h < 0), h.mean(), ci(h), e.mean(), ci(e), len(L)
lab = {"energy": "effort-min", "greedy": "greedy DAAM", "lowpass": "low-pass DAAM", "cbf": "Algorithm~\\ref{alg:filter}"}
rows = []
for plat, nm in [("hex", "hexarotor"), ("oct", "octorotor")]:
    ne = len(mc[plat + "_env"]["missions"]); ns = len(mc[plat + "_scarce"]["missions"])
    rows.append("    \\midrule\n    \\multicolumn{7}{l}{\\emph{%s} ($%d$ + $%d$ missions; %d + %d rejected)}\\\\" %
                (nm, ne, ns, mc[plat + "_env"]["rejected"], mc[plat + "_scarce"]["rejected"]))
    for m in M:
        c = []
        for s in ["env", "scarce"]:
            v, hm, hc, em, ec, n = stats(plat + "_" + s, m)
            vv = "$\\mathbf{%.2f}$" % v if v > 0 else "$%.2f$" % v
            c += [vv, "$%+.2f{\\pm}%.2f$" % (hm, hc), "$%.1f{\\pm}%.1f$" % (em, ec)]
        rows.append("    %s & %s\\\\" % (lab[m], " & ".join(c)))
T["MC_ROWS"] = "\n".join(rows)
T["NHEX"] = "$%d$" % len(mc["hex_env"]["missions"]); T["NOCT"] = "$%d$" % len(mc["oct_env"]["missions"])
T["NMCTOT"] = "$%d$" % sum(len(mc[k]["missions"]) for k in mc)
gv = max(stats(k, "greedy")[0] for k in mc if k.endswith("scarce"))
T["GVMAX"] = "$%d\\%%$" % round(100 * gv)
# ---------------- robust
pc = lambda x: "$%d\\%%$" % round(100 * x)
c = rb["campaign"]; a = rb["ablation"]
T["NOMV10"] = pc(c["0.10_cbf-nominal"]["viol"]); T["NOMW10"] = "$%.3f$" % c["0.10_cbf-nominal"]["worst"]
T["NOMV20"] = pc(c["0.20_cbf-nominal"]["viol"]); T["NOMV30"] = pc(c["0.30_cbf-nominal"]["viol"])
T["ENV10"] = pc(min(c["%.2f_energy" % p]["viol"] for p in [.1, .2, .3]))
T["ENV30"] = pc(max(c["%.2f_energy" % p]["viol"] for p in [.1, .2, .3]))
T["ABLV20"] = pc(a["0.20_metric"]["viol"]); T["ABLV30"] = pc(a["0.30_metric"]["viol"])
T["ABLWR"] = "%.1f" % (a["0.30_metric"]["werr"] / a["0.30_full"]["werr"])
rows = []
fl = {0.1: -12.71, 0.2: -14.47, 0.3: -16.39}
for p in [0.1, 0.2, 0.3]:
    rows.append("    \\midrule")
    for i, (k, nm) in enumerate([("energy", "effort-min QP"), ("cbf-nominal", "nominal barrier"), ("cbf-robust", "\\textbf{robust (Thm.~\\ref{thm:robust})}" if p == 0.1 else "\\textbf{robust}")]):
        r = c["%.2f_%s" % (p, k)]; floor = "$%.2f$" % (rb["floors"]["%.2f" % p]["robust"] if k == "cbf-robust" else rb["eb_nom"])
        v = "$\\mathbf{%.2f}$" % r["viol"] if k == "cbf-robust" else "$%.2f$" % r["viol"]
        rows.append("    %s & %s & $%+.3f$ & $%+.3f$ & %s & $%.4f$ & %s\\\\" % ("$%d\\%%$" % round(100 * p) if i == 0 else "", nm, r["mean"], r["worst"], v, r["werr"], floor))
rows.append("    \\midrule\n    \\multicolumn{7}{l}{\\emph{$\\kappa{=}0.9$ ablation: metric bound only vs.\\ full robust filter}}\\\\")
for p in [0.1, 0.2, 0.3]:
    for i, (k, nm) in enumerate([("metric", "metric bound only"), ("full", "\\textbf{robust}")]):
        r = a["%.2f_%s" % (p, k)]
        w = "$\\mathbf{%+.3f}$" % r["worst"] if r["worst"] < 0 else "$%+.3f$" % r["worst"]
        v = "$\\mathbf{%.2f}$" % r["viol"] if (r["viol"] > 0 or k == "full") else "$%.2f$" % r["viol"]
        rows.append("    %s & %s & $%+.3f$ & %s & %s & $%.4f$ & $%.2f$\\\\" % ("$%d\\%%$" % round(100 * p) if i == 0 else "", nm, r["mean"], w, v, r["werr"], rb["floors"]["%.2f" % p]["ablation"]))
T["ROB_ROWS"] = "\n".join(rows)
# ---------------- timing
T["QPH"] = "%d" % round(tm["hex"]["filter_osqp_solve"]["mean"]); T["QPH99"] = "%d" % round(tm["hex"]["filter_osqp_solve"]["p99"])
T["QPO"] = "%d" % round(tm["oct"]["filter_osqp_solve"]["mean"]); T["QPO99"] = "%d" % round(tm["oct"]["filter_osqp_solve"]["p99"])
# ---------------- MC text and FF text (written from numbers)
hs = lambda k, m: stats(k, m)
mn = lambda k, m: min(x["res"][m]["hmin"] for x in mc[k]["missions"])
nv = lambda k, m: sum(x["res"][m]["hmin"] < 0 for x in mc[k]["missions"])
T["MC_TEXT"] = (
    "Over the hexarotor envelope the barrier is active only $%d\\%%$ of the time and the filter tracks like its nominal allocator, yet greedy and low-pass DAAM already violate the floor in "
    "$%d\\%%$ and $%d\\%%$ of missions (effort-min once, by $%.3f$). In the scarce corner they violate in $%d\\%%$ and $%d\\%%$ of missions, as deep as $%.1f$ and $%.1f$, "
    "with mean wrench error $%.0f$ and $%.0f\\times10^{-3}$; the proposed filter never violates (lowest $h_{\\min}=%+.2f$). "
    "The octorotor, with a 6-D task, rarely forces a handover under single-axis reversal: one scarce mission in $%d$ for greedy and low-pass, but that one collapses to $h_{\\min}=%.0f$ and $%.0f$, "
    "while their wrench error stays $%.0f$--$%.0f\\times$ that of the filter throughout. "
    "Effort-min seldom violates in these maneuvers; its failures appear under model error and forward flight below."
    % (round(100 * np.mean([x["res"]["cbf"]["active"] for x in mc["hex_env"]["missions"]])),
       round(100 * hs("hex_env", "greedy")[0]), round(100 * hs("hex_env", "lowpass")[0]), mn("hex_env", "energy"),
       round(100 * hs("hex_scarce", "greedy")[0]), round(100 * hs("hex_scarce", "lowpass")[0]),
       mn("hex_scarce", "greedy"), mn("hex_scarce", "lowpass"),
       hs("hex_scarce", "greedy")[3], hs("hex_scarce", "lowpass")[3], mn("hex_scarce", "cbf"),
       len(mc["oct_scarce"]["missions"]), mn("oct_scarce", "greedy"), mn("oct_scarce", "lowpass"),
       hs("oct_env", "greedy")[3] / hs("oct_env", "cbf")[3], hs("oct_env", "lowpass")[3] / hs("oct_env", "cbf")[3]))
# forward flight figure data + text
Vs = [0, 3, 6, 9, 12, 15]
def ffdat(plat):
    o = ff[plat]
    with open(PAPER_DATA_DIR / "ff.dat", "w") as f:
        f.write("V energy greedy cbf aware\n")
        for V in Vs:
            s_ = o["speeds"][str(V)]
            f.write("%d %.4f %.4f %.4f %.4f\n" % (V, s_["energy"]["htrue_min"], s_["greedy"]["htrue_min"], s_["cbf"]["htrue_min"], s_["cbf_aware"]["htrue_min"]))
    vals = [o["speeds"][str(V)][m]["htrue_min"] for V in Vs for m in ["energy", "greedy", "cbf", "cbf_aware"]]
    w = [(V, o["speeds"][str(V)]["Lop_s"] - o["eb"]) for V in Vs]
    vprev, l0 = max((V, l) for V, l in w if l >= 0); v1, l1 = min((V, l) for V, l in w if l < 0)
    vc = vprev + (v1 - vprev) * l0 / (l0 - l1)
    return vals, vc
vals, vc = ffdat("hex")
T["FFYMIN"] = "%.1f" % (np.floor(10 * (min(vals) - 0.1)) / 10); T["FFYMAX"] = "%.1f" % (np.ceil(10 * (max(vals) + 0.15)) / 10)
T["FFNC"] = "%.2f" % vc; T["FFNCLAB"] = "%.1f" % ((vc + 15.5) / 2)
oc = ff["oct"]["speeds"]; hx = ff["hex"]["speeds"]
woc = [(V, oc[str(V)]["Lop_s"] - ff["oct"]["eb"]) for V in Vs]
vp, l0 = max((V, l) for V, l in woc if l >= 0); v1, l1 = min((V, l) for V, l in woc if l < 0)
vco = vp + (v1 - vp) * l0 / (l0 - l1)
blind = min(V for V in Vs if hx[str(V)]["cbf"]["htrue_min"] < 0)
T["FF_TEXT"] = (
"Proposition~\\ref{prop:achiev} already answers which missions can be certified at all; Figure~\\ref{fig:applic} (left) maps it for the hexarotor. "
"The window closes from two sides, through low collective (rotors near zero spin) and high moment demand. Beyond an amplitude of about $2.5\\times$ no collective admits a floor: "
"the moment cannot be produced without slowing some rotor toward dropout. Inside the frontier every commanded fiber meets $\\Cset$, so no wrench of the mission is excluded by the barrier. "
"Outside it no allocator can hold any floor above $\\ldrop$, and the filter, which stays feasible through $\\delta$, trades tracking for authority explicitly. "
"Maneuvers that need a rotor to reverse, such as inverted flight, are excluded by design.\n\n"
"\\emph{Forward flight.} To test non-hover operation we accelerate each vehicle from hover to airspeed $V\\le15$\\,m/s over $10\\,T_0$ and hold it. The plant uses a first-order inflow model, "
"$C_T(J)=C_{T0}(1-J/J_0)$ in the advance ratio $J$, with $J_0=0.6$ (about the pitch ratio of a $10{\\times}4.7$ propeller) and $v^{\\mathrm{sat}}=10^4$\\,rpm. This makes the slope factor of Corollary~\\ref{cor:slope} "
"$s_i=1-J_i/(2J_0)$, with $J_i$ the freestream component along rotor $i$'s induced flow. The commanded wrench adds a drag force $0.25\\,W(V/15)^2$, with the hexarotor tilting to produce it, "
"and an integral loop on the true wrench stands in for the position controller. The floor is the hover design of Proposition~\\ref{prop:achiev}. "
"The motor-level barrier stays invariant at every speed, but on the hexarotor the hover model is optimistic: true readiness $h_s$ falls below the floor from $V=%d$\\,m/s for effort-min, greedy DAAM, and the hover-model filter alike (Figure~\\ref{fig:applic}, right). "
"Scheduling the barrier on the measured airspeed restores $h_s\\ge0$ wherever the mission is certifiable in the true metric ($V\\lesssim%.0f$\\,m/s), at a wrench error below $%.0f\\times10^{-3}$. "
"On the octorotor the drag force consumes lateral authority and certifiability ends near $%.0f$\\,m/s. Beyond it greedy DAAM drives a rotor through zero spin ($h_s=%.1f$ at $12$\\,m/s) and effort-min collapses at $15$\\,m/s ($h_s=%.1f$). "
"Both variants of the filter keep $h_s\\ge-0.003$, the sampled-data dip of Remark~\\ref{rem:sampled}, by giving up tracking ($\\mathrm{RMS}\\le%.2f$). "
"The method thus remains useful in forward flight, provided the slope shift is either budgeted in the reserve or scheduled, and its certificate ends exactly where Proposition~\\ref{prop:achiev} says no allocator can succeed."
% (blind, vc, 1e3 * max(hx[str(V)]["cbf_aware"]["werr"] for V in Vs if V <= 12), vco,
   oc["12"]["greedy"]["htrue_min"], oc["15"]["energy"]["htrue_min"], max(oc[str(V)]["cbf_aware"]["werr"] for V in Vs)))
for k, v in T.items():
    tex = tex.replace("@%s@" % k, v)
PAPER_TEX.write_text(tex)
fa = APPLIC_FIGURE.read_text()
for k in ["FFYMIN", "FFYMAX", "FFNC", "FFNCLAB"]: fa = fa.replace("@%s@" % k, T[k])
APPLIC_FIGURE.write_text(fa)
print("crossing speed", vc)
