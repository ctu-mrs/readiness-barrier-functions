"""verify_numbers.py -- check the numbers quoted in the manuscript against the results.

Every number of Section IV that comes from an experiment is recomputed from
results/generated/*.json, formatted as in the text, and searched for in
main.tex. Missing strings are listed and the exit code is 1. The manuscript is
only read.

Usage: python verify_numbers.py PATH_TO_MAIN_TEX
"""
import json, re, numpy as np, sys
from pathlib import Path
from project_paths import RESULTS_DIR, ROOT

if len(sys.argv) != 2:
    raise SystemExit("Usage: python verify_numbers.py PATH_TO_MAIN_TEX")

PAPER_TEX = Path(sys.argv[1]).resolve()
tex = PAPER_TEX.read_text()
sw = json.load(open(RESULTS_DIR / "sweep.json")); mc = json.load(open(RESULTS_DIR / "mc.json")); rb = json.load(open(RESULTS_DIR / "robust.json"))
kp = json.load(open(RESULTS_DIR / "kappa.json")); dl = json.load(open(RESULTS_DIR / "delay.json")); ff = json.load(open(RESULTS_DIR / "ff.json")); tm = json.load(open(RESULTS_DIR / "timing.json"))
checks = []
def need(s, why): checks.append((s in tex, s, why))   # s must appear verbatim in main.tex
g = lambda th, m, k: sw["runs"]["%s_%s" % (th, m)][k]
# Table I rows
for th in ["0.7", "0.8", "0.9", "1.0", "1.1", "1.2"]:
    for m in ["energy", "greedy", "lowpass", "cbf"]:
        need("%.4f" % g(th, m, "werr"), "TabI werr %s %s" % (th, m)); need("%.1f" % g(th, m, "TV"), "TabI TV %s %s" % (th, m))
        h = "%+.2f" % g(th, m, "hmin"); need(h, "TabI hmin %s %s" % (th, m))
need("%.3f" % g("0.7", "greedy", "werr"), "greedy werr 0.7"); need("%+.2f" % g("0.7", "lowpass", "hmin"), "lowpass hmin 0.7")
need("%+.2f" % g("0.7", "greedy", "hmean"), "greedy hmean"); need("%+.2f" % g("0.7", "cbf", "hmean"), "cbf hmean")
# MC
def st(k, m):
    L = mc[k]["missions"]; h = np.array([x["res"][m]["hmin"] for x in L]); return np.mean(h < 0), h.min()
need("$%d\\%%$ and $%d\\%%$ of missions (effort" % (round(100*st("hex_env","greedy")[0]), round(100*st("hex_env","lowpass")[0])), "MC env viol")
need("$%d\\%%$ and $%d\\%%$ of missions, as deep as $%.1f$ and $%.1f$" % (round(100*st("hex_scarce","greedy")[0]), round(100*st("hex_scarce","lowpass")[0]), st("hex_scarce","greedy")[1], st("hex_scarce","lowpass")[1]), "MC scarce")
need("$h_{\\min}=%+.2f$)" % st("hex_scarce", "cbf")[1], "MC cbf lowest")
need("=%.0f$ and $%.0f$" % (st("oct_scarce","greedy")[1], st("oct_scarce","lowpass")[1]), "oct collapse")
# robust
c = rb["campaign"]; a = rb["ablation"]
for p in ["0.10", "0.20", "0.30"]:
    for k in ["energy", "cbf-nominal", "cbf-robust"]:
        need("$%+.3f$ & $%+.3f$" % (c[p+"_"+k]["mean"], c[p+"_"+k]["worst"]), "robust row %s %s" % (p, k))
    for k in ["metric", "full"]:
        need("$%+.3f$" % a[p+"_"+k]["mean"], "abl mean %s %s" % (p, k))
need("$%d\\%%$ at $p=10\\%%$" % round(100*c["0.10_cbf-nominal"]["viol"]), "nominal viol 10")
# kappa
r1 = kp["rows"]["0.1"]; r9 = kp["rows"]["0.9"]
need("$%.2f\\to%.2f$" % (r1["auth"], r9["auth"]), "kappa auth"); need("from $%.1f$ to $%.1f\\times10^{-3}T_0$" % (r1["tdstar"]*1e3, r9["tdstar"]*1e3), "kappa tdstar")
need("from $%d\\%%$ to $%d\\%%$" % (round(100*r1["cl_0.70"]["active"]), round(100*r9["cl_0.70"]["active"])), "kappa activity")
ok04 = [k for k in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"] if kp["rows"][k]["delay_0.1"] >= 0]
need("$\\kappa\\le%s$" % ok04[-1], "kappa delay survive")
pv = [kp["rows"][k]["par_viol"] for k in kp["rows"]]; need("$%d$--$%d\\%%$ of plants" % (round(100*min(pv)), round(100*max(pv))), "kappa par viol")
# delay
rows = {(r["kind"], r["init"], round(r["td"]*1e3, 4)): r for r in dl["rows"]}
need("$\\tau_d^\\star=%.2f\\times10^{-3}" % (dl["tdstar"]*1e3), "tdstar"); need("+%.3f$ at $0.02" % rows[("plain","hold",20.0)]["hmin"], "delay 20")
need("$+%.3f$ at $0.05" % rows[("plain","hold",50.0)]["hmin"], "delay 50"); need("KV\\approx%d" % round(dl["K"]*dl["V"], -2), "KV")
# ff (normalised study, data-based v_sat and J0)
hx = ff["hex"]["speeds"]; oc = ff["oct"]["speeds"]
need("from $%d$\\,m/s for effort-min" % min(int(V) for V in hx if hx[V]["cbf"]["htrue_min"] < 0), "ff hex blind speed")
need("$h_s=%.1f$ at $12$" % oc["12"]["greedy"]["htrue_min"], "ff oct greedy"); need("($h_s=%.1f$)" % oc["15"]["energy"]["htrue_min"], "ff oct energy")
need("up to $%.2f$" % max(max(oc[V]["cbf"]["werr"], oc[V]["cbf_aware"]["werr"]) for V in oc), "ff rms max")
assert hx["12"]["cbf_aware"]["htrue_min"] >= -2e-3 and max(hx[V]["cbf_aware"]["werr"] for V in ["0","3","6","9","12"]) < 3e-2
assert min(oc[V]["cbf_aware"]["htrue_min"] for V in oc) >= -0.0035 and min(oc[V]["cbf"]["htrue_min"] for V in oc) >= -0.0035
# high-fidelity campaign
sys.path.insert(0, str(ROOT / "hifi"))
from analyze_hifi import main as HM, summ, D as HD
fam = lambda k, m: summ([r for r in HM if r["kind"] == k and r["method"] == m])
assert fam("rock", "cbf")["vt"] == 0 and fam("load", "cbf")["vt"] == 0
need("$%d$--$%d\\%%$ of external-moment runs" % (round(100*fam("load","greedy")["vt"]), round(100*fam("load","lowpass")["vt"])), "hifi greedy/lowpass viol")
need("($%.2f$ vs.\\ $%.1f$)" % (fam("cruise","cbf")["htmin"], fam("cruise","energy")["htmin"]), "hifi cruise")
rows_tab = PAPER_TEX.read_text()
for k in ["rock", "load", "cruise"]:
    for m in ["energy", "greedy", "lowpass", "cbf", "robust"]:
        f_ = fam(k, m); need("%+.2f$" % f_["htmin"], "hifi table %s %s" % (k, m))
dl = [r for r in HD if r["tag"] == "delay"]
for d in [0.005, 0.01, 0.02]:
    assert all(r["res"]["htrue_min"] >= 0 for r in dl if r["method"] == "cbf" and r["delay"] == d)
assert all(r["res"]["htrue_min"] < 0 for r in dl if r["method"] == "energy" and r["delay"] == 0.02)
assert sum(r["res"]["htrue_min"] < 0 for r in dl if r["method"] == "cbf" and r["delay"] == 0.04) == 1
# timing
need("$%d$\\,$\\mu$s on average ($99$th percentile $%d$" % (round(tm["hex"]["filter_osqp_solve"]["mean"]), round(tm["hex"]["filter_osqp_solve"]["p99"])), "timing")
bad = [c for c in checks if not c[0]]
print("%d/%d checks passed" % (len(checks) - len(bad), len(checks)))
for b in bad: print("MISSING:", b[1], "<-", b[2])
sys.exit(1 if bad else 0)
