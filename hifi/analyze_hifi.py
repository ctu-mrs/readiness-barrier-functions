"""analyze_hifi.py -- summarise results/hifi.json per mission and per mission family."""
import json, numpy as np, sys
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import RESULTS_DIR
D = json.load(open(RESULTS_DIR / "hifi.json"))
METH = ["energy", "greedy", "lowpass", "cbf", "robust"]
def key(r): return (r["kind"], r["mass"], tuple(sorted(r["kw"].items())))
main = [r for r in D if r["tag"] == "main"]
confs = sorted({key(r) for r in main}, key=str)
def summ(rows):
    c = [r["res"] for r in rows if r["res"]["certifiable"]]
    if not c: return None
    ht = np.array([x["htrue_min"] for x in c]); hn = np.array([x["hmin"] for x in c])
    return dict(n=len(c), nc=len(rows) - len(c), vt=float(np.mean(ht < 0)), vn=float(np.mean(hn < 0)), htmin=float(ht.min()),
                hnmin=float(hn.min()), cross=float(np.mean([x["crossings"] for x in c])), eR=float(np.degrees(np.mean([x["eR"] for x in c]))),
                ez=float(np.mean([x["ez"] for x in c])), werr=float(np.mean([x["werr"] for x in c])), act=float(np.mean([x["active"] for x in c])))
if __name__ == "__main__":
    for cf in confs:
        print(cf)
        for m in METH:
            s = summ([r for r in main if key(r) == cf and r["method"] == m])
            if s is None:
                rows = [r for r in main if key(r) == cf and r["method"] == m]
                print("   %-7s %s" % (m, "not certifiable" if rows else "-")); continue
            print("   %-7s n%d vt %.2f vn %.2f htmin %+.2f hnmin %+.2f cross %.1f eR %.1f ez %.3f werr %.3f act %.2f" %
                  (m, s["n"], s["vt"], s["vn"], s["htmin"], s["hnmin"], s["cross"], s["eR"], s["ez"], s["werr"], s["act"]))
    fam = {"agile": "rock", "interaction": "load", "cruise": "cruise"}
    print("\nFAMILIES")
    for f, k in fam.items():
        for m in METH:
            s = summ([r for r in main if r["kind"] == k and r["method"] == m])
            if s: print("%-12s %-7s n%2d (+%d NC) viol_true %.2f viol_nom %.2f min_htrue %+.2f cross %.1f eR %.1fdeg ez %.3f werr %.3f" %
                        (f, m, s["n"], s["nc"], s["vt"], s["vn"], s["htmin"], s["cross"], s["eR"], s["ez"], s["werr"]))
    dl = [r for r in D if r["tag"] == "delay"]
    if dl:
        print("\nDELAY (load 1.27 kg, 1.0 Nm)")
        for m in ["energy", "cbf"]:
            for d in sorted({r["delay"] for r in dl}):
                s = summ([r for r in dl if r["method"] == m and r["delay"] == d])
                if s: print("  %-6s %4.0f ms viol_true %.2f min_htrue %+.3f hnmin %+.3f werr %.3f" % (m, d * 1e3, s["vt"], s["htmin"], s["hnmin"], s["werr"]))
