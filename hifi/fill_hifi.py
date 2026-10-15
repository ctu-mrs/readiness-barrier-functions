"""fill_hifi.py -- write the body of Table IV as a TeX fragment.

Usage: python hifi/fill_hifi.py PATH_TO_OUTPUT_TEX
Each mission family gets two rows: violation fraction / lowest true readiness,
and mean attitude error (a dagger marks non-certifiable runs).
"""
import json, numpy as np
import sys
from pathlib import Path
from analyze_hifi import D, main, summ
METH = ["energy", "greedy", "lowpass", "cbf", "robust"]
FAM = [("agile", "rock", "agile flight"), ("interaction", "load", "external moment"), ("cruise", "cruise", "cruise $0$--$15$\\,m/s")]
if len(sys.argv) != 2:
    raise SystemExit("Usage: python fill_hifi.py PATH_TO_OUTPUT_TEX")

rows = []
for f, k, lab in FAM:
    cells, cells2 = [], []
    for m in METH:
        rr = [r for r in main if r["kind"] == k and r["method"] == m]
        s = summ(rr)
        nc = sum(1 for r in rr if not r["res"]["certifiable"])
        if s is None: cells.append("--"); cells2.append(""); continue
        v = "$\\mathbf{%.2f}$" % s["vt"] if s["vt"] > 0 else "$%.2f$" % s["vt"]
        cells.append("%s\\,/\\,$%+.2f$" % (v, s["htmin"]))
        cells2.append("$%.0f^\\circ$%s" % (s["eR"], ("$^{\\dagger%d}$" % nc) if nc else ""))
    n = len({(r["mass"], tuple(sorted(r["kw"].items())), r["seed"]) for r in main if r["kind"] == k})
    rows.append("    %s ($%d$) & %s\\\\" % (lab, n, " & ".join(cells)))
    rows.append("    & %s\\\\" % " & ".join(cells2))
Path(sys.argv[1]).resolve().write_text("\n".join(rows) + "\n")
print("\n".join(rows))
