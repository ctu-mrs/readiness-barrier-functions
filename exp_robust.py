"""exp_robust.py -- robustness campaign, NR plants per level (submitted version: 8).
Paired design: within each level all controllers run on the SAME NR random plants
(the submitted campaign drew different plants for each controller).
Upper block: robust_campaign.rollout (energy / nominal barrier / robust barrier).
Lower block: robust_v2.roll at kappa=0.9 (metric bound only vs full robust filter)."""
import numpy as np, json, sys
import multiprocessing as mp
from project_paths import result_path
from platforms import hexarotor
from final_exp import floor_rule
from robust import perturb_params
from robust_campaign import rollout
import robust_v2

P = hexarotor()


def run_task(t):
    kind, p, label, Pt, Pc, eb, mth = t
    if kind == "camp":
        return kind, p, label, rollout(Pt, Pc, eb, mth)
    return kind, p, label, robust_v2.roll(Pt, P, Pc, eb, p, label)


if __name__ == "__main__":
    NR = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    eb_nom, ld, hop = floor_rule(P, 1.5, 0.75, kappa=0.5)
    out = {"eb_nom": eb_nom, "NR": NR, "floors": {}, "campaign": {}, "ablation": {}}
    tasks = []
    for p in [0.10, 0.20, 0.30]:
        Pw = perturb_params(P, p, worst=True)
        eb_rob, _, _ = floor_rule(Pw, 1.5, 0.75, kappa=0.5)
        eb_abl, _, _ = floor_rule(Pw, 1.5, 0.75, kappa=0.9)
        out["floors"]["%.2f" % p] = dict(robust=eb_rob, ablation=eb_abl)
        rng = np.random.default_rng(11)
        plants = [perturb_params(P, p, worst=False, rng=rng) for _ in range(NR)]
        for Pt in plants:
            tasks.append(("camp", p, "energy", Pt, P, eb_nom, "energy"))
            tasks.append(("camp", p, "cbf-nominal", Pt, P, eb_nom, "cbf"))
            tasks.append(("camp", p, "cbf-robust", Pt, Pw, eb_rob, "cbf"))
            tasks.append(("abl", p, "metric", Pt, Pw, eb_abl, None))
            tasks.append(("abl", p, "full", Pt, Pw, eb_abl, None))
    with mp.Pool(2) as pool:
        res = pool.map(run_task, tasks)
    agg = {}
    for kind, p, label, (hm, we) in res:
        agg.setdefault((kind, p, label), []).append((hm, we))
    for (kind, p, label), L in sorted(agg.items()):
        a = np.array(L); h = a[:, 0]; e = a[:, 1]; q = float(np.mean(h < 0))
        # Wilson 95% interval for the violation fraction
        n = len(h); z = 1.96; c = (q + z*z/(2*n)) / (1 + z*z/n)
        hw = z*np.sqrt(q*(1-q)/n + z*z/(4*n*n)) / (1 + z*z/n)
        row = dict(mean=float(h.mean()), worst=float(h.min()), viol=q, viol_lo=float(max(0, c-hw)),
                   viol_hi=float(min(1, c+hw)), werr=float(e.mean()), n=n)
        (out["campaign"] if kind == "camp" else out["ablation"])["%.2f_%s" % (p, label)] = row
        print(kind, p, label, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    json.dump(out, open(result_path("robust.json"), "w"), indent=1)
