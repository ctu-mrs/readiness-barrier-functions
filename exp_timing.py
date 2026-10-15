"""exp_timing.py -- computation time of Algorithm 1 per control step.

Measures, over 2000 steps of the nominal mission on one core: the barrier and
gradient evaluation, the effort QP, the filter QP (wall time) and OSQP's own
solve time. The QP is set up from scratch at every step and never warm
started, so these are upper figures. Results depend on the computer.

Output: results/generated/timing.json.
"""
import numpy as np, time, json, platform
from project_paths import result_path
import allocators as AL
from daam import grad_h, h_value
from platforms import hexarotor, tilted_octorotor
from final_exp import floor_rule
from harden import traj
out = {"cpu": platform.processor() or platform.machine()}
for name, P in [("hex", hexarotor()), ("oct", tilted_octorotor())]:
    eb, _, _ = floor_rule(P, 1.5, 0.9 if name == "hex" else 0.9)
    w, wd = traj(P, 1.5, 0.9); v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"]))
    tg, tn, tf, osq = [], [], [], []
    for k in range(2000):
        t = k * 1e-3
        t0 = time.perf_counter(); gh = grad_h(v, P); h = h_value(v, P, eb); t1 = time.perf_counter()
        tr, _ = AL.energy_qp_tau(v, w(t), wd(t), P); t2 = time.perf_counter()
        tau, dl, rt = AL.cbf_qp(v, w(t), wd(t), P, eb, tau_ref=tr); t3 = time.perf_counter()
        tg.append(t1 - t0); tn.append(t2 - t1); tf.append(t3 - t2); osq.append(rt)
        d = -(P["b"] * v * np.abs(v)) / P["mass"]; v = v + 1e-3 * (d + tau / P["mass"])   # advance the motors
    # statistics in microseconds
    q = lambda x: dict(mean=float(np.mean(x) * 1e6), p99=float(np.percentile(x, 99) * 1e6), max=float(np.max(x) * 1e6))
    out[name] = dict(grad=q(tg), nominal_qp=q(tn), filter_qp_wall=q(tf), filter_osqp_solve=q(osq))
    print(name, out[name])
json.dump(out, open(result_path("timing.json"), "w"), indent=1)
