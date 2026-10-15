"""exp_ff.py -- forward flight up to 15 m/s (Fig. 5, right).

Plant: first-order propeller inflow model, linear in the advance ratio,
    thrust factor  c_i = 1 - J_i / J0,   J_i = V_ax,i / (n_i D),   J0 = 0.65,
where J0 is the zero-thrust advance ratio of the APC 10x4.7SF data in
hifi/apc10x47.py and V_ax,i is the freestream component along rotor i's
induced flow. The drag-torque coefficient is left unchanged. Speeds are scaled
with v_sat = 767.8 rad/s (7330 rpm, identified in hifi/model.py) and D = 0.254 m.

Vehicles:
  octorotor  flies level and commands the drag force F_x = 0.25 W (V/15)^2
             directly; its tilted rotors see different inflow;
  hexarotor  tilts by theta = atan(F_x / W) to produce the drag force: uniform
             axial inflow V sin(theta) and collective W / cos(theta).
Airspeed ramps from hover to V over TRAMP and is then held.

Allocators: effort QP, greedy DAAM, the filter with the hover model, and an
airspeed-scheduled filter whose barrier uses the inflow-aware co-metric. All
use the nominal A; an integral loop on the true wrench (gain ki) stands in for
the vehicle's position/attitude loop. The floor is the hover design.
h_s (htrue) is the readiness of the true plant.

Output: results/generated/ff.json (make_paper_data.py writes data/ff.dat from it).
"""
import numpy as np, json, sys
import multiprocessing as mp
from project_paths import result_path
from scipy.optimize import minimize
from platforms import hexarotor, tilted_octorotor
from daam import h_value
from gate import ell_dagger
from final_exp import floor_rule
from harden import traj
from sim_core import run
phi = lambda v: v * np.abs(v)
NSD = 767.8 * 0.254 / (2 * np.pi); J0 = 0.65   # n_sat * D [m/s] at v_sat, and the zero-thrust advance ratio
SPEEDS = [0, 3, 6, 9, 12, 15]
TEND = 14.0

TRAMP = 10.0   # ramp time in T0 (about 1 s with T0 ~ 0.1 s): 0.9 g to reach 9 m/s, 1.5 g for 15 m/s
def sstep(t):
    """Smoothstep from 0 to 1 over [0, TRAMP]."""
    x = np.clip(t / TRAMP, 0.0, 1.0); return x * x * (3 - 2 * x)

def make(name, V, amp=1.5, thrust=0.9, period=2.0, ramp=True):
    """Airspeed ramps smoothly from hover to V over TRAMP (accelerating flight),
    then is held. ramp=False gives the steady condition (used for L_op)."""
    P = hexarotor() if name == "hex" else tilted_octorotor()
    w0, wd0 = traj(P, amp, thrust, period)
    W = thrust * P["n"] * P["vstar"] ** 2
    Vt = (lambda t: V * sstep(t)) if ramp else (lambda t: V)
    Fd = lambda t: 0.25 * W * (Vt(t) / 15.0) ** 2      # body drag force
    # Aeff: true wrench map (columns scaled by the thrust factor c_i);
    # slope: d(thrust)/d(v) ratio true/nominal, used by the true readiness.
    # Both factors are clipped to [0.2, 1.5].
    if name == "oct":
        xd = P["A"][0, :]  # x-component of each thrust direction d_i
        def w(t): x = w0(t); x[0] += Fd(t); return x
        def Aeff(v, t):
            J = Vt(t) * xd * np.sign(v) / (NSD * np.maximum(np.abs(v), 1e-3))
            return P["A"] * np.clip(1 - J / J0, 0.2, 1.5)[None, :]
        def slope(v, t):
            J = Vt(t) * xd * np.sign(v) / (NSD * np.maximum(np.abs(v), 1e-3))
            return np.clip(1 - J / (2 * J0), 0.2, 1.5)
    else:
        th = lambda t: np.arctan2(Fd(t), W)
        def w(t): x = w0(t); x[0] = W / np.cos(th(t)); return x
        def Aeff(v, t):
            J = Vt(t) * np.sin(th(t)) / (NSD * np.maximum(np.abs(v), 1e-3))
            return P["A"] * np.clip(1 - J / J0, 0.2, 1.5)[None, :]
        def slope(v, t):
            J = Vt(t) * np.sin(th(t)) / (NSD * np.maximum(np.abs(v), 1e-3)) * np.ones_like(v)
            return np.clip(1 - J / (2 * J0), 0.2, 1.5)
    def wd(t, e=1e-5):
        # rate of the forward-flight part by central differences
        return wd0(t) + ((w(t + e) - w0(t + e)) - (w(t - e) - w0(t - e))) / (2 * e) if t > e else wd0(t)
    Aeff.slope = slope
    return P, (w, wd), Aeff

def Lop_true(P, wfun, Aeff, T=2.0, npts=20, metric=None):
    """min over the mission of max_{A_eff(v) phi(v) = w} L(v): the readiness the
    nominal barrier can keep while still producing the TRUE commanded wrench."""
    w, _ = wfun; v0 = np.full(P["n"], P["vstar"]); best = []
    lim = 0.999 * P["vsat"]
    for t in np.linspace(0, T, npts):
        cons = {"type": "eq", "fun": lambda v: Aeff(v, t) @ phi(v) - w(t)}
        f = (lambda v: -h_value(v, P)) if metric is None else (lambda v: -metric(v))
        r = minimize(f, v0, constraints=[cons], bounds=[(-lim, lim)] * P["n"],
                     method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        if not r.success: return -np.inf
        v0 = r.x; best.append(-f(r.x))
    return float(min(best))


# ----------------------------------------------------------------------------------
# Airspeed-scheduled filter: the barrier uses the inflow-aware co-metric
#   D_V(v, t) = 4 A diag(s_i(v_i, V(t))^2 psi_i(v_i)) A^T,  s_i = true/nominal thrust slope,
# the servo row uses the true Jacobian, and dh/dt includes the known airspeed change.
# Gradients are taken by central differences.
import scipy.sparse as sps
from daam import psi as psif
def h_aware(v, t, P, Ae, eb):
    """Inflow-aware readiness ln det D_V(v, t) - eb."""
    s = Ae.slope(v, t); D = 4.0 * (P["A"] * (s ** 2 * psif(v, P))[None, :]) @ P["A"].T
    sg, ld = np.linalg.slogdet(D); return (ld if sg > 0 else -np.inf) - eb
def thrust_true(v, t, Ae): return Ae(v, t) @ phi(v)   # true wrench
def aware_step(v, t, wc, wdc, P, Ae, eb, alpha=5.0, Kw=8.0, rho=1e4, e=1e-6):
    """One step of the airspeed-scheduled filter: returns (tau, |delta|, active, h)."""
    import allocators as AL
    n, m = P["n"], P["m"]
    h = h_aware(v, t, P, Ae, eb)
    gh = np.array([(h_aware(v + e * np.eye(n)[i], t, P, Ae, eb) - h_aware(v - e * np.eye(n)[i], t, P, Ae, eb)) / (2 * e) for i in range(n)])
    ht = (h_aware(v, t + 1e-4, P, Ae, eb) - h_aware(v, max(t - 1e-4, 0), P, Ae, eb)) / (2e-4 if t > 1e-4 else 1e-4)
    Jt = np.column_stack([(thrust_true(v + e * np.eye(n)[i], t, Ae) - thrust_true(v - e * np.eye(n)[i], t, Ae)) / (2 * e) for i in range(n)])
    d = -(P["b"] * phi(v)) / P["mass"]; Minv = np.diag(1 / P["mass"])
    mu = wdc + Kw * (wc - thrust_true(v, t, Ae))
    # nominal (effort-min) with the aware Jacobian, least-squares fallback
    Aeq = Jt @ Minv; beq = mu - Jt @ d
    Pm = np.zeros((n + m, n + m)); Pm[:n, :n] = 2 * np.eye(n); Pm[n:, n:] = 2 * rho * np.eye(m)
    A2 = np.vstack([np.hstack([Aeq, -np.eye(m)]), np.hstack([np.eye(n), np.zeros((n, m))])])
    l2 = np.concatenate([beq, -P["tau_bar"]]); u2 = np.concatenate([beq, P["tau_bar"]])
    x, _, _ = AL._osqp_solve(Pm, np.zeros(n + m), A2, l2, u2)
    tr = np.zeros(n) if x is None else x[:n]
    q = np.concatenate([-2 * tr, np.zeros(m)])
    Ab = np.concatenate([gh @ Minv, np.zeros(m)])[None, :]
    A3 = np.vstack([Ab, A2]); l3 = np.concatenate([[-alpha * h - gh @ d - ht], l2]); u3 = np.concatenate([[np.inf], u2])
    x, _, _ = AL._osqp_solve(Pm, q, A3, l3, u3)
    if x is None:
        tau, delta = AL._barrier_fallback(gh, P["tau_bar"], Jt, d, Minv, mu)
        return tau, np.linalg.norm(delta), int(np.max(np.abs(tau - tr)) > 1e-5), h
    return np.clip(x[:n], -P["tau_bar"], P["tau_bar"]), np.linalg.norm(x[n:]), int(np.max(np.abs(x[:n] - tr)) > 1e-5), h

def run_aware(name, V, eb, T=5.0, dt=1e-3, ki=2.0):
    """Closed loop with the airspeed-scheduled filter; same outputs as job() below."""
    import allocators as AL
    P, (w, wd), Ae = make(name, V)
    v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"])); z = np.zeros(P["m"])
    H, HN, WE, ACT, DL, VV = [], [], [], [], [], []
    for k in range(int(T / dt)):
        t = k * dt; wc = w(t) + z
        tau, dl, act, _ = aware_step(v, t, wc, wd(t), P, Ae, eb)
        d = -(P["b"] * phi(v)) / P["mass"]
        v = np.clip(v + dt * (d + tau / P["mass"]), -0.999, 0.999)
        wt = thrust_true(v, t, Ae); z = z + dt * ki * (w(t) - wt)
        H.append(h_aware(v, t, P, Ae, eb)); HN.append(h_value(v, P, eb)); WE.append(np.linalg.norm(wt - w(t)))
        ACT.append(act); DL.append(dl); VV.append(np.abs(v).min())
    H = np.array(H); WE = np.array(WE); tt = np.arange(len(H)) * dt
    return dict(hmin=float(np.min(HN)), htrue_min=float(H.min()), werr=float(np.sqrt(np.mean(WE ** 2))),
                werr_ss=float(np.sqrt(np.mean(WE[tt >= TRAMP] ** 2))), active=float(np.mean(ACT)),
                dmax=float(np.max(DL)), tbelow_true=float(np.sum(H < 0) * dt), minabsv=float(np.min(VV)),
                vmax=None, gap=None)

def v_lower(P, eb):
    """Prop. 1(i): on C every psi_i >= eps_i, eps_i solving g_i(psi)=ellbar with the
    other rotors at the sweet spot; returns the smallest |v_i| compatible with it."""
    from scipy.optimize import brentq
    from daam import psi as psif
    vs = np.sqrt(P["tau_bar"] / (3 * P["b"])); ps = psif(vs, P); lo = []
    for k in range(P["n"]):
        def g(x):
            pv = ps.copy(); pv[k] = x
            return np.linalg.slogdet(4.0 * (P["A"] * pv[None, :]) @ P["A"].T)[1] - eb
        eps = brentq(g, 1e-12, ps[k])
        # smallest speed in (0, v*] with psi >= eps (the broadcast just evaluates psi on vv)
        vv = np.linspace(1e-4, vs[k], 20001); lo.append(vv[np.argmax(psif(np.full(P["n"], 1.0) * 0 + vv[:, None] * 0 + vv[:, None], P)[:, k] >= eps)])
    return float(min(lo))

def shift_for(name, V, eb):
    """closed-form floor shift -2m ln s_min over the certified set at airspeed V."""
    P, wf, Ae = make(name, V, ramp=False)
    vl = v_lower(P, eb)
    worst = 1.0
    for sg in [1.0, -1.0]:
        vv = np.full(P["n"], sg * vl)
        worst = min(worst, float(np.min(Ae.slope(vv, 10.0))))
    return -2 * P["m"] * np.log(worst), vl, worst

def job(t):
    """Run one allocator at airspeed V; h and h_s are reported against the hover floor."""
    name, V, m, eb, shift = t
    if m == "cbf_aware":
        return t, run_aware(name, V, eb, T=TEND)
    P, wf, Ae = make(name, V)
    meth = "cbf" if m.startswith("cbf") else m
    r = run(P, meth, eb + shift, T=TEND, amp=1.5, thrust=0.9, wfun=wf, Aeff=Ae, ki=2.0, log=True,
            Aslope=lambda v, tt: P["A"] * Ae.slope(v, tt)[None, :])
    h = r["h"] + shift; ht = r["htrue"] + shift      # both w.r.t. the ORIGINAL floor
    return t, dict(hmin=float(h.min()), htrue_min=float(ht.min()),
                   werr=float(np.sqrt(np.mean(r["werr_t"] ** 2))),
                   werr_ss=float(np.sqrt(np.mean(r["werr_t"][r["t"] >= TRAMP] ** 2))),
                   active=float(r["act"].mean()), dmax=float(r["delta"].max()),
                   tbelow_true=float(np.sum(ht < 0) * 1e-3), minabsv=float(np.abs(r["v"]).min()),
                   vmax=float(np.abs(r["v"]).max()), gap=float((ht - h).min()))

if __name__ == "__main__":
    out = {}
    tasks = []
    for name in ["oct", "hex"]:
        P, wf0, _ = make(name, 0)
        eb, ld, hop = floor_rule(P, 1.5, 0.9, kappa=0.5)  # hover-designed floor
        out[name] = dict(eb=eb, ldrop=ld, Lop_hover=hop, speeds={})
        for V in SPEEDS:
            P, wf, Ae = make(name, V, ramp=False)
            lop = Lop_true(P, wf, Ae)                    # nominal metric, true wrench map
            sh, vl, smin = shift_for(name, V, eb)
            # true (inflow-aware) metric at the held airspeed; Lop_s - eb < 0 means not certifiable
            lop_s = Lop_true(P, wf, Ae, metric=lambda v: h_aware(v, 1e9, P, Ae, 0.0))
            out[name]["speeds"][str(V)] = dict(Lop=lop, window=lop - eb, Lop_s=lop_s, shift=sh, vlow=vl, smin=smin)
            print(name, V, "Lop_s-ellbar %+.3f Lop_true %.3f  Lop-ellbar %+.3f  shift %.3f (vlow %.3f smin %.3f)" % (lop_s - eb, lop, lop - eb, sh, vl, smin), flush=True)
            tasks += [(name, V, m, eb, 0.0) for m in ["energy", "greedy", "cbf"]]
            tasks += [(name, V, "cbf_aware", eb, 0.0)]
    with mp.Pool(2) as pool: res = pool.map(job, tasks)
    for (name, V, m, eb, sh), r in res:
        out[name]["speeds"][str(V)][m] = r
        print(name, V, m, {k: (round(x, 4) if x is not None else None) for k, x in r.items()}, flush=True)
    json.dump(out, open(result_path("ff.json"), "w"), indent=1)
