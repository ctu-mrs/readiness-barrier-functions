"""
hifi_sim.py -- high-fidelity closed loop (Table IV).

6-DoF hexarotor with data-driven rotors (model.py), an SE(3) attitude/position
controller and the allocators of the paper running at 500 Hz on the nominal
quadratic model identified from the same data. The plant is integrated at 2 kHz.

The allocators are the code in allocators.py, run in the normalised units of
the identified model (speed / v_sat, time / T0, torque / tau_bar, wrench /
(k_f v_sat^2)); units are converted only at the sensor and actuator interfaces.

Allocators: "energy", "greedy", "lowpass", "cbf" and "robust" (robust filter
with a +-15 % parameter box). Missions: "rock", "load", "cruise" (and "eight",
not used in the campaign).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import allocators as AL
from daam import h_value
from gate import ell_dagger
from robust import perturb_params
import model as M

phi = lambda v: v * np.abs(v)


def nominal_platform(nom):
    """Platform dict in the format of platforms.py, normalised units (tau_bar = b = m = 1)."""
    n = 6
    return dict(A=nom["A"], n=n, m=4, name="hex-hifi", tau_bar=np.ones(n), b=np.ones(n),
                mass=np.ones(n), vsat=1.0, vstar=1.0 / np.sqrt(3.0), fz_row=0, mx_row=1, my_row=2)


# ---------------------------------------------------------------- missions
def mission(kind, p, **kw):
    """Reference generator. Returns f(t) -> dict(pos, vel, acc, R_d, Om_d, dOm_d, mode)."""
    if kind == "rock":            # roll reversals with altitude/position hold
        Phi = np.deg2rad(kw.get("Phi", 45.0)); P = kw.get("P", 1.0); om = 2 * np.pi / P
        def ref(t):
            r = Phi * np.sin(om * t); rd = Phi * om * np.cos(om * t); rdd = -Phi * om ** 2 * np.sin(om * t)
            c, s = np.cos(r), np.sin(r)
            Rd = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            return dict(mode="att", z=0.0, Rd=Rd, Omd=np.array([rd, 0, 0]), dOmd=np.array([rdd, 0, 0]), roll=r)
        return ref
    if kind == "eight":           # lemniscate position tracking
        Ax = kw.get("A", 3.0); P = kw.get("P", 6.0); om = 2 * np.pi / P; dz = kw.get("dz", 0.0)
        def ref(t):
            pos = np.array([Ax * np.sin(om * t), 0.5 * Ax * np.sin(2 * om * t), dz * np.sin(om * t)])
            vel = np.array([Ax * om * np.cos(om * t), Ax * om * np.cos(2 * om * t), dz * om * np.cos(om * t)])
            acc = np.array([-Ax * om ** 2 * np.sin(om * t), -2 * Ax * om ** 2 * np.sin(2 * om * t), -dz * om ** 2 * np.sin(om * t)])
            jrk = np.array([-Ax * om ** 3 * np.cos(om * t), -4 * Ax * om ** 3 * np.cos(2 * om * t), -dz * om ** 3 * np.cos(om * t)])
            return dict(mode="pos", pos=pos, vel=vel, acc=acc, jerk=jrk)
        return ref
    if kind == "cruise":          # accelerate to airspeed V along x, slalom in y
        V = kw.get("V", 10.0); Tr = kw.get("Tr", 4.0); Ay = kw.get("Ay", 0.5); P = kw.get("P", 3.0); om = 2 * np.pi / P
        def ref(t):
            x = np.clip(t / Tr, 0, 1)
            # smoothstep velocity ramp: v = V*(3x^2-2x^3)
            vx = V * (3 * x ** 2 - 2 * x ** 3); ax = V * (6 * x - 6 * x ** 2) / Tr if t < Tr else 0.0
            px = V * Tr * (x ** 3 - 0.5 * x ** 4) if t < Tr else V * Tr * 0.5 + V * (t - Tr)
            pos = np.array([px, Ay * np.sin(om * t), 0.0]); vel = np.array([vx, Ay * om * np.cos(om * t), 0.0])
            acc = np.array([ax, -Ay * om ** 2 * np.sin(om * t), 0.0])
            return dict(mode="pos", pos=pos, vel=vel, acc=acc)
        return ref
    if kind == "load":            # hover + periodic external moment (slung load / tool reaction)
        Ma = kw.get("Mamp", 1.0); P = kw.get("P", 1.0); om = 2 * np.pi / P
        def ref(t):
            return dict(mode="pos", pos=np.zeros(3), vel=np.zeros(3), acc=np.zeros(3),
                        Mdist=np.array([Ma * np.sin(om * t), 0.0, 0.0]))
        return ref
    raise ValueError(kind)


def cross3(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def vee(S): return np.array([S[2, 1], S[0, 2], S[1, 0]])
def hat(w): return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


class Controller:
    """Lee-Leok-McClamroch SE(3) controller -> collective thrust f and body moments."""
    def __init__(self, p):
        self.p = p; self.kx = 6.0; self.kv = 4.0; self.kR = 0.9; self.kW = 0.12
    def __call__(self, s, r):
        p = self.p; R = M.rotmat(s["q"]); Om = s["Om"]; m = p.mass
        if r["mode"] == "pos":
            ex = s["x"] - r["pos"]; ev = s["v"] - r["vel"]
            F = m * (-self.kx * ex - self.kv * ev + M.G * np.array([0, 0, 1.0]) + r["acc"]) + p.cd_body * np.linalg.norm(r["vel"]) * r["vel"]
            b3 = F / np.linalg.norm(F); b1d = np.array([1.0, 0, 0])
            b2 = np.cross(b3, b1d); b2 /= np.linalg.norm(b2); b1 = np.cross(b2, b3)
            Rd = np.column_stack([b1, b2, b3]); Omd = np.zeros(3); dOmd = np.zeros(3)
            f = F @ R[:, 2]
        else:
            Rd, Omd, dOmd = r["Rd"], r["Omd"], r["dOmd"]
            ez = s["x"][2] - r["z"]; evz = s["v"][2]; exy = s["x"][:2]; evxy = s["v"][:2]
            Fz = m * (M.G - self.kx * ez - self.kv * evz)
            f = Fz / max(R[2, 2], 0.3)
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd); eW = Om - R.T @ Rd @ Omd
        Mb = (-self.kR * eR - self.kW * eW + cross3(Om, p.Ib @ Om)
              - p.Ib @ (hat(Om) @ R.T @ Rd @ Omd - R.T @ Rd @ dOmd))
        if "Mdist" in r:
            Mb = Mb - s.get("Mhat", np.zeros(3))
        return np.concatenate([[f], Mb]), eR


def feedforward_wrench(ref, p, T, n=40):
    # Used only to choose the floor before the run; the controller does not see it.
    """Mission wrench set W for Proposition 2, computed offline from the reference.
    Position missions: thrust from differential flatness (incl. body drag) and the
    roll/pitch moments I*(d^2/dt^2) of the flat-output attitude (finite differences)."""
    out = []; h = 0.01
    def att(t):
        r = ref(t)
        F = p.mass * (r["acc"] + np.array([0, 0, M.G])) + p.cd_body * np.linalg.norm(r["vel"]) * r["vel"]
        return F, np.array([np.arctan2(-F[1], F[2]), np.arctan2(F[0], np.hypot(F[1], F[2]))])
    for t in np.linspace(0, T, n):
        r = ref(t)
        if r["mode"] == "att":
            f = p.mass * M.G / np.cos(r["roll"]); Mb = p.Ib @ r["dOmd"]
        elif "Mdist" in r:
            f = p.mass * M.G; Mb = -r["Mdist"]
        else:
            F, a0 = att(t); _, ap = att(t + h); _, am = att(max(t - h, 0.0))
            add = (ap - 2 * a0 + am) / h ** 2 if t > h else (ap - a0) / h ** 2
            f = np.linalg.norm(F); Mb = np.array([p.Ib[0, 0] * add[0], p.Ib[1, 1] * add[1], 0.0])
        out.append(np.concatenate([[f], Mb]))
    return np.array(out)


def floor_from_mission(Pn, Wn, kappa=0.5):
    """Floor rule of final_exp.floor_rule, applied to a list of normalised mission wrenches."""
    ld, _ = ell_dagger(Pn); v0 = np.full(6, Pn["vstar"]); best = []
    for w in Wn:
        v = AL.fiber_argmax_h(w, Pn, v0); v0 = v; best.append(h_value(v, Pn, 0.0))
    hop = float(min(best))
    return (None if hop <= ld else ld + kappa * (hop - ld)), ld, hop


# ---------------------------------------------------------------- simulate
def simulate(method, kind="rock", mass=1.8, T=6.0, seed=0, kappa=0.5, robust_p=0.15,
             extra_delay=0.0, turb=1.0, mkw=None, log=False, floor=None):
    """Run one mission with one allocator; returns the summary used by run_hifi.py.

    mass: vehicle mass [kg]; T: duration [s]; seed: telemetry noise and turbulence;
    extra_delay: added actuation delay [s]; turb: Dryden intensity [m/s];
    mkw: mission parameters; floor: (eb, ld, hop) to skip the floor rule.
    h is the readiness of the nominal model at the true speeds, htrue the
    readiness of the true plant (true_readiness), both relative to the floor.
    """
    p = M.Params(mass=mass, seed=seed); nom = M.identify_nominal(p); Pn = nominal_platform(nom)
    mkw = mkw or {}
    ref = mission(kind, p, **mkw)
    Wn = feedforward_wrench(ref, p, T) / nom["Wscale"]
    if floor is None:
        if method == "robust":
            Pw = perturb_params(Pn, robust_p, worst=True)
            eb, ld, hop = floor_from_mission(Pw, Wn, kappa)
        else:
            eb, ld, hop = floor_from_mission(Pn, Wn, kappa)
    else:
        eb, ld, hop = floor
    if eb is None:
        return dict(certifiable=False, ld=ld, hop=hop)
    # worst-case corner and reduced torque box for the robust filter
    Pw = perturb_params(Pn, robust_p, worst=True); Pbox = dict(Pn); Pbox["tau_bar"] = Pn["tau_bar"] * (1 - robust_p)
    ctrl = Controller(p); rng = p.rng
    gust = M.Dryden(turb, 30.0, 5.0, rng) if turb > 0 else None
    # ---- state
    vs, T0, Ws, tb = nom["vsat"], nom["T0"], nom["Wscale"], nom["tbar"]
    w_hover = np.sqrt(p.mass * M.G / 6 / nom["kf"])
    s = dict(x=np.zeros(3), v=np.zeros(3), q=np.array([1.0, 0, 0, 0]), Om=np.zeros(3))
    r0 = ref(0.0)
    if r0["mode"] == "pos": s["x"] = r0["pos"].copy(); s["v"] = r0["vel"].copy()
    wn0 = np.array([p.mass * M.G, 0, 0, 0]) / Ws
    v0n = AL.fiber_argmax_h(wn0, Pn, np.full(6, Pn["vstar"]))
    # start in hover at the readiness-optimal speeds, with the current that holds them
    w = v0n * vs; cur = (nom["km"] * w * np.abs(w) + p.kt * p.I0 * np.sign(w)) / p.kt
    ncs = int(round(p.dt_ctrl / p.dt)); upd = 10      # greedy re-solve every 10 control steps (50 Hz)
    tel_buf = [w.copy()] * int(round(1e-3 / p.dt))     # 1 ms telemetry delay
    del_buf = []; kdel = int(round(extra_delay / p.dt_ctrl))
    icmd = cur.copy(); hold = v0n.copy(); vlp = v0n.copy(); prev_sp = v0n.copy()
    wcmd_prev = None; wdot_f = np.zeros(4); Ibatt = 0.0
    a_lp = p.dt_ctrl / (0.05 + p.dt_ctrl)
    LOG = {k: [] for k in ["t", "h", "htrue", "werr", "eR", "ez", "w", "sat", "rate", "act"]}
    k_ctrl = 0; wind = np.zeros(3)
    for k in range(int(round(T / p.dt))):
        t = k * p.dt
        Vbus = p.Voc - p.Rint * Ibatt
        # ------------------------------------------------ controller (500 Hz)
        if k % ncs == 0:
            r = ref(t)
            # speed telemetry: 1 ms old, 0.5 % noise, 10 rpm quantisation
            wm = tel_buf[0] * (1 + p.tel_noise * rng.standard_normal(6))
            wm = np.round(wm / p.tel_quant) * p.tel_quant
            vm = wm / vs
            wc, eR = ctrl(s, r)
            wcn = wc / Ws
            # the controller gives no wrench rate, so wdot is a filtered finite difference (10 ms)
            if wcmd_prev is not None:
                raw = (wcn - wcmd_prev) / p.dt_ctrl * T0
                wdot_f += (p.dt_ctrl / (0.01 + p.dt_ctrl)) * (raw - wdot_f)
            wcmd_prev = wcn
            act = 0; sp = vm.copy()
            if method == "energy":
                u, _ = AL.energy_qp_tau(vm, wcn, wdot_f, Pn)
            elif method in ("greedy", "lowpass"):
                if k_ctrl % upd == 0:
                    hold = AL.fiber_argmax_h_global(wcn, Pn, seed=seed, restarts=2)
                if method == "greedy":
                    sp = hold
                else:
                    vlp = (1 - a_lp) * vlp + a_lp * hold; sp = vlp
                u = AL.track_setpoint(vm, sp, Pn)
            elif method == "cbf":
                ur, _ = AL.energy_qp_tau(vm, wcn, wdot_f, Pn)
                u, _, _ = AL.cbf_qp(vm, wcn, wdot_f, Pn, eb, tau_ref=ur)
                act = int(np.max(np.abs(u - ur)) > 1e-4)
            elif method == "robust":
                ur, _ = AL.energy_qp_tau(vm, wcn, wdot_f, Pbox)
                u, _, _ = AL.cbf_qp_robust(vm, wcn, wdot_f, Pn, Pw, eb, robust_p, tau_ref=ur)
                u = np.clip(u, -Pbox["tau_bar"], Pbox["tau_bar"])
                act = int(np.max(np.abs(u - ur)) > 1e-4)
            rate = float(np.max(np.abs(sp - prev_sp)) / (p.dt_ctrl / T0)); prev_sp = sp
            tau_cmd = u * tb
            if kdel > 0:                     # extra delay; holds the hover torque until the buffer fills
                del_buf.append(tau_cmd)
                tau_cmd = del_buf.pop(0) if len(del_buf) > kdel else (nom["km"] * w * np.abs(w))
            icmd = tau_cmd / p.kt
            k_ctrl += 1
            # ------------- logging at controller rate (true quantities)
            if log or True:                  # always logged: the summary below needs these
                vtrue = w / vs
                LOG["t"].append(t); LOG["h"].append(h_value(vtrue, Pn, eb))
                LOG["htrue"].append(true_readiness(w, s, p, nom, Vbus, wind) - eb)
                Tt, Qt = M.rotor_aero(w, axial_inflow(w, s, p, wind), p)
                wtrue = np.array([Tt.sum(), (p.arm * np.sin(p.beta) * Tt).sum(), (-p.arm * np.cos(p.beta) * Tt).sum(), (p.sig * Qt).sum()]) / Ws
                LOG["werr"].append(np.linalg.norm(wtrue - wcn)); LOG["eR"].append(np.linalg.norm(eR))
                LOG["ez"].append(s["x"][2] - (r["z"] if r["mode"] == "att" else r["pos"][2]))
                LOG["w"].append(w.copy()); LOG["rate"].append(rate); LOG["act"].append(act)
                lo, hi = M.current_limits(w, Vbus, p)
                LOG["sat"].append(float(np.any((icmd >= hi - 1e-6) | (icmd <= lo + 1e-6))))
        # ------------------------------------------------ plant (p.dt = 0.5 ms, i.e. 2 kHz)
        wind = (gust.step(p.dt) if gust is not None else np.zeros(3))
        R = M.rotmat(s["q"])
        vax = axial_inflow(w, s, p, wind)
        Tt, Qt = M.rotor_aero(w, vax, p)
        lo, hi = M.current_limits(w, Vbus, p)
        cur += p.dt / p.tau_esc * (icmd - cur); cur = np.clip(cur, lo, hi)   # ESC current loop
        wdot = (p.kt * cur - Qt - p.kt * p.I0 * np.sign(w)) / p.Jr
        Ibatt = max(0.0, float(np.sum(cur * (p.kt * w + p.Rm * cur)) / max(Vbus, 1.0)))   # battery current, power balance
        # rigid body
        vair_w = s["v"] - wind; vair_b = R.T @ vair_w
        Fb = np.array([0, 0, Tt.sum()])
        Fb[:2] += -p.kh * Tt.sum() * vair_b[:2]
        Fw = R @ Fb - p.mass * M.G * np.array([0, 0, 1.0]) - p.cd_body * np.linalg.norm(vair_w) * vair_w
        Mb = np.array([(p.arm * np.sin(p.beta) * Tt).sum(), (-p.arm * np.cos(p.beta) * Tt).sum(), (p.sig * Qt).sum()])
        # sign convention: sig_i is the sign of the body yaw reaction (matches A row 4), so the
        # rotor spins about -sig_i e3; reaction torque on the body = sig_i (Q_i + J_r wdot_i).
        hrot = -p.Jr * np.sum(p.sig * w)                      # rotor angular momentum along e3
        Om = s["Om"]
        Mb = Mb - np.array([Om[1] * hrot, -Om[0] * hrot, 0.0]) + np.array([0, 0, p.Jr * np.sum(p.sig * wdot)])
        rr = ref(t)
        if "Mdist" in rr:                    # external moment, and the controller's 20 ms estimate of it
            Mb = Mb + rr["Mdist"]
            s["Mhat"] = s.get("Mhat", np.zeros(3)) + p.dt / 0.02 * (rr["Mdist"] - s.get("Mhat", np.zeros(3)))
        dOm = p.Ibinv @ (Mb - cross3(s["Om"], p.Ib @ s["Om"]))
        s["x"] = s["x"] + p.dt * s["v"]; s["v"] = s["v"] + p.dt * Fw / p.mass
        s["Om"] = s["Om"] + p.dt * dOm
        dq = 0.5 * M.quat_mult(s["q"], np.concatenate([[0.0], s["Om"]]))
        s["q"] = s["q"] + p.dt * dq; s["q"] /= np.linalg.norm(s["q"])
        w = w + p.dt * wdot
        tel_buf.append(w.copy()); tel_buf.pop(0)
        if not np.all(np.isfinite(s["x"])) or np.linalg.norm(s["x"]) > 1e3:
            break                            # vehicle lost; "completed" will be False
    L = {k: np.array(v) for k, v in LOG.items()}
    W = L["w"]; sign_changes = int(np.sum(np.abs(np.diff(np.sign(W), axis=0)) > 0))
    out = dict(certifiable=True, eb=eb, ld=ld, hop=hop, T0=T0, vsat=vs,
               hmin=float(L["h"].min()), htrue_min=float(L["htrue"].min()),
               hmean=float(L["h"].mean()), tbelow_true=float(np.sum(L["htrue"] < 0) * p.dt_ctrl),
               werr=float(np.sqrt(np.mean(L["werr"] ** 2))), eR=float(np.sqrt(np.mean(L["eR"] ** 2))),
               ez=float(np.sqrt(np.mean(L["ez"] ** 2))), sat=float(L["sat"].mean()),
               minabsw=float(np.abs(W).min() / vs), crossings=sign_changes,
               rate_peak=float(L["rate"][1:].max()), active=float(L["act"].mean()),
               completed=bool(len(L["t"]) >= int(round(T / p.dt_ctrl)) - 1))
    if log: out["log"] = L
    return out



def axial_inflow(w, s, p, wnd):
    """Freestream component through each disk along its induced-flow direction (m/s)."""
    R = M.rotmat(s["q"]); vb = R.T @ (s["v"] - wnd)
    rpos = np.column_stack([p.arm * np.cos(p.beta), p.arm * np.sin(p.beta), np.zeros(6)])
    Om = s["Om"]
    vrot_z = vb[2] + Om[0] * rpos[:, 1] - Om[1] * rpos[:, 0]
    return vrot_z * np.sign(w)


def true_readiness(w, s, p, nom, Vbus, wnd):
    """ln det of the TRUE wrench-rate ellipsoid in the nominal normalisation:
    true thrust/torque slopes (inflow, Reynolds) and true symmetric acceleration capacity
    (back-EMF current limit, friction, true drag)."""
    vax = axial_inflow(w, s, p, wnd)
    dT, dQ = M.rotor_slopes(w, vax, p)
    a = M.true_sac(w, vax, Vbus, p)
    vs, T0, Ws = nom["vsat"], nom["T0"], nom["Wscale"]
    Jt = np.vstack([dT, p.arm * np.sin(p.beta) * dT, -p.arm * np.cos(p.beta) * dT, p.sig * dQ]) * vs / Ws
    an = a * T0 / vs
    D = (Jt * an[None, :] ** 2) @ Jt.T
    sg, ld = np.linalg.slogdet(D)
    return ld if sg > 0 else -np.inf
