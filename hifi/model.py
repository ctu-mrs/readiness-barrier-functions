"""
model.py -- high-fidelity hexarotor plant built from measured component data, and the
nominal quadratic model a practitioner would identify from the same data.

Components
  propeller  APC 10x4.7SF          UIUC data (apc10x47.py): C_T, C_P vs (J, rpm)
  motor      SunnySky X2212-13 980KV datasheet: KV=980 rpm/V, R=0.133 Ohm,
             I0=0.3 A (10 V), 15 A continuous (flyingtech.co.uk product sheet)
  ESC        current (torque) mode, first-order loop tau_esc, duty saturation:
             the current is bounded by +-I_max AND by the voltage limit (V_bus -+ k_e w)/R
  battery    3S LiPo, V_oc = 11.4 V, R_int = 25 mOhm  (assumption, typical 5 Ah 3S pack)
  rotor      J_r = prop (UIUC chord + measured 11.9 g) + motor bell (22 g shell, r=12.5 mm)
  airframe   planar hexarotor, arm 0.25 m, alternating spin (same A as the paper)
Everything the paper's barrier does NOT model is present here: Reynolds (rpm) and
advance-ratio dependence of C_T and C_P, back-EMF torque limit, battery sag, Coulomb
friction, ESC lag, rotor H-force, body drag, telemetry noise/quantisation, sample-and-hold.
"""
import numpy as np
import apc10x47 as prop

G = 9.81
PI2 = 2 * np.pi


class Params:
    def __init__(self, mass=1.8, seed=0):
        # ---- motor (datasheet)
        self.KV = 980.0
        self.kt = 60.0 / (PI2 * self.KV)          # N m / A  (= k_e in V s/rad)
        self.Rm = 0.133 + 0.020                   # winding + ESC/wiring, Ohm
        self.I0 = 0.3
        self.Imax = 15.0
        self.tau_esc = 0.002                      # s, current-loop time constant
        # ---- battery
        self.Voc = 11.4; self.Rint = 0.025
        # ---- rotor
        self.Jr = prop.prop_inertia() + 0.022 * 0.0125 ** 2
        # ---- airframe
        self.n = 6; self.arm = 0.25
        self.beta = np.deg2rad(np.arange(6) * 60.0 + 30.0)
        self.sig = np.array([+1, -1, +1, -1, +1, -1], float)
        self.mass = mass
        self.Ib = np.diag([0.023, 0.023, 0.040]) * (mass / 1.8); self.Ibinv = np.linalg.inv(self.Ib)
        self.cd_body = 0.03          # N/(m/s)^2: 0.5 rho Cd S with Cd S ~ 0.05 m^2
        self.kh = 0.012              # rotor H-force: F_h = -kh * T * v_inplane  [s/m]
        # ---- sensing / timing
        self.tel_noise = 0.005       # 0.5 % rpm noise (eRPM telemetry)
        self.tel_quant = 10.0 * PI2 / 60.0   # 10 rpm quantisation
        self.dt = 5e-4               # plant integration step (ESC tau/4)
        self.dt_ctrl = 2e-3          # 500 Hz attitude controller + allocator
        self.rng = np.random.default_rng(seed)


def identify_nominal(p, rpm_ref=4000.0):
    """What the barrier is given: quadratic thrust/drag fitted to the STATIC data at
    rpm_ref, torque limit from the current limit, inertia from the estimate."""
    ct, cp = prop.coeffs(rpm_ref, 0.0)
    kf = float(ct[0]) * prop.RHO * prop.D ** 4 / PI2 ** 2          # T = kf w^2
    km = float(cp[0]) * prop.RHO * prop.D ** 5 / PI2 ** 3          # Q = km w^2  (= b)
    tbar = p.kt * (p.Imax - p.I0)
    vsat = np.sqrt(tbar / km)
    T0 = p.Jr / np.sqrt(tbar * km)
    A = np.zeros((4, 6))      # normalised wrench map, rows [f_z, M_x, M_y, M_z] / (kf vsat^2)
    A[0] = 1.0; A[1] = p.arm * np.sin(p.beta); A[2] = -p.arm * np.cos(p.beta); A[3] = (km / kf) * p.sig
    return dict(kf=kf, km=km, tbar=tbar, vsat=vsat, T0=T0, A=A, Wscale=kf * vsat ** 2)


def rotor_aero(w, vax, p):
    """True thrust [N], drag torque [N m] and their slopes for rotor speeds w [rad/s]
    and axial inflow vax [m/s] (positive = along the induced flow, i.e. climb-like)."""
    n = np.abs(w) / PI2 + 1e-9
    J = vax / (n * prop.D)
    ct, cp = prop.coeffs(n * 60.0, J)
    T = ct * prop.RHO * n ** 2 * prop.D ** 4 * np.sign(w)          # symmetric 3D-prop assumption
    Q = cp * prop.RHO * n ** 2 * prop.D ** 5 / PI2 * np.sign(w)
    return T, Q


def rotor_slopes(w, vax, p, e=0.5):
    Tp, Qp = rotor_aero(w + e, vax, p); Tm, Qm = rotor_aero(w - e, vax, p)
    return (Tp - Tm) / (2 * e), (Qp - Qm) / (2 * e)


def current_limits(w, Vbus, p):
    """Achievable motor current: +-I_max intersected with the duty-cycle (voltage) limit."""
    emf = p.kt * w
    hi = np.minimum(p.Imax, (Vbus - emf) / p.Rm)
    lo = np.maximum(-p.Imax, (-Vbus - emf) / p.Rm)
    return lo, hi


def true_sac(w, vax, Vbus, p):
    """True symmetric acceleration capacity [rad/s^2]: min of the deliverable spin-up and
    spin-down accelerations given the voltage-limited current, friction and true drag."""
    lo, hi = current_limits(w, Vbus, p)
    _, Q = rotor_aero(w, vax, p)
    fr = p.kt * p.I0 * np.sign(w)
    up = (p.kt * hi - Q - fr) / p.Jr
    dn = (p.kt * lo - Q - fr) / p.Jr
    # "up" = towards larger |w| for w>0; mirror for w<0
    a_plus = np.where(w >= 0, up, -dn); a_minus = np.where(w >= 0, -dn, up)
    return np.maximum(np.minimum(a_plus, a_minus), 0.0)


def rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mult(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Dryden:
    """Low-altitude Dryden turbulence (MIL-F-8785C, first-order shaping), body-independent."""
    def __init__(self, sigma, L, V, rng):
        self.s = sigma; self.L = L; self.V = max(V, 1.0); self.rng = rng; self.x = np.zeros(3)
    def step(self, dt):
        a = self.V / self.L
        self.x += dt * (-a * self.x) + self.s * np.sqrt(2 * a * dt) * self.rng.standard_normal(3)
        return self.x.copy()
