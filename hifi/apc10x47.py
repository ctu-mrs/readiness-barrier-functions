"""
apc10x47.py -- measured aerodynamic data of the APC 10x4.7 Slow Flyer propeller
(UIUC Propeller Database, Vol. 1, Brandt & Selig; files apcsf_10x4.7_*.txt,
https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html) and a fast
2-D interpolant C_T(J, rpm), C_P(J, rpm).

Conventions (UIUC): T = C_T rho n^2 D^4,  P = C_P rho n^3 D^5,  Q = P/(2 pi n),
J = V/(n D), n in rev/s, D = 0.254 m.

Every advance-ratio table is checked on load against its own efficiency column,
eta = J C_T / C_P, which detects transcription errors in any of the three columns.
"""
import numpy as np

D = 0.254           # m
RHO = 1.225         # kg/m^3
MASS = 0.0119       # kg, manufacturer (motionrc.com product sheet, 11.90 g)

STATIC = np.array([  # rpm, CT, CP   (apcsf_10x4.7_static_kt0835.txt)
    [2377, 0.1059, 0.0431], [2676, 0.1079, 0.0437], [2947, 0.1079, 0.0437],
    [3234, 0.1104, 0.0444], [3494, 0.1117, 0.0450], [3762, 0.1143, 0.0460],
    [4029, 0.1158, 0.0466], [4319, 0.1177, 0.0474], [4590, 0.1200, 0.0484],
    [4880, 0.1223, 0.0494], [5147, 0.1237, 0.0500], [5417, 0.1252, 0.0508],
    [5715, 0.1263, 0.0513], [5960, 0.1278, 0.0520], [6226, 0.1286, 0.0524],
    [6528, 0.1299, 0.0531]])

DYN = {  # rpm: rows J, CT, CP, eta
    4014: np.array([  # apcsf_10x4.7_kt0836_4014.txt
        [0.142, 0.0994, 0.0451, 0.312], [0.177, 0.0944, 0.0446, 0.375], [0.216, 0.0885, 0.0436, 0.438],
        [0.255, 0.0829, 0.0427, 0.495], [0.289, 0.0770, 0.0414, 0.537], [0.325, 0.0719, 0.0407, 0.574],
        [0.361, 0.0652, 0.0392, 0.600], [0.399, 0.0582, 0.0375, 0.620], [0.438, 0.0502, 0.0352, 0.624],
        [0.468, 0.0441, 0.0335, 0.615], [0.510, 0.0343, 0.0308, 0.567], [0.534, 0.0288, 0.0292, 0.526],
        [0.577, 0.0178, 0.0258, 0.398], [0.612, 0.0083, 0.0228, 0.222], [0.647, -0.0011, 0.0199, -0.036],
        [0.685, -0.0122, 0.0162, -0.513], [0.719, -0.0218, 0.0130, -1.204]]),
    5018: np.array([  # apcsf_10x4.7_kt0837_5018.txt
        [0.115, 0.1077, 0.0476, 0.259], [0.143, 0.1038, 0.0472, 0.314], [0.172, 0.0993, 0.0465, 0.368],
        [0.199, 0.0953, 0.0459, 0.414], [0.231, 0.0908, 0.0452, 0.463], [0.262, 0.0856, 0.0442, 0.508],
        [0.289, 0.0811, 0.0433, 0.542], [0.318, 0.0766, 0.0426, 0.573], [0.349, 0.0709, 0.0413, 0.600],
        [0.376, 0.0660, 0.0401, 0.620], [0.408, 0.0597, 0.0383, 0.635], [0.432, 0.0549, 0.0370, 0.641],
        [0.460, 0.0486, 0.0351, 0.637], [0.487, 0.0428, 0.0334, 0.623], [0.519, 0.0352, 0.0312, 0.586],
        [0.547, 0.0280, 0.0291, 0.528], [0.576, 0.0206, 0.0268, 0.443]]),
    6020: np.array([  # apcsf_10x4.7_rd0840_6020.txt
        [0.408, 0.0624, 0.0397, 0.641], [0.432, 0.0576, 0.0383, 0.649], [0.458, 0.0519, 0.0366, 0.650],
        [0.480, 0.0471, 0.0351, 0.645], [0.507, 0.0405, 0.0331, 0.622], [0.528, 0.0354, 0.0315, 0.594],
        [0.551, 0.0296, 0.0297, 0.549], [0.574, 0.0235, 0.0279, 0.484], [0.601, 0.0160, 0.0254, 0.378],
        [0.623, 0.0100, 0.0234, 0.265], [0.652, 0.0017, 0.0207, 0.052], [0.671, -0.0039, 0.0189, -0.139],
        [0.700, -0.0123, 0.0160, -0.540], [0.719, -0.0178, 0.0141, -0.906], [0.747, -0.0265, 0.0112, -1.773],
        [0.774, -0.0347, 0.0085, -3.171]]),
    6513: np.array([  # apcsf_10x4.7_rd0842_6513.txt
        [0.378, 0.0698, 0.0422, 0.625], [0.400, 0.0654, 0.0409, 0.639], [0.420, 0.0616, 0.0398, 0.650],
        [0.440, 0.0570, 0.0383, 0.655], [0.465, 0.0517, 0.0367, 0.656], [0.486, 0.0469, 0.0351, 0.648],
        [0.507, 0.0420, 0.0336, 0.632], [0.531, 0.0359, 0.0318, 0.600], [0.551, 0.0306, 0.0302, 0.560],
        [0.579, 0.0229, 0.0277, 0.480], [0.599, 0.0176, 0.0260, 0.406], [0.618, 0.0120, 0.0242, 0.307],
        [0.646, 0.0044, 0.0217, 0.130], [0.665, -0.0012, 0.0197, -0.042], [0.691, -0.0089, 0.0172, -0.359],
        [0.708, -0.0139, 0.0155, -0.634], [0.735, -0.0219, 0.0128, -1.257], [0.755, -0.0281, 0.0106, -2.001]]),
}

GEOM = np.array([  # r/R, c/R, beta[deg]   (apcsf_10x4.7_geom.txt)
    [0.15, 0.109, 21.11], [0.20, 0.132, 23.90], [0.25, 0.156, 24.65], [0.30, 0.176, 24.11],
    [0.35, 0.193, 22.78], [0.40, 0.206, 21.01], [0.45, 0.216, 19.00], [0.50, 0.223, 17.06],
    [0.55, 0.226, 15.33], [0.60, 0.225, 13.82], [0.65, 0.219, 12.51], [0.70, 0.210, 11.36],
    [0.75, 0.197, 10.27], [0.80, 0.179, 9.32], [0.85, 0.157, 8.36], [0.90, 0.130, 7.27],
    [0.95, 0.087, 6.15], [1.00, 0.042, 5.04]])


def check_tables():
    """eta column vs J*CT/CP. Each column is printed with 3-4 decimals, so the check
    allows the propagated rounding error; returns the worst ratio error/allowance (<1)."""
    worst = 0.0
    for rpm, t in DYN.items():
        J, ct, cp, et = t.T
        eta = J * ct / cp
        allow = np.abs(eta) * (5e-4 / J + 5e-5 / np.maximum(np.abs(ct), 1e-4) + 5e-5 / cp) + 5e-4
        worst = max(worst, float(np.max(np.abs(eta - et) / allow)))
    assert worst < 1.0, "UIUC table transcription check failed (%.3f)" % worst
    return worst


def prop_inertia():
    """Blade mass per unit span ~ c^2 (constant thickness ratio), 15% of mass in the hub."""
    R = D / 2; rR, c = GEOM[:, 0], GEOM[:, 1]; w = c ** 2
    Ib = 0.85 * MASS * np.trapezoid(w * (rR * R) ** 2, rR) / np.trapezoid(w, rR)
    Ih = 0.15 * MASS * 0.5 * (0.15 * R) ** 2
    return float(Ib + Ih)


# ------------------------------------------------------------- interpolant
_RPM_NODES = np.array([4014.0, 5018.0, 6020.0, 6513.0])
_JGRID = np.linspace(0.0, 0.80, 81)


def _static_at(rpm):
    r = np.clip(rpm, STATIC[0, 0], STATIC[-1, 0])
    return np.interp(r, STATIC[:, 0], STATIC[:, 1]), np.interp(r, STATIC[:, 0], STATIC[:, 2])


def _curve(rpm, col):
    """C(J) at a tabulated rpm: static point at J=0 joined to the wind-tunnel table
    (monotone piecewise-linear), linear extrapolation beyond the last J."""
    t = DYN[int(rpm)]; cs = _static_at(rpm)[0 if col == 1 else 1]
    J = np.concatenate([[0.0], t[:, 0]]); C = np.concatenate([[cs], t[:, col]])
    out = np.interp(_JGRID, J, C)
    hi = _JGRID > J[-1]
    slope = (C[-1] - C[-2]) / (J[-1] - J[-2])
    out[hi] = C[-1] + slope * (_JGRID[hi] - J[-1])
    return out


_CT = np.array([_curve(r, 1) for r in _RPM_NODES])   # (4, 81)
_CP = np.array([_curve(r, 2) for r in _RPM_NODES])
# static-only Reynolds trend below 4014 rpm: scale the 4014 curve by the static ratio
def coeffs(rpm, J):
    """Vectorised C_T, C_P at |rpm| and advance ratio J (J<0: descent-like inflow,
    held at the static value: vortex-ring states are outside the data)."""
    rpm = np.abs(np.atleast_1d(np.asarray(rpm, float))); J = np.clip(np.atleast_1d(np.asarray(J, float)), 0.0, 0.80)
    rpm, J = np.broadcast_arrays(rpm, J); rpm = rpm.copy(); J = J.copy()
    rr = np.clip(rpm, _RPM_NODES[0], _RPM_NODES[-1])
    k = np.clip(np.searchsorted(_RPM_NODES, rr) - 1, 0, len(_RPM_NODES) - 2)
    a = (rr - _RPM_NODES[k]) / (_RPM_NODES[k + 1] - _RPM_NODES[k])
    jj = J / 0.01; j0 = np.clip(jj.astype(int), 0, 79); b = jj - j0
    def bil(G):
        g0 = G[k, j0] * (1 - b) + G[k, j0 + 1] * b
        g1 = G[k + 1, j0] * (1 - b) + G[k + 1, j0 + 1] * b
        return g0 * (1 - a) + g1 * a
    ct, cp = bil(_CT), bil(_CP)
    lowr = rpm < _RPM_NODES[0]  # Reynolds correction from the static sweep
    if np.any(lowr):
        cts, cps = _static_at(rpm[lowr]); ct4, cp4 = _static_at(_RPM_NODES[0])
        ct[lowr] *= cts / ct4; cp[lowr] *= cps / cp4
    return ct, cp


if __name__ == "__main__":
    print("eta check, worst |eta - J CT/CP| =", check_tables())
    print("prop inertia [kg m^2] =", prop_inertia())
    for rpm in [3000, 4014, 5018, 6000, 6513]:
        print(rpm, [tuple(np.round(coeffs(rpm, j), 4)) for j in [0, 0.2, 0.4, 0.6]])
