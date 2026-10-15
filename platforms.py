"""
platforms.py -- allocation matrices and motor parameters of the two test vehicles.

Conventions used throughout the code:

    wrench          w = A phi(v),   phi(v) = v * |v|   (signed-quadratic thrust/drag)
    motor dynamics  m_i dv_i/dt = -b_i v_i |v_i| + tau_i,   |tau_i| <= tau_bar_i
    acceleration    a_bar_i(v) = max(0, (tau_bar_i - b_i v_i^2) / m_i)
    capacity (SAC)

v holds the signed rotor speeds. All motor parameters are 1 (nondimensional
units), so the saturation speed is v_sat = 1 and the time unit is
T0 = m / sqrt(tau_bar b). The readiness sweet spot is v* = v_sat / sqrt(3), where
v^2 a_bar^2 is largest.

Each platform is returned as a plain dict, so the rest of the code does not
need to know which vehicle it is working on.
"""
import numpy as np


def _motor_params(n, tau_bar=1.0, b=1.0, m=1.0):
    """Identical motors: torque limit, drag coefficient and rotor inertia."""
    return dict(tau_bar=np.full(n, tau_bar),
                b=np.full(n, b),
                mass=np.full(n, m),
                vsat=np.sqrt(tau_bar / b),          # speed at which drag uses the whole torque
                vstar=np.sqrt(tau_bar / (3 * b)))   # readiness sweet spot


def hexarotor(L=0.25, kappa=0.016):
    """Coplanar hexarotor: n = 6 rotors, m = 4 wrench components, redundancy 2.

    Wrench order is [f_z, tau_x, tau_y, tau_z]. Arms at 30, 90, ..., 330 deg,
    arm length L, rotors spinning in alternating directions, yaw/thrust ratio kappa.
    """
    beta = np.deg2rad(np.arange(6) * 60.0 + 30.0)
    sig = np.array([+1, -1, +1, -1, +1, -1], float)   # spin direction of each rotor
    A = np.zeros((4, 6))
    A[0, :] = 1.0                # f_z
    A[1, :] = L * np.sin(beta)   # tau_x
    A[2, :] = -L * np.cos(beta)  # tau_y
    A[3, :] = kappa * sig        # tau_z (reaction torque)
    p = _motor_params(6)
    p.update(A=A, n=6, m=4, name="hexarotor")
    return p


def tilted_octorotor(L=0.30, theta_t=np.deg2rad(30.0), ctau=0.02):
    """Fully actuated tilted octorotor: n = 8, m = 6 (full wrench), redundancy 2.

    Rotor i sits at azimuth gamma_i on a ring of radius L. Its thrust axis is
    e_z tilted by +-theta_t about the local radial axis (alternating sign), which
    gives lateral force authority. The moment rows are divided by L so that
    force and moment rows have comparable scale.
    """
    n = 8
    gamma = np.arange(n) * (2 * np.pi / n)
    sig = np.array([(-1) ** i for i in range(n)], float)
    A = np.zeros((6, n))
    for i in range(n):
        p_i = L * np.array([np.cos(gamma[i]), np.sin(gamma[i]), 0.0])   # rotor position
        rad = np.array([np.cos(gamma[i]), np.sin(gamma[i]), 0.0])       # radial unit vector
        # rotate e_z about the radial axis by sig_i * theta_t (Rodrigues' formula)
        k = rad
        ez = np.array([0, 0, 1.0])
        ang = sig[i] * theta_t
        d_i = (ez * np.cos(ang)
               + np.cross(k, ez) * np.sin(ang)
               + k * (k @ ez) * (1 - np.cos(ang)))
        d_i /= np.linalg.norm(d_i)
        moment = np.cross(p_i, d_i) + ctau * sig[i] * d_i   # lever-arm moment + reaction torque
        A[0:3, i] = d_i
        A[3:6, i] = moment / L
    p = _motor_params(n)
    p.update(A=A, n=n, m=6, name="tilted_octorotor")
    return p
