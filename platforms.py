"""
platforms.py — allocation matrices A (m x n) and motor parameters for the
overactuated multirotor test platforms used in the readiness-CBF paper.

Convention: wrench w = A phi(v),  phi(v)=v*|v|,  v = signed spin rates.
Motor model (per rotor):  m_i vdot_i = -b_i v_i|v_i| + tau_i,  |tau_i| <= tau_bar_i.
SAC:  a_bar_i(v) = max(0, (tau_bar_i - b_i v_i^2)/m_i).

All params returned in a dict so scripts stay platform-agnostic.
"""
import numpy as np


def _motor_params(n, tau_bar=1.0, b=1.0, m=1.0):
    return dict(tau_bar=np.full(n, tau_bar),
                b=np.full(n, b),
                mass=np.full(n, m),
                vsat=np.sqrt(tau_bar / b),          # critical spin
                vstar=np.sqrt(tau_bar / (3 * b)))   # readiness sweet-spot


def hexarotor(L=0.25, kappa=0.016):
    """Planar hexarotor: n=6, m=4, wrench = [f_z, tau_x, tau_y, tau_z]. Redundancy 2."""
    beta = np.deg2rad(np.arange(6) * 60.0 + 30.0)
    sig = np.array([+1, -1, +1, -1, +1, -1], float)   # alternating spin
    A = np.zeros((4, 6))
    A[0, :] = 1.0                # f_z
    A[1, :] = L * np.sin(beta)   # tau_x
    A[2, :] = -L * np.cos(beta)  # tau_y
    A[3, :] = kappa * sig        # tau_z
    p = _motor_params(6)
    p.update(A=A, n=6, m=4, name="hexarotor")
    return p


def tilted_octorotor(L=0.30, theta_t=np.deg2rad(30.0), ctau=0.02):
    """
    Fully-actuated tilted octorotor: n=8, m=6, full 6-D wrench. Redundancy 2.
    Rotor i at azimuth gamma_i on a ring of radius L; thrust tilted by theta_t
    about the local radial axis, alternating sign for full actuation.
    Moment rows scaled by 1/L (characteristic-length nondimensionalization).
    """
    n = 8
    gamma = np.arange(n) * (2 * np.pi / n)
    sig = np.array([(-1) ** i for i in range(n)], float)
    A = np.zeros((6, n))
    for i in range(n):
        p_i = L * np.array([np.cos(gamma[i]), np.sin(gamma[i]), 0.0])
        # radial axis at rotor i
        rad = np.array([np.cos(gamma[i]), np.sin(gamma[i]), 0.0])
        # tilt e_z about radial axis by sig_i*theta_t (Rodrigues)
        k = rad
        ez = np.array([0, 0, 1.0])
        ang = sig[i] * theta_t
        d_i = (ez * np.cos(ang)
               + np.cross(k, ez) * np.sin(ang)
               + k * (k @ ez) * (1 - np.cos(ang)))
        d_i /= np.linalg.norm(d_i)
        moment = np.cross(p_i, d_i) + ctau * sig[i] * d_i
        A[0:3, i] = d_i
        A[3:6, i] = moment / L        # nondimensionalize moment rows
    p = _motor_params(n)
    p.update(A=A, n=n, m=6, name="tilted_octorotor")
    return p
