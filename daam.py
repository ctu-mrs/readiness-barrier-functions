"""
daam.py -- drag-aware actuator-authority co-metric (DAAM), readiness h and its gradient.

    psi_i(v) = v_i^2 a_bar_i(v_i)^2
    D(v)     = 4 A diag(psi(v)) A^T          (smooth wherever every a_bar_i > 0)
    h(v)     = ln det D(v) - ell_bar         (the readiness barrier, ell_bar = floor)

D(v) describes the set of wrench rates the rotors can produce from the
current speeds, {x : x^T D^-1 x <= 1}; ln det D is twice its log-volume up to
a constant. The closed-form gradient is

    dh/dv_i = 8 v_i a_bar_i (tau_bar_i - 3 b_i v_i^2) / m_i * A_i^T D^-1 A_i.
"""
import numpy as np


def a_bar(v, P):
    """Symmetric acceleration capacity of each rotor (zero at or beyond saturation)."""
    return np.maximum(0.0, (P["tau_bar"] - P["b"] * v**2) / P["mass"])


def psi(v, P):
    return v**2 * a_bar(v, P)**2


def D_matrix(v, P):
    return 4.0 * (P["A"] * psi(v, P)[None, :]) @ P["A"].T


def h_value(v, P, ell_bar=0.0):
    """ln det D(v) - ell_bar; -inf when D is singular (e.g. too many rotors stopped)."""
    s, ld = np.linalg.slogdet(D_matrix(v, P))
    return (ld if s > 0 else -np.inf) - ell_bar


def grad_h(v, P):
    """Analytic gradient of h with respect to the rotor speeds."""
    A = P["A"]
    D = D_matrix(v, P)
    Dinv = np.linalg.inv(D)
    s = np.einsum('ij,jk,ki->i', A.T, Dinv, A)          # s_i = A_i^T D^{-1} A_i
    ab = a_bar(v, P)
    coeff = 8.0 * v * ab * (P["tau_bar"] - 3 * P["b"] * v**2) / P["mass"]
    return coeff * s


def grad_h_numeric(v, P, eps=1e-6):
    """Central-difference gradient, used only to check grad_h (see verify.py)."""
    g = np.zeros_like(v)
    for i in range(len(v)):
        vp = v.copy(); vp[i] += eps
        vm = v.copy(); vm[i] -= eps
        g[i] = (h_value(vp, P) - h_value(vm, P)) / (2 * eps)
    return g
