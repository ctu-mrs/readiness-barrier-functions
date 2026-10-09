"""
daam.py — DAAM co-metric, readiness scalar h, and its closed-form gradient.

D(v)   = 4 A diag(psi) A^T,  psi_i = v_i^2 a_bar_i(v_i)^2   (C-infinity on the regular set)
h(v)   = logdet D(v) - ell_bar
grad h : dh/dv_i = 8 v_i a_bar_i (tau_bar_i - 3 b_i v_i^2)/m_i * (A_i^T D^{-1} A_i)
"""
import numpy as np


def a_bar(v, P):
    return np.maximum(0.0, (P["tau_bar"] - P["b"] * v**2) / P["mass"])


def psi(v, P):
    return v**2 * a_bar(v, P)**2


def D_matrix(v, P):
    return 4.0 * (P["A"] * psi(v, P)[None, :]) @ P["A"].T


def h_value(v, P, ell_bar=0.0):
    s, ld = np.linalg.slogdet(D_matrix(v, P))
    return (ld if s > 0 else -np.inf) - ell_bar


def grad_h(v, P):
    """Closed-form analytic gradient (paper Eq. for grad h)."""
    A = P["A"]
    D = D_matrix(v, P)
    Dinv = np.linalg.inv(D)
    s = np.einsum('ij,jk,ki->i', A.T, Dinv, A)          # s_i = A_i^T D^{-1} A_i
    ab = a_bar(v, P)
    coeff = 8.0 * v * ab * (P["tau_bar"] - 3 * P["b"] * v**2) / P["mass"]
    return coeff * s


def grad_h_numeric(v, P, eps=1e-6):
    g = np.zeros_like(v)
    for i in range(len(v)):
        vp = v.copy(); vp[i] += eps
        vm = v.copy(); vm[i] -= eps
        g[i] = (h_value(vp, P) - h_value(vm, P)) / (2 * eps)
    return g
