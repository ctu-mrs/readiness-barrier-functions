"""
delay_study.py (v2) — input delay, done honestly.

Proposition (paper): with applied torque tau*(v(t-tau_d)) and constants
  K1 = Lip(grad h . d),  K2 = sup||Hess h||,  V = max|vdot|,  Hbar = L_max-ellbar
on K_eps, the undershoot obeys  h(t) >= -eta(tau_d),
  eta(td) = (alpha*Hbar + K*V*td)*td + K*V*td/alpha,   K = K1 + K2*max(taubar/m).
The margined-floor remedy is FEASIBLE only while eta(td) <= L_op - ellbar; the
certified delay ceiling td* solves eta(td*) = L_op - ellbar.

This script computes (K1,K2,V,Hbar,td*), then sweeps the delay empirically to
actual failure, and applies the margined floor only where it is feasible.
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
from platforms import hexarotor
from daam import h_value, grad_h
from harden import traj
from final_exp import floor_rule
import allocators as AL
from project_paths import figure_data_path, figure_path
phi=lambda v: v*np.abs(v)
ALPHA=5.0

def constants(P, lo=0.30, hi=0.85, ntrial=1500, seed=0):
    rng=np.random.default_rng(seed); K1=K2=0.0
    for _ in range(ntrial):
        v=rng.uniform(lo,hi,size=P["n"])*P["vsat"]
        g0=grad_h(v,P); H=np.zeros((P["n"],P["n"])); gd=np.zeros(P["n"])
        f0=g0@(-(P["b"]*phi(v))/P["mass"])
        for i in range(P["n"]):
            vp=v.copy(); vp[i]+=1e-5
            gp=grad_h(vp,P)
            H[:,i]=(gp-g0)/1e-5
            gd[i]=(gp@(-(P["b"]*phi(vp))/P["mass"])-f0)/1e-5
        K2=max(K2,float(np.linalg.norm(H,np.inf))); K1=max(K1,float(np.linalg.norm(gd,1)))
    V=float(np.max(np.abs(P["b"])*hi**2/P["mass"]+P["tau_bar"]/P["mass"]))
    return K1,K2,V

def roll_delay(P,ellbar_enf,ellbar_meas,tau_d,T=2.0,dt=1e-3,amp=1.5,thrust=0.75):
    w,wd=traj(P,amp,thrust)
    v=AL.fiber_argmax_h(w(0.0),P,np.full(P["n"],P["vstar"]))
    kd=int(round(tau_d/dt)); buf=[np.zeros(P["n"])]*kd
    hs=[]
    for k in range(int(T/dt)):
        t=k*dt; wd_,wdd_=w(t),wd(t)
        tr_,_=AL.energy_qp_tau(v,wd_,wdd_,P)
        tau_now,_,_=AL.cbf_qp(v,wd_,wdd_,P,ellbar_enf,tau_ref=tr_,alpha=ALPHA)
        if kd>0: buf.append(tau_now); tau_app=buf.pop(0)
        else: tau_app=tau_now
        d=-(P["b"]*phi(v))/P["mass"]
        v=np.clip(v+dt*(d+tau_app/P["mass"]),-0.999*P["vsat"],0.999*P["vsat"])
        hs.append(h_value(v,P,ellbar_meas))
    return float(np.min(hs))

if __name__=="__main__":
    P=hexarotor(); eb,ld,hop=floor_rule(P,1.5,0.75,kappa=0.5)
    Lmax=-10.128; Hbar=Lmax-eb; avail=hop-eb
    K1,K2,V=constants(P); K=K1+K2*float(np.max(P["tau_bar"]/P["mass"]))
    eta=lambda td: (ALPHA*Hbar+K*V*td)*td + K*V*td/ALPHA
    tdstar=brentq(lambda t: eta(t)-avail,1e-7,0.1)
    print("K1=%.1f K2=%.1f K=%.1f V=%.2f Hbar=%.3f  avail=%.4f"%(K1,K2,K,V,Hbar,avail))
    print("certified delay ceiling td* = %.2f ms (eta(td*)=avail)"%(tdstar*1e3))
    # ---- empirical sweep to failure (plain floor) ----
    delays=[0.0,0.002,0.005,0.010,0.020,0.050,0.100,0.200]
    plain=[]
    print("\ntau_d[ms] | min h (plain floor) | eta bound (clipped)")
    for td in delays:
        mh=roll_delay(P,eb,eb,td); plain.append(mh)
        print("  %6.0f  |     %+8.4f      |  -%.3f%s"%(td*1e3,mh,min(eta(td),3),
              "  [remedy infeasible]" if eta(td)>avail else ""))
    # ---- margined runs, only where feasible ----
    marg_d=[td for td in [0.001,0.0015,tdstar*0.999] ]
    marg=[]
    print("\nmargined floor (feasible range only, td <= td*):")
    for td in marg_d:
        mh=roll_delay(P,eb+eta(td),eb,td); marg.append(mh)
        print("  td=%.2f ms  raised floor %.4f (<= L_op %.4f)  min h vs ORIGINAL floor: %+.4f"
              %(td*1e3,eb+eta(td),hop,mh))
    np.savetxt(figure_data_path("delay.csv"),
        np.column_stack([np.array(delays)*1e3,plain,[-min(eta(td),3) for td in delays]]),
        delimiter=",",header="tau_d_ms,minh_plain,eta_bound_clipped",comments="")
    np.savetxt(figure_data_path("delay_marg.csv"),
        np.column_stack([np.array(marg_d)*1e3,marg]),
        delimiter=",",header="tau_d_ms,minh_margined",comments="")
    # ---- figure ----
    plt.figure(figsize=(3.6,2.6))
    d_ms=np.array(delays)*1e3
    plt.plot(d_ms,plain,"o-",color="tab:red",ms=4,label="measured (plain floor)")
    tt=np.linspace(0,0.2,400)
    plt.plot(tt*1e3,np.maximum(-np.vectorize(eta)(tt),-3),"k--",lw=1,label=r"bound $-\eta(\tau_d)$")
    plt.plot(np.array(marg_d)*1e3,marg,"s",color="tab:blue",ms=5,label=r"margined ($\tau_d \leq \tau_d^\star$)")
    plt.axvline(tdstar*1e3,color="gray",ls=":",lw=1)
    plt.text(tdstar*1e3*1.3,-2.4,r"$\tau_d^\star$",fontsize=8,color="gray")
    plt.axhline(0,color="gray",lw=0.7); plt.ylim(-3.05,0.5)
    plt.xscale("symlog",linthresh=1)
    plt.xlabel(r"input delay $\tau_d$ [ms]"); plt.ylabel(r"$\min_t h$")
    plt.legend(fontsize=6,loc="lower left"); plt.tight_layout()
    plt.savefig(figure_path("fig_delay.pdf")); plt.close()
    print("\nwrote fig_delay.pdf, delay.csv, delay_marg.csv")
