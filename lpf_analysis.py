"""
lpf_analysis.py — is the low-pass critique a strawman? Measure what physically
matters: peak commanded rate (saturation risk), phase lag, and floor-violation
DURATION, not just total variation.
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from platforms import hexarotor
from daam import h_value
from harden import traj
from final_exp import floor_rule
import allocators as AL
from project_paths import figure_data_path, figure_path
phi=lambda v: v*np.abs(v)

def roll(P,method,ellbar,amp=1.5,thrust=0.75,T=2.0,dt=1e-3,tau_f=0.05,seed=0):
    w,wd=traj(P,amp,thrust)
    v=AL.fiber_argmax_h(w(0.0),P,np.full(P["n"],P["vstar"]))
    N=int(T/dt); UPD=max(1,int(0.02/dt)); hold=v.copy(); vlp=v.copy()
    ts=[];sp=[];hs=[]
    for k in range(N):
        t=k*dt; wd_,wdd_=w(t),wd(t); s=v.copy()
        if method=="greedy":
            if k%UPD==0: hold=AL.fiber_argmax_h_global(wd_,P,seed=seed,restarts=3)
            s=hold; tau=AL.track_setpoint(v,s,P)
        elif method=="lowpass":
            if k%UPD==0: hold=AL.fiber_argmax_h_global(wd_,P,seed=seed,restarts=3)
            a=dt/(tau_f+dt); vlp=(1-a)*vlp+a*hold; s=vlp; tau=AL.track_setpoint(v,s,P)
        else:
            tr_,_=AL.energy_qp_tau(v,wd_,wdd_,P)
            tau,_,_=AL.cbf_qp(v,wd_,wdd_,P,ellbar,tau_ref=tr_)
        d=-(P["b"]*phi(v))/P["mass"]
        v=np.clip(v+dt*(d+tau/P["mass"]),-0.999*P["vsat"],0.999*P["vsat"])
        ts.append(t);sp.append(s.copy());hs.append(h_value(v,P,ellbar))
    ts=np.array(ts);sp=np.array(sp);hs=np.array(hs)
    rate=np.abs(np.diff(sp,axis=0)).max(axis=1)/dt      # peak per-step commanded rate
    return ts,sp,hs,rate

if __name__=="__main__":
    P=hexarotor(); eb,ld,hop=floor_rule(P,1.5,0.75,kappa=0.5); dt=1e-3
    res={}
    print("method   | max cmd rate [1/s] | TV    | floor viol. time [s] | min h")
    for m in ["greedy","lowpass","cbf"]:
        ts,sp,hs,rate=roll(P,m,eb)
        viol=float(np.sum(hs<0)*dt); TV=float(np.abs(np.diff(sp,axis=0)).sum())
        res[m]=(ts,sp,hs,rate)
        print("%-8s | %18.1f | %5.1f | %20.3f | %+.3f"%(m,rate.max(),TV,viol,hs.min()))
    # save data + figure
    np.savetxt(figure_data_path("lpf_h.csv"),
        np.column_stack([res["greedy"][0],res["greedy"][2],res["lowpass"][2],res["cbf"][2]]),
        delimiter=",",header="t,h_greedy,h_lowpass,h_cbf",comments="")
    clamp=lambda r: np.maximum(r,1e-2)   # log-axis safe
    np.savetxt(figure_data_path("lpf_rate.csv"),
        np.column_stack([res["greedy"][0][1:],clamp(res["greedy"][3]),clamp(res["lowpass"][3]),clamp(res["cbf"][3])]),
        delimiter=",",header="t,rate_greedy,rate_lowpass,rate_cbf",comments="")
    fig,ax=plt.subplots(1,2,figsize=(7.0,2.3))
    col={"greedy":"tab:red","lowpass":"tab:orange","cbf":"tab:blue"}
    lab={"greedy":"greedy","lowpass":"low-pass (50 ms)","cbf":"proposed"}
    for m in res:
        ax[0].semilogy(res[m][0][1:],np.maximum(res[m][3],1e-2),color=col[m],lw=0.9,label=lab[m])
        ax[1].plot(res[m][0],res[m][2],color=col[m],lw=0.9,label=lab[m])
    # physically achievable rate: max |vdot| = (taubar + b v^2)/m <= 2 in normalized units
    ax[0].axhline(2.0,color="k",ls="--",lw=0.8)
    ax[0].text(0.05,2.6,"achievable rate bound",fontsize=6)
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"peak commanded rate $\max_i|\dot v_i^{\rm cmd}|$")
    ax[0].legend(fontsize=6); ax[1].axhline(0,color="k",ls="--",lw=0.8)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel(r"$h(t)$")
    plt.tight_layout(); plt.savefig(figure_path("fig_lpf.pdf")); plt.close()
    print("wrote fig_lpf.pdf + lpf_*.csv")
