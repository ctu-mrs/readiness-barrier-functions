"""
sweep.py — demand sweep: torque amplitude vs allocator behavior.
Shows greedy DAAM total variation blowing up (stratum jumps switching on) while
the CBF filter stays bounded and floor-respecting. Produces figure + table.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from platforms import hexarotor
from daam import h_value
from gate import ell_dagger
import allocators as AL
from project_paths import figure_path, result_path

phi = lambda v: v * np.abs(v)


def run_amp(P, amp, ellbar, T=2.0, dt=2e-3, thrust=0.5, method="cbf", seed=0):
    phistar = P["vstar"]*abs(P["vstar"]); fz_nom = P["n"]*phistar; tau_ref = 0.25*phistar
    w  = lambda t: np.array([thrust*fz_nom, amp*tau_ref*np.sin(2*np.pi*t/2.0), 0., 0.])
    wd = lambda t: np.array([0., amp*tau_ref*(2*np.pi/2.0)*np.cos(2*np.pi*t/2.0), 0., 0.])
    v = AL.fiber_argmax_h(w(0.0), P, np.full(P["n"], P["vstar"]))
    N=int(T/dt); vsp_f=v.copy(); vsp_hold=v.copy()
    UPD=max(1,int(0.02/dt))   # global setpoint recomputed at 50 Hz
    TVc=0.0; werr=[]; hmin=np.inf; vprev_sp=v.copy(); maxrate=0.0
    for k in range(N):
        t=k*dt; wd_,wdd_=w(t),wd(t); vsp=v.copy()
        if method=="energy":
            tau,_=AL.energy_qp_tau(v,wd_,wdd_,P)
        elif method=="greedy":
            if k%UPD==0: vsp_hold=AL.fiber_argmax_h_global(wd_,P,seed=seed,restarts=3)
            vsp=vsp_hold; tau=AL.track_setpoint(v,vsp,P)
        elif method=="lowpass":
            if k%UPD==0: vsp_hold=AL.fiber_argmax_h_global(wd_,P,seed=seed,restarts=3)
            a=(UPD*dt)/(0.05+UPD*dt); vsp_f=(1-a)*vsp_f+a*vsp_hold; vsp=vsp_f
            tau=AL.track_setpoint(v,vsp,P)
        elif method=="cbf":
            tr,_=AL.energy_qp_tau(v,wd_,wdd_,P)
            tau,_,_=AL.cbf_qp(v,wd_,wdd_,P,ellbar,tau_ref=tr)
        TVc+=np.sum(np.abs(vsp-vprev_sp)); vprev_sp=vsp
        d=-(P["b"]*phi(v))/P["mass"]; vn=np.clip(v+dt*(d+tau/P["mass"]),-0.999*P["vsat"],0.999*P["vsat"])
        maxrate=max(maxrate,np.max(np.abs(vn-v))/dt); v=vn
        werr.append(np.linalg.norm(P["A"]@phi(v)-wd_)); hmin=min(hmin,h_value(v,P,ellbar))
    return dict(TVc=TVc,werr_rms=float(np.sqrt(np.mean(np.array(werr)**2))),
                hmin=hmin,maxrate=maxrate)


def main():
    P=hexarotor(); ldag,_=ell_dagger(P); ellbar=ldag+0.4   # fixed floor, above saddle
    amps=[1.0,2.0,3.0,4.0]
    methods=["energy","greedy","lowpass","cbf"]
    data={m:{"TVc":[],"werr":[],"hmin":[],"rate":[]} for m in methods}
    print("floor ellbar = ell_dagger + 0.4 = %.3f  (saddle %.3f)\n"%(ellbar,ldag))
    print("amp | " + " | ".join("%-28s"%m for m in methods))
    for a in amps:
        row=[]
        for m in methods:
            r=run_amp(P,a,ellbar,method=m)
            data[m]["TVc"].append(r["TVc"]); data[m]["werr"].append(r["werr_rms"])
            data[m]["hmin"].append(r["hmin"]); data[m]["rate"].append(r["maxrate"])
            row.append("TV%7.1f wErr%.3f hmin%6.2f"%(r["TVc"],r["werr_rms"],r["hmin"]))
        print("%.1f | "%a + " | ".join(row))
    # figure: TVc and hmin vs amplitude
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.6))
    col={"energy":"gray","greedy":"tab:red","lowpass":"tab:orange","cbf":"tab:blue"}
    for m in methods:
        ax[0].plot(amps,data[m]["TVc"],"o-",color=col[m],ms=3,label=m)
        ax[1].plot(amps,data[m]["hmin"],"o-",color=col[m],ms=3,label=m)
    ax[0].set_yscale("log"); ax[0].set_xlabel("torque amplitude (× ref)")
    ax[0].set_ylabel("allocation total variation"); ax[0].legend(fontsize=6)
    ax[1].axhline(0,color="k",lw=0.8,ls="--"); ax[1].set_xlabel("torque amplitude (× ref)")
    ax[1].set_ylabel(r"$\min_t\,(\ln\det D-\bar\ell)$")
    ax[1].text(1.05,0.15,"authority floor",fontsize=6)
    plt.tight_layout()
    plt.savefig(figure_path("fig_sweep.pdf")); plt.close()
    print("\nfigure -> figs/fig_sweep.pdf")
    np.save(result_path("sweep.npy"),{"amps":amps,"data":data,"ellbar":ellbar,"ldag":ldag},allow_pickle=True)


if __name__=="__main__":
    main()
