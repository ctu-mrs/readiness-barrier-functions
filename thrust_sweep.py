"""thrust_sweep.py — where does each allocator lose authority? (certifiability-aware)"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from platforms import hexarotor
from gate import ell_dagger
from harden import run
from final_exp import floor_rule
from project_paths import figure_data_path, figure_path, result_path

P=hexarotor(); ld,_=ell_dagger(P); amp=1.5
ths=[0.6,0.7,0.8,0.9,1.0,1.1,1.2]
res={m:{"hmin":[],"TV":[],"werr":[]} for m in ["energy","greedy","lowpass","cbf"]}
cert=[]; floors=[]
print("thrust | cert  ellbar | energy / greedy / lowpass / cbf(ours)")
for th in ths:
    eb,ld_,hop=floor_rule(P,amp,th,kappa=0.5)
    ok = eb is not None
    cert.append(ok); floors.append(eb if ok else np.nan)
    if not ok:
        for m in res: res[m]["hmin"].append(np.nan); res[m]["TV"].append(np.nan); res[m]["werr"].append(np.nan)
        print("%6.1f |  NO (h_op=%.2f<=%.2f)"%(th,hop,ld_)); continue
    line=""
    for m in ["energy","greedy","lowpass","cbf"]:
        r=run(P,m,eb,dt=1e-3,T=2.0,amp=amp,thrust=th)
        res[m]["hmin"].append(r["hmin"]); res[m]["TV"].append(r["TV"]); res[m]["werr"].append(r["werr"])
        line+=" h%+.3f TV%5.2f e%.4f |"%(r["hmin"],r["TV"],r["werr"])
    print("%6.1f |  yes  %7.3f |%s"%(th,eb,line))

fig,ax=plt.subplots(1,2,figsize=(6.8,2.5))
col={"energy":"gray","greedy":"tab:red","lowpass":"tab:orange","cbf":"tab:blue"}
lab={"energy":"energy-QP","greedy":"greedy DAAM","lowpass":"low-pass DAAM","cbf":"proposed"}
for m in res:
    ax[0].plot(ths,res[m]["hmin"],"o-",color=col[m],ms=3.5,label=lab[m])
    ax[1].plot(ths,res[m]["werr"],"o-",color=col[m],ms=3.5,label=lab[m])
ax[0].axhline(0,color="k",ls="--",lw=0.9); ax[0].set_xlabel("collective thrust (x nominal)")
ax[0].set_ylabel(r"$\min_t h$"); ax[0].legend(fontsize=6); ax[0].set_title("authority floor",fontsize=8)
ax[1].set_yscale("log"); ax[1].set_xlabel("collective thrust (x nominal)")
ax[1].set_ylabel("RMS wrench error"); ax[1].set_title("task tracking",fontsize=8)
plt.tight_layout()
plt.savefig(figure_path("fig_thrust.pdf")); plt.close()
np.save(result_path("thrust.npy"),{"ths":ths,"res":res,"floors":floors},allow_pickle=True)
np.savetxt(figure_data_path("thrust_sweep.csv"),
    np.column_stack([ths,floors]+[res[m][k] for m in ["energy","greedy","lowpass","cbf"] for k in ["hmin","TV","werr"]]),
    delimiter=",",header="thrust,ellbar,"+",".join("%s_%s"%(m,k) for m in ["energy","greedy","lowpass","cbf"] for k in ["hmin","TV","werr"]),comments="")
print("wrote figdata/thrust_sweep.csv")
print("\nfigure -> fig_thrust.pdf")
