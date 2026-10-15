# Reference results

These are the experiment outputs from which the figures and tables of the paper were made. The scripts never write here. New runs go to `results/generated/`.

| File | Written by | Content |
|---|---|---|
| `sweep.json` | `exp_sweep.py` | Collective sweep and the floor chosen at each collective (Table I, Fig. 3). |
| `h_all.dat` | `exp_fig5.py` | Readiness histories of the four allocators (Fig. 4, right). |
| `rate_all.dat` | `exp_fig5.py` | Commanded-rate histories of the four allocators (Fig. 4, left). |
| `mc.json` | `exp_mc.py` | Randomised missions for both platforms (Table II). |
| `certmap.dat` | `exp_certmap.py` | Certifiability window over collective and moment amplitude (Fig. 5, left). |
| `ff.json` | `exp_ff.py` | Forward flight and certifiability as a function of airspeed (Fig. 5, right). |
| `delay.json` | `exp_delay.py` | Delay constants, certified ceiling and delay sweep (Fig. 6). |
| `robust.json` | `exp_robust.py` | Parameter-uncertainty campaign and ablation (Table III). |
| `kappa.json` | `exp_kappa.py` | Floor choice versus delay and parameter sensitivity. |
| `timing.json` | `exp_timing.py` | QP timing on the computer used for that run. |
| `hifi.json` | `hifi/run_hifi.py` | High-fidelity campaign, including the delay runs (Table IV). |

The `.dat` files are whitespace-separated with a header row. The JSON files load with Python's `json` module. `python make_paper_data.py --results reference` turns these files into the plotting tables in `data/`. The main README describes how a new run compares with them.
