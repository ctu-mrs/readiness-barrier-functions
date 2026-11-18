# Readiness Barrier Functions

Simulation code for the paper *"Readiness Barrier Functions: Forward-Invariant Control Authority for Overactuated Multirotor Allocation"*, accepted for publication in the *IEEE Robotics and Automation Letters* (RA-L). If you use this code, please cite the paper (see [Citation](#citation)).

The paper uses the log-determinant of a drag-aware actuator-authority co-metric (DAAM) as a control barrier function. A torque-level quadratic program keeps this readiness measure above a floor while the vehicle tracks its wrench command. This repository contains the models, the allocators and every experiment in the paper. It also contains the data and TikZ sources of the figures, and a script that rebuilds those data from the experiment outputs.

All results come from simulation. The code is research code, not a flight controller.

## The model in brief

Each rotor *i* has a signed speed $v_i$, produces thrust and drag torque proportional to $v_i|v_i|$, and obeys

$$m_i\dot v_i = -b_i v_i|v_i| + \tau_i,\qquad |\tau_i|\le\bar\tau_i .$$

The wrench is $w = A\,\phi(v)$ with $\phi(v)=v\odot|v|$. The symmetric acceleration capacity $\bar a_i(v)=(\bar\tau_i-b_iv_i^2)/m_i$ gives the co-metric

$$D(v) = 4A\,\mathrm{diag}(v_i^2\bar a_i^2)\,A^\top ,\qquad h(v)=\ln\det D(v)-\bar\ell .$$

$D(v)$ describes the wrench rates the rotors can still produce from their current speeds. The filter (Algorithm 1, `allocators.cbf_qp`) enforces $\dot h\ge-\alpha h$ together with the wrench servo and the torque limits. A slack variable lets tracking give way when the two conflict.

The simulators use nondimensional units: $\bar\tau=b=m=1$, so speeds are in units of $v^{\mathrm{sat}}$ and time is in units of $T_0=m/\sqrt{\bar\tau b}$. The sweet spot is $v^\star=v^{\mathrm{sat}}/\sqrt3$. The continuous-time guarantees in the paper apply to the model. The simulators check them in discrete time, and the high-fidelity study checks what is left of them on a plant the model does not describe exactly.

## Repository layout

| Path | Content |
|---|---|
| `platforms.py` | Allocation matrices of the coplanar hexarotor (n = 6, m = 4) and the tilted octorotor (n = 8, m = 6). |
| `daam.py` | Co-metric $D(v)$, readiness $h(v)$ and its analytic gradient. |
| `gate.py` | Maximum readiness, dropout level $\ell_{\mathrm{drop}}$ and dropout gaps. |
| `allocators.py` | Effort-minimising QP, greedy and low-pass DAAM setpoints, the readiness filter and its robust version. |
| `harden.py`, `sim_core.py` | Reference mission and closed-loop simulators. `sim_core.run` is the one used by the experiments. |
| `final_exp.py` | Mission achievability $L^{\mathrm{op}}$ and the floor rule $\bar\ell=\ell_{\mathrm{drop}}+\kappa(L^{\mathrm{op}}-\ell_{\mathrm{drop}})$. |
| `robust.py`, `robust_campaign.py`, `robust_v2.py` | Parameter boxes, perturbed-plant rollouts and the robust ablation. |
| `delay_study.py` | Constants of the delay bound (used by `exp_delay.py` and `exp_kappa.py`). |
| `exp_*.py` | The experiments of the paper, one script per study (see below). |
| `hifi/` | High-fidelity 6-DoF simulator built from measured propeller and motor data ([hifi/README.md](hifi/README.md)). |
| `make_paper_data.py` | Rebuilds every table in `data/` from the experiment outputs. |
| `figures/tikz/`, `data/` | TikZ/PGFPlots sources of the paper figures and the tables they plot. |
| `results/reference/` | The experiment outputs used for the paper ([results/reference/README.md](results/reference/README.md)). |
| `verify.py` | Checks the analytic gradient and the dropout-gap law. |
| `verify_numbers.py`, `fill_paper.py`, `hifi/fill_hifi.py` | Helpers that check or write the numbers in the manuscript (the manuscript is not included). |
| `export_results_mat.py` | Copies all results into one MATLAB `.mat` file. |
| `project_paths.py` | Output locations. |
| `dev/` | Development and diagnostic scripts (see below). |

`dev/` holds scripts written during development. They print diagnostics or draw quick matplotlib plots, which go to `figures/generated/` and `figures/data/`. They are not needed to reproduce the paper, and some use older settings than the paper (for example, `dev/sweep.py` keeps the original low-pass constant). Run them from anywhere, e.g. `python dev/make_all_figs.py`.

## Installation

The code needs Python 3.10 or newer with NumPy (2.0 or newer), SciPy, Matplotlib and OSQP. No other solver is required. The versions in `requirements.txt` are minimums, not pins. The results in `results/generated` were last checked with Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, contourpy 1.4.0 and OSQP 1.1.3 on Windows 11.

Windows (PowerShell), from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use `.venv\Scripts\activate.bat` from a Command Prompt.

Linux or WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the experiments

Run everything from the repository root. New results go to `results/generated/`, which is not tracked by git. `results/reference/` is never written.

| Study | Command | Output | Used for |
|---|---|---|---|
| Collective sweep | `python exp_sweep.py` | `sweep.json` | Table I, Fig. 3 |
| Moment-reversal time histories | `python exp_fig5.py` | `rate_all.dat`, `h_all.dat` | Fig. 4 |
| Randomised missions (N per box) | `python exp_mc.py 100` | `mc.json` | Table II |
| Certifiability map | `python exp_certmap.py` | `certmap.dat` | Fig. 5 (left) |
| Forward flight | `python exp_ff.py` | `ff.json` | Fig. 5 (right) |
| Input delay | `python exp_delay.py` | `delay.json` | Fig. 6 |
| Parameter uncertainty (plants per level) | `python exp_robust.py 50` | `robust.json` | Table III |
| Choice of $\kappa$ | `python exp_kappa.py` | `kappa.json` | Sec. IV text |
| QP timing on this computer | `python exp_timing.py` | `timing.json` | Sec. IV text |
| High-fidelity campaign | `python hifi/run_hifi.py` | `hifi.json` | Table IV |

`exp_fig5.py` keeps the name it had when the reversal plot was Fig. 5 of the submitted manuscript.

To run everything in order, use `powershell -ExecutionPolicy Bypass -File run_revision.ps1` on Windows or `bash run_revision.sh` on Linux. Logs are written to `results/generated/logs/`. The whole campaign takes a few hours on two cores, and the high-fidelity part takes the longest. The scripts run natively on Windows, where Python's multiprocessing uses spawned worker processes.

The runners finish by calling `make_paper_data.py` (see below). If the environment variable `PAPER_TEX` points to a local copy of the manuscript's `main.tex`, they also run `verify_numbers.py`, which only reads the manuscript. `fill_paper.py` writes into the manuscript and is never called by the runners.

## Reproducing the figures

The figures are drawn with PGFPlots from the tables in `data/`. `make_paper_data.py` rebuilds all of these tables:

```bash
python make_paper_data.py --compare data                       # from results/generated
python make_paper_data.py --results reference --compare data   # from results/reference
```

The new tables are written to `results/generated/paper_data/`. `--compare data` prints the largest difference to the tables used in the paper, and `--out data` replaces them.

| Figure | TikZ source | Tables | Produced from |
|---|---|---|---|
| 1, wrench-rate ellipses | `fig_ellipse.tex` | `ell_sweet.dat`, `ell_stopped.dat`, `ell_sat.dat` | the model (hexarotor, $(\tau_x,\tau_y)$ block of $D$) |
| 2, certified set | `fig_nested.tex` | `n2_C.dat`, `n2_Keps.dat` | the model (two-rotor illustration, see below) |
| 3, collective sweep | `fig_sweep.tex` | `sweep.dat` | `exp_sweep.py` |
| 4, reversal histories | `fig_lpf2.tex` | `rate_all.dat`, `h_all.dat` | `exp_fig5.py` |
| 5, certifiability and forward flight | `fig_applic.tex` | `certmap.dat`, `certfront.dat`, `ff.dat` | `exp_certmap.py`, `exp_ff.py` |
| 6, input delay | `fig_delayregimes.tex` | `delay.dat`, `delay_marg.dat`, `delay_eta.dat` | `exp_delay.py` |

Fig. 2 shows a two-rotor system with $A=I$, so $D=4\,\mathrm{diag}(\psi_1,\psi_2)$. The set $\mathcal C$ is $\ln\psi_1+\ln\psi_2\ge L^{\max}-1.3$, and $K_\varepsilon$ is the region where every $\psi_i\ge0.02$. `fig_section.tex` is a hand-drawn sketch with no data; it is commented out in the manuscript.

A few values are typed directly into the TikZ files: the shaded regions of Fig. 6 ($\tau_d^\star=1.9656$, failure at $91.6$, delay-free margin $0.1636$, all in $10^{-3}T_0$ or nats) and the start of the non-certifiable band in Fig. 5 ($9.33$ m/s). `make_paper_data.py` prints these values so they can be checked against the sources.

The TikZ files are meant to be `\input` from the manuscript. Its preamble defines what they use: the lengths `\panelwd` and `\panelht`, the macros `\lbar`, `\ldrop` and `\Cset`, the colours `cBlue`, `cOrange`, `cRed`, `cGreen` and `cGrey`, the TikZ libraries `arrows.meta` and `patterns`, and PGFPlots with the `colorbar` support. The paths `data/...` are relative to the manuscript directory.

## Reproducibility

Rebuilt from `results/reference/`, every table matches `data/` to the printed precision. The two contour tables differ by less than 0.004 in axis units, because the originals were traced on a coarser grid. The paper figures can therefore be regenerated from the reference outputs.

All random numbers come from seeded NumPy generators, so a run is repeatable on a given machine. Results can still differ between environments, because SLSQP (SciPy) and OSQP may converge to different points with other library versions or platforms. A full re-run with the versions listed above gave the following:

* The proposed filter and the effort QP agree with the reference to within $3\times10^{-3}$ in every study, except for the one forward-flight case listed below.
* The certifiability map, the delay study and the $\kappa$ study agree. The only exceptions are two certifiability cells deep in the non-certifiable corner, and both lie below the colour-bar range.
* The greedy and low-pass baselines differ where they drive a rotor through zero speed, which is a discontinuous event. At collective 0.7 in the sweep, low-pass DAAM reaches $\min h=-0.52$ instead of $-1.23$, and greedy DAAM $-0.51$ instead of $-0.54$. In Table II the violation rates and worst cases are unchanged, but the mean $h_{\min}$ of low-pass DAAM in the scarce hexarotor box moves from $-0.51$ to $-0.49$. In the high-fidelity campaign the greedy violation rate on agile missions is 20 % instead of 25 %. Both baselines still violate the floor in the same cases.
* Forward flight agrees, except for the airspeed-scheduled filter on the hexarotor at 15 m/s, which is beyond the certifiable speed ($\min h_s=-0.53$ instead of $-0.56$).
* In the robust ablation (lower block of Table III), the metric-only rows differ: at $p=30\%$ the violation rate is 10 % instead of 48 %. All other rows of Table III agree.

`exp_timing.py` measures the computer it runs on, so its numbers will be different elsewhere.

## Limitations

The barrier is built on the signed-quadratic rotor model and enforced in continuous time. The simulators integrate with a fixed step of $10^{-3}T_0$. The high-fidelity plant adds Reynolds-number and inflow effects on the propeller coefficients, back-EMF current limits, ESC lag, battery sag, friction, rotor H-force, body drag, telemetry noise and delay, and rigid-body dynamics. It does not model unsteady inflow or rotor-rotor interaction. Manoeuvres that need a rotor to reverse its spin are outside the scope of the method by design.

## Citation

If you use this code or build on the paper, please cite:

> G. Silano, "Readiness Barrier Functions: Forward-Invariant Control Authority for Overactuated Multirotor Allocation," *IEEE Robotics and Automation Letters*, vol. XX, no. X, pp. XXXX–XXXX, MONTH YEAR, doi: 10.1109/LRA.YYYY.XXXXXXX.

```bibtex
@article{Silano2026RAL,
  author  = {Silano, Giuseppe},
  title   = {Readiness Barrier Functions: Forward-Invariant Control Authority for Overactuated Multirotor Allocation},
  journal = {IEEE Robotics and Automation Letters},
  year    = {YYYY},
  volume  = {XX},
  number  = {X},
  pages   = {XXXX--XXXX},
  doi     = {10.1109/LRA.YYYY.XXXXXXX}
}
```

## License

MIT License, copyright (c) 2026 Giuseppe Silano; see [LICENSE](LICENSE). The propeller data in `hifi/apc10x47.py` come from the UIUC Propeller Database (Brandt and Selig), and the source is given in that file.
