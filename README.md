# Readiness Barrier Functions

This repository contains Python simulations for the manuscript “Readiness Barrier Functions: Forward-Invariant Control Authority for Overactuated Multirotor Allocation,” which is under review. The central method treats the log-determinant of a drag-aware actuator-authority co-metric as a control barrier and enforces an authority floor in a torque-level quadratic program (QP). The simulations compare that filter with effort-minimizing, greedy readiness-maximizing, and low-pass-filtered allocators.

The code covers a coplanar hexarotor and a tilted octorotor, randomized missions, parameter uncertainty, delay, forward flight, and a higher-fidelity 6-DoF simulation using propeller and motor data. It is simulation research code, not a flight-ready controller. The manuscript and its figure source are not part of this repository. The supplied reference outputs are retained separately from new runs.

## Model and code

The signed-quadratic thrust map is $f(v)=A(v\odot|v|)$, and the readiness metric is computed from $D(v)=4A\operatorname{diag}(v_i^2\bar a_i^2)A^\top$. The implementation follows that model directly; the paper’s continuous-time invariance statements should not be read as guarantees for the discretized simulator, the high-fidelity plant, or real hardware.

| File or folder | Role |
|---|---|
| `platforms.py` | Hexarotor/octorotor allocation matrices and nominal motor parameters. |
| `daam.py`, `gate.py` | Co-metric, readiness and gradient; dropout gap and mission-floor calculations. |
| `allocators.py` | Effort QP, fiberwise readiness optimizer, low-pass baseline, nominal and robust barrier QPs. |
| `harden.py`, `sim_core.py` | Wrench trajectories and closed-loop simulation used by the revision experiments. |
| `final_exp.py` | Mission-achievability calculation and floor selection. |
| `exp_*.py` | Main sweep, mission, robustness, delay, forward-flight, and timing runs. |
| `robust.py`, `robust_campaign.py`, `robust_v2.py` | Uncertainty sampling, robust invariance campaign, and metric-only ablation. |
| `hifi/` | Measured-component model, 6-DoF simulator, campaign runner, and result summaries; see [hifi/README.md](hifi/README.md). |
| `make_all_figs.py`, `concept_figs.py`, `make_figs.py`, `thrust_sweep.py`, `make_sim_fig.py`, `delay_study.py`, `lpf_analysis.py` | Standalone diagnostic plots and their data. They do not regenerate every paper figure. |
| `verify.py` | Analytic-gradient and gap-law checks; no test framework is configured. |
| `verify_numbers.py`, `fill_paper.py`, `hifi/fill_hifi.py` | Optional manuscript consistency/export helpers. They require explicit local manuscript paths; `fill_paper.py` writes to the manuscript and is not run by the experiment runner. |
| `project_paths.py` | Repository-local generated result and figure paths. |
| `results/reference/` | Curated, read-only comparison outputs from the supplied result set. |
| `figures/tikz/`, `data/` | TikZ figure fragments referenced by the manuscript and their required plotting tables. The manuscript itself is not included. |

The current implementation includes several corrections documented in the earlier project notes: the low-pass baseline applies its 50 ms time constant at every integration step; `_osqp_solve` checks OSQP's status before accepting a solution and the effort allocator relaxes an infeasible wrench request; and `exp_fig5.py` records the maximum commanded rate over each greedy update interval so short setpoint jumps are not skipped.

## Requirements

Python 3.10 or newer is recommended. The project uses NumPy, SciPy, Matplotlib, and OSQP; the minimum NumPy version is 2.0 because the high-fidelity propeller model uses `numpy.trapezoid`. Requirements are listed in `requirements.txt` but are not pinned to exact versions, so this is not a bit-for-bit locked environment. No external optimization solver is needed beyond OSQP. Rights to distribute the included code and embedded propeller data were confirmed by the user; the propeller source and attribution are documented in `hifi/apc10x47.py`.
Python 3.10 or newer is recommended. The project uses NumPy, SciPy, Matplotlib, and OSQP; the minimum NumPy version is 2.0 because the high-fidelity propeller model uses `numpy.trapezoid`. Requirements are listed in `requirements.txt` but are not pinned to exact versions, so this is not a bit-for-bit locked environment. The Windows run in this workspace used Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, and OSQP 1.1.3. No external optimization solver is needed beyond OSQP. Rights to distribute the included code and embedded propeller data were confirmed by the user; the propeller source and attribution are documented in `hifi/apc10x47.py`.

### Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either use a new Command Prompt and run `.venv\Scripts\activate.bat`, or allow activation for the current PowerShell process with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. The experiment scripts use Python's default multiprocessing context and can run natively on Windows. `run_revision.ps1` runs the full campaign from the repository root.

### Ubuntu or WSL2

Install Python and venv support if needed, then run from the repository root:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The same commands apply in a Linux environment with Python 3.10+. Use the virtual environment’s Python for all commands below.

## Running experiments

Run commands from the repository root unless a command says otherwise. Every new result is written beneath `results/generated/`; logs from the full campaign go in `results/generated/logs/`. Existing files in `results/reference/` are not overwritten.

| Purpose | Command | Main output |
|---|---|---|
| Collective sweep | `python exp_sweep.py` | `sweep.json` |
| Time histories at the reversal case | `python exp_fig5.py` | `rate_all.dat`, `h_all.dat` |
| Randomized missions; optional mission count argument | `python exp_mc.py 100` | `mc.json` |
| Floor-selection study | `python exp_kappa.py` | `kappa.json` |
| Certifiability map | `python exp_certmap.py` | `certmap.dat` |
| Forward flight | `python exp_ff.py` | `ff.json` |
| Delay study | `python exp_delay.py` | `delay.json` |
| Robust campaign; plants per uncertainty level | `python exp_robust.py 50` | `robust.json` |
| QP timing on the current computer | `python exp_timing.py` | `timing.json` |
| High-fidelity campaign | `python hifi/run_hifi.py` | `hifi.json` |
| Full ordered campaign (Windows PowerShell) | `powershell -ExecutionPolicy Bypass -File run_revision.ps1` | All experiments; logs go to `results/generated/logs/`. |
| Full ordered campaign (Linux/WSL2) | `bash run_revision.sh` | All experiments; logs go to `results/generated/logs/`. |

The campaign uses fixed NumPy generator seeds in its scripts, but solver/library versions, OS, CPU timing, and floating-point behavior are not locked. OSQP uses tolerances and iteration settings in `allocators.py`. Results should be compared numerically with tolerances; timing results are machine-specific. The corrected robust formulation and solver-failure behavior may cause robust outputs to differ from the supplied reference JSON. Regenerate that campaign before using it to support revised claims.

Both full-suite runners leave the manuscript alone. To opt in to the read-only number check, set `PAPER_TEX` to a local `main.tex` path before running either script (`$env:PAPER_TEX = 'C:\path\to\main.tex'` in PowerShell, or `PAPER_TEX=/path/to/main.tex bash run_revision.sh` in Bash). The standalone checker is `python verify_numbers.py PATH_TO_MAIN_TEX`; it requires the generated JSON outputs. `fill_paper.py PATH_TO_MAIN_TEX` is different: it writes to the specified manuscript and its adjacent figure/data sources, and is not called by either runner. `hifi/fill_hifi.py PATH_TO_OUTPUT_TEX` writes a generated table fragment to the specified path.

## Figures and paper mapping

The seven TikZ/PGFPlots fragments referenced by the manuscript, together with their required plotting tables, are included under `figures/tikz/` and `data/`. The manuscript and bibliography are not included. `fig_section.tex` and its data are included for completeness, although that figure is commented out in the supplied manuscript.

| Manuscript item | Available inputs or script | What is and is not reproduced |
|---|---|---|
| Fig. 1, achievable wrench-rate ellipsoids | `figures/tikz/fig_ellipse.tex`; `data/ell_*.dat`; `concept_figs.py` | Manuscript plot source/data included; `concept_figs.py` also renders a standalone diagnostic. |
| Fig. 2, certified set | `figures/tikz/fig_nested.tex`; `data/n2_C.dat`, `data/n2_Keps.dat` | Manuscript plot source and data included. |
| Fig. 3, collective sweep | `figures/tikz/fig_sweep.tex`; `data/sweep.dat`; `exp_sweep.py` | Manuscript plot source and data included. `thrust_sweep.py` makes a related diagnostic with a different sweep. |
| Fig. 4, reversal histories | `figures/tikz/fig_lpf2.tex`; `data/rate_all.dat`, `data/h_all.dat`; `exp_fig5.py` | Manuscript plot source and data included. `lpf_analysis.py` makes a separate per-step-rate diagnostic. |
| Fig. 5, certifiability and forward flight | `figures/tikz/fig_applic.tex`; `data/certmap.dat`, `data/certfront.dat`, `data/ff.dat`; `exp_certmap.py`, `exp_ff.py` | Manuscript plot source and data included. |
| Fig. 6, delay sweep | `figures/tikz/fig_delayregimes.tex`; `data/delay.dat`, `data/delay_marg.dat`, `data/delay_eta.dat`; `exp_delay.py` | Manuscript plot source and data included. `delay_study.py` renders a related diagnostic with a different sweep. |
| Optional allocation-section concept | `figures/tikz/fig_section.tex`; `data/alloc_greedy.dat`, `data/alloc_cbf.dat` | Source and data are included, but this figure is commented out in the manuscript. |
| Tables I–IV | `exp_sweep.py`, `exp_mc.py`, `exp_robust.py`, `hifi/run_hifi.py` | Numerical result files are available; manuscript tables themselves are not in the repository. |

To make the included diagnostic plots and data, run `python make_all_figs.py`. This runs simulations and writes PDFs to `figures/generated/` and CSV data to `figures/data/`. The manuscript TikZ fragments can be compiled from a copy of the manuscript project with its `figures/tikz/` and `data/` paths preserved. The delay and low-pass diagnostics can be run separately with `python delay_study.py` and `python lpf_analysis.py`. MATLAB export of figure data is available with `python export_mat.py` after generating those CSV files.

## Reference outputs

The curated snapshot in `results/reference/` contains eight JSON and three DAT files (about 0.5 MB). The JSON files cover the collective sweep, Monte Carlo, robustness, floor choice, delay, forward flight, timing, and high-fidelity campaigns. The DAT files contain the certifiability map and Figure 4 readiness/rate histories. Each DAT file starts with column names; JSON can be loaded with Python’s standard `json` module. `hifi.json` also contains delay cases. Timing measurements describe the original machine and are not expected to match elsewhere.

The originals were preserved, and the integrated files were checked against them by SHA-256. New runs go into the separate generated directory. For comparisons, first check that the script, parameters, seeds, and allocator version match; then compare corresponding JSON values or aligned DAT columns with tolerances appropriate to the solver and platform. Do not use image equality as evidence of numerical reproducibility. The current robust campaign’s implementation has been corrected to include the worst-case drift and torque reserve, so the old `robust.json` is a historical reference, not a passing target for that corrected implementation. Token-named files and logs were not included.

## Scope and release status

The nominal theory uses a signed-quadratic thrust/drag model and continuous-time barrier enforcement. The main simulator uses time stepping; the high-fidelity model adds quasi-steady aerodynamics, motor/ESC/battery limits, delay, sensing effects, and rigid-body dynamics. All evidence is simulation-based; it does not establish flight validation.

The repository is distributed under the [MIT License](LICENSE), with copyright attributed to Giuseppe Silano (2026), following confirmation of the required distribution rights. The license applies to project-owned code and materials; dependencies and third-party data remain subject to their own licenses and attribution terms. No citation metadata is included while the manuscript remains under review.

Before public release, rerun and review the corrected robustness results, reconcile any result differences with the manuscript, and confirm the complete manuscript build works from a release snapshot. This repository is prepared locally only; it has not been published.
