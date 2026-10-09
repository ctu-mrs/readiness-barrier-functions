# Reference results

These files are the supplied reference outputs for comparison. They are kept separate from `results/generated/`, where experiment scripts write new results. Do not overwrite this directory when running experiments.

| File | Contents |
|---|---|
| `sweep.json` | Collective-thrust sweep runs and selected floor values (Table I data). |
| `mc.json` | Randomized mission outcomes for both platforms (Table II data). |
| `robust.json` | Parameter-mismatch campaign and ablation (Table III data). |
| `hifi.json` | High-fidelity plant campaign, including delay runs (Table IV data). |
| `kappa.json` | Floor-selection and delay/parameter sensitivity. |
| `delay.json` | Delay margin calculations and delay sweep outputs. |
| `ff.json` | Forward-flight outcomes and airspeed-dependent certifiability. |
| `timing.json` | Timing measurements from the machine used for that run. |
| `certmap.dat` | Whitespace-delimited `thrust amp window` certifiability map. |
| `h_all.dat` | Whitespace-delimited readiness histories for the four allocator variants. |
| `rate_all.dat` | Whitespace-delimited commanded-rate histories for the four variants. |

The JSON files are machine-readable; the DAT files have a header row with column names. Timing values are hardware-dependent. The files document outputs, not a complete execution environment: consult the repository README for scripts, dependencies, and current limitations. Their original files were left unchanged. No run logs or token-named artifacts are included.
