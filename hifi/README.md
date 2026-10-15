# High-fidelity simulator

A closed-loop 6-DoF hexarotor that includes the effects the readiness barrier leaves out, built from measured component data. The allocators run unchanged on the quadratic model a practitioner would identify from the same data. This is the study of Table IV.

| File | Content |
|---|---|
| `apc10x47.py` | APC 10x4.7SF propeller data from the UIUC Propeller Database, Vol. 1 (Brandt and Selig, AIAA 2011-1255). It contains static $C_T$ and $C_P$ against rpm, $C_T$ and $C_P$ against advance ratio at 4014, 5018, 6020 and 6513 rpm, and the chord distribution. Each table row is checked against its own efficiency column when the file is loaded. The file also provides a bilinear interpolant $C(J,\mathrm{rpm})$ and the rotor inertia, computed from the measured propeller mass (11.9 g) and chord. |
| `model.py` | SunnySky X2212-13 980KV motor (KV, R = 133 mOhm, I0 = 0.3 A, 15 A), back-EMF current limit, Coulomb friction, 2 ms ESC current loop, 3S battery sag, true thrust and torque and their slopes, true acceleration capacity, and Dryden turbulence. It also identifies the nominal model from the static data at 4000 rpm, which gives T0 = 0.32 s and v_sat = 7330 rpm. |
| `hifi_sim.py` | Rigid-body dynamics with rotor reaction and gyroscopic torques, H-force and body drag. Telemetry has a 1 ms delay, 0.5 % noise and 10 rpm quantisation. The file also contains the SE(3) controller, the five allocators at 500 Hz (effort QP, greedy, low-pass, filter, robust filter), the missions `rock` (roll reversals), `load` (external moment) and `cruise` (acceleration to 0-15 m/s with a slalom), the floor computed from the mission feedforward wrench (Proposition 2), and the true readiness $h_s$. |
| `run_hifi.py` | The campaign: 13 mission configurations x 5 allocators x 5 seeds, plus a delay sweep. Writes `results/generated/hifi.json`; takes 1-2 h on two cores. |
| `analyze_hifi.py` | Prints per-mission and per-family summaries. |
| `fill_hifi.py` | Writes the body of Table IV as a TeX fragment. |

Run from the repository root:

```bash
python hifi/run_hifi.py
python hifi/analyze_hifi.py
python hifi/fill_hifi.py table_hifi.tex
```

The scripts find the rest of the repository by themselves and work from any directory, on Windows as well as Linux.

Modelling assumptions (stated in the paper or here):

* the propeller is treated as symmetric (a "3D" propeller) when it spins backwards;
* for J < 0 (descent-like inflow) the static coefficients are used;
* below 4014 rpm the coefficients follow the static Reynolds-number trend;
* battery: open-circuit voltage 11.4 V, internal resistance 25 mOhm;
* motor bell: 22 g at radius 12.5 mm;
* rotor aerodynamics are quasi-steady, with no unsteady inflow and no rotor-rotor interaction.
