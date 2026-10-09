# High-fidelity simulator (measured component data)

Closed-loop 6-DoF hexarotor in which every effect the readiness-barrier theory omits is present,
while Algorithm 1 runs on the quadratic model identified from the same data.

| file | content |
|---|---|
| `apc10x47.py` | APC 10x4.7SF data, UIUC Propeller Database Vol. 1 (Brandt & Selig, AIAA 2011-1255): static C_T, C_P vs rpm; C_T, C_P vs J at 4014/5018/6020/6513 rpm; chord geometry. Each table row is checked against its own efficiency column. Fast bilinear interpolant C(J, rpm). Rotor inertia from measured prop mass (11.9 g) and chord distribution. |
| `model.py` | SunnySky X2212-13 980KV (KV, R = 133 mOhm, I0 = 0.3 A, 15 A), back-EMF current limit, Coulomb friction, 2 ms ESC current loop, 3S battery sag; true thrust/torque/slopes; true symmetric acceleration capacity; Dryden turbulence; identification of the nominal model (static data at 4000 rpm -> T0 = 0.32 s, v_sat = 7330 rpm). |
| `hifi_sim.py` | 6-DoF rigid body, rotor reaction and gyroscopic torques, H-force, body drag, telemetry (1 ms delay, 0.5 % noise, 10 rpm quantisation), SE(3) controller, the five allocators at 500 Hz, missions `rock` (agile roll reversals), `load` (external moment), `cruise` (accelerated forward flight + slalom), Proposition-2 floor from the mission feedforward wrench, true readiness h_true. |
| `run_hifi.py` | campaign: 13 missions x 5 allocators x 5 turbulence seeds + delay sweep -> `results/generated/hifi.json` (~1.2 h on 2 cores) |
| `analyze_hifi.py`, `fill_hifi.py` | summaries and the rows of Table IV |

Assumptions (stated in the paper or here): symmetric ("3D") propeller for reversed rotation;
J < 0 (descent-like inflow) held at static coefficients; rpm < 4014 scaled by the static
Reynolds trend; battery V_oc = 11.4 V, R_int = 25 mOhm; motor bell 22 g at r = 12.5 mm;
quasi-steady rotor aerodynamics (no unsteady inflow or rotor-rotor interaction).

Run from the repository root with `python hifi/run_hifi.py`. The runner writes to the generated-results directory regardless of the current working directory. It uses multiprocessing with the Unix `fork` start method, so use Linux or WSL2 rather than native Windows.
