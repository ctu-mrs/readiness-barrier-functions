"""
export_mat.py — export generated figure-data CSV files as MATLAB .mat files.
"""
import os, numpy as np
from scipy.io import savemat
from project_paths import FIGURE_DATA_DIR, figure_data_path

def load_csv(name):
    path = figure_data_path(name)
    with open(path) as f:
        header = f.readline().strip().lstrip("# ")
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    cols = [c.strip().replace(".", "_").replace("-", "_") for c in header.split(",")]
    return cols, np.atleast_2d(arr)

def main():
    bundle = {}
    for csv, key in [("flip_curve.csv", "flip_curve"),
                     ("thrust_sweep.csv", "thrust_sweep"),
                     ("timehist_h.csv", "timehist_h"),
                     ("timehist_greedy_cmd.csv", "timehist_greedy_cmd"),
                     ("timehist_cbf_cmd.csv", "timehist_cbf_cmd"),
                     ("lpf_h.csv", "lpf_h"),
                     ("lpf_rate.csv", "lpf_rate"),
                     ("delay.csv", "delay"),
                     ("delay_marg.csv", "delay_marg"),
                     ("jump_profile.csv", "jump_profile")]:
        p = figure_data_path(csv)
        if not os.path.exists(p):
            print("  skip (missing):", csv); continue
        cols, arr = load_csv(csv)
        d = {c: arr[:, i] for i, c in enumerate(cols) if i < arr.shape[1]}
        d["columns"] = np.array(cols, dtype=object)
        d["matrix"] = arr
        savemat(figure_data_path(key + ".mat"), d)
        bundle[key] = d
        print("  wrote %s.mat  (%d rows x %d cols)" % (key, arr.shape[0], arr.shape[1]))
    # applicability map is a categorical matrix (no header row of names)
    ap = figure_data_path("applicability.csv")
    if os.path.exists(ap):
        M = np.genfromtxt(ap, delimiter=",", skip_header=1)
        d = {"map": M,
             "rows_thrust": np.array([0.25, 0.5, 0.75, 1.0, 1.25]),
             "cols_torque": np.array([0.5, 1.0, 2.0, 3.0]),
             "legend": np.array(["2=OK (single component)", "1=JMP (crossing forced)",
                                 "0=INF (infeasible)"], dtype=object)}
        savemat(figure_data_path("applicability.mat"), d)
        bundle["applicability"] = d
        print("  wrote applicability.mat  (%dx%d)" % M.shape)
    savemat(figure_data_path("figdata_all.mat"), bundle)
    print("  wrote figdata_all.mat (all figures in one struct)")

if __name__ == "__main__":
    print("exporting figdata -> MATLAB .mat")
    main()
