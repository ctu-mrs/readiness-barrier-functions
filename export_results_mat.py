"""export_results_mat.py -- copy every result in results/generated (*.json, *.dat) into one MATLAB file.

JSON keys are made valid MATLAB field names (dots and dashes become
underscores, a "k_" prefix is added when a key starts with a digit).
Output: results/generated/revision_results.mat.
"""
import json, os, glob, numpy as np
from scipy.io import savemat
from project_paths import RESULTS_DIR, result_path
def clean(x):
    if isinstance(x, dict): return {("k_" + k.replace(".", "_").replace("-", "_")) if not k[0].isalpha() else k.replace(".", "_").replace("-", "_"): clean(v) for k, v in x.items()}
    if isinstance(x, list):
        if x and isinstance(x[0], dict): return np.array([clean(v) for v in x], dtype=object)
        return np.array([np.nan if v is None else v for v in x], dtype=float) if all(isinstance(v, (int, float, type(None))) for v in x) else np.array(x, dtype=object)
    return np.nan if x is None else x
bundle = {}
for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
    key = os.path.splitext(os.path.basename(f))[0]
    if key == "tokens": continue
    bundle[key] = clean(json.load(open(f)))
for f in sorted(glob.glob(str(RESULTS_DIR / "*.dat"))):
    key = os.path.splitext(os.path.basename(f))[0]
    with open(f) as fh: hdr = fh.readline().split()
    arr = np.genfromtxt(f, skip_header=1)
    bundle["dat_" + key] = {"columns": np.array(hdr, dtype=object), "data": arr}
savemat(result_path("revision_results.mat"), bundle)
print("wrote revision_results.mat with", list(bundle))
