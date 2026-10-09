#!/bin/bash
# Regenerates experiment outputs under results/generated/ (about 1.5 h on 2 cores).
# Set PAPER_TEX to opt in to checking those results against a local manuscript.
set -e
cd "$(dirname "$0")"
LOG_DIR="$(pwd)/results/generated/logs"
mkdir -p "$LOG_DIR"
for s in exp_timing exp_sweep exp_fig5 exp_certmap exp_kappa exp_delay exp_ff; do
  echo "== $s $(date +%T)"; python3 "$s.py" > "$LOG_DIR/$s.log" 2>&1
done
echo "== exp_robust $(date +%T)"; python3 exp_robust.py 50 > "$LOG_DIR/exp_robust.log" 2>&1
echo "== exp_mc $(date +%T)";     python3 exp_mc.py 100    > "$LOG_DIR/exp_mc.log" 2>&1
echo "== hifi $(date +%T)";      (cd hifi && python3 run_hifi.py > "$LOG_DIR/hifi.log" 2>&1)
python3 export_results_mat.py
if [ -n "${PAPER_TEX:-}" ]; then
  python3 verify_numbers.py "$PAPER_TEX"
else
  echo "Skipping manuscript number check; set PAPER_TEX to a local main.tex to run it."
fi
echo "== done $(date +%T)"
