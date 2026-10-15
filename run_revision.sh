#!/bin/bash
# Run every experiment of the paper in order and write the results to
# results/generated/ (logs in results/generated/logs/). Takes a few hours on two cores.
# Set PAPER_TEX=/path/to/main.tex to also check the numbers quoted in the manuscript.
set -e
cd "$(dirname "$0")"
LOG_DIR="$(pwd)/results/generated/logs"
mkdir -p "$LOG_DIR"
for s in exp_timing exp_sweep exp_fig5 exp_certmap exp_kappa exp_delay exp_ff; do
  echo "== $s $(date +%T)"; python3 "$s.py" > "$LOG_DIR/$s.log" 2>&1
done
echo "== exp_robust $(date +%T)"; python3 exp_robust.py 50 > "$LOG_DIR/exp_robust.log" 2>&1
echo "== exp_mc $(date +%T)";     python3 exp_mc.py 100    > "$LOG_DIR/exp_mc.log" 2>&1
echo "== hifi $(date +%T)";       python3 hifi/run_hifi.py > "$LOG_DIR/hifi_run_hifi.log" 2>&1
python3 export_results_mat.py
# plotting tables for the figures, compared with the ones in data/
python3 make_paper_data.py --compare data | tee "$LOG_DIR/make_paper_data.log"
if [ -n "${PAPER_TEX:-}" ]; then
  python3 verify_numbers.py "$PAPER_TEX"
else
  echo "Skipping the manuscript check; set PAPER_TEX to a local main.tex to run it."
fi
echo "== done $(date +%T)"
