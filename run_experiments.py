"""run_experiments.py — closed-loop evaluation (slow: ~5 min total).
Order: certifiability diagnosis -> floor rule -> thrust sweep -> figures.
"""
import runpy
print("### 1. certifiability diagnosis (which operating points admit a floor)")
runpy.run_module("diagnose", run_name="__main__")
print("\n### 2. closed loop at certifiable floors")
runpy.run_module("final_exp", run_name="__main__")
print("\n### 3. collective-thrust sweep + figure")
runpy.run_module("thrust_sweep", run_name="__main__")
print("\n### 4. hardening diagnostics (margin, alignment ratio, octorotor)")
runpy.run_module("harden", run_name="__main__")
