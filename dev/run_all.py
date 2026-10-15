"""run_all.py — reproduce every Phase-1 numerical result in order."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
import gate, verify, envelope, make_figs
from platforms import hexarotor, tilted_octorotor

print("\n########## GATE (both platforms) ##########")
for P in (hexarotor(), tilted_octorotor()):
    gate.report(P); print()
print("########## VERIFICATION ##########")
import runpy
runpy.run_path(os.path.join(sys.path[0], "verify.py"), run_name="__main__")
print("\n########## OPERATIONAL ENVELOPE (hexarotor) ##########")
envelope.run()
print("\n########## FIGURES ##########")
make_figs  # figures already generated on import
print("done.")
