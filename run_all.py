"""run_all.py — reproduce every Phase-1 numerical result in order."""
import gate, verify, envelope, make_figs
from platforms import hexarotor, tilted_octorotor

print("\n########## GATE (both platforms) ##########")
for P in (hexarotor(), tilted_octorotor()):
    gate.report(P); print()
print("########## VERIFICATION ##########")
exec(open("verify.py").read())
print("\n########## OPERATIONAL ENVELOPE (hexarotor) ##########")
envelope.run()
print("\n########## FIGURES ##########")
make_figs  # figures already generated on import
print("done.")
