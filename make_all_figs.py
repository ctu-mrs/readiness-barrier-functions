"""
make_all_figs.py — generate the diagnostic figures available in this repository.
Figure data are written to figures/data/; manuscript plotting sources are not included.

Figures produced (figures/generated/):
  fig_ellipsoid.pdf      achievable wrench-rate set at three configurations
  fig_concept.pdf        the jump: equal-readiness strata and the valley between
  fig_applicability.pdf  where single-component confinement is compatible w/ task
  fig_thrust.pdf         collective-thrust sweep (authority + tracking)
  fig_timehist.pdf       mechanism: commanded allocation + authority histories
Runtime: ~4-6 min (fig_thrust and fig_timehist run closed-loop simulations).
"""
import runpy, time
JOBS = [("concept_figs",  "fig_ellipsoid + fig_concept    (fast, <5 s)"),
        ("make_figs",     "fig_applicability              (fast, ~1 min)"),
        ("thrust_sweep",  "fig_thrust                     (slow, ~2-3 min)"),
        ("make_sim_fig",  "fig_timehist                   (slow, ~2 min)")]
if __name__ == "__main__":
    for mod, desc in JOBS:
        print("\n" + "="*66 + "\n### %s -> %s\n" % (mod, desc) + "="*66)
        t0 = time.time(); namespace = runpy.run_module(mod, run_name="__main__")
        if mod == "concept_figs":
            namespace["render_pdfs"]()
        print("[%s done in %.1f s]" % (mod, time.time()-t0))
    print("\nFigures are in figures/generated/; data are in figures/data/.")
