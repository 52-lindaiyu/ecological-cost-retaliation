"""Pick representative distilled MARL rules for the evolutionary tournament."""
import json
import numpy as np
import pandas as pd

def _median_table(sub):
    """Median distilled effort per stock bin, ignoring uncovered bins, then
    interpolated across any bin no replicate covered."""
    tabs = np.array([[np.nan if v is None else float(v) for v in json.loads(s)]
                     for s in sub["eff_x"]], dtype=float)
    med = np.nanmedian(tabs, axis=0)
    ok = np.isfinite(med)
    idx = np.arange(len(med))
    return [float(v) for v in np.interp(idx, idx[ok], med[ok])]


d = pd.read_csv("results/marl_objective_sweep.csv")
out = {}

# the source-paper dual objective
sub = d[d["mode"] == "dual"]
if len(sub):
    out["MARL-dual"] = _median_table(sub)

# a competitive and a strongly prosocial learner
for lam, tag in [(0.0, "MARL-selfish"), (4.0, "MARL-prosocial")]:
    sub = d[(d["mode"] == "social") & (np.isclose(d["lam"], lam))]
    if len(sub):
        out[tag] = _median_table(sub)

json.dump(out, open("results/distilled_rules.json", "w"), indent=1)
print({k: v for k, v in out.items()})
