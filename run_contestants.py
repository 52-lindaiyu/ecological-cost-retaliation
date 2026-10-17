"""Same-model certification of the three distilled contestants used in the evolutionary
tournament (Extended Data): self-play efficiency, worst-case collapse, exploitability."""
import json, numpy as np, pandas as pd
from commons.core import Params, optimal_escapement
from commons.exploit import Tabular, evaluate
from commons.marl import SBIN_EDGES
from commons.config import BASE
pr = Params(**BASE); opt = optimal_escapement(pr)
rules = json.load(open("results/distilled_rules.json", encoding="utf-8"))
rows = []
for name, spec in rules.items():
    eff = np.array(spec["eff"] if isinstance(spec, dict) and "eff" in spec else spec, dtype=float)
    R = Tabular(pr, SBIN_EDGES, eff, name=name)
    r = evaluate(R, pr, delta=0.99, nx=251, na=21)
    rows.append(dict(rule=name, rent_self=r["rent_self"], rent_frac=r["rent_self"]/opt["rent_pc"],
                     collapse_self=r["collapse_self"], v_self=r["v_self"], br=r["br_value"],
                     exploitability=r["exploitability"], collapse_worst=r["collapse_worst"], collapse_br=r["collapse_br"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/contestants_certified.csv", index=False)
