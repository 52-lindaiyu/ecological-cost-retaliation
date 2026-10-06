"""Higher-seed replication of the fragility and group-size sweeps (review response)."""
import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd
from commons.core import Params, optimal_escapement, x_sharp
from commons.marl import rollout
from commons.marl_batch import train_batch, distil_v, fit_escapement_v
from commons.config import BASE

OUT = "results"

def sweep(param, values, lams, seeds, tag):
    rows = []
    for v in values:
        pr = Params(**{**BASE, param: v})
        opt = optimal_escapement(pr); xs = x_sharp(pr)
        specs = [dict(mode="social", lam=l, seed=s) for l in lams for s in range(seeds)]
        Qs, Vs = train_batch(pr, specs, episodes=12000)
        for k, (sp, Q, V) in enumerate(zip(specs, Qs, Vs)):
            ev = rollout(Q, pr, episodes=200, T=100, seed=31000 + k)
            d = distil_v(Q, V, pr)
            y_fit, r2 = fit_escapement_v(d["eff_x"], d["centres"], pr, x_floor=xs)
            rows.append(dict(**{param: v}, lam=sp["lam"], seed=sp["seed"],
                             collapse=ev["collapse"], rent_pc=ev["rent_pc"],
                             rent_frac=ev["rent_pc"]/opt["rent_pc"], y_fit=y_fit, r2=r2))
        print("  ", param, v, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/{tag}.csv", index=False)

if __name__ == "__main__":
    print("D lambda x theta, 8 seeds", flush=True)
    sweep("theta", [0.08, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30], [0.0, 0.5, 1.0, 2.0, 4.0, 6.0], 8,
          "review_lambda_theta8")
    print("E group size, 8 seeds", flush=True)
    sweep("N", [3, 5, 8, 12], [0.0, 2.0, 4.0], 8, "review_group_size8")
    print("review2 done", flush=True)
