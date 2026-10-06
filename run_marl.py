"""
MARL strategy discovery (batched), symbolic distillation, and certified
exploitability of the distilled rules.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import numpy as np
import pandas as pd

from commons.core import Params, optimal_escapement, x_sharp
from commons.marl import rollout, SBIN_EDGES
from commons.marl_batch import (train_batch, distil_v, fit_escapement_v,
                                liquidation_index)
from commons.exploit import Tabular, evaluate
from commons.config import BASE

OUT = "results"
EPISODES = 12000


def _fill(eff_x):
    """Forward/backward fill uncovered stock bins so the rule is well defined."""
    v = np.array(eff_x, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros_like(v)
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok])


def run_batch(pr, specs, tag, rollout_eps=300, certify=True):
    Qs, Vs = train_batch(pr, specs, episodes=EPISODES)
    xs = x_sharp(pr)
    opt = optimal_escapement(pr)
    rows = []
    for k, (sp, Q, V) in enumerate(zip(specs, Qs, Vs)):
        ev = rollout(Q, pr, episodes=rollout_eps, T=100, seed=9000 + k)
        d = distil_v(Q, V, pr)
        y_fit, r2 = fit_escapement_v(d["eff_x"], d["centres"], pr, x_floor=xs)
        liq = liquidation_index(d["eff_x"], d["centres"], pr, xs) if xs else np.nan
        on = d["eff_x"][d["centres"] > (xs if xs else 0)]
        on = on[np.isfinite(on)]
        mono = bool(len(on) > 1 and np.all(np.diff(on) >= -1e-9))
        row = dict(tag=tag, mode=sp.get("mode", "social"), lam=sp.get("lam", 0.0),
                   seed=sp.get("seed", 0), exec_error=sp.get("error", 0.0),
                   theta=pr.theta, N=pr.N,
                   rent_pc=ev["rent_pc"], collapse=ev["collapse"],
                   x_end=ev["x_end"], gini=ev["gini"],
                   pi_star=opt["rent_pc"], y_star=opt["y"],
                   y_fit=y_fit, r2=r2, liq_index=liq, monotone_onpath=mono,
                   frac_covered=d["frac_covered"],
                   sanction_mean=float(np.nanmean(d["sanction"])),
                   eff_x=json.dumps([None if not np.isfinite(v) else float(v)
                                     for v in d["eff_x"]]),
                   y_implied=json.dumps([None if not np.isfinite(v) else float(v)
                                         for v in d["y_implied"]]))
        if certify:
            R = Tabular(pr, SBIN_EDGES, _fill(d["eff_x"]), name="Distilled")
            ex = evaluate(R, pr, delta=0.99, nx=251, na=21)
            row.update(d_rent_self=ex["rent_self"], d_expl=ex["exploitability"],
                       d_col_worst=ex["collapse_worst"], d_col_br=ex["collapse_br"])
        rows.append(row)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pr = Params(**BASE)

    # ---- A. objective sweep at the baseline ------------------------------ #
    lams = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    specs = [dict(mode="social", lam=l, seed=s) for l in lams for s in range(5)]
    specs += [dict(mode="dual", lam=0.0, seed=s) for s in range(5)]
    rows = run_batch(pr, specs, "objective")
    pd.DataFrame(rows).to_csv(f"{OUT}/marl_objective_sweep.csv", index=False)
    print("objective sweep done", flush=True)

    # ---- B. critical prosociality vs fragility --------------------------- #
    out = []
    lams2 = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    for th in [0.08, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]:
        p2 = Params(**{**BASE, "theta": th})
        sp = [dict(mode="social", lam=l, seed=s) for l in lams2 for s in range(3)]
        out += run_batch(p2, sp, f"theta{th}", rollout_eps=200, certify=False)
        print("  theta", th, "done", flush=True)
    pd.DataFrame(out).to_csv(f"{OUT}/marl_lambda_theta.csv", index=False)
    print("lambda x theta done", flush=True)

    # ---- C. group-size sweep --------------------------------------------- #
    out = []
    for n in [3, 5, 8, 12]:
        p2 = Params(**{**BASE, "N": n})
        sp = [dict(mode="social", lam=l, seed=s) for l in [0.0, 2.0, 4.0]
              for s in range(3)]
        out += run_batch(p2, sp, f"N{n}", rollout_eps=200, certify=False)
        print("  N", n, "done", flush=True)
    pd.DataFrame(out).to_csv(f"{OUT}/marl_group_size.csv", index=False)
    print("group size done", flush=True)

    # ---- D. execution error during learning ------------------------------ #
    sp = [dict(mode="social", lam=l, seed=s, error=e)
          for l in [0.0, 2.0, 4.0] for e in [0.0, 0.02, 0.05, 0.10]
          for s in range(3)]
    rows = run_batch(pr, sp, "exec_error", rollout_eps=200, certify=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/marl_exec_error.csv", index=False)
    print("exec error done", flush=True)


if __name__ == "__main__":
    main()
