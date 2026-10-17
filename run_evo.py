"""
Evolutionary validation: basins of attraction rather than dominant strategies.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from commons.core import Params, optimal_escapement, steward_window
from commons.exploit import (MaxEffort, Constant, Escapement, MatchAverage,
                             GrimMax, Steward, Tabular)
from commons.evo import evolve
from commons.marl import SBIN_EDGES
from commons.config import BASE

OUT = "results"


def pool(pr, distilled=None):
    opt = optimal_escapement(pr)
    y_d, e_d = opt["y"], opt["E"] / pr.N
    win = steward_window(pr, opt["rent_pc"])
    lo = win["lo"] if win["lo"] else 0.25
    hi = win["hi"]
    xh = 0.5 * (lo + hi) if hi > lo else lo * 1.05
    facs = [lambda: MaxEffort(pr),
            lambda: Constant(pr, e_d),
            lambda: Escapement(pr, y_d),
            lambda: MatchAverage(pr, e_d),
            lambda: GrimMax(pr, y_d),
            lambda: Steward(pr, y_d, xh)]
    names = ["MaxEffort", "Moderate", "Escapement", "MatchAverage",
             "GrimMax", "Steward"]
    if distilled:
        for tag, tab in distilled.items():
            facs.append(lambda t=tab, n=tag: Tabular(pr, SBIN_EDGES,
                                                     np.array(t), name=n))
            names.append(tag)
    return facs, names


def run_full(args):
    kw, seed = args
    pr = Params(**{**BASE, **kw})
    dist = None
    p = f"{OUT}/distilled_rules.json"
    if os.path.exists(p):
        dist = json.load(open(p))
    facs, names = pool(pr, dist)
    res = evolve(facs, names, pr, pop=180, gens=150, T=70, beta=40.0, seed=seed)
    row = dict(seed=seed, **kw, collapse_final=float(res["collapse"][-20:].mean()),
               welfare_final=float(res["welfare"][-20:].mean()))
    for n, v in zip(names, res["final"]):
        row[f"f_{n}"] = float(v)
    return row, res["freq"], res["collapse"], names


def run_basin(args):
    kw, f_stew, seed = args
    pr = Params(**{**BASE, **kw})
    opt = optimal_escapement(pr)
    y_d = opt["y"]
    win = steward_window(pr, opt["rent_pc"])
    lo = win["lo"] if win["lo"] else 0.25
    xh = 0.5 * (lo + win["hi"]) if win["hi"] > lo else lo * 1.05
    facs = [lambda: Steward(pr, y_d, xh), lambda: MaxEffort(pr),
            lambda: Escapement(pr, y_d)]
    names = ["Steward", "MaxEffort", "Escapement"]
    pop = 120
    n_s = int(round(f_stew * pop))
    rest = pop - n_s
    init = [n_s, rest // 2, rest - rest // 2]
    res = evolve(facs, names, pr, pop=pop, gens=80, T=60, beta=40.0,
                 seed=seed, init=init, mu_mut=0.0)
    return dict(**kw, f_init=f_stew, seed=seed,
                f_steward_final=float(res["final"][0]),
                f_max_final=float(res["final"][1]),
                f_esc_final=float(res["final"][2]),
                collapse_final=float(res["collapse"][-20:].mean()),
                welfare_final=float(res["welfare"][-20:].mean()))


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- full pool tournaments, several fragilities, many replicates ----- #
    jobs = [] if os.path.exists(f"{OUT}/evo_tournament.csv") else \
        [({"theta": t}, s) for t in [0.10, 0.15, 0.20, 0.25] for s in range(12)]
    if jobs:
        with ThreadPoolExecutor(max_workers=2) as ex:
            out = list(ex.map(run_full, jobs, chunksize=1))
        rows = [o[0] for o in out]
        pd.DataFrame(rows).to_csv(f"{OUT}/evo_tournament.csv", index=False)
        traj = {}
        for (kw, s), o in zip(jobs, out):
            if s == 0:
                traj[f"theta{kw['theta']}"] = dict(freq=o[1].tolist(),
                                                   collapse=o[2].tolist(),
                                                   names=o[3])
        json.dump(traj, open(f"{OUT}/evo_trajectories.json", "w"))
        print("tournaments done", flush=True)
    else:
        print("tournaments already present - skipped", flush=True)

    # ---- basins of attraction -------------------------------------------- #
    fr = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
    jobs = [({"theta": t}, f, s) for t in [0.10, 0.15, 0.20, 0.25]
            for f in fr for s in range(4)]
    with ThreadPoolExecutor(max_workers=2) as ex:
        rows = list(ex.map(run_basin, jobs, chunksize=2))
    pd.DataFrame(rows).to_csv(f"{OUT}/evo_basins.csv", index=False)
    print("basins done", flush=True)


if __name__ == "__main__":
    main()
