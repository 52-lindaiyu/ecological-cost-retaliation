"""
Full-factorial robustness: discount factor x regeneration noise x group size x
monitoring tolerance, for the three diagnostic rules; plus execution error and
finite-horizon checks in the main text rather than the supplement.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from commons.core import Params, optimal_escapement, steward_window, x_sharp
from commons.exploit import Escapement, GrimMax, Steward, evaluate, self_play
from commons.config import BASE

OUT = "results"


def build(kind, pr, tol):
    opt = optimal_escapement(pr)
    y_d = opt["y"]
    win = steward_window(pr, opt["rent_pc"])
    lo = win["lo"] if win["lo"] else 0.25
    hi = win["hi"]
    xh = 0.5 * (lo + hi) if hi > lo else lo * 1.05
    if kind == "Escapement":
        return Escapement(pr, y_d), opt, win
    if kind == "GrimMax":
        return GrimMax(pr, y_d, tol=tol), opt, win
    return Steward(pr, y_d, xh, tol=tol), opt, win


def cell(args):
    kind, delta, sigma, N, tol, err = args
    pr = Params(**{**BASE, "sigma": sigma, "N": N})
    R, opt, win = build(kind, pr, tol)
    r = evaluate(R, pr, delta=delta, nx=251, na=21)
    sp = self_play(R, pr, error=err, reps=250, T=200, seed=7)
    sp_short = self_play(R, pr, error=err, reps=250, T=20, seed=8)
    xs = x_sharp(pr)
    return dict(rule=kind, delta=delta, sigma=sigma, N=N, tol=tol, exec_err=err,
                horizon_eff=1.0 / (1.0 - delta),
                rent_self=r["rent_self"], v_self=r["v_self"], br=r["br_value"],
                exploitability=r["exploitability"], tol_bound=r["tol_bound"],
                collapse_br=r["collapse_br"], collapse_worst=r["collapse_worst"],
                collapse_sim=sp["collapse"], rent_sim=sp["rent_pc"],
                collapse_sim_T20=sp_short["collapse"],
                pi_star=opt["rent_pc"], x_sharp=(np.nan if xs is None else xs),
                win_width=win["width"])


def main():
    os.makedirs(OUT, exist_ok=True)
    deltas = [0.90, 0.99, 0.999]
    sigmas = [0.0, 0.02, 0.05, 0.10]
    Ns = [3, 5, 8, 12]
    tols = [0.001, 0.05]
    jobs = [(k, d, s, n, t, 0.0)
            for k in ["Escapement", "GrimMax", "Steward"]
            for d in deltas for s in sigmas for n in Ns for t in tols]
    jobs += [(k, 0.99, s, n, 0.001, e)
             for k in ["Escapement", "GrimMax", "Steward"]
             for s in [0.0, 0.02, 0.05] for n in [3, 5, 8, 12]
             for e in [0.02, 0.05, 0.10]]
    print("cells:", len(jobs), flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(cell, jobs, chunksize=4))
    pd.DataFrame(rows).to_csv(f"{OUT}/robustness_grid.csv", index=False)
    print("robustness done", flush=True)


if __name__ == "__main__":
    main()
