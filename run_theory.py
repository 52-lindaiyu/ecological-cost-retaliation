"""Analytic grids: regimes, sanction window, critical group size."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from commons.core import (Params, x_sharp, critical_effort, optimal_escapement,
                          min_defector_rent, critical_group_size, classify,
                          steward_window, E_hold, Phi, phi_inv)
from commons.config import BASE

OUT = "results"


def cell(args):
    th, N = args
    pr = Params(**{**BASE, "theta": float(th), "N": int(N)})
    f = classify(pr)
    opt = optimal_escapement(pr)
    win = steward_window(pr, opt["rent_pc"])
    Nst, Rt, pis = critical_group_size(pr)
    return dict(theta=th, N=N, regime=f["regime"], feasible=bool(f["feasible"]),
                E_crit=f["E_crit"], x_crit=f["x_crit"],
                x_sharp=(np.nan if f["x_sharp"] is None else f["x_sharp"]),
                pi_sharp=f["pi_sharp"], pi_star=opt["rent_pc"],
                R_star=opt["rent_total"], y_star=opt["y"], x_star=opt["x"],
                E_star=opt["E"], e_star=opt["E"] / N,
                N_star=Nst, win_lo=win["lo"] if win["lo"] else np.nan,
                win_hi=win["hi"], win_width=win["width"],
                x_enf=win["x_enf"] if win["x_enf"] is not None else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    thetas = np.round(np.arange(0.04, 0.4501, 0.005), 5)
    Ns = list(range(2, 25))
    jobs = [(t, n) for t in thetas for n in Ns]
    with ThreadPoolExecutor(max_workers=2) as ex:
        rows = list(ex.map(cell, jobs, chunksize=8))
    pd.DataFrame(rows).to_csv(f"{OUT}/theory_grid.csv", index=False)

    # 1-D slices at the baseline
    rows = []
    for th in np.round(np.arange(0.04, 0.4501, 0.0025), 6):
        rows.append(cell((th, BASE["N"])))
    pd.DataFrame(rows).to_csv(f"{OUT}/theory_theta_slice.csv", index=False)

    rows = [cell((BASE["theta"], n)) for n in range(2, 41)]
    pd.DataFrame(rows).to_csv(f"{OUT}/theory_N_slice.csv", index=False)

    # the regeneration map and the effort-space fold, for figure 1
    pr = Params(**BASE)
    yv = np.linspace(0, 1, 801)
    pd.DataFrame(dict(y=yv, Phi=Phi(yv, pr))).to_csv(f"{OUT}/phi_map.csv", index=False)
    Ev = np.linspace(0.0, 3.2, 321)
    ufp, lfp = [], []
    from commons.core import upper_fixed_point, lower_fixed_point
    for E in Ev:
        u = upper_fixed_point(E, pr)
        l = lower_fixed_point(E, pr)
        ufp.append(np.nan if u is None else u)
        lfp.append(np.nan if l is None else l)
    pd.DataFrame(dict(E=Ev, x_upper=ufp, x_lower=lfp)).to_csv(
        f"{OUT}/effort_bifurcation.csv", index=False)

    # sustainable rent curve
    ys = np.linspace(pr.A + 1e-6, 1 - 1e-6, 2001)
    from commons.core import sustainable_state
    x, H, E, rent = sustainable_state(ys, pr)
    pd.DataFrame(dict(y=ys, x=x, H=H, E=E, rent=rent,
                      rent_pc=rent / pr.N)).to_csv(f"{OUT}/rent_curve.csv", index=False)
    print("theory grids written")


if __name__ == "__main__":
    main()
