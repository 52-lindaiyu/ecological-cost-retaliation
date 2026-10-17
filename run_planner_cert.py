import json, numpy as np
from commons.core import Params, planner_optimum, optimal_escapement, min_defector_rent
from commons.config import BASE
rows = []
for th in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
    pr = Params(**{**BASE, "theta": th}); opt = optimal_escapement(pr)
    pl = planner_optimum(pr, nx=3201, ny=3201, check_grid=12801)
    pis, xs = min_defector_rent(pr)
    Rst = opt["rent_total"]; g = pl["gain_total"]; eps = pl["eps_cert"]
    rows.append(dict(theta=th, R_star=Rst, g_planner=g, rel_diff=(g-Rst)/Rst, iters=pl["iters"], converged=pl["converged"],
                     span=pl["span"], eps_cert=eps, eps_cert_frac=eps/g, lipschitz_h=pl["lipschitz_h"],
                     pi_sharp=pis, N_star=Rst/pis if pis>0 else np.inf, N_star_upper=(g+eps)/pis if pis>0 else np.inf))
    print(th, f"g={g:.6f} R*={Rst:.6f} eps={eps:.2e} N*={rows[-1]['N_star']:.3f}..{rows[-1]['N_star_upper']:.3f}", flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv("results/planner_certificate.csv", index=False)
