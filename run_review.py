"""
Additional experiments added in response to the pre-submission review.

A. Monitoring noise.  The main text assumes the aggregate monitoring signal is
   observed without error.  Here the signal itself is corrupted (the residents
   see total effort with multiplicative noise), which is the failure mode a
   referee asks about: noise can fire a sanction that nobody deserved.  Reports
   self-play collapse probability, realised rent and the fraction of periods
   spent in sanction mode, per rule and noise level.

B. Fragility control.  The claim is that the tipping point, not the
   parameterisation, is what makes retaliation ecologically costly.  The control
   is to weaken the threshold: certify the same three rules along theta and show
   that uncapped retaliation is indistinguishable from the steward strategy
   where the resource cannot be tipped, and separates only above a fragility
   threshold.

C. Replication of the learning result.  The lambda sweep is rerun with more
   seeds and a finer grid around the reported critical prosocial weight, so the
   threshold is bracketed rather than read off a coarse grid, and collapse
   probabilities carry Wilson intervals.

Outputs: results/review_monitoring_noise.csv, results/review_fragility_control.csv,
         results/review_lambda_fine.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import numpy as np
import pandas as pd

from commons.core import (Params, optimal_escapement, steward_window, x_sharp, payoff, harvest_fraction, shock,
                          Phi, K_CAP)
from commons.exploit import Escapement, GrimMax, Steward, Rule, evaluate
from commons.config import BASE

OUT = "results"


# --------------------------------------------------------------------------- #
# A. monitoring noise
# --------------------------------------------------------------------------- #
class NoisyMonitor(Rule):
    """
    Wrap a rule so that the aggregate signal it conditions on is observed with
    multiplicative noise: the residents see E_obs = E (1 + nu xi) and
    y_obs = x (1 - q E_obs) instead of the true values.  Effort choice and
    ecology are unaffected - only the monitoring channel is degraded.
    """

    def __init__(self, base: Rule, nu: float, rng):
        self.base, self.nu, self.rng = base, float(nu), rng
        self.name = f"{base.name}|nu={nu:.2f}"
        self.n_mode = base.n_mode
        self.coop_mode = base.coop_mode

    def eff(self, x, mode):
        return self.base.eff(x, mode)

    def nxt(self, mode, x, y, E):
        if self.nu > 0:
            fac = 1.0 + self.nu * self.rng.standard_normal(np.shape(E))
            E_obs = np.maximum(np.asarray(E) * fac, 0.0)
            y_obs = np.maximum(np.asarray(x) * (1.0 - self.base.pr.q * E_obs), 0.0)
        else:
            E_obs, y_obs = E, y
        return self.base.nxt(mode, x, y_obs, E_obs)


def self_play_modes(rule: Rule, pr: Params, T=200, reps=400, seed=0, x0=None):
    """Self-play that also records how much time the group spends sanctioning."""
    rng = np.random.default_rng(seed)
    x_start = K_CAP * 0.9 if x0 is None else float(x0)
    rent = np.zeros(reps)
    collapsed = np.zeros(reps, dtype=bool)
    sanction_frac = np.zeros(reps)
    one = np.array([0.0])
    for rp in range(reps):
        x, mode, tot, nsanc, steps = x_start, rule.coop_mode, 0.0, 0, 0
        for t in range(T):
            one[0] = x
            e0 = float(np.asarray(rule.eff(one, mode))[0])
            e = np.full(pr.N, e0)
            E = float(e.sum())
            y = max(x * (1.0 - float(harvest_fraction(E, pr))), 0.0)
            tot += float(payoff(e, E, x, pr).mean())
            steps += 1
            nsanc += int(mode != rule.coop_mode)
            xn = float(Phi(y, pr)) * (1.0 + pr.sigma * shock(rng))
            xn = float(np.clip(xn, 0.0, K_CAP))
            mode = int(np.asarray(rule.nxt(mode, one, np.array([y]),
                                           np.array([E])))[0])
            if y <= pr.A or xn <= pr.A:
                collapsed[rp] = True
                break
            x = xn
        rent[rp] = tot / T
        sanction_frac[rp] = nsanc / max(steps, 1)
    return dict(rent_pc=float(rent.mean()), collapse=float(collapsed.mean()),
                sanction_frac=float(sanction_frac.mean()), reps=reps)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def build_rules(pr, tol=1e-3):
    opt = optimal_escapement(pr)
    win = steward_window(pr, opt["rent_pc"])
    lo, hi = win["lo"], win["hi"]
    xh = 0.5 * (lo + hi) if (lo is not None and hi > lo) else None
    rules = [("Constant escapement", Escapement(pr, opt["y"])),
             ("Uncapped retaliation", GrimMax(pr, opt["y"], tol=tol))]
    if xh is not None:
        rules.append(("Steward", Steward(pr, opt["y"], xh, tol=tol)))
        rules.append(("Steward + forgiveness",
                      Steward(pr, opt["y"], xh, tol=tol, forgive_x=0.75)))
    return rules, opt, win, xh


def part_a():
    pr = Params(**BASE)
    rules, opt, win, xh = build_rules(pr)
    rows = []
    for nu in [0.0, 0.02, 0.05, 0.10, 0.20]:
        for nm, R in rules:
            rng = np.random.default_rng(12345)
            res = self_play_modes(NoisyMonitor(R, nu, rng), pr, seed=101)
            k = int(round(res["collapse"] * res["reps"]))
            lo_ci, hi_ci = wilson(k, res["reps"])
            rows.append(dict(rule=nm, nu=nu, collapse=res["collapse"],
                             collapse_lo=lo_ci, collapse_hi=hi_ci,
                             rent_pc=res["rent_pc"],
                             rent_frac=res["rent_pc"] / opt["rent_pc"],
                             sanction_frac=res["sanction_frac"],
                             reps=res["reps"], x_hold=xh))
        print("  nu", nu, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/review_monitoring_noise.csv", index=False)


def part_b():
    rows = []
    for th in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]:
        pr = Params(**{**BASE, "theta": th})
        rules, opt, win, xh = build_rules(pr)
        xs = x_sharp(pr)
        for nm, R in rules:
            if nm == "Steward + forgiveness":
                continue
            ev = evaluate(R, pr, delta=0.99, nx=251, na=21)
            rows.append(dict(theta=th, rule=nm, x_sharp=xs, x_hold=xh,
                             win_lo=win["lo"], win_hi=win["hi"],
                             pi_star=opt["rent_pc"],
                             rent_self=ev["rent_self"], br_value=ev["br_value"],
                             exploitability=ev["exploitability"], v_self=ev["v_self"], tol_bound=ev["tol_bound"],
                             collapse_worst=ev["collapse_worst"],
                             collapse_br=ev["collapse_br"]))
        print("  theta", th, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/review_fragility_control.csv", index=False)


def part_c():
    from commons.marl import SBIN_EDGES, rollout
    from commons.marl_batch import train_batch, distil_v, fit_escapement_v
    from commons.exploit import Tabular
    from run_marl import _fill
    pr = Params(**BASE)
    opt = optimal_escapement(pr)
    xs = x_sharp(pr)
    lams = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 1.00]
    seeds = range(16)
    specs = [dict(mode="social", lam=l, seed=s) for l in lams for s in seeds]
    Qs, Vs = train_batch(pr, specs, episodes=12000)
    rows = []
    for k, (sp, Q, V) in enumerate(zip(specs, Qs, Vs)):
        ev = rollout(Q, pr, episodes=200, T=100, seed=20000 + k)
        d = distil_v(Q, V, pr)
        y_fit, r2 = fit_escapement_v(d["eff_x"], d["centres"], pr, x_floor=xs)
        row = dict(lam=sp["lam"], seed=sp["seed"], collapse=ev["collapse"],
                   rent_pc=ev["rent_pc"], rent_frac=ev["rent_pc"] / opt["rent_pc"],
                   gini=ev["gini"], y_fit=y_fit, r2=r2,
                   sanction_mean=float(np.nanmean(d["sanction"])))
        if sp["lam"] in (0.25, 1.00) and sp["seed"] < 8:
            R = Tabular(pr, SBIN_EDGES, _fill(d["eff_x"]), name="Distilled")
            ex = evaluate(R, pr, delta=0.99, nx=251, na=21)
            row.update(d_expl=ex["exploitability"], d_col_worst=ex["collapse_worst"])
        rows.append(row)
        if (k + 1) % 24 == 0:
            print("  trained", k + 1, "/", len(specs), flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/review_lambda_fine.csv", index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("A monitoring noise", flush=True); part_a()
    print("B fragility control", flush=True); part_b()
    if "--with-c" in __import__("sys").argv:
        print("C lambda replication", flush=True); part_c()
    print("review runs done", flush=True)
