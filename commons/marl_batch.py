"""
Batched independent tabular Q-learning.

Each "run" in a batch carries its OWN Q table, its own exploration stream and
its own environment, so a batch of R runs is exactly R independent learners --
identical in distribution to running them one at a time -- but advanced with
numpy over the batch axis instead of a Python loop per run.  This matters here
only for speed: the per-step work is tiny, so the pure-Python loop was
dominated by interpreter overhead.

All runs in one batch must share the resource parameters (Phi, q, N); the
reward shaping (mode, lambda) and the execution-error scale may differ per run.
"""
from __future__ import annotations

import numpy as np

from .core import Params, phi_table, K_CAP
from .marl import SBIN_EDGES, EBIN_EDGES, N_SBIN, N_EBIN, N_STATE, EFFORTS, NA


def _phi_interp(pr: Params, ngrid=40001):
    yt, zt = phi_table(pr.r, pr.theta)
    yg = np.linspace(0.0, 1.30 * K_CAP, ngrid)
    return yg, np.interp(yg, yt, zt)


def train_batch(pr: Params, specs, episodes=12000, T=60, alpha=0.15,
                gamma=0.95, eps0=0.20, eps_min=0.02, eps_every=2000,
                eps_decay=0.85):
    """
    specs : list of dicts with keys
              mode  : 'social' | 'dual'
              lam   : float (social axis; ignored for 'dual')
              seed  : int
              error : float, optional execution-error sd (default 0)
    Returns a list of Q tables, each of shape (N, N_STATE, NA).
    """
    Rn = len(specs)
    N = pr.N
    yg, zg = _phi_interp(pr)
    lam = np.array([float(s.get("lam", 0.0)) for s in specs])
    is_dual = np.array([s.get("mode", "social") == "dual" for s in specs])
    err = np.array([float(s.get("error", 0.0)) for s in specs])
    rngs = [np.random.default_rng(int(s.get("seed", 0)) + 7919) for s in specs]

    Q = np.zeros((Rn, N, N_STATE, NA))
    vis = np.zeros((Rn, N, N_STATE))
    ar = np.arange(Rn)[:, None]
    ai = np.arange(N)[None, :]
    w_dual = 0.8

    for ep in range(episodes):
        eps = max(eps_min, eps0 * (eps_decay ** (ep // eps_every)))
        # one common stream per run keeps runs independent and reproducible
        u = np.stack([r.random(4 + T * (3 * N + 1)) for r in rngs])
        x = 0.35 + 0.60 * u[:, 0]
        s = (N_SBIN * N_EBIN + np.searchsorted(SBIN_EDGES, x)).astype(int)
        alive = np.ones(Rn, dtype=bool)
        ptr = 1
        for t in range(T):
            Qs = Q[ar, ai, s[:, None], :]                     # (Rn, N, NA)
            greedy = Qs.argmax(axis=2)
            rnd = u[:, ptr:ptr + N]; ptr += N
            explore = rnd < eps
            rnd2 = u[:, ptr:ptr + N]; ptr += N
            act = np.where(explore, (rnd2 * NA).astype(int).clip(0, NA - 1), greedy)
            e = EFFORTS[act]
            rnd3 = u[:, ptr:ptr + N]; ptr += N
            if err.any():
                e = np.clip(e + err[:, None] * (rnd3 - 0.5) * 3.4641, 0.0, pr.ebar)
            E = e.sum(axis=1)
            frac = np.minimum(pr.q * E, 1.0)                    # rationed removal
            y = np.maximum(x * (1.0 - frac), 0.0)
            share = np.where(E[:, None] > 0, e / np.where(E[:, None] > 0, E[:, None], 1.0), 0.0)
            ui = pr.p * share * (frac * x)[:, None] - pr.c * e  # (Rn, N) payoffs, sum_i h_i <= x
            ubar = ui.mean(axis=1)
            # reward shaping
            W_soc = np.where(lam[:, None] >= 0,
                             ui + lam[:, None] * ubar[:, None],
                             ui + np.abs(lam)[:, None] * (ui - ubar[:, None]))
            W_dual = w_dual * ui + (1 - w_dual) * ubar[:, None] * (ui >= ubar[:, None])
            W = np.where(is_dual[:, None], W_dual, W_soc)
            xn = np.interp(y, yg, zg)
            xn = np.clip(xn * (1.0 + pr.sigma * (u[:, ptr] - 0.5) * 3.4641), 0.0, K_CAP)
            ptr += 1
            dead = (y <= pr.A) | (xn <= pr.A)
            e_pc = E / N
            sn = (np.searchsorted(SBIN_EDGES, xn) * N_EBIN
                  + np.searchsorted(EBIN_EDGES, e_pc)).astype(int)
            Qn = Q[ar, ai, sn[:, None], :].max(axis=2)          # (Rn, N)
            tgt = W + gamma * (~dead)[:, None] * Qn
            cur = Q[ar, ai, s[:, None], act]
            upd = alpha * (tgt - cur) * alive[:, None]
            Q[ar, ai, s[:, None], act] = cur + upd
            vis[ar, ai, s[:, None]] += alive[:, None]
            x, s = xn, sn
            alive = alive & (~dead)
            if not alive.any():
                break
    return [Q[i] for i in range(Rn)], [vis[i] for i in range(Rn)]


def distil_v(Q, vis, pr: Params, min_visits=30):
    """
    Symbolic read-out of a learned Q table, weighted by state visitation.

    Unvisited states have an all-zero Q row, whose argmax is effort 0; taking
    an unweighted mode over monitoring bins therefore reports "stop harvesting"
    for states the learner never saw.  We aggregate only over (agent,
    monitoring-bin) cells with at least `min_visits` visits and report the
    coverage so the reader can see which stock bins are actually supported.
    """
    pol = Q.argmax(axis=2)                                  # (N, N_STATE)
    centres = np.r_[SBIN_EDGES[0] / 2,
                    (SBIN_EDGES[:-1] + SBIN_EDGES[1:]) / 2,
                    (SBIN_EDGES[-1] + 1.0) / 2]
    eff_x = np.full(N_SBIN, np.nan)
    cover = np.zeros(N_SBIN)
    eff_tab = np.full((N_SBIN, N_EBIN), np.nan)
    for b in range(N_SBIN):
        wtot = 0.0
        acc = np.zeros(NA)
        for m in range(N_EBIN):
            st = b * N_EBIN + m
            w = vis[:, st]
            ok = w >= min_visits
            if ok.any():
                np.add.at(acc, pol[ok, st], w[ok])
                wtot += float(w[ok].sum())
                cnt = np.zeros(NA)
                np.add.at(cnt, pol[ok, st], w[ok])
                eff_tab[b, m] = EFFORTS[int(cnt.argmax())]
        cover[b] = wtot
        if wtot > 0:
            eff_x[b] = EFFORTS[int(acc.argmax())]
    y_impl = centres * (1.0 - pr.q * pr.N * eff_x)
    sanction = eff_tab[:, -1] - eff_tab[:, 0]
    return dict(eff_tab=eff_tab, eff_x=eff_x, centres=centres, coverage=cover,
                y_implied=y_impl, sanction=sanction,
                frac_covered=float(np.mean(np.isfinite(eff_x))))


def liquidation_index(eff_x, centres, pr: Params, x_floor):
    """
    Fraction of covered stock bins BELOW x_floor whose distilled effort is the
    effort cap: the learners' response to an already-doomed stock.
    """
    lo = np.isfinite(eff_x) & (centres <= x_floor)
    if not lo.any():
        return np.nan
    return float(np.mean(eff_x[lo] >= pr.ebar - 1e-9))


def fit_escapement_v(eff_x, centres, pr: Params, x_floor=None):
    """
    Best constant-escapement fit over the covered stock bins above `x_floor`.

    Restricting to x > x_floor (we pass the unilateral tipping stock x_sharp) is
    not curve-fitting convenience: below x_sharp the resource is doomed whatever
    anyone does, so the value of every continuation is ~0 there and liquidation
    is individually rational.  Mixing that region into the fit would test the
    learned rule against a rule shape that cannot apply there.
    """
    ok = np.isfinite(eff_x)
    if x_floor is not None:
        ok = ok & (centres > x_floor)
    if ok.sum() < 3:
        return np.nan, np.nan
    ex, cx = eff_x[ok], centres[ok]
    cand = np.linspace(0.02, 0.98, 481)
    ss_tot = float(((ex - ex.mean()) ** 2).sum())
    best, bt = -np.inf, np.nan
    for yt in cand:
        pred = np.clip((1.0 / (pr.N * pr.q)) * (1.0 - yt / np.maximum(cx, 1e-9)),
                       0.0, pr.ebar)
        pred = EFFORTS[np.abs(EFFORTS[None, :] - pred[:, None]).argmin(axis=1)]
        r2 = 1.0 - float(((ex - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        if r2 > best:
            best, bt = r2, yt
    return float(bt), float(best)
