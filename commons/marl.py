"""
Strategy discovery by multi-agent reinforcement learning under
aggregate-only monitoring, plus symbolic distillation of the learned policies.

Observation available to every agent (identical across agents, by construction:
aggregate monitoring reveals the stock and last period's total effort, because
y/x = 1 - qE, but never who did what):
        (stock bin, last-period per-capita effort bin)  +  a first-round state.

Reward shaping variants
    'dual'      : the source paper's dual objective, ported to N players -
                  W = w*u_i + (1-w)*avg_i  if avg_i >= group mean, else w*u_i
    'social'    : single social-preference axis -
                  lam < 0  competitive  W = u_i + |lam| (avg_i - group mean)
                  lam >= 0 prosocial    W = u_i + lam * group mean
    'resilient' : 'social' plus an explicit terminal penalty on collapse.
"""
from __future__ import annotations

import numpy as np

from .core import Params, phi_table, K_CAP, payoff, harvest_fraction, shock

# ---- observation space ---------------------------------------------------- #
SBIN_EDGES = np.array([0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90])   # 8 stock bins
EBIN_EDGES = np.array([0.20, 0.45, 0.70])                           # 4 effort bins
N_SBIN, N_EBIN = len(SBIN_EDGES) + 1, len(EBIN_EDGES) + 1
N_STATE = N_SBIN * N_EBIN + N_SBIN          # + first-round states
EFFORTS = np.array([0.0, 0.25, 0.50, 0.75, 1.00])
NA = len(EFFORTS)


def sbin(x):
    return int(np.searchsorted(SBIN_EDGES, x))


def ebin(e_pc):
    return int(np.searchsorted(EBIN_EDGES, e_pc))


def state_of(x, e_pc, first):
    b = sbin(x)
    return (N_SBIN * N_EBIN + b) if first else (b * N_EBIN + ebin(e_pc))


# ---- fast regeneration lookup -------------------------------------------- #
class Fast:
    def __init__(self, pr: Params, ngrid=40001):
        yt, zt = phi_table(pr.r, pr.theta)
        self.yg = np.linspace(0.0, 1.30 * K_CAP, ngrid)
        self.zg = np.interp(self.yg, yt, zt)
        self.dy = self.yg[1] - self.yg[0]
        self.n = ngrid

    def phi(self, y):
        if y <= 0.0:
            return 0.0
        j = int(y / self.dy)
        if j >= self.n - 1:
            return float(self.zg[-1])
        f = (y - self.yg[j]) / self.dy
        return float(self.zg[j] * (1.0 - f) + self.zg[j + 1] * f)


# ---- training ------------------------------------------------------------ #
def train(pr: Params, mode="social", lam=0.0, w_dual=0.8, mu=0.0,
          episodes=20000, T=60, alpha=0.15, gamma=0.95, eps0=0.20,
          eps_min=0.02, seed=0, exec_error=0.0):
    rng = np.random.default_rng(seed)
    fast = Fast(pr)
    N = pr.N
    Q = rng.normal(0.0, 1e-3, (N, N_STATE, NA))
    eps = eps0
    for ep in range(episodes):
        if ep and ep % 2000 == 0:
            eps = max(eps_min, eps * 0.85)
        x = float(rng.uniform(0.35, 0.95))
        e_pc, first = 0.0, True
        S, A, U = [], [], []
        pay = np.zeros(N)
        dead = False
        for t in range(T):
            s = state_of(x, e_pc, first)
            greedy = Q[:, s, :].argmax(axis=1)
            rnd = rng.integers(0, NA, N)
            act = np.where(rng.random(N) < eps, rnd, greedy)
            e = EFFORTS[act]
            if exec_error > 0:
                flip = rng.random(N) < exec_error
                if flip.any():
                    e = e.copy()
                    e[flip] = EFFORTS[rng.integers(0, NA, int(flip.sum()))]
            E = float(e.sum())
            u = payoff(e, e.sum(), x, pr)
            pay += u
            y = max(x * (1.0 - float(harvest_fraction(E, pr))), 0.0)
            xn = fast.phi(y) * (1.0 + pr.sigma * shock(rng)) if pr.sigma > 0 \
                else fast.phi(y)
            xn = min(max(xn, 0.0), K_CAP)
            dead = (y <= pr.A) or (xn <= pr.A)
            S.append(s)
            A.append(act.copy())
            U.append(u.copy())
            e_pc, first = E / N, False
            x = 0.0 if dead else xn
            if dead:
                break
        L = len(S)
        avg = pay / L
        grp = float(avg.mean())
        # shaped reward
        for t in range(L):
            u = U[t]
            if mode == "dual":
                Wv = w_dual * u + np.where(avg >= grp, (1.0 - w_dual) * avg, 0.0)
            elif mode == "social":
                Wv = u + (lam * float(u.mean()) if lam >= 0
                          else abs(lam) * (avg - grp))
            else:  # resilient
                Wv = u + lam * float(u.mean())
            if mode == "resilient" and dead and t == L - 1:
                Wv = Wv - mu
            s, a = S[t], A[t]
            if t == L - 1:
                nxt = np.zeros(pr.N) if dead else Q[:, S[t], :].max(axis=1)
            else:
                nxt = Q[:, S[t + 1], :].max(axis=1)
            idx = np.arange(N)
            Q[idx, s, a] += alpha * (Wv + gamma * nxt - Q[idx, s, a])
    return Q


# ---- evaluation of a learned joint policy -------------------------------- #
def rollout(Q, pr: Params, episodes=400, T=100, seed=1234, x0=0.9,
            exec_error=0.0):
    rng = np.random.default_rng(seed)
    fast = Fast(pr)
    N = pr.N
    pol = Q.argmax(axis=2)
    rent = np.zeros(episodes)
    dead = np.zeros(episodes, dtype=bool)
    xend = np.zeros(episodes)
    gini = np.zeros(episodes)
    for k in range(episodes):
        x = float(x0)
        e_pc, first = 0.0, True
        pay = np.zeros(N)
        for t in range(T):
            s = state_of(x, e_pc, first)
            e = EFFORTS[pol[np.arange(N), s]]
            if exec_error > 0:
                flip = rng.random(N) < exec_error
                if flip.any():
                    e = e.copy()
                    e[flip] = EFFORTS[rng.integers(0, NA, int(flip.sum()))]
            E = float(e.sum())
            pay += payoff(e, e.sum(), x, pr)
            y = max(x * (1.0 - float(harvest_fraction(E, pr))), 0.0)
            xn = fast.phi(y) * (1.0 + pr.sigma * shock(rng))
            xn = min(max(xn, 0.0), K_CAP)
            e_pc, first = E / N, False
            if y <= pr.A or xn <= pr.A:
                dead[k] = True
                x = 0.0
                break
            x = xn
        rent[k] = pay.mean() / T
        xend[k] = x
        srt = np.sort(pay)
        n = len(srt)
        gini[k] = (2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum()) \
            if srt.sum() > 0 else 0.0
    return dict(rent_pc=float(rent.mean()), collapse=float(dead.mean()),
                x_end=float(xend.mean()), gini=float(np.nanmean(gini)))


# ---- symbolic distillation ----------------------------------------------- #
def distil(Q, pr: Params):
    """
    Read the learned joint policy out as (a) an effort-vs-stock table obtained
    by taking the modal greedy action across agents and monitoring bins, and
    (b) the implied escapement  y(x) = x (1 - q N e(x)).
    """
    pol = Q.argmax(axis=2)                       # (N, N_STATE)
    eff_tab = np.zeros((N_SBIN, N_EBIN))
    for b in range(N_SBIN):
        for m in range(N_EBIN):
            s = b * N_EBIN + m
            cnt = np.bincount(pol[:, s], minlength=NA)
            eff_tab[b, m] = EFFORTS[int(cnt.argmax())]
    eff_x = np.array([EFFORTS[int(np.bincount(pol[:, b * N_EBIN:(b + 1) * N_EBIN]
                                              .ravel(), minlength=NA).argmax())]
                      for b in range(N_SBIN)])
    centres = np.r_[SBIN_EDGES[0] / 2,
                    (SBIN_EDGES[:-1] + SBIN_EDGES[1:]) / 2,
                    (SBIN_EDGES[-1] + 1.0) / 2]
    y_impl = centres * (1.0 - pr.q * pr.N * eff_x)
    # sanction gradient: extra effort when others were observed harvesting hard
    sanction = eff_tab[:, -1] - eff_tab[:, 0]
    return dict(eff_tab=eff_tab, eff_x=eff_x, centres=centres,
                y_implied=y_impl, sanction=sanction)


def fit_escapement(eff_x, centres, pr: Params):
    """
    Best constant-escapement rule approximating a distilled effort profile.
    Returns (y_target, R2) where R2 is computed on effort.
    """
    cand = np.linspace(0.02, 0.98, 481)
    best, bt = -np.inf, np.nan
    ybar = eff_x.mean()
    ss_tot = float(((eff_x - ybar) ** 2).sum())
    for yt in cand:
        e_pred = np.clip((1.0 / (pr.N * pr.q)) * (1.0 - yt / np.maximum(centres, 1e-9)),
                         0.0, pr.ebar)
        e_pred = EFFORTS[np.abs(EFFORTS[None, :] - e_pred[:, None]).argmin(axis=1)]
        ss_res = float(((eff_x - e_pred) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if r2 > best:
            best, bt = r2, yt
    return float(bt), float(best)
