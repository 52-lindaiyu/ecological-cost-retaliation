"""
Evolutionary validation: pairwise-comparison imitation dynamics over a pool of
interpretable strategies in the stochastic commons, tracking frequencies,
collapse rate and welfare.  The point of this module is NOT to find a dominant
strategy but to measure basins of attraction and path dependence.
"""
from __future__ import annotations

import numpy as np

from .core import Params, Phi, K_CAP, payoff, harvest_fraction, shock


def group_episode(rules, pr: Params, rng, T=80, x0=0.9):
    """
    One group of len(rules) agents, each an independent Rule instance with its
    own mode.  Returns (per-period payoffs, collapsed, final stock).
    """
    n = len(rules)
    x = float(x0)
    mode = [r.coop_mode for r in rules]
    pay = np.zeros(n)
    one = np.array([0.0])
    dead = False
    steps = 0
    for t in range(T):
        one[0] = x
        e = np.array([float(np.asarray(rules[i].eff(one, mode[i]))[0])
                      for i in range(n)])
        E = float(e.sum())
        y = max(x * (1.0 - float(harvest_fraction(E, pr))), 0.0)
        pay += payoff(e, e.sum(), x, pr)
        xn = float(Phi(y, pr)) * (1.0 + pr.sigma * shock(rng))
        xn = float(np.clip(xn, 0.0, K_CAP))
        mode = [int(np.asarray(rules[i].nxt(mode[i], one, np.array([y]),
                                            np.array([E])))[0]) for i in range(n)]
        steps = t + 1
        if y <= pr.A or xn <= pr.A:
            dead = True
            x = 0.0
            break
        x = xn
    return pay / T, dead, x


def _type_of(counts, k):
    """Type of the k-th individual when individuals are ordered by type."""
    return int(np.searchsorted(np.cumsum(counts), k, side="right"))


def evolve(factories, names, pr: Params, pop=200, gens=300, T=80,
           beta=30.0, seed=0, mu_mut=0.001, init=None):
    """
    factories : list of callables () -> Rule  (fresh instance per episode)
    Returns dict of trajectories.  `collapse[g]` is the fraction of the
    groups formed in generation g whose freshly started T-period game
    collapsed; every game starts from the default stock, so this is a
    per-generation game-collapse rate, not a persistence process across
    generations.
    """
    rng = np.random.default_rng(seed)
    S = len(factories)
    if init is None:
        counts = np.full(S, pop // S)
        counts[0] += pop - counts.sum()
    else:
        counts = np.array(init, dtype=int)
    n_group = pr.N
    freq_hist = np.zeros((gens, S))
    col_hist = np.zeros(gens)
    wel_hist = np.zeros(gens)
    for g in range(gens):
        types = np.repeat(np.arange(S), counts)
        rng.shuffle(types)
        tot = np.zeros(S)
        num = np.zeros(S)
        ncol = ngrp = 0
        wel = []
        for k in range(0, pop - n_group + 1, n_group):
            grp = types[k:k + n_group]
            inst = [factories[t]() for t in grp]
            pay, dead, xend = group_episode(inst, pr, rng, T=T)
            np.add.at(tot, grp, pay)
            np.add.at(num, grp, 1.0)
            wel.append(float(pay.mean()))
            ncol += int(dead)
            ngrp += 1
        fit = np.divide(tot, num, out=np.zeros_like(tot), where=num > 0)
        freq_hist[g] = counts / pop
        col_hist[g] = ncol / max(ngrp, 1)
        wel_hist[g] = float(np.mean(wel))
        # pairwise-comparison imitation at the INDIVIDUAL level: a focal
        # individual and a role model are drawn uniformly from the population
        # (so each type is sampled in proportion to its frequency), and the
        # focal adopts the model's type with the Fermi probability.  Updates
        # within a generation are sequential (asynchronous), pop // 2 of them.
        for _ in range(pop // 2):
            a, b = rng.choice(pop, 2, replace=False)
            i, j = _type_of(counts, a), _type_of(counts, b)
            if i == j:
                continue
            if rng.random() < 1.0 / (1.0 + np.exp(-beta * (fit[j] - fit[i]))):
                counts[i] -= 1
                counts[j] += 1
        if mu_mut > 0:                      # rare exploration keeps all types alive
            for _ in range(int(pop * mu_mut) + 1):
                if rng.random() < mu_mut * pop:
                    i = rng.integers(S)
                    j = rng.integers(S)
                    if counts[i] > 0:
                        counts[i] -= 1
                        counts[j] += 1
    return dict(freq=freq_hist, collapse=col_hist, welfare=wel_hist,
                names=list(names), final=freq_hist[-1])
