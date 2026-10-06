"""
Core model for "Partners, rivals, and stewards".

N-player stochastic common-pool resource game:
  - stock x in [0, K], strong Allee regeneration (A = theta*K is a repelling
    fixed point of the no-harvest flow, 0 and K are attracting)
  - within a period: simultaneous efforts e_i in [0, ebar] -> pulse harvest
    h_i = q e_i x  -> escapement y = x (1 - q E), E = sum_i e_i
  - regeneration: x' = Phi(y), where Phi is the time-1 flow map of
        dz/dt = r z (1 - z/K)(z/A - 1)
    (monotone increasing, Phi(A)=A, Phi(K)=K, Phi(y)<y for y<A)
  - collapse is absorbing: once y < A the stock is doomed under ANY subsequent
    behaviour (strong Allee => no positive growth below A), payoffs 0 forever.
  - payoff: u_i = p q e_i x - c e_i          (Gordon-Schaefer rent)

Everything in this module is deterministic theory unless a rng is passed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

K_CAP = 1.0


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Params:
    r: float = 0.60         # intrinsic regeneration rate
    theta: float = 0.15     # fragility A/K  (= config.BASE)
    q: float = 0.20         # catchability   (= config.BASE)
    p: float = 1.00         # price
    c: float = 0.02         # unit effort cost (= config.BASE)
    N: int = 5              # group size
    ebar: float = 1.00      # individual effort cap
    sigma: float = 0.02     # multiplicative regeneration noise (simulation only)
    eps_margin: float = 1e-3   # safety margin used by adaptive-safe punishment

    @property
    def A(self) -> float:
        return self.theta * K_CAP

    @property
    def x0(self) -> float:
        """Gordon bionomic equilibrium / zero-rent stock c/(p q)."""
        return self.c / (self.p * self.q)


# --------------------------------------------------------------------------- #
# regeneration map Phi (time-1 flow map of the strong-Allee ODE)
# --------------------------------------------------------------------------- #
_PHI_CACHE: dict = {}


def phi_table(r: float, theta: float, ngrid: int = 6001, nstep: int = 600):
    key = (round(r, 8), round(theta, 8), ngrid, nstep)
    if key in _PHI_CACHE:
        return _PHI_CACHE[key]
    A = theta * K_CAP
    y = np.linspace(0.0, 1.30 * K_CAP, ngrid)
    z = y.copy()
    dt = 1.0 / nstep

    def f(u):
        return r * u * (1.0 - u / K_CAP) * (u / A - 1.0)

    for _ in range(nstep):
        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        z = z + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        np.clip(z, 0.0, 1.30 * K_CAP, out=z)
    z[0] = 0.0
    z = np.maximum.accumulate(z)           # enforce monotonicity numerically
    _PHI_CACHE[key] = (y, z)
    return y, z


def Phi(yv, pr: Params):
    y, z = phi_table(pr.r, pr.theta)
    return np.interp(yv, y, z)


# --------------------------------------------------------------------------- #
# basic identities
# --------------------------------------------------------------------------- #
def escapement(x, E, pr: Params):
    """Post-harvest stock given pre-harvest stock x and total effort E."""
    return x * (1.0 - pr.q * E)


def E_safe(x, pr: Params, margin: float = 0.0):
    """Largest total effort whose escapement stays above A (+margin)."""
    return np.maximum(0.0, (1.0 / pr.q) * (1.0 - (pr.A + margin) / np.maximum(x, 1e-12)))


def Psi(x, E, pr: Params):
    """One-period stock map under constant total effort E."""
    return Phi(escapement(x, E, pr), pr)


# --------------------------------------------------------------------------- #
# critical effort: saddle-node of the constant-effort map Psi_E
# --------------------------------------------------------------------------- #
def upper_fixed_point(E: float, pr: Params, ngrid: int = 4001):
    """Upper (stable) fixed point of Psi_E, or None if none exists."""
    if pr.q * E >= 1.0:
        return None
    x = np.linspace(pr.A, K_CAP, ngrid)
    d = Psi(x, E, pr) - x
    if d.max() < 0.0:
        return None
    idx = np.where(d >= 0.0)[0]
    j = int(idx[-1])
    if j == ngrid - 1:
        return float(x[-1])
    lo, hi = x[j], x[j + 1]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Psi(mid, E, pr) - mid >= 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def lower_fixed_point(E: float, pr: Params, ngrid: int = 4001):
    """
    Lower (unstable) fixed point x_sharp of Psi_E: the tipping stock under a
    constant total effort E.  Returns None if Psi_E has no positive fixed point.
    """
    if pr.q * E >= 1.0:
        return None
    x = np.linspace(pr.A, K_CAP, ngrid)
    d = Psi(x, E, pr) - x
    if d.max() < 0.0:
        return None
    idx = np.where(d >= 0.0)[0]
    j = int(idx[0])
    if j == 0:
        return float(x[0])
    lo, hi = x[j - 1], x[j]          # d<0 at lo, d>=0 at hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Psi(mid, E, pr) - mid >= 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def critical_effort(pr: Params, tol: float = 1e-9):
    """
    E_crit: total effort at which the positive fixed points of Psi_E collide
    (saddle-node).  Returns (E_crit, x_crit).
    """
    lo, hi = 0.0, (1.0 - 1e-9) / pr.q
    if upper_fixed_point(lo, pr) is None:
        return 0.0, pr.A
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if upper_fixed_point(mid, pr) is not None:
            lo = mid
        else:
            hi = mid
    return float(lo), float(upper_fixed_point(lo, pr))


# --------------------------------------------------------------------------- #
# sustainable rent and the social optimum
# --------------------------------------------------------------------------- #
def sustainable_state(yv, pr: Params):
    """
    For a steady state with escapement y: pre-harvest stock x = Phi(y),
    harvest H = x - y, total effort E = H/(q x), total rent = p H - c E.
    Returns (x, H, E, rent_total).
    """
    yv = np.atleast_1d(np.asarray(yv, dtype=float))
    x = Phi(yv, pr)
    H = x - yv
    with np.errstate(divide="ignore", invalid="ignore"):
        E = H / (pr.q * x)
    rent = pr.p * H - pr.c * E
    return x, H, E, rent


def optimal_escapement(pr: Params, ngrid: int = 20001):
    """
    Social optimum over sustainable steady states subject to E <= N*ebar.
    """
    y = np.linspace(pr.A + 1e-9, K_CAP - 1e-9, ngrid)
    x, H, E, rent = sustainable_state(y, pr)
    ok = (E <= pr.N * pr.ebar + 1e-12) & (H >= 0.0)
    rent = np.where(ok, rent, -np.inf)
    j = int(np.argmax(rent))
    return dict(y=float(y[j]), x=float(x[j]), H=float(H[j]), E=float(E[j]),
                rent_total=float(rent[j]), rent_pc=float(rent[j] / pr.N))


# --------------------------------------------------------------------------- #
# the two theory bounds
# --------------------------------------------------------------------------- #
def larder_rent(pr: Params):
    """
    Proposition 1 bound: rent a single max-effort defector is GUARANTEED to
    collect in any viable trajectory, because viability forces x_t > A.
    """
    return max(0.0, pr.ebar * (pr.p * pr.q * pr.A - pr.c))


def x_sharp(pr: Params):
    """
    Unilateral tipping stock: the lower (unstable) fixed point of the map
    Psi_ebar induced by ONE agent harvesting at the effort cap while all others
    abstain.  From any stock at or below x_sharp a single maximal defector
    drives the system to collapse no matter what the others do, because the
    others can at best abstain.  None  <=>  ebar >= E_crit (Regime I).
    """
    return lower_fixed_point(pr.ebar, pr)


def viability_floor(pr: Params):
    """
    Smallest stock compatible with invasion-proof viability.  Equals x_sharp.
    Returns (x_floor, exists).
    """
    xs = x_sharp(pr)
    return (xs, xs is not None)


def min_defector_rent(pr: Params):
    """
    pi_sharp: rent a single maximal defector is guaranteed along ANY
    invasion-proof viable path, since such a path must satisfy x_t > x_sharp.
    """
    xs = x_sharp(pr)
    if xs is None:
        return np.inf, None
    return max(0.0, pr.ebar * (pr.p * pr.q * xs - pr.c)), xs


def critical_group_size(pr: Params):
    """
    N* = R*_max / pi_sharp : the largest group for which unexploitable
    stewardship is not ruled out by Proposition 1.  R*_max is the maximal
    sustainable total rent (computed without the effort-cap restriction that
    depends on N, i.e. using E <= N*ebar of the given pr).
    """
    pi_sharp, xs = min_defector_rent(pr)
    opt = optimal_escapement(pr)
    if xs is None:                      # Regime I: nothing is invasion-proof viable
        return np.nan, opt["rent_total"], np.inf
    if pi_sharp <= 0:                   # defector rent at the floor is non-positive
        return np.inf, opt["rent_total"], pi_sharp
    return opt["rent_total"] / pi_sharp, opt["rent_total"], pi_sharp


# --------------------------------------------------------------------------- #
# adaptive-safe punishment path (viability check of Theorem 1)
# --------------------------------------------------------------------------- #
def safe_punish_effort(x, pr: Params):
    """Per-resident punishment effort that stays viable even if the defector
    harvests at ebar: (E_safe(x) - ebar)/(N-1), clipped to [0, ebar]."""
    Es = E_safe(x, pr, margin=pr.eps_margin)
    return float(np.clip((Es - pr.ebar) / (pr.N - 1), 0.0, pr.ebar))


def punishment_path(x_start: float, pr: Params, T: int = 3000):
    """
    Deterministic path with 1 defector at ebar and N-1 residents playing the
    adaptive-safe punishment.  Returns (viable, mean_defector_rent, x_limit).
    """
    x = float(x_start)
    rents = []
    for _ in range(T):
        ep = safe_punish_effort(x, pr)
        E = (pr.N - 1) * ep + pr.ebar
        y = escapement(x, E, pr)
        if y <= pr.A:
            return False, float(np.mean(rents) if rents else 0.0), x
        rents.append(pr.ebar * (pr.p * pr.q * x - pr.c))
        x = float(Phi(y, pr))
    tail = max(1, int(0.2 * T))
    return True, float(np.mean(rents[-tail:])), x


# --------------------------------------------------------------------------- #
# Theorem 1: feasibility program for a steward strategy
# --------------------------------------------------------------------------- #
def phi_inv(xv, pr: Params):
    """Inverse of the regeneration map (Phi is monotone increasing)."""
    y, z = phi_table(pr.r, pr.theta)
    return np.interp(xv, z, y)


def E_hold(xv, pr: Params):
    """Total effort needed to hold the stock stationary at level x."""
    xv = np.asarray(xv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / pr.q) * (1.0 - phi_inv(xv, pr) / np.maximum(xv, 1e-12))


def planner_optimum(pr: Params, nx: int = 401, ny: int = 401, iters: int = 20000,
                    tol: float = 1e-10, check_grid: int = 2001, margin: float = 0.005):
    """
    Long-run-average optimum of a single planner controlling total effort,
    over ALL sustainable paths (periodic orbits included), by relative value
    iteration with an aperiodicity transformation, on a grid of states
    x in (A, K] and escapements y in (A, x].  y = A itself is excluded because
    y <= A is collapse by definition.

    Returns the gain g (per period, total), the relative value h, the number
    of iterations used, the final span of the Bellman residual on the solve
    grid, and a CERTIFICATE on a finer grid: eps_cert is the largest violation
    of the Bellman inequality
        g + h(x) >= r(x, y) + h(Phi(y))
    over check_grid x check_grid feasible (x, y) pairs with h linearly
    interpolated.  Any sustainable path visiting only grid-interpolated states
    then has average total rent <= g + eps_cert; the continuum statement needs
    in addition a Lipschitz bound on h, which we report but do not prove.
    """
    # States within `margin` of the Allee threshold are excluded from the grid:
    # Phi has a fixed point at A, so a state at A + 1e-6 leaves only after
    # thousands of periods and its relative value diverges numerically.  Any
    # path inside (A, A + margin] earns at most margin * p per period there,
    # far below the gain, so excluding the band cannot raise the upper bound.
    eps = float(margin)
    xg = np.linspace(pr.A + eps, K_CAP, nx)
    yg = np.linspace(pr.A + eps, K_CAP, ny)

    def reward_and_feas(X, Y):
        H = X - Y
        E = H / (pr.q * np.maximum(X, 1e-12))
        feas = (H >= 0.0) & (E <= pr.N * pr.ebar + 1e-12) & (Y > pr.A)
        R = np.where(feas, H * (pr.p - pr.c / (pr.q * np.maximum(X, 1e-12))), -np.inf)
        return R, feas

    R, feas = reward_and_feas(xg[:, None], yg[None, :])
    nxt = np.clip(Phi(yg, pr), xg[0], xg[-1])
    dx = xg[1] - xg[0]
    jl = np.clip(np.floor((nxt - xg[0]) / dx).astype(int), 0, nx - 2)
    frac = np.clip((nxt - xg[jl]) / dx, 0.0, 1.0)
    h = np.zeros(nx)
    ref = nx // 2
    g = 0.0
    span = np.inf
    for it in range(iters):
        cont = h[jl] * (1.0 - frac) + h[jl + 1] * frac
        Th = (R + cont[None, :]).max(axis=1)          # Bellman operator
        Th = 0.5 * h + 0.5 * Th                         # aperiodicity transform
        diff = Th - h
        span = float(diff.max() - diff.min())
        g_new = float(diff[ref]) * 2.0                  # undo the 1/2 damping on the gain
        h = Th - Th[ref]
        g = g_new
        if span < tol:
            break
    # certificate on a finer grid
    xf = np.linspace(pr.A + eps, K_CAP, check_grid)
    yf = np.linspace(pr.A + eps, K_CAP, check_grid)   # y > A + margin as well
    Rf, ff = reward_and_feas(xf[:, None], yf[None, :])
    hx = np.interp(xf, xg, h)
    hn = np.interp(np.clip(Phi(yf, pr), xg[0], xg[-1]), xg, h)
    viol = np.where(ff, Rf + hn[None, :] - hx[:, None] - g, -np.inf)
    eps_cert = float(max(viol.max(), 0.0))
    lip = float(np.max(np.abs(np.diff(h)) / dx))
    cont = h[jl] * (1.0 - frac) + h[jl + 1] * frac
    pol = yg[(R + cont[None, :]).argmax(axis=1)]
    return dict(gain_total=float(g), gain_pc=float(g / pr.N), policy_escapement=pol,
                x_grid=xg, h=h, iters=it, span=span, eps_cert=eps_cert, lipschitz_h=lip,
                converged=bool(span < tol))


def sanction_ceiling(x_hold: float, pr: Params, ngrid: int = 4001):
    """
    Stock level the graded sanction settles at when the deviator stops
    harvesting altogether.  Residents' sanction effort is calibrated so the
    WORST case (deviator at ebar) leaves next stock at x_hold; if the deviator
    abstains, the stock rises to the fixed point of
        x  ->  Phi( Phi^{-1}(x_hold) + q * ebar * x ).
    Every stock visited during the sanction lies in [x_hold, x_1], so the
    deviator's rent during the sanction is at most ebar (p q x_1 - c).
    """
    y_req = float(phi_inv(x_hold, pr))
    xg = np.linspace(x_hold, K_CAP, ngrid)
    f = Phi(np.minimum(y_req + pr.q * pr.ebar * xg, 1.3 * K_CAP), pr) - xg
    idx = np.where(f <= 0.0)[0]
    if len(idx) == 0:
        return float(K_CAP)
    j = int(idx[0])
    if j == 0:
        return float(x_hold)
    lo, hi = xg[j - 1], xg[j]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(Phi(min(y_req + pr.q * pr.ebar * mid, 1.3 * K_CAP), pr)) - mid <= 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def punishment_value(x_hold: float, pr: Params, nx: int = 601, na: int = 21,
                     iters: int = 20000, tol: float = 1e-11, margin: float = 0.005):
    """
    pi_pun(x_hold): the largest long-run-average rent a single violator can
    extract while the N-1 residents run the graded sanction calibrated to hold
    level x_hold, where the supremum is taken over all violator strategies AND
    over all initial stocks in (A + margin, K].  Taking the supremum over
    every viable initial stock (not only x >= x_hold) is what makes the
    quantity usable in Theorem 1 after a raid that leaves the stock in
    [x_sharp, x_hold).

    Residents' total sanction effort at stock x is
        E_res(x) = clip( (1/q)(1 - Phi^{-1}(x_hold)/x) - ebar , 0 , (N-1) ebar ),
    the largest effort whose worst case leaves next stock at x_hold.  The
    violator faces the 1-D deterministic control problem
        y = x (1 - min(q (E_res(x) + e_m), 1)),  x' = Phi(y),
        reward = payoff(e_m, E_res(x) + e_m, x)
    solved by relative value iteration with an aperiodicity transformation.
    Collapsing (y <= A) is assigned a large negative reward: since the average
    criterion gives zero forever after collapse, no optimal violator collapses
    the stock when any positive-rent alternative exists.
    """
    y_req = float(phi_inv(x_hold, pr))
    xg = np.linspace(pr.A + margin, K_CAP, nx)
    ea = np.linspace(0.0, pr.ebar, na)
    with np.errstate(divide="ignore", invalid="ignore"):
        E_res = np.clip((1.0 / pr.q) * (1.0 - y_req / np.maximum(xg, 1e-12)) - pr.ebar,
                        0.0, (pr.N - 1) * pr.ebar)
    E_tot = E_res[:, None] + ea[None, :]
    y = xg[:, None] * (1.0 - harvest_fraction(E_tot, pr))
    y = np.maximum(y, 0.0)
    xn = np.clip(Phi(y, pr), xg[0], xg[-1])
    dead = (y <= pr.A)
    R = payoff(np.broadcast_to(ea[None, :], y.shape), E_tot, xg[:, None], pr)
    R = np.where(dead, -1e6, R)
    dx = xg[1] - xg[0]
    jl = np.clip(np.floor((xn - xg[0]) / dx).astype(int), 0, nx - 2)
    frac = np.clip((xn - xg[jl]) / dx, 0.0, 1.0)
    h = np.zeros(nx)
    ref = nx // 2
    g = 0.0
    for it in range(iters):
        cont = h[jl] * (1.0 - frac) + h[jl + 1] * frac
        Th = 0.5 * h + 0.5 * (R + cont).max(axis=1)
        diff = Th - h
        g = 2.0 * float(diff[ref])
        hn = Th - Th[ref]
        if float(diff.max() - diff.min()) < tol:
            h = hn
            break
        h = hn
    return float(g)


def deterrence_ceiling(pr: Params, rent_pc: float, lo: float, ngrid: int = 41):
    """
    Largest hold level x_h whose punishment value still deters, i.e.
    punishment_value(x_h) <= rent_pc.  Monotone in x_h, so bisect.
    """
    if punishment_value(lo, pr) > rent_pc:
        return -np.inf
    hi = K_CAP - 1e-3
    if punishment_value(hi, pr) <= rent_pc:
        return float(hi)
    a, b = lo, hi
    for _ in range(24):
        m = 0.5 * (a + b)
        if punishment_value(m, pr) <= rent_pc:
            a = m
        else:
            b = m
    return float(a)


def steward_window(pr: Params, rent_pc: float, exact: bool = False):
    """
    The sanction window.  A graded sanction must hold the stock at some level
    x_h that is simultaneously

        x_h >  x_sharp                      (ecological floor: survivable)
        x_h <= (rent_pc + c*ebar)/(p*q*ebar) (deterrence ceiling: unexploitable)
        E_hold(x_h) <= (N-1)*ebar           (enforceable by the residents alone)

    Returns dict with the floor, the ceiling, the enforceability-implied floor,
    and the resulting window width (<=0 means no steward strategy exists).
    """
    xs = x_sharp(pr)
    x_flat = (rent_pc + pr.c * pr.ebar) / (pr.p * pr.q * pr.ebar)
    if xs is None:
        return dict(x_sharp=None, x_flat=float(x_flat), x_enf=None,
                    hi_exact=-np.inf, width_exact=-np.inf,
                    lo=None, hi=float(x_flat), width=-np.inf)
    # enforceability: smallest x the residents alone can hold
    xg = np.linspace(pr.A + 1e-4, K_CAP - 1e-4, 4001)
    ok = E_hold(xg, pr) <= (pr.N - 1) * pr.ebar + 1e-12
    x_enf = float(xg[ok][0]) if ok.any() else np.inf
    lo = max(float(xs), x_enf)
    hi_exact = deterrence_ceiling(pr, rent_pc, lo) if exact else float(x_flat)
    return dict(x_sharp=float(xs), x_flat=float(x_flat), x_enf=x_enf,
                lo=lo, hi=float(x_flat), width=float(x_flat - lo),
                hi_exact=float(hi_exact), width_exact=float(hi_exact - lo))


def steward_feasibility(pr: Params, ngrid: int = 2001):
    """
    Feasibility program for a steward strategy with cooperative escapement
    target y_dag (Theorem 1):
      (i)   implementable     : e_dag <= ebar and x_dag > x_sharp
      (ii)  detection lag safe: after one period in which a single agent grabs
            at ebar while residents are still on-path, the stock is still
            above x_sharp (otherwise collapse is already unavoidable)
      (iii) unexploitable     : pi_sharp <= pi_dag
    """
    E_crit, x_crit = critical_effort(pr)
    xs = x_sharp(pr)
    regime_I = xs is None
    pi_sharp = np.inf if xs is None else max(0.0, pr.ebar * (pr.p * pr.q * xs - pr.c))
    out = dict(feasible=False, regime_I=bool(regime_I), E_crit=float(E_crit),
               x_crit=float(x_crit), x_sharp=(None if xs is None else float(xs)),
               pi_sharp=float(pi_sharp), witness=None, best_rent_pc=-np.inf,
               n_pass_i=0, n_pass_ii=0)
    if regime_I:
        return out

    ys = np.linspace(pr.A + 1e-6, K_CAP - 1e-6, ngrid)
    xsq, Hs, Es, rents = sustainable_state(ys, pr)
    e_dag = Es / pr.N
    ok_i = (e_dag <= pr.ebar + 1e-12) & (rents > 0) & (xsq > xs)
    E_lag = (pr.N - 1) * e_dag + pr.ebar
    y_lag = xsq * (1.0 - pr.q * E_lag)
    x_after = Phi(np.maximum(y_lag, 0.0), pr)
    ok_ii = ok_i & (y_lag > pr.A) & (x_after > xs)
    rent_pc = rents / pr.N
    # sanction window: need an enforceable, survivable, deterring hold level
    xg = np.linspace(pr.A + 1e-4, K_CAP - 1e-4, 4001)
    enf = E_hold(xg, pr) <= (pr.N - 1) * pr.ebar + 1e-12
    x_enf = float(xg[enf][0]) if enf.any() else np.inf
    lo = max(float(xs), x_enf)
    x_flat = (rent_pc + pr.c * pr.ebar) / (pr.p * pr.q * pr.ebar)
    ok_iii = ok_ii & (x_flat > lo)
    out["x_enf"] = x_enf
    out["n_pass_i"] = int(ok_i.sum())
    out["n_pass_ii"] = int(ok_ii.sum())
    if ok_iii.any():
        cand = np.where(ok_iii)[0]
        j = int(cand[np.argmax(rent_pc[cand])])
        out.update(feasible=True, best_rent_pc=float(rent_pc[j]),
                   witness=dict(y_dag=float(ys[j]), x_dag=float(xsq[j]),
                                e_dag=float(e_dag[j]), rent_pc=float(rent_pc[j]),
                                x_hold_lo=lo, x_hold_hi=float(x_flat[j]),
                                window=float(x_flat[j] - lo)))
    return out


def classify(pr: Params):
    """Regime I / II / III."""
    f = steward_feasibility(pr)
    if f["regime_I"]:
        f["regime"] = 1
    elif not f["feasible"]:
        f["regime"] = 2
    else:
        f["regime"] = 3
    return f


# --------------------------------------------------------------------------- #
# stochastic simulation primitive
# --------------------------------------------------------------------------- #
EFFORT_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.00])



XI_MAX = 3.0   # regeneration shocks are standard normal truncated to [-XI_MAX, XI_MAX]


def shock(rng, size=None):
    """Bounded multiplicative regeneration shock xi ~ N(0,1) truncated to
    [-XI_MAX, XI_MAX] (by clipping; 0.27% of draws are affected).  Bounded
    support is what makes 'zero collapse probability' a well-defined event and
    makes the 5-node Gauss-Hermite quadrature (nodes within +-2.857) a
    quadrature of the actual noise model rather than a truncation of it."""
    return np.clip(rng.standard_normal(size), -XI_MAX, XI_MAX)


def harvest_fraction(E, pr: Params):
    """Fraction of the current stock actually removed by total effort E.
    Nominal removal q*E is capped at 1: the group cannot harvest more than
    the stock that exists.  Below the cap this is the Gordon-Schaefer form."""
    return np.minimum(pr.q * np.asarray(E, dtype=float), 1.0)


def payoff(e_i, E, x, pr: Params):
    """Per-player rent with proportional rationing when q*E > 1.

    h_i = (e_i / E) * min(qE, 1) * x   (equals q e_i x whenever qE <= 1)
    u_i = p h_i - c e_i
    Resource conservation holds by construction: sum_i h_i <= x.
    """
    e_i = np.asarray(e_i, dtype=float); E = np.asarray(E, dtype=float)
    share = np.where(E > 0, e_i / np.where(E > 0, E, 1.0), 0.0)
    h = share * harvest_fraction(E, pr) * np.asarray(x, dtype=float)
    return pr.p * h - pr.c * e_i


def step(x: float, E: float, pr: Params, rng=None):
    """One period. Returns (x_next, escapement, collapsed)."""
    y = max(x * (1.0 - float(harvest_fraction(E, pr))), 0.0)
    xn = float(Phi(y, pr))
    if rng is not None and pr.sigma > 0:
        xn = xn * (1.0 + pr.sigma * shock(rng))
    xn = float(np.clip(xn, 0.0, K_CAP))
    if y <= pr.A or xn <= pr.A:
        return 0.0, y, True
    return xn, y, False
