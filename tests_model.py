import numpy as np, importlib
from commons.core import Params, payoff, harvest_fraction, x_sharp, optimal_escapement, min_defector_rent, critical_group_size, step
from commons.config import BASE
pr = Params(**BASE); assert Params() == pr, "Params defaults != BASE"
rng = np.random.default_rng(0)
for N in [5, 8, 12]:
    p2 = Params(**{**BASE, "N": N})
    for x in [0.3, 0.6, 0.9]:
        e = rng.uniform(0, 1, N); E = e.sum()
        h = (payoff(e, E, x, p2) + p2.c * e) / p2.p
        assert h.sum() <= x + 1e-12, (N, x, h.sum())
        if p2.q * E <= 1: assert np.allclose(h, p2.q * e * x)
        e = np.ones(N); h = (payoff(e, N, x, p2) + p2.c * e) / p2.p
        assert abs(h.sum() - min(p2.q * N, 1) * x) < 1e-12
xs = x_sharp(pr); opt = optimal_escapement(pr); pis, _ = min_defector_rent(pr)
y_, R_ = opt["y"], opt["rent_total"]
print(f"x_sharp={xs:.6f} pi_sharp={pis:.6f} y**={y_:.6f} R*={R_:.6f} N*={R_/pis:.4f}")
print("conservation tests passed")
