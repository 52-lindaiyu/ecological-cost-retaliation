"""Baseline parameterisation used throughout the paper."""
BASE = dict(r=0.60, q=0.20, c=0.02, p=1.00, ebar=1.00, N=5,
            theta=0.15, sigma=0.02)

# secondary parameterisations for the robustness section
ALT = {
    "fragile":      dict(theta=0.25),
    "very_fragile": dict(theta=0.30),
    "robust":       dict(theta=0.08),
    "slow_growth":  dict(r=0.40),
    "fast_growth":  dict(r=0.90),
    "cheap_effort": dict(c=0.01),
    "dear_effort":  dict(c=0.04),
    "efficient_gear": dict(q=0.28),
    "large_group":  dict(N=12),
    "small_group":  dict(N=3),
    "high_noise":   dict(sigma=0.05),
    "no_noise":     dict(sigma=0.0),
}
