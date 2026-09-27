"""Sensitivity engine: one-at-a-time tornado + seeded Monte Carlo."""
from __future__ import annotations

from typing import Any

import numpy as np

from .feasibility import Evaluate

# Variation ranges from the spec: defect density +-20 %, die area +-10 %, process time +-15 %.
RANGES = {"defect_density": ("d_scale", 0.20), "die_area": ("area_scale", 0.10), "process_time": ("time_scale", 0.15)}


def tornado(evaluate: Evaluate) -> list[dict[str, Any]]:
    """Swing each parameter to its low/high bound alone; sort by yield swing (tornado order)."""
    _, base, base_g = evaluate()
    rows = []
    for name, (kw, frac) in RANGES.items():
        _, lo, g_lo = evaluate(**{kw: 1 - frac})
        _, hi, g_hi = evaluate(**{kw: 1 + frac})
        rows.append({"parameter": name, "variation": f"+-{frac:.0%}",
                     "yield_at_low": lo.overall_yield, "yield_at_high": hi.overall_yield,
                     "good_dies_month_at_low": g_lo, "good_dies_month_at_high": g_hi,
                     "yield_swing_points": abs(hi.overall_yield - lo.overall_yield) * 100,
                     "base_yield": base.overall_yield, "base_good_dies_month": base_g})
    return sorted(rows, key=lambda r: r["yield_swing_points"], reverse=True)


def monte_carlo(evaluate: Evaluate, *, samples: int, seed: int, target: float) -> dict[str, Any]:
    """Sample all three parameters uniformly inside their ranges with a fixed seed.

    Uniform (not normal) because the spec gives bounds, not standard deviations;
    np.random.default_rng(seed) makes the draw bit-identical on every run.
    """
    rng = np.random.default_rng(seed)
    draws = {kw: rng.uniform(1 - frac, 1 + frac, samples) for kw, frac in RANGES.values()}
    ys = np.empty(samples)
    gs = np.empty(samples)
    for i in range(samples):
        _, yr, g = evaluate(**{kw: float(v[i]) for kw, v in draws.items()})
        ys[i], gs[i] = yr.overall_yield, g
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return {
        "samples": samples, "seed": seed, "distribution": "uniform within +-bounds",
        "yield": {"mean": float(ys.mean()), "std": float(ys.std()), "p5": pct(ys, 5), "p50": pct(ys, 50),
                  "p95": pct(ys, 95), "ci90": [pct(ys, 5), pct(ys, 95)]},
        "good_dies_per_month": {"mean": float(gs.mean()), "p5": pct(gs, 5), "p50": pct(gs, 50), "p95": pct(gs, 95)},
        "probability_meet_target": float((gs >= target).mean()) if target > 0 else 1.0,
    }
