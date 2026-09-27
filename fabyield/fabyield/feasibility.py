"""Feasibility analysis, bottleneck identification and prioritised improvements."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from .audit import AuditTrail
from .physics import RETICLE_FIELD_MM
from .process import FlowTiming
from .yieldcalc import YieldResult

# Verdict thresholds (documented in docs/TECHNICAL.md, section "Feasibility rules")
VOLUME_MARGIN = 0.10          # need >= 10 % headroom over target to call it ACHIEVABLE
CYCLE_HARD_LIMIT = 1.25       # cycle time > 1.25 x max allowed -> INFEASIBLE (else MARGINAL)
MC_CONFIDENCE = 0.90          # P(meet target) below this -> at best MARGINAL
PACKAGING_YIELD = 0.98        # assumed assembly yield when splitting into 2 chiplets
_RANK = {"ACHIEVABLE": 0, "MARGINAL": 1, "INFEASIBLE": 2}


class Evaluate(Protocol):
    def __call__(self, **kw: Any) -> tuple[FlowTiming, YieldResult, float]: ...


@dataclass(frozen=True)
class Feasibility:
    verdict: str
    limiting_factors: list[str]
    target_good_dies_per_month: float
    achievable_good_dies_per_month: float
    gap_good_dies_per_month: float
    wafer_starts_capacity: float
    required_wafer_starts: float | None
    required_overall_yield: float
    achievable_overall_yield: float
    theoretical_max_yield: float
    cycle_days: float
    max_cycle_days: float
    probability_meet_target: float
    required_defect_density: float | None
    bottlenecks: dict[str, str]


def assess(*, target: float, capacity: float, max_cycle_days: float, die_w: float, die_h: float,
           timing: FlowTiming, yr: YieldResult, good_month: float, p_meet: float,
           evaluate: Evaluate, audit: AuditTrail) -> Feasibility:
    reasons: list[tuple[str, str]] = []

    def flag(level: str, why: str) -> None:
        reasons.append((level, why))

    theo_max = yr.y_parametric * yr.y_edge * yr.y_line
    required_yield = target / (capacity * yr.dies_on_wafer) if yr.dies_on_wafer else math.inf
    req_starts = math.ceil(target / yr.good_dies_per_wafer) if yr.good_dies_per_wafer > 0 else None
    audit.record("required_yield", "Y_req = target / (capacity * dies_on_wafer)",
                 {"target": target, "capacity": capacity, "dies_on_wafer": yr.dies_on_wafer},
                 required_yield if math.isfinite(required_yield) else "inf")
    audit.record("required_wafer_starts", "ceil(target / good_dies_per_wafer_start)",
                 {"target": target, "good_per_wafer": round(yr.good_dies_per_wafer, 3)}, req_starts)

    fits = all(a <= b for a, b in zip(sorted((die_w, die_h)), sorted(RETICLE_FIELD_MM)))
    if not fits:
        flag("INFEASIBLE", f"die {die_w:.1f} x {die_h:.1f} mm exceeds the {RETICLE_FIELD_MM[0]:.0f} x "
                           f"{RETICLE_FIELD_MM[1]:.0f} mm scanner field - cannot be printed monolithically")
    if yr.dies_usable == 0:
        flag("INFEASIBLE", "no complete die fits inside the usable wafer area")
    if target > 0:
        if required_yield > theo_max:
            flag("INFEASIBLE", f"required yield {required_yield:.1%} exceeds theoretical maximum {theo_max:.1%} "
                               "(zero-defect ceiling set by edge loss, parametric yield and line scrap)")
        ratio = good_month / target
        if ratio < 1.0:
            flag("INFEASIBLE", f"achievable {good_month:,.0f} good dies/month is {1 - ratio:.1%} short of target")
        elif ratio < 1.0 + VOLUME_MARGIN:
            flag("MARGINAL", f"only {ratio - 1:.1%} volume headroom (< {VOLUME_MARGIN:.0%} margin)")
        if ratio >= 1.0 and p_meet < MC_CONFIDENCE:
            flag("MARGINAL", f"Monte Carlo: only {p_meet:.0%} probability of meeting target under +-variation")
    if timing.cycle_days > max_cycle_days * CYCLE_HARD_LIMIT:
        flag("INFEASIBLE", f"cycle time {timing.cycle_days:.0f} d is > {CYCLE_HARD_LIMIT:.0%} of the "
                           f"{max_cycle_days:.0f} d limit")
    elif timing.cycle_days > max_cycle_days:
        flag("MARGINAL", f"cycle time {timing.cycle_days:.0f} d exceeds the {max_cycle_days:.0f} d limit")

    verdict = max((lvl for lvl, _ in reasons), key=_RANK.__getitem__, default="ACHIEVABLE")
    if not reasons:
        reasons.append(("ACHIEVABLE", "target met with margin on volume, cycle time and Monte Carlo confidence"
                        if target > 0 else "zero target volume - nothing to build"))

    req_d = _required_defect_density(target, yr, evaluate) if target > 0 else None
    worst_loss = max(yr.loss_breakdown.items(), key=lambda kv: kv[1]["loss_points"])[0]
    dens = {c: v for c, v in yr.d_contributions.items() if c != "base_D0"}
    worst_defect_step = max(dens, key=dens.__getitem__) if dens else "none"
    slowest_cat = max(timing.hours_by_category, key=timing.hours_by_category.__getitem__)
    slowest_mod = max(timing.hours_by_module, key=timing.hours_by_module.__getitem__)
    bottlenecks = {
        "yield_limiting_mechanism": worst_loss,
        "yield_limiting_process_step": worst_defect_step,
        "cycle_time_driver_step": slowest_cat,
        "cycle_time_driver_module": slowest_mod,
    }
    audit.record("verdict", "worst of all flagged conditions", {"flags": reasons}, verdict)
    return Feasibility(verdict, [why for _, why in reasons], target, good_month, good_month - target, capacity,
                       req_starts, required_yield, yr.overall_yield, theo_max, timing.cycle_days, max_cycle_days,
                       p_meet, req_d, bottlenecks)


def _required_defect_density(target: float, yr: YieldResult, evaluate: Evaluate) -> float | None:
    """Bisection on a D multiplier s in [0, 1] until good dies/month just meets target.

    Returns the D that is needed, the current D if already met, or None when even
    D = 0 would not reach target (the gap is not a defect problem).
    """
    if evaluate()[2] >= target:
        return yr.d_total
    if evaluate(d_scale=0.0)[2] < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if evaluate(d_scale=mid)[2] >= target:
            lo = mid
        else:
            hi = mid
    return lo * yr.d_total


# ------------------------------------------------------------------ improvements
@dataclass(frozen=True)
class Improvement:
    name: str
    action: str
    cost: str                 # LOW / MEDIUM / HIGH
    good_dies_gain_per_month: float
    yield_gain_points: float
    priority_score: float     # relative gain per unit of relative cost


COST_INDEX = {"LOW": 1.0, "MEDIUM": 3.0, "HIGH": 10.0}   # relative implementation cost (assumption)


def recommend(evaluate: Evaluate, yr: YieldResult, timing: FlowTiming, good_month: float,
              die_w: float, die_h: float, edge_mm: float, node_param: float, node_sys: float) -> list[Improvement]:
    dens = {c: v for c, v in yr.d_contributions.items() if c != "base_D0"}
    top_cat = max(dens, key=dens.__getitem__) if dens else next(iter(timing.hours_by_category))
    cands: list[tuple[str, str, str, dict[str, Any]]] = [
        ("Defect reduction program", "cut killer-defect density 10% (particle control, chamber seasoning, filtration)",
         "MEDIUM", {"d_scale": 0.9}),
        ("DFM / OPC hotspot fixes", "remove 25% of systematic (layout-driven) defects",
         "LOW", {"node_overrides": {"systematic_fraction": node_sys * 0.75}}),
        (f"Faster {top_cat} processing", f"cut {top_cat} time 10% (the largest time-driven defect contributor)",
         "HIGH", {"category_time_scale": {top_cat: 0.9}}),
        ("Tighter edge control", f"reduce edge exclusion {edge_mm:g} -> {max(edge_mm - 1, 0):g} mm (edge bead/bevel control)",
         "MEDIUM", {"die_overrides": {"edge_exclusion_mm": max(edge_mm - 1.0, 0.0)}}),
        ("Advanced process control", "raise parametric yield by 1 point (run-to-run control, tighter Vt/CD targeting)",
         "MEDIUM", {"node_overrides": {"parametric_yield": min(1.0, node_param + 0.01)}}),
    ]
    out: list[Improvement] = []
    base_yield = yr.overall_yield
    for name, action, cost, kw in cands:
        _, y2, g2 = evaluate(**kw)
        out.append(_improvement(name, action, cost, g2 - good_month, (y2.overall_yield - base_yield) * 100, good_month))
    if die_w * die_h > 400.0:
        # two chiplets of half the area; one product = 2 good chiplets * packaging yield
        half = {"width_mm": die_w, "height_mm": die_h / 2} if die_h >= die_w else {"width_mm": die_w / 2, "height_mm": die_h}
        _, y2, g2 = evaluate(die_overrides=half)
        products = g2 / 2 * PACKAGING_YIELD
        out.append(_improvement("Split into 2 chiplets", "re-architect as two half-area dies + advanced packaging "
                                f"(assumes {PACKAGING_YIELD:.0%} assembly yield)", "HIGH", products - good_month,
                                (y2.overall_yield - base_yield) * 100, good_month))
    return sorted(out, key=lambda i: i.priority_score, reverse=True)


def _improvement(name: str, action: str, cost: str, gain: float, ypts: float, base: float) -> Improvement:
    rel = gain / base if base > 0 else 0.0
    return Improvement(name, action, cost, gain, ypts, rel / COST_INDEX[cost])
