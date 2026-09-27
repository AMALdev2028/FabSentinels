"""Yield calculator: die -> wafer -> lot -> monthly good dies, with loss breakdown."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .audit import AuditTrail
from .physics import (ALPHA_RANDOM, ALPHA_SYSTEMATIC, MM2_PER_CM2, count_dies, die_yield)
from .process import FlowTiming, ProcessNode, defect_density_for

# Technology-type modifiers applied on top of the node (logic = reference).
#   d_mult     - multiplier on random defect density that actually kills a die
#   param_mult - multiplier on parametric yield
#   sys_mult   - multiplier on the systematic share of D
TECH_MODIFIERS: dict[str, dict[str, float | str]] = {
    "logic":  {"d_mult": 1.0, "param_mult": 1.0, "sys_mult": 1.0,
               "why": "reference case - node D0 figures are quoted for logic test chips"},
    "memory": {"d_mult": 0.6, "param_mult": 1.0, "sys_mult": 0.8,
               "why": "row/column redundancy repairs a large share of random defects; regular arrays have fewer litho hotspots"},
    "analog": {"d_mult": 1.0, "param_mult": 0.97, "sys_mult": 1.1,
               "why": "matching/offset specs fail parametrically; mixed layouts add systematic hotspots"},
    "power":  {"d_mult": 0.8, "param_mult": 0.98, "sys_mult": 0.9,
               "why": "relaxed pitches on power devices lower killer-defect sensitivity; breakdown/Rds(on) tails cost parametric yield"},
}


@dataclass(frozen=True)
class DieSpec:
    width_mm: float
    height_mm: float
    wafer_diameter_mm: float
    technology: str
    model: str
    edge_exclusion_mm: float
    scribe_um: float

    @property
    def area_mm2(self) -> float:
        return self.width_mm * self.height_mm


@dataclass(frozen=True)
class YieldResult:
    d_total: float
    d_random: float
    d_systematic: float
    d_contributions: dict[str, float]
    y_random: float
    y_systematic: float
    y_parametric: float
    y_edge: float
    y_line: float
    die_yield: float           # random x systematic x parametric
    overall_yield: float       # good dies / complete dies on the wafer
    dies_on_wafer: int
    dies_usable: int
    good_dies_per_wafer: float  # per wafer START (includes line scrap)
    good_dies_per_lot: float
    loss_breakdown: dict[str, dict[str, float]]


def compute_yield(node: ProcessNode, die: DieSpec, timing: FlowTiming, k: dict[str, float],
                  *, d_scale: float = 1.0, audit: AuditTrail | None = None) -> YieldResult:
    a = audit or AuditTrail()
    tech = TECH_MODIFIERS[die.technology]
    area_cm2 = die.area_mm2 / MM2_PER_CM2

    d_raw, contrib = defect_density_for(node, timing, k)
    a.record("defect_density", "D(t) = D0_base + sum_i k_i * t_i^alpha_i",
             {"contributions_per_cm2": {c: round(v, 5) for c, v in contrib.items()},
              "hours_by_category": {c: round(v, 2) for c, v in timing.hours_by_category.items()}},
             round(d_raw, 5))
    d_total = d_raw * d_scale * float(tech["d_mult"])
    sys_frac = min(1.0, node.systematic_fraction * float(tech["sys_mult"]))
    d_sys = d_total * sys_frac
    d_rand = d_total - d_sys
    a.record("defect_split", "D_sys = D * f_sys * sys_mult; D_rand = D - D_sys",
             {"D_raw": round(d_raw, 5), "d_scale": d_scale, "tech_d_mult": tech["d_mult"], "f_sys": round(sys_frac, 4)},
             {"D_total": round(d_total, 5), "D_random": round(d_rand, 5), "D_systematic": round(d_sys, 5)})

    alpha_r = ALPHA_RANDOM if die.model == "negative_binomial" else float("nan")
    alpha_s = ALPHA_SYSTEMATIC if die.model == "negative_binomial" else float("nan")
    y_rand = die_yield(die.model, d_rand, area_cm2, alpha_r)
    y_sys = die_yield(die.model, d_sys, area_cm2, alpha_s)
    a.record("random_defect_yield", f"{die.model}(D_random, A)" + (", alpha=0.5" if die.model == "negative_binomial" else ""),
             {"D_random": round(d_rand, 5), "A_cm2": area_cm2}, round(y_rand, 6))
    a.record("systematic_defect_yield", f"{die.model}(D_systematic, A)" + (", alpha=2.0" if die.model == "negative_binomial" else ""),
             {"D_systematic": round(d_sys, 5), "A_cm2": area_cm2}, round(y_sys, 6))

    y_param = min(1.0, node.parametric_yield * float(tech["param_mult"]))
    a.record("parametric_yield", "Y_param = node_param * tech_param_mult",
             {"node": node.parametric_yield, "tech_mult": tech["param_mult"]}, round(y_param, 6))

    on_wafer, usable = count_dies(die.wafer_diameter_mm, die.width_mm, die.height_mm,
                                  die.edge_exclusion_mm, die.scribe_um)
    y_edge = usable / on_wafer if on_wafer else 0.0
    a.record("die_count", "exact grid layout, best of 4 offsets",
             {"wafer_mm": die.wafer_diameter_mm, "die_mm": [die.width_mm, die.height_mm],
              "edge_exclusion_mm": die.edge_exclusion_mm, "scribe_um": die.scribe_um},
             {"dies_on_wafer": on_wafer, "dies_usable": usable, "Y_edge": round(y_edge, 6)})

    y_die = y_rand * y_sys * y_param
    overall = y_die * y_edge * timing.line_yield
    good_wafer = on_wafer * overall
    a.record("wafer_yield", "good/wafer start = dies_on_wafer * Y_edge * Y_line * Y_rand * Y_sys * Y_param",
             {"dies_on_wafer": on_wafer, "Y_edge": round(y_edge, 6), "Y_line": round(timing.line_yield, 6),
              "Y_die": round(y_die, 6)},
             {"overall_yield": round(overall, 6), "good_dies_per_wafer_start": round(good_wafer, 3)})
    good_lot = good_wafer * node.lot_size
    a.record("lot_yield", "good/lot = good/wafer * lot_size", {"lot_size": node.lot_size}, round(good_lot, 2))

    factors = {"random_defects": y_rand, "systematic_defects": y_sys, "parametric": y_param,
               "wafer_edge": y_edge, "line_scrap": timing.line_yield}
    return YieldResult(d_total, d_rand, d_sys, contrib, y_rand, y_sys, y_param, y_edge, timing.line_yield,
                       y_die, overall, on_wafer, usable, good_wafer, good_lot, loss_breakdown(factors))


def loss_breakdown(factors: dict[str, float]) -> dict[str, dict[str, float]]:
    """Split total yield loss across mechanisms.

    Yields multiply, so losses do not simply add. Each mechanism gets a share of
    -ln(Y_total) equal to -ln(Y_i) / -ln(Y_total); `loss_points` is that share of the
    total percentage-point loss, so the points add up exactly to (1 - Y_total) * 100.
    """
    total = math.prod(factors.values())
    total_loss_pts = (1.0 - total) * 100.0
    zeros = [k for k, v in factors.items() if v <= 0]
    logs = {k: -math.log(v) for k, v in factors.items() if v > 0}
    s = sum(logs.values())
    out: dict[str, dict[str, float]] = {}
    for k, v in factors.items():
        if zeros:  # a zero factor owns the whole loss
            share = 1.0 / len(zeros) if k in zeros else 0.0
        else:
            share = logs[k] / s if s > 0 else 0.0
        out[k] = {"yield": v, "standalone_loss_pct": (1.0 - v) * 100.0, "loss_points": share * total_loss_pts}
    return out
