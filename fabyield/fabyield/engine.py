"""Engine: wires process -> yield -> feasibility -> sensitivity into one deterministic run."""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

from .audit import AuditTrail
from .feasibility import assess, recommend
from .io import RunConfig, load_node
from .process import FlowTiming, ProcessNode, calibrate_k, flow_timing
from .sensitivity import monte_carlo, tornado
from .yieldcalc import TECH_MODIFIERS, DieSpec, YieldResult, compute_yield

log = logging.getLogger("fabyield")
__version__ = "1.0.0"


class YieldEngine:
    def __init__(self, cfg: RunConfig, node: ProcessNode | None = None) -> None:
        self.cfg = cfg
        self.node = node or load_node(cfg.custom_node_file if cfg.process_node == "custom" else cfg.process_node)
        self.die = DieSpec(cfg.die_width_mm, cfg.die_height_mm, cfg.wafer_diameter_mm, cfg.technology,
                           cfg.yield_model, cfg.edge_exclusion_mm, cfg.scribe_um)
        self.nominal = flow_timing(self.node, include_optional=cfg.backside_power_delivery)
        # k_i fixed from the NOMINAL flow; every what-if reuses these coefficients.
        self.k = calibrate_k(self.node, self.nominal)

    def evaluate(self, *, d_scale: float = 1.0, area_scale: float = 1.0, time_scale: float = 1.0,
                 category_time_scale: dict[str, float] | None = None, node_overrides: dict[str, Any] | None = None,
                 die_overrides: dict[str, Any] | None = None,
                 audit: AuditTrail | None = None) -> tuple[FlowTiming, YieldResult, float]:
        """One deterministic evaluation. Returns (timing, yield result, good dies per month)."""
        node = dataclasses.replace(self.node, **node_overrides) if node_overrides else self.node
        die = dataclasses.replace(self.die, **die_overrides) if die_overrides else self.die
        if area_scale != 1.0:
            s = math.sqrt(area_scale)
            die = dataclasses.replace(die, width_mm=die.width_mm * s, height_mm=die.height_mm * s)
        timing = flow_timing(node, include_optional=self.cfg.backside_power_delivery, time_scale=time_scale,
                             category_time_scale=category_time_scale)
        yr = compute_yield(node, die, timing, self.k, d_scale=d_scale, audit=audit)
        return timing, yr, yr.good_dies_per_wafer * self.cfg.wafer_starts_capacity_per_month

    def run(self) -> dict[str, Any]:
        cfg, node = self.cfg, self.node
        audit = AuditTrail()
        self._log_assumptions(audit)
        audit.record("cycle_time", "calendar_days = sum(hours_per_lot_i) * X_factor / 24",
                     {"raw_process_hours": round(self.nominal.raw_process_hours, 2), "x_factor": node.x_factor},
                     round(self.nominal.cycle_days, 2))
        audit.record("k_calibration", "k_i = D0_pub * f_time * share_i / T_i,nom^alpha_i",
                     {"D0_pub": node.d0_published, "f_time": node.time_defect_fraction},
                     {c: float(f"{v:.6g}") for c, v in self.k.items()})
        timing, yr, good_month = self.evaluate(audit=audit)
        audit.record("monthly_throughput", "good/month = good_per_wafer_start * wafer_starts_capacity",
                     {"good_per_wafer": round(yr.good_dies_per_wafer, 3), "capacity": cfg.wafer_starts_capacity_per_month},
                     round(good_month, 1))

        mc = monte_carlo(self.evaluate, samples=cfg.mc_samples, seed=cfg.mc_seed, target=cfg.target_good_dies_per_month)
        audit.record("monte_carlo", "uniform draws: D +-20%, A +-10%, t +-15%",
                     {"samples": cfg.mc_samples, "seed": cfg.mc_seed},
                     {"yield_p5": round(mc["yield"]["p5"], 5), "yield_p95": round(mc["yield"]["p95"], 5)})
        tor = tornado(self.evaluate)
        max_ct = cfg.max_cycle_time_days or node.max_cycle_days
        feas = assess(target=cfg.target_good_dies_per_month, capacity=cfg.wafer_starts_capacity_per_month,
                      max_cycle_days=max_ct, die_w=self.die.width_mm, die_h=self.die.height_mm, timing=timing,
                      yr=yr, good_month=good_month, p_meet=mc["probability_meet_target"],
                      evaluate=self.evaluate, audit=audit)
        recs = recommend(self.evaluate, yr, timing, good_month, self.die.width_mm, self.die.height_mm,
                         self.die.edge_exclusion_mm, node.parametric_yield, node.systematic_fraction)
        log.info("verdict=%s overall_yield=%.4f good/month=%.0f", feas.verdict, yr.overall_yield, good_month)
        return {
            "engine_version": __version__,
            "inputs": dataclasses.asdict(cfg),
            "node": {"name": node.name, "architecture": node.architecture, "d0_published": node.d0_published,
                     "sources": node.sources},
            "yield": _yield_block(yr),
            "fabrication_time": _time_block(timing, node),
            "feasibility": dataclasses.asdict(feas),
            "sensitivity": {"tornado": tor, "monte_carlo": mc},
            "recommendations": [dataclasses.asdict(r) for r in recs],
            "audit": audit.to_dict(),
        }

    def _log_assumptions(self, a: AuditTrail) -> None:
        n, c = self.node, self.cfg
        a.assume(f"D0 at nominal flow = {n.d0_published}/cm^2 for {n.name}", n.sources.get("d0", "see node config"))
        a.assume(f"{n.time_defect_fraction:.0%} of D0 is time-driven; k_i calibrated to reproduce D0 at nominal flow",
                 "modelling assumption (docs/TECHNICAL.md s3)")
        a.assume(f"systematic share of D = {n.systematic_fraction:.0%}; parametric yield = {n.parametric_yield:.1%}",
                 n.sources.get("systematic_parametric", "modelling assumption"))
        a.assume(f"X-factor {n.x_factor} (cycle time / raw process time)", n.sources.get("cycle_time", "calibration"))
        a.assume(f"technology '{c.technology}': {TECH_MODIFIERS[c.technology]['why']}", "modelling assumption")
        a.assume(f"edge exclusion {c.edge_exclusion_mm} mm, scribe {c.scribe_um} um", "typical 300 mm practice")
        a.assume(f"wafer starts capacity {c.wafer_starts_capacity_per_month:,.0f}/month", "user input")
        if c.yield_model == "negative_binomial":
            a.assume("negative binomial cluster alpha = 0.5 (random) and 2.0 (systematic)", "specification")
        if c.wafer_diameter_mm == 450:
            a.assume("450 mm: per-lot tool times taken equal to 300 mm (450 mm never reached HVM)", "modelling assumption")
        for p in n.physics.values():
            a.assume(f"{p.category}: alpha={p.alpha}, share={p.share} - {p.mechanism}", "step_library.json")


def _yield_block(yr: YieldResult) -> dict[str, Any]:
    return {
        "overall_yield": yr.overall_yield, "die_yield": yr.die_yield,
        "defect_density_per_cm2": {"total": yr.d_total, "random": yr.d_random, "systematic": yr.d_systematic,
                                   "by_source": yr.d_contributions},
        "loss_breakdown": yr.loss_breakdown,
        "dies_on_wafer": yr.dies_on_wafer, "dies_usable": yr.dies_usable,
        "good_dies_per_wafer_start": yr.good_dies_per_wafer, "good_dies_per_lot": yr.good_dies_per_lot,
    }


def _time_block(t: FlowTiming, node: ProcessNode) -> dict[str, Any]:
    return {
        "calendar_days": t.cycle_days, "raw_process_hours": t.raw_process_hours, "x_factor": node.x_factor,
        "mask_layers": t.mask_layers, "days_per_mask_layer": t.days_per_mask_layer, "process_steps": t.step_count,
        "critical_path": {
            "note": "wafers pass modules strictly in sequence, so the whole flow is the critical path; "
                    "modules/steps are ranked by their share of it",
            "modules_days": {m: h * node.x_factor / 24 for m, h in t.hours_by_module.items()},
            "top_layers_days": dict(sorted(((k, v * node.x_factor / 24) for k, v in t.hours_by_layer.items()),
                                           key=lambda kv: kv[1], reverse=True)[:5]),
            "step_categories_days": dict(sorted(((k, v * node.x_factor / 24) for k, v in t.hours_by_category.items()),
                                                key=lambda kv: kv[1], reverse=True)),
        },
        "published_reference": {"cycle_days": node.published_cycle_days, "mask_layers": node.published_mask_layers},
    }
