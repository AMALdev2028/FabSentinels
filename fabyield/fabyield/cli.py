"""Command-line interface: python -m fabyield <config.json> [--out result.json] [-v]."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from .engine import YieldEngine
from .io import ConfigError, load_json, parse_run_config


def render_text(r: dict[str, Any]) -> str:
    y, t, f = r["yield"], r["fabrication_time"], r["feasibility"]
    i = r["inputs"]
    L = [
        f"FabYield {r['engine_version']} - {r['node']['name']} ({r['node']['architecture']}), {i['technology']}, "
        f"{i['yield_model']}",
        f"Die {i['die_width_mm']:.2f} x {i['die_height_mm']:.2f} mm ({i['die_width_mm'] * i['die_height_mm']:.0f} mm^2) "
        f"on {i['wafer_diameter_mm']:.0f} mm wafer",
        "",
        f"VERDICT: {f['verdict']}",
        *[f"  - {why}" for why in f["limiting_factors"]],
        "",
        f"Overall yield {y['overall_yield']:.2%}   (die yield {y['die_yield']:.2%})",
        f"D total {y['defect_density_per_cm2']['total']:.4f}/cm^2  (random {y['defect_density_per_cm2']['random']:.4f}, "
        f"systematic {y['defect_density_per_cm2']['systematic']:.4f})",
        "Yield loss by mechanism (points of total loss):",
        *[f"  {k:<20} Y={v['yield']:.4f}  loss {v['loss_points']:6.2f} pts" for k, v in y["loss_breakdown"].items()],
        f"Dies on wafer {y['dies_on_wafer']}, usable {y['dies_usable']}, good/wafer start "
        f"{y['good_dies_per_wafer_start']:.2f}, good/lot {y['good_dies_per_lot']:.1f}",
        "",
        f"Fabrication time {t['calendar_days']:.1f} calendar days ({t['mask_layers']} masks, "
        f"{t['days_per_mask_layer']:.2f} d/mask; published ref {t['published_reference']['cycle_days']} d)",
        "  Critical path by module: " + ", ".join(f"{m} {d:.1f} d" for m, d in t["critical_path"]["modules_days"].items()),
        "",
        f"Target {f['target_good_dies_per_month']:,.0f} vs achievable {f['achievable_good_dies_per_month']:,.0f} "
        f"good dies/month (gap {f['gap_good_dies_per_month']:+,.0f})",
        f"Required wafer starts {f['required_wafer_starts']} vs capacity {f['wafer_starts_capacity']:,.0f}; "
        f"required yield {f['required_overall_yield']:.2%} vs achievable {f['achievable_overall_yield']:.2%} "
        f"(zero-defect ceiling {f['theoretical_max_yield']:.2%})",
        f"P(meet target) from Monte Carlo: {f['probability_meet_target']:.1%}",
        "Bottlenecks: " + ", ".join(f"{k}={v}" for k, v in f["bottlenecks"].items()),
        "",
        "Sensitivity (tornado, yield swing):",
        *[f"  {row['parameter']:<15} {row['variation']:>6}  {row['yield_at_low']:.2%} .. {row['yield_at_high']:.2%}"
          f"  ({row['yield_swing_points']:.2f} pts)" for row in r["sensitivity"]["tornado"]],
        f"Monte Carlo yield 90% CI: {r['sensitivity']['monte_carlo']['yield']['p5']:.2%} .. "
        f"{r['sensitivity']['monte_carlo']['yield']['p95']:.2%}",
        "",
        "Recommended improvements (by impact / cost):",
        *[f"  {n}. [{rec['cost']}] {rec['name']}: {rec['good_dies_gain_per_month']:+,.0f} good dies/month - {rec['action']}"
          for n, rec in enumerate(r["recommendations"], 1)],
    ]
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fabyield", description="Deterministic semiconductor yield & feasibility engine")
    p.add_argument("config", help="run configuration JSON")
    p.add_argument("--out", help="write full JSON result (incl. audit trail) here")
    p.add_argument("-v", "--verbose", action="store_true", help="log every calculation step")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        result = YieldEngine(parse_run_config(load_json(a.config))).run()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(render_text(result))
    return 0
