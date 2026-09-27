# FabYield — deterministic yield, cycle-time & feasibility engine

A companion to FabSentinel. FabSentinel **learns** yield risk from sensor data
(XGBoost + SHAP). FabYield **computes** yield from first principles: defect density,
die area, the process flow and fabrication time. It tells you whether a production
target is achievable.

Same inputs → same outputs, every time. Every number is traceable through a JSON
audit trail.

## Quick start

```bash
pip install numpy                      # only runtime dependency
python -m fabyield examples/n5_800mm2_50k.json --out result.json
pip install pytest pytest-cov && pytest --cov=fabyield   # 75 tests, 99 % coverage
```

## Input (JSON)

```json
{
  "target_good_dies_per_month": 50000,
  "die_width_mm": 25.0, "die_height_mm": 32.0,     // or "die_area_mm2" (+ optional "die_aspect_ratio")
  "wafer_diameter_mm": 300,                         // 300 | 450
  "process_node": "5nm",                            // 7nm | 5nm | 3nm | custom (+ "custom_node_file")
  "technology": "logic",                            // logic | memory | analog | power
  "yield_model": "negative_binomial",               // poisson | murphy | seeds | negative_binomial
  "wafer_starts_capacity_per_month": 2500,
  "max_cycle_time_days": null,                      // null = node default
  "edge_exclusion_mm": 3.0, "scribe_um": 80,
  "backside_power_delivery": false,                 // 3nm only: adds the BSPDN module
  "monte_carlo": {"samples": 5000, "seed": 42}
}
```
(Comments are shown for explanation only; real JSON cannot contain them.)

## What you get

Verdict (ACHIEVABLE / MARGINAL / INFEASIBLE) with reasons · overall yield broken
down by random defects, systematic defects, parametric, wafer edge and line scrap ·
defect density by process step · calendar days with a module/layer breakdown ·
required vs achievable good dies/month · required wafer starts and required defect
density · tornado + Monte Carlo · ranked improvements · full audit trail.
See `examples/n5_800mm2_50k_report.txt`.

## Layout

```
fabyield/
  physics.py      yield equations, D(t), exact die-grid counting
  process.py      process-flow types, raw process time, cycle time, k_i calibration
  yieldcalc.py    die -> wafer -> lot yield chain, loss breakdown, technology modifiers
  feasibility.py  verdict rules, bottlenecks, required D0, ranked improvements
  sensitivity.py  tornado + seeded Monte Carlo
  audit.py        calculation steps + assumptions log
  io.py           input validation, node/step-library loading
  engine.py       orchestration (the reusable library entry point: YieldEngine)
  cli.py          command-line interface
configs/          step_library.json + 7nm FinFET, 5nm, 3nm GAA node flows
tests/            75 tests
docs/TECHNICAL.md model rationale, time derivation, assumptions, validation, deployment
examples/         the 50k-dies/month 5 nm run: input, text report, full JSON
```

## Library use

```python
from fabyield import YieldEngine, parse_run_config
result = YieldEngine(parse_run_config({...})).run()
print(result["feasibility"]["verdict"])
```

**Before trusting any number, read "What this model is and is not" at the top of
`docs/TECHNICAL.md`.** Public data anchors D0, mask counts and cycle times; the rest
are logged assumptions meant to be replaced with real fab data.
