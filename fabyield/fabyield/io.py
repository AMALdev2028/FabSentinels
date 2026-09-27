"""I/O: validate the run JSON, load node + step-library JSON into typed objects."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .physics import SUPPORTED_WAFER_DIAMETERS_MM, YIELD_MODELS
from .process import MODULE_ORDER, DefectPhysics, Layer, ProcessNode, StepType
from .yieldcalc import TECH_MODIFIERS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
NODE_FILES = {"7nm": "node_7nm_finfet.json", "5nm": "node_5nm.json", "3nm": "node_3nm_gaa.json"}


class ConfigError(ValueError):
    """Raised for any invalid input or node configuration."""


@dataclass(frozen=True)
class RunConfig:
    target_good_dies_per_month: float
    die_width_mm: float
    die_height_mm: float
    wafer_diameter_mm: float
    process_node: str
    technology: str
    yield_model: str
    wafer_starts_capacity_per_month: float
    max_cycle_time_days: float | None
    edge_exclusion_mm: float
    scribe_um: float
    backside_power_delivery: bool
    mc_samples: int
    mc_seed: int
    custom_node_file: str | None = None


def _num(d: dict[str, Any], key: str, *, default: Any = ..., lo: float | None = None,
         lo_inclusive: bool = True) -> float:
    if key not in d or d[key] is None:
        if default is ...:
            raise ConfigError(f"missing required field {key!r}")
        return default
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
        raise ConfigError(f"{key!r} must be a finite number (got {v!r})")
    if lo is not None and (v < lo or (not lo_inclusive and v == lo)):
        raise ConfigError(f"{key!r} must be {'>=' if lo_inclusive else '>'} {lo} (got {v})")
    return float(v)


def parse_run_config(raw: dict[str, Any]) -> RunConfig:
    if not isinstance(raw, dict):
        raise ConfigError("run configuration must be a JSON object")
    target = _num(raw, "target_good_dies_per_month", lo=0)
    if "die_width_mm" in raw or "die_height_mm" in raw:
        w = _num(raw, "die_width_mm", lo=0, lo_inclusive=False)
        h = _num(raw, "die_height_mm", lo=0, lo_inclusive=False)
    else:
        area = _num(raw, "die_area_mm2", lo=0, lo_inclusive=False)
        ar = _num(raw, "die_aspect_ratio", default=1.0, lo=0, lo_inclusive=False)  # width / height
        h = math.sqrt(area / ar)
        w = area / h
    wafer = _num(raw, "wafer_diameter_mm", default=300.0)
    if wafer not in SUPPORTED_WAFER_DIAMETERS_MM:
        raise ConfigError(f"wafer_diameter_mm must be one of {SUPPORTED_WAFER_DIAMETERS_MM} (got {wafer})")
    node = str(raw.get("process_node", ""))
    if node not in NODE_FILES and node != "custom":
        raise ConfigError(f"process_node must be one of {sorted(NODE_FILES)} or 'custom' (got {node!r})")
    custom = raw.get("custom_node_file")
    if node == "custom" and not custom:
        raise ConfigError("process_node 'custom' needs 'custom_node_file'")
    tech = str(raw.get("technology", "logic"))
    if tech not in TECH_MODIFIERS:
        raise ConfigError(f"technology must be one of {sorted(TECH_MODIFIERS)} (got {tech!r})")
    model = str(raw.get("yield_model", "negative_binomial"))
    if model not in YIELD_MODELS:
        raise ConfigError(f"yield_model must be one of {YIELD_MODELS} (got {model!r})")
    mc = raw.get("monte_carlo", {}) or {}
    samples = int(_num(mc, "samples", default=5000, lo=1))
    seed = int(_num(mc, "seed", default=42, lo=0))
    max_ct = raw.get("max_cycle_time_days")
    return RunConfig(
        target_good_dies_per_month=target, die_width_mm=w, die_height_mm=h, wafer_diameter_mm=wafer,
        process_node=node, technology=tech, yield_model=model,
        wafer_starts_capacity_per_month=_num(raw, "wafer_starts_capacity_per_month", lo=0, lo_inclusive=False),
        max_cycle_time_days=None if max_ct is None else _num(raw, "max_cycle_time_days", lo=0, lo_inclusive=False),
        edge_exclusion_mm=_num(raw, "edge_exclusion_mm", default=3.0, lo=0),
        scribe_um=_num(raw, "scribe_um", default=80.0, lo=0),
        backside_power_delivery=bool(raw.get("backside_power_delivery", False)),
        mc_samples=samples, mc_seed=seed, custom_node_file=custom)


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"invalid JSON in {path}: {e}") from e


def load_node(name_or_path: str, library_path: str | Path | None = None) -> ProcessNode:
    path = Path(name_or_path) if name_or_path not in NODE_FILES else CONFIG_DIR / NODE_FILES[name_or_path]
    raw = load_json(path)
    lib = load_json(library_path or CONFIG_DIR / "step_library.json")
    try:
        steps = {n: StepType(n, s["category"], s.get("throughput_wph"), s.get("batch_hours"),
                             s.get("overhead_hours", 0.0), s["scrap_rate"]) for n, s in lib["steps"].items()}
        physics = {c: DefectPhysics(c, p["alpha"], p["share"], p["mechanism"]) for c, p in lib["defect_physics"].items()}
        layers = tuple(Layer(l["name"], l["module"], l["cd_nm"], l["overlay_nm"], l["inspection_nm"],
                             dict(l["steps"]), l.get("repeat", 1), l.get("optional", False)) for l in raw["layers"])
        node = ProcessNode(
            name=raw["name"], architecture=raw["architecture"], d0_published=raw["d0_published"],
            time_defect_fraction=raw["time_defect_fraction"], systematic_fraction=raw["systematic_fraction"],
            parametric_yield=raw["parametric_yield"], x_factor=raw["x_factor"], lot_size=raw["lot_size"],
            max_cycle_days=raw["max_cycle_days"], published_cycle_days=raw["published_cycle_days"],
            published_mask_layers=raw["published_mask_layers"], steps=steps, physics=physics, layers=layers,
            sources=raw.get("sources", {}))
    except (KeyError, TypeError) as e:
        raise ConfigError(f"malformed node/library config ({path}): missing or bad field {e}") from e
    _validate_node(node)
    return node


def _validate_node(n: ProcessNode) -> None:
    for st in n.steps.values():
        if st.category not in n.physics:
            raise ConfigError(f"step {st.name!r} has category {st.category!r} with no defect_physics entry")
        if st.batch_hours is None and not (st.throughput_wph and st.throughput_wph > 0):
            raise ConfigError(f"step {st.name!r} needs throughput_wph > 0 or batch_hours")
        if not 0 <= st.scrap_rate < 1:
            raise ConfigError(f"step {st.name!r} scrap_rate must be in [0, 1)")
    for layer in n.layers:
        if layer.module not in MODULE_ORDER:
            raise ConfigError(f"layer {layer.name!r}: module must be one of {MODULE_ORDER}")
        for s in layer.steps:
            if s not in n.steps:
                raise ConfigError(f"layer {layer.name!r} uses unknown step {s!r}")
    checks = {"d0_published": n.d0_published >= 0, "time_defect_fraction": 0 <= n.time_defect_fraction <= 1,
              "systematic_fraction": 0 <= n.systematic_fraction <= 1, "parametric_yield": 0 < n.parametric_yield <= 1,
              "x_factor": n.x_factor >= 1, "lot_size": n.lot_size >= 1, "max_cycle_days": n.max_cycle_days > 0}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ConfigError(f"node {n.name!r} has out-of-range fields: {bad}")
