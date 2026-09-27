"""Process-flow definitions: step library, node flows, raw process time, cycle time, D(t).

A node flow is a list of layers grouped into modules (FEOL -> MOL -> BEOL -> BACKSIDE).
Each layer lists how many times each step type runs. Step times come from tool
throughput (wafers/hour) or batch duration, so fab time is built bottom-up from
equipment specs. Calendar cycle time = raw process time x X-factor (queueing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .physics import defect_density

MODULE_ORDER = ("FEOL", "MOL", "BEOL", "BACKSIDE")


@dataclass(frozen=True)
class StepType:
    name: str
    category: str
    throughput_wph: float | None      # wafers per hour (single-wafer tools)
    batch_hours: float | None         # hours per batch (furnaces); used instead of wph
    overhead_hours: float             # per-lot load/setup/reticle change
    scrap_rate: float                 # probability a wafer is scrapped at this step

    def hours_per_lot(self, lot_size: int) -> float:
        if self.batch_hours is not None:
            return self.batch_hours
        assert self.throughput_wph  # guaranteed by io validation
        return lot_size / self.throughput_wph + self.overhead_hours


@dataclass(frozen=True)
class DefectPhysics:
    category: str
    alpha: float        # acceleration exponent alpha_i in k_i * t_i^alpha_i
    share: float        # share of the time-driven defect density at nominal flow
    mechanism: str


@dataclass(frozen=True)
class Layer:
    name: str
    module: str
    cd_nm: float
    overlay_nm: float
    inspection_nm: float
    steps: dict[str, int]
    repeat: int = 1
    optional: bool = False


@dataclass(frozen=True)
class ProcessNode:
    name: str
    architecture: str
    d0_published: float
    time_defect_fraction: float
    systematic_fraction: float
    parametric_yield: float
    x_factor: float
    lot_size: int
    max_cycle_days: float
    published_cycle_days: float
    published_mask_layers: float
    steps: dict[str, StepType]
    physics: dict[str, DefectPhysics]
    layers: tuple[Layer, ...]
    sources: dict[str, str] = field(default_factory=dict)

    def active_layers(self, include_optional: bool) -> list[Layer]:
        return [l for l in self.layers if include_optional or not l.optional]


@dataclass(frozen=True)
class FlowTiming:
    hours_by_category: dict[str, float]
    hours_by_module: dict[str, float]
    hours_by_layer: dict[str, float]
    raw_process_hours: float
    cycle_days: float
    mask_layers: int
    days_per_mask_layer: float
    line_yield: float
    step_count: int


def flow_timing(node: ProcessNode, *, include_optional: bool = False, time_scale: float = 1.0,
                category_time_scale: dict[str, float] | None = None) -> FlowTiming:
    """Accumulate raw process hours per lot through the whole flow.

    time_scale multiplies every step (used for +-15% sensitivity);
    category_time_scale multiplies one category (used for improvement what-ifs).
    """
    if time_scale <= 0:
        raise ValueError("time_scale must be > 0")
    cat_scale = category_time_scale or {}
    by_cat: dict[str, float] = {}
    by_mod: dict[str, float] = {m: 0.0 for m in MODULE_ORDER}
    by_layer: dict[str, float] = {}
    masks = 0
    steps_run = 0
    survive = 1.0
    for layer in node.active_layers(include_optional):
        layer_hours = 0.0
        for step_name, count in layer.steps.items():
            st = node.steps[step_name]
            n = count * layer.repeat
            h = st.hours_per_lot(node.lot_size) * n * time_scale * cat_scale.get(st.category, 1.0)
            by_cat[st.category] = by_cat.get(st.category, 0.0) + h
            layer_hours += h
            steps_run += n
            survive *= (1.0 - st.scrap_rate) ** n
            if st.category == "litho":
                masks += n
        by_layer[layer.name] = layer_hours
        by_mod[layer.module] += layer_hours
    raw = sum(by_cat.values())
    cycle_days = raw * node.x_factor / 24.0
    return FlowTiming(by_cat, {m: h for m, h in by_mod.items() if h > 0}, by_layer, raw,
                      cycle_days, masks, cycle_days / masks if masks else 0.0, survive, steps_run)


def calibrate_k(node: ProcessNode, nominal: FlowTiming) -> dict[str, float]:
    """Solve k_i so the NOMINAL flow reproduces the published D0 exactly.

    time-driven part = D0_pub * time_defect_fraction, split by category share:
        k_i = D0_pub * f_time * share_i / T_i,nom^alpha_i
    Categories with share 0 (clean, metrology) get k = 0: zero defect contribution.
    k is calibrated once and then held fixed, so a longer or shorter flow moves D(t).
    """
    d_time = node.d0_published * node.time_defect_fraction
    shares = {c: p.share for c, p in node.physics.items() if p.share > 0 and nominal.hours_by_category.get(c, 0) > 0}
    total = sum(shares.values())
    k: dict[str, float] = {c: 0.0 for c in node.physics}
    for c, s in shares.items():
        k[c] = d_time * (s / total) / nominal.hours_by_category[c] ** node.physics[c].alpha
    return k


def defect_density_for(node: ProcessNode, timing: FlowTiming, k: dict[str, float]) -> tuple[float, dict[str, float]]:
    """D(t) = D0_base + sum_i k_i * t_i^alpha_i. Returns (D_total, per-category contribution)."""
    d0_base = node.d0_published * (1.0 - node.time_defect_fraction)
    contrib = {c: k.get(c, 0.0) * timing.hours_by_category.get(c, 0.0) ** node.physics[c].alpha
               for c in node.physics}
    total = defect_density(d0_base, [(k.get(c, 0.0), timing.hours_by_category.get(c, 0.0), node.physics[c].alpha)
                                     for c in node.physics])
    return total, {"base_D0": d0_base, **contrib}
