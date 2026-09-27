"""End-to-end engine, yield breakdown, feasibility boundaries, sensitivity, determinism."""
import json
import math

import pytest

from fabyield import YieldEngine
from fabyield.audit import AuditTrail
from fabyield.feasibility import assess
from fabyield.yieldcalc import loss_breakdown

from conftest import make_cfg


def test_example_numbers_match_hand_calculation(cfg):
    r = YieldEngine(cfg).run()
    y = r["yield"]
    # D = 0.10 (nominal), 25 % systematic, A = 8 cm^2, NB alpha 0.5 / 2.0
    y_rand = (1 + 0.075 * 8 / 0.5) ** -0.5
    y_sys = (1 + 0.025 * 8 / 2.0) ** -2
    lb = y["loss_breakdown"]
    assert lb["random_defects"]["yield"] == pytest.approx(y_rand)
    assert lb["systematic_defects"]["yield"] == pytest.approx(y_sys)
    expected = y_rand * y_sys * 0.96 * lb["wafer_edge"]["yield"] * lb["line_scrap"]["yield"]
    assert y["overall_yield"] == pytest.approx(expected)
    assert y["good_dies_per_wafer_start"] == pytest.approx(y["dies_on_wafer"] * expected)
    assert sum(v["loss_points"] for v in lb.values()) == pytest.approx((1 - expected) * 100)


def test_fully_deterministic(cfg):
    a = json.dumps(YieldEngine(cfg).run(), default=str, sort_keys=True)
    b = json.dumps(YieldEngine(cfg).run(), default=str, sort_keys=True)
    assert a == b


def test_audit_trail_records_every_stage(cfg):
    r = YieldEngine(cfg).run()
    steps = {e["step"] for e in r["audit"]["calculation_steps"]}
    for s in ("cycle_time", "k_calibration", "defect_density", "random_defect_yield", "systematic_defect_yield",
              "parametric_yield", "die_count", "wafer_yield", "lot_yield", "monthly_throughput",
              "monte_carlo", "required_yield", "verdict"):
        assert s in steps
    assert len(r["audit"]["assumptions"]) >= 10


@pytest.mark.parametrize("model", ["poisson", "murphy", "seeds", "negative_binomial"])
def test_every_yield_model_runs(model):
    r = YieldEngine(make_cfg(yield_model=model)).run()
    assert 0 < r["yield"]["overall_yield"] < 1


def test_technology_and_wafer_options():
    logic = YieldEngine(make_cfg()).run()["yield"]["overall_yield"]
    memory = YieldEngine(make_cfg(technology="memory")).run()["yield"]["overall_yield"]
    analog = YieldEngine(make_cfg(technology="analog")).run()["yield"]["overall_yield"]
    power = YieldEngine(make_cfg(technology="power")).run()["yield"]["overall_yield"]
    assert memory > logic > analog and power > logic
    big = YieldEngine(make_cfg(wafer_diameter_mm=450)).run()
    assert big["yield"]["dies_on_wafer"] > 2 * 71
    assert any("450 mm" in a["text"] for a in big["audit"]["assumptions"])


def test_3nm_with_backside_power_is_slower():
    a = YieldEngine(make_cfg(process_node="3nm")).run()
    b = YieldEngine(make_cfg(process_node="3nm", backside_power_delivery=True)).run()
    assert b["fabrication_time"]["calendar_days"] > a["fabrication_time"]["calendar_days"]
    assert b["yield"]["defect_density_per_cm2"]["total"] > a["yield"]["defect_density_per_cm2"]["total"]


# ---- feasibility boundaries
def test_zero_target_volume():
    f = YieldEngine(make_cfg(target_good_dies_per_month=0)).run()["feasibility"]
    assert f["verdict"] == "ACHIEVABLE"
    assert f["required_wafer_starts"] == 0 and f["required_defect_density"] is None


def test_verdict_boundaries_on_volume():
    base = YieldEngine(make_cfg()).run()["feasibility"]["achievable_good_dies_per_month"]
    assert YieldEngine(make_cfg(target_good_dies_per_month=base / 1.2)).run()["feasibility"]["verdict"] == "ACHIEVABLE"
    marginal = YieldEngine(make_cfg(target_good_dies_per_month=base / 1.05)).run()["feasibility"]
    assert marginal["verdict"] == "MARGINAL"
    short = YieldEngine(make_cfg(target_good_dies_per_month=base * 1.1)).run()["feasibility"]
    assert short["verdict"] == "INFEASIBLE"
    # the gap is closable by defect reduction: required D is below current D
    assert 0 < short["required_defect_density"] < 0.10


def test_target_beyond_zero_defect_ceiling():
    f = YieldEngine(make_cfg(target_good_dies_per_month=10_000_000)).run()["feasibility"]
    assert f["verdict"] == "INFEASIBLE"
    assert f["required_defect_density"] is None
    assert any("theoretical maximum" in m for m in f["limiting_factors"])


def test_cycle_time_limits():
    assert YieldEngine(make_cfg(max_cycle_time_days=90)).run()["feasibility"]["verdict"] == "MARGINAL"
    f = YieldEngine(make_cfg(max_cycle_time_days=70)).run()["feasibility"]
    assert f["verdict"] == "INFEASIBLE" and any("cycle time" in m for m in f["limiting_factors"])


def test_reticle_limit_and_no_usable_die():
    f = YieldEngine(make_cfg(die_width_mm=30, die_height_mm=30)).run()["feasibility"]
    assert f["verdict"] == "INFEASIBLE" and any("scanner field" in m for m in f["limiting_factors"])
    g = YieldEngine(make_cfg(edge_exclusion_mm=140)).run()
    assert g["yield"]["dies_usable"] == 0 and g["feasibility"]["verdict"] == "INFEASIBLE"
    assert g["yield"]["loss_breakdown"]["wafer_edge"]["loss_points"] == pytest.approx(100.0)


def test_monte_carlo_confidence_downgrades_to_marginal():
    eng = YieldEngine(make_cfg())
    timing, yr, g = eng.evaluate()
    f = assess(target=g / 1.5, capacity=2500, max_cycle_days=115, die_w=25, die_h=32, timing=timing, yr=yr,
               good_month=g, p_meet=0.5, evaluate=eng.evaluate, audit=AuditTrail())
    assert f.verdict == "MARGINAL" and any("Monte Carlo" in m for m in f.limiting_factors)


def test_loss_breakdown_edge_cases():
    lb = loss_breakdown({"a": 1.0, "b": 1.0})
    assert lb["a"]["loss_points"] == 0 and lb["b"]["loss_points"] == 0
    lb = loss_breakdown({"a": 0.0, "b": 0.5})
    assert lb["a"]["loss_points"] == pytest.approx(100) and lb["b"]["loss_points"] == 0


# ---- sensitivity + recommendations
def test_tornado_sorted_and_directional(cfg):
    tor = YieldEngine(cfg).run()["sensitivity"]["tornado"]
    swings = [r["yield_swing_points"] for r in tor]
    assert swings == sorted(swings, reverse=True)
    d = next(r for r in tor if r["parameter"] == "defect_density")
    assert d["yield_at_low"] > d["base_yield"] > d["yield_at_high"]
    assert tor[0]["parameter"] == "defect_density"


def test_monte_carlo_seeded():
    a = YieldEngine(make_cfg()).run()["sensitivity"]["monte_carlo"]
    b = YieldEngine(make_cfg(monte_carlo={"samples": 200, "seed": 8})).run()["sensitivity"]["monte_carlo"]
    assert a["yield"]["mean"] != b["yield"]["mean"]
    assert a["yield"]["p5"] < a["yield"]["p50"] < a["yield"]["p95"]
    assert 0 <= a["probability_meet_target"] <= 1


def test_recommendations_ranked_and_chiplet_only_for_large_dies():
    recs = YieldEngine(make_cfg()).run()["recommendations"]
    scores = [r["priority_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert any(r["name"] == "Split into 2 chiplets" for r in recs)
    small = YieldEngine(make_cfg(die_width_mm=10, die_height_mm=10)).run()["recommendations"]
    assert not any(r["name"] == "Split into 2 chiplets" for r in small)
    wide = YieldEngine(make_cfg(die_width_mm=32, die_height_mm=25, target_good_dies_per_month=1000)).run()
    assert wide["feasibility"]["verdict"] == "ACHIEVABLE"


def test_nothing_is_nan(cfg):
    def walk(x):
        if isinstance(x, float):
            assert not math.isnan(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    r = YieldEngine(cfg).run()
    walk({k: v for k, v in r.items() if k != "audit"})
