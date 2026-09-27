"""Process time accumulation vs published cycle times, and the D(t) calibration."""
import pytest

from fabyield.io import load_node
from fabyield.process import calibrate_k, defect_density_for, flow_timing

NODES = ["7nm", "5nm", "3nm"]


@pytest.mark.parametrize("name, masks", [("7nm", 82), ("5nm", 100), ("3nm", 108)])
def test_mask_count_matches_published(name, masks):
    assert flow_timing(load_node(name)).mask_layers == masks


@pytest.mark.parametrize("name", NODES)
def test_cycle_time_matches_published(name):
    node = load_node(name)
    t = flow_timing(node)
    assert t.cycle_days == pytest.approx(node.published_cycle_days, rel=0.05)
    # published range: 0.8 d/layer (best in class) .. 1.5 d/layer (average)
    assert 0.8 <= t.days_per_mask_layer <= 1.5


def test_published_ordering_7_5_3():
    days = [flow_timing(load_node(n)).cycle_days for n in NODES]
    assert days == sorted(days)


def test_time_scaling_is_linear():
    node = load_node("5nm")
    assert flow_timing(node, time_scale=1.15).cycle_days == pytest.approx(flow_timing(node).cycle_days * 1.15)
    with pytest.raises(ValueError):
        flow_timing(node, time_scale=0)


@pytest.mark.parametrize("name", NODES)
def test_k_calibration_reproduces_published_d0(name):
    node = load_node(name)
    t = flow_timing(node)
    d, parts = defect_density_for(node, t, calibrate_k(node, t))
    assert d == pytest.approx(node.d0_published, rel=1e-12)
    assert parts["clean"] == 0 and parts["metrology"] == 0     # zero-contribution steps


def test_longer_flow_accumulates_more_defects():
    node = load_node("5nm")
    nominal = flow_timing(node)
    k = calibrate_k(node, nominal)
    d_long, _ = defect_density_for(node, flow_timing(node, time_scale=1.15), k)
    d_thermal, _ = defect_density_for(node, flow_timing(node, category_time_scale={"thermal": 2.0}), k)
    assert d_long > node.d0_published
    assert node.d0_published < d_thermal < d_long


def test_backside_module_is_optional():
    node = load_node("3nm")
    base, bspdn = flow_timing(node), flow_timing(node, include_optional=True)
    assert bspdn.mask_layers == base.mask_layers + 6
    assert "BACKSIDE" in bspdn.hours_by_module and "BACKSIDE" not in base.hours_by_module
    assert bspdn.line_yield < base.line_yield
