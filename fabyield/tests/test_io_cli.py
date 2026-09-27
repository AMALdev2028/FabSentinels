"""Input validation, node-config validation, and the command-line interface."""
import json
import runpy
import sys

import pytest

from fabyield.cli import main
from fabyield.io import CONFIG_DIR, ConfigError, load_json, load_node, parse_run_config

from conftest import make_cfg

GOOD = {"target_good_dies_per_month": 1000, "die_area_mm2": 100, "process_node": "7nm",
        "wafer_starts_capacity_per_month": 100}


@pytest.mark.parametrize("bad, msg", [
    ({"target_good_dies_per_month": None}, "missing"),
    ({"target_good_dies_per_month": -1}, ">="),
    ({"target_good_dies_per_month": "lots"}, "finite number"),
    ({"target_good_dies_per_month": True}, "finite number"),
    ({"die_area_mm2": 0}, ">"),
    ({"wafer_diameter_mm": 200}, "wafer_diameter_mm"),
    ({"process_node": "65nm"}, "process_node"),
    ({"process_node": "custom"}, "custom_node_file"),
    ({"technology": "quantum"}, "technology"),
    ({"yield_model": "gauss"}, "yield_model"),
    ({"wafer_starts_capacity_per_month": 0}, ">"),
    ({"max_cycle_time_days": 0}, ">"),
    ({"monte_carlo": {"samples": 0}}, ">="),
])
def test_invalid_inputs_rejected(bad, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_run_config({**GOOD, **bad})


def test_run_config_must_be_object():
    with pytest.raises(ConfigError):
        parse_run_config([1, 2])


def test_area_and_aspect_ratio_input():
    c = parse_run_config({**GOOD, "die_aspect_ratio": 4.0})
    assert c.die_width_mm * c.die_height_mm == pytest.approx(100)
    assert c.die_width_mm / c.die_height_mm == pytest.approx(4.0)
    assert c.wafer_diameter_mm == 300 and c.yield_model == "negative_binomial"


def test_load_json_errors(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_json(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_json(bad)


def _node(tmp_path, **over):
    raw = json.loads((CONFIG_DIR / "node_7nm_finfet.json").read_text())
    raw.update(over)
    p = tmp_path / "node.json"
    p.write_text(json.dumps(raw))
    return p


def test_custom_node_file(tmp_path):
    from fabyield import YieldEngine
    p = _node(tmp_path, name="my custom node", d0_published=0.2)
    r = YieldEngine(make_cfg(process_node="custom", custom_node_file=str(p))).run()
    assert r["node"]["name"] == "my custom node"
    assert r["yield"]["defect_density_per_cm2"]["total"] == pytest.approx(0.2)


@pytest.mark.parametrize("over, msg", [
    ({"x_factor": 0.5}, "out-of-range"),
    ({"layers": [{"name": "x", "module": "MIDDLE", "cd_nm": 1, "overlay_nm": 1, "inspection_nm": 1, "steps": {}}]}, "module"),
    ({"layers": [{"name": "x", "module": "FEOL", "cd_nm": 1, "overlay_nm": 1, "inspection_nm": 1, "steps": {"teleport": 1}}]}, "unknown step"),
    ({"layers": [{"name": "x"}]}, "malformed"),
])
def test_bad_node_configs(tmp_path, over, msg):
    with pytest.raises(ConfigError, match=msg):
        load_node(str(_node(tmp_path, **over)))


@pytest.mark.parametrize("mutate, msg", [
    (lambda lib: lib["steps"]["etch"].update(category="magic"), "defect_physics"),
    (lambda lib: lib["steps"]["etch"].update(throughput_wph=0), "throughput_wph"),
    (lambda lib: lib["steps"]["etch"].update(scrap_rate=1.5), "scrap_rate"),
])
def test_bad_step_library(tmp_path, mutate, msg):
    lib = json.loads((CONFIG_DIR / "step_library.json").read_text())
    mutate(lib)
    p = tmp_path / "lib.json"
    p.write_text(json.dumps(lib))
    with pytest.raises(ConfigError, match=msg):
        load_node("7nm", library_path=p)


def test_cli_writes_json_and_report(tmp_path, capsys):
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({**GOOD, "monte_carlo": {"samples": 50}}))
    out = tmp_path / "out.json"
    assert main([str(cfg), "--out", str(out), "-v"]) == 0
    assert "VERDICT:" in capsys.readouterr().out
    assert json.loads(out.read_text())["feasibility"]["verdict"] in {"ACHIEVABLE", "MARGINAL", "INFEASIBLE"}


def test_cli_config_error_exit_code(tmp_path, capsys):
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({**GOOD, "process_node": "65nm"}))
    assert main([str(cfg)]) == 2
    assert "config error" in capsys.readouterr().err


def test_python_dash_m_entrypoint(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({**GOOD, "monte_carlo": {"samples": 20}}))
    monkeypatch.setattr(sys, "argv", ["fabyield", str(cfg)])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("fabyield", run_name="__main__")
    assert e.value.code == 0
