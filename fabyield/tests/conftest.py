import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fabyield.io import parse_run_config  # noqa: E402


def make_cfg(**over):
    base = {"target_good_dies_per_month": 50000, "die_width_mm": 25.0, "die_height_mm": 32.0,
            "wafer_diameter_mm": 300, "process_node": "5nm", "technology": "logic",
            "yield_model": "negative_binomial", "wafer_starts_capacity_per_month": 2500,
            "monte_carlo": {"samples": 200, "seed": 7}}
    base.update(over)
    return parse_run_config(base)


@pytest.fixture
def cfg():
    return make_cfg()
