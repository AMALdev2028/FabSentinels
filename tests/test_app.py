"""Headless tests for the Streamlit app: router, landing page interactions, other pages.

Run from the repo root:  pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "views"))
ENTRY = str(ROOT / "src" / "dashboard.py")


@pytest.fixture
def app():
    return AppTest.from_file(ENTRY, default_timeout=120).run()


def test_home_page_renders_without_errors(app):
    assert not app.exception
    assert app.session_state["dark"] is False
    assert any(b.label == "Scan another lot" for b in app.button)


def test_dark_mode_toggle(app):
    app.button(key="darkbtn").click().run()
    assert app.session_state["dark"] is True
    app.button(key="darkbtn").click().run()
    assert app.session_state["dark"] is False


def test_scan_button_changes_wafer(app):
    before = app.session_state["seed"]
    next(b for b in app.button if b.label == "Scan another lot").click().run()
    assert app.session_state["seed"] > before


def test_live_scan_can_be_paused(app):
    app.toggle(key="live").set_value(False).run()
    assert app.session_state["live"] is False and not app.exception


@pytest.mark.parametrize("page", ["views/overview.py", "views/yield_planner.py"])
def test_other_pages_render(app, page):
    app.switch_page(page).run()
    assert not app.exception


def test_wafer_is_deterministic_and_stats_fallback(tmp_path, monkeypatch):
    import landing
    assert landing.make_wafer(3) == landing.make_wafer(3) != landing.make_wafer(4)
    assert len(landing.make_wafer(3)) == landing.CONFIG["wafer_grid"] ** 2
    stats, live = landing.load_stats()
    assert live and stats == {"runs": 1567, "sensors": 590, "failures": 104}
    landing.load_stats.clear()
    monkeypatch.setattr(landing, "ROOT", tmp_path)          # no artifacts -> mock data, no crash
    assert landing.load_stats() == (landing.MOCK_STATS, False)
    landing.load_stats.clear()
