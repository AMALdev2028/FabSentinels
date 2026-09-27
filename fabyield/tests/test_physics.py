"""Yield equations against closed-form reference values, D(t), and die-count geometry."""
import math

import pytest

from fabyield import physics as p


# ---- reference values: every number here is the textbook formula evaluated by hand
@pytest.mark.parametrize("da, poisson, murphy, seeds", [
    (0.0, 1.0, 1.0, 1.0),
    (0.5, math.exp(-0.5), ((1 - math.exp(-0.5)) / 0.5) ** 2, 1 / 1.5),
    (1.0, 0.36787944, 0.39957640, 0.5),          # DA = 1: e^-1, (1-e^-1)^2, 1/2
    (2.0, 0.13533528, 0.18691127, 1 / 3),         # ((1 - 0.135335)/2)^2 = 0.432332^2
])
def test_models_match_reference_values(da, poisson, murphy, seeds):
    assert p.poisson_yield(da, 1.0) == pytest.approx(poisson, abs=1e-8)
    assert p.murphy_yield(da, 1.0) == pytest.approx(murphy, abs=1e-8)
    assert p.seeds_yield(da, 1.0) == pytest.approx(seeds, abs=1e-8)


def test_murphy_classic_example():
    # D0 = 0.5 /cm^2 on a 1 cm^2 die (typical early-ramp mature-node figure used in
    # textbook Murphy examples): Y = ((1 - 0.606531)/0.5)^2 = 0.786939^2 = 0.619273
    assert p.murphy_yield(0.5, 1.0) == pytest.approx(0.619273, abs=1e-6)


def test_model_ordering_poisson_most_pessimistic():
    for da in (0.1, 0.5, 1, 3):
        assert p.poisson_yield(da, 1) < p.murphy_yield(da, 1) < p.seeds_yield(da, 1)


def test_negative_binomial_limits():
    assert p.negative_binomial_yield(0.7, 1.3, 1.0) == pytest.approx(p.seeds_yield(0.7, 1.3))
    assert p.negative_binomial_yield(0.7, 1.3, 1e7) == pytest.approx(p.poisson_yield(0.7, 1.3), rel=1e-5)
    # clustering (small alpha) raises yield at fixed D*A
    assert p.negative_binomial_yield(0.8, 1, 0.5) > p.negative_binomial_yield(0.8, 1, 2.0)
    assert p.negative_binomial_yield(0.6, 1, 0.5) == pytest.approx(2.2 ** -0.5)


def test_dispatch_and_errors():
    assert p.die_yield("murphy", 0.2, 2, float("nan")) == p.murphy_yield(0.2, 2)
    assert p.die_yield("poisson", 0.2, 2, 0) == p.poisson_yield(0.2, 2)
    assert p.die_yield("seeds", 0.2, 2, 0) == p.seeds_yield(0.2, 2)
    assert p.die_yield("negative_binomial", 0.2, 2, 2.0) == p.negative_binomial_yield(0.2, 2, 2.0)
    with pytest.raises(ValueError):
        p.die_yield("bose-einstein", 0.1, 1, 1)
    with pytest.raises(ValueError):
        p.poisson_yield(-0.1, 1)
    with pytest.raises(ValueError):
        p.murphy_yield(0.1, float("nan"))
    with pytest.raises(ValueError):
        p.negative_binomial_yield(0.1, 1, 0)


def test_defect_density_time_dependence():
    assert p.defect_density(0.07, []) == 0.07
    assert p.defect_density(0.07, [(0.01, 4.0, 0.5), (0.002, 10.0, 1.0)]) == pytest.approx(0.07 + 0.02 + 0.02)
    assert p.defect_density(0.07, [(0.0, 99.0, 1.0)]) == 0.07          # zero-contribution step
    with pytest.raises(ValueError):
        p.defect_density(-1, [])
    with pytest.raises(ValueError):
        p.defect_density(0.1, [(0.1, -1.0, 1.0)])


# ---- geometry
def test_die_count_close_to_textbook_formula():
    on, usable = p.count_dies(300, 10, 10, 0, 0)
    assert on == usable
    assert abs(on - p.dpw_approximation(300, 100)) / on < 0.05


def test_maximum_theoretical_die_count_tiny_die():
    on, usable = p.count_dies(300, 1, 1, 3, 0)
    ideal = math.pi * 147 ** 2
    assert 0.98 * ideal < usable < ideal


def test_450mm_scales_with_area():
    a300 = p.count_dies(300, 5, 5, 0, 0)[0]
    a450 = p.count_dies(450, 5, 5, 0, 0)[0]
    assert a450 / a300 == pytest.approx(2.25, rel=0.03)


def test_edge_exclusion_and_scribe_reduce_count():
    on, usable = p.count_dies(300, 25, 32, 3, 80)
    assert usable < on
    assert p.count_dies(300, 25, 32, 3, 500)[1] <= usable


def test_extreme_aspect_ratio_dies():
    on, usable = p.count_dies(300, 2, 200, 3, 80)     # 100:1 strip
    assert usable > 0
    assert p.count_dies(300, 1, 400, 0, 0) == (0, 0)   # longer than the wafer


def test_geometry_errors_and_degenerate_wafer():
    with pytest.raises(ValueError):
        p.count_dies(300, 0, 10, 3, 80)
    with pytest.raises(ValueError):
        p.count_dies(300, 10, 10, -1, 80)
    assert p.count_dies(300, 10, 10, 150, 0) == (0, 0)
