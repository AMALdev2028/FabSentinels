"""Physics models: die-yield equations, time-dependent defect density, wafer geometry.

Units used everywhere in this package:
  defect density D  -> defects / cm^2
  die area A        -> cm^2 inside the yield equations (mm^2 at the API edge)
  lengths           -> mm (scribe width in um)
  time              -> hours (calendar days only in the cycle-time report)
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# --- Physical / industry constants (documented in docs/TECHNICAL.md) ---------
MM2_PER_CM2 = 100.0
# Full scanner field for 0.33NA EUV and 193i DUV scanners: 26 mm x 33 mm = 858 mm^2.
# A monolithic die larger than this cannot be printed in one exposure.
RETICLE_FIELD_MM = (26.0, 33.0)
# Negative-binomial cluster parameters requested by the spec.
ALPHA_RANDOM = 0.5
ALPHA_SYSTEMATIC = 2.0
SUPPORTED_WAFER_DIAMETERS_MM = (300.0, 450.0)
# Die-map optimiser resolution: grid origin shifted in pitch/16 steps on each axis.
GRID_OFFSET_STEPS = 16

YIELD_MODELS = ("poisson", "murphy", "seeds", "negative_binomial")


def _check_da(d: float, a_cm2: float) -> float:
    if d < 0 or a_cm2 < 0 or math.isnan(d) or math.isnan(a_cm2):
        raise ValueError(f"defect density and area must be >= 0 (got D={d}, A={a_cm2})")
    return d * a_cm2


def poisson_yield(d: float, a_cm2: float) -> float:
    """Y = exp(-D*A). Uniform, uncorrelated defects (pessimistic for large dies)."""
    return math.exp(-_check_da(d, a_cm2))


def murphy_yield(d: float, a_cm2: float) -> float:
    """Murphy (1964), triangular D distribution: Y = ((1 - exp(-DA)) / DA)^2."""
    da = _check_da(d, a_cm2)
    return 1.0 if da == 0 else ((1.0 - math.exp(-da)) / da) ** 2


def seeds_yield(d: float, a_cm2: float) -> float:
    """Seeds (1967), exponential D distribution: Y = 1 / (1 + DA)."""
    return 1.0 / (1.0 + _check_da(d, a_cm2))


def negative_binomial_yield(d: float, a_cm2: float, alpha: float) -> float:
    """Stapper negative binomial: Y = (1 + DA/alpha)^-alpha.

    Small alpha = strong clustering (defects pile onto few dies, so yield is higher);
    alpha -> inf recovers Poisson; alpha = 1 equals Seeds.
    """
    if alpha <= 0:
        raise ValueError(f"cluster parameter alpha must be > 0 (got {alpha})")
    return (1.0 + _check_da(d, a_cm2) / alpha) ** (-alpha)


def die_yield(model: str, d: float, a_cm2: float, alpha: float) -> float:
    """Dispatch to one yield model. `alpha` is only used by negative_binomial."""
    if model == "poisson":
        return poisson_yield(d, a_cm2)
    if model == "murphy":
        return murphy_yield(d, a_cm2)
    if model == "seeds":
        return seeds_yield(d, a_cm2)
    if model == "negative_binomial":
        return negative_binomial_yield(d, a_cm2, alpha)
    raise ValueError(f"unknown yield model {model!r}; choose one of {YIELD_MODELS}")


def defect_density(d0: float, terms: Iterable[tuple[float, float, float]]) -> float:
    """D(t) = D0 + sum_i k_i * t_i^alpha_i, with terms given as (k_i, t_i, alpha_i)."""
    if d0 < 0:
        raise ValueError(f"D0 must be >= 0 (got {d0})")
    total = d0
    for k, t, a in terms:
        if t < 0 or k < 0:
            raise ValueError(f"k and t must be >= 0 (got k={k}, t={t})")
        total += k * t ** a
    return total


def count_dies(wafer_diameter_mm: float, die_w_mm: float, die_h_mm: float,
               edge_exclusion_mm: float, scribe_um: float) -> tuple[int, int]:
    """Count complete dies on a wafer by laying out the real die grid.

    Returns (dies_on_wafer, dies_usable):
      dies_on_wafer - dies entirely inside the physical wafer
      dies_usable   - dies entirely inside the edge-exclusion radius
    The grid origin is shifted in GRID_OFFSET_STEPS steps per axis (fractions of the
    die pitch) and the placement with the most usable dies is kept - what a
    stepper-map optimiser does. Exact geometry, so it stays correct for extreme
    aspect ratios where the textbook DPW approximation breaks down.
    """
    if wafer_diameter_mm <= 0 or die_w_mm <= 0 or die_h_mm <= 0:
        raise ValueError("wafer diameter and die dimensions must be > 0")
    if edge_exclusion_mm < 0 or scribe_um < 0:
        raise ValueError("edge exclusion and scribe width must be >= 0")
    r = wafer_diameter_mm / 2.0
    r_use = r - edge_exclusion_mm
    if r_use <= 0:
        return 0, 0
    px = die_w_mm + scribe_um / 1000.0
    py = die_h_mm + scribe_um / 1000.0
    nx, ny = int(math.ceil(r / px)) + 1, int(math.ceil(r / py)) + 1
    off = np.arange(GRID_OFFSET_STEPS) / GRID_OFFSET_STEPS
    # die lower-left corners for every (offset, index): shape (offsets, dies) per axis
    x0 = (np.arange(-nx, nx + 1)[None, :] + off[:, None]) * px
    y0 = (np.arange(-ny, ny + 1)[None, :] + off[:, None]) * py
    # farthest corner of each die from the wafer centre, squared
    fx2 = np.maximum(np.abs(x0), np.abs(x0 + die_w_mm)) ** 2
    fy2 = np.maximum(np.abs(y0), np.abs(y0 + die_h_mm)) ** 2
    far2 = fx2[:, None, :, None] + fy2[None, :, None, :]        # (ox, oy, i, j)
    usable = np.count_nonzero(far2 <= r_use ** 2, axis=(2, 3))
    on_wafer = np.count_nonzero(far2 <= r ** 2, axis=(2, 3))
    bi = np.unravel_index(int(np.argmax(usable)), usable.shape)   # first max -> deterministic
    return int(on_wafer[bi]), int(usable[bi])


def dpw_approximation(wafer_diameter_mm: float, die_area_mm2: float) -> float:
    """Textbook dies-per-wafer estimate: pi*r^2/A - pi*d/sqrt(2A). Used only to cross-check."""
    r = wafer_diameter_mm / 2.0
    return math.pi * r * r / die_area_mm2 - math.pi * wafer_diameter_mm / math.sqrt(2 * die_area_mm2)
