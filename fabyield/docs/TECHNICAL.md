# FabYield — Technical Documentation

Deterministic semiconductor yield, cycle-time and feasibility engine. This document
covers (1) the yield-model choice, (2) how fabrication time is built, (3) the
time-dependent defect model, (4) every assumption and its source, (5) feasibility
rules, (6) validation, (7) the worked 5 nm example, and (8) running it in production.

> **Read this first — what this model is and is not.** Real fab yield models are
> calibrated on proprietary inline-inspection and wafer-sort data that no public
> source provides. FabYield is therefore a *physics-structured, literature-anchored*
> model: the equations are standard, the headline numbers (D0, mask count, cycle
> time, tool throughput) come from public sources where they exist, and everything
> else is an explicit, logged assumption you can replace with your own fab data.
> Treat outputs as engineering estimates for planning and what-if analysis, not
> as a prediction for a specific fab.

---

## 1. Yield model selection

All models take the killer-defect density **D** (defects/cm²) and die area **A** (cm²).

| Model | Formula | Assumed defect distribution | Use |
|---|---|---|---|
| Poisson | Y = e^(−DA) | uniform, no clustering | pessimistic bound for large dies |
| Murphy (1964) | Y = ((1 − e^(−DA)) / DA)² | triangular D across wafers | classic industry default |
| Seeds (1967) | Y = 1 / (1 + DA) | exponential D | optimistic bound |
| **Negative binomial** (Stapper) | Y = (1 + DA/α)^(−α) | gamma-distributed D, cluster parameter α | **default** |

**Why negative binomial is the default.** It is the only one of the four with a
clustering parameter, so it can treat *random* particle defects (strongly
clustered, α = 0.5) and *systematic* layout/process defects (more uniform, α = 2.0)
differently, as the spec requires. It contains the others as limits: α = 1 is
Seeds, α → ∞ is Poisson (both proven in `tests/test_physics.py`). Murphy is kept
because it is the historical reference.

Total D is split into random and systematic parts (`systematic_fraction` per node),
and each part gets its own yield term:

```
Y_die = Y_NB(D_random, A, α=0.5) × Y_NB(D_systematic, A, α=2.0) × Y_parametric
```

**How much the choice matters** (800 mm² die, D = 0.10/cm²):

| Model | Overall yield | Good dies/month @ 2,500 wafer starts |
|---|---|---|
| Poisson | 42.4 % | 72,010 |
| Murphy | 43.8 % | 74,444 |
| Seeds | 49.1 % | 83,469 |
| Negative binomial | 52.5 % | 89,296 |

A 10-point spread from model choice alone is normal for an 8 cm² die. Pick the
model your fab's historical wafer-sort data fits best (see §8).

## 2. Yield chain: die → wafer → lot → month

| Level | Formula |
|---|---|
| Die yield | Y_random × Y_systematic × Y_parametric |
| Edge yield | dies_usable / dies_on_wafer (complete dies lost in the edge-exclusion ring) |
| Line yield | Π (1 − scrap_rate_i)^(count_i) over every step (wafers scrapped in the line) |
| **Overall yield** | Y_die × Y_edge × Y_line = good dies / complete dies on the wafer |
| Good dies per wafer start | dies_on_wafer × overall yield |
| Good dies per lot | good per wafer × lot size (25) |
| Good dies per month | good per wafer × wafer-start capacity |

**Die count** is not the textbook formula. The engine lays out the real die grid
(die + scribe pitch) and slides its origin in pitch/16 steps on both axes, keeping
the placement with the most usable dies. This is exact for any aspect ratio. It was
cross-checked against a separately written brute-force counter (1 mm offset mesh,
corner-in-circle test): both give 67 usable dies at 3 mm exclusion and 69 at 2 mm
for a 25 × 32 mm die.

**Loss breakdown.** Yields multiply, so losses cannot simply be added. Each
mechanism gets the share −ln(Yᵢ) / −ln(Y_total) of the total loss, which makes the
per-mechanism "loss points" add exactly to (1 − Y_total) × 100.

## 3. Fabrication time and time-dependent defect density

### 3.1 Raw process time from equipment throughput

Each node flow is a list of layers in modules **FEOL → MOL → BEOL → (BACKSIDE)**.
Each layer states how many times each step type runs, and carries its critical
dimension, overlay tolerance and inspection sensitivity. Time per step, per lot of 25:

```
single-wafer tool:  hours = 25 / throughput_wph + overhead_hours
batch tool:         hours = batch_hours
raw process time (RPT) = Σ over all steps
calendar cycle time    = RPT × X-factor / 24
```

### 3.2 X-factor calibration (read this)

Published data give ≈ 1 day per mask layer at 7 nm and 5 nm (80–85 days / 80–85
layers; ~100 days / ~100 layers). The flows here list about 400 module-level steps,
while real 5 nm flows are reported to have well over 1,000. The un-listed sub-steps
(resist coat/bake/develop, extra cleans and inspections, transport) are not modelled
one by one, so the X-factor was **calibrated** to reproduce the published cycle
time: 6.9 (7 nm), 7.5 (5 nm), 7.35 (3 nm).

That is higher than the textbook *queueing* X-factor of ~3, because it absorbs
both queueing and the un-listed sub-steps (roughly 3 × 2.4). If you have your fab's
real step list, list it in full and set the X-factor to your measured queueing value.

### 3.3 D(t) — the spec's time-dependent defect model

```
D(t) = D0_base + Σ_i  k_i × t_i^α_i          (i = litho, etch, dep, cmp, thermal, implant, clean, metrology)
```

`t_i` is the total hours a lot spends in category *i*. The exponents come from the
physics of each mechanism:

| Category | α_i | Mechanism | Share of time-driven D |
|---|---|---|---|
| litho | 0.5 | overlay error accumulates as a random walk over passes: σ ∝ √N | 30 % |
| etch | 1.0 | profile non-uniformity / micro-loading grows with etch time | 25 % |
| dep | 1.0 | chamber particle adders scale with exposure time | 20 % |
| cmp | 1.0 | scratches, dishing, erosion scale with polish time | 15 % |
| thermal | 0.5 | dopant diffusion length L = √(D·t) | 7 % |
| implant | 1.0 | damage / contamination per pass | 3 % |
| clean, metrology | — | remove / non-destructive: **zero contribution** (k = 0) | 0 % |

**Calibration of k_i.** Public sources give one number per node (D0), not a
per-step breakdown. The engine assumes 30 % of D0 is time-driven
(`time_defect_fraction`), splits that by the shares above, and solves

```
k_i = D0_published × 0.30 × share_i / t_i,nominal^α_i
```

so that the **nominal** flow reproduces the published D0 exactly (tested). The k_i
are then frozen: a flow that runs 15 % longer, or a longer anneal, raises D through
the exponents. This is what makes fabrication time move yield.

## 4. Assumptions register

Every run writes these into `audit.assumptions` in the JSON output as well.

| Parameter | 7 nm | 5 nm | 3 nm GAA | Source / status |
|---|---|---|---|---|
| D0 (/cm²) | 0.09 | 0.10 | 0.15 | 7/5 nm: TSMC-reported values [1]; **3 nm: assumption** (no public figure found) |
| Mask layers | 82 | 100 | 108 | 7/5 nm: match published 80–85 / ~100 [2]; **3 nm: assumption** |
| Cycle time (days) | 82.5 | 100 | 108 | 7/5 nm published [2]; **3 nm: assumption** (1 day/layer) |
| EUV layers | 0 | 14 | 20 | approximate public figures |
| Systematic fraction | 20 % | 25 % | 30 % | assumption: rises with scaling |
| Parametric yield | 97 % | 96 % | 95 % | assumption |
| Time-driven share of D0 | 30 % | 30 % | 30 % | assumption |

| Step | Throughput | Source |
|---|---|---|
| EUV scanner | 160 wph | ASML NXE:3600D spec [3] |
| DUV immersion | 275 wph | ASML NXT immersion class [4] (nominal) |
| etch / dep / CMP / implant / clean / metrology | 30 / 25 / 40 / 20 / 60 / 30 wph | assumption (cluster-tool class) |
| thermal (furnace batch) | 3 h per batch | assumption |
| scrap per step | 0–2×10⁻⁵ | assumption (gives ~99.7 % line yield) |

Also assumed: lot = 25 wafers; edge exclusion 3 mm; scribe 80 µm; scanner field
26 × 33 mm (858 mm² reticle limit); memory repairs 40 % of random defects via
redundancy; 450 mm tool times equal to 300 mm (450 mm never reached volume
production); chiplet packaging yield 98 %.

**Note on "5 nm GAA".** Public 5 nm-class processes (TSMC N5, Samsung 5LPE) are
FinFET; gate-all-around first ships at the 3 nm class. The 5 nm config is therefore
FinFET, and the GAA flow (superlattice epi, inner spacers, channel release) is in the
3 nm config. The 3 nm config also has an optional backside-power module
(`"backside_power_delivery": true`), a 2 nm-class feature included for what-if studies.

## 5. Feasibility rules

| Condition | Verdict |
|---|---|
| die larger than the 26 × 33 mm scanner field | INFEASIBLE |
| no complete die inside the usable wafer area | INFEASIBLE |
| required yield > zero-defect ceiling (Y_param × Y_edge × Y_line) | INFEASIBLE |
| achievable good dies/month < target | INFEASIBLE |
| volume headroom < 10 % | MARGINAL |
| Monte Carlo P(meet target) < 90 % | MARGINAL |
| cycle time > limit | MARGINAL (INFEASIBLE if > 125 % of limit) |
| none of the above | ACHIEVABLE |

The verdict is the worst condition triggered, and every triggered condition is
listed. Other outputs:

- **Required wafer starts**: ⌈target / good dies per wafer start⌉.
- **Required yield**: target / (capacity × dies on wafer).
- **Required defect density**: found by bisection. It's the D that just meets target
  at the current capacity. It is `null` when even D = 0 cannot meet target, which
  means the gap is not a defect problem.

**Bottlenecks.** The yield-limiting mechanism is the one with the largest loss
points. The yield-limiting step is the category adding the most to D(t). The
cycle-time driver is the category and module with the most hours. Wafers pass
modules in strict sequence, so the whole flow is the critical path, and its modules
and layers are ranked by their share of it.

**Improvements.** Each candidate is re-run through the engine and ranked by
(relative good-die gain) ÷ (relative cost index: LOW 1, MEDIUM 3, HIGH 10 — an
assumption). The candidates are: defect-reduction program, DFM/OPC hotspot fixes,
faster processing in the top time-driven category, tighter edge exclusion, advanced
process control, and a 2-chiplet split (for dies > 400 mm²).

## 6. Sensitivity

- **Tornado**: defect density ±20 %, die area ±10 % (aspect ratio kept), process
  time ±15 %, one at a time. Sorted by yield swing.
- **Monte Carlo**: all three varied together, drawn uniformly within the bounds,
  using `numpy.random.default_rng(seed)`. Uniform because the spec gives bounds,
  not standard deviations. The same seed gives bit-identical output. The output has
  the mean, standard deviation, P5/P50/P95, a 90 % confidence interval and
  P(meet target).

## 7. Validation

| Check | Test | Result |
|---|---|---|
| Model equations vs closed-form reference values (DA = 0, 0.5, 1, 2) | `test_models_match_reference_values` | exact to 1e-8 |
| Murphy textbook example (D0 = 0.5/cm², 1 cm²) → 0.6193 | `test_murphy_classic_example` | pass |
| Model ordering Poisson < Murphy < Seeds; NB limits | `test_model_ordering…`, `test_negative_binomial_limits` | pass |
| Mask count = published (82, 100) | `test_mask_count_matches_published` | pass |
| Cycle time within 5 % of published, 0.8–1.5 days/layer | `test_cycle_time_matches_published` | pass |
| Nominal D(t) = published D0 | `test_k_calibration_reproduces_published_d0` | pass |
| Die count vs textbook DPW (< 5 %) and area scaling 300 → 450 mm | `test_die_count…`, `test_450mm…` | pass |
| Feasibility boundaries (zero target, ±10 % margin, ceiling, reticle, cycle limit) | `tests/test_engine.py` | pass |
| Full determinism (two runs, byte-identical JSON) | `test_fully_deterministic` | pass |

**Honest limits of this validation.** The cycle-time and D0 tests check that
calibrated inputs reproduce the public numbers they were calibrated to. They are
consistency checks, not independent validation. No verifiable public 90 nm
wafer-sort dataset was found, so the Murphy "reference" tests use closed-form
values, not historical fab data. True validation needs a fab's own die-yield
history: fit D0, α and the systematic fraction to it, then compare predictions.

Test suite: 75 tests, 99 % line coverage (`pytest --cov=fabyield`).

## 8. Operating it in production

1. **Install**: Python ≥ 3.10, `pip install numpy` (pytest + pytest-cov for tests).
2. **Run**: `python -m fabyield run.json --out result.json` (`-v` logs every step).
   Exit code 0 means success; exit code 2 means a configuration error, with the
   message on stderr.
3. **Calibrate to your fab** (this matters more than anything else in this document):
   - Replace `d0_published` with your inline defect density per node.
   - Fit `systematic_fraction` and the yield model to wafer-sort history.
   - Replace the step counts with your real route, and throughputs with your tools.
   - Set `x_factor` from measured cycle time ÷ measured raw process time.
   - Check that `tests/test_process.py` still passes against your published targets.
4. **Keep the audit trail.** Store `result.json`. It holds every input, assumption
   and intermediate value, so any number in a report can be traced back.
5. **Pin versions.** Output is deterministic for a given numpy version. As with the
   FabSentinel ML pipeline, pin numpy in `requirements.txt`.

## References

1. TSMC D0 disclosures for N7 (0.09/cm²) and N5 (0.10–0.11/cm²):
   https://ljdevice.com.tw/en/better-yield-5nm-7nm-tsmc-update-defect-rates-n5/
2. Cycle times and layer counts by node, days per layer:
   https://semiengineering.com/battling-fab-cycle-times/
3. ASML NXE:3600D, 160 wph: https://www.asml.com/en/products/euv-lithography-systems/twinscan-nxe-3600d
4. ASML DUV immersion systems: https://www.asml.com/en/products/duv-lithography-systems
5. Yield-model forms (Poisson, Murphy, Seeds, negative binomial): standard texts, e.g.
   Stapper, "Integrated circuit yield statistics," Proc. IEEE, 1983.
