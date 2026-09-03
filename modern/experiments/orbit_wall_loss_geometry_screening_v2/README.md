# Orbit wall-loss geometry screening v2

**Classification: `SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS`.** Full-orbit collisionless
test-particle electron wall-loss probabilities (orbit_mc v1.7, numpy CPU) integrated in the
accepted L1a linear-vacuum equivalent-current fields of the geometry sweep v2, launched in the
separatrix-bounded wall cells of the accepted cusp topology search v3.1 catalogue. The fields
are L1a screening maps, **not P2-qualified**; no number here is accepted physical-orbit evidence
and none is a plasma or performance claim. One additional row, the v4 divergent-exit design on
its NUMERICAL_P2_QUALIFIED field, carries the label
`P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN` (screening launch design on a qualified field, not a
v4 replication). Every per-cell number is a collisionless geometric wall-access fraction of the
launch distribution, never a cusp-loss probability.

## Why v2 (literature synthesis step 2; surrogate/MDO review blocker 1)

v1 launched at four fixed fractions of the straight span with 128 launches per cell, so its
labels were binomial-noise limited (floor about 0.035 per cell) and its cells were not physical.
v2 keeps every one of the 96 designs (option B, 40 designs x 2048, was rejected), makes the
**v3.1 catalogue cells** the launch cells, replaces the 8-gyrophase grid by **scrambled-Sobol**
launch sets with a frozen seed per (design, cell, stratum), and applies the **frozen two-stage
allocation rule**: 128 launches per cell for every design, then a top-up to 512 only in the
cells whose stage-1 Wilson 95 % width exceeds 0.10. A frozen-seed control of one eighth of every
cell's final launches is re-integrated at the 2N time step (MLMC logic). Independent launch sets
per design.

## Field provenance

Exactly as v1 (`designs.py` reuses `experiments.orbit_wall_loss_geometry_screening_v1.designs` by
import): every sweep design is rebuilt with the sweep's own `build_case`, re-solved with the
accepted L1a CPU solver, and identity-proven against the sealed sweep record (geometry / source /
config / case hashes, QoI replay, node-wise agreement for the four representatives). **Coverage
of the 2x refined re-solve and the cross-resolution diagnostic: every sweep design (96/96)**
(v1 ran it for its four representatives only; v1's README overstated that). No refined-field
orbit case is run. The P2 row uses the hash-bound v4 adapter (level-1 accepted, level-2 as the
cross-resolution partner).

## Cells and launch design (per design)

* Cells: the catalogue cells of `cusp_topology_search_v3_1` (bound by the catalogue bytes; 373
  sweep cells: 96 anode-side partials, 181 interior, 96 exit-side partials; P2: 4 cells incl. a
  28 um exit-side partial). The catalogue's field identity is v3.1's scheme and is carried beside
  this campaign's v1-scheme identity; equality is not asserted.
* Launch plane: the cell midpoint (between its two wall cusps; between the channel end and the
  cusp for partials). One sweep cell (`l1a-gs-v2-088`, 0.16 mm anode-side) has its midpoint inside
  the injector zone and is flagged, not moved; 13 sweep cells shorter than 1 mm are flagged.
* Strata: (5 / 25 eV) x (20 / 70 deg) x (-1 / +1) = 8 per cell, exactly equal counts.
* Scrambled Sobol' (own implementation, `sobol.py`: Joe-Kuo direction numbers, Gray code, LMS +
  digital shift): dimension 1 selects the radius band ([0.650, 0.700] or [0.775, 0.825] r_w),
  dimension 2 the radius inside the band, dimension 3 the gyrophase. Stage 1 = indices 0..15 per
  stratum; stage 2 = indices 16..63 of the same sequences (the union is a (t, 6, 3)-net).
* Launch ids follow orbit_mc's grammar `<campaign>:E<e>:P<p>:X<cell index>:D<+-1>:G<Sobol index>`.

## Case structure and the orbit_mc v1.7 defect

One orbit_mc case = one (design, cell, 16-index block) at one time step: `stage1` (block 0,
128 launches, frozen authority), `stage2b1..3` (blocks 1..3, 128 each, only for topped-up
cells), `control` (16 of 128 or 64 of 512 launches at 2N). **Every case has 128, 64 or 16
launches on purpose**: orbit_mc v1.7's artifact validator requires `lower <= p <= upper`
verbatim, but `wilson_interval(0, n).lower` is a positive round-off for 734 of the first 4000 n
(384 among them) and `wilson_interval(n, n).upper` is `1 - ulp` for 1238 of them (512, 640,
...). A zero-count category (timeouts are always zero) at such an n aborts sealing. The plan
constructors refuse any case size that is not exact at both ends; the tests pin the safe set.

## Gates (per design, binding for the dataset)

v1's integrity gates (field adapter incl. cross-resolution, campaign preflight, zero numerical
failures, energy drift exactly 0, final velocity = event velocity, wall endpoint <= 1e-8 m,
earliest-event ordering, runtime rotation bound, relativistic phase, material quarantine,
exact deterministic replay when sealed, cross-process determinism sample, handoff consumed)
plus **allocation-rule replay** (the main process recomputes every cell's top-up decision, the
stage-2 case authorities and the control selection/authority from the endpoint terminations),
**catalogue binding**, **stage-1 authority frozen**, **case sizes Wilson-exact**. Campaign
gate: pooled paired 2N control `|P_2N - P_N| <= 0.02`. Per-design flag `timestep_passed`
(the design's own paired control within 0.02) decides sealing and is reported; timeouts and the
v1 comparison are reported, never gated.

## Estimators

Per cell: stage-1, stage-2 and pooled counts with Wilson 95 % intervals for P(wall) / P(reflect)
/ P(escape) / P(timeout); binomial floor `sqrt(p(1-p)/n)` and Jeffreys floor at `(k+1/2)/(n+1)`;
surrogate-v3 readiness iff Jeffreys floor <= 0.02. Per design: wall-area-weighted average of the
per-cell P(wall) (declared value) and launch-weighted average (v1-comparable); a direct per-cell
comparison with v1 is impossible (different cells), so only pooled values are compared.

## Lifecycle

```
python -m experiments.orbit_wall_loss_geometry_screening_v2.run shakedown  # 3 sweep designs + P2, 16 launches/cell, temp root
python -m experiments.orbit_wall_loss_geometry_screening_v2.run prepare    # refuses without a valid shakedown.json
# commit "preregister orbit wall-loss geometry screening v2", push, then from a clean detached worktree
python -m experiments.orbit_wall_loss_geometry_screening_v2.run execute
python -m experiments.orbit_wall_loss_geometry_screening_v2.run validate
```

## Outcome (single execution, preregistration `cef1ee59`, result `26029b72`)

`accepted_result` / `accepted_screening_dataset`: 97 designs (96 sweep-v2 + the P2 row), 377
catalogue cells, 1105 cases, 104,832 orbits (48,256 stage 1, 44,928 stage 2, 11,648 control),
70 min of orbit wall time on 12 workers, 16,549 validators passed / 0 failed, no exclusion, no
timeout, energy drift exactly 0. Allocation: 117 cells topped up to 512 (32 anode-side, 83
exit-side, 2 of the P2 row), 260 saturated at 128 (all 181 interior cells at 128/128 wall hits,
34 anode-side and 11 exit-side at 128/128, 32 anode-side cells at 118-127/128); replay of the
rule and of the control selection reproduced every case authority. Per-cell P(wall) by position:
anode-side median 0.984 (min 0.31), interior 1.000 for every cell, exit-side median 0.50 (range
0.25-1.0; 0.50 = the direction/pitch structure of the 90 divergent-exit designs, 1.0 for the 6
straight-exit designs). Reflections: 10,407 (11.2 % of the final launches), 65 of the 96
exit-side cells and 1 anode-side cell, no interior cell; 66 designs. Floors: Jeffreys floor
median 0.0055, max 0.0242; **294 / 377 cells (78.0 %) are surrogate-v3 ready** (253 of 258
saturated sweep cells, 38 of 115 topped-up sweep cells; a topped-up cell at p ~ 0.5 has floor
0.022 by the frozen rule). N -> 2N control: 2 discordant orbits in 11,648 (0.017 %), estimated
bias -0.9e-4 +- 0.9e-4, gate passed, control flag true for 97/97 designs, all sealed. v1 vs v2
pooled (launch-weighted): Spearman 0.15, mean difference +0.038, mean |difference| 0.11,
interval overlap 45 % - the same fields ranked by different cells give a different ranking, as
the protocol anticipated. P2 row: anode-side 0.605 (n 512), interior 1.0 / 1.0, exit-side 0.17
with 350 reflections (v4's launch design had none: the 28 um exit-side cell sits at the exit
cusp).

**Manifest published post hoc.** The runtime's own publication step failed with EMFILE (Windows
CRT 8192-descriptor cap, 16,957 files) after the terminal record and every sidecar-attested
artifact were durable; see `POSTHOC_FINALIZATION.md`. Nothing was rerun.

## Outputs (`results/artifacts/`)

`geometry-wall-loss-dataset-v2.json` (per design: identities, geometry, catalogue cells with
their descriptors, per-cell stage counts / pooled intervals / floors / control, pooled design
values, v1 comparison, gates) and `geometry-wall-loss-dataset-v2.csv` (one row per cell),
`allocation-decisions.json`, `coupling-consumer-record.json`, `v1-comparison.json`, per-case
summaries / endpoints / sidecars / handoffs, per-design fields and field evidence,
`design-exclusions.json`, `gates.json`, `campaign-result.json`.

Dashboard: `modern/visualization/wall-loss-geometry-screening-v2.html`.
