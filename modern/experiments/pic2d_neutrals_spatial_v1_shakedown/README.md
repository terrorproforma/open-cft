# pic2d_neutrals_spatial_v1_shakedown — NON-EVIDENTIARY

Box shakedown of PIC model v2.5.0 (`neutrals_spatial_v1` test-particle neutrals + `metastables_v1`; spec
`modern/spec/pic2d/pic2d-model-v2.5.json`) on the ss-v4 33 µm channel-only protocol with `xe_collision_set_v2` on.
Not preregistered, not a result.

## What differs from the ss-v4 / R3 shakedown protocol

* `operating_point.neutral_inventory` (0-D, τ_g 30 ns, recycling at 500 K) is REPLACED by `operating_point.neutrals`
  (`neutrals_spatial_v1`): the same feed 8.551e16 atoms/s, wall temperature 500 K, recycling γ = 1, accommodation 1,
  Knudsen initial profile, macro-weight 2.2e7 (~4 M macro-neutrals), one sub-step per sync interval (200 steps),
  time acceleration F = 100 (declared, artificial), metastables with branching (0.45, 0.35, 0.50, 0.35).
* `operating_point.neutral_density_per_m3` (the MCC null-collision ceiling) 5.5e19 → 1e21: the Knudsen anode density
  at this feed is 5.5e20 (10 × the exit density for the 2 mm bore / 3 mm exit) and the nearest-cell deposit carries ~13 %
  shot noise per cell (clamped above the ceiling; fail-closed at 1e-3 of the cell-substeps per interval).
* identity / status / model fields.

## Stages

```
python -m experiments.pic2d_neutrals_spatial_v1_shakedown.run shakedown --backend warp-cuda \
    --reference experiments/pic2d_xe_collision_set_v2_shakedown/results-shakedown
python -m experiments.pic2d_neutrals_spatial_v1_shakedown.run readings --results <dir> [--reference <0-D results>]
```

`shakedown` = the v4 shakedown stages (100 000 steps with the shrunk cadences, finalize + assess) plus the neutral
readings: trailing-half means of the neutral record, the window-mean axis density profile, the metastable fraction
profile, where the ionisation sits (axial centroid / quartiles of the ionisation-rate map, next to the 0-D shakedown's
when given), the atom-ledger identities and the cumulative neutral ledger. Results directories are gitignored; the
record is `shakedown.json`.

## Reading the record honestly

100 000 steps = 0.14 µs of plasma time: the plasma is the seed transient and the neutral field is the initial Knudsen
profile after 14 µs of (accelerated) neutral time — ~7 % of the tube residence time. The readings exercise the code
path (transport, depletion, recycling, CEX hand-off, metastable channels, ledger identities, cost) and record early
directions only.
