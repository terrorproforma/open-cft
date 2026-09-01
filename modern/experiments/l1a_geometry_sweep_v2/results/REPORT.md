# Preregistered L1a geometry sweep v2

- Preregistration commit: `092f5fae692ee7d6711e0c7e1c94dac6a345f37c`
- Classification: `L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID`
- Terminal acceptance: `ACCEPTED`
- Evaluated: 96
- Failed: 0
- Nondominated: 25
- Representative roles: 5
- Unique representative artifacts: 4

## Seven terminal gates

- `boundary`: PASS (0 failures; observed `0.011652580349992949`)
- `residual`: PASS (0 failures; observed `9.998083198193306e-11`)
- `cpu_cuda_parity`: PASS (0 failures; observed `{'psi': 1.633321727281456e-15, 'br': 5.3105473081447324e-15, 'bz': 2.893550379622971e-15}`)
- `flux_identity`: PASS (0 failures; observed `4.547473508864641e-13`)
- `source_representation`: PASS (0 failures; observed `0.017476475601085336`)
- `topology_confidence`: PASS (0 failures; observed `0.9117788360760948`)
- `manufacturability`: PASS (0 failures; observed `4.9999999999999806e-05`)

## QoI ranges

- `axis_cusp_count`: 3 to 5
- `axis_null_count`: 4 to 6
- `boundary_to_peak_ratio`: 0.00178704804309 to 0.01165258035
- `centreline_abs_bz_peak_t`: 0.0574998036336 to 0.371886747611
- `centreline_bz_max_t`: 0.0572069249254 to 0.371886747611
- `centreline_bz_min_t`: -0.343414708556 to -0.0574998036336
- `centreline_mid_abs_bz_t`: 0.000849439316522 to 0.343414708556
- `field_energy_j`: 0.0123317295577 to 0.953791290443
- `field_peak_t`: 0.383190466061 to 0.967831921464
- `flux_reconstruction_identity_t_per_m`: 1.42108547152e-13 to 4.54747350886e-13
- `maximum_mirror_ratio`: 4.36779915958 to 2672.765829
- `minimum_mirror_ratio`: 3.31950981942 to 22.2333717906
- `relative_residual_l2`: 8.71313937195e-11 to 9.99808319819e-11
- `source_representation_error`: 0.00212944308142 to 0.0174764756011
- `stage_gradient_max_abs_t_per_m`: 3.71774503634 to 48.0476219577
- `stage_gradient_rms_t_per_m`: 2.51739143719 to 39.169259182
- `topology_confidence`: 0.911778836076 to 0.95799341559

## Representative roles

- `strongest-centreline`: `l1a-gs-v2-065-9e98f08f3b`
- `strongest-mirror`: `l1a-gs-v2-032-570ad83ba6`
- `steepest-stage-gradient`: `l1a-gs-v2-068-375d1b1b13`
- `lowest-field-energy`: `l1a-gs-v2-000-48d2ccedd5`
- `best-boundary-isolation`: `l1a-gs-v2-000-48d2ccedd5`

## Replay and claim boundary

Sampling, geometry, source and configuration identities are bitwise hash-bound.
CUDA floating output is not claimed or required to be bitwise reproducible;
future replay uses the preregistered scale-aware tolerances. Artifact hashes
identify this single run. The six CPU comparisons are independent parity evidence.

This is L1a field-only screening. It is not a material-aware permanent-magnet
model, propulsion calculation, hardware-valid prediction or build qualification.
