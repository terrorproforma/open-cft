# pic2d plume development run v2.1 - axially extended plume box (PREPARED, NOT LAUNCHED)

**Status: development / screening. Not preregistered, not validated, not a performance
prediction. Not launched (2026-09-04): the GPU carries plume attempt 8 (PID 51256, model v2.0.1,
a resume of attempt 7) until ~20:00 AEST.** Model spec `modern/spec/pic2d/pic2d-model-v2.1.json`;
field source `modern/spec/pic2d/p2-field-plume-extension-v2.json`.

## Why a longer box

Attempt 7 (`../pic2d_cft_plume_v1/results-attempt7-wall-budget-no-plateau`, window 3.0-3.6 us): the
axis ion density rises past the exit to 4.0x the exit value at z = 27.45 mm (a focus past the axis
null), then falls with an e-folding length of 2.6-3.1 mm and is still 14.6 % of the exit value on
the 36 mm far plane. Extrapolation: 10 % of the exit value at z = 37-39 mm (robust), 1 % at 43-45 mm
(exponential fits) to 46-58 mm (conical power law) - a lower bound, because the 0 V far plane at
36 mm was itself inside the acceleration region. The far-plane ion current holds 95 % inside a 25 deg
cone (5.6 mm at 12 mm behind the exit -> 11.2 mm at 24 mm); the recorded 60 deg "95 % half-angle"
is a side-wall population (6.8 % of the crossings leave at > 45 deg within 7 mm of the front face)
that no far plane of sane radius captures, so `R_plume` stays at the 12 mm return-yoke radius.

## What v2.1 changes

| item | v2.0.2 | v2.1 |
| --- | --- | --- |
| plume box | 12 x 12 mm (far plane z = 36 mm, bounded by the level-1 FEM domain z <= 36.25 mm) | **12 x 24 mm (far plane z = 48 mm = 1.0 L_channel)**, uniform 50 um: 240 x 960 cells, 135,540 plasma cells, 135,828 unknowns, 721 far-field nodes |
| static B | level-1 authority checkpoint, direct node evaluation (`p2-field-plume-extension-v1.json`) | **domain-padding-1.5 P2 solution** of the same design (FEM box to z = 60.75 mm) over the whole L-shaped domain (`p2-field-plume-extension-v2.json`, protocol key `field_plume_extension`); channel agreement with the qualified field 0.74 mT max (gate 20 mT); the v2.0 far plume carried a 15 % FEM truncation at 36 mm |
| mesh | exit-plane column by coordinate comparison (0.044 / 880 misclassified node 480) | exit-plane column **by index** (`mesh.py`); v2.0 masks unchanged |
| plume gate arming | 2.4 us (channel residence) | 3.8 us (one ion transit of the new box); v2.0.2 window statistic unchanged |
| transit / budget | 3.1 us; 4 h (attempt 8: 50 400 s) | 3.8 us; 79 200 s (22 h) for >= 3 transits |
| runner | - | resume-state hygiene: `run_state.json` `finished` / `stop_reason` are the current session's; the previous terminal block lives in `history` |

Everything else (cathode region and continuity rule, two-zone neutrals, ledgers, histograms, gates,
frame recorder, CUDA-graph step) is v2.0.2. `config_sha256` differs from the v2.0.2 protocol's
(`1937f379...`): **a v2.1 run is a fresh start; no v2.0.x checkpoint can be resumed under it.**

## Cost (projected; anchors: attempt 8 7.0-7.15 ms/step at ~4.4 M particles, 5 min factorisation at 721 nodes/row)

| box (z_far x r_far) | plasma cells | inverse blocks | factorisation | ms/step | transit | 3 transits |
| --- | --- | --- | --- | --- | --- | --- |
| 36 x 12 (v2.0, measured) | 77,940 | 1.00 GB | 5 min | 7.1 | 3.11 us | 12.2 h |
| 44 x 12 | 116,160 | 1.50 GB | ~9 min | 7.8 | 3.58 us | 15.5 h |
| **48 x 12 (proposal)** | **135,540** | **1.78 GB** | **~12 min** | **8.2** | **3.81 us** | **17.4 h (+2 h with the particle-count trend)** |
| 48 x 16 | 173,940 | 2.37 GB | ~16 min | 9.9 | 3.81 us | 20.9 h |
| 60 x 12 | 193,140 | 2.78 GB | ~23 min | 9.6 | 4.52 us | 24.2 h |
| 60 x 24 | 365,940 | 5.55 GB | ~46 min | 15.9 | 4.52 us | 39.8 h |

GPU memory ~8.8 GB projected for the proposal (attempt-8 footprint 8.0 + 0.78 GB of inverse blocks);
host ~1.9 GB during the factorisation (one at a time). The solver and kernels are uniform-spacing
only; the graded-mesh and two-domain alternatives are costed in the spec and are not needed at 48 mm.

## Commands (from `modern/`; do NOT launch while another PIC run holds the GPU)

    $env:PYTHONPATH="$PWD\src;$PWD"
    python -m experiments.pic2d_cft_plume_v2_1.run run        # fresh start; ~12 min host factorisation first
    python -m experiments.pic2d_cft_plume_v2_1.run status
    python -m experiments.pic2d_cft_plume_v2_1.run finalize

## Tests

`tests/pic2d/test_pic2d_v21_domain_extension.py` (17) and
`tests/pic2d/test_pic2d_steady_state_runner.py::test_resume_resets_the_terminal_state_and_keeps_the_previous_stop_in_history`.

## Launch log

* 2026-09-04 (prepared): configuration, field source, mesh fix, runner hygiene, spec and cost table
  committed; no run.
