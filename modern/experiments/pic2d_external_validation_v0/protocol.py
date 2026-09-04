"""Run protocols of external validation v0, composed on the steady-state v4 template (PREREGISTERED option ``channel-20um`` for the Lambda H100).

Template: ``pic2d_cft_steady_state_v4/protocol.json`` (model v1.3 closure: v1.2 runner physics, NO wall-ion recycling; v2.0.3 gates: window-mode
peak-Debye gate hard pi / soft 2.5, one-sided windowed residual-power gate 5 %; frame recorder ON; the accepted plateau rule).  Only the blocks the
reference case dictates change:

* geometry / grid  - the reconstructed channel 14 x 1.5 mm, channel-only, at the PUBLISHED resolution 20 um (75 x 700 cells; approximation A9);
* operating point  - anode 400 V, exit plane 0 V, STATIC uniform xenon background 2e20 m^-3 at 500 K (the v1.2 mode: ``neutral_inventory`` removed, so the
                     MCC density is the constant background exactly as the reference's static DSMC import), electron injection 1.8 mA / 1 eV at the exit plane
                     (the reference's continuity-derived effective source), seed plasma 5e16 m^-3 / 5 eV (the template's, unchanged);
* numerics         - dt 0.7 ps (omega_pe dt = 0.2 at 2.5e19 = 2.5x the published typical density; 0.125 at 1e19; Courant 0.42 at 400 eV; omega_ce dt <= 0.09 at
                     0.7 T), a-priori stability reference = the published density / temperature (1e19 / 10 eV / 400 eV), the a-priori cell-Debye limit set to pi
                     (= the v2.0.3 hard gate; the template's 2.0 was its own reference-density ratio 1.0, disclosed), frames every 40 000 steps (= 28 ns, the
                     template's time cadence), window / checkpoint step cadences verbatim (400 000 / 40 000 steps = 0.28 / 0.028 us at 0.7 ps);
* stopping rule    - plateau after >= 3 transits of 1.4 us (2.4 us x 14 / 24 mm), wall budget 1.5x the projected 3-transit wall at the H100 MPS-4 per-process rate;
* acceptance       - (a) plateau, (b) windowed residual power < +2 %, (c) the comparison spec applied to the trailing-window S (comparison.py), verdicts.

Variants (``VARIANTS``): ``base`` (primary: the accepted v1.3 closure, no anomalous transport) and ``bohm-0.4`` (the v1.4 Bohm-scattering hook at alpha = 0.4,
the coefficient of the reference's D_perp = 0.4 k T_e / e B; sealed, NOT the primary run; the discriminating follow-up if the transport-sensitive rows are
discrepant).  Grids (``GRIDS``): ``20um`` (primary), ``33um`` (the ss-v4 option: inadmissible a priori at the published density, kept for the argument) and
``15um`` (the resolution follow-up with the soft margin met at 1e19 / 10 eV).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.pic2d_design_mini_sweep_v1 import cost as sweep_cost

from . import comparison, reference
from . import geometry as geometry_module
from .geometry import CONFIG_ID, PicMapping, channel_volume_m3, pic_mapping

EXPERIMENT_ID = "pic2d-external-validation-v0"
EXPERIMENT_DIR = Path(__file__).resolve().parent
MODERN = EXPERIMENT_DIR.parents[1]
REPOSITORY = MODERN.parent
TEMPLATE_PATH = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "protocol.json"
STATUS = "preregistered_external_validation_v0_channel_20um_h100_mps4_code_to_code_brandt2016_not_validated"
RUN_PROTOCOL_SCHEMA = "cft.pic2d.external-validation-v0.run-protocol/1.0.0"
EXPERIMENT_PROTOCOL_SCHEMA = "cft.pic2d.external-validation-v0.protocol/1.0.0"
PROTOCOLS_DIR = EXPERIMENT_DIR / "protocols"
PREFLIGHT_RECORD = EXPERIMENT_DIR / "preflight-channel-20um.json"
SHAKEDOWN_RECORD = EXPERIMENT_DIR / "shakedown-channel-20um.json"


def preflight_record_path(variant: str = "base", grid: str = "20um") -> Path:
    """The whole-set preflight record carrying the launch-box GPU timing of ``option``: the base keeps the launch-1 file name (sealed at 3dc12cf6)."""

    if variant == "base" and grid == "20um":
        return PREFLIGHT_RECORD
    return EXPERIMENT_DIR / f"preflight-{option_tag(variant, grid)}.json"

# the launch box (recorded in every sealed run protocol under `execution`; the mini-sweep's record for the same box)
LAUNCH_BOX = {
    "gpu": "NVIDIA H100 80GB HBM3",
    "instance": "Lambda gpu_1x_h100_sxm5 (us-southeast-1), ubuntu@68.209.75.2, driver 580.105.08 (CUDA 13.0), 26 vCPU, 221 GiB RAM",
    "software": "Python 3.12.14, warp-lang 1.14.0 (PyPI CUDA 12.9 build, sm_90 JIT), numpy 2.5.2 (scipy-openblas 0.3.34 Haswell DYNAMIC_ARCH), Ubuntu 22.04",
    "concurrency": ("CUDA MPS (nvidia-cuda-mps-control -d; CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps, CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log): this run takes ONE of the four "
                    "MPS slots of tools/cloud/jobs.yaml (slots_per_gpu 4) beside the preregistered design mini-sweep v1 runs (291a9227) that are still executing at launch; "
                    "benchmark anchor 3.37 ms/step alone -> 8.71 ms/step per process at N = 4 for the v4 configuration (aggregate 1.54x)"),
    "determinism": ("MPS does not change a process's own kernel order: the mini-sweep's replay on this box (mps-replay.json, 291a9227) showed the physics state bitwise between "
                    "concurrent MPS processes and solo processes, float-atomic diagnostic accumulators at <= 2.2e-13 - the same pattern solo-vs-solo"),
    "provenance_caveats": [
        "the runner's gpu_utilisation_percent_samples (nvidia-smi, 300 s) read the WHOLE GPU under MPS - shared-GPU readings, not per process",
        ("ms/step per process under MPS depends on how many slots are busy: the measured launch-box rate below was taken with the sweep runs active and the run "
         "speeds up as they finish; wall-clock projections are per-slot upper bounds, the budget is a cap, not a cost"),
        "same-seed replay across the RTX 5090 (sm_120) and the H100 (sm_90) is numerical, never bitwise (tools/cloud/PLAN.md s.6)",
    ],
}
# ms/step measured on the launch box by `preflight --gpu-timing` (>= 2000 production steps at the seed load and at the projected plateau
# load, under the MPS contention present at the time). The wall budget is derived from max(cost-model MPS-4 rate, measured plateau-load
# rate) so a slower-than-modelled box can never starve the 3-transit plateau rule; a faster measurement leaves the cost-model budget.
LAUNCH_BOX_TIMING: dict[str, Any] | None = {
    "utc": "2026-09-04T11:56:14Z",
    "host": "ubuntu@68.209.75.2 (NVIDIA H100 80GB HBM3, GPU-a800b021-6364-473f-5177-cd6ae7ce0005, driver 580.105.08)",
    "record": "modern/experiments/pic2d_external_validation_v0/preflight-channel-20um.json launch_box_timing (first box preflight at 42e30aaa; the sealed record is the re-run at the "
              "preregistration commit and may differ in the last digits and in the client count)",
    "timing_steps": 2000,
    "ms_per_step_at_seed_load": 5.58,                 # 60 000 seed electrons (5e16 m^-3 / 5 eV), 2000 steps after 200 warm-up
    "ms_per_step_at_plateau_load": 13.07,             # 6.0 M e- + 6.0 M i (synthetic uniform 5e18 seed = the 12 M-particle cap), 2000 steps after 200 warm-up
    "concurrent_mps_clients": 5,                      # three mini-sweep runs (reference, 047, 009) + the steady-state v5 shakedown + a profiling job of other agents
    "factorisation_seconds": 7.1,
    "hours_to_3_transits_at_plateau_load": 21.8,      # 6.0 M steps x 13.07 ms; the cost model's MPS-4 projection is 30.6 h at 18.3 ms/step
    "note": ("measured under SIX-way GPU sharing (five other CUDA-MPS clients), i.e. heavier contention than the four-slot configuration the run executes in; the per-process "
             "rate falls toward the solo rate (~7 ms/step by the cost model) as the sweep runs finish, so the measured value is an upper bound for the launch configuration "
             "and the cost-model MPS-4 rate (slower) stays the budget basis: 46.0 h"),
}
# amendment 1: the bohm-0.4 option's own launch-box timing (`preflight --variant bohm-0.4 --gpu-timing`, written to preflight-channel-20um-bohm-0.4.json); None until
# measured (the composition then falls back to the cost model + the base measurement). Filled before the amendment commit; the sealed record may differ in the last digits.
LAUNCH_BOX_TIMING_BOHM: dict[str, Any] | None = {
    "utc": "2026-09-04T16:56:52Z",
    "host": "ubuntu@68.209.75.2 (NVIDIA H100 80GB HBM3, GPU-a800b021-6364-473f-5177-cd6ae7ce0005, driver 580.105.08)",
    "record": "modern/experiments/pic2d_external_validation_v0/preflight-channel-20um-bohm-0.4.json launch_box_timing (code c1508c06: model v2.1.0 rotation closure, v2.0.6 gates, K = 5)",
    "timing_steps": 2000,
    "ms_per_step_at_seed_load": 2.71,                 # 60 000 seed electrons (5e16 m^-3 / 5 eV), 2000 steps after 200 warm-up
    "ms_per_step_at_plateau_load": 7.22,              # 6.0 M e- + 6.0 M i (synthetic uniform 5e18 seed = the 12 M-particle cap), 2000 steps after 200 warm-up
    "concurrent_mps_clients": 3,                      # ss25-base, sweep-056-launch2, ss33-fast (the four-client configuration with this process)
    "factorisation_seconds": 1.3,
    "device_used_by_loaded_run_gb": 2.62,
    "hours_to_3_transits_at_plateau_load": 12.0,      # 6.0 M steps x 7.22 ms; the cost model's MPS-4 projection is 30.6 h at 18.3 ms/step
    "note": ("measured in the FOUR-client configuration the run executes in (three preregistered runs + this process); faster than the launch-1 base measurement (13.07 ms/step "
             "under six-way sharing, pre-v2.0.5 code) and than the cost model, so by the budget rule (the measured rate may only RAISE the budget) the cost-model MPS-4 rate "
             "stays the basis: 46.0 h = 3.8x the measured 3-transit wall"),
}
LAUNCH_BOX_TIMINGS: dict[str, dict[str, Any] | None] = {"base": LAUNCH_BOX_TIMING, "bohm-0.4": LAUNCH_BOX_TIMING_BOHM}

AMENDMENTS: list[dict[str, Any]] = [
    {
        "version": "v0 amendment 1 (bohm-0.4 launch 2: closure model v2.1.0, gates v2.0.6)",
        "kind": "closure_event_model_and_gate_version_before_first_execution",
        "utc": "2026-09-04T17:30:00Z",
        "option": "channel-20um-bohm-0.4",
        "reason": ("Launch 1 (the base option, no anomalous transport) stopped at 0.52 transits on genuine finite-grid heating: under the v1.3 closure at 2e20 static neutrals the "
                   "discharge avalanched (S = 2.5x the feed, inventory doubling 0.24 us, corrected residual +61.7 % at the stop) - README section 10 concluded that the one "
                   "sealed option that can plausibly reach a resolvable plateau at 20 um is the bohm-0.4 variant (the reference's own D_perp coefficient confines less), after "
                   "(i) the v2.0.6 ledger fix and (ii) the accumulated-particle-step Debye floor. Both landed (4b53012d, 8c70cff0). The physics completeness audit (0901138a, "
                   "section 4.c) further found that the sealed variant used the v1.4 ISOTROPIC redirect, which also randomises v_parallel, whereas Brandt et al. 2016 rotate only "
                   "the perpendicular velocity - so the sealed run was a bracket of the reference's model, not the model; model v2.1.0 (cft_revival.pic2d.sensitivity, "
                   "tests/pic2d/test_pic2d_v210_anomalous_transport.py) implements the reference's event (bohm_perpendicular_rotation) and verifies D_perp = (kT_e/eB) "
                   "alpha/(1+alpha^2) on both backends. For a code-to-code comparison the reference's own model is "
                   "the right closure to seal; alpha stays 0.4 (nu_an = 0.4 omega_ce: the D_perp coefficient read as a rate, the natural mapping of a selection probability that "
                   "depends on |B| only, Pb_237; exact Green-Kubo factor 0.345 disclosed as before)."),
        "changes": [
            "numerics.anomalous_collisions gains model = bohm_perpendicular_rotation (alpha 0.4 unchanged); the alpha_note states the reference's event model and the exact factor",
            "numerics.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak = 64000 (model v2.0.6; the near-axis column that launch 1 densified past pi unresolved is now gate-able)",
            ("the energy ledger is the v2.0.6 W-corrected one (code; no protocol key): acceptance (b) and the 5 % residual-power gate read the corrected statistic natively (launch 1 read "
             "the biased one: recorded +7.4 % = corrected +61.7 %)"),
            "numerics.performance.moment_sample_interval = 5 (v2.0.5; physics bitwise, declared identity)",
            "stopping_rule.wall_budget_seconds from 1.5 x max(cost model, the bohm option's OWN launch-box plateau-load rate) (preflight-channel-20um-bohm-0.4.json)",
            "LAUNCH_SET = (bohm-0.4, 20um); the base option's one execution is on record (LAUNCH_HISTORY / results/channel-20um-launch1-triad-gate-stop/) and is not relaunched",
            ("model_version / classification / case.id / simplifications text of the bohm-0.4 sealed protocol name v2.1.0 transport + v2.0.6 gates; its sha256 in "
             "preregistration.sealed_run_protocols changes accordingly; the base and 15um sealed protocols are byte-identical to the preregistration commit 3dc12cf6"),
            "scheduler job ext-val-v0-channel-20um-bohm-0.4 (tools/cloud/jobs.yaml); records preflight-channel-20um-bohm-0.4.json + shakedown-channel-20um-bohm-0.4.json",
        ],
        "unchanged": ("grid (75 x 700 at 20 um), dt 0.7 ps, W 82 466.8 (the 12 M-particle cap: parity would be 103 M particles, beyond the cap - README section 10 names the parity weight "
                      "as the alternative to a weaker closure, not a lower W within the cap; the 8.6x parity weight stays a declared limit), seed 20260903, operating point (400 V, static "
                      "Xe 2e20 / 500 K, 1.8 mA / 1 eV), frames, the v2.0.3 gate thresholds (hard pi / soft 2.5, 5 % residual power, triad drift bounds with their 1.0-transit arming - "
                      "'nothing about the drift members' arming needs to change', README section 10), the v2.0.4 omega_pe dt statistic, the plateau rule, acceptance (a)-(e), the "
                      "comparison spec (byte-identical) and its inconclusiveness conditions"),
        "expected_outcome": ("declared inconclusiveness stays: the 20 um / W 82 466.8 envelope (hard pi at 1.36e19 x T_e/10 eV) may still be reached if the closure does not bound n_e "
                             "enough; the discriminating outcomes are (i) a plateau under the reference's closure with I_a toward 4.3 mA and n_i toward 1e19 (rows inside V&V20 "
                             "tolerance or recorded misses), or (ii) another heating / envelope stop, which would put the remaining difference on the SEE / neutral-profile / W-parity "
                             "side (audit R2, R5a)"),
    },
]
EXPERIMENT_PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.json"
COMPARISON_SPEC_PATH = EXPERIMENT_DIR / "comparison-spec.json"

GRIDS: dict[str, dict[str, float]] = {
    "20um": {"cell_m": 20.0e-6, "dt_s": 0.7e-12},          # the published resolution (primary)
    "33um": {"cell_m": 14.0e-3 / 420.0, "dt_s": 0.7e-12},  # the ss-v4 option (inadmissible a priori at the published density; kept for the argument / cost table)
    "15um": {"cell_m": 15.0e-6, "dt_s": 0.5e-12},          # the resolution follow-up (soft margin met at 1e19 / 10 eV; dt for omega_pe dt <= 0.2 at 5e19)
}
PRIMARY_GRID = "20um"
VARIANTS: dict[str, dict[str, Any]] = {
    "base": {"anomalous_alpha": None, "role": "PRIMARY: the accepted v1.3 closure (no anomalous transport, no SEE); the closure difference to the reference is declared"},
    "bohm-0.4": {"anomalous_alpha": 0.4, "anomalous_model": "bohm_perpendicular_rotation",
                 "role": ("DISCRIMINATING RUN (amendment 1, launch 2 of the campaign): model v2.1.0 Bohm-type closure with the reference's own event model - the perpendicular velocity "
                          "rotated about the local B by a random angle (v_parallel and |v| unchanged; Brandt et al. 2016 p. Pb_237) at nu_an = 0.4 omega_ce = the reference's D_perp "
                          "coefficient 0.4 k T_e / e B read as a rate (exact Green-Kubo factor alpha / (1 + alpha^2) = 0.345 of k T_e / e B; both readings recorded). Sealed at the "
                          "preregistration as the v1.4 isotropic hook (a bracket of the model); amended to the reference's model before its first execution")},
}
PRIMARY_VARIANT = "base"
# amendment 1 (2026-09-05): launch 1 = the base option (STOPPED, genuine heating, recorded under results/channel-20um-launch1-triad-gate-stop/; README section 10:
# "no launch 2 at 20 um" for the base); launch 2 = the sealed bohm-0.4 option brought to model v2.0.6 + the v2.1.0 rotation closure. The launch set names
# what `launch` may execute NOW; the history keeps the base's one execution on record.
LAUNCH_SET = (("bohm-0.4", "20um"),)
LAUNCH_HISTORY: dict[str, str] = {
    "channel-20um": ("launch 1 (PID 31588, 12:26:44-13:56:18 UTC 2026-09-04, prereg 3dc12cf6): STOPPED by the windowed residual-power gate at 0.52 transits - genuine "
                     "finite-grid heating under the v1.3 closure at W 82 466.8 (README section 10; corrected residual +61.7 % at the stop, section 11); INCONCLUSIVE; "
                     "no launch 2 of the base option at 20 um (the README's own rule)"),
}
ANOMALOUS_MODEL_ROTATION = "bohm_perpendicular_rotation"

ANODE_POTENTIAL_V = 400.0
NEUTRAL_DENSITY_PER_M3 = 2.0e20
NEUTRAL_TEMPERATURE_K = 500.0
INJECTION_CURRENT_A = 1.8e-3
INJECTION_TEMPERATURE_EV = 1.0
STABILITY_REFERENCE = {"density_per_m3": 1.0e19, "electron_temperature_ev": 10.0, "max_electron_energy_ev": 400.0}
EXPECTED_PEAK_DENSITY_PER_M3 = 2.5e19           # dt policy: omega_pe dt <= 0.2 here (2.5x the published typical density); NOTE the peak-Debye hard gate binds first on 20 um:
                                                # pi cells / lambda_D at 10 eV is reached at 1.36e19 (protocol.density_at_cells_per_debye) - the resolvable envelope of the primary grid
EXPECTED_MEAN_DENSITY_PER_M3 = 5.0e18           # particle projection: half the published typical density over the channel volume (declared)
MAX_PROJECTED_PARTICLES_M = 12.0                # the H100 80 GB cap of the mini-sweep
BUDGET_FACTOR = 1.5
FRAME_CADENCE_STEPS = 40000                     # 28 ns at 0.7 ps = the template's frame TIME cadence
TRANSIT_S = sweep_cost.REFERENCE_CHANNEL_RESIDENCE_S * geometry_module.CHANNEL_LENGTH_M / sweep_cost.REFERENCE_CHANNEL_LENGTH_M   # 1.4 us
SEED = 20260903
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
EPSILON_0 = 8.8541878128e-12


def load_template() -> dict[str, Any]:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def option_tag(variant: str, grid: str) -> str:
    return f"channel-{grid}" + ("" if variant == "base" else f"-{variant}")


def protocol_bytes(protocol: dict[str, Any]) -> bytes:
    return json.dumps(protocol, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"


def composed_protocol_path(variant: str = PRIMARY_VARIANT, grid: str = PRIMARY_GRID, *, root: Path = PROTOCOLS_DIR) -> Path:
    return root / f"{CONFIG_ID}-{option_tag(variant, grid)}.json"


# --------------------------------------------------------------------------------------------------------------------------
# a-priori numbers the composition records
# --------------------------------------------------------------------------------------------------------------------------


def omega_pe(density_per_m3: float) -> float:
    return math.sqrt(density_per_m3 * ELEMENTARY_CHARGE_C**2 / (EPSILON_0 * ELECTRON_MASS_KG))


def debye_length_m(density_per_m3: float, temperature_ev: float) -> float:
    return math.sqrt(EPSILON_0 * temperature_ev / (density_per_m3 * ELEMENTARY_CHARGE_C))


def density_at_omega_pe_dt(limit: float, dt_s: float) -> float:
    return (limit / dt_s) ** 2 * EPSILON_0 * ELECTRON_MASS_KG / ELEMENTARY_CHARGE_C**2


def density_at_cells_per_debye(ratio: float, cell_m: float, temperature_ev: float) -> float:
    return EPSILON_0 * temperature_ev / ELEMENTARY_CHARGE_C * (ratio / cell_m) ** 2


def grid_argument() -> dict[str, Any]:
    """Why 20 um: the three candidate grids against the published density (1e19 / 10 eV) and the v2.0.3 gate levels."""

    rows = []
    for name, grid in GRIDS.items():
        cell, dt = grid["cell_m"], grid["dt_s"]
        ld = debye_length_m(STABILITY_REFERENCE["density_per_m3"], STABILITY_REFERENCE["electron_temperature_ev"])
        mapping = pic_mapping("channel", target_cell_m=cell)
        rows.append({
            "grid": name, "cell_m": cell, "dt_s": dt, "cells": [mapping.grid.radial_cells, mapping.grid.axial_cells], "nodes": list(mapping.grid.node_shape),
            "cells_per_debye_at_published": max(mapping.grid.dr_m, mapping.grid.dz_m) / ld,
            "admissible_hard_pi": max(mapping.grid.dr_m, mapping.grid.dz_m) / ld <= math.pi, "soft_2p5_met": max(mapping.grid.dr_m, mapping.grid.dz_m) / ld <= 2.5,
            "hard_gate_density_at_10ev_per_m3": density_at_cells_per_debye(math.pi, max(mapping.grid.dr_m, mapping.grid.dz_m), STABILITY_REFERENCE["electron_temperature_ev"]),
            "soft_level_density_at_10ev_per_m3": density_at_cells_per_debye(2.5, max(mapping.grid.dr_m, mapping.grid.dz_m), STABILITY_REFERENCE["electron_temperature_ev"]),
            "omega_pe_dt_at_published": omega_pe(STABILITY_REFERENCE["density_per_m3"]) * dt, "omega_pe_dt_gate_density_per_m3": density_at_omega_pe_dt(0.2, dt),
            "electron_courant_400ev": math.sqrt(2.0 * 400.0 * ELEMENTARY_CHARGE_C / ELECTRON_MASS_KG) * dt / min(mapping.grid.dr_m, mapping.grid.dz_m),
            "relative_cell_count": (mapping.grid.radial_cells * mapping.grid.axial_cells) / (75 * 700),
            "relative_steps_to_3_transits": (0.7e-12 / dt),
        })
    return {
        "published_density_per_m3": STABILITY_REFERENCE["density_per_m3"], "published_temperature_ev": STABILITY_REFERENCE["electron_temperature_ev"],
        "debye_length_at_published_m": debye_length_m(STABILITY_REFERENCE["density_per_m3"], STABILITY_REFERENCE["electron_temperature_ev"]),
        "rows": rows,
        "decision": (
            "PRIMARY = 20 um (the published resolution; 75 x 700 channel cells = the reference's channel block exactly, 1024 x 256 = its whole box): at the published density and "
            "temperature it sits at 2.69 cells / lambda_D - under the v2.0.3 HARD gate (pi) and over the SOFT plateau precondition (2.5) - so a plateau at exactly the published "
            "density would be recorded as 'resolution margin not met' but the code-to-code numbers exist at the same grid as the reference. The ss-v4 option (33.3 um) is "
            "INADMISSIBLE a priori: 4.48 cells / lambda_D at the published density, i.e. our own hard gate stops the run once the peak reaches ~5e18 at 10 eV; a comparison "
            "with the reference could never form. 15 um / 0.5 ps meets the soft margin (2.02) at the published density and is the declared resolution follow-up at 1.8x the cells "
            "and 1.4x the steps (2.5x the cost)."
        ),
        "reference_scaling_note": ("the reference's own 20 um is the ORIGINAL-SYSTEM value of a self-similar scaled run (factor 4 / 8): in its scaled plasma the cell was ~1 lambda_D "
                                   "(thesis: dr / lambda_D = 0.95); at device scale the same 20 um is 2.7 lambda_D at 1e19 / 10 eV - the reference resolved its Debye length by "
                                   "scaling, we resolve ours by the gate; neither run is 'better resolved' than the other in its own frame (closure_differences)"),
    }


def macro_weight_policy(mapping: PicMapping) -> tuple[float, dict[str, Any]]:
    """Parity W (6e4 x dr dz / (50 um)^2) unless the projected count at the declared mean density exceeds the 12 M cap; then W raised (disclosed)."""

    parity = sweep_cost.parity_macro_weight(mapping)
    volume = channel_volume_m3(mapping)
    physical = 2.0 * EXPECTED_MEAN_DENSITY_PER_M3 * volume           # electrons + ions
    projected_m = physical / parity / 1e6
    if projected_m <= MAX_PROJECTED_PARTICLES_M:
        weight = parity
        rule = f"parity W = 6e4 x dr dz / (50 um)^2 = {parity:.6g}"
    else:
        weight = float(round(parity * projected_m / MAX_PROJECTED_PARTICLES_M, 1))
        rule = (f"W raised from the parity value {parity:.6g} to cap the projected count at {MAX_PROJECTED_PARTICLES_M:g} M ({projected_m:.1f} M projected at parity for the declared mean "
                f"density {EXPECTED_MEAN_DENSITY_PER_M3:.1e} m^-3 over the channel volume {volume:.3e} m^3)")
    cell_volume_mid = 2.0 * math.pi * 0.5 * mapping.geometry.bore_radius_m * mapping.grid.dr_m * mapping.grid.dz_m
    return weight, {"rule": rule, "parity_weight": parity, "projected_total_m_at_parity": projected_m, "projected_total_m": min(projected_m, MAX_PROJECTED_PARTICLES_M),
                    "expected_mean_density_per_m3": EXPECTED_MEAN_DENSITY_PER_M3, "channel_volume_m3": volume,
                    "macro_electrons_per_cell_at_mid_radius_at_published_density": STABILITY_REFERENCE["density_per_m3"] * cell_volume_mid / weight,
                    "reference_macro_ratio_original_units": {"paper_factor_4": 2618 * 16, "thesis_factor_8": 2618 * 64,
                                                             "note": "the reference's 1:2618 applies in its scaled system; x s^2 in original-system units"}}


def admissible_dt(dt_s: float, max_b_t: float, limit: float) -> tuple[float, dict[str, Any]]:
    """The composed dt unless omega_ce dt at the map's max |B| would exceed 0.95 x the gate (then the largest 0.1 ps multiple below it)."""

    omega_ce = ELEMENTARY_CHARGE_C * float(max_b_t) / ELECTRON_MASS_KG
    at = omega_ce * dt_s
    if at <= 0.95 * limit:
        return float(dt_s), {"rule": f"composed dt kept: omega_ce dt {at:.3f} <= 0.95 x {limit}", "omega_ce_dt": float(f"{at:.9g}"), "max_b_t": float(f"{max_b_t:.9g}")}
    dt = math.floor(0.95 * limit / omega_ce / 1.0e-13) * 1.0e-13
    return float(dt), {"rule": "dt reduced so omega_ce dt <= 0.95 x gate at the map's max |B|, 0.1 ps quantum", "omega_ce_dt_at_composed": float(f"{at:.9g}"), "omega_ce_dt": float(f"{omega_ce*dt:.9g}"),
                       "max_b_t": float(f"{max_b_t:.9g}")}


def cost_row(mapping: PicMapping, *, dt_s: float, macro_weight: float, projected_total_m: float) -> dict[str, Any]:
    """ms/step on the 5090 model and on one of four H100 MPS slots (the mini-sweep cost module scaled by cells + particles), steps and hours to 3 transits."""

    nodes = mapping.grid.node_shape
    fixed = sweep_cost.fixed_ms_per_step(nodes)
    ms_5090 = fixed["fixed_ms"] + sweep_cost.PARTICLE_SLOPE_MS_PER_M * projected_total_m
    ms_mps4 = sweep_cost.h100_mps4_ms_per_step(nodes, projected_total_m)
    ms_solo = ms_mps4 * sweep_cost.H100_MPS4_ANCHOR["ms_per_step_solo"] / sweep_cost.H100_MPS4_ANCHOR["ms_per_step_per_process"]
    steps = 3.0 * TRANSIT_S / dt_s
    factorisation_s = sweep_cost.FACTORISATION_REFERENCE_S * (nodes[0] * nodes[1] ** 3) / sweep_cost.FACTORISATION_REFERENCE_ROWS_M3
    field_map_s = sweep_cost.FIELD_MAP_S_PER_NODE * nodes[0] * nodes[1]
    hours = steps * ms_mps4 / 3.6e6 + (factorisation_s + field_map_s) / 3600.0
    return {"nodes": list(nodes), "cells": [mapping.grid.radial_cells, mapping.grid.axial_cells], "dt_s": dt_s, "macro_weight": macro_weight, "particles_projected_m": projected_total_m,
            **fixed, "ms_per_step_rtx5090_model": ms_5090, "ms_per_step_h100_mps4_per_process": ms_mps4, "ms_per_step_h100_solo_equivalent": ms_solo, "platform": "h100-mps4",
            "transit_s": TRANSIT_S, "steps_to_3_transits": steps, "factorisation_s": factorisation_s, "field_map_s": field_map_s, "hours_to_3_transits_mps4": hours,
            "hours_to_3_transits_h100_solo_equivalent": steps * ms_solo / 3.6e6 + (factorisation_s + field_map_s) / 3600.0,
            "device_gb_projected": sweep_cost.DEVICE_GB_BASE + sweep_cost.DEVICE_GB_PER_M_PARTICLES * projected_total_m + fixed["inverse_blocks_gb"],
            "reference_run_steps": reference.SETUP["steps"]["value"]["total"], "reference_run_time_s": reference.SETUP["steps"]["value"]["quasi_steady_time_s"],
            "hours_to_reference_time_mps4": (reference.SETUP["steps"]["value"]["quasi_steady_time_s"] / dt_s) * ms_mps4 / 3.6e6,
            "cost_model": "experiments.pic2d_design_mini_sweep_v1.cost (5090 anchors; H100 MPS-4 anchor 8.71 ms/step for the v4 configuration; solo anchor 3.37) scaled by nodes and particles"}


def budget_basis_hours(cost: dict[str, Any], timing: dict[str, Any] | None = None) -> tuple[float, dict[str, Any]]:
    """Hours to 3 transits the wall budget is derived from: the cost-model MPS-4 rate, or the measured launch-box plateau-load rate when it is slower.

    The measured rate (``LAUNCH_BOX_TIMING``, from ``preflight --gpu-timing`` on the box under the MPS contention present at the time) can only
    RAISE the budget: the run speeds up as sweep slots free, so the faster of the two would understate the wall the plateau rule needs.
    """

    timing = LAUNCH_BOX_TIMING if timing is None else timing
    model_ms = float(cost["ms_per_step_h100_mps4_per_process"])
    overhead_h = (float(cost["factorisation_s"]) + float(cost["field_map_s"])) / 3600.0
    steps = float(cost["steps_to_3_transits"])
    measured_ms = None if not timing else timing.get("ms_per_step_at_plateau_load")
    if measured_ms is not None and float(measured_ms) > model_ms:
        ms = float(measured_ms)
        basis = {"basis": "measured launch-box plateau-load rate (slower than the cost model)", "ms_per_step": ms, "cost_model_ms_per_step": model_ms,
                 "measured_ms_per_step_at_plateau_load": float(measured_ms), "concurrent_mps_clients_at_measurement": timing.get("concurrent_mps_clients")}
    else:
        ms = model_ms
        basis = {"basis": "cost model (H100 MPS-4 anchor scaled by nodes + particles)", "ms_per_step": ms, "cost_model_ms_per_step": model_ms,
                 "measured_ms_per_step_at_plateau_load": None if measured_ms is None else float(measured_ms),
                 "concurrent_mps_clients_at_measurement": None if not timing else timing.get("concurrent_mps_clients")}
    return steps * ms / 3.6e6 + overhead_h, basis


# --------------------------------------------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------------------------------------------


def build_protocol(variant: str = PRIMARY_VARIANT, grid: str = PRIMARY_GRID, *, field_map=None) -> tuple[dict[str, Any], PicMapping]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known {tuple(VARIANTS)}")
    if grid not in GRIDS:
        raise ValueError(f"unknown grid {grid!r}; known {tuple(GRIDS)}")
    cell_m, dt_s = GRIDS[grid]["cell_m"], GRIDS[grid]["dt_s"]
    mapping = pic_mapping("channel", target_cell_m=cell_m)
    template = load_template()
    protocol = copy.deepcopy(template)
    budget_keys = [key for key in protocol if key.startswith("budget")]
    for key in budget_keys + ["field_authority", "reference_run", "preregistration", "design_id"]:
        protocol.pop(key, None)
    protocol["schema_version"] = RUN_PROTOCOL_SCHEMA
    protocol["experiment_id"] = EXPERIMENT_ID
    protocol["status"] = STATUS
    protocol["classification"] = "axisymmetric_electrostatic_pic_mcc_code_to_code_comparison_channel_only_static_neutrals_v1_3_closure_v2_0_3_gates_not_validated"
    protocol["model_version"] = ("pic2d v1.3 runner with a STATIC neutral background (the inventory removed = the v1.2 static-background mode; NO wall-ion recycling, no SEE, no anomalous "
                                 "transport unless the variant switches the v1.4 Bohm hook on) with the v2.0.3 gates (window-mode peak-Debye gate, windowed residual-power gate) and the "
                                 "v2.0.4 runtime omega_pe dt statistic (peak over nodes holding >= the peak-Debye floor of 32 macro-electrons; the raw single-node peak recorded alongside)")
    protocol["option"] = option_tag(variant, grid)
    protocol["variant"] = {"id": variant, **VARIANTS[variant]}
    protocol["template_protocol"] = {"path": TEMPLATE_PATH.relative_to(REPOSITORY).as_posix(), "experiment_id": template.get("experiment_id"), "model_version": template.get("model_version"),
                                     "sha256": hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()}
    protocol["design_id"] = CONFIG_ID
    protocol["reference"] = {"citation": reference.CITATION, "doi": reference.DOI, "thesis_urn": reference.THESIS["urn"], "role": "published PIC-MCC model output (code-to-code); not hardware"}
    protocol["field_binding"] = f"modern/experiments/pic2d_external_validation_v0/fields/{CONFIG_ID}/binding.json"
    protocol["field_source"] = ("external-validation-v0 binding: direct node evaluation of the reconstructed micro-HEMPT's padded level-0 material-aware P2 checkpoint at z_FEM = z + 2.5 mm, "
                                "scaled to the published axis anchor (source_strength_scale, gate G1)")
    protocol["geometry_mapping"] = {"approximations": [a["id"] for a in geometry_module.APPROXIMATIONS], "table": "protocol.json geometry_mapping (experiment level) / README section 2"}
    # geometry / grid
    geometry = mapping.geometry.to_dict()
    geometry["source"] = f"Brandt 2016 channel (Z_thr 14 mm, R_thr 1.5 mm) channel-only at {cell_m*1e6:.4g} um; every reference line on a grid line (snaps zero)"
    geometry["snaps"] = mapping.snaps
    geometry["axial_offset_m_field_frame"] = mapping.axial_offset_m
    protocol["geometry"] = geometry
    # macro weight and cost
    weight, weight_policy = macro_weight_policy(mapping)
    numerics = protocol["numerics"]
    numerics["dt_s"] = float(dt_s)
    dt_policy: dict[str, Any] = {"rule": f"composed {dt_s*1e12:.2f} ps: omega_pe dt = 0.2 at {density_at_omega_pe_dt(0.2, dt_s):.3g} m^-3 (declared expected peak {EXPECTED_PEAK_DENSITY_PER_M3:.1e}); "
                                         f"{omega_pe(STABILITY_REFERENCE['density_per_m3']) * dt_s:.3f} at the published 1e19"}
    if field_map is not None:
        dt_s, admitted = admissible_dt(dt_s, field_map.max_b_t, float(numerics["stability_limits"]["max_omega_ce_dt"]))
        numerics["dt_s"] = float(dt_s)
        dt_policy["omega_ce"] = admitted
    numerics["dt_policy"] = dt_policy
    numerics["dt_justification"] = (f"{dt_s*1e12:.2f} ps: the runtime omega_pe dt gate (0.2) trips at {density_at_omega_pe_dt(0.2, dt_s):.3g} m^-3 = {density_at_omega_pe_dt(0.2, dt_s)/STABILITY_REFERENCE['density_per_m3']:.1f}x "
                                    f"the published typical density (the reference's 3.17 ps is the original-system value of its factor-4/8 scaled run: omega_pe dt 0.2 in ITS frame, 0.56 in ours); "
                                    f"electron Courant at 400 eV {math.sqrt(2*400*ELEMENTARY_CHARGE_C/ELECTRON_MASS_KG)*dt_s/min(mapping.grid.dr_m, mapping.grid.dz_m):.2f} cells; omega_ce dt at 0.7 T "
                                    f"{ELEMENTARY_CHARGE_C*0.7/ELECTRON_MASS_KG*dt_s:.3f}. Step-count conversions: 400000-step window = {400000*dt_s*1e6:.3f} us, 40000-step checkpoint / frame = "
                                    f"{40000*dt_s*1e9:.1f} ns, one transit ({TRANSIT_S*1e6:.2f} us) = {round(TRANSIT_S/dt_s):,} steps")
    numerics["stability_reference"] = dict(STABILITY_REFERENCE)
    numerics["stability_reference_note"] = "the PUBLISHED typical density and the thesis' estimated electron temperature (Brandt 2016 Pb_240; thesis ch. 7); anode 400 V sets the maximum electron energy"
    numerics["stability_limits"]["max_cell_debye_ratio"] = math.pi
    numerics["stability_limits_note"] = ("max_cell_debye_ratio raised from the template's 2.0 to pi: the template value was its own reference-density ratio (33 um at 4e17 / 8 eV = 1.0 cell per lambda_D); "
                                         "here the a-priori check is made at the PUBLISHED density (2.69 cells / lambda_D on 20 um) against the v2.0.3 HARD level pi, and the runtime window-mode gate "
                                         "(hard pi / soft 2.5 on the interval-averaged peak) is the protective one (v2.0.3 lesson). Every other limit is the template's")
    numerics["frame_recorder"] = {"cadence_steps": FRAME_CADENCE_STEPS, "precision": "float32"}
    numerics["frame_recorder_note"] = (f"frame recorder ON: {FRAME_CADENCE_STEPS}-step frames = {FRAME_CADENCE_STEPS*dt_s*1e9:.1f} ns exact interval averages (the template's 28 ns time cadence at half "
                                       f"the time step); 3 transits = {int(3*TRANSIT_S/dt_s/FRAME_CADENCE_STEPS)} frames on the {mapping.grid.node_shape[0]} x {mapping.grid.node_shape[1]} node grid; "
                                       "frames are diagnostics (ionisation structure at the cusp planes, the potential staircase) and feed the comparison's qualitative rows")
    numerics["peak_debye_gate"]["max_cells_per_debye_note"] = (
        "v2.0.3 window-mode gate verbatim (hard pi fail-closed once the window is complete; soft 2.5 = plateau precondition). At the PUBLISHED density / temperature (1e19 / 10 eV) the 20 um grid "
        f"reads {20e-6/debye_length_m(1e19, 10.0):.2f}: a plateau at exactly the published state is recorded as 'resolution margin not met' (acceptance (a) fails on the soft precondition, the run "
        f"continues to the budget, the comparison rows carry the flag); the hard level is reached at {density_at_cells_per_debye(math.pi, 20e-6, 10.0):.3g} m^-3 at 10 eV")
    amended = VARIANTS[variant].get("anomalous_model") is not None       # amendment 1: the bohm-0.4 option carries v2.1.0 transport + v2.0.6 gates
    if VARIANTS[variant]["anomalous_alpha"] is not None:
        alpha = float(VARIANTS[variant]["anomalous_alpha"])
        if amended:
            numerics["anomalous_collisions"] = {
                "model": VARIANTS[variant]["anomalous_model"], "alpha": alpha,
                "alpha_note": (f"model v2.1.0 Bohm-type anomalous transport with the reference's event model (Brandt et al. 2016, Pb_237): every electron has its velocity rotated about the "
                               f"local B by a uniform random angle - v_parallel and |v| unchanged to round-off, gyro-centre shifted - with probability 1 - exp(-alpha omega_ce dt) per step at "
                               f"the particle's |B|; nu_an = {alpha:g} omega_ce = the reference's D_perp = 0.4 k T_e / e B read as a rate (a selection probability that depends on |B| only, "
                               f"as the reference describes); exact Green-Kubo D_perp = (k T_e / e B) alpha / (1 + alpha^2) = {alpha / (1 + alpha**2):.3f} k T_e / e B (verified on both "
                               f"backends, tests/pic2d/test_pic2d_v210_anomalous_transport.py). At 0.7 T nu_an = {alpha * 1.7588e11 * 0.7:.3g} s^-1, nu_an dt = "
                               f"{alpha * 1.7588e11 * 0.7 * float(numerics['dt_s']):.3f}. Elastic (no ledger energy term), count tallied (cumulative.anomalous), axial momentum in "
                               f"pz_collisions; outside the MCC null-collision budget as a separate exact-Poisson process (amendment 1)"),
            }
        else:
            numerics["anomalous_collisions"] = {"alpha": alpha,
                                                "alpha_note": "v1.4 Bohm-scattering hook: isotropic redirect at nu_an = alpha omega_ce; alpha 0.4 reproduces the reference's D_perp coefficient 0.4 k T_e / e B "
                                                              "in the small-alpha identity (exact factor 0.345); it also randomises the parallel speed, which the reference's perpendicular-rotation model "
                                                              "does not - a sensitivity bracket, not the reference's model"}
    else:
        numerics.pop("anomalous_collisions", None)
    if amended:
        numerics["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak"] = 64000
        numerics["peak_debye_gate"]["min_accumulated_macro_particle_steps_at_peak_note"] = (
            "model v2.0.6 (spec gates_v2_0.peak_debye_gate_accumulated_floor_v2_0_6; amendment 1): the gated node is the densest node whose ACCUMULATED macro-electron-steps over the "
            "400 000-step window reach 64 000, so the near-axis column launch 1 densified past pi while the 32-macro-electron mean-occupancy floor hid it (README section 10/11: "
            "the axis node held 0.72 macro-electrons per step and 172 000 macro-electron-steps) is gate-able; the mean-occupancy floor stays recorded alongside")
        numerics["performance"] = {"moment_sample_interval": 5,
                                   "moment_sample_interval_note": "v2.0.5 (amendment 1): electron window moments sampled every 5th accumulated step; physics bitwise, enters config_sha256 by the "
                                                                  "v2.0.5 identity policy; launch 1 (base option) ran K = 1"}
        protocol["model_version"] = ("pic2d v1.3 runner with a STATIC neutral background (the inventory removed = the v1.2 static-background mode; NO wall-ion recycling, no SEE) + model "
                                     "v2.1.0 anomalous cross-field transport (Bohm-type perpendicular-velocity rotation, nu_an = alpha omega_ce, Brandt et al. 2016 event model) with the "
                                     "v2.0.6 gates (window-mode peak-Debye gate with the accumulated-particle-step floor, windowed residual-power gate on the W-corrected ledger), the "
                                     "v2.0.4 runtime omega_pe dt statistic and the v2.0.5 K = 5 moment sampling (amendment 1)")
        protocol["classification"] = "axisymmetric_electrostatic_pic_mcc_code_to_code_comparison_channel_only_static_neutrals_v1_3_closure_v2_1_0_bohm_rotation_v2_0_6_gates_not_validated"
        protocol["amendments"] = copy.deepcopy(AMENDMENTS)
        protocol["launch_history"] = dict(LAUNCH_HISTORY)
    # operating point
    operating = protocol["operating_point"]
    operating["anode_potential_v"] = ANODE_POTENTIAL_V
    operating["exit_plane_potential_v"] = 0.0
    operating["neutral_density_per_m3"] = NEUTRAL_DENSITY_PER_M3
    operating["neutral_density_role"] = "STATIC uniform background (v1.2 mode: no inventory): the MCC density is this constant, as the reference's static DSMC import (mean 'about 2e20 m^-3')"
    operating["neutral_temperature_k"] = NEUTRAL_TEMPERATURE_K
    operating.pop("neutral_inventory", None)
    operating["neutral_inventory"] = None
    operating["neutral_inventory_note"] = ("REMOVED (null): the reference keeps its neutrals static and neglects the 25 % depletion; the v1.3 inventory would move the operating point to its own fixed "
                                           "point and confound the comparison. Under static neutrals gross = net utilisation is undefined; Brandt's 'net ionisation' is I_a / (e Q_in) with Q_in = 1.1e17 /s "
                                           "(comparison spec)")
    operating["electron_injection_current_a"] = INJECTION_CURRENT_A
    operating["electron_injection_temperature_ev"] = INJECTION_TEMPERATURE_EV
    operating["electron_injection_justification"] = ("1.8 mA at 1 eV: the reference's effective electron source into the channel by continuity (I_anode - I_beam = 4.3 - 2.5 mA; its own estimate 'a third "
                                                     "of 5.85 mA' = 1.95 mA) at its source temperature (1 eV). TAKEN FROM THE REFERENCE'S RESULTS: the anode-current row is conditioned on it "
                                                     "(comparison spec); the template's 3 mA / 2 eV is our operating point, not theirs")
    operating["seed_justification"] = ("the template's seed plasma (5e16 m^-3, 5 eV, cold ions) unchanged: at 2e20 background and 0.6 T confinement ignition is expected to be faster than in the "
                                       "accepted runs; the reference seeded a whole plasma column plus an exit electron source for 1.5e6 steps - a different transient, the same steady state sought")
    operating["unchanged_note"] = ("operating point = the reference's: anode 400 V, static xenon 2e20 at 500 K, effective source 1.8 mA / 1 eV at the exit plane; exit plane 0 V (channel-only); "
                                   "seed plasma = the template's")
    operating["reference_feed_atoms_per_s"] = reference.SETUP["mass_flow"]["value"]["atoms_per_s"]
    # case
    projected_m = weight_policy["projected_total_m"]
    protocol["case"] = {
        "id": f"{CONFIG_ID}-{option_tag(variant, grid)}-w{weight:.6g}-ng2e20-static-inj1.8mA-1eV-v1.3-closure-" + ("v2.1.0-bohm-rotation-v2.0.6-gates" if amended else "v2.0.3-gates"),
        "radial_cells": int(mapping.grid.radial_cells), "axial_cells": int(mapping.grid.axial_cells), "macro_weight": weight, "macro_weight_policy": weight_policy, "seed": SEED,
        "grid_policy": {"target_cell_m": cell_m, "dt_s": float(numerics["dt_s"]), "source": f"grid option {grid}: " + ("the published resolution" if grid == "20um" else "see grid_argument")},
        "grid_note": f"dr {mapping.grid.dr_m*1e6:.3f} um x dz {mapping.grid.dz_m*1e6:.3f} um: {mapping.grid.radial_cells} x {mapping.grid.axial_cells} cells, nodes {list(mapping.grid.node_shape)}",
    }
    # stopping rule / acceptance / budget
    cost = cost_row(mapping, dt_s=float(numerics["dt_s"]), macro_weight=weight, projected_total_m=projected_m)
    rule = protocol["stopping_rule"]
    # amendment 1: an amended option's budget reads its OWN launch-box timing (the bohm kernel adds a per-electron pass); the base keeps its record
    timing = LAUNCH_BOX_TIMINGS.get(variant) if amended else LAUNCH_BOX_TIMING
    budget_hours, budget_basis = budget_basis_hours(cost, timing=timing if timing is not None else {})
    wall_budget = max(3600.0, math.ceil(BUDGET_FACTOR * budget_hours * 3600.0 / 600.0) * 600.0)
    rule["wall_budget_seconds"] = float(wall_budget)
    rule["wall_budget_basis"] = budget_basis
    rule["wall_budget_note"] = (f"{BUDGET_FACTOR} x the projected wall to 3 transits at one of four H100 CUDA-MPS slots ({budget_hours:.1f} h at {budget_basis['ms_per_step']:.1f} ms/step, basis "
                                f"'{budget_basis['basis']}'; cost model {cost['hours_to_3_transits_mps4']:.1f} h at {cost['ms_per_step_h100_mps4_per_process']:.1f} ms/step; "
                                f"{cost['steps_to_3_transits']/1e6:.2f} M steps; solo-H100 equivalent {cost['hours_to_3_transits_h100_solo_equivalent']:.1f} h), rounded up to 10 min; cumulative over resumes. "
                                f"The reference's 76 us would cost {cost['hours_to_reference_time_mps4']:.0f} h at this rate - not budgeted; the plateau rule decides")
    rule["plateau"] = (f"relative drift < 5 % over the trailing 20 % of the elapsed simulated time for the discharge current AND the plasma electron count (linear fit, drift = slope x window / |mean|; "
                       f"the neutral density is static and drops out of the rule), evaluated at every checkpoint; may only be declared after >= 3 ion transit times (3 x {TRANSIT_S*1e6:.2f} us = "
                       f"{3*TRANSIT_S*1e6:.2f} us = {round(3*TRANSIT_S/float(numerics['dt_s'])):,} steps); additionally the grid-heating triad must be inside its soft bounds AND the window-mode "
                       "peak-Debye soft margin must hold (<= 2.5) - the ss-v4 rule verbatim")
    rule["ignition_check"] = ("S growing and N_e rising within ~0.5 us at 2e20 / 400 V / 0.6 T is expected (the reference ignited from a seeded column); a non-ignition under the frozen seed and "
                              "injection is a recorded outcome, not a reason to adjust (comparison-spec inconclusive_conditions)")
    rule["acceptance"] = {
        "declared": "predeclared in this DRAFT; evaluated by `run.py assess` and `run.py compare` on the trailing-window quantities",
        "a_plateau": f"stop_reason == plateau_reached_after_min_transit_times under stopping_rule.plateau (>= 3 x {TRANSIT_S*1e6:.2f} us, drifts of I_d and N_e < 5 %, triad soft, peak-Debye soft 2.5)",
        "b_residual_power": "summary.grid_heating_triad.windowed_energy_residual_over_electrode_work < +0.02 (one-sided) at the stop",
        "c_comparison": "every row of comparison-spec.json with comparable_under containing 'channel' evaluated with S from the trailing window (E, u_val, statement); rows needing the plume box are reported as not comparable",
        "d_verdicts": {
            "comparison_quotable": "(a) AND (b): E +- u_val per row is quotable as CROSS-MODEL agreement/disagreement under the declared closure differences",
            "comparison_resolution_flagged": "(a) fails ONLY on the peak-Debye soft precondition (plateau drifts met, peak between 2.5 and pi) AND (b): E reported with 'resolution margin not met' on every row",
            "plateau_with_heating": "(a) but NOT (b): E reported, not quotable",
            "no_plateau": "NOT (a) otherwise: inconclusive within the budget; the trailing-window quantities are reported as transient",
        },
        "e_claim_ceiling": comparison.CLAIM_CEILING + ": a published model output can support cross-model agreement at most; nothing here opens GATE-L3 or validates against hardware",
    }
    hard_debye_density = density_at_cells_per_debye(math.pi, max(mapping.grid.dr_m, mapping.grid.dz_m), STABILITY_REFERENCE["electron_temperature_ev"])
    protocol["budget_external_validation_v0"] = {
        "n_max_per_m3": hard_debye_density, "n_eq_projected_per_m3": STABILITY_REFERENCE["density_per_m3"],
        "n_max_note": (f"n_max = the density at which this grid reaches the v2.0.3 HARD peak-Debye level pi at the published 10 eV ({hard_debye_density:.3g} m^-3 = "
                       f"{hard_debye_density/STABILITY_REFERENCE['density_per_m3']:.2f}x the published typical density; scales with T_e): the resolvable envelope of the run. The omega_pe dt gate at "
                       f"{float(numerics['dt_s'])*1e12:.2f} ps trips later, at {density_at_omega_pe_dt(0.2, float(numerics['dt_s'])):.3g} m^-3. n_eq_projected = the reference's published typical density"),
        "expected_peak_density_dt_policy_per_m3": EXPECTED_PEAK_DENSITY_PER_M3,
        "ion_transit_time_s": TRANSIT_S, "ion_transit_note": "2.4 us x 14 mm / 24 mm (the measured reference residence scaled with the channel length); superseded by the measured N_i / L residence",
        "expected_mean_density_per_m3": EXPECTED_MEAN_DENSITY_PER_M3, "particles_projected_m": projected_m, "macro_weight": weight, "macro_weight_parity": weight_policy["parity_weight"],
        "dr_m": mapping.grid.dr_m, "dz_m": mapping.grid.dz_m, "dt_s": float(numerics["dt_s"]), "wall_budget_factor": BUDGET_FACTOR, **{k: v for k, v in cost.items() if k not in ("nodes", "cells", "dt_s", "macro_weight")},
        "launch_box_timing": timing if amended else LAUNCH_BOX_TIMING,
    }
    protocol["execution"] = {
        **LAUNCH_BOX,
        "launch_box_timing": timing if amended else LAUNCH_BOX_TIMING,
        "scheduler": ("modern/tools/cloud/schedule.py (tmux, detached worktree at the preregistration commit, per-job results directory, Warp cuda:0 UUID cross-check, prereg "
                      "ancestor + byte-identical protocol checks) with slots_per_gpu 4 and the MPS variables exported; job "
                      + (f"`ext-val-v0-{option_tag(variant, grid)}`" if amended else "`ext-val-v0-channel-20um`") + " in tools/cloud/jobs.yaml"),
        "launch_discipline": ("run.py launch --expect-commit <prereg sha> --require-mps: HEAD == prereg commit, clean worktree, protocol.json and protocols/<option>.json blobs == HEAD, "
                              "recomposed protocol == sealed file byte for byte, preflight-channel-20um.json (whole set passed + launch-box GPU timing passed) and "
                              "shakedown-channel-20um.json (passed) present, O_EXCL execution-lock.json in results/<option>/, MPS pipe directory present"),
        "one_execution": ("ONE detached launch from a worktree at the preregistration commit; a wall-budget stop may be resumed (--resume: new session, same identity, disclosed in "
                          "run_state.sessions); no parameter change after the freeze; a non-ignition or a gate stop is a recorded outcome (comparison-spec inconclusive_conditions)"),
    }
    protocol["simplifications"] = [s for s in protocol.get("simplifications", []) if not s.startswith(("neutrals:", "3 mA injection", "single seed and a single refined grid", "preregistered resolution-convergence"))] + [
        "neutrals: STATIC uniform xenon background 2e20 m^-3 at 500 K (the reference's static DSMC mean); no depletion, no profile (the reference's drops 6e20 -> 1e20 along the channel)",
        "electron source: 1.8 mA / 1 eV at the 0 V exit plane = the reference's continuity-derived effective source (an input taken from its results)",
        "channel-only box: the reference's 6.48 mm plume, its grounded body and its 0 V far boundaries are replaced by the Dirichlet exit plane (approximation A9); plume rows not compared",
        ("Bohm-type anomalous transport with the reference's perpendicular-rotation event model at nu_an = 0.4 omega_ce (D_perp = 0.345 k T_e / e B exact) - the reference's coefficient "
         "and event model, IMPOSED as a constant-alpha closure as the reference does; no SEE versus its 50 % / 90 % model" if amended else
         "no anomalous transport (primary) / isotropic Bohm scattering at alpha 0.4 (variant) versus the reference's perpendicular-rotation D_perp = 0.4 k T_e / e B; no SEE versus its 50 % / 90 % model"),
        "reconstructed field: level-0 material-aware P2 of the thesis' magnet stack with approximations A1-A8, scaled to the published axis anchor; not P2-qualified",
        "one grid (the published 20 um) and one seed: the grid caveat is carried, not measured; the 15 um sibling and a seed replicate are the declared follow-ups",
        ("one preregistered execution on one of four CUDA-MPS slots of a shared H100 (the design mini-sweep runs beside it): a code-to-code comparison of two development models, "
         "not a validation against hardware"),
    ]
    protocol["claim_boundary"] = {
        "preregistration": ("PREREGISTERED protocol composed on the steady-state v4 template for a code-to-code comparison with Brandt et al. 2016; ONE execution from the preregistration "
                            "commit (run.py launch --expect-commit, tools/cloud/schedule.py); the acceptance, the comparison spec and the inconclusiveness conditions are frozen here"),
        "comparison": f"cross-model (claim ceiling {comparison.CLAIM_CEILING}); every row conditional on the unpropagated inputs (B scale, neutral profile, effective source); opens no physics level",
        "geometry_approximation": "approximations A1-A9 of experiments/pic2d_external_validation_v0/geometry.py; field gates G1-G7 of fields.py",
    }
    return protocol, mapping


def compose_run_protocol(variant: str = PRIMARY_VARIANT, grid: str = PRIMARY_GRID) -> tuple[dict[str, Any], PicMapping, Any]:
    """Field first, then the protocol (the dt rule reads the node field's maximum)."""

    from . import fields as field_module

    mapping = pic_mapping("channel", target_cell_m=GRIDS[grid]["cell_m"])
    binding = field_module.load_binding()
    field_map = field_module.brandt_field_map(mapping, binding)
    protocol, _ = build_protocol(variant, grid, field_map=field_map)
    return protocol, mapping, field_map


COMPOSED_OPTIONS: tuple[tuple[str, str], ...] = (("base", "20um"), ("bohm-0.4", "20um"), ("base", "15um"))   # primary, transport sensitivity, resolution follow-up


def compose_all(*, root: Path = PROTOCOLS_DIR, options: tuple[tuple[str, str], ...] = COMPOSED_OPTIONS, log=print) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for variant, grid in options:
        protocol, _, _ = compose_run_protocol(variant, grid)
        path = composed_protocol_path(variant, grid, root=root)
        data = protocol_bytes(protocol)
        path.write_bytes(data)
        written[path.relative_to(REPOSITORY).as_posix()] = hashlib.sha256(data).hexdigest()
        log(f"[compose] {path.name}: W {protocol['case']['macro_weight']:.6g}, dt {protocol['numerics']['dt_s']*1e12:.2f} ps, cells {protocol['case']['radial_cells']} x {protocol['case']['axial_cells']}, "
            f"budget {protocol['stopping_rule']['wall_budget_seconds']/3600:.1f} h, sha256 {written[path.relative_to(REPOSITORY).as_posix()][:12]}")
    return written


def _file_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    return {"path": path.relative_to(REPOSITORY).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


DECISIONS_VS_DRAFT: list[dict[str, str]] = [
    {"item": "launch box / GPU model", "draft": "H100 MPS-4 assumed for the cost row only; no GPU recorded in the run protocol",
     "preregistered": "NVIDIA H100 80GB HBM3 (Lambda gpu_1x_h100_sxm5, ubuntu@68.209.75.2, driver 580.105.08) under CUDA MPS, ONE of four slots, recorded in every sealed run "
                      "protocol (`execution`) with the launch discipline and the scheduler job",
     "why": "the mini-sweep record (291a9227) established that the GPU model, the MPS slot and the shared-GPU provenance caveats belong in the preregistration"},
    {"item": "slot disclosure", "draft": "launch when design 047's slot frees (~01:00 AEST 5 Sep) or solo after the sweep",
     "preregistered": "the run enters the slot design 056 freed when its run stopped on the grid-heating triad gate at 10:52 UTC 2026-09-04 (omega_pe dt drift 0.283 > 0.25 at 2.07 "
                      "transits; the sweep is NOT assessed here); three sweep runs (reference, 047, 009) are active at launch and finish during this run, so the per-process "
                      "rate rises from the measured MPS-4 value toward the solo rate",
     "why": "a slot became free earlier than planned; the launch is still a four-client configuration (never a fifth client)"},
    {"item": "shakedown GPU use", "draft": "a labelled shakedown on the launch box before the freeze (no slot named)",
     "preregistered": "the shakedown ran as the FOURTH MPS client (056's freed slot) for ~10-20 min while three sweep runs executed; disclosed in the launch log",
     "why": "the sweep's own preregistration budgets are MPS-4 upper bounds, so a fourth client costs them nothing beyond their declared configuration"},
    {"item": "wall budget", "draft": "46.0 h = 1.5 x the cost-model MPS-4 projection (30.6 h at 18.3 ms/step)",
     "preregistered": "1.5 x max(cost-model MPS-4 rate, MEASURED launch-box plateau-load rate) rounded up to 10 min (`stopping_rule.wall_budget_basis` names which one bound); "
                      "the measured rate comes from `preflight --gpu-timing` (>= 2000 production steps at the seed load and at the projected 12 M-particle plateau load)",
     "why": "a preflight timing at production load beats a cost-table extrapolation (v2.0.3 lesson); the measured rate may only raise the budget, never lower it"},
    {"item": "seed", "draft": "20260903 (the template's)", "preregistered": "20260903 unchanged", "why": "one seed; the seed replicate is a declared follow-up"},
    {"item": "frames", "draft": "ON, 40 000 steps = 28 ns", "preregistered": "unchanged", "why": "the qualitative rows of the comparison spec read the frames"},
    {"item": "gates", "draft": "v2.0.3 verbatim (window-mode peak-Debye hard pi / soft 2.5, windowed residual power 5 %, triad, omega_pe dt 0.2, Courant, Poisson)",
     "preregistered": "v2.0.3 thresholds unchanged; the runtime omega_pe dt STATISTIC is the v2.0.4 resolved-node peak (nodes whose single-step deposit holds >= 32 macro-electrons, the "
                      "peak-Debye gate's own floor; the raw single-node peak is recorded alongside as peak_omega_pe_dt_raw and feeds nothing)",
     "why": ("found by the launch-box preflight / shakedown (2026-09-04 11:39 UTC): at 20 um / W 82 467 one macro-electron on a small-volume axis node reads 1.3e19 m^-3 (omega_pe dt 0.14 at "
             "0.7 ps), two read 0.20 - the raw statistic stopped the 12 M-particle timing seed at 0.212 before its first step and read 5.5e18 in the shakedown at 60 000 electrons over "
             "53 000 nodes (mean 5e14): a shot-noise extreme value decided by the smallest node (the plume-boundary lesson), which would have ended the production run as a spurious "
             "'omega_pe dt stop' long before any physical densification. The window-mode peak-Debye hard gate (pi at 1.36e19 / 10 eV, interval-averaged, 32-particle floor) binds first "
             "on this grid and stays the protective density gate; the floored omega_pe dt gate is the fast-transient backstop on resolved nodes. Physics untouched (same-seed replay bitwise)")},
    {"item": "comparison spec", "draft": "12 rows (10 channel-comparable), tolerances 20 % / +-5 V / 0.3 dex, u_D + u_num predicted, u_input declared not propagated",
     "preregistered": "unchanged (comparison-spec.json byte-identical to the draft, see preregistration.records)", "why": "the estimands and tolerances were fixed before any run"},
    {"item": "launch stages", "draft": "`run.py launch` REFUSED unconditionally", "preregistered": "`launch --expect-commit --require-mps` with the mini-sweep discipline; `shakedown` writes "
     "shakedown-channel-20um.json (run -> assess -> compare -> re-finalize); `preflight --gpu-timing` records the launch-box rate", "why": "the v4 / mini-sweep preregistration discipline"},
]


def experiment_protocol_document(*, preflight_summary: dict[str, Any] | None = None, sealed: dict[str, str] | None = None, field_binding_summary: dict[str, Any] | None = None,
                                 shakedown_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    if sealed is None and PROTOCOLS_DIR.is_dir():
        sealed = {p.relative_to(REPOSITORY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROTOCOLS_DIR.glob("*.json"))}
    rows = {}
    for grid in GRIDS:
        mapping = pic_mapping("channel", target_cell_m=GRIDS[grid]["cell_m"])
        weight, policy = macro_weight_policy(mapping)
        rows[grid] = cost_row(mapping, dt_s=GRIDS[grid]["dt_s"], macro_weight=weight, projected_total_m=policy["projected_total_m"])
    plume = pic_mapping("plume-brandt")
    plume_weight, plume_policy = macro_weight_policy(plume)
    plume_particles = plume_policy["projected_total_m"] * 1.75          # the v2.0 plume model's channel factor (mini-sweep cost)
    plume_cost = cost_row(plume, dt_s=GRIDS["20um"]["dt_s"], macro_weight=plume_weight, projected_total_m=min(plume_particles, MAX_PROJECTED_PARTICLES_M))
    return {
        "schema_version": EXPERIMENT_PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "preregistration": {
            "state": ("PREREGISTERED: the launch set is sealed by this file (sha256 of every run protocol under protocols/, of the whole-set preflight with the launch-box GPU timing, of the "
                      "shakedown record, of the comparison spec and of the field binding). ONE execution of `channel-20um` from the commit that carries this file, through "
                      "`run.py launch --expect-commit <sha> --require-mps` (tools/cloud/schedule.py job `ext-val-v0-channel-20um`); the sensitivity variant and the 15 um follow-up are "
                      "sealed, not launched"),
            "launch_set": [option_tag(*o) for o in LAUNCH_SET],
            "launch_set_note": ("amendment 1: the launch set names what `launch` may execute now - the bohm-0.4 option (launch 2 of the campaign); the base option's one execution "
                                "(launch 1) is on record under launch_history and is not relaunched"),
            "launch_history": dict(LAUNCH_HISTORY),
            "sealed_run_protocols": sealed,
            "records": {"preflight": _file_record(PREFLIGHT_RECORD), "shakedown": _file_record(SHAKEDOWN_RECORD), "comparison_spec": _file_record(COMPARISON_SPEC_PATH),
                        "field_binding": _file_record(EXPERIMENT_DIR / "fields" / CONFIG_ID / "binding.json"),
                        "preflight_bohm_0_4": _file_record(preflight_record_path("bohm-0.4", "20um")), "shakedown_bohm_0_4": _file_record(EXPERIMENT_DIR / "shakedown-channel-20um-bohm-0.4.json")},
            "launch_discipline": ("HEAD == --expect-commit; clean worktree; protocol.json and the sealed run protocol equal HEAD's blobs; the recomposition on this platform equals the sealed "
                                  "bytes; the preflight passed every option and its launch-box timing passed (budget covers the measured 3-transit wall); the shakedown passed; "
                                  "CUDA_MPS_PIPE_DIRECTORY present; O_EXCL execution-lock.json in results/channel-20um/"),
            "decisions_vs_draft": DECISIONS_VS_DRAFT,
        },
        "execution": {**LAUNCH_BOX, "launch_box_timing": LAUNCH_BOX_TIMING, "launch_box_timing_bohm_0_4": LAUNCH_BOX_TIMING_BOHM},
        "amendments": copy.deepcopy(AMENDMENTS),
        "shakedown_summary": shakedown_summary,
        "reference": reference.reference_document(),
        "geometry_mapping": geometry_module.mapping_table(),
        "field_binding_summary": field_binding_summary,
        "grid_argument": grid_argument(),
        "variants": VARIANTS, "primary": {"variant": PRIMARY_VARIANT, "grid": PRIMARY_GRID, "option": option_tag(PRIMARY_VARIANT, PRIMARY_GRID)},
        "operating_point_policy": {"anode_potential_v": ANODE_POTENTIAL_V, "neutral_density_per_m3": NEUTRAL_DENSITY_PER_M3, "neutral_temperature_k": NEUTRAL_TEMPERATURE_K, "neutrals": "static uniform (no inventory)",
                                   "electron_injection": {"current_a": INJECTION_CURRENT_A, "temperature_ev": INJECTION_TEMPERATURE_EV, "source": "reference continuity I_a - I_beam"},
                                   "seed": SEED, "stability_reference": STABILITY_REFERENCE, "expected_peak_density_per_m3": EXPECTED_PEAK_DENSITY_PER_M3, "expected_mean_density_per_m3": EXPECTED_MEAN_DENSITY_PER_M3},
        "cost_table": {**{f"channel-{grid}": row for grid, row in rows.items()},
                       "plume-brandt-20um": {**plume_cost, "note": "the reference's full box (1024 x 256 at 20 um) on the v2.1 plume model; particles = 1.75x the channel projection (mini-sweep plume factor) capped at 12 M; "
                                                             "the cathode model (annulus, continuity) differs from the reference's rim source - a follow-up protocol on the v2.1 template, NOT composed here"}},
        "comparison_spec": COMPARISON_SPEC_PATH.relative_to(REPOSITORY).as_posix(),
        "preflight_summary": preflight_summary,
        "inconclusive_conditions": comparison.inconclusive_conditions(),
    }


def write_experiment_protocol(path: Path = EXPERIMENT_PROTOCOL_PATH, **kwargs: Any) -> Path:
    path.write_bytes(protocol_bytes(experiment_protocol_document(**kwargs)))
    return path


def write_comparison_spec(path: Path = COMPARISON_SPEC_PATH) -> Path:
    document = comparison.comparison_document()
    problems = comparison.validate_comparison_spec(document)
    if problems:
        raise ValueError(f"comparison spec invalid: {problems}")
    path.write_bytes(protocol_bytes(document))
    return path


__all__ = ["AMENDMENTS", "ANODE_POTENTIAL_V", "ANOMALOUS_MODEL_ROTATION", "BUDGET_FACTOR", "COMPARISON_SPEC_PATH", "COMPOSED_OPTIONS", "DECISIONS_VS_DRAFT", "EXPECTED_MEAN_DENSITY_PER_M3",
           "EXPECTED_PEAK_DENSITY_PER_M3", "EXPERIMENT_ID", "EXPERIMENT_PROTOCOL_PATH", "EXPERIMENT_PROTOCOL_SCHEMA", "GRIDS", "INJECTION_CURRENT_A", "INJECTION_TEMPERATURE_EV", "LAUNCH_BOX",
           "LAUNCH_BOX_TIMING", "LAUNCH_BOX_TIMINGS", "LAUNCH_BOX_TIMING_BOHM", "LAUNCH_HISTORY", "LAUNCH_SET", "MAX_PROJECTED_PARTICLES_M", "NEUTRAL_DENSITY_PER_M3", "NEUTRAL_TEMPERATURE_K",
           "PREFLIGHT_RECORD", "PRIMARY_GRID", "PRIMARY_VARIANT", "PROTOCOLS_DIR", "RUN_PROTOCOL_SCHEMA", "SHAKEDOWN_RECORD", "STABILITY_REFERENCE", "STATUS", "TEMPLATE_PATH", "TRANSIT_S", "VARIANTS",
           "admissible_dt", "budget_basis_hours", "build_protocol", "compose_all", "compose_run_protocol", "composed_protocol_path", "cost_row", "debye_length_m", "density_at_cells_per_debye",
           "density_at_omega_pe_dt", "experiment_protocol_document", "grid_argument", "load_template", "macro_weight_policy", "omega_pe", "option_tag", "preflight_record_path", "protocol_bytes",
           "write_comparison_spec", "write_experiment_protocol"]
