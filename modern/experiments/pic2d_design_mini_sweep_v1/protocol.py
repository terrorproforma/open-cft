"""Per-design run protocols composed from the accepted pic2d templates.

Run options (``build_protocol(design_id, domain, grid=...)``):

* ``channel`` / ``50um`` - DRAFT option: ``pic2d_cft_steady_state_v3/protocol.json`` (model v1.4 channel-only: the v1.3
  accepted plateau physics + wall-ion recycling, peak-node Debye gate 4.5, grid-heating triad, CUDA-graph step), 50 um / 1.5 ps,
  W 6e4.  Costed, never preregistered (the attempt-8 verdict retired the 50 um grid for a dense peak).
* ``channel`` / ``33um`` - THE PREREGISTERED OPTION (``channel-33um``): ``pic2d_cft_steady_state_v4/protocol.json`` (the
  preregistered 33.3 um / 1.4 ps grid refinement of the accepted v2 base plateau: model v1.3 closure - NO wall-ion recycling,
  bit-for-bit the accepted v2 physics - with the v2.0.3 gates: window-mode peak-Debye gate hard pi / soft 2.5 on the
  interval-averaged peak, one-sided windowed residual-power gate 5 %, frame recorder ON).  Grid target 24 mm / 720 =
  33.333 um (the reference design reproduces v4's 90 x 720 cells exactly), dt 1.4 ps unless the design's own field asks
  less, macro weight with particles-per-cell PARITY to the 50 um runs (W = 6e4 x dr dz / (50 um)^2 = 26 666.7 for the
  reference = v4's value), wall budget from the H100 / CUDA-MPS four-slot rate.
* ``plume-12mm`` - ``pic2d_cft_plume_v1/protocol.json`` (model v2.0.2, 12 mm plume box), DRAFT, costed only;
* ``plume-24mm`` - ``pic2d_cft_plume_v2_1/protocol.json`` (model v2.1: 24 mm box on the padded field), DRAFT, costed only.

Only the design-dependent blocks change under the declared operating-point policy:

* geometry / grid: ``designs.pic_geometry`` (dr, dz, snapped radii; the snapping record is copied in);
* neutral feed: ``Q_in = c n_g0`` with ``c = v_bar A_exit / 4`` at the design's (snapped) exit area, n_g0 = 5.5e19 m^-3
  (equal initial neutral density and equal null-collision ceiling headroom for every design);
* cathode annulus (plume options): r in [0.25 r_w, min(r_w, r_exit)], z in [L + 0.3 mm, L + 1.0 mm] - the reference's
  own values (0.5-2.0 mm, 24.3-25.0 mm) reproduced exactly; the preflight re-traces its connectivity;
* time step: the template dt unless omega_ce dt at the design map's max |B| would exceed 0.95 x the 0.2 gate (then the
  largest 0.1 ps multiple below it; the decision and the field maximum are recorded in ``numerics.dt_policy``);
* transit time (budget): 2.4 us x L_channel / 24 mm (the measured reference residence scaled with the channel
  length) + L_plume / 17 km/s; plume-gate arming = one transit;
* wall budget: ``budget_factor`` x the projected wall to 3 transits on the declared platform (``rtx5090`` model for the
  draft options; ``h100-mps4`` = one of four CUDA-MPS slots on the H100 for the preregistered option), rounded up to
  10 min, never below 1 h;
* macro weight: 6e4 at 50 um, the parity weight at 33 um; if the projected particle count exceeds the cap (12 M on the
  80 GB H100) W is scaled so it does not (disclosed in ``case.macro_weight_policy``);
* seed: the template's 20260903 (``case`` = "base") or 20260904 (``case`` = "seed-replicate", the 056 replicate of the
  replication policy).

Everything else (anode potential 300 V, seed plasma, gates, series / checkpoint / window cadence, plateau rule, frame
recorder) is the template's, verbatim.  The template's field declarations are replaced by ``field_binding`` (the
hash-bound per-design field of ``fields.py``; the runner receives the node map directly).

``compose_all`` writes the sealed per-design run protocols (``protocols/<design>-channel-33um[-seed-replicate].json``) and
``write_experiment_protocol`` the experiment-level ``protocol.json`` that binds them together with the preflight, shakedown
and MPS-replay records: the preregistration commit carries all of them, ``run.py launch`` re-composes and refuses to start
unless the recomposition equals the sealed file byte for byte.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from cft_revival.pic2d.neutrals import feed_for_density

from . import cost as cost_module
from . import designs as design_module
from .designs import DOMAIN_OPTIONS, MODERN, REPOSITORY, BuiltDesign, PicMapping, exit_area_m2, pic_geometry

EXPERIMENT_ID = "pic2d-design-mini-sweep-v1"
EXPERIMENT_DIR = Path(__file__).resolve().parent
STATUS = "DRAFT_NOT_PREREGISTERED_NO_PRODUCTION_LAUNCH"                      # the 50 um / plume options (costed, never launched)
STATUS_PREREGISTERED = "preregistered_design_mini_sweep_v1_channel_33um_h100_mps4_not_validated"
TEMPLATES = {
    "channel": MODERN / "experiments" / "pic2d_cft_steady_state_v3" / "protocol.json",
    "plume-12mm": MODERN / "experiments" / "pic2d_cft_plume_v1" / "protocol.json",
    "plume-24mm": MODERN / "experiments" / "pic2d_cft_plume_v2_1" / "protocol.json",
}
REFINED_CHANNEL_TEMPLATE = MODERN / "experiments" / "pic2d_cft_steady_state_v4" / "protocol.json"
STEADY_STATE_V4_PREREG_COMMIT = "392129e5"
GRID_VARIANTS: dict[str, tuple[float | None, float | None]] = {"50um": (None, None), "33um": (cost_module.REFINED_CHANNEL_CELL_M, cost_module.REFINED_CHANNEL_DT_S)}
PREREGISTERED_OPTION = ("channel", "33um")
NEUTRAL_DENSITY_N_G0 = 5.5e19
NEUTRAL_TEMPERATURE_K = 300.0
MACRO_WEIGHT = 60000.0
MAX_PROJECTED_PARTICLES_M = 8.0                    # the draft cap (RTX 5090, 32 GB)
MAX_PROJECTED_PARTICLES_M_H100 = 12.0              # the preregistered cap: 80 GB H100, four MPS slots (~1.3 GB per M particles)
BUDGET_FACTOR = 1.25                               # draft options
BUDGET_FACTOR_PREREGISTERED = 1.5                  # margin on the four-slot MPS rate (per-process speed improves as slots empty)
CATHODE_R_INNER_OVER_RW = 0.25
CATHODE_Z_START_BEHIND_EXIT_M = 0.0003
CATHODE_Z_END_BEHIND_EXIT_M = 0.001
CATHODE_SHRINK_LADDER = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
OMEGA_CE_MARGIN = 0.95            # keep omega_ce dt at the map's max |B| below 0.95 x the gate (0.2)
DT_QUANTUM_S = 1.0e-13
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
DRAFT_PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.json"
PROTOCOLS_DIR = EXPERIMENT_DIR / "protocols"
SEEDS = {"base": 20260903, "seed-replicate": 20260904}
CASES = tuple(SEEDS)
LAUNCH_SET = tuple(d.design_id for d in design_module.SWEEP_DESIGNS if not d.optional)
GPU_RECORD = {
    "model": "NVIDIA H100 80GB HBM3",
    "instance": "Lambda gpu_1x_h100_sxm5 (us-southeast-1), ubuntu@68.209.75.2, driver 580.105.08 (CUDA 13.0), 26 vCPU, 221 GiB RAM",
    "software": "Python 3.12.14, warp-lang 1.14.0 (PyPI CUDA 12.9 build, sm_90 JIT), numpy 2.5.2 (scipy-openblas 0.3.34 Haswell DYNAMIC_ARCH), Ubuntu 22.04",
    "concurrency": "CUDA MPS (nvidia-cuda-mps-control -d; CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps, CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log) with four PIC "
                   "processes on the one GPU (tools/cloud/jobs.yaml slots_per_gpu 4); benchmark: 3.37 ms/step alone -> 8.71 ms/step per process at N = 4 "
                   "(aggregate 1.54x) for the v4 configuration at the production load",
    "determinism": "MPS time-slices the GPU between processes; each process's own kernel order is unchanged, so a same-seed run replays bitwise "
                   "under MPS - VERIFIED on this box before the freeze (mps-replay.json: two concurrent processes and one solo process, series / status "
                   "records and the final checkpoint arrays compared bitwise)",
    "provenance_caveats": [
        "the runner's gpu_utilisation_percent_samples (nvidia-smi, 300 s) read the WHOLE GPU under MPS - shared-GPU readings, not per process",
        "ms/step per process under MPS depends on how many slots are busy (8.71 at N = 4, 3.37 alone): wall-clock projections are per-slot upper bounds, "
        "the budget is a cap, not a cost",
        "same-seed replay across the RTX 5090 (sm_120) and the H100 (sm_90) is numerical, never bitwise (tools/cloud/PLAN.md s.6); the reference design's "
        "sweep run is a numerical replication of steady-state v4 (same grid, dt, W, seed, operating point; different experiment / case ids, frame cadence "
        "identical) on a different GPU",
    ],
}
CONVERGENCE_CAVEAT = (
    "PREDECLARED PER-DESIGN CAVEAT: the 50 um -> 33 um grid-convergence verdict for the reference design is PENDING at this preregistration - "
    f"pic2d_cft_steady_state_v4 (preregistered {STEADY_STATE_V4_PREREG_COMMIT}, launch 1 PID 18068 at 13:11:55 AEST 2026-09-04 on the local RTX 5090, "
    "verdict expected 18:45-19:30 AEST 2026-09-04, budget end 13:15 AEST 2026-09-05) classifies the accepted 50 um plateau as converged / "
    "resolution_limited / refinement_heating / no_plateau.  Every sweep design runs at ONE grid (33.33 um / 1.4 ps, parity W), so no design carries a "
    "convergence statement of its own.  When the sweep is assessed (run.py assess) the v4 verdict MUST be cited per design: 'converged' -> the sweep's "
    "33 um values are quoted with the v4 tolerances (10 % I_d / S / n_g / utilisation / I_beam, 20 % peak n_e / T_e,peak) as their grid band; "
    "'resolution_limited' -> the sweep values are the resolved numbers but carry NO grid band (the 50 um numbers are superseded, the 33 um grid is not "
    "itself certified); 'refinement_heating' or 'no_plateau' -> the reference grid is not certified, every design's quotability rests on its own windowed "
    "residual-power and peak-Debye readings alone, and a design-to-design comparison is reported as 'at the 33 um grid, uncertified'.  The HEMP-like "
    "designs (0.27 T cusp fields) may reach denser peaks than the reference: their own peak-Debye window readings (soft 2.5 precondition, hard pi stop) "
    "are the per-design resolution statement, disclosed with the targets."
)


def load_template(domain: str, grid: str = "50um") -> dict[str, Any]:
    if domain not in DOMAIN_OPTIONS:
        raise ValueError(f"unknown domain option {domain!r}")
    if grid not in GRID_VARIANTS:
        raise ValueError(f"unknown grid variant {grid!r}; known {tuple(GRID_VARIANTS)}")
    return json.loads(template_path(domain, grid).read_text(encoding="utf-8"))


def template_path(domain: str, grid: str = "50um") -> Path:
    if (domain, grid) == PREREGISTERED_OPTION:
        return REFINED_CHANNEL_TEMPLATE
    return TEMPLATES[domain]


def option_tag(domain: str, grid: str = "50um") -> str:
    """The results / protocols directory suffix of an option: ``channel``, ``channel-33um``, ``plume-24mm``."""

    return domain if grid == "50um" else f"{domain}-{grid}"


def _strip_notes(block: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if not k.endswith(("_note", "_justification", "_statement"))}


def macro_weight_for(projected_total_m: float, *, base_weight: float = MACRO_WEIGHT, cap_m: float = MAX_PROJECTED_PARTICLES_M) -> tuple[float, str]:
    """``base_weight`` unless the particle count projected AT that weight exceeds ``cap_m``; then W scaled so it does not (disclosed)."""

    if projected_total_m <= cap_m:
        if base_weight == MACRO_WEIGHT:
            return base_weight, "W = 6e4 (the accepted runs; equal particles per cell at equal density across designs)"
        return base_weight, (f"W = 6e4 x dr dz / (50 um)^2 = {base_weight:.6g}: the same macro-particles per cell as the accepted 50 um / W 6e4 runs at equal "
                             "density (the steady-state v4 rule, W = 6e4 / 2.25 = 26 666.7 on the reference's 33.33 um grid)")
    scaled = base_weight * projected_total_m / cap_m
    return float(round(scaled, 1)), (f"W scaled from the parity value {base_weight:.6g} to cap the projected particle count at {cap_m:g} M (disclosed: "
                                     f"{projected_total_m:.2f} M projected at parity; the W x 0.7 pair bounds the W sensitivity at <= 5.7 % on I_d)")


def admissible_dt(template_dt_s: float, max_b_t: float, limit: float, *, margin: float = OMEGA_CE_MARGIN) -> tuple[float, dict[str, Any]]:
    """The template dt unless omega_ce dt at the map's max |B| would exceed ``margin x limit``; then the largest 0.1 ps multiple below it.

    The recorded floats are rounded to 9 significant digits: ``max_b_t`` is a CPU-derived array maximum that differs at ULP level between
    the Windows (msvc) and Linux (gcc/OpenBLAS) numpy wheels; the sealed protocol must not carry a platform's round-off.
    """

    omega_ce = ELEMENTARY_CHARGE_C * float(max_b_t) / ELECTRON_MASS_KG
    at_template = omega_ce * template_dt_s
    sig = _sig9
    if at_template <= margin * limit:
        return float(template_dt_s), {"rule": f"template dt kept: omega_ce dt {at_template:.3f} <= {margin} x {limit}", "omega_ce_dt_at_template": sig(at_template),
                                      "max_b_t": sig(max_b_t), "template_dt_s": float(template_dt_s)}
    dt = math.floor(margin * limit / omega_ce / DT_QUANTUM_S) * DT_QUANTUM_S
    return float(dt), {"rule": f"dt reduced so omega_ce dt <= {margin} x {limit} at the map's max |B| (pole faces of the plume front face), 0.1 ps quantum",
                       "omega_ce_dt_at_template": sig(at_template), "omega_ce_dt": sig(omega_ce * dt), "template_dt_s": float(template_dt_s), "max_b_t": sig(max_b_t)}


def _sig9(value: float) -> float:
    return float(f"{float(value):.9g}")


def connected_cathode_annulus(mapping: PicMapping, field_map, required: float) -> tuple[dict[str, float], dict[str, Any]]:
    """The sweep's annulus rule, shrunk radially (0.9x steps of r_outer) until the channel-connected fraction reaches ``required``."""

    from cft_revival.pic2d.fieldlines import annulus_connectivity
    from cft_revival.pic2d.mesh import build_mesh_masks

    geometry = mapping.geometry
    r_w, length = float(geometry.bore_radius_m), float(geometry.z_max_m)
    masks = build_mesh_masks(mapping.grid)
    r_inner = CATHODE_R_INNER_OVER_RW * r_w
    base_outer = min(r_w, float(geometry.exit_radius_m))
    attempts = []
    for factor in CATHODE_SHRINK_LADDER:
        r_outer = base_outer * factor
        if r_outer <= r_inner * 1.2:
            break
        result = annulus_connectivity(field_map, masks, r_inner, r_outer, length + CATHODE_Z_START_BEHIND_EXIT_M, length + CATHODE_Z_END_BEHIND_EXIT_M, n_r=6, n_z=4)
        attempts.append({"r_outer_factor": factor, "r_outer_m": r_outer, "connected_fraction": result["connected_fraction"], "terminations": result["terminations"]})
        if result["connected_fraction"] >= required:
            return ({"r_inner_m": r_inner, "r_outer_m": r_outer, "z_start_m": length + CATHODE_Z_START_BEHIND_EXIT_M, "z_end_m": length + CATHODE_Z_END_BEHIND_EXIT_M},
                    {"rule": "r in [0.25 r_w, f x min(r_w, r_exit)] with f the first of the shrink ladder whose 6 x 4 sample is connected at the required fraction", "attempts": attempts, "selected_factor": factor})
    return ({"r_inner_m": r_inner, "r_outer_m": base_outer, "z_start_m": length + CATHODE_Z_START_BEHIND_EXIT_M, "z_end_m": length + CATHODE_Z_END_BEHIND_EXIT_M},
            {"rule": "no ladder factor reached the required fraction: base annulus kept, the runner's gate will refuse the launch", "attempts": attempts, "selected_factor": None})


def _sweep_acceptance(design_id: str, transit_s: float) -> dict[str, Any]:
    return {
        "declared": "predeclared before the launch; evaluated per design by `run.py assess --design ID --grid 33um`; the verdict is one of the d_verdicts values and "
                    "is recorded in results/<design>-channel-33um/assessment.json (results-only commit)",
        "a_plateau": f"stop_reason == plateau_reached_after_min_transit_times under stopping_rule.plateau (>= 3 design transits = 3 x {transit_s*1e6:.3f} us, "
                     "trailing-20 % drifts of I_d, N_e and n_g < 5 %, grid-heating triad soft bounds, window-mode peak-Debye soft margin 2.5)",
        "b_residual_power": "summary.grid_heating_triad.windowed_energy_residual_over_electrode_work (the trailing 400000-step ratio at the stop) < +0.02, "
                            "one-sided (positive = heating; the accepted 50 um runs read -0.2 / -1.5 / -4.2 %): the run is itself free of grid heating at the 2 % level",
        "c_closure_targets": "every closure target of protocol.json closure_targets is extracted by `run.py targets` from the trailing-window maps.npz + summary.json "
                             "(closure.extract_targets: cusp windows |z - z_c| <= min(1 mm, pitch / 4), near-wall band 0.5 mm, Kornfeld chain seeded with injected - "
                             "returned exit electron current, anode-edge band 0.25 mm reported separately); a target is QUOTABLE only under (a) AND (b)",
        "d_verdicts": {
            "closure_quotable": "(a) AND (b): the design's closure targets enter plasma-network v2 as calibration / reproduction values with their block standard errors",
            "plateau_with_heating": "(a) but NOT (b): the plateau heats above 2 % of the electrode work; targets are reported, not quotable; a finer grid or a lower "
                                    "operating point is the follow-up for this design",
            "no_plateau": "NOT (a): budget stop / gate stop / soft-margin block -> 'inconclusive within the budget' with the stop reason, the trailing drifts and the "
                          "trailing-window quantities reported; a resume is a new session under the same identity (disclosed), never a re-preregistration",
        },
        "e_design_effect": "a design effect between two designs is reported only if it exceeds BOTH the reference's seed-b / W x 0.7 spread (<= 1.1 % / <= 5.7 % on the "
                           "plateau quantities) AND, when the 056 seed replicate has run, that replicate's spread; with four points the sweep tests a monotone trend of "
                           "p_k and of the ion wall-loss fraction with rho (0.38 -> 0.61 -> 0.92 -> 2.36) and a qualitative change in the HEMP-like regime, nothing finer",
        "f_convergence_caveat": CONVERGENCE_CAVEAT,
        "g_design_specific": {
            "l1a-gs-v2-047-e3196a8aa5": "KEPT with the disclosed anode-edge boundary cusp (under linear iron the anode-side axis null moves from -1.40 to -0.11 mm and its "
                                        "separatrix reaches the dielectric 0.073 mm from the anode plane, 13 deg to the wall normal; boundary classification under the v3.1 "
                                        "0.25 mm ambiguity tolerance; the three interior cusps match L1a within 0.30 mm of the 0.45 mm tolerance; interior rho 0.377). In the "
                                        "PIC that 73 um sits inside the anode sheath (2 cells of 33 um); its electron loss is reported as anode_edge_electron_wall_current_a, "
                                        "never as an interior cusp. The substitute 061 (rho 0.381, 5.9 mm exit taper) was NOT taken: the taper would confound the rho ladder "
                                        "with a long cone the other three designs do not have.",
        }.get(design_id, "none"),
    }


def build_protocol(design_id: str, domain: str, *, built: BuiltDesign | None = None, field_map=None,
                   target_cell_m: float | None = None, dt_s: float | None = None, grid: str | None = None,
                   platform: str | None = None, budget_factor: float | None = None, case: str = "base") -> tuple[dict[str, Any], PicMapping]:
    """Compose the per-design run protocol.  With ``field_map`` (the design's node field) the dt rule and the cathode placement
    search run on the actual field; without it the template dt and the base annulus are used (the preflight passes the map).
    ``grid`` (``50um`` / ``33um``) selects the template and the grid / dt overrides (``target_cell_m`` / ``dt_s`` may be given
    directly instead); ``platform`` / ``budget_factor`` default to the option's (H100 MPS-4 x 1.5 for the preregistered option,
    RTX 5090 model x 1.25 otherwise); ``case`` is ``base`` or ``seed-replicate``."""

    if grid is None:
        grid = "33um" if (target_cell_m, dt_s) == GRID_VARIANTS["33um"] else "50um"
    if target_cell_m is None and dt_s is None:
        target_cell_m, dt_s = GRID_VARIANTS[grid]
    if case not in SEEDS:
        raise ValueError(f"unknown case {case!r}; known {CASES}")
    preregistered = (domain, grid) == PREREGISTERED_OPTION
    platform = platform if platform is not None else ("h100-mps4" if preregistered else "rtx5090")
    budget_factor = budget_factor if budget_factor is not None else (BUDGET_FACTOR_PREREGISTERED if preregistered else BUDGET_FACTOR)
    built = design_module.build_design(design_id) if built is None else built
    mapping = pic_geometry(built, domain) if target_cell_m is None else pic_geometry(built, domain, target_cell_m=target_cell_m)
    template = load_template(domain, grid)
    protocol = copy.deepcopy(template)
    if dt_s is not None:
        protocol["numerics"]["dt_s"] = float(dt_s)
    budget_keys = [key for key in protocol if key.startswith("budget")]
    template_budget = protocol[budget_keys[0]] if budget_keys else {}
    for key in budget_keys + ["field_authority", "field_plume_extension", "lineage", "v2_reference", "validation_v1_observable", "reference_run", "preregistration"]:
        protocol.pop(key, None)
    protocol["schema_version"] = "cft.pic2d.design-mini-sweep.run-protocol/1.0.0" if preregistered else "cft.pic2d.design-mini-sweep.run-protocol/0.1.0-draft"
    protocol["experiment_id"] = EXPERIMENT_ID
    protocol["status"] = STATUS_PREREGISTERED if preregistered else STATUS
    protocol["option"] = option_tag(domain, grid)
    protocol["template_protocol"] = {"domain_option": domain, "grid_variant": grid, "path": template_path(domain, grid).relative_to(REPOSITORY).as_posix(),
                                     "experiment_id": template.get("experiment_id"), "model_version": template.get("model_version"),
                                     "sha256": hashlib.sha256(template_path(domain, grid).read_bytes()).hexdigest()}
    protocol["design_id"] = design_id
    protocol["design"] = built.design.to_dict()
    protocol["design_identity"] = built.identity
    protocol["field_binding"] = f"modern/experiments/pic2d_design_mini_sweep_v1/fields/{design_id}/binding.json"
    protocol["field_source"] = ("existing reference artifacts (authority level-1 / padding-1.5) through the v2.0 / v2.1 field pipeline"
                                if design_id == design_module.REFERENCE_DESIGN_ID else
                                "design-mini-sweep binding: direct node evaluation of the design's padded level-0 material-aware P2 checkpoint, scaled by source_strength_scale")
    if preregistered:
        protocol["classification"] = "axisymmetric_electrostatic_pic_mcc_preregistered_design_sweep_channel_only_33um_v1_3_closure_v2_0_3_gates_not_validated"
    # geometry / grid
    geometry = mapping.geometry.to_dict()
    geometry["source"] = f"catalogue design {design_id} mapped onto the PIC straight-bore + cone (+ plume) representation; snaps recorded"
    geometry["snaps"] = mapping.snaps
    protocol["geometry"] = geometry
    # macro weight: 6e4 (50 um) or particles-per-cell parity (33 um); cost at that weight on the declared platform
    parity = cost_module.parity_macro_weight(mapping) if grid == "33um" else MACRO_WEIGHT
    cap = MAX_PROJECTED_PARTICLES_M_H100 if platform == "h100-mps4" else MAX_PROJECTED_PARTICLES_M
    projected_at_parity = cost_module.projected_particles_m(mapping, macro_weight=parity)["total_m"]
    weight, weight_policy = macro_weight_for(projected_at_parity, base_weight=parity, cap_m=cap)
    cost = cost_module.design_cost(mapping, dt_s=float(protocol["numerics"]["dt_s"]), macro_weight=weight, platform=platform)
    case_block = protocol["case"]
    seed = SEEDS[case]
    case_tag = "" if case == "base" else f"-{case}"
    case_block.update({
        "id": (f"{design_id}-{option_tag(domain, grid)}-w{weight:.6g}-ng0-{NEUTRAL_DENSITY_N_G0:.2g}-seed5e16-v1.3-closure-v2.0.3-gates{case_tag}" if preregistered
               else f"{design_id}-{option_tag(domain, grid)}-w{weight:.3g}-ng0-{NEUTRAL_DENSITY_N_G0:.2g}-seed5e16-draft{case_tag}"),
        "radial_cells": int(mapping.grid.radial_cells), "axial_cells": int(mapping.grid.axial_cells), "macro_weight": weight,
        "macro_weight_policy": weight_policy, "seed": seed, "case": case,
        "grid_policy": {"target_cell_m": mapping.snaps["target_cell_m"], "dt_s": float(protocol["numerics"]["dt_s"]),
                        "source": ("template (50 um / 1.5 ps)" if grid == "50um" else
                                   "the preregistered steady-state v4 grid (392129e5): target 24 mm / 720 = 33.333 um, dt 1.4 ps (attempt-8 verdict ac248e05: "
                                   "Delta <= 32.4 um, dt <= 1.48 ps); the reference design reproduces v4's 90 x 720 cells exactly")},
        "grid_note": f"dr {mapping.grid.dr_m*1e6:.3f} um x dz {mapping.grid.dz_m*1e6:.3f} um: {mapping.grid.radial_cells} x {mapping.grid.axial_cells} cells, nodes {mapping.grid.node_shape}; bore {mapping.snaps['bore_cells']} cells",
    })
    for key in ("seed_note", "variant", "variant_note"):
        case_block.pop(key, None)
    if case == "seed-replicate":
        case_block["seed_note"] = "seed b (20260904): the single seed replicate of the replication policy (the HEMP-like design 056); the RNG streams differ from the base run only through the seed"
    # operating point: feed for n_g0 at the design's exit area
    operating = protocol["operating_point"]
    area = exit_area_m2(mapping)
    feed = feed_for_density(NEUTRAL_DENSITY_N_G0, area, NEUTRAL_TEMPERATURE_K)
    inventory = operating["neutral_inventory"]
    inventory["feed_atoms_per_s"] = float(feed)
    inventory["feed_justification"] = (f"Q_in = c n_g0 with c = v_bar A_exit / 4 (xenon 300 K), A_exit = pi ({mapping.geometry.exit_radius_m*1e3:.3f} mm)^2 = {area:.4e} m^2, "
                                       f"n_g0 = {NEUTRAL_DENSITY_N_G0:.2e} m^-3 -> {feed:.4e} atoms/s (equal initial neutral density for every design; the reference's 8.551e16 /s reproduced)")
    if preregistered:
        operating["unchanged_note"] = ("operating-point POLICY of the sweep: anode 300 V, exit plane 0 V, n_g0 5.5e19 m^-3 (initial density and MCC ceiling), tau_g 30 ns, "
                                       "exit-plane injection 3 mA @ 2 eV, seed plasma 5e16 m^-3 @ 5 eV, cold ions - every key bit-for-bit the accepted v2 attempt-2 / v4 "
                                       "value; ONLY the neutral feed changes with the design (Q_in = c n_g0 at the design's exit area). Equal n_g0 and equal injection "
                                       "current, NOT equal mass flow or current density, is the declared comparison")
    if operating.get("cathode") is not None:
        r_w = float(mapping.geometry.bore_radius_m)
        length = float(mapping.geometry.z_max_m)
        cathode = operating["cathode"]
        annulus = {"r_inner_m": CATHODE_R_INNER_OVER_RW * r_w, "r_outer_m": min(r_w, float(mapping.geometry.exit_radius_m)),
                   "z_start_m": length + CATHODE_Z_START_BEHIND_EXIT_M, "z_end_m": length + CATHODE_Z_END_BEHIND_EXIT_M}
        search: dict[str, Any] | None = None
        if field_map is not None and cathode.get("require_channel_connected_fraction") is not None:
            annulus, search = connected_cathode_annulus(mapping, field_map, float(cathode["require_channel_connected_fraction"]))
        cathode.update(annulus)
        cathode["placement_search_note"] = search
        cathode["position_justification"] = ("sweep policy: annulus r in [0.25 r_w, min(r_w, r_exit)], z in [L + 0.3 mm, L + 1.0 mm] inside the channel's exit flux tube "
                                             "(the reference's 0.5-2.0 mm / 24.3-25.0 mm reproduced), r_outer shrunk in 0.9x steps until the 6 x 4 sample is connected at the "
                                             "required fraction (placement_search_note); the runner re-traces at launch and refuses to start below the fraction")
    # numerics: plume gate arming = one transit of the design's box; dt admissible at the map's max |B|
    numerics = protocol["numerics"]
    if numerics.get("plume_boundary_gate") is not None:
        numerics["plume_boundary_gate"]["enforce_after_s"] = float(cost["transit_s"])
    if field_map is not None:
        dt, dt_policy = admissible_dt(float(numerics["dt_s"]), field_map.max_b_t, float(numerics["stability_limits"]["max_omega_ce_dt"]))
        numerics["dt_s"] = dt
        numerics["dt_policy"] = dt_policy
        if dt != cost["dt_s"]:
            cost = cost_module.design_cost(mapping, dt_s=dt, macro_weight=weight, platform=platform)
    if preregistered:
        numerics["dt_justification"] = ("1.4 ps = the steady-state v4 time step (omega_pe dt 0.101 at the v2 window peak 1.64e18; attempt-8 analysis dt <= 1.48 ps), reduced "
                                        "only where the design's own channel field would put omega_ce dt above 0.95 x 0.2 (numerics.dt_policy; none of the four primary "
                                        "channel-only maps needs it: the 1.3 ps requirement of design 056 belongs to the 24 mm plume box's 0.821 T pole faces)")
        numerics["peak_debye_gate"]["max_cells_per_debye_note"] = (
            "v2.0.3 window-mode gate (spec/pic2d/pic2d-model-v2.0.json gates_v2_0.peak_debye_gate_v2_0_3): the gated statistic is max(dr, dz) / lambda_D at the densest node "
            "of the trailing 400000-step interval average among nodes with mean occupancy >= 32 macro-electrons, with the window's moment T_e. HARD pi (Birdsall-Langdon CIC "
            "threshold; plume attempt 8 heated from ~3.2), fail-closed once the window is complete. SOFT 2.5 is a plateau precondition (plateau.peak_debye_soft_ok), never a "
            "stop. Expected at the reference's v2 peak (1.64e18 / 7.39 eV) on 33.33 um: 2.11; the HEMP-like designs may read higher - a plateau above 2.5 is recorded as "
            "'resolution margin not met' (acceptance (a) fails, run continues to the budget), a legitimate outcome of the design, not a defect")
        numerics["peak_debye_gate"]["min_macro_particles_at_peak_note"] = ("mean occupancy over the window; with particles-per-cell parity the dense off-axis nodes hold "
                                                                            "~300 macro-electrons as in the 50 um runs; axis nodes stay below the floor")
        numerics["frame_recorder_note"] = (f"frame recorder ON: {int(numerics['frame_recorder']['cadence_steps'])}-step frames = "
                                           f"{int(numerics['frame_recorder']['cadence_steps']) * float(numerics['dt_s']) * 1e9:.1f} ns exact interval averages from the window "
                                           "accumulators (float32), for the per-design ionisation structure at the cusp planes and a video; frames are diagnostics, not gates")
    # stopping rule: plateau verbatim; budget from the cost model on the declared platform
    rule = protocol["stopping_rule"]
    wall_budget = max(3600.0, math.ceil(budget_factor * cost["wall_hours"] * 3600.0 / 600.0) * 600.0)
    rule["wall_budget_seconds"] = float(wall_budget)
    rule["wall_budget_note"] = (f"{budget_factor} x the projected wall to 3 transits ({cost['wall_hours']:.1f} h at {cost['ms_per_step']:.2f} ms/step on {platform}"
                                f"{' = one of four CUDA-MPS slots of the H100 (8.71 ms/step anchor at N = 4)' if platform == 'h100-mps4' else ''}, "
                                f"{cost['steps_to_transits']/1e6:.2f} M steps), rounded up to 10 min; cumulative over resumes; a budget stop without a plateau is a "
                                "recorded outcome (resumable as a new session under the same identity, disclosed)")
    transit_s = float(cost["transit_s"])
    if preregistered:
        rule["plateau"] = (f"relative drift < 5 % over the trailing 20 % of the elapsed simulated time for the discharge current, the plasma electron count AND the neutral "
                           f"density (linear fit, drift = slope x window / |mean|), evaluated at every checkpoint; may only be declared after >= 3 ion transit times "
                           f"(3 x {transit_s*1e6:.3f} us = {3*transit_s*1e6:.3f} us = {int(round(3*transit_s/float(numerics['dt_s']))):,} steps); additionally the grid-heating "
                           "triad must be inside its soft bounds (plateau.triad_soft_ok) AND the window-mode peak-Debye soft margin must hold (plateau.peak_debye_soft_ok: "
                           "trailing-window Delta/lambda_D at the peak <= 2.5) - the accepted v2 base rule plus the v2.0.3 preconditions, verbatim from steady-state v4")
        rule["ignition_check"] = ("the seed and injection are the ones that ignited v2 attempt 2 (and seed-b, w-0.7, the v4 refinement); for the three L1a designs this is "
                                  "the FIRST PIC run on their fields: S growing and N_e rising by ~1 us with n_g heading toward its fixed point is expected, NOT guaranteed; "
                                  "a non-ignition is a recorded outcome of the design under this operating-point policy - NO adjustment is allowed under this preregistration")
        rule["acceptance"] = _sweep_acceptance(design_id, transit_s)
    else:
        rule["plateau"] = rule["plateau"].split(";")[0] + f"; may only be declared after >= 3 ion transit times (3 x {transit_s*1e6:.2f} us); the grid-heating triad must be inside its soft bounds"
    protocol["budget_design_mini_sweep"] = {
        # the runner's summary.budget_check reads these two from the (single) budget block: the template's a-priori design ceiling
        # (= numerics.stability_reference density, the same for every design) and the reference's projected equilibrium density
        "n_max_per_m3": float(template_budget.get("n_max_per_m3", protocol["numerics"]["stability_reference"]["density_per_m3"])),
        "n_eq_projected_per_m3": float(template_budget.get("n_eq_projected_per_m3", protocol["numerics"]["stability_reference"]["density_per_m3"])),
        "n_max_note": "template values (reference design): n_max = the a-priori design ceiling used by the stability reference; n_eq_projected = the reference's "
                      "projected fixed-point density; for the other designs they are the REFERENCE numbers the ratios in summary.budget_check are formed against",
        "ion_transit_time_s": transit_s,
        "ion_transit_note": "2.4 us x L_channel / 24 mm (measured reference residence scaled with the channel length) + L_plume / 17 km/s; replaced by the measured N_i / L residence at the first checkpoint past 1 us when the run reports it",
        "particles_projected_m": cost["particles_projected_m"], "ms_per_step_projected": cost["ms_per_step"], "ms_per_step_rtx5090_model": cost["ms_per_step_rtx5090_model"],
        "ms_per_step_h100_mps4_per_process": cost["ms_per_step_h100_mps4_per_process"], "platform": platform, "steps_to_3_transits": cost["steps_to_transits"],
        "hours_to_3_transits_projected": cost["wall_hours"], "factorisation_s_projected": cost["factorisation_s"], "device_gb_projected": cost["device_gb_projected"],
        "macro_weight": weight, "macro_weight_parity": parity, "dr_m": mapping.grid.dr_m, "dz_m": mapping.grid.dz_m, "dt_s": float(protocol["numerics"]["dt_s"]),
        "wall_budget_factor": budget_factor,
        "cost_model": "experiments.pic2d_design_mini_sweep_v1.cost (anchors: steady-state v2 base / W x 0.7, plume attempt 8; H100 MPS-4 anchor 8.71 ms/step for the v4 configuration)",
    }
    if preregistered:
        protocol["execution"] = {
            "gpu": GPU_RECORD, "scheduler": "modern/tools/cloud/schedule.py (tmux, detached worktree at the preregistration commit, per-job results directory, "
            "Warp cuda:0 UUID cross-check, prereg ancestor + byte-identical protocol checks) with slots_per_gpu 4 and the MPS variables exported",
            "launch_discipline": "run.py launch --design ID --grid 33um --expect-commit <prereg sha> --require-mps: HEAD == prereg commit, clean worktree, protocol.json and "
                                 "protocols/<design>-channel-33um.json blobs == HEAD, recomposed protocol == sealed file byte for byte, preflight-channel-33um.json + "
                                 "shakedown-channel-33um.json + mps-replay.json present, O_EXCL execution-lock.json in the results directory, MPS pipe directory present",
            "one_execution": "one detached launch per design (and per declared case) from its own worktree at the preregistration commit; a wall-budget stop may be resumed "
                             "(--resume: new session, same identity, disclosed in run_state.sessions); no parameter changes after the freeze",
        }
        simplifications = [s for s in protocol.get("simplifications", []) if not s.startswith(("single seed and a single refined grid", "preregistered resolution-convergence study"))]
        simplifications += [
            "one grid per design (33.33 um / 1.4 ps, particles-per-cell parity with the 50 um runs): no per-design convergence study; the reference's 50 -> 33 um verdict "
            "(steady-state v4, pending at the freeze) is the only grid statement and is cited per design (stopping_rule.acceptance.f_convergence_caveat)",
            "catalogue geometry snapped to the grid (geometry.snaps, <= half a cell); the dielectric tube / magnet clearance gap is not represented in the channel-only box",
            "material-aware level-0 P2 field of the design (linear iron), scaled by the L1a source_strength_scale; not P2-qualified for these designs (fields/<id>/binding.json gates)",
            "equal n_g0 and equal 3 mA / 2 eV exit-plane injection across designs (operating-point policy), not equal mass flow, mass-flow density or current density",
            "four processes share one H100 through CUDA MPS: ms/step and gpu_utilisation samples are shared-GPU readings; the physics of each process is single-process "
            "deterministic (mps-replay.json)",
            "preregistered design sweep of a development model: no experimental validation, not a performance prediction; the closure targets calibrate plasma-network v2 "
            "under the declared closure and are properties of that closure",
        ]
        protocol["simplifications"] = simplifications
    inherited = protocol.get("claim_boundary")
    inherited_block = inherited if isinstance(inherited, dict) else {"template_claim_boundary": inherited}
    protocol["claim_boundary"] = {
        **inherited_block,
        **({"preregistered": "preregistered channel-only 33.33 um / 1.4 ps PIC-MCC run of ONE catalogue design under the sweep's operating-point policy, model v1.3 closure "
                              "with the v2.0.3 gates, on a Lambda H100 80GB (CUDA MPS, four slots); yields the design's closure targets for plasma-network v2 under the "
                              "predeclared acceptance; not validated against experiment; not a thruster performance prediction; the neutral transient is artificial and "
                              "only the fixed point is physical"}
           if preregistered else
           {"draft": "DRAFT protocol composed from the template; NOT preregistered; no production launch (the preregistered option is channel-33um)"}),
        "geometry_approximation": f"catalogue radii snapped to the {mapping.snaps['target_cell_m']*1e6:.2f} um grid (geometry.snaps); the clearance gap between the dielectric tube and the magnets is treated as dielectric front face",
        "field": "material-aware level-0 P2 field of the design (linear iron), scaled by the L1a source_strength_scale; not P2-qualified for these designs",
    }
    return protocol, mapping


# --------------------------------------------------------------------------
# Sealed per-design run protocols and the experiment-level protocol.json
# --------------------------------------------------------------------------


def composed_protocol_path(design_id: str, domain: str = "channel", grid: str = "33um", case: str = "base", *, root: Path = PROTOCOLS_DIR) -> Path:
    suffix = "" if case == "base" else f"-{case}"
    return root / f"{design_id}-{option_tag(domain, grid)}{suffix}.json"


def protocol_bytes(protocol: dict[str, Any]) -> bytes:
    """The one serialisation every sealed / launched protocol file uses (LF, sorted keys, indent 1, no NaN)."""

    return json.dumps(protocol, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"


def compose_run_protocol(design_id: str, domain: str = "channel", grid: str = "33um", case: str = "base") -> tuple[dict[str, Any], PicMapping, Any]:
    """Field first, then the protocol (the dt rule and the cathode placement read the design's own node field)."""

    from . import fields as field_module

    target_cell_m, dt_s = GRID_VARIANTS[grid]
    built = design_module.build_design(design_id)
    mapping = pic_geometry(built, domain) if target_cell_m is None else pic_geometry(built, domain, target_cell_m=target_cell_m)
    binding = field_module.load_binding(design_id)
    field_map = field_module.design_field_map(mapping, binding)
    protocol, _ = build_protocol(design_id, domain, built=built, field_map=field_map, target_cell_m=target_cell_m, dt_s=dt_s, grid=grid, case=case)
    return protocol, mapping, field_map


def compose_all(domain: str = "channel", grid: str = "33um", *, root: Path = PROTOCOLS_DIR, replicate_design_ids: tuple[str, ...] = ("l1a-gs-v3-056-effcbc8686",),
                log=print) -> dict[str, str]:
    """Write the sealed per-design run protocols of the option (every design, base case; plus the declared seed replicates); returns path -> sha256."""

    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    items = [(design_id, "base") for design_id in design_module.design_ids()] + [(design_id, "seed-replicate") for design_id in replicate_design_ids]
    for design_id, case in items:
        protocol, _, _ = compose_run_protocol(design_id, domain, grid, case)
        path = composed_protocol_path(design_id, domain, grid, case, root=root)
        data = protocol_bytes(protocol)
        path.write_bytes(data)
        written[path.relative_to(REPOSITORY).as_posix()] = hashlib.sha256(data).hexdigest()
        log(f"[compose] {path.name}: W {protocol['case']['macro_weight']:.6g}, dt {protocol['numerics']['dt_s']*1e12:.2f} ps, cells {protocol['case']['radial_cells']} x "
            f"{protocol['case']['axial_cells']}, budget {protocol['stopping_rule']['wall_budget_seconds']/3600:.1f} h, sha256 {written[path.relative_to(REPOSITORY).as_posix()][:12]}")
    return written


def _file_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = path.read_bytes()
    return {"path": path.relative_to(REPOSITORY).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def experiment_protocol_document(*, preflight_summary: dict[str, Any] | None = None, records: dict[str, Path] | None = None,
                                 sealed: dict[str, str] | None = None, preregistered: bool = True) -> dict[str, Any]:
    """The experiment-level protocol.json: design list, field bindings, policies, closure targets, cost table, the sealed run
    protocols and the preflight / shakedown / MPS-replay records of the preregistered option."""

    from .closure import closure_target_table

    built = [design_module.build_design(d.design_id) for d in design_module.SWEEP_DESIGNS]
    table = cost_module.cost_table(built)
    recommended = cost_module.serial_schedule(table, option="channel", replicate_design_ids=("l1a-gs-v3-056-effcbc8686",),
                                              extra=((design_module.REFERENCE_DESIGN_ID, "plume-24mm"),))
    refined = cost_module.serial_schedule(table, option=cost_module.REFINED_CHANNEL_KEY, replicate_design_ids=("l1a-gs-v3-056-effcbc8686",))
    refined["note"] = ("per-process hours at the H100 CUDA-MPS four-slot rate (8.71 ms/step anchor for the v4 configuration); the four primary designs run "
                       "CONCURRENTLY, so the campaign wall time is the longest row (+ its budget margin), not the sum; per-process speed improves as slots empty")
    launch_rows = {row["design_id"]: row for row in table[cost_module.REFINED_CHANNEL_KEY]}
    records = records or {}
    if not records and preregistered:
        records = {"preflight": EXPERIMENT_DIR / "preflight-channel-33um.json", "shakedown": EXPERIMENT_DIR / "shakedown-channel-33um.json",
                   "mps_replay": EXPERIMENT_DIR / "mps-replay.json"}
    if sealed is None and preregistered and PROTOCOLS_DIR.is_dir():
        sealed = {p.relative_to(REPOSITORY).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROTOCOLS_DIR.glob("*.json"))}
    preregistration = {
        "state": ("PREREGISTERED at the commit that carries this file (the prereg commit); protocols/<design>-channel-33um[-seed-replicate].json are the sealed per-design "
                  "run protocols (byte-bound at launch); run.py launch refuses a dirty worktree, an unexpected commit, a recomposition that differs from the sealed file, "
                  "a missing preflight / shakedown / MPS-replay record, a missing MPS pipe directory (--require-mps) or an existing execution lock"
                  if preregistered else "NOT preregistered; this file is a DRAFT and is not hash-bound to any run"),
        "option": option_tag(*PREREGISTERED_OPTION),
        "launch_set": list(LAUNCH_SET),
        "not_launched_in_this_campaign": {
            "l1a-gs-v3-106-ccec1c8b2f": "optional fifth design (four cusps, rho 2.93): composed and sealed, launched only if the budget allows after the four primary designs",
            "l1a-gs-v3-056-effcbc8686 seed-replicate": "the replication policy's single seed replicate (seed 20260904): composed and sealed, launched after the four primary "
                                                       "designs free a slot; a design effect is reported only above its spread once it exists",
        },
        "decisions": {
            "grid": "channel-only 33.33 um / 1.4 ps (target 24 mm / 720 = the steady-state v4 grid; the reference reproduces v4's 90 x 720 exactly; the draft's 3.33e-5 m "
                    "target gave 90 x 721) - the resolved grid of the attempt-8 verdict (ac248e05); 50 um retired for a dense peak",
            "template_and_physics": "steady-state v4 protocol (392129e5): model v1.3 closure (NO wall-ion recycling; bit-for-bit the accepted v2 base physics) with the v2.0.3 "
                                    "gates - CHANGED from the draft's v1.4 (recycling) channel template. Reasons: (1) the sweep's reference run must be comparable with the "
                                    "only 33 um reference run whose convergence verdict it must cite (ss-v4, same physics, same grid / dt / W / seed); (2) v1.4 has no "
                                    "accepted plateau anywhere (steady-state v3 was never launched) and its recycled fixed point projects the reference peak at "
                                    "Delta/lambda_D ~2.45 on this grid - on the 2.5 soft precondition - so every denser design would risk 'no plateau by soft margin'; "
                                    "the v1.3 peak projects 2.11; (3) budgets and ignition expectations anchor on the accepted v1.3 plateaus. Wall-ion recycling stays a "
                                    "declared follow-up closure (gross = net utilisation under v1.3, recorded as such)",
            "gates": "PIC model v2.0.3: window-mode peak-Debye gate hard pi / soft 2.5 on the interval-averaged peak (>= 32 macro-electrons); one-sided windowed "
                     "residual-power gate >= 5 % of the electrode work over the trailing 400000 steps; v1.4 triad drift members; omega_pe dt, Courant, Poisson, inventory gates",
            "plateau_rule": "the accepted steady-state rule (>= 3 design transits, trailing-20 % drifts of I_d / N_e / n_g < 5 %) + the v2.0.3 preconditions (triad soft, "
                            "peak-Debye soft 2.5); frames ON (20000-step = 28 ns interval averages)",
            "macro_weight": "particles-per-cell parity with the 50 um runs: W = 6e4 x dr dz / (50 um)^2 (26 666.7 for the reference = v4; within 1 % of it for the others; "
                            "the draft kept W = 6e4, i.e. 2.25x fewer particles per cell); cap 12 M projected particles (H100 80 GB) - none of the four primary designs hits it",
            "design_047": "KEPT with the disclosed anode-edge boundary cusp (see stopping_rule.acceptance.g_design_specific of its run protocol): boundary classification "
                          "under the v3.1 0.25 mm tolerance, interior cusps within tolerance, the 73 um separatrix foot sits in the anode sheath (2 cells); its electron "
                          "loss is reported as anode_edge_electron_wall_current_a. Substitute 061 NOT taken: its 5.9 mm exit taper would confound the rho ladder",
            "dt": "1.4 ps for every primary design (channel-only maps: omega_ce dt <= 0.08 at the maps' max |B|); design 056's 1.3 ps requirement is a property of the "
                  "24 mm plume box (0.821 T pole faces), not of the channel-only option (preflight-channel-33um.json dt_policy per design)",
            "budget": f"{BUDGET_FACTOR_PREREGISTERED} x the projected wall to 3 transits at the H100 CUDA-MPS four-slot per-process rate (cost model scaled to the 8.71 ms/step "
                      "N = 4 anchor), rounded up to 10 min; cumulative over resumes; the per-design hours are in refined_grid_schedule",
            "gpu": GPU_RECORD["model"] + " with CUDA MPS, four slots (execution.gpu of every run protocol)",
            "operating_point": "equal n_g0 5.5e19 (feed Q_in = c n_g0 at the design's exit area) and equal 3 mA / 2 eV exit-plane injection (the draft's choice, frozen)",
            "estimands": "closure.extract_targets as frozen: cusp window half-width min(1 mm, pitch / 4), near-wall band 0.5 mm, Kornfeld chain seeded with (injected - "
                         "returned) exit electron current, anode-edge band 0.25 mm; cusp planes and cells from the design's own material-aware topology (binding.json)",
            "field_level": "level-0 padded material-aware solves (binding.json gates: mesh >= 5 deg, residual <= 2e-10, coverage, topology within tolerance, |dB| vs L1b "
                           "<= 1.5 mT); a level-1 refinement is not needed by the gates and is not part of this preregistration",
        },
        "before_prereg_done": [
            "whole-set preflight of the channel-33um option over every design ON THE LAUNCH BOX (preflight-channel-33um.json: identity, field binding via file hashes + "
            "field_source_sha256, grid snaps, field map + a-priori stability at the composed dt, mesh masks, composed protocol accepted by the runner, cost)",
            "MPS determinism replay on the launch box (mps-replay.json: two concurrent same-seed processes + one solo, bitwise series / status / checkpoint comparison)",
            "labelled NON-EVIDENTIARY shakedown of ONE design on its real field through run -> finalize -> assess -> targets at shrunk cadences on the launch box "
            "(shakedown-channel-33um.json; results directory not committed)",
            "sealed per-design run protocols (protocols/), their hashes below",
        ],
        "records": {name: _file_record(path) for name, path in records.items()},
        "sealed_run_protocols": sealed,
        "convergence_caveat": CONVERGENCE_CAVEAT,
        "attempt_8_verdict": "recorded in ac248e05 (2026-09-04 11:57 AEST): grid-heating triad gate stop at 4.98 us (S drift 0.253), NO plateau; "
                             "finite-grid heating from ~2.4 us once the peak-node Delta/lambda_D crossed ~3.2 (CIC threshold pi); the accepted channel-only "
                             "plateau sat at 3.17 with a residual closing to +0.4 %; resolution decision Delta <= 32.4 um, dt <= 1.48 ps; v2.1 NOT launched",
    }
    return {
        "schema_version": "cft.pic2d.design-mini-sweep.protocol/1.0.0" if preregistered else "cft.pic2d.design-mini-sweep.protocol/0.1.0-draft",
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS_PREREGISTERED if preregistered else STATUS,
        "preregistration": preregistration,
        "designs": [{**d.to_dict(), "field_binding": f"modern/experiments/pic2d_design_mini_sweep_v1/fields/{d.design_id}/binding.json",
                     "launched_in_this_campaign": d.design_id in LAUNCH_SET} for d in design_module.SWEEP_DESIGNS],
        "domain_options": {**{domain: {"template": path.relative_to(REPOSITORY).as_posix(), "status": "draft (costed, not launched)"} for domain, path in TEMPLATES.items()},
                           option_tag(*PREREGISTERED_OPTION): {"template": REFINED_CHANNEL_TEMPLATE.relative_to(REPOSITORY).as_posix(), "status": "PREREGISTERED"}},
        "recommended_option": {"closure_runs": option_tag(*PREREGISTERED_OPTION), "grid": "33.33 um / 1.4 ps (the steady-state v4 grid; attempt-8 verdict)",
                               "plume_run": "plume-24mm for the reference design only, and only at an operating point / grid that resolves the peak (attempt 8 did not); not part of this preregistration",
                               "reason": "the closure targets are channel quantities (cusp losses, cell potentials, ionisation shares); the plume box costs 3.6-6x per transit and answers a different question (thrust, divergence); the accepted plateaus are channel-only"},
        "operating_point_policy": {
            "anode_potential_v": 300.0, "n_g0_per_m3": NEUTRAL_DENSITY_N_G0, "feed_rule": "Q_in = c n_g0, c = v_bar A_exit / 4 at the design's exit area (equal initial neutral density)",
            "electron_source": {"channel": "3 mA / 2 eV exit-plane injection (v1.x rule, template)", "plume": "cathode annulus in the exit flux tube, continuity rule (v2.0 template)"},
            "macro_weight_50um": MACRO_WEIGHT, "macro_weight_33um": "parity: 6e4 x dr dz / (50 um)^2 (26 666.7 for the reference)", "macro_weight_cap_m_particles": MAX_PROJECTED_PARTICLES_M_H100,
            "seeds": SEEDS, "cathode_annulus_rule": "r in [0.25 r_w, min(r_w, r_exit)], z in [L + 0.3 mm, L + 1.0 mm]",
        },
        "grid_policy": {"target_cell_m_50um": design_module.TARGET_CELL_M, "target_cell_m_33um": cost_module.REFINED_CHANNEL_CELL_M, "dr": "r_w / round(r_w / target)", "dz": "L / round(L / target)",
                        "dt_s_33um": cost_module.REFINED_CHANNEL_DT_S,
                        "snapped": ["exit radius", "cone start", "front-face dielectric radius = magnet inner radius", "plume radius = return-yoke outer radius", "plume length"]},
        "stopping_rule": {"plateau": "relative drift < 5 % over the trailing 20 % of elapsed simulated time for I_d, N_e and n_g (linear fit), evaluated at every checkpoint, only after >= 3 ion transits, triad inside soft bounds AND peak-Debye soft margin 2.5 (the accepted steady-state rule + the v2.0.3 preconditions, verbatim from steady-state v4)",
                          "transit_rule": "2.4 us x L / 24 mm + L_plume / 17 km/s a priori; the measured N_i / L residence supersedes it",
                          "fail_closed": "omega_pe dt gate (0.2), v2.0.3 window-mode peak-Debye gate (hard pi), triad hard drifts (0.25 after one transit), windowed residual-power >= 5 % (one-sided, from the first complete window), Courant, Poisson residual contract, neutral inventory exhaustion / ceiling",
                          "wall_budget": f"{BUDGET_FACTOR_PREREGISTERED} x projected wall to 3 transits at the H100 MPS-4 per-process rate (33 um option); {BUDGET_FACTOR} x the 5090 model (draft options)",
                          "acceptance": "per design: (a) plateau, (b) windowed residual power < +2 %, (c) closure targets extracted, verdicts closure_quotable / plateau_with_heating / no_plateau, (e) design-effect rule, (f) the convergence caveat, (g) design-specific disclosures - stopping_rule.acceptance of every sealed run protocol"},
        "replication_policy": {"base": "one run per design (seed 20260903)", "seed_replicate": "l1a-gs-v3-056-effcbc8686 (seed 20260904) - the HEMP-like design; sealed as protocols/l1a-gs-v3-056-effcbc8686-channel-33um-seed-replicate.json; the reference's seed-b / W x 0.7 pair exists (<= 1.1 % / <= 5.7 %)",
                               "decision_rule": "a design effect counts only if it exceeds the seed replicate spread AND the reference's seed/W spread", "w_replicate": "none in budget; the reference W x 0.7 record is the W-sensitivity statement"},
        "closure_targets": closure_target_table(),
        "cost_table_hours_to_3_transits": {domain: [{k: row[k] for k in ("design_id", "nodes", "dt_s", "macro_weight", "platform", "ms_per_step", "ms_per_step_rtx5090_model",
                                                                            "ms_per_step_h100_mps4_per_process", "transit_s", "steps_to_transits", "wall_hours", "device_gb_projected")}
                                                    for row in rows] for domain, rows in table.items()},
        "cost_anchors": cost_module.anchor_residuals(),
        "h100_mps4_anchor": cost_module.H100_MPS4_ANCHOR,
        "launch_projection": {design_id: {"particles_projected_m": launch_rows[design_id]["particles_projected_m"]["total_m"], "ms_per_step_h100_mps4": launch_rows[design_id]["ms_per_step"],
                                          "steps_to_3_transits": launch_rows[design_id]["steps_to_transits"], "hours_to_3_transits_at_mps4_rate": launch_rows[design_id]["wall_hours"],
                                          "device_gb_projected": launch_rows[design_id]["device_gb_projected"]} for design_id in LAUNCH_SET},
        "recommended_schedule": recommended,
        "refined_grid_schedule": refined,
        "preflight_summary": preflight_summary,
    }


def write_experiment_protocol(path: Path = DRAFT_PROTOCOL_PATH, *, preflight_summary: dict[str, Any] | None = None, records: dict[str, Path] | None = None,
                              sealed: dict[str, str] | None = None, preregistered: bool = True) -> Path:
    document = experiment_protocol_document(preflight_summary=preflight_summary, records=records, sealed=sealed, preregistered=preregistered)
    path.write_bytes(protocol_bytes(document))
    return path


# backwards-compatible names (the draft's)
draft_protocol_document = experiment_protocol_document
write_draft_protocol = write_experiment_protocol


__all__ = ["BUDGET_FACTOR", "BUDGET_FACTOR_PREREGISTERED", "CASES", "CONVERGENCE_CAVEAT", "DRAFT_PROTOCOL_PATH", "EXPERIMENT_ID", "GPU_RECORD", "GRID_VARIANTS",
           "LAUNCH_SET", "MAX_PROJECTED_PARTICLES_M_H100", "PREREGISTERED_OPTION", "PROTOCOLS_DIR", "REFINED_CHANNEL_TEMPLATE", "SEEDS", "STATUS", "STATUS_PREREGISTERED",
           "TEMPLATES", "admissible_dt", "build_protocol", "compose_all", "compose_run_protocol", "composed_protocol_path", "connected_cathode_annulus",
           "draft_protocol_document", "experiment_protocol_document", "load_template", "macro_weight_for", "option_tag", "protocol_bytes", "template_path",
           "write_draft_protocol", "write_experiment_protocol"]
