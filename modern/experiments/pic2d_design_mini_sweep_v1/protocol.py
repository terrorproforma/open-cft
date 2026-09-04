"""Per-design run protocols composed from the accepted pic2d templates (DRAFT: not preregistered).

``build_protocol(design_id, domain)`` takes the template protocol of the domain option

* ``channel``    - ``pic2d_cft_steady_state_v3/protocol.json`` (model v1.4 channel-only: the v1.3 accepted plateau
                   physics + wall-ion recycling, peak-node Debye gate, grid-heating triad, CUDA-graph step);
* ``plume-12mm`` - ``pic2d_cft_plume_v1/protocol.json`` (model v2.0.2, 12 mm plume box, channel-connected cathode,
                   continuity rule, plume-boundary gate);
* ``plume-24mm`` - ``pic2d_cft_plume_v2_1/protocol.json`` (model v2.1: 24 mm box on the padded field, prepared not launched)

and substitutes ONLY the design-dependent blocks under the declared operating-point policy:

* geometry / grid: ``designs.pic_geometry`` (dr, dz, snapped radii; the snapping record is copied in);
* neutral feed: ``Q_in = c n_g0`` with ``c = v_bar A_exit / 4`` at the design's (snapped) exit area, n_g0 = 5.5e19 m^-3
  (equal initial neutral density and equal null-collision ceiling headroom for every design);
* cathode annulus (plume options): r in [0.25 r_w, min(r_w, r_exit)], z in [L + 0.3 mm, L + 1.0 mm] - the reference's
  own values (0.5-2.0 mm, 24.3-25.0 mm) reproduced exactly; the preflight re-traces its connectivity;
* transit time (budget): 2.4 us x L_channel / 24 mm (the measured reference residence scaled with the channel
  length) + L_plume / 17 km/s; plume-gate arming = one transit;
* wall budget: 1.25 x the projected wall to 3 transits (cost model), rounded up to 10 min, never below 1 h;
* macro weight: 6e4 (the accepted runs) unless the projected particle count exceeds 8 M, then scaled so it does not
  (disclosed in ``case.macro_weight_policy``).

Everything else (anode potential 300 V, seed plasma, dt, gates, series / checkpoint / window cadence, plateau rule,
frame recorder) is the template's, verbatim.  The template's field declarations are replaced by ``field_binding``
(the hash-bound per-design field of ``fields.py``; the runner receives the node map directly).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from cft_revival.pic2d.neutrals import feed_for_density

from . import cost as cost_module
from . import designs as design_module
from .designs import DOMAIN_OPTIONS, MODERN, REPOSITORY, BuiltDesign, PicMapping, exit_area_m2, pic_geometry

EXPERIMENT_ID = "pic2d-design-mini-sweep-v1"
STATUS = "DRAFT_NOT_PREREGISTERED_NO_PRODUCTION_LAUNCH"
TEMPLATES = {
    "channel": MODERN / "experiments" / "pic2d_cft_steady_state_v3" / "protocol.json",
    "plume-12mm": MODERN / "experiments" / "pic2d_cft_plume_v1" / "protocol.json",
    "plume-24mm": MODERN / "experiments" / "pic2d_cft_plume_v2_1" / "protocol.json",
}
NEUTRAL_DENSITY_N_G0 = 5.5e19
NEUTRAL_TEMPERATURE_K = 300.0
MACRO_WEIGHT = 60000.0
MAX_PROJECTED_PARTICLES_M = 8.0
BUDGET_FACTOR = 1.25
CATHODE_R_INNER_OVER_RW = 0.25
CATHODE_Z_START_BEHIND_EXIT_M = 0.0003
CATHODE_Z_END_BEHIND_EXIT_M = 0.001
CATHODE_SHRINK_LADDER = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
OMEGA_CE_MARGIN = 0.95            # keep omega_ce dt at the map's max |B| below 0.95 x the gate (0.2)
DT_QUANTUM_S = 1.0e-13
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
DRAFT_PROTOCOL_PATH = Path(__file__).resolve().parent / "protocol.json"


def load_template(domain: str) -> dict[str, Any]:
    if domain not in DOMAIN_OPTIONS:
        raise ValueError(f"unknown domain option {domain!r}")
    return json.loads(TEMPLATES[domain].read_text(encoding="utf-8"))


def _strip_notes(block: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in block.items() if not k.endswith(("_note", "_justification", "_statement"))}


def macro_weight_for(projected_total_m: float) -> tuple[float, str]:
    if projected_total_m <= MAX_PROJECTED_PARTICLES_M:
        return MACRO_WEIGHT, "W = 6e4 (the accepted runs; equal particles per cell at equal density across designs)"
    scaled = MACRO_WEIGHT * projected_total_m / MAX_PROJECTED_PARTICLES_M
    return float(scaled), f"W scaled to cap the projected particle count at {MAX_PROJECTED_PARTICLES_M} M (disclosed; the W x 0.7 pair bounds the W sensitivity at <= 5.7 % on I_d)"


def admissible_dt(template_dt_s: float, max_b_t: float, limit: float, *, margin: float = OMEGA_CE_MARGIN) -> tuple[float, dict[str, Any]]:
    """The template dt unless omega_ce dt at the map's max |B| would exceed ``margin x limit``; then the largest 0.1 ps multiple below it."""

    omega_ce = ELEMENTARY_CHARGE_C * float(max_b_t) / ELECTRON_MASS_KG
    at_template = omega_ce * template_dt_s
    if at_template <= margin * limit:
        return float(template_dt_s), {"rule": f"template dt kept: omega_ce dt {at_template:.3f} <= {margin} x {limit}", "omega_ce_dt_at_template": at_template, "max_b_t": float(max_b_t)}
    dt = math.floor(margin * limit / omega_ce / DT_QUANTUM_S) * DT_QUANTUM_S
    return float(dt), {"rule": f"dt reduced so omega_ce dt <= {margin} x {limit} at the map's max |B| (pole faces of the plume front face), 0.1 ps quantum",
                       "omega_ce_dt_at_template": at_template, "omega_ce_dt": omega_ce * dt, "template_dt_s": float(template_dt_s), "max_b_t": float(max_b_t)}


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


def build_protocol(design_id: str, domain: str, *, built: BuiltDesign | None = None, field_map=None,
                   target_cell_m: float | None = None, dt_s: float | None = None) -> tuple[dict[str, Any], PicMapping]:
    """Compose the per-design run protocol.  With ``field_map`` (the design's node field) the dt rule and the cathode placement
    search run on the actual field; without it the template dt and the base annulus are used (the preflight passes the map).
    ``target_cell_m`` / ``dt_s`` override the template grid spacing / time step (the attempt-8 grid-refinement variant:
    33.3 um / 1.4 ps); both are recorded in ``case.grid_policy``."""

    built = design_module.build_design(design_id) if built is None else built
    mapping = pic_geometry(built, domain) if target_cell_m is None else pic_geometry(built, domain, target_cell_m=target_cell_m)
    template = load_template(domain)
    protocol = copy.deepcopy(template)
    if dt_s is not None:
        protocol["numerics"]["dt_s"] = float(dt_s)
    budget_keys = [key for key in protocol if key.startswith("budget")]
    for key in budget_keys + ["field_authority", "field_plume_extension", "lineage", "v2_reference", "validation_v1_observable"]:
        protocol.pop(key, None)
    protocol["schema_version"] = "cft.pic2d.design-mini-sweep.run-protocol/0.1.0-draft"
    protocol["experiment_id"] = EXPERIMENT_ID
    protocol["status"] = STATUS
    protocol["template_protocol"] = {"domain_option": domain, "path": TEMPLATES[domain].relative_to(REPOSITORY).as_posix(),
                                     "experiment_id": template.get("experiment_id"), "model_version": template.get("model_version")}
    protocol["design_id"] = design_id
    protocol["design"] = built.design.to_dict()
    protocol["design_identity"] = built.identity
    protocol["field_binding"] = f"modern/experiments/pic2d_design_mini_sweep_v1/fields/{design_id}/binding.json"
    protocol["field_source"] = ("existing reference artifacts (authority level-1 / padding-1.5) through the v2.0 / v2.1 field pipeline"
                                if design_id == design_module.REFERENCE_DESIGN_ID else
                                "design-mini-sweep binding: direct node evaluation of the design's padded level-0 material-aware P2 checkpoint, scaled by source_strength_scale")
    # geometry / grid
    geometry = mapping.geometry.to_dict()
    geometry["source"] = f"catalogue design {design_id} mapped onto the PIC straight-bore + cone (+ plume) representation; snaps recorded"
    geometry["snaps"] = mapping.snaps
    protocol["geometry"] = geometry
    cost = cost_module.design_cost(mapping, dt_s=float(protocol["numerics"]["dt_s"]))
    weight, weight_policy = macro_weight_for(cost["particles_projected_m"]["total_m"])
    case = protocol["case"]
    grid_tag = "" if target_cell_m is None else f"-{target_cell_m*1e6:.0f}um"
    case.update({
        "id": f"{design_id}-{domain}{grid_tag}-w{weight:.3g}-ng0-{NEUTRAL_DENSITY_N_G0:.2g}-seed5e16-draft",
        "radial_cells": int(mapping.grid.radial_cells), "axial_cells": int(mapping.grid.axial_cells), "macro_weight": weight,
        "macro_weight_policy": weight_policy, "seed": int(case.get("seed", 20260903)),
        "grid_policy": {"target_cell_m": mapping.snaps["target_cell_m"], "dt_s": float(protocol["numerics"]["dt_s"]),
                        "source": "template (50 um / 1.5 ps)" if target_cell_m is None and dt_s is None else "override (attempt-8 grid-refinement variant: Delta <= 32.4 um, dt <= 1.48 ps)"},
        "grid_note": f"dr {mapping.grid.dr_m*1e6:.2f} um x dz {mapping.grid.dz_m*1e6:.2f} um: {mapping.grid.radial_cells} x {mapping.grid.axial_cells} cells, nodes {mapping.grid.node_shape}; bore {mapping.snaps['bore_cells']} cells",
    })
    for key in ("seed_note",):
        case.pop(key, None)
    # operating point: feed for n_g0 at the design's exit area
    operating = protocol["operating_point"]
    area = exit_area_m2(mapping)
    feed = feed_for_density(NEUTRAL_DENSITY_N_G0, area, NEUTRAL_TEMPERATURE_K)
    inventory = operating["neutral_inventory"]
    inventory["feed_atoms_per_s"] = float(feed)
    inventory["feed_justification"] = (f"Q_in = c n_g0 with c = v_bar A_exit / 4 (xenon 300 K), A_exit = pi ({mapping.geometry.exit_radius_m*1e3:.3f} mm)^2 = {area:.4e} m^2, "
                                       f"n_g0 = {NEUTRAL_DENSITY_N_G0:.2e} m^-3 -> {feed:.4e} atoms/s (equal initial neutral density for every design; the reference's 8.551e16 /s reproduced)")
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
        cost["steps_to_transits"] = cost["transits"] * cost["transit_s"] / dt
        cost["stepping_hours"] = cost["steps_to_transits"] * cost["ms_per_step"] / 3.6e6
        cost["wall_hours"] = cost["stepping_hours"] + (cost["factorisation_s"] + cost["field_map_s"]) / 3600.0
    # stopping rule: plateau verbatim; budget from the cost model
    rule = protocol["stopping_rule"]
    wall_budget = max(3600.0, math.ceil(BUDGET_FACTOR * cost["wall_hours"] * 3600.0 / 600.0) * 600.0)
    rule["wall_budget_seconds"] = float(wall_budget)
    rule["wall_budget_note"] = (f"{BUDGET_FACTOR} x the projected wall to 3 transits ({cost['wall_hours']:.1f} h at {cost['ms_per_step']:.2f} ms/step, "
                                f"{cost['steps_to_transits']/1e6:.2f} M steps), rounded up to 10 min; cumulative over resumes")
    rule["plateau"] = rule["plateau"].split(";")[0] + f"; may only be declared after >= 3 ion transit times (3 x {cost['transit_s']*1e6:.2f} us); the grid-heating triad must be inside its soft bounds"
    protocol["budget_design_mini_sweep"] = {
        "ion_transit_time_s": float(cost["transit_s"]),
        "ion_transit_note": "2.4 us x L_channel / 24 mm (measured reference residence scaled with the channel length) + L_plume / 17 km/s; replaced by the measured N_i / L residence at the first checkpoint past 1 us when the run reports it",
        "particles_projected_m": cost["particles_projected_m"], "ms_per_step_projected": cost["ms_per_step"], "steps_to_3_transits": cost["steps_to_transits"],
        "hours_to_3_transits_projected": cost["wall_hours"], "factorisation_s_projected": cost["factorisation_s"], "device_gb_projected": cost["device_gb_projected"],
        "macro_weight": weight, "dr_m": mapping.grid.dr_m, "dz_m": mapping.grid.dz_m, "dt_s": float(protocol["numerics"]["dt_s"]),
        "cost_model": "experiments.pic2d_design_mini_sweep_v1.cost (anchors: steady-state v2 base / W x 0.7, plume attempt 8)",
    }
    inherited = protocol.get("claim_boundary")
    inherited_block = inherited if isinstance(inherited, dict) else {"template_claim_boundary": inherited}
    protocol["claim_boundary"] = {
        **inherited_block,
        "draft": "DRAFT protocol composed from the template; NOT preregistered; no production launch until the preregistration commit that follows attempt 8's plateau verdict",
        "geometry_approximation": "catalogue radii snapped to the 50 um grid (geometry.snaps); the clearance gap between the dielectric tube and the magnets is treated as dielectric front face",
        "field": "material-aware level-0 P2 field of the design (linear iron), scaled by the L1a source_strength_scale; not P2-qualified for these designs",
    }
    return protocol, mapping


def draft_protocol_document(*, preflight_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """The experiment-level DRAFT protocol.json: design list, field bindings, policies, closure targets, cost table."""

    from .closure import closure_target_table

    built = [design_module.build_design(d.design_id) for d in design_module.SWEEP_DESIGNS]
    table = cost_module.cost_table(built)
    recommended = cost_module.serial_schedule(table, option="channel", replicate_design_ids=("l1a-gs-v3-056-effcbc8686",),
                                              extra=((design_module.REFERENCE_DESIGN_ID, "plume-24mm"),))
    refined = cost_module.serial_schedule(table, option=cost_module.REFINED_CHANNEL_KEY, replicate_design_ids=("l1a-gs-v3-056-effcbc8686",))
    return {
        "schema_version": "cft.pic2d.design-mini-sweep.protocol/0.1.0-draft",
        "experiment_id": EXPERIMENT_ID,
        "status": STATUS,
        "preregistration": {
            "state": "NOT preregistered; this file is a DRAFT and is not hash-bound to any run",
            "attempt_8_verdict": "recorded in ac248e05 (2026-09-04 11:57 AEST): grid-heating triad gate stop at 4.98 us (S drift 0.253), NO plateau; "
                                 "finite-grid heating from ~2.4 us once the peak-node Delta/lambda_D crossed ~3.2 (CIC threshold pi); the accepted channel-only "
                                 "plateau sat at 3.17 with a residual closing to +0.4 %; resolution decision Delta <= 32.4 um, dt <= 1.48 ps; v2.1 NOT launched",
            "blocked_on": "the operating-point / grid decision that follows the verdict: the channel-only 33 um / 1.4 ps grid-refinement run (or a lower operating point at 50 um) and the recalibrated peak-Debye / residual-power gates",
            "before_prereg": ["whole-set preflight over every design (fields, mesh, connectivity, stability) - the L1b lesson", "labelled non-evidentiary shakedown of one design on the real field through run/finalize/targets",
                              "decide the open decisions listed in the README", "seal: protocol semantic hash, field bindings, template protocols, code hashes"],
        },
        "designs": [{**d.to_dict(), "field_binding": f"modern/experiments/pic2d_design_mini_sweep_v1/fields/{d.design_id}/binding.json"} for d in design_module.SWEEP_DESIGNS],
        "domain_options": {domain: {"template": path.relative_to(REPOSITORY).as_posix()} for domain, path in TEMPLATES.items()},
        "recommended_option": {"closure_runs": "channel", "grid": "the grid the attempt-8 follow-up validates: 50 um / 1.5 ps if the recalibrated peak-Debye gate admits the design's peak, else the 33 um / 1.4 ps refinement variant (cost row channel-33um-1.4ps)",
                               "plume_run": "plume-24mm for the reference design only, and only at an operating point / grid that resolves the peak (attempt 8 did not)",
                               "reason": "the closure targets are channel quantities (cusp losses, cell potentials, ionisation shares); the plume box costs 3.6-6x per transit and answers a different question (thrust, divergence); the accepted plateaus are channel-only"},
        "operating_point_policy": {
            "anode_potential_v": 300.0, "n_g0_per_m3": NEUTRAL_DENSITY_N_G0, "feed_rule": "Q_in = c n_g0, c = v_bar A_exit / 4 at the design's exit area (equal initial neutral density)",
            "electron_source": {"channel": "3 mA / 2 eV exit-plane injection (v1.x rule, template)", "plume": "cathode annulus in the exit flux tube, continuity rule (v2.0 template)"},
            "macro_weight": MACRO_WEIGHT, "macro_weight_cap_m_particles": MAX_PROJECTED_PARTICLES_M, "seed": 20260903,
            "cathode_annulus_rule": "r in [0.25 r_w, min(r_w, r_exit)], z in [L + 0.3 mm, L + 1.0 mm]",
        },
        "grid_policy": {"target_cell_m": design_module.TARGET_CELL_M, "dr": "r_w / round(r_w / 50 um)", "dz": "L / round(L / 50 um)",
                        "snapped": ["exit radius", "cone start", "front-face dielectric radius = magnet inner radius", "plume radius = return-yoke outer radius", "plume length"]},
        "stopping_rule": {"plateau": "relative drift < 5 % over the trailing 20 % of elapsed simulated time for I_d, N_e and n_g (linear fit), evaluated at every checkpoint, only after >= 3 ion transits, triad inside soft bounds (the accepted steady-state rule, verbatim)",
                          "transit_rule": "2.4 us x L / 24 mm + L_plume / 17 km/s a priori; the measured N_i / L residence supersedes it",
                          "wall_budget": f"{BUDGET_FACTOR} x projected wall to 3 transits"},
        "replication_policy": {"base": "one run per design (seed 20260903)", "seed_replicate": "l1a-gs-v3-056-effcbc8686 (seed 20260904) - the HEMP-like design; the reference's seed-b / W x 0.7 pair exists (<= 1.1 % / <= 5.7 %)",
                               "decision_rule": "a design effect counts only if it exceeds the seed replicate spread AND the reference's seed/W spread", "w_replicate": "none in budget; the reference W x 0.7 record is the W-sensitivity statement"},
        "closure_targets": closure_target_table(),
        "cost_table_hours_to_3_transits": {domain: [{k: row[k] for k in ("design_id", "nodes", "dt_s", "ms_per_step", "transit_s", "steps_to_transits", "wall_hours", "device_gb_projected")} for row in rows] for domain, rows in table.items()},
        "cost_anchors": cost_module.anchor_residuals(),
        "recommended_schedule": recommended,
        "refined_grid_schedule": refined,
        "preflight_summary": preflight_summary,
    }


def write_draft_protocol(path: Path = DRAFT_PROTOCOL_PATH, *, preflight_summary: dict[str, Any] | None = None) -> Path:
    document = draft_protocol_document(preflight_summary=preflight_summary)
    path.write_bytes(json.dumps(document, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    return path


__all__ = ["DRAFT_PROTOCOL_PATH", "EXPERIMENT_ID", "STATUS", "TEMPLATES", "admissible_dt", "build_protocol", "connected_cathode_annulus",
           "draft_protocol_document", "load_template", "macro_weight_for", "write_draft_protocol"]
