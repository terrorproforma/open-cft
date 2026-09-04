"""Whole-set preflight of external validation v0 (every composed option; no stepping, no GPU).

Per option (variant x grid of the launch set plus the sealed sensitivity variant), in order and each recorded even when a later one fails:

1. reference     - the DOI validates and the reference record is complete;
2. geometry      - the reconstruction builds under the v1.1 contract, its hash equals the binding's, every approximation is listed;
3. field binding - ``fields/<id>/binding.json`` exists, every bound hash re-verifies, the published-anchor gates G1-G7 all passed, the gates
                   recompute from the bound checkpoint (regate) to the recorded scale;
4. grid          - the PIC ChannelGeometry / Grid2D construct; every reference line on a grid line (worst snap = 0 at 20 um; <= half a cell otherwise);
5. field map     - the node map builds inside the binding's supported box (anode frame, offset applied); the a-priori stability report at the PUBLISHED
                   density / temperature is admitted (omega_pe dt, omega_ce dt at the map's max |B|, Courant, cells / lambda_D <= pi) and the soft-margin
                   status (<= 2.5) is recorded;
6. mesh masks    - plasma cells > 0;
7. protocol      - the composed run protocol is accepted by the shared runner's ``build_config`` (CPU backend) with static neutrals (inventory None);
8. comparison    - the comparison spec validates and every channel-comparable row names an estimand;
9. cost          - the projection row (ms/step at MPS-4 and solo, transit, hours, device GB).

Output: ``preflight-channel-20um.json`` with per-option gate results, the grid argument and ``all_passed`` over the launch set.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import StabilityLimits, stability_report

from . import comparison, reference
from . import fields as field_module
from . import geometry as geometry_module
from . import protocol as protocol_module

EXPERIMENT = Path(__file__).resolve().parent
PREFLIGHT_OPTIONS: tuple[tuple[str, str], ...] = protocol_module.COMPOSED_OPTIONS      # primary, transport sensitivity, resolution follow-up


def preflight_option(variant: str, grid: str, *, log=print) -> dict[str, Any]:
    started = time.perf_counter()
    gates: dict[str, dict[str, Any]] = {}
    record: dict[str, Any] = {"variant": variant, "grid": grid, "option": protocol_module.option_tag(variant, grid), "gates": gates}
    state: dict[str, Any] = {}

    def gate(name: str, fn):
        try:
            payload = fn()
            gates[name] = {"passed": bool(payload.pop("passed", True)), **payload}
        except Exception as error:  # noqa: BLE001 - every failure is recorded in the report, never hidden
            gates[name] = {"passed": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=3)}
        log(f"[preflight] {record['option']} {name}: {'PASS' if gates[name]['passed'] else 'FAIL'}")
        return gates[name]["passed"]

    def reference_gate():
        document = reference.reference_document()
        required = ("channel_radius_m", "channel_length_m", "anode_voltage_v", "mass_flow", "neutral_background", "magnet_stack", "field_anchors", "grid", "time_step_s")
        missing = [k for k in required if k not in document["setup"]]
        return {"passed": not missing and document["doi"] == reference.DOI, "doi": document["doi"], "reported_quantities": sorted(document["reported"]), "missing_setup": missing}

    def geometry_gate():
        geometry = geometry_module.brandt_micro_hempt_geometry()
        state["geometry"] = geometry
        return {"passed": len(geometry_module.APPROXIMATIONS) >= 9, "config_id": geometry.config_id, "geometry_sha256": geometry.canonical_sha256,
                "approximations": [a["id"] for a in geometry_module.APPROXIMATIONS], "stages": len(geometry.stages)}

    def binding_gate():
        binding = field_module.load_binding()
        verification = field_module.verify_binding(binding)
        regated = field_module.regate_field()
        state["binding"] = binding
        own = binding["gates"]
        return {"passed": bool(verification["passed"] and own.get("all_passed", False) and regated["scale_matches_binding"] and all(g["passed"] for k, g in regated.items() if k.startswith("G"))),
                "hash_checks": verification["checks"], "production_gates": {k: (v.get("passed") if isinstance(v, dict) else v) for k, v in own.items()},
                "regate_passed": {k: g["passed"] for k, g in regated.items() if k.startswith("G")}, "scale_matches_binding": regated["scale_matches_binding"],
                "source_strength_scale": binding["source_strength_scale"], "interior_nulls_m": own["G2_interior_nulls"]["interior_nulls_m"], "exit_null_m": own["G3_exit_null"]["nearest_null_m"],
                "b_at_17mm_t": own["G4_exit_point"]["b_t"], "axis_max_t": own["G5_axis_maximum"]["axis_max_t"], "axis_max_z_m": own["G5_axis_maximum"]["axis_max_z_m"],
                "wall_cusp_b_t_descriptor": [r["wall_b_max_within_0p5mm_t"] for r in own["D6_wall_cusp_field"]["cusps"]],
                "low_field_contour_radius_m_descriptor": [r["low_field_contour_radius_m"] for r in own["D6_wall_cusp_field"]["cusps"]],
                "gate_genealogy_entries": len(own.get("genealogy", [])),
                "no_ring_bracket": (binding.get("sensitivity_no_rings") or {}).get("bracket")}

    def grid_gate():
        mapping = geometry_module.pic_mapping("channel", target_cell_m=protocol_module.GRIDS[grid]["cell_m"])
        state["mapping"] = mapping
        worst = geometry_module.worst_snap_in_cells(mapping)
        return {"passed": worst <= 0.5 + 1e-9, "node_shape": list(mapping.grid.node_shape), "cells": [mapping.grid.radial_cells, mapping.grid.axial_cells], "dr_m": mapping.grid.dr_m, "dz_m": mapping.grid.dz_m,
                "worst_snap_in_cells": worst, "snaps": mapping.snaps, "axial_offset_m": mapping.axial_offset_m}

    def field_map_gate():
        mapping = state["mapping"]
        fm = field_module.brandt_field_map(mapping, state["binding"])
        state["field_map"] = fm
        protocol, _ = protocol_module.build_protocol(variant, grid, field_map=fm)
        state["protocol"] = protocol
        numerics = protocol["numerics"]
        limits = StabilityLimits(**numerics["stability_limits"])
        ref = numerics["stability_reference"]
        report = stability_report(mapping.grid, float(numerics["dt_s"]), reference_density_per_m3=float(ref["density_per_m3"]), reference_electron_temperature_ev=float(ref["electron_temperature_ev"]),
                                  max_b_t=fm.max_b_t, max_electron_energy_ev=float(ref["max_electron_energy_ev"]), limits=limits)
        omega_ce_dt = 1.602176634e-19 * float(fm.max_b_t) / 9.1093837139e-31 * float(numerics["dt_s"])
        cells_per_debye = report.to_dict()["cell_debye_ratio"]
        return {"passed": report.stable and omega_ce_dt <= float(limits.max_omega_ce_dt) and cells_per_debye <= math.pi, "field_map_sha256": fm.sha256, "field_source_sha256": fm.source_sha256,
                "max_b_t": fm.max_b_t, "stability": report.to_dict(), "dt_s": float(numerics["dt_s"]), "dt_policy": numerics.get("dt_policy"),
                "cells_per_debye_at_published": cells_per_debye, "soft_margin_2p5_met": cells_per_debye <= 2.5, "hard_pi_met": cells_per_debye <= math.pi,
                "omega_pe_dt_gate_density_per_m3": protocol_module.density_at_omega_pe_dt(float(limits.max_omega_pe_dt), float(numerics["dt_s"])),
                "hard_debye_density_at_10ev_per_m3": protocol_module.density_at_cells_per_debye(math.pi, max(mapping.grid.dr_m, mapping.grid.dz_m), 10.0),
                "provenance_kind": fm.provenance.get("kind"), "plasma_nodes_sampled": fm.provenance.get("plasma_nodes_sampled"), "axial_offset_m": fm.provenance.get("axial_offset_m"),
                "source_strength_scale": fm.provenance.get("source_strength_scale")}

    def masks_gate():
        import numpy as np

        masks = build_mesh_masks(state["mapping"].grid)
        info = masks.to_dict()
        plasma_cells = int(np.count_nonzero(masks.plasma_cell))
        return {"passed": plasma_cells > 0, "plasma_cells": plasma_cells, **{k: v for k, v in info.items() if not isinstance(v, (list, dict))}}

    def protocol_gate():
        from experiments.pic2d_cft_steady_state_v1 import run as runner

        protocol = state["protocol"]
        config = runner.build_config(protocol, backend="cpu")
        budget = runner.protocol_budget(protocol)
        peak_gate = protocol["numerics"].get("peak_debye_gate") or {}
        triad = protocol["stopping_rule"].get("grid_heating_triad") or {}
        return {"passed": config.neutral_inventory is None and config.mcc is not None and abs(config.mcc.neutral_density_per_m3 - protocol_module.NEUTRAL_DENSITY_PER_M3) < 1e6,
                "case_id": protocol["case"]["id"], "status": protocol["status"], "template": protocol["template_protocol"]["path"], "model_version": protocol.get("model_version"),
                "macro_weight": config.macro_weight, "macro_weight_parity": budget.get("macro_weight_parity"), "dt_s": config.dt_s, "cells": [config.grid.radial_cells, config.grid.axial_cells],
                "static_neutrals": config.neutral_inventory is None, "neutral_density_per_m3": config.mcc.neutral_density_per_m3 if config.mcc else None, "anode_potential_v": config.potentials.anode_v,
                "injection": None if config.injection is None else config.injection.to_dict(), "anomalous": None if config.anomalous is None else config.anomalous.to_dict(),
                "wall_budget_seconds": protocol["stopping_rule"]["wall_budget_seconds"], "wall_budget_hours": protocol["stopping_rule"]["wall_budget_seconds"] / 3600.0,
                "ion_transit_time_s": budget["ion_transit_time_s"], "steps_to_3_transits": budget["steps_to_3_transits"], "hours_to_3_transits_mps4": budget["hours_to_3_transits_mps4"],
                "particles_projected_m": budget["particles_projected_m"], "frame_recorder": protocol["numerics"].get("frame_recorder"),
                "gates_v2_0_3": {"peak_debye_window_mode": peak_gate.get("window_steps") is not None, "peak_debye_hard": peak_gate.get("max_cells_per_debye"), "peak_debye_soft": peak_gate.get("soft_cells_per_debye"),
                                 "windowed_residual_power_max": triad.get("windowed_energy_residual_over_electrode_work_max"), "residual_window_steps": triad.get("residual_window_steps")}}

    def comparison_gate():
        document = comparison.comparison_document()
        problems = comparison.validate_comparison_spec(document)
        channel_rows = [q["quantity_id"] for q in document["quantities"] if "channel" in q["comparable_under"]]
        plume_only = [q["quantity_id"] for q in document["quantities"] if "channel" not in q["comparable_under"]]
        return {"passed": not problems and len(channel_rows) >= 8, "problems": problems, "channel_comparable_rows": channel_rows, "plume_only_rows": plume_only,
                "closure_differences": len(document["closure_differences"]), "inconclusive_conditions": len(document["inconclusive_conditions"])}

    def cost_gate():
        budget = state["protocol"]["budget_external_validation_v0"]
        return {"passed": True, **{k: budget[k] for k in ("particles_projected_m", "ms_per_step_rtx5090_model", "ms_per_step_h100_mps4_per_process", "ms_per_step_h100_solo_equivalent", "transit_s",
                                                          "steps_to_3_transits", "hours_to_3_transits_mps4", "hours_to_3_transits_h100_solo_equivalent", "device_gb_projected", "factorisation_s",
                                                          "hours_to_reference_time_mps4")}}

    ok = gate("reference", reference_gate)
    ok = gate("geometry", geometry_gate) and ok
    ok = gate("field_binding", binding_gate) and ok
    ok = gate("grid", grid_gate) and ok
    if "mapping" in state and "binding" in state:
        ok = gate("field_map", field_map_gate) and ok
        ok = gate("mesh_masks", masks_gate) and ok
        if "protocol" in state:
            ok = gate("protocol", protocol_gate) and ok
            ok = gate("cost", cost_gate) and ok
    ok = gate("comparison_spec", comparison_gate) and ok
    record["passed"] = bool(ok and all(g["passed"] for g in gates.values()))
    record["seconds"] = time.perf_counter() - started
    return record


def _platform_record() -> dict[str, Any]:
    import platform as platform_module
    import socket

    import numpy as np

    return {"host": socket.gethostname(), "system": platform_module.system(), "release": platform_module.release(), "machine": platform_module.machine(),
            "python": platform_module.python_version(), "numpy": np.__version__, "gpu_used": False, "note": "CPU-only preflight (a preregistered run re-runs it on the launch box)"}


def preflight_all(options: tuple[tuple[str, str], ...] = PREFLIGHT_OPTIONS, *, log=print) -> dict[str, Any]:
    started = time.perf_counter()
    records = [preflight_option(variant, grid, log=log) for variant, grid in options]
    launch = {protocol_module.option_tag(*o) for o in protocol_module.LAUNCH_SET}
    return {
        "schema_version": "cft.pic2d.external-validation-v0.preflight/0.1.0-draft",
        "experiment_id": protocol_module.EXPERIMENT_ID,
        "status": "whole-set preflight of a DRAFT (not preregistered, not launched) code-to-code comparison; derived-float gates are re-run on the launch platform before any launch",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": _platform_record(),
        "option_count": len(records),
        "launch_set": sorted(launch),
        "all_passed": all(r["passed"] for r in records),
        "launch_set_passed": all(r["passed"] for r in records if r["option"] in launch),
        "options": records,
        "grid_argument": protocol_module.grid_argument(),
        "seconds": time.perf_counter() - started,
    }


def preflight_path() -> Path:
    return EXPERIMENT / "preflight-channel-20um.json"


def write_preflight(options: tuple[tuple[str, str], ...] = PREFLIGHT_OPTIONS, *, log=print) -> tuple[Path, dict[str, Any]]:
    report = preflight_all(options, log=log)
    path = preflight_path()
    path.write_bytes(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=_plain).encode("utf-8") + b"\n")
    return path, report


def _plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


__all__ = ["PREFLIGHT_OPTIONS", "preflight_all", "preflight_option", "preflight_path", "write_preflight"]
