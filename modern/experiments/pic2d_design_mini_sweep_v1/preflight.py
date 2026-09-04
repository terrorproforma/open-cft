"""Whole-set preflight of the design mini-sweep (every design, one domain option; no stepping, no GPU).

The L1b v1 lesson: every cheap deterministic gate must run over the WHOLE design set before any preregistration
commit.  Per design and domain option, in order and each recorded even when a later one fails:

1. identity      - the accepted geometry rebuilds and its hashes equal the sealed authorities;
2. field binding - ``fields/<design>/binding.json`` exists, every bound hash re-verifies, its own production gates
                   (mesh angle, convergence, coverage, topology agreement, L1b agreement) all passed;
3. grid          - the PIC ChannelGeometry / Grid2D construct (bore on a grid line, exit plane on a grid line, ...);
                   every snap error <= half a cell;
4. field map     - the node map builds on the PIC grid inside the binding's supported box; the a-priori stability report
                   (omega_pe dt at the reference density, omega_ce dt at the map's max |B|, Courant, Debye ratio) is admitted
                   with the template's limits;
5. mesh masks    - plasma cells > 0, far-field / body-face nodes present for plume boxes;
6. protocol      - the per-design run protocol composes and ``runner.build_config`` accepts it (CPU backend, no device);
7. connectivity  - plume options: the cathode annulus is channel-connected at the required fraction on the design's own field
                   (the runner's gate, run here so a failure is known before any launch);
8. cost          - the projection row (ms/step, transit, hours, device GB).

Output: ``preflight-<domain>.json`` (50 um) or ``preflight-<domain>-<grid>.json`` (``preflight-channel-33um.json`` for the
preregistered option) with per-design gate results and ``all_passed``.  The field-map gate records the platform-independent
``field_source_sha256`` next to the platform's ``field_map_sha256``; the protocol gate records the composed dt (the design's
dt admissibility at its own map maximum), the parity macro weight and the wall budget; the run refuses to start unless the
preflight of ITS option passed on the launch box (PLAN.md s.6.5: derived-float gates are re-run where the launch happens).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
from typing import Any

from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import StabilityLimits, stability_report

from . import cost as cost_module
from . import designs as design_module
from . import fields as field_module
from .protocol import GRID_VARIANTS, build_protocol, option_tag

EXPERIMENT = Path(__file__).resolve().parent


def preflight_design(design_id: str, domain: str, *, grid: str = "50um", log=print) -> dict[str, Any]:
    started = time.perf_counter()
    gates: dict[str, dict[str, Any]] = {}
    record: dict[str, Any] = {"design_id": design_id, "domain": domain, "grid": grid, "option": option_tag(domain, grid), "gates": gates}
    target_cell_m, dt_override = GRID_VARIANTS[grid]

    def gate(name: str, fn):
        try:
            payload = fn()
            gates[name] = {"passed": bool(payload.pop("passed", True)), **payload}
        except Exception as error:  # recorded, never hidden
            gates[name] = {"passed": False, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=3)}
        log(f"[preflight] {design_id} {domain} {name}: {'PASS' if gates[name]['passed'] else 'FAIL'}")
        return gates[name]["passed"]

    state: dict[str, Any] = {}

    def identity():
        built = design_module.build_design(design_id)
        state["built"] = built
        checks = built.identity.get("identity_checks", {"reference": True})
        return {"passed": all(checks.values()), "identity": built.identity}

    def binding():
        value = field_module.load_binding(design_id)
        verification = field_module.verify_binding(value)
        state["binding"] = value
        own = value["gates"]
        return {"passed": bool(verification["passed"] and own.get("all_passed", False)), "hash_checks": verification["checks"],
                "production_gates": {k: (v.get("passed") if isinstance(v, dict) else v) for k, v in own.items()},
                "rho_under_iron": (value.get("topology_under_iron") or {}).get("min_rho_conservative"),
                "cusps_under_iron": [c["z_c_m"] for c in (value.get("topology_under_iron") or {}).get("wall_cusps", [])]}

    def grid_gate():
        mapping = design_module.pic_geometry(state["built"], domain) if target_cell_m is None else design_module.pic_geometry(state["built"], domain, target_cell_m=target_cell_m)
        state["mapping"] = mapping
        worst = max((abs(v["error_m"]) / (mapping.grid.dr_m if "radius" in k else mapping.grid.dz_m)) for k, v in mapping.snaps.items() if isinstance(v, dict))
        return {"passed": worst <= 0.5 + 1e-9, "node_shape": list(mapping.grid.node_shape), "dr_m": mapping.grid.dr_m, "dz_m": mapping.grid.dz_m,
                "worst_snap_in_cells": worst, "snaps": mapping.snaps}

    def field_map():
        mapping = state["mapping"]
        fm = field_module.design_field_map(mapping, state["binding"])
        state["field_map"] = fm
        protocol, _ = build_protocol(design_id, domain, built=state["built"], field_map=fm, target_cell_m=target_cell_m, dt_s=dt_override, grid=grid)
        state["protocol"] = protocol
        numerics = protocol["numerics"]
        limits = StabilityLimits(**numerics["stability_limits"])
        reference = numerics["stability_reference"]
        report = stability_report(mapping.grid, float(numerics["dt_s"]), reference_density_per_m3=float(reference["density_per_m3"]),
                                  reference_electron_temperature_ev=float(reference["electron_temperature_ev"]), max_b_t=fm.max_b_t,
                                  max_electron_energy_ev=float(reference["max_electron_energy_ev"]), limits=limits)
        # field_map_sha256 = content hash of the sampled arrays (this platform's bitwise identity, provenance only);
        # field_source_sha256 = the platform-independent binding (checkpoint bundle hashes + grid + scale)
        omega_ce_dt = 1.602176634e-19 * float(fm.max_b_t) / 9.1093837139e-31 * float(numerics["dt_s"])
        return {"passed": report.stable and omega_ce_dt <= float(limits.max_omega_ce_dt), "field_map_sha256": fm.sha256, "field_source_sha256": fm.source_sha256,
                "max_b_t": fm.max_b_t, "stability": report.to_dict(), "dt_s": float(numerics["dt_s"]), "dt_template_s": dt_override, "dt_policy": numerics.get("dt_policy"),
                "dt_admissibility": {"omega_ce_dt_at_composed_dt": omega_ce_dt, "limit": float(limits.max_omega_ce_dt), "reduced_from_template": bool(dt_override is not None and float(numerics["dt_s"]) < float(dt_override))},
                "cathode_placement": (protocol["operating_point"].get("cathode") or {}).get("placement_search_note"),
                "provenance_kind": fm.provenance.get("kind"), "plasma_nodes_sampled": fm.provenance.get("plasma_nodes_sampled")}

    def masks():
        import numpy as np

        m = build_mesh_masks(state["mapping"].grid)
        info = m.to_dict()
        far_field = int(np.count_nonzero(m.far_field_node)) if m.far_field_node.size else 0
        body_face = int(np.count_nonzero(m.body_face_node)) if m.body_face_node.size else 0
        ok = int(np.count_nonzero(m.plasma_cell)) > 0
        if state["mapping"].geometry.has_plume:
            ok = ok and far_field > 0 and body_face > 0
        return {"passed": ok, "far_field_nodes": far_field, "body_face_nodes": body_face, **{k: v for k, v in info.items() if not isinstance(v, (list, dict))}}

    def protocol_gate():
        from experiments.pic2d_cft_steady_state_v1 import run as runner

        protocol = state["protocol"]
        config = runner.build_config(protocol, backend="cpu")
        budget = runner.protocol_budget(protocol)
        peak_gate = protocol["numerics"].get("peak_debye_gate") or {}
        triad = protocol["stopping_rule"].get("grid_heating_triad") or {}
        return {"passed": True, "case_id": protocol["case"]["id"], "status": protocol["status"], "template": protocol["template_protocol"]["path"],
                "model_version": protocol.get("model_version"), "macro_weight": config.macro_weight, "macro_weight_parity": budget.get("macro_weight_parity"),
                "macro_weight_policy": protocol["case"]["macro_weight_policy"], "dt_s": config.dt_s, "cells": [config.grid.radial_cells, config.grid.axial_cells],
                "wall_budget_seconds": protocol["stopping_rule"]["wall_budget_seconds"], "wall_budget_hours": protocol["stopping_rule"]["wall_budget_seconds"] / 3600.0,
                "ion_transit_time_s": budget["ion_transit_time_s"], "steps_to_3_transits": budget["steps_to_3_transits"], "platform": budget.get("platform"),
                "ms_per_step_projected": budget["ms_per_step_projected"], "hours_to_3_transits_projected": budget["hours_to_3_transits_projected"],
                "particles_projected_m": budget["particles_projected_m"]["total_m"], "feed_atoms_per_s": protocol["operating_point"]["neutral_inventory"]["feed_atoms_per_s"],
                "wall_recycling": bool((protocol["operating_point"].get("neutral_inventory") or {}).get("wall_recycling", False)),
                "frame_recorder": protocol["numerics"].get("frame_recorder"),
                "gates_v2_0_3": {"peak_debye_window_mode": peak_gate.get("window_steps") is not None, "peak_debye_hard": peak_gate.get("max_cells_per_debye"),
                                 "peak_debye_soft": peak_gate.get("soft_cells_per_debye"), "windowed_residual_power_max": triad.get("windowed_energy_residual_over_electrode_work_max"),
                                 "residual_window_steps": triad.get("residual_window_steps")}}

    def connectivity():
        from experiments.pic2d_cft_steady_state_v1 import run as runner

        if not state["mapping"].geometry.has_plume:
            return {"passed": True, "skipped": "channel-only domain has no cathode region"}
        summary = runner.cathode_connectivity_check(state["protocol"], state["field_map"], build_mesh_masks(state["mapping"].grid))
        return {"passed": True, **(summary or {})}

    def cost():
        budget = state["protocol"]["budget_design_mini_sweep"] if "protocol" in state else {}
        return {"passed": True, **cost_module.design_cost(state["mapping"], dt_s=float(budget.get("dt_s", cost_module.DT_S)),
                                                          macro_weight=float(budget.get("macro_weight", cost_module.MACRO_WEIGHT_REFERENCE)),
                                                          platform=str(budget.get("platform", "rtx5090")))}

    ok = gate("identity", identity)
    ok = gate("field_binding", binding) and ok
    ok = gate("grid", grid_gate) and ok if "built" in state else False
    if "mapping" in state and "binding" in state:
        ok = gate("field_map", field_map) and ok
        ok = gate("mesh_masks", masks) and ok
        if "protocol" in state:
            ok = gate("protocol", protocol_gate) and ok
            ok = gate("cathode_connectivity", connectivity) and ok if "field_map" in state else False
        ok = gate("cost", cost) and ok
    record["passed"] = bool(ok and all(g["passed"] for g in gates.values()))
    record["seconds"] = time.perf_counter() - started
    return record


def _platform_record() -> dict[str, Any]:
    import platform as platform_module
    import socket

    import numpy as np

    record: dict[str, Any] = {"host": socket.gethostname(), "system": platform_module.system(), "release": platform_module.release(), "machine": platform_module.machine(),
                              "python": platform_module.python_version(), "numpy": np.__version__}
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        record["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()] if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - telemetry only
        record["gpus"] = None
    import os

    record["cuda_mps_pipe_directory"] = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    return record


def preflight_all(domain: str, *, grid: str = "50um", design_ids: tuple[str, ...] | None = None, log=print) -> dict[str, Any]:
    ids = design_module.design_ids() if design_ids is None else design_ids
    started = time.perf_counter()
    records = [preflight_design(design_id, domain, grid=grid, log=log) for design_id in ids]
    from .protocol import PREREGISTERED_OPTION

    preregistered = (domain, grid) == PREREGISTERED_OPTION
    return {
        "schema_version": "cft.pic2d.design-mini-sweep.preflight/1.0.0" if preregistered else "cft.pic2d.design-mini-sweep.preflight/0.1.0-draft",
        "experiment_id": "pic2d-design-mini-sweep-v1",
        "status": ("whole-set preflight of the PREREGISTERED option (non-evidentiary: derived-float gates re-run on the launch platform)" if preregistered
                   else "whole-set preflight of a DRAFT (not preregistered) option"),
        "domain": domain,
        "grid": grid,
        "option": option_tag(domain, grid),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": _platform_record(),
        "design_count": len(records),
        "all_passed": all(r["passed"] for r in records),
        "designs": records,
        "seconds": time.perf_counter() - started,
    }


def preflight_path(domain: str, grid: str = "50um") -> Path:
    return EXPERIMENT / f"preflight-{option_tag(domain, grid)}.json"


def write_preflight(domain: str, *, grid: str = "50um", design_ids: tuple[str, ...] | None = None, log=print) -> tuple[Path, dict[str, Any]]:
    report = preflight_all(domain, grid=grid, design_ids=design_ids, log=log)
    path = preflight_path(domain, grid)
    path.write_bytes(json.dumps(report, indent=1, sort_keys=True, allow_nan=False, default=_plain).encode("utf-8") + b"\n")
    return path, report


def _plain(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


__all__ = ["preflight_all", "preflight_design", "preflight_path", "write_preflight"]
