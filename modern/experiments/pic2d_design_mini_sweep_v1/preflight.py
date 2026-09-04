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

Output: ``preflight-<domain>.json`` with per-design gate results and ``all_passed``.
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
from .protocol import build_protocol

EXPERIMENT = Path(__file__).resolve().parent


def preflight_design(design_id: str, domain: str, *, log=print) -> dict[str, Any]:
    started = time.perf_counter()
    gates: dict[str, dict[str, Any]] = {}
    record: dict[str, Any] = {"design_id": design_id, "domain": domain, "gates": gates}

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

    def grid():
        mapping = design_module.pic_geometry(state["built"], domain)
        state["mapping"] = mapping
        worst = max((abs(v["error_m"]) / (mapping.grid.dr_m if "radius" in k else mapping.grid.dz_m)) for k, v in mapping.snaps.items() if isinstance(v, dict))
        return {"passed": worst <= 0.5 + 1e-9, "node_shape": list(mapping.grid.node_shape), "dr_m": mapping.grid.dr_m, "dz_m": mapping.grid.dz_m,
                "worst_snap_in_cells": worst, "snaps": mapping.snaps}

    def field_map():
        mapping = state["mapping"]
        fm = field_module.design_field_map(mapping, state["binding"])
        state["field_map"] = fm
        protocol, _ = build_protocol(design_id, domain, built=state["built"], field_map=fm)
        state["protocol"] = protocol
        numerics = protocol["numerics"]
        limits = StabilityLimits(**numerics["stability_limits"])
        reference = numerics["stability_reference"]
        report = stability_report(mapping.grid, float(numerics["dt_s"]), reference_density_per_m3=float(reference["density_per_m3"]),
                                  reference_electron_temperature_ev=float(reference["electron_temperature_ev"]), max_b_t=fm.max_b_t,
                                  max_electron_energy_ev=float(reference["max_electron_energy_ev"]), limits=limits)
        return {"passed": report.stable, "field_map_sha256": fm.sha256, "max_b_t": fm.max_b_t, "stability": report.to_dict(),
                "dt_s": float(numerics["dt_s"]), "dt_policy": numerics.get("dt_policy"), "cathode_placement": protocol["operating_point"].get("cathode", {}).get("placement_search_note"),
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
        return {"passed": True, "case_id": protocol["case"]["id"], "macro_weight": config.macro_weight, "wall_budget_seconds": protocol["stopping_rule"]["wall_budget_seconds"],
                "ion_transit_time_s": budget["ion_transit_time_s"], "feed_atoms_per_s": protocol["operating_point"]["neutral_inventory"]["feed_atoms_per_s"]}

    def connectivity():
        from experiments.pic2d_cft_steady_state_v1 import run as runner

        if not state["mapping"].geometry.has_plume:
            return {"passed": True, "skipped": "channel-only domain has no cathode region"}
        summary = runner.cathode_connectivity_check(state["protocol"], state["field_map"], build_mesh_masks(state["mapping"].grid))
        return {"passed": True, **(summary or {})}

    def cost():
        return {"passed": True, **cost_module.design_cost(state["mapping"])}

    ok = gate("identity", identity)
    ok = gate("field_binding", binding) and ok
    ok = gate("grid", grid) and ok if "built" in state else False
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


def preflight_all(domain: str, *, design_ids: tuple[str, ...] | None = None, log=print) -> dict[str, Any]:
    ids = design_module.design_ids() if design_ids is None else design_ids
    started = time.perf_counter()
    records = [preflight_design(design_id, domain, log=log) for design_id in ids]
    return {
        "schema_version": "cft.pic2d.design-mini-sweep.preflight/0.1.0-draft",
        "experiment_id": "pic2d-design-mini-sweep-v1",
        "status": "whole-set preflight of a DRAFT (not preregistered) sweep",
        "domain": domain,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "design_count": len(records),
        "all_passed": all(r["passed"] for r in records),
        "designs": records,
        "seconds": time.perf_counter() - started,
    }


def preflight_path(domain: str) -> Path:
    return EXPERIMENT / f"preflight-{domain}.json"


def write_preflight(domain: str, *, design_ids: tuple[str, ...] | None = None, log=print) -> tuple[Path, dict[str, Any]]:
    report = preflight_all(domain, design_ids=design_ids, log=log)
    path = preflight_path(domain)
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
