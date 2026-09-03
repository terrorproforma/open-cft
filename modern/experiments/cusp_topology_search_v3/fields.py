"""Design sets, identity-proven field re-solves and tracing grids for cusp topology search v3.

Four design sets, each bound to its sealed source records before any field is solved:

* ``sweep_v2`` - the 96 accepted L1a geometry-sweep-v2 designs, rebuilt and re-solved on
  CPU with the wall-loss geometry screening's identity-proven pipeline
  (``experiments.orbit_wall_loss_geometry_screening_v1.designs``: rebuilt case hashes must
  equal the sealed raw record, the re-solved QoIs must replay the recorded QoIs, and the
  four stored representatives must reproduce node-wise);
* ``four_cell_v2`` - the 128 sealed candidates of the preregistered four-cell topology
  search v2, rebuilt with the v2 sampler/builder (source and material hashes must equal the
  sealed record; the geometry hash must equal the sealed record once the recorded CRLF-era
  protocol byte hash is substituted into the geometry evidence note, see the v2
  ``POSTHOC_AUDIT.md``) and re-solved on CPU at the v2 primary resolution;
* ``characterization_v1`` - the 56 cases of the topology characterization v1 (geometry,
  material, source and family hashes must equal the sealed dataset) re-solved on CPU at the
  v1 primary resolution; their sealed axis roots are the held-out reference for v3's axis
  nulls;
* ``p2_divergent_exit`` - the P2-qualified divergent-exit-stack FEM field (iron poles and
  return yoke) through the hash-bound orbit v4 adapter (level-1 as accepted, level-2 as
  refined), sampled on the v4 regular plasma domain.

For every L1a design the "refined" map is the same problem re-solved with twice the radial
and axial intervals. All maps are restricted to the radial nodes 0..first node >= wall
radius and the full axial range (``topology.tracing_grid``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

import cft_revival.coupling as coupling_package
import cft_revival.experiment_runtime as runtime_package
import cft_revival.orbit_mc as orbit_mc_package
from cft_revival.experiment_runtime import strict_json_file
from cft_revival.fields import AxisymmetricDomain, AxisymmetricProblem, FieldMap, solve_problem_cpu, validate_field_artifact

from experiments.cft_orbit_wall_loss_v4 import adapter as v4_adapter
from experiments.cft_topology_characterization_v1 import experiment as characterization_v1
from experiments.four_cell_topology_search_v2 import experiment as four_cell_v2
from experiments.l1a_geometry_sweep_v2 import experiment as sweep
from experiments.orbit_wall_loss_geometry_screening_v1 import designs as screening_designs

from .topology import ChannelGeometry, TracingGrid, tracing_grid

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
REPOSITORY = MODERN.parent

SET_SWEEP = "sweep_v2"
SET_FOUR_CELL = "four_cell_v2"
SET_CHARACTERIZATION = "characterization_v1"
SET_P2 = "p2_divergent_exit"
DESIGN_SETS = (SET_SWEEP, SET_FOUR_CELL, SET_CHARACTERIZATION, SET_P2)
P2_DESIGN_ID = "divergent-exit-stack"

SCREENING_PROTOCOL_PATH = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "protocol.json"
V2_RESULTS = MODERN / "experiments" / "four_cell_topology_search_v2" / "results"
V1_RESULTS = MODERN / "experiments" / "cft_topology_characterization_v1" / "results"
V4_PROTOCOL_PATH = MODERN / "experiments" / "cft_orbit_wall_loss_v4" / "protocol.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Source binding of everything the topology depends on
# --------------------------------------------------------------------------


def dependency_source_files() -> list[Path]:
    """Imported experiment modules and packages whose bytes determine the v3 topology."""

    files: list[Path] = []
    for package in (coupling_package, runtime_package):
        root = Path(package.__file__).resolve().parent
        if root.parent != (MODERN / "src" / "cft_revival").resolve():
            raise RuntimeError(f"{package.__name__} is imported from {root}, not from this worktree")
        files.extend(sorted(root.glob("*.py")))
    orbit_root = Path(orbit_mc_package.__file__).resolve().parent
    files.extend(orbit_root / name for name in ("fields.py", "models.py"))
    files.extend(
        (
            MODERN / "experiments" / "cft_topology_characterization_v1" / "experiment.py",
            MODERN / "experiments" / "cft_topology_characterization_v1" / "protocol.json",
            MODERN / "experiments" / "four_cell_topology_search_v2" / "experiment.py",
            MODERN / "experiments" / "four_cell_topology_search_v2" / "protocol.json",
            MODERN / "experiments" / "cft_orbit_wall_loss_v4" / "adapter.py",
            V4_PROTOCOL_PATH,
            MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "designs.py",
            SCREENING_PROTOCOL_PATH,
        )
    )
    return files


def dependency_source_sha256() -> str:
    """SHA-256 over (posix path, LF bytes) of the dependency sources (fails closed on CR)."""

    digest = hashlib.sha256()
    for path in dependency_source_files():
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"dependency source {path.relative_to(MODERN).as_posix()} contains CR bytes")
        digest.update(path.relative_to(MODERN).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def sealed_source_binding() -> dict[str, Any]:
    """Byte identities of the sealed records every design set is bound to."""

    screening_protocol = strict_json_file(SCREENING_PROTOCOL_PATH)
    binding = screening_designs.load_sweep_binding(screening_protocol["field_source"])
    v2_dataset_path = V2_RESULTS / "dataset.json"
    four_cell_v2._verify_sidecar(v2_dataset_path)
    v4_protocol = strict_json_file(V4_PROTOCOL_PATH)
    return {
        "sweep_v2": {
            "manifest_file_sha256": binding.manifest_file_sha256,
            "raw_results_file_sha256": binding.raw_file_sha256,
            "summary_file_sha256": binding.summary_file_sha256,
            "preregistration_commit": binding.manifest["preregistration_commit_sha"],
            "screening_protocol_file_sha256": _file_sha256(SCREENING_PROTOCOL_PATH),
        },
        "four_cell_v2": {
            "dataset_file_sha256": _file_sha256(v2_dataset_path),
            "recorded_protocol_sha256": strict_json_file(v2_dataset_path)["protocol_sha256"],
            "lf_protocol_sha256": four_cell_v2.PROTOCOL_SHA256,
            "preregistration_commit": strict_json_file(v2_dataset_path)["preregistration_commit_sha"],
        },
        "characterization_v1": {
            "dataset_file_sha256": _file_sha256(V1_RESULTS / "dataset.json"),
            "manifest_file_sha256": _file_sha256(V1_RESULTS / "manifest.json"),
            "preregistration_commit": strict_json_file(V1_RESULTS / "dataset.json")["preregistration_commit_sha"],
        },
        "p2_divergent_exit": {
            "v4_protocol_file_sha256": _file_sha256(V4_PROTOCOL_PATH),
            "maps": {
                role: {
                    key: v4_protocol["field_adapter"]["maps"][role][key]
                    for key in ("checkpoint_path", "checkpoint_file_sha256", "sidecar_file_sha256", "mesh_sha256", "run_sha256")
                }
                for role in ("primary", "refined")
            },
        },
    }


# --------------------------------------------------------------------------
# Design specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignSpec:
    set_id: str
    design_id: str
    ordinal: int
    representative: bool

    @property
    def key(self) -> str:
        return f"{self.set_id}:{self.design_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_V2_DATASET_CACHE: dict[str, Any] | None = None
_V1_DATASET_CACHE: dict[str, Any] | None = None
_SWEEP_BINDING_CACHE: screening_designs.SweepBinding | None = None


def v2_dataset() -> dict[str, Any]:
    global _V2_DATASET_CACHE
    if _V2_DATASET_CACHE is None:
        _V2_DATASET_CACHE = four_cell_v2.load_sealed_json(V2_RESULTS / "dataset.json")
    return _V2_DATASET_CACHE


def v1_dataset() -> dict[str, Any]:
    global _V1_DATASET_CACHE
    if _V1_DATASET_CACHE is None:
        value = strict_json_file(V1_RESULTS / "dataset.json")
        manifest = strict_json_file(V1_RESULTS / "manifest.json")
        entry = next(item for item in manifest["artifacts"] if item["path"] == "dataset.json")
        if characterization_v1.semantic_hash(value) != entry["semantic_sha256"]:
            raise ValueError("characterization v1 dataset does not match its manifest identity")
        _V1_DATASET_CACHE = value
    return _V1_DATASET_CACHE


def sweep_binding() -> screening_designs.SweepBinding:
    global _SWEEP_BINDING_CACHE
    if _SWEEP_BINDING_CACHE is None:
        screening_protocol = strict_json_file(SCREENING_PROTOCOL_PATH)
        _SWEEP_BINDING_CACHE = screening_designs.load_sweep_binding(screening_protocol["field_source"])
    return _SWEEP_BINDING_CACHE


def design_specs(protocol: Mapping[str, Any]) -> tuple[DesignSpec, ...]:
    """Every declared design of every included set, in a fixed order, checked against the sources."""

    declaration = protocol["design_sets"]
    specs: list[DesignSpec] = []
    if declaration[SET_SWEEP]["included"]:
        binding = sweep_binding()
        ids = sorted(binding.cases_by_id)
        representatives = set(screening_designs.representative_case_ids(binding))
        if len(ids) != int(declaration[SET_SWEEP]["design_count"]):
            raise ValueError("sweep-v2 design count differs from the protocol")
        specs.extend(DesignSpec(SET_SWEEP, case_id, index, case_id in representatives) for index, case_id in enumerate(ids))
    if declaration[SET_FOUR_CELL]["included"]:
        candidates = four_cell_v2.sample_candidates()
        recorded = {case["candidate_id"] for case in v2_dataset()["cases"]}
        ids = [item["candidate_id"] for item in candidates]
        if set(ids) != recorded or len(ids) != int(declaration[SET_FOUR_CELL]["design_count"]):
            raise ValueError("four-cell v2 candidate set differs from the sealed dataset or the protocol")
        representatives = set(declaration[SET_FOUR_CELL]["representative_ids"])
        specs.extend(DesignSpec(SET_FOUR_CELL, cid, index, cid in representatives) for index, cid in enumerate(ids))
    if declaration[SET_CHARACTERIZATION]["included"]:
        definitions = characterization_v1.case_definitions()
        recorded = {case["case_id"] for case in v1_dataset()["cases"]}
        ids = [item.case_id for item in definitions]
        if set(ids) != recorded or len(ids) != int(declaration[SET_CHARACTERIZATION]["design_count"]):
            raise ValueError("characterization v1 case set differs from the sealed dataset or the protocol")
        representatives = set(v1_dataset()["representative_case_ids"])
        specs.extend(DesignSpec(SET_CHARACTERIZATION, cid, index, cid in representatives) for index, cid in enumerate(ids))
    if declaration[SET_P2]["included"]:
        specs.append(DesignSpec(SET_P2, P2_DESIGN_ID, 0, True))
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("design keys are not unique")
    return tuple(specs)


# --------------------------------------------------------------------------
# Per-set rebuild + identity + solve
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedDesign:
    spec: DesignSpec
    geometry: ChannelGeometry
    accepted: TracingGrid
    refined: TracingGrid
    identity: dict[str, Any]
    evidence: dict[str, Any]
    reference: dict[str, Any]
    solve_seconds: float

    @property
    def accepted_identity_sha256(self) -> str:
        return self.identity["accepted_field_identity_sha256"]

    @property
    def refined_identity_sha256(self) -> str:
        return self.identity["refined_field_identity_sha256"]


def _grid(field: FieldMap, wall_radius_m: float) -> TracingGrid:
    return tracing_grid(field.r_m, field.z_m, field.psi_wb, field.b_r_t, field.b_z_t, wall_radius_m)


def _doubled(domain: AxisymmetricDomain, factor: int) -> AxisymmetricDomain:
    return AxisymmetricDomain(
        domain.radius_m, domain.z_min_m, domain.z_max_m, int(domain.radial_intervals) * factor, int(domain.axial_intervals) * factor
    )


def _field_identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _solver_evidence(field: FieldMap) -> dict[str, Any]:
    return {
        "backend": field.diagnostics.backend,
        "iterations": int(field.diagnostics.iterations),
        "converged": bool(field.diagnostics.converged),
        "relative_residual_l2": float(field.diagnostics.relative_residual_l2),
        "flux_reconstruction_identity_t_per_m": float(field.diagnostics.max_flux_reconstruction_identity_t_per_m),
    }


def _node_agreement(field: FieldMap, mapping: Mapping[str, Any], tolerance: Mapping[str, float], label: str) -> dict[str, Any]:
    if list(mapping["r_m"]) != list(field.r_m) or list(mapping["z_m"]) != list(field.z_m):
        raise ValueError(f"{label}: stored representative grid differs from the re-solve")
    psi_difference = float(np.max(np.abs(np.asarray(mapping["psi_wb"]) - np.asarray(field.psi_wb))))
    b_difference = float(
        max(
            np.max(np.abs(np.asarray(mapping["b_r_t"]) - np.asarray(field.b_r_t))),
            np.max(np.abs(np.asarray(mapping["b_z_t"]) - np.asarray(field.b_z_t))),
        )
    )
    checks = {
        "psi_nodes": psi_difference <= float(tolerance["psi_max_abs_wb"]),
        "b_nodes": b_difference <= float(tolerance["b_max_abs_t"]),
    }
    return {
        "psi_max_abs_difference_wb": psi_difference,
        "b_max_abs_difference_t": b_difference,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---- sweep_v2 --------------------------------------------------------------


def _resolve_sweep(spec: DesignSpec, protocol: Mapping[str, Any]) -> ResolvedDesign:
    declaration = protocol["design_sets"][SET_SWEEP]
    screening_protocol = strict_json_file(SCREENING_PROTOCOL_PATH)
    field_source = screening_protocol["field_source"]
    binding = sweep_binding()
    recorded = binding.cases_by_id[spec.design_id]
    case = screening_designs.rebuild_case(binding, spec.design_id)
    geometry_record = screening_designs.design_geometry(case)
    domain = case.problem.domain
    resolve = field_source["resolve"]
    if (
        domain.radius_m != resolve["domain"]["radius_m"]
        or domain.z_min_m != resolve["domain"]["z_min_m"]
        or domain.z_max_m != resolve["domain"]["z_max_m"]
        or domain.radial_intervals != resolve["domain"]["radial_intervals"]
        or domain.axial_intervals != resolve["domain"]["axial_intervals"]
        or asdict(sweep.SOLVER) != dict(resolve["solver_config"])
    ):
        raise ValueError(f"{spec.design_id}: sweep solver inputs differ from the screening's declared authority")
    started = time.perf_counter()
    accepted_field = solve_problem_cpu(case.problem, sweep.SOLVER)
    qoi_report = screening_designs.verify_resolved_qois(case, accepted_field, recorded)
    if not qoi_report["passed"]:
        raise ValueError(f"{spec.design_id}: re-solved QoIs differ from the sealed sweep record")
    stored = screening_designs.verify_stored_representative(spec.design_id, accepted_field, resolve["stored_map_node_tolerance"])
    if stored is not None and not stored["passed"]:
        raise ValueError(f"{spec.design_id}: re-solved field differs from the stored representative map")
    refined_field = solve_problem_cpu(screening_designs.refined_problem(case, int(declaration["refinement"])), sweep.SOLVER)
    solve_seconds = time.perf_counter() - started
    geometry = ChannelGeometry(
        wall_radius_m=geometry_record.wall_radius_m,
        straight_z_min_m=0.0,
        straight_z_max_m=geometry_record.exit_start_m,
        chamber_length_m=geometry_record.chamber_length_m,
        stage_pitch_m=geometry_record.stage_pitch_m,
        stage_centres_m=tuple(geometry_record.stage_centers_m),
        injector_length_m=geometry_record.injector_length_m,
    )
    base_identity = {
        "set_id": SET_SWEEP,
        "design_id": spec.design_id,
        "case_sha256": case.case_sha256,
        "geometry_sha256": case.geometry_sha256,
        "source_sha256": case.source_sha256,
        "config_sha256": case.config_sha256,
        "solver": "cft_revival.fields.solve_problem_cpu",
        "solver_config": asdict(sweep.SOLVER),
        "domain": {"radius_m": domain.radius_m, "z_min_m": domain.z_min_m, "z_max_m": domain.z_max_m, "radial_intervals": domain.radial_intervals, "axial_intervals": domain.axial_intervals},
    }
    identity = {
        **base_identity,
        "sweep_preregistration_commit": binding.manifest["preregistration_commit_sha"],
        "accepted_field_identity_sha256": _field_identity({**base_identity, "role": "accepted", "refinement": 1}),
        "refined_field_identity_sha256": _field_identity({**base_identity, "role": "refined", "refinement": int(declaration["refinement"])}),
    }
    evidence = {
        "design_values": sweep.design_values(case.design),
        "design_geometry": geometry_record.to_dict(),
        "accepted_solve": _solver_evidence(accepted_field),
        "refined_solve": _solver_evidence(refined_field),
        "qoi_replay": {"passed": qoi_report["passed"], "checks": qoi_report["checks"]},
        "stored_representative": stored,
        "identity_proven": True,
    }
    reference = {
        "sweep_axis_null_positions_m": list(recorded["qois"]["axis_null_positions_m"]),
        "sweep_axis_bz_peak_positions_m": list(recorded["qois"]["axis_cusp_positions_m"]),
        "sweep_axis_bz_peak_definition": "local maxima of |B_z| along the axis inside the chamber (the sweep's 'axis cusp' QoI)",
        "sweep_minimum_mirror_ratio": recorded["qois"]["minimum_mirror_ratio"],
        "sweep_maximum_mirror_ratio": recorded["qois"]["maximum_mirror_ratio"],
    }
    return ResolvedDesign(spec, geometry, _grid(accepted_field, geometry.wall_radius_m), _grid(refined_field, geometry.wall_radius_m), identity, evidence, reference, solve_seconds)


# ---- four_cell_v2 -----------------------------------------------------------


@contextlib.contextmanager
def _recorded_v2_protocol_hash(recorded: str) -> Iterator[None]:
    """Substitute the sealed (CRLF-era) protocol byte hash into the v2 geometry evidence note."""

    original = four_cell_v2.PROTOCOL_SHA256
    four_cell_v2.PROTOCOL_SHA256 = recorded
    try:
        yield
    finally:
        four_cell_v2.PROTOCOL_SHA256 = original


def _resolve_four_cell(spec: DesignSpec, protocol: Mapping[str, Any]) -> ResolvedDesign:
    declaration = protocol["design_sets"][SET_FOUR_CELL]
    dataset = v2_dataset()
    recorded = next(case for case in dataset["cases"] if case["candidate_id"] == spec.design_id)
    candidate_declaration = next(item for item in four_cell_v2.sample_candidates() if item["candidate_id"] == spec.design_id)
    if candidate_declaration["sampling_identity_sha256"] != recorded["sampling"]["sampling_identity_sha256"]:
        raise ValueError(f"{spec.design_id}: sampled candidate identity differs from the sealed record")
    lf_candidate = four_cell_v2.build_candidate(candidate_declaration)
    with _recorded_v2_protocol_hash(dataset["protocol_sha256"]):
        candidate = four_cell_v2.build_candidate(candidate_declaration)
    checks = {
        "geometry_sha256_with_recorded_protocol_hash": candidate.geometry_sha256 == recorded["geometry_sha256"],
        "source_sha256": candidate.source_sha256 == recorded["source_sha256"] == lf_candidate.source_sha256,
        "material_sha256": candidate.material_sha256 == recorded["material_sha256"] == lf_candidate.material_sha256,
        "derived_geometry": (
            recorded["derived_geometry"]["chamber_radius_m"] == candidate.chamber_radius_m
            and list(recorded["derived_geometry"]["stage_centres_m"]) == list(candidate.stage_centres_m)
            and recorded["derived_geometry"]["pitch_m"] == candidate.pitch_m
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"{spec.design_id}: rebuilt candidate differs from the sealed v2 record: {checks}")
    domain = four_cell_v2._domain(candidate, "primary")
    recorded_domain = recorded["maps"]["primary"]["domain"]
    if (
        domain.radius_m != recorded_domain["radius_m"]
        or domain.z_min_m != recorded_domain["z_min_m"]
        or domain.z_max_m != recorded_domain["z_max_m"]
        or domain.radial_intervals != recorded_domain["radial_intervals"]
        or domain.axial_intervals != recorded_domain["axial_intervals"]
    ):
        raise ValueError(f"{spec.design_id}: rebuilt primary domain differs from the sealed record")
    config = four_cell_v2._solver_config()
    started = time.perf_counter()
    accepted_field = solve_problem_cpu(AxisymmetricProblem(f"{spec.design_id}-primary", domain, candidate.sources), config)
    stored = None
    if spec.representative:
        artifact_path = V2_RESULTS / "representatives" / f"{spec.design_id}-primary-field.json"
        artifact = four_cell_v2.load_sealed_json(artifact_path)
        validate_field_artifact(artifact)
        stored = _node_agreement(accepted_field, artifact["field_map"], declaration["stored_map_node_tolerance"], spec.design_id)
        stored["artifact_payload_sha256"] = artifact["integrity"]["payload_sha256"]
        # The v2 dataset records the sha256 of the artifact BYTES (== the sidecar-verified file).
        stored["artifact_file_sha256"] = _file_sha256(artifact_path)
        if stored["artifact_file_sha256"] != recorded["maps"]["primary"]["artifact_sha256"]:
            raise ValueError(f"{spec.design_id}: stored representative artifact identity differs from the dataset record")
        if not stored["passed"]:
            raise ValueError(f"{spec.design_id}: re-solved field differs from the stored representative map")
    refined_field = solve_problem_cpu(
        AxisymmetricProblem(f"{spec.design_id}-refined", _doubled(domain, int(declaration["refinement"])), candidate.sources), config
    )
    solve_seconds = time.perf_counter() - started
    chamber = candidate.geometry.chamber
    geometry = ChannelGeometry(
        wall_radius_m=float(candidate.chamber_radius_m),
        straight_z_min_m=0.0,
        straight_z_max_m=float(chamber.exit_start_m),
        chamber_length_m=float(chamber.length_m),
        stage_pitch_m=float(candidate.pitch_m),
        stage_centres_m=tuple(float(value) for value in candidate.stage_centres_m),
        injector_length_m=float(chamber.injector_length_m),
    )
    base_identity = {
        "set_id": SET_FOUR_CELL,
        "design_id": spec.design_id,
        "sampling_identity_sha256": candidate_declaration["sampling_identity_sha256"],
        "geometry_sha256": candidate.geometry_sha256,
        "geometry_sha256_lf_checkout": lf_candidate.geometry_sha256,
        "source_sha256": candidate.source_sha256,
        "material_sha256": candidate.material_sha256,
        "solver": "cft_revival.fields.solve_problem_cpu",
        "solver_config": asdict(config),
        "domain": {"radius_m": domain.radius_m, "z_min_m": domain.z_min_m, "z_max_m": domain.z_max_m, "radial_intervals": domain.radial_intervals, "axial_intervals": domain.axial_intervals},
    }
    identity = {
        **base_identity,
        "v2_preregistration_commit": dataset["preregistration_commit_sha"],
        "accepted_field_identity_sha256": _field_identity({**base_identity, "role": "accepted", "refinement": 1}),
        "refined_field_identity_sha256": _field_identity({**base_identity, "role": "refined", "refinement": int(declaration["refinement"])}),
    }
    evidence = {
        "design_values": dict(candidate_declaration["values"]),
        "identity_checks": checks,
        "accepted_solve": _solver_evidence(accepted_field),
        "refined_solve": _solver_evidence(refined_field),
        "stored_representative": stored,
        "identity_proven": True,
        "geometry_hash_note": "sealed geometry_sha256 embeds the v2 protocol BYTE hash of the executing (CRLF) checkout through an evidence note; the rebuilt hash matches once that recorded hash is substituted (POSTHOC_AUDIT.md of the v2 experiment)",
    }
    reference = {
        "v2_recorded_interior_cusp_z_m": {role: list(recorded["maps"][role]["interior_cusp_z_m"]) for role in recorded["maps"]},
        "v2_recorded_failures": list(recorded["failures"]),
        "v2_recorded_stable": bool(recorded["stable"]),
        "v2_cusp_targets_m": list(recorded["derived_geometry"]["cusp_targets_m"]),
        "v2_wall_sample_radius_m": recorded["derived_geometry"]["wall_radius_m"],
    }
    return ResolvedDesign(spec, geometry, _grid(accepted_field, geometry.wall_radius_m), _grid(refined_field, geometry.wall_radius_m), identity, evidence, reference, solve_seconds)


# ---- characterization_v1 ----------------------------------------------------


def _resolve_characterization(spec: DesignSpec, protocol: Mapping[str, Any]) -> ResolvedDesign:
    declaration = protocol["design_sets"][SET_CHARACTERIZATION]
    dataset = v1_dataset()
    recorded = next(case for case in dataset["cases"] if case["case_id"] == spec.design_id)
    definition = next(item for item in characterization_v1.case_definitions() if item.case_id == spec.design_id)
    case = characterization_v1.build_case(definition)
    checks = {
        "geometry_sha256": case.geometry_sha256 == recorded["geometry_sha256"],
        "material_semantic_sha256": case.material_semantic_sha256 == recorded["material_semantic_sha256"],
        "source_semantic_sha256": case.source_semantic_sha256 == recorded["source_semantic_sha256"],
        "family_semantic_sha256": definition.family_semantic_sha256 == recorded["family_semantic_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"{spec.design_id}: rebuilt case differs from the sealed v1 record: {checks}")
    domain = characterization_v1.domain_for(case, "primary")
    recorded_domain = recorded["maps"]["primary"]["domain"]
    if (
        domain.radius_m != recorded_domain["radius_m"]
        or domain.z_min_m != recorded_domain["z_min_m"]
        or domain.z_max_m != recorded_domain["z_max_m"]
        or domain.radial_intervals != recorded_domain["radial_intervals"]
        or domain.axial_intervals != recorded_domain["axial_intervals"]
    ):
        raise ValueError(f"{spec.design_id}: rebuilt primary domain differs from the sealed record")
    config = characterization_v1.solver_config()
    started = time.perf_counter()
    accepted_field = solve_problem_cpu(AxisymmetricProblem(f"{spec.design_id}-primary", domain, case.sources), config)
    stored = None
    if spec.representative:
        path = V1_RESULTS / "representatives" / spec.design_id / "primary-field.json"
        raw = path.read_bytes()
        manifest = strict_json_file(V1_RESULTS / "manifest.json")
        relative = path.relative_to(V1_RESULTS).as_posix()
        entry = next(item for item in manifest["artifacts"] if item["path"] == relative)
        if hashlib.sha256(raw).hexdigest() != entry["semantic_sha256"]:
            raise ValueError(f"{spec.design_id}: stored representative field does not match the v1 manifest identity")
        artifact = json.loads(raw.decode("utf-8"))
        validate_field_artifact(artifact)
        stored = _node_agreement(accepted_field, artifact["field_map"], declaration["stored_map_node_tolerance"], spec.design_id)
        stored["artifact_file_sha256"] = entry["semantic_sha256"]
        if not stored["passed"]:
            raise ValueError(f"{spec.design_id}: re-solved field differs from the stored representative map")
    refined_field = solve_problem_cpu(
        AxisymmetricProblem(f"{spec.design_id}-refined", _doubled(domain, int(declaration["refinement"])), case.sources), config
    )
    solve_seconds = time.perf_counter() - started
    chamber = case.geometry.chamber
    geometry = ChannelGeometry(
        wall_radius_m=float(definition.chamber_radius_m),
        straight_z_min_m=0.0,
        straight_z_max_m=float(chamber.exit_start_m),
        chamber_length_m=float(case.chamber_length_m),
        stage_pitch_m=float(definition.pitch_m),
        stage_centres_m=tuple(float(value) for value in case.stage_centres_m),
        injector_length_m=float(chamber.injector_length_m),
    )
    base_identity = {
        "set_id": SET_CHARACTERIZATION,
        "design_id": spec.design_id,
        "family_semantic_sha256": definition.family_semantic_sha256,
        "geometry_sha256": case.geometry_sha256,
        "material_semantic_sha256": case.material_semantic_sha256,
        "source_semantic_sha256": case.source_semantic_sha256,
        "solver": "cft_revival.fields.solve_problem_cpu",
        "solver_config": asdict(config),
        "domain": {"radius_m": domain.radius_m, "z_min_m": domain.z_min_m, "z_max_m": domain.z_max_m, "radial_intervals": domain.radial_intervals, "axial_intervals": domain.axial_intervals},
    }
    identity = {
        **base_identity,
        "v1_preregistration_commit": dataset["preregistration_commit_sha"],
        "accepted_field_identity_sha256": _field_identity({**base_identity, "role": "accepted", "refinement": 1}),
        "refined_field_identity_sha256": _field_identity({**base_identity, "role": "refined", "refinement": int(declaration["refinement"])}),
    }
    primary_roots = [
        {
            "root_id": root["root_id"],
            "z_m": root["z_m"],
            "classification": root["local_topology"]["classification"],
            "jacobian_converged": root["local_topology"].get("jacobian_converged"),
            "zone": root["geometry_association"]["zone"],
            "inside_plasma_channel": root["geometry_association"]["inside_plasma_channel"],
            "eligible_cusp": root["eligible_cusp"],
            "exclusion_reason": root["exclusion_reason"],
        }
        for root in recorded["maps"]["primary"]["roots"]
        if root["r_m"] == 0.0 and not root["finite_box_boundary"]
    ]
    evidence = {
        "identity_checks": checks,
        "accepted_solve": _solver_evidence(accepted_field),
        "refined_solve": _solver_evidence(refined_field),
        "stored_representative": stored,
        "identity_proven": True,
        "definition": {"stage_count": definition.stage_count, "pitch_m": definition.pitch_m, "chamber_radius_m": definition.chamber_radius_m, "first_polarity": definition.first_polarity},
    }
    reference = {
        "v1_primary_axis_roots": primary_roots,
        "v1_recorded_failures": list(recorded["failures"]),
        "v1_stable_root_count": recorded["cross_map"]["stable_root_count"],
        "v1_stable_eligible_cusp_count": recorded["cross_map"]["stable_eligible_cusp_count"],
        "v1_stable_eligible_cell_count": recorded["cross_map"]["stable_eligible_cell_count"],
    }
    return ResolvedDesign(spec, geometry, _grid(accepted_field, geometry.wall_radius_m), _grid(refined_field, geometry.wall_radius_m), identity, evidence, reference, solve_seconds)


# ---- p2_divergent_exit -------------------------------------------------------


def _p2_grid(v4_protocol: Mapping[str, Any], role: str) -> tuple[TracingGrid, dict[str, Any]]:
    adapter = v4_protocol["field_adapter"]
    declaration = adapter["maps"][role]
    bounds = adapter["regular_plasma_domain"]
    evaluator = v4_adapter.BoundP2Evaluator(
        REPOSITORY / declaration["checkpoint_path"],
        declaration,
        allowed_regions=set(adapter["plasma_region_ids"]),
        bounds=bounds,
    )
    radii = np.linspace(bounds["r_min_m"], bounds["r_max_m"], int(declaration["radial_intervals"]) + 1)
    axial = np.linspace(bounds["z_min_m"], bounds["z_max_m"], int(declaration["axial_intervals"]) + 1)
    psi = np.empty((len(radii), len(axial)), dtype=np.float64)
    br = np.empty_like(psi)
    bz = np.empty_like(psi)
    for i, radius in enumerate(radii):
        for j, z_value in enumerate(axial):
            psi[i, j], br[i, j], bz[i, j] = evaluator.evaluate(float(radius), float(z_value))
    psi[0, :] = 0.0  # exact axis value of r*A_phi; the evaluator returns 0.0 there already
    grid = tracing_grid(radii, axial, psi, br, bz, float(bounds["r_max_m"]))
    evidence = {
        "role": role,
        "checkpoint_path": declaration["checkpoint_path"],
        "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
        "sidecar_file_sha256": declaration["sidecar_file_sha256"],
        "mesh_sha256": declaration["mesh_sha256"],
        "run_sha256": declaration["run_sha256"],
        "regular_grid": {"radial_samples": int(len(radii)), "axial_samples": int(len(axial)), "r_max_m": float(radii[-1]), "z_min_m": float(axial[0]), "z_max_m": float(axial[-1])},
        "sampling": "BoundP2Evaluator.evaluate (quadratic A_phi FEM checkpoint) at every regular node: psi = r*A_phi, B_r = -dA_phi/dz, B_z = A_phi/r + dA_phi/dr",
    }
    return grid, evidence


def _resolve_p2(spec: DesignSpec, protocol: Mapping[str, Any]) -> ResolvedDesign:
    declaration = protocol["design_sets"][SET_P2]
    v4_protocol = strict_json_file(V4_PROTOCOL_PATH)
    if v4_protocol["authority"]["design_id"] != P2_DESIGN_ID:
        raise ValueError("v4 protocol does not describe the divergent-exit-stack design")
    started = time.perf_counter()
    accepted, accepted_evidence = _p2_grid(v4_protocol, "primary")
    refined, refined_evidence = _p2_grid(v4_protocol, "refined")
    solve_seconds = time.perf_counter() - started
    from cft_revival.geometry.generators import divergent_exit_stack

    design = divergent_exit_stack()
    chamber = design.chamber
    wall = v4_protocol["orbit"]["wall"]
    # chamber.exit_start_m = 0.024 - 0.006 = 0.018000000000000002 in binary64; the v4 authority is 0.018.
    if float(wall["radius_m"]) != float(chamber.outer_radius_m) or abs(float(wall["z_max_m"]) - float(chamber.exit_start_m)) > 1.0e-12:
        raise ValueError("v4 straight-wall authority differs from the divergent-exit-stack geometry")
    geometry = ChannelGeometry(
        wall_radius_m=float(chamber.outer_radius_m),
        straight_z_min_m=float(wall["z_min_m"]),
        straight_z_max_m=float(wall["z_max_m"]),
        chamber_length_m=float(chamber.length_m),
        stage_pitch_m=float(design.stages[0].pitch_m),
        stage_centres_m=tuple(float(stage.center_z_m) for stage in design.stages),
        injector_length_m=float(chamber.injector_length_m),
    )
    base_identity = {
        "set_id": SET_P2,
        "design_id": P2_DESIGN_ID,
        "geometry_sha256": design.canonical_sha256,
        "v4_protocol_file_sha256": _file_sha256(V4_PROTOCOL_PATH),
        "primary": {key: accepted_evidence[key] for key in ("checkpoint_file_sha256", "sidecar_file_sha256", "mesh_sha256", "run_sha256")},
        "refined": {key: refined_evidence[key] for key in ("checkpoint_file_sha256", "sidecar_file_sha256", "mesh_sha256", "run_sha256")},
    }
    identity = {
        **base_identity,
        "accepted_field_identity_sha256": _field_identity({**base_identity, "role": "accepted"}),
        "refined_field_identity_sha256": _field_identity({**base_identity, "role": "refined"}),
    }
    evidence = {
        "accepted_map": accepted_evidence,
        "refined_map": refined_evidence,
        "identity_proven": True,
        "field_level": "P2 adaptive FEM (NUMERICAL_P2_QUALIFIED), iron poles and return yoke present in the source model",
        "refinement_note": "the refined map is the level-2 FEM checkpoint sampled on a 2x regular grid, not a re-solve of the same discrete problem",
        "straight_section_note": "the regular plasma domain starts at z = 1 mm; the straight dielectric spans [1, 18] mm in the field data, the physical chamber [0, 24] mm",
    }
    reference = {
        "p2_consistency_references": dict(declaration["consistency_references"]),
    }
    return ResolvedDesign(spec, geometry, accepted, refined, identity, evidence, reference, solve_seconds)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def resolve_design(spec: DesignSpec, protocol: Mapping[str, Any]) -> ResolvedDesign:
    """Rebuild, prove identity and solve/sample the accepted and refined maps of one design."""

    if spec.set_id == SET_SWEEP:
        return _resolve_sweep(spec, protocol)
    if spec.set_id == SET_FOUR_CELL:
        return _resolve_four_cell(spec, protocol)
    if spec.set_id == SET_CHARACTERIZATION:
        return _resolve_characterization(spec, protocol)
    if spec.set_id == SET_P2:
        return _resolve_p2(spec, protocol)
    raise ValueError(f"unknown design set {spec.set_id}")


def design_identity_without_solving(spec: DesignSpec, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Identity facts for the design authorities (rebuild only; no field solve)."""

    if spec.set_id == SET_SWEEP:
        binding = sweep_binding()
        case = screening_designs.rebuild_case(binding, spec.design_id)
        return {
            "set_id": SET_SWEEP,
            "design_id": spec.design_id,
            "case_sha256": case.case_sha256,
            "geometry_sha256": case.geometry_sha256,
            "source_sha256": case.source_sha256,
            "config_sha256": case.config_sha256,
            "representative": spec.representative,
        }
    if spec.set_id == SET_FOUR_CELL:
        dataset = v2_dataset()
        recorded = next(case for case in dataset["cases"] if case["candidate_id"] == spec.design_id)
        candidate_declaration = next(item for item in four_cell_v2.sample_candidates() if item["candidate_id"] == spec.design_id)
        with _recorded_v2_protocol_hash(dataset["protocol_sha256"]):
            candidate = four_cell_v2.build_candidate(candidate_declaration)
        if candidate.geometry_sha256 != recorded["geometry_sha256"] or candidate.source_sha256 != recorded["source_sha256"]:
            raise ValueError(f"{spec.design_id}: rebuilt candidate differs from the sealed v2 record")
        return {
            "set_id": SET_FOUR_CELL,
            "design_id": spec.design_id,
            "sampling_identity_sha256": candidate_declaration["sampling_identity_sha256"],
            "geometry_sha256": candidate.geometry_sha256,
            "source_sha256": candidate.source_sha256,
            "material_sha256": candidate.material_sha256,
            "representative": spec.representative,
        }
    if spec.set_id == SET_CHARACTERIZATION:
        dataset = v1_dataset()
        recorded = next(case for case in dataset["cases"] if case["case_id"] == spec.design_id)
        definition = next(item for item in characterization_v1.case_definitions() if item.case_id == spec.design_id)
        case = characterization_v1.build_case(definition)
        if case.geometry_sha256 != recorded["geometry_sha256"] or case.source_semantic_sha256 != recorded["source_semantic_sha256"]:
            raise ValueError(f"{spec.design_id}: rebuilt case differs from the sealed v1 record")
        return {
            "set_id": SET_CHARACTERIZATION,
            "design_id": spec.design_id,
            "family_semantic_sha256": definition.family_semantic_sha256,
            "geometry_sha256": case.geometry_sha256,
            "material_semantic_sha256": case.material_semantic_sha256,
            "source_semantic_sha256": case.source_semantic_sha256,
            "representative": spec.representative,
        }
    if spec.set_id == SET_P2:
        v4_protocol = strict_json_file(V4_PROTOCOL_PATH)
        return {
            "set_id": SET_P2,
            "design_id": P2_DESIGN_ID,
            "v4_protocol_file_sha256": _file_sha256(V4_PROTOCOL_PATH),
            "maps": {
                role: {key: v4_protocol["field_adapter"]["maps"][role][key] for key in ("checkpoint_file_sha256", "sidecar_file_sha256", "mesh_sha256", "run_sha256")}
                for role in ("primary", "refined")
            },
            "representative": True,
        }
    raise ValueError(f"unknown design set {spec.set_id}")
