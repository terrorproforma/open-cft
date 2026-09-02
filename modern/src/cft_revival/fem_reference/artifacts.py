"""Hash-anchored result and viewer contracts for the independent FEM reference."""

from __future__ import annotations

from dataclasses import asdict
import copy
from hashlib import sha256
import json
from math import isfinite, log, pi, sqrt
from pathlib import Path
import zipfile

import numpy as np

from .assembly import _QUADRATURE, assemble, p2_shape
from .evidence import evaluate_phase_matched_domain_expansion
from .mesh import adjacent_size_growth, mesh_quality
from .models import (
    Domain,
    FEMProblem,
    FEMResult,
    FEMValidationError,
    P2Mesh,
    Region,
    SheetSource,
    SolverDiagnostics,
    canonical_bytes,
)
from .solver import qois
from .resource_policy import guard_allocation

SCHEMA_VERSION = "cft_revival.fem_reference.result/1.3.0"
LEGACY_SCHEMA_VERSION = "cft_revival.fem_reference.result/1.1.0"
PRIOR_SCHEMA_VERSION = "cft_revival.fem_reference.result/1.2.0"
VIEWER_SCHEMA_VERSION = "cft_revival.fem_reference.viewer/1.1.0"
CLASSIFICATION = "independent_numerical_reference_not_hardware_validation"
MAXIMUM_CHECKPOINT_METADATA_BYTES = 8 * 1024**2

_CHECKPOINT_BINARY_PATHS = (
    ("problem", "free_current_quadrature_a_per_m2"),
    ("mesh", "vertices_rz_m"),
    ("mesh", "triangles"),
    ("mesh", "p2_nodes_rz_m"),
    ("mesh", "element_dofs"),
    ("mesh", "edges"),
    ("mesh", "edge_midpoint_dofs"),
    ("mesh", "element_parent_ids"),
    ("mesh", "interface_edges"),
    ("solution", "a_phi_dofs_t_m"),
    ("solution", "psi_dofs_wb_per_rad"),
)


def _acceptance_code_sha256() -> str:
    digest = sha256()
    root = Path(__file__).parent
    for filename in (
        "adaptivity.py",
        "artifacts.py",
        "assembly.py",
        "evidence.py",
        "mesh.py",
        "models.py",
        "resource_policy.py",
        "solver.py",
    ):
        digest.update(filename.encode())
        digest.update((root / filename).read_bytes())
    return digest.hexdigest()


def _seal(payload: dict[str, object]) -> dict[str, object]:
    return {
        **payload,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": sha256(canonical_bytes(payload)).hexdigest(),
        },
    }


def _chain_sha256(anchors: list[dict[str, object]]) -> str:
    return sha256(canonical_bytes(anchors)).hexdigest()


def _problem_identity_sha256(problem: dict[str, object]) -> str:
    normalized = copy.deepcopy(problem)
    normalized.pop("free_current_quadrature_a_per_m2", None)
    return sha256(canonical_bytes(normalized)).hexdigest()


def _bound_config_id(problem: dict[str, object]) -> str:
    metadata = problem["metadata"]
    return str(
        metadata.get(
            "config_id",
            metadata.get("geometry_config_id", problem["problem_id"]),
        )
    )


def _checkpoint_authority(payload: dict[str, object]) -> dict[str, object]:
    evidence = payload["acceptance_evidence"]
    levels = evidence["level_evidence"]
    domains = evidence["domain_studies"]
    if levels and (
        not isinstance(levels[-1], dict) or "file_sha256" not in levels[-1]
    ):
        raise FEMValidationError("level evidence is not a complete checkpoint anchor")
    problem_sha = _problem_identity_sha256(payload["problem"])
    root_payload = {
        "artifact_schema": payload["schema_version"],
        "classification": payload["classification"],
        "design_id": payload["problem"]["problem_id"],
        "geometry_sha256": payload["anchors"]["geometry_sha256"],
        "magnetics_sha256": payload["anchors"]["magnetics_sha256"],
        "config_id": _bound_config_id(payload["problem"]),
        "implementation_sha256": evidence["implementation_sha256"],
        "acceptance_code_sha256": evidence["acceptance_code_sha256"],
        "problem_sha256": problem_sha,
    }
    return {
        **root_payload,
        "authority_root_sha256": sha256(canonical_bytes(root_payload)).hexdigest(),
        "ordered_level_chain_sha256": _chain_sha256(levels),
        "ordered_domain_chain_sha256": _chain_sha256(domains),
        "final_checkpoint_file_sha256": (
            levels[-1]["file_sha256"] if levels else "0" * 64
        ),
        "final_checkpoint_run_sha256": (
            payload["anchors"]["run_sha256"] if levels else "0" * 64
        ),
        "final_checkpoint_mesh_sha256": (
            payload["anchors"]["mesh_sha256"] if levels else "0" * 64
        ),
    }


def _array_descriptor(values: np.ndarray, dtype: str) -> dict[str, object]:
    array = np.asarray(values, dtype=np.dtype(dtype), order="C")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "order": "C",
        "sha256": sha256(array.tobytes(order="C")).hexdigest(),
    }


def _array_contract(result: FEMResult) -> dict[str, dict[str, object]]:
    mesh = result.mesh
    arrays = {
        "mesh.vertices_rz_m": _array_descriptor(mesh.vertices_rz_m, "<f8"),
        "mesh.triangles": _array_descriptor(mesh.triangles, "<i8"),
        "mesh.p2_nodes_rz_m": _array_descriptor(mesh.p2_nodes_rz_m, "<f8"),
        "mesh.element_dofs": _array_descriptor(mesh.element_dofs, "<i8"),
        "mesh.edges": _array_descriptor(mesh.edges, "<i8"),
        "mesh.edge_midpoint_dofs": _array_descriptor(
            mesh.edge_midpoint_dofs, "<i8"
        ),
        "mesh.element_parent_ids": _array_descriptor(
            mesh.element_parent_ids, "<i8"
        ),
        "mesh.interface_edges": _array_descriptor(mesh.interface_edges, "<i8"),
        "solution.a_phi_dofs_t_m": _array_descriptor(
            result.a_phi_dofs_t_m, "<f8"
        ),
        "solution.psi_dofs_wb_per_rad": _array_descriptor(
            result.psi_dofs_wb_per_rad, "<f8"
        ),
    }
    for name, values in sorted(mesh.boundary_edges.items()):
        arrays[f"mesh.boundary_edges.{name}"] = _array_descriptor(values, "<i8")
    arrays["problem.free_current_quadrature_a_per_m2"] = _array_descriptor(
        _free_current_samples(result.problem, mesh), "<f8"
    )
    return arrays


def _free_current_samples(problem: FEMProblem, mesh: P2Mesh) -> np.ndarray:
    samples = np.empty((len(mesh.triangles), len(_QUADRATURE)), dtype=np.float64)
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        for quadrature, (barycentric, _weight) in enumerate(_QUADRATURE):
            point = np.asarray(barycentric) @ points
            samples[element, quadrature] = problem.free_current_phi(
                float(point[0]), float(point[1])
            )
    if not np.isfinite(samples).all():
        raise FEMValidationError("free-current quadrature evidence is nonfinite")
    return samples


def _guard_result_phase(
    phase: str, result: FEMResult, *, serialized_bytes: int = 0
) -> None:
    guard_allocation(
        phase,
        p2_dofs=len(result.mesh.p2_nodes_rz_m),
        triangles=len(result.mesh.triangles),
        robin_edges=sum(
            len(result.mesh.boundary_edges[name])
            for name in ("outer_radial", "z_min", "z_max")
        ),
        serialized_bytes=serialized_bytes,
    )


def _guard_artifact_phase(
    phase: str, artifact: dict[str, object], *, serialized_bytes: int = 0
) -> None:
    mesh = artifact["mesh"]
    guard_allocation(
        phase,
        p2_dofs=len(mesh["p2_nodes_rz_m"]),
        triangles=len(mesh["triangles"]),
        robin_edges=sum(
            len(mesh["boundary_edges"][name])
            for name in ("outer_radial", "z_min", "z_max")
        ),
        serialized_bytes=serialized_bytes,
    )


def _recovered_vertex_fields(result: FEMResult) -> list[tuple[float, float, float]]:
    """Area-average element traces for lightweight viewer data, not interface maxima."""

    mesh = result.mesh
    br = np.zeros(len(mesh.vertices_rz_m))
    bz = np.zeros(len(mesh.vertices_rz_m))
    weights = np.zeros(len(mesh.vertices_rz_m))
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        jacobian = np.column_stack((points[1] - points[0], points[2] - points[0]))
        determinant = float(np.linalg.det(jacobian))
        inverse_transpose = np.linalg.inv(jacobian).T
        grad_lambda = np.empty((3, 2))
        grad_lambda[1] = inverse_transpose[:, 0]
        grad_lambda[2] = inverse_transpose[:, 1]
        grad_lambda[0] = -grad_lambda[1] - grad_lambda[2]
        coefficients = result.a_phi_dofs_t_m[mesh.element_dofs[element]]
        for local_vertex in range(3):
            barycentric = np.zeros(3)
            barycentric[local_vertex] = 1.0
            values, gradients = p2_shape(barycentric, grad_lambda)
            a_phi = float(np.dot(values, coefficients))
            gradient = coefficients @ gradients
            r_m = float(points[local_vertex, 0])
            local_br = 0.0 if r_m == 0.0 else -float(gradient[1])
            local_bz = (
                2.0 * float(gradient[0])
                if r_m == 0.0
                else a_phi / r_m + float(gradient[0])
            )
            vertex = int(triangle[local_vertex])
            br[vertex] += determinant * local_br
            bz[vertex] += determinant * local_bz
            weights[vertex] += determinant
    psi = mesh.vertices_rz_m[:, 0] * result.a_phi_dofs_t_m[: len(mesh.vertices_rz_m)]
    return [
        (float(psi[index]), float(br[index] / weights[index]), float(bz[index] / weights[index]))
        for index in range(len(mesh.vertices_rz_m))
    ]


def artifact_from_result(
    result: FEMResult,
    *,
    qoi_values: dict[str, float] | None = None,
    qoi_windows: tuple[tuple[str, float, float, float], ...] = (),
    level_evidence: list[dict[str, object]] | None = None,
    domain_studies: list[dict[str, object]] | None = None,
    evidence_base_path: str = ".",
    convergence: dict[str, object] | None = None,
    comparisons: dict[str, object] | None = None,
) -> dict[str, object]:
    mesh = result.mesh
    binary_bytes = (
        mesh.vertices_rz_m.nbytes
        + mesh.triangles.nbytes
        + mesh.p2_nodes_rz_m.nbytes
        + mesh.element_dofs.nbytes
        + result.a_phi_dofs_t_m.nbytes
    )
    _guard_result_phase(
        "artifact_serialization_construction",
        result,
        serialized_bytes=binary_bytes,
    )
    if qoi_windows:
        recomputed_qois = qois(result, qoi_windows)
        if qoi_values is None:
            qoi_values = recomputed_qois
        elif qoi_values != recomputed_qois:
            raise FEMValidationError("provided QoIs differ from bound solution")
    vertex_fields = _recovered_vertex_fields(result)
    diagnostics = asdict(result.diagnostics)
    diagnostics["residual_history_l2"] = list(result.diagnostics.residual_history_l2)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_level": "independent-P2-FEM-reference",
        "classification": CLASSIFICATION,
        "method": {
            "unknown": "A_phi in continuous quadratic Lagrange P2",
            "represented_flux": "psi=r*A_phi with psi=O(r^2) on axis",
            "weak_operator": "integral nu/r grad(psi).grad(v_psi) d(r,z)",
            "remanence_action": "integral nu*(Br_z*d_r(v_psi)-Br_r*d_z(v_psi)) d(r,z)",
            "outer_boundary": "corrected local dipole Robin",
            "quadrature": "Dunavant degree-5 volume; 3-point Gauss edge",
            "linear_algebra": "native vectorized CSR + IC(0)/Jacobi-PCG, binary64 CPU",
        },
        "anchors": {
            "geometry_sha256": result.problem.geometry_sha256,
            "magnetics_sha256": result.problem.magnetics_sha256,
            "mesh_sha256": mesh.sha256,
            "solution_sha256": sha256(result.a_phi_dofs_t_m.tobytes()).hexdigest(),
            "run_sha256": result.run_sha256,
        },
        "problem": {
            "problem_id": result.problem.problem_id,
            "domain": result.problem.domain.to_dict(),
            "regions": [asdict(region) for region in result.problem.regions],
            "sheets": [asdict(sheet) for sheet in result.problem.sheets],
            "free_current_quadrature_a_per_m2": _free_current_samples(
                result.problem, mesh
            ).tolist(),
            "source_center_z_m": result.problem.source_center_z_m,
            "outer_boundary": result.problem.outer_boundary,
            "metadata": dict(result.problem.metadata),
        },
        "mesh": {
            "layout": "P2: vertices then globally unique edge midpoints",
            "vertices_rz_m": mesh.vertices_rz_m.tolist(),
            "triangles": mesh.triangles.tolist(),
            "triangle_region_ids": list(mesh.triangle_region_ids),
            "p2_nodes_rz_m": mesh.p2_nodes_rz_m.tolist(),
            "element_dofs": mesh.element_dofs.tolist(),
            "edges": mesh.edges.tolist(),
            "edge_midpoint_dofs": mesh.edge_midpoint_dofs.tolist(),
            "boundary_edges": {
                name: values.tolist() for name, values in sorted(mesh.boundary_edges.items())
            },
            "element_parent_ids": mesh.element_parent_ids.tolist(),
            "interface_edges": mesh.interface_edges.tolist(),
            "interface_region_pairs": [list(pair) for pair in mesh.interface_region_pairs],
            "refinement_level": mesh.refinement_level,
            "parent_mesh_sha256": mesh.parent_mesh_sha256,
            "protected_radii_m": list(mesh.protected_radii_m),
            "protected_z_m": list(mesh.protected_z_m),
            "quality": mesh_quality(mesh),
        },
        "solution": {
            "a_phi_dofs_t_m": result.a_phi_dofs_t_m.tolist(),
            "psi_dofs_wb_per_rad": result.psi_dofs_wb_per_rad.tolist(),
            "vertex_psi_wb_per_rad": [value[0] for value in vertex_fields],
            "vertex_b_r_t": [value[1] for value in vertex_fields],
            "vertex_b_z_t": [value[2] for value in vertex_fields],
        },
        "diagnostics": diagnostics,
        "qois_bz_t": qoi_values or {},
        "acceptance_evidence": {
            "qoi_windows": [list(window) for window in qoi_windows],
            "level_evidence": level_evidence or [],
            "domain_studies": domain_studies or [],
            "evidence_base_path": evidence_base_path,
            "solver_controls": dict(result.solver_controls),
            "implementation_sha256": result.implementation_sha256,
            "acceptance_code_sha256": _acceptance_code_sha256(),
            "array_contract": _array_contract(result),
            "initial_solution_sha256": result.initial_solution_sha256,
            "authority": "recompute_from_bound_mesh_solution_and_inputs",
        },
        "convergence": convergence or {},
        "comparisons": comparisons or {},
        "limitations": [
            "Independent numerical reference, not hardware validation.",
            "Linear isotropic reluctivity and linear recoil remanence only.",
            "Local interface maxima are screening-only; use fixed/bore integral QoIs.",
            "The local dipole Robin condition remains a finite-domain approximation.",
        ],
    }
    payload["acceptance_evidence"]["checkpoint_authority"] = _checkpoint_authority(
        payload
    )
    artifact = _seal(payload)
    validate_artifact(artifact)
    return artifact


def artifact_from_bound_chain(
    bound_artifact: dict[str, object],
    *,
    level_evidence: list[dict[str, object]],
    domain_studies: list[dict[str, object]],
    evidence_base_path: str,
    convergence: dict[str, object],
    comparisons: dict[str, object],
) -> dict[str, object]:
    """Promote a completed bound checkpoint without rerunning its solve."""

    artifact = copy.deepcopy(bound_artifact)
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact["acceptance_evidence"]["level_evidence"]
        or artifact["acceptance_evidence"]["domain_studies"]
    ):
        raise FEMValidationError("chain promotion requires a current leaf artifact")
    evidence = artifact["acceptance_evidence"]
    evidence["level_evidence"] = copy.deepcopy(level_evidence)
    evidence["domain_studies"] = copy.deepcopy(domain_studies)
    evidence["evidence_base_path"] = evidence_base_path
    artifact["convergence"] = copy.deepcopy(convergence)
    artifact["comparisons"] = copy.deepcopy(comparisons)
    payload = {key: value for key, value in artifact.items() if key != "integrity"}
    evidence["checkpoint_authority"] = _checkpoint_authority(payload)
    promoted = _seal(payload)
    validate_artifact(promoted)
    return promoted


def refresh_bound_artifact_authority(
    bound_artifact: dict[str, object],
) -> dict[str, object]:
    """Migrate a numerical leaf to the current acceptance-code authority."""

    artifact = copy.deepcopy(bound_artifact)
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise FEMValidationError("only current-schema leaves can be refreshed")
    evidence = artifact["acceptance_evidence"]
    if evidence["level_evidence"] or evidence["domain_studies"]:
        raise FEMValidationError("only leaf artifacts can be refreshed")
    evidence["acceptance_code_sha256"] = _acceptance_code_sha256()
    payload = {key: value for key, value in artifact.items() if key != "integrity"}
    evidence["checkpoint_authority"] = _checkpoint_authority(payload)
    return _seal(payload)


def replay_artifact(artifact: dict[str, object]) -> dict[str, object]:
    _guard_artifact_phase("artifact_replay", artifact)
    validate_artifact(artifact, replay=False)
    mesh = artifact["mesh"]
    solution = artifact["solution"]
    domain = Domain(**artifact["problem"]["domain"])
    rebuilt = P2Mesh(
        np.asarray(mesh["vertices_rz_m"], dtype=np.float64),
        np.asarray(mesh["triangles"], dtype=np.int64),
        tuple(mesh["triangle_region_ids"]),
        np.asarray(mesh["p2_nodes_rz_m"], dtype=np.float64),
        np.asarray(mesh["element_dofs"], dtype=np.int64),
        np.asarray(mesh["edges"], dtype=np.int64),
        np.asarray(mesh["edge_midpoint_dofs"], dtype=np.int64),
        {
            name: np.asarray(values, dtype=np.int64)
            for name, values in mesh["boundary_edges"].items()
        },
        np.asarray(mesh["element_parent_ids"], dtype=np.int64),
        np.asarray(mesh["interface_edges"], dtype=np.int64),
        tuple(tuple(pair) for pair in mesh["interface_region_pairs"]),
        int(mesh["refinement_level"]),
        mesh["parent_mesh_sha256"],
        tuple(mesh["protected_radii_m"]),
        tuple(mesh["protected_z_m"]),
    )
    expected_coordinates = {
        "axis": (0, domain.r_min_m),
        "inner_radial": (0, domain.r_min_m),
        "outer_radial": (0, domain.r_max_m),
        "z_min": (1, domain.z_min_m),
        "z_max": (1, domain.z_max_m),
    }
    for name, edge_indices in rebuilt.boundary_edges.items():
        axis, expected_coordinate = expected_coordinates[name]
        points = rebuilt.vertices_rz_m[rebuilt.edges[edge_indices], axis]
        if not np.allclose(points, expected_coordinate, rtol=0.0, atol=1.0e-13):
            raise FEMValidationError("artifact boundary ownership geometry differs")
    if mesh_quality(rebuilt) != mesh["quality"]:
        raise FEMValidationError("artifact mesh quality does not replay")
    a_phi = np.asarray(solution["a_phi_dofs_t_m"], dtype="<f8")
    nodes = np.asarray(mesh["p2_nodes_rz_m"], dtype=np.float64)
    psi = nodes[:, 0] * a_phi
    return {
        "mesh_sha256": rebuilt.sha256,
        "solution_sha256": sha256(a_phi.tobytes()).hexdigest(),
        "psi_max_absolute_replay_error": float(
            np.max(np.abs(psi - np.asarray(solution["psi_dofs_wb_per_rad"])))
        ),
        "passed": bool(
            rebuilt.sha256 == artifact["anchors"]["mesh_sha256"]
            and sha256(a_phi.tobytes()).hexdigest()
            == artifact["anchors"]["solution_sha256"]
            and np.array_equal(psi, np.asarray(solution["psi_dofs_wb_per_rad"]))
        ),
        "acceptance_authority": (
            "recomputed"
            if artifact["schema_version"] == SCHEMA_VERSION
            else "legacy_integrity_only_screening"
        ),
    }


def _rebuild_result(artifact: dict[str, object], mesh: P2Mesh) -> FEMResult:
    problem_data = artifact["problem"]
    anchors = artifact["anchors"]
    current_samples = np.asarray(
        problem_data["free_current_quadrature_a_per_m2"], dtype=np.float64
    )
    if current_samples.shape != (len(mesh.triangles), len(_QUADRATURE)):
        raise FEMValidationError("free-current quadrature shape differs")
    current_lookup: dict[tuple[float, float], float] = {}
    for element, triangle in enumerate(mesh.triangles):
        points = mesh.vertices_rz_m[triangle]
        for quadrature, (barycentric, _weight) in enumerate(_QUADRATURE):
            point = np.asarray(barycentric) @ points
            current_lookup[(float(point[0]), float(point[1]))] = float(
                current_samples[element, quadrature]
            )

    def bound_current(r_m: float, z_m: float) -> float:
        try:
            return current_lookup[(float(r_m), float(z_m))]
        except KeyError as error:
            raise FEMValidationError(
                "free-current evaluation lies outside bound quadrature"
            ) from error

    problem = FEMProblem(
        problem_data["problem_id"],
        Domain(**problem_data["domain"]),
        tuple(Region(**region) for region in problem_data["regions"]),
        lambda _r, _z: "ambient-background",
        free_current_phi=bound_current,
        sheets=tuple(SheetSource(**sheet) for sheet in problem_data["sheets"]),
        source_center_z_m=float(problem_data["source_center_z_m"]),
        outer_boundary=problem_data["outer_boundary"],
        geometry_sha256=anchors["geometry_sha256"],
        magnetics_sha256=anchors["magnetics_sha256"],
        metadata=tuple(sorted(problem_data["metadata"].items())),
    )
    diagnostics = SolverDiagnostics(
        True, 0, 0.0, 0.0, 0.0, (), 0.0, 0.0, 0.0, 0.0, 0.0, 0
    )
    evidence = artifact["acceptance_evidence"]
    if evidence["acceptance_code_sha256"] != _acceptance_code_sha256():
        raise FEMValidationError("artifact acceptance code evidence is stale")
    return FEMResult(
        problem,
        mesh,
        np.asarray(artifact["solution"]["a_phi_dofs_t_m"], dtype=np.float64),
        diagnostics,
        artifact["anchors"]["run_sha256"],
        tuple(sorted(evidence["solver_controls"].items())),
        evidence["implementation_sha256"],
        evidence["initial_solution_sha256"],
    )


def _relative(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def _mesh_from_artifact(artifact: dict[str, object]) -> P2Mesh:
    mesh = artifact["mesh"]
    return P2Mesh(
        np.asarray(mesh["vertices_rz_m"], dtype=np.float64),
        np.asarray(mesh["triangles"], dtype=np.int64),
        tuple(mesh["triangle_region_ids"]),
        np.asarray(mesh["p2_nodes_rz_m"], dtype=np.float64),
        np.asarray(mesh["element_dofs"], dtype=np.int64),
        np.asarray(mesh["edges"], dtype=np.int64),
        np.asarray(mesh["edge_midpoint_dofs"], dtype=np.int64),
        {
            name: np.asarray(values, dtype=np.int64)
            for name, values in mesh["boundary_edges"].items()
        },
        np.asarray(mesh["element_parent_ids"], dtype=np.int64),
        np.asarray(mesh["interface_edges"], dtype=np.int64),
        tuple(tuple(pair) for pair in mesh["interface_region_pairs"]),
        int(mesh["refinement_level"]),
        mesh["parent_mesh_sha256"],
        tuple(mesh["protected_radii_m"]),
        tuple(mesh["protected_z_m"]),
    )


def _qoi_h(mesh: P2Mesh, windows: tuple[tuple[str, float, float, float], ...]):
    points = mesh.vertices_rz_m
    sizes = np.empty(len(mesh.triangles))
    centroids = np.empty((len(mesh.triangles), 2))
    for element, triangle in enumerate(mesh.triangles):
        triangle_points = points[triangle]
        first = triangle_points[1] - triangle_points[0]
        second = triangle_points[2] - triangle_points[0]
        sizes[element] = sqrt(
            abs(float(first[0] * second[1] - first[1] * second[0]))
        )
        centroids[element] = np.mean(triangle_points, axis=0)
    output: dict[str, float] = {}
    for name, radius, z_min, z_max in windows:
        selected = (
            (centroids[:, 0] <= radius)
            & (centroids[:, 1] >= z_min)
            & (centroids[:, 1] <= z_max)
        )
        if not np.any(selected):
            raise FEMValidationError("bound QoI window contains no mesh elements")
        value = float(sqrt(float(np.mean(sizes[selected] ** 2))))
        if not isfinite(value) or value <= 0.0:
            raise FEMValidationError("bound QoI local h is invalid")
        output[f"{name}-bore-average"] = value
    return output


def _bound_local_h(artifact: dict[str, object], mesh: P2Mesh, qoi_h):
    points = mesh.vertices_rz_m
    sizes = np.empty(len(mesh.triangles))
    for element, triangle in enumerate(mesh.triangles):
        triangle_points = points[triangle]
        first = triangle_points[1] - triangle_points[0]
        second = triangle_points[2] - triangle_points[0]
        sizes[element] = sqrt(
            abs(float(first[0] * second[1] - first[1] * second[0]))
        )
    source_regions = {
        region["region_id"]
        for region in artifact["problem"]["regions"]
        if (
            abs(float(region["remanence_r_t"])) > 0.0
            or abs(float(region["remanence_z_t"])) > 0.0
        )
    }
    selected = np.fromiter(
        (region_id in source_regions for region_id in mesh.triangle_region_ids),
        dtype=bool,
        count=len(mesh.triangles),
    )
    current_samples = np.asarray(
        artifact["problem"]["free_current_quadrature_a_per_m2"], dtype=np.float64
    )
    selected |= np.any(current_samples != 0.0, axis=1)
    if not np.any(selected):
        raise FEMValidationError("bound domain study contains no source-region cells")
    source_h = float(sqrt(float(np.mean(sizes[selected] ** 2))))
    if not isfinite(source_h) or source_h <= 0.0:
        raise FEMValidationError("bound source local h is invalid")
    return {"source": source_h, **qoi_h}


def domain_study_evidence(
    bound_artifact: dict[str, object], padding_factor: float
) -> dict[str, object]:
    """Derive domain-study values exclusively from a validated bound artifact."""

    validate_artifact(bound_artifact)
    mesh = _mesh_from_artifact(bound_artifact)
    windows = tuple(
        (str(item[0]), float(item[1]), float(item[2]), float(item[3]))
        for item in bound_artifact["acceptance_evidence"]["qoi_windows"]
    )
    qoi_h = _qoi_h(mesh, windows)
    return {
        "padding_factor": float(padding_factor),
        "qois_bz_t": dict(bound_artifact["qois_bz_t"]),
        "qoi_h_m": qoi_h,
        "local_h_m": _bound_local_h(bound_artifact, mesh, qoi_h),
        "domain": dict(bound_artifact["problem"]["domain"]),
    }


def _read_bounded_checkpoint_metadata(path: Path) -> tuple[dict[str, object], int]:
    size = path.stat().st_size
    if size > MAXIMUM_CHECKPOINT_METADATA_BYTES:
        guard_allocation(
            "legacy_checkpoint_migration",
            p2_dofs=1_500_000,
            triangles=1_500_000,
            serialized_bytes=size,
        )
    else:
        guard_allocation(
            "checkpoint_metadata_parse",
            p2_dofs=1,
            triangles=1,
            serialized_bytes=size,
        )
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise FEMValidationError("checkpoint metadata must be an object")
    return value, size


def _inspect_checkpoint_sidecar(
    checkpoint: dict[str, object], metadata_path: Path
) -> tuple[Path, dict[str, tuple[tuple[int, ...], str, int]], int]:
    metadata = checkpoint["array_sidecar"]
    sidecar_path = (metadata_path.parent / metadata["file"]).resolve()
    if not sidecar_path.is_relative_to(metadata_path.parent.resolve()):
        raise FEMValidationError("checkpoint array sidecar escapes evidence root")
    headers: dict[str, tuple[tuple[int, ...], str, int]] = {}
    with zipfile.ZipFile(sidecar_path) as archive:
        members = {
            member.filename[:-4]: member
            for member in archive.infolist()
            if member.filename.endswith(".npy")
        }
        if set(members) != set(metadata["array_keys"]):
            raise FEMValidationError("checkpoint array sidecar keys differ")
        for key, member in members.items():
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                        source
                    )
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran, dtype = np.lib.format._read_array_header(
                        source, version
                    )
                else:
                    raise FEMValidationError("unsupported checkpoint NPY version")
            count = 1
            for dimension in shape:
                if (
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 0
                ):
                    raise FEMValidationError("checkpoint array shape is invalid")
                count *= dimension
            data_bytes = count * dtype.itemsize
            if data_bytes > member.file_size:
                raise FEMValidationError("checkpoint NPY member size is invalid")
            descriptor = checkpoint["bound_artifact"]["acceptance_evidence"][
                "array_contract"
            ][key]
            expected_order = "F" if fortran else "C"
            if (
                list(shape) != descriptor["shape"]
                or dtype.str != descriptor["dtype"]
                or expected_order != descriptor["order"]
            ):
                raise FEMValidationError("checkpoint binary header contract differs")
            headers[key] = (tuple(shape), dtype.str, data_bytes)
    total_data_bytes = sum(item[2] for item in headers.values())
    if total_data_bytes != metadata["uncompressed_array_bytes"]:
        raise FEMValidationError("checkpoint sidecar byte count differs")
    return sidecar_path, headers, total_data_bytes


def _actual_checkpoint_topology(
    checkpoint: dict[str, object],
    headers: dict[str, tuple[tuple[int, ...], str, int]] | None,
) -> tuple[int, int, int]:
    if headers is None:
        mesh = checkpoint["bound_artifact"]["mesh"]
        p2_dofs = len(mesh["p2_nodes_rz_m"])
        triangles = len(mesh["triangles"])
        robin_edges = sum(
            len(mesh["boundary_edges"][name])
            for name in ("outer_radial", "z_min", "z_max")
        )
        return p2_dofs, triangles, robin_edges
    p2_shape = headers["mesh.p2_nodes_rz_m"][0]
    triangle_shape = headers["mesh.triangles"][0]
    if len(p2_shape) != 2 or p2_shape[1] != 2:
        raise FEMValidationError("checkpoint P2-node header shape differs")
    if len(triangle_shape) != 2 or triangle_shape[1] != 3:
        raise FEMValidationError("checkpoint triangle header shape differs")
    robin_edges = 0
    for name in ("outer_radial", "z_min", "z_max"):
        shape = headers[f"mesh.boundary_edges.{name}"][0]
        if len(shape) != 1:
            raise FEMValidationError("checkpoint boundary-edge header shape differs")
        robin_edges += shape[0]
    return p2_shape[0], triangle_shape[0], robin_edges


def load_checkpoint_bundle(
    path: Path,
    *,
    expected_counts: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, int | str]]:
    """Load a bounded checkpoint using verified binary headers, never anchors."""

    checkpoint, metadata_bytes = _read_bounded_checkpoint_metadata(path)
    file_hash = _stream_file_sha256(path)
    payload = {key: value for key, value in checkpoint.items() if key != "integrity"}
    payload_hash = sha256(canonical_bytes(payload)).hexdigest()
    if checkpoint.get("integrity") != {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "payload_sha256": payload_hash,
    }:
        raise FEMValidationError("checkpoint payload hash differs")
    sidecar_metadata = checkpoint.get("array_sidecar")
    sidecar_path = None
    headers = None
    array_bytes = 0
    if sidecar_metadata is not None:
        sidecar_path, headers, array_bytes = _inspect_checkpoint_sidecar(
            checkpoint, path
        )
    actual_p2, actual_triangles, actual_robin = _actual_checkpoint_topology(
        checkpoint, headers
    )
    if expected_counts is not None:
        for name, actual in (
            ("p2_dofs", actual_p2),
            ("triangles", actual_triangles),
            ("robin_edges", actual_robin),
        ):
            claimed = expected_counts[name]
            if (
                isinstance(claimed, bool)
                or not isinstance(claimed, int)
                or claimed != actual
            ):
                raise FEMValidationError(
                    f"checkpoint anchor {name} differs from verified headers"
                )
    guard_allocation(
        "checkpoint_verified_load",
        p2_dofs=actual_p2,
        triangles=actual_triangles,
        robin_edges=actual_robin,
        serialized_bytes=metadata_bytes + array_bytes,
    )
    if sidecar_metadata is not None:
        if _stream_file_sha256(sidecar_path) != sidecar_metadata["file_sha256"]:
            raise FEMValidationError("checkpoint array sidecar hash differs")
        with np.load(sidecar_path, allow_pickle=False) as archive:
            bound_skeleton = checkpoint["bound_artifact"]
            paths = list(_CHECKPOINT_BINARY_PATHS)
            paths.extend(
                ("mesh", "boundary_edges", name)
                for name in sorted(bound_skeleton["mesh"]["boundary_edges"])
            )
            for parts in paths:
                target = bound_skeleton
                for part in parts[:-1]:
                    target = target[part]
                key = ".".join(parts)
                if target[parts[-1]] != {"$binary_array": key}:
                    raise FEMValidationError("checkpoint binary reference differs")
                array = archive[key]
                descriptor = bound_skeleton["acceptance_evidence"][
                    "array_contract"
                ][key]
                if _array_descriptor(array, descriptor["dtype"]) != descriptor:
                    raise FEMValidationError("checkpoint binary array contract differs")
                target[parts[-1]] = array.tolist()
    return checkpoint, {
        "file_sha256": file_hash,
        "payload_sha256": payload_hash,
        "p2_dofs": actual_p2,
        "triangles": actual_triangles,
        "robin_edges": actual_robin,
        "metadata_bytes": metadata_bytes,
        "array_bytes": array_bytes,
    }


def checkpoint_metadata_summary(path: Path) -> dict[str, int | str]:
    checkpoint, metadata_bytes = _read_bounded_checkpoint_metadata(path)
    payload = {key: value for key, value in checkpoint.items() if key != "integrity"}
    payload_hash = sha256(canonical_bytes(payload)).hexdigest()
    if checkpoint.get("integrity", {}).get("payload_sha256") != payload_hash:
        raise FEMValidationError("checkpoint metadata payload hash differs")
    return {
        "file_sha256": _stream_file_sha256(path),
        "payload_sha256": payload_hash,
        "metadata_bytes": metadata_bytes,
    }


def _load_bound_checkpoint(
    anchor: dict[str, object],
    base: Path,
    previous_file_hash: str,
    previous_mesh_hash: str,
    authority: dict[str, object],
    chain_kind: str,
) -> tuple[dict[str, object], str, str]:
    required = {
        "level",
        "file",
        "file_sha256",
        "payload_sha256",
        "mesh_sha256",
        "parent_mesh_sha256",
        "previous_checkpoint_file_sha256",
        "p2_dofs",
        "triangles",
        "robin_edges",
        "chain_final_run_sha256",
        "chain_final_mesh_sha256",
        "run_sha256",
        "problem_sha256",
    }
    optional = {"padding_factor", "local_h_m", "domain"}
    if not required.issubset(anchor) or set(anchor) - required - optional:
        raise FEMValidationError("level evidence is not a complete checkpoint anchor")
    path = (base / str(anchor["file"])).resolve()
    if not path.is_relative_to(base.resolve()):
        raise FEMValidationError("checkpoint path escapes evidence root")
    checkpoint, verified = load_checkpoint_bundle(path, expected_counts=anchor)
    file_hash = str(verified["file_sha256"])
    if file_hash != anchor["file_sha256"]:
        raise FEMValidationError("checkpoint file hash differs")
    if verified["payload_sha256"] != anchor["payload_sha256"]:
        raise FEMValidationError("checkpoint payload hash differs")
    if (
        anchor["previous_checkpoint_file_sha256"] != previous_file_hash
        or checkpoint["previous_checkpoint_file_sha256"] != previous_file_hash
        or checkpoint["mesh_sha256"] != anchor["mesh_sha256"]
        or checkpoint["parent_mesh_sha256"] != anchor["parent_mesh_sha256"]
    ):
        raise FEMValidationError("checkpoint chain authority differs")
    expected_chain_authority = {
        "authority_root_sha256": authority["authority_root_sha256"],
        "artifact_schema": authority["artifact_schema"],
        "classification": authority["classification"],
        "design_id": authority["design_id"],
        "geometry_sha256": authority["geometry_sha256"],
        "magnetics_sha256": authority["magnetics_sha256"],
        "config_id": authority["config_id"],
        "implementation_sha256": authority["implementation_sha256"],
        "acceptance_code_sha256": authority["acceptance_code_sha256"],
        "base_problem_sha256": authority["problem_sha256"],
        "chain_kind": chain_kind,
        "final_checkpoint_run_sha256": anchor["chain_final_run_sha256"],
        "final_checkpoint_mesh_sha256": anchor["chain_final_mesh_sha256"],
    }
    if checkpoint.get("chain_authority") != expected_chain_authority:
        raise FEMValidationError("checkpoint belongs to unrelated authority chain")
    if int(anchor["level"]) > 0 and anchor["parent_mesh_sha256"] != previous_mesh_hash:
        raise FEMValidationError("checkpoint parent mesh ancestry differs")
    if "padding_factor" in anchor:
        domain_study = checkpoint.get("domain_study")
        try:
            bound_padding = float(domain_study["padding_factor"])
            anchor_padding = float(anchor["padding_factor"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise FEMValidationError("domain padding evidence is invalid") from error
        if not isfinite(bound_padding) or bound_padding != anchor_padding:
            raise FEMValidationError("domain padding differs from bound checkpoint")
    bound = checkpoint.get("bound_artifact")
    if not isinstance(bound, dict):
        raise FEMValidationError("checkpoint lacks complete bound mesh/solution evidence")
    validate_artifact(bound)
    bound_evidence = bound["acceptance_evidence"]
    bound_authority = bound_evidence["checkpoint_authority"]
    bound_problem_sha = _problem_identity_sha256(bound["problem"])
    bound_config = _bound_config_id(bound["problem"])
    identity_checks = (
        (bound["schema_version"], authority["artifact_schema"], "artifact schema"),
        (bound["classification"], authority["classification"], "classification"),
        (bound["problem"]["problem_id"], authority["design_id"], "design"),
        (
            bound["anchors"]["geometry_sha256"],
            authority["geometry_sha256"],
            "geometry",
        ),
        (
            bound["anchors"]["magnetics_sha256"],
            authority["magnetics_sha256"],
            "magnetics",
        ),
        (bound_config, authority["config_id"], "config"),
        (checkpoint["config_id"], authority["config_id"], "checkpoint config"),
        (
            bound_evidence["implementation_sha256"],
            authority["implementation_sha256"],
            "implementation code",
        ),
        (
            bound_evidence["acceptance_code_sha256"],
            authority["acceptance_code_sha256"],
            "acceptance code",
        ),
        (bound_problem_sha, anchor["problem_sha256"], "problem"),
        (bound["anchors"]["run_sha256"], anchor["run_sha256"], "run"),
        (checkpoint["run_sha256"], anchor["run_sha256"], "checkpoint run"),
    )
    for actual, expected, name in identity_checks:
        if actual != expected:
            raise FEMValidationError(
                f"checkpoint bound {name} identity differs from chain authority"
            )
    if chain_kind == "adaptive" and bound_problem_sha != authority["problem_sha256"]:
        raise FEMValidationError(
            "checkpoint bound problem identity differs from top-level artifact"
        )
    for key in (
        "artifact_schema",
        "classification",
        "design_id",
        "geometry_sha256",
        "magnetics_sha256",
        "config_id",
        "implementation_sha256",
        "acceptance_code_sha256",
        "problem_sha256",
    ):
        expected = bound_problem_sha if key == "problem_sha256" else (
            bound["problem"]["problem_id"] if key == "design_id" else
            bound_config if key == "config_id" else
            authority[key]
        )
        if bound_authority[key] != expected:
            raise FEMValidationError(
                "bound artifact authority identity differs from checkpoint"
            )
    if bound["anchors"]["mesh_sha256"] != anchor["mesh_sha256"]:
        raise FEMValidationError("checkpoint bound mesh hash differs")
    windows = tuple(
        (str(item[0]), float(item[1]), float(item[2]), float(item[3]))
        for item in bound["acceptance_evidence"]["qoi_windows"]
    )
    mesh = _mesh_from_artifact(bound)
    recomputed_h = _qoi_h(mesh, windows)
    local_h = (
        _bound_local_h(bound, mesh, recomputed_h)
        if chain_kind == "domain"
        else {}
    )
    recorded_h = checkpoint["run"]["resolution"]["qoi_h_m"]
    if set(recomputed_h) != set(recorded_h) or any(
        not np.isclose(recomputed_h[key], recorded_h[key], rtol=2.0e-13, atol=0.0)
        for key in recomputed_h
    ):
        raise FEMValidationError("checkpoint bound local h differs")
    if checkpoint["run"]["qois_bz_t"] != bound["qois_bz_t"]:
        raise FEMValidationError("checkpoint QoIs differ from bound solution")
    return (
        {
            "qois_bz_t": bound["qois_bz_t"],
            "resolution": {"qoi_h_m": recomputed_h},
            "adjacent_area_size_growth": adjacent_size_growth(mesh),
            "local_h_m": local_h,
            "domain": bound["problem"]["domain"],
            "run_sha256": bound["anchors"]["run_sha256"],
        },
        file_hash,
        str(anchor["mesh_sha256"]),
    )


def _validate_acceptance_replay(
    artifact: dict[str, object], mesh: P2Mesh
) -> None:
    evidence = artifact["acceptance_evidence"]
    payload_without_integrity = {
        key: value for key, value in artifact.items() if key != "integrity"
    }
    if evidence["checkpoint_authority"] != _checkpoint_authority(
        payload_without_integrity
    ):
        raise FEMValidationError("top-level checkpoint chain authority differs")
    authority = evidence["checkpoint_authority"]
    result = _rebuild_result(artifact, mesh)
    if evidence["array_contract"] != _array_contract(result):
        raise FEMValidationError("artifact dtype/shape/endian array contract differs")
    windows = tuple(
        (str(item[0]), float(item[1]), float(item[2]), float(item[3]))
        for item in evidence["qoi_windows"]
    )
    if windows:
        recomputed_qois = qois(result, windows)
        if set(recomputed_qois) != set(artifact["qois_bz_t"]):
            raise FEMValidationError("artifact QoI keys differ from bound windows")
        for key, value in recomputed_qois.items():
            if not np.isclose(
                value, artifact["qois_bz_t"][key], rtol=2.0e-12, atol=1.0e-14
            ):
                raise FEMValidationError("artifact QoI differs from bound solution")

    system = assemble(result.problem, mesh)
    solution = result.a_phi_dofs_t_m
    if not np.allclose(
        solution[system.prescribed_dofs],
        system.prescribed_values,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise FEMValidationError("artifact prescribed solution values differ")
    free_solution = solution[system.free_dofs]
    true_residual = system.rhs - system.matrix.matvec(free_solution)
    true_norm = float(np.linalg.norm(true_residual))
    relative_residual = true_norm / max(
        float(np.linalg.norm(system.rhs)), 1.0e-300
    )
    magnetic = pi * float(
        np.dot(free_solution, system.matrix.matvec(free_solution))
    )
    source = pi * float(np.dot(solution, system.physical_load))
    energy_relative = abs(magnetic - source) / max(
        abs(magnetic), abs(source), 1.0e-300
    )
    diagnostics = artifact["diagnostics"]
    for recomputed, stored, name in (
        (true_norm, diagnostics["final_true_residual_l2"], "true residual"),
        (
            relative_residual,
            diagnostics["relative_true_residual_l2"],
            "relative true residual",
        ),
        (magnetic, diagnostics["magnetic_action_j"], "magnetic action"),
        (source, diagnostics["source_action_j"], "source action"),
        (
            energy_relative,
            diagnostics["energy_action_relative"],
            "energy action",
        ),
    ):
        if not np.isclose(recomputed, stored, rtol=2.0e-11, atol=1.0e-13):
            raise FEMValidationError(f"artifact {name} does not replay")

    controls = evidence["solver_controls"]
    run_payload = {
        "problem_id": result.problem.problem_id,
        "mesh_sha256": mesh.sha256,
        "geometry_sha256": result.problem.geometry_sha256,
        "magnetics_sha256": result.problem.magnetics_sha256,
        "implementation_sha256": evidence["implementation_sha256"],
        "relative_tolerance": controls["relative_tolerance"],
        "absolute_tolerance": controls["absolute_tolerance"],
        "max_iterations": controls["max_iterations"],
        "required_available_ram_bytes": controls["required_available_ram_bytes"],
        "initial_solution_sha256": evidence["initial_solution_sha256"],
        "solution_sha256": sha256(solution.tobytes()).hexdigest(),
    }
    if sha256(canonical_bytes(run_payload)).hexdigest() != artifact["anchors"]["run_sha256"]:
        raise FEMValidationError("artifact run evidence hash differs")

    comparison_group = artifact["comparisons"]
    comparisons = comparison_group.get("l1b_fixed_and_volume_qois", {})
    authoritative_l1b: dict[str, float] = {}
    if comparisons:
        l1b_path = Path(__file__).resolve().parents[3] / comparison_group["l1b_artifact"]
        l1b_bytes = l1b_path.read_bytes()
        if sha256(l1b_bytes).hexdigest() != comparison_group["l1b_artifact_sha256"]:
            raise FEMValidationError("artifact L1b authority hash differs")
        l1b_payload = json.loads(l1b_bytes)
        authoritative_l1b = {
            key: float(value)
            for key, value in l1b_payload["summary"]["fixed_qois_bz_t"].items()
        }
    for comparison_key, comparison in comparisons.items():
        fem_key = comparison["fem_qoi_key"]
        fem_value = float(artifact["qois_bz_t"][fem_key])
        l1b_value = float(comparison["l1b_structured_grid_bz_t"])
        if (
            comparison_key not in authoritative_l1b
            or authoritative_l1b[comparison_key] != l1b_value
        ):
            raise FEMValidationError("artifact L1b comparison value lacks authority")
        if not np.isclose(
            fem_value,
            comparison["fem_reference_bz_t"],
            rtol=0.0,
            atol=0.0,
        ) or not np.isclose(
            _relative(fem_value, l1b_value),
            comparison["relative_difference"],
            rtol=2.0e-14,
            atol=1.0e-16,
        ):
            raise FEMValidationError("artifact comparison does not replay")

    base = Path(__file__).resolve().parents[3] / evidence["evidence_base_path"]
    levels = []
    previous_file_hash = "0" * 64
    previous_mesh_hash = "0" * 64
    for anchor in evidence["level_evidence"]:
        level, previous_file_hash, previous_mesh_hash = _load_bound_checkpoint(
            anchor,
            base,
            previous_file_hash,
            previous_mesh_hash,
            authority,
            "adaptive",
        )
        levels.append(level)
    convergence = artifact["convergence"]
    if levels:
        if (
            previous_file_hash != authority["final_checkpoint_file_sha256"]
            or levels[-1]["run_sha256"]
            != authority["final_checkpoint_run_sha256"]
            or previous_mesh_hash != authority["final_checkpoint_mesh_sha256"]
        ):
            raise FEMValidationError("final checkpoint identity differs")
        keys = convergence["acceptance_qois"]
        changes = [
            {
                key: _relative(
                    float(left["qois_bz_t"][key]),
                    float(right["qois_bz_t"][key]),
                )
                for key in keys
            }
            for left, right in zip(levels, levels[1:])
        ]
        if changes != convergence["successive_volume_qoi_relative_changes"]:
            raise FEMValidationError("artifact convergence changes do not replay")
        two_successive = len(changes) >= 2 and all(
            value < 0.01 for change in changes[-2:] for value in change.values()
        )
        if two_successive != convergence["two_successive_less_than_one_percent"]:
            raise FEMValidationError("artifact successive-change gate differs")
        orders: dict[str, float | None] = {key: None for key in keys}
        if len(levels) >= 3:
            for key in keys:
                first_delta = abs(
                    float(levels[0]["qois_bz_t"][key])
                    - float(levels[1]["qois_bz_t"][key])
                )
                second_delta = abs(
                    float(levels[1]["qois_bz_t"][key])
                    - float(levels[2]["qois_bz_t"][key])
                )
                h0 = float(levels[0]["resolution"]["qoi_h_m"][key])
                h2 = float(levels[2]["resolution"]["qoi_h_m"][key])
                denominator = log(sqrt(h0 / h2)) if h0 > h2 else 0.0
                orders[key] = (
                    log(first_delta / second_delta) / denominator
                    if first_delta > 0.0
                    and second_delta > 0.0
                    and denominator > 0.0
                    else None
                )
        if orders != convergence["observed_orders_from_actual_qoi_h"]:
            raise FEMValidationError("artifact observed orders do not replay")
        stable_positive = all(
            value is not None and value > 0.0 for value in orders.values()
        )
        if stable_positive != convergence["stable_positive_order"]:
            raise FEMValidationError("artifact positive-order gate differs")
        growth_gate = all(
            float(level["adjacent_area_size_growth"]) <= 1.3 + 1.0e-12
            for level in levels
        )
        if growth_gate != convergence["adjacent_size_growth_gate"]:
            raise FEMValidationError("artifact mesh-growth gate differs")
        domain_inputs = []
        domain_file_hash = "0" * 64
        domain_mesh_hash = "0" * 64
        for anchor in evidence["domain_studies"]:
            domain_level, domain_file_hash, domain_mesh_hash = _load_bound_checkpoint(
                anchor,
                base,
                domain_file_hash,
                domain_mesh_hash,
                authority,
                "domain",
            )
            domain_inputs.append(
                {
                    "padding_factor": anchor["padding_factor"],
                    "qois_bz_t": {
                        key: domain_level["qois_bz_t"][key] for key in keys
                    },
                    "qoi_h_m": {
                        key: domain_level["resolution"]["qoi_h_m"][key]
                        for key in keys
                    },
                    "local_h_m": domain_level["local_h_m"],
                    "domain": domain_level["domain"],
                }
            )
        domain_gate = False
        if domain_inputs:
            domain_gate = bool(
                evaluate_phase_matched_domain_expansion(tuple(domain_inputs))["passed"]
            )
        if domain_gate != convergence["phase_matched_domain_expansion_gate"]:
            raise FEMValidationError("artifact domain-expansion gate differs")
        expected_status = bool(
            two_successive
            and stable_positive
            and growth_gate
            and domain_gate
        )
        if expected_status != convergence["less_than_one_percent_reached"]:
            raise FEMValidationError("artifact acceptance status does not replay")


def validate_artifact(artifact: dict[str, object], *, replay: bool = True) -> None:
    required = {
        "schema_version",
        "model_level",
        "classification",
        "method",
        "anchors",
        "problem",
        "mesh",
        "solution",
        "diagnostics",
        "qois_bz_t",
        "acceptance_evidence",
        "convergence",
        "comparisons",
        "limitations",
        "integrity",
    }
    legacy_required = required - {"acceptance_evidence"}
    if not isinstance(artifact, dict) or (
        set(artifact) != required and set(artifact) != legacy_required
    ):
        raise FEMValidationError("artifact top-level contract differs")
    _guard_artifact_phase("artifact_validation", artifact)
    if artifact["schema_version"] not in {
        SCHEMA_VERSION,
        LEGACY_SCHEMA_VERSION,
        PRIOR_SCHEMA_VERSION,
    } or artifact["classification"] != CLASSIFICATION:
        raise FEMValidationError("artifact identity or claim boundary is invalid")
    if artifact["schema_version"] == SCHEMA_VERSION and set(artifact) != required:
        raise FEMValidationError("authoritative artifact lacks acceptance evidence")
    if artifact["schema_version"] == LEGACY_SCHEMA_VERSION and set(artifact) != legacy_required:
        raise FEMValidationError("legacy artifact layout differs")
    integrity = artifact["integrity"]
    payload = {key: value for key, value in artifact.items() if key != "integrity"}
    expected = sha256(canonical_bytes(payload)).hexdigest()
    if (
        integrity
        != {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": expected,
        }
    ):
        raise FEMValidationError("artifact integrity hash differs")
    numeric_groups = (
        artifact["mesh"]["vertices_rz_m"],
        artifact["mesh"]["p2_nodes_rz_m"],
        artifact["solution"]["a_phi_dofs_t_m"],
        artifact["solution"]["psi_dofs_wb_per_rad"],
    )
    try:
        if any(
            not isfinite(float(value))
            for group in numeric_groups
            for row in group
            for value in (row if isinstance(row, list) else [row])
        ):
            raise FEMValidationError("artifact contains nonfinite numerical state")
    except (TypeError, ValueError, OverflowError) as error:
        raise FEMValidationError("artifact numerical layout is invalid") from error
    if replay:
        replay_report = replay_artifact(artifact)
        if not replay_report["passed"]:
            raise FEMValidationError("artifact deterministic replay differs")
        if artifact["schema_version"] == SCHEMA_VERSION:
            mesh_data = artifact["mesh"]
            rebuilt = P2Mesh(
                np.asarray(mesh_data["vertices_rz_m"], dtype=np.float64),
                np.asarray(mesh_data["triangles"], dtype=np.int64),
                tuple(mesh_data["triangle_region_ids"]),
                np.asarray(mesh_data["p2_nodes_rz_m"], dtype=np.float64),
                np.asarray(mesh_data["element_dofs"], dtype=np.int64),
                np.asarray(mesh_data["edges"], dtype=np.int64),
                np.asarray(mesh_data["edge_midpoint_dofs"], dtype=np.int64),
                {
                    name: np.asarray(values, dtype=np.int64)
                    for name, values in mesh_data["boundary_edges"].items()
                },
                np.asarray(mesh_data["element_parent_ids"], dtype=np.int64),
                np.asarray(mesh_data["interface_edges"], dtype=np.int64),
                tuple(tuple(pair) for pair in mesh_data["interface_region_pairs"]),
                int(mesh_data["refinement_level"]),
                mesh_data["parent_mesh_sha256"],
                tuple(mesh_data["protected_radii_m"]),
                tuple(mesh_data["protected_z_m"]),
            )
            _validate_acceptance_replay(artifact, rebuilt)


def viewer_contract(artifact: dict[str, object]) -> dict[str, object]:
    validate_artifact(artifact)
    payload = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "artifact_payload_sha256": artifact["integrity"]["payload_sha256"],
        "coordinates_rz_m": artifact["mesh"]["vertices_rz_m"],
        "triangles": artifact["mesh"]["triangles"],
        "triangle_region_ids": artifact["mesh"]["triangle_region_ids"],
        "vertex_fields": {
            "psi_wb_per_rad": artifact["solution"]["vertex_psi_wb_per_rad"],
            "b_r_t": artifact["solution"]["vertex_b_r_t"],
            "b_z_t": artifact["solution"]["vertex_b_z_t"],
        },
        "qois_bz_t": artifact["qois_bz_t"],
        "limitations": artifact["limitations"],
    }
    return _seal(payload)


def _stream_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checkpoint_bundle(path: Path, checkpoint: dict[str, object]) -> str:
    """Atomically publish JSON metadata plus a compressed strict-array sidecar."""

    bound = checkpoint.get("bound_artifact")
    if not isinstance(bound, dict):
        raise FEMValidationError("checkpoint bundle requires a bound artifact")
    _guard_artifact_phase("checkpoint_binary_serialization", bound)
    skeleton = copy.deepcopy(checkpoint)
    bound_skeleton = skeleton["bound_artifact"]
    arrays: dict[str, np.ndarray] = {}
    paths = list(_CHECKPOINT_BINARY_PATHS)
    paths.extend(
        ("mesh", "boundary_edges", name)
        for name in sorted(bound["mesh"]["boundary_edges"])
    )
    for parts in paths:
        source = bound
        target = bound_skeleton
        for part in parts[:-1]:
            source = source[part]
            target = target[part]
        key = ".".join(parts)
        descriptor = bound["acceptance_evidence"]["array_contract"].get(key)
        dtype = descriptor["dtype"] if descriptor else "<f8"
        array = np.asarray(source[parts[-1]], dtype=np.dtype(dtype), order="C")
        arrays[key] = array
        target[parts[-1]] = {"$binary_array": key}
    sidecar_path = path.with_name(path.name + ".arrays.npz")
    temporary_sidecar = path.with_name(path.name + ".arrays.tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary_sidecar, **arrays)
    sidecar_hash = _stream_file_sha256(temporary_sidecar)
    skeleton["array_sidecar"] = {
        "format": "numpy-npz-deflate",
        "file": sidecar_path.name,
        "file_sha256": sidecar_hash,
        "uncompressed_array_bytes": sum(array.nbytes for array in arrays.values()),
        "array_keys": sorted(arrays),
    }
    payload = {key: value for key, value in skeleton.items() if key != "integrity"}
    skeleton["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "payload_sha256": sha256(canonical_bytes(payload)).hexdigest(),
    }
    encoded = canonical_bytes(skeleton)
    temporary_json = path.with_name(path.name + ".tmp")
    temporary_json.write_bytes(encoded)
    temporary_sidecar.replace(sidecar_path)
    temporary_json.replace(path)
    return sha256(encoded).hexdigest()


def write_json(path: Path, value: dict[str, object]) -> str:
    bound = value.get("bound_artifact")
    if isinstance(bound, dict):
        _guard_artifact_phase("checkpoint_serialization", bound)
    elif "mesh" in value and "solution" in value:
        _guard_artifact_phase("artifact_publication", value)
    encoded = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return sha256(encoded).hexdigest()
