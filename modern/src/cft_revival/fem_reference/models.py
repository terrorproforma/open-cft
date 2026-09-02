"""Strict data models for the independent axisymmetric P2 FEM reference."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Callable

import numpy as np


class FEMReferenceError(Exception):
    """Base error for the independent numerical reference."""


class FEMValidationError(FEMReferenceError, ValueError):
    """Input, topology, or artifact validation failed."""


class FEMConvergenceError(FEMReferenceError, RuntimeError):
    """The sparse solve failed its true-residual contract."""


ScalarField = Callable[[float, float], float]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode("utf-8")


def content_hash(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class Domain:
    r_min_m: float
    r_max_m: float
    z_min_m: float
    z_max_m: float

    def __post_init__(self) -> None:
        values = (self.r_min_m, self.r_max_m, self.z_min_m, self.z_max_m)
        if any(not isfinite(value) for value in values):
            raise FEMValidationError("domain coordinates must be finite")
        if self.r_min_m < 0.0 or self.r_max_m <= self.r_min_m:
            raise FEMValidationError("domain radial interval is invalid")
        if self.z_max_m <= self.z_min_m:
            raise FEMValidationError("domain axial interval is invalid")

    def to_dict(self) -> dict[str, float]:
        return {
            "r_min_m": self.r_min_m,
            "r_max_m": self.r_max_m,
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
        }


@dataclass(frozen=True, slots=True)
class Region:
    region_id: str
    material_id: str
    reluctivity_per_m_h: float
    remanence_r_t: float = 0.0
    remanence_z_t: float = 0.0

    def __post_init__(self) -> None:
        if not self.region_id or not self.material_id:
            raise FEMValidationError("region and material IDs must be non-empty")
        values = (
            self.reluctivity_per_m_h,
            self.remanence_r_t,
            self.remanence_z_t,
        )
        if any(not isfinite(value) for value in values) or self.reluctivity_per_m_h <= 0.0:
            raise FEMValidationError("region constitutive values are invalid")


@dataclass(frozen=True, slots=True)
class SheetSource:
    source_id: str
    orientation: str
    coordinate_m: float
    span_min_m: float
    span_max_m: float
    k_phi_a_per_m: float

    def __post_init__(self) -> None:
        if not self.source_id or self.orientation not in {"constant_r", "constant_z"}:
            raise FEMValidationError("sheet source identity/orientation is invalid")
        values = (
            self.coordinate_m,
            self.span_min_m,
            self.span_max_m,
            self.k_phi_a_per_m,
        )
        if any(not isfinite(value) for value in values) or self.span_max_m <= self.span_min_m:
            raise FEMValidationError("sheet source geometry is invalid")


@dataclass(frozen=True, slots=True)
class FEMProblem:
    problem_id: str
    domain: Domain
    regions: tuple[Region, ...]
    region_at: Callable[[float, float], str]
    free_current_phi: ScalarField = field(
        default=lambda _r, _z: 0.0, compare=False, repr=False
    )
    sheets: tuple[SheetSource, ...] = ()
    source_center_z_m: float = 0.0
    outer_boundary: str = "dipole_robin"
    dirichlet_a_phi: ScalarField | None = field(default=None, compare=False, repr=False)
    geometry_sha256: str = "0" * 64
    magnetics_sha256: str = "0" * 64
    classification: str = "independent_numerical_reference_not_hardware_validation"
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.problem_id or not self.regions:
            raise FEMValidationError("problem identity and regions are required")
        if len({region.region_id for region in self.regions}) != len(self.regions):
            raise FEMValidationError("region IDs must be unique")
        if self.outer_boundary not in {"dipole_robin", "dirichlet"}:
            raise FEMValidationError("unsupported outer boundary")
        if self.outer_boundary == "dirichlet" and self.dirichlet_a_phi is None:
            raise FEMValidationError("Dirichlet outer boundary requires prescribed A_phi")
        if not isfinite(self.source_center_z_m):
            raise FEMValidationError("source center must be finite")
        for digest in (self.geometry_sha256, self.magnetics_sha256):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise FEMValidationError("provenance hashes must be lowercase SHA-256")

    @property
    def regions_by_id(self) -> dict[str, Region]:
        return {region.region_id: region for region in self.regions}


@dataclass(slots=True)
class P2Mesh:
    vertices_rz_m: np.ndarray
    triangles: np.ndarray
    triangle_region_ids: tuple[str, ...]
    p2_nodes_rz_m: np.ndarray
    element_dofs: np.ndarray
    edges: np.ndarray
    edge_midpoint_dofs: np.ndarray
    boundary_edges: dict[str, np.ndarray]
    element_parent_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    interface_edges: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    interface_region_pairs: tuple[tuple[str, str], ...] = ()
    refinement_level: int = 0
    parent_mesh_sha256: str = "0" * 64
    protected_radii_m: tuple[float, ...] = ()
    protected_z_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        integer_arrays = (
            self.triangles,
            self.element_dofs,
            self.edges,
            self.edge_midpoint_dofs,
        )
        if self.vertices_rz_m.ndim != 2 or self.vertices_rz_m.shape[1] != 2:
            raise FEMValidationError("vertices must have shape (n,2)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise FEMValidationError("triangles must have shape (m,3)")
        if any(not np.issubdtype(values.dtype, np.integer) for values in integer_arrays):
            raise FEMValidationError("mesh topology arrays must use integer indices")
        if not np.isfinite(self.vertices_rz_m).all():
            raise FEMValidationError("vertex coordinates must be finite")
        if len(self.triangle_region_ids) != len(self.triangles):
            raise FEMValidationError("every triangle requires exactly one region tag")
        if any(not isinstance(tag, str) or not tag for tag in self.triangle_region_ids):
            raise FEMValidationError("triangle region tags must be non-empty strings")
        if self.element_dofs.shape != (len(self.triangles), 6):
            raise FEMValidationError("P2 element topology must have six local DOFs")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise FEMValidationError("edge topology must have shape (e,2)")
        if self.edge_midpoint_dofs.shape != (len(self.edges),):
            raise FEMValidationError("every edge requires one midpoint DOF")
        if (
            self.p2_nodes_rz_m.ndim != 2
            or self.p2_nodes_rz_m.shape[1] != 2
            or not np.isfinite(self.p2_nodes_rz_m).all()
        ):
            raise FEMValidationError("mesh coordinates must be finite")
        vertex_count = len(self.vertices_rz_m)
        node_count = len(self.p2_nodes_rz_m)
        if node_count != vertex_count + len(self.edges):
            raise FEMValidationError("P2 node count must equal vertices plus edges")
        if (
            np.any(self.triangles < 0)
            or np.any(self.triangles >= vertex_count)
            or np.any(self.edges < 0)
            or np.any(self.edges >= vertex_count)
            or np.any(self.element_dofs < 0)
            or np.any(self.element_dofs >= node_count)
        ):
            raise FEMValidationError("mesh topology index lies outside its node range")
        if any(len(set(map(int, triangle))) != 3 for triangle in self.triangles):
            raise FEMValidationError("triangle vertices must be distinct")
        edge_tuples = [tuple(map(int, edge)) for edge in self.edges]
        if any(first >= second for first, second in edge_tuples):
            raise FEMValidationError("edges must be stored in increasing vertex order")
        if edge_tuples != sorted(set(edge_tuples)):
            raise FEMValidationError("edges must be globally sorted and unique")
        expected_midpoints = np.arange(
            vertex_count, vertex_count + len(self.edges), dtype=np.int64
        )
        if not np.array_equal(self.edge_midpoint_dofs, expected_midpoints):
            raise FEMValidationError("edge midpoint DOFs must use deterministic contiguous IDs")
        if not np.array_equal(self.p2_nodes_rz_m[:vertex_count], self.vertices_rz_m):
            raise FEMValidationError("P2 vertex coordinates differ from mesh vertices")
        geometric_midpoints = 0.5 * (
            self.vertices_rz_m[self.edges[:, 0]] + self.vertices_rz_m[self.edges[:, 1]]
        )
        if not np.array_equal(
            self.p2_nodes_rz_m[self.edge_midpoint_dofs], geometric_midpoints
        ):
            raise FEMValidationError("P2 midpoint coordinates differ from edge midpoints")
        edge_to_dof = {
            edge: int(dof) for edge, dof in zip(edge_tuples, self.edge_midpoint_dofs)
        }
        for triangle, dofs in zip(self.triangles, self.element_dofs):
            v0, v1, v2 = map(int, triangle)
            expected = (
                v0,
                v1,
                v2,
                edge_to_dof[tuple(sorted((v0, v1)))],
                edge_to_dof[tuple(sorted((v1, v2)))],
                edge_to_dof[tuple(sorted((v2, v0)))],
            )
            if tuple(map(int, dofs)) != expected:
                raise FEMValidationError("element P2 DOFs do not match triangle edges")
        edge_use: dict[tuple[int, int], list[int]] = {edge: [] for edge in edge_tuples}
        for element, triangle in enumerate(self.triangles):
            v0, v1, v2 = map(int, triangle)
            for edge in ((v0, v1), (v1, v2), (v2, v0)):
                edge_use[tuple(sorted(edge))].append(element)
        if any(len(owners) not in (1, 2) for owners in edge_use.values()):
            raise FEMValidationError("mesh edge must have one or two owning elements")
        boundary_indices: list[int] = []
        for name, values in self.boundary_edges.items():
            if not name or values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise FEMValidationError("boundary edge collection is malformed")
            if np.any(values < 0) or np.any(values >= len(self.edges)):
                raise FEMValidationError("boundary edge index is outside edge range")
            boundary_indices.extend(map(int, values))
        if len(boundary_indices) != len(set(boundary_indices)):
            raise FEMValidationError("boundary edge ownership must be unique")
        if any(len(edge_use[edge_tuples[index]]) != 1 for index in boundary_indices):
            raise FEMValidationError("boundary edge must have exactly one owning element")
        expected_boundary = {
            index
            for index, edge in enumerate(edge_tuples)
            if len(edge_use[edge]) == 1
        }
        if set(boundary_indices) != expected_boundary:
            raise FEMValidationError("boundary edge collections must be exhaustive")
        if not self.element_parent_ids.size:
            self.element_parent_ids = np.arange(len(self.triangles), dtype=np.int64)
        if (
            self.element_parent_ids.shape != (len(self.triangles),)
            or not np.issubdtype(self.element_parent_ids.dtype, np.integer)
            or np.any(self.element_parent_ids < 0)
        ):
            raise FEMValidationError("element parent ownership is invalid")
        if self.interface_edges.ndim != 1 or not np.issubdtype(
            self.interface_edges.dtype, np.integer
        ):
            raise FEMValidationError("interface edge collection is malformed")
        if np.any(self.interface_edges < 0) or np.any(self.interface_edges >= len(self.edges)):
            raise FEMValidationError("interface edge index is outside edge range")
        if len(self.interface_region_pairs) != len(self.interface_edges):
            raise FEMValidationError("interface edge ownership pair count differs")
        expected_interfaces = {
            index
            for index, edge in enumerate(edge_tuples)
            if len(edge_use[edge]) == 2
            and self.triangle_region_ids[edge_use[edge][0]]
            != self.triangle_region_ids[edge_use[edge][1]]
        }
        if set(map(int, self.interface_edges)) != expected_interfaces:
            raise FEMValidationError("material interface edge collection must be exhaustive")
        for edge_index, pair in zip(self.interface_edges, self.interface_region_pairs):
            owners = edge_use[edge_tuples[int(edge_index)]]
            if len(owners) != 2:
                raise FEMValidationError("material interface edge must have two owners")
            expected_pair = tuple(
                sorted(self.triangle_region_ids[element] for element in owners)
            )
            if pair != expected_pair or pair[0] == pair[1]:
                raise FEMValidationError("material interface ownership differs from region tags")
        if (
            isinstance(self.refinement_level, bool)
            or not isinstance(self.refinement_level, int)
            or self.refinement_level < 0
        ):
            raise FEMValidationError("refinement level must be a non-negative integer")
        if len(self.parent_mesh_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.parent_mesh_sha256
        ):
            raise FEMValidationError("parent mesh hash must be lowercase SHA-256")
        for collection in (self.protected_radii_m, self.protected_z_m):
            if any(not isfinite(value) for value in collection):
                raise FEMValidationError("protected mesh coordinates must be finite")

    @property
    def sha256(self) -> str:
        payload = {
            "vertices": self.vertices_rz_m.tolist(),
            "triangles": self.triangles.tolist(),
            "triangle_region_ids": list(self.triangle_region_ids),
            "p2_nodes": self.p2_nodes_rz_m.tolist(),
            "element_dofs": self.element_dofs.tolist(),
            "edges": self.edges.tolist(),
            "edge_midpoint_dofs": self.edge_midpoint_dofs.tolist(),
            "boundary_edges": {
                key: value.tolist() for key, value in sorted(self.boundary_edges.items())
            },
            "element_parent_ids": self.element_parent_ids.tolist(),
            "interface_edges": self.interface_edges.tolist(),
            "interface_region_pairs": [list(pair) for pair in self.interface_region_pairs],
            "refinement_level": self.refinement_level,
            "parent_mesh_sha256": self.parent_mesh_sha256,
            "protected_radii_m": list(self.protected_radii_m),
            "protected_z_m": list(self.protected_z_m),
        }
        return content_hash(payload)


@dataclass(frozen=True, slots=True)
class CSRMatrix:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape != (self.shape[1],):
            raise FEMValidationError("CSR vector shape mismatch")
        products = self.data * vector[self.indices]
        return np.add.reduceat(products, self.indptr[:-1])


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    converged: bool
    iterations: int
    initial_residual_l2: float
    final_true_residual_l2: float
    relative_true_residual_l2: float
    residual_history_l2: tuple[float, ...]
    magnetic_action_j: float
    source_action_j: float
    energy_action_relative: float
    assembly_seconds: float
    solve_seconds: float
    peak_working_set_bytes: int
    backend: str = "numpy-csr-ic0-pcg"


@dataclass(frozen=True, slots=True)
class FEMResult:
    problem: FEMProblem
    mesh: P2Mesh
    a_phi_dofs_t_m: np.ndarray
    diagnostics: SolverDiagnostics
    run_sha256: str
    solver_controls: tuple[tuple[str, float | int], ...] = ()
    implementation_sha256: str = "0" * 64
    initial_solution_sha256: str = "0" * 64

    @property
    def psi_dofs_wb_per_rad(self) -> np.ndarray:
        return self.mesh.p2_nodes_rz_m[:, 0] * self.a_phi_dofs_t_m
