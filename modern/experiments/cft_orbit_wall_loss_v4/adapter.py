"""Experiment-local, hash-bound P2 FEM to regular ψ-grid adapter (v4; unchanged from v3)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cft_revival.orbit_mc import PsiBicubicField
from cft_revival.orbit_mc.artifacts import content_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BoundP2Evaluator:
    """Evaluate a bound quadratic A_phi checkpoint inside the plasma bore."""

    def __init__(
        self,
        checkpoint_path: Path,
        declaration: Mapping[str, Any],
        *,
        allowed_regions: set[str],
        bounds: Mapping[str, float],
    ) -> None:
        if file_sha256(checkpoint_path) != declaration["checkpoint_file_sha256"]:
            raise ValueError("P2 checkpoint file authority differs")
        metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            metadata["integrity"]["payload_sha256"]
            != declaration["checkpoint_payload_sha256"]
            or metadata["mesh_sha256"] != declaration["mesh_sha256"]
            or metadata["run_sha256"] != declaration["run_sha256"]
        ):
            raise ValueError("P2 checkpoint payload/mesh/run authority differs")
        sidecar = checkpoint_path.parent / metadata["array_sidecar"]["file"]
        if (
            metadata["array_sidecar"]["file_sha256"]
            != declaration["sidecar_file_sha256"]
            or file_sha256(sidecar) != declaration["sidecar_file_sha256"]
        ):
            raise ValueError("P2 checkpoint array sidecar authority differs")
        with np.load(sidecar, allow_pickle=False) as archive:
            self.vertices = np.asarray(
                archive["mesh.vertices_rz_m"], dtype=np.float64
            )
            self.triangles = np.asarray(archive["mesh.triangles"], dtype=np.int64)
            self.element_dofs = np.asarray(
                archive["mesh.element_dofs"], dtype=np.int64
            )
            self.a_phi = np.asarray(
                archive["solution.a_phi_dofs_t_m"], dtype=np.float64
            )
        region_ids = np.asarray(
            metadata["bound_artifact"]["mesh"]["triangle_region_ids"], dtype=object
        )
        if len(region_ids) != len(self.triangles):
            raise ValueError("P2 triangle material authority differs")
        points = self.vertices[self.triangles]
        r_min = np.min(points[:, :, 0], axis=1)
        r_max = np.max(points[:, :, 0], axis=1)
        z_min = np.min(points[:, :, 1], axis=1)
        z_max = np.max(points[:, :, 1], axis=1)
        selected = np.flatnonzero(
            (r_max >= bounds["r_min_m"])
            & (r_min <= bounds["r_max_m"])
            & (z_max >= bounds["z_min_m"])
            & (z_min <= bounds["z_max_m"])
        )
        self.global_elements = selected
        self.region_ids = region_ids[selected]
        self.allowed_regions = allowed_regions
        self.p0 = points[selected, 0]
        first = points[selected, 1] - self.p0
        second = points[selected, 2] - self.p0
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        if np.any(np.abs(determinant) <= np.finfo(float).tiny):
            raise ValueError("degenerate P2 triangle in plasma search region")
        self.inverse = np.empty((len(selected), 2, 2), dtype=np.float64)
        self.inverse[:, 0, 0] = second[:, 1] / determinant
        self.inverse[:, 0, 1] = -second[:, 0] / determinant
        self.inverse[:, 1, 0] = -first[:, 1] / determinant
        self.inverse[:, 1, 1] = first[:, 0] / determinant
        self.element_r_min = r_min[selected]
        self.element_r_max = r_max[selected]
        self.element_z_min = z_min[selected]
        self.element_z_max = z_max[selected]
        self.bounds = dict(bounds)
        self.bucket_r = 64
        self.bucket_z = 256
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for local in range(len(selected)):
            i0, j0 = self._bucket(
                float(self.element_r_min[local]), float(self.element_z_min[local])
            )
            i1, j1 = self._bucket(
                float(self.element_r_max[local]), float(self.element_z_max[local])
            )
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    self.buckets.setdefault((i, j), []).append(local)

    def _bucket(self, radius: float, axial: float) -> tuple[int, int]:
        r_span = self.bounds["r_max_m"] - self.bounds["r_min_m"]
        z_span = self.bounds["z_max_m"] - self.bounds["z_min_m"]
        i = int(
            (radius - self.bounds["r_min_m"]) / r_span * self.bucket_r
        )
        j = int(
            (axial - self.bounds["z_min_m"]) / z_span * self.bucket_z
        )
        return (
            max(0, min(self.bucket_r - 1, i)),
            max(0, min(self.bucket_z - 1, j)),
        )

    def _matches(
        self, radius: float, axial: float
    ) -> list[tuple[int, np.ndarray]]:
        tolerance = 2.0e-11
        matches: list[tuple[int, np.ndarray]] = []
        for local in self.buckets.get(self._bucket(radius, axial), ()):
            if (
                radius < self.element_r_min[local] - tolerance
                or radius > self.element_r_max[local] + tolerance
                or axial < self.element_z_min[local] - tolerance
                or axial > self.element_z_max[local] + tolerance
            ):
                continue
            coordinate = self.inverse[local] @ (
                np.asarray((radius, axial)) - self.p0[local]
            )
            barycentric = np.asarray(
                (1.0 - coordinate[0] - coordinate[1], coordinate[0], coordinate[1])
            )
            if float(np.min(barycentric)) >= -tolerance:
                matches.append((local, barycentric))
        if not matches:
            raise ValueError(f"P2 point ({radius}, {axial}) is outside the bound mesh")
        plasma = [
            item for item in matches if self.region_ids[item[0]] in self.allowed_regions
        ]
        if not plasma:
            observed = sorted({str(self.region_ids[item[0]]) for item in matches})
            raise ValueError(f"P2 point entered quarantined material regions: {observed}")
        return plasma

    def _evaluate_element(
        self, local: int, barycentric: np.ndarray, radius: float
    ) -> tuple[float, float, float]:
        l0, l1, l2 = map(float, barycentric)
        values = np.asarray(
            (
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            )
        )
        inverse_transpose = self.inverse[local].T
        gradients_lambda = np.empty((3, 2), dtype=np.float64)
        gradients_lambda[1] = inverse_transpose[:, 0]
        gradients_lambda[2] = inverse_transpose[:, 1]
        gradients_lambda[0] = -gradients_lambda[1] - gradients_lambda[2]
        gradients = np.vstack(
            (
                (4.0 * l0 - 1.0) * gradients_lambda[0],
                (4.0 * l1 - 1.0) * gradients_lambda[1],
                (4.0 * l2 - 1.0) * gradients_lambda[2],
                4.0 * (l0 * gradients_lambda[1] + l1 * gradients_lambda[0]),
                4.0 * (l1 * gradients_lambda[2] + l2 * gradients_lambda[1]),
                4.0 * (l2 * gradients_lambda[0] + l0 * gradients_lambda[2]),
            )
        )
        coefficients = self.a_phi[
            self.element_dofs[self.global_elements[local]]
        ]
        a_phi = float(values @ coefficients)
        gradient = coefficients @ gradients
        if radius == 0.0:
            return 0.0, 0.0, 2.0 * float(gradient[0])
        return (
            radius * a_phi,
            -float(gradient[1]),
            a_phi / radius + float(gradient[0]),
        )

    def evaluate(self, radius: float, axial: float) -> tuple[float, float, float]:
        query_r = radius
        query_z = axial
        if query_r == self.bounds["r_max_m"]:
            query_r = math.nextafter(query_r, self.bounds["r_min_m"])
        if query_z == self.bounds["z_min_m"]:
            query_z = math.nextafter(query_z, self.bounds["z_max_m"])
        elif query_z == self.bounds["z_max_m"]:
            query_z = math.nextafter(query_z, self.bounds["z_min_m"])
        matches = self._matches(query_r, query_z)
        values = [
            self._evaluate_element(local, barycentric, query_r)
            for local, barycentric in matches
        ]
        return tuple(float(np.mean([item[index] for item in values])) for index in range(3))


def build_regular_field(
    repository_root: Path,
    protocol: Mapping[str, Any],
    role: str,
) -> tuple[PsiBicubicField, dict[str, Any], dict[str, Any]]:
    adapter = protocol["field_adapter"]
    declaration = adapter["maps"][role]
    bounds = adapter["regular_plasma_domain"]
    checkpoint_path = repository_root / declaration["checkpoint_path"]
    evaluator = BoundP2Evaluator(
        checkpoint_path,
        declaration,
        allowed_regions=set(adapter["plasma_region_ids"]),
        bounds=bounds,
    )
    radii = np.linspace(
        bounds["r_min_m"], bounds["r_max_m"], declaration["radial_intervals"] + 1
    )
    axial = np.linspace(
        bounds["z_min_m"], bounds["z_max_m"], declaration["axial_intervals"] + 1
    )
    psi = np.empty((len(radii), len(axial)), dtype=np.float64)
    for i, radius in enumerate(radii):
        for j, z_value in enumerate(axial):
            psi[i, j] = evaluator.evaluate(float(radius), float(z_value))[0]
    material = np.full(psi.shape, adapter["plasma_material_id"], dtype=object)
    source_identity = content_hash(
        {
            "role": role,
            "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
            "sidecar_file_sha256": declaration["sidecar_file_sha256"],
            "mesh_sha256": declaration["mesh_sha256"],
            "run_sha256": declaration["run_sha256"],
        }
    )
    field = PsiBicubicField(
        radii,
        axial,
        psi,
        material_id=material,
        plasma_material_id=adapter["plasma_material_id"],
        minimum_certificate_tightness_ratio=protocol["gates"][
            "minimum_certificate_dense_to_bound_ratio"
        ],
        source_identity_sha256=source_identity,
    )
    psi_errors: list[float] = []
    br_errors: list[float] = []
    bz_errors: list[float] = []
    stride = int(adapter["withheld_midcell_stride"])
    for i in range(0, len(radii) - 1, stride):
        radius = 0.5 * (radii[i] + radii[i + 1])
        for j in range(0, len(axial) - 1, stride):
            z_value = 0.5 * (axial[j] + axial[j + 1])
            reference = evaluator.evaluate(float(radius), float(z_value))
            interpolated_psi, _, _ = field.psi_gradient(float(radius), float(z_value))
            interpolated_br, interpolated_bz = field.field_cylindrical(
                float(radius), float(z_value)
            )
            psi_errors.append(interpolated_psi - reference[0])
            br_errors.append(interpolated_br - reference[1])
            bz_errors.append(interpolated_bz - reference[2])
    squared = np.square(br_errors) + np.square(bz_errors)
    report = {
        "sample_count": len(squared),
        "psi_node_max_abs_wb": max(map(abs, psi_errors)),
        "br_max_abs_t": max(map(abs, br_errors)),
        "bz_max_abs_t": max(map(abs, bz_errors)),
        "b_rms_t": float(np.sqrt(np.mean(squared))),
        "b_relative_rms": float(
            np.sqrt(np.mean(squared)) / max(field.max_b_t, np.finfo(float).tiny)
        ),
    }
    passed = bool(
        report["b_relative_rms"] <= adapter["maximum_b_relative_rms"]
        and max(report["br_max_abs_t"], report["bz_max_abs_t"])
        <= adapter["maximum_b_component_absolute_error_t"]
    )
    evidence = {
        "role": role,
        "source_identity_sha256": source_identity,
        "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
        "checkpoint_payload_sha256": declaration["checkpoint_payload_sha256"],
        "checkpoint_sidecar_sha256": declaration["sidecar_file_sha256"],
        "mesh_sha256": declaration["mesh_sha256"],
        "run_sha256": declaration["run_sha256"],
        "regular_grid": {
            "radial_samples": len(radii),
            "axial_samples": len(axial),
            "r_min_m": float(radii[0]),
            "r_max_m": float(radii[-1]),
            "z_min_m": float(axial[0]),
            "z_max_m": float(axial[-1]),
        },
        "field_error_report": report,
        "certificate": field.certificate_tightness.to_dict(),
        "material_map_sha256": field.material_map_sha256,
        "passed": passed,
    }
    serialized = {
        "r_m": radii.tolist(),
        "z_m": axial.tolist(),
        "psi_wb": psi.tolist(),
        "material_id": material.tolist(),
        "source_identity_sha256": source_identity,
    }
    return field, evidence, serialized
