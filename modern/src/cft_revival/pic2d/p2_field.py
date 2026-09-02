"""Hash-bound P2 FEM checkpoint to regular ψ-grid adapter.

This is the same quadratic ``A_phi`` element evaluation and regular-grid
sampling that the full-orbit wall-loss campaign uses
(``experiments/cft_orbit_wall_loss_v3/adapter.py`` on branch
``exp/cft-orbit-wall-loss-v3``), carried into the package so the PIC binds the
identical qualified P2 evidence.  Differences from the campaign adapter are
documented in ``fields.py``: the PIC samples the whole channel bounding box
(including the μ0 dielectric and injector regions) instead of the conservative
``r <= 2 mm, 1 <= z <= 23 mm`` plasma subdomain.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import PIC2DValidationError


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BoundP2Evaluator:
    """Evaluate a bound quadratic A_phi checkpoint inside declared regions.

    Returns ``(psi, B_r, B_z)`` with ``psi = r * A_phi`` (Wb/rad),
    ``B_r = -dA_phi/dz`` and ``B_z = A_phi / r + dA_phi/dr``; on the axis
    ``B_z = 2 dA_phi/dr`` by regularity.  Every declared hash must match.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        declaration: Mapping[str, Any],
        *,
        allowed_regions: set[str],
        bounds: Mapping[str, float],
    ) -> None:
        if file_sha256(checkpoint_path) != declaration["checkpoint_file_sha256"]:
            raise PIC2DValidationError("P2 checkpoint file authority differs")
        metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            metadata["integrity"]["payload_sha256"] != declaration["checkpoint_payload_sha256"]
            or metadata["mesh_sha256"] != declaration["mesh_sha256"]
            or metadata["run_sha256"] != declaration["run_sha256"]
        ):
            raise PIC2DValidationError("P2 checkpoint payload/mesh/run authority differs")
        sidecar = checkpoint_path.parent / metadata["array_sidecar"]["file"]
        if (
            metadata["array_sidecar"]["file_sha256"] != declaration["sidecar_file_sha256"]
            or file_sha256(sidecar) != declaration["sidecar_file_sha256"]
        ):
            raise PIC2DValidationError("P2 checkpoint array sidecar authority differs")
        with np.load(sidecar, allow_pickle=False) as archive:
            self.vertices = np.asarray(archive["mesh.vertices_rz_m"], dtype=np.float64)
            self.triangles = np.asarray(archive["mesh.triangles"], dtype=np.int64)
            self.element_dofs = np.asarray(archive["mesh.element_dofs"], dtype=np.int64)
            self.a_phi = np.asarray(archive["solution.a_phi_dofs_t_m"], dtype=np.float64)
        region_ids = np.asarray(metadata["bound_artifact"]["mesh"]["triangle_region_ids"], dtype=object)
        if len(region_ids) != len(self.triangles):
            raise PIC2DValidationError("P2 triangle material authority differs")
        self.qualification_status = str(
            metadata.get("bound_artifact", {}).get("acceptance_evidence", {}).get("authority", "")
        )
        self.classification = str(metadata.get("classification", ""))
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
        self.allowed_regions = set(allowed_regions)
        self.p0 = points[selected, 0]
        first = points[selected, 1] - self.p0
        second = points[selected, 2] - self.p0
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        if np.any(np.abs(determinant) <= np.finfo(float).tiny):
            raise PIC2DValidationError("degenerate P2 triangle in search region")
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
            i0, j0 = self._bucket(float(self.element_r_min[local]), float(self.element_z_min[local]))
            i1, j1 = self._bucket(float(self.element_r_max[local]), float(self.element_z_max[local]))
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    self.buckets.setdefault((i, j), []).append(local)

    def _bucket(self, radius: float, axial: float) -> tuple[int, int]:
        r_span = self.bounds["r_max_m"] - self.bounds["r_min_m"]
        z_span = self.bounds["z_max_m"] - self.bounds["z_min_m"]
        i = int((radius - self.bounds["r_min_m"]) / r_span * self.bucket_r)
        j = int((axial - self.bounds["z_min_m"]) / z_span * self.bucket_z)
        return (max(0, min(self.bucket_r - 1, i)), max(0, min(self.bucket_z - 1, j)))

    def _matches(self, radius: float, axial: float) -> list[tuple[int, np.ndarray]]:
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
            coordinate = self.inverse[local] @ (np.asarray((radius, axial)) - self.p0[local])
            barycentric = np.asarray((1.0 - coordinate[0] - coordinate[1], coordinate[0], coordinate[1]))
            if float(np.min(barycentric)) >= -tolerance:
                matches.append((local, barycentric))
        if not matches:
            raise PIC2DValidationError(f"P2 point ({radius}, {axial}) is outside the bound mesh")
        allowed = [item for item in matches if self.region_ids[item[0]] in self.allowed_regions]
        if not allowed:
            observed = sorted({str(self.region_ids[item[0]]) for item in matches})
            raise PIC2DValidationError(f"P2 point entered disallowed material regions: {observed}")
        return allowed

    def _evaluate_element(self, local: int, barycentric: np.ndarray, radius: float) -> tuple[float, float, float]:
        l0, l1, l2 = map(float, barycentric)
        values = np.asarray(
            (
                l0 * (2.0 * l0 - 1.0), l1 * (2.0 * l1 - 1.0), l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1, 4.0 * l1 * l2, 4.0 * l2 * l0,
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
        coefficients = self.a_phi[self.element_dofs[self.global_elements[local]]]
        a_phi = float(values @ coefficients)
        gradient = coefficients @ gradients
        if radius == 0.0:
            return 0.0, 0.0, 2.0 * float(gradient[0])
        return radius * a_phi, -float(gradient[1]), a_phi / radius + float(gradient[0])

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
        values = [self._evaluate_element(local, barycentric, query_r) for local, barycentric in matches]
        return tuple(float(np.mean([item[index] for item in values])) for index in range(3))


__all__ = ["BoundP2Evaluator", "file_sha256"]
