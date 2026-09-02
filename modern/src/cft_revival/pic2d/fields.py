"""Prescribed static magnetic field on the PIC node mesh.

The production field is the qualified P2 divergent-exit ``A_phi`` solution,
sampled onto a regular ψ grid by ``BoundP2Evaluator`` and interpolated with the
same axis-regular C1 bicubic (``cft_revival.orbit_mc.fields.PsiBicubicField``)
the orbit campaign uses, then evaluated at every PIC node.  Particles gather
``(B_r, B_z)`` bilinearly from the nodes; the bilinear interpolation error is
second order in the PIC cell size and is measured against the bicubic field by
``tests/pic2d``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..orbit_mc.artifacts import content_hash
from ..orbit_mc.fields import PsiBicubicField
from .models import Grid2D, PIC2DValidationError
from .p2_field import BoundP2Evaluator, file_sha256

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec" / "pic2d"
DEFAULT_AUTHORITY_PATH = SPEC_DIR / "p2-field-authority-v1.json"


@dataclass(frozen=True, slots=True)
class MagneticFieldMap:
    """Node-centred ``(B_r, B_z)`` in tesla with bound provenance."""

    grid: Grid2D
    b_r_t: np.ndarray
    b_z_t: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        shape = self.grid.node_shape
        for name, values in (("b_r_t", self.b_r_t), ("b_z_t", self.b_z_t)):
            if values.shape != shape or values.dtype != np.float64 or not np.isfinite(values).all():
                raise PIC2DValidationError(f"{name} must be a finite float64 node array")
        if np.any(self.b_r_t[0, :] != 0.0):
            raise PIC2DValidationError("B_r must vanish on the axis")

    @property
    def max_b_t(self) -> float:
        return float(np.max(np.hypot(self.b_r_t, self.b_z_t)))

    @property
    def sha256(self) -> str:
        return content_hash(
            {
                "grid": self.grid.to_dict(),
                "b_r_t": self.b_r_t.tolist(),
                "b_z_t": self.b_z_t.tolist(),
                "provenance": dict(self.provenance),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_b_t": self.max_b_t,
            "provenance": dict(self.provenance),
            "field_map_sha256": self.sha256,
        }


def load_authority(path: Path = DEFAULT_AUTHORITY_PATH) -> dict[str, Any]:
    authority = json.loads(path.read_text(encoding="utf-8"))
    if authority.get("schema") != "cft.pic2d.p2-field-authority.v1":
        raise PIC2DValidationError("unsupported P2 field authority schema")
    return authority


def build_p2_psi_field(
    repository_root: Path,
    *,
    role: str = "primary",
    authority: Mapping[str, Any] | None = None,
) -> tuple[PsiBicubicField, dict[str, Any]]:
    """Sample the bound P2 checkpoint onto the declared regular ψ grid.

    Returns the bicubic field and an evidence record (hashes, grid, withheld
    mid-cell error report).  Raises if any declared hash differs.
    """

    authority = dict(load_authority() if authority is None else authority)
    declaration = authority["maps"][role]
    psi_grid = authority["psi_grids"][role]
    bounds = authority["bounding_box"]
    checkpoint_path = Path(repository_root) / declaration["checkpoint_path"]
    evaluator = BoundP2Evaluator(
        checkpoint_path,
        declaration,
        allowed_regions=set(authority["sampling_regions"]),
        bounds=bounds,
    )
    radii = np.linspace(bounds["r_min_m"], bounds["r_max_m"], int(psi_grid["radial_intervals"]) + 1)
    axial = np.linspace(bounds["z_min_m"], bounds["z_max_m"], int(psi_grid["axial_intervals"]) + 1)
    psi = np.empty((len(radii), len(axial)), dtype=np.float64)
    reference_br = np.empty_like(psi)
    reference_bz = np.empty_like(psi)
    for i, radius in enumerate(radii):
        for j, z_value in enumerate(axial):
            psi[i, j], reference_br[i, j], reference_bz[i, j] = evaluator.evaluate(float(radius), float(z_value))
    material = np.full(psi.shape, authority["plasma_material_id"], dtype=object)
    source_identity = content_hash(
        {
            "role": role,
            "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
            "sidecar_file_sha256": declaration["sidecar_file_sha256"],
            "mesh_sha256": declaration["mesh_sha256"],
            "run_sha256": declaration["run_sha256"],
            "bounding_box": dict(bounds),
            "psi_grid": dict(psi_grid),
        }
    )
    field = PsiBicubicField(
        radii,
        axial,
        psi,
        material_id=material,
        plasma_material_id=authority["plasma_material_id"],
        source_identity_sha256=source_identity,
    )
    stride = int(authority["withheld_midcell_stride"])
    br_errors: list[float] = []
    bz_errors: list[float] = []
    psi_errors: list[float] = []
    for i in range(0, len(radii) - 1, stride):
        radius = 0.5 * (radii[i] + radii[i + 1])
        for j in range(0, len(axial) - 1, stride):
            z_value = 0.5 * (axial[j] + axial[j + 1])
            reference = evaluator.evaluate(float(radius), float(z_value))
            interpolated_psi, _, _ = field.psi_gradient(float(radius), float(z_value))
            interpolated_br, interpolated_bz = field.field_cylindrical(float(radius), float(z_value))
            psi_errors.append(interpolated_psi - reference[0])
            br_errors.append(interpolated_br - reference[1])
            bz_errors.append(interpolated_bz - reference[2])
    squared = np.square(br_errors) + np.square(bz_errors)
    report = {
        "sample_count": len(squared),
        "psi_midcell_max_abs_wb": float(max(map(abs, psi_errors))),
        "br_max_abs_t": float(max(map(abs, br_errors))),
        "bz_max_abs_t": float(max(map(abs, bz_errors))),
        "b_rms_t": float(np.sqrt(np.mean(squared))),
        "b_relative_rms": float(np.sqrt(np.mean(squared)) / max(field.max_b_t, np.finfo(float).tiny)),
    }
    passed = bool(
        report["b_relative_rms"] <= authority["maximum_b_relative_rms"]
        and max(report["br_max_abs_t"], report["bz_max_abs_t"]) <= authority["maximum_b_component_absolute_error_t"]
    )
    if not passed:
        raise PIC2DValidationError(f"P2 ψ-grid withheld mid-cell error gate failed: {report}")
    node_reference_error = float(
        np.max(
            np.hypot(
                np.array([[field.field_cylindrical(float(r), float(z))[0] for z in axial] for r in radii]) - reference_br,
                np.array([[field.field_cylindrical(float(r), float(z))[1] for z in axial] for r in radii]) - reference_bz,
            )
        )
    )
    evidence = {
        "role": role,
        "design_id": authority["design_id"],
        "checkpoint_path": declaration["checkpoint_path"],
        "checkpoint_file_sha256": declaration["checkpoint_file_sha256"],
        "checkpoint_payload_sha256": declaration["checkpoint_payload_sha256"],
        "checkpoint_sidecar_sha256": declaration["sidecar_file_sha256"],
        "mesh_sha256": declaration["mesh_sha256"],
        "run_sha256": declaration["run_sha256"],
        "authority_file_sha256": file_sha256(DEFAULT_AUTHORITY_PATH),
        "source_identity_sha256": source_identity,
        "p2_classification": evaluator.classification,
        "psi_grid": {
            "radial_samples": len(radii),
            "axial_samples": len(axial),
            "r_min_m": float(radii[0]),
            "r_max_m": float(radii[-1]),
            "z_min_m": float(axial[0]),
            "z_max_m": float(axial[-1]),
        },
        "withheld_midcell_error": report,
        "node_reference_b_max_abs_error_t": node_reference_error,
        "certificate": field.certificate_tightness.to_dict(),
        "certified_max_b_t": field.certified_max_b_t,
        "material_map_sha256": field.material_map_sha256,
        "sampling_regions": list(authority["sampling_regions"]),
        "sampling_region_justification": authority["sampling_region_justification"],
    }
    return field, evidence


def sample_field_map(field: PsiBicubicField, grid: Grid2D, provenance: Mapping[str, Any]) -> MagneticFieldMap:
    """Evaluate the bicubic field at every PIC node."""

    r = grid.r_m
    z = grid.z_m
    b_r = np.empty(grid.node_shape, dtype=np.float64)
    b_z = np.empty(grid.node_shape, dtype=np.float64)
    for i, radius in enumerate(r):
        for j, axial in enumerate(z):
            b_r[i, j], b_z[i, j] = field.field_cylindrical(float(radius), float(axial))
    b_r[0, :] = 0.0
    return MagneticFieldMap(grid, b_r, b_z, dict(provenance) | {"kind": "p2-psi-bicubic-node-sample"})


def p2_field_map(repository_root: Path, grid: Grid2D, *, role: str = "primary") -> tuple[MagneticFieldMap, PsiBicubicField]:
    field, evidence = build_p2_psi_field(repository_root, role=role)
    return sample_field_map(field, grid, evidence), field


def uniform_field_map(grid: Grid2D, b_z_t: float) -> MagneticFieldMap:
    b_r = np.zeros(grid.node_shape, dtype=np.float64)
    b_z = np.full(grid.node_shape, float(b_z_t), dtype=np.float64)
    return MagneticFieldMap(grid, b_r, b_z, {"kind": "analytic-uniform", "b_z_t": float(b_z_t)})


def linear_psi_field_map(grid: Grid2D, coefficient_t_per_m: float) -> MagneticFieldMap:
    """ψ = a r² z  ->  B_r = -a r, B_z = 2 a z; divergence-free and exactly bilinear."""

    a = float(coefficient_t_per_m)
    rr, zz = np.meshgrid(grid.r_m, grid.z_m, indexing="ij")
    return MagneticFieldMap(grid, -a * rr, 2.0 * a * zz, {"kind": "analytic-linear-psi", "coefficient_t_per_m": a})


def zero_field_map(grid: Grid2D) -> MagneticFieldMap:
    zeros = np.zeros(grid.node_shape, dtype=np.float64)
    return MagneticFieldMap(grid, zeros, zeros.copy(), {"kind": "analytic-zero"})


__all__ = [
    "DEFAULT_AUTHORITY_PATH",
    "MagneticFieldMap",
    "build_p2_psi_field",
    "linear_psi_field_map",
    "load_authority",
    "p2_field_map",
    "sample_field_map",
    "uniform_field_map",
    "zero_field_map",
]
