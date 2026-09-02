"""C1 bicubic interpolation of axisymmetric flux with consistent B derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import hypot, isfinite
from numbers import Real
from typing import Callable, Mapping, Sequence

import numpy as np

from .models import OrbitNumericsError, OrbitValidationError

_HERMITE_TO_POWER = np.array(
    [
        [1.0, 0.0, -3.0, 2.0],
        [0.0, 0.0, 3.0, -2.0],
        [0.0, 1.0, -2.0, 1.0],
        [0.0, 0.0, -1.0, 1.0],
    ]
)


def _coordinates(name: str, values: Sequence[float]) -> np.ndarray:
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise OrbitValidationError(f"{name} must not contain booleans")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 4 or not np.isfinite(array).all():
        raise OrbitValidationError(f"{name} must contain at least four finite coordinates")
    if np.any(np.diff(array) <= 0.0):
        raise OrbitValidationError(f"{name} must be strictly increasing")
    return array


def _basis(value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = value
    v = np.array(
        [2*t**3 - 3*t**2 + 1, -2*t**3 + 3*t**2, t**3 - 2*t**2 + t, t**3 - t**2]
    )
    d = np.array(
        [6*t**2 - 6*t, -6*t**2 + 6*t, 3*t**2 - 4*t + 1, 3*t**2 - 2*t]
    )
    d2 = np.array([12*t - 6, -12*t + 6, 6*t - 4, 6*t - 2])
    return v, d, d2


@dataclass(frozen=True, slots=True)
class InterpolationErrorReport:
    sample_count: int
    psi_node_max_abs_wb: float
    br_max_abs_t: float
    bz_max_abs_t: float
    b_rms_t: float
    b_relative_rms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_count": self.sample_count,
            "psi_node_max_abs_wb": self.psi_node_max_abs_wb,
            "br_max_abs_t": self.br_max_abs_t,
            "bz_max_abs_t": self.bz_max_abs_t,
            "b_rms_t": self.b_rms_t,
            "b_relative_rms": self.b_relative_rms,
        }


@dataclass(frozen=True, slots=True)
class CertificateTightness:
    dense_diagnostic_max_b_t: float
    certified_max_b_t: float
    dense_to_bound_ratio: float
    minimum_ratio: float
    preflight_passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "dense_diagnostic_max_b_t": self.dense_diagnostic_max_b_t,
            "certified_max_b_t": self.certified_max_b_t,
            "dense_to_bound_ratio": self.dense_to_bound_ratio,
            "minimum_ratio": self.minimum_ratio,
            "preflight_passed": self.preflight_passed,
        }


class PsiBicubicField:
    """Axis-regular C1 interpolation of g=(ψ-ψ_axis)/r² in plasma cells."""

    def __init__(
        self,
        r_m: Sequence[float],
        z_m: Sequence[float],
        psi_wb: Sequence[Sequence[float]],
        *,
        material_id: Sequence[Sequence[str]],
        plasma_material_id: str,
        reference_br_t: Sequence[Sequence[float]] | None = None,
        reference_bz_t: Sequence[Sequence[float]] | None = None,
        reference_max_b_t: float | None = None,
        reference_consistency_relative_tolerance: float = 0.05,
        minimum_certificate_tightness_ratio: float = 1.0e-3,
        source_identity_sha256: str = "0" * 64,
    ) -> None:
        self.r_m = _coordinates("r_m", r_m)
        self.z_m = _coordinates("z_m", z_m)
        if self.r_m[0] != 0.0:
            raise OrbitValidationError("axisymmetric interpolation requires r_m[0] == 0")
        self.psi_wb = np.asarray(psi_wb, dtype=np.float64)
        shape = (len(self.r_m), len(self.z_m))
        if self.psi_wb.shape != shape or not np.isfinite(self.psi_wb).all():
            raise OrbitValidationError("psi_wb shape/finite contract failed")
        axis_scale = max(1.0, float(np.max(np.abs(self.psi_wb))))
        if float(np.ptp(self.psi_wb[0])) > 64.0 * np.finfo(float).eps * axis_scale:
            raise OrbitValidationError("ψ must be constant along the symmetry axis")
        if not isinstance(plasma_material_id, str) or not plasma_material_id:
            raise OrbitValidationError("plasma_material_id must be non-empty")
        self.plasma_material_id = plasma_material_id
        self.material_id = np.asarray(material_id, dtype=object)
        if self.material_id.shape != shape or any(
            not isinstance(value, str) or not value
            for value in self.material_id.flat
        ):
            raise OrbitValidationError("material_id must be a non-empty string map")
        self.traversable_cells = np.zeros((shape[0] - 1, shape[1] - 1), dtype=bool)
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                corners = self.material_id[i : i + 2, j : j + 2]
                self.traversable_cells[i, j] = bool(
                    np.all(corners == self.plasma_material_id)
                )
        if not np.any(self.traversable_cells):
            raise OrbitValidationError("material map has no homogeneous plasma cell")
        self.material_map_sha256 = sha256(
            json.dumps(
                self.material_id.tolist(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        axis_psi = self.psi_wb[0].copy()
        self.g_wb_per_m2 = np.empty(shape, dtype=np.float64)
        radii2 = self.r_m[1:] ** 2
        self.g_wb_per_m2[1:, :] = (
            self.psi_wb[1:, :] - axis_psi[np.newaxis, :]
        ) / radii2[:, np.newaxis]
        # Even-axis extrapolation in x=r². It exactly preserves ψ=a r²+b r⁴.
        for j in range(shape[1]):
            coefficients = np.polyfit(radii2[:3], self.g_wb_per_m2[1:4, j], 2)
            self.g_wb_per_m2[0, j] = float(np.polyval(coefficients, 0.0))
        self.dg_dr = np.gradient(self.g_wb_per_m2, self.r_m, axis=0, edge_order=2)
        self.dg_dz = np.gradient(self.g_wb_per_m2, self.z_m, axis=1, edge_order=2)
        self.dg_drdz = np.gradient(self.dg_dr, self.z_m, axis=1, edge_order=2)
        self.dg_dr[0, :] = 0.0
        self.dg_drdz[0, :] = 0.0
        self.reference_br_t = self._optional_map(reference_br_t, shape, "reference_br_t")
        self.reference_bz_t = self._optional_map(reference_bz_t, shape, "reference_bz_t")
        if (self.reference_br_t is None) != (self.reference_bz_t is None):
            raise OrbitValidationError("reference Br and Bz maps must be supplied together")
        if (
            not isinstance(source_identity_sha256, str)
            or len(source_identity_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_identity_sha256)
        ):
            raise OrbitValidationError("source_identity_sha256 must be lowercase SHA-256")
        self.source_identity_sha256 = source_identity_sha256
        self.certified_max_b_t = self._certified_field_bound()
        if not isfinite(self.certified_max_b_t) or self.certified_max_b_t <= 0.0:
            raise OrbitValidationError(
                "certified field maximum must be finite and positive"
            )
        if (
            isinstance(minimum_certificate_tightness_ratio, bool)
            or not isinstance(minimum_certificate_tightness_ratio, Real)
        ):
            raise OrbitValidationError(
                "minimum certificate tightness ratio must be a real scalar"
            )
        minimum_tightness = float(minimum_certificate_tightness_ratio)
        if not isfinite(minimum_tightness) or not 0.0 <= minimum_tightness <= 1.0:
            raise OrbitValidationError(
                "minimum certificate tightness ratio must lie in [0,1]"
            )
        self.certificate_tightness = self._certificate_tightness(
            minimum_tightness
        )
        if not self.certificate_tightness.preflight_passed:
            raise OrbitValidationError(
                "certificate preflight NOT_EVALUATED: dense/bound ratio "
                f"{self.certificate_tightness.dense_to_bound_ratio:.6g} is below "
                f"{minimum_tightness:.6g}; refine the certificate before orbit work"
            )
        if (
            isinstance(reference_consistency_relative_tolerance, bool)
            or not isinstance(reference_consistency_relative_tolerance, Real)
        ):
            raise OrbitValidationError(
                "reference consistency tolerance must not be boolean"
            )
        try:
            tolerance = float(reference_consistency_relative_tolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise OrbitValidationError(
                "reference consistency tolerance must be a real scalar"
            ) from error
        if not isfinite(tolerance) or tolerance < 0.0:
            raise OrbitValidationError(
                "reference consistency tolerance must be finite and nonnegative"
            )
        self.reference_consistency_relative_tolerance = tolerance
        observed_reference = 0.0
        if self.reference_br_t is not None:
            observed_reference = float(
                np.max(np.hypot(self.reference_br_t, self.reference_bz_t))
            )
            maximum_error = 0.0
            for i, radius in enumerate(self.r_m):
                for j, axial in enumerate(self.z_m):
                    cell_i = min(i, len(self.r_m) - 2)
                    cell_j = min(j, len(self.z_m) - 2)
                    if not self.traversable_cells[cell_i, cell_j]:
                        continue
                    br, bz = self.field_cylindrical(float(radius), float(axial))
                    maximum_error = max(
                        maximum_error,
                        hypot(
                            br - float(self.reference_br_t[i, j]),
                            bz - float(self.reference_bz_t[i, j]),
                        ),
                    )
            scale = max(observed_reference, self.certified_max_b_t, np.finfo(float).tiny)
            if maximum_error > tolerance * scale:
                raise OrbitValidationError(
                    "reference Br/Bz map is inconsistent with ψ-derived field"
                )
        if reference_max_b_t is not None:
            if isinstance(reference_max_b_t, bool) or not isinstance(
                reference_max_b_t, Real
            ):
                raise OrbitValidationError("reference_max_b_t must not be boolean")
            try:
                declared = float(reference_max_b_t)
            except (TypeError, ValueError, OverflowError) as error:
                raise OrbitValidationError(
                    "reference_max_b_t must be a real scalar"
                ) from error
            if (
                not isfinite(declared)
                or declared <= 0.0
                or declared + 32.0 * np.spacing(declared) < observed_reference
            ):
                raise OrbitValidationError(
                    "reference_max_b_t is nonfinite, nonpositive, or underdeclared"
                )
        else:
            declared = 0.0
        self.reference_max_b_t = declared if reference_max_b_t is not None else None
        self.max_b_t = max(self.certified_max_b_t, observed_reference, declared)
        if not isfinite(self.max_b_t) or self.max_b_t <= 0.0:
            raise OrbitValidationError("field maximum must be finite and positive")

    @staticmethod
    def _optional_map(
        values: Sequence[Sequence[float]] | None, shape: tuple[int, int], name: str
    ) -> np.ndarray | None:
        if values is None:
            return None
        result = np.asarray(values, dtype=np.float64)
        if result.shape != shape or not np.isfinite(result).all():
            raise OrbitValidationError(f"{name} shape/finite contract failed")
        return result

    @classmethod
    def from_field_artifact(
        cls, artifact: Mapping[str, object], *, source_identity_sha256: str
    ) -> "PsiBicubicField":
        mapping = artifact.get("field_map")
        if not isinstance(mapping, Mapping):
            raise OrbitValidationError("artifact.field_map is required")
        if "material_id" not in mapping:
            raise OrbitValidationError(
                "artifact field map requires an explicit material_id map"
            )
        summary = artifact.get("summary")
        reference_max = (
            summary.get("b_magnitude_max_t")
            if isinstance(summary, Mapping)
            else None
        )
        return cls(
            mapping["r_m"], mapping["z_m"], mapping["psi_wb"],
            material_id=mapping["material_id"],
            plasma_material_id="plasma",
            reference_br_t=mapping.get("b_r_t"),
            reference_bz_t=mapping.get("b_z_t"),
            reference_max_b_t=reference_max,
            source_identity_sha256=source_identity_sha256,
        )

    def _cell(self, r_m: float, z_m: float) -> tuple[int, int, float, float, float, float]:
        if not isfinite(r_m) or not isfinite(z_m):
            raise OrbitNumericsError("field query is nonfinite")
        if not self.r_m[0] <= r_m <= self.r_m[-1] or not self.z_m[0] <= z_m <= self.z_m[-1]:
            raise OrbitNumericsError("field query lies outside interpolation domain")
        i = min(int(np.searchsorted(self.r_m, r_m, side="right") - 1), len(self.r_m) - 2)
        j = min(int(np.searchsorted(self.z_m, z_m, side="right") - 1), len(self.z_m) - 2)
        if not self.traversable_cells[i, j]:
            raise OrbitNumericsError(
                "field query enters an interface or non-plasma interpolation cell"
            )
        dr = float(self.r_m[i + 1] - self.r_m[i])
        dz = float(self.z_m[j + 1] - self.z_m[j])
        return i, j, (r_m - self.r_m[i]) / dr, (z_m - self.z_m[j]) / dz, dr, dz

    def _g_values(
        self, r_m: float, z_m: float
    ) -> tuple[float, float, float]:
        i, j, s, t, dr, dz = self._cell(r_m, z_m)
        br, ds, _ = _basis(s)
        bz, dt, _ = _basis(t)
        f = np.empty((4, 4), dtype=np.float64)
        f[:2, :2] = self.g_wb_per_m2[i:i+2, j:j+2]
        f[2:, :2] = dr * self.dg_dr[i:i+2, j:j+2]
        f[:2, 2:] = dz * self.dg_dz[i:i+2, j:j+2]
        f[2:, 2:] = dr * dz * self.dg_drdz[i:i+2, j:j+2]
        g = float(br @ f @ bz)
        dg_dr = float(ds @ f @ bz / dr)
        dg_dz = float(br @ f @ dt / dz)
        if not all(isfinite(value) for value in (g, dg_dr, dg_dz)):
            raise OrbitNumericsError("interpolated regular flux variable is nonfinite")
        return g, dg_dr, dg_dz

    def psi_gradient(self, r_m: float, z_m: float) -> tuple[float, float, float]:
        g, dg_dr, dg_dz = self._g_values(r_m, z_m)
        psi_axis = float(self.psi_wb[0, 0])
        return (
            psi_axis + r_m * r_m * g,
            2.0 * r_m * g + r_m * r_m * dg_dr,
            r_m * r_m * dg_dz,
        )

    def field_cylindrical(self, r_m: float, z_m: float) -> tuple[float, float]:
        g, dg_dr, dg_dz = self._g_values(r_m, z_m)
        return -r_m * dg_dz, 2.0 * g + r_m * dg_dr

    def _raw_gradient(self, r_m: float, z_m: float) -> tuple[float, float, float]:
        return self.psi_gradient(r_m, z_m)

    def _cell_power_coefficients(self, i: int, j: int) -> np.ndarray:
        dr = float(self.r_m[i + 1] - self.r_m[i])
        dz = float(self.z_m[j + 1] - self.z_m[j])
        values = np.empty((4, 4), dtype=np.float64)
        values[:2, :2] = self.g_wb_per_m2[i:i+2, j:j+2]
        values[2:, :2] = dr * self.dg_dr[i:i+2, j:j+2]
        values[:2, 2:] = dz * self.dg_dz[i:i+2, j:j+2]
        values[2:, 2:] = dr * dz * self.dg_drdz[i:i+2, j:j+2]
        return _HERMITE_TO_POWER.T @ values @ _HERMITE_TO_POWER

    def _certified_field_bound(self) -> float:
        bounds: list[float] = []
        for i in range(len(self.r_m) - 1):
            for j in range(len(self.z_m) - 1):
                if not self.traversable_cells[i, j]:
                    continue
                coefficients = self._cell_power_coefficients(i, j)
                dr = float(self.r_m[i + 1] - self.r_m[i])
                dz = float(self.z_m[j + 1] - self.z_m[j])
                g_bound = float(np.sum(np.abs(coefficients)))
                gr_bound = float(
                    sum(
                        radial * abs(coefficients[radial, axial]) / dr
                        for radial in range(1, 4)
                        for axial in range(4)
                    )
                )
                gz_bound = float(
                    sum(
                        axial * abs(coefficients[radial, axial]) / dz
                        for radial in range(4)
                        for axial in range(1, 4)
                    )
                )
                radius = float(self.r_m[i + 1])
                bounds.append(hypot(radius * gz_bound, 2.0 * g_bound + radius * gr_bound))
        return max(bounds)

    def _certificate_tightness(
        self, minimum_ratio: float, *, samples_per_axis: int = 9
    ) -> CertificateTightness:
        dense_max = 0.0
        for i in range(len(self.r_m) - 1):
            for j in range(len(self.z_m) - 1):
                if not self.traversable_cells[i, j]:
                    continue
                coefficients = self._cell_power_coefficients(i, j)
                dr = float(self.r_m[i + 1] - self.r_m[i])
                dz = float(self.z_m[j + 1] - self.z_m[j])
                for s in np.linspace(0.0, 1.0, samples_per_axis):
                    radial_powers = np.array([1.0, s, s*s, s*s*s])
                    radial_derivative = np.array([0.0, 1.0, 2.0*s, 3.0*s*s])
                    radius = float(self.r_m[i] + s * dr)
                    for t in np.linspace(0.0, 1.0, samples_per_axis):
                        axial_powers = np.array([1.0, t, t*t, t*t*t])
                        axial_derivative = np.array([0.0, 1.0, 2.0*t, 3.0*t*t])
                        g = float(radial_powers @ coefficients @ axial_powers)
                        dg_dr = float(
                            radial_derivative @ coefficients @ axial_powers / dr
                        )
                        dg_dz = float(
                            radial_powers @ coefficients @ axial_derivative / dz
                        )
                        dense_max = max(
                            dense_max,
                            hypot(-radius * dg_dz, 2.0*g + radius*dg_dr),
                        )
        ratio = (
            dense_max / self.certified_max_b_t
            if self.certified_max_b_t > 0.0
            else 0.0
        )
        return CertificateTightness(
            dense_max,
            self.certified_max_b_t,
            ratio,
            minimum_ratio,
            bool(ratio >= minimum_ratio),
        )

    def magnetic_cartesian(self, position_m: np.ndarray) -> np.ndarray:
        x, y, z = map(float, position_m)
        radius = hypot(x, y)
        br, bz = self.field_cylindrical(radius, z)
        magnitude = hypot(br, bz)
        if magnitude > self.max_b_t * (1.0 + 64.0 * np.finfo(float).eps):
            raise OrbitNumericsError("runtime field exceeds certified maximum")
        if radius == 0.0:
            return np.array([0.0, 0.0, bz])
        return np.array([br * x / radius, br * y / radius, bz])

    def electric_cartesian(self, position_m: np.ndarray, time_s: float) -> np.ndarray:
        del position_m, time_s
        return np.zeros(3, dtype=np.float64)

    def reference_error(self) -> InterpolationErrorReport:
        if self.reference_br_t is None or self.reference_bz_t is None:
            raise OrbitValidationError("reference Br/Bz maps were not supplied")
        br_error: list[float] = []
        bz_error: list[float] = []
        psi_error: list[float] = []
        for i, radius in enumerate(self.r_m):
            for j, axial in enumerate(self.z_m):
                psi, _, _ = self._raw_gradient(float(radius), float(axial))
                br, bz = self.field_cylindrical(float(radius), float(axial))
                psi_error.append(psi - self.psi_wb[i, j])
                br_error.append(br - self.reference_br_t[i, j])
                bz_error.append(bz - self.reference_bz_t[i, j])
        squared = np.square(br_error) + np.square(bz_error)
        scale = max(self.max_b_t, np.finfo(float).tiny)
        return InterpolationErrorReport(
            len(squared), max(map(abs, psi_error)), max(map(abs, br_error)),
            max(map(abs, bz_error)), float(np.sqrt(np.mean(squared))),
            float(np.sqrt(np.mean(squared)) / scale),
        )


def compare_maps(
    first: PsiBicubicField, second: PsiBicubicField, *, samples_r: int = 17, samples_z: int = 31
) -> dict[str, float | int]:
    """Deterministic common-domain ψ/B resolution or domain comparison."""

    if min(samples_r, samples_z) < 2:
        raise OrbitValidationError("map comparison requires at least two samples per axis")
    r_max = min(float(first.r_m[-1]), float(second.r_m[-1]))
    z_min = max(float(first.z_m[0]), float(second.z_m[0]))
    z_max = min(float(first.z_m[-1]), float(second.z_m[-1]))
    if r_max <= 0.0 or z_max <= z_min:
        raise OrbitValidationError("maps have no common axisymmetric domain")
    psi_errors: list[float] = []
    b_errors: list[float] = []
    for radius in np.linspace(0.0, r_max, samples_r):
        for axial in np.linspace(z_min, z_max, samples_z):
            p1, _, _ = first._raw_gradient(float(radius), float(axial))
            p2, _, _ = second._raw_gradient(float(radius), float(axial))
            br1, bz1 = first.field_cylindrical(float(radius), float(axial))
            br2, bz2 = second.field_cylindrical(float(radius), float(axial))
            psi_errors.append(abs(p1 - p2))
            b_errors.append(hypot(br1 - br2, bz1 - bz2))
    scale = max(first.max_b_t, second.max_b_t, np.finfo(float).tiny)
    return {
        "sample_count": len(b_errors),
        "psi_max_abs_wb": max(psi_errors),
        "b_max_abs_t": max(b_errors),
        "b_rms_t": float(np.sqrt(np.mean(np.square(b_errors)))),
        "b_max_relative": max(b_errors) / scale,
    }


class AnalyticField:
    """Verified callable field for manufactured orbit tests."""

    def __init__(
        self,
        magnetic: Callable[[np.ndarray], np.ndarray],
        electric: Callable[[np.ndarray, float], np.ndarray] | None,
        max_b_t: float,
    ) -> None:
        self._magnetic = magnetic
        self._electric = electric
        if isinstance(max_b_t, bool):
            raise OrbitValidationError("analytic max_b_t must be a real scalar")
        self.max_b_t = float(max_b_t)
        if not isfinite(self.max_b_t) or self.max_b_t < 0.0:
            raise OrbitValidationError("analytic max_b_t must be finite and nonnegative")

    def magnetic_cartesian(self, position_m: np.ndarray) -> np.ndarray:
        return np.asarray(self._magnetic(position_m), dtype=np.float64)

    def electric_cartesian(self, position_m: np.ndarray, time_s: float) -> np.ndarray:
        if self._electric is None:
            return np.zeros(3)
        return np.asarray(self._electric(position_m, time_s), dtype=np.float64)
