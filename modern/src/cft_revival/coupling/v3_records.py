"""Accepted v3 records built only from connected constant-ψ surfaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from enum import Enum
from math import hypot, isfinite, sqrt
from sys import float_info
from typing import Any

from .models import (
    CouplingValidationError,
    EvidenceVerificationError,
    TopologyResolutionError,
    TopologyStatus,
    UncertaintyModel,
)
from .surfaces import (
    bilinear_sample,
    certify_contour_field,
    magnetic_null_geometry,
    trace_flux_contours,
    validate_simple_contour,
)
from .profiles import validate_uncertainty_model
from .v3_evidence import (
    AcceptedTopologyStabilityEvidence,
    AcceptedV3FieldEvidence,
    reverify_v3_evidence,
    reverify_v3_topology_stability,
)
from .v3_models import (
    BoundedMirrorProbability,
    CellRegistration,
    ElectronAdiabaticInputs,
    FluxCell,
    FluxContour,
    FluxQuantileOutcome,
    FluxSurfaceMirror,
    FluxSurfacePolicy,
    StabilityCase,
    SurfaceStatus,
    TopologyStabilityStudy,
    V3CouplingRecord,
    V3EvidenceIdentity,
    ValidatedPsiMap,
)

COUPLING_SCHEMA_VERSION = "cft-field-plasma-coupling/3.0.0"
_ELECTRON_MASS_KG = 9.1093837139e-31
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_ELECTRON_REST_ENERGY_EV = 510_998.95069


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise CouplingValidationError("v3 records cannot contain nonfinite numbers")
    return value


def _record_hash(record: V3CouplingRecord) -> str:
    payload = _json_value(record)
    payload["record_hash"] = ""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(b"cft-coupling-record-v3\0" + encoded).hexdigest()


def _check_case(case: StabilityCase, expected_role: str) -> None:
    if case.role != expected_role:
        raise EvidenceVerificationError(
            f"stability case role must be {expected_role!r}"
        )
    if len(case.full_map_hash) != 64 or any(
        character not in "0123456789abcdef" for character in case.full_map_hash.lower()
    ):
        raise EvidenceVerificationError("stability map hashes must be SHA-256")
    if case.cell_count < 0 or len(case.interior_cusp_z_m) != case.cell_count:
        raise EvidenceVerificationError("stability cell count/cusp list mismatch")
    if case.radial_samples < 2 or case.axial_samples < 3:
        raise EvidenceVerificationError("stability mesh dimensions are undersampled")
    values = (
        case.radius_m,
        case.z_min_m,
        case.z_max_m,
        *case.interior_cusp_z_m,
    )
    if any(not isfinite(float(value)) for value in values):
        raise EvidenceVerificationError("stability geometry must be finite")
    if case.radius_m <= 0.0 or case.z_max_m <= case.z_min_m:
        raise EvidenceVerificationError("stability domain is invalid")


def verify_topology_stability(
    study: TopologyStabilityStudy,
    *,
    field: ValidatedPsiMap,
    observed_cusp_z_m: tuple[float, ...],
) -> None:
    """Fail closed unless count/location survive mesh and domain changes."""

    _check_case(study.full_resolution, "full_resolution")
    _check_case(study.downsampled, "downsampled")
    _check_case(study.enlarged_domain, "enlarged_domain")
    if study.full_resolution.full_map_hash != field.full_map_hash:
        raise EvidenceVerificationError(
            "full-resolution stability case is not bound to accepted map"
        )
    counts = {
        study.full_resolution.cell_count,
        study.downsampled.cell_count,
        study.enlarged_domain.cell_count,
        len(observed_cusp_z_m),
    }
    if len(counts) != 1:
        raise TopologyResolutionError(
            "topology ambiguous: cell count changes with mesh/domain"
        )
    if study.maximum_cusp_shift_m < 0.0 or not isfinite(
        study.maximum_cusp_shift_m
    ):
        raise EvidenceVerificationError("maximum_cusp_shift_m must be finite")
    if not (
        study.downsampled.radial_samples < study.full_resolution.radial_samples
        or study.downsampled.axial_samples < study.full_resolution.axial_samples
    ):
        raise EvidenceVerificationError("downsampled case is not lower resolution")
    enlarged = study.enlarged_domain
    full = study.full_resolution
    if not (
        enlarged.radius_m > full.radius_m
        or enlarged.z_min_m < full.z_min_m
        or enlarged.z_max_m > full.z_max_m
    ):
        raise EvidenceVerificationError("enlarged-domain case is not enlarged")
    references = (
        observed_cusp_z_m,
        study.downsampled.interior_cusp_z_m,
        study.enlarged_domain.interior_cusp_z_m,
    )
    for candidate in references:
        if any(
            abs(left - right) > study.maximum_cusp_shift_m
            for left, right in zip(
                study.full_resolution.interior_cusp_z_m, candidate, strict=True
            )
        ):
            raise TopologyResolutionError(
                "topology ambiguous: cusp positions are not stable"
            )


def _probability(field_ratio: float) -> float:
    ratio = min(1.0, max(0.0, field_ratio))
    root = sqrt(max(0.0, 1.0 - ratio))
    return ratio / (1.0 + root)


def _invalid_probability(
    status: SurfaceStatus, reason: str
) -> BoundedMirrorProbability:
    return BoundedMirrorProbability(
        status,
        reason,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "bounded-extrema-v3",
    )


def _safe_nonnegative_sum(*values: float) -> float | None:
    if any(not isfinite(value) or value < 0.0 for value in values):
        return None
    scale = max(values, default=0.0)
    if scale == 0.0:
        return 0.0
    normalized = sum(value / scale for value in values)
    if normalized > float_info.max / scale:
        return None
    result = scale * normalized
    return result if isfinite(result) else None


def _safe_product(first: float, second: float) -> float | None:
    if not isfinite(first) or not isfinite(second):
        return None
    if first == 0.0 or second == 0.0:
        return 0.0
    if abs(first) > float_info.max / abs(second):
        return None
    result = first * second
    return result if isfinite(result) else None


def _field_scale_length(
    points: tuple[tuple[float, float], ...],
    magnitudes: tuple[float, ...],
    low_index: int,
) -> float | None:
    derivatives: list[float] = []
    for neighbor in (low_index - 1, low_index + 1):
        if 0 <= neighbor < len(points):
            ds = hypot(
                points[neighbor][0] - points[low_index][0],
                points[neighbor][1] - points[low_index][1],
            )
            if ds > 0.0:
                derivatives.append(
                    abs(magnitudes[neighbor] - magnitudes[low_index]) / ds
                )
    gradient = max(derivatives, default=0.0)
    if gradient == 0.0:
        return None
    result = magnitudes[low_index] / gradient
    return result if isfinite(result) and result > 0.0 else None


def _surface_mirror(
    field: ValidatedPsiMap,
    contour: FluxContour,
    *,
    cell_id: str,
    quantile: float,
    component: int,
    uncertainty: UncertaintyModel,
    surface_policy: FluxSurfacePolicy,
    electron_inputs: ElectronAdiabaticInputs | None,
) -> FluxSurfaceMirror:
    map_field_scale = max(
        hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    null_floor = max(
        surface_policy.null_field_absolute_floor_t,
        surface_policy.null_field_relative_floor * map_field_scale,
        uncertainty.absolute_independent_sigma_t,
    )
    certificate = certify_contour_field(
        field,
        contour,
        null_floor_t=null_floor,
        absolute_tolerance_t=surface_policy.segment_bound_absolute_tolerance_t,
        relative_tolerance=surface_policy.segment_bound_relative_tolerance,
        maximum_depth=surface_policy.segment_max_depth,
    )
    magnitudes = certificate.sampled_b_t
    points = certificate.sampled_points_rz_m
    low_index = min(range(len(magnitudes)), key=magnitudes.__getitem__)
    high_index = max(range(len(magnitudes)), key=magnitudes.__getitem__)
    low = certificate.sampled_b_low_upper_t
    high = certificate.sampled_b_high_lower_t
    scale_length = (
        None
        if certificate.maximum_gradient_t_per_m == 0.0
        else certificate.certified_b_low_lower_t
        / certificate.maximum_gradient_t_per_m
    )
    gyroradius: float | None = None
    epsilon: float | None = None
    status: SurfaceStatus | None = None
    reason = ""
    relative_error = _safe_nonnegative_sum(
        uncertainty.relative_independent_sigma,
        surface_policy.interpolation_relative_error,
        surface_policy.surface_relative_error,
    )
    low_relative = (
        None if relative_error is None else _safe_product(relative_error, low)
    )
    high_relative = (
        None if relative_error is None else _safe_product(relative_error, high)
    )
    low_uncovered = (
        None
        if low_relative is None
        else _safe_nonnegative_sum(
            uncertainty.absolute_independent_sigma_t,
            uncertainty.common_mode_sigma_t,
            low_relative,
        )
    )
    high_uncovered = (
        None
        if high_relative is None
        else _safe_nonnegative_sum(
            uncertainty.absolute_independent_sigma_t,
            uncertainty.common_mode_sigma_t,
            high_relative,
        )
    )
    field_error_low = (
        None
        if low_uncovered is None
        else _safe_product(uncertainty.coverage_factor, low_uncovered)
    )
    field_error_high = (
        None
        if high_uncovered is None
        else _safe_product(uncertainty.coverage_factor, high_uncovered)
    )
    geometry_simple, geometry_reason, _, _ = validate_simple_contour(
        contour.points_rz_m,
        tolerance_m=surface_policy.connectivity_tolerance_m,
        domain_bounds=(
            field.r_m[0],
            field.r_m[-1],
            field.z_m[0],
            field.z_m[-1],
        ),
    )
    if not contour.simple or not geometry_simple:
        status = SurfaceStatus.DISCONNECTED
        reason = contour.topology_reason if not contour.simple else geometry_reason
    elif len(contour.points_rz_m) < surface_policy.minimum_contour_points:
        status = SurfaceStatus.DISCONNECTED
        reason = "contour has too few connected points"
    elif not contour.closed or contour.touches_boundary:
        status = SurfaceStatus.OPEN_BOUNDARY
        reason = "constant-psi contour is truncated by the finite map"
    elif not certificate.regular or certificate.certified_b_low_lower_t <= null_floor:
        status = SurfaceStatus.EXACT_NULL
        reason = certificate.reason
    elif field_error_low is None or field_error_high is None:
        status = SurfaceStatus.NUMERICALLY_INVALID
        reason = "covered field uncertainty is not representable"
    elif electron_inputs is None:
        status = SurfaceStatus.MISSING_ADIABATIC_INPUTS
        reason = "electron energy inputs are required for mirror validity"
    else:
        energy = float(electron_inputs.kinetic_energy_ev)
        fraction = float(electron_inputs.perpendicular_energy_fraction)
        threshold = float(electron_inputs.maximum_gyroradius_to_scale_length)
        if (
            not isfinite(energy)
            or energy <= 0.0
            or not isfinite(fraction)
            or not 0.0 <= fraction <= 1.0
            or not isfinite(threshold)
            or threshold <= 0.0
        ):
            raise CouplingValidationError("electron adiabatic inputs are invalid")
        if energy >= _ELECTRON_REST_ENERGY_EV:
            status = SurfaceStatus.PHYSICALLY_INVALID
            reason = "nonrelativistic gyroradius model is invalid at this energy"
        perpendicular_energy_j = _safe_product(
            energy, _ELEMENTARY_CHARGE_C * fraction
        )
        momentum_squared = (
            None
            if perpendicular_energy_j is None
            else _safe_product(
                2.0 * _ELECTRON_MASS_KG, perpendicular_energy_j
            )
        )
        denominator = _safe_product(
            _ELEMENTARY_CHARGE_C, certificate.certified_b_low_lower_t
        )
        if status is None and (
            momentum_squared is None
            or denominator is None
            or denominator <= 0.0
        ):
            status = SurfaceStatus.NUMERICALLY_INVALID
            reason = "gyroradius inputs are not representable"
        elif status is None:
            gyroradius = sqrt(momentum_squared) / denominator
            epsilon = 0.0 if scale_length is None else gyroradius / scale_length
        if status is None and (
            not isfinite(gyroradius) or not isfinite(epsilon)
        ):
            status = SurfaceStatus.NUMERICALLY_INVALID
            reason = "gyroradius/scale-length calculation is nonfinite"
        elif status is None and epsilon > threshold:
            status = SurfaceStatus.NONADIABATIC
            reason = "electron gyroradius is not small relative to field scale length"
    if status is not None:
        probability = _invalid_probability(status, reason)
    else:
        low_lower = max(
            0.0, certificate.certified_b_low_lower_t - field_error_low
        )
        low_upper = _safe_nonnegative_sum(
            certificate.sampled_b_low_upper_t, field_error_low
        )
        high_lower = max(
            0.0, certificate.sampled_b_high_lower_t - field_error_high
        )
        high_upper = _safe_nonnegative_sum(
            certificate.certified_b_high_upper_t, field_error_high
        )
        if low_upper is None or high_upper is None:
            probability = _invalid_probability(
                SurfaceStatus.NUMERICALLY_INVALID,
                "covered mirror extrema bounds overflowed",
            )
        elif high_lower <= 0.0 or high_upper <= 0.0 or low_lower <= 0.0:
            probability = _invalid_probability(
                SurfaceStatus.UNCERTAINTY_DOMINATED,
                "covered mirror extrema bounds include zero",
            )
        else:
            ratio_lower = min(1.0, low_lower / high_upper)
            ratio_upper = min(1.0, low_upper / high_lower)
            nominal = _probability(low / high)
            probability_lower = _probability(ratio_lower)
            probability_upper = _probability(ratio_upper)
            interval_width = probability_upper - probability_lower
            dominance = interval_width / max(nominal, 1.0e-300)
            mirror_lower = 1.0 / ratio_upper if ratio_upper > 0.0 else None
            mirror_upper = 1.0 / ratio_lower if ratio_lower > 0.0 else None
            if (
                any(
                    not isfinite(value)
                    for value in (
                        ratio_lower,
                        ratio_upper,
                        nominal,
                        probability_lower,
                        probability_upper,
                        dominance,
                    )
                )
                or mirror_lower is None
                or mirror_upper is None
                or not isfinite(mirror_lower)
                or not isfinite(mirror_upper)
            ):
                probability = _invalid_probability(
                    SurfaceStatus.NUMERICALLY_INVALID,
                    "mirror/probability bounds are not finitely representable",
                )
            elif dominance > surface_policy.uncertainty_dominance_factor:
                probability = _invalid_probability(
                    SurfaceStatus.UNCERTAINTY_DOMINATED,
                    "bounded probability uncertainty dominates the nominal value",
                )
            else:
                probability = BoundedMirrorProbability(
                    SurfaceStatus.VALID,
                    "bounded same-surface mirror estimate passed all gates",
                    nominal,
                    probability_lower,
                    probability_upper,
                    ratio_lower,
                    ratio_upper,
                    mirror_lower,
                    mirror_upper,
                    "bounded-extrema-v3",
                )
    return FluxSurfaceMirror(
        cell_id,
        quantile,
        contour.psi_wb,
        component,
        contour,
        certificate,
        low,
        high,
        points[low_index],
        points[high_index],
        scale_length,
        gyroradius,
        epsilon,
        probability.status is SurfaceStatus.VALID,
        probability,
    )


def _cell_flux_range(
    field: ValidatedPsiMap, z_start: float, z_end: float
) -> tuple[float, float]:
    values = tuple(
        field.psi_wb[i][j]
        for i in range(1, len(field.r_m) - 1)
        for j, z in enumerate(field.z_m)
        if z_start < z < z_end
    )
    if not values:
        raise TopologyResolutionError("cell has no interior ψ samples")
    low, high = min(values), max(values)
    if high <= low:
        raise TopologyResolutionError("cell ψ range is degenerate")
    return low, high


def build_coupling_record(
    evidence: AcceptedV3FieldEvidence,
    *,
    stability_evidence: AcceptedTopologyStabilityEvidence,
    cell_registrations: tuple[CellRegistration, ...],
    electron_inputs: ElectronAdiabaticInputs | None,
    surface_policy: FluxSurfacePolicy = FluxSurfacePolicy(),
    uncertainty_model: UncertaintyModel = UncertaintyModel(),
    reference_time_utc: datetime | None = None,
) -> V3CouplingRecord:
    """Build an accepted record; every mirror pair lies on one connected ψ contour."""

    numeric_policy = (
        surface_policy.psi_absolute_tolerance_wb,
        surface_policy.psi_relative_tolerance,
        surface_policy.connectivity_tolerance_m,
        surface_policy.interpolation_relative_error,
        surface_policy.surface_relative_error,
        surface_policy.uncertainty_dominance_factor,
        surface_policy.null_field_absolute_floor_t,
        surface_policy.null_field_relative_floor,
        surface_policy.segment_bound_absolute_tolerance_t,
        surface_policy.segment_bound_relative_tolerance,
    )
    if (
        any(not isfinite(value) or value < 0.0 for value in numeric_policy)
        or surface_policy.connectivity_tolerance_m <= 0.0
        or surface_policy.uncertainty_dominance_factor <= 0.0
        or surface_policy.boundary_exclusion_cells < 1
        or surface_policy.minimum_contour_points < 2
        or isinstance(surface_policy.segment_max_depth, bool)
        or surface_policy.segment_max_depth < 1
        or surface_policy.saddle_tie_policy
        not in ("reject", "pair_01_23", "pair_03_12")
    ):
        raise CouplingValidationError("flux-surface policy is invalid")
    validate_uncertainty_model(uncertainty_model)
    snapshot = reverify_v3_evidence(
        evidence, reference_time_utc=reference_time_utc
    )
    field = snapshot.field_map
    stability_study = reverify_v3_topology_stability(
        stability_evidence,
        full_map_hash=field.full_map_hash,
        reference_time_utc=reference_time_utc,
    )
    null_points, boundary_nulls = magnetic_null_geometry(
        field, boundary_exclusion_cells=surface_policy.boundary_exclusion_cells
    )
    cusp_points: list[tuple[float, float]] = []
    for point in null_points:
        if not cusp_points or abs(point[1] - cusp_points[-1][1]) > (
            surface_policy.connectivity_tolerance_m
        ):
            cusp_points.append(point)
    cusp_z = tuple(point[1] for point in cusp_points)
    if not cusp_points:
        raise TopologyResolutionError("no geometry-identified interior cusp/separatrix")
    verify_topology_stability(
        stability_study, field=field, observed_cusp_z_m=cusp_z
    )
    if len(cell_registrations) != len(cusp_points):
        raise CouplingValidationError(
            "preregistered cell count must equal stable interior cusp count"
        )
    z_bounds = [field.z_m[0]]
    z_bounds.extend(
        (left[1] + right[1]) * 0.5
        for left, right in zip(cusp_points, cusp_points[1:])
    )
    z_bounds.append(field.z_m[-1])
    cells: list[FluxCell] = []
    for index, (registration, cusp) in enumerate(
        zip(cell_registrations, cusp_points, strict=True)
    ):
        if not registration.cell_id.strip():
            raise CouplingValidationError("cell_id must not be empty")
        if (
            not registration.flux_quantiles
            or any(
                not isfinite(value) or not 0.0 < value < 1.0
                for value in registration.flux_quantiles
            )
            or any(
                right <= left
                for left, right in zip(
                    registration.flux_quantiles,
                    registration.flux_quantiles[1:],
                )
            )
        ):
            raise CouplingValidationError(
                "flux quantiles must be unique, increasing, and strictly interior"
            )
        z_start, z_end = z_bounds[index], z_bounds[index + 1]
        psi_low, psi_high = _cell_flux_range(field, z_start, z_end)
        mirrors: list[FluxSurfaceMirror] = []
        outcomes: list[FluxQuantileOutcome] = []
        for quantile in registration.flux_quantiles:
            target = psi_low + quantile * (psi_high - psi_low)
            try:
                contours = trace_flux_contours(field, target, surface_policy)
            except TopologyResolutionError as error:
                outcomes.append(
                    FluxQuantileOutcome(
                        quantile,
                        target,
                        SurfaceStatus.DISCONNECTED,
                        str(error),
                        (),
                        (),
                    )
                )
                continue
            local = tuple(
                contour
                for contour in contours
                if z_start
                < sum(point[1] for point in contour.points_rz_m)
                / len(contour.points_rz_m)
                < z_end
            )
            quantile_mirrors: list[FluxSurfaceMirror] = []
            for component, contour in enumerate(local):
                mirror = _surface_mirror(
                    field,
                    contour,
                    cell_id=registration.cell_id,
                    quantile=quantile,
                    component=component,
                    uncertainty=uncertainty_model,
                    surface_policy=surface_policy,
                    electron_inputs=electron_inputs,
                )
                quantile_mirrors.append(mirror)
                mirrors.append(mirror)
            accepted_components = tuple(
                mirror.contour_component
                for mirror in quantile_mirrors
                if mirror.probability.status is SurfaceStatus.VALID
            )
            if not quantile_mirrors:
                outcome_status = SurfaceStatus.DISCONNECTED
                outcome_reason = "no local contour component for required quantile"
            elif len(accepted_components) != len(quantile_mirrors):
                failed = next(
                    mirror
                    for mirror in quantile_mirrors
                    if mirror.probability.status is not SurfaceStatus.VALID
                )
                outcome_status = failed.probability.status
                outcome_reason = (
                    "required quantile has a failed component: "
                    + failed.probability.reason
                )
            else:
                outcome_status = SurfaceStatus.VALID
                outcome_reason = "every component for required quantile passed"
            outcomes.append(
                FluxQuantileOutcome(
                    quantile,
                    target,
                    outcome_status,
                    outcome_reason,
                    tuple(mirror.contour_component for mirror in quantile_mirrors),
                    accepted_components,
                )
            )
        all_quantiles_valid = len(outcomes) == len(registration.flux_quantiles) and all(
            outcome.status is SurfaceStatus.VALID for outcome in outcomes
        )
        status = (
            SurfaceStatus.VALID
            if all_quantiles_valid
            else next(
                (
                    outcome.status
                    for outcome in outcomes
                    if outcome.status is not SurfaceStatus.VALID
                ),
                SurfaceStatus.DISCONNECTED,
            )
        )
        reason = (
            "every preregistered quantile and connected component passed all gates"
            if all_quantiles_valid
            else "at least one preregistered quantile failed atomically"
        )
        cells.append(
            FluxCell(
                registration.cell_id,
                z_start,
                z_end,
                cusp[1],
                bilinear_sample(field, field.psi_wb, cusp),
                tuple(mirrors),
                tuple(outcomes),
                status,
                reason,
            )
        )
    resolved = all(cell.status is SurfaceStatus.VALID for cell in cells)
    identity = V3EvidenceIdentity(
        artifact_hash=snapshot.claims.artifact_hash,
        full_map_hash=field.full_map_hash,
        source_hash=snapshot.claims.source_hash,
        geometry_hash=snapshot.claims.geometry_hash,
        material_hash=snapshot.claims.material_hash,
        mesh_hash=snapshot.claims.mesh_hash,
        domain_hash=snapshot.claims.domain_hash,
        evidence_binding_hash=snapshot.claims.evidence_binding_hash,
        artifact_schema_version=snapshot.claims.artifact_schema_version,
        model_level=snapshot.claims.model_level,
        field_model_id=snapshot.claims.field_model_id,
        field_model_hash=snapshot.claims.field_model_hash,
        code_hash=snapshot.claims.code_hash,
        config_hash=snapshot.claims.config_hash,
        backend_id=snapshot.claims.backend_id,
        backend_version=snapshot.claims.backend_version,
        adapter_id=snapshot.adapter_id,
        adapter_code_hash=snapshot.adapter_code_hash,
        adapter_contract=snapshot.adapter_contract,
        generated_at_utc=snapshot.claims.generated_at_utc,
        diagnostics=snapshot.claims.diagnostics,
        validation_policy=snapshot.validation_policy,
    )
    provisional = V3CouplingRecord(
        COUPLING_SCHEMA_VERSION,
        "",
        TopologyStatus.RESOLVED if resolved else TopologyStatus.AMBIGUOUS,
        (
            "stable cells contain accepted connected same-psi mirror distributions"
            if resolved
            else "stable geometry found, but at least one cell has no accepted mirror surface"
        ),
        identity,
        stability_study,
        surface_policy,
        uncertainty_model,
        electron_inputs,
        cell_registrations,
        boundary_nulls,
        cusp_z,
        tuple(cells),
    )
    return replace(provisional, record_hash=_record_hash(provisional))


def coupling_record_dict(record: V3CouplingRecord) -> dict[str, Any]:
    payload = _json_value(record)
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload


def global_solver_inputs(
    record: V3CouplingRecord,
) -> tuple[dict[str, float | str], ...]:
    """Expose only accepted bounded surfaces; proxies and invalid surfaces cannot pass."""

    if (
        record.schema_version != COUPLING_SCHEMA_VERSION
        or record.topology_status is not TopologyStatus.RESOLVED
    ):
        return ()
    rows: list[dict[str, float | str]] = []
    for cell in record.cells:
        for surface in cell.surfaces:
            probability = surface.probability
            if (
                probability.status is not SurfaceStatus.VALID
                or probability.nominal_probability is None
                or probability.probability_lower is None
                or probability.probability_upper is None
            ):
                continue
            rows.append(
                {
                    "record_hash": record.record_hash,
                    "cell_id": cell.cell_id,
                    "flux_quantile": surface.flux_quantile,
                    "psi_wb": surface.psi_wb,
                    "loss_cone_probability": probability.nominal_probability,
                    "loss_cone_probability_lower": probability.probability_lower,
                    "loss_cone_probability_upper": probability.probability_upper,
                    "b_low_t": surface.b_low_t,
                    "b_high_t": surface.b_high_t,
                    "field_map_hash": record.identity.full_map_hash,
                    "geometry_hash": record.identity.geometry_hash,
                    "material_hash": record.identity.material_hash,
                    "mesh_hash": record.identity.mesh_hash,
                    "domain_hash": record.identity.domain_hash,
                    "field_model_id": record.identity.field_model_id,
                    "coverage_factor": record.uncertainty_model.coverage_factor,
                    "primary_artifact_hash": (
                        record.stability_study.full_resolution.artifact_hash
                    ),
                    "downsampled_artifact_hash": (
                        record.stability_study.downsampled.artifact_hash
                    ),
                    "enlarged_artifact_hash": (
                        record.stability_study.enlarged_domain.artifact_hash
                    ),
                }
            )
    return tuple(rows)
