"""Source-backed HEMP/CFT wall-cusp and inter-cusp-cell assessment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from math import hypot, isfinite, sin, sqrt
from typing import Any

from .models import CouplingValidationError, EvidenceVerificationError, UncertaintyModel
from .profiles import validate_uncertainty_model
from .surfaces import bilinear_sample, certify_contour_field, trace_flux_contours
from .v3_evidence import _V3Snapshot
from .v3_models import FluxContour, FluxSurfacePolicy, V3EvidenceIdentity
from .v4_evidence import (
    AcceptedV4MapSet,
    reverify_v4_map_set,
    v4_map_set_evidence_fingerprints,
)
from .v4_validation import (
    AcceptedHeldOutValidationEvidence,
    reverify_held_out_validation,
)
from .v4_models import (
    AxialDominanceMetrics,
    AxialDominancePolicy,
    CFTCell,
    CFTCellRegistration,
    CFTGeometry,
    CFTStabilityPolicy,
    CFT_V4_DEVELOPMENT_MANIFEST,
    ClosedIslandDiagnostic,
    ElectronOrbitSample,
    FieldLineSeed,
    FieldLineTracePolicy,
    HeldOutValidationRegistration,
    OrbitAssessment,
    OrbitVerificationAdapter,
    OrbitVerificationClaims,
    OrbitVerificationIdentity,
    SeedPathOutcome,
    V4CouplingRecord,
    V4Criterion,
    V4MapAssessment,
    V4StabilityAssessment,
    V4Status,
    WallConnectedMirrorPath,
    WallCusp,
    WallCuspPolicy,
    validation_set_manifest_hash,
)

COUPLING_V4_SCHEMA_VERSION = "cft-field-plasma-coupling/4.2.0"
_ELECTRON_MASS_KG = 9.1093837139e-31
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_ELECTRON_REST_ENERGY_EV = 510_998.95069
_HEX = frozenset("0123456789abcdef")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value.lower())
    )


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
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not isfinite(value):
        raise CouplingValidationError("v4 records cannot contain nonfinite values")
    return value


def _hash_payload(label: bytes, payload: Any) -> str:
    encoded = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(label + b"\0" + encoded).hexdigest()


def _path_identity_hash(
    full_map_hash: str,
    seed_id: str,
    direction: int,
    psi_start_wb: float,
    points_rz_m: tuple[tuple[float, float], ...],
) -> str:
    return _hash_payload(
        b"cft-v4-field-line-path",
        {
            "full_map_hash": full_map_hash,
            "seed_id": seed_id,
            "direction": direction,
            "psi_start_wb": psi_start_wb,
            "points": points_rz_m,
        },
    )


def _identity(snapshot: _V3Snapshot) -> V3EvidenceIdentity:
    claims = snapshot.claims
    return V3EvidenceIdentity(
        artifact_hash=claims.artifact_hash,
        full_map_hash=snapshot.field_map.full_map_hash,
        source_hash=claims.source_hash,
        geometry_hash=claims.geometry_hash,
        material_hash=claims.material_hash,
        mesh_hash=claims.mesh_hash,
        domain_hash=claims.domain_hash,
        evidence_binding_hash=claims.evidence_binding_hash,
        artifact_schema_version=claims.artifact_schema_version,
        model_level=claims.model_level,
        field_model_id=claims.field_model_id,
        field_model_hash=claims.field_model_hash,
        code_hash=claims.code_hash,
        config_hash=claims.config_hash,
        backend_id=claims.backend_id,
        backend_version=claims.backend_version,
        adapter_id=snapshot.adapter_id,
        adapter_code_hash=snapshot.adapter_code_hash,
        adapter_contract=snapshot.adapter_contract,
        generated_at_utc=claims.generated_at_utc,
        diagnostics=claims.diagnostics,
        validation_policy=snapshot.validation_policy,
    )


def _validate_inputs(
    geometry: CFTGeometry,
    cusp: WallCuspPolicy,
    trace: FieldLineTracePolicy,
    axial: AxialDominancePolicy,
    stability: CFTStabilityPolicy,
    uncertainty: UncertaintyModel,
) -> None:
    validate_uncertainty_model(uncertainty)
    numeric = (
        geometry.channel_wall_radius_m,
        geometry.plasma_z_min_m,
        geometry.plasma_z_max_m,
        geometry.core_radius_m,
        cusp.minimum_prominence_t,
        cusp.prominence_support_half_width_m,
        cusp.minimum_cusp_separation_m,
        cusp.minimum_wall_radial_fraction,
        cusp.endpoint_plane_tolerance_m,
        cusp.axial_boundary_margin_m,
        cusp.minimum_endpoint_high_field_fraction,
        trace.step_m,
        trace.wall_tolerance_m,
        trace.maximum_psi_drift_wb,
        trace.minimum_b_t,
        trace.interpolation_relative_error,
        trace.path_relative_error,
        trace.uncertainty_dominance_factor,
        axial.pointwise_axial_fraction_threshold,
        axial.minimum_passing_fraction,
        axial.minimum_mean_axial_fraction,
        stability.maximum_cusp_shift_m,
        stability.maximum_cusp_strength_relative_change,
        stability.maximum_endpoint_shift_m,
        stability.maximum_cell_bound_shift_m,
        stability.maximum_axial_metric_change,
    )
    if any(not isfinite(value) for value in numeric):
        raise CouplingValidationError("v4 policies must be finite")
    nonnegative = (
        geometry.channel_wall_radius_m,
        geometry.core_radius_m,
        *numeric[4:],
    )
    if any(value < 0.0 for value in nonnegative):
        raise CouplingValidationError("v4 policies must be non-negative")
    if not (
        0.0 < geometry.core_radius_m < geometry.channel_wall_radius_m
        and geometry.plasma_z_min_m < geometry.plasma_z_max_m
        and _nonempty_text(geometry.geometry_id)
        and trace.step_m > 0.0
        and type(trace.maximum_steps) is int
        and trace.maximum_steps > 0
        and trace.minimum_b_t > 0.0
        and trace.uncertainty_dominance_factor > 0.0
        and cusp.prominence_support_half_width_m > 0.0
        and cusp.minimum_cusp_separation_m > 0.0
        and type(cusp.minimum_bundle_paths) is int
        and cusp.minimum_bundle_paths > 0
    ):
        raise CouplingValidationError("v4 geometry or integer policy is invalid")
    fractions = (
        cusp.minimum_wall_radial_fraction,
        cusp.minimum_endpoint_high_field_fraction,
        axial.pointwise_axial_fraction_threshold,
        axial.minimum_passing_fraction,
        axial.minimum_mean_axial_fraction,
    )
    if any(value > 1.0 for value in fractions):
        raise CouplingValidationError("v4 fraction policies must be in [0,1]")


def _validate_registrations(
    registrations: tuple[CFTCellRegistration, ...],
) -> None:
    if not registrations:
        raise CouplingValidationError("v4 requires preregistered cells")
    cell_ids: set[str] = set()
    seed_ids: set[str] = set()
    for registration in registrations:
        if (
            not _nonempty_text(registration.cell_id)
            or registration.cell_id in cell_ids
        ):
            raise CouplingValidationError("v4 cell IDs must be nonempty and unique")
        cell_ids.add(registration.cell_id)
        if not registration.seeds:
            raise CouplingValidationError("each v4 cell requires seeds")
        for seed in registration.seeds:
            sample_ids: set[str] = set()
            if not _nonempty_text(seed.seed_id) or seed.seed_id in seed_ids:
                raise CouplingValidationError(
                    "v4 seed IDs must be nonempty and globally unique"
                )
            seed_ids.add(seed.seed_id)
            if not seed.electron_samples:
                raise CouplingValidationError(
                    "each v4 seed requires prescribed electron samples"
                )
            for sample in seed.electron_samples:
                if (
                    not _nonempty_text(sample.sample_id)
                    or sample.sample_id in sample_ids
                ):
                    raise CouplingValidationError(
                    "v4 sample IDs must be nonempty and unique per seed"
                    )
                sample_ids.add(sample.sample_id)


def _wall_samples(snapshot: _V3Snapshot, wall_radius: float) -> tuple[tuple[float, float, float], ...]:
    field = snapshot.field_map
    return tuple(
        (
            z,
            bilinear_sample(field, field.b_r_t, (wall_radius, z)),
            bilinear_sample(field, field.b_z_t, (wall_radius, z)),
        )
        for z in field.z_m
    )


def _quadratic_peak(
    samples: tuple[tuple[float, float, float], ...],
    index: int,
) -> tuple[float, float]:
    x0, x1, x2 = (
        samples[index - 1][0],
        samples[index][0],
        samples[index + 1][0],
    )
    y0, y1, y2 = (
        abs(samples[index - 1][1]),
        abs(samples[index][1]),
        abs(samples[index + 1][1]),
    )
    left_slope = (y1 - y0) / (x1 - x0)
    right_slope = (y2 - y1) / (x2 - x1)
    curvature = (right_slope - left_slope) / (x2 - x0)
    if not isfinite(curvature) or curvature >= 0.0:
        return x1, y1
    derivative = left_slope + curvature * (x1 - x0)
    location = x1 - derivative / (2.0 * curvature)
    if not x0 <= location <= x2:
        return x1, y1
    delta = location - x1
    value = y1 + derivative * delta + curvature * delta * delta
    if not isfinite(value) or value < y1:
        return x1, y1
    return location, value


def _topographic_side_minimum(
    snapshot: _V3Snapshot,
    samples: tuple[tuple[float, float, float], ...],
    index: int,
    direction: int,
    peak_value: float,
    support_boundary_z: float,
    wall_radius_m: float,
) -> float:
    minimum = abs(samples[index][1])
    cursor = index + direction
    reached_higher_peak = False
    while 0 <= cursor < len(samples):
        z, br, _ = samples[cursor]
        if (
            direction < 0
            and z < support_boundary_z
            or direction > 0
            and z > support_boundary_z
        ):
            break
        value = abs(br)
        minimum = min(minimum, value)
        if value > peak_value:
            reached_higher_peak = True
            break
        cursor += direction
    if not reached_higher_peak:
        boundary_value = abs(
            bilinear_sample(
                snapshot.field_map,
                snapshot.field_map.b_r_t,
                (wall_radius_m, support_boundary_z),
            )
        )
        minimum = min(minimum, boundary_value)
    return minimum


def _detect_cusps(
    snapshot: _V3Snapshot,
    geometry: CFTGeometry,
    policy: WallCuspPolicy,
) -> tuple[WallCusp, ...]:
    samples = _wall_samples(snapshot, geometry.channel_wall_radius_m)
    detected: list[WallCusp] = []
    lower_limit = (
        geometry.plasma_z_min_m + policy.axial_boundary_margin_m
    )
    upper_limit = (
        geometry.plasma_z_max_m - policy.axial_boundary_margin_m
    )
    for index in range(1, len(samples) - 1):
        sample_z, sample_br, _ = samples[index]
        sample_value = abs(sample_br)
        if not lower_limit < sample_z < upper_limit:
            continue
        left, right = abs(samples[index - 1][1]), abs(samples[index + 1][1])
        if sample_value <= left or sample_value <= right:
            continue
        z, peak_value = _quadratic_peak(samples, index)
        left_boundary = max(
            lower_limit,
            z - policy.prominence_support_half_width_m,
        )
        right_boundary = min(
            upper_limit,
            z + policy.prominence_support_half_width_m,
        )
        if not left_boundary < z < right_boundary:
            continue
        left_minimum = _topographic_side_minimum(
            snapshot,
            samples,
            index,
            -1,
            peak_value,
            left_boundary,
            geometry.channel_wall_radius_m,
        )
        right_minimum = _topographic_side_minimum(
            snapshot,
            samples,
            index,
            1,
            peak_value,
            right_boundary,
            geometry.channel_wall_radius_m,
        )
        prominence = peak_value - max(left_minimum, right_minimum)
        sign = -1.0 if sample_br < 0.0 else 1.0
        br = sign * peak_value
        bz = bilinear_sample(
            snapshot.field_map,
            snapshot.field_map.b_z_t,
            (geometry.channel_wall_radius_m, z),
        )
        magnitude = hypot(br, bz)
        radial_fraction = 0.0 if magnitude == 0.0 else peak_value / magnitude
        if (
            prominence >= policy.minimum_prominence_t
            and radial_fraction >= policy.minimum_wall_radial_fraction
        ):
            detected.append(
                WallCusp(
                    "",
                    z,
                    br,
                    magnitude,
                    prominence,
                    radial_fraction,
                    0,
                    False,
                )
            )
    selected: list[WallCusp] = []
    for candidate in sorted(
        detected,
        key=lambda item: (-item.prominence_t, -abs(item.wall_br_t), item.z_m),
    ):
        if any(
            abs(candidate.z_m - retained.z_m)
            < policy.minimum_cusp_separation_m
            for retained in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item.z_m)
    return tuple(
        replace(cusp, cusp_id=f"wall-cusp-{index + 1:03d}")
        for index, cusp in enumerate(selected)
    )


def _unit_field(snapshot: _V3Snapshot, point: tuple[float, float], direction: int) -> tuple[float, float]:
    field = snapshot.field_map
    br = bilinear_sample(field, field.b_r_t, point)
    bz = bilinear_sample(field, field.b_z_t, point)
    magnitude = hypot(br, bz)
    if not isfinite(magnitude) or magnitude == 0.0:
        raise CouplingValidationError("field-line trace encountered a magnetic null")
    return direction * br / magnitude, direction * bz / magnitude


class _WallStageCrossing(RuntimeError):
    pass


def _rk4_step(
    snapshot: _V3Snapshot,
    point: tuple[float, float],
    direction: int,
    step: float,
    wall_radius_m: float,
) -> tuple[float, float]:
    def moved(base: tuple[float, float], vector: tuple[float, float], scale: float) -> tuple[float, float]:
        return base[0] + scale * vector[0], base[1] + scale * vector[1]

    def sampled(candidate: tuple[float, float]) -> tuple[float, float]:
        if candidate[0] > wall_radius_m:
            raise _WallStageCrossing
        return _unit_field(snapshot, candidate, direction)

    k1 = sampled(point)
    k2 = sampled(moved(point, k1, step * 0.5))
    k3 = sampled(moved(point, k2, step * 0.5))
    k4 = sampled(moved(point, k3, step))
    return (
        point[0] + step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        point[1] + step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _probability(ratio: float) -> float:
    bounded = min(1.0, max(0.0, ratio))
    return bounded / (1.0 + sqrt(max(0.0, 1.0 - bounded)))


def _orbit_identity(
    adapter: OrbitVerificationAdapter | None,
) -> OrbitVerificationIdentity | None:
    if adapter is None:
        return None
    if not isinstance(adapter, OrbitVerificationAdapter):
        raise EvidenceVerificationError(
            "orbit_adapter must implement OrbitVerificationAdapter"
        )
    text = (
        adapter.adapter_id,
        adapter.adapter_version,
        adapter.orbit_model_id,
        adapter.orbit_model_version,
        adapter.convergence_id,
        adapter.convergence_version,
    )
    hashes = (
        adapter.adapter_code_hash,
        adapter.orbit_code_hash,
        adapter.orbit_config_hash,
        adapter.convergence_config_hash,
    )
    if any(not _nonempty_text(value) for value in text):
        raise EvidenceVerificationError(
            "orbit adapter/model/convergence identities and versions are required"
        )
    if any(not _is_sha256(value) for value in hashes):
        raise EvidenceVerificationError(
            "orbit adapter/model/config/convergence hashes must be SHA-256"
        )
    return OrbitVerificationIdentity(
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        adapter_code_hash=adapter.adapter_code_hash,
        orbit_model_id=adapter.orbit_model_id,
        orbit_model_version=adapter.orbit_model_version,
        orbit_code_hash=adapter.orbit_code_hash,
        orbit_config_hash=adapter.orbit_config_hash,
        convergence_id=adapter.convergence_id,
        convergence_version=adapter.convergence_version,
        convergence_config_hash=adapter.convergence_config_hash,
    )


def _orbit_assessment(
    adapter: OrbitVerificationAdapter | None,
    path: tuple[tuple[float, float], ...],
    path_hash: str,
    sample: ElectronOrbitSample,
    b_low_lower: float,
    scale_length: float | None,
) -> OrbitAssessment:
    if (
        not _nonempty_text(sample.sample_id)
        or not isfinite(sample.kinetic_energy_ev)
        or sample.kinetic_energy_ev <= 0.0
        or sample.kinetic_energy_ev >= _ELECTRON_REST_ENERGY_EV
        or not isfinite(sample.pitch_angle_rad)
        or not 0.0 <= sample.pitch_angle_rad <= 1.5707963267948966
        or not isfinite(sample.maximum_rho_over_scale)
        or sample.maximum_rho_over_scale <= 0.0
        or not isfinite(sample.maximum_mu_relative_variation)
        or sample.maximum_mu_relative_variation < 0.0
    ):
        raise CouplingValidationError("invalid prescribed electron orbit sample")
    if not isfinite(b_low_lower) or b_low_lower <= 0.0:
        return OrbitAssessment(
            sample=sample,
            path_hash=path_hash,
            rho_over_scale=None,
            maximum_mu_relative_variation=None,
            adapter_id=None,
            adapter_version=None,
            adapter_code_hash=None,
            orbit_model_id=None,
            orbit_model_version=None,
            orbit_code_hash=None,
            orbit_config_hash=None,
            convergence_id=None,
            convergence_version=None,
            convergence_config_hash=None,
            status=V4Status.NONADIABATIC,
            reason=(
                "positive conservative B_low is required for gyroradius ordering"
            ),
        )
    perpendicular_j = (
        sample.kinetic_energy_ev
        * _ELEMENTARY_CHARGE_C
        * sin(sample.pitch_angle_rad) ** 2
    )
    rho = sqrt(2.0 * _ELECTRON_MASS_KG * perpendicular_j) / (
        _ELEMENTARY_CHARGE_C * b_low_lower
    )
    rho_ratio = 0.0 if scale_length is None else rho / scale_length
    if not isfinite(rho_ratio) or rho_ratio > sample.maximum_rho_over_scale:
        return OrbitAssessment(
            sample=sample,
            path_hash=path_hash,
            rho_over_scale=rho_ratio if isfinite(rho_ratio) else None,
            maximum_mu_relative_variation=None,
            adapter_id=None,
            adapter_version=None,
            adapter_code_hash=None,
            orbit_model_id=None,
            orbit_model_version=None,
            orbit_code_hash=None,
            orbit_config_hash=None,
            convergence_id=None,
            convergence_version=None,
            convergence_config_hash=None,
            status=V4Status.NONADIABATIC,
            reason="rho_e/L_B exceeds the preregistered limit",
        )
    if adapter is None:
        return OrbitAssessment(
            sample=sample,
            path_hash=path_hash,
            rho_over_scale=rho_ratio,
            maximum_mu_relative_variation=None,
            adapter_id=None,
            adapter_version=None,
            adapter_code_hash=None,
            orbit_model_id=None,
            orbit_model_version=None,
            orbit_code_hash=None,
            orbit_config_hash=None,
            convergence_id=None,
            convergence_version=None,
            convergence_config_hash=None,
            status=V4Status.ORBIT_UNVERIFIED,
            reason="orbit verification adapter is required",
        )
    identity = _orbit_identity(adapter)
    assert identity is not None
    claims = adapter.verify_orbit(path, path_hash, sample)
    if not isinstance(claims, OrbitVerificationClaims):
        raise EvidenceVerificationError("orbit adapter returned invalid claims")
    if claims.path_hash != path_hash or claims.sample_id != sample.sample_id:
        raise EvidenceVerificationError("orbit claims are not bound to path/sample")
    claimed_identity = OrbitVerificationIdentity(
        claims.adapter_id,
        claims.adapter_version,
        claims.adapter_code_hash,
        claims.orbit_model_id,
        claims.orbit_model_version,
        claims.orbit_code_hash,
        claims.orbit_config_hash,
        claims.convergence_id,
        claims.convergence_version,
        claims.convergence_config_hash,
    )
    if claimed_identity != identity:
        raise EvidenceVerificationError(
            "orbit claims do not match preregistered implementation identity"
        )
    variation = float(claims.maximum_mu_relative_variation)
    if claims.converged is not True or not isfinite(variation) or variation < 0.0:
        return OrbitAssessment(
            sample=sample,
            path_hash=path_hash,
            rho_over_scale=rho_ratio,
            maximum_mu_relative_variation=None,
            adapter_id=identity.adapter_id,
            adapter_version=identity.adapter_version,
            adapter_code_hash=identity.adapter_code_hash,
            orbit_model_id=identity.orbit_model_id,
            orbit_model_version=identity.orbit_model_version,
            orbit_code_hash=identity.orbit_code_hash,
            orbit_config_hash=identity.orbit_config_hash,
            convergence_id=identity.convergence_id,
            convergence_version=identity.convergence_version,
            convergence_config_hash=identity.convergence_config_hash,
            status=V4Status.ORBIT_UNVERIFIED,
            reason="orbit evidence is nonconverged or nonfinite",
        )
    status = (
        V4Status.RESOLVED
        if variation <= sample.maximum_mu_relative_variation
        else V4Status.NONADIABATIC
    )
    return OrbitAssessment(
        sample=sample,
        path_hash=path_hash,
        rho_over_scale=rho_ratio,
        maximum_mu_relative_variation=variation,
        adapter_id=identity.adapter_id,
        adapter_version=identity.adapter_version,
        adapter_code_hash=identity.adapter_code_hash,
        orbit_model_id=identity.orbit_model_id,
        orbit_model_version=identity.orbit_model_version,
        orbit_code_hash=identity.orbit_code_hash,
        orbit_config_hash=identity.orbit_config_hash,
        convergence_id=identity.convergence_id,
        convergence_version=identity.convergence_version,
        convergence_config_hash=identity.convergence_config_hash,
        status=status,
        reason=(
            "orbit and adiabatic-invariant gates passed"
            if status is V4Status.RESOLVED
            else "adiabatic-invariant variation exceeds the preregistered limit"
        ),
    )


def _trace_path(
    snapshot: _V3Snapshot,
    seed: FieldLineSeed,
    direction: int,
    geometry: CFTGeometry,
    trace_policy: FieldLineTracePolicy,
    uncertainty: UncertaintyModel,
    orbit_adapter: OrbitVerificationAdapter | None,
    minimum_endpoint_high_field_fraction: float,
) -> WallConnectedMirrorPath:
    field = snapshot.field_map
    points = [(seed.r_m, seed.z_m)]
    termination = "maximum_steps"
    endpoint_error = None
    for _ in range(trace_policy.maximum_steps):
        current = points[-1]
        try:
            local_direction = _unit_field(snapshot, current, direction)
        except CouplingValidationError:
            termination = "magnetic_null"
            break
        distance_to_wall = geometry.channel_wall_radius_m - current[0]
        radial_speed = local_direction[0]
        remaining_length = (
            distance_to_wall / radial_speed
            if radial_speed > 0.0
            else None
        )
        if (
            remaining_length is not None
            and 0.0 <= remaining_length <= trace_policy.wall_tolerance_m
        ):
            endpoint = (
                geometry.channel_wall_radius_m,
                current[1] + remaining_length * local_direction[1],
            )
            points.append(endpoint)
            endpoint_error = remaining_length
            termination = "channel_wall"
            break
        step = trace_policy.step_m
        if (
            remaining_length is not None
            and remaining_length < 2.0 * step
        ):
            step = max(remaining_length * 0.5, trace_policy.wall_tolerance_m * 0.25)
        try:
            for _attempt in range(64):
                try:
                    candidate = _rk4_step(
                        snapshot,
                        current,
                        direction,
                        step,
                        geometry.channel_wall_radius_m,
                    )
                    break
                except _WallStageCrossing:
                    step *= 0.5
            else:
                termination = "wall_event_unresolved"
                break
        except CouplingValidationError:
            termination = "magnetic_null"
            break
        except ValueError:
            termination = "computational_boundary"
            break
        if candidate[0] >= geometry.channel_wall_radius_m:
            endpoint = (
                geometry.channel_wall_radius_m,
                candidate[1],
            )
            points.append(endpoint)
            endpoint_error = 0.0
            termination = "channel_wall"
            break
        if not (
            geometry.plasma_z_min_m < candidate[1] < geometry.plasma_z_max_m
            and candidate[0] >= field.r_m[0]
        ):
            points.append(candidate)
            termination = "plasma_axial_boundary"
            break
        if (
            candidate[0] >= field.r_m[-1]
            or candidate[1] <= field.z_m[0]
            or candidate[1] >= field.z_m[-1]
        ):
            points.append(candidate)
            termination = "computational_boundary"
            break
        points.append(candidate)
    path = tuple(points)
    length = sum(
        hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(path, path[1:])
    )
    psi_start = bilinear_sample(field, field.psi_wb, path[0])
    try:
        psi_drift = max(
            abs(bilinear_sample(field, field.psi_wb, point) - psi_start)
            for point in path
            if field.r_m[0] <= point[0] <= field.r_m[-1]
            and field.z_m[0] <= point[1] <= field.z_m[-1]
        )
    except ValueError:
        psi_drift = 0.0
    contour = FluxContour(
        psi_start, path, False, termination == "computational_boundary",
        psi_drift, 0.0, True, "open connected field-line path",
        len(path), max(0, len(path) - 1),
    )
    certificate = certify_contour_field(
        field,
        contour,
        null_floor_t=trace_policy.minimum_b_t,
        absolute_tolerance_t=trace_policy.minimum_b_t * 0.1,
        relative_tolerance=trace_policy.interpolation_relative_error,
        maximum_depth=18,
    )
    path_hash = _path_identity_hash(
        field.full_map_hash,
        seed.seed_id,
        direction,
        psi_start,
        path,
    )
    status = V4Status.RESOLVED
    reason = "wall-connected field-line path passed"
    if termination != "channel_wall":
        status, reason = V4Status.INVALID, f"path terminated at {termination}"
    elif psi_drift > trace_policy.maximum_psi_drift_wb:
        status, reason = V4Status.INVALID, "field-line psi drift exceeds tolerance"
    elif not certificate.regular or certificate.certified_b_low_lower_t <= trace_policy.minimum_b_t:
        status, reason = V4Status.NONADIABATIC, "path contains a near-null/nonregular field segment"
    elif hypot(
        bilinear_sample(field, field.b_r_t, path[-1]),
        bilinear_sample(field, field.b_z_t, path[-1]),
    ) < minimum_endpoint_high_field_fraction * certificate.sampled_b_high_lower_t:
        status, reason = V4Status.INVALID, "wall endpoint is not in the declared high-field region"
    b_low = certificate.sampled_b_low_upper_t
    b_high = certificate.sampled_b_high_lower_t
    low_index = min(
        range(len(certificate.sampled_b_t)),
        key=certificate.sampled_b_t.__getitem__,
    )
    high_index = max(
        range(len(certificate.sampled_b_t)),
        key=certificate.sampled_b_t.__getitem__,
    )
    b_low_location = certificate.sampled_points_rz_m[low_index]
    b_high_location = certificate.sampled_points_rz_m[high_index]
    scale_length = (
        None
        if certificate.maximum_gradient_t_per_m == 0.0
        else certificate.certified_b_low_lower_t / certificate.maximum_gradient_t_per_m
    )
    relative_error = (
        uncertainty.relative_independent_sigma
        + trace_policy.interpolation_relative_error
        + trace_policy.path_relative_error
    )
    interpolation_error = uncertainty.coverage_factor * (
        uncertainty.absolute_independent_sigma_t
        + uncertainty.common_mode_sigma_t
        + relative_error * b_high
    )
    low_lower = max(
        0.0,
        certificate.certified_b_low_lower_t
        - uncertainty.coverage_factor
        * (
            uncertainty.absolute_independent_sigma_t
            + uncertainty.common_mode_sigma_t
            + relative_error * b_low
        ),
    )
    high_upper = certificate.certified_b_high_upper_t + interpolation_error
    probability = lower_probability = upper_probability = None
    if any(not isfinite(value) for value in (interpolation_error, low_lower, high_upper)):
        status, reason = V4Status.INVALID, "path uncertainty bounds overflowed"
    elif status is V4Status.RESOLVED and low_lower <= trace_policy.minimum_b_t:
        status, reason = (
            V4Status.UNCERTAINTY_DOMINATED,
            "uncertainty removes the positive field lower bound",
        )
    elif status is V4Status.RESOLVED and low_lower > 0.0 and high_upper > 0.0:
        ratio = min(1.0, b_low / b_high)
        lower_ratio = min(1.0, low_lower / high_upper)
        upper_ratio = min(
            1.0,
            (b_low + interpolation_error)
            / max(trace_policy.minimum_b_t, b_high - interpolation_error),
        )
        probability = _probability(ratio)
        lower_probability = _probability(lower_ratio)
        upper_probability = _probability(upper_ratio)
        if (
            upper_probability - lower_probability
            > max(probability, 1e-300)
            * trace_policy.uncertainty_dominance_factor
        ):
            status, reason = V4Status.UNCERTAINTY_DOMINATED, "path uncertainty dominates mirror probability"
            probability = lower_probability = upper_probability = None
    else:
        probability = lower_probability = upper_probability = None
    assessments = tuple(
        _orbit_assessment(
            orbit_adapter, path, path_hash, sample, low_lower,
            scale_length,
        )
        for sample in seed.electron_samples
    )
    if not assessments:
        status, reason = V4Status.ORBIT_UNVERIFIED, "seed has no prescribed electron samples"
    elif status is V4Status.RESOLVED and any(
        item.status is not V4Status.RESOLVED for item in assessments
    ):
        failed = next(item for item in assessments if item.status is not V4Status.RESOLVED)
        status, reason = failed.status, failed.reason
        probability = lower_probability = upper_probability = None
    endpoint = path[-1] if termination == "channel_wall" else None
    return WallConnectedMirrorPath(
        seed.seed_id, direction, path, endpoint, endpoint_error,
        termination, length, psi_start,
        psi_drift, b_low, b_high, b_low_location, b_high_location,
        low_lower, high_upper, scale_length,
        interpolation_error, probability, lower_probability, upper_probability,
        path_hash, assessments, status, reason,
    )


def _axial_metrics(
    snapshot: _V3Snapshot,
    geometry: CFTGeometry,
    z_start: float,
    z_end: float,
    policy: AxialDominancePolicy,
) -> AxialDominanceMetrics:
    values: list[float] = []
    field = snapshot.field_map
    for i, radius in enumerate(field.r_m):
        if radius > geometry.core_radius_m:
            continue
        for j, z in enumerate(field.z_m):
            if not z_start < z < z_end:
                continue
            magnitude = hypot(field.b_r_t[i][j], field.b_z_t[i][j])
            if magnitude > 0.0:
                values.append(abs(field.b_z_t[i][j]) / magnitude)
    if not values:
        return AxialDominanceMetrics(0.0, 0.0, 0, False)
    mean = sum(values) / len(values)
    passing = sum(value >= policy.pointwise_axial_fraction_threshold for value in values) / len(values)
    return AxialDominanceMetrics(
        mean, passing, len(values),
        mean >= policy.minimum_mean_axial_fraction
        and passing >= policy.minimum_passing_fraction,
    )


def _islands(
    snapshot: _V3Snapshot, cell_id: str, z_start: float, z_end: float
) -> tuple[ClosedIslandDiagnostic, ...]:
    field = snapshot.field_map
    values = [
        field.psi_wb[i][j]
        for i in range(1, len(field.r_m) - 1)
        for j, z in enumerate(field.z_m)
        if z_start < z < z_end
    ]
    if not values or max(values) <= min(values):
        return ()
    result: list[ClosedIslandDiagnostic] = []
    for quantile in (0.25, 0.5, 0.75):
        level = min(values) + quantile * (max(values) - min(values))
        try:
            contours = trace_flux_contours(
                field, level, FluxSurfacePolicy(saddle_tie_policy="reject")
            )
        except Exception:
            # Islands are explicitly diagnostic and never define/gate a CFT cell.
            continue
        count = sum(
            contour.closed
            and contour.simple
            and not contour.touches_boundary
            and z_start
            < sum(point[1] for point in contour.points_rz_m)
            / len(contour.points_rz_m)
            < z_end
            for contour in contours
        )
        if count:
            result.append(ClosedIslandDiagnostic(level, count, cell_id))
    return tuple(result)


def _assess_map(
    role: str,
    snapshot: _V3Snapshot,
    geometry: CFTGeometry,
    registrations: tuple[CFTCellRegistration, ...],
    cusp_policy: WallCuspPolicy,
    trace_policy: FieldLineTracePolicy,
    axial_policy: AxialDominancePolicy,
    uncertainty: UncertaintyModel,
    orbit_adapter: OrbitVerificationAdapter | None,
) -> V4MapAssessment:
    field = snapshot.field_map
    if not (
        field.r_m[-1] > geometry.channel_wall_radius_m + trace_policy.wall_tolerance_m
        and field.z_m[0] < geometry.plasma_z_min_m
        and field.z_m[-1] > geometry.plasma_z_max_m
    ):
        raise CouplingValidationError(
            "accepted map must extend beyond the declared plasma wall/domain"
        )
    candidates = list(_detect_cusps(snapshot, geometry, cusp_policy))
    expected_cusp_count = len(registrations) + 1
    if len(candidates) != expected_cusp_count:
        return V4MapAssessment(
            role=role,
            identity=_identity(snapshot),
            validation_policy=snapshot.validation_policy,
            cusps=tuple(candidates),
            cells=(),
            status=V4Status.AMBIGUOUS,
            reason=(
                f"detected {len(candidates)} wall cusps; "
                f"preregistration requires {expected_cusp_count}"
            ),
            detected_cusp_count=len(candidates),
            expected_cusp_count=expected_cusp_count,
        )
    cells: list[CFTCell] = []
    all_endpoints: list[tuple[float, float]] = []
    for index, registration in enumerate(registrations):
        upstream, downstream = candidates[index], candidates[index + 1]
        if not _nonempty_text(registration.cell_id) or not registration.seeds:
            raise CouplingValidationError("each v4 cell requires ID and seeds")
        outcomes: list[SeedPathOutcome] = []
        for seed in registration.seeds:
            if not (
                0.0 <= seed.r_m < geometry.channel_wall_radius_m
                and upstream.z_m < seed.z_m < downstream.z_m
                and _nonempty_text(seed.seed_id)
            ):
                return V4MapAssessment(
                    role=role,
                    identity=_identity(snapshot),
                    validation_policy=snapshot.validation_policy,
                    cusps=tuple(candidates),
                    cells=tuple(cells),
                    status=V4Status.AMBIGUOUS,
                    reason=(
                        f"seed {seed.seed_id!r} lies outside detected "
                        f"cell {registration.cell_id!r}"
                    ),
                    detected_cusp_count=len(candidates),
                    expected_cusp_count=expected_cusp_count,
                )
            negative = _trace_path(
                snapshot, seed, -1, geometry, trace_policy, uncertainty, orbit_adapter,
                cusp_policy.minimum_endpoint_high_field_fraction,
            )
            positive = _trace_path(
                snapshot, seed, 1, geometry, trace_policy, uncertainty, orbit_adapter,
                cusp_policy.minimum_endpoint_high_field_fraction,
            )
            for path in (negative, positive):
                if path.wall_endpoint_rz_m is not None:
                    all_endpoints.append(path.wall_endpoint_rz_m)
            path_status = (
                V4Status.RESOLVED
                if negative.status is V4Status.RESOLVED
                and positive.status is V4Status.RESOLVED
                else next(
                    path.status
                    for path in (negative, positive)
                    if path.status is not V4Status.RESOLVED
                )
            )
            outcomes.append(
                SeedPathOutcome(
                    seed, negative, positive, path_status,
                    "both wall-connected directions passed"
                    if path_status is V4Status.RESOLVED
                    else "one or both required wall paths failed",
                )
            )
        metrics = _axial_metrics(
            snapshot, geometry, upstream.z_m, downstream.z_m, axial_policy
        )
        status = (
            V4Status.RESOLVED
            if metrics.passed
            and all(item.status is V4Status.RESOLVED for item in outcomes)
            else V4Status.AMBIGUOUS
        )
        cells.append(
            CFTCell(
                registration.cell_id, upstream.z_m, downstream.z_m,
                upstream.cusp_id, downstream.cusp_id, metrics, tuple(outcomes),
                _islands(snapshot, registration.cell_id, upstream.z_m, downstream.z_m),
                status,
                "inter-cusp cell passed axial and wall-path gates"
                if status is V4Status.RESOLVED
                else "inter-cusp cell failed axial or wall-path gate",
            )
        )
    for index, cusp in enumerate(candidates):
        endpoints = [
            point for point in all_endpoints
            if abs(point[1] - cusp.z_m) <= cusp_policy.endpoint_plane_tolerance_m
        ]
        candidates[index] = replace(
            cusp,
            bundle_endpoint_count=len(endpoints),
            stable=len(endpoints) >= cusp_policy.minimum_bundle_paths,
        )
    map_status = (
        V4Status.RESOLVED
        if all(cusp.stable for cusp in candidates)
        and all(cell.status is V4Status.RESOLVED for cell in cells)
        else V4Status.AMBIGUOUS
    )
    return V4MapAssessment(
        role=role,
        identity=_identity(snapshot),
        validation_policy=snapshot.validation_policy,
        cusps=tuple(candidates),
        cells=tuple(cells),
        status=map_status,
        reason=(
            "map passed cusp, cell, path, orbit, and axial gates"
            if map_status is V4Status.RESOLVED
            else "map failed one or more cusp, cell, path, orbit, or axial gates"
        ),
        detected_cusp_count=len(candidates),
        expected_cusp_count=expected_cusp_count,
    )


def _endpoint_map(assessment: V4MapAssessment) -> dict[tuple[str, str, int], tuple[float, float]]:
    result = {}
    for cell in assessment.cells:
        for outcome in cell.seed_outcomes:
            for path in (outcome.negative_path, outcome.positive_path):
                if path.wall_endpoint_rz_m is not None:
                    result[(cell.cell_id, outcome.seed.seed_id, path.direction)] = path.wall_endpoint_rz_m
    return result


def _matched_cusp_assignment(
    primary: V4MapAssessment,
    refined: V4MapAssessment,
    enlarged: V4MapAssessment,
    maximum_shift_m: float,
) -> tuple[tuple[int, int, int], ...]:
    result: list[tuple[int, int, int]] = []
    used_refined: set[int] = set()
    used_enlarged: set[int] = set()
    for primary_index, cusp in enumerate(primary.cusps):
        refined_choices = [
            (abs(cusp.z_m - candidate.z_m), index)
            for index, candidate in enumerate(refined.cusps)
            if index not in used_refined
        ]
        enlarged_choices = [
            (abs(cusp.z_m - candidate.z_m), index)
            for index, candidate in enumerate(enlarged.cusps)
            if index not in used_enlarged
        ]
        if not refined_choices or not enlarged_choices:
            continue
        refined_shift, refined_index = min(refined_choices)
        enlarged_shift, enlarged_index = min(enlarged_choices)
        if (
            refined_shift > maximum_shift_m
            or enlarged_shift > maximum_shift_m
        ):
            continue
        used_refined.add(refined_index)
        used_enlarged.add(enlarged_index)
        result.append((primary_index, refined_index, enlarged_index))
    return tuple(result)


def _stability(
    assessments: tuple[V4MapAssessment, V4MapAssessment, V4MapAssessment],
    policy: CFTStabilityPolicy,
) -> V4StabilityAssessment:
    primary, refined, enlarged = assessments
    cusp_counts = tuple(item.detected_cusp_count for item in assessments)
    assignment = _matched_cusp_assignment(
        primary,
        refined,
        enlarged,
        policy.maximum_cusp_shift_m,
    )
    if len(set(cusp_counts)) != 1:
        return V4StabilityAssessment(
            primary, refined, enlarged, assignment, cusp_counts, None, None,
            None, None, None, False,
            (
                "wall cusp count changes across accepted maps: "
                f"primary={cusp_counts[0]}, refined={cusp_counts[1]}, "
                f"enlarged={cusp_counts[2]}"
            ),
        )
    if any(item.status is not V4Status.RESOLVED for item in assessments):
        reasons = "; ".join(
            f"{item.role}: {item.reason}" for item in assessments
        )
        return V4StabilityAssessment(
            primary,
            refined,
            enlarged,
            assignment,
            cusp_counts,
            None,
            None,
            None,
            None,
            None,
            False,
            f"map classification failure: {reasons}",
        )
    cusp_shift = max(
        abs(reference.z_m - candidate.z_m)
        for reference, candidate in (
            *((a, b) for a, b in zip(primary.cusps, refined.cusps)),
            *((a, b) for a, b in zip(primary.cusps, enlarged.cusps)),
        )
    )
    strength_change = max(
        abs(abs(reference.wall_br_t) - abs(candidate.wall_br_t))
        / max(abs(reference.wall_br_t), 1e-300)
        for reference, candidate in (
            *((a, b) for a, b in zip(primary.cusps, refined.cusps)),
            *((a, b) for a, b in zip(primary.cusps, enlarged.cusps)),
        )
    )
    endpoint_maps = tuple(_endpoint_map(item) for item in assessments)
    keys = set(endpoint_maps[0])
    endpoint_shift = None
    if keys and all(set(item) == keys for item in endpoint_maps[1:]):
        endpoint_shift = max(
            hypot(reference[0] - candidate[0], reference[1] - candidate[1])
            for key in keys
            for reference, candidate in (
                (endpoint_maps[0][key], endpoint_maps[1][key]),
                (endpoint_maps[0][key], endpoint_maps[2][key]),
            )
        )
    cell_shift = max(
        max(abs(a.z_start_m - b.z_start_m), abs(a.z_end_m - b.z_end_m))
        for base, other in ((primary.cells, refined.cells), (primary.cells, enlarged.cells))
        for a, b in zip(base, other)
    )
    axial_change = max(
        abs(a.axial_metrics.mean_axial_fraction - b.axial_metrics.mean_axial_fraction)
        for base, other in ((primary.cells, refined.cells), (primary.cells, enlarged.cells))
        for a, b in zip(base, other)
    )
    passed = (
        cusp_shift <= policy.maximum_cusp_shift_m
        and strength_change <= policy.maximum_cusp_strength_relative_change
        and endpoint_shift is not None
        and endpoint_shift <= policy.maximum_endpoint_shift_m
        and cell_shift <= policy.maximum_cell_bound_shift_m
        and axial_change <= policy.maximum_axial_metric_change
        and all(cusp.stable for item in assessments for cusp in item.cusps)
        and all(cell.status is V4Status.RESOLVED for item in assessments for cell in item.cells)
    )
    return V4StabilityAssessment(
        primary, refined, enlarged, assignment, cusp_counts,
        cusp_shift, strength_change,
        endpoint_shift, cell_shift, axial_change, passed,
        "all cusp, endpoint, cell, and axial metrics are stable"
        if passed else "one or more v4 stability/physics gates failed",
    )


def _validate_criterion(criterion: V4Criterion) -> None:
    if criterion != V4Criterion():
        raise CouplingValidationError(
            "v4 criterion metadata is frozen; validation status is evidence-derived"
        )


def _validate_validation_registration(
    registration: HeldOutValidationRegistration,
    criterion: V4Criterion,
) -> None:
    if not isinstance(registration, HeldOutValidationRegistration):
        raise CouplingValidationError(
            "HeldOutValidationRegistration is required"
        )
    development = registration.development_manifest
    held_out = registration.held_out_manifest
    for name, manifest in (
        ("development", development),
        ("held_out", held_out),
    ):
        if (
            not _nonempty_text(manifest.manifest_id)
            or not isinstance(manifest.case_ids, tuple)
            or not isinstance(manifest.geometry_family_ids, tuple)
            or not manifest.case_ids
            or not manifest.geometry_family_ids
            or len(set(manifest.case_ids)) != len(manifest.case_ids)
            or len(set(manifest.geometry_family_ids))
            != len(manifest.geometry_family_ids)
            or any(
                not _nonempty_text(value)
                for value in (
                    *manifest.case_ids,
                    *manifest.geometry_family_ids,
                )
            )
            or manifest.manifest_hash
            != validation_set_manifest_hash(
                manifest.manifest_id,
                manifest.case_ids,
                manifest.geometry_family_ids,
            )
        ):
            raise CouplingValidationError(
                f"{name} validation manifest is invalid"
            )
    if (
        development != CFT_V4_DEVELOPMENT_MANIFEST
        or development.manifest_id != criterion.development_evidence_id
    ):
        raise CouplingValidationError(
            "development manifest is not the frozen 56-case characterization"
        )
    if (
        development.manifest_hash == held_out.manifest_hash
        or set(development.case_ids) & set(held_out.case_ids)
        or set(development.geometry_family_ids)
        & set(held_out.geometry_family_ids)
    ):
        raise CouplingValidationError(
            "held-out manifest must be disjoint from development"
        )
    if (
        not _nonempty_text(registration.evaluated_case_id)
        or not _nonempty_text(registration.evaluated_geometry_family_id)
        or not _nonempty_text(registration.validation_adapter_id)
        or not _is_sha256(registration.validation_adapter_code_hash)
        or not _is_sha256(registration.validation_code_hash)
        or not _is_sha256(registration.validation_config_hash)
        or type(registration.required_case_count) is not int
        or registration.required_case_count != len(held_out.case_ids)
        or not isinstance(registration.required_outcomes, tuple)
        or registration.required_case_count
        != len(registration.required_outcomes)
        or any(
            not _nonempty_text(item.case_id)
            or not _nonempty_text(item.geometry_family_id)
            or item.geometry_family_id
            not in held_out.geometry_family_ids
            for item in registration.required_outcomes
        )
        or len({item.case_id for item in registration.required_outcomes})
        != len(registration.required_outcomes)
        or {item.case_id for item in registration.required_outcomes}
        != set(held_out.case_ids)
        or registration.evaluated_case_id not in held_out.case_ids
        or registration.evaluated_geometry_family_id
        not in held_out.geometry_family_ids
        or not any(
            item.case_id == registration.evaluated_case_id
            and item.geometry_family_id
            == registration.evaluated_geometry_family_id
            for item in registration.required_outcomes
        )
    ):
        raise CouplingValidationError(
            "held-out evaluated/required membership is not preregistered"
        )
    if (
        not isfinite(registration.policy.maximum_age_s)
        or registration.policy.maximum_age_s <= 0.0
        or not isfinite(registration.policy.maximum_future_skew_s)
        or registration.policy.maximum_future_skew_s < 0.0
    ):
        raise CouplingValidationError(
            "held-out freshness/future-skew policy is invalid"
        )


def cft_preregistration_hash(
    *,
    geometry: CFTGeometry,
    registrations: tuple[CFTCellRegistration, ...],
    validation_registration: HeldOutValidationRegistration,
    three_map_hashes: tuple[str, str, str],
    three_map_evidence_fingerprints: tuple[str, str, str],
    orbit_identity: OrbitVerificationIdentity | None,
    cusp_policy: WallCuspPolicy = WallCuspPolicy(),
    trace_policy: FieldLineTracePolicy = FieldLineTracePolicy(),
    axial_policy: AxialDominancePolicy = AxialDominancePolicy(),
    stability_policy: CFTStabilityPolicy = CFTStabilityPolicy(),
    uncertainty_model: UncertaintyModel = UncertaintyModel(),
    criterion: V4Criterion = V4Criterion(),
) -> str:
    """Hash every frozen choice before generating held-out validation data."""

    _validate_inputs(
        geometry,
        cusp_policy,
        trace_policy,
        axial_policy,
        stability_policy,
        uncertainty_model,
    )
    _validate_registrations(registrations)
    _validate_criterion(criterion)
    _validate_validation_registration(validation_registration, criterion)
    if len(three_map_hashes) != 3 or any(
        not _is_sha256(value) for value in three_map_hashes
    ):
        raise CouplingValidationError(
            "preregistration requires three exact map hashes"
        )
    if len(three_map_evidence_fingerprints) != 3 or any(
        not _is_sha256(value)
        for value in three_map_evidence_fingerprints
    ):
        raise CouplingValidationError(
            "preregistration requires three complete evidence fingerprints"
        )
    if orbit_identity is not None:
        orbit_text = (
            orbit_identity.adapter_id,
            orbit_identity.adapter_version,
            orbit_identity.orbit_model_id,
            orbit_identity.orbit_model_version,
            orbit_identity.convergence_id,
            orbit_identity.convergence_version,
        )
        orbit_hashes = (
            orbit_identity.adapter_code_hash,
            orbit_identity.orbit_code_hash,
            orbit_identity.orbit_config_hash,
            orbit_identity.convergence_config_hash,
        )
        if any(not _nonempty_text(value) for value in orbit_text) or any(
            not _is_sha256(value) for value in orbit_hashes
        ):
            raise CouplingValidationError(
                "preregistered orbit identity is invalid"
            )
    return _hash_payload(
        b"cft-v4-held-out-preregistration",
        {
            "criterion_id": criterion.criterion_id,
            "criterion_version": criterion.criterion_version,
            "development_evidence_id": criterion.development_evidence_id,
            "validation_registration": validation_registration,
            "three_map_hashes": three_map_hashes,
            "three_map_evidence_fingerprints": (
                three_map_evidence_fingerprints
            ),
            "geometry": geometry,
            "cusp_policy": cusp_policy,
            "trace_policy": trace_policy,
            "axial_policy": axial_policy,
            "stability_policy": stability_policy,
            "uncertainty_model": uncertainty_model,
            "registrations": registrations,
            "directions": (-1, 1),
            "orbit_identity": orbit_identity,
        },
    )


def _record_hash(record: V4CouplingRecord) -> str:
    return _hash_payload(
        b"cft-coupling-record-v4",
        replace(record, record_hash=""),
    )


def build_cft_coupling_record(
    evidence: AcceptedV4MapSet,
    *,
    geometry: CFTGeometry,
    registrations: tuple[CFTCellRegistration, ...],
    validation_registration: HeldOutValidationRegistration,
    orbit_adapter: OrbitVerificationAdapter | None,
    cusp_policy: WallCuspPolicy = WallCuspPolicy(),
    trace_policy: FieldLineTracePolicy = FieldLineTracePolicy(),
    axial_policy: AxialDominancePolicy = AxialDominancePolicy(),
    stability_policy: CFTStabilityPolicy = CFTStabilityPolicy(),
    uncertainty_model: UncertaintyModel = UncertaintyModel(),
    criterion: V4Criterion = V4Criterion(),
    held_out_validation_evidence: AcceptedHeldOutValidationEvidence | None = None,
    reference_time_utc: datetime | None = None,
) -> V4CouplingRecord:
    _validate_inputs(
        geometry, cusp_policy, trace_policy, axial_policy,
        stability_policy, uncertainty_model,
    )
    _validate_registrations(registrations)
    _validate_criterion(criterion)
    _validate_validation_registration(validation_registration, criterion)
    orbit_identity = _orbit_identity(orbit_adapter)
    evaluation_time = reference_time_utc or datetime.now(timezone.utc)
    snapshots = reverify_v4_map_set(
        evidence, reference_time_utc=evaluation_time
    )
    three_map_hashes = tuple(
        snapshot.field_map.full_map_hash for snapshot in snapshots
    )
    evidence_fingerprints = v4_map_set_evidence_fingerprints(
        evidence,
        reference_time_utc=evaluation_time,
    )
    migration_manifest_hashes = tuple(
        hashlib.sha256(snapshot.migration_manifest_bytes).hexdigest()
        if snapshot.migration_manifest_bytes is not None
        else None
        for snapshot in snapshots
    )
    migration_source_artifact_hashes = tuple(
        hashlib.sha256(snapshot.migration_source_artifact_bytes).hexdigest()
        if snapshot.migration_source_artifact_bytes is not None
        else None
        for snapshot in snapshots
    )
    expected_registration_hash = cft_preregistration_hash(
        geometry=geometry,
        registrations=registrations,
        validation_registration=validation_registration,
        three_map_hashes=three_map_hashes,
        three_map_evidence_fingerprints=evidence_fingerprints,
        orbit_identity=orbit_identity,
        cusp_policy=cusp_policy,
        trace_policy=trace_policy,
        axial_policy=axial_policy,
        stability_policy=stability_policy,
        uncertainty_model=uncertainty_model,
        criterion=criterion,
    )
    validation_identity = None
    if held_out_validation_evidence is not None:
        claims, validation_identity = reverify_held_out_validation(
            held_out_validation_evidence,
            reference_time_utc=evaluation_time,
        )
        if (
            claims.criterion_id != criterion.criterion_id
            or claims.criterion_version != criterion.criterion_version
            or claims.development_manifest
            != validation_registration.development_manifest
            or claims.held_out_manifest
            != validation_registration.held_out_manifest
            or claims.evaluated_case_id
            != validation_registration.evaluated_case_id
            or claims.evaluated_geometry_family_id
            != validation_registration.evaluated_geometry_family_id
            or claims.preregistration_hash != expected_registration_hash
            or validation_identity.policy != validation_registration.policy
            or validation_identity.adapter_id
            != validation_registration.validation_adapter_id
            or validation_identity.adapter_code_hash
            != validation_registration.validation_adapter_code_hash
            or claims.validation_code_hash
            != validation_registration.validation_code_hash
            or claims.validation_config_hash
            != validation_registration.validation_config_hash
            or {
                (outcome.case_id, outcome.geometry_family_id)
                for outcome in claims.outcomes
            }
            != {
                (required.case_id, required.geometry_family_id)
                for required in validation_registration.required_outcomes
            }
        ):
            raise EvidenceVerificationError(
                "held-out evidence is not bound to this frozen criterion/preregistration"
            )
        matching_outcomes = tuple(
            outcome
            for outcome in claims.outcomes
            if outcome.case_id == validation_registration.evaluated_case_id
            and outcome.geometry_family_id
            == validation_registration.evaluated_geometry_family_id
        )
        if (
            len(matching_outcomes) != 1
            or matching_outcomes[0].three_map_hashes != three_map_hashes
            or matching_outcomes[0].three_map_evidence_fingerprints
            != evidence_fingerprints
        ):
            raise EvidenceVerificationError(
                "evaluated three-map record is not a held-out manifest member"
            )
        criterion = replace(
            criterion,
            held_out_validation_status="validated_new_geometry_family",
        )
    assessments = tuple(
        _assess_map(
            role, snapshot, geometry, registrations, cusp_policy, trace_policy,
            axial_policy, uncertainty_model, orbit_adapter,
        )
        for role, snapshot in zip(("primary", "refined", "enlarged"), snapshots)
    )
    stability = _stability(assessments, stability_policy)
    status = V4Status.RESOLVED if stability.passed else V4Status.AMBIGUOUS
    provisional = V4CouplingRecord(
        schema_version=COUPLING_V4_SCHEMA_VERSION,
        record_hash="",
        status=status,
        reason=(
            "source-backed wall cusps and inter-cusp cells passed"
            if status is V4Status.RESOLVED
            else "v4 CFT assessment failed one or more atomic gates"
        ),
        criterion=criterion,
        geometry=geometry,
        cusp_policy=cusp_policy,
        trace_policy=trace_policy,
        axial_policy=axial_policy,
        stability_policy=stability_policy,
        uncertainty_model=uncertainty_model,
        registrations=registrations,
        validation_registration=validation_registration,
        evidence_fingerprints=evidence_fingerprints,
        field_migration_manifest_hashes=migration_manifest_hashes,
        field_migration_source_artifact_hashes=(
            migration_source_artifact_hashes
        ),
        stability=stability,
        orbit_identity=orbit_identity,
        held_out_validation=validation_identity,
    )
    return replace(
        provisional,
        record_hash=_record_hash(provisional),
    )


def cft_coupling_record_dict(record: V4CouplingRecord) -> dict[str, Any]:
    result = _json_value(record)
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


@dataclass(frozen=True, slots=True)
class _AcceptedProjectionSnapshot:
    record: V4CouplingRecord
    evidence: AcceptedV4MapSet
    held_out_evidence: AcceptedHeldOutValidationEvidence
    orbit_adapter: OrbitVerificationAdapter
    accepted_at_utc: datetime


_PROJECTION_KEY = object()


class AcceptedCFTProjection:
    """Opaque authority retaining the exact artifacts needed for reverification."""

    __slots__ = ("__snapshot", "__invariant_hash")

    def __new__(
        cls,
        snapshot: _AcceptedProjectionSnapshot,
        invariant_hash: str,
        *,
        _factory_key: object | None = None,
    ) -> AcceptedCFTProjection:
        if _factory_key is not _PROJECTION_KEY:
            raise TypeError("use accept_cft_projection")
        instance = super().__new__(cls)
        object.__setattr__(
            instance,
            "_AcceptedCFTProjection__snapshot",
            snapshot,
        )
        object.__setattr__(
            instance,
            "_AcceptedCFTProjection__invariant_hash",
            invariant_hash,
        )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AcceptedCFTProjection is immutable")

    def _components(
        self,
        *,
        _factory_key: object,
    ) -> tuple[_AcceptedProjectionSnapshot, str]:
        if _factory_key is not _PROJECTION_KEY:
            raise TypeError("accepted projection is private")
        return self.__snapshot, self.__invariant_hash


def _projection_invariant(snapshot: _AcceptedProjectionSnapshot) -> str:
    return _hash_payload(
        b"cft-v4-accepted-projection",
        {
            "record": snapshot.record,
            "accepted_at_utc": snapshot.accepted_at_utc,
        },
    )


def _rebuild_projection(
    snapshot: _AcceptedProjectionSnapshot,
    evaluation_time: datetime,
) -> V4CouplingRecord:
    record = snapshot.record
    return build_cft_coupling_record(
        snapshot.evidence,
        geometry=record.geometry,
        registrations=record.registrations,
        validation_registration=record.validation_registration,
        orbit_adapter=snapshot.orbit_adapter,
        cusp_policy=record.cusp_policy,
        trace_policy=record.trace_policy,
        axial_policy=record.axial_policy,
        stability_policy=record.stability_policy,
        uncertainty_model=record.uncertainty_model,
        held_out_validation_evidence=snapshot.held_out_evidence,
        reference_time_utc=evaluation_time,
    )


def _valid_diagnostics(value: object) -> bool:
    try:
        return bool(
            value.converged is True
            and all(
                isfinite(item)
                for item in (
                    value.residual_norm,
                    value.residual_tolerance,
                    value.relative_residual,
                    value.relative_tolerance,
                )
            )
            and value.residual_norm >= 0.0
            and value.residual_tolerance >= 0.0
            and value.residual_norm <= value.residual_tolerance
            and value.relative_residual >= 0.0
            and value.relative_tolerance >= 0.0
            and value.relative_residual <= value.relative_tolerance
            and type(value.iterations) is int
            and value.iterations >= 0
        )
    except (AttributeError, TypeError):
        return False


def _point_on_path(
    point: tuple[float, float],
    path: tuple[tuple[float, float], ...],
    tolerance_m: float,
) -> bool:
    if len(path) < 2 or not all(isfinite(value) for value in point):
        return False
    px, py = point
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        parameter = (
            0.0
            if denominator == 0.0
            else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
        )
        if hypot(px - (ax + parameter * dx), py - (ay + parameter * dy)) <= tolerance_m:
            return True
    return False


def _projection_record_is_complete(record: V4CouplingRecord) -> bool:
    if (
        record.status is not V4Status.RESOLVED
        or not record.stability.passed
        or record.orbit_identity is None
        or record.held_out_validation is None
        or len(record.evidence_fingerprints) != 3
        or any(not _is_sha256(value) for value in record.evidence_fingerprints)
        or len(record.field_migration_manifest_hashes) != 3
        or len(record.field_migration_source_artifact_hashes) != 3
        or any(
            value is not None and not _is_sha256(value)
            for value in (
                *record.field_migration_manifest_hashes,
                *record.field_migration_source_artifact_hashes,
            )
        )
        or any(
            (manifest is None) != (source is None)
            for manifest, source in zip(
                record.field_migration_manifest_hashes,
                record.field_migration_source_artifact_hashes,
                strict=True,
            )
        )
        or not isfinite(record.uncertainty_model.coverage_factor)
        or record.uncertainty_model.coverage_factor <= 0.0
        or not _valid_diagnostics(record.held_out_validation.diagnostics)
    ):
        return False
    identity_tuple = (
        record.orbit_identity.adapter_id,
        record.orbit_identity.adapter_version,
        record.orbit_identity.adapter_code_hash,
        record.orbit_identity.orbit_model_id,
        record.orbit_identity.orbit_model_version,
        record.orbit_identity.orbit_code_hash,
        record.orbit_identity.orbit_config_hash,
        record.orbit_identity.convergence_id,
        record.orbit_identity.convergence_version,
        record.orbit_identity.convergence_config_hash,
    )
    assessments = (
        record.stability.primary,
        record.stability.refined,
        record.stability.enlarged,
    )
    for assessment, registration in (
        (assessment, registration)
        for assessment in assessments
        for registration in (record.registrations,)
    ):
        if (
            assessment.status is not V4Status.RESOLVED
            or assessment.detected_cusp_count != assessment.expected_cusp_count
            or len(assessment.cusps) != assessment.expected_cusp_count
            or len(assessment.cells) != len(registration)
            or not _valid_diagnostics(assessment.identity.diagnostics)
            or any(not cusp.stable for cusp in assessment.cusps)
        ):
            return False
        for cell, cell_registration in zip(
            assessment.cells,
            registration,
            strict=True,
        ):
            if (
                cell.cell_id != cell_registration.cell_id
                or cell.status is not V4Status.RESOLVED
                or not cell.axial_metrics.passed
                or len(cell.seed_outcomes) != len(cell_registration.seeds)
            ):
                return False
            for outcome, seed in zip(
                cell.seed_outcomes,
                cell_registration.seeds,
                strict=True,
            ):
                if (
                    outcome.seed != seed
                    or outcome.status is not V4Status.RESOLVED
                ):
                    return False
                for path, direction in (
                    (outcome.negative_path, -1),
                    (outcome.positive_path, 1),
                ):
                    values = (
                        path.path_length_m,
                        path.psi_start_wb,
                        path.maximum_psi_drift_wb,
                        path.b_low_t,
                        path.b_high_t,
                        path.b_low_lower_t,
                        path.b_high_upper_t,
                        path.interpolation_error_t,
                        path.wall_endpoint_error_m,
                        path.mirror_probability,
                        path.probability_lower,
                        path.probability_upper,
                    )
                    if (
                        path.seed_id != seed.seed_id
                        or path.direction != direction
                        or path.status is not V4Status.RESOLVED
                        or path.termination != "channel_wall"
                        or path.wall_endpoint_rz_m is None
                        or any(value is None or not isfinite(value) for value in values)
                        or path.wall_endpoint_error_m < 0.0
                        or path.wall_endpoint_error_m
                        > record.trace_policy.wall_tolerance_m
                        or abs(
                            path.wall_endpoint_rz_m[0]
                            - record.geometry.channel_wall_radius_m
                        )
                        > record.trace_policy.wall_tolerance_m
                        or hypot(
                            path.points_rz_m[-1][0] - path.wall_endpoint_rz_m[0],
                            path.points_rz_m[-1][1] - path.wall_endpoint_rz_m[1],
                        )
                        > record.trace_policy.wall_tolerance_m
                        or path.maximum_psi_drift_wb
                        > record.trace_policy.maximum_psi_drift_wb
                        or not (
                            0.0
                            < path.b_low_lower_t
                            <= path.b_low_t
                            <= path.b_high_t
                            <= path.b_high_upper_t
                        )
                        or not (
                            0.0
                            <= path.probability_lower
                            <= path.mirror_probability
                            <= path.probability_upper
                            <= 1.0
                        )
                        or path.path_hash
                        != _path_identity_hash(
                            assessment.identity.full_map_hash,
                            seed.seed_id,
                            direction,
                            path.psi_start_wb,
                            path.points_rz_m,
                        )
                        or not _point_on_path(
                            path.b_low_location_rz_m,
                            path.points_rz_m,
                            record.trace_policy.wall_tolerance_m,
                        )
                        or not _point_on_path(
                            path.b_high_location_rz_m,
                            path.points_rz_m,
                            record.trace_policy.wall_tolerance_m,
                        )
                        or len(path.orbit_assessments)
                        != len(seed.electron_samples)
                    ):
                        return False
                    for orbit, sample in zip(
                        path.orbit_assessments,
                        seed.electron_samples,
                        strict=True,
                    ):
                        orbit_identity = (
                            orbit.adapter_id,
                            orbit.adapter_version,
                            orbit.adapter_code_hash,
                            orbit.orbit_model_id,
                            orbit.orbit_model_version,
                            orbit.orbit_code_hash,
                            orbit.orbit_config_hash,
                            orbit.convergence_id,
                            orbit.convergence_version,
                            orbit.convergence_config_hash,
                        )
                        if (
                            orbit.sample != sample
                            or orbit.path_hash != path.path_hash
                            or orbit.status is not V4Status.RESOLVED
                            or orbit_identity != identity_tuple
                            or orbit.rho_over_scale is None
                            or not isfinite(orbit.rho_over_scale)
                            or orbit.rho_over_scale
                            > sample.maximum_rho_over_scale
                            or orbit.maximum_mu_relative_variation is None
                            or not isfinite(orbit.maximum_mu_relative_variation)
                            or orbit.maximum_mu_relative_variation
                            > sample.maximum_mu_relative_variation
                        ):
                            return False
    return True


def accept_cft_projection(
    record: V4CouplingRecord,
    evidence: AcceptedV4MapSet,
    *,
    held_out_validation_evidence: AcceptedHeldOutValidationEvidence,
    orbit_adapter: OrbitVerificationAdapter,
    reference_time_utc: datetime,
) -> AcceptedCFTProjection:
    """Accept only a record reproducible from exact private source evidence."""

    if (
        reference_time_utc.tzinfo is None
        or reference_time_utc.utcoffset() is None
    ):
        raise CouplingValidationError(
            "projection acceptance requires an explicit timezone-aware clock"
        )
    snapshot = _AcceptedProjectionSnapshot(
        record=record,
        evidence=evidence,
        held_out_evidence=held_out_validation_evidence,
        orbit_adapter=orbit_adapter,
        accepted_at_utc=reference_time_utc,
    )
    rebuilt = _rebuild_projection(snapshot, reference_time_utc)
    if rebuilt != record or not _projection_record_is_complete(rebuilt):
        raise EvidenceVerificationError(
            "record is not reproducible as a complete accepted projection"
        )
    authoritative = replace(snapshot, record=rebuilt)
    return AcceptedCFTProjection(
        authoritative,
        _projection_invariant(authoritative),
        _factory_key=_PROJECTION_KEY,
    )


def cft_solver_inputs(
    accepted: AcceptedCFTProjection,
    *,
    reference_time_utc: datetime,
) -> tuple[dict[str, Any], ...]:
    """Rebuild from exact artifacts, then project complete held-out evidence."""

    if not isinstance(accepted, AcceptedCFTProjection):
        return ()
    try:
        if (
            reference_time_utc.tzinfo is None
            or reference_time_utc.utcoffset() is None
        ):
            return ()
        snapshot, invariant_hash = accepted._components(
            _factory_key=_PROJECTION_KEY
        )
        if _projection_invariant(snapshot) != invariant_hash:
            return ()
        record = _rebuild_projection(snapshot, reference_time_utc)
        if (
            record != snapshot.record
            or not _projection_record_is_complete(record)
            or record.schema_version != COUPLING_V4_SCHEMA_VERSION
            or record.criterion.held_out_validation_status
            != "validated_new_geometry_family"
            or record.held_out_validation is None
        ):
            return ()
        canonical_hash = _record_hash(record)
        map_hashes = (
            record.stability.primary.identity.full_map_hash,
            record.stability.refined.identity.full_map_hash,
            record.stability.enlarged.identity.full_map_hash,
        )
        expected_preregistration_hash = cft_preregistration_hash(
            geometry=record.geometry,
            registrations=record.registrations,
            validation_registration=record.validation_registration,
            three_map_hashes=map_hashes,
            three_map_evidence_fingerprints=record.evidence_fingerprints,
            orbit_identity=record.orbit_identity,
            cusp_policy=record.cusp_policy,
            trace_policy=record.trace_policy,
            axial_policy=record.axial_policy,
            stability_policy=record.stability_policy,
            uncertainty_model=record.uncertainty_model,
            criterion=replace(
                record.criterion,
                held_out_validation_status="awaiting_new_geometry_family",
            ),
        )
    except (CouplingValidationError, TypeError, ValueError, OverflowError):
        return ()
    held_out = record.held_out_validation
    evaluation_time = reference_time_utc
    age = (
        evaluation_time.astimezone(timezone.utc)
        - held_out.generated_at_utc.astimezone(timezone.utc)
    ).total_seconds()
    matching_outcomes = tuple(
        outcome
        for outcome in held_out.outcomes
        if outcome.case_id
        == record.validation_registration.evaluated_case_id
        and outcome.geometry_family_id
        == record.validation_registration.evaluated_geometry_family_id
    )
    if (
        record.record_hash != canonical_hash
        or not record.stability.passed
        or record.held_out_validation.criterion_id
        != record.criterion.criterion_id
        or record.held_out_validation.criterion_version
        != record.criterion.criterion_version
        or held_out.development_manifest
        != record.validation_registration.development_manifest
        or held_out.held_out_manifest
        != record.validation_registration.held_out_manifest
        or held_out.evaluated_case_id
        != record.validation_registration.evaluated_case_id
        or held_out.evaluated_geometry_family_id
        != record.validation_registration.evaluated_geometry_family_id
        or held_out.preregistration_hash
        != expected_preregistration_hash
        or held_out.policy != record.validation_registration.policy
        or held_out.adapter_id
        != record.validation_registration.validation_adapter_id
        or held_out.adapter_code_hash
        != record.validation_registration.validation_adapter_code_hash
        or held_out.validation_code_hash
        != record.validation_registration.validation_code_hash
        or held_out.validation_config_hash
        != record.validation_registration.validation_config_hash
        or record.validation_registration.required_case_count
        != len(held_out.outcomes)
        or {
            (outcome.case_id, outcome.geometry_family_id)
            for outcome in held_out.outcomes
        }
        != {
            (required.case_id, required.geometry_family_id)
            for required in record.validation_registration.required_outcomes
        }
        or any(outcome.passed is not True for outcome in held_out.outcomes)
        or len(matching_outcomes) != 1
        or matching_outcomes[0].three_map_hashes != map_hashes
        or matching_outcomes[0].three_map_evidence_fingerprints
        != record.evidence_fingerprints
        or age < -held_out.policy.maximum_future_skew_s
        or age > held_out.policy.maximum_age_s
        or record.orbit_identity is None
        or any(
            not cusp.stable
            for assessment in (
                record.stability.primary,
                record.stability.refined,
                record.stability.enlarged,
            )
            for cusp in assessment.cusps
        )
        or any(
            cell.status is not V4Status.RESOLVED
            for assessment in (
                record.stability.primary,
                record.stability.refined,
                record.stability.enlarged,
            )
            for cell in assessment.cells
        )
        or any(
            (
                item.adapter_id,
                item.adapter_version,
                item.adapter_code_hash,
                item.orbit_model_id,
                item.orbit_model_version,
                item.orbit_code_hash,
                item.orbit_config_hash,
                item.convergence_id,
                item.convergence_version,
                item.convergence_config_hash,
            )
            != (
                record.orbit_identity.adapter_id,
                record.orbit_identity.adapter_version,
                record.orbit_identity.adapter_code_hash,
                record.orbit_identity.orbit_model_id,
                record.orbit_identity.orbit_model_version,
                record.orbit_identity.orbit_code_hash,
                record.orbit_identity.orbit_config_hash,
                record.orbit_identity.convergence_id,
                record.orbit_identity.convergence_version,
                record.orbit_identity.convergence_config_hash,
            )
            for assessment in (
                record.stability.primary,
                record.stability.refined,
                record.stability.enlarged,
            )
            for cell in assessment.cells
            for outcome in cell.seed_outcomes
            for path in (outcome.negative_path, outcome.positive_path)
            for item in path.orbit_assessments
        )
    ):
        return ()
    identity = record.stability.primary.identity
    return tuple(
        {
            "schema_version": record.schema_version,
            "record_hash": record.record_hash,
            "criterion_id": record.criterion.criterion_id,
            "criterion_version": record.criterion.criterion_version,
            "cell_id": cell.cell_id,
            "z_start_m": cell.z_start_m,
            "z_end_m": cell.z_end_m,
            "mean_axial_fraction": cell.axial_metrics.mean_axial_fraction,
            "axial_passing_fraction": cell.axial_metrics.passing_fraction,
            "seed_id": outcome.seed.seed_id,
            "direction": path.direction,
            "path_hash": path.path_hash,
            "path_status": path.status.value,
            "path_termination": path.termination,
            "wall_endpoint_rz_m": path.wall_endpoint_rz_m,
            "wall_endpoint_error_m": path.wall_endpoint_error_m,
            "b_low_t": path.b_low_t,
            "b_high_t": path.b_high_t,
            "b_low_lower_t": path.b_low_lower_t,
            "b_high_upper_t": path.b_high_upper_t,
            "b_low_location_rz_m": path.b_low_location_rz_m,
            "b_high_location_rz_m": path.b_high_location_rz_m,
            "field_scale_length_m": path.field_scale_length_m,
            "mirror_probability": path.mirror_probability,
            "mirror_probability_lower": path.probability_lower,
            "mirror_probability_upper": path.probability_upper,
            "orbit_assessments": tuple(
                {
                    "sample_id": item.sample.sample_id,
                    "rho_over_scale": item.rho_over_scale,
                    "maximum_mu_relative_variation": (
                        item.maximum_mu_relative_variation
                    ),
                    "adapter_id": item.adapter_id,
                    "adapter_version": item.adapter_version,
                    "adapter_code_hash": item.adapter_code_hash,
                    "orbit_model_id": item.orbit_model_id,
                    "orbit_model_version": item.orbit_model_version,
                    "orbit_code_hash": item.orbit_code_hash,
                    "orbit_config_hash": item.orbit_config_hash,
                    "convergence_id": item.convergence_id,
                    "convergence_version": item.convergence_version,
                    "convergence_config_hash": item.convergence_config_hash,
                }
                for item in path.orbit_assessments
            ),
            "artifact_hash": identity.artifact_hash,
            "source_hash": identity.source_hash,
            "geometry_hash": identity.geometry_hash,
            "material_hash": identity.material_hash,
            "field_model_id": identity.field_model_id,
            "field_model_hash": identity.field_model_hash,
            "code_hash": identity.code_hash,
            "config_hash": identity.config_hash,
            "backend_id": identity.backend_id,
            "backend_version": identity.backend_version,
            "adapter_id": identity.adapter_id,
            "adapter_code_hash": identity.adapter_code_hash,
            "primary_map_hash": identity.full_map_hash,
            "refined_map_hash": (
                record.stability.refined.identity.full_map_hash
            ),
            "enlarged_map_hash": (
                record.stability.enlarged.identity.full_map_hash
            ),
            "primary_evidence_fingerprint": record.evidence_fingerprints[0],
            "refined_evidence_fingerprint": record.evidence_fingerprints[1],
            "enlarged_evidence_fingerprint": record.evidence_fingerprints[2],
            "field_migration_manifest_hashes": (
                record.field_migration_manifest_hashes
            ),
            "field_migration_source_artifact_hashes": (
                record.field_migration_source_artifact_hashes
            ),
            "projection_evaluated_at_utc": evaluation_time.astimezone(
                timezone.utc
            ).isoformat(),
            "held_out_manifest_id": held_out.held_out_manifest.manifest_id,
            "held_out_manifest_hash": held_out.held_out_manifest.manifest_hash,
            "held_out_case_id": held_out.evaluated_case_id,
            "held_out_geometry_family_id": (
                held_out.evaluated_geometry_family_id
            ),
            "held_out_preregistration_hash": held_out.preregistration_hash,
            "held_out_validation_artifact_hash": (
                held_out.validation_artifact_hash
            ),
            "held_out_validation_code_hash": held_out.validation_code_hash,
            "held_out_validation_config_hash": held_out.validation_config_hash,
            "held_out_validation_adapter_id": held_out.adapter_id,
            "held_out_validation_adapter_code_hash": (
                held_out.adapter_code_hash
            ),
        }
        for cell in record.stability.primary.cells
        for outcome in cell.seed_outcomes
        for path in (outcome.negative_path, outcome.positive_path)
    )
