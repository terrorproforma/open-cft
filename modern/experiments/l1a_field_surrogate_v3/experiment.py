"""Geometry, staged L1a solves, transformed scalars and weighted residual POD."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    SolverConfig,
    solve_problem_warp,
)
from cft_revival.fields.numerics import current_density_grid, source_discretization_diagnostics
from cft_revival.fields.warp_solver import solve_current_density_warp
from cft_revival.geometry import (
    EvidenceNote,
    GeometryValidationError,
    PPMStackParameters,
    compute_descriptors,
    generate_twt_inspired_ppm_stack,
    to_l1a_current_equivalent_preview,
)
from cft_revival.optimization.sampling import initial_designs
from experiments.l1a_geometry_sweep_v2.experiment import (
    BuiltCase,
    VARIABLES,
    _material_registry,
    _stable_pitch,
)
from experiments.l1a_field_surrogate_v1.experiment import sample_designs as v1_designs
from experiments.l1a_field_surrogate_v2.experiment import raw_designs as v2_raw_designs

from .protocol import PROTOCOL, PROTOCOL_HASH, canonical_hash, percentile

QOIS = tuple(PROTOCOL["scalar_model"]["qois"])
INPUT_NAMES = tuple(variable.name for variable in VARIABLES)
HIGH_DOMAIN = AxisymmetricDomain(
    **PROTOCOL["fidelities"]["domain"],
    radial_intervals=PROTOCOL["fidelities"]["high"]["radial_intervals"],
    axial_intervals=PROTOCOL["fidelities"]["high"]["axial_intervals"],
)
LOW_DOMAIN = AxisymmetricDomain(
    **PROTOCOL["fidelities"]["domain"],
    radial_intervals=PROTOCOL["fidelities"]["low"]["radial_intervals"],
    axial_intervals=PROTOCOL["fidelities"]["low"]["axial_intervals"],
)
SOLVER = SolverConfig(**PROTOCOL["fidelities"]["solver"])
MU0 = 1.2566370614359173e-6


def raw_designs() -> tuple[Any, ...]:
    designs = initial_designs(
        VARIABLES,
        PROTOCOL["sampling"]["raw_rows"],
        seed=PROTOCOL["sampling"]["seed"],
        include_boundary_challenges=False,
    )
    prior = {tuple(item.values) for item in v1_designs()}
    prior.update(tuple(item.values) for item in v2_raw_designs())
    if prior.intersection(tuple(item.values) for item in designs):
        raise RuntimeError("v3 raw candidates overlap v1/v2 role-coordinate evidence")
    return designs


def design_row(design: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in design.values)


def construct_geometry(design: Any, raw_index: int) -> tuple[BuiltCase, dict[str, Any]]:
    values = dict(zip(INPUT_NAMES, design.values, strict=True))
    stage_count = min(5, 3 + int(values["stage_count_selector"] * 3.0))
    pitch = _stable_pitch(values["stage_pitch_m"], stage_count)
    chamber_length = stage_count * pitch
    requested_exit_length = chamber_length * values["exit_length_fraction"]
    exit_length = 0.0 if requested_exit_length < 0.00025 else requested_exit_length
    chamber_radius = values["chamber_outer_radius_m"]
    dielectric = values["dielectric_thickness_m"]
    requested_radius = (
        chamber_radius
        if exit_length == 0.0
        else chamber_radius * (1.15 + 0.35 * values["exit_expansion_descriptor"])
    )
    magnet_inner = max(chamber_radius, requested_radius) + dielectric + values["radial_clearance_m"]
    magnet_outer = magnet_inner + values["magnet_radial_thickness_m"]
    magnet_axial = pitch * values["magnet_axial_fraction"]
    centres = tuple((index + 0.5) * pitch for index in range(stage_count))
    radius = requested_radius
    trace: list[dict[str, Any]] = []
    geometry = None
    preview = None
    for step in range(PROTOCOL["geometry_preflight"]["maximum_nextafter_steps"] + 1):
        try:
            geometry = generate_twt_inspired_ppm_stack(
                PPMStackParameters(
                    config_id=f"l1a-fs-v3-raw-{raw_index:03d}",
                    title=f"L1a field-surrogate v3 raw {raw_index:03d}",
                    chamber_inner_radius_m=0.0,
                    chamber_outer_radius_m=chamber_radius,
                    chamber_length_m=chamber_length,
                    injector_length_m=0.08 * chamber_length,
                    dielectric_thickness_m=dielectric,
                    thermal_clearance_m=0.00025,
                    magnet_inner_radius_m=magnet_inner,
                    magnet_outer_radius_m=magnet_outer,
                    stage_pitch_m=pitch,
                    stage_centers_m=centres,
                    magnet_axial_thicknesses_m=(magnet_axial,) * stage_count,
                    shield_outer_radius_m=magnet_outer + 0.00075,
                    yoke_outer_radius_m=magnet_outer + 0.00175,
                    exit_length_m=exit_length,
                    exit_outer_radius_m=radius,
                    first_polarity=1 if values["first_polarity_selector"] < 0.5 else -1,
                    radial_tolerance_m=0.000025,
                    axial_tolerance_m=0.000025,
                    minimum_thickness_m=0.00025,
                    minimum_clearance_m=0.0001,
                ),
                evidence=(
                    EvidenceNote(
                        f"v3-raw-{raw_index:03d}",
                        "assumption",
                        "Fresh preregistered L1a numerical-emulation geometry.",
                        "L1a field-surrogate v3 protocol.",
                    ),
                ),
            )
            preview = to_l1a_current_equivalent_preview(
                geometry,
                material_registry=_material_registry(geometry),
                radial_smear_thickness_m=PROTOCOL["fidelities"]["physical_source_radial_smear_m"],
            )
            break
        except GeometryValidationError as error:
            trace.append({"step": step, "radius_m": radius, "reason": str(error)})
            if exit_length == 0.0 or not str(error).startswith("divergent wall "):
                raise
            radius = math.nextafter(radius, chamber_radius)
    if geometry is None or preview is None:
        raise GeometryValidationError(
            "nextafter sequence exhausted: " + (trace[-1]["reason"] if trace else "unknown")
        )
    bands = tuple(
        replace(band, ampere_turns_a=band.ampere_turns_a * values["source_strength_scale"])
        for band in preview.bands
    )
    problem = AxisymmetricProblem(f"l1a-fs-v3-{raw_index:03d}", HIGH_DOMAIN, bands)
    source_payload = {
        **preview.to_dict(),
        "source_strength_scale": values["source_strength_scale"],
        "scaled_bands": [asdict(band) for band in bands],
    }
    source_hash = canonical_hash(source_payload)
    config_hash = canonical_hash(
        {
            "protocol_hash": PROTOCOL_HASH,
            "design_id": design.design_id,
            "fidelities": PROTOCOL["fidelities"],
            "gates": PROTOCOL["gates"],
        }
    )
    case_hash = canonical_hash(
        {
            "geometry_sha256": geometry.canonical_sha256,
            "source_sha256": source_hash,
            "config_sha256": config_hash,
        }
    )
    case = BuiltCase(
        f"l1a-fs-v3-{raw_index:03d}",
        design,
        geometry,
        preview,
        problem,
        geometry.canonical_sha256,
        source_hash,
        config_hash,
        case_hash,
        {
            "stage_count": stage_count,
            "stage_centers_m": list(centres),
            "represented_stage_pitch_m": pitch,
            "chamber_length_m": chamber_length,
            "requested_exit_outer_radius_m": requested_radius,
            "represented_exit_outer_radius_m": radius,
            "geometry_descriptors": compute_descriptors(geometry).to_dict(),
        },
    )
    return case, {
        "raw_index": raw_index,
        "design_id": design.design_id,
        "valid": True,
        "attempt_count": len(trace) + 1,
        "rejection_trace": trace,
        "geometry_sha256": geometry.canonical_sha256,
        "preview_sha256": canonical_hash(preview.to_dict()),
        "source_sha256": source_hash,
        "requested_exit_outer_radius_m": requested_radius,
        "represented_exit_outer_radius_m": radius,
    }


def preflight_candidates() -> tuple[list[dict[str, Any]], dict[int, BuiltCase]]:
    records: list[dict[str, Any]] = []
    valid: dict[int, BuiltCase] = {}
    for index, design in enumerate(raw_designs()):
        try:
            case, record = construct_geometry(design, index)
            records.append(record)
            valid[index] = case
        except Exception as error:
            records.append(
                {
                    "raw_index": index,
                    "design_id": design.design_id,
                    "valid": False,
                    "rejection_type": type(error).__name__,
                    "rejection_reason": str(error),
                }
            )
    return records, valid


def select_frozen(valid: Mapping[int, BuiltCase]) -> tuple[int, ...]:
    ordered = sorted(valid)
    if len(ordered) < 240:
        raise RuntimeError("fewer than 240 geometry-valid raw candidates")
    fixed = ordered[:144]
    pool = ordered[144:]

    def normalized(index: int) -> tuple[float, ...]:
        return tuple(
            (value - variable.lower) / (variable.upper - variable.lower)
            for value, variable in zip(valid[index].design.values, VARIABLES, strict=True)
        )

    candidate = [normalized(index) for index in fixed[:128]]

    def take_strata(available: list[int]) -> tuple[list[int], list[int]]:
        boundary = sorted(
            available,
            key=lambda index: (
                min(min(value, 1.0 - value) for value in normalized(index)),
                valid[index].design.design_id,
            ),
        )[:16]
        remainder = [index for index in available if index not in boundary]
        ood = sorted(
            remainder,
            key=lambda index: (
                -min(math.dist(normalized(index), row) for row in candidate),
                valid[index].design.design_id,
            ),
        )[:16]
        excluded = set(boundary) | set(ood)
        interpolation = [index for index in available if index not in excluded][:16]
        selected = interpolation + boundary + ood
        return selected, [index for index in available if index not in set(selected)]

    calibration, remaining = take_strata(pool)
    assessment, _ = take_strata(remaining)
    frozen = tuple(fixed + calibration + assessment)
    if len(frozen) != 240 or len(set(frozen)) != 240:
        raise RuntimeError("frozen role selection is not 240 unique rows")
    return frozen


def rebuild_frozen(indices: Sequence[int]) -> tuple[dict[int, BuiltCase], list[dict[str, Any]]]:
    designs = raw_designs()
    cases = {}
    records = []
    for frozen_index, raw_index in enumerate(indices):
        case, record = construct_geometry(designs[raw_index], raw_index)
        cases[frozen_index] = case
        records.append(
            {
                "frozen_index": frozen_index,
                "raw_index": raw_index,
                "geometry_sha256": case.geometry_sha256,
                "preview_sha256": record["preview_sha256"],
                "source_sha256": case.source_sha256,
            }
        )
    return cases, records


def role_indices(name: str) -> tuple[int, ...]:
    return tuple(range(*PROTOCOL["sampling"]["roles"][name]))


def stratum_indices(role: str) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(range(*bounds))
        for name, bounds in PROTOCOL["sampling"][f"{role}_strata"].items()
    }


def _fake_problem(case: BuiltCase, domain: AxisymmetricDomain) -> Any:
    return SimpleNamespace(domain=domain, sources=case.problem.sources)


def solve_fidelity(case: BuiltCase, fidelity: str) -> Any:
    if fidelity == "high":
        return solve_problem_warp(case.problem, device=PROTOCOL["execution"]["device"], config=SOLVER)
    if fidelity != "low":
        raise ValueError("fidelity must be low or high")
    fake = _fake_problem(case, LOW_DOMAIN)
    source = current_density_grid(fake)
    return solve_current_density_warp(
        LOW_DOMAIN,
        source,
        permeability_h_per_m=case.problem.permeability_h_per_m,
        device=PROTOCOL["execution"]["device"],
        config=SOLVER,
    )


def field_vector(field: Any) -> np.ndarray:
    return np.concatenate(
        (np.asarray(field.b_r_t, dtype=np.float64).ravel(), np.asarray(field.b_z_t, dtype=np.float64).ravel())
    )


def prolong_low(field: Any) -> np.ndarray:
    def nested(values: np.ndarray) -> np.ndarray:
        nr, nz = values.shape
        output = np.empty((2 * nr - 1, 2 * nz - 1), dtype=np.float64)
        output[::2, ::2] = values
        output[1::2, ::2] = 0.5 * (values[:-1] + values[1:])
        output[::2, 1::2] = 0.5 * (values[:, :-1] + values[:, 1:])
        output[1::2, 1::2] = 0.25 * (
            values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:] + values[1:, 1:]
        )
        return output

    return np.concatenate((nested(np.asarray(field.b_r_t)).ravel(), nested(np.asarray(field.b_z_t)).ravel()))


def _field_arrays(vector: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    nr, nz = HIGH_DOMAIN.shape
    values = np.asarray(vector, dtype=np.float64)
    return values[: nr * nz].reshape((nr, nz)), values[nr * nz :].reshape((nr, nz))


def field_energy(vector: Sequence[float]) -> float:
    br, bz = _field_arrays(vector)
    nr, nz = HIGH_DOMAIN.shape
    radial = np.arange(nr) * HIGH_DOMAIN.dr_m
    rw = np.ones(nr); rw[[0, -1]] = 0.5
    zw = np.ones(nz); zw[[0, -1]] = 0.5
    weighted = np.sum((br * br + bz * bz) * (radial * rw)[:, None] * zw[None, :])
    return float(math.pi * HIGH_DOMAIN.dr_m * HIGH_DOMAIN.dz_m * weighted / MU0)


def topology(vector: Sequence[float]) -> dict[str, Any]:
    _, bz = _field_arrays(vector)
    axis = bz[0]
    scale = max(float(np.max(np.abs(axis))), 1e-300)
    tolerance = max(1e-12, 1e-6 * scale)
    nulls = []
    for j in range(len(axis) - 1):
        left, right = float(axis[j]), float(axis[j + 1])
        if abs(left) <= tolerance:
            nulls.append(HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m)
        elif left * right < 0.0:
            fraction = abs(left) / (abs(left) + abs(right))
            nulls.append(HIGH_DOMAIN.z_min_m + (j + fraction) * HIGH_DOMAIN.dz_m)
    cusps = [
        HIGH_DOMAIN.z_min_m + j * HIGH_DOMAIN.dz_m
        for j in range(1, len(axis) - 1)
        if abs(axis[j]) >= abs(axis[j - 1]) and abs(axis[j]) > abs(axis[j + 1])
    ]
    return {
        "status": "sampled-null" if any(abs(value) <= tolerance for value in axis[1:-1]) else "resolved",
        "null_positions_m": nulls,
        "cusp_positions_m": cusps,
        "null_count": len(nulls),
        "cusp_count": len(cusps),
    }


def topology_match(prediction: Sequence[float], truth: Sequence[float]) -> bool:
    left, right = topology(prediction), topology(truth)
    tolerance = PROTOCOL["gates"]["topology"]["position_tolerance_m"]
    return (
        left["status"] == right["status"]
        and left["null_count"] == right["null_count"]
        and left["cusp_count"] == right["cusp_count"]
        and all(abs(a - b) <= tolerance for a, b in zip(left["null_positions_m"], right["null_positions_m"], strict=True))
        and all(abs(a - b) <= tolerance for a, b in zip(left["cusp_positions_m"], right["cusp_positions_m"], strict=True))
    )


def qois(case: BuiltCase, field: Any, fidelity: str) -> dict[str, float]:
    vector = field_vector(field)
    domain = HIGH_DOMAIN if fidelity == "high" else LOW_DOMAIN
    axis = np.asarray(field.b_z_t)[0]
    peak = max(
        math.hypot(br, bz)
        for br_row, bz_row in zip(field.b_r_t, field.b_z_t, strict=True)
        for br, bz in zip(br_row, bz_row, strict=True)
    )
    boundary = max(
        math.hypot(field.b_r_t[i][j], field.b_z_t[i][j])
        for i in range(len(field.r_m))
        for j in range(len(field.z_m))
        if i == len(field.r_m) - 1 or j in (0, len(field.z_m) - 1)
    ) / max(peak, 1e-300)
    nearest = lambda value: min(range(len(field.z_m)), key=lambda j: abs(field.z_m[j] - value))
    stage_indices = [nearest(stage.center_z_m) for stage in case.geometry.stages]
    stage_values = [float(axis[index]) for index in stage_indices]
    gradients = [
        float(axis[min(index + 1, len(axis) - 1)] - axis[max(index - 1, 0)])
        / (field.z_m[min(index + 1, len(axis) - 1)] - field.z_m[max(index - 1, 0)])
        for index in stage_indices
    ]
    mirror = []
    for left, right, left_value, right_value in zip(
        case.geometry.stages[:-1], case.geometry.stages[1:], stage_values[:-1], stage_values[1:], strict=True
    ):
        midpoint = 0.5 * (left.center_z_m + right.center_z_m)
        mirror.append(max(abs(left_value), abs(right_value)) / max(abs(float(axis[nearest(midpoint)])), 1e-12))
    diagnostics = source_discretization_diagnostics(_fake_problem(case, domain))
    source_errors = []
    for source, item in zip(case.problem.sources, diagnostics, strict=True):
        area = float(item["requested_area_m2"])
        current = abs(float(item["requested_signed_ampere_turns_a"]))
        thickness = min(source.r_outer_m - source.r_inner_m, source.z_max_m - source.z_min_m)
        source_errors.extend(
            (
                abs(float(item["area_error_m2"])) / max(area, 1e-300),
                abs(float(item["ampere_turn_error_a"])) / max(current, 1e-300),
                math.hypot(float(item["centroid_r_error_m"]), float(item["centroid_z_error_m"])) / thickness,
            )
        )
    return {
        "centreline_mid_abs_bz_t": abs(float(axis[nearest(0.5 * case.geometry.chamber.length_m)])),
        "centreline_abs_bz_peak_t": float(np.max(np.abs(axis))),
        "minimum_mirror_ratio": min(mirror),
        "maximum_mirror_ratio": max(mirror),
        "stage_gradient_rms_t_per_m": math.sqrt(sum(value * value for value in gradients) / len(gradients)),
        "stage_gradient_max_abs_t_per_m": max(abs(value) for value in gradients),
        "field_energy_j": field_energy(vector if fidelity == "high" else prolong_low(field)),
        "boundary_to_peak_ratio": boundary,
        "source_representation_error": max(source_errors, default=0.0),
    }


def numerical_record(case: BuiltCase, field: Any, values: Mapping[str, float], index: int, fidelity: str) -> dict[str, Any]:
    return {
        "index": index,
        "fidelity": fidelity,
        "geometry_sha256": case.geometry_sha256,
        "pairing_sha256": canonical_hash(
            {
                "geometry_sha256": case.geometry_sha256,
                "signed_turns": [(source.polarity, source.ampere_turns_a) for source in case.problem.sources],
            }
        ),
        "backend": field.diagnostics.backend,
        "shape": list(field_vector(field).shape),
        "iterations": field.diagnostics.iterations,
        "relative_residual_l2": field.diagnostics.relative_residual_l2,
        "flux_identity_t_per_m": field.diagnostics.max_flux_reconstruction_identity_t_per_m,
        "boundary_to_peak_ratio": values["boundary_to_peak_ratio"],
        "source_representation_error": values["source_representation_error"],
    }


def model_features(row: Sequence[float], coarse_qois: Mapping[str, float], use_coarse: bool) -> np.ndarray:
    base = list(float(value) for value in row)
    if use_coarse:
        base.extend(math.log(max(float(coarse_qois[name]), 1e-15)) for name in QOIS)
    return np.asarray(base, dtype=np.float64)


class SharedKernelGP:
    def __init__(self, train_x: np.ndarray, alpha: np.ndarray, lower: np.ndarray, span: np.ndarray, length: float):
        self.train_x, self.alpha, self.lower, self.span, self.length = train_x, alpha, lower, span, length

    @staticmethod
    def _kernel(left: np.ndarray, right: np.ndarray, length: float) -> np.ndarray:
        distance = np.linalg.norm((left[:, None, :] - right[None, :, :]) / length, axis=2)
        scaled = math.sqrt(5.0) * distance
        return (1.0 + scaled + scaled * scaled / 3.0) * np.exp(-scaled)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, length: float) -> "SharedKernelGP":
        lower = np.min(x, axis=0)
        span = np.maximum(np.max(x, axis=0) - lower, 1e-12)
        normalized = (x - lower) / span
        kernel = cls._kernel(normalized, normalized, length)
        kernel.flat[:: len(kernel) + 1] += PROTOCOL["scalar_model"]["ridge"]
        alpha = np.linalg.solve(kernel, y)
        return cls(normalized, alpha, lower, span, length)

    def predict(self, x: np.ndarray) -> np.ndarray:
        normalized = (x - self.lower) / self.span
        return self._kernel(normalized, self.train_x, self.length) @ self.alpha

    def to_dict(self) -> dict[str, Any]:
        value = {
            "model_type": "shared-kernel-matern52-gp",
            "train_x": self.train_x.tolist(),
            "alpha": self.alpha.tolist(),
            "lower": self.lower.tolist(),
            "span": self.span.tolist(),
            "length": self.length,
            "ridge": PROTOCOL["scalar_model"]["ridge"],
        }
        value["model_hash"] = canonical_hash(value)
        return value


def align_vector(vector: Sequence[float], case: BuiltCase) -> np.ndarray:
    br, bz = _field_arrays(vector)
    physical_z = np.linspace(HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1])
    canonical = np.linspace(-1.0, 2.0, HIGH_DOMAIN.shape[1])
    chamber = case.geometry.chamber.length_m
    source_z = np.where(
        canonical < 0.0,
        -canonical * HIGH_DOMAIN.z_min_m,
        np.where(canonical <= 1.0, canonical * chamber, chamber + (canonical - 1.0) * (HIGH_DOMAIN.z_max_m - chamber)),
    )
    aligned_br = np.vstack([np.interp(source_z, physical_z, row) for row in br])
    aligned_bz = np.vstack([np.interp(source_z, physical_z, row) for row in bz])
    return np.concatenate((aligned_br.ravel(), aligned_bz.ravel()))


def unalign_vector(vector: Sequence[float], case: BuiltCase) -> np.ndarray:
    br, bz = _field_arrays(vector)
    physical_z = np.linspace(HIGH_DOMAIN.z_min_m, HIGH_DOMAIN.z_max_m, HIGH_DOMAIN.shape[1])
    chamber = case.geometry.chamber.length_m
    canonical_at_z = np.where(
        physical_z < 0.0,
        physical_z / -HIGH_DOMAIN.z_min_m,
        np.where(physical_z <= chamber, physical_z / chamber, 1.0 + (physical_z - chamber) / (HIGH_DOMAIN.z_max_m - chamber)),
    )
    canonical_grid = np.linspace(-1.0, 2.0, HIGH_DOMAIN.shape[1])
    result_br = np.vstack([np.interp(canonical_at_z, canonical_grid, row) for row in br])
    result_bz = np.vstack([np.interp(canonical_at_z, canonical_grid, row) for row in bz])
    return np.concatenate((result_br.ravel(), result_bz.ravel()))


def cylindrical_weights() -> np.ndarray:
    nr, nz = HIGH_DOMAIN.shape
    radial = np.maximum(np.arange(nr) * HIGH_DOMAIN.dr_m, 0.25 * HIGH_DOMAIN.dr_m)
    rw = np.ones(nr); rw[[0, -1]] = 0.5
    zw = np.ones(nz); zw[[0, -1]] = 0.5
    weights = (radial * rw)[:, None] * zw[None, :]
    return np.concatenate((weights.ravel(), weights.ravel()))


class WeightedPOD:
    def __init__(self, mean: np.ndarray, modes: np.ndarray, weights: np.ndarray, retained: float, rank: int):
        self.mean, self.modes, self.weights, self.retained, self.rank = mean, modes, weights, retained, rank

    @classmethod
    def fit(cls, snapshots: np.ndarray) -> "WeightedPOD | None":
        weights = cylindrical_weights()
        mean = np.mean(snapshots, axis=0)
        centered = snapshots - mean
        weighted = centered * np.sqrt(weights)[None, :]
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        energy = np.cumsum(singular * singular) / max(float(np.sum(singular * singular)), 1e-300)
        indices = np.flatnonzero(energy >= PROTOCOL["field_model"]["retained_training_energy_min"])
        if not len(indices) or int(indices[0]) + 1 > PROTOCOL["field_model"]["rank_cap"]:
            return None
        rank = int(indices[0]) + 1
        modes = vt[:rank] / np.sqrt(weights)[None, :]
        return cls(mean, modes, weights, float(energy[rank - 1]), rank)

    def project(self, snapshots: np.ndarray) -> np.ndarray:
        return ((snapshots - self.mean) * self.weights[None, :]) @ self.modes.T

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.mean + coefficients @ self.modes

    def to_dict(self) -> dict[str, Any]:
        value = {
            "rank": self.rank,
            "retained_energy": self.retained,
            "mean": self.mean.tolist(),
            "modes": self.modes.tolist(),
            "weights_hash": canonical_hash(self.weights.tolist()),
        }
        value["basis_hash"] = canonical_hash(value)
        return value


def scalar_predictions(
    family: str,
    model: SharedKernelGP,
    features: np.ndarray,
    coarse: Sequence[Mapping[str, float]],
) -> np.ndarray:
    latent = model.predict(features)
    if family == "observed_coarse_log_ratio_matern52_gp":
        baseline = np.asarray([[math.log(max(row[name], 1e-15)) for name in QOIS] for row in coarse])
        latent = latent + baseline
    return np.exp(latent)


def metric_summary(
    truth_qois: np.ndarray,
    predicted_qois: np.ndarray,
    truth_fields: Sequence[np.ndarray],
    predicted_fields: Sequence[np.ndarray],
) -> dict[str, Any]:
    scalar = {}
    for column, name in enumerate(QOIS):
        truth = truth_qois[:, column]
        scale = max(float(np.ptp(truth)), float(np.max(np.abs(truth))) * 1e-12, 1e-15)
        errors = np.abs(predicted_qois[:, column] - truth) / scale
        scalar[name] = {
            "nrmse": float(np.sqrt(np.mean(errors * errors))),
            "worst_range_normalized_error": float(np.max(errors)),
            "range": scale,
        }
    fields = []
    for truth, prediction in zip(truth_fields, predicted_fields, strict=True):
        fields.append(
            {
                "relative_l2": float(np.linalg.norm(prediction - truth) / max(np.linalg.norm(truth), 1e-300)),
                "relative_energy_error": abs(field_energy(prediction) - field_energy(truth)) / max(field_energy(truth), 1e-300),
                "topology_match": topology_match(prediction, truth),
            }
        )
    return {
        "scalar": scalar,
        "worst_scalar_nrmse": max(item["nrmse"] for item in scalar.values()),
        "worst_scalar_error": max(item["worst_range_normalized_error"] for item in scalar.values()),
        "field_rows": fields,
        "worst_field_l2": max(item["relative_l2"] for item in fields),
        "worst_field_energy": max(item["relative_energy_error"] for item in fields),
        "topology_matches": sum(item["topology_match"] for item in fields),
    }
