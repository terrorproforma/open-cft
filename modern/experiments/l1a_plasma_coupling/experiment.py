"""Strict L1a field -> topology -> four-cell global-plasma experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.coupling import (
    COUPLING_V2_SCHEMA_VERSION,
    CouplingValidationError,
    ProfileRole,
    TopologyStatus,
    UncertaintyModel,
    build_screening_proxy,
    describe_profile,
    validate_profile,
)

# ``cft_revival.coupling.build_coupling_record`` now names the coupling v4 CFT
# builder.  This experiment's topology stage is the coupling v2 same-z
# axis/wall comparison, which the accepted package re-exports only as the
# deprecated ``build_screening_proxy`` (no acceptance authority).  Its record
# serializers stay in ``cft_revival.coupling.records``.
from cft_revival.coupling.records import (
    coupling_record_dict as screening_proxy_record_dict,
    global_solver_inputs as screening_proxy_solver_inputs,
)
from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    evaluate_performance,
    operating_point_from_config,
)
from cft_revival.plasma import (
    PlasmaMultiStartResult,
    PlasmaState,
    SolverOptions,
    XenonGlobalInputs,
    solve_global_discharge_multistart,
)

from .adapter import (
    ACCEPTANCE_TIME_UTC,
    ACCEPTED_MAP_POLICY,
    accepted_artifact_document,
    accepted_manifest_document,
    canonical_bytes,
    load_accepted_evidence,
    stable_hash,
    strict_json_bytes,
)

SCHEMA_VERSION = "cft-revival.experiment.l1a-plasma-coupling/1.1.0"
MANIFEST_VERSION = "cft-revival.experiment.l1a-plasma-coupling-manifest/1.0.0"
CLASSIFICATION = "HYPOTHETICAL_L1A_TO_REDUCED_GLOBAL_PLASMA_SCREENING"
PROJECTION_POLICY = {
    "builder": "cft_revival.coupling.build_screening_proxy",
    "record_schema": COUPLING_V2_SCHEMA_VERSION,
    "status": (
        "deprecated coupling v2 same-z axis/wall screening proxy; "
        "no acceptance authority; probabilities are screening inputs only"
    ),
}
LEGACY_ASSUMPTIONS = (0.060, 0.119, 0.160, 0.254)
LEGACY_SOURCE = (
    "FYP/Power_B_EQs.m lines 63-68; commented Kornfeld DM9.2 values, "
    "preserved only as descriptive legacy assumptions"
)
UNCERTAINTY = UncertaintyModel(
    absolute_independent_sigma_t=2.0e-5,
    relative_independent_sigma=0.01,
    common_mode_sigma_t=1.0e-5,
    residual_correlation=0.25,
    coverage_factor=2.0,
)
PLASMA_OPTIONS = SolverOptions(
    max_iterations=250,
    residual_tolerance=1.0e-9,
    gradient_tolerance=1.0e-10,
    step_tolerance=1.0e-12,
    initial_damping=1.0e-3,
)
PLASMA_START_COUNT = 9

DESIGNS = (
    ("compact", "hypothetical-compact-mirror-l1a-v1.json"),
    ("opposed-cusp", "hypothetical-opposed-cusp-l1a-v1.json"),
    ("triplet", "hypothetical-thick-outer-triplet-l1a-v1.json"),
)

RESIDUAL_NAMES = (
    "cathode-emission-current",
    "electron-transport-cell-1",
    "electron-transport-cell-2",
    "electron-transport-cell-3",
    "ionization-source-cell-1",
    "ionization-source-cell-2",
    "ionization-source-cell-3",
    "ionization-source-cell-4",
    "ion-transport-cell-1",
    "ion-transport-cell-2",
    "ion-transport-cell-3",
    "anode-ion-current",
    "electron-power-transport-cell-2",
    "electron-power-transport-cell-3",
    "electron-power-transport-cell-4",
    "interface-current-0",
    "interface-current-1",
    "interface-current-2",
    "interface-current-3",
    "interface-current-4",
    "cusp-current-1",
    "cusp-current-2",
    "cusp-current-3",
    "cell-energy-1",
    "cell-energy-2",
    "cell-energy-3",
    "cell-energy-4",
    "global-energy",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_with_sidecar(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _json_value(dict(payload))
    return {
        **body,
        "integrity": {
            "algorithm": "sha256",
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "payload_sha256": stable_hash(body),
        },
    }


def write_sealed_json(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    value = _seal(payload)
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return value, _write_with_sidecar(path, data)


def _verify_sidecar(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = f"{digest}  {path.name}\n"
    if path.with_name(path.name + ".sha256").read_text(encoding="ascii") != expected:
        raise ValueError(f"invalid SHA-256 sidecar for {path.name}")
    return digest


def load_sealed_json(path: Path) -> dict[str, Any]:
    _verify_sidecar(path)
    value = strict_json_bytes(path.read_bytes(), label=path.name)
    integrity = value.get("integrity")
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "canonicalization",
        "payload_sha256",
    }:
        raise ValueError(f"invalid integrity object in {path.name}")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != "json-sort-keys-compact-utf8-v1"
        or integrity["payload_sha256"] != stable_hash(payload)
    ):
        raise ValueError(f"sealed payload mismatch in {path.name}")
    return value


def topology_compatibility(record) -> tuple[bool, str]:
    """Require a direct ordered one-segment-per-cell map; never invent windows."""

    if record.topology_status is not TopologyStatus.RESOLVED:
        return (
            False,
            f"topology status is {record.topology_status.value}: "
            f"{record.topology_reason}",
        )
    if len(record.segments) != 4:
        return (
            False,
            f"resolved topology has {len(record.segments)} segments; "
            "the global model requires exactly four directly ordered cells",
        )
    cusps = [segment.representative_cusp_z_m for segment in record.segments]
    if any(right <= left for left, right in zip(cusps, cusps[1:])):
        return False, "representative cusp positions are not strictly ordered"
    domain_start = record.segments[0].z_start_m
    domain_end = record.segments[-1].z_end_m
    if any(position <= domain_start or position >= domain_end for position in cusps):
        return (
            False,
            "one or more apparent cusps lie on the finite Dirichlet map boundary; "
            "boundary zeros are not physical four-cell topology evidence",
        )
    for left, right in zip(record.segments, record.segments[1:]):
        if left.z_end_m != right.z_start_m:
            return False, "topology segments are not exactly contiguous"
    mirror_ratios = [
        segment.mirror_loss.mirror_ratio_high_to_low
        for segment in record.segments
    ]
    if any(
        value is None or not isfinite(value) or value < 1.0
        for value in mirror_ratios
    ):
        return False, "one or more cells has no finite physical mirror ratio"
    probabilities = [
        segment.mirror_loss.probability.value for segment in record.segments
    ]
    if any(not isfinite(value) or not 0.0 <= value < 1.0 for value in probabilities):
        return False, "one or more field-derived probabilities are outside [0,1)"
    return True, "direct upstream-to-downstream one-segment-per-cell mapping"


def _operating_point(config_path: Path):
    config_bytes = config_path.read_bytes()
    raw = strict_json_bytes(config_bytes, label=config_path.name)
    point = operating_point_from_config(raw)
    fractions = point.charge_state_fractions
    total_rate = point.propellant_mass_flow.kg_per_s / point.xenon_atom_mass_kg
    beam_current = (
        ELEMENTARY_CHARGE_C
        * total_rate
        * fractions.charge_weighted_ion_fraction
    )
    anode_current = (
        beam_current
        / point.beam_divergence_factors.beam_current_fraction_of_anode_current
    )
    return point, anode_current, {
        "path": "config/l0-representative-point.json",
        "file_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "hypothetical_inputs": raw["hypothetical_inputs"],
        "derivation": (
            "I_a = e*(mdot/m_Xe)*(f_Xe+ + 2*f_Xe2+) / "
            "beam_current_fraction_of_anode_current"
        ),
    }


def build_plasma_inputs(
    probabilities: Sequence[float],
    *,
    anode_voltage_v: float,
    anode_current_a: float,
) -> XenonGlobalInputs:
    if len(probabilities) != 4:
        raise ValueError("four direct topology probabilities are required")
    return XenonGlobalInputs(
        anode_voltage_v=anode_voltage_v,
        anode_current_a=anode_current_a,
        cusp_arrival_probabilities=tuple(float(value) for value in probabilities),
    )


def run_plasma_case(
    inputs: XenonGlobalInputs,
    *,
    initial_states: Sequence[PlasmaState] | None = None,
) -> PlasmaMultiStartResult:
    return solve_global_discharge_multistart(
        inputs,
        initial_states,
        start_count=PLASMA_START_COUNT,
        options=PLASMA_OPTIONS,
        use_analytic_jacobian=True,
    )


def _residual_rows(
    normalized: Sequence[float],
    raw: Sequence[float] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "name": name,
            "normalized": float(normalized[index]),
            "raw_si": None if raw is None else float(raw[index]),
        }
        for index, name in enumerate(RESIDUAL_NAMES)
    ]


def serialize_plasma(result: PlasmaMultiStartResult) -> dict[str, Any]:
    attempts = []
    for index, attempt in enumerate(result.attempts):
        diagnostics = attempt.diagnostics
        raw = None if attempt.evaluation is None else attempt.evaluation.raw
        attempts.append(
            {
                "start_index": index,
                "branch_status": "accepted" if diagnostics.converged else "failed",
                "state_published": attempt.state is not None,
                "state": None if attempt.state is None else _json_value(attempt.state),
                "diagnostics": _json_value(diagnostics),
                "residual_rows": _residual_rows(diagnostics.normalized_residuals, raw),
                "conservation": (
                    None
                    if attempt.evaluation is None
                    else {
                        "powers": _json_value(attempt.evaluation.powers),
                        "closures": _json_value(attempt.evaluation.closures),
                    }
                ),
            }
        )
    valid = result.best.state is not None and result.best.diagnostics.converged
    return {
        "valid_state": valid,
        "selected_start_index": result.selected_start_index,
        "residual_floor": result.residual_floor,
        "rank_status": {
            "jacobian_rank": result.best.diagnostics.jacobian_rank,
            "state_size": 25,
            "full_column_rank": result.best.diagnostics.jacobian_rank == 25,
            "condition_estimate": result.best.diagnostics.jacobian_condition_estimate,
        },
        "attempts": attempts,
    }


def _profile_summary(record_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "centreline": record_dict["inner_profile"],
        "wall": record_dict["wall"],
        "uncertainty_model": record_dict["uncertainty_model"],
        "selected_segments": [
            {
                "segment_id": segment["segment_id"],
                "z_start_m": segment["z_start_m"],
                "z_end_m": segment["z_end_m"],
                "representative_cusp_z_m": segment["representative_cusp_z_m"],
                "mirror_ratio_high_to_low": segment["mirror_loss"][
                    "mirror_ratio_high_to_low"
                ],
                "loss_cone_probability": segment["mirror_loss"]["probability"],
            }
            for segment in record_dict["segments"]
        ],
    }


def _artifact_profile_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Describe accepted full-resolution profiles when mirror projection rejects."""

    class Profile:
        def __init__(self, value: Mapping[str, Any]) -> None:
            self.z_m = value["z_m"]
            self.b_r_t = value["b_r_t"]
            self.b_z_t = value["b_z_t"]

    summaries: dict[str, Any] = {
        "uncertainty_model": _json_value(UNCERTAINTY),
        "selected_segments": [],
    }
    for name, role, radius_key in (
        ("centreline", ProfileRole.CENTRELINE, "r_m"),
        ("wall", ProfileRole.WALL, "sampled_r_m"),
    ):
        raw = artifact["profiles"][name]
        magnitude = [
            (float(br) ** 2 + float(bz) ** 2) ** 0.5
            for br, bz in zip(raw["b_r_t"], raw["b_z_t"], strict=True)
        ]
        independent = [
            UNCERTAINTY.absolute_independent_sigma_t
            + UNCERTAINTY.relative_independent_sigma * value
            for value in magnitude
        ]
        profile = validate_profile(
            Profile(raw),
            name=name,
            sampled_r_m=float(raw[radius_key]),
            role=role,
            independent_sigma_b_t=independent,
            common_mode_sigma_t=UNCERTAINTY.common_mode_sigma_t,
        )
        summaries[name] = _json_value(describe_profile(profile))
    return summaries


def _design_result(
    design_id: str,
    artifact_path: Path,
    manifest_path: Path,
    *,
    point,
    anode_current_a: float,
) -> dict[str, Any]:
    evidence, acceptance_identity = load_accepted_evidence(
        artifact_path,
        manifest_path,
        policy=ACCEPTED_MAP_POLICY,
        reference_time_utc=ACCEPTANCE_TIME_UTC,
    )
    # Same exact bytes the evidence was issued from, reloaded through the
    # public serialization v1.2 loader (never hand-parsed).
    artifact_bytes = artifact_path.read_bytes()
    artifact = accepted_artifact_document(artifact_bytes, source=artifact_path.name)
    artifact_file_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_file_sha256 != acceptance_identity["artifact_hash"]:
        raise ValueError(f"accepted artifact bytes changed during replay: {artifact_path.name}")
    wall_radius = float(artifact["profiles"]["wall"]["requested_r_m"])
    try:
        # Deprecated v2 same-z screening proxy; its DeprecationWarning is left
        # to propagate on purpose and the policy is recorded in the dataset.
        record = build_screening_proxy(
            evidence,
            wall_radius_m=wall_radius,
            uncertainty_model=UNCERTAINTY,
            reference_time_utc=ACCEPTANCE_TIME_UTC,
        )
    except CouplingValidationError as error:
        reason = f"coupling v2 rejected topology projection: {error}"
        return {
            "design_id": design_id,
            "artifact": {
                "path": f"examples/axisymmetric/results/{artifact_path.name}",
                "file_sha256": artifact_file_sha256,
                "payload_sha256": artifact["integrity"]["payload_sha256"],
                "manifest_file_sha256": acceptance_identity["manifest_file_sha256"],
                "manifest_payload_sha256": acceptance_identity[
                    "manifest_payload_sha256"
                ],
            },
            "coupling_identity": {
                **{
                    key: acceptance_identity[key]
                    for key in (
                        "field_map_hash",
                        "artifact_hash",
                        "source_hash",
                        "source_map_binding_hash",
                        "artifact_schema_version",
                        "field_model_hash",
                        "code_hash",
                        "config_hash",
                        "adapter_code_hash",
                        "diagnostics",
                    )
                },
                "record_hash": None,
                "coupling_model_hash": None,
            },
            "topology": {
                "status": "projection_rejected",
                "reason": str(error),
                "compatible_with_four_cell_global_model": False,
                "compatibility_reason": reason,
                "profile_summary": _artifact_profile_summary(artifact),
                "solver_projection_rows": [],
            },
            "legacy_comparison": {
                "assumptions": list(LEGACY_ASSUMPTIONS),
                "source": LEGACY_SOURCE,
                "status": "not comparable because strict coupling projection failed",
                "field_minus_legacy": None,
            },
            "plasma": None,
            "screening_performance": {
                "status": "not_published",
                "reason": reason,
            },
            "status": "failed",
            "failure_reason": reason,
        }
    record_dict = screening_proxy_record_dict(record)
    rows = [dict(row) for row in screening_proxy_solver_inputs(record)]
    compatible, reason = topology_compatibility(record)
    probabilities = [
        float(row["loss_cone_probability"]) for row in rows
    ]
    legacy = {
        "assumptions": list(LEGACY_ASSUMPTIONS),
        "source": LEGACY_SOURCE,
        "status": (
            "index-aligned descriptive comparison"
            if compatible
            else "not index-aligned because topology is incompatible"
        ),
        "field_minus_legacy": (
            [
                probability - legacy_probability
                for probability, legacy_probability in zip(
                    probabilities, LEGACY_ASSUMPTIONS, strict=True
                )
            ]
            if compatible
            else None
        ),
    }
    plasma = None
    performance = None
    if compatible:
        inputs = build_plasma_inputs(
            probabilities,
            anode_voltage_v=point.discharge_voltage_v,
            anode_current_a=anode_current_a,
        )
        plasma_result = run_plasma_case(inputs)
        plasma = {
            "input": _json_value(inputs),
            "input_hash": stable_hash(_json_value(inputs)),
            **serialize_plasma(plasma_result),
        }
        if plasma["valid_state"]:
            performance = {
                "status": "published_after_valid_plasma_state",
                "l0_screening": _json_value(evaluate_performance(point)),
            }
        else:
            performance = {
                "status": "not_published",
                "reason": "global plasma residual tolerance was not satisfied",
            }
    else:
        performance = {
            "status": "not_published",
            "reason": f"topology incompatible: {reason}",
        }
    accepted = bool(plasma and plasma["valid_state"])
    return {
        "design_id": design_id,
        "artifact": {
            "path": f"examples/axisymmetric/results/{artifact_path.name}",
            "file_sha256": artifact_file_sha256,
            "payload_sha256": artifact["integrity"]["payload_sha256"],
            "manifest_file_sha256": acceptance_identity["manifest_file_sha256"],
            "manifest_payload_sha256": acceptance_identity[
                "manifest_payload_sha256"
            ],
        },
        "coupling_identity": {
            key: record_dict[key]
            for key in (
                "record_hash",
                "field_map_hash",
                "artifact_hash",
                "source_hash",
                "source_map_binding_hash",
                "artifact_schema_version",
                "field_model_hash",
                "code_hash",
                "config_hash",
                "adapter_code_hash",
                "coupling_model_hash",
                "diagnostics",
            )
        },
        "topology": {
            "status": record.topology_status.value,
            "reason": record.topology_reason,
            "compatible_with_four_cell_global_model": compatible,
            "compatibility_reason": reason,
            "profile_summary": _profile_summary(record_dict),
            "solver_projection_rows": rows,
        },
        "legacy_comparison": legacy,
        "plasma": plasma,
        "screening_performance": performance,
        "status": "accepted" if accepted else "failed",
        "failure_reason": (
            None
            if accepted
            else reason
            if not compatible
            else "global plasma residual tolerance was not satisfied"
        ),
    }


def _report(dataset: Mapping[str, Any]) -> str:
    lines = [
        "# Strict L1a field-to-global-plasma experiment",
        "",
        f"- Classification: `{CLASSIFICATION}`",
        f"- Accepted plasma states: {dataset['summary']['accepted_count']}",
        f"- Failed designs: {dataset['summary']['failed_count']}",
        f"- Four-cell-compatible topologies: {dataset['summary']['compatible_count']}",
        "",
        "## Per-design outcome",
        "",
    ]
    for result in dataset["designs"]:
        topology = result["topology"]
        segments = topology["profile_summary"]["selected_segments"]
        probabilities = [
            segment["loss_cone_probability"]["value"] for segment in segments
        ]
        mirrors = [
            segment["mirror_ratio_high_to_low"] for segment in segments
        ]
        plasma_detail = (
            "not run"
            if result["plasma"] is None
            else (
                f"valid={result['plasma']['valid_state']}, "
                f"residual_floor={result['plasma']['residual_floor']:.6g}, "
                f"rank={result['plasma']['rank_status']['jacobian_rank']}/25, "
                f"condition={result['plasma']['rank_status']['condition_estimate']:.6g}"
            )
        )
        lines.extend(
            (
                f"### {result['design_id']}",
                "",
                f"- Topology: `{topology['status']}`",
                f"- Four-cell compatibility: "
                f"`{topology['compatible_with_four_cell_global_model']}`",
                f"- Derived segments: "
                f"{len(segments)}",
                f"- Mirror ratios: `{mirrors}`",
                f"- Field-derived probabilities: `{probabilities}`",
                f"- Plasma: `{plasma_detail}`",
                f"- Outcome: `{result['status']}` — "
                f"{result['failure_reason'] or 'strict state accepted'}",
                "",
            )
        )
    lines.extend(
        (
            "## Claim boundary",
            "",
            "The field artifacts are accepted L1a equivalent-current finite-box results",
            "loaded through field serialization v1.2. The topology projection is the",
            "coupling v2 same-z axis/wall comparison, which the accepted coupling package",
            "marks as a deprecated screening proxy without acceptance authority.",
            "The operating point is explicitly hypothetical. Field-derived loss-cone",
            "probabilities are screening inputs, not measured transport probabilities.",
            "Legacy DM9.2 values are shown descriptively and are not treated as truth.",
            "No plasma state or L0 performance is published after topology or residual failure.",
            "",
            "The L1a schema has no generation timestamp. The coupling evidence therefore",
            "uses the experiment-local, hash-covered acceptance registry timestamp",
            f"`{ACCEPTANCE_TIME_UTC.isoformat()}` rather than claiming a solve timestamp.",
            "",
            "## Next corrections",
            "",
            "- Generate accepted geometries with four physically identified ordered cells,",
            "  or generalize the plasma model to topology-variable cell count.",
            "- Replace equivalent-current fields with accepted material-aware magnetization,",
            "  nonlinear permeability, open-boundary, and mesh-convergence evidence.",
            "- Calibrate field uncertainty and transport probability against measurements or",
            "  a validated kinetic model; retain correlations and geometry identities.",
            "",
        )
    )
    return "\n".join(lines)


def run_experiment(output: Path, *, modern_root: Path | None = None) -> dict[str, Any]:
    root = (
        Path(__file__).resolve().parents[2]
        if modern_root is None
        else modern_root.resolve()
    )
    accepted_root = root / "examples" / "axisymmetric" / "results"
    manifest_path = accepted_root / "manifest-l1a-v1.json"
    point, anode_current, operating_identity = _operating_point(
        root / "config" / "l0-representative-point.json"
    )
    results = [
        _design_result(
            design_id,
            accepted_root / filename,
            manifest_path,
            point=point,
            anode_current_a=anode_current,
        )
        for design_id, filename in DESIGNS
    ]
    summary = {
        "design_count": len(results),
        "compatible_count": sum(
            result["topology"]["compatible_with_four_cell_global_model"]
            for result in results
        ),
        "accepted_count": sum(result["status"] == "accepted" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "plasma_solve_count": sum(result["plasma"] is not None for result in results),
        "performance_publication_count": sum(
            result["screening_performance"]["status"]
            == "published_after_valid_plasma_state"
            for result in results
        ),
    }
    manifest_bytes = manifest_path.read_bytes()
    accepted_manifest = accepted_manifest_document(
        manifest_bytes, source=manifest_path.name
    )
    dataset_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "accepted_field_manifest": {
            "path": "examples/axisymmetric/results/manifest-l1a-v1.json",
            "file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "payload_sha256": accepted_manifest["integrity"]["payload_sha256"],
        },
        "operating_point": {
            **operating_identity,
            "anode_voltage_v": point.discharge_voltage_v,
            "derived_anode_current_a": anode_current,
        },
        "coupling_policy": {
            "projection": dict(PROJECTION_POLICY),
            "uncertainty": _json_value(UNCERTAINTY),
            "compatibility": (
                "exactly four resolved, ordered, contiguous coupling-v2 segments; "
                "no inferred or forced windows"
            ),
        },
        "plasma_policy": {
            "solver_options": _json_value(PLASMA_OPTIONS),
            "start_count": PLASMA_START_COUNT,
            "closure_factors": {
                "excitation_fraction": 0.25,
                "ionization_fraction": 0.07,
                "thermalization_fraction": 0.68,
                "tuned_for_convergence": False,
            },
            "publication": "state only when solver diagnostics report strict convergence",
        },
        "designs": results,
        "summary": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    dataset, dataset_hash = write_sealed_json(output / "dataset.json", dataset_payload)
    report_data = _report(dataset).encode("utf-8")
    report_hash = _write_with_sidecar(output / "report.md", report_data)
    manifest_payload = {
        "schema_version": MANIFEST_VERSION,
        "classification": CLASSIFICATION,
        "dataset_payload_sha256": dataset["integrity"]["payload_sha256"],
        "files": [
            {
                "path": "dataset.json",
                "file_sha256": dataset_hash,
                "payload_sha256": dataset["integrity"]["payload_sha256"],
            },
            {
                "path": "report.md",
                "file_sha256": report_hash,
                "payload_sha256": None,
            },
        ],
        "source_artifacts": [
            result["artifact"] for result in results
        ],
        "operating_point_file_sha256": operating_identity["file_sha256"],
    }
    manifest, _ = write_sealed_json(output / "manifest.json", manifest_payload)
    validate_bundle(output, modern_root=root)
    return {"dataset": dataset, "manifest": manifest}


def validate_bundle(output: Path, *, modern_root: Path | None = None) -> dict[str, Any]:
    root = (
        Path(__file__).resolve().parents[2]
        if modern_root is None
        else modern_root.resolve()
    )
    manifest = load_sealed_json(output / "manifest.json")
    dataset = load_sealed_json(output / "dataset.json")
    if manifest["schema_version"] != MANIFEST_VERSION:
        raise ValueError("unsupported experiment manifest schema")
    if manifest["dataset_payload_sha256"] != dataset["integrity"]["payload_sha256"]:
        raise ValueError("manifest/dataset payload identity mismatch")
    for entry in manifest["files"]:
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("manifest path escapes output directory")
        if _verify_sidecar(path) != entry["file_sha256"]:
            raise ValueError(f"manifest file hash mismatch for {entry['path']}")
    accepted_root = root / "examples" / "axisymmetric" / "results"
    accepted_manifest = accepted_root / "manifest-l1a-v1.json"
    accepted_manifest_hash = hashlib.sha256(accepted_manifest.read_bytes()).hexdigest()
    for entry in manifest["source_artifacts"]:
        path = accepted_root / Path(entry["path"]).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_sha256"]:
            raise ValueError(f"accepted source artifact changed: {path.name}")
        if accepted_manifest_hash != entry["manifest_file_sha256"]:
            raise ValueError("accepted source manifest changed")
    config_path = root / "config" / "l0-representative-point.json"
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != (
        manifest["operating_point_file_sha256"]
    ):
        raise ValueError("operating-point configuration changed")
    return {"manifest": manifest, "dataset": dataset}
