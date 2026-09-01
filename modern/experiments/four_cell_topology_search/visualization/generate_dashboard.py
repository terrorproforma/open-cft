"""Build the sealed, standalone four-cell topology-search dashboard."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import hypot, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RESULTS = EXPERIMENT / "results"
TEMPLATE = HERE / "dashboard.template.html"
DEFAULT_OUTPUT = HERE / "four-cell-topology-search.html"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"

EXPECTED_DATASET_FILE_SHA256 = "145decd7f1856b9e00429e07f06d47beb2a1de24523d75e6eb10c8f0856344de"
EXPECTED_DATASET_PAYLOAD_SHA256 = "fced0eb33b9073a83911e8ec7717a88e7764708b344020007e9a516a40836491"
EXPECTED_MANIFEST_FILE_SHA256 = "3ddae4a7254e832126a3e368b9e30cc29c25fe8fdafd127fcbbaafe27def7b53"
EXPECTED_MANIFEST_PAYLOAD_SHA256 = "2ff65a4887058ebfb9e23542caca066515c2ba03b04efa6f0b9da4cfd787f38d"
EXPECTED_COMPATIBLE = ("four-cell-005-8885e09139", "four-cell-029-52bb37501f")
FAILURES = (
    "FIELD_GATE_FAILURE",
    "TOPOLOGY_COUNT",
    "BOUNDARY_LEAKAGE",
    "MIRROR_INVERTED",
)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {label}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid readable JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _verify_integrity(value: Mapping[str, Any], label: str) -> str:
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "algorithm",
        "canonicalization",
        "payload_sha256",
    }:
        raise ValueError(f"{label} integrity declaration is not closed")
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != CANONICALIZATION
    ):
        raise ValueError(f"{label} integrity declaration is unsupported")
    claimed = _digest(integrity["payload_sha256"], f"{label} payload digest")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if _canonical_hash(payload) != claimed:
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return claimed


def _verify_file(path: Path, label: str, expected: str | None = None) -> str:
    try:
        digest = sha256(path.read_bytes()).hexdigest()
        sidecar = path.with_name(path.name + ".sha256").read_text(encoding="ascii")
    except OSError as error:
        raise ValueError(f"{label} or its SHA-256 sidecar is unreadable") from error
    if expected is not None and digest != expected:
        raise ValueError(f"{label} file SHA-256 mismatch")
    if sidecar != f"{digest}  {path.name}\n":
        raise ValueError(f"{label} SHA-256 sidecar is invalid")
    return digest


def _safe_result_path(raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or "\\" in raw:
        raise ValueError(f"{label} must be a portable relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"{label} escapes results")
    path = RESULTS.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(RESULTS.resolve()):
        raise ValueError(f"{label} escapes results")
    return path


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _matrix(value: Any, nr: int, nz: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != nr:
        raise ValueError(f"{label} radial dimension mismatch")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != nz:
            raise ValueError(f"{label}[{index}] axial dimension mismatch")
        result.append([_number(item, label) for item in row])
    return result


def _validate_field(field: Mapping[str, Any], label: str) -> None:
    if (
        field.get("schema_version") != "cft-axisymmetric-field-map/1.1.0"
        or field.get("model_level") != "L1a"
    ):
        raise ValueError(f"{label} field schema/model mismatch")
    field_map = field.get("field_map")
    if not isinstance(field_map, Mapping):
        raise ValueError(f"{label} field map is missing")
    r_values = [_number(item, f"{label}.r") for item in field_map["r_m"]]
    z_values = [_number(item, f"{label}.z") for item in field_map["z_m"]]
    if (
        len(r_values) < 2
        or len(z_values) < 2
        or any(b <= a for a, b in zip(r_values, r_values[1:]))
        or any(b <= a for a, b in zip(z_values, z_values[1:]))
    ):
        raise ValueError(f"{label} field coordinates are invalid")
    nr, nz = len(r_values), len(z_values)
    maps = {
        key: _matrix(field_map[key], nr, nz, f"{label}.{key}")
        for key in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t")
    }
    for i in range(nr):
        for j in range(nz):
            expected = hypot(maps["b_r_t"][i][j], maps["b_z_t"][i][j])
            if abs(maps["b_magnitude_t"][i][j] - expected) > max(
                2e-15, 2e-12 * expected
            ):
                raise ValueError(f"{label} |B| component identity mismatch")
    for name in ("centreline", "wall"):
        profile = field["profiles"][name]
        count = len(profile["z_m"])
        if count < 2 or any(
            len(profile[key]) != count for key in ("b_r_t", "b_z_t")
        ):
            raise ValueError(f"{label} {name} profile shape mismatch")


def _validate_geometry(geometry: Mapping[str, Any], label: str) -> None:
    if geometry.get("schema_version") != "cft_revival.geometry.axisymmetric_cft/1.1.0":
        raise ValueError(f"{label} geometry schema mismatch")
    if not geometry.get("regions") or not geometry.get("stages"):
        raise ValueError(f"{label} geometry is incomplete")
    for region in geometry["regions"]:
        for key in (
            "z_min_m",
            "z_max_m",
            "r_inner_start_m",
            "r_inner_end_m",
            "r_outer_start_m",
            "r_outer_end_m",
        ):
            _number(region[key], f"{label}.{key}")


def _topology_projection(case: Mapping[str, Any]) -> dict[str, Any]:
    topology = case["topology"]
    segments = topology["segments"]
    finite_mirrors = [
        float(segment["mirror_ratio_high_to_low"])
        for segment in segments
        if segment["mirror_ratio_high_to_low"] is not None
    ]
    boundary_ratio = float(case["field_quality"]["boundary_to_peak_ratio"])
    return {
        "case_id": case["case_id"],
        "compatible": bool(topology["compatible"]),
        "status": case["status"],
        "backend": case["backend"],
        "stage_count": case["derived_geometry"]["stage_count"],
        "segment_count": topology["segment_count"],
        "confidence": topology["overall_confidence"],
        "minimum_mirror_ratio": min(finite_mirrors) if finite_mirrors else None,
        "boundary_to_peak_ratio": boundary_ratio,
        "field_peak_t": case["field_quality"]["field_peak_t"],
        "relative_residual_l2": case["field_quality"]["relative_residual_l2"],
        "source_representation_error": case["field_quality"][
            "source_representation_error"
        ],
        "failure_codes": list(case["failure_codes"]),
        "gates": {**case["field_gates"], **topology["gates"]},
        "design_values": case["design_values"],
        "identity": case["identity"],
        "sampling_provenance": case["sampling_provenance"],
    }


def _selected_residual_roots(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcomes = []
    for item in case["plasma"]:
        selected = int(item["selected_start_index"])
        attempt = next(
            attempt for attempt in item["attempts"] if attempt["start_index"] == selected
        )
        diagnostics = attempt["diagnostics"]
        conservation = attempt["conservation_diagnostics"]
        if conservation is None:
            raise ValueError(f"{case['case_id']} selected root lacks conservation diagnostics")
        if not item["residual_root_found"] or not attempt["residual_root_found"]:
            raise ValueError(f"{case['case_id']} selected attempt is not a residual root")
        outcomes.append(
            {
                "case_id": case["case_id"],
                "operating_point": item["operating_point"],
                "input_hash": item["input_hash"],
                "selected_start_index": selected,
                "start_count": len(item["attempts"]),
                "residual_root_found": item["residual_root_found"],
                "outcome_classification": item["outcome_classification"],
                "identifiability": item["identifiability"],
                "conservation_diagnostics": conservation,
                "residual_diagnostics": {
                    "residual_inf_norm": diagnostics["residual_inf_norm"],
                    "residual_floor": item["residual_floor"],
                    "jacobian_rank": diagnostics["jacobian_rank"],
                    "jacobian_condition_estimate": diagnostics[
                        "jacobian_condition_estimate"
                    ],
                },
            }
        )
    return outcomes


def _representative_projection(
    case: Mapping[str, Any],
    binding: Mapping[str, Any],
    geometry: Mapping[str, Any],
    field: Mapping[str, Any],
) -> dict[str, Any]:
    segments = case["topology"]["segments"]
    return {
        "case_id": case["case_id"],
        "rank": binding["rank"],
        "geometry": {
            "regions": geometry["regions"],
            "stages": geometry["stages"],
            "chamber": geometry["chamber"],
        },
        "field": {
            "map": field["field_map"],
            "profiles": field["profiles"],
            "sources": field["input"]["sources"],
            "diagnostics": field["diagnostics"],
            "limitations": field["limitations"],
        },
        "topology": [
            {
                "segment_id": segment["segment_id"],
                "z_start_m": segment["z_start_m"],
                "z_end_m": segment["z_end_m"],
                "cusp": segment["cusp"],
                "wall_radius_m": segment["wall_radius_m"],
                "wall_b_t": segment["wall_b_t"],
                "mirror_ratio": segment["mirror_ratio_high_to_low"],
                "probability": segment["loss_cone_probability"],
                "probability_lower": segment["loss_cone_probability_lower"],
                "probability_upper": segment["loss_cone_probability_upper"],
            }
            for segment in segments
        ],
        "identity": {
            "geometry_file_sha256": binding["geometry"]["file_sha256"],
            "geometry_payload_sha256": binding["geometry"]["payload_sha256"],
            "field_file_sha256": binding["field"]["file_sha256"],
            "field_payload_sha256": binding["field"]["payload_sha256"],
        },
    }


def build_payload() -> dict[str, Any]:
    dataset_path = RESULTS / "dataset.json"
    manifest_path = RESULTS / "manifest.json"
    dataset_file_hash = _verify_file(
        dataset_path, "dataset", EXPECTED_DATASET_FILE_SHA256
    )
    manifest_file_hash = _verify_file(
        manifest_path, "manifest", EXPECTED_MANIFEST_FILE_SHA256
    )
    dataset = _load_object(dataset_path, "dataset")
    manifest = _load_object(manifest_path, "manifest")
    if _verify_integrity(dataset, "dataset") != EXPECTED_DATASET_PAYLOAD_SHA256:
        raise ValueError("dataset payload differs from reviewed evidence")
    if _verify_integrity(manifest, "manifest") != EXPECTED_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("manifest payload differs from reviewed evidence")
    if manifest["dataset_payload_sha256"] != EXPECTED_DATASET_PAYLOAD_SHA256:
        raise ValueError("manifest/dataset payload binding mismatch")
    if manifest["protocol_status"] != dataset["protocol_status"]:
        raise ValueError("manifest/dataset protocol-status binding mismatch")
    if manifest["semantic_correction"] != dataset["semantic_correction"]:
        raise ValueError("manifest/dataset semantic-correction binding mismatch")
    if len(dataset["cases"]) != 128 or dataset["summary"]["evaluated_count"] != 128:
        raise ValueError("dataset does not contain the exact 128 evaluated cases")
    if (
        dataset["summary"]["plasma_residual_root_count"] != 6
        or dataset["summary"]["identifiable_state_count"] != 0
        or dataset["summary"]["performance_publication_count"] != 0
    ):
        raise ValueError("corrected v1 root/identifiability/publication counts mismatch")
    by_id = {case["case_id"]: case for case in dataset["cases"]}
    compatible = tuple(
        sorted(case_id for case_id, case in by_id.items() if case["topology"]["compatible"])
    )
    if compatible != EXPECTED_COMPATIBLE:
        raise ValueError("compatible-candidate identity mismatch")

    deterministic = {item["path"]: item for item in manifest["deterministic_files"]}
    representatives = []
    for binding in manifest["representatives"]:
        for kind in ("geometry", "field"):
            record = binding[kind]
            path = _safe_result_path(record["path"], f"{binding['case_id']} {kind}")
            file_hash = _verify_file(path, f"{binding['case_id']} {kind}")
            if file_hash != record["file_sha256"]:
                raise ValueError(f"{binding['case_id']} {kind} manifest file mismatch")
            listed = deterministic.get(record["path"])
            expected_binding = {
                "path": record["path"],
                "file_sha256": record["file_sha256"],
                "payload_sha256": record["payload_sha256"],
                "kind": kind,
            }
            if listed != expected_binding:
                raise ValueError(f"{binding['case_id']} {kind} deterministic binding mismatch")
            value = _load_object(path, f"{binding['case_id']} {kind}")
            if _verify_integrity(value, f"{binding['case_id']} {kind}") != record[
                "payload_sha256"
            ]:
                raise ValueError(f"{binding['case_id']} {kind} payload mismatch")
            if kind == "field":
                _validate_field(value, f"{binding['case_id']} field")
                field = value
            else:
                _validate_geometry(value, f"{binding['case_id']} geometry")
                geometry = value
        if binding["case_id"] in compatible:
            representatives.append(
                _representative_projection(
                    by_id[binding["case_id"]], binding, geometry, field
                )
            )
    representatives.sort(key=lambda item: item["rank"])

    cases = [_topology_projection(case) for case in dataset["cases"]]
    outcomes = [
        outcome
        for case_id in compatible
        for outcome in _selected_residual_roots(by_id[case_id])
    ]
    payload = {
        "title": "Four-cell topology search",
        "warning": (
            "V1 IS SUPERSEDED DEVELOPMENT EVIDENCE. Coupling v2 used a deprecated "
            "same-z centreline-low/wall-high mirror proxy with roundoff-scale null "
            "lows. These six rank-22 roots are non-identifiable screening-equation "
            "diagnostics only; physical mirror interpretation is invalid."
        ),
        "summary": dataset["summary"],
        "failure_taxonomy": [
            {"code": code, "count": dataset["summary"]["failure_counts"][code]}
            for code in FAILURES
        ],
        "cases": cases,
        "representatives": representatives,
        "residual_roots": outcomes,
        "parity": dataset["parity"],
        "declared_gates": dataset["declared_gates"],
        "protocol_status": dataset["protocol_status"],
        "semantic_correction": dataset["semantic_correction"],
        "limitations": dataset["limitations"],
        "provenance": {
            "classification": manifest["classification"],
            "schema_version": dataset["schema_version"],
            "model_chain": dataset["model_chain"],
            "sampling": dataset["sampling"],
            "dataset_file_sha256": dataset_file_hash,
            "dataset_payload_sha256": EXPECTED_DATASET_PAYLOAD_SHA256,
            "manifest_file_sha256": manifest_file_hash,
            "manifest_payload_sha256": EXPECTED_MANIFEST_PAYLOAD_SHA256,
            "canonicalization": CANONICALIZATION,
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    if len(payload["cases"]) != 128:
        raise ValueError("embedded payload must contain 128 cases")
    compatible = tuple(
        sorted(case["case_id"] for case in payload["cases"] if case["compatible"])
    )
    if compatible != EXPECTED_COMPATIBLE:
        raise ValueError("embedded compatible candidates mismatch")
    if len(payload["representatives"]) != 2 or len(payload["residual_roots"]) != 6:
        raise ValueError("embedded representative/residual-root count mismatch")
    if any(
        not item["residual_root_found"]
        or item["residual_diagnostics"]["jacobian_rank"] != 22
        or item["identifiability"]["status"] != "non_identifiable"
        or item["identifiability"]["publication_allowed"]
        for item in payload["residual_roots"]
    ):
        raise ValueError("embedded residual-root semantics changed")
    summary = payload["summary"]
    if (
        summary["plasma_residual_root_count"] != 6
        or summary["identifiable_state_count"] != 0
        or summary["performance_publication_count"] != 0
    ):
        raise ValueError("embedded corrected summary mismatch")
    if (
        payload["protocol_status"]["experiment_version"] != "v1"
        or payload["protocol_status"]["status"] != "development_evidence_only"
        or payload["protocol_status"]["valid_for_physical_mirror_claims"]
        or payload["semantic_correction"]["kind"]
        != "semantic_publication_metadata_correction"
    ):
        raise ValueError("embedded protocol/correction semantics mismatch")
    if payload["provenance"]["dataset_payload_sha256"] != EXPECTED_DATASET_PAYLOAD_SHA256:
        raise ValueError("embedded dataset identity mismatch")
    if payload["provenance"]["manifest_payload_sha256"] != EXPECTED_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("embedded manifest identity mismatch")
    expected_counts = {"FIELD_GATE_FAILURE": 68, "TOPOLOGY_COUNT": 118,
                       "BOUNDARY_LEAKAGE": 36, "MIRROR_INVERTED": 61}
    if {item["code"]: item["count"] for item in payload["failure_taxonomy"]} != expected_counts:
        raise ValueError("overlapping failure taxonomy mismatch")


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).replace("</", "<\\/")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("template must contain exactly one data marker")
    return template.replace("__DATA__", data)


def generate(output: Path = DEFAULT_OUTPUT) -> str:
    html = render_html(build_payload())
    output.write_text(html, encoding="utf-8", newline="\n")
    return sha256(html.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"{generate(args.output)}  {args.output.name}")


if __name__ == "__main__":
    main()
