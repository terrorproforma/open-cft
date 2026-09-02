"""Generate the deterministic standalone L1a geometry-sweep dashboard.

The generator validates every sealed deterministic input before embedding a
compact, display-only projection. Runtime diagnostics and timestamps are
intentionally excluded so identical reviewed inputs produce identical bytes.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import hypot, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DEFAULT_MANIFEST = RESULTS / "manifest.json"
DEFAULT_OUTPUT = HERE / "l1a-geometry-sweep.html"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
EXPECTED_MANIFEST_FILE_SHA256 = (
    "eb73fc6916cb1166ddd4209ba9580ca57bf63ca71f4bcc8d18baf86242f6cb8d"
)
EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "e76d1b7c7bb5fa3ff2088f0f193868e9341e215a9fb4150eae6272ba8262e82d"
)
EXPECTED_DATASET_FILE_SHA256 = (
    "2fb0c19df2e575e119e3a8c1b3280c93389674b55347fdc749d8cf929dad34dd"
)
EXPECTED_DATASET_PAYLOAD_SHA256 = (
    "fb6c34ffe4baac03e2d89fba2530eb8b594ae69b41fdf42ab5af0f33c4832f95"
)
EXPECTED_REPRESENTATIVES = (
    ("strongest-centreline", "l1a-gs-065-9e98f08f3b"),
    ("strongest-mirror", "l1a-gs-032-570ad83ba6"),
    ("steepest-stage-gradient", "l1a-gs-068-375d1b1b13"),
    ("lowest-field-energy", "l1a-gs-000-48d2ccedd5"),
    ("additional-nondominated-5", "l1a-gs-005-942c2d458e"),
)
OBJECTIVES = (
    ("centreline_mid_abs_bz_t", "maximize", "T"),
    ("minimum_mirror_ratio", "maximize", "1"),
    ("stage_gradient_rms_t_per_m", "maximize", "T/m"),
    ("field_energy_j", "minimize", "J"),
)
METRICS = (
    ("centreline_mid_abs_bz_t", "Centreline |Bz|", "T"),
    ("minimum_mirror_ratio", "Minimum mirror ratio", "1"),
    ("axis_cusp_count", "Sampled cusp count", "count"),
    ("axis_null_count", "Sampled null count", "count"),
    ("stage_gradient_rms_t_per_m", "Stage gradient RMS", "T/m"),
    ("field_energy_j", "Field energy", "J"),
    ("source_representation_error", "Source error", "1"),
    ("topology_confidence", "Topology confidence", "1"),
    ("boundary_to_peak_ratio", "Boundary / peak", "1"),
)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    def no_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {path.name}")

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid readable JSON: {path.name}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be one JSON object")
    return parsed


def _closed(value: Any, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys do not match the closed schema")
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


def _verify_integrity(value: dict[str, Any], label: str) -> str:
    integrity = _closed(
        value.get("integrity"),
        f"{label}.integrity",
        {"algorithm", "canonicalization", "payload_sha256"},
    )
    if (
        integrity["algorithm"] != "sha256"
        or integrity["canonicalization"] != CANONICALIZATION
    ):
        raise ValueError(f"{label} integrity declaration is unsupported")
    digest = _digest(integrity["payload_sha256"], f"{label} payload digest")
    payload = {key: item for key, item in value.items() if key != "integrity"}
    if _canonical_hash(payload) != digest:
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return digest


def _verify_file(path: Path, label: str, expected: str | None = None) -> str:
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"{label} file is not readable: {path.name}") from error
    if expected is not None and digest != expected:
        raise ValueError(f"{label} file SHA-256 mismatch")
    sidecar = path.with_name(path.name + ".sha256")
    try:
        sidecar_text = sidecar.read_text(encoding="ascii")
    except OSError as error:
        raise ValueError(f"{label} is missing its SHA-256 sidecar") from error
    if sidecar_text != f"{digest}  {path.name}\n":
        raise ValueError(f"{label} SHA-256 sidecar is invalid")
    return digest


def _safe_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{label} must be a portable relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"{label} escapes the results directory")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the results directory")
    return path


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _matrix(
    value: Any, rows: int, columns: int, label: str
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} radial dimension mismatch")
    result: list[list[float]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{label}[{index}] axial dimension mismatch")
        result.append([_number(item, f"{label}[{index}]") for item in row])
    return result


def _validate_field(field: dict[str, Any], label: str) -> None:
    if field.get("schema_version") != "cft-axisymmetric-field-map/1.1.0":
        raise ValueError(f"{label} field schema is unsupported")
    if field.get("model_level") != "L1a":
        raise ValueError(f"{label} is not an L1a artifact")
    r_values = [_number(item, f"{label}.r_m") for item in field["field_map"]["r_m"]]
    z_values = [_number(item, f"{label}.z_m") for item in field["field_map"]["z_m"]]
    if len(r_values) < 2 or len(z_values) < 2:
        raise ValueError(f"{label} field grid is too small")
    if any(b <= a for a, b in zip(r_values, r_values[1:])) or any(
        b <= a for a, b in zip(z_values, z_values[1:])
    ):
        raise ValueError(f"{label} coordinates must be strictly increasing")
    shape = (len(r_values), len(z_values))
    maps = {
        key: _matrix(field["field_map"][key], *shape, f"{label}.{key}")
        for key in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t")
    }
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(maps["b_magnitude_t"][i][j] - hypot(
                maps["b_r_t"][i][j], maps["b_z_t"][i][j]
            )) > max(2e-15, 2e-12 * maps["b_magnitude_t"][i][j]):
                raise ValueError(f"{label} |B| component identity mismatch")
    for profile_name in ("centreline", "wall"):
        profile = field["profiles"][profile_name]
        count = len(profile["z_m"])
        if count < 2 or any(
            len(profile[key]) != count for key in ("b_r_t", "b_z_t")
        ):
            raise ValueError(f"{label} {profile_name} profile shape mismatch")
        for key in ("z_m", "b_r_t", "b_z_t"):
            for item in profile[key]:
                _number(item, f"{label}.{profile_name}.{key}")


def _case_hash(case: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            "geometry_sha256": case["geometry_sha256"],
            "source_sha256": case["source_sha256"],
            "config_sha256": case["config_sha256"],
        }
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    comparisons: list[tuple[float, float]] = []
    for name, direction, _units in OBJECTIVES:
        a, b = float(left["qois"][name]), float(right["qois"][name])
        comparisons.append((-a, -b) if direction == "minimize" else (a, b))
    return all(a >= b for a, b in comparisons) and any(a > b for a, b in comparisons)


def _field_projection(field: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "map": field["field_map"],
        "profiles": field["profiles"],
        "sources": field["input"]["sources"],
        "domain": field["input"]["domain"],
        "diagnostics": field["diagnostics"],
        "summary": field["summary"],
        "limitations": field["limitations"],
        "model_description": field["model_description"],
    }


def build_payload(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Validate the complete sealed bundle and return its display projection."""

    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    is_reviewed = manifest_path == DEFAULT_MANIFEST.resolve()
    manifest_file_hash = _verify_file(
        manifest_path,
        "manifest",
        EXPECTED_MANIFEST_FILE_SHA256 if is_reviewed else None,
    )
    manifest = _load_object(manifest_path, "manifest")
    manifest_payload_hash = _verify_integrity(manifest, "manifest")
    if is_reviewed and manifest_payload_hash != EXPECTED_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("manifest payload SHA-256 differs from the reviewed sweep")
    _closed(
        manifest,
        "manifest",
        {
            "schema_version",
            "classification",
            "dataset_payload_sha256",
            "deterministic_files",
            "representative_artifacts",
            "runtime_diagnostics",
            "integrity",
        },
    )
    if (
        manifest["schema_version"]
        != "cft-revival.experiment.l1a-geometry-sweep-manifest/1.0.0"
        or manifest["classification"]
        != "L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID"
    ):
        raise ValueError("manifest schema/classification binding mismatch")

    entries: dict[str, Mapping[str, Any]] = {}
    for index, raw_entry in enumerate(manifest["deterministic_files"]):
        entry = _closed(
            raw_entry,
            f"manifest.deterministic_files[{index}]",
            {"path", "kind", "file_sha256", "payload_sha256"},
        )
        path_text = entry["path"]
        if path_text in entries:
            raise ValueError("manifest contains duplicate deterministic paths")
        path = _safe_path(root, path_text, f"manifest path {index}")
        expected_file_hash = _digest(entry["file_sha256"], f"{path_text} file digest")
        actual_file_hash = _verify_file(path, path_text, expected_file_hash)
        if actual_file_hash != expected_file_hash:
            raise ValueError(f"manifest file hash mismatch for {path_text}")
        if entry["payload_sha256"] is not None:
            artifact = _load_object(path, path_text)
            if _verify_integrity(artifact, path_text) != _digest(
                entry["payload_sha256"], f"{path_text} payload digest"
            ):
                raise ValueError(f"manifest payload hash mismatch for {path_text}")
        entries[path_text] = entry

    dataset_entry = entries.get("dataset.json")
    if dataset_entry is None or dataset_entry["kind"] != "sealed_dataset":
        raise ValueError("manifest has no sealed dataset entry")
    dataset_path = root / "dataset.json"
    if is_reviewed and dataset_entry["file_sha256"] != EXPECTED_DATASET_FILE_SHA256:
        raise ValueError("dataset file SHA-256 differs from the reviewed sweep")
    dataset = _load_object(dataset_path, "dataset")
    dataset_payload_hash = _verify_integrity(dataset, "dataset")
    if is_reviewed and dataset_payload_hash != EXPECTED_DATASET_PAYLOAD_SHA256:
        raise ValueError("dataset payload SHA-256 differs from the reviewed sweep")
    if (
        dataset_payload_hash != manifest["dataset_payload_sha256"]
        or dataset_payload_hash != dataset_entry["payload_sha256"]
    ):
        raise ValueError("manifest/dataset payload binding mismatch")
    _closed(
        dataset,
        "dataset",
        {
            "schema_version",
            "classification",
            "model_level",
            "sampling",
            "domain",
            "solver",
            "current_equivalent_preview",
            "objectives",
            "constraints",
            "qoi_policy",
            "failure_taxonomy",
            "cases",
            "parity",
            "representatives",
            "summary",
            "limitations",
            "integrity",
        },
    )
    if (
        dataset["classification"] != manifest["classification"]
        or dataset["model_level"] != "L1a"
        or dataset["current_equivalent_preview"]["authoritative"] is not False
    ):
        raise ValueError("dataset model/classification binding mismatch")
    if [
        (item["name"], item["direction"], item["units"])
        for item in dataset["objectives"]
    ] != list(OBJECTIVES):
        raise ValueError("dataset objective directions differ from the reviewed policy")
    constraint_names = {item["name"] for item in dataset["constraints"]}
    required_constraints = {
        "boundary_to_peak_ratio",
        "relative_residual_l2",
        "flux_reconstruction_identity_t_per_m",
        "source_representation_error",
        "topology_confidence",
        "worst_case_radial_manufacturing_margin_m",
        "worst_case_axial_manufacturing_margin_m",
    }
    if constraint_names != required_constraints:
        raise ValueError("dataset constraint policy is incomplete")

    cases = dataset["cases"]
    summary = dataset["summary"]
    if not isinstance(cases, list) or len(cases) != 96:
        raise ValueError("reviewed sweep must contain exactly 96 cases")
    if (
        summary["requested_count"],
        summary["evaluated_count"],
        summary["failed_count"],
        summary["feasible_count"],
        summary["nondominated_count"],
    ) != (96, 96, 0, 96, 25):
        raise ValueError("reviewed sweep summary must be 96/96 feasible with 25 nondominated")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError(f"case {index} has an invalid or duplicate identity")
        seen.add(case_id)
        if (
            case.get("status") != "success"
            or case.get("failure") is not None
            or case.get("feasible") is not True
            or case.get("classification") != dataset["classification"]
        ):
            raise ValueError(f"{case_id} is not a successful feasible reviewed case")
        for name, _label, _units in METRICS:
            _number(case["qois"][name], f"{case_id}.{name}")
        for digest_name in (
            "design_id",
            "geometry_sha256",
            "source_sha256",
            "config_sha256",
            "case_sha256",
        ):
            _digest(case[digest_name], f"{case_id}.{digest_name}")
        if case["case_sha256"] != _case_hash(case):
            raise ValueError(f"{case_id} identity does not bind geometry/source/config")
        declared_feasible = all(
            case["constraints"][item["name"]] <= item["threshold"]
            if item["sense"] == "<="
            else case["constraints"][item["name"]] >= item["threshold"]
            for item in dataset["constraints"]
        )
        if not declared_feasible:
            raise ValueError(f"{case_id} contradicts its feasibility flag")

    computed_front = [
        case["case_id"]
        for case in cases
        if not any(
            other is not case and _dominates(other, case) for other in cases
        )
    ]
    if computed_front != summary["nondominated_case_ids"] or len(computed_front) != 25:
        raise ValueError("nondominated identities do not reproduce exact ranking")

    representatives = [
        (item["label"], item["case_id"]) for item in dataset["representatives"]
    ]
    if representatives != list(EXPECTED_REPRESENTATIVES):
        raise ValueError("dataset representative identities differ from reviewed selection")
    artifact_representatives = [
        (item["label"], item["case_id"])
        for item in manifest["representative_artifacts"]
    ]
    if artifact_representatives != representatives:
        raise ValueError("manifest representative identities disagree with dataset")

    parity_by_case: dict[str, Mapping[str, Any]] = {}
    if len(dataset["parity"]) != 6:
        raise ValueError("reviewed sweep must contain six parity records")
    for item in dataset["parity"]:
        if item.get("passed") is not True or item.get("case_id") not in seen:
            raise ValueError("parity records must be passing and case-bound")
        parity_by_case[item["case_id"]] = item

    artifact_by_case = {
        item["case_id"]: item for item in manifest["representative_artifacts"]
    }
    representative_payloads: list[dict[str, Any]] = []
    for label, case_id in representatives:
        artifact_entry = artifact_by_case[case_id]
        for kind in ("geometry", "full_field", "downsampled_field"):
            binding = artifact_entry[kind]
            manifest_entry = entries.get(binding["path"])
            if (
                manifest_entry is None
                or manifest_entry["file_sha256"] != binding["file_sha256"]
                or manifest_entry["payload_sha256"] != binding["payload_sha256"]
            ):
                raise ValueError(f"{case_id} {kind} manifest binding mismatch")
        geometry_path = _safe_path(root, artifact_entry["geometry"]["path"], "geometry")
        down_path = _safe_path(
            root, artifact_entry["downsampled_field"]["path"], "downsampled field"
        )
        geometry = _load_object(geometry_path, f"{case_id} geometry")
        field = _load_object(down_path, f"{case_id} downsampled field")
        if _verify_integrity(geometry, f"{case_id} geometry") != artifact_entry[
            "geometry"
        ]["payload_sha256"]:
            raise ValueError(f"{case_id} geometry payload identity mismatch")
        _validate_field(field, f"{case_id} field")
        if field["input"]["name"] != case_id:
            raise ValueError(f"{case_id} field input identity mismatch")
        representative_payloads.append(
            {
                "label": label,
                "case_id": case_id,
                "geometry": {
                    "chamber": geometry["chamber"],
                    "regions": geometry["regions"],
                    "stages": geometry["stages"],
                    "classification": geometry["classification"],
                },
                "field": _field_projection(field),
                "identity": {
                    "geometry_file_sha256": artifact_entry["geometry"]["file_sha256"],
                    "geometry_payload_sha256": artifact_entry["geometry"]["payload_sha256"],
                    "full_field_file_sha256": artifact_entry["full_field"]["file_sha256"],
                    "full_field_payload_sha256": artifact_entry["full_field"]["payload_sha256"],
                    "downsampled_field_file_sha256": artifact_entry[
                        "downsampled_field"
                    ]["file_sha256"],
                    "downsampled_field_payload_sha256": artifact_entry[
                        "downsampled_field"
                    ]["payload_sha256"],
                },
            }
        )

    representative_labels = {case_id: label for label, case_id in representatives}
    compact_cases = []
    front_set = set(computed_front)
    for case in cases:
        compact_cases.append(
            {
                "case_id": case["case_id"],
                "design_id": case["design_id"],
                "sampling_provenance": case["sampling_provenance"],
                "design_values": case["design_values"],
                "derived_geometry": case["derived_geometry"],
                "qois": case["qois"],
                "constraints": case["constraints"],
                "geometry_sha256": case["geometry_sha256"],
                "source_sha256": case["source_sha256"],
                "config_sha256": case["config_sha256"],
                "case_sha256": case["case_sha256"],
                "backend": case["backend"],
                "iterations": case["iterations"],
                "feasible": True,
                "nondominated": case["case_id"] in front_set,
                "representative": representative_labels.get(case["case_id"]),
                "parity": parity_by_case.get(case["case_id"]),
            }
        )

    payload = {
        "schema": "cft-revival.l1a-geometry-sweep-visualization/1.0.0",
        "warning": (
            "L1a equivalent-current field-only screening: no material-aware permanent-"
            "magnet model, plasma solution, thrust, efficiency, or hardware validity."
        ),
        "manifest": {
            "file_sha256": manifest_file_hash,
            "payload_sha256": manifest_payload_hash,
            "dataset_file_sha256": dataset_entry["file_sha256"],
            "dataset_payload_sha256": dataset_payload_hash,
        },
        "summary": summary,
        "sampling": dataset["sampling"],
        "domain": dataset["domain"],
        "solver": dataset["solver"],
        "objectives": dataset["objectives"],
        "constraints": dataset["constraints"],
        "limitations": dataset["limitations"],
        "metrics": [
            {"name": name, "label": label, "units": units}
            for name, label, units in METRICS
        ],
        "cases": compact_cases,
        "representatives": representative_payloads,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    _closed(
        payload,
        "visualization payload",
        {
            "schema",
            "warning",
            "manifest",
            "summary",
            "sampling",
            "domain",
            "solver",
            "objectives",
            "constraints",
            "limitations",
            "metrics",
            "cases",
            "representatives",
        },
    )
    if payload["schema"] != "cft-revival.l1a-geometry-sweep-visualization/1.0.0":
        raise ValueError("visualization schema is unsupported")
    identity = payload["manifest"]
    expected = (
        EXPECTED_MANIFEST_FILE_SHA256,
        EXPECTED_MANIFEST_PAYLOAD_SHA256,
        EXPECTED_DATASET_FILE_SHA256,
        EXPECTED_DATASET_PAYLOAD_SHA256,
    )
    actual = (
        identity["file_sha256"],
        identity["payload_sha256"],
        identity["dataset_file_sha256"],
        identity["dataset_payload_sha256"],
    )
    if actual != expected:
        raise ValueError("embedded manifest/dataset identity is not reviewed")
    if len(payload["cases"]) != 96 or sum(
        bool(case["nondominated"]) for case in payload["cases"]
    ) != 25:
        raise ValueError("visualization case/front count mismatch")
    if [
        (item["label"], item["case_id"]) for item in payload["representatives"]
    ] != list(EXPECTED_REPRESENTATIVES):
        raise ValueError("visualization representative identity mismatch")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L1a geometry sweep · field-only dashboard</title>
<style>
:root{color-scheme:dark;--bg:#07101d;--panel:#0d1928;--panel2:#111f31;--ink:#edf5ff;--muted:#96a9bf;--line:#263b53;--cyan:#48d5e8;--blue:#6e8cff;--gold:#ffc857;--pink:#ff6b9d;--green:#6ee7a8;--danger:#ff8f70;--shadow:0 18px 50px #0006}
:root[data-theme="light"]{color-scheme:light;--bg:#eef4f8;--panel:#fff;--panel2:#f6f9fc;--ink:#132338;--muted:#536b82;--line:#cad8e5;--cyan:#057f91;--blue:#3d56c7;--gold:#a56800;--pink:#c42968;--green:#087b4c;--danger:#aa331c;--shadow:0 18px 50px #17324a20}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 85% -10%,#21426755,transparent 35%),var(--bg);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}button,select,input{font:inherit;color:inherit;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:.45rem .6rem}button{cursor:pointer}button:hover,button:focus-visible,select:focus-visible,input:focus-visible{border-color:var(--cyan);outline:2px solid color-mix(in srgb,var(--cyan) 30%,transparent);outline-offset:2px}header,main,footer{width:min(1480px,calc(100% - 28px));margin:auto}header{padding:30px 0 18px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;color:var(--cyan);font-weight:750;font-size:.76rem}h1{font-size:clamp(2rem,5vw,4.6rem);line-height:.96;max-width:950px;margin:.35rem 0 1rem;letter-spacing:-.055em}.lede{font-size:1.05rem;color:var(--muted);max-width:880px}.warning{border:1px solid var(--danger);border-left-width:5px;background:color-mix(in srgb,var(--danger) 8%,var(--panel));padding:12px 14px;border-radius:8px;margin-top:18px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-top:15px}.cards{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin:0 0 12px}.stat,.panel{background:linear-gradient(150deg,var(--panel),color-mix(in srgb,var(--panel) 88%,var(--blue)));border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow)}.stat{padding:14px}.stat strong{font-size:1.7rem;display:block}.stat span{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}.panel{padding:14px;min-width:0}.span12{grid-column:span 12}.span8{grid-column:span 8}.span7{grid-column:span 7}.span6{grid-column:span 6}.span5{grid-column:span 5}.span4{grid-column:span 4}h2{font-size:1.08rem;margin:0 0 10px}h3{font-size:.93rem;margin:12px 0 6px}.subtle,.caption{color:var(--muted)}.controls{display:flex;gap:10px;align-items:end;flex-wrap:wrap}.controls label{display:grid;gap:4px;color:var(--muted);font-size:.8rem}.controls label select{color:var(--ink);min-width:180px}.check{display:flex!important;grid-auto-flow:column;align-items:center;gap:6px!important}.check input{width:16px;height:16px}.canvas-wrap{height:410px;min-height:260px;position:relative;margin-top:10px}.canvas-wrap.short{height:270px}.canvas-wrap.tall{height:520px}canvas{width:100%;height:100%;display:block;border-radius:9px;background:color-mix(in srgb,var(--panel2) 80%,transparent);touch-action:none}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.78rem;margin-top:8px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.filters{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}.filter{display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:8px;border:1px solid var(--line);border-radius:9px}.filter b{grid-column:1/-1;font-size:.78rem}.filter input{min-width:0;width:100%;font-variant-numeric:tabular-nums}.policy{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.policy article{background:var(--panel2);padding:10px;border-radius:9px;border:1px solid var(--line)}.policy code{color:var(--cyan)}.case-title{display:flex;justify-content:space-between;gap:8px;align-items:start}.badges{display:flex;gap:5px;flex-wrap:wrap}.badge{border:1px solid var(--line);padding:2px 7px;border-radius:99px;font-size:.7rem}.badge.front{border-color:var(--gold);color:var(--gold)}.badge.rep{border-color:var(--pink);color:var(--pink)}table{width:100%;border-collapse:collapse;font-size:.78rem}th,td{text-align:left;vertical-align:top;padding:5px 7px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}td:last-child{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.scroll{max-height:420px;overflow:auto}.hash{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.72rem;overflow-wrap:anywhere}.field-head{display:flex;gap:8px;justify-content:space-between;align-items:end;flex-wrap:wrap}.field-head label{display:grid;gap:4px;color:var(--muted);font-size:.78rem}.limits li{margin:.3rem 0}.sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}footer{padding:22px 0 40px;color:var(--muted);font-size:.8rem}
@media(max-width:960px){.span8,.span7,.span6,.span5,.span4{grid-column:span 12}.cards{grid-template-columns:repeat(2,1fr)}.filters{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){header,main,footer{width:min(100% - 16px,1480px)}.cards,.filters,.policy{grid-template-columns:1fr}.canvas-wrap{height:330px}.canvas-wrap.tall{height:390px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style>
</head>
<body>
<header>
<div class="eyebrow">Deterministic evidence viewer · 96-case field sweep</div>
<h1>Geometry changes the field. Nothing more is claimed.</h1>
<p class="lede">Linked objective-space, topology, field-map and source-geometry views for the reviewed L1a axisymmetric sweep. Every displayed value is embedded after strict hash validation.</p>
<div class="warning" role="note"><strong>Field-only boundary:</strong> <span id="warning"></span></div>
<div class="toolbar"><button id="reset" aria-keyshortcuts="Escape">Reset view</button><button id="theme" aria-pressed="false">Switch to light</button><span id="status" role="status" aria-live="polite"></span></div>
</header>
<main>
<section class="cards" aria-label="Sweep summary">
<div class="stat"><strong id="evaluated"></strong><span>evaluated</span></div><div class="stat"><strong id="feasible"></strong><span>feasible</span></div><div class="stat"><strong id="frontCount"></strong><span>nondominated</span></div><div class="stat"><strong id="visibleCount"></strong><span>visible after filters</span></div><div class="stat"><strong>6 / 6</strong><span>CPU/CUDA parity gates</span></div>
</section>
<div class="grid">
<section class="panel span8" aria-labelledby="scatterTitle"><h2 id="scatterTitle">Linked scatter view</h2><div class="controls"><label>X axis<select id="xAxis"></select></label><label>Y axis<select id="yAxis"></select></label><label class="check"><input id="frontOnly" type="checkbox">Nondominated only</label><label class="check"><input id="repsOnly" type="checkbox">Named representatives only</label></div><div class="canvas-wrap"><canvas id="scatter" tabindex="0" role="img" aria-label="Interactive linked scatter plot. Use arrow keys, Home and End to select visible cases."></canvas></div><div class="legend"><span><i class="dot" style="background:var(--gold)"></i>nondominated</span><span><i class="dot" style="background:var(--pink)"></i>named representative</span><span><i class="dot" style="background:var(--cyan)"></i>selected</span><span>Mouse: hover/select · Keyboard: arrows/Home/End</span></div></section>
<aside class="panel span4" aria-labelledby="caseTitle"><div class="case-title"><h2 id="caseTitle">Selected case</h2><div class="badges" id="badges"></div></div><div class="scroll" id="caseDetails"></div></aside>
<section class="panel span12" aria-labelledby="filterTitle"><h2 id="filterTitle">Explicit metric filters</h2><p class="subtle">Inclusive lower/upper bounds. Counts are sampled topology descriptors, not continuous critical-point proofs.</p><div class="filters" id="filters"></div></section>
<section class="panel span8" aria-labelledby="parallelTitle"><h2 id="parallelTitle">Linked parallel coordinates</h2><p class="subtle">All nine requested field, topology, source-transfer and boundary dimensions; axes normalize only for display.</p><div class="canvas-wrap"><canvas id="parallel" tabindex="0" role="img" aria-label="Parallel-coordinate view linked to the selected scatter case."></canvas></div></section>
<section class="panel span4" aria-labelledby="policyTitle"><h2 id="policyTitle">Objective and constraint directions</h2><div id="policies"></div></section>
<section class="panel span7" aria-labelledby="fieldTitle"><div class="field-head"><div><h2 id="fieldTitle">Representative field map and ψ contours</h2><p class="subtle">Radial-major sampled grid. Geometry outlines and equivalent-current source bands are overlaid.</p></div><label>Representative<select id="representative"></select></label><label>Component<select id="component"><option value="b_magnitude_t">|B| (T)</option><option value="b_z_t">Bz (T)</option><option value="b_r_t">Br (T)</option><option value="psi_wb">ψ (Wb)</option></select></label></div><div class="canvas-wrap tall"><canvas id="field" tabindex="0" role="img" aria-label="Representative axisymmetric field raster with flux contours, geometry and equivalent-current source bands."></canvas></div><div class="legend"><span><i class="dot" style="background:var(--gold)"></i>positive source</span><span><i class="dot" style="background:var(--pink)"></i>negative source</span><span>thin lines: sampled ψ contours</span></div></section>
<section class="panel span5" aria-labelledby="profileTitle"><h2 id="profileTitle">Centreline and wall Bz profiles</h2><div class="canvas-wrap short"><canvas id="profile" tabindex="0" role="img" aria-label="Representative centreline and wall axial field profiles."></canvas></div><h3>Representative identity</h3><div class="scroll hash" id="repIdentity"></div><h3>Field limitations</h3><ul class="limits" id="fieldLimits"></ul></section>
<section class="panel span12" aria-labelledby="limitTitle"><h2 id="limitTitle">Interpretation limits</h2><ul class="limits" id="limitations"></ul><p class="caption">The finite-box boundary ratio, source transfer, residual, topology confidence and manufacturing margins are screening gates. Passing them does not establish material, plasma, propulsion, thermal, structural, lifetime, procurement, or build validity.</p></section>
</div>
</main>
<footer>Self-contained offline HTML · no network requests · no per-point DOM · deterministic projection of hash-sealed evidence.</footer>
<script id="sweep-data" type="application/json">__DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("sweep-data").textContent),$=id=>document.getElementById(id),metrics=D.metrics,cases=D.cases,reps=D.representatives;
const colors={bg:"#101d2c",grid:"#6b829744",text:"#a9bbce",base:"#5f7893",front:"#ffc857",rep:"#ff6b9d",selected:"#48d5e8",positive:"#ffc857",negative:"#ff6b9d"};
const INITIAL=Object.freeze({x:0,y:1,frontOnly:false,repsOnly:false,selected:0,representative:0,component:"b_magnitude_t"});
let state={...INITIAL},bounds={},visible=[],hover=-1,dark=true,pendingFrame=0,scatterLayout=null;
function fmt(v){if(v===null||v===undefined)return "—";if(typeof v!=="number")return String(v);if(v===0)return "0";const a=Math.abs(v);return a>=1000||a<.001?v.toExponential(4):v.toPrecision(6)}
function metricValue(c,m){return c.qois[m.name]}
function setup(){
 $("warning").textContent=D.warning;$("evaluated").textContent=D.summary.evaluated_count;$("feasible").textContent=D.summary.feasible_count;$("frontCount").textContent=D.summary.nondominated_count;
 metrics.forEach((m,i)=>{for(const id of ["xAxis","yAxis"]){const o=document.createElement("option");o.value=String(i);o.textContent=`${m.label} [${m.units}]`;$(id).append(o)}const values=cases.map(c=>metricValue(c,m)),lo=Math.min(...values),hi=Math.max(...values);bounds[m.name]=[lo,hi];const row=document.createElement("div");row.className="filter";row.innerHTML=`<b>${m.label} <span class="subtle">[${m.units}]</span></b><label><span class="sr">Minimum ${m.label}</span><input data-metric="${m.name}" data-side="0" type="number" step="any"></label><label><span class="sr">Maximum ${m.label}</span><input data-metric="${m.name}" data-side="1" type="number" step="any"></label>`;$("filters").append(row)});
 reps.forEach((r,i)=>{const o=document.createElement("option");o.value=String(i);o.textContent=`${r.label} · ${r.case_id}`;$("representative").append(o)});
 document.querySelectorAll("#filters input").forEach(input=>input.addEventListener("change",()=>{const m=input.dataset.metric,side=Number(input.dataset.side);bounds[m][side]=Number(input.value);filterAndDraw()}));
 $("xAxis").onchange=()=>{state.x=Number($("xAxis").value);schedule()};$("yAxis").onchange=()=>{state.y=Number($("yAxis").value);schedule()};
 $("frontOnly").onchange=()=>{state.frontOnly=$("frontOnly").checked;filterAndDraw()};$("repsOnly").onchange=()=>{state.repsOnly=$("repsOnly").checked;filterAndDraw()};
 $("representative").onchange=()=>{state.representative=Number($("representative").value);const id=reps[state.representative].case_id,idx=cases.findIndex(c=>c.case_id===id);if(idx>=0)selectCase(idx);schedule()};
 $("component").onchange=()=>{state.component=$("component").value;schedule()};
 $("reset").onclick=reset;$("theme").onclick=toggleTheme;window.addEventListener("keydown",e=>{if(e.key==="Escape"){e.preventDefault();reset()}});
 for(const id of ["scatter","parallel"]){$(id).addEventListener("keydown",navigateCases)}
 $("scatter").addEventListener("pointermove",scatterPointer);$("scatter").addEventListener("pointerleave",()=>{hover=-1;drawScatter()});$("scatter").addEventListener("click",e=>{const i=nearestScatter(e);if(i>=0)selectCase(i)});
 new ResizeObserver(schedule).observe(document.querySelector("main"));window.addEventListener("pageshow",schedule);
 renderPolicies();reset();
}
function reset(){state={...INITIAL};$("xAxis").value="0";$("yAxis").value="1";$("frontOnly").checked=false;$("repsOnly").checked=false;$("representative").value="0";$("component").value=INITIAL.component;metrics.forEach(m=>{const values=cases.map(c=>metricValue(c,m));bounds[m.name]=[Math.min(...values),Math.max(...values)]});syncFilterInputs();filterAndDraw();selectCase(0)}
function syncFilterInputs(){document.querySelectorAll("#filters input").forEach(input=>{input.value=String(bounds[input.dataset.metric][Number(input.dataset.side)])})}
function filterAndDraw(){visible=cases.map((c,i)=>[c,i]).filter(([c])=>{if(state.frontOnly&&!c.nondominated)return false;if(state.repsOnly&&!c.representative)return false;return metrics.every(m=>{const v=metricValue(c,m),b=bounds[m.name];return v>=b[0]&&v<=b[1]})}).map(([,i])=>i);$("visibleCount").textContent=visible.length;$("status").textContent=`${visible.length} of ${cases.length} cases visible`;schedule()}
function selectCase(i){state.selected=i;const c=cases[i];if(c.representative){const ri=reps.findIndex(r=>r.case_id===c.case_id);if(ri>=0){state.representative=ri;$("representative").value=String(ri)}}updateCase();schedule()}
function navigateCases(e){if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Home","End"].includes(e.key))return;e.preventDefault();if(!visible.length)return;let p=visible.indexOf(state.selected);if(e.key==="Home")p=0;else if(e.key==="End")p=visible.length-1;else p=Math.max(0,Math.min(visible.length-1,(p<0?0:p)+(["ArrowRight","ArrowDown"].includes(e.key)?1:-1)));selectCase(visible[p])}
function updateCase(){const c=cases[state.selected];$("badges").innerHTML=`<span class="badge">feasible</span>${c.nondominated?'<span class="badge front">nondominated</span>':""}${c.representative?`<span class="badge rep">${c.representative}</span>`:""}`;let html=`<h3>${c.case_id}</h3><table><tr><th>Sampling</th><td>${c.sampling_provenance}</td></tr><tr><th>Backend / iterations</th><td>${c.backend} / ${c.iterations}</td></tr></table><h3>Full inputs</h3>${objectTable(c.design_values)}<h3>Field outputs</h3>${objectTable(Object.fromEntries(metrics.map(m=>[`${m.label} [${m.units}]`,metricValue(c,m)])))}<h3>Residual and gates</h3>${gateTable(c)}<h3>Identities</h3>${objectTable({design_sha256:c.design_id,geometry_sha256:c.geometry_sha256,source_sha256:c.source_sha256,config_sha256:c.config_sha256,case_sha256:c.case_sha256})}<h3>Derived geometry</h3>${objectTable(c.derived_geometry)}<h3>CPU/CUDA parity</h3>${c.parity?objectTable(c.parity):'<p class="subtle">Not one of the six distributed parity cases.</p>'}`;$("caseDetails").innerHTML=html}
function objectTable(obj){return `<table>${Object.entries(obj).map(([k,v])=>`<tr><th>${k}</th><td>${typeof v==="object"&&v!==null?escapeText(JSON.stringify(v)):fmt(v)}</td></tr>`).join("")}</table>`}
function escapeText(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function gateTable(c){const defs=Object.fromEntries(D.constraints.map(x=>[x.name,x]));return `<table>${Object.entries(c.constraints).map(([k,v])=>{const d=defs[k],pass=d.sense==="<="?v<=d.threshold:v>=d.threshold;return `<tr><th>${k}</th><td>${fmt(v)} ${d.sense} ${fmt(d.threshold)} ${d.units} · ${pass?"PASS":"FAIL"}</td></tr>`}).join("")}</table>`}
function renderPolicies(){$("policies").innerHTML=`<h3>Objectives</h3><div class="policy">${D.objectives.map(o=>`<article><code>${o.name}</code><br><strong>${o.direction==="maximize"?"↑ maximize":"↓ minimize"}</strong> [${o.units}]</article>`).join("")}</div><h3>Feasibility constraints</h3><div class="policy">${D.constraints.map(o=>`<article><code>${o.name}</code><br><strong>${o.sense} ${fmt(o.threshold)}</strong> [${o.units}]</article>`).join("")}</div>`;$("limitations").innerHTML=D.limitations.map(x=>`<li>${x}</li>`).join("")}
function fitCanvas(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.min(3,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);return [ctx,r.width,r.height]}
function themeColors(){const css=getComputedStyle(document.documentElement);colors.text=css.getPropertyValue("--muted").trim();colors.grid=css.getPropertyValue("--line").trim();colors.base=dark?"#6f87a0":"#7890a8";colors.front=css.getPropertyValue("--gold").trim();colors.rep=css.getPropertyValue("--pink").trim();colors.selected=css.getPropertyValue("--cyan").trim()}
function schedule(){cancelAnimationFrame(pendingFrame);pendingFrame=requestAnimationFrame(drawAll)}
function drawAll(){themeColors();drawScatter();drawParallel();drawField();drawProfile();updateRepresentativeText()}
function scales(values,pad=.08){let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=1;hi+=1}const d=(hi-lo)*pad;return [lo-d,hi+d]}
function axes(ctx,w,h,xlabel,ylabel){ctx.strokeStyle=colors.grid;ctx.fillStyle=colors.text;ctx.lineWidth=1;ctx.font="12px system-ui";ctx.beginPath();ctx.moveTo(54,16);ctx.lineTo(54,h-42);ctx.lineTo(w-16,h-42);ctx.stroke();ctx.fillText(xlabel,Math.max(58,w-190),h-13);ctx.save();ctx.translate(16,Math.min(h-50,170));ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,0,0);ctx.restore()}
function drawScatter(){const [ctx,w,h]=fitCanvas($("scatter"));ctx.clearRect(0,0,w,h);const xm=metrics[state.x],ym=metrics[state.y],xs=scales(cases.map(c=>metricValue(c,xm))),ys=scales(cases.map(c=>metricValue(c,ym))),px=v=>54+(v-xs[0])/(xs[1]-xs[0])*(w-70),py=v=>h-42-(v-ys[0])/(ys[1]-ys[0])*(h-58);axes(ctx,w,h,`${xm.label} [${xm.units}]`,`${ym.label} [${ym.units}]`);scatterLayout={px,py,points:new Map};for(const i of visible){const c=cases[i],x=px(metricValue(c,xm)),y=py(metricValue(c,ym));scatterLayout.points.set(i,[x,y]);ctx.beginPath();ctx.arc(x,y,c.representative?5:c.nondominated?4:3,0,Math.PI*2);ctx.fillStyle=c.representative?colors.rep:c.nondominated?colors.front:colors.base;ctx.globalAlpha=i===state.selected||i===hover?1:.74;ctx.fill();if(i===state.selected||i===hover){ctx.strokeStyle=colors.selected;ctx.lineWidth=2;ctx.stroke()}}ctx.globalAlpha=1;if(!visible.length){ctx.fillStyle=colors.text;ctx.fillText("No cases match the filters.",70,70)}}
function nearestScatter(e){if(!scatterLayout)return-1;const r=$("scatter").getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=-1,dist=100;for(const [i,p] of scatterLayout.points){const d=(p[0]-x)**2+(p[1]-y)**2;if(d<dist){dist=d;best=i}}return best}
function scatterPointer(e){const i=nearestScatter(e);if(i!==hover){hover=i;drawScatter();$("scatter").setAttribute("aria-label",i<0?"Interactive linked scatter plot":`Hover ${cases[i].case_id}`)}}
function drawParallel(){const [ctx,w,h]=fitCanvas($("parallel"));ctx.clearRect(0,0,w,h);const left=40,right=w-18,top=28,bottom=h-42,step=(right-left)/(metrics.length-1),ranges=metrics.map(m=>scales(cases.map(c=>metricValue(c,m)),0));ctx.font="10px system-ui";ctx.textAlign="center";metrics.forEach((m,j)=>{const x=left+j*step;ctx.strokeStyle=colors.grid;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle=colors.text;const words=m.label.split(" ");ctx.fillText(words.slice(0,2).join(" "),x,h-25);ctx.fillText(words.slice(2).join(" "),x,h-13)});const order=visible.filter(i=>i!==state.selected).concat(visible.includes(state.selected)?[state.selected]:[]);for(const i of order){const c=cases[i];ctx.beginPath();metrics.forEach((m,j)=>{const [lo,hi]=ranges[j],x=left+j*step,y=bottom-(metricValue(c,m)-lo)/(hi-lo)*(bottom-top);j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=i===state.selected?colors.selected:c.representative?colors.rep:c.nondominated?colors.front:colors.base;ctx.globalAlpha=i===state.selected?1:.16;ctx.lineWidth=i===state.selected?2.5:1;ctx.stroke()}ctx.globalAlpha=1}
function mapColor(t,signed){t=Math.max(0,Math.min(1,t));if(signed){const r=t<.5?Math.round(50+180*(1-2*t)):Math.round(50+205*(2*t-1)),g=Math.round(65+120*(1-Math.abs(2*t-1))),b=t<.5?Math.round(90+165*(1-2*t)):Math.round(90-50*(2*t-1));return[r,g,b,255]}return[Math.round(12+240*t),Math.round(42+180*Math.sqrt(t)),Math.round(80+120*(1-t)),255]}
function drawField(){const rep=reps[state.representative],[ctx,w,h]=fitCanvas($("field"));ctx.clearRect(0,0,w,h);const f=rep.field.map,arr=f[state.component],rs=f.r_m,zs=f.z_m,nr=rs.length,nz=zs.length,flat=arr.flat(),lo=Math.min(...flat),hi=Math.max(...flat),signed=lo<0,off=document.createElement("canvas");off.width=nz;off.height=nr;const oc=off.getContext("2d"),img=oc.createImageData(nz,nr);for(let i=0;i<nr;i++)for(let j=0;j<nz;j++){const t=(arr[i][j]-lo)/(hi-lo||1),c=mapColor(t,signed),p=((nr-1-i)*nz+j)*4;img.data.set(c,p)}oc.putImageData(img,0,0);const p={l:54,r:18,t:20,b:44},dw=w-p.l-p.r,dh=h-p.t-p.b;ctx.imageSmoothingEnabled=true;ctx.drawImage(off,p.l,p.t,dw,dh);const x=z=>p.l+(z-zs[0])/(zs.at(-1)-zs[0])*dw,y=r=>p.t+dh-(r-rs[0])/(rs.at(-1)-rs[0])*dh;ctx.save();ctx.beginPath();ctx.rect(p.l,p.t,dw,dh);ctx.clip();const psi=f.psi_wb,pflat=psi.flat(),plo=Math.min(...pflat),phi=Math.max(...pflat);for(let k=1;k<9;k++){const level=plo+(phi-plo)*k/9;ctx.strokeStyle="#ffffff88";ctx.lineWidth=.7;ctx.beginPath();for(const seg of contourSegments(psi,rs,zs,level)){ctx.moveTo(x(seg[0]),y(seg[1]));ctx.lineTo(x(seg[2]),y(seg[3]))}ctx.stroke()}for(const region of rep.geometry.regions){ctx.strokeStyle=region.role==="permanent_magnet"?"#ffffffcc":"#ffffff66";ctx.lineWidth=region.role==="permanent_magnet"?1.2:.7;const points=[[region.z_min_m,region.r_inner_start_m],[region.z_max_m,region.r_inner_end_m],[region.z_max_m,region.r_outer_end_m],[region.z_min_m,region.r_outer_start_m]];ctx.beginPath();points.forEach((q,i)=>i?ctx.lineTo(x(q[0]),y(q[1])):ctx.moveTo(x(q[0]),y(q[1])));ctx.closePath();ctx.stroke()}for(const s of rep.field.sources){ctx.fillStyle=s.polarity>0?colors.positive:colors.negative;ctx.globalAlpha=.48;ctx.fillRect(x(s.z_min_m),y(s.r_outer_m),Math.max(1,x(s.z_max_m)-x(s.z_min_m)),Math.max(1,y(s.r_inner_m)-y(s.r_outer_m)))}ctx.restore();ctx.globalAlpha=1;ctx.strokeStyle=colors.grid;ctx.strokeRect(p.l,p.t,dw,dh);ctx.fillStyle=colors.text;ctx.font="12px system-ui";ctx.fillText(`z [m] · ${state.component}: ${fmt(lo)} … ${fmt(hi)}`,p.l,h-14);ctx.save();ctx.translate(16,h/2);ctx.rotate(-Math.PI/2);ctx.fillText("r [m]",0,0);ctx.restore()}
function interp(a,b,level){const d=b[2]-a[2],t=d===0?.5:(level-a[2])/d;return[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]}
function contourSegments(v,rs,zs,level){const out=[];for(let i=0;i<rs.length-1;i++)for(let j=0;j<zs.length-1;j++){const q=[[zs[j],rs[i],v[i][j]],[zs[j+1],rs[i],v[i][j+1]],[zs[j+1],rs[i+1],v[i+1][j+1]],[zs[j],rs[i+1],v[i+1][j]]],hits=[];for(let e=0;e<4;e++){const a=q[e],b=q[(e+1)%4];if((a[2]<level&&b[2]>=level)||(b[2]<level&&a[2]>=level))hits.push(interp(a,b,level))}if(hits.length===2)out.push([...hits[0],...hits[1]]);else if(hits.length===4){out.push([...hits[0],...hits[1]],[...hits[2],...hits[3]])}}return out}
function drawProfile(){const rep=reps[state.representative],[ctx,w,h]=fitCanvas($("profile"));ctx.clearRect(0,0,w,h);const a=rep.field.profiles.centreline,b=rep.field.profiles.wall,vals=a.b_z_t.concat(b.b_z_t),range=scales(vals),zs=a.z_m,p={l:48,r:14,t:18,b:38},x=z=>p.l+(z-zs[0])/(zs.at(-1)-zs[0])*(w-p.l-p.r),y=v=>h-p.b-(v-range[0])/(range[1]-range[0])*(h-p.t-p.b);axes(ctx,w,h,"z [m]","Bz [T]");if(range[0]<0&&range[1]>0){ctx.strokeStyle=colors.grid;ctx.beginPath();ctx.moveTo(p.l,y(0));ctx.lineTo(w-p.r,y(0));ctx.stroke()}[[a,colors.selected,"centreline"],[b,colors.front,"wall"]].forEach(([s,c])=>{ctx.beginPath();s.z_m.forEach((z,i)=>i?ctx.lineTo(x(z),y(s.b_z_t[i])):ctx.moveTo(x(z),y(s.b_z_t[i])));ctx.strokeStyle=c;ctx.lineWidth=1.6;ctx.stroke()});ctx.fillStyle=colors.selected;ctx.fillText("centreline",60,18);ctx.fillStyle=colors.front;ctx.fillText("wall",140,18)}
function updateRepresentativeText(){const r=reps[state.representative];$("repIdentity").innerHTML=objectTable(r.identity);$("fieldLimits").innerHTML=r.field.limitations.map(x=>`<li>${x}</li>`).join("");$("field").setAttribute("aria-label",`${r.label} ${state.component} field map with flux contours and ${r.field.sources.length} sampled source bands`)}
function toggleTheme(){dark=!dark;document.documentElement.dataset.theme=dark?"dark":"light";$("theme").textContent=dark?"Switch to light":"Switch to dark";$("theme").setAttribute("aria-pressed",String(!dark));schedule()}
setup();
</script>
</body>
</html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", encoded)


def generate(
    output_path: Path = DEFAULT_OUTPUT, manifest_path: Path = DEFAULT_MANIFEST
) -> str:
    html = render_html(build_payload(manifest_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8", newline="\n")
    return sha256(html.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(generate(args.output, args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
