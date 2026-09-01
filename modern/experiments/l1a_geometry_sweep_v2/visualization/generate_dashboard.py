"""Generate the committed preregistered L1a geometry-sweep v2 dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
from math import hypot, isfinite
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
RESULTS = EXPERIMENT / "results"
PROTOCOL_PATH = EXPERIMENT / "protocol.json"
DEFAULT_OUTPUT = HERE / "l1a-geometry-sweep-v2.html"
TEMPLATE_PATH = HERE / "dashboard.template.html"
REPO = HERE.parents[3]
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"

PREREGISTRATION_COMMIT_SHA = "092f5fae692ee7d6711e0c7e1c94dac6a345f37c"
RESULTS_COMMIT_SHA = "f30cb42ec4a8633bf634a3d32ffa5b11f66be97a"
PREREGISTRATION_COMMIT_TIME = "2026-09-02T03:03:16+10:00"
RESULTS_COMMIT_TIME = "2026-09-02T03:06:19+10:00"
EXPECTED_PROTOCOL_FILE_SHA256 = (
    "64b2c58c3cecb2ea1836d2bf48e23ff83dffb114866bf21e7135b411beaa2b2c"
)
EXPECTED_PROTOCOL_PAYLOAD_SHA256 = (
    "da319f2271d56b0d0c883b76d3106b094359a608b560d58ac7801de1293ecbc8"
)
EXPECTED_MANIFEST_FILE_SHA256 = (
    "768b345e946a45e623f83aaa18e01f8ec5bc7f823e81858a0a8c3a3e2e448754"
)
EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "1ba8c3ed4da694afe8f660c9083e2ddbb4e8621f1768c4c88794e65909030e92"
)
EXPECTED_RAW_FILE_SHA256 = (
    "76f145f816a187c36c3260c184a07d8979c244999f40c2562f4207dbca90b4c1"
)
EXPECTED_RAW_PAYLOAD_SHA256 = (
    "c53fdbf1cb1cb1d90447ceccf0f2e2f81a03a2096446a4509f537f27be0712fa"
)
EXPECTED_SUMMARY_FILE_SHA256 = (
    "942852295d3b2ee04f968734a31e15694e46d389838259a01113ae090ad30e29"
)
EXPECTED_SUMMARY_PAYLOAD_SHA256 = (
    "2b7643890a365e6b2f86597f30f138dcbf8f58baf754fb0f36e6a27dfe3f09dc"
)
EXPECTED_LOCK_FILE_SHA256 = (
    "23e581d549c4cee64267d379e969b1eb99a5f00203e17fe50c7b05a9a864b9c1"
)
EXPECTED_LOCK_PAYLOAD_SHA256 = (
    "c49e4e20d515d7bbd5f9407cd1f41bd76506b86f139730932b318888a62f49a3"
)
EXPECTED_ROLES = (
    ("strongest-centreline", "l1a-gs-v2-065-9e98f08f3b"),
    ("strongest-mirror", "l1a-gs-v2-032-570ad83ba6"),
    ("steepest-stage-gradient", "l1a-gs-v2-068-375d1b1b13"),
    ("lowest-field-energy", "l1a-gs-v2-000-48d2ccedd5"),
    ("best-boundary-isolation", "l1a-gs-v2-000-48d2ccedd5"),
)
EXPECTED_UNIQUE_REPRESENTATIVES = (
    "l1a-gs-v2-000-48d2ccedd5",
    "l1a-gs-v2-032-570ad83ba6",
    "l1a-gs-v2-065-9e98f08f3b",
    "l1a-gs-v2-068-375d1b1b13",
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
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {path.name}")

    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid readable JSON") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain one object")
    return loaded


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
        raise ValueError(f"{label} file is not readable") from error
    if expected is not None and digest != expected:
        raise ValueError(f"{label} file SHA-256 mismatch")
    sidecar = path.with_name(path.name + ".sha256")
    try:
        sidecar_text = sidecar.read_text(encoding="ascii")
    except OSError as error:
        raise ValueError(f"{label} SHA-256 sidecar is missing") from error
    if sidecar_text != f"{digest}  {path.name}\n":
        raise ValueError(f"{label} SHA-256 sidecar is invalid")
    return digest


def _safe_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or "\\" in raw:
        raise ValueError(f"{label} must be a portable relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"{label} escapes the result directory")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the result directory")
    return path


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPO,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if completed.returncode:
        stderr = (
            completed.stderr.decode("utf-8", "replace")
            if binary
            else completed.stderr
        )
        raise ValueError(f"Git evidence check failed: {stderr.strip()}")
    return completed.stdout


def _verify_git_clean(commit: str, *relative_paths: str) -> None:
    completed = subprocess.run(
        ("git", "diff", "--quiet", commit, "--", *relative_paths),
        cwd=REPO,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"working evidence differs from commit {commit}: "
            + ", ".join(relative_paths)
        )


def _verify_temporal_git_evidence() -> dict[str, str]:
    parents = str(_git("rev-list", "--parents", "-n", "1", RESULTS_COMMIT_SHA)).strip()
    if parents != f"{RESULTS_COMMIT_SHA} {PREREGISTRATION_COMMIT_SHA}":
        raise ValueError("results commit is not the direct child of preregistration")
    expected_meta = (
        (
            PREREGISTRATION_COMMIT_SHA,
            PREREGISTRATION_COMMIT_TIME,
            "preregister L1a geometry sweep v2",
        ),
        (
            RESULTS_COMMIT_SHA,
            RESULTS_COMMIT_TIME,
            "record L1a geometry sweep v2 results",
        ),
    )
    for commit, expected_time, expected_subject in expected_meta:
        actual = str(_git("show", "-s", "--format=%H%n%aI%n%s", commit)).splitlines()
        if actual != [commit, expected_time, expected_subject]:
            raise ValueError(f"committed temporal metadata mismatch for {commit}")
    prereg_rel = "modern/experiments/l1a_geometry_sweep_v2/protocol.json"
    _verify_git_clean(
        PREREGISTRATION_COMMIT_SHA,
        prereg_rel,
        prereg_rel + ".sha256",
    )
    return {
        "preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
        "preregistration_commit_time": PREREGISTRATION_COMMIT_TIME,
        "results_commit_sha": RESULTS_COMMIT_SHA,
        "results_commit_time": RESULTS_COMMIT_TIME,
        "relationship": "results commit is the direct child of preregistration",
    }


def _verify_committed_result(path: Path) -> None:
    relative = path.relative_to(REPO).as_posix()
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_relative = sidecar.relative_to(REPO).as_posix()
    _verify_git_clean(RESULTS_COMMIT_SHA, relative, sidecar_relative)


def _matrix(value: Any, nr: int, nz: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != nr:
        raise ValueError(f"{label} radial dimension mismatch")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != nz:
            raise ValueError(f"{label}[{index}] axial dimension mismatch")
        result.append([_number(item, label) for item in row])
    return result


def _validate_field(field: dict[str, Any], label: str) -> None:
    if (
        field.get("schema_version") != "cft-axisymmetric-field-map/1.1.0"
        or field.get("model_level") != "L1a"
    ):
        raise ValueError(f"{label} field schema/model mismatch")
    r_values = [_number(item, f"{label}.r") for item in field["field_map"]["r_m"]]
    z_values = [_number(item, f"{label}.z") for item in field["field_map"]["z_m"]]
    if (
        len(r_values) < 2
        or len(z_values) < 2
        or any(b <= a for a, b in zip(r_values, r_values[1:]))
        or any(b <= a for a, b in zip(z_values, z_values[1:]))
    ):
        raise ValueError(f"{label} coordinates are invalid")
    nr, nz = len(r_values), len(z_values)
    maps = {
        key: _matrix(field["field_map"][key], nr, nz, f"{label}.{key}")
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
        if count < 2 or any(len(profile[key]) != count for key in ("b_r_t", "b_z_t")):
            raise ValueError(f"{label} {name} profile shape mismatch")


def _comparison(left: float, right: float, objective: Mapping[str, Any]) -> int:
    tolerance = max(
        float(objective["absolute_tolerance"]),
        float(objective["relative_tolerance"])
        * max(abs(left), abs(right), 1e-300),
    )
    if objective["direction"] == "minimize":
        left, right = -left, -right
    if left > right + tolerance:
        return 1
    if right > left + tolerance:
        return -1
    return 0


def _front(cases: Sequence[Mapping[str, Any]], objectives: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered = sorted(cases, key=lambda case: case["case_id"])

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        comparisons = [
            _comparison(
                float(left["qois"][item["name"]]),
                float(right["qois"][item["name"]]),
                item,
            )
            for item in objectives
        ]
        return all(value >= 0 for value in comparisons) and any(
            value > 0 for value in comparisons
        )

    return [
        candidate["case_id"]
        for candidate in ordered
        if not any(
            other["case_id"] != candidate["case_id"] and dominates(other, candidate)
            for other in ordered
        )
    ]


def _roles(
    cases: Sequence[Mapping[str, Any]],
    front_ids: Sequence[str],
    protocol: Mapping[str, Any],
) -> list[dict[str, str]]:
    by_id = {case["case_id"]: case for case in cases}
    front = [by_id[case_id] for case_id in front_ids]
    result = []
    for definition in protocol["representative_policy"]["roles"]:
        ordered = sorted(front, key=lambda case: case["case_id"])
        select = max if definition["selection"] == "maximum" else min
        selected = select(
            ordered, key=lambda case: float(case["qois"][definition["qoi"]])
        )
        result.append({"role": definition["role"], "case_id": selected["case_id"]})
    return result


def _gates(
    cases: Sequence[Mapping[str, Any]],
    parity: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = []
    parity_by_id = {item["case_id"]: item for item in parity}
    for definition in protocol["terminal_acceptance"]["gates"]:
        gate_id = definition["gate_id"]
        failed: list[str] = []
        values: list[float] = []
        observed: Any
        if gate_id == "cpu_cuda_parity":
            observed = {"psi": 0.0, "br": 0.0, "bz": 0.0}
            for index in protocol["execution"]["parity_case_indices"]:
                case_id = cases[index]["case_id"]
                item = parity_by_id.get(case_id)
                if item is None:
                    failed.append(case_id)
                    continue
                differences = item["differences"]
                for component, source_key in (
                    ("psi", "psi_scale_relative"),
                    ("br", "br_scale_relative"),
                    ("bz", "bz_scale_relative"),
                ):
                    value = float(differences[source_key])
                    observed[component] = max(observed[component], value)
                    if value > definition["limits"][component]:
                        failed.append(case_id)
        else:
            for case in cases:
                if gate_id == "manufacturability":
                    value = min(
                        case["derived_geometry"][
                            "worst_case_radial_manufacturing_margin_m"
                        ],
                        case["derived_geometry"][
                            "worst_case_axial_manufacturing_margin_m"
                        ],
                    )
                else:
                    value = float(case["qois"][definition["metric"]])
                values.append(value)
                if (
                    definition["comparator"] == "<=" and value > definition["limit"]
                ) or (
                    definition["comparator"] == ">=" and value < definition["limit"]
                ):
                    failed.append(case["case_id"])
            observed = (
                max(values) if definition["aggregation"] == "maximum" else min(values)
            )
        results.append(
            {
                "gate_id": gate_id,
                "failed_case_ids": sorted(set(failed)),
                "failure_count": len(set(failed)),
                "passed": not failed,
                "observed": observed,
            }
        )
    return results


def _field_projection(field: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "map": field["field_map"],
        "profiles": field["profiles"],
        "sources": field["input"]["sources"],
        "diagnostics": field["diagnostics"],
        "summary": field["summary"],
        "limitations": field["limitations"],
        "model_description": field["model_description"],
    }


def build_payload() -> dict[str, Any]:
    git_evidence = _verify_temporal_git_evidence()
    protocol_file_hash = _verify_file(
        PROTOCOL_PATH, "protocol", EXPECTED_PROTOCOL_FILE_SHA256
    )
    protocol = _load_object(PROTOCOL_PATH, "protocol")
    protocol_payload_hash = _verify_integrity(protocol, "protocol")
    if protocol_payload_hash != EXPECTED_PROTOCOL_PAYLOAD_SHA256:
        raise ValueError("protocol payload differs from committed preregistration")

    pinned_files = (
        ("manifest.json", EXPECTED_MANIFEST_FILE_SHA256, EXPECTED_MANIFEST_PAYLOAD_SHA256),
        ("raw-results.json", EXPECTED_RAW_FILE_SHA256, EXPECTED_RAW_PAYLOAD_SHA256),
        ("summary.json", EXPECTED_SUMMARY_FILE_SHA256, EXPECTED_SUMMARY_PAYLOAD_SHA256),
        ("execution-lock.json", EXPECTED_LOCK_FILE_SHA256, EXPECTED_LOCK_PAYLOAD_SHA256),
    )
    loaded: dict[str, dict[str, Any]] = {}
    for name, file_hash, payload_hash in pinned_files:
        path = RESULTS / name
        _verify_file(path, name, file_hash)
        _verify_committed_result(path)
        value = _load_object(path, name)
        if _verify_integrity(value, name) != payload_hash:
            raise ValueError(f"{name} payload differs from committed evidence")
        loaded[name] = value
    manifest = loaded["manifest.json"]
    raw = loaded["raw-results.json"]
    summary = loaded["summary.json"]
    lock = loaded["execution-lock.json"]

    if (
        manifest["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA
        or raw["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA
    ):
        raise ValueError("preregistration revision chain mismatch")
    revision_values = {
        manifest["preregistration_commit_sha"],
        raw["preregistration_commit_sha"],
        summary["preregistration_commit_sha"],
        lock["preregistration_commit_sha"],
        summary["environment"]["code_revision"],
        PREREGISTRATION_COMMIT_SHA,
    }
    if revision_values != {PREREGISTRATION_COMMIT_SHA}:
        raise ValueError("temporal preregistration SHA binding mismatch")
    protocol_values = {
        manifest["protocol_file_sha256"],
        raw["protocol_file_sha256"],
        lock["protocol_file_sha256"],
        protocol_file_hash,
    }
    if protocol_values != {EXPECTED_PROTOCOL_FILE_SHA256}:
        raise ValueError("protocol file SHA binding mismatch")
    payload_values = {
        manifest["protocol_payload_sha256"],
        raw["protocol_payload_sha256"],
        summary["protocol_payload_sha256"],
        lock["protocol_payload_sha256"],
        protocol_payload_hash,
    }
    if payload_values != {EXPECTED_PROTOCOL_PAYLOAD_SHA256}:
        raise ValueError("protocol payload SHA binding mismatch")
    if (
        manifest["raw_results_payload_sha256"] != EXPECTED_RAW_PAYLOAD_SHA256
        or summary["raw_results_payload_sha256"] != EXPECTED_RAW_PAYLOAD_SHA256
        or manifest["summary_payload_sha256"] != EXPECTED_SUMMARY_PAYLOAD_SHA256
    ):
        raise ValueError("raw/summary temporal payload chain mismatch")
    started = datetime.fromisoformat(lock["started_at_utc"])
    prereg_time = datetime.fromisoformat(PREREGISTRATION_COMMIT_TIME)
    results_time = datetime.fromisoformat(RESULTS_COMMIT_TIME)
    if not prereg_time < started < results_time:
        raise ValueError("execution lock is not temporally between prereg and results")

    entries: dict[str, Mapping[str, Any]] = {}
    for index, raw_entry in enumerate(manifest["deterministic_files"]):
        entry = _closed(
            raw_entry,
            f"manifest entry {index}",
            {"path", "kind", "file_sha256", "payload_sha256"},
        )
        if entry["path"] in entries:
            raise ValueError("manifest deterministic paths are duplicated")
        path = _safe_path(RESULTS, entry["path"], f"manifest entry {index}")
        _verify_file(path, entry["path"], _digest(entry["file_sha256"], "file hash"))
        _verify_committed_result(path)
        if entry["payload_sha256"] is not None:
            artifact = _load_object(path, entry["path"])
            if _verify_integrity(artifact, entry["path"]) != entry["payload_sha256"]:
                raise ValueError(f"{entry['path']} payload binding mismatch")
        entries[entry["path"]] = entry

    cases = raw["cases"]
    if (
        len(cases) != 96
        or len(raw["sampling_design_ids"]) != 96
        or len(set(raw["sampling_design_ids"])) != 96
    ):
        raise ValueError("raw results do not contain 96 unique preregistered samples")
    if any(case["status"] != "success" or case["failure"] is not None for case in cases):
        raise ValueError("reviewed v2 run must contain 96 successes and zero failures")
    for index, case in enumerate(cases):
        if case["design_id"] != raw["sampling_design_ids"][index]:
            raise ValueError(f"case {index} does not match preregistered sampling order")
        for digest_name in (
            "design_id",
            "geometry_sha256",
            "source_sha256",
            "config_sha256",
            "case_sha256",
        ):
            _digest(case[digest_name], f"{case['case_id']}.{digest_name}")
        case_hash = _canonical_hash(
            {
                "geometry_sha256": case["geometry_sha256"],
                "source_sha256": case["source_sha256"],
                "config_sha256": case["config_sha256"],
            }
        )
        if case["case_sha256"] != case_hash:
            raise ValueError(f"{case['case_id']} identity binding mismatch")
        for name, _label, _units in METRICS:
            _number(case["qois"][name], f"{case['case_id']}.{name}")

    front = _front(cases, protocol["objectives"])
    if front != summary["nondominated_case_ids"] or len(front) != 25:
        raise ValueError("tolerance-based nondominated front does not reproduce")
    roles = _roles(cases, front, protocol)
    if roles != summary["representative_roles"] or roles != manifest["representative_roles"]:
        raise ValueError("five representative roles do not reproduce")
    if [(item["role"], item["case_id"]) for item in roles] != list(EXPECTED_ROLES):
        raise ValueError("representative role identities differ from committed evidence")
    unique_ids = tuple(sorted({item["case_id"] for item in roles}))
    if (
        unique_ids != EXPECTED_UNIQUE_REPRESENTATIVES
        or summary["unique_representative_count"] != 4
    ):
        raise ValueError("representative coalescence must yield four unique cases")

    recomputed_gates = _gates(cases, raw["parity"], protocol)
    recorded_gates = [
        {
            "gate_id": item["gate_id"],
            "failed_case_ids": item["failed_case_ids"],
            "failure_count": item["failure_count"],
            "passed": item["passed"],
            "observed": item["observed"],
        }
        for item in summary["terminal_gates"]
    ]
    if recomputed_gates != recorded_gates or not all(item["passed"] for item in recomputed_gates):
        raise ValueError("seven terminal gates do not reproduce exactly")
    if (
        summary["requested_count"],
        summary["evaluated_count"],
        summary["failed_count"],
        summary["nondominated_count"],
        summary["terminal_status"],
    ) != (96, 96, 0, 25, "ACCEPTED"):
        raise ValueError("committed summary is not exact accepted 96/0/25 evidence")

    role_map: dict[str, list[str]] = {}
    for role in roles:
        role_map.setdefault(role["case_id"], []).append(role["role"])
    artifacts_by_id = {
        item["case_id"]: item for item in manifest["representative_artifacts"]
    }
    if tuple(sorted(artifacts_by_id)) != EXPECTED_UNIQUE_REPRESENTATIVES:
        raise ValueError("manifest has incorrect unique representative artifacts")
    representative_payloads = []
    for case_id in EXPECTED_UNIQUE_REPRESENTATIVES:
        binding = artifacts_by_id[case_id]
        if binding["roles"] != sorted(role_map[case_id]):
            raise ValueError(f"{case_id} coalesced roles are inconsistent")
        for kind in ("geometry", "full_field", "downsampled_field"):
            linked = entries.get(binding[kind]["path"])
            if (
                linked is None
                or linked["file_sha256"] != binding[kind]["file_sha256"]
                or linked["payload_sha256"] != binding[kind]["payload_sha256"]
            ):
                raise ValueError(f"{case_id} {kind} manifest binding mismatch")
        geometry = _load_object(
            _safe_path(RESULTS, binding["geometry"]["path"], "geometry"), "geometry"
        )
        field = _load_object(
            _safe_path(
                RESULTS, binding["downsampled_field"]["path"], "downsampled field"
            ),
            "field",
        )
        _validate_field(field, case_id)
        if field["input"]["name"] != case_id:
            raise ValueError(f"{case_id} field identity mismatch")
        representative_payloads.append(
            {
                "case_id": case_id,
                "roles": sorted(role_map[case_id]),
                "geometry": {
                    "chamber": geometry["chamber"],
                    "regions": geometry["regions"],
                    "stages": geometry["stages"],
                },
                "field": _field_projection(field),
                "identity": {
                    kind: {
                        "file_sha256": binding[kind]["file_sha256"],
                        "payload_sha256": binding[kind]["payload_sha256"],
                    }
                    for kind in ("geometry", "full_field", "downsampled_field")
                },
            }
        )

    front_set = set(front)
    compact_cases = [
        {
            "case_id": case["case_id"],
            "design_id": case["design_id"],
            "sampling_provenance": case["sampling_provenance"],
            "design_values": case["design_values"],
            "derived_geometry": case["derived_geometry"],
            "qois": case["qois"],
            "geometry_sha256": case["geometry_sha256"],
            "source_sha256": case["source_sha256"],
            "config_sha256": case["config_sha256"],
            "case_sha256": case["case_sha256"],
            "backend": case["backend"],
            "iterations": case["iterations"],
            "nondominated": case["case_id"] in front_set,
            "roles": sorted(role_map.get(case["case_id"], [])),
            "parity": next(
                (item for item in raw["parity"] if item["case_id"] == case["case_id"]),
                None,
            ),
        }
        for case in cases
    ]
    payload = {
        "schema": "cft-revival.l1a-geometry-sweep-v2-visualization/1.0.0",
        "warning": (
            "L1a equivalent-current field-only screening: no material-aware permanent-"
            "magnet model, plasma solution, thrust, efficiency, or hardware validity."
        ),
        "identity": {
            **git_evidence,
            "execution_lock_started_at_utc": lock["started_at_utc"],
            "protocol_file_sha256": protocol_file_hash,
            "protocol_payload_sha256": protocol_payload_hash,
            "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
            "manifest_payload_sha256": EXPECTED_MANIFEST_PAYLOAD_SHA256,
            "raw_file_sha256": EXPECTED_RAW_FILE_SHA256,
            "raw_payload_sha256": EXPECTED_RAW_PAYLOAD_SHA256,
            "summary_file_sha256": EXPECTED_SUMMARY_FILE_SHA256,
            "summary_payload_sha256": EXPECTED_SUMMARY_PAYLOAD_SHA256,
        },
        "summary": summary,
        "protocol": {
            "execution": protocol["execution"],
            "sampling": protocol["sampling"],
            "field": protocol["field"],
            "qoi_policy": protocol["qoi_policy"],
            "objectives": protocol["objectives"],
            "representative_policy": protocol["representative_policy"],
            "terminal_acceptance": protocol["terminal_acceptance"],
            "replay_contract": protocol["replay_contract"],
            "claim_limits": protocol["claim_limits"],
        },
        "metrics": [
            {"name": name, "label": label, "units": units}
            for name, label, units in METRICS
        ],
        "gates": recorded_gates,
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
            "identity",
            "summary",
            "protocol",
            "metrics",
            "gates",
            "cases",
            "representatives",
        },
    )
    if payload["schema"] != "cft-revival.l1a-geometry-sweep-v2-visualization/1.0.0":
        raise ValueError("visualization payload schema is unsupported")
    identity = payload["identity"]
    expected = {
        "preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
        "results_commit_sha": RESULTS_COMMIT_SHA,
        "protocol_file_sha256": EXPECTED_PROTOCOL_FILE_SHA256,
        "protocol_payload_sha256": EXPECTED_PROTOCOL_PAYLOAD_SHA256,
        "manifest_file_sha256": EXPECTED_MANIFEST_FILE_SHA256,
        "manifest_payload_sha256": EXPECTED_MANIFEST_PAYLOAD_SHA256,
        "raw_file_sha256": EXPECTED_RAW_FILE_SHA256,
        "raw_payload_sha256": EXPECTED_RAW_PAYLOAD_SHA256,
        "summary_file_sha256": EXPECTED_SUMMARY_FILE_SHA256,
        "summary_payload_sha256": EXPECTED_SUMMARY_PAYLOAD_SHA256,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("embedded committed identity evidence is invalid")
    if (
        len(payload["cases"]) != 96
        or sum(case["nondominated"] for case in payload["cases"]) != 25
        or len(payload["gates"]) != 7
        or not all(gate["passed"] for gate in payload["gates"])
    ):
        raise ValueError("visualization counts or gate evidence mismatch")
    if len(payload["representatives"]) != 4 or sum(
        len(item["roles"]) for item in payload["representatives"]
    ) != 5:
        raise ValueError("visualization representative coalescence mismatch")


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("dashboard template must contain one data placeholder")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).replace("</", "<\\/")
    return template.replace("__DATA__", encoded)


def generate(output: Path = DEFAULT_OUTPUT) -> str:
    html = render_html(build_payload())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return sha256(html.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(generate(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
