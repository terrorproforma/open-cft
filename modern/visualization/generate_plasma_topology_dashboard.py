"""Generate the standalone plasma / magnetic-topology results dashboard.

The dashboard embeds only committed, accepted or recorded artifacts. Every
embedded source is hash-verified before projection (sidecar, manifest byte or
canonical-semantic identity, and Git-committed equivalence); the generator
refuses to run when any anchor mismatches. No physics is regenerated. Field
rasters are downsampled/rounded projections of hashed source files; every
scalar count, probability, position and gate value is embedded exactly.

Determinism: the generator omits wall-clock time and runtime measurements. The
footer "evidence snapshot" time is the author time of the newest pinned
evidence commit (or ``SOURCE_DATE_EPOCH`` when set), so identical inputs
produce identical bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from math import hypot, isfinite
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
REPO = MODERN.parent
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEMPLATE_PATH = HERE / "plasma-topology-results.template.html"
DEFAULT_OUTPUT = HERE / "plasma-topology-results.html"
SCHEMA = "cft-revival.plasma-topology-results-dashboard/1.0.0"
CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
DATA_SCRIPT_ID = "plasma-topology-data"
SIZE_CAP_BYTES = 15 * 1024 * 1024

# --------------------------------------------------------------------------
# Pinned evidence identities (all on the feat/sota-foundation lineage except the
# orbit v4 result, which is read from its own recorded branch via ``git show``).
# --------------------------------------------------------------------------
EVIDENCE_BASE_COMMIT = "3ab50ef5c31cfa45f2256ddba18dafa965010c7a"

CHARACTERIZATION_ID = "cft-topology-characterization-v1"
CHARACTERIZATION_RESULTS = (
    MODERN / "experiments" / "cft_topology_characterization_v1" / "results"
)
CHARACTERIZATION_RESULTS_COMMIT = "3ce6c546194e1d3e943d0b3d0951d03e15e354d9"
CHARACTERIZATION_PREREGISTRATION_COMMIT = "af88470b86fd95882ae7fddc48e2860cbfba1219"
CHARACTERIZATION_MANIFEST_FILE_SHA256 = (
    "bd2ad68e6cc68dd547847f8e07817dc86d35e7a45075fcee9102a2e9eca938a8"
)
CHARACTERIZATION_MANIFEST_PAYLOAD_SHA256 = (
    "f42bb4286a6c76f7b3aaaab19e5dd7792b944efb83d0b96f1930868c10e445d0"
)
CHARACTERIZATION_DATASET_SEMANTIC_SHA256 = (
    "046364dcf427b435b1621921e8463c7276f093eb2d4fb3c8f1ccd2c3c748be44"
)

FOUR_CELL_V2_ID = "four-cell-topology-search-v2"
FOUR_CELL_V2_RESULTS = MODERN / "experiments" / "four_cell_topology_search_v2" / "results"
FOUR_CELL_V2_RESULTS_COMMIT = "7120e8edcb74c02c1df968c730d1f93b3758b4e1"
FOUR_CELL_V2_PREREGISTRATION_COMMIT = "d6317910703de91ca6dc25c4d4d855e36cc3b14d"
FOUR_CELL_V2_MANIFEST_FILE_SHA256 = (
    "f5e26373d72bd13aa5631516009797567e0f66812941d993d5fad534be07240a"
)
FOUR_CELL_V2_MANIFEST_PAYLOAD_SHA256 = (
    "f444dee5c7731df8f5a80c3dfce3dfb7f4333609cdce36ee39d2eed51df08be5"
)
FOUR_CELL_V2_DATASET_FILE_SHA256 = (
    "f01c044b99349ee1fc207702b4b7ab73c6eab49362a5b0249c3a99f501252d68"
)

FOUR_CELL_V1_ID = "four-cell-topology-search-v1"
FOUR_CELL_V1_RESULTS = MODERN / "experiments" / "four_cell_topology_search" / "results"
FOUR_CELL_V1_RESULTS_COMMIT = "4afcecfb024cd06e79d0ce8e063fc863ba3f79dc"
FOUR_CELL_V1_MANIFEST_FILE_SHA256 = (
    "3ddae4a7254e832126a3e368b9e30cc29c25fe8fdafd127fcbbaafe27def7b53"
)
FOUR_CELL_V1_MANIFEST_PAYLOAD_SHA256 = (
    "2ff65a4887058ebfb9e23542caca066515c2ba03b04efa6f0b9da4cfd787f38d"
)
FOUR_CELL_V1_DATASET_FILE_SHA256 = (
    "145decd7f1856b9e00429e07f06d47beb2a1de24523d75e6eb10c8f0856344de"
)
FOUR_CELL_V1_DATASET_PAYLOAD_SHA256 = (
    "fced0eb33b9073a83911e8ec7717a88e7764708b344020007e9a516a40836491"
)

SWEEP_ID = "l1a-geometry-sweep-v2"
SWEEP_RESULTS = MODERN / "experiments" / "l1a_geometry_sweep_v2" / "results"
SWEEP_RESULTS_COMMIT = "f30cb42ec4a8633bf634a3d32ffa5b11f66be97a"
SWEEP_PREREGISTRATION_COMMIT = "092f5fae692ee7d6711e0c7e1c94dac6a345f37c"
SWEEP_MANIFEST_FILE_SHA256 = (
    "768b345e946a45e623f83aaa18e01f8ec5bc7f823e81858a0a8c3a3e2e448754"
)
SWEEP_MANIFEST_PAYLOAD_SHA256 = (
    "1ba8c3ed4da694afe8f660c9083e2ddbb4e8621f1768c4c88794e65909030e92"
)
SWEEP_SUMMARY_FILE_SHA256 = (
    "942852295d3b2ee04f968734a31e15694e46d389838259a01113ae090ad30e29"
)
SWEEP_RAW_FILE_SHA256 = (
    "76f145f816a187c36c3260c184a07d8979c244999f40c2562f4207dbca90b4c1"
)

WCVAL_V1_ID = "cft-wall-cusp-validation-v1"
WCVAL_V1_RESULTS = MODERN / "experiments" / "cft_wall_cusp_validation_v1" / "results"
WCVAL_V1_RESULTS_COMMIT = "2504175ea845dca1b57fef159961f335a2b546ee"
WCVAL_V1_MANIFEST_FILE_SHA256 = (
    "43a5bbe83a05ba7802d8dd78ed2d539d97f85fbff879e92e6405e81fdf3dcaf3"
)
WCVAL_V1_MANIFEST_PAYLOAD_SHA256 = (
    "d1ddcfaef7a084e9612d0952f7dec72db8959c06bf325ea3722ce3e8f3d79fc6"
)
WCVAL_V2_ID = "cft-wall-cusp-validation-v2"
WCVAL_V2_RESULTS = MODERN / "experiments" / "cft_wall_cusp_validation_v2" / "results"
WCVAL_V2_RESULTS_COMMIT = "7e1246b5b76830fe09afb10ffe076e953a3c2905"
WCVAL_V2_MANIFEST_FILE_SHA256 = (
    "8e0be571d43111caeb45c3a5d87f2e104ea49693848d4fd520a18d058ae3c5cc"
)
WCVAL_V2_MANIFEST_PAYLOAD_SHA256 = (
    "dbcf394f809da5a869acac7d257942b2d41ac3a99020b4688327a3a715b92bf3"
)
COUPLING_V4_COMMIT = "f10d8213117fbafd8c2b69bdc103b6ef7b5d6d8c"
COUPLING_V3_COMMIT = "f80a360fd740a30017cdac1874cedbfa2806874a"

AXISYMMETRIC_ID = "axisymmetric-l1a-v1.2"
AXISYMMETRIC_RESULTS_COMMIT = "dbcab64603edba357acc3eff13519965dcad4187"
AXISYMMETRIC_GENERATOR = HERE / "generate_axisymmetric_results.py"

P2_ID = "fem-reference-divergent-exit-p2"
P2_DESIGN_ID = "divergent-exit-stack"
P2_ROOT = MODERN / "examples" / "fem_reference" / "artifacts" / "third-level" / P2_DESIGN_ID
P2_RESULTS_COMMIT = "a1158bad5eac3dd27ca6464a7649ce359524d8db"
P2_MANIFEST_FILE_SHA256 = (
    "0defabb5bf2aa7750bc4a39ce3392fcd6b23ef22470b4560bc0e37d37bb03da1"
)
P2_RESULT_FILE_SHA256 = (
    "6c3261208e04e4d10f5a711e35441b4366716878c56d1c0f5ae7558f0fc2f133"
)
P2_VIEWER_FILE_SHA256 = (
    "c1bf7bb1a8876a815a14e54b4dc91666b969b1eefe980352fee3d64ee107f1aa"
)
FEM_GENERATOR = MODERN / "examples" / "fem_reference" / "visualization" / "generate_dashboard.py"
P2_WALL_RADIUS_M = 0.002
P2_WALL_Z_RANGE_M = (0.001, 0.023)
P2_PROFILE_SAMPLES = 220

ORBIT_V4_ID = "cft-orbit-wall-loss-v4"
ORBIT_V4_BRANCH = "exp/cft-orbit-wall-loss-v4"
ORBIT_V4_RESULTS_COMMIT = "6922a3cf97d261735266aa1a5a0c0c9683e021ca"
ORBIT_V4_PREREGISTRATION_COMMIT = "757e365f9f667620c7610663574294c3b71e1f51"
ORBIT_V4_PREFIX = "modern/experiments/cft_orbit_wall_loss_v4/"
ORBIT_V4_MANIFEST_FILE_SHA256 = (
    "ef3863b0a3ba0a1d74187b05daf81d5d94d3838a7e33ecf82c485dccd162929f"
)
ORBIT_V4_CASES = (
    "primary-N", "primary-2N", "primary-4N",
    "refined-N", "refined-2N", "refined-4N",
    "enlarged-N", "enlarged-2N", "enlarged-4N",
)
ORBIT_PRIOR_CAMPAIGNS = ("v1", "v2", "v3")

EXPECTED_CHARACTERIZATION_REPRESENTATIVES = tuple(
    f"topology-s0{n}-p0-r0-neg" for n in range(2, 9)
)
EXPECTED_FOUR_CELL_V2_REPRESENTATIVES = ("v2-006", "v2-010")
EXPECTED_FOUR_CELL_V1_COMPATIBLE = ("four-cell-005-8885e09139", "four-cell-029-52bb37501f")
EXPECTED_SWEEP_REPRESENTATIVES = (
    "l1a-gs-v2-000-48d2ccedd5",
    "l1a-gs-v2-032-570ad83ba6",
    "l1a-gs-v2-065-9e98f08f3b",
    "l1a-gs-v2-068-375d1b1b13",
)
RASTER_SIGNIFICANT_DIGITS = 7


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
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
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} is not readable: {path.name}") from error


def _load_object(path: Path, label: str) -> dict[str, Any]:
    return _load_json_bytes(_read_bytes(path, label), label)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
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


def _commit(value: str, label: str) -> str:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a full lowercase Git commit SHA-1")
    return value


def _verify_integrity(value: Mapping[str, Any], label: str, key: str = "integrity") -> str:
    integrity = value.get(key)
    if not isinstance(integrity, Mapping) or set(integrity) != {
        "algorithm", "canonicalization", "payload_sha256"
    }:
        raise ValueError(f"{label} {key} declaration is not closed")
    if integrity["algorithm"] != "sha256" or integrity["canonicalization"] != CANONICALIZATION:
        raise ValueError(f"{label} {key} declaration is unsupported")
    claimed = _digest(integrity["payload_sha256"], f"{label} payload digest")
    payload = {item: content for item, content in value.items() if item != key}
    if _canonical_hash(payload) != claimed:
        raise ValueError(f"{label} canonical payload SHA-256 mismatch")
    return claimed


def _file_sha256(path: Path, label: str) -> str:
    digest = sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 22), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"{label} is not readable: {path.name}") from error
    return digest.hexdigest()


def _verify_sidecar_file(path: Path, label: str, expected: str | None = None) -> str:
    """Verify ``<name>.sha256`` sidecar (``"{digest}  {name}\\n"``) and optional pin."""

    digest = _file_sha256(path, label)
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
    if not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw:
        raise ValueError(f"{label} must be a portable relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{label} escapes its evidence directory")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes its evidence directory")
    return path


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _round(value: float) -> float:
    return float(f"{float(value):.{RASTER_SIGNIFICANT_DIGITS}g}")


def _round_rows(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[_round(item) for item in row] for row in rows]


def _round_list(values: Sequence[float]) -> list[float]:
    return [_round(item) for item in values]


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments), cwd=REPO, check=False, capture_output=True
    )
    if completed.returncode:
        raise ValueError(
            "Git evidence check failed: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return completed.stdout


def _git_show(commit: str, repo_path: str) -> bytes:
    _commit(commit, "git show commit")
    return _git("show", f"{commit}:{repo_path}")


def _git_commit_time(commit: str) -> str:
    return _git("show", "-s", "--format=%aI", commit).decode("ascii").strip()


def _verify_git_clean(commit: str, *repo_paths: str) -> None:
    completed = subprocess.run(
        ("git", "diff", "--quiet", commit, "--", *repo_paths),
        cwd=REPO, check=False, capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"working evidence differs from commit {commit[:12]}: " + ", ".join(repo_paths)
        )


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def _import_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import sibling generator {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Source ledger: every embedded number points back to one of these entries.
# --------------------------------------------------------------------------
class SourceLedger:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def add(
        self,
        source_id: str,
        *,
        path: str,
        sha256_hex: str,
        commit: str,
        identity_method: str,
        identity_matches: bool,
        experiment_id: str,
        note: str | None = None,
    ) -> str:
        if source_id in self._entries:
            raise ValueError(f"duplicate source id {source_id}")
        entry: dict[str, Any] = {
            "id": source_id,
            "path": path,
            "sha256": _digest(sha256_hex, f"{source_id} sha256"),
            "commit": _commit(commit, f"{source_id} commit"),
            "identity_method": identity_method,
            "identity_matches": bool(identity_matches),
            "experiment_id": experiment_id,
        }
        if note:
            entry["note"] = note
        self._entries[source_id] = entry
        return source_id

    def entries(self) -> list[dict[str, Any]]:
        return [self._entries[key] for key in sorted(self._entries)]


# --------------------------------------------------------------------------
# Field helpers
# --------------------------------------------------------------------------
def _validate_l1a_field(field: Mapping[str, Any], label: str) -> tuple[list[float], list[float]]:
    if field.get("model_level") != "L1a":
        raise ValueError(f"{label} field model level is not L1a")
    if field.get("schema_version") not in {
        "cft-axisymmetric-field-map/1.1.0", "cft-axisymmetric-field-map/1.2.0"
    }:
        raise ValueError(f"{label} field schema is unsupported")
    field_map = field.get("field_map")
    if not isinstance(field_map, Mapping):
        raise ValueError(f"{label} field map is missing")
    r_values = [_number(item, f"{label}.r") for item in field_map["r_m"]]
    z_values = [_number(item, f"{label}.z") for item in field_map["z_m"]]
    if (
        len(r_values) < 2 or len(z_values) < 2
        or any(b <= a for a, b in zip(r_values, r_values[1:]))
        or any(b <= a for a, b in zip(z_values, z_values[1:]))
    ):
        raise ValueError(f"{label} field coordinates are invalid")
    nr, nz = len(r_values), len(z_values)
    for key in ("psi_wb", "b_r_t", "b_z_t", "b_magnitude_t"):
        rows = field_map[key]
        if not isinstance(rows, list) or len(rows) != nr or any(
            not isinstance(row, list) or len(row) != nz for row in rows
        ):
            raise ValueError(f"{label}.{key} dimensions do not match r_m/z_m")
    for i in range(nr):
        for j in range(nz):
            br = _number(field_map["b_r_t"][i][j], f"{label}.b_r_t")
            bz = _number(field_map["b_z_t"][i][j], f"{label}.b_z_t")
            expected = hypot(br, bz)
            if abs(_number(field_map["b_magnitude_t"][i][j], f"{label}.|B|") - expected) > max(
                2e-15, 2e-12 * expected
            ):
                raise ValueError(f"{label} |B| component identity mismatch")
            _number(field_map["psi_wb"][i][j], f"{label}.psi")
    for name in ("centreline", "wall"):
        profile = field["profiles"][name]
        count = len(profile["z_m"])
        if count < 2 or any(len(profile[key]) != count for key in ("b_r_t", "b_z_t")):
            raise ValueError(f"{label} {name} profile shape mismatch")
    return r_values, z_values


def _field_projection(field: Mapping[str, Any], label: str, stride: int = 1) -> dict[str, Any]:
    """|B| and psi raster plus exact wall/centreline profiles from an L1a artifact."""

    r_values, z_values = _validate_l1a_field(field, label)
    field_map = field["field_map"]
    r_index = list(range(0, len(r_values), stride))
    z_index = list(range(0, len(z_values), stride))
    if r_index[-1] != len(r_values) - 1:
        r_index.append(len(r_values) - 1)
    if z_index[-1] != len(z_values) - 1:
        z_index.append(len(z_values) - 1)
    wall = field["profiles"]["wall"]
    centre = field["profiles"]["centreline"]
    wall_magnitude = [
        hypot(_number(br, "wall br"), _number(bz, "wall bz"))
        for br, bz in zip(wall["b_r_t"], wall["b_z_t"], strict=True)
    ]
    return {
        "layout": "radial-major; values[r_index][z_index]",
        "source_grid": [len(r_values), len(z_values)],
        "embedded_stride": stride,
        "rounding": f"{RASTER_SIGNIFICANT_DIGITS} significant digits (display raster only)",
        "r_m": [r_values[i] for i in r_index],
        "z_m": [z_values[j] for j in z_index],
        "b_magnitude_t": _round_rows(
            [[field_map["b_magnitude_t"][i][j] for j in z_index] for i in r_index]
        ),
        "psi_wb": _round_rows([[field_map["psi_wb"][i][j] for j in z_index] for i in r_index]),
        "b_magnitude_max_t": _number(field["summary"]["b_magnitude_max_t"], f"{label} max"),
        "wall_profile": {
            "sampled_r_m": wall["sampled_r_m"],
            "requested_r_m": wall["requested_r_m"],
            "z_m": list(wall["z_m"]),
            "b_r_t": list(wall["b_r_t"]),
            "b_z_t": list(wall["b_z_t"]),
            "b_magnitude_t": wall_magnitude,
        },
        "centreline_profile": {
            "z_m": list(centre["z_m"]),
            "b_z_t": list(centre["b_z_t"]),
            "b_r_t": list(centre["b_r_t"]),
        },
        "sources": [
            {
                "name": source["name"],
                "polarity": source["polarity"],
                "r_inner_m": source["r_inner_m"],
                "r_outer_m": source["r_outer_m"],
                "z_min_m": source["z_min_m"],
                "z_max_m": source["z_max_m"],
                "ampere_turns_a": source["ampere_turns_a"],
            }
            for source in field["input"]["sources"]
        ],
        "diagnostics": {
            "converged": field["diagnostics"]["converged"],
            "iterations": field["diagnostics"]["iterations"],
            "relative_residual_l2": field["diagnostics"]["relative_residual_l2"],
            "backend": field["diagnostics"]["backend"],
        },
        "axis_topology": field["summary"]["topology"],
    }


def _geometry_projection(geometry: Mapping[str, Any], label: str) -> dict[str, Any]:
    if geometry.get("schema_version") != "cft_revival.geometry.axisymmetric_cft/1.1.0":
        raise ValueError(f"{label} geometry schema mismatch")
    regions = geometry.get("regions")
    stages = geometry.get("stages")
    if not regions or not stages:
        raise ValueError(f"{label} geometry is incomplete")
    projected_regions = []
    for region in regions:
        for key in (
            "z_min_m", "z_max_m", "r_inner_start_m", "r_inner_end_m",
            "r_outer_start_m", "r_outer_end_m",
        ):
            _number(region[key], f"{label}.{key}")
        projected_regions.append(
            {
                "region_id": region["region_id"],
                "role": region["role"],
                "polarity": region.get("polarity"),
                "z_min_m": region["z_min_m"],
                "z_max_m": region["z_max_m"],
                "r_inner_start_m": region["r_inner_start_m"],
                "r_inner_end_m": region["r_inner_end_m"],
                "r_outer_start_m": region["r_outer_start_m"],
                "r_outer_end_m": region["r_outer_end_m"],
            }
        )
    chamber = geometry["chamber"]
    return {
        "regions": projected_regions,
        "stages": [
            {
                "stage_id": stage["stage_id"],
                "center_z_m": stage["center_z_m"],
                "z_min_m": stage["z_min_m"],
                "z_max_m": stage["z_max_m"],
                "pitch_m": stage["pitch_m"],
                "magnetization": stage["magnetization"],
            }
            for stage in stages
        ],
        "chamber": {
            "outer_radius_m": chamber["outer_radius_m"],
            "length_m": chamber["length_m"],
            "dielectric_thickness_m": chamber["dielectric_thickness_m"],
            "exit_length_m": chamber.get("exit_length_m"),
            "exit_outer_radius_m": chamber.get("exit_outer_radius_m"),
        },
    }


P2_MAXIMA_MERGE_DISTANCE_M = 0.001


def _local_maxima(z_values: Sequence[float], values: Sequence[float]) -> list[dict[str, float]]:
    """Interior local maxima merged within 1 mm (dashboard-derived display diagnostic)."""

    candidates = [
        {"z_m": z_values[index], "value": values[index]}
        for index in range(1, len(values) - 1)
        if values[index] > values[index - 1] and values[index] >= values[index + 1]
    ]
    merged: list[dict[str, float]] = []
    for candidate in candidates:
        if merged and candidate["z_m"] - merged[-1]["z_m"] < P2_MAXIMA_MERGE_DISTANCE_M:
            if candidate["value"] > merged[-1]["value"]:
                merged[-1] = candidate
        else:
            merged.append(candidate)
    return merged


# --------------------------------------------------------------------------
# Section 1: CFT topology characterization v1 (56 cases)
# --------------------------------------------------------------------------
def _characterization(ledger: SourceLedger) -> dict[str, Any]:
    results = CHARACTERIZATION_RESULTS
    manifest_path = results / "manifest.json"
    manifest_hash = _file_sha256(manifest_path, "characterization manifest")
    if manifest_hash != CHARACTERIZATION_MANIFEST_FILE_SHA256:
        raise ValueError("characterization manifest file SHA-256 mismatch")
    manifest = _load_object(manifest_path, "characterization manifest")
    if _verify_integrity(manifest, "characterization manifest", "semantic_integrity") != (
        CHARACTERIZATION_MANIFEST_PAYLOAD_SHA256
    ):
        raise ValueError("characterization manifest payload differs from reviewed evidence")
    if (
        manifest["experiment_id"] != CHARACTERIZATION_ID
        or manifest["preregistration_commit_sha"] != CHARACTERIZATION_PREREGISTRATION_COMMIT
        or manifest["accepted_coupling_v3_commit_sha"] != COUPLING_V3_COMMIT
        or manifest["single_execution"] is not True
    ):
        raise ValueError("characterization manifest identity mismatch")
    listed = {item["path"]: item for item in manifest["artifacts"]}
    _verify_git_clean(
        CHARACTERIZATION_RESULTS_COMMIT,
        _repo_relative(manifest_path),
        _repo_relative(results / "dataset.json"),
        _repo_relative(results / "report.md"),
    )
    ledger.add(
        "char-manifest", path=_repo_relative(manifest_path), sha256_hex=manifest_hash,
        commit=CHARACTERIZATION_RESULTS_COMMIT,
        identity_method="pinned-file-sha256+canonical-semantic-payload",
        identity_matches=True, experiment_id=CHARACTERIZATION_ID,
    )

    def semantic_json(relative: str, label: str, source_id: str) -> dict[str, Any]:
        entry = listed.get(relative)
        if entry is None or entry["identity_method"] != "canonical-json-sha256":
            raise ValueError(f"{label} is not listed as canonical JSON in the manifest")
        path = _safe_path(results, relative, label)
        raw = _read_bytes(path, label)
        value = _load_json_bytes(raw, label)
        if _canonical_hash(value) != entry["semantic_sha256"]:
            raise ValueError(f"{label} canonical semantic SHA-256 mismatch")
        ledger.add(
            source_id, path=_repo_relative(path), sha256_hex=sha256(raw).hexdigest(),
            commit=CHARACTERIZATION_RESULTS_COMMIT,
            identity_method="manifest canonical-json-sha256",
            identity_matches=True, experiment_id=CHARACTERIZATION_ID,
        )
        return value

    dataset = semantic_json("dataset.json", "characterization dataset", "char-dataset")
    if listed["dataset.json"]["semantic_sha256"] != CHARACTERIZATION_DATASET_SEMANTIC_SHA256:
        raise ValueError("characterization dataset semantic anchor is superseded")
    _verify_integrity(dataset, "characterization dataset", "semantic_integrity")
    report_entry = listed["report.md"]
    report_raw = _read_bytes(results / "report.md", "characterization report")
    normalized = report_raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if (
        report_entry["identity_method"] != "normalized-lf-text-sha256"
        or sha256(normalized).hexdigest() != report_entry["semantic_sha256"]
    ):
        raise ValueError("characterization report normalized-text SHA-256 mismatch")
    ledger.add(
        "char-report", path=_repo_relative(results / "report.md"),
        sha256_hex=sha256(report_raw).hexdigest(), commit=CHARACTERIZATION_RESULTS_COMMIT,
        identity_method="manifest normalized-lf-text-sha256", identity_matches=True,
        experiment_id=CHARACTERIZATION_ID,
    )

    summary = dataset["summary"]
    if summary != manifest["summary"]:
        raise ValueError("characterization dataset/manifest summary mismatch")
    cases = dataset["cases"]
    if len(cases) != 56 or summary["evaluated_count"] != 56:
        raise ValueError("characterization dataset does not contain 56 evaluated cases")
    if summary["stable_eligible_cusp_count"] != 0 or summary["stable_eligible_cell_count"] != 0:
        raise ValueError("characterization stable eligible counts differ from recorded evidence")

    class_counts: Counter[str] = Counter()
    zone_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    compact_cases = []
    for case in cases:
        primary = case["maps"]["primary"]
        roots = primary["roots"]
        classes = Counter(root["local_topology"]["classification"] for root in roots)
        zones = Counter(root["geometry_association"]["zone"] for root in roots)
        channel = [
            root for root in roots
            if root["geometry_association"]["zone"] == "plasma_channel"
        ]
        channel_classes = Counter(root["local_topology"]["classification"] for root in channel)
        class_counts.update(classes)
        zone_counts.update(zones)
        exclusion_counts.update(root["exclusion_reason"] for root in roots)
        if any(root["eligible_cusp"] or root["eligible_cell"] for root in roots):
            raise ValueError(f"{case['case_id']} contains an eligible root not in the summary")
        cross = case["cross_map"]
        compact_cases.append(
            {
                "case_id": case["case_id"],
                "stage_count": case["stage_count"],
                "pitch_m": case["pitch_m"],
                "chamber_radius_m": case["chamber_radius_m"],
                "first_polarity": case["first_polarity"],
                "chamber_length_m": case["geometry"]["chamber_length_m"],
                "field_peak_t": primary["quality"]["field_peak_t"],
                "raw_detection_count": primary["raw_detection_count"],
                "clustered_root_count": primary["clustered_root_count"],
                "root_classes": {
                    "X": classes.get("X", 0),
                    "O": classes.get("O", 0),
                    "degenerate": classes.get("degenerate", 0),
                },
                "channel_roots": {
                    "total": len(channel),
                    "X": channel_classes.get("X", 0),
                    "O": channel_classes.get("O", 0),
                },
                "zones": dict(sorted(zones.items())),
                "eligible_cusp_count": primary["eligible_cusp_count"],
                "eligible_cell_count": primary["eligible_cell_count"],
                "stable_root_count": cross["stable_root_count"],
                "stable_eligible_cusp_count": cross["stable_eligible_cusp_count"],
                "stable_eligible_cell_count": cross["stable_eligible_cell_count"],
                "complete_primary_correspondence": cross["complete_primary_correspondence"],
                "refined_correspondence_count": cross["primary_to_refined"]["correspondence_count"],
                "enlarged_correspondence_count": cross["primary_to_enlarged"]["correspondence_count"],
                "refined_max_shift_m": cross["primary_to_refined"]["maximum_shift_m"],
                "enlarged_max_shift_m": cross["primary_to_enlarged"]["maximum_shift_m"],
                "failures": list(case["failures"]),
                "topology_class": "no_stable_eligible_cusp_or_cell",
            }
        )
    if sum(class_counts.values()) != sum(
        case["clustered_root_count"] for case in compact_cases
    ):
        raise ValueError("characterization root class tally is inconsistent")

    representatives = []
    by_id = {case["case_id"]: case for case in cases}
    if tuple(dataset["representative_case_ids"]) != EXPECTED_CHARACTERIZATION_REPRESENTATIVES:
        raise ValueError("characterization representative identities differ")
    for case_id in EXPECTED_CHARACTERIZATION_REPRESENTATIVES:
        base = f"representatives/{case_id}"
        geometry = semantic_json(f"{base}/geometry.json", f"{case_id} geometry", f"char-{case_id}-geometry")
        field = semantic_json(f"{base}/primary-field.json", f"{case_id} primary field", f"char-{case_id}-primary-field")
        for role in ("refined-field.json", "enlarged_domain-field.json"):
            entry = listed[f"{base}/{role}"]
            path = _safe_path(results, f"{base}/{role}", role)
            raw = _read_bytes(path, role)
            if _canonical_hash(_load_json_bytes(raw, role)) != entry["semantic_sha256"]:
                raise ValueError(f"{case_id} {role} canonical semantic SHA-256 mismatch")
        case = by_id[case_id]
        primary = case["maps"]["primary"]
        if primary["artifact_semantic_sha256"] != _canonical_hash(
            {k: v for k, v in field.items() if k != "integrity"}
        ):
            raise ValueError(f"{case_id} primary field does not match the dataset record")
        if field["input"]["name"] != f"{case_id}-primary":
            raise ValueError(f"{case_id} primary field identity mismatch")
        representatives.append(
            {
                "case_id": case_id,
                "label": f"{case['stage_count']}-stage · pitch {case['pitch_m'] * 1e3:g} mm · "
                f"channel r {case['chamber_radius_m'] * 1e3:g} mm · first polarity {case['first_polarity']:+d}",
                "stage_count": case["stage_count"],
                "pitch_m": case["pitch_m"],
                "chamber_radius_m": case["chamber_radius_m"],
                "first_polarity": case["first_polarity"],
                "geometry": _geometry_projection(geometry, case_id),
                "field": _field_projection(field, case_id, stride=1),
                "roots": [
                    {
                        "root_id": root["root_id"],
                        "r_m": root["r_m"],
                        "z_m": root["z_m"],
                        "classification": root["local_topology"]["classification"],
                        "zone": root["geometry_association"]["zone"],
                        "exclusion_reason": root["exclusion_reason"],
                        "field_magnitude_t": root["field_magnitude_t"],
                        "eligible_cusp": root["eligible_cusp"],
                        "eligible_cell": root["eligible_cell"],
                        "cell_bounding": root["separatrix_connectivity"]["cell_bounding"],
                    }
                    for root in primary["roots"]
                ],
                "mesh_scale_m": primary["mesh_scale_m"],
                "cluster_tolerance_m": primary["cluster_tolerance_m"],
                "quality": primary["quality"],
                "mirror_ratio": {
                    "status": "not_computed",
                    "reason": "protocol.publication.mirror_probability=false; the study characterizes roots only",
                },
                "sources": [f"char-{case_id}-geometry", f"char-{case_id}-primary-field", "char-dataset"],
            }
        )

    return {
        "experiment_id": CHARACTERIZATION_ID,
        "classification": dataset["classification"],
        "purpose": dataset["purpose"],
        "status": "recorded_developmental_characterization",
        "commits": {
            "preregistration": CHARACTERIZATION_PREREGISTRATION_COMMIT,
            "results": CHARACTERIZATION_RESULTS_COMMIT,
            "accepted_coupling_v3": COUPLING_V3_COMMIT,
        },
        "families": {
            "stage_counts": sorted({case["stage_count"] for case in cases}),
            "pitch_m": sorted({case["pitch_m"] for case in cases}),
            "chamber_radius_m": sorted({case["chamber_radius_m"] for case in cases}),
            "first_polarity": sorted({case["first_polarity"] for case in cases}),
        },
        "summary": summary,
        "root_class_counts": {
            "X": class_counts.get("X", 0),
            "O": class_counts.get("O", 0),
            "degenerate": class_counts.get("degenerate", 0),
        },
        "zone_counts": dict(sorted(zone_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "stage_relation": dataset["analyses"]["stage_relation"],
        "factor_scores": dataset["analyses"]["factor_scores"],
        "search_v3_recommendation": dataset["analyses"]["search_v3_recommendation"],
        "gpu_replay": [
            {
                "case_id": item["case_id"],
                "map_role": item["map_role"],
                "passed": item["passed"],
                "field_equality_passed": item["field_equality_passed"],
                "residual_reproducibility_passed": item["residual_reproducibility_passed"],
                "br_max_abs_t": item["field_differences"]["br_max_abs_t"],
                "bz_max_abs_t": item["field_differences"]["bz_max_abs_t"],
                "psi_max_abs_wb": item["field_differences"]["psi_max_abs_wb"],
            }
            for item in dataset["gpu_replay"]
        ],
        "runtime_identity": {
            key: dataset["runtime_identity"][key]
            for key in ("gpu_name", "compute_capability", "driver_version", "warp_version", "python_version", "platform")
        },
        "publication": dataset["publication"],
        "cases": compact_cases,
        "representatives": representatives,
        "sources": ["char-manifest", "char-dataset", "char-report"],
    }


# --------------------------------------------------------------------------
# Section 2: four-cell topology search v2 (preregistered, 128 candidates)
# --------------------------------------------------------------------------
def _four_cell_v2(ledger: SourceLedger) -> dict[str, Any]:
    results = FOUR_CELL_V2_RESULTS
    manifest_path = results / "manifest.json"
    manifest_hash = _verify_sidecar_file(manifest_path, "four-cell v2 manifest", FOUR_CELL_V2_MANIFEST_FILE_SHA256)
    manifest = _load_object(manifest_path, "four-cell v2 manifest")
    if _verify_integrity(manifest, "four-cell v2 manifest") != FOUR_CELL_V2_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("four-cell v2 manifest payload differs from reviewed evidence")
    if (
        manifest["experiment_id"] != FOUR_CELL_V2_ID
        or manifest["preregistration_commit_sha"] != FOUR_CELL_V2_PREREGISTRATION_COMMIT
        or manifest["accepted_coupling_v3_commit_sha"] != COUPLING_V3_COMMIT
    ):
        raise ValueError("four-cell v2 manifest identity mismatch")
    listed = {item["path"]: item for item in manifest["artifacts"]}
    dataset_path = results / "dataset.json"
    dataset_hash = _verify_sidecar_file(dataset_path, "four-cell v2 dataset", FOUR_CELL_V2_DATASET_FILE_SHA256)
    if listed["dataset.json"]["sha256"] != dataset_hash:
        raise ValueError("four-cell v2 dataset is not the manifest-listed file")
    dataset = _load_object(dataset_path, "four-cell v2 dataset")
    _verify_integrity(dataset, "four-cell v2 dataset")
    report_path = results / "report.md"
    report_hash = _verify_sidecar_file(report_path, "four-cell v2 report")
    if listed["report.md"]["sha256"] != report_hash:
        raise ValueError("four-cell v2 report is not the manifest-listed file")
    _verify_git_clean(
        FOUR_CELL_V2_RESULTS_COMMIT,
        _repo_relative(manifest_path), _repo_relative(dataset_path), _repo_relative(report_path),
    )
    for source_id, path, digest in (
        ("fc2-manifest", manifest_path, manifest_hash),
        ("fc2-dataset", dataset_path, dataset_hash),
        ("fc2-report", report_path, report_hash),
    ):
        ledger.add(
            source_id, path=_repo_relative(path), sha256_hex=digest,
            commit=FOUR_CELL_V2_RESULTS_COMMIT,
            identity_method="sidecar-sha256+manifest-listing" + (
                "+canonical-payload" if path.suffix == ".json" else ""
            ),
            identity_matches=True, experiment_id=FOUR_CELL_V2_ID,
        )
    summary = dataset["summary"]
    if summary != manifest["summary"]:
        raise ValueError("four-cell v2 dataset/manifest summary mismatch")
    cases = dataset["cases"]
    if len(cases) != 128 or summary["evaluated_count"] != 128 or summary["stable_count"] != 0:
        raise ValueError("four-cell v2 counts differ from recorded null result")
    ranking = {item["candidate_id"]: index + 1 for index, item in enumerate(dataset["ranking"])}
    compact = []
    for case in cases:
        topology = case["topology"]
        compact.append(
            {
                "candidate_id": case["candidate_id"],
                "rank": ranking[case["candidate_id"]],
                "design_values": case["sampling"]["values"],
                "stage_count": len(case["derived_geometry"]["stage_centres_m"]),
                "pitch_m": case["derived_geometry"]["pitch_m"],
                "chamber_radius_m": case["derived_geometry"]["chamber_radius_m"],
                "wall_radius_m": case["derived_geometry"]["wall_radius_m"],
                "stage_centres_m": case["derived_geometry"]["stage_centres_m"],
                "cusp_targets_m": case["derived_geometry"]["cusp_targets_m"],
                "stable": bool(case["stable"]),
                "adiabatic": bool(case["adiabatic"]),
                "coupled": bool(case["coupled"]),
                "failures": list(case["failures"]),
                "count_by_role": topology["count_by_role"],
                "exact_count": topology["exact_count"],
                "geometry_registered": topology["geometry_registered"],
                "all_field_gates": topology["all_field_gates"],
                "interior_cusp_z_m": {
                    role: list(case["maps"][role]["interior_cusp_z_m"])
                    for role in ("primary", "downsampled", "enlarged_domain")
                },
                "boundary_null_count": case["maps"]["primary"]["boundary_null_count"],
                "field_peak_t": case["maps"]["primary"]["quality"]["field_peak_t"],
                "topology_class": "not_four_cell_stable",
            }
        )
    if any(case["stable"] or case["exact_count"] for case in compact):
        raise ValueError("four-cell v2 contains a stable or exact-count candidate not in the summary")

    representatives = []
    by_id = {case["candidate_id"]: case for case in cases}
    rep_ids = tuple(sorted({item["candidate_id"] for item in manifest["representatives"]}))
    if rep_ids != EXPECTED_FOUR_CELL_V2_REPRESENTATIVES:
        raise ValueError("four-cell v2 representative identities differ")
    for candidate_id in EXPECTED_FOUR_CELL_V2_REPRESENTATIVES:
        loaded: dict[str, dict[str, Any]] = {}
        for item in manifest["representatives"]:
            if item["candidate_id"] != candidate_id:
                continue
            path = _safe_path(results, item["path"], f"{candidate_id} {item['role']}")
            digest = _verify_sidecar_file(path, f"{candidate_id} {item['role']}", item["sha256"])
            if listed[item["path"]]["sha256"] != digest:
                raise ValueError(f"{candidate_id} {item['role']} manifest listing mismatch")
            _verify_git_clean(FOUR_CELL_V2_RESULTS_COMMIT, _repo_relative(path))
            ledger.add(
                f"fc2-{candidate_id}-{item['role']}", path=_repo_relative(path), sha256_hex=digest,
                commit=FOUR_CELL_V2_RESULTS_COMMIT,
                identity_method="sidecar-sha256+manifest-listing", identity_matches=True,
                experiment_id=FOUR_CELL_V2_ID,
            )
            if item["role"] in ("geometry", "downsampled"):
                loaded[item["role"]] = _load_object(path, f"{candidate_id} {item['role']}")
        case = by_id[candidate_id]
        field = loaded["downsampled"]
        if case["maps"]["downsampled"]["artifact_sha256"] != sha256(
            _read_bytes(_safe_path(results, f"representatives/{candidate_id}-downsampled-field.json", "field"), "field")
        ).hexdigest():
            raise ValueError(f"{candidate_id} downsampled field does not match the dataset record")
        representatives.append(
            {
                "candidate_id": candidate_id,
                "rank": ranking[candidate_id],
                "label": f"{candidate_id} · rank {ranking[candidate_id]} · "
                f"{len(case['derived_geometry']['stage_centres_m'])}-stage · pitch "
                f"{case['derived_geometry']['pitch_m'] * 1e3:.3g} mm",
                "geometry": _geometry_projection(loaded["geometry"], candidate_id),
                "field": _field_projection(field, candidate_id, stride=1),
                "interior_cusp_z_m": {
                    role: list(case["maps"][role]["interior_cusp_z_m"])
                    for role in ("primary", "downsampled", "enlarged_domain")
                },
                "cusp_targets_m": case["derived_geometry"]["cusp_targets_m"],
                "wall_radius_m": case["derived_geometry"]["wall_radius_m"],
                "failures": list(case["failures"]),
                "mirror_ratio": {
                    "status": "not_published",
                    "reason": "zero candidates passed the four-cusp stability gate, so no connected constant-psi mirror distribution was accepted",
                },
                "sources": [f"fc2-{candidate_id}-geometry", f"fc2-{candidate_id}-downsampled", "fc2-dataset"],
            }
        )
    return {
        "experiment_id": FOUR_CELL_V2_ID,
        "classification": dataset["classification"],
        "status": "preregistered_null_result",
        "commits": {
            "preregistration": FOUR_CELL_V2_PREREGISTRATION_COMMIT,
            "results": FOUR_CELL_V2_RESULTS_COMMIT,
            "accepted_coupling_v3": COUPLING_V3_COMMIT,
        },
        "summary": summary,
        "claim_boundary": dataset["claim_boundary"],
        "ranking_top": [item["candidate_id"] for item in dataset["ranking"][:10]],
        "gpu_replay": [
            {key: item[key] for key in item if not isinstance(item[key], (dict, list))}
            for item in dataset["gpu_replay"]
        ],
        "candidates": compact,
        "representatives": representatives,
        "sources": ["fc2-manifest", "fc2-dataset", "fc2-report"],
    }


# --------------------------------------------------------------------------
# Section 3: four-cell topology search v1 (superseded screening evidence)
# --------------------------------------------------------------------------
def _four_cell_v1(ledger: SourceLedger) -> dict[str, Any]:
    results = FOUR_CELL_V1_RESULTS
    manifest_path = results / "manifest.json"
    manifest_hash = _verify_sidecar_file(manifest_path, "four-cell v1 manifest", FOUR_CELL_V1_MANIFEST_FILE_SHA256)
    manifest = _load_object(manifest_path, "four-cell v1 manifest")
    if _verify_integrity(manifest, "four-cell v1 manifest") != FOUR_CELL_V1_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("four-cell v1 manifest payload differs from reviewed evidence")
    dataset_path = results / "dataset.json"
    dataset_hash = _verify_sidecar_file(dataset_path, "four-cell v1 dataset", FOUR_CELL_V1_DATASET_FILE_SHA256)
    dataset = _load_object(dataset_path, "four-cell v1 dataset")
    if _verify_integrity(dataset, "four-cell v1 dataset") != FOUR_CELL_V1_DATASET_PAYLOAD_SHA256:
        raise ValueError("four-cell v1 dataset payload differs from reviewed evidence")
    if manifest["dataset_payload_sha256"] != FOUR_CELL_V1_DATASET_PAYLOAD_SHA256:
        raise ValueError("four-cell v1 manifest/dataset binding mismatch")
    _verify_git_clean(FOUR_CELL_V1_RESULTS_COMMIT, _repo_relative(manifest_path), _repo_relative(dataset_path))
    for source_id, path, digest in (
        ("fc1-manifest", manifest_path, manifest_hash),
        ("fc1-dataset", dataset_path, dataset_hash),
    ):
        ledger.add(
            source_id, path=_repo_relative(path), sha256_hex=digest,
            commit=FOUR_CELL_V1_RESULTS_COMMIT,
            identity_method="sidecar-sha256+canonical-payload", identity_matches=True,
            experiment_id=FOUR_CELL_V1_ID,
        )
    protocol_status = dataset["protocol_status"]
    if (
        protocol_status["status"] != "development_evidence_only"
        or protocol_status["valid_for_physical_mirror_claims"]
        or protocol_status["preregistered"]
    ):
        raise ValueError("four-cell v1 protocol status differs from archived semantics")
    cases = dataset["cases"]
    if len(cases) != 128:
        raise ValueError("four-cell v1 must contain 128 cases")
    compatible = tuple(sorted(case["case_id"] for case in cases if case["topology"]["compatible"]))
    if compatible != EXPECTED_FOUR_CELL_V1_COMPATIBLE:
        raise ValueError("four-cell v1 compatible candidates differ")
    compact = []
    for case in cases:
        topology = case["topology"]
        compact.append(
            {
                "case_id": case["case_id"],
                "compatible": bool(topology["compatible"]),
                "status": case["status"],
                "stage_count": case["derived_geometry"]["stage_count"],
                "pitch_m": case["design_values"]["stage_pitch_m"],
                "chamber_wall_radius_m": case["design_values"]["chamber_wall_radius_m"],
                "segment_count": topology["segment_count"],
                "overall_confidence": topology["overall_confidence"],
                "failure_codes": list(case["failure_codes"]),
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "cusp_z_m": segment["cusp"]["z_m"],
                        "cusp_kind": segment["cusp"]["kind"],
                        "z_start_m": segment["z_start_m"],
                        "z_end_m": segment["z_end_m"],
                        "wall_b_t": segment["wall_b_t"],
                        "mirror_ratio_high_to_low": segment["mirror_ratio_high_to_low"],
                        "loss_cone_probability": segment["loss_cone_probability"],
                        "loss_cone_probability_upper": segment["loss_cone_probability_upper"],
                    }
                    for segment in topology["segments"]
                ],
                "topology_class": "screening_compatible_four_segment" if topology["compatible"] else "not_compatible",
            }
        )
    return {
        "experiment_id": FOUR_CELL_V1_ID,
        "classification": manifest["classification"],
        "status": "superseded_screening_only",
        "commits": {"results": FOUR_CELL_V1_RESULTS_COMMIT},
        "protocol_status": protocol_status,
        "semantic_correction": {
            key: value for key, value in manifest["semantic_correction"].items()
            if not key.startswith("prior_")
        },
        "summary": {
            key: dataset["summary"][key]
            for key in (
                "evaluated_count", "plasma_residual_root_count", "identifiable_state_count",
                "performance_publication_count", "failure_counts",
            )
        },
        "compatible_case_ids": list(compatible),
        "cases": compact,
        "mirror_proxy_warning": (
            "Coupling v2 used a deprecated same-z centreline-low/wall-high mirror proxy with "
            "roundoff-scale null lows. Mirror ratios and loss-cone probabilities are audit-visible "
            "screening diagnostics only and are not physical mirror evidence."
        ),
        "sources": ["fc1-manifest", "fc1-dataset"],
    }


# --------------------------------------------------------------------------
# Section 4: preregistered L1a geometry sweep v2 (accepted field-only screening)
# --------------------------------------------------------------------------
def _l1a_sweep(ledger: SourceLedger) -> dict[str, Any]:
    results = SWEEP_RESULTS
    manifest_path = results / "manifest.json"
    manifest_hash = _verify_sidecar_file(manifest_path, "sweep manifest", SWEEP_MANIFEST_FILE_SHA256)
    manifest = _load_object(manifest_path, "sweep manifest")
    if _verify_integrity(manifest, "sweep manifest") != SWEEP_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("sweep manifest payload differs from reviewed evidence")
    summary_path = results / "summary.json"
    summary_hash = _verify_sidecar_file(summary_path, "sweep summary", SWEEP_SUMMARY_FILE_SHA256)
    summary = _load_object(summary_path, "sweep summary")
    if _verify_integrity(summary, "sweep summary") != manifest["summary_payload_sha256"]:
        raise ValueError("sweep summary payload binding mismatch")
    raw_path = results / "raw-results.json"
    raw_hash = _verify_sidecar_file(raw_path, "sweep raw results", SWEEP_RAW_FILE_SHA256)
    raw = _load_object(raw_path, "sweep raw results")
    if _verify_integrity(raw, "sweep raw results") != manifest["raw_results_payload_sha256"]:
        raise ValueError("sweep raw payload binding mismatch")
    if (
        manifest["preregistration_commit_sha"] != SWEEP_PREREGISTRATION_COMMIT
        or summary["preregistration_commit_sha"] != SWEEP_PREREGISTRATION_COMMIT
        or summary["terminal_status"] != "ACCEPTED"
        or summary["failed_count"] != 0
        or summary["evaluated_count"] != 96
    ):
        raise ValueError("sweep identity or terminal status differs from accepted evidence")
    _verify_git_clean(
        SWEEP_RESULTS_COMMIT, _repo_relative(manifest_path), _repo_relative(summary_path), _repo_relative(raw_path)
    )
    for source_id, path, digest in (
        ("sweep-manifest", manifest_path, manifest_hash),
        ("sweep-summary", summary_path, summary_hash),
        ("sweep-raw", raw_path, raw_hash),
    ):
        ledger.add(
            source_id, path=_repo_relative(path), sha256_hex=digest, commit=SWEEP_RESULTS_COMMIT,
            identity_method="sidecar-sha256+canonical-payload", identity_matches=True,
            experiment_id=SWEEP_ID,
        )
    cases = raw["cases"]
    if len(cases) != 96 or any(case["status"] != "success" for case in cases):
        raise ValueError("sweep raw results must contain 96 successful cases")
    front = set(summary["nondominated_case_ids"])
    role_map: dict[str, list[str]] = {}
    for role in summary["representative_roles"]:
        role_map.setdefault(role["case_id"], []).append(role["role"])
    compact = []
    status_counts: Counter[str] = Counter()
    cusp_count_counts: Counter[int] = Counter()
    for case in cases:
        qois = case["qois"]
        status_counts[qois["topology_status"]] += 1
        cusp_count_counts[int(qois["axis_cusp_count"])] += 1
        compact.append(
            {
                "case_id": case["case_id"],
                "design_values": case["design_values"],
                "stage_count": case["derived_geometry"]["stage_count"],
                "stage_pitch_m": case["derived_geometry"]["represented_stage_pitch_m"],
                "chamber_length_m": case["derived_geometry"]["chamber_length_m"],
                "chamber_outer_radius_m": case["design_values"]["chamber_outer_radius_m"],
                "axis_cusp_count": qois["axis_cusp_count"],
                "axis_cusp_positions_m": list(qois["axis_cusp_positions_m"]),
                "axis_null_count": qois["axis_null_count"],
                "axis_null_positions_m": list(qois["axis_null_positions_m"]),
                "mirror_ratios": list(qois["mirror_ratios"]),
                "minimum_mirror_ratio": qois["minimum_mirror_ratio"],
                "maximum_mirror_ratio": qois["maximum_mirror_ratio"],
                "field_peak_t": qois["field_peak_t"],
                "centreline_mid_abs_bz_t": qois["centreline_mid_abs_bz_t"],
                "topology_confidence": qois["topology_confidence"],
                "topology_status": qois["topology_status"],
                "boundary_to_peak_ratio": qois["boundary_to_peak_ratio"],
                "field_energy_j": qois["field_energy_j"],
                "nondominated": case["case_id"] in front,
                "roles": sorted(role_map.get(case["case_id"], [])),
                "topology_class": f"axis_cusp_count_{int(qois['axis_cusp_count'])}",
            }
        )
    if sum(case["nondominated"] for case in compact) != 25:
        raise ValueError("sweep nondominated front does not contain 25 cases")
    representatives = []
    artifacts_by_id = {item["case_id"]: item for item in manifest["representative_artifacts"]}
    if tuple(sorted(artifacts_by_id)) != EXPECTED_SWEEP_REPRESENTATIVES:
        raise ValueError("sweep representative identities differ")
    listed = {item["path"]: item for item in manifest["deterministic_files"]}
    by_id = {case["case_id"]: case for case in cases}
    for case_id in EXPECTED_SWEEP_REPRESENTATIVES:
        binding = artifacts_by_id[case_id]
        loaded: dict[str, dict[str, Any]] = {}
        for kind in ("geometry", "downsampled_field", "full_field"):
            record = binding[kind]
            path = _safe_path(results, record["path"], f"{case_id} {kind}")
            digest = _verify_sidecar_file(path, f"{case_id} {kind}", record["file_sha256"])
            if listed[record["path"]]["file_sha256"] != digest:
                raise ValueError(f"{case_id} {kind} manifest listing mismatch")
            _verify_git_clean(SWEEP_RESULTS_COMMIT, _repo_relative(path))
            ledger.add(
                f"sweep-{case_id}-{kind}", path=_repo_relative(path), sha256_hex=digest,
                commit=SWEEP_RESULTS_COMMIT, identity_method="sidecar-sha256+manifest-listing",
                identity_matches=True, experiment_id=SWEEP_ID,
            )
            if kind != "full_field":
                value = _load_object(path, f"{case_id} {kind}")
                if _verify_integrity(value, f"{case_id} {kind}") != record["payload_sha256"]:
                    raise ValueError(f"{case_id} {kind} payload binding mismatch")
                loaded[kind] = value
        field = loaded["downsampled_field"]
        if field["input"]["name"] != case_id:
            raise ValueError(f"{case_id} field identity mismatch")
        case = by_id[case_id]
        representatives.append(
            {
                "case_id": case_id,
                "roles": sorted(role_map[case_id]),
                "label": f"{case_id} · {', '.join(sorted(role_map[case_id]))}",
                "geometry": _geometry_projection(loaded["geometry"], case_id),
                "field": _field_projection(field, case_id, stride=1),
                "axis_cusp_positions_m": list(case["qois"]["axis_cusp_positions_m"]),
                "axis_null_positions_m": list(case["qois"]["axis_null_positions_m"]),
                "mirror_ratio": {
                    "status": "recorded_l1a_screening_qoi",
                    "values": list(case["qois"]["mirror_ratios"]),
                    "reason": "per-cell centreline mirror ratios recorded by the preregistered sweep QoI policy; field-only screening, not a plasma confinement claim",
                },
                "sources": [f"sweep-{case_id}-geometry", f"sweep-{case_id}-downsampled_field", "sweep-raw"],
            }
        )
    return {
        "experiment_id": SWEEP_ID,
        "classification": summary["classification"],
        "status": "accepted_field_only_screening",
        "commits": {
            "preregistration": SWEEP_PREREGISTRATION_COMMIT,
            "results": SWEEP_RESULTS_COMMIT,
        },
        "summary": {
            key: summary[key]
            for key in (
                "requested_count", "evaluated_count", "failed_count", "nondominated_count",
                "unique_representative_count", "terminal_status", "qoi_ranges",
                "representative_roles", "screening_level",
            )
        },
        "gates": [
            {
                "gate_id": gate["gate_id"],
                "passed": gate["passed"],
                "failure_count": gate["failure_count"],
                "observed": gate["observed"],
                "definition": gate["definition"],
            }
            for gate in summary["terminal_gates"]
        ],
        "topology_status_counts": dict(sorted(status_counts.items())),
        "axis_cusp_count_counts": {str(k): v for k, v in sorted(cusp_count_counts.items())},
        "environment": summary["environment"],
        "cases": compact,
        "representatives": representatives,
        "sources": ["sweep-manifest", "sweep-summary", "sweep-raw"],
    }


# --------------------------------------------------------------------------
# Section 5: accepted L1a axisymmetric v1.2 example artifacts
# --------------------------------------------------------------------------
def _axisymmetric(ledger: SourceLedger) -> dict[str, Any]:
    module = _import_script(AXISYMMETRIC_GENERATOR, "plasma_topology_axisymmetric_source")
    payload = module.build_payload()
    manifest_path = module.DEFAULT_MANIFEST
    _verify_git_clean(
        AXISYMMETRIC_RESULTS_COMMIT,
        _repo_relative(manifest_path),
        *[_repo_relative(manifest_path.with_name(item[1])) for item in module.EXPECTED_DESIGNS],
    )
    ledger.add(
        "axi-manifest", path=_repo_relative(manifest_path),
        sha256_hex=payload["manifest"]["file_sha256"], commit=AXISYMMETRIC_RESULTS_COMMIT,
        identity_method="sidecar-sha256+canonical-v1.2-payload (authoritative reload)",
        identity_matches=True, experiment_id=AXISYMMETRIC_ID,
    )
    designs = []
    for design in payload["designs"]:
        ledger.add(
            f"axi-{design['id']}", path=_repo_relative(manifest_path.with_name(design["artifact"])),
            sha256_hex=design["file_sha256"], commit=AXISYMMETRIC_RESULTS_COMMIT,
            identity_method="sidecar-sha256+canonical-v1.2-payload (authoritative reload)",
            identity_matches=True, experiment_id=AXISYMMETRIC_ID,
        )
        field = design["field"]
        wall = design["profiles"]["wall"]
        centre = design["profiles"]["centreline"]
        designs.append(
            {
                "id": design["id"],
                "label": design["label"],
                "field": {
                    "layout": field["layout"],
                    "source_grid": [
                        design["input"]["domain"]["radial_intervals"] + 1,
                        design["input"]["domain"]["axial_intervals"] + 1,
                    ],
                    "embedded_stride": field["downsample_stride"],
                    "rounding": "exact artifact values (artifact is already the downsampled v1.2 map)",
                    "r_m": field["r_m"],
                    "z_m": field["z_m"],
                    "b_magnitude_t": field["b_magnitude_t"],
                    "psi_wb": field["psi_wb"],
                    "b_magnitude_max_t": design["summary"]["b_magnitude_max_t"],
                    "wall_profile": {
                        "sampled_r_m": wall["sampled_r_m"],
                        "requested_r_m": wall["requested_r_m"],
                        "z_m": wall["z_m"],
                        "b_r_t": wall["b_r_t"],
                        "b_z_t": wall["b_z_t"],
                        "b_magnitude_t": [
                            hypot(br, bz) for br, bz in zip(wall["b_r_t"], wall["b_z_t"], strict=True)
                        ],
                    },
                    "centreline_profile": {
                        "z_m": centre["z_m"], "b_z_t": centre["b_z_t"], "b_r_t": centre["b_r_t"],
                    },
                    "sources": design["input"]["sources"],
                    "diagnostics": {
                        "converged": design["diagnostics"]["converged"],
                        "iterations": design["diagnostics"]["iterations"],
                        "relative_residual_l2": design["diagnostics"]["relative_residual_l2"],
                        "backend": design["diagnostics"]["backend"],
                    },
                    "axis_topology": design["summary"]["topology"],
                },
                "geometry": None,
                "mirror_ratio": {
                    "status": "not_computed",
                    "reason": "the accepted v1.2 example artifacts record axis-null topology only",
                },
                "limitations": design["limitations"],
                "sources": [f"axi-{design['id']}", "axi-manifest"],
            }
        )
    return {
        "experiment_id": AXISYMMETRIC_ID,
        "status": "accepted_serialization_examples",
        "commits": {"results": AXISYMMETRIC_RESULTS_COMMIT},
        "schema_version": payload["manifest"]["schema_version"],
        "warning": payload["warning"],
        "designs": designs,
        "sources": ["axi-manifest"] + [f"axi-{design['id']}" for design in designs],
    }


# --------------------------------------------------------------------------
# Section 6: qualified P2 divergent-exit FEM field (the orbit campaigns' field)
# --------------------------------------------------------------------------
def _line_profile(
    coordinates: np.ndarray,
    fields: Mapping[str, np.ndarray],
    radial_target: float,
    z_min: float,
    z_max: float,
    samples: int,
) -> dict[str, Any]:
    """Nearest-vertex sampling of recovered vertex fields along r = radial_target."""

    in_range = (coordinates[:, 1] >= z_min) & (coordinates[:, 1] <= z_max)
    subset = coordinates[in_range]
    if len(subset) < samples:
        raise ValueError("P2 viewer has too few vertices in the requested wall window")
    bins = np.clip(((subset[:, 1] - z_min) / (z_max - z_min) * samples).astype(np.int64), 0, samples - 1)
    distance = np.abs(subset[:, 0] - radial_target)
    order = np.lexsort((distance, bins))
    ordered_bins = bins[order]
    _, first = np.unique(ordered_bins, return_index=True)
    chosen_local = order[first]
    chosen_bins = bins[chosen_local]
    chosen = np.flatnonzero(in_range)[chosen_local]
    targets = np.arange(samples)
    z_centres = z_min + (targets + 0.5) * (z_max - z_min) / samples
    result: dict[str, Any] = {
        "radial_target_m": radial_target,
        "z_window_m": [z_min, z_max],
        "sample_count": samples,
        "method": "nearest recovered P2 vertex per axial bin, linearly interpolated across empty bins",
        "maximum_radial_offset_m": _round(float(np.max(np.abs(coordinates[chosen, 0] - radial_target)))),
        "z_m": _round_list(z_centres),
    }
    for key, values in fields.items():
        result[key] = _round_list(np.interp(targets, chosen_bins, values[chosen]))
    return result


def _p2_divergent_exit(ledger: SourceLedger) -> dict[str, Any]:
    fem = _import_script(FEM_GENERATOR, "plasma_topology_fem_source")
    manifest_path = P2_ROOT / "manifest.json"
    manifest_hash = fem._stream_hash(manifest_path, "P2 manifest", P2_MANIFEST_FILE_SHA256)
    manifest = fem._load_object(manifest_path, "P2 manifest")
    manifest_payload_hash = fem._verify_integrity(manifest, "P2 manifest")
    if (
        manifest.get("classification") != fem.CLASSIFICATION
        or len(manifest.get("designs", [])) != 1
        or manifest["designs"][0].get("config_id") != f"{P2_DESIGN_ID}-v1"
        or manifest["designs"][0].get("qualification_status") != "NUMERICAL_P2_QUALIFIED"
    ):
        raise ValueError("P2 divergent-exit manifest is not the qualified accepted record")
    record = manifest["designs"][0]
    artifact_path = fem._safe_path(P2_ROOT, record["artifact"], "P2 artifact")
    artifact_hash = fem._stream_hash(artifact_path, "P2 artifact", record["artifact_file_sha256"])
    if artifact_hash != P2_RESULT_FILE_SHA256:
        raise ValueError("P2 result artifact SHA-256 differs from the orbit-campaign authority")
    viewer_path = fem._safe_path(P2_ROOT, record["viewer"], "P2 viewer")
    viewer_hash = fem._stream_hash(viewer_path, "P2 viewer", record["viewer_file_sha256"])
    if viewer_hash != P2_VIEWER_FILE_SHA256:
        raise ValueError("P2 viewer SHA-256 differs from the pinned qualified viewer")
    viewer = fem._load_object(viewer_path, "P2 viewer")
    viewer_payload_hash = fem._verify_integrity(viewer, "P2 viewer")
    if viewer.get("artifact_payload_sha256") != record["artifact_payload_sha256"]:
        raise ValueError("P2 viewer-to-artifact payload binding differs")
    _verify_git_clean(P2_RESULTS_COMMIT, _repo_relative(manifest_path))
    raster = fem._viewer_projection(viewer, P2_DESIGN_ID)
    ledger.add(
        "p2-manifest", path=_repo_relative(manifest_path), sha256_hex=manifest_hash,
        commit=P2_RESULTS_COMMIT, identity_method="pinned-file-sha256+canonical-payload",
        identity_matches=True, experiment_id=P2_ID,
    )
    ledger.add(
        "p2-artifact", path=_repo_relative(artifact_path), sha256_hex=artifact_hash,
        commit=P2_RESULTS_COMMIT, identity_method="manifest-file-sha256 (streamed) + orbit v4 authority binding",
        identity_matches=True, experiment_id=P2_ID,
        note="Git LFS-backed result artifact; verified as materialised bytes",
    )
    ledger.add(
        "p2-viewer", path=_repo_relative(viewer_path), sha256_hex=viewer_hash,
        commit=P2_RESULTS_COMMIT, identity_method="manifest-file-sha256 (streamed) + canonical-payload",
        identity_matches=True, experiment_id=P2_ID,
    )
    coordinates = np.asarray(viewer["coordinates_rz_m"], dtype=np.float64)
    fields = {
        key: np.asarray(viewer["vertex_fields"][key], dtype=np.float64)
        for key in ("psi_wb_per_rad", "b_r_t", "b_z_t")
    }
    fields["b_magnitude_t"] = np.hypot(fields["b_r_t"], fields["b_z_t"])
    fields["abs_b_r_t"] = np.abs(fields["b_r_t"])
    wall_profile = _line_profile(
        coordinates, fields, P2_WALL_RADIUS_M, P2_WALL_Z_RANGE_M[0], P2_WALL_Z_RANGE_M[1], P2_PROFILE_SAMPLES
    )
    axis_profile = _line_profile(
        coordinates, fields, 0.0, P2_WALL_Z_RANGE_M[0], P2_WALL_Z_RANGE_M[1], P2_PROFILE_SAMPLES
    )
    convergence = record["convergence"]
    return {
        "experiment_id": P2_ID,
        "design_id": P2_DESIGN_ID,
        "status": "NUMERICAL_P2_QUALIFIED",
        "classification": record["classification"],
        "commits": {"results": P2_RESULTS_COMMIT},
        "identity": {
            "manifest_file_sha256": manifest_hash,
            "manifest_payload_sha256": manifest_payload_hash,
            "artifact_file_sha256": artifact_hash,
            "artifact_payload_sha256": record["artifact_payload_sha256"],
            "viewer_file_sha256": viewer_hash,
            "viewer_payload_sha256": viewer_payload_hash,
        },
        "levels": [
            {
                "level": int(run["level"]),
                "p2_dofs": int(run["mesh_quality"]["p2_dofs"]),
                "triangles": int(run["mesh_quality"]["triangles"]),
                "relative_true_residual_l2": run["relative_true_residual_l2"],
            }
            for run in record["runs"]
        ],
        "gates": {
            "two_successive_changes_below_one_percent": convergence["two_successive_less_than_one_percent"],
            "stable_positive_order": convergence["stable_positive_order"],
            "adjacent_size_growth": convergence["adjacent_size_growth_gate"],
            "phase_matched_domain_expansion": convergence["phase_matched_domain_expansion_gate"],
            "less_than_one_percent_reached": convergence["less_than_one_percent_reached"],
        },
        "observed_orders": convergence["observed_orders_from_actual_qoi_h"],
        "qois_bz_t": viewer["qois_bz_t"],
        "raster": {
            "grid": raster["grid"],
            "extent_rz_m": raster["extent_rz_m"],
            "fields": {
                key: raster["fields"][key] for key in ("b_magnitude_t", "psi_wb_per_rad", "b_r_t")
            },
            "regions": raster["regions"],
            "source_vertices": raster["source_vertices"],
            "source_triangles": raster["source_triangles"],
            "method": (
                f"{raster['grid']['width']}x{raster['grid']['height']} cell-mean raster of "
                f"{raster['source_vertices']} recovered P2 vertices, 8 significant digits"
            ),
        },
        "wall_profile": wall_profile,
        "axis_profile": axis_profile,
        "wall_normal_maxima": {
            "authority": "dashboard_derived_display_diagnostic_not_accepted_cusp_evidence",
            "definition": (
                "strict interior local maxima of |B_r| along the sampled r = 2 mm dielectric wall "
                f"line, merged within {P2_MAXIMA_MERGE_DISTANCE_M * 1e3:g} mm keeping the larger value"
            ),
            "maxima": _local_maxima(wall_profile["z_m"], wall_profile["abs_b_r_t"]),
        },
        "regular_plasma_domain_m": {
            "r_min": 0.0, "r_max": P2_WALL_RADIUS_M,
            "z_min": P2_WALL_Z_RANGE_M[0], "z_max": P2_WALL_Z_RANGE_M[1],
        },
        "mirror_ratio": {
            "status": "not_published",
            "reason": "orbit v4 publication boundary forbids mirror-formula publication; wall-loss probabilities are the direct estimands",
        },
        "limitations": sorted(set(manifest["limitations"]) | set(viewer["limitations"])),
        "sources": ["p2-manifest", "p2-artifact", "p2-viewer"],
    }


# --------------------------------------------------------------------------
# Section 7: coupling v4 wall-cusp held-out validation (v1, v2 failed immutable)
# --------------------------------------------------------------------------
def _manifest_listed_file(
    results: Path, relative: str, listed: Mapping[str, Mapping[str, Any]], label: str
) -> tuple[bytes, dict[str, Any], bool]:
    entry = listed.get(relative)
    if entry is None:
        raise ValueError(f"{label} is not listed in its manifest")
    path = _safe_path(results, relative, label)
    raw = _read_bytes(path, label)
    byte_matches = sha256(raw).hexdigest() == entry["byte_sha256"]
    method = entry["identity_method"]
    if method == "byte-and-canonical-json-sha256":
        value = _load_json_bytes(raw, label)
        semantic = _canonical_hash(value)
    elif method == "byte-and-normalized-lf-text-sha256":
        value = {}
        semantic = sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    else:
        raise ValueError(f"{label} identity method {method!r} is unsupported")
    if semantic != entry["semantic_sha256"]:
        raise ValueError(f"{label} semantic SHA-256 mismatch")
    return raw, value, byte_matches


def _wall_cusp_validation(ledger: SourceLedger) -> dict[str, Any]:
    runs = []
    for version, results, commit, manifest_file_hash, manifest_payload_hash, experiment_id in (
        ("v1", WCVAL_V1_RESULTS, WCVAL_V1_RESULTS_COMMIT, WCVAL_V1_MANIFEST_FILE_SHA256, WCVAL_V1_MANIFEST_PAYLOAD_SHA256, WCVAL_V1_ID),
        ("v2", WCVAL_V2_RESULTS, WCVAL_V2_RESULTS_COMMIT, WCVAL_V2_MANIFEST_FILE_SHA256, WCVAL_V2_MANIFEST_PAYLOAD_SHA256, WCVAL_V2_ID),
    ):
        manifest_path = results / "manifest.json"
        digest = _file_sha256(manifest_path, f"{experiment_id} manifest")
        if digest != manifest_file_hash:
            raise ValueError(f"{experiment_id} manifest file SHA-256 mismatch")
        manifest = _load_object(manifest_path, f"{experiment_id} manifest")
        if _verify_integrity(manifest, f"{experiment_id} manifest", "semantic_integrity") != manifest_payload_hash:
            raise ValueError(f"{experiment_id} manifest payload differs from recorded evidence")
        if (
            manifest["experiment_id"] != experiment_id
            or manifest["accepted_coupling_commit_sha"] != COUPLING_V4_COMMIT
            or manifest["summary"]["criterion_numerically_promoted"] is not False
        ):
            raise ValueError(f"{experiment_id} manifest identity mismatch")
        listed = {item["path"]: item for item in manifest["artifacts"]}
        _verify_git_clean(
            commit, _repo_relative(manifest_path), _repo_relative(results / "failure.json"),
            _repo_relative(results / "report.md"),
        )
        ledger.add(
            f"wcval-{version}-manifest", path=_repo_relative(manifest_path), sha256_hex=digest, commit=commit,
            identity_method="pinned-file-sha256+canonical-semantic-payload", identity_matches=True,
            experiment_id=experiment_id,
        )
        failure_raw, failure, failure_bytes_match = _manifest_listed_file(results, "failure.json", listed, f"{experiment_id} failure")
        report_raw, _, report_bytes_match = _manifest_listed_file(results, "report.md", listed, f"{experiment_id} report")
        crlf_note = (
            "manifest byte_sha256 was recorded on CRLF worktree bytes; the committed LF blob "
            "matches only the recorded canonical/normalized semantic identity"
        )
        ledger.add(
            f"wcval-{version}-failure", path=_repo_relative(results / "failure.json"),
            sha256_hex=sha256(failure_raw).hexdigest(), commit=commit,
            identity_method="manifest byte-and-canonical-json-sha256",
            identity_matches=failure_bytes_match, experiment_id=experiment_id,
            note=None if failure_bytes_match else crlf_note,
        )
        ledger.add(
            f"wcval-{version}-report", path=_repo_relative(results / "report.md"),
            sha256_hex=sha256(report_raw).hexdigest(), commit=commit,
            identity_method="manifest byte-and-normalized-lf-text-sha256",
            identity_matches=report_bytes_match, experiment_id=experiment_id,
            note=None if report_bytes_match else crlf_note,
        )
        run: dict[str, Any] = {
            "version": version,
            "experiment_id": experiment_id,
            "status": manifest["status"],
            "commits": {
                "preregistration": manifest["preregistration_commit_sha"],
                "results": commit,
                "accepted_coupling_v4": COUPLING_V4_COMMIT,
            },
            "summary": manifest["summary"],
            "failure": failure["failure"],
            "sources": [f"wcval-{version}-manifest", f"wcval-{version}-failure", f"wcval-{version}-report"],
        }
        if version == "v1":
            run["coverage"] = failure["coverage"]
            run["materialized_maps"] = [
                {
                    key: item[key]
                    for key in (
                        "role", "full_map_sha256", "relative_residual_l2", "iterations",
                        "boundary_to_peak_ratio", "field_gates_passed",
                    )
                }
                for item in failure["materialized_maps"]
            ]
            run["read_only_reconstruction"] = failure["read_only_reconstruction_from_exact_maps"]
            run["promotion"] = failure["promotion"]
            run["case_id"] = failure["failure"]["case_id"]
        else:
            run["coverage"] = {
                key: failure["summary"][key]
                for key in (
                    "declared_case_count", "declared_map_count", "attempted_case_count",
                    "materialized_map_count", "held_out_outcome_count", "opaque_projection_count",
                )
            }
            run["launcher_finalization_failure"] = failure["launcher_finalization_failure"]
            run["v1_disclosure"] = failure["v1_disclosure"]
            run["promotion"] = {
                "criterion_numerically_promoted": failure["summary"]["criterion_numerically_promoted"],
                "search_v3_ready": failure["summary"]["search_v3_ready"],
                "plasma_coupling_ready": failure["summary"]["plasma_coupling_ready"],
                "hardware_validated": False,
            }
        runs.append(run)
    return {
        "criterion": {
            "id": "cft-hemp-wall-cusp-v4",
            "accepted_coupling_commit": COUPLING_V4_COMMIT,
            "definition": (
                "a cusp candidate is a quadratically interpolated local maximum of wall-normal "
                "|B_r(r_wall, z)| with physical prominence and separation; consecutive stable wall-cusp "
                "planes define a cell whose core must be predominantly axial (|B_z|/|B| thresholds); "
                "X/O/null and closed-island results are diagnostics only"
            ),
            "development_manifest": "frozen 56-case characterization family (development_non_validation)",
            "promotion_status": "not_promoted",
        },
        "runs": runs,
        "sources": [source for run in runs for source in run["sources"]],
    }


# --------------------------------------------------------------------------
# Section 8: full-orbit wall-loss campaign v4 (read from its recorded branch)
# --------------------------------------------------------------------------
def _orbit_wall_loss_v4(ledger: SourceLedger) -> dict[str, Any]:
    prefix = ORBIT_V4_PREFIX
    manifest_raw = _git_show(ORBIT_V4_RESULTS_COMMIT, prefix + "results/manifest.json")
    manifest_hash = sha256(manifest_raw).hexdigest()
    if manifest_hash != ORBIT_V4_MANIFEST_FILE_SHA256:
        raise ValueError("orbit v4 result manifest SHA-256 differs from the pinned recorded result")
    manifest = _load_json_bytes(manifest_raw, "orbit v4 manifest")
    if (
        manifest["experiment_id"] != ORBIT_V4_ID
        or manifest["state"] != "accepted_result"
        or manifest["manifest_is_sole_completion_marker"] is not True
    ):
        raise ValueError("orbit v4 manifest is not an accepted completion marker")
    listed = {item["path"]: item for item in manifest["artifacts"] if item["type"] == "file"}
    ledger.add(
        "orbit4-manifest", path=prefix + "results/manifest.json", sha256_hex=manifest_hash,
        commit=ORBIT_V4_RESULTS_COMMIT, identity_method="pinned-file-sha256 via git show",
        identity_matches=True, experiment_id=ORBIT_V4_ID,
        note=f"read from branch {ORBIT_V4_BRANCH} without checkout",
    )
    terminal_raw = _git_show(ORBIT_V4_RESULTS_COMMIT, prefix + "results/terminal.json")
    terminal_hash = sha256(terminal_raw).hexdigest()
    if terminal_hash != manifest["terminal_byte_sha256"]:
        raise ValueError("orbit v4 terminal.json bytes differ from the manifest marker")
    terminal = _load_json_bytes(terminal_raw, "orbit v4 terminal")
    payload = terminal["payload"]
    if (
        terminal["state"] != "accepted_result"
        or payload["status"] != "accepted"
        or payload["evidentiary"] is not True
        or payload["gates"]["passed"] is not True
        or payload["gates_binding"] is not True
        or payload["campaign_count"] != 9
        or payload["orbit_count"] != 4608
        or terminal["primary_error"] is not None
    ):
        raise ValueError("orbit v4 terminal record is not an accepted evidentiary result")
    ledger.add(
        "orbit4-terminal", path=prefix + "results/terminal.json", sha256_hex=terminal_hash,
        commit=ORBIT_V4_RESULTS_COMMIT, identity_method="manifest terminal_byte_sha256 via git show",
        identity_matches=True, experiment_id=ORBIT_V4_ID,
    )

    def listed_artifact(relative: str, source_id: str) -> dict[str, Any]:
        entry = listed.get(relative)
        if entry is None or entry["contract"] != "hash-sidecar":
            raise ValueError(f"orbit v4 {relative} is not a hash-sidecar artifact")
        raw = _git_show(ORBIT_V4_RESULTS_COMMIT, prefix + "results/" + relative)
        digest = sha256(raw).hexdigest()
        if digest != entry["byte_sha256"]:
            raise ValueError(f"orbit v4 {relative} bytes differ from the manifest sidecar")
        sidecar = _load_json_bytes(
            _git_show(ORBIT_V4_RESULTS_COMMIT, prefix + "results/" + entry["sidecar"]), f"{relative} sidecar"
        )
        if sidecar["byte_sha256"] != digest or sidecar["artifact"] != relative:
            raise ValueError(f"orbit v4 {relative} sidecar binding mismatch")
        ledger.add(
            source_id, path=prefix + "results/" + relative, sha256_hex=digest,
            commit=ORBIT_V4_RESULTS_COMMIT, identity_method="manifest byte_sha256 + sidecar via git show",
            identity_matches=True, experiment_id=ORBIT_V4_ID,
        )
        return _load_json_bytes(raw, relative)

    protocol = listed_artifact("artifacts/protocol.json", "orbit4-protocol")
    authority = listed_artifact("artifacts/p2-input-authority.json", "orbit4-p2-authority")
    convergence = listed_artifact("artifacts/probability-convergence.json", "orbit4-convergence")
    if (
        authority["design_id"] != P2_DESIGN_ID
        or authority["manifest_file_sha256"] != P2_MANIFEST_FILE_SHA256
        or authority["result_file_sha256"] != P2_RESULT_FILE_SHA256
        or authority["qualification_status"] != "NUMERICAL_P2_QUALIFIED"
    ):
        raise ValueError("orbit v4 P2 input authority does not match the qualified divergent-exit field")
    if (
        protocol["authority"]["orbit_mc_commit"] != EVIDENCE_BASE_COMMIT
        or protocol["authority"]["p2_evidence_commit"] != P2_RESULTS_COMMIT
    ):
        raise ValueError("orbit v4 protocol authorities differ from the pinned evidence commits")
    campaigns = []
    for case in ORBIT_V4_CASES:
        summary = listed_artifact(f"artifacts/summaries/{case}.json", f"orbit4-summary-{case}")
        recorded = payload["campaigns"][case]
        overall = summary["summary"]
        for key in ("wall_hit", "reflected", "escaped", "incomplete", "termination_counts", "trial_count"):
            if overall[key] != recorded[key]:
                raise ValueError(f"orbit v4 {case} summary differs from the terminal record")
        strata = summary["strata"]
        if len(strata) != 32 or sum(stratum["trials"] for stratum in strata) != 512:
            raise ValueError(f"orbit v4 {case} does not contain 32 strata of 512 launches")
        for stratum in strata:
            counts = stratum["termination_counts"]
            if (
                stratum["wall_hit"]["successes"] != counts["wall_hit"]
                or stratum["reflected"]["successes"] != counts["reflected"]
                or stratum["domain_escape"]["successes"] != counts["domain_escape"]
                or stratum["wall_hit"]["method"] != "wilson-95"
            ):
                raise ValueError(f"orbit v4 {case} stratum interval bookkeeping is inconsistent")
        role, policy = case.split("-")
        campaigns.append(
            {
                "case_id": case,
                "map_role": role,
                "timestep_policy": policy,
                "ensemble_id": overall["ensemble_id"],
                "trial_count": overall["trial_count"],
                "wall_hit": overall["wall_hit"],
                "reflected": overall["reflected"],
                "escaped": overall["escaped"],
                "incomplete": overall["incomplete"],
                "termination_counts": overall["termination_counts"],
                "diagnostics": {
                    "maximum_relative_energy_error": summary["diagnostics"]["maximum_relative_energy_error"],
                    "wall_endpoint_error_max_m": summary["diagnostics"]["wall_endpoint_error_max_m"],
                    "final_velocity_equals_event_velocity_count": summary["diagnostics"]["final_velocity_equals_event_velocity_count"],
                    "magnetic_moment_variation": summary["diagnostics"]["magnetic_moment_variation_diagnostic"],
                    "steps": summary["diagnostics"]["steps"],
                    "runtime_max_b_t": summary["diagnostics"]["runtime_max_b_t"],
                },
                "preflight": summary["preflight"],
                "strata": [
                    {
                        "cell_id": stratum["cell_id"],
                        "kinetic_energy_ev": stratum["kinetic_energy_ev"],
                        "pitch_angle_deg": stratum["pitch_angle_deg"],
                        "parallel_direction": stratum["parallel_direction"],
                        "trials": stratum["trials"],
                        "wall_hit": stratum["wall_hit"],
                        "reflected": stratum["reflected"],
                        "domain_escape": stratum["domain_escape"],
                        "timeout": stratum["timeout"],
                    }
                    for stratum in strata
                ],
                "sources": [f"orbit4-summary-{case}", "orbit4-terminal"],
            }
        )
    launches = protocol["launches"]
    return {
        "experiment_id": ORBIT_V4_ID,
        "status": "accepted_evidentiary_result",
        "classification": payload["classification"],
        "branch": ORBIT_V4_BRANCH,
        "commits": {
            "preregistration": ORBIT_V4_PREREGISTRATION_COMMIT,
            "results": ORBIT_V4_RESULTS_COMMIT,
            "orbit_mc": protocol["authority"]["orbit_mc_commit"],
            "p2_evidence": protocol["authority"]["p2_evidence_commit"],
        },
        "prior_campaigns": {
            version: {
                key: protocol["prior_campaign_disclosure"][version][key]
                for key in (
                    "branch", "terminal_state", "primary_error_type", "primary_error_message",
                    "root_cause", "preregistration_commit", "result_commit",
                )
            }
            for version in ORBIT_PRIOR_CAMPAIGNS
        },
        "prior_pattern": protocol["prior_campaign_disclosure"]["common_pattern"],
        "p2_input_authority": authority,
        "design": {
            "launches_per_case": launches["launches_per_case"],
            "strata_per_case": launches["strata_per_case"],
            "stratum_dimensions": launches["stratum_dimensions"],
            "energies_ev": launches["energies_ev"],
            "pitch_angles_deg": launches["pitch_angles_deg"],
            "directions": launches["directions"],
            "gyrophase_count": launches["gyrophase_count"],
            "independent_repeats_per_stratum": launches["independent_repeats_per_stratum"],
            "estimator_policy": launches["estimator_policy"],
            "position_seeds": launches["position_seeds"],
            "regular_plasma_domain_m": protocol["field_adapter"]["regular_plasma_domain"],
            "field_adapter_maps": {
                role: {
                    "radial_intervals": item["radial_intervals"],
                    "axial_intervals": item["axial_intervals"],
                    "checkpoint_path": item["checkpoint_path"],
                    "checkpoint_file_sha256": item["checkpoint_file_sha256"],
                }
                for role, item in protocol["field_adapter"]["maps"].items()
            },
        },
        "gates": payload["gates"],
        "validators": payload["validators"],
        "orbit_count": payload["orbit_count"],
        "campaign_count": payload["campaign_count"],
        "limitations": payload["limitations"],
        "publication_boundary": protocol["publication_boundary"],
        "convergence": convergence,
        "campaigns": campaigns,
        "sources": [
            "orbit4-manifest", "orbit4-terminal", "orbit4-protocol", "orbit4-p2-authority", "orbit4-convergence",
        ] + [f"orbit4-summary-{case}" for case in ORBIT_V4_CASES],
    }


# --------------------------------------------------------------------------
# Validation ledger and provenance
# --------------------------------------------------------------------------
def _claims(sections: Mapping[str, Any]) -> list[dict[str, Any]]:
    char = sections["characterization"]
    fc2 = sections["four_cell_v2"]
    fc1 = sections["four_cell_v1"]
    sweep = sections["l1a_sweep"]
    p2 = sections["p2_divergent_exit"]
    wcval = sections["coupling_v4_validation"]
    orbit = sections["orbit_wall_loss_v4"]
    v1_run, v2_run = wcval["runs"]
    return [
        {
            "claim": "Accepted L1a equivalent-current field solver artifacts (v1.2 canonical serialization) for three hypothetical designs",
            "status": "accepted_numerical_evidence",
            "experiment_id": AXISYMMETRIC_ID,
            "evidence": "3 artifacts reload byte-identically under the authoritative v1.2 canonical path; axis-null topology recorded",
            "sources": sections["axisymmetric"]["sources"],
        },
        {
            "claim": f"Preregistered L1a geometry sweep v2: {sweep['summary']['evaluated_count']} designs evaluated, {sweep['summary']['failed_count']} failures, all 7 terminal gates passed, {sweep['summary']['nondominated_count']} nondominated",
            "status": "accepted_numerical_evidence",
            "experiment_id": SWEEP_ID,
            "evidence": "field-only screening; axis cusp counts 3-5 per design; not hardware-valid",
            "sources": sweep["sources"],
        },
        {
            "claim": f"P2 FEM divergent-exit field is NUMERICAL_P2_QUALIFIED ({len(p2['levels'])} adaptive levels, {p2['levels'][-1]['p2_dofs']} P2 DOFs)",
            "status": "accepted_numerical_evidence",
            "experiment_id": P2_ID,
            "evidence": "two successive sub-percent QoI changes, stable positive observed order, phase-matched domain expansion",
            "sources": p2["sources"],
        },
        {
            "claim": f"Full-orbit wall-loss v4: {orbit['orbit_count']} collisionless test-particle orbits across {orbit['campaign_count']} campaigns; all binding gates passed",
            "status": "accepted_numerical_evidence",
            "experiment_id": ORBIT_V4_ID,
            "evidence": "wall-hit / reflection / escape probabilities with Wilson 95% intervals; not PIC, not self-consistent, no mirror-formula or performance publication",
            "sources": orbit["sources"],
        },
        {
            "claim": f"CFT topology characterization v1: {char['summary']['evaluated_count']} designs, {char['summary']['stable_eligible_cusp_count']} stable eligible cusps, {char['summary']['stable_eligible_cell_count']} stable eligible cells",
            "status": "recorded_developmental_characterization",
            "experiment_id": CHARACTERIZATION_ID,
            "evidence": f"{sum(char['root_class_counts'].values())} clustered primary-map vector nulls were all excluded (hardware, finite box, outside channel, or unresolved separatrix)",
            "sources": char["sources"],
        },
        {
            "claim": f"Four-cell topology search v2 (preregistered): {fc2['summary']['stable_count']} of {fc2['summary']['evaluated_count']} candidates achieve a stable four-cell topology",
            "status": "preregistered_null_result",
            "experiment_id": FOUR_CELL_V2_ID,
            "evidence": "every candidate failed TOPOLOGY_COUNT and TOPOLOGY_UNSTABLE; no physical mirror or plasma claim",
            "sources": fc2["sources"],
        },
        {
            "claim": f"Four-cell topology search v1: {len(fc1['compatible_case_ids'])} of {fc1['summary']['evaluated_count']} candidates screened as four-segment compatible",
            "status": "superseded_screening_only",
            "experiment_id": FOUR_CELL_V1_ID,
            "evidence": "not preregistered; coupling v2 same-z mirror proxy deprecated; invalid for physical mirror or performance claims",
            "sources": fc1["sources"],
        },
        {
            "claim": "Coupling v4 wall-cusp criterion promoted by held-out validation",
            "status": "rejected_failed_immutable_runs",
            "experiment_id": f"{WCVAL_V1_ID}, {WCVAL_V2_ID}",
            "evidence": (
                f"v1 failed at {v1_run['failure']['phase']} after 1 of 24 cases ({v1_run['failure']['exception_type']}); "
                f"v2 failed at {v2_run['failure']['phase']} before any held-out access; criterion not promoted"
            ),
            "sources": wcval["sources"],
        },
        {
            "claim": "Full-orbit wall-loss campaigns v1-v3",
            "status": "rejected_code_failures",
            "experiment_id": "cft-orbit-wall-loss-v1, v2, v3",
            "evidence": orbit["prior_pattern"],
            "sources": ["orbit4-protocol"],
        },
        {
            "claim": "Kinetic PIC / self-consistent plasma results",
            "status": "foundation_only_no_results",
            "experiment_id": "pic workstream",
            "evidence": "package foundation exists; no accepted PIC result is recorded in any committed artifact",
            "sources": [],
        },
    ]


def _generated_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
    return _git_commit_time(ORBIT_V4_RESULTS_COMMIT)


def _generator_sha256() -> str:
    raw = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return sha256(raw).hexdigest()


def _template_sha256() -> str:
    return sha256(TEMPLATE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_payload() -> dict[str, Any]:
    """Verify every pinned source and assemble the embedded dashboard payload."""

    for commit in (
        EVIDENCE_BASE_COMMIT, CHARACTERIZATION_RESULTS_COMMIT, FOUR_CELL_V2_RESULTS_COMMIT,
        FOUR_CELL_V1_RESULTS_COMMIT, SWEEP_RESULTS_COMMIT, WCVAL_V1_RESULTS_COMMIT,
        WCVAL_V2_RESULTS_COMMIT, AXISYMMETRIC_RESULTS_COMMIT, P2_RESULTS_COMMIT,
        ORBIT_V4_RESULTS_COMMIT, ORBIT_V4_PREREGISTRATION_COMMIT,
    ):
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    if _git("rev-list", "--parents", "-n", "1", ORBIT_V4_RESULTS_COMMIT).decode("ascii").split() != [
        ORBIT_V4_RESULTS_COMMIT, ORBIT_V4_PREREGISTRATION_COMMIT
    ]:
        raise ValueError("orbit v4 result commit is not the direct child of its preregistration")
    ledger = SourceLedger()
    sections = {
        "characterization": _characterization(ledger),
        "four_cell_v2": _four_cell_v2(ledger),
        "four_cell_v1": _four_cell_v1(ledger),
        "l1a_sweep": _l1a_sweep(ledger),
        "axisymmetric": _axisymmetric(ledger),
        "p2_divergent_exit": _p2_divergent_exit(ledger),
        "coupling_v4_validation": _wall_cusp_validation(ledger),
        "orbit_wall_loss_v4": _orbit_wall_loss_v4(ledger),
    }
    payload = {
        "schema": SCHEMA,
        "title": "Plasma / magnetic topology results",
        "warning": (
            "Linear-vacuum L1a equivalent-current fields, one qualified P2 FEM reference field and "
            "collisionless prescribed-field test-particle orbits only. No permanent-magnet nonlinear-iron "
            "model beyond P2, no self-consistent plasma, no PIC result, no experimental or hardware validation, "
            "no thrust, efficiency, mirror-formula or device-performance claim."
        ),
        "overview": {
            "concepts": [
                {
                    "term": "Vector null (X / O)",
                    "definition": "A point where (B_r, B_z) = 0. Converged Jacobian with negative determinant and winding index -1 is an X point (a cusp candidate that may bound cells); positive determinant with index +1 is an O point (a closed-flux cell centre).",
                },
                {
                    "term": "Wall cusp (coupling v4)",
                    "definition": "A quadratically interpolated local maximum of wall-normal |B_r| at the dielectric wall with physical prominence and separation, receiving field-line endpoints on every map. Consecutive stable wall-cusp planes bound a cell.",
                },
                {
                    "term": "Separatrix / field line",
                    "definition": "Field lines in the (r, z) half-plane are level sets of the flux function psi = r A_phi. A cell-bounding separatrix passes through an X point; the dashboard draws psi isolines by marching squares directly from embedded psi.",
                },
                {
                    "term": "Mirror ratio",
                    "definition": "B_high / B_low along one connected field line between cusps. Only the L1a sweep v2 records per-cell centreline mirror ratios as field-only screening QoIs; no experiment publishes a mirror-formula confinement claim.",
                },
                {
                    "term": "Design parameterisation",
                    "definition": "Stage count, axial pitch, chamber (channel) radius and first-stage polarity define the characterization family; search v2 adds magnet thickness/fraction, clearances, paddings, stack offset and alternating-strength ratio (13 variables); sweep v2 uses 11 variables including exit expansion and source strength scale.",
                },
            ],
            "fidelity_ladder": [
                {"rung": "L1a", "name": "Linear-vacuum equivalent-current axisymmetric FDM", "status": "accepted screening artifacts and preregistered sweep/search/characterization results"},
                {"rung": "P2", "name": "Independent adaptive P2 FEM reference", "status": "divergent-exit stack NUMERICAL_P2_QUALIFIED; two other designs SCREENING_ONLY"},
                {"rung": "Full orbit", "name": "Collisionless prescribed-field test-particle wall loss (orbit_mc v1.6)", "status": "v4 accepted evidentiary result; v1-v3 failed on code"},
                {"rung": "PIC", "name": "Kinetic particle-in-cell / self-consistent plasma", "status": "foundation only; no results recorded"},
            ],
        },
        **sections,
        "ledger": _claims(sections),
        "sources": ledger.entries(),
        "provenance": {
            "evidence_base_commit": EVIDENCE_BASE_COMMIT,
            "orbit_v4_branch": ORBIT_V4_BRANCH,
            "generator": "modern/visualization/generate_plasma_topology_dashboard.py",
            "generator_sha256": _generator_sha256(),
            "template_sha256": _template_sha256(),
            "evidence_snapshot_time": _generated_at(),
            "time_policy": (
                "deterministic: the snapshot time is the author time of the newest pinned evidence "
                "commit (or SOURCE_DATE_EPOCH); no wall-clock or runtime measurement is embedded"
            ),
            "downsampling": (
                "L1a rasters embed |B| and psi at the archived grid (stride 1) rounded to "
                f"{RASTER_SIGNIFICANT_DIGITS} significant digits; the P2 field is a 144x92 cell-mean raster of "
                "300233 vertices; profiles, counts, positions, probabilities and gate values are exact"
            ),
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the embedded envelope, identities and claim-boundary semantics."""

    required = {
        "schema", "title", "warning", "overview", "characterization", "four_cell_v2", "four_cell_v1",
        "l1a_sweep", "axisymmetric", "p2_divergent_exit", "coupling_v4_validation",
        "orbit_wall_loss_v4", "ledger", "sources", "provenance",
    }
    if set(payload) != required:
        raise ValueError("dashboard payload top-level contract differs")
    if payload["schema"] != SCHEMA:
        raise ValueError("dashboard payload schema differs")
    source_ids = {entry["id"] for entry in payload["sources"]}
    if len(source_ids) != len(payload["sources"]):
        raise ValueError("source ledger contains duplicate ids")
    for entry in payload["sources"]:
        _digest(entry["sha256"], f"source {entry['id']}")
        _commit(entry["commit"], f"source {entry['id']}")
        if not entry["path"] or "\\" in entry["path"]:
            raise ValueError(f"source {entry['id']} path is not portable")

    def check_sources(section: Mapping[str, Any], label: str) -> None:
        listed = section.get("sources")
        if not isinstance(listed, list) or not listed or any(item not in source_ids for item in listed):
            raise ValueError(f"{label} does not reference verified sources")

    for key in (
        "characterization", "four_cell_v2", "four_cell_v1", "l1a_sweep", "axisymmetric",
        "p2_divergent_exit", "coupling_v4_validation", "orbit_wall_loss_v4",
    ):
        check_sources(payload[key], key)
    for representative in payload["characterization"]["representatives"]:
        check_sources(representative, representative["case_id"])
    for representative in payload["four_cell_v2"]["representatives"]:
        check_sources(representative, representative["candidate_id"])
    for representative in payload["l1a_sweep"]["representatives"]:
        check_sources(representative, representative["case_id"])
    for design in payload["axisymmetric"]["designs"]:
        check_sources(design, design["id"])
    for campaign in payload["orbit_wall_loss_v4"]["campaigns"]:
        check_sources(campaign, campaign["case_id"])
    for claim in payload["ledger"]:
        if claim["status"] != "foundation_only_no_results" and (
            not claim["sources"] or any(item not in source_ids for item in claim["sources"])
        ):
            raise ValueError(f"ledger claim {claim['claim'][:40]!r} is not traceable")

    char = payload["characterization"]
    if (
        char["summary"]["evaluated_count"] != 56
        or char["summary"]["stable_eligible_cusp_count"] != 0
        or char["summary"]["stable_eligible_cell_count"] != 0
        or len(char["cases"]) != 56
        or len(char["representatives"]) != 7
    ):
        raise ValueError("characterization embedded counts differ from recorded evidence")
    fc2 = payload["four_cell_v2"]
    if (
        fc2["summary"]["evaluated_count"] != 128 or fc2["summary"]["stable_count"] != 0
        or len(fc2["candidates"]) != 128 or len(fc2["representatives"]) != 2
        or any(candidate["stable"] for candidate in fc2["candidates"])
    ):
        raise ValueError("four-cell v2 embedded counts differ from the recorded null result")
    fc1 = payload["four_cell_v1"]
    if (
        tuple(fc1["compatible_case_ids"]) != EXPECTED_FOUR_CELL_V1_COMPATIBLE
        or fc1["protocol_status"]["valid_for_physical_mirror_claims"]
        or fc1["status"] != "superseded_screening_only"
    ):
        raise ValueError("four-cell v1 embedded semantics differ from archived evidence")
    sweep = payload["l1a_sweep"]
    if (
        sweep["summary"]["terminal_status"] != "ACCEPTED" or len(sweep["cases"]) != 96
        or len(sweep["gates"]) != 7 or not all(gate["passed"] for gate in sweep["gates"])
        or len(sweep["representatives"]) != 4
    ):
        raise ValueError("sweep embedded counts or gates differ from accepted evidence")
    if len(payload["axisymmetric"]["designs"]) != 3:
        raise ValueError("axisymmetric embedded designs differ")
    p2 = payload["p2_divergent_exit"]
    if (
        p2["status"] != "NUMERICAL_P2_QUALIFIED"
        or p2["identity"]["manifest_file_sha256"] != P2_MANIFEST_FILE_SHA256
        or p2["identity"]["artifact_file_sha256"] != P2_RESULT_FILE_SHA256
        or p2["wall_normal_maxima"]["authority"] != "dashboard_derived_display_diagnostic_not_accepted_cusp_evidence"
    ):
        raise ValueError("P2 embedded identity or authority semantics differ")
    wcval = payload["coupling_v4_validation"]
    if (
        wcval["criterion"]["promotion_status"] != "not_promoted"
        or len(wcval["runs"]) != 2
        or any(run["summary"]["criterion_numerically_promoted"] for run in wcval["runs"])
    ):
        raise ValueError("coupling v4 validation embedded semantics differ")
    orbit = payload["orbit_wall_loss_v4"]
    if (
        orbit["status"] != "accepted_evidentiary_result"
        or orbit["commits"]["results"] != ORBIT_V4_RESULTS_COMMIT
        or len(orbit["campaigns"]) != 9
        or any(len(campaign["strata"]) != 32 for campaign in orbit["campaigns"])
        or orbit["gates"]["passed"] is not True
        or orbit["limitations"]["forbid_mirror_formula_publication"] is not True
        or orbit["limitations"]["forbid_pic_or_self_consistent_claim"] is not True
    ):
        raise ValueError("orbit v4 embedded semantics differ from the accepted result")
    statuses = [claim["status"] for claim in payload["ledger"]]
    if statuses.count("accepted_numerical_evidence") != 4 or "foundation_only_no_results" not in statuses:
        raise ValueError("validation ledger tallies differ")
    provenance = payload["provenance"]
    if provenance["evidence_base_commit"] != EVIDENCE_BASE_COMMIT:
        raise ValueError("embedded evidence base commit differs")
    _digest(provenance["generator_sha256"], "generator hash")
    _digest(provenance["template_sha256"], "template hash")


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("dashboard template must contain exactly one data placeholder")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).replace("</", "<\\/")
    html = template.replace("__DATA__", encoded)
    if len(html.encode("utf-8")) > SIZE_CAP_BYTES:
        raise ValueError("rendered dashboard exceeds the 15 MiB size cap")
    return html


def generate(output: Path = DEFAULT_OUTPUT) -> str:
    html = render_html(build_payload())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8", newline="\n")
    return sha256(html.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(f"{generate(args.output)}  {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
