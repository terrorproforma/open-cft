"""Generate the standalone CFT full-orbit wall-loss v4 results dashboard.

Every number shown by the dashboard is read from the immutable, hash-bound
results bundle of ``modern/experiments/cft_orbit_wall_loss_v4`` (or from the
committed protocol/README/DEVLOG of that experiment for verbatim strings).
Nothing is typed by hand.  The generator verifies the whole bundle against
``results/manifest.json`` before rendering, cross-checks every artifact
against every other artifact that repeats the same quantity, and refuses to
render on any inconsistency.

Known bundle defect (tolerated exactly, nothing else): the nine
``artifacts/orbits/<case>.json.sha256`` text sidecars were hashed with CRLF
line endings when the bundle was sealed but are stored LF after the
repository-wide ``eol=lf`` pin.  For those nine paths only, the generator
accepts ``sha256(bytes.replace(b"\\n", b"\\r\\n")) == recorded``.  Any other
byte mismatch is an error.

The generator omits wall-clock timestamps of its own and machine paths so
identical inputs produce identical bytes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
from hashlib import sha256
import json
from math import hypot, isfinite
from pathlib import Path, PurePosixPath
from statistics import median
import sys
from typing import Any, Mapping, Sequence

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cft_revival.orbit_mc import PsiBicubicField  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERIMENT = MODERN / "experiments" / "cft_orbit_wall_loss_v4"
RESULTS = EXPERIMENT / "results"
TEMPLATE_PATH = HERE / "wall-loss-v4-results.template.html"
DEFAULT_OUTPUT = HERE / "wall-loss-v4-results.html"

SCHEMA = "cft-revival.wall-loss-v4-results-dashboard/1.0.0"
MAX_HTML_BYTES = 1_200_000

# Committed identities of the accepted campaign (feat/sota-foundation).
RESULTS_COMMIT_SHA = "6922a3cf97d261735266aa1a5a0c0c9683e021ca"
PREREGISTRATION_COMMIT_SHA = "757e365f9f667620c7610663574294c3b71e1f51"
EXPECTED_MANIFEST_SHA256 = (
    "ef3863b0a3ba0a1d74187b05daf81d5d94d3838a7e33ecf82c485dccd162929f"
)
EXPECTED_TERMINAL_SHA256 = (
    "6d23c4af9d1645aa8983054b9d8bc3b40d773ae088f7b1987d77cf77736921f1"
)
EXPECTED_LOCK_SHA256 = (
    "6232324add3ecfa3b4b400026dc15825e5df89ee9388e1ee0f3cca961f493952"
)

CASES = (
    "primary-N", "primary-2N", "primary-4N",
    "refined-N", "refined-2N", "refined-4N",
    "enlarged-N", "enlarged-2N", "enlarged-4N",
)
MAP_ROLES = ("primary", "refined", "enlarged")
TIMESTEP_POLICIES = ("N", "2N", "4N")
CELLS = ("v4-cell-1", "v4-cell-2", "v4-cell-3", "v4-cell-4")
TERMINATION_CODES = ("wall_hit", "domain_escape")
ESTIMANDS = ("wall_hit", "escaped", "reflected", "incomplete")
ESTIMAND_LABELS = {
    "wall_hit": "wall hit (dielectric)",
    "escaped": "domain escape",
    "reflected": "reflected",
    "incomplete": "timeout / incomplete",
}
GATE_CHECKS = (
    "campaign_preflight", "cross_map_probability_convergence", "earliest_event",
    "energy", "field_adapter", "field_map_convergence",
    "final_velocity_equals_event_velocity", "independent_repeats", "manufactured",
    "material_quarantine", "relativistic_phase", "runtime_rotation",
    "timestep_probability_convergence", "wall_endpoint",
    "zero_incomplete_or_numerical_failures",
)
TOLERATED_CRLF_SIDECARS = frozenset(
    f"artifacts/orbits/{case}.json.sha256" for case in CASES
)
CLASSIFICATION = "collisionless_prescribed_field_test_particle_wall_loss_not_pic"
COUPLING_STATUS = "export_only_pending_consumer_integration"
PLASMA_MATERIAL_ID = "qualified-divergent-exit-homogeneous-plasma"

# Verbatim lineage statements that live only in the experiment DEVLOG; the
# generator asserts each one is present in the committed file and records
# that file's hash.
DEVLOG_QUOTES = {
    "second_latent_v3_bug": (
        "the assessment raised `zip() argument 2 is shorter than\n"
        "  argument 1` inside `_convergence` — a latent v3 bug (`zip(ordered,\n"
        "  ordered[1:], strict=True)`) that v3 never reached. Fixed to\n"
        "  `zip(ordered[:-1], ordered[1:], strict=True)`; added a regression test."
    ),
    "v1_6_shakedown_energy": (
        "**max relative energy error 0.0, 0/576 orbits\n"
        "  with non-zero energy error, 576/576 final-velocity == event-velocity**"
    ),
}


# --------------------------------------------------------------------------- #
# Strict loading helpers
# --------------------------------------------------------------------------- #
def _load_json_bytes(raw: bytes, label: str) -> Any:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {label}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is not readable: {path.name}") from exc
    value = _load_json_bytes(raw, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _lf_digest(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF (checkout-neutral)."""

    return _digest(path.read_bytes().replace(b"\r\n", b"\n"))


def _canonical_hash(value: Any) -> str:
    return _digest(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _hex64(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _closed(value: Any, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys do not match the closed schema")
    return value


def _safe_relative(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or ":" in raw or "\\" in raw:
        raise ValueError(f"{label} is not a relative bundle path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"{label} escapes the bundle")
    path = root.joinpath(*pure.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the bundle")
    return path


def _sig(value: float, digits: int) -> float:
    if value == 0.0:
        return 0.0
    return float(f"{value:.{digits}g}")


def _wilson_95(successes: int, trials: int) -> dict[str, Any]:
    """Wilson score interval (z = 1.959963984540054), as used by orbit_mc."""

    z = 1.959963984540054
    if trials <= 0:
        raise ValueError("Wilson interval requires trials > 0")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half = z * ((p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) ** 0.5)
    half /= denominator
    return {
        "method": "wilson-95",
        "successes": successes,
        "trials": trials,
        "probability": p,
        "lower": max(0.0, centre - half),
        "upper": min(1.0, centre + half),
    }


# --------------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------------- #
class Bundle:
    """The verified results bundle: every file hashed against the manifest."""

    def __init__(self, results_root: Path) -> None:
        self.root = results_root.resolve()
        manifest_path = self.root / "manifest.json"
        manifest_raw = manifest_path.read_bytes()
        self.manifest_sha256 = _digest(manifest_raw)
        self.manifest = _load_json_bytes(manifest_raw, "results manifest")
        if not isinstance(self.manifest, dict):
            raise ValueError("results manifest must be a JSON object")
        if self.manifest.get("schema_version") != "cft-revival.experiment-manifest/1.0.0":
            raise ValueError("results manifest schema is unsupported")
        if self.manifest.get("state") != "accepted_result":
            raise ValueError("results manifest state is not accepted_result")
        if self.manifest.get("experiment_id") != "cft-orbit-wall-loss-v4":
            raise ValueError("results manifest experiment identity differs")
        if self.manifest.get("manifest_is_sole_completion_marker") is not True:
            raise ValueError("results manifest completion marker contract differs")
        entries = self.manifest.get("artifacts")
        if not isinstance(entries, list) or len(entries) != self.manifest.get(
            "artifact_count"
        ):
            raise ValueError("results manifest artifact count differs")
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        self.tolerated: list[dict[str, Any]] = []
        files: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"results manifest artifacts[{index}] is not an object")
            kind = entry.get("type")
            relative = _text(entry.get("path"), f"artifacts[{index}].path")
            if kind == "directory":
                if not (self.root / relative).is_dir():
                    raise ValueError(f"required directory missing: {relative}")
                continue
            if kind != "file":
                raise ValueError(f"unsupported manifest artifact type {kind!r}")
            if relative in files:
                raise ValueError(f"duplicate manifest path {relative}")
            files[relative] = entry
        for relative, entry in files.items():
            path = _safe_relative(self.root, relative, "manifest artifact path")
            raw = path.read_bytes()
            expected = _hex64(entry.get("byte_sha256"), f"{relative} byte_sha256")
            expected_bytes = _integer(entry.get("bytes"), f"{relative} bytes")
            actual = _digest(raw)
            if actual != expected:
                crlf = raw.replace(b"\n", b"\r\n")
                if (
                    relative in TOLERATED_CRLF_SIDECARS
                    and b"\r" not in raw
                    and _digest(crlf) == expected
                    and len(crlf) == expected_bytes
                ):
                    self.tolerated.append(
                        {
                            "path": relative,
                            "recorded_sha256": expected,
                            "recorded_bytes": expected_bytes,
                            "checkout_sha256": actual,
                            "checkout_bytes": len(raw),
                            "rule": "sha256(bytes.replace(LF, CRLF)) == recorded",
                        }
                    )
                else:
                    raise ValueError(f"bundle file SHA-256 mismatch: {relative}")
            elif len(raw) != expected_bytes:
                raise ValueError(f"bundle file size mismatch: {relative}")
            self.hashes[relative] = actual
            self.sizes[relative] = len(raw)
            if "canonical_json_sha256" in entry and entry["canonical_json_sha256"] is not None:
                if _canonical_hash(_load_json_bytes(raw, relative)) != entry[
                    "canonical_json_sha256"
                ]:
                    raise ValueError(f"bundle canonical JSON hash mismatch: {relative}")
            sidecar = entry.get("sidecar")
            if sidecar is not None:
                sidecar_entry = files.get(sidecar)
                if sidecar_entry is None or sidecar_entry.get("contract") != (
                    "hash-sidecar-metadata"
                ):
                    raise ValueError(f"missing hash sidecar for {relative}")
                sidecar_value = _load_object(
                    _safe_relative(self.root, sidecar, "sidecar path"), sidecar
                )
                if (
                    sidecar_value.get("artifact") != relative
                    or sidecar_value.get("byte_sha256") != expected
                    or sidecar_value.get("bytes") != expected_bytes
                ):
                    raise ValueError(f"hash sidecar disagrees with manifest: {relative}")
        self.tolerated.sort(key=lambda item: item["path"])
        if {item["path"] for item in self.tolerated} - TOLERATED_CRLF_SIDECARS:
            raise ValueError("unexpected tolerated sidecar")
        if self.hashes.get("terminal.json") != self.manifest.get("terminal_byte_sha256"):
            raise ValueError("terminal.json hash differs from manifest binding")
        if self.hashes.get("execution-lock.json") != self.manifest.get("lock_byte_sha256"):
            raise ValueError("execution-lock.json hash differs from manifest binding")

    def read(self, relative: str) -> bytes:
        if relative not in self.hashes:
            raise ValueError(f"{relative} is not a manifest-bound artifact")
        return (self.root / relative).read_bytes()

    def load(self, relative: str) -> dict[str, Any]:
        value = _load_json_bytes(self.read(relative), relative)
        if not isinstance(value, dict):
            raise ValueError(f"{relative} must be a JSON object")
        return value

    def load_gzip(self, relative: str) -> tuple[dict[str, Any], bytes]:
        try:
            raw = gzip.decompress(self.read(relative))
        except (OSError, EOFError) as exc:
            raise ValueError(f"{relative} is not a valid gzip member") from exc
        value = _load_json_bytes(raw, relative)
        if not isinstance(value, dict):
            raise ValueError(f"{relative} must be a JSON object")
        return value, raw


# --------------------------------------------------------------------------- #
# Artifact cross-checks and projections
# --------------------------------------------------------------------------- #
def _interval(value: Any, label: str, trials: int) -> dict[str, Any]:
    item = _closed(
        value, label, {"lower", "method", "probability", "successes", "trials", "upper"}
    )
    if item["method"] != "wilson-95":
        raise ValueError(f"{label} interval method is not wilson-95")
    successes = _integer(item["successes"], f"{label}.successes")
    if _integer(item["trials"], f"{label}.trials", 1) != trials:
        raise ValueError(f"{label} trials differ from the stratum/case size")
    probability = _finite(item["probability"], f"{label}.probability")
    lower = _finite(item["lower"], f"{label}.lower")
    upper = _finite(item["upper"], f"{label}.upper")
    if not (0.0 <= lower <= probability <= upper <= 1.0):
        raise ValueError(f"{label} interval ordering is invalid")
    if probability != successes / trials:
        raise ValueError(f"{label} probability is not successes/trials")
    recomputed = _wilson_95(successes, trials)
    if abs(recomputed["lower"] - lower) > 1e-12 or abs(recomputed["upper"] - upper) > 1e-12:
        raise ValueError(f"{label} Wilson bounds do not reproduce")
    return {
        "successes": successes,
        "trials": trials,
        "probability": probability,
        "lower": lower,
        "upper": upper,
    }


def _termination_counts(value: Any, label: str, total: int) -> dict[str, int]:
    counts = _closed(
        value,
        label,
        {
            "domain_escape", "extreme_relativity", "field_failure",
            "initial_state_invalid", "nonfinite_state", "path_timeout", "reflected",
            "step_limit", "time_timeout", "wall_hit",
        },
    )
    parsed = {key: _integer(item, f"{label}.{key}") for key, item in counts.items()}
    if sum(parsed.values()) != total:
        raise ValueError(f"{label} termination counts do not sum to {total}")
    return dict(sorted(parsed.items()))


def _case_block(value: Any, label: str) -> dict[str, Any]:
    block = _closed(
        value,
        label,
        {"escaped", "incomplete", "reflected", "termination_counts", "trial_count",
         "wall_hit"},
    )
    trials = _integer(block["trial_count"], f"{label}.trial_count", 1)
    counts = _termination_counts(block["termination_counts"], f"{label}.termination", trials)
    intervals = {
        key: _interval(block[key], f"{label}.{key}", trials) for key in ESTIMANDS
    }
    if intervals["wall_hit"]["successes"] != counts["wall_hit"]:
        raise ValueError(f"{label} wall_hit successes differ from termination counts")
    if intervals["escaped"]["successes"] != counts["domain_escape"]:
        raise ValueError(f"{label} escape successes differ from termination counts")
    if intervals["reflected"]["successes"] != counts["reflected"]:
        raise ValueError(f"{label} reflected successes differ from termination counts")
    incomplete = (
        counts["path_timeout"] + counts["time_timeout"] + counts["step_limit"]
        + counts["field_failure"] + counts["nonfinite_state"] + counts["extreme_relativity"]
        + counts["initial_state_invalid"]
    )
    if intervals["incomplete"]["successes"] != incomplete:
        raise ValueError(f"{label} incomplete successes differ from termination counts")
    return {"trials": trials, "termination_counts": counts, "estimands": intervals}


def _stratum_key(cell_id: str, energy: float, pitch: float, direction: int) -> str:
    return f"{cell_id}|{energy:g}|{pitch:g}|{direction:+d}"


def _strata(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 32:
        raise ValueError(f"{label} must contain 32 strata")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_order = [
        _stratum_key(cell, energy, pitch, direction)
        for cell in CELLS
        for energy in (5.0, 25.0)
        for pitch in (20.0, 70.0)
        for direction in (-1, 1)
    ]
    for index, item in enumerate(value):
        row = _closed(
            item,
            f"{label}[{index}]",
            {
                "cell_id", "domain_escape", "kinetic_energy_ev", "parallel_direction",
                "physical_position_repeat_count", "pitch_angle_deg", "reflected",
                "termination_counts", "timeout", "trials", "wall_hit",
            },
        )
        cell = _text(row["cell_id"], "stratum cell_id")
        if cell not in CELLS:
            raise ValueError(f"{label}[{index}] cell is not a v4 cell")
        energy = _finite(row["kinetic_energy_ev"], "stratum energy")
        pitch = _finite(row["pitch_angle_deg"], "stratum pitch")
        direction = row["parallel_direction"]
        if direction not in (-1, 1) or isinstance(direction, bool):
            raise ValueError(f"{label}[{index}] direction is invalid")
        trials = _integer(row["trials"], "stratum trials", 1)
        if trials != 16 or _integer(row["physical_position_repeat_count"], "repeats") != 2:
            raise ValueError(f"{label}[{index}] design size differs from the protocol")
        key = _stratum_key(cell, energy, pitch, direction)
        if key in seen:
            raise ValueError(f"{label} contains a duplicate stratum")
        seen.add(key)
        if key != expected_order[index]:
            raise ValueError(f"{label} stratum order differs from the design order")
        counts = _termination_counts(row["termination_counts"], f"{label}[{index}]", trials)
        intervals = {
            name: _interval(row[name], f"{label}[{index}].{name}", trials)
            for name in ("wall_hit", "domain_escape", "reflected", "timeout")
        }
        if (
            intervals["wall_hit"]["successes"] != counts["wall_hit"]
            or intervals["domain_escape"]["successes"] != counts["domain_escape"]
            or intervals["reflected"]["successes"] != counts["reflected"]
        ):
            raise ValueError(f"{label}[{index}] intervals disagree with counts")
        rows.append(
            {
                "index": index,
                "key": key,
                "cell_id": cell,
                "cell": CELLS.index(cell) + 1,
                "kinetic_energy_ev": energy,
                "pitch_angle_deg": pitch,
                "parallel_direction": direction,
                "trials": trials,
                "wall_hit": counts["wall_hit"],
                "domain_escape": counts["domain_escape"],
                "reflected": counts["reflected"],
                "timeout": intervals["timeout"]["successes"],
                "wall_hit_lower": intervals["wall_hit"]["lower"],
                "wall_hit_upper": intervals["wall_hit"]["upper"],
                "domain_escape_lower": intervals["domain_escape"]["lower"],
                "domain_escape_upper": intervals["domain_escape"]["upper"],
            }
        )
    return rows


def _diagnostics(value: Any, label: str, trials: int) -> dict[str, Any]:
    block = _closed(
        value,
        label,
        {
            "event_resolution_counts", "final_velocity_equals_event_velocity_count",
            "magnetic_moment_variation_diagnostic", "maximum_relative_energy_error",
            "orbits_with_nonzero_energy_error", "runtime_max_b_t", "steps",
            "termination_counts", "tolerance_close_conditions",
            "tolerance_close_event_count", "wall_endpoint_error_max_m",
        },
    )
    mu = _closed(
        block["magnetic_moment_variation_diagnostic"],
        f"{label}.mu",
        {
            "binding", "count_above_0p1", "count_above_0p5", "informational_gate", "max",
            "median", "min", "orbit_count_with_mu", "orbit_count_without_mu", "role",
        },
    )
    if mu["binding"] is not False or mu["informational_gate"] is not False:
        raise ValueError(f"{label} mu diagnostic is presented as a gate")
    if mu["role"] != "diagnostic_only":
        raise ValueError(f"{label} mu diagnostic role differs")
    if _integer(mu["orbit_count_with_mu"], "mu count") != trials or mu[
        "orbit_count_without_mu"
    ] != 0:
        raise ValueError(f"{label} mu orbit counts differ from the case size")
    resolution = _closed(
        block["event_resolution_counts"],
        f"{label}.event_resolution_counts",
        {"interpolated", "tolerance_close_fraction_zero"},
    )
    interpolated = _integer(resolution["interpolated"], "interpolated")
    tolerance_close = _integer(resolution["tolerance_close_fraction_zero"], "tolerance close")
    if interpolated + tolerance_close != trials:
        raise ValueError(f"{label} event resolution counts do not sum to the case size")
    if _integer(block["tolerance_close_event_count"], "tolerance-close count") != tolerance_close:
        raise ValueError(f"{label} tolerance-close counts disagree")
    conditions = {
        key: _integer(item, f"{label}.tolerance_close_conditions.{key}")
        for key, item in _closed(
            block["tolerance_close_conditions"],
            f"{label}.tolerance_close_conditions",
            {
                "tolerance_close_domain_radial", "tolerance_close_domain_z_max",
                "tolerance_close_domain_z_min", "tolerance_close_wall_radial",
            },
        ).items()
    }
    if sum(conditions.values()) != tolerance_close:
        raise ValueError(f"{label} tolerance-close conditions do not sum")
    steps = _closed(block["steps"], f"{label}.steps", {"max", "median", "min", "total"})
    if _finite(block["maximum_relative_energy_error"], "energy error") != 0.0:
        raise ValueError(f"{label} maximum relative energy error is not exactly zero")
    if _integer(block["orbits_with_nonzero_energy_error"], "nonzero energy") != 0:
        raise ValueError(f"{label} reports orbits with non-zero energy error")
    if _integer(block["final_velocity_equals_event_velocity_count"], "velocity identity") != trials:
        raise ValueError(f"{label} final-velocity identity count differs from the case size")
    return {
        "mu": {
            "min": _finite(mu["min"], "mu min"),
            "median": _finite(mu["median"], "mu median"),
            "max": _finite(mu["max"], "mu max"),
            "count_above_0p1": _integer(mu["count_above_0p1"], "mu >0.1"),
            "count_above_0p5": _integer(mu["count_above_0p5"], "mu >0.5"),
            "orbit_count": trials,
            "role": "diagnostic_only",
            "binding": False,
        },
        "event_resolution": {
            "interpolated": interpolated,
            "tolerance_close_fraction_zero": tolerance_close,
        },
        "tolerance_close_conditions": dict(sorted(conditions.items())),
        "steps": {
            "min": _integer(steps["min"], "steps min", 1),
            "median": _finite(steps["median"], "steps median"),
            "max": _integer(steps["max"], "steps max", 1),
            "total": _integer(steps["total"], "steps total", 1),
        },
        "maximum_relative_energy_error": 0.0,
        "orbits_with_nonzero_energy_error": 0,
        "final_velocity_equals_event_velocity_count": trials,
        "runtime_max_b_t": _finite(block["runtime_max_b_t"], "runtime max |B|"),
        "wall_endpoint_error_max_m": _finite(
            block["wall_endpoint_error_max_m"], "wall endpoint error"
        ),
        "termination_counts": _termination_counts(
            block["termination_counts"], f"{label}.termination_counts", trials
        ),
    }


def _orbit_columns(
    artifact: Mapping[str, Any],
    case_index: int,
    strata: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    launches = artifact["launches"]
    results = artifact["results"]
    if not isinstance(launches, list) or not isinstance(results, list):
        raise ValueError(f"{label} launches/results must be arrays")
    if len(launches) != 512 or len(results) != 512:
        raise ValueError(f"{label} must contain 512 launches and results")
    key_to_index = {row["key"]: row["index"] for row in strata}
    columns: dict[str, list[Any]] = {
        "case": [], "stratum": [], "termination": [], "condition": [],
        "final_r_mm": [], "final_z_mm": [], "steps": [], "mu_variation": [],
        "path_mm": [], "max_b_t": [],
    }
    conditions: list[str] = []
    stratum_counts: dict[int, dict[str, int]] = {}
    wall = geometry["wall"]
    domain = geometry["domain"]
    tolerance = 1.0e-6
    for launch, result in zip(launches, results, strict=True):
        if launch.get("launch_id") != result.get("launch_id"):
            raise ValueError(f"{label} launch/result order differs")
        cell = str(launch["flux_surface_id"]).rsplit("-r", 1)[0]
        pitch_deg = round(float(launch["pitch_angle_rad"]) * 180.0 / 3.141592653589793, 6)
        key = _stratum_key(
            cell, float(launch["kinetic_energy_ev"]), pitch_deg, int(launch["parallel_direction"])
        )
        if key not in key_to_index:
            raise ValueError(f"{label} launch {launch['launch_id']} has no stratum")
        stratum = key_to_index[key]
        termination = result["termination"]
        if termination not in TERMINATION_CODES:
            raise ValueError(f"{label} contains a termination outside the accepted set")
        witness = result["event_witness"]
        if witness.get("kind") != termination:
            raise ValueError(f"{label} witness kind differs from termination")
        condition = _text(witness.get("condition"), "witness condition")
        if condition not in conditions:
            conditions.append(condition)
        position = result["final_position_m"]
        if witness.get("event_position_m") != position:
            raise ValueError(f"{label} final position differs from the event witness")
        if result.get("final_velocity_m_per_s") != witness.get("event_velocity_m_per_s"):
            raise ValueError(f"{label} final velocity differs from the event velocity")
        if _finite(result["maximum_relative_energy_error"], "energy error") != 0.0:
            raise ValueError(f"{label} contains a non-zero energy error")
        x, y, z = (_finite(item, "final position") for item in position)
        radius = hypot(x, y)
        if termination == "wall_hit":
            if abs(radius - wall["radius_m"]) > tolerance or not (
                wall["z_min_m"] - tolerance <= z <= wall["z_max_m"] + tolerance
            ):
                raise ValueError(f"{label} wall hit lies off the dielectric")
        else:
            on_axial = (
                abs(z - domain["z_min_m"]) <= tolerance
                or abs(z - domain["z_max_m"]) <= tolerance
            )
            on_radial = abs(radius - domain["radius_m"]) <= tolerance
            if not (on_axial or on_radial):
                raise ValueError(f"{label} domain escape lies inside the domain")
            if on_radial and not on_axial and z < wall["z_max_m"] - tolerance:
                raise ValueError(f"{label} radial escape inside the dielectric span")
        counts = stratum_counts.setdefault(stratum, {"wall_hit": 0, "domain_escape": 0})
        counts[termination] += 1
        columns["case"].append(case_index)
        columns["stratum"].append(stratum)
        columns["termination"].append(TERMINATION_CODES.index(termination))
        columns["condition"].append(conditions.index(condition))
        columns["final_r_mm"].append(round(radius * 1e3, 5))
        columns["final_z_mm"].append(round(z * 1e3, 5))
        columns["steps"].append(_integer(result["steps"], "steps", 1))
        columns["mu_variation"].append(
            _sig(_finite(result["maximum_instantaneous_mu_relative_variation"], "mu"), 5)
        )
        columns["path_mm"].append(round(_finite(result["path_length_m"], "path") * 1e3, 4))
        columns["max_b_t"].append(_sig(_finite(result["maximum_b_t"], "max |B|"), 6))
    for row in strata:
        counts = stratum_counts.get(row["index"], {"wall_hit": 0, "domain_escape": 0})
        if counts["wall_hit"] != row["wall_hit"] or counts["domain_escape"] != row["domain_escape"]:
            raise ValueError(f"{label} per-orbit outcomes disagree with the stratum summary")
    return columns, {"conditions": conditions}


def _field_projection(
    bundle: Bundle, role: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    relative = f"artifacts/fields/{role}.json"
    grid = bundle.load(relative)
    if set(grid) != {"material_id", "psi_wb", "r_m", "source_identity_sha256", "z_m"}:
        raise ValueError(f"{relative} keys differ from the adapter contract")
    if grid["source_identity_sha256"] != evidence["source_identity_sha256"]:
        raise ValueError(f"{relative} source identity differs from field evidence")
    r_m = [_finite(item, "r_m") for item in grid["r_m"]]
    z_m = [_finite(item, "z_m") for item in grid["z_m"]]
    regular = evidence["regular_grid"]
    if (
        len(r_m) != regular["radial_samples"]
        or len(z_m) != regular["axial_samples"]
        or r_m[0] != regular["r_min_m"]
        or r_m[-1] != regular["r_max_m"]
        or z_m[0] != regular["z_min_m"]
        or z_m[-1] != regular["z_max_m"]
    ):
        raise ValueError(f"{relative} grid differs from field evidence")
    field = PsiBicubicField(
        r_m,
        z_m,
        grid["psi_wb"],
        material_id=grid["material_id"],
        plasma_material_id=PLASMA_MATERIAL_ID,
        minimum_certificate_tightness_ratio=evidence["certificate"]["minimum_ratio"],
        source_identity_sha256=grid["source_identity_sha256"],
    )
    if field.certificate_tightness.to_dict() != evidence["certificate"]:
        raise ValueError(f"{relative} certificate does not reproduce the field evidence")
    if field.material_map_sha256 != evidence["material_map_sha256"]:
        raise ValueError(f"{relative} material map hash differs from field evidence")
    magnitude: list[list[float]] = []
    grid_max = 0.0
    for radius in r_m:
        row: list[float] = []
        for axial in z_m:
            br, bz = field.field_cylindrical(float(radius), float(axial))
            value = hypot(br, bz)
            grid_max = max(grid_max, value)
            row.append(_sig(value, 5))
        magnitude.append(row)
    if grid_max > field.certified_max_b_t:
        raise ValueError(f"{relative} grid |B| exceeds the certified bound")
    return {
        "role": role,
        "artifact": relative,
        "artifact_sha256": bundle.hashes[relative],
        "evidence_artifact": f"artifacts/field-evidence/{role}.json",
        "evidence_sha256": bundle.hashes[f"artifacts/field-evidence/{role}.json"],
        "r_m": r_m,
        "z_m": z_m,
        "psi_wb": [[_sig(_finite(v, "psi"), 6) for v in row] for row in grid["psi_wb"]],
        "b_magnitude_t": magnitude,
        "b_magnitude_layout": "radial-major; values[r_index][z_index]; |B| from the "
        "orbit_mc PsiBicubicField reconstruction of the sealed ψ grid, 5 significant digits; "
        "ψ rounded to 6 significant digits for display",
        "grid_max_b_t": grid_max,
        "certified_max_b_t": field.certified_max_b_t,
        "certificate": dict(evidence["certificate"]),
        "field_error_report": dict(evidence["field_error_report"]),
        "checkpoint_file_sha256": evidence["checkpoint_file_sha256"],
        "checkpoint_payload_sha256": evidence["checkpoint_payload_sha256"],
        "checkpoint_sidecar_sha256": evidence["checkpoint_sidecar_sha256"],
        "mesh_sha256": evidence["mesh_sha256"],
        "run_sha256": evidence["run_sha256"],
        "material_map_sha256": evidence["material_map_sha256"],
        "source_identity_sha256": grid["source_identity_sha256"],
        "passed": evidence["passed"],
    }


def _timestamp(value: Any, label: str) -> str:
    stamp = _closed(value, label, {"__cft_type__", "value"})
    if stamp["__cft_type__"] != "aware-utc-datetime":
        raise ValueError(f"{label} is not an aware UTC datetime")
    return _text(stamp["value"], label)


def _seconds_between(start: str, end: str) -> float:
    first = datetime.fromisoformat(start.replace("Z", "+00:00"))
    last = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (last - first).total_seconds()


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def build_payload(
    results_root: Path = RESULTS, experiment_root: Path = EXPERIMENT
) -> dict[str, Any]:
    """Verify the bundle and project it into the dashboard payload."""

    bundle = Bundle(results_root)
    terminal = bundle.load("terminal.json")
    if terminal.get("state") != "accepted_result" or terminal.get("primary_error") is not None:
        raise ValueError("terminal record is not an accepted result")
    if terminal.get("secondary_errors") != []:
        raise ValueError("terminal record carries secondary errors")
    counts = terminal["counts"]
    if counts.get("attempt_count") != 1:
        raise ValueError("terminal record is not a single-attempt execution")
    campaign = bundle.load("artifacts/campaign-result.json")
    if terminal.get("payload") != campaign:
        raise ValueError("terminal payload differs from campaign-result.json")
    gates = bundle.load("artifacts/gates.json")
    if campaign.get("gates") != gates:
        raise ValueError("campaign-result gates differ from gates.json")
    if (
        campaign.get("status") != "accepted"
        or campaign.get("evidentiary") is not True
        or campaign.get("plan_kind") != "evidentiary"
        or campaign.get("gates_binding") is not True
        or campaign.get("classification") != CLASSIFICATION
        or campaign.get("coupling") != COUPLING_STATUS
        or campaign.get("campaign_count") != 9
        or campaign.get("launches_per_case") != 512
        or campaign.get("orbit_count") != 4608
        or campaign.get("total_case_launch_count") != 4608
    ):
        raise ValueError("campaign-result headline identity differs from the accepted campaign")
    validators = _closed(campaign["validators"], "validators", {"failed", "passed"})
    if validators["failed"] != 0:
        raise ValueError("campaign-result records validator failures")
    checks = _closed(gates["checks"], "gate checks", set(GATE_CHECKS))
    if any(value is not True for value in checks.values()) or gates.get("passed") is not True:
        raise ValueError("a binding gate did not pass")
    if gates.get("binding") is not True:
        raise ValueError("gates are not binding")
    if gates.get("maximum_relative_energy_error") != 0.0 or gates.get(
        "orbits_exceeding_energy_gate"
    ) != 0:
        raise ValueError("energy gate evidence differs")
    mu_pooled = _closed(
        gates["diagnostics_not_gates"]["magnetic_moment_variation"],
        "pooled mu diagnostic",
        {
            "binding", "count_above_0p1", "count_above_0p5", "informational_gate", "max",
            "median", "min", "orbit_count_with_mu", "orbit_count_without_mu", "role",
        },
    )
    if mu_pooled["binding"] is not False or mu_pooled["role"] != "diagnostic_only":
        raise ValueError("pooled mu diagnostic is presented as a gate")

    protocol_artifact = bundle.load("artifacts/protocol.json")
    protocol_path = experiment_root / "protocol.json"
    protocol = _load_object(protocol_path, "committed protocol")
    if {k: v for k, v in protocol.items() if k != "status"} != {
        k: v for k, v in protocol_artifact.items() if k != "status"
    }:
        raise ValueError("committed protocol differs from the sealed protocol artifact")
    if protocol["classification"] != CLASSIFICATION:
        raise ValueError("protocol classification differs")
    orbit_cfg = protocol["orbit"]
    geometry = {
        "wall": {
            "radius_m": _finite(orbit_cfg["wall"]["radius_m"], "wall radius"),
            "z_min_m": _finite(orbit_cfg["wall"]["z_min_m"], "wall z_min"),
            "z_max_m": _finite(orbit_cfg["wall"]["z_max_m"], "wall z_max"),
            "scope": _text(orbit_cfg["wall"]["scope"], "wall scope"),
        },
        "domain": {
            "radius_m": _finite(orbit_cfg["domain"]["radius_m"], "domain radius"),
            "z_min_m": _finite(orbit_cfg["domain"]["z_min_m"], "domain z_min"),
            "z_max_m": _finite(orbit_cfg["domain"]["z_max_m"], "domain z_max"),
            "scope": _text(orbit_cfg["domain"]["scope"], "domain scope"),
        },
        "max_time_s": _finite(orbit_cfg["max_time_s"], "max time"),
        "max_path_m": _finite(orbit_cfg["max_path_m"], "max path"),
        "max_steps": _integer(orbit_cfg["max_steps"], "max steps", 1),
        "event_tolerance_m": _finite(orbit_cfg["event_tolerance_m"], "event tolerance"),
        "maximum_gamma": _finite(orbit_cfg["maximum_gamma"], "maximum gamma"),
        "timestep_policies": {
            key: _finite(item["max_rotation_rad"], f"policy {key}")
            for key, item in orbit_cfg["timestep_policies"].items()
        },
        "timestep_policy_order": list(TIMESTEP_POLICIES),
        "map_role_order": list(MAP_ROLES),
        "plasma_domain": dict(protocol["field_adapter"]["regular_plasma_domain"]),
    }
    if set(geometry["timestep_policies"]) != set(TIMESTEP_POLICIES):
        raise ValueError("protocol timestep policies differ from the accepted N/2N/4N set")
    rotations = [geometry["timestep_policies"][p] for p in TIMESTEP_POLICIES]
    if any(b >= a for a, b in zip(rotations[:-1], rotations[1:])):
        raise ValueError("timestep policies must halve the rotation bound in N, 2N, 4N order")
    if geometry["wall"]["z_max_m"] >= geometry["domain"]["z_max_m"]:
        raise ValueError("protocol wall span must end before the domain exit")
    launches_cfg = protocol["launches"]
    plan = bundle.load("artifacts/campaign-plan.json")
    if plan["kind"] != "evidentiary" or plan["launches_per_case"] != 512 or plan[
        "strata_per_case"
    ] != 32 or plan["batches_per_case"] != 8 or plan["batch_size"] != 64:
        raise ValueError("campaign plan differs from the preregistered design")
    seeds = launches_cfg["position_seeds"]
    if [
        {"flux_surface_id": item["flux_surface_id"], "position_m": item["position_m"]}
        for item in seeds
    ] != plan["positions"]:
        raise ValueError("campaign plan positions differ from the protocol")
    cells = []
    for cell in CELLS:
        positions = [item for item in seeds if item["cell_id"] == cell]
        axial = {float(item["position_m"][2]) for item in positions}
        if len(axial) != 1 or len(positions) != 2:
            raise ValueError("cell position seeds do not form one axial sample per cell")
        cells.append(
            {
                "cell_id": cell,
                "z_m": axial.pop(),
                "radii_m": sorted(float(item["position_m"][0]) for item in positions),
                "flux_surface_ids": [item["flux_surface_id"] for item in positions],
            }
        )
    design = {
        "cells": cells,
        "energies_ev": list(launches_cfg["energies_ev"]),
        "pitch_angles_deg": list(launches_cfg["pitch_angles_deg"]),
        "directions": list(launches_cfg["directions"]),
        "gyrophase_count": launches_cfg["gyrophase_count"],
        "gyrophase_offset_rad": launches_cfg["gyrophase_offset_rad"],
        "gyrophase_offset_rule": launches_cfg["gyrophase_offset_rule"],
        "gyrophases_rad": list(plan["gyrophases_rad"]),
        "independent_repeats_per_stratum": launches_cfg["independent_repeats_per_stratum"],
        "estimator_policy": launches_cfg["estimator_policy"],
        "equal_weights": launches_cfg["equal_weights"],
        "launches_per_case": 512,
        "strata_per_case": 32,
        "launches_per_stratum": 16,
        "batches_per_case": 8,
        "batch_size": 64,
        "stratum_dimensions": list(launches_cfg["stratum_dimensions"]),
    }
    if len(plan["gyrophases_rad"]) != design["gyrophase_count"]:
        raise ValueError("gyrophase grid differs from the protocol")
    if (
        len(cells) * len(design["energies_ev"]) * len(design["pitch_angles_deg"])
        * len(design["directions"]) != 32
        or 32 * design["independent_repeats_per_stratum"] * design["gyrophase_count"] != 512
    ):
        raise ValueError("stratum design does not factor into 512 launches")

    gate_cfg = protocol["gates"]
    convergence = bundle.load("artifacts/probability-convergence.json")
    if convergence.get("timestep_passed") is not True or convergence.get(
        "cross_map_passed"
    ) is not True:
        raise ValueError("probability convergence did not pass")
    threshold = _finite(gate_cfg["maximum_successive_probability_change"], "gate threshold")

    cases: list[dict[str, Any]] = []
    orbit_columns: dict[str, list[Any]] = {}
    condition_codes: list[str] = []
    mu_values_by_case: list[list[float]] = []
    for index, case_id in enumerate(CASES):
        role, policy = case_id.split("-")
        block = _case_block(campaign["campaigns"][case_id], f"campaign-result {case_id}")
        summary_rel = f"artifacts/summaries/{case_id}.json"
        summary = bundle.load(summary_rel)
        if set(summary) != {
            "diagnostics", "final_checkpoint_file_sha256", "partial_checkpoint_file_sha256",
            "preflight", "sequential_batch_checkpoint_file_sha256", "strata", "summary",
            "timing_s", "worker_process_id",
        }:
            raise ValueError(f"{summary_rel} keys differ from the summary contract")
        case_summary = summary["summary"]
        expected_ensemble = f"cft-orbit-wall-loss-v4:{role}:{policy}"
        if case_summary.get("ensemble_id") != expected_ensemble:
            raise ValueError(f"{summary_rel} ensemble identity differs")
        result_identity = _hex64(case_summary.get("result_identity_sha256"), "result identity")
        stripped = {
            k: v for k, v in case_summary.items()
            if k not in {"ensemble_id", "result_identity_sha256"}
        }
        if stripped != campaign["campaigns"][case_id]:
            raise ValueError(f"{summary_rel} summary differs from campaign-result")
        strata = _strata(summary["strata"], f"{summary_rel}.strata")
        if sum(row["wall_hit"] for row in strata) != block["termination_counts"]["wall_hit"]:
            raise ValueError(f"{summary_rel} strata do not sum to the case wall hits")
        if sum(row["domain_escape"] for row in strata) != block["termination_counts"][
            "domain_escape"
        ]:
            raise ValueError(f"{summary_rel} strata do not sum to the case escapes")
        diagnostics = _diagnostics(summary["diagnostics"], f"{summary_rel}.diagnostics", 512)
        if diagnostics["termination_counts"] != block["termination_counts"]:
            raise ValueError(f"{summary_rel} diagnostics counts differ from campaign-result")
        preflight = _closed(
            summary["preflight"],
            f"{summary_rel}.preflight",
            {"launch_count", "maximum_declared_b_t", "maximum_launch_b_t", "status", "timestep_s"},
        )
        if preflight["status"] != "passed" or preflight["launch_count"] != 512:
            raise ValueError(f"{summary_rel} preflight did not pass for 512 launches")
        declared = _finite(preflight["maximum_declared_b_t"], "declared |B| bound")
        if diagnostics["runtime_max_b_t"] > declared:
            raise ValueError(f"{summary_rel} runtime |B| exceeds the declared bound")
        timing = _closed(
            summary["timing_s"],
            f"{summary_rel}.timing_s",
            {"checkpoints", "integration", "per_orbit_ms", "preflight", "total"},
        )
        orbits_rel = f"artifacts/orbits/{case_id}.json.gz"
        artifact, raw = bundle.load_gzip(orbits_rel)
        sidecar_rel = f"artifacts/orbits/{case_id}.json.sha256"
        sidecar_text = bundle.read(sidecar_rel).decode("ascii")
        if sidecar_text != f"{_digest(raw)}  {case_id}-orbit.json\n":
            raise ValueError(f"{sidecar_rel} does not record the decompressed orbit hash")
        if artifact.get("schema_version") != "cft-revival-orbit-mc-result/1.6.0":
            raise ValueError(f"{orbits_rel} schema differs from the bound contract")
        if artifact.get("campaign_id") != expected_ensemble:
            raise ValueError(f"{orbits_rel} campaign identity differs")
        integrity = _closed(
            artifact["integrity"], "orbit integrity", {"algorithm", "canonicalization", "payload_sha256"}
        )
        if integrity["canonicalization"] != "json-sort-keys-compact-utf8-v1":
            raise ValueError(f"{orbits_rel} canonicalization is unsupported")
        if _canonical_hash({k: v for k, v in artifact.items() if k != "integrity"}) != integrity[
            "payload_sha256"
        ]:
            raise ValueError(f"{orbits_rel} payload hash mismatch")
        if _canonical_hash(artifact["results"]) != artifact["identities"]["results_sha256"]:
            raise ValueError(f"{orbits_rel} results hash mismatch")
        if _canonical_hash(artifact["launches"]) != artifact["identities"]["launches_sha256"]:
            raise ValueError(f"{orbits_rel} launches hash mismatch")
        if artifact["summary"] != case_summary:
            raise ValueError(f"{orbits_rel} summary differs from the case summary")
        if artifact["convergence_evidence"] != gates["artifact_convergence_flags"]:
            raise ValueError(f"{orbits_rel} convergence flags differ from the gate report")
        columns, extra = _orbit_columns(artifact, index, strata, geometry, orbits_rel)
        for condition in extra["conditions"]:
            if condition not in condition_codes:
                condition_codes.append(condition)
        remap = [condition_codes.index(name) for name in extra["conditions"]]
        columns["condition"] = [remap[code] for code in columns["condition"]]
        for key, values in columns.items():
            orbit_columns.setdefault(key, []).extend(values)
        mu_values_by_case.append(columns["mu_variation"])
        mu_from_orbits = [
            float(r["maximum_instantaneous_mu_relative_variation"]) for r in artifact["results"]
        ]
        if (
            abs(min(mu_from_orbits) - diagnostics["mu"]["min"]) > 1e-15
            or abs(max(mu_from_orbits) - diagnostics["mu"]["max"]) > 1e-15
            or abs(median(mu_from_orbits) - diagnostics["mu"]["median"]) > 1e-15
            or sum(v > 0.1 for v in mu_from_orbits) != diagnostics["mu"]["count_above_0p1"]
            or sum(v > 0.5 for v in mu_from_orbits) != diagnostics["mu"]["count_above_0p5"]
        ):
            raise ValueError(f"{orbits_rel} mu statistics differ from the case diagnostics")
        steps_from_orbits = [int(r["steps"]) for r in artifact["results"]]
        if (
            min(steps_from_orbits) != diagnostics["steps"]["min"]
            or max(steps_from_orbits) != diagnostics["steps"]["max"]
            or sum(steps_from_orbits) != diagnostics["steps"]["total"]
            or median(steps_from_orbits) != diagnostics["steps"]["median"]
        ):
            raise ValueError(f"{orbits_rel} step statistics differ from the case diagnostics")
        cases.append(
            {
                "id": case_id,
                "index": index,
                "map_role": role,
                "timestep_policy": policy,
                "max_rotation_rad": geometry["timestep_policies"][policy],
                "ensemble_id": expected_ensemble,
                "result_identity_sha256": result_identity,
                "trials": block["trials"],
                "termination_counts": block["termination_counts"],
                "estimands": block["estimands"],
                "strata": strata,
                "diagnostics": diagnostics,
                "preflight": {
                    "timestep_s": _finite(preflight["timestep_s"], "timestep"),
                    "maximum_declared_b_t": declared,
                    "maximum_launch_b_t": _finite(preflight["maximum_launch_b_t"], "launch |B|"),
                },
                "timing_s": {key: _finite(value, f"timing {key}") for key, value in timing.items()},
                "artifacts": {
                    "summary": summary_rel,
                    "summary_sha256": bundle.hashes[summary_rel],
                    "orbits": orbits_rel,
                    "orbits_sha256": bundle.hashes[orbits_rel],
                    "orbits_payload_sha256": integrity["payload_sha256"],
                    "final_checkpoint_file_sha256": _hex64(
                        summary["final_checkpoint_file_sha256"], "final checkpoint"
                    ),
                },
            }
        )

    # Convergence panel: verify against campaign probabilities.
    wall_p = {case["id"]: case["estimands"]["wall_hit"]["probability"] for case in cases}
    wall_iv = {
        case["id"]: (case["estimands"]["wall_hit"]["lower"], case["estimands"]["wall_hit"]["upper"])
        for case in cases
    }

    def _check_chain(item: Mapping[str, Any], ids: Sequence[str], label: str) -> dict[str, Any]:
        probabilities = [_finite(v, f"{label} probability") for v in item["probabilities"]]
        if probabilities != [wall_p[i] for i in ids]:
            raise ValueError(f"{label} probabilities differ from the campaign estimands")
        changes = [_finite(v, f"{label} change") for v in item["successive_changes"]]
        if changes != [abs(b - a) for a, b in zip(probabilities[:-1], probabilities[1:])]:
            raise ValueError(f"{label} successive changes do not reproduce")
        overlaps = list(item["adjacent_wilson_overlap"])
        expected_overlap = [
            max(wall_iv[a][0], wall_iv[b][0]) <= min(wall_iv[a][1], wall_iv[b][1])
            for a, b in zip(ids[:-1], ids[1:])
        ]
        if overlaps != expected_overlap:
            raise ValueError(f"{label} interval-overlap flags do not reproduce")
        if item["passed"] is not True or max(changes) > threshold or not all(overlaps):
            raise ValueError(f"{label} did not pass the preregistered gate")
        return {
            "cases": list(ids),
            "probabilities": probabilities,
            "successive_changes": changes,
            "adjacent_wilson_overlap": overlaps,
            "passed": True,
        }

    timestep_chains = []
    for item in convergence["timestep"]:
        role = item["map_role"]
        chain = _check_chain(item, [f"{role}-{p}" for p in TIMESTEP_POLICIES], f"timestep {role}")
        chain["map_role"] = role
        timestep_chains.append(chain)
    cross_chains = []
    for item in convergence["cross_map"]:
        policy = item["timestep_policy"]
        chain = _check_chain(item, [f"{r}-{policy}" for r in MAP_ROLES], f"cross-map {policy}")
        chain["timestep_policy"] = policy
        cross_chains.append(chain)
    if [c["map_role"] for c in timestep_chains] != list(MAP_ROLES) or [
        c["timestep_policy"] for c in cross_chains
    ] != list(TIMESTEP_POLICIES):
        raise ValueError("convergence chains do not cover every map and policy")

    # Pooled (derived) counts.
    pooled_counts = {key: 0 for key in cases[0]["termination_counts"]}
    for case in cases:
        for key, value in case["termination_counts"].items():
            pooled_counts[key] += value
    pooled_trials = sum(case["trials"] for case in cases)
    if pooled_trials != 4608:
        raise ValueError("pooled trials differ from the orbit count")
    pooled_incomplete = pooled_trials - pooled_counts["wall_hit"] - pooled_counts[
        "domain_escape"
    ] - pooled_counts["reflected"]
    pooled = {
        "derived": True,
        "definition": "equal-weight sum of the nine per-case termination counts; the "
        "Wilson intervals treat the 4608 orbits as one binomial sample and are a "
        "descriptive summary, not a preregistered estimand",
        "trials": pooled_trials,
        "wall_hit": _wilson_95(pooled_counts["wall_hit"], pooled_trials),
        "escaped": _wilson_95(pooled_counts["domain_escape"], pooled_trials),
        "reflected": _wilson_95(pooled_counts["reflected"], pooled_trials),
        "incomplete": _wilson_95(pooled_incomplete, pooled_trials),
        "termination_counts": pooled_counts,
    }
    pooled_strata = []
    for stratum_index in range(32):
        rows = [case["strata"][stratum_index] for case in cases]
        base = rows[0]
        if any(row["key"] != base["key"] for row in rows):
            raise ValueError("stratum order differs between cases")
        wall = sum(row["wall_hit"] for row in rows)
        escape = sum(row["domain_escape"] for row in rows)
        trials = sum(row["trials"] for row in rows)
        wall_interval = _wilson_95(wall, trials)
        escape_interval = _wilson_95(escape, trials)
        pooled_strata.append(
            {
                **{k: base[k] for k in (
                    "index", "key", "cell_id", "cell", "kinetic_energy_ev",
                    "pitch_angle_deg", "parallel_direction",
                )},
                "trials": trials,
                "wall_hit": wall,
                "domain_escape": escape,
                "reflected": sum(row["reflected"] for row in rows),
                "timeout": sum(row["timeout"] for row in rows),
                "wall_hit_lower": wall_interval["lower"],
                "wall_hit_upper": wall_interval["upper"],
                "domain_escape_lower": escape_interval["lower"],
                "domain_escape_upper": escape_interval["upper"],
            }
        )

    # Diagnostics pooled across cases (derived from per-case artifact values).
    tolerance_close_total = sum(
        case["diagnostics"]["event_resolution"]["tolerance_close_fraction_zero"] for case in cases
    )
    mu_all = [value for values in mu_values_by_case for value in values]
    if (
        sum(v > 0.1 for v in mu_all) != mu_pooled["count_above_0p1"]
        or sum(v > 0.5 for v in mu_all) != mu_pooled["count_above_0p5"]
        or len(mu_all) != mu_pooled["orbit_count_with_mu"]
    ):
        raise ValueError("pooled mu counts differ between gates.json and the orbit artifacts")
    lock = bundle.load("execution-lock.json")
    first_transition = bundle.load("transitions/0001-lock-acquired.json")
    last_transition = bundle.load("transitions/0009-terminal.json")
    if first_transition.get("transition") != "lock-acquired" or last_transition.get(
        "transition"
    ) != "terminal":
        raise ValueError("transition log endpoints differ from the lifecycle contract")
    if last_transition["details"].get("state") != "accepted_result":
        raise ValueError("terminal transition state differs")
    lock_time = _timestamp(first_transition["recorded_at_utc"], "lock time")
    terminal_time = _timestamp(last_transition["recorded_at_utc"], "terminal time")
    wall_seconds = _seconds_between(lock_time, terminal_time)
    if wall_seconds <= 0:
        raise ValueError("lifecycle wall time is not positive")
    execution_mode = _closed(
        campaign["execution_mode"],
        "execution_mode",
        {"assessment_wall_s", "export_wall_s", "integration_wall_s", "parallel_cases",
         "worker_pool_size"},
    )
    runtime = bundle.load("artifacts/runtime.json")
    diagnostics = {
        "mu_variation": {
            "pooled": {
                "min": _finite(mu_pooled["min"], "mu min"),
                "median": _finite(mu_pooled["median"], "mu median"),
                "max": _finite(mu_pooled["max"], "mu max"),
                "count_above_0p1": mu_pooled["count_above_0p1"],
                "count_above_0p5": mu_pooled["count_above_0p5"],
                "orbit_count": mu_pooled["orbit_count_with_mu"],
                "fraction_above_0p1": mu_pooled["count_above_0p1"] / mu_pooled["orbit_count_with_mu"],
                "role": "diagnostic_only",
                "binding": False,
                "source": "artifacts/gates.json#diagnostics_not_gates.magnetic_moment_variation",
            },
            "statement": protocol["diagnostics"]["magnetic_moment_variation"]["statement"],
            "quantity": protocol["diagnostics"]["magnetic_moment_variation"]["quantity"],
        },
        "tolerance_close": {
            "derived": True,
            "total_events": tolerance_close_total,
            "orbit_count": pooled_trials,
            "share": tolerance_close_total / pooled_trials,
            "statement": "tolerance-close (event fraction 0) terminations are physical "
            "events under orbit_mc >= 1.5 (protocol.orbit_mc_contract."
            "tolerance_close_events_are_physical = true)",
            "physical": protocol["orbit_mc_contract"]["tolerance_close_events_are_physical"],
        },
        "energy": {
            "gate_limit": _finite(gates["energy_gate_limit"], "energy gate limit"),
            "maximum_relative_energy_error": 0.0,
            "orbits_exceeding_energy_gate": 0,
            "final_velocity_event_velocity_mismatches": _integer(
                gates["final_velocity_event_velocity_mismatches"], "velocity mismatches"
            ),
            "note": protocol["gates"]["energy_gate_note"],
        },
        "wall_endpoint": {
            "maximum_error_m": _finite(gates["maximum_wall_endpoint_error_m"], "wall endpoint"),
            "gate_m": _finite(gate_cfg["maximum_wall_endpoint_error_m"], "wall endpoint gate"),
        },
        "timing": {
            "lifecycle_wall_s": wall_seconds,
            "lifecycle_wall_definition": "transitions/0001-lock-acquired → transitions/0009-terminal "
            "recorded_at_utc difference (derived)",
            "lock_acquired_utc": lock_time,
            "terminal_utc": terminal_time,
            "integration_wall_s": _finite(execution_mode["integration_wall_s"], "integration wall"),
            "assessment_wall_s": _finite(execution_mode["assessment_wall_s"], "assessment wall"),
            "export_wall_s": _finite(execution_mode["export_wall_s"], "export wall"),
            "worker_pool_size": _integer(execution_mode["worker_pool_size"], "workers", 1),
            "parallel_cases": execution_mode["parallel_cases"] is True,
            "sequential_case_total_s": sum(case["timing_s"]["total"] for case in cases),
            "cpu_count": _integer(runtime["cpu_count"], "cpu count", 1),
            "platform": _text(runtime["platform"], "platform"),
            "python": _text(runtime["python"], "python"),
            "numpy": _text(runtime["numpy"], "numpy"),
            "device": _text(lock["device"], "device"),
            "statement": "Timings are diagnostic wall-clock observations of one shared "
            "workstation run; they are not a benchmark and establish no speed-up.",
        },
        "incomplete_and_failure_counts": dict(gates["incomplete_and_failure_counts"]),
    }
    if any(v != 0 for v in diagnostics["incomplete_and_failure_counts"].values()):
        raise ValueError("gate report records incomplete or failed orbits")

    # Field panel.
    field_maps = []
    for role in MAP_ROLES:
        evidence = bundle.load(f"artifacts/field-evidence/{role}.json")
        if evidence.get("role") != role or evidence.get("passed") is not True:
            raise ValueError(f"field evidence for {role} did not pass")
        declared = protocol["field_adapter"]["maps"][role]
        for key in ("checkpoint_file_sha256", "checkpoint_payload_sha256", "mesh_sha256", "run_sha256"):
            if evidence[key] != declared[key]:
                raise ValueError(f"field evidence {role} {key} differs from the protocol")
        if evidence["checkpoint_sidecar_sha256"] != declared["sidecar_file_sha256"]:
            raise ValueError(f"field evidence {role} sidecar differs from the protocol")
        projection = _field_projection(bundle, role, evidence)
        projection["checkpoint_path"] = declared["checkpoint_path"]
        projection["runtime_max_b_t_by_case"] = {
            case["id"]: case["diagnostics"]["runtime_max_b_t"]
            for case in cases if case["map_role"] == role
        }
        projection["maximum_declared_b_t_by_case"] = {
            case["id"]: case["preflight"]["maximum_declared_b_t"]
            for case in cases if case["map_role"] == role
        }
        if any(
            abs(value - projection["certified_max_b_t"]) > 1e-15
            for value in projection["maximum_declared_b_t_by_case"].values()
        ):
            raise ValueError(f"declared |B| bound for {role} differs from the certificate")
        field_maps.append(projection)
    field_convergence = bundle.load("artifacts/field-map-convergence.json")
    p2 = bundle.load("artifacts/p2-input-authority.json")
    if (
        p2["qualification_status"] != protocol["authority"]["required_qualification"]
        or p2["design_id"] != protocol["authority"]["design_id"]
        or p2["manifest_file_sha256"] != protocol["authority"]["manifest"]["file_sha256"]
        or p2["result_file_sha256"] != protocol["authority"]["result"]["file_sha256"]
    ):
        raise ValueError("P2 input authority differs from the protocol")
    manufactured = bundle.load("artifacts/manufactured-gates.json")
    if manufactured.get("passed") is not True or any(
        v is not True for v in manufactured["checks"].values()
    ):
        raise ValueError("manufactured gates did not pass")
    field = {
        "p2_authority": {
            "design_id": p2["design_id"],
            "qualification_status": p2["qualification_status"],
            "manifest_path": protocol["authority"]["manifest"]["path"],
            "manifest_file_sha256": p2["manifest_file_sha256"],
            "result_path": protocol["authority"]["result"]["path"],
            "result_file_sha256": p2["result_file_sha256"],
            "p2_evidence_commit": protocol["authority"]["p2_evidence_commit"],
            "excluded_designs": list(protocol["authority"]["excluded_designs"]),
        },
        "plasma_material_id": protocol["field_adapter"]["plasma_material_id"],
        "plasma_region_ids": list(protocol["field_adapter"]["plasma_region_ids"]),
        "maps": field_maps,
        "cross_map_convergence": field_convergence,
        "adapter_limits": {
            "maximum_b_relative_rms": protocol["field_adapter"]["maximum_b_relative_rms"],
            "maximum_b_component_absolute_error_t": protocol["field_adapter"][
                "maximum_b_component_absolute_error_t"
            ],
            "maximum_cross_map_b_relative_rms": protocol["field_adapter"][
                "maximum_cross_map_b_relative_rms"
            ],
            "withheld_midcell_stride": protocol["field_adapter"]["withheld_midcell_stride"],
        },
        "manufactured": {
            "checks": dict(manufactured["checks"]),
            "helix_orders": list(manufactured["helix_convergence"]["observed_orders"]),
            "varying_e_orders": list(manufactured["varying_e_convergence"]["observed_orders"]),
            "cpu_parity_max_relative": manufactured["cpu_parity"][
                "maximum_relative_velocity_difference"
            ],
            "cuda_parity_max_relative": manufactured["cuda_parity"][
                "maximum_relative_velocity_difference"
            ],
            "cuda_device": manufactured["cuda_parity"]["device"],
        },
        "runtime_max_b_t_min": min(case["diagnostics"]["runtime_max_b_t"] for case in cases),
        "runtime_max_b_t_max": max(case["diagnostics"]["runtime_max_b_t"] for case in cases),
    }

    # Claim boundary (verbatim).
    coupling = bundle.load("artifacts/coupling-export-only.json")
    if coupling["integration_status"] != COUPLING_STATUS:
        raise ValueError("coupling export status differs")
    handoff_matches = [
        case for case in cases
        if case["result_identity_sha256"] == coupling["result_identity_sha256"]
    ]
    if len(handoff_matches) != 1:
        raise ValueError("coupling export does not identify exactly one case")
    handoff_case = handoff_matches[0]
    if (
        handoff_case["estimands"]["wall_hit"]["probability"] != coupling["probability"]
        or handoff_case["estimands"]["wall_hit"]["lower"] != coupling["confidence_interval_95"][0]
        or handoff_case["estimands"]["wall_hit"]["upper"] != coupling["confidence_interval_95"][1]
        or handoff_case["trials"] != coupling["trial_count"]
    ):
        raise ValueError("coupling export values differ from the identified case")
    limitations = _closed(
        campaign["limitations"],
        "limitations",
        {
            "coupling_status", "direct_estimands", "forbid_mirror_formula_publication",
            "forbid_pic_or_self_consistent_claim", "forbid_plasma_performance_publication",
            "hardware_or_experimental_validation", "overall_estimand_only_under_equal_weights",
            "shakedown_outcomes_are_not_evidence",
        },
    )
    if protocol["publication_boundary"] != dict(limitations):
        raise ValueError("campaign limitations differ from the protocol publication boundary")
    readme_path = experiment_root / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    readme_boundary = (
        "No result is PIC, self-consistent plasma, experimental validation, hardware\n"
        "validation, total thruster performance, or mirror-formula publication."
    )
    if readme_boundary not in readme_text:
        raise ValueError("README claim boundary sentence is not verbatim")
    first_artifact, _ = bundle.load_gzip(cases[0]["artifacts"]["orbits"])
    orbit_limitations = list(first_artifact["limitations"])
    claim_boundary = {
        "classification": CLASSIFICATION,
        "orbit_artifact_classification": first_artifact["classification"],
        "limitations": dict(limitations),
        "orbit_artifact_limitations": orbit_limitations,
        "readme_statement": readme_boundary,
        "readme_sha256": _lf_digest(readme_path),
        "coupling": {
            "integration_status": coupling["integration_status"],
            "plasma_network_role": coupling["plasma_network_role"],
            "coupling_target": coupling["coupling_target"],
            "quantity": coupling["quantity"],
            "probability": coupling["probability"],
            "confidence_interval_95": list(coupling["confidence_interval_95"]),
            "standard_uncertainty": coupling["standard_uncertainty"],
            "trial_count": coupling["trial_count"],
            "verification_status": coupling["verification_status"],
            "schema_version": coupling["schema_version"],
            "handoff_case": handoff_case["id"],
            "classification": coupling["classification"],
        },
        "pooled_note": "The pooled wall-hit fraction is an equal-weight design average over "
        "the 32 preregistered strata; the per-cell result is bimodal (cells 2–3 and the "
        "exit cell are saturated), so the pooled value characterises this launch design, "
        "not a physical loss rate.",
        "reflection_note": "Zero reflections were observed in all 4608 orbits, so the "
        "mirror-formula picture does not apply to this field and design; "
        "forbid_mirror_formula_publication is true.",
    }

    # Lineage (verbatim strings from the protocol and DEVLOG).
    disclosure = protocol["prior_campaign_disclosure"]
    prior = []
    for version in ("v1", "v2", "v3"):
        entry = disclosure[version]
        prior.append(
            {
                "version": version,
                "branch": entry["branch"],
                "preregistration_commit": entry["preregistration_commit"],
                "result_commit": entry["result_commit"],
                "orbit_mc_commit": entry["orbit_mc_commit"],
                "terminal_state": entry["terminal_state"],
                "primary_error_type": entry["primary_error_type"],
                "primary_error_message": entry["primary_error_message"],
                "root_cause": entry["root_cause"],
                "persisted_orbit_outcome_count": entry.get(
                    "persisted_orbit_outcome_count", entry.get("orbit_outcome_access_count")
                ),
            }
        )
    devlog_path = experiment_root / "DEVLOG.md"
    devlog_text = devlog_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    for key, quote in DEVLOG_QUOTES.items():
        if quote not in devlog_text:
            raise ValueError(f"DEVLOG quote {key} is not verbatim")
    shakedown = bundle.load("artifacts/shakedown.json")
    authorities = bundle.load("artifacts/authorities.json")
    if (
        shakedown.get("passed") is not True
        or shakedown.get("evidentiary") is not False
        or shakedown.get("outcomes_enter_estimand") is not False
        or shakedown["disjointness"].get("proven") is not True
        or authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"]
    ):
        raise ValueError("shakedown record differs from the bound authorities")
    if (
        authorities["orbit_mc_source_sha256"] != shakedown["orbit_mc_source_sha256"]
        or authorities["protocol_semantic_sha256"] != shakedown["protocol_semantic_sha256"]
        or authorities["orbit_mc_package_version"] != protocol["orbit_mc_contract"]["package_version"]
    ):
        raise ValueError("authorities differ from the shakedown/protocol bindings")
    shakedown_cases = shakedown["cases"]
    shakedown_summary = {
        "passed": True,
        "evidentiary": False,
        "git_head": shakedown["git"]["head"],
        "git_dirty_entry_count": shakedown["git"]["dirty_entry_count"],
        "runtime_seconds": shakedown["runtime"]["runtime_seconds"],
        "terminal_state": shakedown["runtime"]["terminal_state"],
        "bundle_validated": shakedown["runtime"]["bundle_validated"],
        "manifest_artifact_count": shakedown["runtime"]["manifest_artifact_count"],
        "launch_count": shakedown["disjointness"]["shakedown_launch_count"],
        "energy_summary": dict(shakedown["energy_summary"]),
        "informational_gates_passed": shakedown["informational_gates"]["passed"],
        "per_case_validator_calls": sum(
            _integer(case["validators"]["passed"], "shakedown validators")
            for case in shakedown_cases.values()
        ),
        "per_case_validator_failures": sum(
            _integer(case["validators"]["failed"], "shakedown validator failures")
            for case in shakedown_cases.values()
        ),
        "development_accepted": shakedown["development"]["accepted"] is True,
        "per_case_termination": {
            case_id: dict(shakedown_cases[case_id]["diagnostics"]["termination_counts"])
            for case_id in CASES
        },
        "disjointness_reports": {
            name: {"disjoint": report["disjoint"], "overlap_counts": dict(report["overlap_counts"])}
            for name, report in shakedown["disjointness"]["reports"].items()
        },
        "disclosure": shakedown["disclosure"],
        "execution_mode": dict(shakedown["execution_mode"]),
    }
    lineage = {
        "prior_campaigns": prior,
        "common_pattern": disclosure["common_pattern"],
        "shakedown_rule": disclosure["shakedown_rule"],
        "v4_design_freshness": {
            key: disclosure[key]
            for key in (
                "v4_launch_grid_reused", "v4_fresh_positions", "v4_rotated_gyrophase_grid",
                "v4_case_ids_and_seed_ids_recomputed",
            )
        },
        "v1_5_fix": disclosure["v3"]["root_cause"],
        "v1_6_fix": protocol["gates"]["energy_gate_note"],
        "second_latent_v3_bug": DEVLOG_QUOTES["second_latent_v3_bug"],
        "v1_6_shakedown_energy": DEVLOG_QUOTES["v1_6_shakedown_energy"],
        "devlog_sha256": _lf_digest(devlog_path),
        "shakedown": shakedown_summary,
        "orbit_mc_contract": {
            "package_version": protocol["orbit_mc_contract"]["package_version"],
            "result_schema_version": protocol["orbit_mc_contract"]["result_schema_version"],
            "source_sha256": authorities["orbit_mc_source_sha256"],
            "code_identity_sha256": authorities["orbit_mc_code_identity_sha256"],
            "event_velocity_definition": protocol["orbit_mc_contract"]["event_velocity_definition"],
        },
        "authority_commits": {
            "base_commit": protocol["authority"]["base_commit"],
            "orbit_mc_commit": protocol["authority"]["orbit_mc_commit"],
            "experiment_runtime_commit": protocol["authority"]["experiment_runtime_commit"],
            "p2_evidence_commit": protocol["authority"]["p2_evidence_commit"],
        },
    }

    headline = {
        "state": terminal["state"],
        "status": campaign["status"],
        "classification": CLASSIFICATION,
        "case_count": 9,
        "launches_per_case": 512,
        "orbit_count": 4608,
        "strata_per_case": 32,
        "launches_per_stratum": 16,
        "binding_gate_count": len(GATE_CHECKS),
        "binding_gates_passed": sum(1 for v in checks.values() if v is True),
        "gate_checks": {key: checks[key] for key in GATE_CHECKS},
        "validators": {"passed": validators["passed"], "failed": validators["failed"]},
        "exact_authority_replay_count": gates["exact_authority_replay_count"],
        "attempt_count": counts["attempt_count"],
        "label_access_count": counts["label_access_count"],
        "expensive_operation_count": counts["expensive_operation_count"],
        "pooled": pooled,
        "gate_threshold_successive_change": threshold,
        "require_adjacent_wilson_overlap": gate_cfg["require_adjacent_wilson_overlap"],
    }

    payload = {
        "schema": SCHEMA,
        "title": "CFT full-orbit electron wall loss — campaign v4",
        "warning": (
            "Collisionless prescribed-field test-particle wall-loss evidence only: not PIC, "
            "not a self-consistent plasma, not experimental or hardware validation, not "
            "thruster performance. Coupling is export-only pending consumer integration."
        ),
        "identity": {
            "results_commit_sha": RESULTS_COMMIT_SHA,
            "preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
            "execution_lock_commit": lock["commit"],
            "manifest_file_sha256": bundle.manifest_sha256,
            "terminal_file_sha256": bundle.hashes["terminal.json"],
            "lock_file_sha256": bundle.hashes["execution-lock.json"],
            "protocol_semantic_sha256": authorities["protocol_semantic_sha256"],
            "protocol_file_sha256": bundle.hashes["artifacts/protocol.json"],
            "campaign_result_sha256": bundle.hashes["artifacts/campaign-result.json"],
            "gates_sha256": bundle.hashes["artifacts/gates.json"],
            "probability_convergence_sha256": bundle.hashes["artifacts/probability-convergence.json"],
            "authorities_sha256": bundle.hashes["artifacts/authorities.json"],
            "shakedown_sha256": bundle.hashes["artifacts/shakedown.json"],
            "coupling_export_sha256": bundle.hashes["artifacts/coupling-export-only.json"],
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "artifact_hashes": dict(sorted(bundle.hashes.items())),
            "sidecar_tolerance": {
                "policy": "exactly the nine artifacts/orbits/<case>.json.sha256 text sidecars "
                "may mismatch by CRLF→LF checkout normalisation; verified as "
                "sha256(bytes.replace(LF, CRLF)) == recorded; every other byte mismatch fails",
                "tolerated": bundle.tolerated,
            },
            "generator_sha256": _lf_digest(Path(__file__).resolve()),
            "template_sha256": _lf_digest(TEMPLATE_PATH),
            "canonicalization": bundle.manifest["canonicalization"],
            "max_html_bytes": MAX_HTML_BYTES,
        },
        "headline": headline,
        "cases": cases,
        "convergence": {
            "threshold": threshold,
            "require_adjacent_wilson_overlap": gate_cfg["require_adjacent_wilson_overlap"],
            "timestep": timestep_chains,
            "cross_map": cross_chains,
            "timestep_passed": True,
            "cross_map_passed": True,
        },
        "design": design,
        "geometry": geometry,
        "pooled_strata": pooled_strata,
        "orbits": {
            "policy": "all 4608 orbit endpoints are embedded (one row per orbit, column-"
            "oriented); no downsampling was applied",
            "count": len(orbit_columns["case"]),
            "termination_codes": list(TERMINATION_CODES),
            "condition_codes": condition_codes,
            "columns": orbit_columns,
            "column_units": {
                "final_r_mm": "mm (hypot of final x,y), 5 decimals",
                "final_z_mm": "mm, 5 decimals",
                "steps": "Boris steps",
                "mu_variation": "maximum_instantaneous_mu_relative_variation, 5 significant digits",
                "path_mm": "path length, mm, 4 decimals",
                "max_b_t": "maximum |B| seen along the orbit, T, 6 significant digits",
            },
        },
        "diagnostics": diagnostics,
        "field": field,
        "claim_boundary": claim_boundary,
        "lineage": lineage,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate the dashboard-specific envelope and its embedded identities."""

    _closed(
        payload,
        "dashboard payload",
        {
            "schema", "title", "warning", "identity", "headline", "cases", "convergence",
            "design", "geometry", "pooled_strata", "orbits", "diagnostics", "field",
            "claim_boundary", "lineage",
        },
    )
    if payload["schema"] != SCHEMA:
        raise ValueError("dashboard payload schema is unsupported")
    identity = payload["identity"]
    if (
        identity["results_commit_sha"] != RESULTS_COMMIT_SHA
        or identity["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA
        or identity["execution_lock_commit"] != PREREGISTRATION_COMMIT_SHA
        or identity["manifest_file_sha256"] != EXPECTED_MANIFEST_SHA256
        or identity["terminal_file_sha256"] != EXPECTED_TERMINAL_SHA256
        or identity["lock_file_sha256"] != EXPECTED_LOCK_SHA256
    ):
        raise ValueError("embedded committed identity is invalid or superseded")
    tolerated = {item["path"] for item in identity["sidecar_tolerance"]["tolerated"]}
    if tolerated != set(TOLERATED_CRLF_SIDECARS):
        raise ValueError("tolerated sidecar set differs from the disclosed defect")
    headline = payload["headline"]
    if (
        headline["state"] != "accepted_result"
        or headline["status"] != "accepted"
        or headline["classification"] != CLASSIFICATION
        or headline["case_count"] != 9
        or headline["orbit_count"] != 4608
        or headline["binding_gate_count"] != 15
        or headline["binding_gates_passed"] != 15
        or headline["validators"]["failed"] != 0
    ):
        raise ValueError("embedded headline differs from the accepted campaign")
    cases = payload["cases"]
    if [case["id"] for case in cases] != list(CASES):
        raise ValueError("embedded case identities/order are invalid")
    for case in cases:
        if case["estimands"]["reflected"]["successes"] != 0:
            raise ValueError("embedded reflection count contradicts the zero-reflection result")
        if len(case["strata"]) != 32:
            raise ValueError("embedded strata count is invalid")
    orbits = payload["orbits"]
    lengths = {len(values) for values in orbits["columns"].values()}
    if lengths != {4608} or orbits["count"] != 4608:
        raise ValueError("embedded orbit columns are not complete")
    if payload["claim_boundary"]["coupling"]["integration_status"] != COUPLING_STATUS:
        raise ValueError("embedded coupling status differs")
    if payload["diagnostics"]["mu_variation"]["pooled"]["binding"] is not False:
        raise ValueError("embedded mu diagnostic must not be a gate")
    if [m["role"] for m in payload["field"]["maps"]] != list(MAP_ROLES):
        raise ValueError("embedded field maps are incomplete")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    template = TEMPLATE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    if template.count("__DATA__") != 1:
        raise ValueError("dashboard template must contain exactly one data placeholder")
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).replace("</", "<\\/")
    html = template.replace("__DATA__", encoded)
    size = len(html.encode("utf-8"))
    if size > MAX_HTML_BYTES:
        raise ValueError(f"rendered dashboard is {size} bytes, above the {MAX_HTML_BYTES} cap")
    return html


def generate(
    output_path: Path = DEFAULT_OUTPUT,
    results_root: Path = RESULTS,
    experiment_root: Path = EXPERIMENT,
) -> str:
    html = render_html(build_payload(results_root, experiment_root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8", newline="\n")
    return _digest(html.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    args = parser.parse_args(argv)
    print(generate(args.output, args.results, args.experiment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
