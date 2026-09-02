"""Generate hash-bound paper evidence for the CFT full-orbit wall-loss v4 campaign.

Reads the sealed results bundle of ``modern/experiments/cft_orbit_wall_loss_v4``
(verified byte-for-byte against ``results/manifest.json``), binds it to the
committed results revision, and writes:

* ``paper/evidence/wall-loss-v4.json`` — every macro value with the artifact
  path, JSON pointer, formatter and artifact SHA-256 it was read from;
* ``paper/generated/wall-loss-v4.tex`` — ``\\newcommand`` macros and two
  generated tables for the draft results subsection;
* ``paper/generated/wall-loss-v4.provenance.json`` — generator/input/output
  hashes in the same shape as the L0 table sidecar.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The disclosed bundle defect (nine
``artifacts/orbits/<case>.json.sha256`` sidecars recorded with CRLF bytes) is
tolerated exactly as ``sha256(bytes.replace(LF, CRLF)) == recorded`` and
nothing else.
"""

from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Callable

EXPERIMENT = Path("modern/experiments/cft_orbit_wall_loss_v4")
RESULTS = EXPERIMENT / "results"
EVIDENCE_PATH = Path("paper/evidence/wall-loss-v4.json")
OUTPUT_PATH = Path("paper/generated/wall-loss-v4.tex")
SIDECAR_PATH = Path("paper/generated/wall-loss-v4.provenance.json")
SECTION_PATH = Path("paper/sections/wall-loss-v4.tex")

RESULTS_COMMIT_SHA = "6922a3cf97d261735266aa1a5a0c0c9683e021ca"
PREREGISTRATION_COMMIT_SHA = "757e365f9f667620c7610663574294c3b71e1f51"
CASES = (
    "primary-N", "primary-2N", "primary-4N",
    "refined-N", "refined-2N", "refined-4N",
    "enlarged-N", "enlarged-2N", "enlarged-4N",
)
MAP_ROLES = ("primary", "refined", "enlarged")
POLICIES = ("N", "2N", "4N")
TOLERATED_CRLF_SIDECARS = frozenset(f"artifacts/orbits/{case}.json.sha256" for case in CASES)
CLASSIFICATION = "collisionless_prescribed_field_test_particle_wall_loss_not_pic"
Z = 1.959963984540054


# --------------------------------------------------------------------------- #
# Formatting (shared with the test through this module)
# --------------------------------------------------------------------------- #
def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _sci(value: float, digits: int) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


FORMATTERS: dict[str, Callable[[Any], str]] = {
    "int": lambda v: f"{int(v):d}",
    "int_comma": lambda v: f"{int(v):,d}".replace(",", "{,}"),
    "fixed1": lambda v: f"{float(v):.1f}",
    "fixed2": lambda v: f"{float(v):.2f}",
    "fixed3": lambda v: f"{float(v):.3f}",
    "fixed4": lambda v: f"{float(v):.4f}",
    "fixed6": lambda v: f"{float(v):.6f}",
    "pct1": lambda v: f"{100.0 * float(v):.1f}\\%",
    "pct2": lambda v: f"{100.0 * float(v):.2f}\\%",
    "mm1": lambda v: f"{1e3 * float(v):.1f}",
    "mm2": lambda v: f"{1e3 * float(v):.2f}",
    "sci2": lambda v: _sci(float(v), 2),
    "sci3": lambda v: _sci(float(v), 3),
    "g": lambda v: f"{float(v):g}",
    "text": lambda v: _tex_escape(str(v)),
    "ident": lambda v: _tex_escape(str(v)).replace("\\_", "\\_\\allowbreak{}").replace("-", "-\\allowbreak{}"),
    "bool": lambda v: "true" if v is True else "false" if v is False else _tex_escape(str(v)),
    "list_g": lambda v: ", ".join(f"{float(x):g}" for x in v),
    "list_text": lambda v: ", ".join(_tex_escape(str(x)) for x in v),
    "sha_short": lambda v: _tex_escape(str(v)[:12]),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


# --------------------------------------------------------------------------- #
# Strict JSON, hashing, pointers
# --------------------------------------------------------------------------- #
def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_json_bytes(raw: bytes, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate key {key!r}")
            result[key] = value
        return result

    def reject(constant: str) -> None:
        raise ValueError(f"{label}: nonfinite constant {constant!r}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid UTF-8 JSON") from exc


def resolve_pointer(value: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer."""

    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    current = value
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"pointer {pointer!r}: missing key {token!r}")
            current = current[token]
        else:
            raise KeyError(f"pointer {pointer!r}: cannot descend into scalar at {token!r}")
    return current


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


# --------------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------------- #
class Bundle:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.root = repo / RESULTS
        manifest_raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.manifest = load_json_bytes(manifest_raw, "results manifest")
        if self.manifest.get("state") != "accepted_result":
            raise ValueError("results manifest state is not accepted_result")
        if self.manifest.get("experiment_id") != "cft-orbit-wall-loss-v4":
            raise ValueError("results manifest experiment identity differs")
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        self.tolerated: list[str] = []
        entries = self.manifest["artifacts"]
        if len(entries) != self.manifest["artifact_count"]:
            raise ValueError("results manifest artifact count differs")
        for entry in entries:
            if entry["type"] != "file":
                continue
            relative = entry["path"]
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"manifest path escapes the bundle: {relative}")
            raw = (self.root / pure).read_bytes()
            actual = sha256_bytes(raw)
            if actual != entry["byte_sha256"]:
                crlf = raw.replace(b"\n", b"\r\n")
                if (
                    relative in TOLERATED_CRLF_SIDECARS
                    and b"\r" not in raw
                    and sha256_bytes(crlf) == entry["byte_sha256"]
                    and len(crlf) == entry["bytes"]
                ):
                    self.tolerated.append(relative)
                else:
                    raise ValueError(f"bundle file SHA-256 mismatch: {relative}")
            elif len(raw) != entry["bytes"]:
                raise ValueError(f"bundle file size mismatch: {relative}")
            self.hashes[relative] = actual
            self.sizes[relative] = len(raw)
        self.tolerated.sort()
        if self.hashes["terminal.json"] != self.manifest["terminal_byte_sha256"]:
            raise ValueError("terminal.json hash differs from the manifest binding")
        if self.hashes["execution-lock.json"] != self.manifest["lock_byte_sha256"]:
            raise ValueError("execution-lock.json hash differs from the manifest binding")
        self.used: dict[str, dict[str, Any]] = {}

    def load(self, relative: str) -> Any:
        if relative not in self.hashes:
            raise ValueError(f"{relative} is not manifest-bound")
        raw = (self.root / relative).read_bytes()
        if relative.endswith(".gz"):
            raw = gzip.decompress(raw)
        self.used[relative] = {"sha256": self.hashes[relative], "bytes": self.sizes[relative]}
        return load_json_bytes(raw, relative)

    def bind_committed(self) -> dict[str, Any]:
        """Prove the working-tree bundle equals the committed results revision."""

        head = _git(self.repo, "rev-parse", "HEAD")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", RESULTS_COMMIT_SHA, head],
            cwd=self.repo, check=False, capture_output=True,
        ).returncode == 0
        if not ancestor:
            raise ValueError("results commit is not an ancestor of HEAD")
        manifest_rel = (RESULTS / "manifest.json").as_posix()
        committed_blob = _git(self.repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{manifest_rel}")
        working_blob = _git(self.repo, "hash-object", "--", manifest_rel)
        if committed_blob != working_blob:
            raise ValueError("working-tree results manifest differs from the committed blob")
        subject = _git(self.repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA)
        return {
            "results_commit": RESULTS_COMMIT_SHA,
            "results_commit_subject": subject,
            "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
            "manifest_git_blob": committed_blob,
            "manifest_path": manifest_rel,
        }


# --------------------------------------------------------------------------- #
# Macro construction
# --------------------------------------------------------------------------- #
class Macros:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle
        self.items: list[dict[str, Any]] = []
        self.names: set[str] = set()
        self.docs: dict[str, Any] = {}

    def doc(self, relative: str) -> Any:
        if relative not in self.docs:
            self.docs[relative] = self.bundle.load(relative)
        return self.docs[relative]

    def add(self, name: str, artifact: str, pointer: str, fmt: str, description: str) -> str:
        if name in self.names or not name.isalpha():
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        raw = resolve_pointer(self.doc(artifact), pointer)
        value = format_value(fmt, raw)
        self.items.append(
            {
                "name": name,
                "value": value,
                "raw": raw,
                "format": fmt,
                "derived": False,
                "source": {"artifact": artifact, "pointer": pointer},
                "description": description,
            }
        )
        self.names.add(name)
        return value

    def add_derived(
        self, name: str, raw: Any, fmt: str, description: str, derivation: str,
        inputs: list[dict[str, str]],
    ) -> str:
        if name in self.names or not name.isalpha():
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        value = format_value(fmt, raw)
        self.items.append(
            {
                "name": name,
                "value": value,
                "raw": raw,
                "format": fmt,
                "derived": True,
                "derivation": derivation,
                "inputs": inputs,
                "description": description,
            }
        )
        self.names.add(name)
        return value


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    p = successes / trials
    d = 1.0 + Z * Z / trials
    c = (p + Z * Z / (2.0 * trials)) / d
    h = Z * ((p * (1.0 - p) / trials + Z * Z / (4.0 * trials * trials)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def _seconds(start: str, end: str) -> float:
    a = datetime.fromisoformat(start.replace("Z", "+00:00"))
    b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (b - a).total_seconds()


def build(repo: Path) -> tuple[dict[str, Any], str]:
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo)
    binding = bundle.bind_committed()
    m = Macros(bundle)
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    terminal = m.doc("terminal.json")
    protocol = m.doc("artifacts/protocol.json")
    convergence = m.doc("artifacts/probability-convergence.json")
    coupling = m.doc("artifacts/coupling-export-only.json")
    authorities = m.doc("artifacts/authorities.json")
    field_convergence = m.doc("artifacts/field-map-convergence.json")
    if terminal["payload"] != campaign or campaign["gates"] != gates:
        raise ValueError("terminal/campaign/gates artifacts disagree")
    if campaign["classification"] != CLASSIFICATION or campaign["status"] != "accepted":
        raise ValueError("campaign is not the accepted wall-loss classification")
    if len(gates["checks"]) != 15 or not all(v is True for v in gates["checks"].values()):
        raise ValueError("binding gate checks are not all true")
    if campaign["limitations"] != protocol["publication_boundary"]:
        raise ValueError("campaign limitations differ from the protocol publication boundary")

    # Identity and design.
    m.add("WlfClassification", "artifacts/campaign-result.json", "/classification", "ident", "campaign classification string")
    m.add("WlfTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("WlfStatus", "artifacts/campaign-result.json", "/status", "text", "assessment status")
    m.add("WlfCaseCount", "artifacts/campaign-result.json", "/campaign_count", "int", "number of cases")
    m.add("WlfLaunchesPerCase", "artifacts/campaign-result.json", "/launches_per_case", "int", "launches per case")
    m.add("WlfOrbitCount", "artifacts/campaign-result.json", "/orbit_count", "int", "orbits integrated")
    m.add("WlfStrataPerCase", "artifacts/protocol.json", "/launches/strata_per_case", "int", "strata per case")
    m.add("WlfRepeatsPerStratum", "artifacts/protocol.json", "/launches/independent_repeats_per_stratum", "int", "position repeats per stratum")
    m.add("WlfGyrophaseCount", "artifacts/protocol.json", "/launches/gyrophase_count", "int", "gyrophases per position")
    m.add("WlfEnergies", "artifacts/protocol.json", "/launches/energies_ev", "list_g", "kinetic energies (eV)")
    m.add("WlfPitches", "artifacts/protocol.json", "/launches/pitch_angles_deg", "list_g", "pitch angles (deg)")
    m.add("WlfEstimatorPolicy", "artifacts/protocol.json", "/launches/estimator_policy", "ident", "estimator policy")
    m.add("WlfPlasmaRegions", "artifacts/protocol.json", "/field_adapter/plasma_region_ids", "list_text", "plasma region ids")
    if set(protocol["field_adapter"]["maps"]) != set(MAP_ROLES):
        raise ValueError("protocol field maps differ from primary/refined/enlarged")
    m.add_derived(
        "WlfMapRoleList", list(MAP_ROLES), "list_text", "field map roles in campaign order",
        "keys of field_adapter.maps ordered primary, refined, enlarged",
        [{"artifact": "artifacts/protocol.json", "pointer": "/field_adapter/maps"}],
    )
    m.add_derived(
        "WlfPolicyList", list(POLICIES), "list_text", "timestep policies in refinement order",
        "keys of orbit.timestep_policies ordered N, 2N, 4N",
        [{"artifact": "artifacts/protocol.json", "pointer": "/orbit/timestep_policies"}],
    )
    if set(protocol["orbit"]["timestep_policies"]) != set(POLICIES):
        raise ValueError("protocol timestep policies differ from N/2N/4N")
    m.add("WlfWallRadiusMm", "artifacts/protocol.json", "/orbit/wall/radius_m", "mm1", "dielectric wall radius (mm)")
    m.add("WlfWallZMinMm", "artifacts/protocol.json", "/orbit/wall/z_min_m", "mm1", "wall start (mm)")
    m.add("WlfWallZMaxMm", "artifacts/protocol.json", "/orbit/wall/z_max_m", "mm1", "wall end (mm)")
    m.add("WlfDomainZMaxMm", "artifacts/protocol.json", "/orbit/domain/z_max_m", "mm1", "domain exit plane (mm)")
    m.add("WlfCellOneZMm", "artifacts/protocol.json", "/launches/position_seeds/0/position_m/2", "mm1", "cell 1 axial position (mm)")
    m.add("WlfCellTwoZMm", "artifacts/protocol.json", "/launches/position_seeds/2/position_m/2", "mm1", "cell 2 axial position (mm)")
    m.add("WlfCellThreeZMm", "artifacts/protocol.json", "/launches/position_seeds/4/position_m/2", "mm1", "cell 3 axial position (mm)")
    m.add("WlfCellFourZMm", "artifacts/protocol.json", "/launches/position_seeds/6/position_m/2", "mm1", "cell 4 axial position (mm)")
    m.add("WlfRadiusInnerMm", "artifacts/protocol.json", "/launches/position_seeds/0/position_m/0", "mm2", "inner launch radius (mm)")
    m.add("WlfRadiusOuterMm", "artifacts/protocol.json", "/launches/position_seeds/1/position_m/0", "mm2", "outer launch radius (mm)")
    m.add("WlfPolicyN", "artifacts/protocol.json", "/orbit/timestep_policies/N/max_rotation_rad", "g", "max gyro-rotation per step, policy N (rad)")
    m.add("WlfPolicyTwoN", "artifacts/protocol.json", "/orbit/timestep_policies/2N/max_rotation_rad", "g", "policy 2N (rad)")
    m.add("WlfPolicyFourN", "artifacts/protocol.json", "/orbit/timestep_policies/4N/max_rotation_rad", "g", "policy 4N (rad)")
    m.add("WlfMaxTimeS", "artifacts/protocol.json", "/orbit/max_time_s", "sci2", "integration time limit (s)")
    m.add("WlfMaxPathMm", "artifacts/protocol.json", "/orbit/max_path_m", "mm1", "integration path limit (mm)")
    m.add("WlfMaxSteps", "artifacts/protocol.json", "/orbit/max_steps", "int_comma", "integration step limit")
    m.add("WlfOrbitMcVersion", "artifacts/authorities.json", "/orbit_mc_package_version", "text", "orbit_mc package version")
    m.add("WlfOrbitMcSourceSha", "artifacts/authorities.json", "/orbit_mc_source_sha256", "sha_short", "orbit_mc source hash prefix")
    m.add("WlfProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("WlfPTwoDesign", "artifacts/p2-input-authority.json", "/design_id", "text", "P2 design id")
    m.add("WlfPTwoQualification", "artifacts/p2-input-authority.json", "/qualification_status", "ident", "P2 qualification status")
    m.add("WlfPTwoManifestSha", "artifacts/p2-input-authority.json", "/manifest_file_sha256", "sha_short", "P2 manifest hash prefix")
    m.add("WlfPrimaryGrid", "artifacts/field-evidence/primary.json", "/regular_grid/radial_samples", "int", "primary map radial samples")
    m.add("WlfPrimaryGridAxial", "artifacts/field-evidence/primary.json", "/regular_grid/axial_samples", "int", "primary map axial samples")
    m.add("WlfRefinedGrid", "artifacts/field-evidence/refined.json", "/regular_grid/radial_samples", "int", "refined map radial samples")
    m.add("WlfRefinedGridAxial", "artifacts/field-evidence/refined.json", "/regular_grid/axial_samples", "int", "refined map axial samples")
    m.add("WlfPrimaryCertifiedB", "artifacts/field-evidence/primary.json", "/certificate/certified_max_b_t", "fixed4", "certified max |B|, primary map (T)")
    m.add("WlfAdapterRmsLimit", "artifacts/protocol.json", "/field_adapter/maximum_b_relative_rms", "g", "adapter relative RMS limit")
    m.add("WlfPrimaryAdapterRms", "artifacts/field-evidence/primary.json", "/field_error_report/b_relative_rms", "sci2", "primary adapter relative RMS error")
    m.add("WlfPrimaryToRefinedRms", "artifacts/field-map-convergence.json", "/primary_to_refined/b_relative_rms", "sci2", "primary→refined field RMS")
    m.add("WlfRefinedToEnlargedRms", "artifacts/field-map-convergence.json", "/refined_to_enlarged/b_relative_rms", "sci2", "refined→enlarged field RMS")

    # Gates and validators.
    m.add_derived(
        "WlfGateCount", len(gates["checks"]), "int", "number of binding gate checks",
        "len(gates.checks)", [{"artifact": "artifacts/gates.json", "pointer": "/checks"}],
    )
    m.add_derived(
        "WlfGatesTrue", sum(1 for v in gates["checks"].values() if v is True), "int",
        "binding gate checks that are true", "count(gates.checks == true)",
        [{"artifact": "artifacts/gates.json", "pointer": "/checks"}],
    )
    m.add("WlfValidatorsPassed", "artifacts/campaign-result.json", "/validators/passed", "int", "validator calls passed")
    m.add("WlfValidatorsFailed", "artifacts/campaign-result.json", "/validators/failed", "int", "validator calls failed")
    m.add("WlfAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("WlfReplayCount", "artifacts/gates.json", "/exact_authority_replay_count", "int", "exact authority replays")
    m.add("WlfGateThreshold", "artifacts/protocol.json", "/gates/maximum_successive_probability_change", "g", "successive-change gate")
    m.add("WlfEnergyGate", "artifacts/protocol.json", "/gates/maximum_relative_energy_error", "sci2", "relative energy gate")
    m.add("WlfEnergyErrorMax", "artifacts/gates.json", "/maximum_relative_energy_error", "g", "maximum relative energy error")
    m.add("WlfEnergyExceed", "artifacts/gates.json", "/orbits_exceeding_energy_gate", "int", "orbits exceeding the energy gate")
    m.add("WlfVelocityMismatch", "artifacts/gates.json", "/final_velocity_event_velocity_mismatches", "int", "final/event velocity mismatches")
    m.add("WlfWallEndpointErr", "artifacts/gates.json", "/maximum_wall_endpoint_error_m", "sci2", "maximum wall endpoint error (m)")
    m.add("WlfWallEndpointGate", "artifacts/protocol.json", "/gates/maximum_wall_endpoint_error_m", "sci2", "wall endpoint gate (m)")
    m.add("WlfManufacturedPassed", "artifacts/manufactured-gates.json", "/passed", "bool", "manufactured gates passed")
    m.add("WlfHelixOrders", "artifacts/manufactured-gates.json", "/helix_convergence/observed_orders", "list_g", "helix observed orders")
    m.add("WlfCudaParity", "artifacts/manufactured-gates.json", "/cuda_parity/maximum_relative_velocity_difference", "g", "CUDA parity max relative velocity difference")

    # Per-case estimands and the pooled (derived) counts.
    rows: list[str] = []
    wall_total = escape_total = reflected_total = trials_total = 0
    for case_id in CASES:
        block = campaign["campaigns"][case_id]
        role, policy = case_id.split("-")
        base = f"/campaigns/{case_id}"
        wall = block["wall_hit"]
        esc_ = block["escaped"]
        rows.append(
            f"{role} & {policy} & {wall['successes']} & {format_value('fixed4', wall['probability'])} & "
            f"[{format_value('fixed3', wall['lower'])}, {format_value('fixed3', wall['upper'])}] & "
            f"{esc_['successes']} & {format_value('fixed4', esc_['probability'])} & "
            f"[{format_value('fixed3', esc_['lower'])}, {format_value('fixed3', esc_['upper'])}] & "
            f"{block['reflected']['successes']} & {block['incomplete']['successes']}\\\\"
        )
        wall_total += wall["successes"]
        escape_total += esc_["successes"]
        reflected_total += block["reflected"]["successes"]
        trials_total += block["trial_count"]
        if block["reflected"]["successes"] != 0 or block["incomplete"]["successes"] != 0:
            raise ValueError(f"{case_id} records reflections or incomplete orbits")
        for key in ("wall_hit", "escaped"):
            lo, hi = _wilson(block[key]["successes"], block["trial_count"])
            if abs(lo - block[key]["lower"]) > 1e-12 or abs(hi - block[key]["upper"]) > 1e-12:
                raise ValueError(f"{case_id} {key} Wilson interval does not reproduce")
        m.add(f"Wlf{_camel(case_id)}WallP", "artifacts/campaign-result.json", f"{base}/wall_hit/probability", "fixed4", f"{case_id} wall-hit probability")
    if trials_total != campaign["orbit_count"]:
        raise ValueError("per-case trials do not sum to the orbit count")
    pooled_inputs = [
        {"artifact": "artifacts/campaign-result.json", "pointer": f"/campaigns/{c}/termination_counts"} for c in CASES
    ]
    m.add_derived("WlfPooledWall", wall_total, "int", "pooled wall hits", "sum over cases of termination_counts.wall_hit", pooled_inputs)
    m.add_derived("WlfPooledEscape", escape_total, "int", "pooled domain escapes", "sum over cases of termination_counts.domain_escape", pooled_inputs)
    m.add_derived("WlfPooledReflected", reflected_total, "int", "pooled reflections", "sum over cases of termination_counts.reflected", pooled_inputs)
    m.add_derived(
        "WlfPooledIncomplete", trials_total - wall_total - escape_total - reflected_total, "int",
        "pooled timeouts and numerical failures", "trials - wall_hit - domain_escape - reflected, summed over cases", pooled_inputs,
    )
    m.add_derived("WlfPooledTrials", trials_total, "int", "pooled trials", "sum over cases of trial_count", pooled_inputs)
    m.add_derived("WlfPooledWallP", wall_total / trials_total, "fixed3", "pooled wall-hit fraction", "WlfPooledWall / WlfPooledTrials", pooled_inputs)
    lo, hi = _wilson(wall_total, trials_total)
    m.add_derived("WlfPooledWallLo", lo, "fixed3", "pooled wall-hit Wilson lower", "wilson95(WlfPooledWall, WlfPooledTrials).lower", pooled_inputs)
    m.add_derived("WlfPooledWallHi", hi, "fixed3", "pooled wall-hit Wilson upper", "wilson95(WlfPooledWall, WlfPooledTrials).upper", pooled_inputs)
    m.add_derived("WlfPooledEscapeP", escape_total / trials_total, "fixed3", "pooled escape fraction", "WlfPooledEscape / WlfPooledTrials", pooled_inputs)
    wall_ps = [campaign["campaigns"][c]["wall_hit"]["probability"] for c in CASES]
    m.add_derived("WlfWallPMin", min(wall_ps), "fixed4", "minimum per-case wall-hit probability", "min over cases", [{"artifact": "artifacts/campaign-result.json", "pointer": f"/campaigns/{c}/wall_hit/probability"} for c in CASES])
    m.add_derived("WlfWallPMax", max(wall_ps), "fixed4", "maximum per-case wall-hit probability", "max over cases", [{"artifact": "artifacts/campaign-result.json", "pointer": f"/campaigns/{c}/wall_hit/probability"} for c in CASES])

    # Convergence.
    all_changes = [c for chain in convergence["timestep"] + convergence["cross_map"] for c in chain["successive_changes"]]
    for chain in convergence["timestep"]:
        ids = [f"{chain['map_role']}-{p}" for p in POLICIES]
        if chain["probabilities"] != [campaign["campaigns"][i]["wall_hit"]["probability"] for i in ids]:
            raise ValueError("timestep convergence probabilities differ from the campaign")
    for chain in convergence["cross_map"]:
        ids = [f"{r}-{chain['timestep_policy']}" for r in MAP_ROLES]
        if chain["probabilities"] != [campaign["campaigns"][i]["wall_hit"]["probability"] for i in ids]:
            raise ValueError("cross-map convergence probabilities differ from the campaign")
    m.add_derived(
        "WlfMaxSuccessiveChange", max(all_changes), "fixed6", "largest successive wall-hit probability change",
        "max over all timestep and cross-map successive_changes",
        [{"artifact": "artifacts/probability-convergence.json", "pointer": "/timestep"}, {"artifact": "artifacts/probability-convergence.json", "pointer": "/cross_map"}],
    )
    m.add("WlfTimestepPassed", "artifacts/probability-convergence.json", "/timestep_passed", "bool", "timestep convergence gate")
    m.add("WlfCrossMapPassed", "artifacts/probability-convergence.json", "/cross_map_passed", "bool", "cross-map convergence gate")
    if not all(all(chain["adjacent_wilson_overlap"]) for chain in convergence["timestep"] + convergence["cross_map"]):
        raise ValueError("an adjacent Wilson overlap flag is false")

    # Strata bimodality (from the per-case summaries; pooled over the nine cases).
    cell_wall = {1: 0, 2: 0, 3: 0, 4: 0}
    cell_trials = {1: 0, 2: 0, 3: 0, 4: 0}
    cell_one = {-1: [0, 0], 1: [0, 0]}
    tolerance_close = 0
    mu_all: list[float] = []
    runtime_max_b: list[float] = []
    stratum_inputs = []
    for case_id in CASES:
        summary = m.doc(f"artifacts/summaries/{case_id}.json")
        stratum_inputs.append({"artifact": f"artifacts/summaries/{case_id}.json", "pointer": "/strata"})
        if {k: v for k, v in summary["summary"].items() if k not in {"ensemble_id", "result_identity_sha256"}} != campaign["campaigns"][case_id]:
            raise ValueError(f"{case_id} summary differs from campaign-result")
        for stratum in summary["strata"]:
            cell = int(stratum["cell_id"].rsplit("-", 1)[1])
            cell_wall[cell] += stratum["termination_counts"]["wall_hit"]
            cell_trials[cell] += stratum["trials"]
            if cell == 1:
                cell_one[stratum["parallel_direction"]][0] += stratum["termination_counts"]["wall_hit"]
                cell_one[stratum["parallel_direction"]][1] += stratum["trials"]
        tolerance_close += summary["diagnostics"]["tolerance_close_event_count"]
        runtime_max_b.append(summary["diagnostics"]["runtime_max_b_t"])
        if summary["diagnostics"]["maximum_relative_energy_error"] != 0.0:
            raise ValueError(f"{case_id} energy error is not exactly zero")
    if cell_wall[2] != cell_trials[2] or cell_wall[3] != cell_trials[3] or cell_wall[4] != 0:
        raise ValueError("per-cell bimodality differs from the accepted result")
    if cell_one[-1][0] != cell_one[-1][1]:
        raise ValueError("cell-1 minus-z launches are not all wall hits")
    m.add_derived("WlfCellTwoThreeWall", cell_wall[2] + cell_wall[3], "int", "wall hits in cells 2–3 (all cases)", "sum of stratum wall_hit for cells 2 and 3", stratum_inputs)
    m.add_derived("WlfCellTwoThreeTrials", cell_trials[2] + cell_trials[3], "int", "launches in cells 2–3 (all cases)", "sum of stratum trials for cells 2 and 3", stratum_inputs)
    m.add_derived("WlfCellFourEscape", cell_trials[4] - cell_wall[4], "int", "domain escapes in cell 4 (all cases)", "sum of stratum (trials - wall_hit) for cell 4", stratum_inputs)
    m.add_derived("WlfCellFourTrials", cell_trials[4], "int", "launches in cell 4 (all cases)", "sum of stratum trials for cell 4", stratum_inputs)
    m.add_derived("WlfCellOneMinusWallP", cell_one[-1][0] / cell_one[-1][1], "pct1", "cell-1 −z wall-hit fraction", "cell-1 strata with parallel_direction=-1: wall_hit/trials", stratum_inputs)
    m.add_derived("WlfCellOnePlusWallP", cell_one[1][0] / cell_one[1][1], "pct1", "cell-1 +z wall-hit fraction", "cell-1 strata with parallel_direction=+1: wall_hit/trials", stratum_inputs)
    m.add_derived("WlfCellOnePlusWall", cell_one[1][0], "int", "cell-1 +z wall hits", "cell-1 strata with parallel_direction=+1: sum wall_hit", stratum_inputs)
    m.add_derived("WlfCellOnePlusTrials", cell_one[1][1], "int", "cell-1 +z launches", "cell-1 strata with parallel_direction=+1: sum trials", stratum_inputs)
    cell_rows = []
    for cell in (1, 2, 3, 4):
        lo, hi = _wilson(cell_wall[cell], cell_trials[cell])
        z_mm = format_value("mm1", protocol["launches"]["position_seeds"][2 * (cell - 1)]["position_m"][2])
        cell_rows.append(
            f"{cell} & {z_mm} & {cell_trials[cell]} & {cell_wall[cell]} & "
            f"{format_value('fixed3', cell_wall[cell] / cell_trials[cell])} & "
            f"[{format_value('fixed3', lo)}, {format_value('fixed3', hi)}] & {cell_trials[cell] - cell_wall[cell]}\\\\"
        )

    # Diagnostics (not gates).
    mu = gates["diagnostics_not_gates"]["magnetic_moment_variation"]
    if mu["binding"] is not False or mu["role"] != "diagnostic_only":
        raise ValueError("mu diagnostic is presented as a gate")
    m.add("WlfMuMin", "artifacts/gates.json", "/diagnostics_not_gates/magnetic_moment_variation/min", "fixed4", "mu variation minimum")
    m.add("WlfMuMedian", "artifacts/gates.json", "/diagnostics_not_gates/magnetic_moment_variation/median", "fixed4", "mu variation median")
    m.add("WlfMuMax", "artifacts/gates.json", "/diagnostics_not_gates/magnetic_moment_variation/max", "fixed4", "mu variation maximum")
    m.add("WlfMuAboveTenth", "artifacts/gates.json", "/diagnostics_not_gates/magnetic_moment_variation/count_above_0p1", "int", "orbits with mu variation > 0.1")
    m.add("WlfMuAboveHalf", "artifacts/gates.json", "/diagnostics_not_gates/magnetic_moment_variation/count_above_0p5", "int", "orbits with mu variation > 0.5")
    m.add("WlfMuRole", "artifacts/gates.json", "/diagnostics_not_gates/magnetic_moment_variation/role", "ident", "mu diagnostic role")
    m.add_derived(
        "WlfMuAboveTenthP", mu["count_above_0p1"] / mu["orbit_count_with_mu"], "pct1",
        "share of orbits with mu variation > 0.1", "count_above_0p1 / orbit_count_with_mu",
        [{"artifact": "artifacts/gates.json", "pointer": "/diagnostics_not_gates/magnetic_moment_variation"}],
    )
    m.add_derived(
        "WlfToleranceCloseShare", tolerance_close / campaign["orbit_count"], "pct1",
        "share of tolerance-close terminations", "sum(summaries.diagnostics.tolerance_close_event_count) / orbit_count",
        [{"artifact": f"artifacts/summaries/{c}.json", "pointer": "/diagnostics/tolerance_close_event_count"} for c in CASES],
    )
    m.add_derived("WlfToleranceCloseCount", tolerance_close, "int", "tolerance-close terminations", "sum(summaries.diagnostics.tolerance_close_event_count)", [{"artifact": f"artifacts/summaries/{c}.json", "pointer": "/diagnostics/tolerance_close_event_count"} for c in CASES])
    m.add_derived("WlfRuntimeBMin", min(runtime_max_b), "fixed4", "minimum runtime max |B| (T)", "min(summaries.diagnostics.runtime_max_b_t)", [{"artifact": f"artifacts/summaries/{c}.json", "pointer": "/diagnostics/runtime_max_b_t"} for c in CASES])
    m.add_derived("WlfRuntimeBMax", max(runtime_max_b), "fixed4", "maximum runtime max |B| (T)", "max(summaries.diagnostics.runtime_max_b_t)", [{"artifact": f"artifacts/summaries/{c}.json", "pointer": "/diagnostics/runtime_max_b_t"} for c in CASES])
    m.add("WlfIntegrationWallS", "artifacts/campaign-result.json", "/execution_mode/integration_wall_s", "fixed1", "integration wall time (s)")
    m.add("WlfWorkers", "artifacts/campaign-result.json", "/execution_mode/worker_pool_size", "int", "parallel case workers")
    first = m.doc("transitions/0001-lock-acquired.json")
    last = m.doc("transitions/0009-terminal.json")
    m.add_derived(
        "WlfLifecycleWallS", _seconds(first["recorded_at_utc"]["value"], last["recorded_at_utc"]["value"]), "fixed1",
        "lock-acquired to terminal wall time (s)", "terminal.recorded_at_utc - lock_acquired.recorded_at_utc",
        [{"artifact": "transitions/0001-lock-acquired.json", "pointer": "/recorded_at_utc/value"}, {"artifact": "transitions/0009-terminal.json", "pointer": "/recorded_at_utc/value"}],
    )

    # Claim boundary and lineage.
    m.add("WlfCouplingStatus", "artifacts/coupling-export-only.json", "/integration_status", "ident", "coupling integration status")
    m.add("WlfCouplingQuantity", "artifacts/coupling-export-only.json", "/quantity", "ident", "coupling quantity")
    m.add("WlfCouplingP", "artifacts/coupling-export-only.json", "/probability", "fixed4", "exported wall-loss probability")
    m.add("WlfCouplingU", "artifacts/coupling-export-only.json", "/standard_uncertainty", "fixed4", "exported standard uncertainty")
    m.add("WlfForbidPic", "artifacts/campaign-result.json", "/limitations/forbid_pic_or_self_consistent_claim", "bool", "PIC/self-consistent claims forbidden")
    m.add("WlfForbidMirror", "artifacts/campaign-result.json", "/limitations/forbid_mirror_formula_publication", "bool", "mirror-formula publication forbidden")
    m.add("WlfForbidPerformance", "artifacts/campaign-result.json", "/limitations/forbid_plasma_performance_publication", "bool", "plasma performance publication forbidden")
    m.add("WlfHardwareValidation", "artifacts/campaign-result.json", "/limitations/hardware_or_experimental_validation", "bool", "hardware/experimental validation claimed")
    m.add("WlfVOneState", "artifacts/protocol.json", "/prior_campaign_disclosure/v1/terminal_state", "ident", "v1 terminal state")
    m.add("WlfVOneError", "artifacts/protocol.json", "/prior_campaign_disclosure/v1/primary_error_message", "text", "v1 error message")
    m.add("WlfVTwoState", "artifacts/protocol.json", "/prior_campaign_disclosure/v2/terminal_state", "ident", "v2 terminal state")
    m.add("WlfVTwoError", "artifacts/protocol.json", "/prior_campaign_disclosure/v2/primary_error_message", "text", "v2 error message")
    m.add("WlfVThreeState", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/terminal_state", "ident", "v3 terminal state")
    m.add("WlfVThreeError", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/primary_error_message", "text", "v3 error message")
    m.add("WlfShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("WlfShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add("WlfShakedownLaunches", "artifacts/shakedown.json", "/disjointness/shakedown_launch_count", "int", "shakedown launches")
    m.add("WlfShakedownDisjoint", "artifacts/shakedown.json", "/disjointness/proven", "bool", "shakedown disjointness proven")
    m.add("WlfBaseCommit", "artifacts/protocol.json", "/authority/base_commit", "sha_short", "orbit_mc/base commit prefix")
    m.add("WlfPTwoCommit", "artifacts/protocol.json", "/authority/p2_evidence_commit", "sha_short", "P2 evidence commit prefix")
    m.add("WlfPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    if m.doc("execution-lock.json")["commit"] != PREREGISTRATION_COMMIT_SHA:
        raise ValueError("execution lock commit differs from the preregistration commit")
    m.add_derived(
        "WlfResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix",
        "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)",
        [{"artifact": "manifest.json", "pointer": ""}],
    )
    m.add("WlfIntervalMethod", "artifacts/campaign-result.json", "/campaigns/primary-N/wall_hit/method", "text", "interval method identifier")
    if any(campaign["campaigns"][c][k]["method"] != "wilson-95" for c in CASES for k in ("wall_hit", "escaped", "reflected", "incomplete")):
        raise ValueError("interval method is not wilson-95 for every estimand")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"]:
        raise ValueError("shakedown artifact differs from the bound authority")
    if coupling["integration_status"] != "export_only_pending_consumer_integration":
        raise ValueError("coupling status is not export-only")

    # Generated TeX.
    lines = [
        "% Generated by paper/scripts/generate_wall_loss_v4_evidence.py; do not hand edit.",
        f"% Evidence: {RESULTS.as_posix()} at commit {RESULTS_COMMIT_SHA} (manifest SHA-256 {bundle.manifest_sha256}).",
        "% Every macro value traces to an artifact path and JSON pointer recorded in paper/evidence/wall-loss-v4.json.",
    ]
    for item in m.items:
        lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    lines.append("\\newcommand{\\WlfCaseTable}{%")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Termination probabilities of the nine preregistered cases (map role $\\times$ timestep policy); "
        "\\WlfLaunchesPerCase{} equal-weight launches per case; Wilson 95\\% intervals as sealed by orbit\\_mc. "
        "Reflection and timeout counts are exact zeros.}"
    )
    lines.append("\\label{tab:wall-loss-v4-cases}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{llrrlrrlrr}")
    lines.append("\\toprule")
    lines.append("map & $\\Delta t$ & wall & $p$ & 95\\% & escape & $p$ & 95\\% & refl. & timeout\\\\")
    lines.append("\\midrule")
    lines.extend(rows)
    lines.append("\\midrule")
    lines.append(
        f"pooled & --- & \\WlfPooledWall{{}} & \\WlfPooledWallP{{}} & [\\WlfPooledWallLo{{}}, \\WlfPooledWallHi{{}}] & "
        f"\\WlfPooledEscape{{}} & \\WlfPooledEscapeP{{}} & --- & \\WlfPooledReflected{{}} & \\WlfPooledIncomplete{{}}\\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}")
    lines.append("\\newcommand{\\WlfCellTable}{%")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Wall-hit fraction by launch cell, pooled over the nine cases (equal weights). "
        "Cells 2--3 are saturated at total wall loss; cell 4 lies beyond the dielectric end and escapes entirely; "
        "cell 1 depends on the launch direction.}"
    )
    lines.append("\\label{tab:wall-loss-v4-cells}")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{rrrrrlr}")
    lines.append("\\toprule")
    lines.append("cell & $z$ (mm) & launches & wall & fraction & 95\\% & escape\\\\")
    lines.append("\\midrule")
    lines.extend(cell_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}")
    tex = "\n".join(lines) + "\n"

    evidence = {
        "document_type": "paper-wall-loss-v4-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "manuscript_integration": {
            "status": "draft-section-not-wired-into-manuscript",
            "section_path": SECTION_PATH.as_posix(),
            "generated_tex_path": OUTPUT_PATH.as_posix(),
            "reason": (
                "manuscript.tex pins its evidence boundary to the L0 evidence revision and admits "
                "results only through claims.json/result-gates.json; this section can be \\input "
                "only after a claim-matrix entry and gate for the wall-loss campaign are accepted."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_crlf_sidecars": bundle.tolerated,
            "tolerance_rule": "sha256(bytes.replace(LF, CRLF)) == recorded byte_sha256; every other mismatch fails",
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": {
            "WlfCaseTable": {"rows": len(rows), "source": "artifacts/campaign-result.json#/campaigns"},
            "WlfCellTable": {"rows": len(cell_rows), "source": "artifacts/summaries/<case>.json#/strata"},
        },
        "generator": {
            "path": "paper/scripts/generate_wall_loss_v4_evidence.py",
            "sha256": sha256_bytes((repo / "paper/scripts/generate_wall_loss_v4_evidence.py").read_bytes().replace(b"\r\n", b"\n")),
            "command": "python paper/scripts/generate_wall_loss_v4_evidence.py",
        },
        "output": {"path": OUTPUT_PATH.as_posix(), "sha256": sha256_bytes(tex.encode("utf-8"))},
    }
    if len({item["name"] for item in m.items}) != len(m.items):
        raise ValueError("duplicate macro names")
    return evidence, tex


def _camel(case_id: str) -> str:
    role, policy = case_id.split("-")
    return role.capitalize() + {"N": "N", "2N": "TwoN", "4N": "FourN"}[policy]


def render(repo: Path) -> tuple[bytes, bytes, bytes]:
    evidence, tex = build(repo)
    tex_bytes = tex.encode("utf-8")
    build_config = json.loads((repo / "paper/build-config.json").read_text("utf-8"))
    sidecar = {
        "document_type": "paper-generated-artifact-provenance",
        "schema_version": "1.0",
        "artifact_id": "TAB-WALL-LOSS-V4",
        "claim_ids": [],
        "claim_status": "draft; no claims.json entry authorizes manuscript prose yet",
        "evidence_revision": RESULTS_COMMIT_SHA,
        "source_date_epoch": build_config["source_date_epoch"],
        "generator": evidence["generator"],
        "manifest": {
            "path": EVIDENCE_PATH.as_posix(),
            "sha256": sha256_bytes(canonical_json(evidence)),
            "manifest_id": "WALL-LOSS-V4-20260902-4608-V1",
        },
        "inputs": [
            {"path": (RESULTS / path).as_posix(), "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["artifacts"].items()
        ],
        "bundle_manifest": {
            "path": evidence["bundle"]["manifest_path"],
            "sha256": evidence["bundle"]["manifest_sha256"],
            "git_blob": evidence["binding"]["manifest_git_blob"],
        },
        "output": {"path": OUTPUT_PATH.as_posix(), "sha256": sha256_bytes(tex_bytes)},
    }
    return canonical_json(evidence), tex_bytes, canonical_json(sidecar)


def write_generated(repo: Path) -> None:
    evidence, tex, sidecar = render(repo)
    (repo / EVIDENCE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / EVIDENCE_PATH).write_bytes(evidence)
    (repo / OUTPUT_PATH).write_bytes(tex)
    (repo / SIDECAR_PATH).write_bytes(sidecar)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    try:
        write_generated(repo)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Wall-loss v4 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
