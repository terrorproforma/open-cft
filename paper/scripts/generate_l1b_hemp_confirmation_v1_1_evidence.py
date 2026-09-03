"""Generate hash-bound paper evidence for the L1b/P2 material-aware HEMP confirmation v1.1.

Reads the sealed results bundle of ``modern/experiments/l1b_hemp_confirmation_v1_1``
(every manifest file verified byte for byte; no end-of-line tolerance is needed or
granted), binds it to the committed results revision, recomputes the experiment-code,
dependency-source and field-pipeline hashes the bundle sealed from the blobs committed
at the preregistration revision, re-derives the verdict, both confirmation gates, the
reported HEMP-like preservation and every headline statistic from the per-design rows
and their matched-cusp pairs, cross-checks every design record, sampled P2 field grid,
agreement-table row and CSV row against its dataset row, byte-verifies the recorded
development rejection of the predecessor campaign (v1) and verifies its post-hoc
rejection note against that bundle, proves that the v1 -> v1.1 protocol differs only in
the disclosed declarations, cross-checks the committed results dashboard against both
bundles, and writes:

* ``paper/evidence/l1b-hemp-confirmation-v1-1.json`` -- every macro value with the
  artifact path, JSON pointer, formatter and artifact SHA-256 it was read from, or the
  derivation and inputs of a derived macro;
* ``paper/generated/l1b-hemp-confirmation-v1-1.tex`` -- ``\\newcommand`` macros and
  four generated tables (each wrapped in ``\\ArtifactClaim``) for the admitted results
  subsection ``paper/sections/l1b-hemp-confirmation-v1-1.tex``;
* ``paper/generated/l1b-hemp-confirmation-v1-1.provenance.json`` -- generator/input/
  output hashes in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path enters
any output.  The study is a material-aware confirmation: the adaptive quadratic
finite-element magnetostatic field with LINEAR soft-iron poles and yoke and
recoil-remanence magnets (two nested levels; no saturation, no B-H curve) is solved for
the fifteen HEMP-like designs of the accepted L1a geometry sweep v3, the literature cusp
definition of the accepted topology screening is applied to it verbatim, and the wall-cusp
count and positions are compared with the sealed L1a records under predeclared
tolerances.  The verdict is admitted exactly as recorded (CONFIRMED under the frozen
rule); the wall-field and Koch-ratio ratios are field ratios of two field models, never
probabilities, and nothing here is a saturation, plasma, mirror-probability, thrust,
efficiency or design-recommendation claim.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from generate_cusp_topology_v3_1_evidence import _bound_file
from generate_mdo_l0_v1_evidence import (
    FORMATTERS as _BASE_FORMATTERS,
    _git,
    _lf,
    _sci,
    _tex_escape,
    canonical_json,
    dashboard_payload,
    load_json_bytes,
    resolve_pointer,
    sha256_bytes,
)

EXPERIMENT = Path("modern/experiments/l1b_hemp_confirmation_v1_1")
RESULTS = EXPERIMENT / "results"
V1_EXPERIMENT = Path("modern/experiments/l1b_hemp_confirmation_v1")
V1_RESULTS = V1_EXPERIMENT / "results"
EVIDENCE_PATH = Path("paper/evidence/l1b-hemp-confirmation-v1-1.json")
OUTPUT_PATH = Path("paper/generated/l1b-hemp-confirmation-v1-1.tex")
SIDECAR_PATH = Path("paper/generated/l1b-hemp-confirmation-v1-1.provenance.json")
SECTION_PATH = Path("paper/sections/l1b-hemp-confirmation-v1-1.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_l1b_hemp_confirmation_v1_dashboard.py")
DASHBOARD_TEMPLATE = Path("modern/visualization/l1b-hemp-confirmation-v1.template.html")
DASHBOARD_HTML = Path("modern/visualization/l1b-hemp-confirmation-v1.html")

# Revisions on feat/sota-foundation.  The campaign chain was developed on the pushed branch
# origin/exp/l1b-hemp-confirmation-v1 and rebased onto feat/sota-foundation before this
# admission; the bundle's execution lock and the protocol's predecessor block therefore
# carry the pre-rebase commit identifiers of that branch.  The rebased commits below are
# the ones reachable from HEAD; the experiment-code, dependency-source and field-pipeline
# hashes the bundle sealed are recomputed from the blobs at the rebased preregistration
# commit, which proves that it carries exactly the code that produced the bundle.
RESULTS_COMMIT_SHA = "54cd3e82b7c879110cb7242c5f6210d1ac59fc92"
PREREGISTRATION_COMMIT_SHA = "c8692ff2cb6d4b8605fc00176f919b830ff9e685"
CODE_COMMIT_SHA = "b6125fe7a3c4f310ff564c0102c1617ccc68b00a"  # v1.1 code + tests + the v1 post-hoc rejection note
DASHBOARD_COMMIT_SHA = "560909f77ad64ff0215437ecd85e12bee6e8d4ea"
V1_RESULTS_COMMIT_SHA = "2d8d670593f11045601f911acd1fa11ccf9bda21"
V1_PREREGISTRATION_COMMIT_SHA = "fb143eb2280abafc1bce3910f4311f8fdff5fd7f"
V1_CODE_COMMIT_SHA = "6e9f056cc01bb00d6c9cdba3f98b997ee01b4dcc"
# References bound at their own admitted revisions.
SWEEP_V3_RESULTS_COMMIT_SHA = "2cfe8223630fbef6bfe8099a5dcecaf4eb8c6b44"
CUSP_TOPOLOGY_RESULTS_COMMIT_SHA = "cec47f12f5909c5886424bf5d46ac20ce06f1ac5"

SWEEP_V3_CATALOGUE_PATH = Path("modern/experiments/l1a_geometry_sweep_v3/results/artifacts/cusp-cell-catalogue-v3.json")
SWEEP_V3_MANIFEST_PATH = Path("modern/experiments/l1a_geometry_sweep_v3/results/manifest.json")
SWEEP_V3_DESIGN_AUTHORITIES_PATH = Path("modern/experiments/l1a_geometry_sweep_v3/design-authorities.json")
CUSP_TOPOLOGY_PROTOCOL_PATH = Path("modern/experiments/cusp_topology_search_v3_1/protocol.json")
V1_REJECTION_PATH = V1_EXPERIMENT / "POSTHOC_REJECTION.md"

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "L1B-HEMP-CONFIRMATION-V1-1-20260904-15-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/l1b-hemp-confirmation-v1-1.json")
GATE_ID = "GATE-L1B-HEMP-CONFIRMATION-V1-1"
GATE_KIND = "numerical-screening"
RECORDED_OUTCOME = "accepted-material-aware-confirmation"
ARTIFACT_ID = "TAB-L1B-HEMP-CONFIRMATION-V1-1"
ARTIFACT_CLAIM_ID = "CLM-088"
PROSE_CLAIM_IDS = ("CLM-086", "CLM-087", "CLM-089", "CLM-090", "CLM-091", "CLM-092", "CLM-093")
SECTION_BINDING = "\\input{sections/l1b-hemp-confirmation-v1-1.tex}"
GENERATED_BINDING = "\\input{generated/l1b-hemp-confirmation-v1-1.tex}"
SECTION_HEADING = "Material-aware confirmation of the HEMP-like designs under a linear-iron P2 field"
TABLE_MACROS = ("HmcDesignTable", "HmcGateTable", "HmcFieldTable", "HmcDisclosureTable")
REVISION_MACRO = "HempConfirmationEvidenceRevision"
MACRO_PREFIX = "Hmc"

EXPERIMENT_ID = "l1b-hemp-confirmation-v1-1"
V1_EXPERIMENT_ID = "l1b-hemp-confirmation-v1"
CLASSIFICATION = "P2_MATERIAL_AWARE_FIELD_CONFIRMATION_NOT_HARDWARE_VALID"
TOPOLOGY_LABEL = "SCREENING_P2_MATERIAL_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
CAMPAIGN_STATUS = "accepted_l1b_confirmation_confirmed"
VERDICT = "CONFIRMED"
V1_TERMINAL_STATE = "development_rejection"
SCREENING_MODEL = (
    "adaptive quadratic finite-element magnetostatics (cft_revival.fem_reference, numpy CPU) with LINEAR "
    "soft-iron poles and return yoke and recoil-remanence SmCo-like magnets on two nested levels, sampled into "
    "the cusp topology search v3.1 definition and compared with the sealed linear-vacuum L1a records of the "
    "fifteen HEMP-like sweep-v3 designs (a confirmation of field-map descriptors under two field models; not the "
    "three-level P2-qualified chain, no saturation or B-H curve, not hardware-valid)"
)
FROZEN_FILES = ("protocol.json", "authorities.json", "shakedown.json", "design-authorities.json")
SET_ID = "hemp_like_v3"
DESIGN_ID_PREFIX = "l1a-gs-v3-"
FLOAT_TOLERANCE = 1e-9
# Declared v1 -> v1.1 protocol changes (paths whose values may differ; everything else must be identical).
ALLOWED_PROTOCOL_CHANGES = (
    "execution/git_common_lock",
    "experiment_id",
    "p2/mesh/angle_gate_disclosure",
    "p2/mesh/reject_below_angle_deg",
    "p2/mesh/whole_set_preflight",
    "predecessor",
    "schema_version",
    "shakedown/design_rule",
    "shakedown/designs/hemp_like_v3",
    "shakedown/prepare_requires",
    "shakedown/whole_set_mesh_preflight",
    "title",
)
DECLARATIONS_CHANGED = ("p2/mesh/reject_below_angle_deg", "shakedown/whole_set_mesh_preflight")
SLIVER_THRESHOLD_DEG = 10.0
BINDING_GATE_NAMES = (
    "all_declared_designs_resolved",
    "axis_window_reproduced",
    "determinism_replay",
    "every_null_converged",
    "every_trace_terminates_cleanly",
    "every_wall_trace_flux_consistent",
    "hash_bindings",
    "identity_proven",
    "ram_policy_respected",
    "sampling_stability",
    "solver_converged_all_levels",
)
TRANSITION_NAMES = (
    (1, "lock-acquired"), (2, "cache-prepared"), (3, "prebundle-started"), (4, "prebundle-completed"),
    (5, "development-started"), (6, "development-accepted"), (7, "assessment-started"), (8, "assessment-accepted"),
    (9, "terminal"),
)
V1_TRANSITION_NAMES = (
    (1, "lock-acquired"), (2, "cache-prepared"), (3, "prebundle-started"), (4, "prebundle-completed"),
    (5, "development-started"), (6, "development-rejected"), (7, "terminal"),
)
COUNT_TOKENS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _pct_of_ratio(value: float, digits: int) -> str:
    """Render a ratio as a signed percentage change (1.23 -> +23%)."""

    text = f"{100.0 * (float(value) - 1.0):+.{digits}f}"
    return text.replace("-", "$-$").replace("+", "$+$") + "\\%"


FORMATTERS: dict[str, Callable[[Any], str]] = {
    **_BASE_FORMATTERS,
    "mm1": lambda v: f"{1e3 * float(v):.1f}",
    "mm2": lambda v: f"{1e3 * float(v):.2f}",
    "mm3": lambda v: f"{1e3 * float(v):.3f}",
    "um0": lambda v: f"{1e6 * float(v):.0f}",
    "um1": lambda v: f"{1e6 * float(v):.1f}",
    "nm0": lambda v: f"{1e9 * float(v):.0f}",
    "pct0": lambda v: f"{100.0 * float(v):.0f}\\%",
    "pct2": lambda v: f"{100.0 * float(v):.2f}\\%",
    "ratio_pct0": lambda v: _pct_of_ratio(float(v), 0),
    "deg1": lambda v: f"{float(v):.1f}",
    "gb1": lambda v: f"{float(v) / 1e9:.1f}",
    "mb0": lambda v: f"{float(v) / 1e6:.0f}",
    "sec0": lambda v: f"{float(v):.0f}",
    "sci3": lambda v: _sci(float(v), 3),
    "symbol": lambda v: str(v),
    "list_ident_tt": lambda v: ", ".join(f"\\texttt{{{_BASE_FORMATTERS['ident'](x)}}}" for x in v),
    "list_clauses": lambda v: "; ".join(_tex_escape(str(x)) for x in v),
    "list_short_designs": lambda v: ", ".join(_short(x) for x in v),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def _utc(value: dict[str, Any]) -> datetime:
    if value.get("__cft_type__") != "aware-utc-datetime":
        raise ValueError("timestamp is not an aware UTC datetime record")
    return datetime.fromisoformat(str(value["value"]).replace("Z", "+00:00"))


def _short(design_id: str) -> str:
    """The three-digit sweep ordinal of a design identifier (l1a-gs-v3-028-f012c0bf33 -> 028)."""

    if not design_id.startswith(DESIGN_ID_PREFIX):
        raise ValueError(f"design id {design_id!r} is not a sweep-v3 identifier")
    return design_id[len(DESIGN_ID_PREFIX):len(DESIGN_ID_PREFIX) + 3]


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values]
    if not clean:
        raise ValueError("empty distribution")
    return {"count": len(clean), "max": max(clean), "mean": statistics.fmean(clean), "median": statistics.median(clean), "min": min(clean)}


def _check_distribution(recorded: dict[str, Any], values: list[float], label: str) -> None:
    computed = _distribution(values)
    if recorded["count"] != computed["count"]:
        raise ValueError(f"{label}: distribution count differs")
    for key in ("max", "mean", "median", "min"):
        if not _close(recorded[key], computed[key]):
            raise ValueError(f"{label}: distribution {key} does not recompute")


def _histogram(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: int(pair[0])))


def _semantic_sha256(value: Any) -> str:
    """The experiment runtime's semantic hash of a plain JSON document (canonical compact form)."""

    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Paths at which two JSON documents differ (nested objects compared key by key)."""

    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{prefix}{key}")
            else:
                out.extend(_diff_paths(a[key], b[key], f"{prefix}{key}/"))
        return out
    return [] if a == b else [prefix.rstrip("/")]


# --------------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------------- #
class Bundle:
    """A sealed results bundle, verified file by file against its own manifest."""

    def __init__(self, repo: Path, results: Path, experiment_id: str, state: str) -> None:
        self.repo = repo
        self.results = results
        self.root = repo / results
        manifest_raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.manifest = load_json_bytes(manifest_raw, "results manifest")
        if self.manifest.get("state") != state:
            raise ValueError(f"results manifest state is not {state}")
        if self.manifest.get("experiment_id") != experiment_id:
            raise ValueError("results manifest experiment identity differs")
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        entries = self.manifest["artifacts"]
        if len(entries) != self.manifest["artifact_count"]:
            raise ValueError("results manifest artifact count differs")
        self.directory_count = 0
        for entry in entries:
            if entry["type"] != "file":
                self.directory_count += 1
                continue
            relative = entry["path"]
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"manifest path escapes the bundle: {relative}")
            raw = (self.root / pure).read_bytes()
            actual = sha256_bytes(raw)
            if actual != entry["byte_sha256"] or len(raw) != entry["bytes"]:
                raise ValueError(f"bundle file SHA-256 or size mismatch: {relative}")
            self.hashes[relative] = actual
            self.sizes[relative] = len(raw)
        on_disk = sorted(
            path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file() and path.name != "manifest.json"
        )
        if on_disk != sorted(self.hashes):
            raise ValueError("the results tree carries files the manifest does not bind (or lacks bound files)")
        if self.hashes["terminal.json"] != self.manifest["terminal_byte_sha256"]:
            raise ValueError("terminal.json hash differs from the manifest binding")
        if self.hashes["execution-lock.json"] != self.manifest["lock_byte_sha256"]:
            raise ValueError("execution-lock.json hash differs from the manifest binding")
        for relative in list(self.hashes):
            if relative.endswith(".sha256.json") or relative == "execution-lock.json":
                continue
            sidecar_rel = f"{relative}.sha256.json"
            if sidecar_rel not in self.hashes:
                raise ValueError(f"artifact without manifest-bound sidecar: {relative}")
            sidecar = load_json_bytes((self.root / sidecar_rel).read_bytes(), sidecar_rel)
            if sidecar["artifact"] != relative or sidecar["byte_sha256"] != self.hashes[relative]:
                raise ValueError(f"sidecar disagrees with the manifest: {sidecar_rel}")
            if sidecar["bytes"] != self.sizes[relative]:
                raise ValueError(f"sidecar size disagrees with the manifest: {sidecar_rel}")
        self.used: dict[str, dict[str, Any]] = {}

    def raw(self, relative: str) -> bytes:
        if relative not in self.hashes:
            raise ValueError(f"{relative} is not manifest-bound")
        self.used[relative] = {"sha256": self.hashes[relative], "bytes": self.sizes[relative]}
        return (self.root / relative).read_bytes()

    def load(self, relative: str) -> Any:
        return load_json_bytes(self.raw(relative), relative)

    def load_gz(self, relative: str, expected_payload_sha256: str) -> Any:
        payload = gzip.decompress(self.raw(relative))
        if sha256_bytes(payload) != expected_payload_sha256:
            raise ValueError(f"{relative}: payload hash differs from the record binding")
        return load_json_bytes(payload, relative)


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo, check=False, capture_output=True,
    ).returncode == 0


def _committed_source_hash(repo: Path, revision: str, names: list[str], prefix: str) -> str:
    """The experiment's source-binding hash (name, NUL, bytes, NUL per file) over blobs at a revision."""

    request = "".join(f"{revision}:{prefix}{name}\n" for name in names).encode("utf-8")
    output = subprocess.run(["git", "cat-file", "--batch"], cwd=repo, check=True, capture_output=True, input=request).stdout
    digest = hashlib.sha256()
    cursor = 0
    for name in names:
        header_end = output.index(b"\n", cursor)
        header = output[cursor:header_end].split(b" ")
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"committed source {name} is missing at {revision[:8]}")
        size = int(header[2])
        data = output[header_end + 1:header_end + 1 + size]
        cursor = header_end + 1 + size + 1
        if b"\r" in data:
            raise ValueError(f"committed source {name} contains CR bytes")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def bind_committed(repo: Path, bundle: Bundle, v1_bundle: Bundle, source_binding: dict[str, Any], authorities: dict[str, Any]) -> dict[str, Any]:
    """Prove the working-tree bundles equal their committed revisions and the chain of commits holds."""

    head = _git(repo, "rev-parse", "HEAD")
    for commit, label in (
        (RESULTS_COMMIT_SHA, "results"),
        (PREREGISTRATION_COMMIT_SHA, "preregistration"),
        (CODE_COMMIT_SHA, "v1.1 code and v1 post-hoc note"),
        (DASHBOARD_COMMIT_SHA, "dashboard"),
        (V1_RESULTS_COMMIT_SHA, "v1 results"),
        (V1_PREREGISTRATION_COMMIT_SHA, "v1 preregistration"),
        (V1_CODE_COMMIT_SHA, "v1 code"),
        (SWEEP_V3_RESULTS_COMMIT_SHA, "sweep v3 results"),
        (CUSP_TOPOLOGY_RESULTS_COMMIT_SHA, "cusp topology v3.1 results"),
    ):
        if not _is_ancestor(repo, commit, head):
            raise ValueError(f"{label} commit is not an ancestor of HEAD")
    for earlier, later, label in (
        (V1_CODE_COMMIT_SHA, V1_PREREGISTRATION_COMMIT_SHA, "v1 code -> v1 preregistration"),
        (V1_PREREGISTRATION_COMMIT_SHA, V1_RESULTS_COMMIT_SHA, "v1 preregistration -> v1 results"),
        (V1_RESULTS_COMMIT_SHA, CODE_COMMIT_SHA, "v1 results -> v1.1 code"),
        (CODE_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "v1.1 code -> preregistration"),
        (PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA, "preregistration -> results"),
        (RESULTS_COMMIT_SHA, DASHBOARD_COMMIT_SHA, "results -> dashboard"),
        (SWEEP_V3_RESULTS_COMMIT_SHA, V1_CODE_COMMIT_SHA, "sweep v3 results -> v1 code"),
        (CUSP_TOPOLOGY_RESULTS_COMMIT_SHA, SWEEP_V3_RESULTS_COMMIT_SHA, "cusp topology v3.1 results -> sweep v3 results"),
    ):
        if not _is_ancestor(repo, earlier, later) or earlier == later:
            raise ValueError(f"revision chain {label} does not hold strictly")
    manifest_rel = (RESULTS / "manifest.json").as_posix()
    committed_blob = _git(repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{manifest_rel}")
    working_blob = _git(repo, "hash-object", "--", manifest_rel)
    if committed_blob != working_blob:
        raise ValueError("working-tree results manifest differs from the committed blob")
    results_tree = _git(repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{RESULTS.as_posix()}")
    head_tree = _git(repo, "rev-parse", f"HEAD:{RESULTS.as_posix()}")
    if results_tree != head_tree:
        raise ValueError("results tree changed after the results revision")
    v1_manifest_rel = (V1_RESULTS / "manifest.json").as_posix()
    v1_committed_blob = _git(repo, "rev-parse", f"{V1_RESULTS_COMMIT_SHA}:{v1_manifest_rel}")
    if v1_committed_blob != _git(repo, "hash-object", "--", v1_manifest_rel):
        raise ValueError("working-tree v1 results manifest differs from the committed blob")
    v1_results_tree = _git(repo, "rev-parse", f"{V1_RESULTS_COMMIT_SHA}:{V1_RESULTS.as_posix()}")
    if v1_results_tree != _git(repo, "rev-parse", f"HEAD:{V1_RESULTS.as_posix()}"):
        raise ValueError("v1 results tree changed after its record revision")
    for name in FROZEN_FILES:
        relative = (EXPERIMENT / name).as_posix()
        frozen = _git(repo, "rev-parse", f"{PREREGISTRATION_COMMIT_SHA}:{relative}")
        recorded = _git(repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{relative}")
        working = _git(repo, "hash-object", "--", relative)
        if not frozen == recorded == working:
            raise ValueError(f"frozen {name} differs between preregistration, results and the working tree")
    for name in FROZEN_FILES:
        relative = (V1_EXPERIMENT / name).as_posix()
        frozen = _git(repo, "rev-parse", f"{V1_PREREGISTRATION_COMMIT_SHA}:{relative}")
        recorded = _git(repo, "rev-parse", f"{V1_RESULTS_COMMIT_SHA}:{relative}")
        working = _git(repo, "hash-object", "--", relative)
        if not frozen == recorded == working:
            raise ValueError(f"v1 frozen {name} differs between preregistration, results and the working tree")
    # Both record commits carry only their results trees.
    for commit, results, label in ((RESULTS_COMMIT_SHA, RESULTS, "results"), (V1_RESULTS_COMMIT_SHA, V1_RESULTS, "v1 results")):
        files = _git(repo, "diff", "--name-only", f"{commit}~1", commit).split()
        if not files or any(not path.startswith(results.as_posix() + "/") for path in files):
            raise ValueError(f"the {label} commit changes files outside its results tree")
    results_commit_files = len(_git(repo, "diff", "--name-only", f"{RESULTS_COMMIT_SHA}~1", RESULTS_COMMIT_SHA).split())
    v1_results_commit_files = len(_git(repo, "diff", "--name-only", f"{V1_RESULTS_COMMIT_SHA}~1", V1_RESULTS_COMMIT_SHA).split())
    if results_commit_files != len(bundle.hashes) + 1 or v1_results_commit_files != len(v1_bundle.hashes) + 1:
        raise ValueError("a record commit does not carry exactly its bundle's files plus the manifest")
    # The preregistration commits freeze the two authority files and nothing else.
    for commit, experiment, label in ((PREREGISTRATION_COMMIT_SHA, EXPERIMENT, "preregistration"), (V1_PREREGISTRATION_COMMIT_SHA, V1_EXPERIMENT, "v1 preregistration")):
        files = sorted(_git(repo, "diff", "--name-only", f"{commit}~1", commit).split())
        if files != sorted((experiment / name).as_posix() for name in ("authorities.json", "design-authorities.json")):
            raise ValueError(f"the {label} commit does not freeze exactly the two authority files")
    # The dashboard commit changes no code.
    dashboard_files = _git(repo, "diff", "--name-only", f"{DASHBOARD_COMMIT_SHA}~1", DASHBOARD_COMMIT_SHA).split()
    if not dashboard_files or any(path.endswith((".py", ".json")) for path in dashboard_files):
        raise ValueError("the dashboard commit changes a code or JSON file")
    # The sealed source-binding hashes recompute from the blobs at the rebased preregistration commit.
    experiment_prefix = EXPERIMENT.as_posix() + "/"
    recomputed = {
        "experiment_code_sha256": _committed_source_hash(repo, PREREGISTRATION_COMMIT_SHA, list(source_binding["experiment_code_files"]), experiment_prefix),
        "dependency_source_sha256": _committed_source_hash(repo, PREREGISTRATION_COMMIT_SHA, list(source_binding["dependency_source_files"]), "modern/"),
        "field_pipeline_source_sha256": _committed_source_hash(repo, PREREGISTRATION_COMMIT_SHA, list(source_binding["field_pipeline_source_files"]), "modern/"),
    }
    for key, value in recomputed.items():
        if value != source_binding[key] or value != authorities[key]:
            raise ValueError(f"{key} recomputed from the preregistration blobs differs from the sealed value")
    subject = _git(repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA)
    return {
        "results_commit": RESULTS_COMMIT_SHA,
        "results_commit_subject": subject,
        "results_tree": results_tree,
        "results_commit_files": results_commit_files,
        "results_commit_files_outside_results_tree": 0,
        "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
        "preregistration_commit_subject": _git(repo, "show", "-s", "--format=%s", PREREGISTRATION_COMMIT_SHA),
        "code_commit": CODE_COMMIT_SHA,
        "code_commit_subject": _git(repo, "show", "-s", "--format=%s", CODE_COMMIT_SHA),
        "dashboard_commit": DASHBOARD_COMMIT_SHA,
        "dashboard_commit_code_files_changed": 0,
        "manifest_git_blob": committed_blob,
        "manifest_path": manifest_rel,
        "source_hashes_recomputed_at_preregistration": recomputed,
        "source_hash_rule": (
            "sha256 over (relative name, NUL, bytes, NUL) for every file of the sealed source-binding lists, read as "
            "Git blobs at the rebased preregistration commit; each must equal the value the bundle sealed in "
            "source-binding.json and authorities.json"
        ),
        "lineage": {
            "experiment_id": V1_EXPERIMENT_ID,
            "results_commit": V1_RESULTS_COMMIT_SHA,
            "results_commit_subject": _git(repo, "show", "-s", "--format=%s", V1_RESULTS_COMMIT_SHA),
            "results_tree": v1_results_tree,
            "results_commit_files": v1_results_commit_files,
            "manifest_git_blob": v1_committed_blob,
            "manifest_path": v1_manifest_rel,
            "preregistration_commit": V1_PREREGISTRATION_COMMIT_SHA,
            "code_commit": V1_CODE_COMMIT_SHA,
            "posthoc_rejection_commit": CODE_COMMIT_SHA,
            "terminal_state": V1_TERMINAL_STATE,
        },
        "reference_commits": {
            "sweep_v3_results": SWEEP_V3_RESULTS_COMMIT_SHA,
            "cusp_topology_v3_1_results": CUSP_TOPOLOGY_RESULTS_COMMIT_SHA,
        },
        "rebase_note": (
            "the campaign chain was developed on origin/exp/l1b-hemp-confirmation-v1 and rebased onto "
            "feat/sota-foundation; the bundle's execution lock and the protocol's predecessor block name the "
            "pre-rebase commits of that branch, which are recorded here as strings, while every binding above "
            "uses the rebased commits reachable from HEAD and the sealed source hashes are recomputed from their blobs"
        ),
    }


def cross_check_dashboard(
    repo: Path, bundle: Bundle, v1_bundle: Bundle, dataset: dict[str, Any], campaign: dict[str, Any], gates: dict[str, Any],
    lock: dict[str, Any], protocol: dict[str, Any], v1_failures: dict[str, Any], v1_terminal: dict[str, Any], v1_lock: dict[str, Any],
) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of the same two bundles; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    template_raw = (repo / DASHBOARD_TEMPLATE).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    if f'CLASSIFICATION = "{CLASSIFICATION}"' not in generator_text or f'TOPOLOGY_LABEL = "{TOPOLOGY_LABEL}"' not in generator_text:
        raise ValueError("dashboard generator does not pin the confirmation labels")
    if 'expected_state="development_rejection"' not in generator_text or 'if manifest.get("state") != expected_state' not in generator_text:
        raise ValueError("dashboard generator does not verify both bundle states")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_file_sha256"] != bundle.manifest_sha256:
        raise ValueError("dashboard payload names a different results manifest")
    if identity["preregistration_commit_sha"] != lock["commit"] or identity["experiment_id"] != EXPERIMENT_ID or identity["state"] != "accepted_result":
        raise ValueError("dashboard payload names a different lock commit, experiment or state")
    if identity["verified_file_count"] != len(bundle.hashes) or identity["artifact_count"] != bundle.manifest["artifact_count"]:
        raise ValueError("dashboard payload file counts differ from the bundle")
    if identity["terminal_file_sha256"] != bundle.manifest["terminal_byte_sha256"] or identity["lock_file_sha256"] != bundle.manifest["lock_byte_sha256"]:
        raise ValueError("dashboard payload terminal/lock hashes differ from the bundle")
    for key in ("protocol_semantic_sha256", "experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
        if identity[key] != dataset[key]:
            raise ValueError(f"dashboard payload {key} differs from the sealed dataset")
    if identity["sealed_sources"] != dataset["sealed_sources"]:
        raise ValueError("dashboard payload sealed sources differ from the dataset")
    if identity["generator_sha256"] != sha256_bytes(_lf(generator_raw)) or identity["template_sha256"] != sha256_bytes(_lf(template_raw)):
        raise ValueError("dashboard payload generator/template hashes differ from the checkout")
    if identity["protocol_file_sha256_lf"] != sha256_bytes(_lf((repo / EXPERIMENT / "protocol.json").read_bytes())):
        raise ValueError("dashboard payload protocol hash differs from the frozen protocol")
    if payload["classification"] != CLASSIFICATION or payload["topology_label"] != TOPOLOGY_LABEL:
        raise ValueError("dashboard classification or topology label differs from the registration")
    if payload["verdict"] != campaign["verdict"] or payload["verdict_rule"] != gates["confirmation"]["verdict_rule"] or payload["confirmation"] != gates["confirmation"]:
        raise ValueError("dashboard verdict or confirmation block differs from the sealed gates")
    if payload["headline"] != dataset["headline"] or payload["estimands"] != dataset["estimands"] or payload["agreement_table"] != campaign["agreement_table"]:
        raise ValueError("dashboard headline, estimands or agreement table differ from the sealed artifacts")
    if payload["claim_boundary"] != dataset["claim_boundary"] or payload["classification_statement"] != dataset["classification_statement"]:
        raise ValueError("dashboard claim boundary differs from the sealed dataset")
    if payload["gates"]["campaign"] != gates["campaign"] or payload["gates"]["replays"] != gates["replays"] or payload["gates"]["peak_rss_bytes"] != gates["peak_rss_bytes"] or payload["gates"]["ram_budget"] != gates["ram_budget"]:
        raise ValueError("dashboard gates differ from gates.json")
    if payload["execution"] != campaign["execution_mode"] or payload["paper_admission"] != campaign["paper_admission"]:
        raise ValueError("dashboard execution record differs from the campaign result")
    if payload["gate_definitions"] != protocol["gates"] or payload["comparison_rule"] != protocol["comparison"] or payload["definition_import"] != protocol["definition_v3_import"]["source"]:
        raise ValueError("dashboard protocol blocks differ from the frozen protocol")
    if payload["p2"] != {key: protocol["p2"][key] for key in ("solver", "materials", "mesh", "adaptivity", "resources", "sampling")}:
        raise ValueError("dashboard P2 blocks differ from the frozen protocol")
    rows = {item["id"]: item for item in payload["designs"]}
    if set(rows) != {row["design_id"] for row in dataset["designs"]} or len(payload["designs"]) != dataset["design_count"]:
        raise ValueError("dashboard design rows differ from the sealed dataset")
    for row in dataset["designs"]:
        shown = rows[row["design_id"]]
        comparison = row["comparison"]
        expected = {
            "stages": row["derived"]["stage_count"], "x_w": row["derived"]["x_w"], "rw_over_L": row["derived"]["wall_radius_over_pitch"],
            "l1a_cusps": comparison["l1a_wall_cusp_count"], "p2_cusps": comparison["p2_wall_cusp_count"], "l1a_cells": comparison["l1a_cell_count"],
            "p2_cells": comparison["p2_cell_count"], "strict": comparison["count_agreement_strict"], "boundary_tolerant": comparison["count_agreement_boundary_tolerant"],
            "bijection": comparison["cusp_match"]["bijection"], "matched": comparison["matched_cusp_count"], "max_shift_m": comparison["max_cusp_shift_m"],
            "max_shift_tol": comparison["max_cusp_shift_over_tolerance"], "tolerance_m": comparison["cusp_position_tolerance_m"],
            "wall_ratio": comparison["peak_wall_b_ratio_p2_over_l1a"], "axis_ratio": comparison["axis_peak_b_ratio_p2_over_l1a"],
            "l1a_min_rho": comparison["l1a_min_rho_conservative"], "p2_min_rho": comparison["p2_min_rho_conservative"],
            "l1a_hemp": comparison["l1a_hemp_like_all_cusps"], "p2_hemp": comparison["p2_hemp_like_all_cusps"],
            "dofs": [level["p2_dofs"] for level in row["p2"]["levels"]], "converged": row["p2"]["all_levels_converged"],
            "sampling_stable": row["sampling_stability"]["stable"], "seconds": row["p2"]["total_seconds"], "rss_bytes": row["p2"]["peak_rss_bytes"],
            "gates": all(row["gate_checks"].values()),
        }
        for key, value in expected.items():
            if shown[key] != value:
                raise ValueError(f"dashboard {row['design_id']} {key} differs from the sealed row")
    predecessor = payload["predecessor"]
    if predecessor["manifest_file_sha256"] != v1_bundle.manifest_sha256 or predecessor["state"] != V1_TERMINAL_STATE or predecessor["experiment_id"] != V1_EXPERIMENT_ID:
        raise ValueError("dashboard predecessor block names a different v1 bundle")
    if predecessor["preregistration_commit_sha"] != v1_lock["commit"] or predecessor["verified_file_count"] != len(v1_bundle.hashes) or predecessor["artifact_count"] != v1_bundle.manifest["artifact_count"]:
        raise ValueError("dashboard predecessor identity differs from the v1 bundle")
    if predecessor["stage_wall_s"] != v1_terminal["payload"]["stage_wall_s"] or predecessor["resolved_design_count"] != v1_terminal["payload"]["resolved_design_count"]:
        raise ValueError("dashboard predecessor counts differ from the v1 terminal record")
    if [item["design_id"] for item in predecessor["failed_designs"]] != [item["key"].split(":")[1] for item in v1_failures["failed"]]:
        raise ValueError("dashboard predecessor failures differ from the v1 record")
    if predecessor["protocol_block"] != protocol["predecessor"]:
        raise ValueError("dashboard predecessor protocol block differs from the frozen protocol")
    if payload["angle_gate"]["reject_below_angle_deg"] != protocol["p2"]["mesh"]["reject_below_angle_deg"] or payload["angle_gate"]["disclosure"] != protocol["p2"]["mesh"]["angle_gate_disclosure"]:
        raise ValueError("dashboard angle-gate block differs from the frozen protocol")
    return {
        "generator_path": DASHBOARD_GENERATOR.as_posix(),
        "generator_sha256_lf": sha256_bytes(_lf(generator_raw)),
        "template_path": DASHBOARD_TEMPLATE.as_posix(),
        "template_sha256_lf": sha256_bytes(_lf(template_raw)),
        "html_path": DASHBOARD_HTML.as_posix(),
        "html_sha256_lf": sha256_bytes(_lf(html_raw)),
        "html_schema": payload["schema"],
        "payload_manifest_sha256": identity["manifest_file_sha256"],
        "payload_predecessor_manifest_sha256": predecessor["manifest_file_sha256"],
        "angle_gate_designs_below_ten_degrees": list(payload["angle_gate"]["designs_with_elements_below_10deg"]),
        "rule": (
            "the committed dashboard byte-verifies both bundles against their manifests, re-derives the verdict and the "
            "headline counts from the per-design rows, embeds its own extraction and pins the manifest SHA-256 and the lock "
            "commit of each bundle; the generator requires that extraction (identity, sealed sources, verdict, confirmation, "
            "headline, estimands, agreement table, claim boundary, gates, execution, protocol blocks, every design row, the "
            "predecessor record and the angle-gate block) to equal the sealed artifacts before writing any macro"
        ),
    }


# --------------------------------------------------------------------------- #
# Macro construction
# --------------------------------------------------------------------------- #
class Macros:
    def __init__(self, bundle: Bundle, scope: str) -> None:
        self.bundle = bundle
        self.scope = scope
        self.items: list[dict[str, Any]] = []
        self.names: set[str] = set()
        self.docs: dict[str, Any] = {}

    def doc(self, relative: str) -> Any:
        if relative == "manifest.json":
            return self.bundle.manifest
        if relative not in self.docs:
            self.docs[relative] = self.bundle.load(relative)
        return self.docs[relative]

    def _check(self, name: str) -> None:
        if name in self.names or not name.isalpha() or not name.startswith(MACRO_PREFIX):
            raise ValueError(f"macro name {name!r} is invalid or duplicated")

    def add(self, name: str, artifact: str, pointer: str, fmt: str, description: str) -> Any:
        self._check(name)
        raw = resolve_pointer(self.doc(artifact), pointer)
        self.items.append(
            {
                "name": name, "value": format_value(fmt, raw), "raw": raw, "format": fmt,
                "derived": False, "source": {"artifact": artifact, "pointer": pointer},
                "bundle": self.scope, "description": description,
            }
        )
        self.names.add(name)
        return raw

    def add_derived(
        self, name: str, raw: Any, fmt: str, description: str, derivation: str, inputs: list[dict[str, str]]
    ) -> Any:
        self._check(name)
        self.items.append(
            {
                "name": name, "value": format_value(fmt, raw), "raw": raw, "format": fmt,
                "derived": True, "derivation": derivation, "inputs": inputs, "bundle": self.scope,
                "description": description,
            }
        )
        self.names.add(name)
        return raw


def _ident(value: str) -> str:
    return format_value("ident", value)


def _p(width_cm: float) -> str:
    return f">{{\\raggedright\\arraybackslash}}p{{{width_cm:g}cm}}"


def _artifact_claim() -> str:
    return f"\\ArtifactClaim{{{ARTIFACT_CLAIM_ID}}}{{{ARTIFACT_ID}}}{{%"


def _table(macro: str, caption: str, label: str, columns: str, header: str, rows: list[str], *, size: str = "\\footnotesize", extra: str = "") -> list[str]:
    lines = [f"\\newcommand{{\\{macro}}}{{%", _artifact_claim(), "\\begin{table}[ht]", "\\centering"]
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(size)
    if extra:
        lines.append(extra)
    lines.append(f"\\begin{{tabular}}{{{columns}}}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    lines.extend(rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    return lines


def _comma(value: int) -> str:
    return f"{int(value):,d}".replace(",", "{,}")


def _yes(value: bool) -> str:
    return "yes" if value else "no"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(repo: Path) -> tuple[dict[str, Any], str]:  # noqa: C901 - one linear verification pass
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo, RESULTS, EXPERIMENT_ID, "accepted_result")
    v1_bundle = Bundle(repo, V1_RESULTS, V1_EXPERIMENT_ID, V1_TERMINAL_STATE)
    m = Macros(bundle, "results")
    v1 = Macros(v1_bundle, "lineage")
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    dataset = m.doc("artifacts/confirmation-dataset.json")
    protocol = m.doc("artifacts/protocol.json")
    authorities = m.doc("artifacts/authorities.json")
    shakedown = m.doc("artifacts/shakedown.json")
    plan = m.doc("artifacts/campaign-plan.json")
    runtime = m.doc("artifacts/runtime.json")
    design_authorities = m.doc("artifacts/design-authorities.json")
    failures = m.doc("artifacts/design-failures.json")
    source_binding = m.doc("artifacts/source-binding.json")
    csv_text = bundle.raw("artifacts/confirmation-dataset.csv").decode("utf-8")
    transitions = {index: m.doc(f"transitions/{index:04d}-{name}.json") for index, name in TRANSITION_NAMES}
    phases = {name: m.doc(f"phases/{name}.json") for name in ("prebundle", "development", "assessment")}
    v1_terminal = v1.doc("terminal.json")
    v1_lock = v1.doc("execution-lock.json")
    v1_failures = v1.doc("artifacts/design-failures.json")
    v1_protocol = v1.doc("artifacts/protocol.json")
    v1_authorities = v1.doc("artifacts/authorities.json")
    v1_shakedown = v1.doc("artifacts/shakedown.json")
    v1_transitions = {index: v1.doc(f"transitions/{index:04d}-{name}.json") for index, name in V1_TRANSITION_NAMES}
    v1_development = v1.doc("phases/development.json")
    binding = bind_committed(repo, bundle, v1_bundle, source_binding, authorities)
    designs = dataset["designs"]
    headline = dataset["headline"]
    estimands = dataset["estimands"]
    confirmation = gates["confirmation"]
    dashboard = cross_check_dashboard(repo, bundle, v1_bundle, dataset, campaign, gates, lock, protocol, v1_failures, v1_terminal, v1_lock)

    # ---- reference files bound at their own admitted revisions ----------------------
    catalogue_file = _bound_file(repo, SWEEP_V3_CATALOGUE_PATH, SWEEP_V3_RESULTS_COMMIT_SHA, "reference-sweep-v3-catalogue", lf_equal=False)
    sweep_manifest_file = _bound_file(repo, SWEEP_V3_MANIFEST_PATH, SWEEP_V3_RESULTS_COMMIT_SHA, "reference-sweep-v3-manifest", lf_equal=False)
    sweep_design_authorities_file = _bound_file(repo, SWEEP_V3_DESIGN_AUTHORITIES_PATH, SWEEP_V3_RESULTS_COMMIT_SHA, "reference-sweep-v3-design-authorities", lf_equal=False)
    topology_protocol_file = _bound_file(repo, CUSP_TOPOLOGY_PROTOCOL_PATH, CUSP_TOPOLOGY_RESULTS_COMMIT_SHA, "reference-cusp-topology-protocol", lf_equal=False)
    sealed = dataset["sealed_sources"]["l1a_geometry_sweep_v3"]
    if catalogue_file["sha256"] != sealed["catalogue_byte_sha256"] or sweep_manifest_file["sha256"] != sealed["manifest_file_sha256"]:
        raise ValueError("the sweep-v3 catalogue or manifest on disk differs from the identity the campaign sealed")
    if sweep_design_authorities_file["sha256"] != sealed["design_authorities_file_sha256"]:
        raise ValueError("the sweep-v3 design authorities on disk differ from the identity the campaign sealed")
    if authorities["sealed_sources"] != dataset["sealed_sources"] or shakedown["sealed_sources"] != dataset["sealed_sources"]:
        raise ValueError("sealed sources differ between the authorities, the shakedown and the dataset")
    catalogue = load_json_bytes((repo / SWEEP_V3_CATALOGUE_PATH).read_bytes(), "sweep-v3 catalogue")
    if catalogue["experiment_id"] != "l1a-geometry-sweep-v3" or catalogue["protocol_semantic_sha256"] != sealed["protocol_semantic_sha256"]:
        raise ValueError("sweep-v3 catalogue identity differs from the sealed source")
    hemp_entries = [entry for entry in catalogue["entries"] if entry["set_id"] == "sobol_v3" and entry["hemp_like_all_cusps"] is True]
    if len(hemp_entries) != catalogue["hemp_like_design_count"] or catalogue["hemp_like_design_count"] != sealed["hemp_like_design_count"]:
        raise ValueError("HEMP-like design count differs between the catalogue and the sealed source")
    declared_ids = list(protocol["design_sets"][SET_ID]["design_ids"])
    if declared_ids != [entry["design_id"] for entry in hemp_entries] or len(declared_ids) != protocol["design_sets"][SET_ID]["design_count"]:
        raise ValueError("declared designs differ from the HEMP-like entries of the sealed catalogue")
    hemp_by_id = {entry["design_id"]: entry for entry in hemp_entries}
    topology_protocol = load_json_bytes((repo / CUSP_TOPOLOGY_PROTOCOL_PATH).read_bytes(), "cusp topology v3.1 protocol")
    imported = protocol["definition_v3_import"]
    if imported["numerical_parameters"] != topology_protocol["definition_v3"]["numerical_parameters"]:
        raise ValueError("the imported definition parameters differ from the frozen cusp topology v3.1 protocol")
    if imported["stability_tolerance_m"] != topology_protocol["definition_v3"]["stability_tolerance_m"] or imported["minimum_certificate_dense_to_bound_ratio"] != topology_protocol["definition_v3"]["minimum_certificate_dense_to_bound_ratio"]:
        raise ValueError("the imported stability tolerance differs from the frozen cusp topology v3.1 protocol")
    rejection_file = _bound_file(repo, V1_REJECTION_PATH, CODE_COMMIT_SHA, "lineage-posthoc-rejection", lf_equal=True)
    rejection_text = _lf((repo / V1_REJECTION_PATH).read_bytes()).decode("utf-8")

    # ---- internal consistency of the sealed bundle (fail closed on any disagreement) ----
    if campaign["status"] != CAMPAIGN_STATUS or campaign["verdict"] != VERDICT or campaign["evidentiary"] is not True or campaign["plan_kind"] != "evidentiary":
        raise ValueError("campaign result is not the accepted evidentiary confirmation")
    if not (campaign["classification"] == dataset["classification"] == protocol["classification"] == authorities["classification"] == shakedown["classification"] == phases["prebundle"]["classification"] == CLASSIFICATION):
        raise ValueError("classification differs between the sealed artifacts")
    if not (campaign["topology_label"] == dataset["topology_label"] == phases["prebundle"]["topology_label"] == TOPOLOGY_LABEL):
        raise ValueError("topology label differs between the sealed artifacts")
    if campaign["campaign_gates"] != gates["campaign"] or gates["passed"] is not True or gates["binding"] is not True or any(gates["campaign"][name] is not True for name in BINDING_GATE_NAMES):
        raise ValueError("gates.json disagrees with the campaign result or records a failure")
    if set(gates["campaign"]) != set(BINDING_GATE_NAMES) or set(gates["definitions"]["binding_integrity"]) != set(BINDING_GATE_NAMES):
        raise ValueError("binding gate set differs from the registered set")
    if campaign["confirmation_gates"] != {"cusp_count_unchanged": confirmation["cusp_count_unchanged"]["passed"], "cusp_position_shift": confirmation["cusp_position_shift"]["passed"]}:
        raise ValueError("confirmation gate flags differ between gates.json and the campaign result")
    if confirmation != estimands["confirmation"] or confirmation != dataset["gates"]["confirmation"] or gates["campaign"] != dataset["gates"]["campaign"]:
        raise ValueError("confirmation block differs between gates.json and the dataset")
    if campaign["headline"] != headline or headline["verdict"] != VERDICT or confirmation["verdict"] != VERDICT or gates["confirmation"]["verdict_rule"] != protocol["gates"]["confirmation"]["verdict_rule"]:
        raise ValueError("headline or verdict differs between the sealed artifacts")
    if not (len(designs) == dataset["design_count"] == campaign["design_count"] == gates["design_count"] == headline["design_count"] == authorities["design_count"] == plan["design_keys"].__len__() == len(campaign["agreement_table"])):
        raise ValueError("design count differs between the sealed artifacts")
    if failures["failed"] != [] or campaign["execution_mode"]["worker_pool_size"] != 1 or runtime["worker_pool_size"] != 1:
        raise ValueError("the bundle records a failed design or a parallel worker pool")
    if not (dataset["protocol_semantic_sha256"] == authorities["protocol_semantic_sha256"] == shakedown["protocol_semantic_sha256"] == campaign["protocol_semantic_sha256"] == source_binding["protocol_semantic_sha256"]):
        raise ValueError("protocol semantic hash differs between the sealed artifacts")
    if _semantic_sha256(protocol) != dataset["protocol_semantic_sha256"] or _semantic_sha256(load_json_bytes((repo / EXPERIMENT / "protocol.json").read_bytes(), "protocol")) != dataset["protocol_semantic_sha256"]:
        raise ValueError("the protocol semantic hash does not recompute from the sealed or the frozen protocol")
    for key in ("experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
        if not (dataset[key] == authorities[key] == shakedown[key] == source_binding[key] == phases["prebundle"][key]):
            raise ValueError(f"{key} differs between the sealed artifacts")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"] or authorities["shakedown_semantic_sha256"] != _semantic_sha256(shakedown):
        raise ValueError("shakedown artifact differs from the bound authority")
    if authorities["design_authorities_sha256"] != _semantic_sha256(design_authorities):
        raise ValueError("design authorities differ from the bound authority")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if authorities["shakedown_timing_projection"] != shakedown["timing_projection"] or any(v is not True for v in authorities["shakedown_gate_checks"].values()):
        raise ValueError("shakedown gate checks differ between authorities and shakedown")
    preflight = shakedown["mesh_preflight"]
    if preflight["all_passed"] is not True or preflight["passed_count"] != len(designs) or preflight["design_count"] != len(designs) or preflight["failed_designs"] != []:
        raise ValueError("the whole-set mesh preflight does not record every design as passed")
    if sorted(preflight["designs_with_elements_below_10deg"]) != sorted(item["key"].split(":")[1] for item in v1_failures["failed"]):
        raise ValueError("the designs with sliver elements differ from the designs the v1 run rejected")
    if not _close(preflight["minimum_angle_deg"], min(item["minimum_angle_deg"] for item in preflight["designs"])) or any(item["reject_below_angle_deg"] != protocol["p2"]["mesh"]["reject_below_angle_deg"] for item in preflight["designs"]):
        raise ValueError("the mesh preflight's minimum angle or gate does not recompute")
    shakedown_ids = list(protocol["shakedown"]["designs"][SET_ID])
    if set(shakedown_ids) - set(declared_ids) or len(shakedown_ids) != shakedown["design_count"] or shakedown["resolved_design_count"] != len(shakedown_ids):
        raise ValueError("shakedown designs are not a resolved subset of the declared set")
    if {item["key"].split(":")[1] for item in v1_failures["failed"]} - set(shakedown_ids):
        raise ValueError("the v1-rejected designs were not exercised by the v1.1 shakedown")
    for frozen in FROZEN_FILES:
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != m.doc(f"artifacts/{frozen}"):
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
        if load_json_bytes((repo / V1_EXPERIMENT / frozen).read_bytes(), frozen) != v1.doc(f"artifacts/{frozen}"):
            raise ValueError(f"v1 frozen {frozen} differs from the sealed copy in the v1 bundle")
    if plan["kind"] != "evidentiary" or plan["binding_gates"] is not True or plan["design_keys"] != [f"{SET_ID}:{design_id}" for design_id in declared_ids]:
        raise ValueError("campaign plan differs from the frozen protocol")
    if lock["experiment_id"] != EXPERIMENT_ID or lock["attempt"] != 1 or lock["immutable"] is not True or lock["clean_worktree_attested"] is not True:
        raise ValueError("execution lock is not the single immutable attempt of this experiment")
    if not re.fullmatch(r"[0-9a-f]{40}", lock["commit"]) or lock["commit"] == PREREGISTRATION_COMMIT_SHA:
        raise ValueError("execution lock commit is not the pre-rebase preregistration identifier")
    if terminal["state"] != bundle.manifest["state"] or terminal["payload"] != phases["assessment"]["fields"]["payload"] or terminal["counts"]["attempt_count"] != 1:
        raise ValueError("terminal record disagrees with the manifest or the assessment phase")
    if terminal["payload"]["verdict"] != VERDICT or terminal["payload"]["status"] != CAMPAIGN_STATUS or terminal["payload"]["gates"] != gates["campaign"]:
        raise ValueError("terminal payload differs from the campaign result")
    if phases["development"]["fields"]["accepted"] is not True or phases["development"]["fields"]["payload"]["failed_design_count"] != 0 or phases["development"]["fields"]["payload"]["resolved_design_count"] != len(designs):
        raise ValueError("development phase does not record every design as resolved")
    if not _close(phases["development"]["fields"]["payload"]["stage_wall_s"], campaign["execution_mode"]["stage_wall_s"]):
        raise ValueError("stage wall time differs between the development phase and the campaign result")
    if transitions[9]["transition"] != "terminal" or transitions[9]["details"]["state"] != "accepted_result" or transitions[1]["transition"] != "lock-acquired":
        raise ValueError("transition log does not run lock-acquired -> terminal")
    execution_wall_s = (_utc(transitions[9]["recorded_at_utc"]) - _utc(transitions[1]["recorded_at_utc"])).total_seconds()
    stage_wall_s = (_utc(transitions[6]["recorded_at_utc"]) - _utc(transitions[5]["recorded_at_utc"])).total_seconds()
    assessment_wall_s = (_utc(transitions[8]["recorded_at_utc"]) - _utc(transitions[7]["recorded_at_utc"])).total_seconds()
    if execution_wall_s <= 0 or execution_wall_s < campaign["execution_mode"]["stage_wall_s"] + campaign["execution_mode"]["assessment_wall_s"]:
        raise ValueError("execution wall time is not longer than the recorded stages")
    if abs(stage_wall_s - campaign["execution_mode"]["stage_wall_s"]) > 5.0 or abs(assessment_wall_s - campaign["execution_mode"]["assessment_wall_s"]) > 5.0:
        raise ValueError("transition timestamps disagree with the recorded stage wall times")
    if gates["ram_budget"] != runtime["ram_budget"] or gates["peak_rss_bytes"] != headline["peak_rss_bytes"] or gates["ram_budget"]["budget_bytes"] != headline["ram_budget_bytes"]:
        raise ValueError("RAM budget or peak RSS differs between the sealed artifacts")
    if not _close(headline["ram_budget_fraction_used"], headline["peak_rss_bytes"] / headline["ram_budget_bytes"]) or headline["peak_rss_bytes"] > headline["ram_budget_bytes"]:
        raise ValueError("RAM budget fraction does not recompute or the budget was exceeded")
    if not _close(gates["ram_budget"]["budget_bytes"], gates["ram_budget"]["fraction"] * gates["ram_budget"]["free_at_start_bytes"]):
        raise ValueError("the RAM budget does not follow from the declared fraction")
    if gates["ram_budget"]["fraction"] != protocol["p2"]["resources"]["ram_budget_fraction_of_free_at_start"] or gates["ram_budget"]["maximum_p2_dofs"] != protocol["p2"]["resources"]["maximum_p2_dofs"]:
        raise ValueError("RAM policy differs from the frozen protocol")
    if len(gates["replays"]) != 1 or gates["replays"][0]["key"] != f"{SET_ID}:{protocol['execution']['replay_designs'][SET_ID][0]}" or not all(gates["replays"][0][k] is True for k in ("accepted_grid_equal", "bit_identical", "field_identity_equal", "p2_run_sha256_equal")):
        raise ValueError("the determinism replay is not the frozen design or is not bit-identical")
    if gates["replays"][0]["replay_comparison_payload_sha256"] != gates["replays"][0]["worker_comparison_payload_sha256"]:
        raise ValueError("the determinism replay's payload hashes differ")
    if set(gates["per_design"]) != {f"{SET_ID}:{design_id}" for design_id in declared_ids} or any(not all(checks.values()) for checks in gates["per_design"].values()):
        raise ValueError("per-design gate checks are incomplete or record a failure")
    if any(gates["failing_designs"][name] != [] for name in gates["failing_designs"]):
        raise ValueError("gates.json records a failing design")

    # ---- per-design replay against the records, grids, agreement table and CSV --------
    agreement = {row["design_id"]: row for row in campaign["agreement_table"]}
    csv_rows = {row["design_id"]: row for row in csv.DictReader(io.StringIO(csv_text))}
    if len(csv_rows) != len(designs) or set(csv_rows) != set(declared_ids):
        raise ValueError("CSV rows differ from the declared designs")
    l1a_dz = protocol["comparison"]["l1a_dz_m"]
    bore_elements = protocol["p2"]["mesh"]["bore_elements"]
    rho_threshold: float | None = None
    matched_pairs: list[dict[str, Any]] = []
    pairs_by_design: dict[str, list[dict[str, Any]]] = {}
    per_design: dict[str, dict[str, Any]] = {}
    strict_agree = tolerant_agree = bijective = preserved = 0
    channel_null_bijections = channel_count_equal = more_p2_nulls = 0
    channel_sorted_shifts: list[float] = []
    channel_matched_shifts: list[float] = []
    pooled_matched_shifts: list[float] = []
    outside_shifts: list[float] = []
    lean_l1a: list[float] = []
    lean_p2: list[float] = []
    level_dofs: dict[int, list[int]] = {0: [], 1: []}
    level_iterations: dict[int, list[int]] = {0: [], 1: []}
    residuals: list[float] = []
    solve_seconds: list[float] = []
    design_rss: list[int] = []
    disc_wall: list[float] = []
    disc_axis: list[float] = []
    disc_rho: list[float] = []
    sampling_wall: list[float] = []
    l1a_b3: list[float] = []
    p2_b3: list[float] = []
    solves_converged = 0
    shift_over_disc_all = True
    shifts_above_stability = 0
    l1a_outside_total = p2_outside_total = 0
    hemp_flag_stable_levels = 0
    for ordinal, (design_id, row) in enumerate(zip(declared_ids, designs, strict=True)):
        if row["design_id"] != design_id or row["ordinal"] != ordinal or row["set_id"] != SET_ID or row["key"] != f"{SET_ID}:{design_id}":
            raise ValueError(f"{design_id}: dataset row identity differs from the declared order")
        if row["classification"] != CLASSIFICATION or row["label"] != TOPOLOGY_LABEL or not all(row["gate_checks"].values()):
            raise ValueError(f"{design_id}: row label or gate checks differ")
        if row["gate_checks"] != gates["per_design"][row["key"]]:
            raise ValueError(f"{design_id}: per-design gate checks differ from gates.json")
        comparison = row["comparison"]
        record_rel = row["record_path"]
        record = bundle.load(record_rel)
        if record["design_id"] != design_id or record["comparison"] != comparison or record["status"] != "resolved" or record["ordinal"] != ordinal:
            raise ValueError(f"{design_id}: design record differs from the dataset row")
        if [c["z_c_m"] for c in record["accepted"]["topology"]["wall_cusps"]] != [c["z_c_m"] for c in row["p2_wall_cusps"]]:
            raise ValueError(f"{design_id}: record cusps differ from the dataset row")
        if record["accepted"]["axis_nulls"]["nulls"].__len__() != comparison["p2_axis_null_count"] or len(record["accepted"]["topology"]["wall_cusps"]) != comparison["p2_wall_cusp_count"]:
            raise ValueError(f"{design_id}: record null or cusp count differs")
        if record["descriptors"]["accepted"]["hemp_like_all_cusps"] != comparison["p2_hemp_like_all_cusps"] or record["descriptors"]["accepted"]["min_rho_conservative"] != comparison["p2_min_rho_conservative"]:
            raise ValueError(f"{design_id}: record descriptors differ from the comparison")
        threshold = record["descriptors"]["accepted"]["hemp_like_threshold"]["rho"]
        if rho_threshold is None:
            rho_threshold = threshold
        elif threshold != rho_threshold:
            raise ValueError("the HEMP-like threshold differs between design records")
        if record["l1a_reference"]["record_byte_sha256"] != row["l1a"]["record_byte_sha256"] or record["identity"]["l1a_record_byte_sha256"] != row["l1a"]["record_byte_sha256"]:
            raise ValueError(f"{design_id}: L1a record binding differs")
        catalogue_entry = hemp_by_id[design_id]
        if catalogue_entry["record_path"] != row["l1a"]["record_path"] or catalogue_entry["accepted_field_identity_sha256"] != record["identity"]["l1a_accepted_field_identity_sha256"]:
            raise ValueError(f"{design_id}: sealed catalogue entry differs from the L1a reference")
        if catalogue_entry["wall_cusp_count"] != comparison["l1a_wall_cusp_count"] or catalogue_entry["cell_count"] != comparison["l1a_cell_count"] or not _close(catalogue_entry["min_rho_conservative"], comparison["l1a_min_rho_conservative"]):
            raise ValueError(f"{design_id}: sealed catalogue topology differs from the L1a reference")
        if [c["z_c_m"] for c in catalogue_entry["wall_cusps"]] != [c["z_c_m"] for c in row["l1a"]["wall_cusps"]] or not _close(catalogue_entry["x_w"], row["derived"]["x_w"]):
            raise ValueError(f"{design_id}: sealed catalogue cusps differ from the L1a reference")
        # Field grid: the accepted P2 sample bound by payload hash and identity.
        grid_rel = row["accepted_grid_path"] if "accepted_grid_path" in row else record["accepted_grid_path"]
        grid = bundle.load_gz(grid_rel, record["accepted_grid_payload_sha256"])
        if grid["identity"] != record["identity"] or grid["key"] != row["key"]:
            raise ValueError(f"{design_id}: field grid identity differs from the record")
        if len(grid["r_m"]) != row["sampling"]["accepted"]["radial_samples"] or len(grid["z_m"]) != row["sampling"]["accepted"]["axial_samples"]:
            raise ValueError(f"{design_id}: field grid shape differs from the sampling record")
        if len(grid["r_m"]) != protocol["p2"]["sampling"]["radial_intervals"] + 1:
            raise ValueError(f"{design_id}: field grid radial samples differ from the frozen sampling")
        if not _close(row["sampling"]["refined"]["dr_m"] * protocol["p2"]["sampling"]["refinement"], row["sampling"]["accepted"]["dr_m"]):
            raise ValueError(f"{design_id}: refined sampling is not the declared refinement of the accepted sampling")
        if not _close(row["sampling"]["accepted"]["source_strength_scale_applied"], row["design_values"]["source_strength_scale"]) or not _close(comparison["source_strength_scale"], row["design_values"]["source_strength_scale"]):
            raise ValueError(f"{design_id}: the source strength scale differs between sampling, design values and comparison")
        # Geometry-derived quantities.
        r_w = row["geometry"]["wall_radius_m"]
        pitch = row["derived"]["represented_stage_pitch_m"]
        if not _close(row["derived"]["x_w"], math.pi * r_w / pitch) or not _close(row["derived"]["wall_radius_over_pitch"], r_w / pitch):
            raise ValueError(f"{design_id}: x_w or r_w / L does not recompute")
        if not _close(comparison["cusp_position_tolerance_m"], max(r_w / bore_elements, l1a_dz)) or not _close(row["l1a"]["grid"]["dz_m"], l1a_dz):
            raise ValueError(f"{design_id}: the cusp-position tolerance does not recompute from max(r_w / bore elements, L1a dz)")
        # Counts and agreement.
        strict = comparison["p2_wall_cusp_count"] == comparison["l1a_wall_cusp_count"]
        if comparison["count_agreement_strict"] is not strict or comparison["cell_count_agreement"] is not (comparison["p2_cell_count"] == comparison["l1a_cell_count"]):
            raise ValueError(f"{design_id}: strict count agreement does not recompute")
        if strict and comparison["count_agreement_boundary_tolerant"] is not True:
            raise ValueError(f"{design_id}: boundary-tolerant agreement contradicts the strict agreement")
        if comparison["l1a_wall_cusp_count"] != len(row["l1a"]["wall_cusps"]) or comparison["p2_wall_cusp_count"] != len(row["p2_wall_cusps"]) or comparison["l1a_cell_count"] != len(row["l1a"]["cells"]) or comparison["p2_cell_count"] != len(row["p2_cells"]):
            raise ValueError(f"{design_id}: cusp or cell counts differ from the listed cusps and cells")
        if comparison["p2_cell_count"] != comparison["p2_wall_cusp_count"] + 1 or comparison["l1a_cell_count"] != comparison["l1a_wall_cusp_count"] + 1:
            raise ValueError(f"{design_id}: cells are not cusps plus one")
        strict_agree += strict
        tolerant_agree += comparison["count_agreement_boundary_tolerant"]
        # Matched cusps: shift, tolerance ratio, wall |B| ratio, rho ratio, bijection.
        pairs = comparison["matched_cusps"]
        if len(pairs) != comparison["matched_cusp_count"] or comparison["cusp_match"]["observed_count"] != comparison["p2_wall_cusp_count"] or comparison["cusp_match"]["reference_count"] != comparison["l1a_wall_cusp_count"]:
            raise ValueError(f"{design_id}: matched cusp count differs")
        is_bijection = len(pairs) == comparison["p2_wall_cusp_count"] == comparison["l1a_wall_cusp_count"] and comparison["unmatched_cusps"] == []
        if comparison["cusp_match"]["bijection"] is not is_bijection or comparison["cusp_match"]["unmatched_observed_z_m"] != [] or comparison["cusp_match"]["unmatched_reference_z_m"] != []:
            raise ValueError(f"{design_id}: cusp bijection flag does not recompute")
        bijective += is_bijection
        l1a_rho = {item["cusp_id"]: item for item in row["l1a"]["rho"]}
        p2_rho = {item["cusp_id"]: item for item in row["p2_rho"]}
        l1a_cusps = {item["cusp_id"]: item for item in row["l1a"]["wall_cusps"]}
        p2_cusps = {item["cusp_id"]: item for item in row["p2_wall_cusps"]}
        for item in list(l1a_rho.values()) + list(p2_rho.values()):
            if not _close(item["rho_conservative"], item["wall_b_t"] / max(item["upstream_axis_peak_t"], item["downstream_axis_peak_t"])):
                raise ValueError(f"{design_id}: the conservative Koch ratio does not recompute from the wall field and the adjacent axis peaks")
            if item["hemp_like_conservative"] is not (item["rho_conservative"] >= threshold):
                raise ValueError(f"{design_id}: the per-cusp HEMP-like flag does not recompute")
        for pair in pairs:
            l1a_cusp = l1a_cusps[pair["l1a_cusp_id"]]
            p2_cusp = p2_cusps[pair["p2_cusp_id"]]
            if not _close(pair["l1a_z_c_m"], l1a_cusp["z_c_m"]) or not _close(pair["p2_z_c_m"], p2_cusp["z_c_m"]) or not _close(pair["l1a_wall_b_t"], l1a_cusp["wall_b_t"]) or not _close(pair["p2_wall_b_t"], p2_cusp["wall_b_t"]):
                raise ValueError(f"{design_id}: matched pair differs from the listed cusps")
            if not _close(pair["shift_m"], abs(pair["p2_z_c_m"] - pair["l1a_z_c_m"])) or not _close(pair["shift_over_tolerance"], pair["shift_m"] / comparison["cusp_position_tolerance_m"]):
                raise ValueError(f"{design_id}: cusp shift does not recompute")
            if not _close(pair["wall_b_ratio_p2_over_l1a"], pair["p2_wall_b_t"] / pair["l1a_wall_b_t"]) or not _close(pair["p2_wall_b_unscaled_t"] * comparison["source_strength_scale"], pair["p2_wall_b_t"]):
                raise ValueError(f"{design_id}: wall field ratio or magnet-strength scaling does not recompute")
            if not _close(pair["l1a_rho_conservative"], l1a_rho[pair["l1a_cusp_id"]]["rho_conservative"]) or not _close(pair["p2_rho_conservative"], p2_rho[pair["p2_cusp_id"]]["rho_conservative"]):
                raise ValueError(f"{design_id}: matched pair rho differs from the rho tables")
            if not _close(pair["rho_conservative_ratio_p2_over_l1a"], pair["p2_rho_conservative"] / pair["l1a_rho_conservative"]):
                raise ValueError(f"{design_id}: rho ratio does not recompute")
            if pair["shift_m"] > comparison["cusp_position_tolerance_m"]:
                raise ValueError(f"{design_id}: a matched cusp exceeds its tolerance")
            if pair["shift_m"] <= row["p2_discretisation"]["max_wall_intersection_shift_m"]:
                shift_over_disc_all = False
            if pair["shift_m"] > imported["stability_tolerance_m"]:
                shifts_above_stability += 1
            matched_pairs.append({**pair, "design_id": design_id, "tolerance_m": comparison["cusp_position_tolerance_m"]})
        pairs_by_design[design_id] = pairs
        if not _close(comparison["max_cusp_shift_m"], max(p["shift_m"] for p in pairs)) or not _close(comparison["max_cusp_shift_over_tolerance"], max(p["shift_over_tolerance"] for p in pairs)) or not _close(comparison["median_cusp_shift_m"], statistics.median(p["shift_m"] for p in pairs)):
            raise ValueError(f"{design_id}: per-design shift statistics do not recompute")
        if comparison["all_matched_within_tolerance"] is not True or comparison["position_gate_passed"] is not True:
            raise ValueError(f"{design_id}: per-design position gate is not recorded as passed")
        if not _close(comparison["l1a_min_rho_conservative"], min(item["rho_conservative"] for item in l1a_rho.values())) or not _close(comparison["p2_min_rho_conservative"], min(item["rho_conservative"] for item in p2_rho.values())):
            raise ValueError(f"{design_id}: minimum rho does not recompute")
        p2_hemp = bool(p2_rho) and all(item["rho_conservative"] >= threshold for item in p2_rho.values())
        if comparison["p2_hemp_like_all_cusps"] is not p2_hemp or comparison["l1a_hemp_like_all_cusps"] is not True or row["l1a"]["hemp_like_all_cusps"] is not True:
            raise ValueError(f"{design_id}: HEMP-like flags do not recompute")
        if comparison["hemp_like_preserved"] is not p2_hemp:
            raise ValueError(f"{design_id}: HEMP-like preservation flag does not recompute")
        preserved += p2_hemp
        if not _close(comparison["peak_wall_b_ratio_p2_over_l1a"], comparison["p2_peak_wall_b_t"] / comparison["l1a_peak_wall_b_t"]) or not _close(comparison["axis_peak_b_ratio_p2_over_l1a"], comparison["p2_axis_peak_b_t"] / comparison["l1a_axis_peak_b_t"]):
            raise ValueError(f"{design_id}: peak field ratios do not recompute")
        if not _close(comparison["peak_wall_b_ratio_unscaled"], comparison["p2_peak_wall_b_unscaled_t"] / comparison["l1a_peak_wall_b_t"]) or not _close(comparison["p2_peak_wall_b_unscaled_t"] * comparison["source_strength_scale"], comparison["p2_peak_wall_b_t"]):
                raise ValueError(f"{design_id}: unscaled peak wall field does not recompute")
        band = comparison["wall_b_ratio_band_descriptive"]
        if comparison["peak_wall_b_ratio_in_band"] is not (band[0] <= comparison["peak_wall_b_ratio_p2_over_l1a"] <= band[1]):
            raise ValueError(f"{design_id}: the descriptive band flag does not recompute")
        # Axis nulls: channel population (sorted pairing) and outside population.
        channel = comparison["channel_axis_nulls"]
        l1a_channel = sorted(null["z_m"] for null in row["l1a"]["axis_nulls"] if null["zone"] == "channel")
        p2_channel = sorted(null["z_m"] for null in row["p2_axis_nulls"] if null["zone"] == "channel")
        if sorted(channel["l1a_z_m"]) != l1a_channel or sorted(channel["p2_z_m"]) != p2_channel or channel["count_equal"] is not (len(l1a_channel) == len(p2_channel)):
            raise ValueError(f"{design_id}: channel axis nulls differ from the listed nulls")
        if channel["count_equal"]:
            shifts = [abs(a - b) for a, b in zip(l1a_channel, p2_channel, strict=True)]
            if not all(_close(a, b) for a, b in zip(channel["sorted_shifts_m"], shifts, strict=True)) or not _close(channel["max_sorted_shift_m"], max(shifts)):
                raise ValueError(f"{design_id}: sorted channel-null shifts do not recompute")
            channel_sorted_shifts.extend(shifts)
            channel_count_equal += 1
        channel_match = comparison["channel_axis_null_match"]
        if channel_match["bijection"]:
            channel_null_bijections += 1
        if channel_match["max_difference_m"] is not None:
            channel_matched_shifts.append(channel_match["max_difference_m"])
        if comparison["axis_null_match"]["max_difference_m"] is not None:
            pooled_matched_shifts.append(comparison["axis_null_match"]["max_difference_m"])
        if comparison["axis_null_match"]["bijection"] is True and comparison["p2_axis_null_count"] != comparison["l1a_axis_null_count"]:
            raise ValueError(f"{design_id}: pooled axis-null bijection contradicts the counts")
        more_p2_nulls += comparison["p2_axis_null_count"] > comparison["l1a_axis_null_count"]
        outside = comparison["outside_channel_axis_nulls"]
        l1a_outside_total += len(outside["l1a_z_m"])
        p2_outside_total += len(outside["p2_z_m"])
        if outside["shifts_m"] is not None:
            outside_shifts.extend(outside["shifts_m"])
        lean = comparison["separatrix_lean_m"]
        expected_l1a_lean = [abs(c["z_c_m"] - c["axis_null_z_m"]) for c in row["l1a"]["wall_cusps"]]
        expected_p2_lean = [abs(c["z_c_m"] - c["axis_null_z_m"]) for c in row["p2_wall_cusps"]]
        if not all(_close(a, b) for a, b in zip(lean["l1a_axis_null_to_cusp"], expected_l1a_lean, strict=True)) or not all(_close(a, b) for a, b in zip(lean["p2_axis_null_to_cusp"], expected_p2_lean, strict=True)):
            raise ValueError(f"{design_id}: the separatrix lean does not recompute from the cusps and their axis nulls")
        if not _close(lean["l1a_max"], max(expected_l1a_lean)) or not _close(lean["p2_max"], max(expected_p2_lean)):
            raise ValueError(f"{design_id}: the maximum separatrix lean does not recompute")
        lean_l1a.extend(expected_l1a_lean)
        lean_p2.extend(expected_p2_lean)
        # P2 solve evidence.
        levels = row["p2"]["levels"]
        if len(levels) != protocol["p2"]["adaptivity"]["levels"] or [level["level"] for level in levels] != [0, 1] or row["p2"]["all_levels_converged"] is not all(level["converged"] for level in levels):
            raise ValueError(f"{design_id}: P2 levels differ from the frozen two-level configuration")
        record_levels = record["evidence"]["p2"]["levels"]
        for level, record_level in zip(levels, record_levels, strict=True):
            if any(record_level[key] != level[key] for key in level):
                raise ValueError(f"{design_id}: record level {level['level']} differs from the dataset row")
            if level["converged"] is not True or level["relative_true_residual_l2"] > protocol["p2"]["solver"]["relative_tolerance"] or level["iterations"] > protocol["p2"]["solver"]["max_iterations"]:
                raise ValueError(f"{design_id}: a P2 level did not converge within the frozen controls")
            if level["p2_dofs"] > protocol["p2"]["resources"]["maximum_p2_dofs"]:
                raise ValueError(f"{design_id}: a P2 level exceeds the DOF cap")
            if record_level["mesh_quality"]["minimum_angle_deg"] < protocol["p2"]["mesh"]["reject_below_angle_deg"] or record_level["mesh_quality"]["sliver"]["threshold_deg"] != SLIVER_THRESHOLD_DEG:
                raise ValueError(f"{design_id}: a recorded level falls below the declared angle gate or reports a different sliver threshold")
            level_dofs[level["level"]].append(level["p2_dofs"])
            level_iterations[level["level"]].append(level["iterations"])
            residuals.append(level["relative_true_residual_l2"])
            solves_converged += 1
        if levels[1]["p2_dofs"] <= levels[0]["p2_dofs"]:
            raise ValueError(f"{design_id}: level 1 is not finer than level 0")
        solve_seconds.append(row["p2"]["total_seconds"])
        design_rss.append(row["p2"]["peak_rss_bytes"])
        if row["p2"]["peak_rss_bytes"] > headline["peak_rss_bytes"]:
            raise ValueError(f"{design_id}: per-design RSS exceeds the campaign peak")
        iron = [region for region in row["p2"]["regions"] if region["relative_permeability"] == protocol["p2"]["materials"]["soft_iron_relative_permeability"]]
        magnets = [region for region in row["p2"]["regions"] if region["remanence_z_t"] != 0.0]
        if len(magnets) != row["derived"]["stage_count"] or len(iron) != row["derived"]["stage_count"] or not any(region["region_id"] == "return-yoke" for region in iron):
            raise ValueError(f"{design_id}: the material regions are not one magnet per stage, one pole per gap and a return yoke")
        if any(not _close(region["relative_permeability"], protocol["p2"]["materials"]["magnet_recoil_relative_permeability"]) for region in magnets):
            raise ValueError(f"{design_id}: magnet recoil permeability differs from the frozen protocol")
        # Discretisation and sampling stability.
        disc = row["p2_discretisation"]
        if disc["stable"] is not True or disc["wall_cusp_count_equal"] is not True or disc["axis_null_count_equal"] is not True:
            raise ValueError(f"{design_id}: P2 discretisation comparison is not stable")
        disc_wall.append(disc["max_wall_intersection_shift_m"])
        disc_axis.append(disc["max_axis_null_shift_m"])
        disc_rho.append(row["p2_discretisation_rho_sensitivity"]["max_relative_rho_difference"])
        if row["p2_discretisation_rho_sensitivity"]["hemp_like_flag_agrees"] is True:
            hemp_flag_stable_levels += 1
        sampling = row["sampling_stability"]
        if sampling["stable"] is not True or sampling["wall_cusp_count_equal"] is not True or sampling["max_wall_intersection_shift_m"] > imported["stability_tolerance_m"]:
            raise ValueError(f"{design_id}: sampling stability is not recorded as stable")
        sampling_wall.append(sampling["max_wall_intersection_shift_m"])
        l1a_b3.append(row["l1a"]["wall_harmonics"]["b3_over_b1"])
        p2_b3.append(row["p2_wall_harmonics"]["b3_over_b1"])
        # Agreement-table row and CSV row.
        table_row = agreement[design_id]
        expected_table = {
            "count_agreement_boundary_tolerant": comparison["count_agreement_boundary_tolerant"], "count_agreement_strict": comparison["count_agreement_strict"],
            "cusp_bijection": comparison["cusp_match"]["bijection"], "cusp_position_tolerance_m": comparison["cusp_position_tolerance_m"],
            "cusp_wall_b_ratios": [p["wall_b_ratio_p2_over_l1a"] for p in pairs], "design_id": design_id, "l1a_cell_count": comparison["l1a_cell_count"],
            "l1a_min_rho_conservative": comparison["l1a_min_rho_conservative"], "l1a_wall_cusp_count": comparison["l1a_wall_cusp_count"],
            "max_channel_axis_null_shift_m": channel["max_sorted_shift_m"], "max_cusp_shift_m": comparison["max_cusp_shift_m"],
            "max_cusp_shift_over_tolerance": comparison["max_cusp_shift_over_tolerance"], "p2_all_levels_converged": row["p2"]["all_levels_converged"],
            "p2_cell_count": comparison["p2_cell_count"], "p2_discretisation_max_cusp_shift_m": disc["max_wall_intersection_shift_m"],
            "p2_hemp_like": comparison["p2_hemp_like_all_cusps"], "p2_level1_dofs": levels[1]["p2_dofs"], "p2_min_rho_conservative": comparison["p2_min_rho_conservative"],
            "p2_wall_cusp_count": comparison["p2_wall_cusp_count"], "peak_wall_b_ratio_p2_over_l1a": comparison["peak_wall_b_ratio_p2_over_l1a"],
            "stage_count": row["derived"]["stage_count"], "x_w": row["derived"]["x_w"],
        }
        if table_row != expected_table:
            raise ValueError(f"{design_id}: agreement-table row differs from the dataset row")
        csv_row = csv_rows[design_id]
        csv_expected = {
            "stage_count": str(row["derived"]["stage_count"]), "x_w": repr(row["derived"]["x_w"]), "wall_radius_m": repr(r_w),
            "l1a_wall_cusp_count": str(comparison["l1a_wall_cusp_count"]), "p2_wall_cusp_count": str(comparison["p2_wall_cusp_count"]),
            "count_agreement_strict": str(comparison["count_agreement_strict"]), "cusp_bijection": str(comparison["cusp_match"]["bijection"]),
            "matched_cusp_count": str(comparison["matched_cusp_count"]), "max_cusp_shift_m": repr(comparison["max_cusp_shift_m"]),
            "cusp_position_tolerance_m": repr(comparison["cusp_position_tolerance_m"]), "p2_hemp_like": str(comparison["p2_hemp_like_all_cusps"]),
            "l1a_hemp_like": str(comparison["l1a_hemp_like_all_cusps"]), "p2_level0_dofs": str(levels[0]["p2_dofs"]), "p2_level1_dofs": str(levels[1]["p2_dofs"]),
            "p2_all_levels_converged": str(row["p2"]["all_levels_converged"]), "sampling_stable": str(sampling["stable"]),
            "cusp_wall_b_ratios": ";".join(repr(p["wall_b_ratio_p2_over_l1a"]) for p in pairs),
            "rho_conservative_ratios": ";".join(repr(p["rho_conservative_ratio_p2_over_l1a"]) for p in pairs),
            "peak_rss_bytes": str(row["p2"]["peak_rss_bytes"]),
        }
        for key, value in csv_expected.items():
            if csv_row[key] != value:
                raise ValueError(f"{design_id}: CSV column {key} differs from the dataset row")
        per_design[design_id] = {
            "short": _short(design_id), "stages": row["derived"]["stage_count"], "x_w": row["derived"]["x_w"], "rw_over_l": row["derived"]["wall_radius_over_pitch"],
            "l1a_cusps": comparison["l1a_wall_cusp_count"], "p2_cusps": comparison["p2_wall_cusp_count"], "matched": len(pairs),
            "max_shift_m": comparison["max_cusp_shift_m"], "max_shift_tol": comparison["max_cusp_shift_over_tolerance"], "tolerance_m": comparison["cusp_position_tolerance_m"],
            "channel_shift_m": channel["max_sorted_shift_m"], "wall_ratios": [p["wall_b_ratio_p2_over_l1a"] for p in pairs],
            "l1a_min_rho": comparison["l1a_min_rho_conservative"], "p2_min_rho": comparison["p2_min_rho_conservative"], "p2_hemp": comparison["p2_hemp_like_all_cusps"],
            "level1_dofs": levels[1]["p2_dofs"], "axis_ratio": comparison["axis_peak_b_ratio_p2_over_l1a"], "peak_wall_ratio": comparison["peak_wall_b_ratio_p2_over_l1a"],
            "representative": row["representative"],
        }
    if rho_threshold is None:
        raise ValueError("no design record carries the HEMP-like threshold")
    if solves_converged != 2 * len(designs):
        raise ValueError("not every level of every design converged")

    # ---- gates (b), (c), reported (d) and the verdict, recomputed ----------------------
    gate_b = confirmation["cusp_count_unchanged"]
    gate_c = confirmation["cusp_position_shift"]
    gate_d = confirmation["hemp_like_preserved"]
    if gate_b["agreeing_designs_strict"] != strict_agree or gate_b["agreeing_designs_boundary_tolerant"] != tolerant_agree or gate_b["design_count"] != len(designs):
        raise ValueError("gate (b) counts do not recompute from the rows")
    if not _close(gate_b["fraction_strict"], strict_agree / len(designs)) or not _close(gate_b["fraction_boundary_tolerant"], tolerant_agree / len(designs)) or gate_b["disagreeing_designs"] != []:
        raise ValueError("gate (b) fractions do not recompute")
    b_passed = gate_b["fraction_boundary_tolerant"] >= gate_b["pass_threshold"]
    if gate_b["passed"] is not b_passed or gate_b["comparator"] != ">=" or gate_b["pass_threshold"] != protocol["gates"]["confirmation"]["cusp_count_unchanged"]["pass_threshold"]:
        raise ValueError("gate (b) verdict does not recompute from its threshold")
    shifts = [pair["shift_m"] for pair in matched_pairs]
    shifts_over = [pair["shift_over_tolerance"] for pair in matched_pairs]
    tolerances = [item["tolerance_m"] for item in per_design.values()]
    if gate_c["matched_cusp_count"] != len(matched_pairs) or gate_c["all_designs_bijective"] is not (bijective == len(designs)) or gate_c["designs_exceeding_tolerance"] != [] or gate_c["non_bijective_designs"] != []:
        raise ValueError("gate (c) counts do not recompute from the matched pairs")
    _check_distribution(gate_c["shift_m"], shifts, "gate (c) shift")
    _check_distribution(gate_c["shift_over_tolerance"], shifts_over, "gate (c) shift over tolerance")
    _check_distribution(gate_c["tolerance_m"], tolerances, "gate (c) tolerance")
    if not _close(gate_c["max_shift_over_tolerance"], max(shifts_over)):
        raise ValueError("gate (c) maximum does not recompute")
    c_passed = bool(gate_c["all_designs_bijective"] and gate_c["max_shift_over_tolerance"] <= gate_c["pass_threshold"])
    if gate_c["passed"] is not c_passed or gate_c["comparator"] != "<=" or gate_c["pass_threshold"] != protocol["gates"]["confirmation"]["cusp_position_shift"]["pass_threshold"]:
        raise ValueError("gate (c) verdict does not recompute from its threshold")
    verdict = VERDICT if (b_passed and c_passed) else ("PARTIALLY_CONFIRMED" if (b_passed or c_passed) else "DISCONFIRMED")
    if verdict != campaign["verdict"] or campaign["status"] != f"accepted_l1b_confirmation_{verdict.lower()}":
        raise ValueError("the verdict does not recompute from the predeclared rule")
    if gate_d["preserved_count"] != preserved or not _close(gate_d["fraction"], preserved / len(designs)) or gate_d["design_count"] != len(designs) or gate_d["pass_threshold"] is not None or gate_d["passed"] is not None:
        raise ValueError("reported (d) counts do not recompute or are recorded as a gate")
    lost = [design_id for design_id, item in per_design.items() if not item["p2_hemp"]]
    if gate_d["lost_designs"] != lost or len(lost) != len(designs) - preserved:
        raise ValueError("reported (d) lost designs do not recompute")
    if len(lost) != 1:
        raise ValueError("the section's wording presumes exactly one design loses the HEMP-like flag")
    _check_distribution(gate_d["wall_b_ratio_p2_over_l1a_per_cusp"], [pair["wall_b_ratio_p2_over_l1a"] for pair in matched_pairs], "wall field ratio")
    _check_distribution(gate_d["rho_conservative_ratio_p2_over_l1a"], [pair["rho_conservative_ratio_p2_over_l1a"] for pair in matched_pairs], "rho ratio")
    _check_distribution(gate_d["peak_wall_b_ratio_p2_over_l1a"], [item["peak_wall_ratio"] for item in per_design.values()], "peak wall ratio")
    _check_distribution(gate_d["axis_peak_b_ratio_p2_over_l1a"], [item["axis_ratio"] for item in per_design.values()], "axis peak ratio")
    _check_distribution(gate_d["l1a_min_rho_conservative"], [item["l1a_min_rho"] for item in per_design.values()], "L1a minimum rho")
    _check_distribution(gate_d["p2_min_rho_conservative"], [item["p2_min_rho"] for item in per_design.values()], "P2 minimum rho")
    _check_distribution(gate_d["peak_wall_b_ratio_unscaled"], [row["comparison"]["peak_wall_b_ratio_unscaled"] for row in designs], "unscaled peak wall ratio")
    if gate_d["peak_wall_b_ratio_in_band_count"] != sum(1 for row in designs if row["comparison"]["peak_wall_b_ratio_in_band"]):
        raise ValueError("descriptive band count does not recompute")
    # Headline statistics recompute.
    if headline["channel_axis_null_bijection_count"] != channel_null_bijections or headline["channel_axis_null_count_equal_count"] != channel_count_equal or headline["axis_null_bijection_count"] != sum(1 for row in designs if row["comparison"]["axis_null_match"]["bijection"]):
        raise ValueError("axis-null headline counts do not recompute")
    _check_distribution(headline["channel_axis_null_sorted_shift_m"], channel_sorted_shifts, "channel null sorted shift")
    _check_distribution(headline["channel_axis_null_shift_m"], channel_matched_shifts, "channel null matched shift")
    _check_distribution(headline["max_axis_null_shift_m"], pooled_matched_shifts, "pooled axis null matched shift")
    _check_distribution(headline["outside_channel_axis_null_shift_m"], outside_shifts, "outside null shift")
    _check_distribution(headline["separatrix_lean_l1a_m"], lean_l1a, "L1a separatrix lean")
    _check_distribution(headline["separatrix_lean_p2_m"], lean_p2, "P2 separatrix lean")
    _check_distribution(headline["p2_discretisation_max_wall_intersection_shift_m"], disc_wall, "P2 discretisation wall shift")
    _check_distribution(headline["p2_total_seconds"], solve_seconds, "P2 seconds")
    _check_distribution(headline["p2_level_dofs"]["level_0"], [float(v) for v in level_dofs[0]], "level 0 DOFs")
    _check_distribution(headline["p2_level_dofs"]["level_1"], [float(v) for v in level_dofs[1]], "level 1 DOFs")
    _check_distribution(estimands["p2_iterations"]["level_0"], [float(v) for v in level_iterations[0]], "level 0 iterations")
    _check_distribution(estimands["p2_iterations"]["level_1"], [float(v) for v in level_iterations[1]], "level 1 iterations")
    _check_distribution(estimands["p2_discretisation_max_axis_null_shift_m"], disc_axis, "P2 discretisation axis shift")
    _check_distribution(estimands["p2_discretisation_rho_sensitivity_max"], disc_rho, "P2 discretisation rho sensitivity")
    _check_distribution(estimands["sampling_max_wall_intersection_shift_m"], sampling_wall, "sampling wall shift")
    _check_distribution(estimands["l1a_wall_b3_over_b1"], l1a_b3, "L1a wall harmonic")
    _check_distribution(estimands["p2_wall_b3_over_b1"], p2_b3, "P2 wall harmonic")
    _check_distribution(estimands["p2_peak_rss_bytes"], [float(v) for v in design_rss], "per-design RSS")
    if not _close(headline["p2_relative_true_residual_max"], max(residuals)) or headline["p2_discretisation_stable_count"] != len(designs) or headline["sampling_stable_count"] != len(designs):
        raise ValueError("residual or stability headline counts do not recompute")
    if estimands["l1a_wall_cusp_count_histogram"] != _histogram([item["l1a_cusps"] for item in per_design.values()]) or estimands["p2_wall_cusp_count_histogram"] != _histogram([item["p2_cusps"] for item in per_design.values()]):
        raise ValueError("cusp-count histograms do not recompute")
    if estimands["l1a_axis_null_count_histogram"] != _histogram([row["comparison"]["l1a_axis_null_count"] for row in designs]) or estimands["p2_axis_null_count_histogram"] != _histogram([row["comparison"]["p2_axis_null_count"] for row in designs]):
        raise ValueError("axis-null histograms do not recompute")
    if authorities["l1a_wall_cusp_count_histogram"] != estimands["l1a_wall_cusp_count_histogram"]:
        raise ValueError("the authorities' cusp histogram differs from the dataset")
    stage_histogram = _histogram([item["stages"] for item in per_design.values()])
    cusp_histogram = estimands["l1a_wall_cusp_count_histogram"]
    l1a_nulls_hist = estimands["l1a_axis_null_count_histogram"]
    p2_nulls_hist = estimands["p2_axis_null_count_histogram"]
    for key in ("angle_to_wall_normal_deg_l1a", "angle_to_wall_normal_deg_p2"):
        side = "l1a" if key.endswith("l1a") else "p2"
        _check_distribution(estimands[key], [pair[f"{side}_angle_to_wall_normal_deg"] for pair in matched_pairs], key)

    # ---- the v1 development rejection (lineage), verified against its own bundle ---------
    if v1_terminal["state"] != V1_TERMINAL_STATE or v1_terminal["counts"]["attempt_count"] != 1 or v1_terminal["counts"]["assessment_access_count"] != 0:
        raise ValueError("the v1 terminal record is not a single development rejection without assessment")
    v1_failed = v1_failures["failed"]
    if len(v1_failed) != v1_terminal["payload"]["failed_design_count"] or v1_terminal["payload"]["resolved_design_count"] + len(v1_failed) != len(designs):
        raise ValueError("the v1 failure count does not add up to the declared set")
    if any(item["stage"] != "resolve" or "minimum-angle rejection gate" not in item["reason"] or item["resource_blocked"] is not False for item in v1_failed):
        raise ValueError("a v1 failure is not the level-0 mesh-angle rejection before any solve")
    v1_failed_ids = [item["key"].split(":")[1] for item in v1_failed]
    v1_records = sorted(path for path in v1_bundle.hashes if path.startswith("artifacts/designs/") and path.endswith(".json") and not path.endswith(".sha256.json"))
    if len(v1_records) != v1_terminal["payload"]["resolved_design_count"] or any(_short(Path(path).stem) in {_short(d) for d in v1_failed_ids} for path in v1_records):
        raise ValueError("the v1 design records do not match the resolved count or include a rejected design")
    if v1_development["fields"]["accepted"] is not False or v1_development["fields"]["payload"] != v1_terminal["payload"]:
        raise ValueError("the v1 development phase does not record the rejection")
    if v1_transitions[6]["transition"] != "development-rejected" or v1_transitions[7]["details"]["state"] != V1_TERMINAL_STATE:
        raise ValueError("the v1 transition log does not end in development-rejected -> terminal")
    v1_execution_wall_s = (_utc(v1_transitions[7]["recorded_at_utc"]) - _utc(v1_transitions[1]["recorded_at_utc"])).total_seconds()
    if v1_protocol["experiment_id"] != V1_EXPERIMENT_ID or v1_protocol["p2"]["mesh"]["reject_below_angle_deg"] <= protocol["p2"]["mesh"]["reject_below_angle_deg"]:
        raise ValueError("the v1 protocol is not the stricter-gate predecessor")
    if v1_lock["experiment_id"] != V1_EXPERIMENT_ID or v1_lock["attempt"] != 1 or v1_lock["immutable"] is not True:
        raise ValueError("the v1 execution lock is not a single immutable attempt")
    predecessor = protocol["predecessor"]
    if predecessor["experiment_id"] != V1_EXPERIMENT_ID or predecessor["preregistration_commit"] != v1_lock["commit"] or predecessor["terminal_state"] != V1_TERMINAL_STATE:
        raise ValueError("the protocol's predecessor block does not name the v1 lock commit and terminal state")
    if v1_authorities["protocol_semantic_sha256"] != _semantic_sha256(v1_protocol) or v1_shakedown["design_count"] >= shakedown["design_count"]:
        raise ValueError("the v1 authorities or shakedown do not match their protocol or the v1.1 shakedown does not widen the set")
    protocol_diff = _diff_paths(v1_protocol, protocol)
    if tuple(protocol_diff) != ALLOWED_PROTOCOL_CHANGES:
        raise ValueError(f"the v1 -> v1.1 protocol differs outside the declared changes: {protocol_diff}")
    for block in ("comparison", "gates", "definition_v3_import", "design_sets", "p2/solver", "p2/materials", "p2/adaptivity", "p2/sampling", "p2/resources", "claim_boundary", "outputs"):
        a, b = v1_protocol, protocol
        for token in block.split("/"):
            a, b = a[token], b[token]
        if a != b:
            raise ValueError(f"protocol block {block} differs between v1 and v1.1")
    # The post-hoc rejection note names the recorded facts.
    for phrase, label in (
        (f"`{v1_lock['commit'][:8]}`", "the v1 preregistration commit"),
        (f"`{predecessor['result_commit']}`", "the v1 result commit"),
        (f"{v1_terminal['payload']['resolved_design_count']}/{len(designs)} designs", "the resolved count"),
        (f"failed at `{v1_failed[0]['stage']}` BEFORE any solve", "the failure stage"),
        (f"{v1_bundle.manifest['artifact_count'] - v1_bundle.directory_count + 1} files", "the v1 bundle file count"),
        (f"({v1_protocol['p2']['mesh']['reject_below_angle_deg']:.0f} deg, inherited", "the v1 angle gate"),
        ("No assessment, gates, verdict or dashboard exist for v1", "the absence of a verdict"),
        (f"{v1_terminal['payload']['stage_wall_s']:.0f} s", "the v1 stage wall time"),
        ("`development_rejection`", "the terminal state"),
    ):
        if phrase not in rejection_text:
            raise ValueError(f"POSTHOC_REJECTION.md does not name {label} as recorded ({phrase!r})")
    for design_id in v1_failed_ids:
        if _short(design_id) not in rejection_text or design_id not in rejection_text.replace("\n", " "):
            raise ValueError(f"POSTHOC_REJECTION.md does not name the rejected design {design_id}")
    if f"{v1_terminal['payload']['resolved_design_count']}/{len(designs)} designs resolved" not in predecessor["statement"] or not all(_short(design_id) in predecessor["statement"] for design_id in v1_failed_ids):
        raise ValueError("predecessor statement does not carry the resolved count and the rejected designs")
    # Sliver record of the two v1-rejected designs on the v1.1 meshes.
    slivers: dict[str, list[dict[str, Any]]] = {}
    for design_id in v1_failed_ids:
        record = bundle.load(next(row["record_path"] for row in designs if row["design_id"] == design_id))
        slivers[design_id] = [
            {"level": level["level"], "min_angle_deg": level["mesh_quality"]["minimum_angle_deg"], "below_threshold": level["mesh_quality"]["sliver"]["elements_below_threshold"], "elements": level["mesh_quality"]["sliver"]["element_count"]}
            for level in record["evidence"]["p2"]["levels"]
        ]
        preflight_entry = next(item for item in preflight["designs"] if item["design_id"] == design_id)
        if not _close(preflight_entry["minimum_angle_deg"], slivers[design_id][0]["min_angle_deg"]) or preflight_entry["sliver"]["elements_below_threshold"] != slivers[design_id][0]["below_threshold"]:
            raise ValueError(f"{design_id}: the shakedown mesh preflight differs from the recorded level-0 mesh")
        if slivers[design_id][0]["min_angle_deg"] >= v1_protocol["p2"]["mesh"]["reject_below_angle_deg"]:
            raise ValueError(f"{design_id}: the level-0 mesh does not fall below the v1 gate, so the v1 rejection is not reproduced")
    if dashboard["angle_gate_designs_below_ten_degrees"] != sorted(v1_failed_ids):
        raise ValueError("the dashboard's sliver designs differ from the v1-rejected designs")
    sliver_a, sliver_b = (slivers[design_id] for design_id in v1_failed_ids)

    # ======================================================================= macros ====
    row_inputs = [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/designs"}]
    pair_inputs = [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/designs"}, {"artifact": "artifacts/gates.json", "pointer": "/confirmation/cusp_position_shift"}]

    # ---- identity, labels, outcome ----
    m.add("HmcExperimentId", "artifacts/campaign-result.json", "/experiment_id", "ident", "experiment identifier")
    m.add("HmcClassification", "artifacts/campaign-result.json", "/classification", "ident", "classification string sealed with the campaign")
    m.add("HmcTopologyLabel", "artifacts/campaign-result.json", "/topology_label", "ident", "topology label carried by every record")
    m.add_derived("HmcRecordedOutcome", RECORDED_OUTCOME, "ident", "recorded outcome at which the gate admits the study", "constant of the admission (paper/evidence/result-gates.json)", [{"artifact": "artifacts/campaign-result.json", "pointer": "/status"}])
    m.add("HmcCampaignStatus", "artifacts/campaign-result.json", "/status", "ident", "sealed campaign status")
    m.add("HmcTerminalState", "terminal.json", "/state", "ident", "terminal state of the single execution")
    m.add("HmcVerdict", "artifacts/campaign-result.json", "/verdict", "ident", "recorded confirmation verdict")
    m.add("HmcVerdictRule", "artifacts/gates.json", "/confirmation/verdict_rule", "text", "predeclared verdict rule")
    m.add_derived("HmcScreeningModel", SCREENING_MODEL, "text", "screening model statement of the admission", "constant of the admission (paper/evidence/manifests)", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/classification_statement"}])
    m.add("HmcPlanKind", "artifacts/campaign-result.json", "/plan_kind", "ident", "plan kind")
    m.add("HmcEvidentiary", "artifacts/campaign-result.json", "/evidentiary", "bool", "the single execution is the evidentiary run")
    m.add("HmcPaperAdmissionRecord", "artifacts/campaign-result.json", "/paper_admission", "text", "the campaign's own statement about paper admission, as sealed")
    m.add("HmcFieldLevelStatement", "artifacts/confirmation-dataset.json", "/claim_boundary/field_level", "text", "field level statement of the sealed claim boundary")
    m.add("HmcWhatConfirmed", "artifacts/confirmation-dataset.json", "/claim_boundary/what_is_confirmed_or_disconfirmed", "text", "what the campaign confirms or disconfirms")
    m.add("HmcWhatNotClaimed", "artifacts/confirmation-dataset.json", "/claim_boundary/what_is_not_claimed", "text", "what the campaign does not claim")
    m.add("HmcForbidPlasmaPerformance", "artifacts/confirmation-dataset.json", "/claim_boundary/forbid_plasma_performance_publication", "bool", "plasma or performance publication forbidden")
    m.add("HmcForbidMirrorProbability", "artifacts/confirmation-dataset.json", "/claim_boundary/forbid_mirror_probability_publication", "bool", "mirror-probability publication forbidden")
    m.add("HmcMirrorDescriptorsNotProbabilities", "artifacts/confirmation-dataset.json", "/claim_boundary/mirror_ratios_are_field_descriptors_not_probabilities", "bool", "mirror ratios are field descriptors, not probabilities")
    m.add("HmcShakedownNotEvidence", "artifacts/confirmation-dataset.json", "/claim_boundary/shakedown_outcomes_are_not_evidence", "bool", "shakedown outcomes are not evidence")
    m.add_derived("HmcFieldModelLevelLOneA", "L1a", "symbol", "field level token of the linear-vacuum reference records", "fixed token of the classification statement (linear-vacuum equivalent-current FDM = L1a)", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/classification_statement"}])
    m.add_derived("HmcTopologyVersion", "v3.1", "symbol", "version token of the imported cusp definition", "token of protocol.definition_v3_import.source (cusp_topology_search_v3_1)", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3_import/source"}])
    m.add_derived("HmcSweepVersion", "v3", "symbol", "version token of the sweep whose HEMP-like designs are confirmed", "token of protocol.design_sets.hemp_like_v3.source (l1a_geometry_sweep_v3)", [{"artifact": "artifacts/protocol.json", "pointer": "/design_sets/hemp_like_v3/source"}])
    m.add_derived("HmcVOneVersion", "v1", "symbol", "version token of the rejected predecessor", "token of protocol.predecessor.experiment_id", [{"artifact": "artifacts/protocol.json", "pointer": "/predecessor/experiment_id"}])
    m.add_derived("HmcVersion", "v1.1", "symbol", "version token of this campaign", "token of campaign-result.experiment_id", [{"artifact": "artifacts/campaign-result.json", "pointer": "/experiment_id"}])
    m.add_derived("HmcIOne", "I_1", "symbol", "modified Bessel function symbol of the single-harmonic prediction", "fixed symbol", [{"artifact": "artifacts/protocol.json", "pointer": "/purpose"}])
    m.add_derived("HmcBRSymbol", "B_r", "symbol", "radial field component symbol", "fixed symbol", [{"artifact": "artifacts/protocol.json", "pointer": "/p2/sampling/statement"}])
    m.add_derived("HmcBZSymbol", "B_z", "symbol", "axial field component symbol", "fixed symbol", [{"artifact": "artifacts/protocol.json", "pointer": "/p2/sampling/statement"}])
    m.add_derived("HmcLevelZeroToken", COUNT_TOKENS[0].lower(), "text", "name of the coarse P2 level", "fixed token", [{"artifact": "artifacts/protocol.json", "pointer": "/p2/adaptivity/levels"}])
    m.add_derived("HmcLevelOneToken", COUNT_TOKENS[1].lower(), "text", "name of the accepted P2 level", "fixed token", [{"artifact": "artifacts/protocol.json", "pointer": "/p2/adaptivity/levels"}])

    # ---- revisions and hashes ----
    m.add_derived("HmcResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results (record) commit prefix", "constant of the admission, verified as the commit at which the results tree first exists", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("HmcPreregCommit", PREREGISTRATION_COMMIT_SHA, "sha_short", "rebased preregistration commit prefix (carries the code whose hashes the bundle sealed)", "constant of the admission, verified by recomputing the sealed source hashes from its blobs", [{"artifact": "artifacts/source-binding.json", "pointer": "/experiment_code_sha256"}])
    m.add("HmcLockCommit", "execution-lock.json", "/commit", "sha_short", "pre-rebase preregistration commit named by the execution lock (origin/exp/l1b-hemp-confirmation-v1)")
    m.add_derived("HmcCodeCommit", CODE_COMMIT_SHA, "sha_short", "commit of the v1.1 code, tests and the v1 post-hoc rejection note", "constant of the admission", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("HmcDashboardCommit", DASHBOARD_COMMIT_SHA, "sha_short", "dashboard commit prefix", "constant of the admission", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("HmcVOneResultsCommit", V1_RESULTS_COMMIT_SHA, "sha_short", "rebased v1 record commit prefix", "constant of the admission", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("HmcVOnePreregCommit", V1_PREREGISTRATION_COMMIT_SHA, "sha_short", "rebased v1 preregistration commit prefix", "constant of the admission", [{"artifact": "manifest.json", "pointer": ""}])
    m.add("HmcVOneLockCommit", "artifacts/protocol.json", "/predecessor/preregistration_commit", "sha_short", "pre-rebase v1 preregistration commit named by the v1 lock and the protocol's predecessor block")
    m.add("HmcVOneRecordedResultCommit", "artifacts/protocol.json", "/predecessor/result_commit", "sha_short", "pre-rebase v1 result commit prefix named by the protocol's predecessor block")
    m.add_derived("HmcSweepResultsCommit", SWEEP_V3_RESULTS_COMMIT_SHA, "sha_short", "sweep-v3 results commit at which the catalogue and manifest are bound", "constant of the admission", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/sealed_sources/l1a_geometry_sweep_v3"}])
    m.add("HmcSweepPreregCommit", "artifacts/confirmation-dataset.json", "/sealed_sources/l1a_geometry_sweep_v3/preregistration_commit", "sha_short", "sweep-v3 preregistration commit sealed as source")
    m.add_derived("HmcTopologyResultsCommit", CUSP_TOPOLOGY_RESULTS_COMMIT_SHA, "sha_short", "cusp topology v3.1 results commit at which the frozen definition protocol is bound", "constant of the admission", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3_import"}])
    m.add_derived("HmcManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add("HmcTerminalSha", "manifest.json", "/terminal_byte_sha256", "sha_short", "terminal record SHA-256 prefix")
    m.add("HmcProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix (recomputed)")
    m.add("HmcExperimentCodeSha", "artifacts/authorities.json", "/experiment_code_sha256", "sha_short", "experiment code hash prefix (recomputed from the preregistration blobs)")
    m.add("HmcDependencySourceSha", "artifacts/authorities.json", "/dependency_source_sha256", "sha_short", "dependency source hash prefix (recomputed from the preregistration blobs)")
    m.add("HmcFieldPipelineSha", "artifacts/authorities.json", "/field_pipeline_source_sha256", "sha_short", "field pipeline source hash prefix (recomputed from the preregistration blobs)")
    m.add_derived("HmcExperimentCodeFiles", len(source_binding["experiment_code_files"]), "int", "experiment code files hashed", "len(source-binding.experiment_code_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/experiment_code_files"}])
    m.add_derived("HmcDependencySourceFiles", len(source_binding["dependency_source_files"]), "int", "dependency source files hashed", "len(source-binding.dependency_source_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/dependency_source_files"}])
    m.add_derived("HmcFieldPipelineFiles", len(source_binding["field_pipeline_source_files"]), "int", "field pipeline source files hashed", "len(source-binding.field_pipeline_source_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/field_pipeline_source_files"}])
    m.add("HmcSweepCatalogueSha", "artifacts/confirmation-dataset.json", "/sealed_sources/l1a_geometry_sweep_v3/catalogue_byte_sha256", "sha_short", "sealed sweep-v3 catalogue hash prefix (the bound file hashes to it)")
    m.add("HmcSweepManifestSha", "artifacts/confirmation-dataset.json", "/sealed_sources/l1a_geometry_sweep_v3/manifest_file_sha256", "sha_short", "sealed sweep-v3 manifest hash prefix (the bound file hashes to it)")
    m.add_derived("HmcVerifiedFiles", len(bundle.hashes), "int", "bundle files verified byte for byte", "count of manifest file entries verified", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add("HmcArtifactCount", "manifest.json", "/artifact_count", "int", "manifest artifact entries (files and directories)")
    m.add_derived("HmcToleratedEolFiles", 0, "int", "files accepted through an end-of-line tolerance", "none: every file hashes to its recorded byte_sha256", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("HmcTransitionCount", len(transitions), "int", "transitions in the lifecycle log", "count of transition records lock-acquired -> terminal", [{"artifact": "transitions/0009-terminal.json", "pointer": "/sequence"}])
    m.add_derived("HmcRecordCommitFiles", binding["results_commit_files"], "int", "files added by the record commit (results tree only)", "git diff --name-only <record>~1 <record>", [{"artifact": "manifest.json", "pointer": "/artifacts"}])

    # ---- execution ----
    m.add("HmcAttemptCount", "terminal.json", "/counts/attempt_count", "int", "attempts")
    m.add("HmcLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutable")
    m.add("HmcCleanWorktree", "execution-lock.json", "/clean_worktree_attested", "bool", "clean detached worktree attested at lock")
    m.add("HmcDevice", "execution-lock.json", "/device", "ident", "device record of the execution")
    m.add("HmcWorkerPool", "artifacts/campaign-result.json", "/execution_mode/worker_pool_size", "int", "worker pool size (one design at a time)")
    m.add("HmcCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "CPU cores of the host")
    m.add("HmcBlasThreads", "artifacts/runtime.json", "/blas_threads/OPENBLAS_NUM_THREADS", "text", "BLAS threads pinned for bitwise determinism")
    m.add("HmcStageWallS", "artifacts/campaign-result.json", "/execution_mode/stage_wall_s", "sec0", "design stage wall time (s)")
    m.add("HmcStageWallMin", "artifacts/campaign-result.json", "/execution_mode/stage_wall_s", "min1", "design stage wall time (min)")
    m.add("HmcAssessmentWallS", "artifacts/campaign-result.json", "/execution_mode/assessment_wall_s", "sec0", "assessment wall time (s)")
    m.add("HmcAssessmentWallMin", "artifacts/campaign-result.json", "/execution_mode/assessment_wall_s", "min1", "assessment wall time (min)")
    m.add_derived("HmcExecutionWallMin", execution_wall_s, "min1", "lock-acquired to terminal wall time (min)", "difference of the transition timestamps 0009 - 0001", [{"artifact": "transitions/0001-lock-acquired.json", "pointer": "/recorded_at_utc"}, {"artifact": "transitions/0009-terminal.json", "pointer": "/recorded_at_utc"}])
    m.add("HmcPeakRssMb", "artifacts/gates.json", "/peak_rss_bytes", "mb0", "peak process RSS over the campaign (MB)")
    m.add("HmcRamBudgetGb", "artifacts/gates.json", "/ram_budget/budget_bytes", "gb1", "RAM budget (GB)")
    m.add("HmcRamFreeAtStartGb", "artifacts/gates.json", "/ram_budget/free_at_start_bytes", "gb1", "free physical RAM at campaign start (GB)")
    m.add("HmcRamBudgetFraction", "artifacts/gates.json", "/ram_budget/fraction", "g", "budget fraction of free RAM")
    m.add("HmcRamFractionUsed", "artifacts/confirmation-dataset.json", "/headline/ram_budget_fraction_used", "pct1", "peak RSS as a share of the budget")
    m.add("HmcDofCap", "artifacts/gates.json", "/ram_budget/maximum_p2_dofs", "int_comma", "P2 DOF cap")
    m.add("HmcGpuNotUsed", "artifacts/protocol.json", "/p2/resources/cpu_only", "bool", "CPU only (GPU not used)")
    m.add("HmcPythonVersion", "artifacts/runtime.json", "/python", "text", "Python version string of the execution")
    m.add("HmcNumpyVersion", "artifacts/runtime.json", "/numpy", "text", "numpy version of the execution")

    # ---- design set ----
    m.add("HmcDesignCount", "artifacts/campaign-result.json", "/design_count", "int", "designs confirmed")
    m.add("HmcDeclaredDesigns", "artifacts/protocol.json", "/design_sets/hemp_like_v3/design_count", "int", "declared designs")
    m.add_derived("HmcFailedDesigns", len(failures["failed"]), "int", "failed designs", "len(design-failures.failed)", [{"artifact": "artifacts/design-failures.json", "pointer": "/failed"}])
    m.add("HmcResolvedDesigns", "phases/development.json", "/fields/payload/resolved_design_count", "int", "resolved designs")
    m.add_derived("HmcRepresentatives", sum(1 for item in per_design.values() if item["representative"]), "int", "representative designs with stored traces", "count of representative rows", row_inputs)
    m.add_derived("HmcRepresentativeIds", [d for d in declared_ids if per_design[d]["representative"]], "list_short_designs", "representative design ordinals", "design ordinals of the representative rows (protocol order)", row_inputs)
    m.add("HmcHempLikeRule", "artifacts/protocol.json", "/design_sets/hemp_like_v3/rule", "text", "HEMP-like rule of the sealed catalogue")
    m.add_derived("HmcRhoThreshold", rho_threshold, "fixed1", "HEMP-like threshold on rho (equal in every design record)", "design record descriptors.accepted.hemp_like_threshold.rho, verified equal across records", [{"artifact": designs[0]["record_path"], "pointer": "/descriptors/accepted/hemp_like_threshold/rho"}])
    m.add("HmcSweepHempLikeCount", "artifacts/confirmation-dataset.json", "/sealed_sources/l1a_geometry_sweep_v3/hemp_like_design_count", "int", "HEMP-like designs of the sealed sweep-v3 catalogue")
    for count, token in ((3, "Three"), (4, "Four"), (5, "Five")):
        m.add_derived(f"HmcStage{token}Designs", stage_histogram.get(str(count), 0), "int", f"designs with {count} magnet stages", "histogram of derived.stage_count", row_inputs)
    for count, token in ((2, "Two"), (3, "Three"), (4, "Four")):
        m.add_derived(f"HmcCusp{token}Designs", cusp_histogram.get(str(count), 0), "int", f"designs with {count} wall cusps (both field models)", "estimands.l1a_wall_cusp_count_histogram (equal to the P2 histogram)", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/estimands/l1a_wall_cusp_count_histogram"}])
    m.add("HmcMatchedCusps", "artifacts/gates.json", "/confirmation/cusp_position_shift/matched_cusp_count", "int", "matched wall cusps over every design")
    m.add_derived("HmcXwMin", min(item["x_w"] for item in per_design.values()), "fixed2", "smallest x_w of the set", "min over rows of derived.x_w (recomputed as pi r_w / L)", row_inputs)
    m.add_derived("HmcXwMax", max(item["x_w"] for item in per_design.values()), "fixed2", "largest x_w of the set", "max over rows of derived.x_w", row_inputs)
    m.add_derived("HmcRwOverLMin", min(item["rw_over_l"] for item in per_design.values()), "fixed3", "smallest r_w / L of the set", "min over rows of derived.wall_radius_over_pitch", row_inputs)
    m.add_derived("HmcRwOverLMax", max(item["rw_over_l"] for item in per_design.values()), "fixed3", "largest r_w / L of the set", "max over rows of derived.wall_radius_over_pitch", row_inputs)
    m.add_derived("HmcLOneARhoMinMin", min(item["l1a_min_rho"] for item in per_design.values()), "fixed3", "smallest per-design minimum rho under L1a", "min over rows of comparison.l1a_min_rho_conservative", row_inputs)
    m.add_derived("HmcLOneARhoMinMax", max(item["l1a_min_rho"] for item in per_design.values()), "fixed3", "largest per-design minimum rho under L1a", "max over rows of comparison.l1a_min_rho_conservative", row_inputs)

    # ---- P2 model ----
    m.add("HmcIronMuR", "artifacts/protocol.json", "/p2/materials/soft_iron_relative_permeability", "fixed0", "soft-iron relative permeability")
    m.add("HmcMagnetRecoilMuR", "artifacts/protocol.json", "/p2/materials/magnet_recoil_relative_permeability", "fixed2", "magnet recoil relative permeability")
    m.add("HmcMaterialsStatement", "artifacts/protocol.json", "/p2/materials/statement", "text", "materials statement of the frozen protocol")
    m.add("HmcVacuumLikeRegions", "artifacts/protocol.json", "/p2/materials/vacuum_like_regions", "text", "regions at unit permeability")
    m.add("HmcBoreElements", "artifacts/protocol.json", "/p2/mesh/bore_elements", "int", "level-0 elements across the bore radius")
    m.add("HmcFeatureElements", "artifacts/protocol.json", "/p2/mesh/feature_elements", "int", "level-0 elements across the thinnest feature")
    m.add("HmcPaddingFactor", "artifacts/protocol.json", "/p2/mesh/padding_factor", "g", "outer-boundary padding factor")
    m.add("HmcAngleGateDeg", "artifacts/protocol.json", "/p2/mesh/reject_below_angle_deg", "fixed0", "level-0 minimum-angle rejection gate of v1.1 (degrees)")
    v1.add("HmcVOneAngleGateDeg", "artifacts/protocol.json", "/p2/mesh/reject_below_angle_deg", "fixed0", "level-0 minimum-angle rejection gate of v1 (degrees)")
    m.add("HmcAdaptiveLevels", "artifacts/protocol.json", "/p2/adaptivity/levels", "int", "nested adaptive levels")
    m.add("HmcDorflerTheta", "artifacts/protocol.json", "/p2/adaptivity/dorfler_theta", "g", "Dorfler marking fraction")
    m.add("HmcSolverRelTol", "artifacts/protocol.json", "/p2/solver/relative_tolerance", "sci1", "relative true-residual tolerance of every solve")
    m.add("HmcSolverMaxIter", "artifacts/protocol.json", "/p2/solver/max_iterations", "int_comma", "iteration cap of every solve")
    m.add("HmcSolverBackend", "artifacts/protocol.json", "/p2/solver/backend", "text", "solver backend")
    m.add("HmcRadialIntervals", "artifacts/protocol.json", "/p2/sampling/radial_intervals", "int", "radial sampling intervals across the bore")
    m.add("HmcSamplingRefinement", "artifacts/protocol.json", "/p2/sampling/refinement", "int", "refinement factor of the second sampling")
    m.add_derived("HmcLevelZeroDofsMin", min(level_dofs[0]), "int_comma", "smallest level-0 DOF count", "min over rows of p2.levels[0].p2_dofs", row_inputs)
    m.add_derived("HmcLevelZeroDofsMax", max(level_dofs[0]), "int_comma", "largest level-0 DOF count", "max over rows of p2.levels[0].p2_dofs", row_inputs)
    m.add_derived("HmcLevelOneDofsMin", min(level_dofs[1]), "int_comma", "smallest level-1 DOF count", "min over rows of p2.levels[1].p2_dofs", row_inputs)
    m.add_derived("HmcLevelOneDofsMax", max(level_dofs[1]), "int_comma", "largest level-1 DOF count", "max over rows of p2.levels[1].p2_dofs", row_inputs)
    m.add_derived("HmcLevelOneDofsMedian", statistics.median(level_dofs[1]), "int_comma", "median level-1 DOF count", "median over rows of p2.levels[1].p2_dofs", row_inputs)
    m.add_derived("HmcLevelZeroItersMin", min(level_iterations[0]), "int_comma", "fewest level-0 PCG iterations", "min over rows of p2.levels[0].iterations", row_inputs)
    m.add_derived("HmcLevelZeroItersMax", max(level_iterations[0]), "int_comma", "most level-0 PCG iterations", "max over rows of p2.levels[0].iterations", row_inputs)
    m.add_derived("HmcLevelOneItersMin", min(level_iterations[1]), "int_comma", "fewest level-1 PCG iterations", "min over rows of p2.levels[1].iterations", row_inputs)
    m.add_derived("HmcLevelOneItersMax", max(level_iterations[1]), "int_comma", "most level-1 PCG iterations", "max over rows of p2.levels[1].iterations", row_inputs)
    m.add("HmcResidualMax", "artifacts/confirmation-dataset.json", "/headline/p2_relative_true_residual_max", "sci2", "largest relative true residual over every solve")
    m.add_derived("HmcSolvesConverged", solves_converged, "int", "converged P2 solves", "count over rows and levels of converged == true", row_inputs)
    m.add_derived("HmcSolvesTotal", 2 * len(designs), "int", "P2 solves (two levels per design)", "2 x design count", row_inputs)
    m.add_derived("HmcSolveSecondsMin", min(solve_seconds), "sec0", "shortest per-design P2 time (s)", "min over rows of p2.total_seconds", row_inputs)
    m.add_derived("HmcSolveSecondsMedian", statistics.median(solve_seconds), "sec0", "median per-design P2 time (s)", "median over rows of p2.total_seconds", row_inputs)
    m.add_derived("HmcSolveSecondsMax", max(solve_seconds), "sec0", "longest per-design P2 time (s)", "max over rows of p2.total_seconds", row_inputs)
    m.add_derived("HmcSolveSecondsTotal", sum(solve_seconds), "sec0", "summed per-design P2 time (s)", "sum over rows of p2.total_seconds", row_inputs)
    m.add_derived("HmcDesignRssMaxMb", max(design_rss), "mb0", "largest per-design peak RSS (MB)", "max over rows of p2.peak_rss_bytes", row_inputs)
    m.add_derived("HmcRegionCount", len(designs[0]["p2"]["regions"]), "int", "material regions of the first design", "len(designs[0].p2.regions)", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/designs/0/p2/regions"}])

    # ---- definition import and comparison rule ----
    m.add("HmcDefinitionSource", "artifacts/protocol.json", "/definition_v3_import/source", "text", "imported definition source")
    m.add("HmcStabilityToleranceMm", "artifacts/protocol.json", "/definition_v3_import/stability_tolerance_m", "mm2", "stability tolerance of the imported definition (mm)")
    m.add("HmcBoundaryAmbiguityMm", "artifacts/protocol.json", "/definition_v3_import/numerical_parameters/boundary_ambiguity_tolerance_m", "mm2", "boundary-ambiguity tolerance (mm)")
    m.add("HmcWallSamplesPerCell", "artifacts/protocol.json", "/definition_v3_import/numerical_parameters/wall_samples_per_cell", "int", "wall samples per cell")
    m.add("HmcFluxRootToleranceUm", "artifacts/protocol.json", "/definition_v3_import/numerical_parameters/trace_flux_root_tolerance_m", "um0", "trace-to-flux-root tolerance (um)")
    m.add_derived("HmcDefinitionParametersEqual", True, "bool", "imported numerical parameters equal the frozen cusp topology v3.1 protocol", "protocol.definition_v3_import.numerical_parameters == cusp_topology_search_v3_1/protocol.json#definition_v3/numerical_parameters at the bound revision", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3_import/numerical_parameters"}])
    m.add("HmcToleranceRule", "artifacts/protocol.json", "/comparison/cusp_position_tolerance_rule", "text", "cusp-position tolerance rule")
    m.add("HmcLOneADzMm", "artifacts/protocol.json", "/comparison/l1a_dz_m", "mm3", "L1a axial grid step (mm)")
    m.add("HmcToleranceMinMm", "artifacts/gates.json", "/confirmation/cusp_position_shift/tolerance_m/min", "mm2", "smallest design tolerance (mm)")
    m.add("HmcToleranceMaxMm", "artifacts/gates.json", "/confirmation/cusp_position_shift/tolerance_m/max", "mm2", "largest design tolerance (mm)")
    m.add("HmcToleranceMedianMm", "artifacts/gates.json", "/confirmation/cusp_position_shift/tolerance_m/median", "mm2", "median design tolerance (mm)")
    m.add("HmcCountAgreementRule", "artifacts/protocol.json", "/comparison/count_agreement", "text", "strict and boundary-tolerant count agreement rule")
    m.add("HmcMatchingRule", "artifacts/protocol.json", "/comparison/matching", "text", "cusp matching rule")

    # ---- binding gates ----
    m.add_derived("HmcBindingGateCount", len(BINDING_GATE_NAMES), "int", "binding integrity gates", "len(gates.campaign)", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("HmcBindingGatesTrue", sum(1 for name in BINDING_GATE_NAMES if gates["campaign"][name] is True), "int", "binding integrity gates true", "count of true values in gates.campaign", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add("HmcGatesPassed", "artifacts/gates.json", "/passed", "bool", "every binding gate passed")
    m.add_derived("HmcReplayDesigns", len(gates["replays"]), "int", "determinism replays", "len(gates.replays)", [{"artifact": "artifacts/gates.json", "pointer": "/replays"}])
    m.add_derived("HmcReplaysBitIdentical", sum(1 for item in gates["replays"] if item["bit_identical"]), "int", "bit-identical replays", "count of replays with bit_identical", [{"artifact": "artifacts/gates.json", "pointer": "/replays"}])
    m.add_derived("HmcReplayDesign", _short(gates["replays"][0]["key"].split(":")[1]), "ident", "replayed design ordinal", "gates.replays[0].key", [{"artifact": "artifacts/gates.json", "pointer": "/replays/0/key"}])
    m.add("HmcSamplingStable", "artifacts/confirmation-dataset.json", "/headline/sampling_stable_count", "int", "designs stable between the 1x and 2x sampling")
    m.add("HmcDiscStable", "artifacts/confirmation-dataset.json", "/headline/p2_discretisation_stable_count", "int", "designs stable between the level-0 and level-1 maps")
    m.add("HmcIdentityProven", "artifacts/gates.json", "/campaign/identity_proven", "bool", "identity proven for every design")
    m.add("HmcHashBindings", "artifacts/gates.json", "/campaign/hash_bindings", "bool", "hash bindings equal the frozen authorities")

    # ---- gate (b) ----
    m.add("HmcGateBAgreeStrict", "artifacts/gates.json", "/confirmation/cusp_count_unchanged/agreeing_designs_strict", "int", "designs with strict cusp-count agreement")
    m.add("HmcGateBAgreeTolerant", "artifacts/gates.json", "/confirmation/cusp_count_unchanged/agreeing_designs_boundary_tolerant", "int", "designs with boundary-tolerant cusp-count agreement")
    m.add("HmcGateBFractionStrict", "artifacts/gates.json", "/confirmation/cusp_count_unchanged/fraction_strict", "fixed2", "strict agreement fraction")
    m.add("HmcGateBThreshold", "artifacts/gates.json", "/confirmation/cusp_count_unchanged/pass_threshold", "fixed1", "gate (b) threshold")
    m.add("HmcGateBPassed", "artifacts/gates.json", "/confirmation/cusp_count_unchanged/passed", "bool", "gate (b) passed")
    m.add_derived("HmcGateBDisagreeing", len(gate_b["disagreeing_designs"]), "int", "designs whose cusp count changed", "len(disagreeing_designs)", [{"artifact": "artifacts/gates.json", "pointer": "/confirmation/cusp_count_unchanged/disagreeing_designs"}])
    m.add_derived("HmcCellCountsAgree", sum(1 for row in designs if row["comparison"]["cell_count_agreement"]), "int", "designs whose cell count agrees", "count over rows of comparison.cell_count_agreement", row_inputs)

    # ---- gate (c) ----
    m.add("HmcGateCMaxShiftOverTol", "artifacts/gates.json", "/confirmation/cusp_position_shift/max_shift_over_tolerance", "fixed2", "largest matched-cusp shift in tolerance units")
    m.add("HmcGateCThreshold", "artifacts/gates.json", "/confirmation/cusp_position_shift/pass_threshold", "fixed1", "gate (c) threshold in tolerance units")
    m.add("HmcGateCPassed", "artifacts/gates.json", "/confirmation/cusp_position_shift/passed", "bool", "gate (c) passed")
    m.add("HmcAllBijective", "artifacts/gates.json", "/confirmation/cusp_position_shift/all_designs_bijective", "bool", "every design's cusp matching is a bijection")
    m.add_derived("HmcBijectiveDesigns", bijective, "int", "designs whose cusp matching is a bijection", "count over rows of comparison.cusp_match.bijection", row_inputs)
    m.add("HmcShiftMaxMm", "artifacts/gates.json", "/confirmation/cusp_position_shift/shift_m/max", "mm3", "largest matched-cusp shift (mm)")
    m.add("HmcShiftMedianMm", "artifacts/gates.json", "/confirmation/cusp_position_shift/shift_m/median", "mm3", "median matched-cusp shift (mm)")
    m.add("HmcShiftMeanMm", "artifacts/gates.json", "/confirmation/cusp_position_shift/shift_m/mean", "mm3", "mean matched-cusp shift (mm)")
    m.add("HmcShiftMinUm", "artifacts/gates.json", "/confirmation/cusp_position_shift/shift_m/min", "um1", "smallest matched-cusp shift (um)")
    m.add("HmcShiftOverTolMedian", "artifacts/gates.json", "/confirmation/cusp_position_shift/shift_over_tolerance/median", "fixed2", "median shift in tolerance units")
    m.add("HmcShiftOverTolMean", "artifacts/gates.json", "/confirmation/cusp_position_shift/shift_over_tolerance/mean", "fixed2", "mean shift in tolerance units")
    m.add_derived("HmcDesignsExceeding", len(gate_c["designs_exceeding_tolerance"]), "int", "designs with a cusp beyond its tolerance", "len(designs_exceeding_tolerance)", [{"artifact": "artifacts/gates.json", "pointer": "/confirmation/cusp_position_shift/designs_exceeding_tolerance"}])
    max_pair = max(matched_pairs, key=lambda pair: pair["shift_m"])
    min_pair = min(matched_pairs, key=lambda pair: pair["shift_m"])
    m.add_derived("HmcMaxShiftDesign", _short(max_pair["design_id"]), "ident", "design of the largest cusp shift", "argmax over matched pairs of shift_m", pair_inputs)
    m.add_derived("HmcMinShiftDesign", _short(min_pair["design_id"]), "ident", "design of the smallest cusp shift", "argmin over matched pairs of shift_m", pair_inputs)
    m.add_derived("HmcShiftsAboveStability", shifts_above_stability, "int", "matched cusps whose shift exceeds the imported stability tolerance", "count over matched pairs of shift_m > definition stability tolerance", pair_inputs)
    if not shift_over_disc_all:
        raise ValueError("a matched-cusp shift lies within the P2 discretisation scale; the wording of the section presumes none does")
    m.add_derived("HmcShiftsAboveDiscretisation", len(matched_pairs), "int", "matched cusps whose shift exceeds the design's level-0 -> level-1 P2 discretisation shift", "count over matched pairs of shift_m > p2_discretisation.max_wall_intersection_shift_m (verified to be every pair)", pair_inputs)
    m.add_derived("HmcMaxShiftInBoreElements", max_pair["shift_m"] / (max_pair["tolerance_m"]), "fixed2", "largest shift in level-0 bore elements (tolerance = one bore element for that design)", "shift_m / tolerance_m of the largest pair (tolerance_m = r_w / bore elements when that exceeds the L1a step)", pair_inputs)
    if not _close(per_design[max_pair["design_id"]]["tolerance_m"], max(designs[declared_ids.index(max_pair["design_id"])]["geometry"]["wall_radius_m"] / bore_elements, l1a_dz)):
        raise ValueError("largest-shift design tolerance does not recompute")

    # ---- reported (d) ----
    m.add("HmcPreservedCount", "artifacts/gates.json", "/confirmation/hemp_like_preserved/preserved_count", "int", "designs that remain HEMP-like under P2")
    m.add("HmcPreservedFraction", "artifacts/gates.json", "/confirmation/hemp_like_preserved/fraction", "pct0", "share of designs that remain HEMP-like")
    m.add_derived("HmcLostDesigns", len(lost), "int", "designs that lose the HEMP-like flag under P2", "len(lost_designs)", [{"artifact": "artifacts/gates.json", "pointer": "/confirmation/hemp_like_preserved/lost_designs"}])
    m.add_derived("HmcLostDesign", _short(lost[0]), "ident", "ordinal of the design that loses the flag", "lost_designs[0]", [{"artifact": "artifacts/gates.json", "pointer": "/confirmation/hemp_like_preserved/lost_designs/0"}])
    m.add_derived("HmcLostDesignLOneARho", per_design[lost[0]]["l1a_min_rho"], "fixed3", "minimum rho of the lost design under L1a", "comparison.l1a_min_rho_conservative of the lost design", row_inputs)
    m.add_derived("HmcLostDesignPTwoRho", per_design[lost[0]]["p2_min_rho"], "fixed3", "minimum rho of the lost design under P2", "comparison.p2_min_rho_conservative of the lost design", row_inputs)
    m.add_derived("HmcLostDesignShiftOverTol", per_design[lost[0]]["max_shift_tol"], "fixed2", "largest cusp shift of the lost design in tolerance units", "comparison.max_cusp_shift_over_tolerance of the lost design", row_inputs)
    m.add("HmcRhoRatioMin", "artifacts/gates.json", "/confirmation/hemp_like_preserved/rho_conservative_ratio_p2_over_l1a/min", "fixed2", "smallest per-cusp rho ratio P2 / L1a")
    m.add("HmcRhoRatioMedian", "artifacts/gates.json", "/confirmation/hemp_like_preserved/rho_conservative_ratio_p2_over_l1a/median", "fixed2", "median per-cusp rho ratio")
    m.add("HmcRhoRatioMax", "artifacts/gates.json", "/confirmation/hemp_like_preserved/rho_conservative_ratio_p2_over_l1a/max", "fixed2", "largest per-cusp rho ratio")
    m.add_derived("HmcRhoRatioBelowOneCusps", sum(1 for pair in matched_pairs if pair["rho_conservative_ratio_p2_over_l1a"] < 1.0), "int", "matched cusps whose rho falls under iron", "count over matched pairs of rho ratio < 1", pair_inputs)
    m.add_derived("HmcRhoRatioBelowOneDesigns", sum(1 for pairs in pairs_by_design.values() if any(p["rho_conservative_ratio_p2_over_l1a"] < 1.0 for p in pairs)), "int", "designs with a cusp whose rho falls under iron", "count over designs with a matched pair of rho ratio < 1", pair_inputs)
    m.add("HmcWallBRatioMin", "artifacts/gates.json", "/confirmation/hemp_like_preserved/wall_b_ratio_p2_over_l1a_per_cusp/min", "fixed2", "smallest per-cusp wall field ratio P2 / L1a at equal magnet strength")
    m.add("HmcWallBRatioMedian", "artifacts/gates.json", "/confirmation/hemp_like_preserved/wall_b_ratio_p2_over_l1a_per_cusp/median", "fixed2", "median per-cusp wall field ratio")
    m.add("HmcWallBRatioMax", "artifacts/gates.json", "/confirmation/hemp_like_preserved/wall_b_ratio_p2_over_l1a_per_cusp/max", "fixed2", "largest per-cusp wall field ratio")
    m.add("HmcWallBRaiseMinPct", "artifacts/gates.json", "/confirmation/hemp_like_preserved/wall_b_ratio_p2_over_l1a_per_cusp/min", "ratio_pct0", "smallest wall field change at a cusp (percent)")
    m.add("HmcWallBRaiseMaxPct", "artifacts/gates.json", "/confirmation/hemp_like_preserved/wall_b_ratio_p2_over_l1a_per_cusp/max", "ratio_pct0", "largest wall field change at a cusp (percent)")
    m.add_derived("HmcWallBRatioAboveOneCusps", sum(1 for pair in matched_pairs if pair["wall_b_ratio_p2_over_l1a"] > 1.0), "int", "matched cusps whose wall field rises under iron", "count over matched pairs of wall field ratio > 1", pair_inputs)
    m.add("HmcPeakWallRatioMin", "artifacts/gates.json", "/confirmation/hemp_like_preserved/peak_wall_b_ratio_p2_over_l1a/min", "fixed2", "smallest per-design peak wall field ratio")
    m.add("HmcPeakWallRatioMedian", "artifacts/gates.json", "/confirmation/hemp_like_preserved/peak_wall_b_ratio_p2_over_l1a/median", "fixed2", "median per-design peak wall field ratio")
    m.add("HmcPeakWallRatioMax", "artifacts/gates.json", "/confirmation/hemp_like_preserved/peak_wall_b_ratio_p2_over_l1a/max", "fixed2", "largest per-design peak wall field ratio")
    m.add("HmcPeakWallUnscaledMin", "artifacts/gates.json", "/confirmation/hemp_like_preserved/peak_wall_b_ratio_unscaled/min", "fixed2", "smallest per-design peak wall field ratio before the magnet-strength scaling")
    m.add("HmcPeakWallUnscaledMax", "artifacts/gates.json", "/confirmation/hemp_like_preserved/peak_wall_b_ratio_unscaled/max", "fixed2", "largest per-design peak wall field ratio before the magnet-strength scaling")
    m.add("HmcAxisPeakRatioMin", "artifacts/gates.json", "/confirmation/hemp_like_preserved/axis_peak_b_ratio_p2_over_l1a/min", "fixed2", "smallest per-design axis peak ratio")
    m.add("HmcAxisPeakRatioMedian", "artifacts/gates.json", "/confirmation/hemp_like_preserved/axis_peak_b_ratio_p2_over_l1a/median", "fixed2", "median per-design axis peak ratio")
    m.add("HmcAxisPeakRatioMax", "artifacts/gates.json", "/confirmation/hemp_like_preserved/axis_peak_b_ratio_p2_over_l1a/max", "fixed2", "largest per-design axis peak ratio")
    m.add("HmcPTwoRhoMinMin", "artifacts/gates.json", "/confirmation/hemp_like_preserved/p2_min_rho_conservative/min", "fixed3", "smallest per-design minimum rho under P2")
    m.add("HmcPTwoRhoMinMax", "artifacts/gates.json", "/confirmation/hemp_like_preserved/p2_min_rho_conservative/max", "fixed3", "largest per-design minimum rho under P2")
    m.add("HmcPTwoRhoMinMedian", "artifacts/gates.json", "/confirmation/hemp_like_preserved/p2_min_rho_conservative/median", "fixed3", "median per-design minimum rho under P2")
    m.add("HmcLOneARhoMinMedian", "artifacts/gates.json", "/confirmation/hemp_like_preserved/l1a_min_rho_conservative/median", "fixed3", "median per-design minimum rho under L1a")
    m.add("HmcBandLo", "artifacts/confirmation-dataset.json", "/designs/0/comparison/wall_b_ratio_band_descriptive/0", "fixed1", "lower edge of the descriptive wall field ratio band")
    m.add("HmcBandHi", "artifacts/confirmation-dataset.json", "/designs/0/comparison/wall_b_ratio_band_descriptive/1", "fixed1", "upper edge of the descriptive wall field ratio band")
    m.add("HmcInBandCount", "artifacts/gates.json", "/confirmation/hemp_like_preserved/peak_wall_b_ratio_in_band_count", "int", "designs inside the descriptive band")
    m.add("HmcReportedDEstimand", "artifacts/gates.json", "/definitions/confirmation/hemp_like_preserved/estimand", "text", "estimand statement of reported (d)")
    m.add_derived("HmcPTwoWallBelowOneCusps", sum(1 for row in designs for item in row["p2_rho"] if item["rho_wall"] < 1.0), "int", "P2 cusps whose wall reading is below one (no cusp is the wall field maximum)", "count over rows of p2_rho.rho_wall < 1", row_inputs)
    m.add_derived("HmcPTwoCuspsTotal", sum(len(row["p2_rho"]) for row in designs), "int", "P2 wall cusps over every design", "sum over rows of len(p2_rho)", row_inputs)
    m.add_derived("HmcPTwoRhoWallMax", max(item["rho_wall"] for row in designs for item in row["p2_rho"]), "fixed2", "largest P2 wall reading", "max over rows of p2_rho.rho_wall", row_inputs)
    m.add_derived("HmcCuspIsWallMaximumPTwo", sum(1 for row in designs for item in row["p2_rho"] if item["cusp_is_wall_maximum"]), "int", "P2 cusps that are the wall field maximum of their neighbourhood", "count over rows of p2_rho.cusp_is_wall_maximum", row_inputs)

    # ---- axis nulls and separatrix lean ----
    m.add("HmcChannelNullCountEqual", "artifacts/confirmation-dataset.json", "/headline/channel_axis_null_count_equal_count", "int", "designs with the same channel axis-null count under both fields")
    m.add("HmcChannelNullBijection", "artifacts/confirmation-dataset.json", "/headline/channel_axis_null_bijection_count", "int", "designs whose channel axis nulls are in bijection within the cusp tolerance")
    m.add("HmcPooledNullBijection", "artifacts/confirmation-dataset.json", "/headline/axis_null_bijection_count", "int", "designs whose pooled axis nulls (all zones) are in bijection")
    m.add("HmcChannelNullShiftMaxMm", "artifacts/confirmation-dataset.json", "/headline/channel_axis_null_sorted_shift_m/max", "mm2", "largest sorted channel axis-null shift (mm)")
    m.add("HmcChannelNullShiftMedianMm", "artifacts/confirmation-dataset.json", "/headline/channel_axis_null_sorted_shift_m/median", "mm2", "median sorted channel axis-null shift (mm)")
    m.add("HmcChannelNullShiftMinUm", "artifacts/confirmation-dataset.json", "/headline/channel_axis_null_sorted_shift_m/min", "um1", "smallest sorted channel axis-null shift (um)")
    m.add_derived("HmcChannelNullsBeyondTolDesigns", sum(1 for row in designs if row["comparison"]["channel_axis_nulls"]["max_sorted_shift_m"] > row["comparison"]["cusp_position_tolerance_m"]), "int", "designs whose channel axis nulls move beyond the cusp tolerance", "count over rows of channel_axis_nulls.max_sorted_shift_m > cusp_position_tolerance_m", row_inputs)
    m.add_derived("HmcMorePTwoNullDesigns", more_p2_nulls, "int", "designs with more axis nulls under P2 than under L1a", "count over rows of p2_axis_null_count > l1a_axis_null_count", row_inputs)
    m.add("HmcOutsideNullShiftMinMm", "artifacts/confirmation-dataset.json", "/headline/outside_channel_axis_null_shift_m/min", "mm2", "smallest shift of an axis null outside the straight section (mm)")
    m.add("HmcOutsideNullShiftMaxMm", "artifacts/confirmation-dataset.json", "/headline/outside_channel_axis_null_shift_m/max", "mm2", "largest shift of an axis null outside the straight section (mm)")
    m.add("HmcOutsideNullPairs", "artifacts/confirmation-dataset.json", "/headline/outside_channel_axis_null_shift_m/count", "int", "paired axis nulls outside the straight section")
    m.add_derived("HmcLOneAOutsideNulls", l1a_outside_total, "int", "axis nulls outside the straight section under L1a", "sum over rows of len(outside_channel_axis_nulls.l1a_z_m)", row_inputs)
    m.add_derived("HmcPTwoOutsideNulls", p2_outside_total, "int", "axis nulls outside the straight section under P2", "sum over rows of len(outside_channel_axis_nulls.p2_z_m)", row_inputs)
    m.add("HmcLeanLOneAMaxMm", "artifacts/confirmation-dataset.json", "/headline/separatrix_lean_l1a_m/max", "mm2", "largest axis-null-to-cusp lean under L1a (mm)")
    m.add("HmcLeanPTwoMaxMm", "artifacts/confirmation-dataset.json", "/headline/separatrix_lean_p2_m/max", "mm2", "largest axis-null-to-cusp lean under P2 (mm)")
    m.add("HmcLeanLOneAMedianUm", "artifacts/confirmation-dataset.json", "/headline/separatrix_lean_l1a_m/median", "um0", "median lean under L1a (um)")
    m.add("HmcLeanPTwoMedianUm", "artifacts/confirmation-dataset.json", "/headline/separatrix_lean_p2_m/median", "um0", "median lean under P2 (um)")
    for count, token in ((2, "Two"), (3, "Three"), (4, "Four"), (5, "Five"), (6, "Six")):
        m.add_derived(f"HmcLOneANull{token}Designs", l1a_nulls_hist.get(str(count), 0), "int", f"designs with {count} axis nulls under L1a", "estimands.l1a_axis_null_count_histogram", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/estimands/l1a_axis_null_count_histogram"}])
        m.add_derived(f"HmcPTwoNull{token}Designs", p2_nulls_hist.get(str(count), 0), "int", f"designs with {count} axis nulls under P2", "estimands.p2_axis_null_count_histogram", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/estimands/p2_axis_null_count_histogram"}])
    m.add("HmcAngleLOneAMaxDeg", "artifacts/confirmation-dataset.json", "/estimands/angle_to_wall_normal_deg_l1a/max", "deg1", "largest separatrix angle to the wall normal under L1a (degrees)")
    m.add("HmcAnglePTwoMaxDeg", "artifacts/confirmation-dataset.json", "/estimands/angle_to_wall_normal_deg_p2/max", "deg1", "largest separatrix angle to the wall normal under P2 (degrees)")
    m.add("HmcAngleLOneAMedianDeg", "artifacts/confirmation-dataset.json", "/estimands/angle_to_wall_normal_deg_l1a/median", "deg1", "median separatrix angle under L1a (degrees)")
    m.add("HmcAnglePTwoMedianDeg", "artifacts/confirmation-dataset.json", "/estimands/angle_to_wall_normal_deg_p2/median", "deg1", "median separatrix angle under P2 (degrees)")
    m.add("HmcLOneABThreeMedian", "artifacts/confirmation-dataset.json", "/estimands/l1a_wall_b3_over_b1/median", "fixed3", "median third-harmonic content of the L1a wall field")
    m.add("HmcPTwoBThreeMedian", "artifacts/confirmation-dataset.json", "/estimands/p2_wall_b3_over_b1/median", "fixed3", "median third-harmonic content of the P2 wall field")

    # ---- discretisation and sampling sensitivity ----
    m.add("HmcDiscShiftMaxUm", "artifacts/confirmation-dataset.json", "/headline/p2_discretisation_max_wall_intersection_shift_m/max", "um1", "largest level-0 -> level-1 cusp shift (um)")
    m.add("HmcDiscShiftMedianUm", "artifacts/confirmation-dataset.json", "/headline/p2_discretisation_max_wall_intersection_shift_m/median", "um1", "median level-0 -> level-1 cusp shift (um)")
    m.add("HmcDiscNullShiftMaxUm", "artifacts/confirmation-dataset.json", "/estimands/p2_discretisation_max_axis_null_shift_m/max", "um1", "largest level-0 -> level-1 axis-null shift (um)")
    m.add("HmcDiscRhoSensitivityMax", "artifacts/confirmation-dataset.json", "/estimands/p2_discretisation_rho_sensitivity_max/max", "sci1", "largest relative rho difference between the two levels")
    m.add_derived("HmcHempFlagStableLevels", hemp_flag_stable_levels, "int", "designs whose HEMP-like flag agrees between the two P2 levels", "count over rows of p2_discretisation_rho_sensitivity.hemp_like_flag_agrees", row_inputs)
    m.add("HmcSamplingShiftMaxNm", "artifacts/confirmation-dataset.json", "/estimands/sampling_max_wall_intersection_shift_m/max", "nm0", "largest 1x -> 2x sampling cusp shift (nm)")
    m.add_derived("HmcShiftMinOverDiscMax", min(shifts) / max(disc_wall), "fixed2", "smallest matched-cusp shift over the largest discretisation shift", "min(shift_m) / max(p2_discretisation.max_wall_intersection_shift_m)", pair_inputs)

    # ---- shakedown ----
    m.add("HmcShakedownDesigns", "artifacts/shakedown.json", "/design_count", "int", "designs exercised by the non-evidentiary shakedown")
    m.add_derived("HmcShakedownDesignIds", shakedown_ids, "list_short_designs", "shakedown design ordinals", "protocol.shakedown.designs.hemp_like_v3", [{"artifact": "artifacts/protocol.json", "pointer": "/shakedown/designs/hemp_like_v3"}])
    m.add("HmcShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("HmcShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary")
    m.add("HmcShakedownEnterEstimand", "artifacts/shakedown.json", "/outcomes_enter_estimand", "bool", "shakedown outcomes enter an estimand")
    m.add("HmcPreflightDesigns", "artifacts/shakedown.json", "/mesh_preflight/design_count", "int", "designs whose level-0 mesh the whole-set preflight built")
    m.add("HmcPreflightPassed", "artifacts/shakedown.json", "/mesh_preflight/passed_count", "int", "designs that passed the whole-set preflight")
    m.add("HmcPreflightMinAngleDeg", "artifacts/shakedown.json", "/mesh_preflight/minimum_angle_deg", "deg1", "smallest level-0 minimum angle over the set (degrees)")
    m.add_derived("HmcPreflightBelowTenDesigns", len(preflight["designs_with_elements_below_10deg"]), "int", "designs with level-0 elements below the qualification's sliver threshold", "len(mesh_preflight.designs_with_elements_below_10deg)", [{"artifact": "artifacts/shakedown.json", "pointer": "/mesh_preflight/designs_with_elements_below_10deg"}])
    m.add_derived("HmcSliverThresholdDeg", SLIVER_THRESHOLD_DEG, "fixed0", "sliver threshold of the mesh-quality record (degrees)", "mesh_quality.sliver.threshold_deg of every recorded level", [{"artifact": designs[0]["record_path"], "pointer": "/evidence/p2/levels/0/mesh_quality/sliver/threshold_deg"}])
    m.add("HmcPreflightSeconds", "artifacts/shakedown.json", "/mesh_preflight/seconds", "sec0", "whole-set preflight wall time (s)")
    m.add("HmcTimingProjectedS", "artifacts/shakedown.json", "/timing_projection/projected_wall_seconds_at_pool", "sec0", "shakedown timing projection (s)")
    m.add("HmcTimingProjectedMin", "artifacts/shakedown.json", "/timing_projection/projected_wall_seconds_at_pool", "min1", "shakedown timing projection (min)")
    m.add("HmcTimingBudgetMin", "artifacts/shakedown.json", "/timing_projection/budget_wall_seconds", "min1", "wall-clock budget (min)")
    m.add("HmcTimingWithinBudget", "artifacts/shakedown.json", "/timing_projection/within_budget", "bool", "projection within budget")
    m.add("HmcTimingContentionFactor", "artifacts/shakedown.json", "/timing_projection/contention_factor", "g", "contention factor of the projection")
    m.add("HmcTimingMeanSecondsPerDesign", "artifacts/shakedown.json", "/timing_projection/mean_seconds_per_design", "sec0", "mean shakedown seconds per design")
    m.add_derived("HmcStageWithinBudget", campaign["execution_mode"]["stage_wall_s"] <= shakedown["timing_projection"]["budget_wall_seconds"], "bool", "the recorded design stage finished within the budget", "campaign-result.execution_mode.stage_wall_s <= shakedown.timing_projection.budget_wall_seconds", [{"artifact": "artifacts/campaign-result.json", "pointer": "/execution_mode/stage_wall_s"}, {"artifact": "artifacts/shakedown.json", "pointer": "/timing_projection/budget_wall_seconds"}])
    m.add_derived("HmcShakedownOverlapDesigns", len(set(shakedown_ids) & set(declared_ids)), "int", "shakedown designs that are also evidentiary designs", "|shakedown designs ∩ declared designs|", [{"artifact": "artifacts/protocol.json", "pointer": "/shakedown/designs/hemp_like_v3"}])
    m.add("HmcShakedownDesignRule", "artifacts/protocol.json", "/shakedown/design_rule", "text", "shakedown design rule")

    # ---- the v1 development rejection (lineage) ----
    v1.add("HmcVOneExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier of the rejected predecessor")
    v1.add("HmcVOneTerminalState", "terminal.json", "/state", "ident", "terminal state of the v1 execution")
    v1.add("HmcVOneResolved", "terminal.json", "/payload/resolved_design_count", "int", "designs resolved by v1 before the rejection")
    v1.add("HmcVOneFailed", "terminal.json", "/payload/failed_design_count", "int", "designs v1 failed at the mesh gate")
    v1.add_derived("HmcVOneFailedDesigns", v1_failed_ids, "list_short_designs", "designs v1 rejected", "design-failures.failed[*].key", [{"artifact": "artifacts/design-failures.json", "pointer": "/failed"}])
    v1.add("HmcVOneFailureStage", "artifacts/design-failures.json", "/failed/0/stage", "ident", "stage at which v1 failed the designs")
    v1.add("HmcVOneFailureReason", "artifacts/design-failures.json", "/failed/0/reason", "text", "recorded failure reason")
    v1.add("HmcVOneStageWallMin", "terminal.json", "/payload/stage_wall_s", "min1", "v1 design stage wall time (min)")
    v1.add_derived("HmcVOneExecutionWallMin", v1_execution_wall_s, "min1", "v1 lock-acquired to terminal wall time (min)", "difference of the v1 transition timestamps 0007 - 0001", [{"artifact": "transitions/0001-lock-acquired.json", "pointer": "/recorded_at_utc"}, {"artifact": "transitions/0007-terminal.json", "pointer": "/recorded_at_utc"}])
    v1.add_derived("HmcVOneVerifiedFiles", len(v1_bundle.hashes), "int", "v1 bundle files verified byte for byte", "count of v1 manifest file entries verified", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    v1.add("HmcVOneArtifactCount", "manifest.json", "/artifact_count", "int", "v1 manifest artifact entries")
    v1.add_derived("HmcVOneRecords", len(v1_records), "int", "v1 design records (resolved designs)", "count of artifacts/designs/*.json in the v1 bundle", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    v1.add("HmcVOneAssessmentAccess", "terminal.json", "/counts/assessment_access_count", "int", "v1 assessment accesses (none: no verdict was produced)")
    v1.add("HmcVOneAttemptCount", "terminal.json", "/counts/attempt_count", "int", "v1 attempts")
    v1.add("HmcVOneLockImmutable", "execution-lock.json", "/immutable", "bool", "v1 lock immutable")
    v1.add("HmcVOneShakedownDesigns", "artifacts/shakedown.json", "/design_count", "int", "designs exercised by the v1 shakedown")
    v1.add_derived("HmcVOneTransitionCount", len(v1_transitions), "int", "v1 transitions", "count of v1 transition records", [{"artifact": "transitions/0007-terminal.json", "pointer": "/sequence"}])
    m.add_derived("HmcVOneManifestSha", v1_bundle.manifest_sha256, "sha_short", "v1 results manifest SHA-256 prefix", "sha256(v1 results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add("HmcPredecessorStatement", "artifacts/protocol.json", "/predecessor/statement", "text", "predecessor statement of the frozen protocol")
    m.add("HmcAngleGateDisclosure", "artifacts/protocol.json", "/p2/mesh/angle_gate_disclosure", "text", "angle-gate disclosure of the frozen protocol")
    m.add_derived("HmcProtocolChangedPaths", len(protocol_diff), "int", "protocol paths that differ between v1 and v1.1", "len(diff(v1 protocol, v1.1 protocol)) == the declared change set", [{"artifact": "artifacts/protocol.json", "pointer": ""}])
    m.add_derived("HmcProtocolChangedPathList", list(protocol_diff), "list_clauses", "the differing protocol paths", "diff(v1 protocol, v1.1 protocol) (identity, predecessor block, angle gate, shakedown set and preflight)", [{"artifact": "artifacts/protocol.json", "pointer": ""}])
    m.add_derived("HmcProtocolDeclarationsChanged", len(DECLARATIONS_CHANGED), "int", "numerical declarations changed between v1 and v1.1 (the angle gate; the whole-set preflight)", "count of the declared declaration changes", [{"artifact": "artifacts/protocol.json", "pointer": "/predecessor/statement"}])
    m.add_derived("HmcProtocolBlocksUnchanged", True, "bool", "comparison, gates, definition import, design sets, solver, materials, adaptivity, sampling, resources, claim boundary and outputs are identical between v1 and v1.1", "block equality of the two sealed protocols", [{"artifact": "artifacts/protocol.json", "pointer": "/comparison"}, {"artifact": "artifacts/protocol.json", "pointer": "/gates"}])
    m.add_derived("HmcSliverDesignA", _short(v1_failed_ids[0]), "ident", "first design v1 rejected", "design-failures.failed[0]", [{"artifact": "artifacts/protocol.json", "pointer": "/predecessor"}])
    m.add_derived("HmcSliverDesignB", _short(v1_failed_ids[1]), "ident", "second design v1 rejected", "design-failures.failed[1]", [{"artifact": "artifacts/protocol.json", "pointer": "/predecessor"}])
    m.add_derived("HmcSliverAMinAngleDeg", sliver_a[0]["min_angle_deg"], "deg1", "level-0 minimum angle of the first rejected design (degrees)", "record evidence.p2.levels[0].mesh_quality.minimum_angle_deg", row_inputs)
    m.add_derived("HmcSliverBMinAngleDeg", sliver_b[0]["min_angle_deg"], "deg1", "level-0 minimum angle of the second rejected design (degrees)", "record evidence.p2.levels[0].mesh_quality.minimum_angle_deg", row_inputs)
    m.add_derived("HmcSliverABelowTen", sliver_a[0]["below_threshold"], "int_comma", "level-0 elements of the first rejected design below the sliver threshold", "record evidence.p2.levels[0].mesh_quality.sliver.elements_below_threshold", row_inputs)
    m.add_derived("HmcSliverBBelowTen", sliver_b[0]["below_threshold"], "int_comma", "level-0 elements of the second rejected design below the sliver threshold", "record evidence.p2.levels[0].mesh_quality.sliver.elements_below_threshold", row_inputs)
    m.add_derived("HmcSliverAElements", sliver_a[0]["elements"], "int_comma", "level-0 elements of the first rejected design", "record evidence.p2.levels[0].mesh_quality.sliver.element_count", row_inputs)
    m.add_derived("HmcSliverBElements", sliver_b[0]["elements"], "int_comma", "level-0 elements of the second rejected design", "record evidence.p2.levels[0].mesh_quality.sliver.element_count", row_inputs)
    m.add_derived("HmcSliverALevelOneMinAngleDeg", sliver_a[1]["min_angle_deg"], "deg1", "level-1 minimum angle of the first rejected design (degrees)", "record evidence.p2.levels[1].mesh_quality.minimum_angle_deg", row_inputs)
    m.add_derived("HmcSliverBLevelOneMinAngleDeg", sliver_b[1]["min_angle_deg"], "deg1", "level-1 minimum angle of the second rejected design (degrees)", "record evidence.p2.levels[1].mesh_quality.minimum_angle_deg", row_inputs)
    m.add_derived("HmcSliverAShiftOverTol", per_design[v1_failed_ids[0]]["max_shift_tol"], "fixed2", "largest cusp shift of the first rejected design in tolerance units", "comparison.max_cusp_shift_over_tolerance", row_inputs)
    m.add_derived("HmcSliverBShiftOverTol", per_design[v1_failed_ids[1]]["max_shift_tol"], "fixed2", "largest cusp shift of the second rejected design in tolerance units", "comparison.max_cusp_shift_over_tolerance", row_inputs)
    m.add_derived("HmcRejectionNoteBound", True, "bool", "POSTHOC_REJECTION.md names the recorded v1 facts", "regex checks of the note against the v1 bundle (commits, counts, gate, stage time, no verdict)", [{"artifact": "artifacts/protocol.json", "pointer": "/predecessor"}])

    # ---- claim boundary flags carried as policy ----
    m.add_derived("HmcNotHardwareValid", True, "bool", "not hardware-valid", "classification token NOT_HARDWARE_VALID", [{"artifact": "artifacts/campaign-result.json", "pointer": "/classification"}])
    m.add_derived("HmcLinearMaterials", True, "bool", "linear isotropic materials only (no saturation, no B-H curve)", "protocol.p2.materials.statement", [{"artifact": "artifacts/protocol.json", "pointer": "/p2/materials/statement"}])
    if "no saturation, no B-H curve" not in protocol["p2"]["materials"]["statement"] or "saturation" not in dataset["claim_boundary"]["what_is_not_claimed"]:
        raise ValueError("the frozen protocol does not declare the linear-material boundary")
    m.add_derived("HmcNotPTwoQualifiedChain", True, "bool", "not the three-level P2-qualified chain", "claim_boundary.field_level names the two-level screening configuration", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/claim_boundary/field_level"}])
    if "not the three-level NUMERICAL_P2_QUALIFIED chain" not in dataset["claim_boundary"]["field_level"]:
        raise ValueError("the claim boundary does not distinguish the two-level configuration from the qualified chain")
    m.add_derived("HmcDesignRecommendation", False, "bool", "a design recommendation is made", "claim_boundary.what_is_not_claimed names design recommendation", [{"artifact": "artifacts/confirmation-dataset.json", "pointer": "/claim_boundary/what_is_not_claimed"}])
    if "design recommendation" not in dataset["claim_boundary"]["what_is_not_claimed"]:
        raise ValueError("the claim boundary does not exclude a design recommendation")

    # ================================================================== tables ====
    tex_lines = [
        "% Generated by paper/scripts/generate_l1b_hemp_confirmation_v1_1_evidence.py; do not hand edit.",
        f"% Evidence: {EXPERIMENT.as_posix()} at commit {RESULTS_COMMIT_SHA} (results manifest SHA-256 {bundle.manifest_sha256}).",
        f"% Every macro value traces to an artifact path and JSON pointer recorded in {EVIDENCE_PATH.as_posix()}.",
    ]
    for item in m.items + v1.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    # (a) per-design agreement table
    design_rows: list[str] = []
    for design_id in declared_ids:
        d = per_design[design_id]
        ratios = d["wall_ratios"]
        low, high = f"{min(ratios):.2f}", f"{max(ratios):.2f}"
        ratio_text = low if low == high else f"{low}--{high}"
        design_rows.append(
            f"\\texttt{{{d['short']}}}{'$^*$' if d['representative'] else ''} & {d['stages']} & {d['x_w']:.2f} & {d['rw_over_l']:.3f} & {d['l1a_cusps']} / {d['p2_cusps']} & {d['matched']} & "
            f"{1e3 * d['max_shift_m']:.3f} ({d['max_shift_tol']:.2f}) & {1e3 * d['channel_shift_m']:.2f} & {ratio_text} & {d['axis_ratio']:.2f} & "
            f"{d['l1a_min_rho']:.3f} $\\to$ {d['p2_min_rho']:.3f} & {_yes(d['p2_hemp'])} & {_comma(d['level1_dofs'])}\\\\"
        )
    tex_lines += _table(
        "HmcDesignTable",
        "Per-design agreement between the sealed \\HmcFieldModelLevelLOneA{} record and the accepted (level-\\HmcLevelOneToken) "
        "material-aware P2 map of the \\HmcDesignCount{} HEMP-like sweep-\\HmcSweepVersion{} designs, as sealed in "
        "\\texttt{campaign-result.json} and recomputed by the evidence generator from the matched cusp pairs: sweep ordinal "
        "(representatives with stored separatrix traces starred), magnet stages, $x_w=\\pi r_w/L$, $r_w/L$, wall cusps under "
        "\\HmcFieldModelLevelLOneA{} / P2, matched cusps, largest matched-cusp shift in mm (in units of the design tolerance "
        "$\\max(r_w/\\HmcBoreElements, \\Delta z_{\\mathrm{L1a}})$), largest sorted channel axis-null shift in mm, the range of the "
        "per-cusp wall $|B|$ ratio P2 / \\HmcFieldModelLevelLOneA{} at equal magnet strength, the axis $|B_z|$ peak ratio, the "
        "per-design minimum Koch ratio $\\rho$ under \\HmcFieldModelLevelLOneA{} $\\to$ P2, whether the design stays HEMP-like "
        "($\\rho \\ge \\HmcRhoThreshold$ at every cusp) and the level-\\HmcLevelOneToken{} P2 degrees of freedom. Every ratio is a "
        "ratio of two field models and never a probability.",
        "tab:l1b-hemp-confirmation-designs", "lrrrcrrrrrrcr",
        "design & stages & $x_w$ & $r_w/L$ & cusps & matched & shift mm (tol.) & null mm & wall $|B|$ & axis & $\\rho_{\\min}$ & HEMP & DOFs\\\\",
        design_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    # (b) verdict and gates
    gate_rows = [
        f"(a) P2 solves converged at relative true residual $\\le$ {format_value('sci1', protocol['p2']['solver']['relative_tolerance'])} & {solves_converged} of {2 * len(designs)} (largest residual {format_value('sci2', headline['p2_relative_true_residual_max'])})\\\\",
        f"(a) level 0 / level 1 P2 degrees of freedom & {_comma(min(level_dofs[0]))}--{_comma(max(level_dofs[0]))} / {_comma(min(level_dofs[1]))}--{_comma(max(level_dofs[1]))} (cap {_comma(protocol['p2']['resources']['maximum_p2_dofs'])})\\\\",
        f"(b) designs with an unchanged wall-cusp count: strict / boundary-tolerant (threshold {gate_b['pass_threshold']:.1f}) & {strict_agree} / {tolerant_agree} of {len(designs)}; passed: {_yes(gate_b['passed'])}\\\\",
        f"(b) designs with an unchanged cell count & {sum(1 for row in designs if row['comparison']['cell_count_agreement'])} of {len(designs)}\\\\",
        f"(c) matched cusps / designs with a bijective matching & {len(matched_pairs)} / {bijective} of {len(designs)}\\\\",
        f"(c) largest shift in tolerance units (threshold {gate_c['pass_threshold']:.1f}) & {gate_c['max_shift_over_tolerance']:.2f} (design \\texttt{{{_short(max_pair['design_id'])}}}); passed: {_yes(gate_c['passed'])}\\\\",
        f"(c) matched-cusp shift: min / median / mean / max (mm) & {1e3 * min(shifts):.4f} / {1e3 * statistics.median(shifts):.3f} / {1e3 * statistics.fmean(shifts):.3f} / {1e3 * max(shifts):.3f}\\\\",
        f"(c) design tolerance $\\max(r_w/\\HmcBoreElements, \\Delta z_{{\\mathrm{{L1a}}}})$: min / median / max (mm) & {1e3 * min(tolerances):.3f} / {1e3 * statistics.median(tolerances):.3f} / {1e3 * max(tolerances):.3f}\\\\",
        f"(c) shifts above the imported stability tolerance ({1e3 * imported['stability_tolerance_m']:.2f}~mm) / above the P2 discretisation shift & {shifts_above_stability} / {len(matched_pairs)} of {len(matched_pairs)}\\\\",
        f"verdict by the predeclared rule & \\texttt{{{_ident(campaign['verdict'])}}}\\\\",
        "\\midrule",
        f"(d) designs HEMP-like under P2 (reported, not gated) & {preserved} of {len(designs)} (lost: \\texttt{{{_short(lost[0])}}}, $\\rho_{{\\min}}$ {per_design[lost[0]]['l1a_min_rho']:.3f} $\\to$ {per_design[lost[0]]['p2_min_rho']:.3f})\\\\",
        f"(d) per-cusp $\\rho$ ratio P2 / L1a: min / median / max & {gate_d['rho_conservative_ratio_p2_over_l1a']['min']:.2f} / {gate_d['rho_conservative_ratio_p2_over_l1a']['median']:.2f} / {gate_d['rho_conservative_ratio_p2_over_l1a']['max']:.2f} (below one at {sum(1 for p in matched_pairs if p['rho_conservative_ratio_p2_over_l1a'] < 1.0)} cusps)\\\\",
        f"(d) per-cusp wall $|B|$ ratio at equal magnet strength: min / median / max & {gate_d['wall_b_ratio_p2_over_l1a_per_cusp']['min']:.2f} / {gate_d['wall_b_ratio_p2_over_l1a_per_cusp']['median']:.2f} / {gate_d['wall_b_ratio_p2_over_l1a_per_cusp']['max']:.2f} (above one at {sum(1 for p in matched_pairs if p['wall_b_ratio_p2_over_l1a'] > 1.0)} cusps)\\\\",
        f"(d) per-design peak wall $|B|$ ratio / axis $|B_z|$ peak ratio: min--max & {gate_d['peak_wall_b_ratio_p2_over_l1a']['min']:.2f}--{gate_d['peak_wall_b_ratio_p2_over_l1a']['max']:.2f} / {gate_d['axis_peak_b_ratio_p2_over_l1a']['min']:.2f}--{gate_d['axis_peak_b_ratio_p2_over_l1a']['max']:.2f}\\\\",
        f"(d) designs inside the descriptive band [{designs[0]['comparison']['wall_b_ratio_band_descriptive'][0]:.1f}, {designs[0]['comparison']['wall_b_ratio_band_descriptive'][1]:.1f}] & {gate_d['peak_wall_b_ratio_in_band_count']} of {len(designs)}\\\\",
        "\\midrule",
        f"channel axis nulls: designs with equal count / in bijection within the cusp tolerance & {channel_count_equal} / {channel_null_bijections} of {len(designs)}\\\\",
        f"channel axis-null sorted shift: median / max (mm); designs beyond the cusp tolerance & {1e3 * statistics.median(channel_sorted_shifts):.2f} / {1e3 * max(channel_sorted_shifts):.2f}; {sum(1 for row in designs if row['comparison']['channel_axis_nulls']['max_sorted_shift_m'] > row['comparison']['cusp_position_tolerance_m'])}\\\\",
        f"axis nulls outside the straight section: L1a / P2 (designs with more nulls under P2) & {l1a_outside_total} / {p2_outside_total} ({more_p2_nulls})\\\\",
        f"axis-null-to-cusp lean, largest: L1a / P2 (mm) & {1e3 * headline['separatrix_lean_l1a_m']['max']:.2f} / {1e3 * headline['separatrix_lean_p2_m']['max']:.2f}\\\\",
        f"binding integrity gates true / determinism replays bit-identical & {sum(1 for name in BINDING_GATE_NAMES if gates['campaign'][name])} of {len(BINDING_GATE_NAMES)} / {sum(1 for item in gates['replays'] if item['bit_identical'])} of {len(gates['replays'])}\\\\",
        f"designs stable under $\\times${protocol['p2']['sampling']['refinement']} sampling / between the two P2 levels & {headline['sampling_stable_count']} / {headline['p2_discretisation_stable_count']} of {len(designs)}\\\\",
        f"P2 level 0 $\\to$ level 1 cusp shift, largest ($\\mu$m) & {1e6 * max(disc_wall):.1f}\\\\",
    ]
    tex_lines += _table(
        "HmcGateTable",
        "Verdict, predeclared gates and reported quantities of the material-aware confirmation as sealed in "
        "\\texttt{gates.json} and \\texttt{confirmation-dataset.json}, each recomputed by the evidence generator from the "
        "per-design rows and matched cusp pairs. Gate (a) is a binding integrity gate (every solve converged within the frozen "
        "controls); gates (b) and (c) are the predeclared confirmation gates whose outcome classifies the result; (d) and the "
        "axis-null rows are reported, never gated. The verdict is a statement about the \\HmcFieldModelLevelLOneA{} field maps' "
        "cusp descriptors under a linear-iron P2 field; no row is a plasma, confinement or performance quantity.",
        "tab:l1b-hemp-confirmation-gates", f"{_p(9.6)}{_p(6.2)}",
        "quantity & value\\\\", gate_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{4pt}",
    )
    # (c) P2 field model and solve evidence
    field_rows = [
        f"soft-iron poles (one per inter-magnet gap) and return yoke: relative permeability & {protocol['p2']['materials']['soft_iron_relative_permeability']:.0f} (linear; no saturation, no B-H curve)\\\\",
        f"magnets: recoil relative permeability; remanence & {protocol['p2']['materials']['magnet_recoil_relative_permeability']:.2f}; synthetic SmCo-like, axial, alternating polarity\\\\",
        f"vacuum-like regions at unit permeability & {_tex_escape(protocol['p2']['materials']['vacuum_like_regions'])}\\\\",
        f"level 0 mesh: bore elements / feature elements / padding factor / angle gate & {protocol['p2']['mesh']['bore_elements']} / {protocol['p2']['mesh']['feature_elements']} / {protocol['p2']['mesh']['padding_factor']:g} / {protocol['p2']['mesh']['reject_below_angle_deg']:.0f}$^\\circ$\\\\",
        f"adaptivity: nested levels / D\\\"orfler fraction & {protocol['p2']['adaptivity']['levels']} / {protocol['p2']['adaptivity']['dorfler_theta']:g}\\\\",
        f"solver: relative tolerance / iteration cap / backend & {format_value('sci1', protocol['p2']['solver']['relative_tolerance'])} / {_comma(protocol['p2']['solver']['max_iterations'])} / numpy CSR PCG, CPU only\\\\",
        f"PCG iterations, level 0 / level 1: min--max & {_comma(min(level_iterations[0]))}--{_comma(max(level_iterations[0]))} / {_comma(min(level_iterations[1]))}--{_comma(max(level_iterations[1]))}\\\\",
        f"sampling of the bore: radial intervals / refinement of the second sampling & {protocol['p2']['sampling']['radial_intervals']} / $\\times${protocol['p2']['sampling']['refinement']}\\\\",
        f"per-design P2 time, min / median / max / sum (s) & {min(solve_seconds):.0f} / {statistics.median(solve_seconds):.0f} / {max(solve_seconds):.0f} / {sum(solve_seconds):.0f}\\\\",
        f"design stage / assessment / lock-to-terminal wall time (min) & {campaign['execution_mode']['stage_wall_s'] / 60:.1f} / {campaign['execution_mode']['assessment_wall_s'] / 60:.1f} / {execution_wall_s / 60:.1f}\\\\",
        f"workers / CPU cores / BLAS threads & {campaign['execution_mode']['worker_pool_size']} / {runtime['cpu_count']} / {_tex_escape(runtime['blas_threads']['OPENBLAS_NUM_THREADS'])}\\\\",
        f"peak RSS (MB) / budget (GB) / share used / DOF cap & {gates['peak_rss_bytes'] / 1e6:.0f} / {gates['ram_budget']['budget_bytes'] / 1e9:.1f} / {100 * headline['ram_budget_fraction_used']:.1f}\\% / {_comma(gates['ram_budget']['maximum_p2_dofs'])}\\\\",
        f"bundle files verified byte for byte / accepted through an end-of-line tolerance & {len(bundle.hashes)} / 0\\\\",
    ]
    tex_lines += _table(
        "HmcFieldTable",
        "The material-aware P2 field model and the solve evidence of the confirmation as frozen in \\texttt{protocol.json} and "
        "sealed in \\texttt{campaign-result.json}, \\texttt{gates.json} and \\texttt{runtime.json}. The materials are linear and "
        "isotropic; the two nested adaptive levels are the screening configuration of the finite-element reference and not the "
        "three-level qualified chain of the wall-loss campaign's field.",
        "tab:l1b-hemp-confirmation-field", f"{_p(8.2)}{_p(7.4)}",
        "quantity & value\\\\", field_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{4pt}",
    )
    # (d) disclosures
    disclosure_rows = [
        f"(i) v1 executed once at its preregistration; terminal state & \\texttt{{{_ident(v1_terminal['state'])}}} (bundle verified byte for byte: {len(v1_bundle.hashes)}; manifest prefix \\texttt{{{v1_bundle.manifest_sha256[:12]}}})\\\\",
        f"(i) designs resolved / failed before any solve; verdict & {v1_terminal['payload']['resolved_design_count']} / {len(v1_failed)} (\\texttt{{{_short(v1_failed_ids[0])}}}, \\texttt{{{_short(v1_failed_ids[1])}}}); none (assessment accesses {v1_terminal['counts']['assessment_access_count']})\\\\",
        f"(i) failure stage / reason & \\texttt{{{_ident(v1_failed[0]['stage'])}}} / level 0 mesh below the {v1_protocol['p2']['mesh']['reject_below_angle_deg']:.0f}$^\\circ$ gate inherited from the reference qualification\\\\",
        f"(i) v1 design stage wall time (min) / protocol paths changed for v1.1 & {v1_terminal['payload']['stage_wall_s'] / 60:.1f} / {len(protocol_diff)} (identity, predecessor block, angle gate, shakedown set and preflight; every tolerance and threshold identical)\\\\",
        "\\midrule",
        f"(ii) angle gate v1 $\\to$ v1.1 & {v1_protocol['p2']['mesh']['reject_below_angle_deg']:.0f}$^\\circ$ $\\to$ {protocol['p2']['mesh']['reject_below_angle_deg']:.0f}$^\\circ$ (disclosed in the frozen protocol)\\\\",
        f"(ii) \\texttt{{{_short(v1_failed_ids[0])}}}: level 0 minimum angle / elements below {SLIVER_THRESHOLD_DEG:.0f}$^\\circ$ of / level 1 minimum angle & {sliver_a[0]['min_angle_deg']:.1f}$^\\circ$ / {_comma(sliver_a[0]['below_threshold'])} of {_comma(sliver_a[0]['elements'])} / {sliver_a[1]['min_angle_deg']:.1f}$^\\circ$\\\\",
        f"(ii) \\texttt{{{_short(v1_failed_ids[1])}}}: level 0 minimum angle / elements below {SLIVER_THRESHOLD_DEG:.0f}$^\\circ$ of / level 1 minimum angle & {sliver_b[0]['min_angle_deg']:.1f}$^\\circ$ / {_comma(sliver_b[0]['below_threshold'])} of {_comma(sliver_b[0]['elements'])} / {sliver_b[1]['min_angle_deg']:.1f}$^\\circ$\\\\",
        f"(ii) whole-set mesh preflight: designs built / passed / smallest minimum angle & {preflight['design_count']} / {preflight['passed_count']} / {preflight['minimum_angle_deg']:.1f}$^\\circ$\\\\",
        f"(ii) largest cusp shift of the two designs in tolerance units & {per_design[v1_failed_ids[0]]['max_shift_tol']:.2f} / {per_design[v1_failed_ids[1]]['max_shift_tol']:.2f}\\\\",
        "\\midrule",
        f"(iii) shakedown designs (all evidentiary designs; outcomes non-evidentiary) & {shakedown['design_count']} of {len(designs)} (\\texttt{{{'}, \\texttt{'.join(_short(d) for d in shakedown_ids)}}})\\\\",
        f"(iii) shakedown timing projection / budget (min); within budget & {shakedown['timing_projection']['projected_wall_seconds_at_pool'] / 60:.1f} / {shakedown['timing_projection']['budget_wall_seconds'] / 60:.1f}; {_yes(shakedown['timing_projection']['within_budget'])} (contention factor {shakedown['timing_projection']['contention_factor']:g})\\\\",
        f"(iii) recorded design stage (min); within budget & {campaign['execution_mode']['stage_wall_s'] / 60:.1f}; {_yes(campaign['execution_mode']['stage_wall_s'] <= shakedown['timing_projection']['budget_wall_seconds'])}\\\\",
        f"(iii) the campaign's own record on paper admission & \\texttt{{{_ident(campaign['paper_admission'].split(';')[0])}}}\\\\",
    ]
    tex_lines += _table(
        "HmcDisclosureTable",
        "Disclosures admitted with the confirmation, each verified by the evidence generator against the two sealed bundles and "
        "the frozen protocols: (i) the recorded development rejection of the predecessor campaign (\\texttt{POSTHOC\\_REJECTION.md}, "
        "bound at commit \\texttt{\\HmcCodeCommit}), (ii) the relaxed level-\\HmcLevelZeroToken{} angle gate with the sliver record "
        "of the two designs that gate had rejected and the whole-set mesh preflight, and (iii) the overlap of the non-evidentiary "
        "shakedown with the evidentiary set, its timing projection and the campaign's own statement that paper admission was out "
        "of its scope.",
        "tab:l1b-hemp-confirmation-disclosures", f"{_p(8.4)}{_p(7.2)}",
        "disclosure & value\\\\", disclosure_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{4pt}",
    )
    tex = "\n".join(tex_lines) + "\n"

    reference_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (catalogue_file, sweep_manifest_file, sweep_design_authorities_file, topology_protocol_file)
    }
    lineage_files = {
        rejection_file["path"]: {"sha256": rejection_file["sha256"], "bytes": rejection_file["bytes"], "revision": rejection_file["revision"], "role": rejection_file["role"], "git_blob": rejection_file["git_blob"], "git_blob_sha256": rejection_file["git_blob_sha256"]},
    }
    evidence = {
        "document_type": "paper-l1b-hemp-confirmation-v1-1-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "topology_label": TOPOLOGY_LABEL,
        "recorded_outcome": RECORDED_OUTCOME,
        "campaign_status": CAMPAIGN_STATUS,
        "verdict": VERDICT,
        "screening_model": SCREENING_MODEL,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
        "reference_artifacts": {
            "rule": (
                "the sweep-v3 wall-cusp catalogue and results manifest and the sweep-v3 design authorities (must hash to the "
                "identities the campaign sealed; the declared designs must be exactly the catalogue's HEMP-like Sobol entries in "
                "catalogue order, and every L1a reference must equal its catalogue entry) and the frozen cusp topology v3.1 "
                "protocol (whose definition parameters the imported definition must equal), each bound at its own admitted revision"
            ),
            "files": reference_files,
        },
        "lineage": {
            **binding["lineage"],
            "manifest_sha256": v1_bundle.manifest_sha256,
            "verified_file_count": len(v1_bundle.hashes),
            "artifact_count": v1_bundle.manifest["artifact_count"],
            "lock_commit_recorded": v1_lock["commit"],
            "result_commit_recorded_prefix": predecessor["result_commit"],
            "resolved_design_count": v1_terminal["payload"]["resolved_design_count"],
            "failed_design_count": len(v1_failed),
            "failed_designs": v1_failed_ids,
            "angle_gate_deg": v1_protocol["p2"]["mesh"]["reject_below_angle_deg"],
            "protocol_paths_changed": protocol_diff,
            "declarations_changed": list(DECLARATIONS_CHANGED),
            "files": lineage_files,
            "cited_for_numbers": False,
            "rule": (
                "the recorded development rejection of l1b_hemp_confirmation_v1 is byte-verified from its own bundle (every "
                "file, sidecar, terminal record, lock and transition), its two failures must be the level-0 mesh-angle rejection at "
                "the resolve stage, its resolved count plus failures must equal the declared set, the v1.1 protocol must differ "
                "from the v1 protocol at exactly the declared paths, the post-hoc rejection note bound at the v1.1 code commit "
                "must name the recorded commits, counts, gate, stage time and the absence of a verdict, and the two rejected "
                "designs' v1.1 level-0 meshes must fall below the v1 gate; the rejection is disclosed as lineage and never cited "
                "for a confirmation number"
            ),
        },
        "manuscript_integration": {
            "status": "admitted",
            "gate_kind": GATE_KIND,
            "section_path": SECTION_PATH.as_posix(),
            "section_heading": SECTION_HEADING,
            "section_binding": SECTION_BINDING,
            "generated_tex_path": OUTPUT_PATH.as_posix(),
            "generated_binding": GENERATED_BINDING,
            "manifest_id": MANIFEST_ID,
            "manifest_path": MANIFEST_PATH.as_posix(),
            "gate_id": GATE_ID,
            "artifact_id": ARTIFACT_ID,
            "artifact_claim_id": ARTIFACT_CLAIM_ID,
            "prose_claim_ids": list(PROSE_CLAIM_IDS),
            "table_macros": list(TABLE_MACROS),
            "rule": (
                "Every number in the section is a macro defined here; each macro is bound below to an artifact path, JSON "
                "pointer, formatter and SHA-256, or to a stated derivation over such inputs. Claim-bearing sentences are exact "
                "EvidenceClaim bodies registered in paper/evidence/claims.json; the numerical-screening gate in "
                "paper/evidence/result-gates.json names the typed manifest that admits the section at its recorded outcome (an "
                "accepted material-aware confirmation at the verdict the campaign recorded) without opening any physics level. "
                "The confirmed object is the L1a cusp topology (count and wall-cusp positions) of fifteen field maps under a "
                "linear-iron P2 field; every wall-field and Koch-ratio ratio is a ratio of two field models and never a "
                "probability, nothing here is a saturation, plasma, mirror-probability, thrust, efficiency or "
                "design-recommendation claim, and the predecessor's development rejection is disclosed alongside."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size, and the results tree carries no file the manifest does not bind",
            "recomputation_rule": (
                "the verdict, gates (b) and (c), reported (d), every headline and estimand distribution, the tolerance rule "
                "max(r_w / bore elements, L1a dz), x_w and r_w / L, every matched cusp's shift, tolerance ratio, wall-field "
                "ratio, magnet-strength scaling and rho ratio, every conservative Koch ratio from the wall field and the adjacent "
                "axis peaks, every HEMP-like flag against the recorded threshold, the channel and outside axis-null populations, "
                "the separatrix lean, the level structure, convergence, DOF cap and mesh-angle gate of every solve, the material "
                "regions, and the agreement-table and CSV rows are recomputed from the per-design rows, records and grids and "
                f"must equal the sealed values (counts exactly; floats within a relative tolerance of {FLOAT_TOLERANCE:g})"
            ),
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "lineage_artifacts": {path: v1_bundle.used[path] for path in sorted(v1_bundle.used)},
        "macros": m.items + v1.items,
        "tables": {
            "HmcDesignTable": {"rows": len(design_rows), "source": "artifacts/campaign-result.json#/agreement_table, artifacts/confirmation-dataset.json#/designs (recomputed)"},
            "HmcGateTable": {"rows": len(gate_rows), "source": "artifacts/gates.json#/confirmation, artifacts/confirmation-dataset.json#/headline, #/designs/*/comparison (recomputed)"},
            "HmcFieldTable": {"rows": len(field_rows), "source": "artifacts/protocol.json#/p2, artifacts/campaign-result.json#/execution_mode, artifacts/gates.json#/ram_budget, artifacts/runtime.json, transitions"},
            "HmcDisclosureTable": {"rows": len(disclosure_rows), "source": "v1 results bundle (terminal.json, design-failures.json, protocol.json), artifacts/protocol.json#/predecessor and #/p2/mesh, artifacts/shakedown.json#/mesh_preflight and #/timing_projection, design records mesh_quality"},
        },
        "generator": {
            "path": "paper/scripts/generate_l1b_hemp_confirmation_v1_1_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_l1b_hemp_confirmation_v1_1_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_l1b_hemp_confirmation_v1_1_evidence.py",
        },
        "output": {"path": OUTPUT_PATH.as_posix(), "sha256": sha256_bytes(tex.encode("utf-8"))},
    }
    names = [item["name"] for item in evidence["macros"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate macro names")
    return evidence, tex


def render(repo: Path) -> tuple[bytes, bytes, bytes]:
    evidence, tex = build(repo)
    tex_bytes = tex.encode("utf-8")
    build_config = json.loads((repo / "paper/build-config.json").read_text("utf-8"))
    sidecar = {
        "document_type": "paper-generated-artifact-provenance",
        "schema_version": "1.0",
        "artifact_id": ARTIFACT_ID,
        "claim_ids": [ARTIFACT_CLAIM_ID],
        "claim_status": (
            f"authorized by {ARTIFACT_CLAIM_ID} (quantitative-generated-table) in paper/evidence/claims.json; "
            f"admitted through {GATE_ID} ({GATE_KIND}, recorded outcome {RECORDED_OUTCOME})"
        ),
        "evidence_revision": RESULTS_COMMIT_SHA,
        "source_date_epoch": build_config["source_date_epoch"],
        "generator": evidence["generator"],
        "manifest": {
            "path": EVIDENCE_PATH.as_posix(),
            "sha256": sha256_bytes(canonical_json(evidence)),
            "manifest_id": MANIFEST_ID,
            "gate_manifest_path": MANIFEST_PATH.as_posix(),
        },
        "inputs": [
            {"path": (RESULTS / path).as_posix(), "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["artifacts"].items()
        ],
        "lineage_inputs": [
            {"path": (V1_RESULTS / path).as_posix(), "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["lineage_artifacts"].items()
        ] + [
            {"path": path, "sha256": meta["sha256"], "bytes": meta["bytes"], "revision": meta["revision"]}
            for path, meta in evidence["lineage"]["files"].items()
        ],
        "reference_inputs": [
            {"path": path, "sha256": meta["sha256"], "bytes": meta["bytes"], "revision": meta["revision"]}
            for path, meta in evidence["reference_artifacts"]["files"].items()
        ],
        "bundle_manifest": {
            "path": evidence["bundle"]["manifest_path"],
            "sha256": evidence["bundle"]["manifest_sha256"],
            "git_blob": evidence["binding"]["manifest_git_blob"],
        },
        "dashboard": evidence["dashboard"],
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
        print(f"L1b HEMP confirmation v1.1 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
