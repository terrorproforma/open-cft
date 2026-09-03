"""Generate hash-bound paper evidence for the orbit wall-loss geometry screening v2.

Reads the sealed results bundle of
``modern/experiments/orbit_wall_loss_geometry_screening_v2`` (every manifest file
verified byte-for-byte; no end-of-line tolerance is needed or granted), binds it
to the committed results revision, cross-checks the committed results dashboard
against the same bundle, replays the frozen two-stage allocation rule and the
N -> 2N control from the sealed endpoint tables, recomputes every reported Wilson
interval, floor, pooled design value, headline statistic and the v1 comparison
from the per-case artifacts, binds and verifies the post-hoc manifest publication
disclosure against the bundle, and writes:

* ``paper/evidence/wall-loss-geometry-screening-v2.json`` -- every macro value
  with the artifact path, JSON pointer, formatter and artifact SHA-256 it was
  read from, or the derivation and inputs of a derived macro;
* ``paper/generated/wall-loss-geometry-screening-v2.tex`` -- ``\\newcommand``
  macros and five generated tables (each wrapped in ``\\ArtifactClaim``) for the
  admitted results subsection ``paper/sections/wall-loss-geometry-screening-v2.tex``;
* ``paper/generated/wall-loss-geometry-screening-v2.provenance.json`` --
  generator/input/output hashes in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The study is a screening dataset: collisionless
prescribed-field test-particle electron orbits launched at the midpoints of the
separatrix-bounded wall cells of the accepted cusp-topology catalogue, integrated
in the re-solved linear-vacuum L1a equivalent-current fields of every accepted
sweep-v2 design (plus one launch-design row on the P2-qualified field).  Those
fields are screening fields (not P2-qualified), so no number below is accepted
physical-orbit evidence and none is a plasma or performance claim; every per-cell
number is a collisionless geometric wall-access fraction of the launch
distribution, never a cusp-loss probability.  The dataset is admitted at its
recorded outcome, ``accepted_screening_dataset``, as surrogate and optimisation
input carrying its label.
"""

from __future__ import annotations

import gzip
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
from generate_wall_loss_geometry_screening_v1_evidence import wilson

EXPERIMENT = Path("modern/experiments/orbit_wall_loss_geometry_screening_v2")
RESULTS = EXPERIMENT / "results"
EVIDENCE_PATH = Path("paper/evidence/wall-loss-geometry-screening-v2.json")
OUTPUT_PATH = Path("paper/generated/wall-loss-geometry-screening-v2.tex")
SIDECAR_PATH = Path("paper/generated/wall-loss-geometry-screening-v2.provenance.json")
SECTION_PATH = Path("paper/sections/wall-loss-geometry-screening-v2.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_wall_loss_geometry_screening_v2_dashboard.py")
DASHBOARD_TEMPLATE = Path("modern/visualization/wall-loss-geometry-screening-v2.template.html")
DASHBOARD_HTML = Path("modern/visualization/wall-loss-geometry-screening-v2.html")

# Revisions.  The results tree first exists at the record commit (results only); the
# runtime fix with the post-hoc publication disclosure and the dashboard follow in two
# sibling commits; the merge into feat/sota-foundation adds nothing under the experiment.
RESULTS_COMMIT_SHA = "26029b72222e2b408e87fca3493940b0516b0f5d"
PREREGISTRATION_COMMIT_SHA = "cef1ee59bce5e7bcd2bc7e696d9c8b52394682b1"
DISCLOSURE_COMMIT_SHA = "bb756418b6281c0193f1580c11fb7c32d9898e2e"
DASHBOARD_COMMIT_SHA = "eef7ac827c2595fb3219bd65f8ea1016c210199b"
# References bound at their own admitted revisions.
CATALOGUE_RESULTS_COMMIT_SHA = "cec47f12f5909c5886424bf5d46ac20ce06f1ac5"
V1_RESULTS_COMMIT_SHA = "ab7c28977963822b2ad6eac451d2bafef5185e6c"
V4_RESULTS_COMMIT_SHA = "6922a3cf97d261735266aa1a5a0c0c9683e021ca"
SWEEP_V2_RESULTS_COMMIT_SHA = "f30cb42ec4a8633bf634a3d32ffa5b11f66be97a"

CATALOGUE_PATH = Path("modern/experiments/cusp_topology_search_v3_1/results/artifacts/cusp-cell-catalogue.json")
CATALOGUE_MANIFEST_PATH = Path("modern/experiments/cusp_topology_search_v3_1/results/manifest.json")
V1_DATASET_PATH = Path("modern/experiments/orbit_wall_loss_geometry_screening_v1/results/artifacts/geometry-wall-loss-dataset.json")
V1_MANIFEST_PATH = Path("modern/experiments/orbit_wall_loss_geometry_screening_v1/results/manifest.json")
V4_EXPORT_PATH = Path("modern/experiments/cft_orbit_wall_loss_v4/results/artifacts/coupling-export-only.json")
SWEEP_V2_MANIFEST_PATH = Path("modern/experiments/l1a_geometry_sweep_v2/results/manifest.json")
DISCLOSURE_PATH = EXPERIMENT / "POSTHOC_FINALIZATION.md"
RECOVERY_MODULE_PATH = Path("modern/src/cft_revival/experiment_runtime/recovery.py")
LIFECYCLE_MODULE_PATH = Path("modern/src/cft_revival/experiment_runtime/lifecycle.py")
RECOVERY_TEST_PATH = Path("modern/tests/experiment_runtime/test_recovery.py")

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "WALL-LOSS-GEOMETRY-SCREENING-V2-20260903-377-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/wall-loss-geometry-screening-v2.json")
GATE_ID = "GATE-WALL-LOSS-GEOMETRY-SCREENING-V2"
GATE_KIND = "numerical-screening"
RECORDED_OUTCOME = "accepted-screening-dataset"
ARTIFACT_ID = "TAB-WALL-LOSS-GEOMETRY-SCREENING-V2"
ARTIFACT_CLAIM_ID = "CLM-079"
PROSE_CLAIM_IDS = ("CLM-077", "CLM-078", "CLM-080", "CLM-081", "CLM-082", "CLM-083", "CLM-084", "CLM-085")
SECTION_BINDING = "\\input{sections/wall-loss-geometry-screening-v2.tex}"
GENERATED_BINDING = "\\input{generated/wall-loss-geometry-screening-v2.tex}"
SECTION_HEADING = "Collisionless wall access from the catalogue cells of the accepted sweep designs"
TABLE_MACROS = ("WlhDatasetTable", "WlhCellClassTable", "WlhControlTable", "WlhComparisonTable", "WlhDisclosureTable")
REVISION_MACRO = "GeometryScreeningTwoEvidenceRevision"
MACRO_PREFIX = "Wlh"

EXPERIMENT_ID = "orbit-wall-loss-geometry-screening-v2"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
P2_LABEL = "P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN"
CATALOGUE_LABEL_SWEEP = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
CATALOGUE_LABEL_P2 = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
CAMPAIGN_STATUS = "accepted_screening_dataset"
SCREENING_MODEL = (
    "collisionless prescribed-field relativistic-Boris test-particle electron orbits (orbit_mc) launched at the "
    "midpoints of the separatrix-bounded catalogue cells in linear-vacuum L1a equivalent-current axisymmetric "
    "screening fields (not P2-qualified; not a permanent-magnet or nonlinear-iron material model), plus one "
    "screening launch-design row on the P2-qualified field (not a replication of the accepted wall-loss campaign)"
)
FROZEN_FILES = ("protocol.json", "authorities.json", "shakedown.json", "design-authorities.json")
SET_SWEEP = "sweep_v2"
SET_P2 = "p2_divergent_exit"
POSITION_CLASSES = ("anode_side", "interior", "exit_side")
POSITION_OF_KIND = {"anode_partial": "anode_side", "interior": "interior", "exit_partial": "exit_side"}
POSITION_TOKENS = {"anode_side": "Anode", "interior": "Interior", "exit_side": "Exit"}
STAGE1 = "stage1"
CONTROL = "control"
STAGE2_BLOCKS = ("stage2b1", "stage2b2", "stage2b3")
ESCAPE_SUBCLASSES = ("upstream_anode_plane", "exit_plane", "divergent_section_radial", "unclassified")
NUMERICAL_FAILURES = ("step_limit", "nonfinite_state", "extreme_relativity", "field_failure", "initial_state_invalid")
TIMEOUTS = ("path_timeout", "time_timeout")
WILSON_Z = 1.959963984540054
FLOAT_TOLERANCE = 1e-9
KNOWN_DEFECT_SCAN_N = 4000
EXACT_CASE_SIZES = (128, 16, 64)
ONE_SIDED_SPLIT = 0.8
INJECTOR_FLAGGED_CELL_DESIGN = "l1a-gs-v2-088"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _fixed_signed(value: float, digits: int) -> str:
    text = f"{float(value):+.{digits}f}"
    return text.replace("-", "$-$").replace("+", "$+$")


def _sci_signed(value: float, digits: int) -> str:
    text = _sci(abs(float(value)), digits)
    return ("$-$" if float(value) < 0 else "") + text


FORMATTERS: dict[str, Callable[[Any], str]] = {
    **_BASE_FORMATTERS,
    "mm1": lambda v: f"{1e3 * float(v):.1f}",
    "mm2": lambda v: f"{1e3 * float(v):.2f}",
    "um0": lambda v: f"{1e6 * float(v):.0f}",
    "pct0": lambda v: f"{100.0 * float(v):.0f}\\%",
    "pct2": lambda v: f"{100.0 * float(v):.2f}\\%",
    "pct3": lambda v: f"{100.0 * float(v):.3f}\\%",
    "signed2": lambda v: _fixed_signed(float(v), 2),
    "signed3": lambda v: _fixed_signed(float(v), 3),
    "sci1_signed": lambda v: _sci_signed(float(v), 1),
    "list_fixed3": lambda v: ", ".join(f"{float(x):.3f}" for x in v),
    "list_mm1": lambda v: ", ".join(f"{1e3 * float(x):.1f}" for x in v),
    "list_ident_tt": lambda v: ", ".join(f"\\texttt{{{_BASE_FORMATTERS['ident'](x)}}}" for x in v),
    "list_clauses": lambda v: "; ".join(_tex_escape(str(x)) for x in v),
    "sci3": lambda v: _sci(float(v), 3),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)


def _check_estimate(estimate: dict[str, Any], successes: int, trials: int, label: str) -> None:
    if set(estimate) != {"lower", "method", "probability", "successes", "trials", "upper"} or estimate["method"] != "wilson-95":
        raise ValueError(f"{label}: estimate is not a closed Wilson-95 record")
    if estimate["successes"] != successes or estimate["trials"] != trials:
        raise ValueError(f"{label}: estimate counts differ from the recomputed counts")
    p, lower, upper = wilson(int(successes), int(trials))
    if estimate["probability"] != p or estimate["lower"] != lower or estimate["upper"] != upper:
        raise ValueError(f"{label}: Wilson interval does not recompute")


def wilson_width(successes: int, trials: int) -> float:
    _p, lower, upper = wilson(int(successes), int(trials))
    return upper - lower


def binomial_floor(successes: int, trials: int) -> float:
    """The experiment's floor: sqrt(p(1-p)/n) at the pooled estimate (zero at p in {0, 1})."""

    p = successes / trials
    return math.sqrt(p * (1.0 - p) / trials)


def jeffreys_floor(successes: int, trials: int) -> float:
    """The experiment's floor at the Jeffreys posterior mean (k+1/2)/(n+1); never zero."""

    p = (successes + 0.5) / (trials + 1.0)
    return math.sqrt(p * (1.0 - p) / trials)


def spearman(left: list[float], right: list[float]) -> float:
    """The experiment's Spearman rank correlation (average ranks for ties), operation for operation."""

    if len(left) != len(right) or len(left) < 3:
        raise ValueError("Spearman needs at least three pairs")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        output = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            rank = 0.5 * (position + end) + 1.0
            for index in order[position : end + 1]:
                output[index] = rank
            position = end + 1
        return output

    a = ranks(left)
    b = ranks(right)
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    return cov / math.sqrt(var_a * var_b)


def design_pooled(cells: list[dict[str, Any]], weight: str) -> dict[str, Any]:
    """The experiment's design average of the per-cell P(wall) with declared weights."""

    if weight == "wall_area":
        weights = [float(cell["wall_area_m2"]) for cell in cells]
    elif weight == "launches":
        weights = [float(cell["final"]["trials"]) for cell in cells]
    else:
        raise ValueError("unknown pooling weight")
    total = sum(weights)
    probabilities = [cell["final"]["p_wall"]["probability"] for cell in cells]
    variances = [p * (1.0 - p) / cell["final"]["trials"] for p, cell in zip(probabilities, cells)]
    mean = sum(w * p for w, p in zip(weights, probabilities)) / total
    standard = math.sqrt(sum((w / total) ** 2 * v for w, v in zip(weights, variances)))
    return {
        "weights": [w / total for w in weights],
        "probability": mean,
        "standard_uncertainty": standard,
        "lower": max(0.0, mean - WILSON_Z * standard),
        "upper": min(1.0, mean + WILSON_Z * standard),
        "trials": sum(int(cell["final"]["trials"]) for cell in cells),
    }


def _utc(value: dict[str, Any]) -> datetime:
    if value.get("__cft_type__") != "aware-utc-datetime":
        raise ValueError("timestamp is not an aware UTC datetime record")
    return datetime.fromisoformat(str(value["value"]).replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------------- #
class Bundle:
    """The sealed results bundle, verified file by file against its own manifest."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.root = repo / RESULTS
        manifest_raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.manifest = load_json_bytes(manifest_raw, "results manifest")
        if self.manifest.get("state") != "accepted_result":
            raise ValueError("results manifest state is not accepted_result")
        if self.manifest.get("experiment_id") != EXPERIMENT_ID:
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
            raise ValueError(f"{relative}: payload hash differs from the dataset binding")
        return load_json_bytes(payload, relative)


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo, check=False, capture_output=True,
    ).returncode == 0


def bind_committed(repo: Path, bundle: Bundle) -> dict[str, Any]:
    """Prove the working-tree bundle equals the committed results revision and the chain of commits holds."""

    head = _git(repo, "rev-parse", "HEAD")
    for commit, label in (
        (RESULTS_COMMIT_SHA, "results"),
        (PREREGISTRATION_COMMIT_SHA, "preregistration"),
        (DISCLOSURE_COMMIT_SHA, "disclosure"),
        (DASHBOARD_COMMIT_SHA, "dashboard"),
        (CATALOGUE_RESULTS_COMMIT_SHA, "cusp topology v3.1 results"),
        (V1_RESULTS_COMMIT_SHA, "screening v1 results"),
        (V4_RESULTS_COMMIT_SHA, "wall-loss v4 results"),
        (SWEEP_V2_RESULTS_COMMIT_SHA, "sweep v2 results"),
    ):
        if not _is_ancestor(repo, commit, head):
            raise ValueError(f"{label} commit is not an ancestor of HEAD")
    for earlier, later, label in (
        (PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA, "preregistration -> results"),
        (RESULTS_COMMIT_SHA, DISCLOSURE_COMMIT_SHA, "results -> disclosure"),
        (DISCLOSURE_COMMIT_SHA, DASHBOARD_COMMIT_SHA, "disclosure -> dashboard"),
        (CATALOGUE_RESULTS_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "cusp topology v3.1 results -> preregistration"),
        (V1_RESULTS_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "screening v1 results -> preregistration"),
        (V4_RESULTS_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "wall-loss v4 results -> preregistration"),
        (SWEEP_V2_RESULTS_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "sweep v2 results -> preregistration"),
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
    for name in FROZEN_FILES:
        relative = (EXPERIMENT / name).as_posix()
        frozen = _git(repo, "rev-parse", f"{PREREGISTRATION_COMMIT_SHA}:{relative}")
        recorded = _git(repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{relative}")
        working = _git(repo, "hash-object", "--", relative)
        if not frozen == recorded == working:
            raise ValueError(f"frozen {name} differs between preregistration, results and the working tree")
    # The results commit carries only the results tree; the disclosure commit changes no
    # code file of the experiment (Markdown only), so the preregistered experiment code
    # hash the bundle sealed is the hash of the code that produced it.
    results_files = _git(repo, "diff", "--name-only", f"{RESULTS_COMMIT_SHA}~1", RESULTS_COMMIT_SHA).split()
    if not results_files or any(not path.startswith(RESULTS.as_posix() + "/") for path in results_files):
        raise ValueError("the results commit changes files outside the results tree")
    changed_after_record = _git(repo, "diff", "--name-only", RESULTS_COMMIT_SHA, DISCLOSURE_COMMIT_SHA, "--", EXPERIMENT.as_posix()).split()
    if not changed_after_record or any(not path.endswith(".md") for path in changed_after_record):
        raise ValueError("the disclosure commit changes a non-Markdown file of the experiment")
    subject = _git(repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA)
    return {
        "results_commit": RESULTS_COMMIT_SHA,
        "results_commit_subject": subject,
        "results_tree": results_tree,
        "results_commit_files_outside_results_tree": 0,
        "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
        "disclosure_commit": DISCLOSURE_COMMIT_SHA,
        "disclosure_commit_experiment_files_changed": sorted(changed_after_record),
        "dashboard_commit": DASHBOARD_COMMIT_SHA,
        "manifest_git_blob": committed_blob,
        "manifest_path": manifest_rel,
        "reference_commits": {
            "cusp_topology_v3_1_results": CATALOGUE_RESULTS_COMMIT_SHA,
            "screening_v1_results": V1_RESULTS_COMMIT_SHA,
            "wall_loss_v4_results": V4_RESULTS_COMMIT_SHA,
            "sweep_v2_results": SWEEP_V2_RESULTS_COMMIT_SHA,
        },
    }


def cross_check_dashboard(
    repo: Path, bundle: Bundle, dataset: dict[str, Any], campaign: dict[str, Any], gates: dict[str, Any],
    consumer: dict[str, Any], comparison: dict[str, Any], exclusions: dict[str, Any],
) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of the same bundle; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    template_raw = (repo / DASHBOARD_TEMPLATE).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    if f'CLASSIFICATION = "{CLASSIFICATION}"' not in generator_text or f'LABEL_P2 = "{P2_LABEL}"' not in generator_text:
        raise ValueError("dashboard generator does not pin the screening labels")
    if 'if manifest.get("state") != "accepted_result"' not in generator_text or 'allowed = {"accepted_screening_dataset"}' not in generator_text:
        raise ValueError("dashboard generator does not verify the bundle state and campaign status")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_file_sha256"] != bundle.manifest_sha256:
        raise ValueError("dashboard payload names a different results manifest")
    if identity["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA or identity["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("dashboard payload names a different preregistration commit or experiment")
    if identity["verified_file_count"] != len(bundle.hashes) or identity["artifact_count"] != bundle.manifest["artifact_count"]:
        raise ValueError("dashboard payload file counts differ from the bundle")
    if identity["terminal_file_sha256"] != bundle.manifest["terminal_byte_sha256"] or identity["lock_file_sha256"] != bundle.manifest["lock_byte_sha256"]:
        raise ValueError("dashboard payload terminal/lock hashes differ from the bundle")
    for key in ("protocol_semantic_sha256", "orbit_mc_source_sha256", "field_pipeline_source_sha256", "catalogue_file_sha256"):
        if identity[key] != dataset[key]:
            raise ValueError(f"dashboard payload {key} differs from the sealed dataset")
    if identity["generator_sha256"] != sha256_bytes(_lf(generator_raw)) or identity["template_sha256"] != sha256_bytes(_lf(template_raw)):
        raise ValueError("dashboard payload generator/template hashes differ from the checkout")
    if identity["protocol_file_sha256_lf"] != sha256_bytes(_lf((repo / EXPERIMENT / "protocol.json").read_bytes())):
        raise ValueError("dashboard payload protocol hash differs from the frozen protocol")
    if payload["classification"] != CLASSIFICATION or payload["label_p2"] != P2_LABEL or payload["campaign_status"] != campaign["status"]:
        raise ValueError("dashboard classification or campaign status differs from the bundle")
    if payload["headline"] != dataset["headline"] or payload["claim_boundary"] != dataset["claim_boundary"]:
        raise ValueError("dashboard headline or claim boundary differs from the sealed dataset")
    if payload["classification_statement"] != dataset["classification_statement"] or payload["control_gate"] != dataset["control_gate"]:
        raise ValueError("dashboard classification statement or control gate differs from the sealed dataset")
    if payload["design_count"] != dataset["design_count"] or payload["cell_count"] != dataset["cell_count"]:
        raise ValueError("dashboard counts differ from the sealed dataset")
    if payload["catalogue"] != dataset["cusp_cell_catalogue"] or payload["launch_design"] != dataset["launch_design"]:
        raise ValueError("dashboard catalogue or launch design differs from the sealed dataset")
    if payload["allocation_rule"] != dataset["allocation_rule"] or payload["control_rule"] != dataset["control_rule"] or payload["estimators"] != dataset["estimators"]:
        raise ValueError("dashboard rules differ from the sealed dataset")
    rows = {item["design_key"]: item for item in payload["designs"]}
    if set(rows) != {design["design_key"] for design in dataset["designs"]}:
        raise ValueError("dashboard design rows differ from the sealed dataset")
    cells = {(item["design_key"], item["cell_id"]): item for item in payload["cells"]}
    if len(cells) != dataset["cell_count"]:
        raise ValueError("dashboard cell rows differ from the sealed dataset")
    for design in dataset["designs"]:
        row = rows[design["design_key"]]
        if row["label"] != design["label"] or row["set_id"] != design["set_id"] or row["sealed"] is not design["sealed"]:
            raise ValueError(f"dashboard {design['design_key']} label or seal differs")
        for weight in ("wall_area", "launches"):
            shown = row["pooled"][weight]
            if (shown["p"], shown["lo"], shown["hi"]) != (design["pooled"][weight]["probability"], design["pooled"][weight]["lower"], design["pooled"][weight]["upper"]):
                raise ValueError(f"dashboard {design['design_key']} pooled {weight} differs")
        if row["control"] != {k: design["control"][k] for k in ("n_control", "wall_N", "wall_2N", "delta_p_wall", "discordant", "quantum", "passed")}:
            raise ValueError(f"dashboard {design['design_key']} control differs")
        if row["timestep_passed"] is not design["convergence_flags"]["timestep_passed"] or row["reflections"] != design["diagnostics"]["reflections_final_n"]:
            raise ValueError(f"dashboard {design['design_key']} flag or reflections differ")
        if row["launches"] != {k: design["launch_design"][k] for k in ("stage1_launches", "stage2_launches", "control_launches", "final_launches")}:
            raise ValueError(f"dashboard {design['design_key']} launch counts differ")
        if row["topped_cells"] != design["allocation"]["topped_up_cell_count"] or row["gates"]["allocation_replay"] is not design["allocation_replay"]["passed"]:
            raise ValueError(f"dashboard {design['design_key']} allocation differs")
        v1 = design["v1_comparison"]
        if (row["v1"] is None) is not (v1 is None):
            raise ValueError(f"dashboard {design['design_key']} v1 comparison presence differs")
        if v1 is not None and (row["v1"]["p"] != v1["v1_probability"] or row["v1"]["diff_launch"] != v1["comparison"]["launches"]["difference_v2_minus_v1"] or row["v1"]["diff_area"] != v1["comparison"]["wall_area"]["difference_v2_minus_v1"]):
            raise ValueError(f"dashboard {design['design_key']} v1 comparison differs")
        for cell in design["cells"]:
            shown = cells[(design["design_key"], cell["cell_id"])]
            final = cell["final"]
            if (shown["n"], shown["n1"], shown["k1"], shown["topped"], shown["ready"], shown["position"], shown["kind"]) != (
                final["trials"], cell["stage1"]["trials"], cell["stage1"]["wall_hit"], cell["topped_up"], final["surrogate_ready"], cell["position_class"], cell["kind"]
            ):
                raise ValueError(f"dashboard {design['design_key']} {cell['cell_id']} allocation or class differs")
            for key, estimand in (("wall", "p_wall"), ("refl", "p_reflected"), ("esc", "p_escape"), ("timeout", "p_timeout")):
                estimate = final[estimand]
                if (shown[key]["p"], shown[key]["lo"], shown[key]["hi"], shown[key]["k"], shown[key]["n"]) != (
                    estimate["probability"], estimate["lower"], estimate["upper"], estimate["successes"], estimate["trials"]
                ):
                    raise ValueError(f"dashboard {design['design_key']} {cell['cell_id']} {key} differs")
            if shown["floor"] != final["binomial_floor"] or shown["jfloor"] != final["jeffreys_floor"] or shown["width"] != final["wilson_width"]:
                raise ValueError(f"dashboard {design['design_key']} {cell['cell_id']} floors differ")
            if shown["launch_z_m"] != cell["launch_z_m"] or shown["short_cell"] is not cell["short_cell"] or shown["injector_flag"] is not cell["launch_plane_inside_injector_zone"]:
                raise ValueError(f"dashboard {design['design_key']} {cell['cell_id']} geometry flags differ")
            control = cell["control"]
            if shown["control"] != {"n": control["n_control"], "wall_N": control["wall_N"], "wall_2N": control["wall_2N"], "discordant": control["discordant"], "delta": control["delta_p_wall"]}:
                raise ValueError(f"dashboard {design['design_key']} {cell['cell_id']} control differs")
    if payload["gates"]["validators"] != campaign["validators"] or payload["gates"]["validator_failures"] != gates["validator_failures"]:
        raise ValueError("dashboard validator counts differ from the sealed artifacts")
    for key in ("passed", "structural_all_passed", "allocation_replay_all_passed", "control_gate", "manufactured", "sealed_case_count", "case_count", "exact_authority_replay_count"):
        if payload["gates"][key] != gates[key]:
            raise ValueError(f"dashboard gate {key} differs from gates.json")
    if payload["execution"]["orbit_count"] != campaign["orbit_count"] or payload["execution"]["case_count"] != campaign["case_count"]:
        raise ValueError("dashboard execution counts differ from the campaign result")
    if payload["execution"]["worker_pool_size"] != campaign["execution_mode"]["worker_pool_size"] or payload["execution"]["cases_wall_s"] != campaign["execution_mode"]["cases_wall_s"]:
        raise ValueError("dashboard execution record differs from the campaign result")
    if payload["consumer"]["cases_consumed"] != sum(1 for c in consumer["screening_cases_consumed"] if c["consumption_status"] == "consumed_verified_handoff") or payload["consumer"]["cases_unsealed"] != 0:
        raise ValueError("dashboard consumer counts differ from the consumer record")
    if payload["consumer"]["v4_reference"] != consumer["v4_reference"]["reference_row"] or payload["consumer"]["catalogue_consumed"] != consumer["catalogue_consumed"]:
        raise ValueError("dashboard reference row or catalogue consumption differs from the consumer record")
    if payload["v1_comparison"] != {key: comparison[key] for key in ("design_count", "statement", "spearman_rank_correlation", "mean_difference_v2_minus_v1", "mean_absolute_difference", "interval_overlap_fraction")}:
        raise ValueError("dashboard v1 comparison differs from the sealed comparison")
    if payload["excluded_designs"] != exclusions["excluded"] or payload["evidentiary"] is not True or payload["plan_kind"] != "evidentiary":
        raise ValueError("dashboard exclusions or plan kind differ from the sealed artifacts")
    return {
        "generator_path": DASHBOARD_GENERATOR.as_posix(),
        "generator_sha256_lf": sha256_bytes(_lf(generator_raw)),
        "template_path": DASHBOARD_TEMPLATE.as_posix(),
        "template_sha256_lf": sha256_bytes(_lf(template_raw)),
        "html_path": DASHBOARD_HTML.as_posix(),
        "html_sha256_lf": sha256_bytes(_lf(html_raw)),
        "html_schema": payload["schema"],
        "payload_manifest_sha256": identity["manifest_file_sha256"],
        "rule": (
            "the committed dashboard byte-verifies the bundle against its manifest, cross-checks every cell's pooled "
            "counts against the sealed per-case summaries, embeds its own extraction and pins the manifest SHA-256 and "
            "the preregistration commit; the generator requires that extraction (identity, headline, claim boundary, "
            "rules, every design row with its pooled values, control and v1 comparison, every cell row with its "
            "estimates, floors, flags and control, gates, execution, consumer and v1-comparison blocks) to equal the "
            "sealed artifacts before writing any macro"
        ),
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
                "description": description,
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
                "derived": True, "derivation": derivation, "inputs": inputs, "description": description,
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


def _short(design_key: str) -> str:
    return design_key.split("-")[3] if design_key.startswith("l1a-gs-v2-") else "P2"


# --------------------------------------------------------------------------- #
# Known orbit_mc v1.7 defect (design constraint), recomputed
# --------------------------------------------------------------------------- #
def known_defect_scan(limit: int) -> dict[str, Any]:
    """Count the case sizes n <= limit whose zero-count or full-count Wilson bound is inexact."""

    zero_inexact = [n for n in range(1, limit + 1) if wilson(0, n)[1] > 0.0]
    full_inexact = [n for n in range(1, limit + 1) if wilson(n, n)[2] < 1.0]
    return {
        "scan_limit": limit,
        "zero_count_lower_inexact": len(zero_inexact),
        "full_count_upper_inexact": len(full_inexact),
        "exact_at_both_ends": {n: (n not in zero_inexact and n not in full_inexact) for n in EXACT_CASE_SIZES},
        "n512_zero_inexact": 512 in zero_inexact,
        "n512_full_inexact": 512 in full_inexact,
        "n384_zero_inexact": 384 in zero_inexact,
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(repo: Path) -> tuple[dict[str, Any], str]:  # noqa: C901 - one linear verification pass
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo)
    binding = bind_committed(repo, bundle)
    m = Macros(bundle)
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    dataset = m.doc("artifacts/geometry-wall-loss-dataset-v2.json")
    consumer = m.doc("artifacts/coupling-consumer-record.json")
    exclusions = m.doc("artifacts/design-exclusions.json")
    protocol = m.doc("artifacts/protocol.json")
    authorities = m.doc("artifacts/authorities.json")
    shakedown = m.doc("artifacts/shakedown.json")
    plan = m.doc("artifacts/campaign-plan.json")
    contract = m.doc("artifacts/orbit-mc-contract.json")
    field_binding = m.doc("artifacts/field-pipeline-binding.json")
    manufactured = m.doc("artifacts/manufactured-gates.json")
    runtime = m.doc("artifacts/runtime.json")
    design_authorities = m.doc("artifacts/design-authorities.json")
    allocation = m.doc("artifacts/allocation-decisions.json")
    comparison = m.doc("artifacts/v1-comparison.json")
    catalogue_binding = m.doc("artifacts/catalogue-binding.json")
    transitions = {index: m.doc(f"transitions/{index:04d}-{name}.json") for index, name in (
        (1, "lock-acquired"), (2, "cache-prepared"), (3, "prebundle-started"), (4, "prebundle-completed"),
        (5, "development-started"), (6, "development-accepted"), (7, "assessment-started"), (8, "assessment-accepted"), (9, "terminal"),
    )}
    designs = dataset["designs"]
    headline = dataset["headline"]
    dashboard = cross_check_dashboard(repo, bundle, dataset, campaign, gates, consumer, comparison, exclusions)

    # ---- reference files bound at their own admitted revisions ----------------------
    catalogue_file = _bound_file(repo, CATALOGUE_PATH, CATALOGUE_RESULTS_COMMIT_SHA, "reference-cusp-cell-catalogue", lf_equal=False)
    catalogue_manifest_file = _bound_file(repo, CATALOGUE_MANIFEST_PATH, CATALOGUE_RESULTS_COMMIT_SHA, "reference-cusp-topology-manifest", lf_equal=False)
    v1_dataset_file = _bound_file(repo, V1_DATASET_PATH, V1_RESULTS_COMMIT_SHA, "reference-screening-v1-dataset", lf_equal=False)
    v1_manifest_file = _bound_file(repo, V1_MANIFEST_PATH, V1_RESULTS_COMMIT_SHA, "reference-screening-v1-manifest", lf_equal=False)
    v4_export_file = _bound_file(repo, V4_EXPORT_PATH, V4_RESULTS_COMMIT_SHA, "reference-wall-loss-export", lf_equal=False)
    sweep_manifest_file = _bound_file(repo, SWEEP_V2_MANIFEST_PATH, SWEEP_V2_RESULTS_COMMIT_SHA, "reference-sweep-manifest", lf_equal=False)
    if catalogue_file["sha256"] != dataset["catalogue_file_sha256"] or catalogue_file["sha256"] != protocol["cusp_cell_catalogue"]["catalogue_file_sha256"]:
        raise ValueError("the cusp-cell catalogue on disk differs from the catalogue the campaign bound")
    if catalogue_manifest_file["sha256"] != protocol["cusp_cell_catalogue"]["manifest_file_sha256"] or catalogue_manifest_file["sha256"] != dataset["cusp_cell_catalogue"]["manifest_file_sha256"]:
        raise ValueError("the cusp topology manifest on disk differs from the manifest the campaign bound")
    if v1_dataset_file["sha256"] != protocol["v1_comparison"]["dataset_file_sha256"] or v1_dataset_file["sha256"] != comparison["declaration"]["dataset_file_sha256"]:
        raise ValueError("the v1 dataset on disk differs from the dataset the comparison bound")
    if v1_manifest_file["sha256"] != protocol["v1_comparison"]["results_manifest_file_sha256"] or v1_manifest_file["sha256"] != comparison["declaration"]["results_manifest_file_sha256"]:
        raise ValueError("the v1 results manifest on disk differs from the manifest the comparison bound")
    if v4_export_file["sha256"] != protocol["coupling_consumer"]["v4_export_file_sha256"] or v4_export_file["sha256"] != consumer["v4_reference"]["consumed_export_file_sha256"]:
        raise ValueError("the v4 coupling export on disk differs from the export the consumer bound")
    if sweep_manifest_file["sha256"] != protocol["field_source"]["manifest_file_sha256"] or sweep_manifest_file["sha256"] != field_binding["sweep_manifest_file_sha256"]:
        raise ValueError("the sweep-v2 results manifest on disk differs from the manifest the field pipeline bound")
    catalogue = load_json_bytes((repo / CATALOGUE_PATH).read_bytes(), "cusp-cell catalogue")
    v1_dataset = load_json_bytes((repo / V1_DATASET_PATH).read_bytes(), "screening v1 dataset")
    if catalogue["experiment_id"] != dataset["cusp_cell_catalogue"]["experiment_id"] or catalogue["protocol_semantic_sha256"] != protocol["cusp_cell_catalogue"]["protocol_semantic_sha256"]:
        raise ValueError("catalogue identity differs from the frozen protocol")
    if catalogue["design_count"] != protocol["cusp_cell_catalogue"]["design_count"] or catalogue["stable_design_count"] != protocol["cusp_cell_catalogue"]["stable_design_count"]:
        raise ValueError("catalogue design counts differ from the frozen protocol")
    catalogue_entries = {(entry["set_id"], entry["design_id"]): entry for entry in catalogue["entries"]}
    v1_rows = {design["case_id"]: design for design in v1_dataset["designs"]}

    # ---- disclosure files bound at the disclosure revision --------------------------
    disclosure_file = _bound_file(repo, DISCLOSURE_PATH, DISCLOSURE_COMMIT_SHA, "disclosure-posthoc-finalization", lf_equal=True)
    recovery_file = _bound_file(repo, RECOVERY_MODULE_PATH, DISCLOSURE_COMMIT_SHA, "disclosure-runtime-recovery-module", lf_equal=True)
    lifecycle_file = _bound_file(repo, LIFECYCLE_MODULE_PATH, DISCLOSURE_COMMIT_SHA, "disclosure-runtime-lifecycle-module", lf_equal=True)
    recovery_test_file = _bound_file(repo, RECOVERY_TEST_PATH, DISCLOSURE_COMMIT_SHA, "disclosure-runtime-recovery-tests", lf_equal=True)
    disclosure_text = _lf((repo / DISCLOSURE_PATH).read_bytes()).decode("utf-8")
    lifecycle_text = _lf((repo / LIFECYCLE_MODULE_PATH).read_bytes()).decode("utf-8")
    recovery_text = _lf((repo / RECOVERY_MODULE_PATH).read_bytes()).decode("utf-8")

    def _disclosure(pattern: str, label: str) -> str:
        match = re.search(pattern, disclosure_text)
        if match is None:
            raise ValueError(f"disclosure lacks {label}")
        return match.group(1)

    disclosed_manifest_sha = _disclosure(r"manifest_byte_sha256 ([0-9a-f]{64})", "the manifest hash")
    disclosed_terminal_sha = _disclosure(r"terminal_byte_sha256 ([0-9a-f]{64})", "the terminal hash")
    disclosed_counts = re.search(r"artifact_count (\d+), file_count (\d+), transition_count (\d+), state (\w+)", disclosure_text)
    if disclosed_counts is None:
        raise ValueError("disclosure lacks the recovery counts")
    disclosed_file_count = int(_disclosure(r"this\s+bundle has ([\d,]+) files", "the bundle file count").replace(",", ""))
    disclosed_cap = int(_disclosure(r"allows (\d+) low-level descriptors", "the descriptor cap"))
    disclosed_pin_cap = int(_disclosure(r"`MAX_PINNED_DESCRIPTORS = (\d+)`", "the pin cap"))
    disclosed_validate_count = int(_disclosure(r"returned `accepted_result` with ([\d,]+)\s+artifacts", "the validate artifact count").replace(",", ""))
    disclosed_results_commit = _disclosure(r"The results commit `([0-9a-f]{8})` contains only `results/`", "the results commit")
    disclosed_rerun = "No orbit was re-integrated and no experiment code was changed" in disclosure_text
    if disclosed_manifest_sha != bundle.manifest_sha256 or disclosed_terminal_sha != bundle.manifest["terminal_byte_sha256"]:
        raise ValueError("the disclosure names a different manifest or terminal record than the committed bundle")
    if (int(disclosed_counts.group(1)), int(disclosed_counts.group(2)), int(disclosed_counts.group(3)), disclosed_counts.group(4)) != (
        bundle.manifest["artifact_count"], len(bundle.hashes), len(transitions), bundle.manifest["state"]
    ):
        raise ValueError("the disclosure's recovery counts differ from the committed bundle")
    if disclosed_file_count != len(bundle.hashes) or disclosed_validate_count != bundle.manifest["artifact_count"]:
        raise ValueError("the disclosure's file or artifact counts differ from the committed bundle")
    if disclosed_results_commit != RESULTS_COMMIT_SHA[:8] or not disclosed_rerun:
        raise ValueError("the disclosure names a different results commit or does not state that nothing was rerun")
    if disclosed_cap <= disclosed_pin_cap or disclosed_file_count <= disclosed_cap:
        raise ValueError("the disclosure's descriptor arithmetic does not describe an overflow that the cap prevents")
    pin_cap_match = re.search(r"^MAX_PINNED_DESCRIPTORS = (\d+)$", lifecycle_text, re.MULTILINE)
    if pin_cap_match is None or int(pin_cap_match.group(1)) != disclosed_pin_cap:
        raise ValueError("the runtime's pin cap differs from the disclosed cap")
    if "def finalize_unpublished_attempt" not in recovery_text or "manifest_override" not in recovery_text:
        raise ValueError("the recovery module does not carry the disclosed fail-closed recovery")
    recovery_refusals = sum(1 for phrase in ("manifest.json", "terminal", "transition", "lock") if phrase in recovery_text)
    if recovery_refusals != 4:
        raise ValueError("the recovery module does not name every refusal condition")
    if lock["commit"] != PREREGISTRATION_COMMIT_SHA or lock["experiment_id"] != EXPERIMENT_ID or lock["attempt"] != 1 or lock["immutable"] is not True:
        raise ValueError("execution lock names a different preregistration commit, experiment or attempt")
    if terminal["state"] != bundle.manifest["state"] or terminal["payload"] != campaign or terminal["counts"]["attempt_count"] != 1:
        raise ValueError("terminal record disagrees with the manifest or the campaign result")
    if transitions[9]["transition"] != "terminal" or transitions[9]["details"]["state"] != "accepted_result" or transitions[1]["transition"] != "lock-acquired":
        raise ValueError("transition log does not run lock-acquired -> terminal")
    execution_wall_s = (_utc(transitions[9]["recorded_at_utc"]) - _utc(transitions[1]["recorded_at_utc"])).total_seconds()
    if execution_wall_s <= 0 or execution_wall_s < campaign["execution_mode"]["cases_wall_s"]:
        raise ValueError("execution wall time is not longer than the case pool")

    # ---- internal consistency of the sealed bundle (fail closed on any disagreement) ----
    if campaign["status"] != CAMPAIGN_STATUS or campaign["evidentiary"] is not True or campaign["plan_kind"] != "evidentiary":
        raise ValueError("campaign result is not the accepted evidentiary screening dataset")
    if not (campaign["classification"] == dataset["classification"] == protocol["classification"] == authorities["classification"] == consumer["classification"] == CLASSIFICATION):
        raise ValueError("classification differs between the sealed artifacts")
    if campaign["gates"] != gates or gates["passed"] is not True or gates["binding"] is not True or gates["structural_all_passed"] is not True or gates["allocation_replay_all_passed"] is not True:
        raise ValueError("gates.json disagrees with the campaign result or records a failure")
    if campaign["headline"] != headline or campaign["limitations"] != dataset["claim_boundary"]:
        raise ValueError("campaign headline or limitations differ from the dataset")
    if not (len(designs) == dataset["design_count"] == campaign["design_count"] == gates["design_count"] == headline["design_count"]):
        raise ValueError("design count differs between the dataset and the campaign result")
    if dataset["excluded_designs"] != [] or exclusions["excluded"] != [] or campaign["excluded_design_count"] != 0:
        raise ValueError("the bundle records an excluded design")
    if not (dataset["protocol_semantic_sha256"] == authorities["protocol_semantic_sha256"] == shakedown["protocol_semantic_sha256"] == design_authorities["protocol_semantic_sha256"]):
        raise ValueError("protocol semantic hash differs between the sealed artifacts")
    if not (dataset["orbit_mc_source_sha256"] == authorities["orbit_mc_source_sha256"] == contract["source_sha256"] == shakedown["orbit_mc_source_sha256"]):
        raise ValueError("orbit_mc source hash differs between the sealed artifacts")
    if not (dataset["field_pipeline_source_sha256"] == authorities["field_pipeline_source_sha256"] == field_binding["field_pipeline_source_sha256"] == shakedown["field_pipeline_source_sha256"]):
        raise ValueError("field pipeline source hash differs between the sealed artifacts")
    if not (dataset["catalogue_file_sha256"] == authorities["catalogue_file_sha256"] == design_authorities["catalogue_file_sha256"] == catalogue_binding["catalogue_file_sha256"] == consumer["catalogue_consumed"]["catalogue_file_sha256"]):
        raise ValueError("catalogue hash differs between the sealed artifacts")
    if contract["matches"] is not True or contract["expected"] != contract["observed"] or contract["expected"]["package_version"] != protocol["orbit_mc_contract"]["package_version"]:
        raise ValueError("orbit_mc code contract does not match the frozen protocol")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"]:
        raise ValueError("shakedown artifact differs from the bound authority")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if shakedown["disjointness"]["proven"] is not True or shakedown["timing_projection"]["within_budget_expected"] is not True or shakedown["timing_projection"]["expected"]["within_budget"] is not True:
        raise ValueError("shakedown disjointness or timing projection is not recorded as passed")
    if authorities["shakedown_timing_projection"] != shakedown["timing_projection"] or any(v is not True for v in authorities["shakedown_gate_checks"].values()):
        raise ValueError("shakedown gate checks differ between authorities and shakedown")
    if manufactured["passed"] is not True or manufactured["checks"] != gates["manufactured"]:
        raise ValueError("manufactured gates differ from gates.json or record a failure")
    if field_binding["sweep_manifest_file_sha256"] != protocol["field_source"]["manifest_file_sha256"] or field_binding["sweep_raw_results_file_sha256"] != protocol["field_source"]["raw_results_file_sha256"]:
        raise ValueError("field pipeline binding names a different sweep record")
    if any(dataset["field_source"][k] != protocol["field_source"][k] for k in ("experiment", "field_status", "manifest_file_sha256", "raw_results_file_sha256")):
        raise ValueError("dataset field source differs from the frozen protocol")
    if dataset["field_source"]["refined_diagnostic_coverage"] != protocol["field_source"]["refined_diagnostic"]["coverage"] or dataset["cusp_cell_catalogue"]["catalogue_file_sha256"] != protocol["cusp_cell_catalogue"]["catalogue_file_sha256"]:
        raise ValueError("dataset refined-diagnostic coverage or catalogue differs from the frozen protocol")
    for frozen in FROZEN_FILES:
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != m.doc(f"artifacts/{frozen}"):
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
    sweep_case_ids = list(protocol["designs"]["sweep_case_ids"])
    p2_design = protocol["designs"]["p2_design"]
    if p2_design["included"] is not True or p2_design["set_id"] != SET_P2 or p2_design["label"] != P2_LABEL or len(sweep_case_ids) != protocol["designs"]["sweep_design_count"]:
        raise ValueError("the frozen protocol's design set differs from the registered set")
    p2_key = p2_design["design_key"]
    declared = sorted(sweep_case_ids + [p2_key])
    if len(declared) != len(set(declared)) or declared != sorted(d["design_key"] for d in designs) or plan["design_keys"] != sorted(d["design_key"] for d in designs):
        raise ValueError("declared designs differ from the dataset or the campaign plan")
    if plan["kind"] != "evidentiary" or plan["binding_gates"] is not True or plan["wilson_width_threshold"] != protocol["allocation"]["wilson_width_threshold"]:
        raise ValueError("campaign plan differs from the frozen protocol")
    if plan["stage1_points_per_stratum"] != protocol["allocation"]["stage1_points_per_stratum"] or plan["stage2_points_per_stratum"] != protocol["allocation"]["stage2_points_per_stratum"] or plan["control_fraction"] != protocol["control"]["fraction_per_cell"]:
        raise ValueError("campaign plan allocation differs from the frozen protocol")
    block_count = plan["stage2_points_per_stratum"] // plan["stage1_points_per_stratum"]
    if block_count != 1 + len(STAGE2_BLOCKS) or plan["stage2_points_per_stratum"] % plan["stage1_points_per_stratum"] != 0:
        raise ValueError("stage-2 block structure differs from the registered three top-up blocks")
    strata = protocol["launches"]["strata_per_cell"]
    stage1_per_cell = protocol["allocation"]["stage1_launches_per_cell"]
    final_per_topped = protocol["allocation"]["stage2_launches_per_cell"]
    if stage1_per_cell != plan["stage1_points_per_stratum"] * strata or final_per_topped != plan["stage2_points_per_stratum"] * strata:
        raise ValueError("launch counts per cell do not follow from the points per stratum")
    case_sizes = protocol["cases"]["case_sizes"]
    if (case_sizes["block"], case_sizes["control_of_stage1_cell"], case_sizes["control_of_topped_up_cell"]) != EXACT_CASE_SIZES:
        raise ValueError("case sizes differ from the registered Wilson-exact sizes")
    if case_sizes["control_of_stage1_cell"] != math.ceil(plan["control_fraction"] * stage1_per_cell) or case_sizes["control_of_topped_up_cell"] != math.ceil(plan["control_fraction"] * final_per_topped):
        raise ValueError("control case sizes do not follow from the control fraction")
    threshold = protocol["allocation"]["wilson_width_threshold"]
    readiness_floor = protocol["estimators"]["surrogate_readiness_floor"]
    short_cell_length = protocol["launches"]["short_cell_length_m"]
    bands = protocol["launches"]["radius_bands_of_wall"]
    energies = protocol["launches"]["energies_ev"]
    pitches = protocol["launches"]["pitch_angles_deg"]
    directions = protocol["launches"]["directions"]
    if len(bands) * len(energies) * len(pitches) * len(directions) != 2 * strata or len(energies) * len(pitches) * len(directions) != strata:
        raise ValueError("stratum structure differs from the frozen protocol")
    control_gate = dataset["control_gate"]
    if control_gate["maximum_allowed_change"] != protocol["control"]["maximum_paired_probability_change"] or gates["control_gate"] != {k: control_gate[k] for k in ("n_control", "estimated_bias_2N_minus_N", "discordant", "maximum_allowed_change", "passed")}:
        raise ValueError("control gate differs between dataset, gates and protocol")
    if design_authorities["design_count"] != len(designs) or design_authorities["cell_count"] != dataset["cell_count"] or design_authorities["stage1_launches"] != headline["stage1_launches"]:
        raise ValueError("design authorities differ from the dataset counts")
    if authorities["design_count"] != len(designs) or authorities["cell_count"] != dataset["cell_count"] or authorities["candidate_launches"] != design_authorities["candidate_launches"]:
        raise ValueError("authorities differ from the dataset counts")
    if authorities["candidate_launches"] != dataset["cell_count"] * final_per_topped:
        raise ValueError("candidate launch superset does not equal cells x 512")
    reference = consumer["v4_reference"]
    if reference["passed"] is not True or reference["design_in_screening_set"] is not False or reference["consumed"]["passed"] is not True:
        raise ValueError("the v4 reference row was not consumed as recorded")
    if reference["v4_result_commit"] != protocol["coupling_consumer"]["v4_result_commit"] or reference["v4_result_commit"] != V4_RESULTS_COMMIT_SHA:
        raise ValueError("the v4 reference export names a different result commit")
    row = reference["reference_row"]
    derived = reference["consumed"]["derived"]
    if row["probability"] != derived["probability"] or row["trial_count"] != derived["trials"] or row["confidence_interval_95"] != [derived["wilson_lower"], derived["wilson_upper"]]:
        raise ValueError("the v4 reference row differs from the consumer's derivation")
    if wilson(int(derived["successes"]), int(derived["trials"]))[1:] != (derived["wilson_lower"], derived["wilson_upper"]) or any(v is not True for v in reference["consumed"]["checks"].values()):
        raise ValueError("the v4 reference Wilson interval does not recompute or a consumer check failed")
    consumed_cases = {item["case_key"]: item for item in consumer["screening_cases_consumed"]}
    if len(consumed_cases) != len(consumer["screening_cases_consumed"]) or consumer["catalogue_consumed"] != {"catalogue_file_sha256": dataset["catalogue_file_sha256"], "designs": len(designs), "cells": dataset["cell_count"]}:
        raise ValueError("consumer record case keys or catalogue consumption differ from the dataset")
    if allocation["summary"]["replay_all_passed"] is not True or allocation["summary"]["cells"] != dataset["cell_count"]:
        raise ValueError("allocation decisions record a failed replay or a different cell count")
    if allocation["rule"]["planning_assumption_topped_up_fraction"] != protocol["allocation"]["planning_assumption_topped_up_fraction"]:
        raise ValueError("allocation planning assumption differs from the frozen protocol")
    if not (comparison["design_count"] == len(sweep_case_ids) == headline["sweep_design_count"]):
        raise ValueError("v1 comparison design count differs from the sweep design count")

    # ---- per-design and per-cell replay against the sealed per-case artifacts ------
    all_cells: list[dict[str, Any]] = []
    sweep_cells: list[dict[str, Any]] = []
    p2_cells: list[dict[str, Any]] = []
    cell_class_of: dict[str, str] = {}
    case_count = 0
    total_orbits = 0
    stage1_total = stage2_total = control_total = 0
    reflections_final = 0
    reflections_control = 0
    termination_final = {key: 0 for key in ("wall_hit", "reflected", "domain_escape", *TIMEOUTS, *NUMERICAL_FAILURES)}
    termination_all = dict(termination_final)
    subclasses_final = {key: 0 for key in ESCAPE_SUBCLASSES}
    energy_errors: list[float] = []
    interpolation: list[float] = []
    cross_resolution: list[float] = []
    identity_proven = 0
    control_records: list[dict[str, Any]] = []
    design_wall_area: list[float] = []
    comparison_rows: list[dict[str, Any]] = []
    stored_psi: list[float] = []
    stored_b: list[float] = []
    mu_medians: list[float] = []
    mu_max: list[float] = []
    steps_max: list[int] = []
    tolerance_close: list[float] = []
    exit_direction: dict[int, dict[str, int]] = {+1: {"wall_hit": 0, "trials": 0}, -1: {"wall_hit": 0, "trials": 0}}
    exit_direction_per_design: dict[int, list[float]] = {+1: [], -1: []}
    exit_direction_split: list[float] = []
    exit_wall_side_matches_last_polarity = 0
    straight_exit_cells: list[float] = []
    divergent_exit_cells: list[float] = []
    injector_flagged: list[tuple[str, str, float]] = []
    short_cells: list[tuple[str, str, str, float]] = []
    anode_saturated_below_one: list[int] = []
    designs_with_reflections: list[str] = []
    least_most_pool: list[tuple[float, str]] = []
    representatives: list[str] = []
    for design in designs:
        key = design["design_key"]
        label = f"design {key}"
        set_id = design["set_id"]
        is_p2 = set_id == SET_P2
        if is_p2 != (key == p2_key) or design["label"] != (P2_LABEL if is_p2 else CLASSIFICATION) or design["classification"] != design["label"]:
            raise ValueError(f"{label}: set or label differs from the frozen protocol")
        if design["sealed"] is not True or design["seal_policy"] != "converged" or any(v is not True for v in design["convergence_flags"].values()):
            raise ValueError(f"{label}: design is not sealed with every convergence flag true")
        if design["representative"]:
            representatives.append(key)
        geometry = design["geometry"]
        wall_radius = geometry["wall_radius_m"]
        pitch = geometry["stage_pitch_m"]
        # Catalogue binding: the dataset's cells are the catalogue's cells, byte-bound.
        catalogue_id = design["design_id"] if is_p2 else key
        entry = catalogue_entries[(set_id, catalogue_id)]
        if entry["label"] != (CATALOGUE_LABEL_P2 if is_p2 else CATALOGUE_LABEL_SWEEP) or entry["stable"] is not True:
            raise ValueError(f"{label}: catalogue entry label or stability differs")
        if design["catalogue"]["label"] != entry["label"] or design["catalogue"]["wall_cusp_count"] != entry["wall_cusp_count"] or design["catalogue"]["cell_count"] != entry["cell_count"]:
            raise ValueError(f"{label}: catalogue block differs from the catalogue entry")
        if design["catalogue"]["wall_cusps_z_m"] != [c["z_c_m"] for c in entry["wall_cusps"]] or design["catalogue"]["accepted_field_identity_sha256"] != entry["accepted_field_identity_sha256"]:
            raise ValueError(f"{label}: catalogue cusps or field identity differ from the catalogue entry")
        if design["identities"]["catalogue_accepted_field_identity_sha256"] != entry["accepted_field_identity_sha256"]:
            raise ValueError(f"{label}: catalogue field identity differs from the dataset identities")
        if entry["geometry"]["wall_radius_m"] != wall_radius or entry["geometry"]["stage_pitch_m"] != pitch or entry["geometry"]["injector_length_m"] != geometry["injector_length_m"]:
            raise ValueError(f"{label}: catalogue geometry differs from the dataset geometry")
        if catalogue_binding["cells_bound"][key] != [c["cell_id"] for c in design["cells"]] or [c["cell_id"] for c in entry["cells"]] != [c["cell_id"] for c in design["cells"]]:
            raise ValueError(f"{label}: bound cell ids differ from the catalogue")
        if len(design["cells"]) != entry["cell_count"] or len(design["cells"]) != design["launch_design"]["cell_count"] or design["launch_design"]["strata_per_cell"] != strata:
            raise ValueError(f"{label}: cell count differs")
        if design["launch_design"]["radius_bands_of_wall"] != bands or len(design["launch_design"]["launch_radii_m"]) != len(bands):
            raise ValueError(f"{label}: launch design differs from the frozen protocol")
        for band, radii in zip(bands, design["launch_design"]["launch_radii_m"]):
            if not _close(radii[0], (band["centre_of_wall"] - band["half_width_of_wall"]) * wall_radius) or not _close(radii[1], (band["centre_of_wall"] + band["half_width_of_wall"]) * wall_radius):
                raise ValueError(f"{label}: launch radii do not follow from the bands and the wall radius")
        # Field evidence.
        evidence = m.doc(f"artifacts/field-evidence/{key}.json")
        if evidence["passed"] is not True or any(v is not True for v in evidence["checks"].values()) or evidence["cross_resolution"] is None:
            raise ValueError(f"{label}: field evidence records a failed check or lacks the cross-resolution diagnostic")
        if evidence["cross_resolution"]["b_relative_rms"] != design["field"]["cross_resolution_b_relative_rms"] or design["field"]["cross_resolution_evaluated"] is not True:
            raise ValueError(f"{label}: cross-resolution report differs from the dataset")
        if evidence["accepted_bore_field"]["interpolation_error_report"]["b_relative_rms"] != design["field"]["interpolation_b_relative_rms"]:
            raise ValueError(f"{label}: interpolation report differs from the dataset")
        if evidence["accepted_bore_field"]["max_b_t"] != design["field"]["bore_max_b_t"] or evidence["accepted_bore_field"]["bore_grid"] != design["field"]["bore_grid"]:
            raise ValueError(f"{label}: bore field differs from the dataset")
        if not is_p2:
            if evidence["case_sha256"] != design["identities"]["case_sha256"] or evidence["geometry_sha256"] != design["identities"]["geometry_sha256"] or evidence["source_sha256"] != design["identities"]["source_sha256"] or evidence["config_sha256"] != design["identities"]["config_sha256"]:
                raise ValueError(f"{label}: field identity differs from the dataset")
            if evidence["resolve"]["qoi_replay"]["passed"] is not True or evidence["resolve"]["converged"] is not True or evidence["sweep_record"] != design["field"]["sweep_qois"]:
                raise ValueError(f"{label}: field re-solve, QoI replay or sweep record did not pass")
            stored = evidence["resolve"]["stored_representative"]
            if (stored is None) is design["representative"]:
                raise ValueError(f"{label}: stored-map comparison presence differs from the representative flag")
            if stored is not None:
                if stored["passed"] is not True:
                    raise ValueError(f"{label}: stored representative map was not reproduced")
                stored_psi.append(float(stored["psi_max_abs_difference_wb"]))
                stored_b.append(float(stored["b_max_abs_difference_t"]))
            if design["field"]["status"] != protocol["field_source"]["field_status"]:
                raise ValueError(f"{label}: field status differs from the frozen protocol")
        else:
            if evidence["field_level"] != design["field"]["field_level"] or evidence["v4_protocol_file_sha256"] != p2_design["v4_protocol_file_sha256"] or design["field"]["sweep_qois"] is not None:
                raise ValueError(f"{label}: P2 field level or v4 protocol binding differs")
            if design["representative"] is not True:
                raise ValueError(f"{label}: the P2 row must be a representative (full orbit artifacts)")
        identity_proven += 1
        interpolation.append(float(design["field"]["interpolation_b_relative_rms"]))
        cross_resolution.append(float(design["field"]["cross_resolution_b_relative_rms"]))
        if interpolation[-1] > protocol["field_source"]["adapter_gates"]["maximum_b_relative_rms"] or cross_resolution[-1] > protocol["field_source"]["adapter_gates"]["maximum_cross_resolution_b_relative_rms"]:
            raise ValueError(f"{label}: field adapter gate exceeded")
        # Per-design gates.
        per_design = gates["per_design"][key]
        if per_design != design["gates"] or per_design["passed"] is not True or per_design["sealed"] is not True or per_design["structural_passed"] is not True or per_design["control_flag"] is not True:
            raise ValueError(f"{label}: per-design gates differ from gates.json or record a failure")
        if any(v is not True for v in per_design["checks"].values()) or per_design["timeout_count"] != 0 or per_design["timeout_free"] is not True:
            raise ValueError(f"{label}: a per-design check failed or a timeout was recorded")
        if design["allocation_replay"]["passed"] is not True or any(v is not True for v in design["allocation_replay"]["checks"].values()):
            raise ValueError(f"{label}: allocation replay did not pass")
        replay = allocation["designs"][key]["replay"]
        if {"checks": replay["checks"], "passed": replay["passed"]} != design["allocation_replay"] or allocation["designs"][key]["worker_decision"] != design["allocation"]:
            raise ValueError(f"{label}: allocation decisions differ from the dataset")
        if replay["replayed_decision"]["cells"] != design["allocation"]["cells"] or replay["replayed_decision"]["topped_up_cell_ids"] != design["allocation"]["topped_up_cell_ids"]:
            raise ValueError(f"{label}: the main-process replay of the allocation rule differs from the worker decision")
        if replay["expected_stage2_launch_count"] != design["allocation"]["stage2_launch_count"] or replay["expected_control_launch_count"] != design["launch_design"]["control_launches"] or replay["control_selection_sha256"] != design["control"]["selection_sha256"]:
            raise ValueError(f"{label}: replayed launch counts or control selection differ from the dataset")
        # Cells: replay the rule, the pooling, the floors and the control from the sealed cases.
        design_cases = design["cases"]
        expected_case_keys: set[str] = set()
        design_control = {"n_control": 0, "wall_N": 0, "wall_2N": 0, "discordant": 0}
        design_reflections = 0
        stage1_launches = stage2_launches = control_launches = 0
        for index, cell in enumerate(design["cells"]):
            cell_id = cell["cell_id"]
            cell_label = f"{label} {cell_id}"
            cat = entry["cells"][index]
            for field in ("cell_id", "kind", "z_start_m", "z_end_m", "length_m", "length_over_pitch", "start_cusp_id", "end_cusp_id", "wall_mirror_ratio", "axis_mirror_ratio", "wall_b_min_t", "cusp_wall_b_min_t", "axis_bz_peak_t"):
                if cell[field] != cat[field]:
                    raise ValueError(f"{cell_label}: catalogue field {field} differs")
            if cell["index"] != index or cell["position_class"] != POSITION_OF_KIND[cell["kind"]]:
                raise ValueError(f"{cell_label}: index or position class differs")
            if not _close(cell["length_m"], cell["z_end_m"] - cell["z_start_m"]) or not _close(cell["launch_z_m"], 0.5 * (cell["z_start_m"] + cell["z_end_m"])):
                raise ValueError(f"{cell_label}: length or midpoint launch plane does not recompute")
            if not _close(cell["wall_area_m2"], 2.0 * math.pi * wall_radius * cell["length_m"]) or not _close(cell["length_over_pitch"], cell["length_m"] / pitch):
                raise ValueError(f"{cell_label}: wall area or length/pitch does not recompute")
            if cell["short_cell"] is not (cell["length_m"] < short_cell_length) or cell["launch_plane_inside_injector_zone"] is not (cell["launch_z_m"] < geometry["injector_length_m"]):
                raise ValueError(f"{cell_label}: short-cell or injector flag does not recompute")
            if cell["launch_plane_inside_injector_zone"]:
                injector_flagged.append((key, cell_id, cell["length_m"]))
            if cell["short_cell"]:
                short_cells.append((key, cell_id, cell["position_class"], cell["length_m"]))
            if cell["launch_z_m"] != design["launch_design"]["launch_planes_z_m"][index]:
                raise ValueError(f"{cell_label}: launch plane differs from the launch design")
            stage1_case = design_cases[f"{key}--{cell_id}--{STAGE1}-N"]
            control_case = design_cases[f"{key}--{cell_id}--{CONTROL}-2N"]
            expected_case_keys.update({f"{key}--{cell_id}--{STAGE1}-N", f"{key}--{cell_id}--{CONTROL}-2N"})
            stage1_counts = stage1_case["termination_counts"]
            stage1 = {"trials": stage1_case["trial_count"], "wall_hit": stage1_counts["wall_hit"], "reflected": stage1_counts["reflected"], "domain_escape": stage1_counts["domain_escape"], "timeout": sum(stage1_counts[t] for t in TIMEOUTS)}
            if stage1["trials"] != stage1_per_cell:
                raise ValueError(f"{cell_label}: stage-1 case is not {stage1_per_cell} launches")
            width1 = wilson_width(stage1["wall_hit"], stage1["trials"])
            topped = width1 > threshold
            if cell["topped_up"] is not topped or cell["saturated_after_stage1"] is topped or cell["stage1"] != {**stage1, "wilson_width": width1}:
                raise ValueError(f"{cell_label}: the frozen allocation rule does not replay")
            decision = design["allocation"]["cells"][cell_id]
            if decision != {"saturated": not topped, "stage1_trials": stage1["trials"], "stage1_wall_hit": stage1["wall_hit"], "stage1_wilson_width": width1, "threshold": threshold, "topped_up": topped}:
                raise ValueError(f"{cell_label}: allocation decision does not replay")
            stage2 = None
            if topped:
                stage2 = {"trials": 0, "wall_hit": 0, "reflected": 0, "domain_escape": 0, "timeout": 0}
                for block in STAGE2_BLOCKS:
                    case = design_cases[f"{key}--{cell_id}--{block}-N"]
                    expected_case_keys.add(f"{key}--{cell_id}--{block}-N")
                    counts = case["termination_counts"]
                    if case["trial_count"] != stage1_per_cell:
                        raise ValueError(f"{cell_label} {block}: block is not {stage1_per_cell} launches")
                    stage2["trials"] += case["trial_count"]
                    stage2["wall_hit"] += counts["wall_hit"]
                    stage2["reflected"] += counts["reflected"]
                    stage2["domain_escape"] += counts["domain_escape"]
                    stage2["timeout"] += sum(counts[t] for t in TIMEOUTS)
                if cell["stage2"] != stage2 or cell["stage2_only"]["p_wall"] != dict(zip(("probability", "lower", "upper"), wilson(stage2["wall_hit"], stage2["trials"])), method="wilson-95", successes=stage2["wall_hit"], trials=stage2["trials"]):
                    raise ValueError(f"{cell_label}: stage-2 counts or stage-2-only estimate do not recompute")
            elif cell["stage2"] is not None or cell["stage2_only"] is not None or any(f"{key}--{cell_id}--{block}-N" in design_cases for block in STAGE2_BLOCKS):
                raise ValueError(f"{cell_label}: a saturated cell carries stage-2 cases")
            final = cell["final"]
            n_final = stage1["trials"] + (0 if stage2 is None else stage2["trials"])
            pooled = {name: stage1[name] + (0 if stage2 is None else stage2[name]) for name in ("wall_hit", "reflected", "domain_escape", "timeout")}
            if final["trials"] != n_final or n_final != (final_per_topped if topped else stage1_per_cell) or sum(pooled.values()) != n_final:
                raise ValueError(f"{cell_label}: final launch count does not recompute")
            for name in pooled:
                if final[name] != pooled[name]:
                    raise ValueError(f"{cell_label}: pooled {name} differs")
            for estimand, name in (("p_wall", "wall_hit"), ("p_reflected", "reflected"), ("p_escape", "domain_escape"), ("p_timeout", "timeout")):
                _check_estimate(final[estimand], pooled[name], n_final, f"{cell_label} {estimand}")
            if final["wilson_width"] != wilson_width(pooled["wall_hit"], n_final) or final["binomial_floor"] != binomial_floor(pooled["wall_hit"], n_final) or final["jeffreys_floor"] != jeffreys_floor(pooled["wall_hit"], n_final):
                raise ValueError(f"{cell_label}: width or floors do not recompute")
            if final["readiness_floor"] != readiness_floor or final["surrogate_ready"] is not (final["jeffreys_floor"] <= readiness_floor):
                raise ValueError(f"{cell_label}: surrogate readiness does not recompute")
            # Control: paired N vs 2N over identical launches, replayed from the endpoint tables.
            n_control = control_case["trial_count"]
            if n_control != math.ceil(plan["control_fraction"] * n_final) or n_control != (case_sizes["control_of_topped_up_cell"] if topped else case_sizes["control_of_stage1_cell"]):
                raise ValueError(f"{cell_label}: control case size does not follow from the control fraction")
            n_terminations: dict[str, str] = {}
            for stage_key in [f"{key}--{cell_id}--{STAGE1}-N"] + ([f"{key}--{cell_id}--{block}-N" for block in STAGE2_BLOCKS] if topped else []):
                case = design_cases[stage_key]
                table = bundle.load_gz(f"artifacts/endpoints/{stage_key}.json.gz", case["endpoints_payload_sha256"])
                if table["case_key"] != stage_key or table["cell_id"] != cell_id or table["orbit_artifact_file_sha256"] != case["orbit_artifact_file_sha256"] or table["sealed"] is not True or len(table["rows"]) != case["trial_count"]:
                    raise ValueError(f"{stage_key}: endpoints table differs from the dataset")
                block_index = 0 if table["stage"] == STAGE1 else STAGE2_BLOCKS.index(table["stage"]) + 1
                per_stratum: dict[tuple[float, float, int], int] = {}
                for endpoint in table["rows"]:
                    launch_key = endpoint["launch_key"]
                    if launch_key in n_terminations or endpoint["cell_id"] != cell_id or endpoint["cell_index"] != index or endpoint["stage"] != table["stage"]:
                        raise ValueError(f"{stage_key}: endpoint row is duplicated or mislabelled")
                    if not block_index * plan["stage1_points_per_stratum"] <= endpoint["sobol_index"] < (block_index + 1) * plan["stage1_points_per_stratum"]:
                        raise ValueError(f"{stage_key}: Sobol index outside the block")
                    if endpoint["launch_z_m"] != cell["launch_z_m"] or endpoint["kinetic_energy_ev"] not in energies or endpoint["pitch_angle_deg"] not in pitches or endpoint["parallel_direction"] not in directions:
                        raise ValueError(f"{stage_key}: launch plane or stratum outside the frozen design")
                    band = next((b for b in bands if b["band_id"] == endpoint["band_id"]), None)
                    if band is None or not (band["centre_of_wall"] - band["half_width_of_wall"]) * wall_radius - 1e-12 <= endpoint["launch_r_m"] <= (band["centre_of_wall"] + band["half_width_of_wall"]) * wall_radius + 1e-12:
                        raise ValueError(f"{stage_key}: launch radius outside its band")
                    if endpoint["maximum_relative_energy_error"] != 0.0:
                        raise ValueError(f"{stage_key}: an orbit records an energy drift")
                    stratum = (endpoint["kinetic_energy_ev"], endpoint["pitch_angle_deg"], endpoint["parallel_direction"])
                    per_stratum[stratum] = per_stratum.get(stratum, 0) + 1
                    n_terminations[launch_key] = endpoint["termination"]
                if len(per_stratum) != strata or set(per_stratum.values()) != {plan["stage1_points_per_stratum"]}:
                    raise ValueError(f"{stage_key}: strata are not equally populated")
                counts = case["termination_counts"]
                table_counts = {name: 0 for name in counts}
                for endpoint in table["rows"]:
                    table_counts[endpoint["termination"]] += 1
                if table_counts != counts:
                    raise ValueError(f"{stage_key}: endpoint terminations differ from the case counts")
            control_table = bundle.load_gz(f"artifacts/endpoints/{key}--{cell_id}--{CONTROL}-2N.json.gz", control_case["endpoints_payload_sha256"])
            if control_table["case_key"] != f"{key}--{cell_id}--{CONTROL}-2N" or control_table["stage"] != CONTROL or control_table["timestep"] != "2N" or len(control_table["rows"]) != n_control:
                raise ValueError(f"{cell_label}: control endpoints table differs from the dataset")
            wall_n = wall_2n = discordant = 0
            seen_control: set[str] = set()
            for endpoint in control_table["rows"]:
                launch_key = endpoint["launch_key"]
                if launch_key in seen_control or launch_key not in n_terminations:
                    raise ValueError(f"{cell_label}: control launch is duplicated or has no N-step partner")
                seen_control.add(launch_key)
                wall_n += n_terminations[launch_key] == "wall_hit"
                wall_2n += endpoint["termination"] == "wall_hit"
                discordant += n_terminations[launch_key] != endpoint["termination"]
            control = cell["control"]
            if control != {"delta_p_wall": (wall_2n - wall_n) / n_control, "discordant": discordant, "n_control": n_control, "quantum": 1.0 / n_control, "wall_2N": wall_2n, "wall_N": wall_n}:
                raise ValueError(f"{cell_label}: the paired N -> 2N control does not replay from the endpoint tables")
            if control_case["termination_counts"]["wall_hit"] != wall_2n or design["control"]["per_cell"][cell_id] != control:
                raise ValueError(f"{cell_label}: control case counts differ from the replayed control")
            reflections_control += control_case["termination_counts"]["reflected"]
            for name in design_control:
                design_control[name] += control[name] if name != "n_control" else n_control
            # Bookkeeping for the headline replay.
            cell_class_of[f"{key}--{cell_id}"] = cell["position_class"]
            all_cells.append(cell)
            (p2_cells if is_p2 else sweep_cells).append(cell)
            design_reflections += pooled["reflected"]
            stage1_launches += stage1["trials"]
            stage2_launches += 0 if stage2 is None else stage2["trials"]
            control_launches += n_control
            if cell["position_class"] == "anode_side" and not topped and pooled["wall_hit"] != n_final:
                anode_saturated_below_one.append(pooled["wall_hit"])
            if cell["position_class"] == "exit_side" and not is_p2:
                (divergent_exit_cells if geometry["has_divergent_exit"] else straight_exit_cells).append(final["p_wall"]["probability"])
                if geometry["has_divergent_exit"]:
                    fraction_by_direction: dict[int, float] = {}
                    for direction in (+1, -1):
                        rows_dir = [s for s in design["per_stratum_final"] if s["cell_id"] == cell_id and s["parallel_direction"] == direction]
                        wall_dir = sum(s["wall_hit"] for s in rows_dir)
                        trials_dir = sum(s["trials"] for s in rows_dir)
                        if len(rows_dir) != strata // 2 or trials_dir != n_final // 2:
                            raise ValueError(f"{cell_label}: direction strata do not split the cell in half")
                        exit_direction[direction]["wall_hit"] += wall_dir
                        exit_direction[direction]["trials"] += trials_dir
                        exit_direction_per_design[direction].append(wall_dir / trials_dir)
                        fraction_by_direction[direction] = wall_dir / trials_dir
                    exit_direction_split.append(abs(fraction_by_direction[+1] - fraction_by_direction[-1]))
                    # The alternating stack's last-stage polarity is first_polarity * (-1)^(N-1).
                    last_polarity = geometry["first_polarity"] * (-1) ** (geometry["stage_count"] - 1)
                    wall_side = +1 if fraction_by_direction[+1] > fraction_by_direction[-1] else -1
                    exit_wall_side_matches_last_polarity += wall_side == last_polarity
        if set(design_cases) != expected_case_keys:
            raise ValueError(f"{label}: case set differs from the allocation")
        # Per-case artifacts: summaries, handoffs, orbit sidecars, consumer rows.
        for case_key, case in design_cases.items():
            summary = m.doc(f"artifacts/summaries/{case_key}.json")
            if summary["campaign_id"] != case["campaign_id"] or summary["sealed"] is not True or case["sealed"] is not True or summary["design_key"] != key or summary["cell_id"] != case["cell_id"]:
                raise ValueError(f"{case_key}: summary identity or seal differs from the dataset")
            if summary["stage"] != case["stage"] or summary["timestep"] != case["timestep"] or summary["label"] != design["label"]:
                raise ValueError(f"{case_key}: summary stage, timestep or label differs")
            if summary["summary"]["wall_hit"] != case["wall_hit"] or summary["summary"]["reflected"] != case["reflected"] or summary["summary"]["escaped"] != case["domain_escape"]:
                raise ValueError(f"{case_key}: summary estimates differ from the dataset")
            if summary["summary"]["termination_counts"] != case["termination_counts"] or summary["summary"]["trial_count"] != case["trial_count"]:
                raise ValueError(f"{case_key}: summary termination counts differ from the dataset")
            if summary["orbit_artifact_file_sha256"] != case["orbit_artifact_file_sha256"] or summary["endpoints_payload_sha256"] != case["endpoints_payload_sha256"]:
                raise ValueError(f"{case_key}: summary artifact hashes differ from the dataset")
            if summary["gate_facts"]["orbits_exceeding_energy_gate"] != 0 or summary["gate_facts"]["final_velocity_event_velocity_mismatches"] != 0 or summary["gate_facts"]["maximum_relative_energy_error"] != 0.0:
                raise ValueError(f"{case_key}: summary records an energy or event-velocity defect")
            if summary["gate_facts"]["maximum_wall_endpoint_error_m"] > protocol["gates"]["maximum_wall_endpoint_error_m"] or summary["gate_facts"]["earliest_event_ordering"] is not True:
                raise ValueError(f"{case_key}: wall endpoint or event ordering gate failed")
            if summary["diagnostics"]["termination_counts"] != case["termination_counts"] or summary["authority"]["launch_count"] != case["trial_count"]:
                raise ValueError(f"{case_key}: summary diagnostics or authority differ from the dataset")
            energy_errors.append(float(summary["diagnostics"]["maximum_relative_energy_error"]))
            steps_max.append(int(case["steps"]["max"]))
            expected_strata = strata if case["timestep"] == "N" else range(1, strata + 1)
            if (len(summary["strata"]) != expected_strata if case["timestep"] == "N" else len(summary["strata"]) not in expected_strata):
                raise ValueError(f"{case_key}: stratum count differs from the launch design")
            if sum(s["trials"] for s in summary["strata"]) != case["trial_count"] or sum(s["termination_counts"]["wall_hit"] for s in summary["strata"]) != case["termination_counts"]["wall_hit"]:
                raise ValueError(f"{case_key}: strata do not sum to the case")
            if case["timestep"] == "N" and {s["trials"] for s in summary["strata"]} != {case["trial_count"] // strata}:
                raise ValueError(f"{case_key}: N-step strata are not equally populated")
            for stratum in summary["strata"]:
                for estimand in ("wall_hit", "reflected", "domain_escape", "timeout"):
                    successes = stratum["termination_counts"][estimand] if estimand != "timeout" else sum(stratum["termination_counts"][t] for t in TIMEOUTS)
                    _check_estimate(stratum[estimand], successes, stratum["trials"], f"{case_key} stratum {estimand}")
            sidecar_text = bundle.raw(f"artifacts/orbits/{case_key}.json.sha256").decode("ascii")
            if sidecar_text.split()[0] != case["orbit_artifact_file_sha256"]:
                raise ValueError(f"{case_key}: orbit artifact sidecar differs from the dataset")
            if design["representative"] and f"artifacts/orbits/{case_key}.json.gz" not in bundle.hashes:
                raise ValueError(f"{case_key}: representative orbit artifact is not in the bundle")
            handoff = m.doc(f"artifacts/handoffs/{case_key}.json")
            if bundle.hashes[f"artifacts/handoffs/{case_key}.json"] != case["handoff_sha256"] or handoff["probability"] != case["wall_hit"]["probability"] or handoff["trial_count"] != case["trial_count"]:
                raise ValueError(f"{case_key}: handoff differs from the dataset")
            if handoff["orbit_result_artifact_sha256"] != case["orbit_artifact_file_sha256"] or handoff["schema_version"] != contract["observed"]["handoff_schema_version"] or handoff["integration_status"] != "export_only_pending_consumer_integration":
                raise ValueError(f"{case_key}: handoff binding differs")
            consumed = consumed_cases[case_key]
            if consumed["consumption_status"] != "consumed_verified_handoff" or consumed["consumed"]["passed"] is not True or consumed["handoff_sha256"] != case["handoff_sha256"] or consumed["probability"] != case["wall_hit"]["probability"]:
                raise ValueError(f"{case_key}: consumer record differs from the case")
            if consumed["label"] != design["label"] or consumed["consumed"]["orbit_result_artifact_sha256"] != case["orbit_artifact_file_sha256"] or any(v is not True for v in consumed["consumed"]["checks"].values()):
                raise ValueError(f"{case_key}: a consumer check failed or the consumer binding differs")
            counts = case["termination_counts"]
            if set(counts) != set(termination_all) or sum(counts.values()) != case["trial_count"]:
                raise ValueError(f"{case_key}: termination counts do not partition the trials")
            if sum(counts[t] for t in TIMEOUTS) != case["timeout"]["successes"] or case["timeout_counts"] != {t: counts[t] for t in TIMEOUTS}:
                raise ValueError(f"{case_key}: timeout counts differ from the estimate")
            if sum(case["domain_escape_subclasses"].values()) != counts["domain_escape"] or not set(case["domain_escape_subclasses"]) <= set(ESCAPE_SUBCLASSES):
                raise ValueError(f"{case_key}: escape sub-classes do not partition the escapes")
            case_count += 1
            total_orbits += case["trial_count"]
            for name, value in counts.items():
                termination_all[name] += value
            if case["timestep"] == "N":
                for name, value in counts.items():
                    termination_final[name] += value
                for name, value in case["domain_escape_subclasses"].items():
                    subclasses_final[name] += value
        if design["diagnostics"]["reflections_final_n"] != design_reflections or design["diagnostics"]["reflections_control_2n"] != sum(design_cases[f"{key}--{c['cell_id']}--{CONTROL}-2N"]["termination_counts"]["reflected"] for c in design["cells"]):
            raise ValueError(f"{label}: reflection diagnostics differ from the cases")
        if design["diagnostics"]["domain_escape_subclasses_final_n"] != {sub: sum(design_cases[k]["domain_escape_subclasses"].get(sub, 0) for k in design_cases if design_cases[k]["timestep"] == "N") for sub in ESCAPE_SUBCLASSES}:
            raise ValueError(f"{label}: escape sub-class diagnostics differ from the cases")
        reflections_final += design_reflections
        if design_reflections > 0:
            designs_with_reflections.append(key)
        if design["launch_design"] != {**design["launch_design"], "stage1_launches": stage1_launches, "stage2_launches": stage2_launches, "control_launches": control_launches, "final_launches": stage1_launches + stage2_launches}:
            raise ValueError(f"{label}: launch counts differ from the cells")
        if design["allocation"]["stage2_launch_count"] != stage2_launches or design["allocation"]["topped_up_cell_count"] != sum(c["topped_up"] for c in design["cells"]) or design["allocation"]["topped_up_cell_ids"] != [c["cell_id"] for c in design["cells"] if c["topped_up"]]:
            raise ValueError(f"{label}: allocation summary differs from the cells")
        stage1_total += stage1_launches
        stage2_total += stage2_launches
        control_total += control_launches
        # Design-level control and pooled values.
        n_ctrl = design_control["n_control"]
        delta = (design_control["wall_2N"] - design_control["wall_N"]) / n_ctrl
        design_control_expected = {
            "n_control": n_ctrl, "wall_N": design_control["wall_N"], "wall_2N": design_control["wall_2N"],
            "p_wall_N": design_control["wall_N"] / n_ctrl, "p_wall_2N": design_control["wall_2N"] / n_ctrl, "delta_p_wall": delta,
            "quantum": 1.0 / n_ctrl, "discordant": design_control["discordant"], "discordance_rate": design_control["discordant"] / n_ctrl,
            "maximum_allowed_change": control_gate["maximum_allowed_change"], "passed": abs(delta) <= control_gate["maximum_allowed_change"],
        }
        if {k: v for k, v in design["control"].items() if k not in ("per_cell", "selection_sha256")} != design_control_expected or design["control"]["passed"] is not True:
            raise ValueError(f"{label}: design control does not recompute from the cells")
        control_records.append(design_control_expected)
        for weight in ("wall_area", "launches"):
            expected = design_pooled(design["cells"], weight)
            recorded = design["pooled"][weight]
            if recorded["weight"] != weight or recorded["trials"] != expected["trials"] or recorded["weights"] != expected["weights"]:
                raise ValueError(f"{label}: pooled {weight} weights differ")
            for field in ("probability", "standard_uncertainty", "lower", "upper"):
                if not _close(recorded[field], expected[field]):
                    raise ValueError(f"{label}: pooled {weight} {field} does not recompute")
        if not is_p2:
            design_wall_area.append(design["pooled"]["wall_area"]["probability"])
            least_most_pool.append((design["pooled"]["wall_area"]["probability"], key))
            v1 = v1_rows[key]
            left = v1["reported"]["wall_hit"]
            recorded_v1 = design["v1_comparison"]
            if recorded_v1["v1_probability"] != left["probability"] or recorded_v1["v1_interval"] != [left["lower"], left["upper"]] or recorded_v1["v1_trials"] != left["trials"] or recorded_v1["v1_case"] != "accepted-2N":
                raise ValueError(f"{label}: v1 reference values differ from the bound v1 dataset")
            if recorded_v1["v1_cells_z_m"] != [c["axial_center_m"] for c in v1["launch_design"]["cells"]] or recorded_v1["v1_per_cell_p_wall"] != [v1["per_cell"]["accepted-2N"][c]["wall_hit"]["probability"] for c in sorted(v1["per_cell"]["accepted-2N"])]:
                raise ValueError(f"{label}: v1 cell values differ from the bound v1 dataset")
            for weight in ("wall_area", "launches"):
                item = design["pooled"][weight]
                block = recorded_v1["comparison"][weight]
                if block["v2_probability"] != item["probability"] or block["difference_v2_minus_v1"] != item["probability"] - left["probability"] or block["intervals_overlap"] is not (max(left["lower"], item["lower"]) <= min(left["upper"], item["upper"])):
                    raise ValueError(f"{label}: v1 comparison {weight} does not recompute")
            comparison_rows.append({"design_key": key, "v1": left["probability"], "v2_wall_area": design["pooled"]["wall_area"]["probability"], "v2_launches": design["pooled"]["launches"]["probability"], "comparison": recorded_v1["comparison"], "v1_interval": [left["lower"], left["upper"]]})
        elif design["v1_comparison"] is not None:
            raise ValueError(f"{label}: the P2 row carries a v1 comparison")
        mu = design["diagnostics"]["magnetic_moment_variation"]
        if mu["binding"] is not False or mu["role"] != "diagnostic_only":
            raise ValueError(f"{label}: magnetic-moment variation is recorded as a gate")
        mu_medians.append(float(mu["median_of_case_medians"]))
        mu_max.append(float(mu["max"]))
        tolerance_close.append(float(design["diagnostics"]["tolerance_close_share"]))
        energy_errors.append(float(design["diagnostics"]["maximum_relative_energy_error"]))
    if case_count != campaign["case_count"] or total_orbits != campaign["orbit_count"] or gates["sealed_case_count"] != case_count or gates["exact_authority_replay_count"] != case_count:
        raise ValueError("case or orbit totals differ from the campaign result")
    if len(all_cells) != dataset["cell_count"] or len(sweep_cells) != headline["sweep_cell_count"] or len(p2_cells) != headline["p2_row"]["cells"].__len__():
        raise ValueError("cell totals differ from the headline")
    if len(consumed_cases) != case_count or sum(termination_all[t] for t in TIMEOUTS) != 0 or sum(termination_all[t] for t in NUMERICAL_FAILURES) != 0 or max(energy_errors) != 0.0:
        raise ValueError("the bundle records an unconsumed case, a timeout, a numerical failure or an energy drift")
    if stage1_total + stage2_total + control_total != total_orbits or stage1_total != headline["stage1_launches"] or stage2_total != headline["stage2_launches"] or control_total != headline["control_launches"]:
        raise ValueError("stage totals differ from the headline")
    if stage1_total != dataset["cell_count"] * stage1_per_cell or stage2_total != allocation["summary"]["stage2_launches"] or control_total != allocation["summary"]["control_launches"]:
        raise ValueError("stage totals differ from the allocation summary")
    if total_orbits != headline["total_orbits"] or termination_final["wall_hit"] + termination_final["reflected"] + termination_final["domain_escape"] != stage1_total + stage2_total:
        raise ValueError("N-step terminations do not partition the final launches")

    # ---- headline replay ----
    topped_cells = [c for c in all_cells if c["topped_up"]]
    saturated_cells = [c for c in all_cells if not c["topped_up"]]
    ready_cells = [c for c in all_cells if c["final"]["surrogate_ready"]]
    floors = [c["final"]["jeffreys_floor"] for c in all_cells]
    expected_headline_counts = {
        "cell_count": len(all_cells), "sweep_cell_count": len(sweep_cells), "cells_topped_up": len(topped_cells),
        "cells_saturated_after_stage1": len(saturated_cells), "cells_surrogate_ready": len(ready_cells),
        "sweep_cells_surrogate_ready": sum(c["final"]["surrogate_ready"] for c in sweep_cells),
        "final_n_per_cell_counts": {str(n): sum(c["final"]["trials"] == n for c in all_cells) for n in sorted({c["final"]["trials"] for c in all_cells})},
        "sealed_design_count": len(designs), "control_flag_true_design_count": len(designs), "timeout_free_design_count": len(designs),
        "total_reflections_final_n": reflections_final, "designs_with_reflections": designs_with_reflections,
        "structural_gates_all_passed": True, "allocation_replay_all_passed": True, "p2_row_present": True, "sweep_design_count": len(sweep_case_ids),
    }
    for key, value in expected_headline_counts.items():
        if headline[key] != value:
            raise ValueError(f"headline {key} does not reproduce")
    for key, value in (
        ("fraction_cells_saturated", len(saturated_cells) / len(all_cells)), ("fraction_cells_surrogate_ready", len(ready_cells) / len(all_cells)),
        ("fraction_sweep_cells_surrogate_ready", sum(c["final"]["surrogate_ready"] for c in sweep_cells) / len(sweep_cells)),
        ("jeffreys_floor_median", statistics.median(floors)), ("jeffreys_floor_max", max(floors)),
        ("reflection_fraction_final_n", reflections_final / (stage1_total + stage2_total)),
        ("design_pooled_wall_area_min", min(design_wall_area)), ("design_pooled_wall_area_median", statistics.median(design_wall_area)), ("design_pooled_wall_area_max", max(design_wall_area)),
    ):
        if not _close(headline[key], value):
            raise ValueError(f"headline {key} does not reproduce")
    if headline["least_wall_loss_design_keys"] != [k for _p_, k in sorted(least_most_pool)[:3]] or headline["most_wall_loss_design_keys"] != [k for _p_, k in sorted(least_most_pool, key=lambda item: (-item[0], item[1]))[:3]]:
        raise ValueError("least/most wall-loss designs do not reproduce")
    position_summary: dict[str, dict[str, Any]] = {}
    for position in POSITION_CLASSES:
        rows = [c for c in sweep_cells if c["position_class"] == position]
        p = sorted(c["final"]["p_wall"]["probability"] for c in rows)
        position_summary[position] = {
            "cell_count": len(rows), "p_wall_min": p[0], "p_wall_q1": p[len(p) // 4], "p_wall_median": statistics.median(p), "p_wall_q3": p[(3 * len(p)) // 4], "p_wall_max": p[-1],
            "p_wall_mean": statistics.fmean(p), "p_reflected_mean": statistics.fmean(c["final"]["p_reflected"]["probability"] for c in rows), "p_escape_mean": statistics.fmean(c["final"]["p_escape"]["probability"] for c in rows),
            "topped_up_count": sum(c["topped_up"] for c in rows), "saturated_count": sum(c["saturated_after_stage1"] for c in rows), "surrogate_ready_count": sum(c["final"]["surrogate_ready"] for c in rows),
            "saturated_at_zero": sum(c["final"]["wall_hit"] == 0 for c in rows), "saturated_at_one": sum(c["final"]["wall_hit"] == c["final"]["trials"] for c in rows),
        }
    recorded_positions = headline["per_cell_by_position"]
    if set(recorded_positions) != {*POSITION_CLASSES, "unbounded"} or recorded_positions["unbounded"] != {"cell_count": 0}:
        raise ValueError("position classes differ from the registered classes")
    for position in POSITION_CLASSES:
        for key, value in position_summary[position].items():
            recorded = recorded_positions[position][key]
            if (recorded != value) if isinstance(value, int) else not _close(recorded, value):
                raise ValueError(f"headline per_cell_by_position {position} {key} does not reproduce")
    if set(recorded_positions["anode_side"]) != set(position_summary["anode_side"]):
        raise ValueError("position summary keys differ")
    # P2 row.
    p2 = next(d for d in designs if d["set_id"] == SET_P2)
    if headline["p2_row"] != {"design_key": p2["design_key"], "label": P2_LABEL, "cells": [{"cell_id": c["cell_id"], "kind": c["kind"], "n": c["final"]["trials"], "p_wall": c["final"]["p_wall"]["probability"]} for c in p2["cells"]], "pooled_wall_area": p2["pooled"]["wall_area"]["probability"]}:
        raise ValueError("P2 row does not reproduce")
    # Pooled control gate with the experiment's standard error (per-design paired differences).
    pooled_n = sum(r["n_control"] for r in control_records)
    pooled_wall_n = sum(r["wall_N"] for r in control_records)
    pooled_wall_2n = sum(r["wall_2N"] for r in control_records)
    pooled_discordant = sum(r["discordant"] for r in control_records)
    pooled_delta = (pooled_wall_2n - pooled_wall_n) / pooled_n
    paired: list[float] = []
    for r in control_records:
        plus = r["wall_2N"] - min(r["wall_2N"], r["wall_N"])
        minus = r["wall_N"] - min(r["wall_2N"], r["wall_N"])
        paired.extend([1.0] * plus + [-1.0] * minus + [0.0] * (r["n_control"] - plus - minus))
    bias_se = statistics.pstdev(paired) / math.sqrt(len(paired))
    expected_control = {
        "n_control": pooled_n, "wall_N": pooled_wall_n, "wall_2N": pooled_wall_2n, "p_wall_N": pooled_wall_n / pooled_n, "p_wall_2N": pooled_wall_2n / pooled_n,
        "estimated_bias_2N_minus_N": pooled_delta, "estimated_bias_standard_error": bias_se, "discordant": pooled_discordant, "discordance_rate": pooled_discordant / pooled_n,
        "maximum_allowed_change": control_gate["maximum_allowed_change"], "passed": abs(pooled_delta) <= control_gate["maximum_allowed_change"], "designs_with_control_flag_false": [],
    }
    for key, value in expected_control.items():
        recorded = control_gate[key]
        if (not _close(recorded, value)) if isinstance(value, float) else recorded != value:
            raise ValueError(f"control gate {key} does not reproduce")
    if pooled_n != control_total or control_gate["passed"] is not True or headline["control"] != {k: control_gate[k] for k in ("n_control", "estimated_bias_2N_minus_N", "estimated_bias_standard_error", "discordant", "discordance_rate", "passed")}:
        raise ValueError("pooled control differs from the headline")
    # v1 comparison replay.
    expected_comparison = {
        "spearman_rank_correlation": {w: spearman([r["v1"] for r in comparison_rows], [r[f"v2_{w}"] for r in comparison_rows]) for w in ("wall_area", "launches")},
        "mean_difference_v2_minus_v1": {w: statistics.fmean(r["comparison"][w]["difference_v2_minus_v1"] for r in comparison_rows) for w in ("wall_area", "launches")},
        "mean_absolute_difference": {w: statistics.fmean(abs(r["comparison"][w]["difference_v2_minus_v1"]) for r in comparison_rows) for w in ("wall_area", "launches")},
        "interval_overlap_fraction": {w: sum(r["comparison"][w]["intervals_overlap"] for r in comparison_rows) / len(comparison_rows) for w in ("wall_area", "launches")},
    }
    for key, block in expected_comparison.items():
        for w, value in block.items():
            if not _close(comparison[key][w], value) or not _close(headline["v1_comparison"][key][w], value):
                raise ValueError(f"v1 comparison {key} {w} does not reproduce")
    if comparison["per_design"] != [{"design_key": r["design_key"], "v1": r["v1"], "v2_wall_area": r["v2_wall_area"], "v2_launches": r["v2_launches"]} for r in comparison_rows]:
        raise ValueError("v1 comparison per-design rows do not reproduce")
    if comparison["declaration"] != protocol["v1_comparison"] or comparison["statement"] != protocol["estimators"]["v1_comparison"]:
        raise ValueError("v1 comparison declaration differs from the frozen protocol")
    v1_pooled = [r["v1"] for r in comparison_rows]
    v2_launch_pooled = [r["v2_launches"] for r in comparison_rows]
    # Reflection structure by class.
    reflections_by_class = {position: sum(c["final"]["reflected"] for c in sweep_cells if c["position_class"] == position) for position in POSITION_CLASSES}
    cells_with_reflections_by_class = {position: sum(1 for c in sweep_cells if c["position_class"] == position and c["final"]["reflected"] > 0) for position in POSITION_CLASSES}
    p2_reflections = sum(c["final"]["reflected"] for c in p2_cells)
    if sum(reflections_by_class.values()) + p2_reflections != reflections_final or termination_final["reflected"] != reflections_final:
        raise ValueError("reflections by class do not sum to the total")
    if len(injector_flagged) != 1 or injector_flagged[0][0].startswith(INJECTOR_FLAGGED_CELL_DESIGN) is False:
        raise ValueError("the injector-zone flag does not name exactly the recorded cell")
    known_defect = known_defect_scan(KNOWN_DEFECT_SCAN_N)
    if not all(known_defect["exact_at_both_ends"].values()) or not known_defect["n512_full_inexact"] or known_defect["n512_zero_inexact"] or not known_defect["n384_zero_inexact"]:
        raise ValueError("the recomputed Wilson-exactness scan contradicts the frozen protocol's defect statement")
    defect_statement = protocol["orbit_mc_contract"]["known_defect_v1_7"]["statement"]
    if f"for {known_defect['zero_count_lower_inexact']} of the first {KNOWN_DEFECT_SCAN_N} n" not in defect_statement or f"for {known_defect['full_count_upper_inexact']} of them" not in defect_statement:
        raise ValueError("the frozen protocol's defect counts differ from the recomputed scan")
    # Divergent / straight exit designs.
    divergent_designs = sum(1 for d in designs if d["set_id"] == SET_SWEEP and d["geometry"]["has_divergent_exit"])
    straight_designs = len(sweep_case_ids) - divergent_designs
    if len(divergent_exit_cells) != divergent_designs or len(straight_exit_cells) != straight_designs:
        raise ValueError("exit-side cell classes do not split by the divergent-exit flag as recorded")
    if sum(1 for p in divergent_exit_cells if p == 1.0) + sum(1 for p in straight_exit_cells if p == 1.0) != position_summary["exit_side"]["saturated_at_one"]:
        raise ValueError("exit-side cells at one do not split by the divergent-exit flag")

    # ================================================================== macros ====
    m.add("WlhClassification", "artifacts/campaign-result.json", "/classification", "ident", "screening classification string")
    m.add("WlhPTwoLabel", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/p2_row/label", "ident", "label of the P2-field launch-design row")
    m.add("WlhCatalogueLabelSweep", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/1/catalogue/label", "ident", "catalogue label of the sweep rows")
    m.add("WlhCatalogueLabelPTwo", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/0/catalogue/label", "ident", "catalogue label of the P2 row")
    if designs[0]["set_id"] != SET_P2 or designs[1]["set_id"] != SET_SWEEP:
        raise ValueError("dataset row order differs from the registered P2-first order")
    m.add("WlhTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("WlhCampaignStatus", "artifacts/campaign-result.json", "/status", "ident", "recorded campaign status")
    m.add_derived("WlhRecordedOutcome", RECORDED_OUTCOME, "ident", "recorded outcome admitted by the numerical-screening gate", "constant of the generator; the gate admits the study at exactly this outcome, which names campaign-result.json#/status", [{"artifact": "artifacts/campaign-result.json", "pointer": "/status"}])
    m.add_derived("WlhScreeningModel", SCREENING_MODEL, "text", "screening model label", "constant of the generator naming the orbit model of protocol.json#/claim_boundary/orbit_model in the field model of protocol.json#/claim_boundary/field_level", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/orbit_model"}, {"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}])
    m.add("WlhExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier")
    m.add("WlhAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("WlhLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutability flag")
    m.add("WlhPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    m.add_derived("WlhResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhDisclosureCommit", DISCLOSURE_COMMIT_SHA, "sha_short", "disclosure and runtime-fix commit prefix", "git commit that adds POSTHOC_FINALIZATION.md, the recovery module, the pin cap and their tests; changes only Markdown under the experiment", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhDashboardCommit", DASHBOARD_COMMIT_SHA, "sha_short", "dashboard commit prefix", "git commit that adds the dashboard generator, template and HTML", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhTerminalSha", bundle.manifest["terminal_byte_sha256"], "sha_short", "terminal record SHA-256 prefix", "manifest.terminal_byte_sha256 (equals sha256(results/terminal.json))", [{"artifact": "manifest.json", "pointer": "/terminal_byte_sha256"}])
    m.add_derived("WlhVerifiedFiles", len(bundle.hashes), "int_comma", "bundle files verified byte-for-byte", "count of manifest file entries whose sha256 and size equal the checkout", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlhArtifactCount", bundle.manifest["artifact_count"], "int_comma", "manifest entries (files and directories)", "manifest.artifact_count", [{"artifact": "manifest.json", "pointer": "/artifact_count"}])
    m.add_derived("WlhDirectoryCount", bundle.directory_count, "int", "manifest directory entries", "count of manifest entries with type directory", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlhToleratedEolFiles", 0, "int", "manifest entries accepted through any end-of-line tolerance", "count of manifest file entries whose recorded byte_sha256 differs from sha256(bytes); the generator grants no tolerance and fails on any difference", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlhTransitionCount", len(transitions), "int", "runtime transitions recorded", "count of transitions/NNNN-*.json (lock-acquired ... terminal)", [{"artifact": "transitions/0009-terminal.json", "pointer": "/sequence"}])
    m.add("WlhOrbitMcVersion", "artifacts/orbit-mc-contract.json", "/observed/package_version", "text", "orbit_mc package version")
    m.add("WlhOrbitMcContractMatches", "artifacts/orbit-mc-contract.json", "/matches", "bool", "orbit_mc code contract matches the frozen protocol")
    m.add("WlhOrbitMcSourceSha", "artifacts/orbit-mc-contract.json", "/source_sha256", "sha_short", "orbit_mc source hash prefix")
    m.add("WlhFieldPipelineSha", "artifacts/field-pipeline-binding.json", "/field_pipeline_source_sha256", "sha_short", "field pipeline source hash prefix")
    m.add_derived("WlhFieldPipelineFiles", len(field_binding["field_pipeline_source_files"]), "int", "field pipeline source files hashed", "len(field-pipeline-binding.field_pipeline_source_files)", [{"artifact": "artifacts/field-pipeline-binding.json", "pointer": "/field_pipeline_source_files"}])
    m.add("WlhProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("WlhExperimentCodeSha", "artifacts/authorities.json", "/experiment_code_sha256", "sha_short", "experiment code hash prefix (frozen at preregistration)")
    m.add_derived("WlhExperimentCodeFiles", len(shakedown["experiment_code_files"]), "int", "experiment code files hashed", "len(shakedown.experiment_code_files)", [{"artifact": "artifacts/shakedown.json", "pointer": "/experiment_code_files"}])
    m.add("WlhCatalogueSha", "artifacts/geometry-wall-loss-dataset-v2.json", "/catalogue_file_sha256", "sha_short", "cusp-cell catalogue hash prefix")
    m.add("WlhCatalogueExperimentId", "artifacts/geometry-wall-loss-dataset-v2.json", "/cusp_cell_catalogue/experiment_id", "ident", "catalogue experiment identifier")
    m.add("WlhCatalogueResultCommit", "artifacts/protocol.json", "/cusp_cell_catalogue/result_commit", "text", "catalogue result commit prefix named by the frozen protocol")
    if protocol["cusp_cell_catalogue"]["result_commit"] != CATALOGUE_RESULTS_COMMIT_SHA[:8]:
        raise ValueError("the frozen protocol names a different catalogue result commit")
    m.add("WlhCatalogueDesigns", "artifacts/protocol.json", "/cusp_cell_catalogue/design_count", "int", "designs in the catalogue")
    m.add("WlhBackend", "artifacts/runtime.json", "/backend", "ident", "integration backend")
    m.add("WlhCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "logical CPUs of the host")
    m.add("WlhWorkerPool", "artifacts/campaign-result.json", "/execution_mode/worker_pool_size", "int", "worker pool size")
    m.add("WlhDevice", "execution-lock.json", "/device", "ident", "device string recorded in the execution lock")
    m.add("WlhCasesWallMin", "artifacts/campaign-result.json", "/execution_mode/cases_wall_s", "min1", "wall time of the case pool (min)")
    m.add("WlhAssessmentWallMin", "artifacts/campaign-result.json", "/execution_mode/assessment_wall_s", "min1", "wall time of the assessment phase (min)")
    m.add_derived("WlhExecutionWallMin", execution_wall_s, "min1", "wall time from lock acquisition to the terminal transition (min)", "transitions/0009-terminal.recorded_at_utc - transitions/0001-lock-acquired.recorded_at_utc", [{"artifact": "transitions/0001-lock-acquired.json", "pointer": "/recorded_at_utc"}, {"artifact": "transitions/0009-terminal.json", "pointer": "/recorded_at_utc"}])
    m.add("WlhShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("WlhShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add_derived("WlhShakedownDesigns", len(protocol["shakedown"]["design_case_ids"]) + (1 if protocol["shakedown"]["include_p2_design"] else 0), "int", "shakedown designs (sweep designs plus the P2 row)", "len(protocol.shakedown.design_case_ids) + protocol.shakedown.include_p2_design", [{"artifact": "artifacts/protocol.json", "pointer": "/shakedown/design_case_ids"}, {"artifact": "artifacts/protocol.json", "pointer": "/shakedown/include_p2_design"}])
    m.add("WlhShakedownBlock", "artifacts/protocol.json", "/shakedown/case_sizes/block", "int", "shakedown launches per cell block")
    m.add("WlhShakedownCases", "artifacts/shakedown.json", "/case_count", "int", "shakedown cases")
    m.add("WlhShakedownValidators", "artifacts/shakedown.json", "/validators/passed", "int", "shakedown validator calls passed")
    m.add("WlhShakedownDisjoint", "artifacts/shakedown.json", "/disjointness/proven", "bool", "shakedown launch design disjoint from the evidentiary design")
    m.add("WlhTimingWithinBudget", "artifacts/shakedown.json", "/timing_projection/within_budget_expected", "bool", "expected projection within the wall-time budget")
    m.add("WlhTimingWorstWithinBudget", "artifacts/shakedown.json", "/timing_projection/within_budget_worst_case", "bool", "worst-case projection within the wall-time budget")
    m.add("WlhTimingBudgetMin", "artifacts/shakedown.json", "/timing_projection/budget_wall_seconds", "min1", "wall-time budget (min)")
    m.add("WlhTimingProjectedMin", "artifacts/shakedown.json", "/timing_projection/expected/projected_wall_seconds_at_pool", "min1", "projected wall time at the pool size under the planning assumption (min)")
    m.add("WlhTimingWorstMin", "artifacts/shakedown.json", "/timing_projection/worst_case/projected_wall_seconds_at_pool", "min1", "worst-case projected wall time (every cell topped up; min)")
    m.add("WlhPlanningToppedFraction", "artifacts/protocol.json", "/allocation/planning_assumption_topped_up_fraction", "pct0", "planning assumption for the topped-up fraction")

    # ---- design set, cells and launch design ----
    m.add("WlhDesignCount", "artifacts/campaign-result.json", "/design_count", "int", "designs screened (sweep designs plus the P2 row)")
    m.add("WlhSweepDesignCount", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/sweep_design_count", "int", "sweep-v2 designs screened")
    m.add_derived("WlhDeclaredDesigns", len(declared), "int", "designs declared by the frozen protocol", "len(protocol.designs.sweep_case_ids) + 1 (the P2 design)", [{"artifact": "artifacts/protocol.json", "pointer": "/designs/sweep_case_ids"}])
    m.add_derived("WlhPTwoRowCount", 1, "int", "P2-field launch-design rows", "count(designs[*].set_id == p2_divergent_exit)", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/p2_row_present"}])
    m.add("WlhExcludedDesigns", "artifacts/campaign-result.json", "/excluded_design_count", "int", "designs excluded before integration")
    m.add("WlhCellCount", "artifacts/geometry-wall-loss-dataset-v2.json", "/cell_count", "int", "catalogue cells launched")
    m.add("WlhSweepCellCount", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/sweep_cell_count", "int", "catalogue cells of the sweep designs")
    m.add_derived("WlhPTwoCellCount", len(p2_cells), "int", "catalogue cells of the P2 row", "len(headline.p2_row.cells)", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/p2_row/cells"}])
    for position in POSITION_CLASSES:
        token = POSITION_TOKENS[position]
        m.add(f"Wlh{token}Cells", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/cell_count", "int", f"{position} catalogue cells of the sweep designs")
    m.add_derived("WlhRepresentativeCount", len(representatives), "int", "designs whose full orbit artifacts are in the bundle", "count(designs[*].representative == true)", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs"}])
    m.add("WlhCaseCount", "artifacts/campaign-result.json", "/case_count", "int_comma", "orbit cases")
    m.add("WlhOrbitCount", "artifacts/campaign-result.json", "/orbit_count", "int_comma", "integrated electron orbits")
    m.add("WlhStageOneLaunches", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/stage1_launches", "int_comma", "stage-1 launches")
    m.add("WlhStageTwoLaunches", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/stage2_launches", "int_comma", "stage-2 launches")
    m.add("WlhControlLaunches", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/control_launches", "int_comma", "control launches at the 2N time step")
    m.add_derived("WlhFinalLaunches", stage1_total + stage2_total, "int_comma", "final N-step launches (stage 1 plus stage 2)", "headline.stage1_launches + headline.stage2_launches", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/stage1_launches"}, {"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/stage2_launches"}])
    m.add("WlhStageOnePerCell", "artifacts/protocol.json", "/allocation/stage1_launches_per_cell", "int", "stage-1 launches per cell")
    m.add("WlhFinalPerTopped", "artifacts/protocol.json", "/allocation/stage2_launches_per_cell", "int", "final launches per topped-up cell")
    m.add("WlhStrataPerCell", "artifacts/protocol.json", "/launches/strata_per_cell", "int", "strata per cell")
    m.add("WlhStageOnePointsPerStratum", "artifacts/protocol.json", "/allocation/stage1_points_per_stratum", "int", "scrambled-Sobol points per stratum at stage 1")
    m.add("WlhStageTwoPointsPerStratum", "artifacts/protocol.json", "/allocation/stage2_points_per_stratum", "int", "scrambled-Sobol points per stratum after a top-up")
    m.add_derived("WlhStageTwoBlocks", len(STAGE2_BLOCKS), "int", "top-up blocks per topped-up cell", "stage2_points_per_stratum / stage1_points_per_stratum - 1", [{"artifact": "artifacts/campaign-plan.json", "pointer": "/stage2_points_per_stratum"}, {"artifact": "artifacts/campaign-plan.json", "pointer": "/stage1_points_per_stratum"}])
    m.add("WlhWidthThreshold", "artifacts/protocol.json", "/allocation/wilson_width_threshold", "g", "stage-1 Wilson width threshold of the allocation rule")
    m.add("WlhControlFraction", "artifacts/protocol.json", "/control/fraction_per_cell", "g", "control fraction of every cell's final launches")
    m.add("WlhControlOfStageOne", "artifacts/protocol.json", "/cases/case_sizes/control_of_stage1_cell", "int", "control launches of a saturated cell")
    m.add("WlhControlOfTopped", "artifacts/protocol.json", "/cases/case_sizes/control_of_topped_up_cell", "int", "control launches of a topped-up cell")
    m.add("WlhEnergies", "artifacts/protocol.json", "/launches/energies_ev", "list_g", "launch kinetic energies (eV)")
    m.add("WlhPitches", "artifacts/protocol.json", "/launches/pitch_angles_deg", "list_g", "launch pitch angles (deg)")
    m.add("WlhDirections", "artifacts/protocol.json", "/launches/directions", "list_int", "parallel launch directions")
    m.add_derived("WlhBandCount", len(bands), "int", "launch radius bands", "len(protocol.launches.radius_bands_of_wall)", [{"artifact": "artifacts/protocol.json", "pointer": "/launches/radius_bands_of_wall"}])
    m.add_derived("WlhBandOneLo", bands[0]["centre_of_wall"] - bands[0]["half_width_of_wall"], "fixed3", "inner edge of the first launch band (fraction of the wall radius)", "centre_of_wall - half_width_of_wall of band 0", [{"artifact": "artifacts/protocol.json", "pointer": "/launches/radius_bands_of_wall/0"}])
    m.add_derived("WlhBandOneHi", bands[0]["centre_of_wall"] + bands[0]["half_width_of_wall"], "fixed3", "outer edge of the first launch band", "centre_of_wall + half_width_of_wall of band 0", [{"artifact": "artifacts/protocol.json", "pointer": "/launches/radius_bands_of_wall/0"}])
    m.add_derived("WlhBandTwoLo", bands[1]["centre_of_wall"] - bands[1]["half_width_of_wall"], "fixed3", "inner edge of the second launch band", "centre_of_wall - half_width_of_wall of band 1", [{"artifact": "artifacts/protocol.json", "pointer": "/launches/radius_bands_of_wall/1"}])
    m.add_derived("WlhBandTwoHi", bands[1]["centre_of_wall"] + bands[1]["half_width_of_wall"], "fixed3", "outer edge of the second launch band", "centre_of_wall + half_width_of_wall of band 1", [{"artifact": "artifacts/protocol.json", "pointer": "/launches/radius_bands_of_wall/1"}])
    m.add("WlhLaunchPlaneRule", "artifacts/protocol.json", "/launches/launch_plane_rule", "ident", "launch plane rule")
    m.add("WlhSobolGenerator", "artifacts/protocol.json", "/launches/sobol/generator", "text", "scrambled-Sobol generator statement")
    m.add("WlhShortCellLengthMm", "artifacts/protocol.json", "/launches/short_cell_length_m", "mm1", "short-cell flag threshold (mm)")
    m.add("WlhEstimator", "artifacts/protocol.json", "/launches/estimator_policy", "ident", "estimator policy")
    m.add("WlhIntervalMethod", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/0/cells/0/final/p_wall/method", "ident", "interval method")
    m.add("WlhReadinessFloor", "artifacts/protocol.json", "/estimators/surrogate_readiness_floor", "g", "Jeffreys-floor threshold for surrogate readiness")
    timestep_names = sorted(protocol["orbit_geometry_rule"]["timestep_policies"], key=len)
    if timestep_names != ["N", "2N"]:
        raise ValueError("timestep policies differ from the registered N and 2N")
    m.add_derived("WlhTimestepN", timestep_names[0], "ident", "reported time-step policy name", "the shorter key of protocol.orbit_geometry_rule.timestep_policies", [{"artifact": "artifacts/protocol.json", "pointer": "/orbit_geometry_rule/timestep_policies"}])
    m.add_derived("WlhTimestepControl", timestep_names[1], "ident", "control time-step policy name", "the longer key of protocol.orbit_geometry_rule.timestep_policies", [{"artifact": "artifacts/protocol.json", "pointer": "/orbit_geometry_rule/timestep_policies"}])
    m.add("WlhRotationN", "artifacts/protocol.json", "/orbit_geometry_rule/timestep_policies/N/max_rotation_rad", "g", "maximum gyro-rotation per step, policy N (rad)")
    m.add("WlhRotationTwoN", "artifacts/protocol.json", "/orbit_geometry_rule/timestep_policies/2N/max_rotation_rad", "g", "maximum gyro-rotation per step, policy 2N (rad)")
    m.add("WlhMaxPathLengths", "artifacts/protocol.json", "/orbit_geometry_rule/max_path_channel_lengths", "g", "path budget in channel lengths")
    m.add("WlhMaxTimeFactor", "artifacts/protocol.json", "/orbit_geometry_rule/max_time_transit_factor", "g", "time budget factor")
    m.add("WlhSlowestEnergy", "artifacts/protocol.json", "/orbit_geometry_rule/slowest_energy_ev", "g", "slowest launch energy for the time budget (eV)")
    m.add("WlhEventTolerance", "artifacts/protocol.json", "/orbit_geometry_rule/event_tolerance_m", "sci1", "event tolerance (m)")
    m.add("WlhMaxSteps", "artifacts/protocol.json", "/orbit_geometry_rule/max_steps", "int_comma", "step budget per orbit")
    m.add("WlhMaxGamma", "artifacts/protocol.json", "/orbit_geometry_rule/maximum_gamma", "g", "Lorentz-factor guard")
    m.add_derived("WlhStepsMax", max(steps_max), "int_comma", "largest step count of any case", "max over cases of steps.max", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs"}])

    # ---- field provenance ----
    m.add("WlhFieldStatus", "artifacts/field-pipeline-binding.json", "/field_status", "ident", "field status label of the sweep rows")
    m.add("WlhPTwoFieldLevel", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/0/field/field_level", "text", "field level statement of the P2 row")
    m.add("WlhSweepClassification", "artifacts/protocol.json", "/field_source/classification", "ident", "classification of the source sweep")
    m.add_derived("WlhFieldModelLevel", protocol["field_source"]["classification"].split("_")[0], "text", "field model level named by the sweep classification", "protocol.field_source.classification.split('_')[0]", [{"artifact": "artifacts/protocol.json", "pointer": "/field_source/classification"}])
    if protocol["field_source"]["classification"].split("_")[0] != "L1a" or "L1A" not in CLASSIFICATION:
        raise ValueError("field model level differs between the sweep classification and the screening classification")
    m.add("WlhFieldLevelStatement", "artifacts/protocol.json", "/claim_boundary/field_level", "text", "field level statement of the claim boundary")
    m.add("WlhOrbitModelStatement", "artifacts/protocol.json", "/claim_boundary/orbit_model", "text", "orbit model statement of the claim boundary")
    m.add("WlhEstimandStatement", "artifacts/protocol.json", "/claim_boundary/estimand", "text", "estimand statement of the claim boundary")
    m.add("WlhSweepManifestSha", "artifacts/field-pipeline-binding.json", "/sweep_manifest_file_sha256", "sha_short", "source sweep manifest hash prefix")
    m.add("WlhSweepRawSha", "artifacts/field-pipeline-binding.json", "/sweep_raw_results_file_sha256", "sha_short", "source sweep raw-results hash prefix")
    m.add("WlhResolveSolver", "artifacts/protocol.json", "/field_source/resolve/solver", "ident", "field re-solve function")
    m.add("WlhGridRadial", "artifacts/protocol.json", "/field_source/resolve/domain/radial_intervals", "int", "radial intervals of the re-solve")
    m.add("WlhGridAxial", "artifacts/protocol.json", "/field_source/resolve/domain/axial_intervals", "int", "axial intervals of the re-solve")
    m.add("WlhInterpolationGate", "artifacts/protocol.json", "/field_source/adapter_gates/maximum_b_relative_rms", "pct0", "interpolation gate on the relative rms field error")
    m.add("WlhCrossResolutionGate", "artifacts/protocol.json", "/field_source/adapter_gates/maximum_cross_resolution_b_relative_rms", "pct0", "cross-resolution gate on the relative rms field error")
    m.add("WlhRefinement", "artifacts/protocol.json", "/field_source/refined_diagnostic/refinement", "int", "refinement factor of the refined re-solve")
    m.add_derived("WlhIdentityProvenDesigns", identity_proven, "int", "designs whose field evidence passed every check", "count of field-evidence records with passed == true and every check true", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/checks"}])
    m.add_derived("WlhInterpolationRmsMax", max(interpolation), "pct2", "largest interpolation relative rms field error over the designs", "max over designs of field.interpolation_b_relative_rms", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs"}])
    m.add_derived("WlhCrossResolutionDesigns", len(cross_resolution), "int", "designs with a refined re-solve and cross-resolution diagnostic (every design)", "count of designs whose field.cross_resolution_b_relative_rms is not null", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs"}])
    m.add_derived("WlhCrossResolutionRmsMax", max(cross_resolution), "pct2", "largest cross-resolution relative rms field error over every design", "max over designs of field.cross_resolution_b_relative_rms", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs"}])
    m.add_derived("WlhStoredMapDesigns", len(stored_psi), "int", "sweep representatives whose stored maps were reproduced node-wise", "count of field-evidence records with resolve.stored_representative.passed == true", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/resolve/stored_representative"}])
    m.add_derived("WlhStoredPsiMaxDiff", max(stored_psi), "sci1", "largest stored-map flux difference over the sweep representatives (Wb)", "max over representative field-evidence records of resolve.stored_representative.psi_max_abs_difference_wb", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/resolve/stored_representative"}])
    m.add_derived("WlhStoredBMaxDiff", max(stored_b), "sci1", "largest stored-map field difference over the sweep representatives (T)", "max over representative field-evidence records of resolve.stored_representative.b_max_abs_difference_t", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/resolve/stored_representative"}])
    sweep_designs = [d for d in designs if d["set_id"] == SET_SWEEP]
    design_inputs = [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs"}]
    m.add_derived("WlhLengthMinMm", min(d["geometry"]["chamber_length_m"] for d in sweep_designs), "mm1", "shortest sweep chamber length (mm)", "min over sweep designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlhLengthMaxMm", max(d["geometry"]["chamber_length_m"] for d in sweep_designs), "mm1", "longest sweep chamber length (mm)", "max over sweep designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlhRadiusMinMm", min(d["geometry"]["wall_radius_m"] for d in sweep_designs), "mm2", "smallest sweep wall radius (mm)", "min over sweep designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlhRadiusMaxMm", max(d["geometry"]["wall_radius_m"] for d in sweep_designs), "mm2", "largest sweep wall radius (mm)", "max over sweep designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlhStageCounts", sorted({d["geometry"]["stage_count"] for d in sweep_designs}), "list_int", "stage counts present in the sweep designs", "sorted(set(geometry.stage_count))", design_inputs)
    m.add_derived("WlhDivergentDesigns", divergent_designs, "int", "sweep designs with a divergent exit section", "count(geometry.has_divergent_exit == true) over sweep designs", design_inputs)
    m.add_derived("WlhStraightDesigns", straight_designs, "int", "sweep designs with a full-length straight channel", "count(geometry.has_divergent_exit == false) over sweep designs", design_inputs)
    m.add("WlhPTwoWallRadiusMm", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/0/geometry/wall_radius_m", "mm2", "wall radius of the P2 design (mm)")
    m.add("WlhPTwoLengthMm", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/0/geometry/chamber_length_m", "mm1", "chamber length of the P2 design (mm)")
    m.add("WlhPTwoWallCusps", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/0/catalogue/wall_cusp_count", "int", "wall cusps of the P2 design in the catalogue")

    # ---- gates and verification ----
    m.add("WlhGatesPassed", "artifacts/gates.json", "/passed", "bool", "binding gates passed")
    m.add("WlhStructuralPassed", "artifacts/gates.json", "/structural_all_passed", "bool", "structural gates passed for every design")
    m.add("WlhAllocationReplayPassed", "artifacts/gates.json", "/allocation_replay_all_passed", "bool", "allocation-rule replay passed for every design")
    m.add("WlhValidatorsPassed", "artifacts/campaign-result.json", "/validators/passed", "int_comma", "validator calls passed")
    m.add("WlhValidatorsFailed", "artifacts/campaign-result.json", "/validators/failed", "int", "validator failures")
    m.add("WlhSealedCases", "artifacts/gates.json", "/sealed_case_count", "int_comma", "sealed orbit cases")
    m.add("WlhReplayCount", "artifacts/gates.json", "/exact_authority_replay_count", "int_comma", "exact authority replays")
    m.add("WlhSealedDesigns", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/sealed_design_count", "int", "designs sealed")
    m.add("WlhControlFlagDesigns", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/control_flag_true_design_count", "int", "designs whose paired control flag is true")
    m.add("WlhTimeoutFreeDesigns", "artifacts/gates.json", "/timeout_free_design_count", "int", "designs without a timeout")
    m.add_derived("WlhFailedCases", campaign["case_count"] - gates["sealed_case_count"], "int", "cases not sealed", "campaign.case_count - gates.sealed_case_count", [{"artifact": "artifacts/campaign-result.json", "pointer": "/case_count"}, {"artifact": "artifacts/gates.json", "pointer": "/sealed_case_count"}])
    m.add_derived("WlhFailedDesigns", sum(1 for v in gates["per_design"].values() if v["passed"] is not True), "int", "designs whose per-design gates failed", "count(gates.per_design[*].passed != true)", [{"artifact": "artifacts/gates.json", "pointer": "/per_design"}])
    m.add_derived("WlhPerDesignChecks", len(next(iter(gates["per_design"].values()))["checks"]), "int", "per-design structural checks", "len(gates.per_design[*].checks)", [{"artifact": "artifacts/gates.json", "pointer": "/per_design"}])
    m.add_derived("WlhManufacturedChecks", len(manufactured["checks"]), "int", "manufactured verification checks", "len(manufactured-gates.checks)", [{"artifact": "artifacts/manufactured-gates.json", "pointer": "/checks"}])
    m.add("WlhManufacturedPassed", "artifacts/manufactured-gates.json", "/passed", "bool", "manufactured checks passed")
    m.add("WlhCpuParityDiff", "artifacts/manufactured-gates.json", "/cpu_parity/maximum_relative_velocity_difference", "g", "numpy versus Warp CPU relative velocity difference")
    m.add("WlhCudaParityStatus", "artifacts/manufactured-gates.json", "/cuda_parity/status", "ident", "CUDA parity status")
    m.add("WlhEnergyGate", "artifacts/protocol.json", "/gates/maximum_relative_energy_error", "sci1", "relative energy drift gate")
    m.add_derived("WlhEnergyErrorMax", max(energy_errors), "g", "largest relative kinetic-energy drift over every case", "max over case summaries and designs of maximum_relative_energy_error", design_inputs)
    m.add("WlhWallEndpointGate", "artifacts/protocol.json", "/gates/maximum_wall_endpoint_error_m", "sci1", "wall endpoint gate (m)")
    m.add("WlhControlGateMax", "artifacts/protocol.json", "/control/maximum_paired_probability_change", "g", "pooled paired N -> 2N control gate")
    m.add_derived("WlhTimeouts", sum(termination_all[t] for t in TIMEOUTS), "int", "timeouts over every case", "sum over cases of termination_counts.path_timeout + time_timeout", design_inputs)
    m.add_derived("WlhNumericalFailures", sum(termination_all[t] for t in NUMERICAL_FAILURES), "int", "numerical failures over every case", "sum over cases of the five numerical-failure termination counts", design_inputs)
    m.add_derived("WlhTotalOrbits", total_orbits, "int_comma", "orbits summed over every case", "sum over cases of trial_count", design_inputs)
    m.add_derived("WlhToleranceCloseShareMin", min(tolerance_close), "pct1", "smallest per-design share of tolerance-close terminations", "min over designs of diagnostics.tolerance_close_share", design_inputs)
    m.add_derived("WlhToleranceCloseShareMax", max(tolerance_close), "pct1", "largest per-design share of tolerance-close terminations", "max over designs of diagnostics.tolerance_close_share", design_inputs)
    m.add_derived("WlhMuMedianMin", min(mu_medians), "fixed2", "smallest per-design median magnetic-moment variation", "min over designs of diagnostics.magnetic_moment_variation.median_of_case_medians", design_inputs)
    m.add_derived("WlhMuMedianMax", max(mu_medians), "fixed2", "largest per-design median magnetic-moment variation", "max over designs of diagnostics.magnetic_moment_variation.median_of_case_medians", design_inputs)
    m.add_derived("WlhMuMaxMax", max(mu_max), "fixed1", "largest magnetic-moment variation of any orbit", "max over designs of diagnostics.magnetic_moment_variation.max", design_inputs)
    m.add("WlhMuRole", "artifacts/protocol.json", "/diagnostics/magnetic_moment_variation/role", "ident", "role of the magnetic-moment diagnostic")

    # ---- allocation and floors ----
    m.add("WlhCellsToppedUp", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/cells_topped_up", "int", "cells topped up to the final launch count")
    m.add("WlhCellsSaturated", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/cells_saturated_after_stage1", "int", "cells saturated after stage 1")
    m.add("WlhFractionSaturated", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/fraction_cells_saturated", "pct0", "share of cells saturated after stage 1")
    m.add_derived("WlhFractionToppedUp", len(topped_cells) / len(all_cells), "pct0", "share of cells topped up", "headline.cells_topped_up / headline.cell_count", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/cells_topped_up"}, {"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/cell_count"}])
    for position in POSITION_CLASSES:
        token = POSITION_TOKENS[position]
        m.add(f"Wlh{token}ToppedUp", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/topped_up_count", "int", f"{position} cells topped up")
        m.add(f"Wlh{token}Saturated", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/saturated_count", "int", f"{position} cells saturated after stage 1")
        m.add(f"Wlh{token}Ready", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/surrogate_ready_count", "int", f"{position} cells surrogate-ready")
        m.add(f"Wlh{token}AtOne", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/saturated_at_one", "int", f"{position} cells whose every launch reached the wall")
        m.add(f"Wlh{token}AtZero", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/saturated_at_zero", "int", f"{position} cells without a wall hit")
        m.add(f"Wlh{token}PMin", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_wall_min", "fixed3", f"smallest {position} cell wall-access fraction")
        m.add(f"Wlh{token}PQOne", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_wall_q1", "fixed3", f"first quartile of the {position} cell wall-access fraction")
        m.add(f"Wlh{token}PMedian", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_wall_median", "fixed3", f"median {position} cell wall-access fraction")
        m.add(f"Wlh{token}PQThree", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_wall_q3", "fixed3", f"third quartile of the {position} cell wall-access fraction")
        m.add(f"Wlh{token}PMax", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_wall_max", "fixed3", f"largest {position} cell wall-access fraction")
        m.add(f"Wlh{token}PMean", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_wall_mean", "fixed3", f"mean {position} cell wall-access fraction")
        m.add(f"Wlh{token}ReflectedMean", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_reflected_mean", "fixed3", f"mean {position} cell reflection fraction")
        m.add(f"Wlh{token}EscapeMean", "artifacts/geometry-wall-loss-dataset-v2.json", f"/headline/per_cell_by_position/{position}/p_escape_mean", "fixed3", f"mean {position} cell escape fraction")
        class_cells = [c for c in sweep_cells if c["position_class"] == position]
        class_floors = [c["final"]["jeffreys_floor"] for c in class_cells]
        m.add_derived(f"Wlh{token}FloorMedian", statistics.median(class_floors), "fixed4", f"median Jeffreys floor of the {position} cells", f"median over {position} sweep cells of final.jeffreys_floor", design_inputs)
        m.add_derived(f"Wlh{token}FloorMax", max(class_floors), "fixed4", f"largest Jeffreys floor of the {position} cells", f"max over {position} sweep cells of final.jeffreys_floor", design_inputs)
        m.add_derived(f"Wlh{token}Reflections", reflections_by_class[position], "int_comma", f"reflections in the {position} cells (final N-step launches)", f"sum over {position} sweep cells of final.reflected", design_inputs)
        m.add_derived(f"Wlh{token}CellsWithReflections", cells_with_reflections_by_class[position], "int", f"{position} cells with at least one reflection", f"count over {position} sweep cells of final.reflected > 0", design_inputs)
    if position_summary["interior"]["saturated_at_one"] != position_summary["interior"]["cell_count"] or position_summary["interior"]["p_wall_min"] != 1.0 or position_summary["interior"]["topped_up_count"] != 0:
        raise ValueError("the interior saturation finding does not hold in the evidence")
    m.add_derived("WlhInteriorAllSaturated", position_summary["interior"]["saturated_at_one"] == position_summary["interior"]["cell_count"], "bool", "every interior cell of every sweep design lost every launch to the wall", "per_cell_by_position.interior.saturated_at_one == per_cell_by_position.interior.cell_count", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/per_cell_by_position/interior"}])
    m.add_derived("WlhInteriorDesignsAllSaturated", sum(1 for d in sweep_designs if all(c["final"]["wall_hit"] == c["final"]["trials"] for c in d["cells"] if c["position_class"] == "interior")), "int", "sweep designs whose every interior cell is saturated at one", "count over sweep designs of all(interior cells final.wall_hit == final.trials)", design_inputs)
    m.add_derived("WlhAnodeSaturatedBelowOne", len(anode_saturated_below_one), "int", "anode-side cells saturated after stage 1 below a full count", "count over anode-side sweep cells of saturated_after_stage1 and final.wall_hit < final.trials", design_inputs)
    m.add_derived("WlhAnodeSaturatedBelowOneMin", min(anode_saturated_below_one), "int", "fewest wall hits of an anode-side cell saturated below a full count", "min over those cells of final.wall_hit", design_inputs)
    m.add_derived("WlhAnodeSaturatedBelowOneMax", max(anode_saturated_below_one), "int", "most wall hits of an anode-side cell saturated below a full count", "max over those cells of final.wall_hit", design_inputs)
    m.add_derived("WlhDivergentExitPMin", min(divergent_exit_cells), "fixed3", "smallest exit-side wall-access fraction over the divergent-exit designs", "min over divergent-exit sweep designs of the exit-side cell final.p_wall.probability", design_inputs)
    m.add_derived("WlhDivergentExitPMedian", statistics.median(divergent_exit_cells), "fixed3", "median exit-side wall-access fraction over the divergent-exit designs", "median over divergent-exit sweep designs of the exit-side cell final.p_wall.probability", design_inputs)
    m.add_derived("WlhDivergentExitPMax", max(divergent_exit_cells), "fixed3", "largest exit-side wall-access fraction over the divergent-exit designs", "max over divergent-exit sweep designs of the exit-side cell final.p_wall.probability", design_inputs)
    m.add_derived("WlhDivergentExitAtOne", sum(1 for p in divergent_exit_cells if p == 1.0), "int", "divergent-exit designs whose exit-side cell lost every launch", "count over divergent-exit sweep designs of exit-side final.p_wall.probability == 1", design_inputs)
    m.add_derived("WlhStraightExitPMin", min(straight_exit_cells), "fixed3", "smallest exit-side wall-access fraction over the straight designs", "min over straight sweep designs of the exit-side cell final.p_wall.probability", design_inputs)
    m.add_derived("WlhStraightExitPMax", max(straight_exit_cells), "fixed3", "largest exit-side wall-access fraction over the straight designs", "max over straight sweep designs of the exit-side cell final.p_wall.probability", design_inputs)
    m.add_derived("WlhStraightExitAtOne", sum(1 for p in straight_exit_cells if p == 1.0), "int", "straight designs whose exit-side cell lost every launch", "count over straight sweep designs of exit-side final.p_wall.probability == 1", design_inputs)
    for direction, token in ((+1, "Plus"), (-1, "Minus")):
        pooled_dir = exit_direction[direction]["wall_hit"] / exit_direction[direction]["trials"]
        m.add_derived(f"WlhExitDir{token}Pooled", pooled_dir, "fixed3", f"pooled exit-side wall-access fraction of the D{direction:+d} launches over the divergent-exit designs", f"sum of per_stratum_final wall_hit / trials over exit-side strata with parallel_direction {direction:+d} of divergent-exit sweep designs", design_inputs)
        m.add_derived(f"WlhExitDir{token}Min", min(exit_direction_per_design[direction]), "fixed3", f"smallest per-design exit-side wall-access fraction of the D{direction:+d} launches", f"min over divergent-exit sweep designs of the exit-side D{direction:+d} fraction", design_inputs)
        m.add_derived(f"WlhExitDir{token}Median", statistics.median(exit_direction_per_design[direction]), "fixed3", f"median per-design exit-side wall-access fraction of the D{direction:+d} launches", f"median over divergent-exit sweep designs of the exit-side D{direction:+d} fraction", design_inputs)
        m.add_derived(f"WlhExitDir{token}Max", max(exit_direction_per_design[direction]), "fixed3", f"largest per-design exit-side wall-access fraction of the D{direction:+d} launches", f"max over divergent-exit sweep designs of the exit-side D{direction:+d} fraction", design_inputs)
    m.add_derived("WlhExitOneSidedThreshold", ONE_SIDED_SPLIT, "g", "direction-split threshold for a one-sided exit-side cell", "constant of the generator", design_inputs)
    m.add_derived("WlhExitOneSidedDesigns", sum(1 for s in exit_direction_split if s >= ONE_SIDED_SPLIT), "int", "divergent-exit designs whose exit-side wall-access fractions of the two launch directions differ by at least the threshold", f"count over divergent-exit sweep designs of |D+1 fraction - D-1 fraction| >= {ONE_SIDED_SPLIT:g}", design_inputs)
    m.add_derived("WlhExitDirectionSplitMedian", statistics.median(exit_direction_split), "fixed3", "median absolute difference between the two launch directions' exit-side fractions", "median over divergent-exit sweep designs of |D+1 fraction - D-1 fraction|", design_inputs)
    m.add_derived("WlhExitWallSideLastPolarity", exit_wall_side_matches_last_polarity, "int", "divergent-exit designs whose wall-reaching launch direction equals the last stage's polarity", "count over divergent-exit sweep designs of sign(D+1 fraction - D-1 fraction) == first_polarity * (-1)^(stage_count - 1)", design_inputs)
    m.add_derived("WlhExitWallSideNotLastPolarity", divergent_designs - exit_wall_side_matches_last_polarity, "int", "divergent-exit designs whose wall-reaching launch direction differs from the last stage's polarity", "divergent-exit designs minus WlhExitWallSideLastPolarity", design_inputs)
    m.add("WlhCellsReady", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/cells_surrogate_ready", "int", "cells surrogate-ready (Jeffreys floor at or below the threshold)")
    m.add("WlhFractionReady", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/fraction_cells_surrogate_ready", "pct1", "share of cells surrogate-ready")
    m.add("WlhSweepCellsReady", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/sweep_cells_surrogate_ready", "int", "sweep cells surrogate-ready")
    m.add_derived("WlhSaturatedReady", sum(1 for c in saturated_cells if c["final"]["surrogate_ready"]), "int", "saturated cells that are surrogate-ready", "count over cells of not topped_up and final.surrogate_ready", design_inputs)
    m.add_derived("WlhToppedReady", sum(1 for c in topped_cells if c["final"]["surrogate_ready"]), "int", "topped-up cells that are surrogate-ready", "count over cells of topped_up and final.surrogate_ready", design_inputs)
    m.add("WlhFloorMedian", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/jeffreys_floor_median", "fixed4", "median Jeffreys floor over every cell")
    m.add("WlhFloorMax", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/jeffreys_floor_max", "fixed4", "largest Jeffreys floor over every cell")
    m.add_derived("WlhFloorHalfAtFinal", jeffreys_floor(final_per_topped // 2, final_per_topped), "fixed4", "Jeffreys floor of a topped-up cell at one half", "sqrt(p(1-p)/n) with p = (n/2 + 1/2)/(n + 1), n = 512", [{"artifact": "artifacts/protocol.json", "pointer": "/allocation/stage2_launches_per_cell"}])
    m.add_derived("WlhFloorFullAtStageOne", jeffreys_floor(stage1_per_cell, stage1_per_cell), "fixed4", "Jeffreys floor of a saturated cell at a full count", "sqrt(p(1-p)/n) with p = (n + 1/2)/(n + 1), n = 128", [{"artifact": "artifacts/protocol.json", "pointer": "/allocation/stage1_launches_per_cell"}])
    m.add_derived("WlhFinalNCounts", [int(k) for k in headline["final_n_per_cell_counts"]], "list_int", "final launch counts per cell present", "sorted keys of headline.final_n_per_cell_counts", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/final_n_per_cell_counts"}])

    # ---- reflections, escapes, control ----
    m.add("WlhReflectionsFinal", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/total_reflections_final_n", "int_comma", "reflections over the final N-step launches")
    m.add("WlhReflectionShareFinal", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/reflection_fraction_final_n", "pct1", "share of final launches that reflected")
    m.add_derived("WlhReflectionsControl", reflections_control, "int_comma", "reflections in the 2N control cases", "sum over control cases of termination_counts.reflected", design_inputs)
    m.add_derived("WlhDesignsWithReflections", len(headline["designs_with_reflections"]), "int", "designs with at least one reflection", "len(headline.designs_with_reflections)", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/designs_with_reflections"}])
    m.add_derived("WlhSweepDesignsWithReflections", sum(1 for k in headline["designs_with_reflections"] if k != p2_key), "int", "sweep designs with at least one reflection", "count of headline.designs_with_reflections excluding the P2 row", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/headline/designs_with_reflections"}])
    m.add_derived("WlhCellsWithReflections", sum(cells_with_reflections_by_class.values()) + sum(1 for c in p2_cells if c["final"]["reflected"] > 0), "int", "cells with at least one reflection", "count over cells of final.reflected > 0", design_inputs)
    m.add_derived("WlhPTwoReflections", p2_reflections, "int", "reflections in the P2 row", "sum over P2 cells of final.reflected", design_inputs)
    m.add_derived("WlhWallHitsFinal", termination_final["wall_hit"], "int_comma", "wall hits over the final N-step launches", "sum over N-step cases of termination_counts.wall_hit", design_inputs)
    m.add_derived("WlhWallShareFinal", termination_final["wall_hit"] / (stage1_total + stage2_total), "pct1", "share of final launches that reached the wall", "WlhWallHitsFinal / WlhFinalLaunches", design_inputs)
    m.add_derived("WlhEscapesFinal", termination_final["domain_escape"], "int_comma", "domain escapes over the final N-step launches", "sum over N-step cases of termination_counts.domain_escape", design_inputs)
    m.add_derived("WlhEscapeAnode", subclasses_final["upstream_anode_plane"], "int_comma", "escapes through the anode plane (final N-step launches)", "sum over N-step cases of domain_escape_subclasses.upstream_anode_plane", design_inputs)
    m.add_derived("WlhEscapeExit", subclasses_final["exit_plane"], "int_comma", "escapes through the exit plane (final N-step launches)", "sum over N-step cases of domain_escape_subclasses.exit_plane", design_inputs)
    m.add_derived("WlhEscapeDivergent", subclasses_final["divergent_section_radial"], "int_comma", "radial escapes into a divergent exit section (final N-step launches)", "sum over N-step cases of domain_escape_subclasses.divergent_section_radial", design_inputs)
    m.add_derived("WlhEscapeUnclassified", subclasses_final["unclassified"], "int", "unclassified escapes (final N-step launches)", "sum over N-step cases of domain_escape_subclasses.unclassified", design_inputs)
    m.add("WlhControlN", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/n_control", "int_comma", "pooled control launches")
    m.add("WlhControlWallN", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/wall_N", "int_comma", "wall hits of the control launches at the N time step")
    m.add("WlhControlWallTwoN", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/wall_2N", "int_comma", "wall hits of the control launches at the 2N time step")
    m.add("WlhControlDiscordant", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/discordant", "int", "control orbits whose termination differs between N and 2N")
    m.add("WlhControlDiscordanceRate", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/discordance_rate", "pct3", "discordance rate of the control")
    m.add("WlhControlBias", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/estimated_bias_2N_minus_N", "sci1_signed", "estimated N -> 2N bias of the wall-access fraction")
    m.add("WlhControlBiasSe", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/estimated_bias_standard_error", "sci1", "standard error of the estimated bias")
    m.add("WlhControlPassed", "artifacts/geometry-wall-loss-dataset-v2.json", "/control_gate/passed", "bool", "pooled control gate passed")
    m.add_derived("WlhControlDeltaAbs", abs(pooled_delta), "sci1", "absolute pooled paired change", "|control_gate.estimated_bias_2N_minus_N|", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/control_gate/estimated_bias_2N_minus_N"}])
    m.add_derived("WlhDesignQuantumMax", max(r["quantum"] for r in control_records), "fixed4", "largest per-design control quantum (one discordant orbit)", "max over designs of control.quantum", design_inputs)
    m.add_derived("WlhDesignControlNMin", min(r["n_control"] for r in control_records), "int", "smallest per-design control size", "min over designs of control.n_control", design_inputs)
    m.add_derived("WlhDesignControlNMax", max(r["n_control"] for r in control_records), "int", "largest per-design control size", "max over designs of control.n_control", design_inputs)
    m.add_derived("WlhDesignsWithDiscordance", sum(1 for r in control_records if r["discordant"] > 0), "int", "designs with a discordant control orbit", "count over designs of control.discordant > 0", design_inputs)
    # ---- design pooled values and v1 comparison ----
    m.add("WlhPooledAreaMin", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/design_pooled_wall_area_min", "fixed3", "smallest wall-area-weighted design value over the sweep designs")
    m.add("WlhPooledAreaMedian", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/design_pooled_wall_area_median", "fixed3", "median wall-area-weighted design value")
    m.add("WlhPooledAreaMax", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/design_pooled_wall_area_max", "fixed3", "largest wall-area-weighted design value")
    m.add_derived("WlhPooledLaunchMin", min(v2_launch_pooled), "fixed3", "smallest launch-weighted design value over the sweep designs", "min over sweep designs of pooled.launches.probability", design_inputs)
    m.add_derived("WlhPooledLaunchMedian", statistics.median(v2_launch_pooled), "fixed3", "median launch-weighted design value", "median over sweep designs of pooled.launches.probability", design_inputs)
    m.add_derived("WlhPooledLaunchMax", max(v2_launch_pooled), "fixed3", "largest launch-weighted design value", "max over sweep designs of pooled.launches.probability", design_inputs)
    m.add_derived("WlhVOnePooledMin", min(v1_pooled), "fixed3", "smallest v1 pooled value over the same designs", "min over sweep designs of v1_comparison.v1_probability", design_inputs)
    m.add_derived("WlhVOnePooledMedian", statistics.median(v1_pooled), "fixed3", "median v1 pooled value", "median over sweep designs of v1_comparison.v1_probability", design_inputs)
    m.add_derived("WlhVOnePooledMax", max(v1_pooled), "fixed3", "largest v1 pooled value", "max over sweep designs of v1_comparison.v1_probability", design_inputs)
    m.add("WlhVOneTrials", "artifacts/geometry-wall-loss-dataset-v2.json", "/designs/1/v1_comparison/v1_trials", "int", "launches behind every v1 pooled value")
    m.add_derived("WlhVOneCells", len(designs[1]["v1_comparison"]["v1_cells_z_m"]), "int", "fixed-fraction launch cells of v1", "len(v1_comparison.v1_cells_z_m)", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs/1/v1_comparison/v1_cells_z_m"}])
    m.add("WlhVOneDatasetSha", "artifacts/v1-comparison.json", "/declaration/dataset_file_sha256", "sha_short", "v1 dataset hash prefix bound by the comparison")
    m.add("WlhVOnePreregCommit", "artifacts/v1-comparison.json", "/declaration/preregistration_commit", "sha_short", "v1 preregistration commit prefix")
    m.add("WlhComparisonDesigns", "artifacts/v1-comparison.json", "/design_count", "int", "designs in the v1 comparison")
    for weight, token in (("launches", "Launch"), ("wall_area", "Area")):
        m.add(f"WlhSpearman{token}", "artifacts/v1-comparison.json", f"/spearman_rank_correlation/{weight}", "signed2", f"Spearman rank correlation of the v1 pooled value with the v2 {weight}-weighted value")
        m.add(f"WlhMeanDiff{token}", "artifacts/v1-comparison.json", f"/mean_difference_v2_minus_v1/{weight}", "signed3", f"mean v2 minus v1 difference ({weight}-weighted)")
        m.add(f"WlhMeanAbsDiff{token}", "artifacts/v1-comparison.json", f"/mean_absolute_difference/{weight}", "fixed3", f"mean absolute v2 minus v1 difference ({weight}-weighted)")
        m.add(f"WlhOverlap{token}", "artifacts/v1-comparison.json", f"/interval_overlap_fraction/{weight}", "pct0", f"share of designs whose v1 and v2 {weight}-weighted intervals overlap")
    m.add("WlhDesignPooledStatement", "artifacts/protocol.json", "/claim_boundary/design_pooled_values", "text", "design pooled values statement of the claim boundary")

    # ---- P2 row ----
    p2_by_kind = {c["kind"]: c for c in p2["cells"]}
    m.add("WlhPTwoDesignId", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/p2_row/design_key", "ident", "P2 design key")
    for kind, token in (("anode_partial", "Anode"), ("exit_partial", "Exit")):
        cell = p2_by_kind[kind]
        index = p2["cells"].index(cell)
        m.add(f"WlhPTwo{token}P", "artifacts/geometry-wall-loss-dataset-v2.json", f"/designs/0/cells/{index}/final/p_wall/probability", "fixed3", f"P2 row {kind} cell wall-access fraction")
        m.add(f"WlhPTwo{token}Lo", "artifacts/geometry-wall-loss-dataset-v2.json", f"/designs/0/cells/{index}/final/p_wall/lower", "fixed3", f"P2 row {kind} cell Wilson lower bound")
        m.add(f"WlhPTwo{token}Hi", "artifacts/geometry-wall-loss-dataset-v2.json", f"/designs/0/cells/{index}/final/p_wall/upper", "fixed3", f"P2 row {kind} cell Wilson upper bound")
        m.add(f"WlhPTwo{token}N", "artifacts/geometry-wall-loss-dataset-v2.json", f"/designs/0/cells/{index}/final/trials", "int", f"P2 row {kind} cell final launches")
        m.add(f"WlhPTwo{token}Reflections", "artifacts/geometry-wall-loss-dataset-v2.json", f"/designs/0/cells/{index}/final/reflected", "int", f"P2 row {kind} cell reflections")
        m.add(f"WlhPTwo{token}LengthMm", "artifacts/geometry-wall-loss-dataset-v2.json", f"/designs/0/cells/{index}/length_m", "mm2", f"P2 row {kind} cell length (mm)")
    m.add_derived("WlhPTwoExitLengthUm", p2_by_kind["exit_partial"]["length_m"], "um0", "P2 row exit-side cell length (um)", "cells[exit_partial].length_m in micrometres", [{"artifact": "artifacts/geometry-wall-loss-dataset-v2.json", "pointer": "/designs/0/cells"}])
    m.add_derived("WlhPTwoInteriorCells", sum(1 for c in p2["cells"] if c["kind"] == "interior"), "int", "P2 row interior cells", "count of P2 cells with kind interior", design_inputs)
    m.add_derived("WlhPTwoInteriorPMin", min(c["final"]["p_wall"]["probability"] for c in p2["cells"] if c["kind"] == "interior"), "fixed3", "smallest P2 interior cell wall-access fraction", "min over P2 interior cells of final.p_wall.probability", design_inputs)
    m.add("WlhPTwoPooledArea", "artifacts/geometry-wall-loss-dataset-v2.json", "/headline/p2_row/pooled_wall_area", "fixed3", "P2 row wall-area-weighted design value")
    m.add("WlhPTwoNotReplication", "artifacts/campaign-result.json", "/limitations/p2_row_is_not_v4_replication", "bool", "the P2 row is not a v4 replication")
    m.add("WlhVFourP", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/probability", "fixed3", "wall-loss probability of the v4 reference row")
    m.add("WlhVFourLo", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/confidence_interval_95/0", "fixed3", "Wilson lower bound of the v4 reference row")
    m.add("WlhVFourHi", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/confidence_interval_95/1", "fixed3", "Wilson upper bound of the v4 reference row")
    m.add("WlhVFourTrials", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/trial_count", "int", "trials of the v4 reference row")
    m.add("WlhVFourEvidenceClass", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/evidence_class", "ident", "evidence class of the v4 reference row")
    m.add("WlhVFourFieldQualification", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/field_qualification", "ident", "field qualification of the v4 reference row")
    m.add("WlhVFourInScreeningSet", "artifacts/coupling-consumer-record.json", "/v4_reference/design_in_screening_set", "bool", "the v4 design is in the screening set")
    m.add("WlhVFourConsumed", "artifacts/coupling-consumer-record.json", "/v4_reference/passed", "bool", "the v4 reference export was consumed as a verified reference row")
    m.add("WlhVFourResultCommit", "artifacts/coupling-consumer-record.json", "/v4_reference/v4_result_commit", "sha_short", "result commit prefix of the v4 campaign")
    m.add("WlhVFourHeadline", "artifacts/protocol.json", "/prior_campaign_disclosure/v4/headline", "text", "v4 headline as disclosed by the frozen protocol")
    m.add("WlhVOneHeadline", "artifacts/protocol.json", "/prior_campaign_disclosure/v1_screening/headline", "text", "v1 headline as disclosed by the frozen protocol")
    # ---- coupling consumer ----
    m.add("WlhConsumerId", "artifacts/coupling-consumer-record.json", "/consumer_id", "ident", "consumer identifier")
    m.add("WlhHandoffSchema", "artifacts/orbit-mc-contract.json", "/observed/handoff_schema_version", "ident", "coupling handoff schema version consumed")
    m.add_derived("WlhConsumedCases", len(consumed_cases), "int_comma", "case handoffs consumed", "len(coupling-consumer-record.screening_cases_consumed)", [{"artifact": "artifacts/coupling-consumer-record.json", "pointer": "/screening_cases_consumed"}])
    m.add_derived("WlhConsumedVerified", sum(1 for c in consumed_cases.values() if c["consumption_status"] == "consumed_verified_handoff"), "int_comma", "case handoffs consumed as verified handoffs", "count(screening_cases_consumed[*].consumption_status == consumed_verified_handoff)", [{"artifact": "artifacts/coupling-consumer-record.json", "pointer": "/screening_cases_consumed"}])
    m.add_derived("WlhConsumerChecks", len(reference["consumed"]["checks"]), "int", "checks the consumer applies to every handoff", "len(v4_reference.consumed.checks)", [{"artifact": "artifacts/coupling-consumer-record.json", "pointer": "/v4_reference/consumed/checks"}])
    m.add("WlhCatalogueConsumedCells", "artifacts/coupling-consumer-record.json", "/catalogue_consumed/cells", "int", "catalogue cells consumed")
    m.add("WlhHandoffIntegrationStatus", "artifacts/handoffs/l1a-gs-v2-000-48d2ccedd5--cell-01--stage1-N.json", "/integration_status", "ident", "integration status string carried by every handoff")

    # ---- disclosures ----
    m.add_derived("WlhDisclosedFileCount", disclosed_file_count, "int_comma", "bundle file count named by the disclosure (equals the verified file count)", "POSTHOC_FINALIZATION.md: 'this bundle has N files'", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlhDescriptorCap", disclosed_cap, "int_comma", "Windows C-runtime low-level descriptor cap named by the disclosure", "POSTHOC_FINALIZATION.md: 'allows N low-level descriptors'", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlhPinCap", disclosed_pin_cap, "int_comma", "descriptor pin cap added to the runtime (MAX_PINNED_DESCRIPTORS)", "lifecycle.py: MAX_PINNED_DESCRIPTORS (equals the disclosed cap)", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlhDisclosedArtifactCount", disclosed_validate_count, "int_comma", "artifact count returned by validate after the recovery (equals the manifest artifact count)", "POSTHOC_FINALIZATION.md: validate returned accepted_result with N artifacts", [{"artifact": "manifest.json", "pointer": "/artifact_count"}])
    m.add_derived("WlhManifestPublishedPosthoc", True, "bool", "the results manifest was published by the recovery function after the locked attempt failed at publication", "POSTHOC_FINALIZATION.md binds the manifest and terminal hashes of the committed bundle", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhEvidenceDurableBeforeRecovery", True, "bool", "terminal record, transitions and every sidecar-attested artifact were durable before the recovery", "manifest.terminal_byte_sha256 equals sha256(terminal.json), the transition log ends at terminal, every artifact carries a manifest-bound sidecar and the disclosure states the order of events", [{"artifact": "manifest.json", "pointer": "/terminal_byte_sha256"}, {"artifact": "transitions/0009-terminal.json", "pointer": "/transition"}])
    m.add_derived("WlhOrbitsRerun", 0, "int", "orbits re-integrated after the failed publication", "POSTHOC_FINALIZATION.md: 'No orbit was re-integrated and no experiment code was changed'; the results commit carries only results/ and the disclosure commit changes only Markdown under the experiment", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhCodeFilesChangedAfterRecord", 0, "int", "experiment code files changed between the record and the disclosure commit", "git diff --name-only <record> <disclosure> -- <experiment> contains Markdown files only", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhRecoveryRefusals", recovery_refusals, "int", "refusal conditions named by the recovery module (existing manifest, missing lock or terminal record, transition log not ending at terminal, tampered file)", "count of refusal conditions in recovery.py", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlhDefectScanN", KNOWN_DEFECT_SCAN_N, "int", "case sizes scanned for Wilson exactness", "constant of the generator (the frozen protocol's 'first 4000 n')", [{"artifact": "artifacts/protocol.json", "pointer": "/orbit_mc_contract/known_defect_v1_7/statement"}])
    m.add_derived("WlhDefectZeroInexact", known_defect["zero_count_lower_inexact"], "int", "case sizes whose zero-count Wilson lower bound is a positive round-off", "count of n <= 4000 with wilson(0, n).lower > 0 (recomputed)", [{"artifact": "artifacts/protocol.json", "pointer": "/orbit_mc_contract/known_defect_v1_7/statement"}])
    m.add_derived("WlhDefectFullInexact", known_defect["full_count_upper_inexact"], "int", "case sizes whose full-count Wilson upper bound is below one", "count of n <= 4000 with wilson(n, n).upper < 1 (recomputed)", [{"artifact": "artifacts/protocol.json", "pointer": "/orbit_mc_contract/known_defect_v1_7/statement"}])
    m.add_derived("WlhDefectCaseSizesExact", all(known_defect["exact_at_both_ends"].values()), "bool", "the three case sizes are exact at both ends", "wilson(0, n).lower == 0 and wilson(n, n).upper == 1 for n in {128, 16, 64}", [{"artifact": "artifacts/protocol.json", "pointer": "/cases/case_sizes"}])
    m.add_derived("WlhDefectFiveTwelveInexact", known_defect["n512_full_inexact"], "bool", "a 512-launch case is inexact at a full count", "wilson(512, 512).upper < 1 (recomputed)", [{"artifact": "artifacts/protocol.json", "pointer": "/orbit_mc_contract/known_defect_v1_7/v1_note"}])
    m.add("WlhDefectDiscovered", "artifacts/protocol.json", "/orbit_mc_contract/known_defect_v1_7/discovered", "text", "where the defect was discovered, as frozen")
    m.add("WlhLaunchIdRule", "artifacts/protocol.json", "/launches/launch_id_rule", "text", "launch-id grammar rule as frozen")
    m.add_derived("WlhInjectorFlaggedCells", len(injector_flagged), "int", "cells whose midpoint lies inside the injector zone", "count over cells of launch_plane_inside_injector_zone", design_inputs)
    m.add_derived("WlhInjectorFlaggedDesign", injector_flagged[0][0], "ident", "design of the injector-zone cell", "the design key of the flagged cell", design_inputs)
    m.add_derived("WlhInjectorFlaggedCellId", injector_flagged[0][1], "ident", "cell id of the injector-zone cell", "the cell id of the flagged cell", design_inputs)
    m.add_derived("WlhInjectorFlaggedLengthMm", injector_flagged[0][2], "mm2", "length of the injector-zone cell (mm)", "length_m of the flagged cell", design_inputs)
    m.add_derived("WlhShortCells", len(short_cells), "int", "cells shorter than the short-cell threshold", "count over cells of short_cell", design_inputs)
    m.add_derived("WlhShortSweepCells", sum(1 for s in short_cells if s[0] != p2_key), "int", "sweep cells shorter than the short-cell threshold", "count over sweep cells of short_cell", design_inputs)
    m.add_derived("WlhShortSweepExitCells", sum(1 for s in short_cells if s[0] != p2_key and s[2] == "exit_side"), "int", "short sweep cells on the exit side", "count over sweep cells of short_cell and position_class exit_side", design_inputs)
    m.add_derived("WlhShortSweepAnodeCells", sum(1 for s in short_cells if s[0] != p2_key and s[2] == "anode_side"), "int", "short sweep cells on the anode side", "count over sweep cells of short_cell and position_class anode_side", design_inputs)
    m.add_derived("WlhShortPTwoCells", sum(1 for s in short_cells if s[0] == p2_key), "int", "short cells of the P2 row", "count over P2 cells of short_cell", design_inputs)
    m.add_derived("WlhShortCellLengthMinUm", min(s[3] for s in short_cells), "um0", "shortest cell (um)", "min over short cells of length_m", design_inputs)

    # ---- claim boundary flags ----
    m.add("WlhNotAcceptedPhysicalOrbit", "artifacts/campaign-result.json", "/limitations/not_accepted_physical_orbit_evidence", "bool", "not accepted physical-orbit evidence")
    m.add("WlhNotPTwoQualified", "artifacts/campaign-result.json", "/limitations/not_p2_qualified", "bool", "sweep fields not P2-qualified")
    m.add("WlhForbidPerformance", "artifacts/campaign-result.json", "/limitations/forbid_plasma_performance_publication", "bool", "plasma or performance publication forbidden")
    m.add("WlhForbidPic", "artifacts/campaign-result.json", "/limitations/forbid_pic_or_self_consistent_claim", "bool", "PIC or self-consistent claim forbidden")
    m.add("WlhForbidMirror", "artifacts/campaign-result.json", "/limitations/forbid_mirror_formula_publication", "bool", "mirror-formula publication forbidden")
    m.add("WlhHardwareValidation", "artifacts/campaign-result.json", "/limitations/hardware_or_experimental_validation", "bool", "hardware or experimental validation claimed")
    m.add("WlhUsableAs", "artifacts/campaign-result.json", "/limitations/usable_as", "list_clauses", "permitted uses of the dataset")
    m.add("WlhShakedownNotEvidence", "artifacts/campaign-result.json", "/limitations/shakedown_outcomes_are_not_evidence", "bool", "shakedown outcomes are not evidence")
    m.add("WlhTwoStageEstimatorStatement", "artifacts/campaign-result.json", "/limitations/two_stage_estimator", "text", "two-stage estimator statement of the claim boundary")

    # ================================================================== tables ====
    tex_lines = [
        "% Generated by paper/scripts/generate_wall_loss_geometry_screening_v2_evidence.py; do not hand edit.",
        f"% Evidence: {EXPERIMENT.as_posix()} at commit {RESULTS_COMMIT_SHA} (results manifest SHA-256 {bundle.manifest_sha256}).",
        f"% Every macro value traces to an artifact path and JSON pointer recorded in {EVIDENCE_PATH.as_posix()}.",
    ]
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    # (a) dataset and allocation summary
    dataset_rows = [
        f"designs screened (sweep-v2 designs + P2 launch-design row) & {len(designs)} ({len(sweep_case_ids)} + 1)\\\\",
        f"designs excluded before integration & {campaign['excluded_design_count']}\\\\",
        f"catalogue cells launched (anode-side / interior / exit-side / P2) & {len(all_cells)} ({position_summary['anode_side']['cell_count']} / {position_summary['interior']['cell_count']} / {position_summary['exit_side']['cell_count']} / {len(p2_cells)})\\\\",
        f"orbit cases (stage 1, stage-2 blocks, 2N control) & {_comma(campaign['case_count'])}\\\\",
        f"electron orbits integrated (stage 1 / stage 2 / control) & {_comma(campaign['orbit_count'])} ({_comma(stage1_total)} / {_comma(stage2_total)} / {_comma(control_total)})\\\\",
        f"cells topped up to {final_per_topped} (anode-side / exit-side / P2) & {len(topped_cells)} ({position_summary['anode_side']['topped_up_count']} / {position_summary['exit_side']['topped_up_count']} / {sum(c['topped_up'] for c in p2_cells)})\\\\",
        f"cells saturated at {stage1_per_cell} (stage-1 Wilson width $\\le$ {threshold:g}) & {len(saturated_cells)} ({100 * len(saturated_cells) / len(all_cells):.0f}\\%)\\\\",
        f"allocation-rule and control-selection replay passed & {sum(1 for d in designs if d['allocation_replay']['passed'])} / {len(designs)} designs\\\\",
        f"validator calls passed / failed & {_comma(campaign['validators']['passed'])} / {campaign['validators']['failed']}\\\\",
        f"cases sealed and replayed by the exact authority & {_comma(gates['sealed_case_count'])} / {_comma(campaign['case_count'])}\\\\",
        f"designs sealed (paired control flag true) & {headline['sealed_design_count']} / {len(designs)}\\\\",
        f"timeouts / numerical failures over every case & {sum(termination_all[t] for t in TIMEOUTS)} / {sum(termination_all[t] for t in NUMERICAL_FAILURES)}\\\\",
        f"largest relative kinetic-energy drift (gate {format_value('sci1', protocol['gates']['maximum_relative_energy_error'])}) & {max(energy_errors):g}\\\\",
        f"largest interpolation rms field error (gate {100 * protocol['field_source']['adapter_gates']['maximum_b_relative_rms']:.0f}\\%) & {100 * max(interpolation):.2f}\\%\\\\",
        f"largest cross-resolution rms field error ({len(cross_resolution)} of {len(designs)} designs; gate {100 * protocol['field_source']['adapter_gates']['maximum_cross_resolution_b_relative_rms']:.0f}\\%) & {100 * max(cross_resolution):.2f}\\%\\\\",
        f"cells surrogate-ready (Jeffreys floor $\\le$ {readiness_floor:g}) & {len(ready_cells)} / {len(all_cells)} ({100 * len(ready_cells) / len(all_cells):.0f}\\%)\\\\",
        f"Jeffreys floor over every cell: median / maximum & {statistics.median(floors):.4f} / {max(floors):.4f}\\\\",
        f"execution wall time: lock to terminal / case pool & {execution_wall_s / 60:.1f} / {campaign['execution_mode']['cases_wall_s'] / 60:.1f}~min\\\\",
        f"bundle files verified byte for byte / accepted through an end-of-line tolerance & {_comma(len(bundle.hashes))} / 0\\\\",
    ]
    tex_lines += _table(
        "WlhDatasetTable",
        "Dataset and allocation summary of the catalogue-cell wall-access screening as sealed in "
        "\\texttt{campaign-result.json}, \\texttt{gates.json}, \\texttt{allocation-decisions.json} and "
        "\\texttt{geometry-wall-loss-dataset-v2.json}. Every count is a screening quantity within the collisionless "
        "test-particle model on linear-vacuum screening fields (the P2 row excepted, which is a launch-design row on "
        "the qualified field); none is a plasma or performance quantity.",
        "tab:wall-loss-geometry-screening-v2-dataset", f"{_p(10.6)}r",
        "quantity & value\\\\", dataset_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    # (b) per-cell-class distributions
    class_rows: list[str] = []
    for position in POSITION_CLASSES:
        s = position_summary[position]
        rows = [c for c in sweep_cells if c["position_class"] == position]
        fl = [c["final"]["jeffreys_floor"] for c in rows]
        class_rows.append(
            f"{position.replace('_', '-')} & {s['cell_count']} & {s['p_wall_min']:.3f} & {s['p_wall_q1']:.3f} & {s['p_wall_median']:.3f} & {s['p_wall_q3']:.3f} & {s['p_wall_max']:.3f} & {s['p_wall_mean']:.3f} & "
            f"{s['saturated_at_one']} & {s['saturated_at_zero']} & {s['topped_up_count']} & {s['surrogate_ready_count']} & {statistics.median(fl):.4f} & {max(fl):.4f} & {reflections_by_class[position]}\\\\"
        )
    p2_p = sorted(c["final"]["p_wall"]["probability"] for c in p2_cells)
    p2_fl = [c["final"]["jeffreys_floor"] for c in p2_cells]
    class_rows.append("\\midrule")
    class_rows.append(
        f"P2 row (all kinds) & {len(p2_cells)} & {p2_p[0]:.3f} & {p2_p[len(p2_p) // 4]:.3f} & {statistics.median(p2_p):.3f} & {p2_p[(3 * len(p2_p)) // 4]:.3f} & {p2_p[-1]:.3f} & {statistics.fmean(p2_p):.3f} & "
        f"{sum(c['final']['wall_hit'] == c['final']['trials'] for c in p2_cells)} & {sum(c['final']['wall_hit'] == 0 for c in p2_cells)} & {sum(c['topped_up'] for c in p2_cells)} & {sum(c['final']['surrogate_ready'] for c in p2_cells)} & {statistics.median(p2_fl):.4f} & {max(p2_fl):.4f} & {p2_reflections}\\\\"
    )
    tex_lines += _table(
        "WlhCellClassTable",
        "Per-cell wall-access fraction $P_{\\mathrm{wall}}$ by catalogue position class over the "
        "\\WlhSweepCellCount{} cells of the \\WlhSweepDesignCount{} sweep designs and over the \\WlhPTwoCellCount{} cells "
        "of the P2 row: cell count, minimum, quartiles, median, maximum and mean of the per-cell fraction, cells "
        "saturated at one and at zero, cells topped up by the frozen rule, cells surrogate-ready (Jeffreys floor "
        "$\\le$ \\WlhReadinessFloor), median and largest Jeffreys floor, and reflections over the final launches. "
        "The cells are the separatrix-bounded catalogue cells launched at their midpoints; a fraction of one means every "
        "collisionless launch reached the dielectric and is not a loss probability.",
        "tab:wall-loss-geometry-screening-v2-cells", f"{_p(1.75)}rrrrrrrrrrrrrr",
        "class & cells & min & $q_1$ & med. & $q_3$ & max & mean & at $1$ & at $0$ & topped & ready & floor med. & floor max & refl.\\\\",
        class_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{2.4pt}",
    )
    # (c) reflections and the 2N control
    control_rows = [
        f"reflections over the final launches (share) & {_comma(reflections_final)} of {_comma(stage1_total + stage2_total)} ({100 * reflections_final / (stage1_total + stage2_total):.1f}\\%)\\\\",
        f"cells with reflections: anode-side / interior / exit-side / P2 & {cells_with_reflections_by_class['anode_side']} / {cells_with_reflections_by_class['interior']} / {cells_with_reflections_by_class['exit_side']} / {sum(1 for c in p2_cells if c['final']['reflected'] > 0)}\\\\",
        f"designs with reflections (sweep + P2) & {len(headline['designs_with_reflections'])} ({sum(1 for k in headline['designs_with_reflections'] if k != p2_key)} + {1 if p2_key in headline['designs_with_reflections'] else 0})\\\\",
        f"exit-side $P_{{\\mathrm{{wall}}}}$ of the {divergent_designs} divergent-exit designs: min / median / max & {min(divergent_exit_cells):.3f} / {statistics.median(divergent_exit_cells):.3f} / {max(divergent_exit_cells):.3f}\\\\",
        f"\\quad by direction, pooled: $D{{+}}1$ / $D{{-}}1$ & {exit_direction[+1]['wall_hit'] / exit_direction[+1]['trials']:.3f} / {exit_direction[-1]['wall_hit'] / exit_direction[-1]['trials']:.3f}\\\\",
        f"\\quad by direction, per-design range: $D{{+}}1$ / $D{{-}}1$ & {min(exit_direction_per_design[+1]):.3f}--{max(exit_direction_per_design[+1]):.3f} / {min(exit_direction_per_design[-1]):.3f}--{max(exit_direction_per_design[-1]):.3f}\\\\",
        f"\\quad designs whose two directions differ by $\\ge {ONE_SIDED_SPLIT:g}$ (median split) & {sum(1 for s in exit_direction_split if s >= ONE_SIDED_SPLIT)} of {divergent_designs} ({statistics.median(exit_direction_split):.3f})\\\\",
        f"\\quad wall-reaching direction equals the last stage's polarity / differs & {exit_wall_side_matches_last_polarity} / {divergent_designs - exit_wall_side_matches_last_polarity}\\\\",
        f"\\quad divergent-exit designs whose exit-side cell is at one & {sum(1 for p in divergent_exit_cells if p == 1.0)}\\\\",
        f"exit-side $P_{{\\mathrm{{wall}}}}$ of the {straight_designs} straight designs: min / max / at one & {min(straight_exit_cells):.3f} / {max(straight_exit_cells):.3f} / {sum(1 for p in straight_exit_cells if p == 1.0)}\\\\",
        f"escapes over the final launches: anode plane / exit plane / divergent radial / unclassified & {_comma(subclasses_final['upstream_anode_plane'])} / {_comma(subclasses_final['exit_plane'])} / {_comma(subclasses_final['divergent_section_radial'])} / {subclasses_final['unclassified']}\\\\",
        "\\midrule",
        f"control launches re-integrated at 2N (one eighth of every cell) & {_comma(pooled_n)}\\\\",
        f"wall hits of the control launches at N / at 2N & {_comma(pooled_wall_n)} / {_comma(pooled_wall_2n)}\\\\",
        f"discordant orbits (termination differs between N and 2N) & {pooled_discordant} ({100 * pooled_discordant / pooled_n:.3f}\\%)\\\\",
        f"estimated bias $P_{{2N}} - P_N$ $\\pm$ standard error & {format_value('sci1_signed', pooled_delta)} $\\pm$ {format_value('sci1', bias_se)}\\\\",
        f"pooled gate $|P_{{2N}} - P_N| \\le {control_gate['maximum_allowed_change']:g}$ & {'passed' if control_gate['passed'] else 'failed'}\\\\",
        f"designs with the paired control flag true / with a discordant orbit & {headline['control_flag_true_design_count']} of {len(designs)} / {sum(1 for r in control_records if r['discordant'] > 0)}\\\\",
        f"reflections in the control cases & {_comma(reflections_control)}\\\\",
    ]
    tex_lines += _table(
        "WlhControlTable",
        "Reflections, escapes and the N $\\to$ 2N control of the catalogue-cell screening over the final N-step "
        "launches (stage 1 plus stage 2). Reflections are orbits that reversed their parallel velocity before "
        "reaching a boundary; the control re-integrates a frozen-seed subset of every cell's final launches at the "
        "halved time step and pairs each orbit with its N-step partner by launch key (replayed by the evidence "
        "generator from the sealed endpoint tables). The direction rows split the exit-side launches by their "
        "parallel launch direction; they are observations of this launch design, not a design rule.",
        "tab:wall-loss-geometry-screening-v2-control", f"{_p(10.2)}r",
        "quantity & value\\\\", control_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    # (d) v1 vs v2 pooled comparison
    comparison_table_rows = [
        f"designs compared (same fields, different cells and launch distributions) & {comparison['design_count']} & --\\\\",
        f"v1 pooled $P_{{\\mathrm{{wall}}}}$ (accepted-2N, {designs[1]['v1_comparison']['v1_trials']} launches, {len(designs[1]['v1_comparison']['v1_cells_z_m'])} fixed-fraction cells): min / median / max & {min(v1_pooled):.3f} / {statistics.median(v1_pooled):.3f} / {max(v1_pooled):.3f} & --\\\\",
        f"v2 launch-weighted design value: min / median / max & {min(v2_launch_pooled):.3f} / {statistics.median(v2_launch_pooled):.3f} / {max(v2_launch_pooled):.3f} & --\\\\",
        f"v2 wall-area-weighted design value: min / median / max & {min(design_wall_area):.3f} / {statistics.median(design_wall_area):.3f} / {max(design_wall_area):.3f} & --\\\\",
        "\\midrule",
        "statistic & launch-weighted & wall-area-weighted\\\\",
        "\\midrule",
        f"Spearman rank correlation of v2 with v1 & {format_value('signed2', comparison['spearman_rank_correlation']['launches'])} & {format_value('signed2', comparison['spearman_rank_correlation']['wall_area'])}\\\\",
        f"mean difference v2 $-$ v1 & {format_value('signed3', comparison['mean_difference_v2_minus_v1']['launches'])} & {format_value('signed3', comparison['mean_difference_v2_minus_v1']['wall_area'])}\\\\",
        f"mean absolute difference & {comparison['mean_absolute_difference']['launches']:.3f} & {comparison['mean_absolute_difference']['wall_area']:.3f}\\\\",
        f"designs whose intervals overlap & {100 * comparison['interval_overlap_fraction']['launches']:.0f}\\% & {100 * comparison['interval_overlap_fraction']['wall_area']:.0f}\\%\\\\",
    ]
    tex_lines += _table(
        "WlhComparisonTable",
        "Pooled comparison of the catalogue-cell screening with the fixed-fraction screening of "
        "Section~\\ref{sec:wall-loss-geometry-screening} over the same \\WlhComparisonDesigns{} re-solved fields, as sealed in "
        "\\texttt{v1-comparison.json} and recomputed by the evidence generator from the bound v1 dataset. A direct per-cell "
        "comparison is impossible because the cells differ; the design values are declared averages of the per-cell "
        "structure (launch counts, the v1-comparable weighting; catalogue wall areas, the declared design value) and not "
        "estimands. The comparison is reported, never gated.",
        "tab:wall-loss-geometry-screening-v2-comparison", f"{_p(8.4)}{_p(2.3)}{_p(2.3)}",
        "quantity & value & \\\\", comparison_table_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    # (e) disclosures
    disclosure_rows = [
        f"(i) files in the bundle / Windows C-runtime descriptor cap & {_comma(disclosed_file_count)} / {_comma(disclosed_cap)}\\\\",
        f"(i) durable before the recovery: terminal record, transitions, sidecar-attested artifacts (file / transition count) & yes ({_comma(len(bundle.hashes))} / {len(transitions)})\\\\",
        f"(i) manifest published post hoc by the fail-closed recovery; manifest SHA-256 prefix disclosed = committed & yes; \\texttt{{{disclosed_manifest_sha[:12]}}}\\\\",
        f"(i) orbits re-integrated / experiment code files changed after the record & {0} / {0}\\\\",
        f"(i) runtime pin cap added (\\texttt{{MAX\\_PINNED\\_DESCRIPTORS}}) / recovery refusal conditions & {_comma(disclosed_pin_cap)} / {recovery_refusals}\\\\",
        f"(i) \\texttt{{validate}} after the recovery: state / artifacts & \\texttt{{{_ident(bundle.manifest['state'])}}} / {_comma(disclosed_validate_count)}\\\\",
        "\\midrule",
        f"(ii) case sizes of the first {KNOWN_DEFECT_SCAN_N} whose zero-count lower bound is a positive round-off & {known_defect['zero_count_lower_inexact']}\\\\",
        f"(ii) case sizes of the first {KNOWN_DEFECT_SCAN_N} whose full-count upper bound is below one & {known_defect['full_count_upper_inexact']}\\\\",
        f"(ii) case sizes used (block / control of a saturated cell / of a topped-up cell), exact at both ends & {case_sizes['block']} / {case_sizes['control_of_stage1_cell']} / {case_sizes['control_of_topped_up_cell']}, yes\\\\",
        f"(ii) launch-id grammar of the frozen package (cell index in \\texttt{{X}}, Sobol index in \\texttt{{G}}) & constraint, no number\\\\",
        "\\midrule",
        f"(iii) cells whose midpoint lies inside the injector zone (flagged, not moved) & {len(injector_flagged)} (\\texttt{{{_ident(_short(injector_flagged[0][0]))}}}, {1e3 * injector_flagged[0][2]:.2f}~mm)\\\\",
        f"(iii) cells shorter than {1e3 * short_cell_length:.0f}~mm: sweep (exit-side / anode-side) / P2 & {sum(1 for s in short_cells if s[0] != p2_key)} ({sum(1 for s in short_cells if s[0] != p2_key and s[2] == 'exit_side')} / {sum(1 for s in short_cells if s[0] != p2_key and s[2] == 'anode_side')}) / {sum(1 for s in short_cells if s[0] == p2_key)} ({1e6 * min(s[3] for s in short_cells):.0f}~$\\mu$m)\\\\",
    ]
    tex_lines += _table(
        "WlhDisclosureTable",
        "Disclosures admitted with the dataset, each verified by the evidence generator: (i) the post-hoc manifest "
        "publication (\\texttt{POSTHOC\\_FINALIZATION.md}, bound at commit \\texttt{\\WlhDisclosureCommit}; the disclosed "
        "manifest and terminal hashes must equal the committed bundle's), (ii) the two design constraints inherited from "
        "the frozen orbit\\_mc package (Wilson-exact case sizes, recomputed here; the launch-id grammar) and (iii) the "
        "launch-plane flags of the catalogue cells.",
        "tab:wall-loss-geometry-screening-v2-disclosures", f"{_p(10.4)}{_p(4.4)}",
        "disclosure & value\\\\", disclosure_rows, size="\\scriptsize",
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    tex = "\n".join(tex_lines) + "\n"

    reference_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (catalogue_file, catalogue_manifest_file, v1_dataset_file, v1_manifest_file, v4_export_file, sweep_manifest_file)
    }
    disclosure_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (disclosure_file, recovery_file, lifecycle_file, recovery_test_file)
    }
    evidence = {
        "document_type": "paper-wall-loss-geometry-screening-v2-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "p2_row_label": P2_LABEL,
        "recorded_outcome": RECORDED_OUTCOME,
        "campaign_status": CAMPAIGN_STATUS,
        "screening_model": SCREENING_MODEL,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
        "reference_artifacts": {
            "rule": (
                "the cusp-cell catalogue and the cusp topology v3.1 manifest (must hash to the identities the campaign "
                "bound; the dataset's cells equal the catalogue's cells field by field), the screening v1 dataset and "
                "manifest (must hash to the identities the comparison declared; the v1 values of every comparison row "
                "are re-read from it), the wall-loss v4 coupling export (must hash to the identity the consumer bound) "
                "and the sweep-v2 results manifest (must hash to the identity the field pipeline bound), each bound at "
                "its own admitted revision"
            ),
            "files": reference_files,
        },
        "disclosure_sources": {
            "revision": DISCLOSURE_COMMIT_SHA,
            "files": disclosure_files,
            "rule": (
                "POSTHOC_FINALIZATION.md, the runtime recovery module, the lifecycle module carrying the pin cap and the "
                "recovery tests are bound at the disclosure commit (LF-normalised); the disclosure's manifest and "
                "terminal hashes, file, artifact and transition counts, results commit and descriptor arithmetic must "
                "equal the committed bundle, the pin cap in the module must equal the disclosed cap, and the disclosure "
                "commit may change only Markdown under the experiment"
            ),
            "verified": {
                "manifest_sha256_matches_bundle": disclosed_manifest_sha == bundle.manifest_sha256,
                "terminal_sha256_matches_bundle": disclosed_terminal_sha == bundle.manifest["terminal_byte_sha256"],
                "file_count": disclosed_file_count,
                "descriptor_cap": disclosed_cap,
                "pin_cap": disclosed_pin_cap,
                "validate_artifact_count": disclosed_validate_count,
                "results_commit_prefix": disclosed_results_commit,
                "nothing_rerun_stated": disclosed_rerun,
                "experiment_files_changed_by_disclosure_commit": binding["disclosure_commit_experiment_files_changed"],
            },
        },
        "known_defect_recomputation": known_defect,
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
                "Every number in the section is a macro defined here; each macro is bound below to an artifact path, "
                "JSON pointer, formatter and SHA-256, or to a stated derivation over such inputs. Claim-bearing "
                "sentences are exact EvidenceClaim bodies registered in paper/evidence/claims.json; the "
                "numerical-screening gate in paper/evidence/result-gates.json names the typed manifest that admits "
                "the section at its recorded outcome (an accepted screening dataset) without opening any physics "
                "level. Every per-cell number is a collisionless geometric wall-access fraction of the launch "
                "distribution on linear-vacuum screening fields (the P2 row on the qualified field with this "
                "campaign's launch design); none is a loss probability, accepted physical-orbit, plasma or "
                "performance evidence, and the post-hoc manifest publication is disclosed with the dataset."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size",
            "wilson_rule": "every per-cell, per-case and per-stratum Wilson-95 interval is recomputed operation for operation and must equal the sealed value exactly",
            "recomputation_rule": (
                "the frozen allocation rule, the stage pooling, both floors, the readiness flag and the paired N -> 2N "
                "control of every cell are replayed from the sealed per-case summaries and endpoint tables (launch keys "
                "paired orbit by orbit); the design pooled values, the pooled control gate with its standard error, "
                "every headline statistic including the position-class summary, the least/most designs and the P2 row, "
                f"and the v1 comparison are recomputed and must equal the sealed values (counts exactly; floats within a "
                f"relative tolerance of {FLOAT_TOLERANCE:g})"
            ),
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": {
            "WlhDatasetTable": {"rows": len(dataset_rows), "source": "artifacts/campaign-result.json, artifacts/gates.json, artifacts/allocation-decisions.json, artifacts/geometry-wall-loss-dataset-v2.json#/headline, transitions"},
            "WlhCellClassTable": {"rows": len(class_rows), "source": "artifacts/geometry-wall-loss-dataset-v2.json#/headline/per_cell_by_position, #/designs/*/cells"},
            "WlhControlTable": {"rows": len(control_rows), "source": "artifacts/geometry-wall-loss-dataset-v2.json#/control_gate, #/designs/*/cells, #/designs/*/per_stratum_final, #/designs/*/cases"},
            "WlhComparisonTable": {"rows": len(comparison_table_rows), "source": "artifacts/v1-comparison.json, artifacts/geometry-wall-loss-dataset-v2.json#/designs/*/pooled, #/designs/*/v1_comparison"},
            "WlhDisclosureTable": {"rows": len(disclosure_rows), "source": "POSTHOC_FINALIZATION.md and lifecycle.py at the disclosure commit, artifacts/protocol.json#/orbit_mc_contract/known_defect_v1_7 (recomputed), #/designs/*/cells flags"},
        },
        "generator": {
            "path": "paper/scripts/generate_wall_loss_geometry_screening_v2_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_wall_loss_geometry_screening_v2_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_wall_loss_geometry_screening_v2_evidence.py",
        },
        "output": {"path": OUTPUT_PATH.as_posix(), "sha256": sha256_bytes(tex.encode("utf-8"))},
    }
    if len({item["name"] for item in m.items}) != len(m.items):
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
        "reference_inputs": [
            {"path": path, "sha256": meta["sha256"], "bytes": meta["bytes"], "revision": meta["revision"]}
            for path, meta in evidence["reference_artifacts"]["files"].items()
        ],
        "disclosure_inputs": [
            {"path": path, "sha256": meta["sha256"], "bytes": meta["bytes"], "revision": meta["revision"]}
            for path, meta in evidence["disclosure_sources"]["files"].items()
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
        print(f"wall-loss geometry screening v2 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
