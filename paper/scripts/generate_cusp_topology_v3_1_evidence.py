"""Generate hash-bound paper evidence for the cusp topology search v3.1.

Reads the sealed results bundle of ``modern/experiments/cusp_topology_search_v3_1``
(every manifest file verified byte-for-byte; no end-of-line tolerance is needed or
granted), binds it to the committed results revision, re-derives every headline
and per-set estimand from the 281 per-design rows and their design records,
cross-checks the committed results dashboard against the same bundle, verifies the
RECORDED ``assessment_rejection`` bundle of the predecessor campaign
``cusp_topology_search_v3`` as lineage (never as a source of numbers), reproduces
that campaign's post-hoc audit from the sealed characterization-v1 dataset, and
writes:

* ``paper/evidence/cusp-topology-v3-1.json`` — every macro value with the
  artifact path, JSON pointer, formatter and artifact SHA-256 it was read from,
  or the derivation and inputs of a derived macro;
* ``paper/generated/cusp-topology-v3-1.tex`` — ``\\newcommand`` macros and four
  generated tables (each wrapped in ``\\ArtifactClaim``) for the admitted results
  subsection ``paper/sections/cusp-topology-v3-1.tex``;
* ``paper/generated/cusp-topology-v3-1.provenance.json`` — generator/input/output
  hashes in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The study is a magnetic-topology screening of prescribed
vacuum field maps under the HEMP/DCFT literature definition of a wall cusp (axis
null -> separatrix -> dielectric-wall intersection): linear-vacuum L1a
equivalent-current screening fields for three design sets (not P2-qualified) and
the P2-qualified finite-element divergent-exit-stack field for one row.  Cells
and mirror ratios are geometric field descriptors; no number below is a plasma,
mirror-probability, wall-loss or performance claim.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable

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

EXPERIMENT = Path("modern/experiments/cusp_topology_search_v3_1")
RESULTS = EXPERIMENT / "results"
LINEAGE_EXPERIMENT = Path("modern/experiments/cusp_topology_search_v3")
LINEAGE_RESULTS = LINEAGE_EXPERIMENT / "results"
LINEAGE_AUDIT = LINEAGE_EXPERIMENT / "POSTHOC_AUDIT.md"
LINEAGE_AUDIT_SCRIPT = LINEAGE_EXPERIMENT / "audit_held_out.py"
V1_DATASET = Path("modern/experiments/cft_topology_characterization_v1/results/dataset.json")
V2_DATASET = Path("modern/experiments/four_cell_topology_search_v2/results/dataset.json")
SWEEP_MANIFEST = Path("modern/experiments/l1a_geometry_sweep_v2/results/manifest.json")
LITERATURE_REVIEW = Path("modern/docs/literature/reduced-models-cusp-topology-blockers.md")
EVIDENCE_PATH = Path("paper/evidence/cusp-topology-v3-1.json")
OUTPUT_PATH = Path("paper/generated/cusp-topology-v3-1.tex")
SIDECAR_PATH = Path("paper/generated/cusp-topology-v3-1.provenance.json")
SECTION_PATH = Path("paper/sections/cusp-topology-v3-1.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_cusp_topology_search_v3_dashboard.py")
DASHBOARD_TEMPLATE = Path("modern/visualization/cusp-topology-search-v3.template.html")
DASHBOARD_HTML = Path("modern/visualization/cusp-topology-search-v3.html")

# Revisions.  The v3.1 results tree first exists at the record commit; the dashboard
# was generated from that bundle two commits later.  The predecessor campaign v3 is
# bound as lineage only: its preregistration, its recorded rejection and the read-only
# post-hoc audit that explains the rejection.  The literature review commit supplies
# the definition sources; the sealed v1/v2 datasets and the sweep manifest are the
# held-out references the campaign compared against (bound at their own admitted
# revisions and required to equal the sealed-source hashes recorded in the bundle).
RESULTS_COMMIT_SHA = "cec47f12f5909c5886424bf5d46ac20ce06f1ac5"
PREREGISTRATION_COMMIT_SHA = "1600cfd3b102980eeba4b070930667d232a1105c"
DASHBOARD_COMMIT_SHA = "9abbd5371b816208d687f1adbf54f31884c8b27f"
LINEAGE_RESULTS_COMMIT_SHA = "8cbcdbe6ede6c55156f300f82d9c85133f06c0dd"
LINEAGE_PREREGISTRATION_COMMIT_SHA = "691599340355818ff64d3834d45110768a751589"
LINEAGE_AUDIT_COMMIT_SHA = "9fa6359a2ba87d14635d80147af4857482afc977"
LITERATURE_COMMIT_SHA = "66879e00834b09e2c7f942358b3ae4a51658cb6b"
V1_RESULTS_COMMIT_SHA = "3ce6c546194e1d3e943d0b3d0951d03e15e354d9"
V2_RESULTS_COMMIT_SHA = "7120e8edcb74c02c1df968c730d1f93b3758b4e1"
SWEEP_RESULTS_COMMIT_SHA = "f30cb42ec4a8633bf634a3d32ffa5b11f66be97a"

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "CUSP-TOPOLOGY-V3-1-20260903-281-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/cusp-topology-v3-1.json")
GATE_ID = "GATE-CUSP-TOPOLOGY-V3-1"
GATE_KIND = "numerical-screening"
RECORDED_OUTCOME = "accepted-topology-screening"
ARTIFACT_ID = "TAB-CUSP-TOPOLOGY-V3-1"
ARTIFACT_CLAIM_ID = "CLM-063"
PROSE_CLAIM_IDS = ("CLM-061", "CLM-062", "CLM-064", "CLM-065", "CLM-066", "CLM-067", "CLM-068", "CLM-028", "CLM-044")
SECTION_BINDING = "\\input{sections/cusp-topology-v3-1.tex}"
GENERATED_BINDING = "\\input{generated/cusp-topology-v3-1.tex}"
SECTION_HEADING = "Wall cusps and cells under the literature definition across the screened design sets"
TABLE_MACROS = ("CtvHistogramTable", "CtvSweepStageTable", "CtvPTwoTable", "CtvLineageTable")
REVISION_MACRO = "CuspTopologyEvidenceRevision"
MACRO_PREFIX = "Ctv"

EXPERIMENT_ID = "cusp-topology-search-v3.1"
LINEAGE_EXPERIMENT_ID = "cusp-topology-search-v3"
CLASSIFICATION = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
P2_CLASSIFICATION = "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
CAMPAIGN_STATUS = "accepted_topology_screening"
LINEAGE_TERMINAL_STATE = "assessment_rejection"
LINEAGE_CAMPAIGN_STATUS = "gates_failed"
LINEAGE_FAILING_GATE = "held_out_correspondence"
SCREENING_MODEL = (
    "separatrix cusp topology of prescribed vacuum field maps under the HEMP/DCFT literature definition "
    "(axis null, separatrix traced to the dielectric wall, wall cusp at the intersection, cells between "
    "consecutive cusps): linear-vacuum L1a equivalent-current screening fields for the sweep-v2, four-cell-v2 "
    "and characterization-v1 sets (not P2-qualified; no permanent-magnet or nonlinear-iron material model) and "
    "the P2-qualified adaptive finite-element divergent-exit-stack field (iron poles and return yoke present) "
    "for the single P2 row"
)
FROZEN_FILES = ("protocol.json", "authorities.json", "shakedown.json", "design-authorities.json")
SET_IDS = ("sweep_v2", "four_cell_v2", "characterization_v1", "p2_divergent_exit")
SET_TOKENS = {"sweep_v2": "Sweep", "four_cell_v2": "FourCell", "characterization_v1": "CharV", "p2_divergent_exit": "PTwo"}
SET_LABELS = {
    "sweep_v2": "geometry sweep",
    "four_cell_v2": "four-cell candidates",
    "characterization_v1": "characterization cases",
    "p2_divergent_exit": "P2 divergent-exit stack",
}
COUNT_TOKENS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight")
STAGE_TOKENS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
CUSP_TOKENS = ("One", "Two", "Three")
BINDING_GATE_NAMES = (
    "all_declared_designs_resolved",
    "determinism_replay",
    "every_null_converged",
    "every_trace_terminates_cleanly",
    "every_wall_trace_flux_consistent",
    "hash_bindings",
    "held_out_correspondence",
    "identity_proven",
    "refinement_stability",
)
AXIS_METHODS = ("axis_sign_change", "axis_grid")
V1_CHANNEL_ZONES = ("plasma_channel", "channel_axial_margin")
AUDIT_ROOT_CAUSE_PATTERN = re.compile(
    r"\((\d+) of the (\d+) sealed axis clusters;\s+(\d+) inside the channel, in exactly the (\d+) failing designs\)"
)
AUDIT_CORRECTED_PATTERN = re.compile(
    r"\| (\d+)/(\d+) bijections, max matched difference ([0-9.eE+-]+) m \(tolerance ([0-9.eE+-]+) m\), all X \|"
)
AUDIT_SCRIPT_FAILING_PATTERN = re.compile(r"^RECORDED_FAILING_DESIGN_COUNT = (\d+)$", re.MULTILINE)
AUDIT_SCRIPT_GATE_PATTERN = re.compile(r'^RECORDED_FAILING_GATE = "([a-z_]+)"$', re.MULTILINE)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _version_token(experiment_id: str) -> str:
    match = re.search(r"-(v\d+(?:\.\d+)?)$", experiment_id)
    if match is None:
        raise ValueError(f"experiment id {experiment_id!r} carries no version token")
    return match.group(1)


def _set_version(set_id: str) -> str:
    match = re.search(r"_(v\d+)$", set_id)
    if match is None:
        raise ValueError(f"design set {set_id!r} carries no version token")
    return match.group(1)


def _histogram_text(values: dict[str, int]) -> str:
    return " / ".join(f"{key}:{values[key]}" for key in sorted(values, key=int))


FORMATTERS: dict[str, Callable[[Any], str]] = {
    **_BASE_FORMATTERS,
    "mm1": lambda v: f"{1e3 * float(v):.1f}",
    "mm2": lambda v: f"{1e3 * float(v):.2f}",
    "mm3": lambda v: f"{1e3 * float(v):.3f}",
    "um0": lambda v: f"{1e6 * float(v):.0f}",
    "um1": lambda v: f"{1e6 * float(v):.1f}",
    "pct0": lambda v: f"{100.0 * float(v):.0f}\\%",
    "deg1": lambda v: f"{float(v):.1f}",
    "deg2": lambda v: f"{float(v):.2f}",
    "list_ident_tt": lambda v: ", ".join(f"\\texttt{{{_BASE_FORMATTERS['ident'](x)}}}" for x in v),
    "list_clauses": lambda v: "; ".join(_tex_escape(str(x)) for x in v),
    "list_mm2": lambda v: ", ".join(f"{1e3 * float(x):.2f}" for x in v),
    "list_mm3": lambda v: ", ".join(f"{1e3 * float(x):.3f}" for x in v),
    "histogram": lambda v: _histogram_text(v),
    "sci3": lambda v: _sci(float(v), 3),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "max": None, "median": None, "min": None}
    return {"count": len(clean), "max": max(clean), "median": statistics.median(clean), "min": min(clean)}


def _histogram(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: int(pair[0])))


def _match_sorted(reference: list[float], observed: list[float], tolerance: float) -> tuple[bool, float | None]:
    """Greedy sorted correspondence used by the post-hoc audit (bijection flag, max difference)."""

    remaining = sorted(observed)
    pairs: list[float] = []
    unmatched = 0
    for value in sorted(reference):
        if remaining:
            nearest = min(remaining, key=lambda item: abs(item - value))
            if abs(nearest - value) <= tolerance:
                pairs.append(abs(nearest - value))
                remaining.remove(nearest)
                continue
        unmatched += 1
    return (unmatched == 0 and not remaining), (max(pairs) if pairs else None)


# --------------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------------- #
class Bundle:
    """A sealed results bundle, verified file by file against its own manifest."""

    def __init__(self, repo: Path, results: Path, *, experiment_id: str, expected_state: str) -> None:
        self.repo = repo
        self.results = results
        self.root = repo / results
        manifest_raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.manifest = load_json_bytes(manifest_raw, "results manifest")
        if self.manifest.get("state") != expected_state:
            raise ValueError(f"results manifest state is not {expected_state}")
        if self.manifest.get("experiment_id") != experiment_id:
            raise ValueError("results manifest experiment identity differs")
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
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
            if actual != entry["byte_sha256"] or len(raw) != entry["bytes"]:
                raise ValueError(f"bundle file SHA-256 or size mismatch: {relative}")
            self.hashes[relative] = actual
            self.sizes[relative] = len(raw)
        if self.hashes["terminal.json"] != self.manifest["terminal_byte_sha256"]:
            raise ValueError("terminal.json hash differs from the manifest binding")
        if self.hashes["execution-lock.json"] != self.manifest["lock_byte_sha256"]:
            raise ValueError("execution-lock.json hash differs from the manifest binding")
        # Every artifact except the lock (bound by lock_byte_sha256 above) carries a
        # sidecar whose byte hash and size must agree with the manifest.
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

    def load_gzip(self, relative: str) -> tuple[Any, str]:
        payload = gzip.decompress(self.raw(relative))
        return load_json_bytes(payload, relative), sha256_bytes(payload)


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo, check=False, capture_output=True
    ).returncode == 0


def bind_committed(repo: Path, bundle: Bundle) -> dict[str, Any]:
    """Prove the working-tree bundle equals the committed results revision (and the lineage its own)."""

    head = _git(repo, "rev-parse", "HEAD")
    for commit, label in (
        (RESULTS_COMMIT_SHA, "results"),
        (PREREGISTRATION_COMMIT_SHA, "preregistration"),
        (DASHBOARD_COMMIT_SHA, "dashboard"),
        (LINEAGE_RESULTS_COMMIT_SHA, "lineage results"),
        (LINEAGE_PREREGISTRATION_COMMIT_SHA, "lineage preregistration"),
        (LINEAGE_AUDIT_COMMIT_SHA, "lineage audit"),
        (LITERATURE_COMMIT_SHA, "literature review"),
        (V1_RESULTS_COMMIT_SHA, "characterization v1 results"),
        (V2_RESULTS_COMMIT_SHA, "four-cell v2 results"),
        (SWEEP_RESULTS_COMMIT_SHA, "sweep v2 results"),
    ):
        if not _is_ancestor(repo, commit, head):
            raise ValueError(f"{label} commit is not an ancestor of HEAD")
    for earlier, later, label in (
        (PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA, "preregistration -> results"),
        (RESULTS_COMMIT_SHA, DASHBOARD_COMMIT_SHA, "results -> dashboard"),
        (LINEAGE_PREREGISTRATION_COMMIT_SHA, LINEAGE_RESULTS_COMMIT_SHA, "lineage preregistration -> lineage results"),
        (LINEAGE_RESULTS_COMMIT_SHA, LINEAGE_AUDIT_COMMIT_SHA, "lineage results -> lineage audit"),
        (LINEAGE_AUDIT_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "lineage audit -> preregistration"),
        (LITERATURE_COMMIT_SHA, LINEAGE_PREREGISTRATION_COMMIT_SHA, "literature review -> lineage preregistration"),
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
    lineage_manifest_rel = (LINEAGE_RESULTS / "manifest.json").as_posix()
    lineage_blob = _git(repo, "rev-parse", f"{LINEAGE_RESULTS_COMMIT_SHA}:{lineage_manifest_rel}")
    if lineage_blob != _git(repo, "hash-object", "--", lineage_manifest_rel):
        raise ValueError("working-tree lineage results manifest differs from the committed blob")
    lineage_tree = _git(repo, "rev-parse", f"{LINEAGE_RESULTS_COMMIT_SHA}:{LINEAGE_RESULTS.as_posix()}")
    if lineage_tree != _git(repo, "rev-parse", f"HEAD:{LINEAGE_RESULTS.as_posix()}"):
        raise ValueError("lineage results tree changed after its results revision")
    # The frozen preregistration files carry the same blob at the preregistration and
    # results revisions; the working tree must equal that blob (both campaigns).
    for experiment, prereg, results in (
        (EXPERIMENT, PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA),
        (LINEAGE_EXPERIMENT, LINEAGE_PREREGISTRATION_COMMIT_SHA, LINEAGE_RESULTS_COMMIT_SHA),
    ):
        for name in FROZEN_FILES:
            relative = (experiment / name).as_posix()
            frozen = _git(repo, "rev-parse", f"{prereg}:{relative}")
            recorded = _git(repo, "rev-parse", f"{results}:{relative}")
            working = _git(repo, "hash-object", "--", relative)
            if not frozen == recorded == working:
                raise ValueError(f"frozen {relative} differs between preregistration, results and the working tree")
    subject = _git(repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA)
    return {
        "results_commit": RESULTS_COMMIT_SHA,
        "results_commit_subject": subject,
        "results_tree": results_tree,
        "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
        "dashboard_commit": DASHBOARD_COMMIT_SHA,
        "manifest_git_blob": committed_blob,
        "manifest_path": manifest_rel,
        "lineage": {
            "experiment_id": LINEAGE_EXPERIMENT_ID,
            "preregistration_commit": LINEAGE_PREREGISTRATION_COMMIT_SHA,
            "results_commit": LINEAGE_RESULTS_COMMIT_SHA,
            "results_tree": lineage_tree,
            "posthoc_audit_commit": LINEAGE_AUDIT_COMMIT_SHA,
            "manifest_git_blob": lineage_blob,
            "manifest_path": lineage_manifest_rel,
        },
        "literature_review_commit": LITERATURE_COMMIT_SHA,
        "reference_commits": {
            "characterization_v1": V1_RESULTS_COMMIT_SHA,
            "four_cell_v2": V2_RESULTS_COMMIT_SHA,
            "sweep_v2": SWEEP_RESULTS_COMMIT_SHA,
        },
    }


def _bound_file(repo: Path, relative: Path, revision: str, role: str, *, lf_equal: bool) -> dict[str, Any]:
    """Bind a repository file at a revision; the checkout must equal the blob (bytes or LF-normalised)."""

    blob = _git(repo, "rev-parse", f"{revision}:{relative.as_posix()}")
    committed = subprocess.run(["git", "cat-file", "blob", blob], cwd=repo, check=True, capture_output=True).stdout
    working = (repo / relative).read_bytes()
    if lf_equal:
        if _lf(working) != _lf(committed):
            raise ValueError(f"{relative.as_posix()} differs (LF-normalised) from the blob at {revision[:8]}")
    elif working != committed:
        raise ValueError(f"{relative.as_posix()} differs from the blob at {revision[:8]}")
    return {
        "path": relative.as_posix(),
        "revision": revision,
        "role": role,
        "git_blob": blob,
        "git_blob_sha256": sha256_bytes(committed),
        "sha256": sha256_bytes(working),
        "bytes": len(working),
    }


def cross_check_dashboard(
    repo: Path, bundle: Bundle, lineage: Bundle, dataset: dict[str, Any], campaign: dict[str, Any], gates: dict[str, Any],
    catalogue: dict[str, Any], sealed_v1_sha: str, sealed_v2_sha: str,
) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of the same bundle; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    template_raw = (repo / DASHBOARD_TEMPLATE).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    if f'CLASSIFICATION = "{CLASSIFICATION}"' not in generator_text or f'P2_CLASSIFICATION = "{P2_CLASSIFICATION}"' not in generator_text:
        raise ValueError("dashboard generator does not pin the topology labels")
    if 'verify_bundle(results, expected_state="accepted_result")' not in generator_text:
        raise ValueError("dashboard generator does not verify the bundle state")
    if 'expected_state="assessment_rejection"' not in generator_text:
        raise ValueError("dashboard generator does not verify the lineage bundle state")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_file_sha256"] != bundle.manifest_sha256 or identity["state"] != "accepted_result":
        raise ValueError("dashboard payload names a different results manifest")
    if identity["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA:
        raise ValueError("dashboard payload names a different preregistration commit")
    if identity["experiment_id"] != EXPERIMENT_ID or identity["verified_file_count"] != len(bundle.hashes):
        raise ValueError("dashboard payload identity differs from the bundle")
    if identity["artifact_count"] != bundle.manifest["artifact_count"] or identity["artifact_hashes"] != bundle.hashes:
        raise ValueError("dashboard payload artifact hashes differ from the bundle")
    if identity["terminal_file_sha256"] != bundle.manifest["terminal_byte_sha256"] or identity["lock_file_sha256"] != bundle.manifest["lock_byte_sha256"]:
        raise ValueError("dashboard payload terminal/lock hashes differ from the bundle")
    for key in ("protocol_semantic_sha256", "experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
        if identity[key] != dataset[key]:
            raise ValueError(f"dashboard payload {key} differs from the sealed dataset")
    if identity["generator_sha256"] != sha256_bytes(_lf(generator_raw)) or identity["template_sha256"] != sha256_bytes(_lf(template_raw)):
        raise ValueError("dashboard payload generator/template hashes differ from the checkout")
    if payload["classification"] != CLASSIFICATION or payload["p2_classification"] != P2_CLASSIFICATION:
        raise ValueError("dashboard classification differs from the bundle")
    if payload["headline"] != dataset["headline"] or payload["held_out"] != dataset["held_out"] or payload["p2_consistency"] != dataset["p2_consistency"]:
        raise ValueError("dashboard headline, held-out or P2 consistency block differs from the sealed dataset")
    if payload["claim_boundary"] != dataset["claim_boundary"] or payload["classification_statement"] != dataset["classification_statement"]:
        raise ValueError("dashboard claim boundary differs from the sealed dataset")
    if payload["gates"]["campaign"] != gates["campaign"] or payload["gates"]["replays"] != gates["replays"]:
        raise ValueError("dashboard gates differ from the sealed gates")
    if payload["execution"] != campaign["execution_mode"]:
        raise ValueError("dashboard execution record differs from the campaign result")
    rows = {(item["set"], item["id"]): item for item in payload["designs"]}
    if set(rows) != {(d["set_id"], d["design_id"]) for d in dataset["designs"]}:
        raise ValueError("dashboard design rows differ from the sealed dataset")
    for design in dataset["designs"]:
        row = rows[(design["set_id"], design["design_id"])]
        if row["cusps"] != design["wall_cusp_count"] or row["cells"] != design["cell_count"] or row["z_c_m"] != [c["z_c_m"] for c in design["wall_cusps"]]:
            raise ValueError(f"dashboard row {design['key']} differs from the sealed dataset")
        if row["stable"] is not design["stability"]["stable"] or row["label"] != design["label"] or row["rep"] is not design["representative"]:
            raise ValueError(f"dashboard row {design['key']} stability, label or representative flag differs")
        if row["axis_nulls_m"] != [n["z_m"] for n in design["axis_nulls"]] or row["four_cusps"] is not design["four_wall_cusps"] or row["four_cells"] is not design["four_cells"]:
            raise ValueError(f"dashboard row {design['key']} axis nulls or legacy-target flags differ")
    if len(payload["representatives"]) != sum(1 for d in dataset["designs"] if d["representative"]):
        raise ValueError("dashboard representative count differs from the sealed dataset")
    if payload["catalogue"]["design_count"] != catalogue["design_count"] or payload["catalogue"]["stable_design_count"] != catalogue["stable_design_count"]:
        raise ValueError("dashboard catalogue counts differ from the sealed catalogue")
    lineage_block = payload["lineage"]["v3_recorded_rejection"]
    if lineage_block["manifest_file_sha256"] != lineage.manifest_sha256 or lineage_block["state"] != LINEAGE_TERMINAL_STATE:
        raise ValueError("dashboard lineage block names a different rejected bundle")
    if lineage_block["preregistration_commit_sha"] != LINEAGE_PREREGISTRATION_COMMIT_SHA or lineage_block["experiment_id"] != LINEAGE_EXPERIMENT_ID:
        raise ValueError("dashboard lineage block names a different rejected campaign")
    frozen = payload["lineage"]["frozen_definition_results"]
    if frozen["characterization_v1"]["dataset_file_sha256"] != sealed_v1_sha or frozen["four_cell_v2"]["dataset_file_sha256"] != sealed_v2_sha:
        raise ValueError("dashboard frozen-definition block names different sealed datasets")
    return {
        "generator_path": DASHBOARD_GENERATOR.as_posix(),
        "generator_sha256_lf": sha256_bytes(_lf(generator_raw)),
        "template_path": DASHBOARD_TEMPLATE.as_posix(),
        "template_sha256_lf": sha256_bytes(_lf(template_raw)),
        "html_path": DASHBOARD_HTML.as_posix(),
        "html_sha256_lf": sha256_bytes(_lf(html_raw)),
        "html_schema": payload["schema"],
        "payload_manifest_sha256": identity["manifest_file_sha256"],
        "payload_lineage_manifest_sha256": lineage_block["manifest_file_sha256"],
        "rule": (
            "the committed dashboard byte-verifies the accepted bundle and the recorded lineage bundle, embeds its "
            "own extraction and pins the manifest SHA-256 values and the preregistration commits; the generator "
            "requires that extraction (identity incl. every artifact hash, headline, held-out, P2 consistency, "
            "gates, execution, every per-design row, catalogue counts and the lineage block) to equal the sealed "
            "artifacts before writing any macro"
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


def _range(values: list[float], fmt: str) -> str:
    return f"{format_value(fmt, min(values))}--{format_value(fmt, max(values))}"


# --------------------------------------------------------------------------- #
# Lineage: reproduce the v3 post-hoc audit from the sealed v1 dataset and the v3 records
# --------------------------------------------------------------------------- #
def reproduce_lineage_audit(lineage: Bundle, v1_dataset: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """Re-derive the recorded v3 held-out failure and its correction from sealed data only."""

    gates = lineage.load("artifacts/gates.json")
    recorded_failures = set(gates["failing_designs"][LINEAGE_FAILING_GATE])
    total_clusters = 0
    dropped = 0
    dropped_in_channel = 0
    max_centroid_r = 0.0
    failing: set[str] = set()
    corrected_pass = 0
    corrected_max = 0.0
    explained = True
    for case in v1_dataset["cases"]:
        record = lineage.load(f"artifacts/designs/characterization_v1/{case['case_id']}.json")
        clusters = [
            root for root in case["maps"]["primary"]["roots"]
            if not root["finite_box_boundary"] and any(member["method"] in AXIS_METHODS for member in root["members"])
        ]
        total_clusters += len(clusters)
        drop = [root for root in clusters if root["r_m"] != 0.0]
        dropped += len(drop)
        max_centroid_r = max([max_centroid_r] + [abs(root["r_m"]) for root in drop])
        in_channel = [root for root in drop if root["geometry_association"]["zone"] in V1_CHANNEL_ZONES]
        dropped_in_channel += len(in_channel)
        key = f"characterization_v1:{case['case_id']}"
        if in_channel:
            failing.add(key)
        reference = [root["z_m"] for root in clusters if root["geometry_association"]["zone"] in V1_CHANNEL_ZONES]
        observed = [null["z_m"] for null in record["accepted"]["axis_nulls"]["nulls"] if null["zone"] == "channel"]
        bijection, max_difference = _match_sorted(reference, observed, tolerance)
        all_x = all(root["local_topology"]["classification"] == "X" for root in clusters if root["geometry_association"]["zone"] in V1_CHANNEL_ZONES)
        if bijection and all_x:
            corrected_pass += 1
        if max_difference is not None:
            corrected_max = max(corrected_max, max_difference)
        held_out = record["held_out"]
        if held_out["passed"] is not (not in_channel) or (key in recorded_failures) is not (not held_out["passed"]):
            explained = False
        if len(held_out.get("unmatched_observed_z_m", [])) != len(in_channel):
            explained = False
    if failing != recorded_failures:
        raise ValueError("the recorded v3 failing designs are not exactly the designs with dropped in-channel clusters")
    if not explained:
        raise ValueError("the recorded v3 held-out failures are not explained by the dropped clusters")
    return {
        "sealed_axis_clusters": total_clusters,
        "dropped_by_recorded_filter": dropped,
        "dropped_in_channel": dropped_in_channel,
        "max_dropped_centroid_r_m": max_centroid_r,
        "recorded_failing_designs": sorted(recorded_failures),
        "corrected_filter_pass_count": corrected_pass,
        "corrected_filter_max_difference_m": corrected_max,
        "failures_explained_by_dropped_clusters": explained,
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(repo: Path) -> tuple[dict[str, Any], str]:
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo, RESULTS, experiment_id=EXPERIMENT_ID, expected_state="accepted_result")
    lineage = Bundle(repo, LINEAGE_RESULTS, experiment_id=LINEAGE_EXPERIMENT_ID, expected_state=LINEAGE_TERMINAL_STATE)
    binding = bind_committed(repo, bundle)
    m = Macros(bundle)
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    dataset = m.doc("artifacts/topology-dataset.json")
    protocol = m.doc("artifacts/protocol.json")
    authorities = m.doc("artifacts/authorities.json")
    shakedown = m.doc("artifacts/shakedown.json")
    design_authorities = m.doc("artifacts/design-authorities.json")
    plan = m.doc("artifacts/campaign-plan.json")
    runtime = m.doc("artifacts/runtime.json")
    failures = m.doc("artifacts/design-failures.json")
    source_binding = m.doc("artifacts/source-binding.json")
    catalogue = m.doc("artifacts/cusp-cell-catalogue.json")
    designs = dataset["designs"]
    headline = dataset["headline"]
    definition = dataset["definition_v3"]

    # ---- reference files: the sealed sources the campaign compared against ----
    sealed = dataset["sealed_sources"]
    v1_file = _bound_file(repo, V1_DATASET, V1_RESULTS_COMMIT_SHA, "reference-characterization-dataset", lf_equal=False)
    v2_file = _bound_file(repo, V2_DATASET, V2_RESULTS_COMMIT_SHA, "reference-four-cell-dataset", lf_equal=False)
    sweep_file = _bound_file(repo, SWEEP_MANIFEST, SWEEP_RESULTS_COMMIT_SHA, "reference-sweep-manifest", lf_equal=False)
    if v1_file["sha256"] != sealed["characterization_v1"]["dataset_file_sha256"]:
        raise ValueError("the characterization-v1 dataset on disk differs from the sealed source the campaign bound")
    if v2_file["sha256"] != sealed["four_cell_v2"]["dataset_file_sha256"]:
        raise ValueError("the four-cell-v2 dataset on disk differs from the sealed source the campaign bound")
    if sweep_file["sha256"] != sealed["sweep_v2"]["manifest_file_sha256"]:
        raise ValueError("the sweep-v2 results manifest on disk differs from the sealed source the campaign bound")
    v1_dataset = load_json_bytes((repo / V1_DATASET).read_bytes(), "characterization v1 dataset")
    v2_dataset = load_json_bytes((repo / V2_DATASET).read_bytes(), "four-cell v2 dataset")
    if v1_dataset["preregistration_commit_sha"] != sealed["characterization_v1"]["preregistration_commit"]:
        raise ValueError("characterization-v1 preregistration commit differs from the sealed source")
    if v2_dataset["preregistration_commit_sha"] != sealed["four_cell_v2"]["preregistration_commit"]:
        raise ValueError("four-cell-v2 preregistration commit differs from the sealed source")
    literature_file = _bound_file(repo, LITERATURE_REVIEW, LITERATURE_COMMIT_SHA, "definition-source-review", lf_equal=True)
    if not definition["review_document"].startswith(LITERATURE_REVIEW.as_posix()) or LITERATURE_COMMIT_SHA[:8] not in definition["review_document"]:
        raise ValueError("the frozen definition does not name the bound literature review at its commit")
    literature_text = _lf((repo / LITERATURE_REVIEW).read_bytes()).decode("utf-8")
    for source in definition["literature_basis"]:
        tokens: set[str] = set()
        for locator in (source.get("doi"), source.get("locator")):
            if not locator:
                continue
            tail = locator.split("://", 1)[-1]
            tokens.update({locator, tail, tail.rsplit("/", 1)[-1], tail.rsplit("/", 1)[-1].removesuffix(".pdf")})
        if not tokens or not any(token in literature_text for token in tokens):
            raise ValueError(f"literature source {source['key']} is not present in the bound review")
    lineage_audit_file = _bound_file(repo, LINEAGE_AUDIT, LINEAGE_AUDIT_COMMIT_SHA, "lineage-posthoc-audit", lf_equal=True)
    lineage_script_file = _bound_file(repo, LINEAGE_AUDIT_SCRIPT, LINEAGE_AUDIT_COMMIT_SHA, "lineage-posthoc-audit-script", lf_equal=True)
    lineage_prereg_file = _bound_file(repo, LINEAGE_EXPERIMENT / "protocol.json", LINEAGE_PREREGISTRATION_COMMIT_SHA, "lineage-rejected-preregistration", lf_equal=False)
    dashboard = cross_check_dashboard(
        repo, bundle, lineage, dataset, campaign, gates, catalogue,
        sealed["characterization_v1"]["dataset_file_sha256"], sealed["four_cell_v2"]["dataset_file_sha256"],
    )

    # ---- internal consistency of the sealed bundle (fail closed on any disagreement) ----
    if terminal["state"] != bundle.manifest["state"] or terminal["counts"]["attempt_count"] != 1:
        raise ValueError("terminal record disagrees with the manifest or records more than one attempt")
    if terminal["payload"] != {"design_count": campaign["design_count"], "gates": campaign["campaign_gates"], "stable_design_count": headline["stable_design_count"], "status": campaign["status"]}:
        raise ValueError("terminal payload differs from the campaign result")
    if lock["attempt"] != 1 or lock["immutable"] is not True or lock["commit"] != PREREGISTRATION_COMMIT_SHA or lock["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("execution lock does not record the single immutable attempt at the preregistration commit")
    if campaign["status"] != CAMPAIGN_STATUS or campaign["evidentiary"] is not True or campaign["plan_kind"] != "evidentiary" or campaign["gates_passed"] is not True:
        raise ValueError("campaign result is not the accepted evidentiary topology screening")
    if not (campaign["classification"] == dataset["classification"] == protocol["classification"] == authorities["classification"] == shakedown["classification"] == CLASSIFICATION):
        raise ValueError("classification differs between the sealed artifacts")
    if not (campaign["p2_row_classification"] == dataset["p2_row_classification"] == protocol["p2_row_classification"] == P2_CLASSIFICATION):
        raise ValueError("P2 row classification differs between the sealed artifacts")
    if campaign["campaign_gates"] != gates["campaign"] or gates["passed"] is not True or gates["binding"] is not True or set(gates["campaign"]) != set(BINDING_GATE_NAMES):
        raise ValueError("gates.json disagrees with the campaign result or names a different gate set")
    if any(gates["campaign"][name] is not True for name in BINDING_GATE_NAMES) or any(gates["failing_designs"].values()):
        raise ValueError("gates.json records a failed binding gate or a failing design")
    if set(gates["definitions"]["binding_integrity"]) != set(BINDING_GATE_NAMES) - {"identity_proven"}:
        raise ValueError("gate definitions differ from the binding gate set")
    if campaign["headline"] != headline or dataset["gates"] != {"campaign": gates["campaign"], "failing_designs": gates["failing_designs"], "passed": True}:
        raise ValueError("campaign headline or dataset gate block differs from the sealed gates")
    if not (len(designs) == dataset["design_count"] == campaign["design_count"] == gates["design_count"] == headline["design_count"] == catalogue["design_count"] == authorities["design_count"] == design_authorities["design_count"] == len(plan["design_keys"])):
        raise ValueError("design count differs between the sealed artifacts")
    if plan["kind"] != "evidentiary" or plan["binding_gates"] is not True or plan["design_keys"] != [d["key"] for d in designs]:
        raise ValueError("campaign plan differs from the dataset order")
    if failures["failed"] != [] or campaign["design_count"] != headline["stable_design_count"]:
        raise ValueError("the bundle records a failed or unstable design")
    if not (campaign["set_counts"] == headline["set_counts"] == authorities["set_counts"] == design_authorities["set_counts"] == {s: sum(1 for d in designs if d["set_id"] == s) for s in SET_IDS}):
        raise ValueError("set counts differ between the sealed artifacts and the rows")
    for key in ("protocol_semantic_sha256", "experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
        if not (dataset[key] == authorities[key] == shakedown[key] == source_binding[key]):
            raise ValueError(f"{key} differs between the sealed artifacts")
    if campaign["protocol_semantic_sha256"] != dataset["protocol_semantic_sha256"] or catalogue["protocol_semantic_sha256"] != dataset["protocol_semantic_sha256"]:
        raise ValueError("protocol semantic hash differs between campaign result, dataset and catalogue")
    if not (sealed == authorities["sealed_sources"] == source_binding["sealed_sources"] == shakedown["sealed_sources"]):
        raise ValueError("sealed-source identities differ between the sealed artifacts")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"] or authorities["design_authorities_sha256"] != bundle.hashes["artifacts/design-authorities.json"]:
        raise ValueError("shakedown or design-authorities artifact differs from the bound authority")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if shakedown["timing_projection"] != authorities["shakedown_timing_projection"] or shakedown["timing_projection"]["within_budget"] is not True:
        raise ValueError("shakedown timing projection differs from the authorities or is out of budget")
    if authorities["shakedown_git_head"] != LINEAGE_AUDIT_COMMIT_SHA or shakedown["git"]["head"] != LINEAGE_AUDIT_COMMIT_SHA:
        raise ValueError("the shakedown was not run at the lineage audit commit the protocol chain records")
    shakedown_designs = [f"{s}:{d}" for s in SET_IDS for d in protocol["shakedown"]["designs"][s]]
    if shakedown["shakedown_plan"]["design_keys"] != shakedown_designs or shakedown["design_count"] != len(shakedown_designs):
        raise ValueError("shakedown design keys differ from the frozen protocol")
    for frozen in FROZEN_FILES:
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != m.doc(f"artifacts/{frozen}"):
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
    if runtime["worker_pool_size"] != campaign["execution_mode"]["worker_pool_size"] or runtime["worker_pool_size"] != protocol["execution"]["max_design_workers"]:
        raise ValueError("worker pool differs between runtime, campaign result and protocol")
    if "gpu-not-used" not in lock["device"] or "CPU only" not in runtime["backend"]:
        raise ValueError("the execution did not record a CPU-only run")
    replay_keys = [f"{s}:{d}" for s in SET_IDS for d in protocol["execution"]["replay_designs"][s]]
    if [r["key"] for r in gates["replays"]] != replay_keys or any(not (r["bit_identical"] and r["field_identity_equal"] and r["accepted_grid_equal"] and r["replay_topology_payload_sha256"] == r["worker_topology_payload_sha256"]) for r in gates["replays"]):
        raise ValueError("determinism replays differ from the frozen protocol or were not bit-identical")
    if definition != protocol["definition_v3"] or dataset["claim_boundary"] != protocol["claim_boundary"]:
        raise ValueError("dataset definition or claim boundary differs from the frozen protocol")
    if catalogue["definition"].split(";")[0].strip() != "wall cusp := intersection of the separatrix of an axis null with the straight dielectric wall":
        raise ValueError("catalogue definition differs from the admitted wording")
    if catalogue["experiment_id"] != EXPERIMENT_ID or catalogue["mirror_descriptor_statement"] is not True or set(catalogue["labels"]) != {CLASSIFICATION, P2_CLASSIFICATION}:
        raise ValueError("catalogue identity, labels or mirror statement differ from the campaign")
    if catalogue["stable_design_count"] != headline["stable_design_count"] or len(catalogue["entries"]) != len(designs):
        raise ValueError("catalogue counts differ from the dataset")

    # ---- per-design cross-checks against the sealed design records and field grids ----
    by_key = {d["key"]: d for d in designs}
    catalogue_by_key = {f"{e['set_id']}:{e['design_id']}": e for e in catalogue["entries"]}
    if set(catalogue_by_key) != set(by_key):
        raise ValueError("catalogue entries do not cover exactly the dataset designs")
    csv_rows = list(csv.DictReader(io.StringIO(bundle.raw("artifacts/topology-dataset.csv").decode("utf-8"))))
    if [f"{r['set_id']}:{r['design_id']}" for r in csv_rows] != [d["key"] for d in designs]:
        raise ValueError("dataset CSV rows differ from the dataset order")
    wall_trace_count = 0
    flux_root_max = 0.0
    representative_count = 0
    replays_bit_identical = sum(1 for r in gates["replays"] if r["bit_identical"])
    v4_bilinear_max = 0.0
    axis_shift_max = 0.0
    for index, design in enumerate(designs):
        key = design["key"]
        label = f"design {key}"
        set_id = design["set_id"]
        if set_id not in SET_IDS or key != f"{set_id}:{design['design_id']}" or design["ordinal"] != sum(1 for d in designs[:index] if d["set_id"] == set_id):
            raise ValueError(f"{label}: key, set or ordinal is inconsistent")
        expected_label = P2_CLASSIFICATION if set_id == "p2_divergent_exit" else CLASSIFICATION
        if design["label"] != expected_label:
            raise ValueError(f"{label}: label differs from the set's classification")
        checks = design["gate_checks"]
        if set(checks) != set(BINDING_GATE_NAMES) - {"all_declared_designs_resolved", "determinism_replay", "hash_bindings"} or any(v is not True for v in checks.values()):
            raise ValueError(f"{label}: a per-design gate check failed or the check set differs")
        if gates["per_design"][key] != checks:
            raise ValueError(f"{label}: gates.json per-design checks differ from the dataset")
        cusps = design["wall_cusps"]
        if design["wall_cusp_count"] != len(cusps) or design["cell_count"] != len(design["cells"]) or design["cell_count"] != (len(cusps) + 1 if cusps else 1):
            raise ValueError(f"{label}: cusp and cell counts are inconsistent")
        if [c["z_c_m"] for c in cusps] != sorted(c["z_c_m"] for c in cusps) or design["axis_null_count"] != len(design["axis_nulls"]):
            raise ValueError(f"{label}: cusps are not sorted or the axis-null count differs")
        if design["channel_axis_null_count"] != sum(1 for n in design["axis_nulls"] if n["zone"] == "channel") or any(n["classification"] != "X" for n in design["axis_nulls"]):
            raise ValueError(f"{label}: channel null count or X classification differs")
        geometry = design["geometry"]
        for cusp in cusps:
            if not geometry["straight_z_min_m"] <= cusp["z_c_m"] <= geometry["straight_z_max_m"]:
                raise ValueError(f"{label}: a wall cusp lies outside the straight dielectric")
            if cusp["boundary_ambiguous"] is not (min(cusp["z_c_m"] - geometry["straight_z_min_m"], geometry["straight_z_max_m"] - cusp["z_c_m"]) <= definition["numerical_parameters"]["boundary_ambiguity_tolerance_m"]):
                raise ValueError(f"{label}: boundary-ambiguity flag does not recompute")
            if cusp["z_c_over_length"] != cusp["z_c_m"] / geometry["chamber_length_m"]:
                raise ValueError(f"{label}: z_c over length does not recompute")
            # Inter-magnet gap centres: midpoints between consecutive stage centres plus the
            # half-pitch positions beyond the first and last stage (the stack's end gaps).
            centres = geometry["stage_centres_m"]
            half_pitch = geometry["stage_pitch_m"] / 2
            gap_centres = [centres[0] - half_pitch, *((a + b) / 2 for a, b in zip(centres, centres[1:])), centres[-1] + half_pitch]
            if cusp["distance_to_nearest_stage_gap_m"] != min(abs(cusp["z_c_m"] - g) for g in gap_centres):
                raise ValueError(f"{label}: distance to the nearest inter-magnet gap does not recompute")
            if cusp["distance_to_nearest_stage_centre_m"] != min(abs(cusp["z_c_m"] - s) for s in geometry["stage_centres_m"]):
                raise ValueError(f"{label}: distance to the nearest stage centre does not recompute")
        if design["four_wall_cusps"] is not (len(cusps) == 4) or design["four_cells"] is not (len(design["cells"]) == 4):
            raise ValueError(f"{label}: legacy-target flags do not recompute")
        for previous, cell in zip([None, *design["cells"]], design["cells"]):
            if cell["length_m"] != cell["z_end_m"] - cell["z_start_m"] or cell["length_over_pitch"] != cell["length_m"] / geometry["stage_pitch_m"]:
                raise ValueError(f"{label}: cell length does not recompute")
            if previous is not None and cell["z_start_m"] != previous["z_end_m"]:
                raise ValueError(f"{label}: cells are not contiguous")
        kinds = [c["kind"] for c in design["cells"]]
        expected_kinds = ["unbounded"] if not cusps else ["anode_partial", *(["interior"] * (len(cusps) - 1)), "exit_partial"]
        if kinds != expected_kinds:
            raise ValueError(f"{label}: cell kinds differ from the definition")
        stability = design["stability"]
        if stability["stable"] is not True or not (stability["axis_null_count_equal"] and stability["wall_cusp_count_equal"] and stability["wall_reaching_count_equal"]):
            raise ValueError(f"{label}: stability flags differ from the recorded acceptance")
        if stability["max_wall_intersection_shift_m"] is not None and stability["max_wall_intersection_shift_m"] > definition["stability_tolerance_m"]:
            raise ValueError(f"{label}: wall-intersection shift exceeds the stability tolerance")
        if stability["max_axis_null_shift_m"] is not None:
            axis_shift_max = max(axis_shift_max, stability["max_axis_null_shift_m"])
        held_out = design["held_out"]
        if held_out["applies"] is not (set_id in ("sweep_v2", "characterization_v1")) or held_out["passed"] is not True:
            raise ValueError(f"{label}: held-out applicability or outcome differs from the protocol")
        if held_out["applies"] and (held_out["observed_count"] != held_out["reference_count"] or (held_out["max_difference_m"] is not None and held_out["max_difference_m"] > definition["held_out_tolerance_m"])):
            raise ValueError(f"{label}: held-out correspondence is not a bijection within tolerance")
        if held_out["applies"] and (held_out["max_difference_m"] is None) is not (held_out["observed_count"] == 0):
            raise ValueError(f"{label}: held-out difference is missing for a matched design or present for an empty one")
        record = m.doc(design["record_path"])
        if record["key"] != key or record["status"] != "resolved" or record["gate_checks"] != checks or record["geometry"] != geometry:
            raise ValueError(f"{label}: design record identity, status, checks or geometry differ from the dataset")
        accepted = record["accepted"]
        if [c["z_c_m"] for c in accepted["topology"]["wall_cusps"]] != [c["z_c_m"] for c in cusps] or accepted["topology"]["cell_count"] != design["cell_count"]:
            raise ValueError(f"{label}: design record topology differs from the dataset row")
        if [n["z_m"] for n in accepted["axis_nulls"]["nulls"]] != [n["z_m"] for n in design["axis_nulls"]] or not (accepted["axis_nulls"]["all_converged"] and accepted["axis_nulls"]["all_x_type"] and accepted["axis_nulls"]["all_classifications_agree"]):
            raise ValueError(f"{label}: design record axis nulls differ from the dataset or were not all converged X nulls")
        if len(accepted["separatrix_traces"]) != len(design["axis_nulls"]) or not (accepted["all_traces_terminate_cleanly"] and accepted["all_wall_traces_flux_consistent"]):
            raise ValueError(f"{label}: separatrix traces are not one per null, clean and flux-consistent")
        for trace in accepted["separatrix_traces"]:
            if trace["termination"] not in ("wall", "domain_z"):
                raise ValueError(f"{label}: a trace terminated in {trace['termination']!r}")
            if trace["reaches_wall"]:
                wall_trace_count += 1
                if trace["flux_root_consistent"] is not True or trace["flux_root_difference_m"] > definition["numerical_parameters"]["trace_flux_root_tolerance_m"]:
                    raise ValueError(f"{label}: a wall trace disagrees with its flux root")
                flux_root_max = max(flux_root_max, abs(trace["flux_root_difference_m"]))
                if trace.get("v4_bilinear_difference_m") is not None:
                    v4_bilinear_max = max(v4_bilinear_max, abs(trace["v4_bilinear_difference_m"]))
            if (trace["path_rz_m"] is not None) is not design["representative"]:
                raise ValueError(f"{label}: sampled separatrix paths are kept for representatives only")
        if record["stability"]["stable"] is not True or record["stability"]["max_wall_intersection_shift_m"] != stability["max_wall_intersection_shift_m"] or record["stability"]["tolerance_m"] != definition["stability_tolerance_m"]:
            raise ValueError(f"{label}: design record stability differs from the dataset")
        if record["held_out"]["passed"] is not True or record["held_out"]["applies"] is not held_out["applies"]:
            raise ValueError(f"{label}: design record held-out block differs from the dataset")
        if record["evidence"]["identity_proven"] is not True or record["identity"]["accepted_field_identity_sha256"] != design["identity"]["accepted_field_identity_sha256"] or record["identity"]["refined_field_identity_sha256"] != design["identity"]["refined_field_identity_sha256"]:
            raise ValueError(f"{label}: field identity is not proven or differs from the dataset")
        if set_id != "p2_divergent_exit":
            for solve in ("accepted_solve", "refined_solve"):
                if record["evidence"][solve]["converged"] is not True or record["evidence"][solve]["relative_residual_l2"] > record["identity"]["solver_config"]["relative_tolerance"]:
                    raise ValueError(f"{label}: {solve} did not converge within the solver tolerance")
        if design["representative"]:
            representative_count += 1
            stored = record["evidence"].get("stored_representative")
            if set_id != "p2_divergent_exit" and (stored is None or stored["passed"] is not True):
                raise ValueError(f"{label}: stored representative map was not reproduced")
        grid, payload_sha = bundle.load_gzip(record["accepted_grid_path"])
        if payload_sha != record["accepted_grid_payload_sha256"] or grid["identity"]["accepted_field_identity_sha256"] != design["identity"]["accepted_field_identity_sha256"]:
            raise ValueError(f"{label}: accepted field grid payload hash or identity differs from the design record")
        if grid["wall_radius_m"] != geometry["wall_radius_m"] or len(grid["z_m"]) != design["grid"]["axial_samples"] or len(grid["r_m"]) != design["grid"]["radial_samples"]:
            raise ValueError(f"{label}: field grid shape differs from the recorded grid")
        entry = catalogue_by_key[key]
        if entry["stable"] is not True or entry["label"] != design["label"] or entry["wall_cusp_count"] != len(cusps) or [c["z_c_m"] for c in entry["wall_cusps"]] != [c["z_c_m"] for c in cusps]:
            raise ValueError(f"{label}: catalogue entry differs from the dataset row")
        if [c["axis_mirror_ratio"] for c in entry["cells"]] != [c["axis_mirror_ratio"] for c in design["cells"]] or [c["wall_mirror_ratio"] for c in entry["cells"]] != [c["wall_mirror_ratio"] for c in design["cells"]]:
            raise ValueError(f"{label}: catalogue mirror descriptors differ from the dataset row")
        if entry["geometry"] != geometry or entry["record_path"] != design["record_path"]:
            raise ValueError(f"{label}: catalogue geometry or record path differs from the dataset row")
        row = csv_rows[index]
        if int(row["wall_cusp_count"]) != len(cusps) or int(row["cell_count"]) != design["cell_count"] or row["stable"] != "True" or row["label"] != design["label"]:
            raise ValueError(f"{label}: CSV row differs from the dataset row")
    if wall_trace_count != dataset["estimands"]["pooled_all"]["v4_bilinear_difference_m"]["count"]:
        raise ValueError("wall-reaching trace count differs from the pooled estimand")

    # ---- re-derive the headline and the per-set estimands from the rows ----
    interior = [c for d in designs for c in d["cells"] if c["kind"] == "interior"]
    all_cusps = [c for d in designs for c in d["wall_cusps"]]
    recomputed_headline = {
        "angle_to_wall_normal_deg": _distribution([c["angle_to_wall_normal_deg"] for c in all_cusps]),
        "design_count": len(designs),
        "four_cell_fraction_by_set": {s: sum(d["four_cells"] for d in designs if d["set_id"] == s) / sum(1 for d in designs if d["set_id"] == s) for s in SET_IDS},
        "four_wall_cusp_fraction_by_set": {s: sum(d["four_wall_cusps"] for d in designs if d["set_id"] == s) / sum(1 for d in designs if d["set_id"] == s) for s in SET_IDS},
        "held_out": {
            s: {
                "applies": any(d["held_out"]["applies"] for d in designs if d["set_id"] == s),
                "design_count": sum(1 for d in designs if d["set_id"] == s),
                "max_difference_m": max([d["held_out"]["max_difference_m"] for d in designs if d["set_id"] == s and d["held_out"]["max_difference_m"] is not None], default=None),
                "observed_null_count": sum(d["held_out"]["observed_count"] or 0 for d in designs if d["set_id"] == s),
                "passed_count": sum(1 for d in designs if d["set_id"] == s and d["held_out"]["passed"]),
                "reference_null_count": sum(d["held_out"]["reference_count"] or 0 for d in designs if d["set_id"] == s),
            }
            for s in SET_IDS
        },
        "interior_axis_mirror_ratio": _distribution([c["axis_mirror_ratio"] for c in interior]),
        "interior_wall_mirror_ratio": _distribution([c["wall_mirror_ratio"] for c in interior]),
        "max_wall_intersection_shift_m": max(d["stability"]["max_wall_intersection_shift_m"] for d in designs if d["stability"]["max_wall_intersection_shift_m"] is not None),
        "p2_consistency": {
            "cusp_count_equals_reference_count": dataset["p2_consistency"]["cusp_count_equals_reference_count"],
            "max_abs_difference_axis_null_to_pic_plane_m": max(abs(c["difference_axis_null_to_pic_plane_m"]) for c in dataset["p2_consistency"]["cusps"]),
            "max_abs_difference_to_dashboard_maximum_m": max(abs(c["difference_to_dashboard_maximum_m"]) for c in dataset["p2_consistency"]["cusps"]),
        },
        "set_counts": {s: sum(1 for d in designs if d["set_id"] == s) for s in SET_IDS},
        "stable_design_count": sum(1 for d in designs if d["stability"]["stable"]),
        "wall_cusp_count_histogram": _histogram([d["wall_cusp_count"] for d in designs]),
        "wall_cusp_count_histogram_by_set": {s: _histogram([d["wall_cusp_count"] for d in designs if d["set_id"] == s]) for s in SET_IDS},
        "z_c_over_length": _distribution([c["z_c_over_length"] for c in all_cusps]),
    }
    if recomputed_headline != headline:
        raise ValueError("the sealed headline does not recompute from the per-design rows")
    for set_id in (*SET_IDS, "pooled_all"):
        rows = designs if set_id == "pooled_all" else [d for d in designs if d["set_id"] == set_id]
        est = dataset["estimands"][set_id]
        cusps = [c for d in rows for c in d["wall_cusps"]]
        cells = [c for d in rows for c in d["cells"]]
        inter = [c for c in cells if c["kind"] == "interior"]
        recomputed = {
            "all_cells_axis_mirror_ratio": _distribution([c["axis_mirror_ratio"] for c in cells]),
            "all_cells_wall_mirror_ratio": _distribution([c["wall_mirror_ratio"] for c in cells]),
            "angle_to_wall_normal_deg": _distribution([c["angle_to_wall_normal_deg"] for c in cusps]),
            "axis_null_count_histogram": _histogram([d["axis_null_count"] for d in rows]),
            "axis_to_wall_shift_m": _distribution([c["z_c_m"] - c["axis_null_z_m"] for c in cusps]),
            "boundary_ambiguous_cusp_count": sum(c["boundary_ambiguous"] for c in cusps),
            "cell_count_histogram": _histogram([d["cell_count"] for d in rows]),
            "channel_axis_null_count_histogram": _histogram([d["channel_axis_null_count"] for d in rows]),
            "design_count": len(rows),
            "designs_with_at_least_one_cusp": sum(1 for d in rows if d["wall_cusp_count"] > 0),
            "distance_to_nearest_stage_centre_m": _distribution([c["distance_to_nearest_stage_centre_m"] for c in cusps]),
            "distance_to_nearest_stage_gap_m": _distribution([c["distance_to_nearest_stage_gap_m"] for c in cusps]),
            "four_cell_count": sum(d["four_cells"] for d in rows),
            "four_cell_fraction": sum(d["four_cells"] for d in rows) / len(rows),
            "four_wall_cusp_count": sum(d["four_wall_cusps"] for d in rows),
            "four_wall_cusp_fraction": sum(d["four_wall_cusps"] for d in rows) / len(rows),
            "interior_axis_mirror_ratio": _distribution([c["axis_mirror_ratio"] for c in inter]),
            "interior_cell_length_m": _distribution([c["length_m"] for c in inter]),
            "interior_cell_length_over_pitch": _distribution([c["length_over_pitch"] for c in inter]),
            "interior_wall_mirror_ratio": _distribution([c["wall_mirror_ratio"] for c in inter]),
            "max_axis_null_shift_m": _distribution([d["stability"]["max_axis_null_shift_m"] for d in rows]),
            "max_wall_intersection_shift_m": _distribution([d["stability"]["max_wall_intersection_shift_m"] for d in rows]),
            "outside_intersection_zones": dict(sorted({z: sum(1 for d in rows for o in d["outside_intersections"] if o["zone"] == z) for z in {o["zone"] for d in rows for o in d["outside_intersections"]}}.items())),
            "stable_design_count": sum(1 for d in rows if d["stability"]["stable"]),
            "wall_b_at_cusp_t": _distribution([c["wall_b_t"] for c in cusps]),
            "wall_cusp_count_histogram": _histogram([d["wall_cusp_count"] for d in rows]),
            "z_c_m": _distribution([c["z_c_m"] for c in cusps]),
            "z_c_over_length": _distribution([c["z_c_over_length"] for c in cusps]),
        }
        recorded = {k: v for k, v in est.items() if k != "v4_bilinear_difference_m"}
        if recomputed != recorded:
            raise ValueError(f"the sealed {set_id} estimands do not recompute from the per-design rows")
    p2_rows = [d for d in designs if d["set_id"] == "p2_divergent_exit"]
    if len(p2_rows) != 1 or p2_rows[0]["p2_consistency"] != dataset["p2_consistency"]:
        raise ValueError("the P2 consistency block differs between the row and the dataset")
    p2 = p2_rows[0]
    p2c = dataset["p2_consistency"]
    if len(p2c["cusps"]) != len(p2["wall_cusps"]) or p2c["cusp_count_equals_reference_count"] is not (len(p2["wall_cusps"]) == len(p2c["references"]["pic_axis_null_planes_m"]) == len(p2c["references"]["topology_dashboard_wall_abs_br_maxima_m"])):
        raise ValueError("the P2 cusp count does not match the consistency references")
    for cusp, entry in zip(p2["wall_cusps"], p2c["cusps"]):
        if entry["z_c_m"] != cusp["z_c_m"] or entry["axis_null_z_m"] != cusp["axis_null_z_m"]:
            raise ValueError("P2 consistency entries differ from the wall cusps")
        plane = min(p2c["references"]["pic_axis_null_planes_m"], key=lambda z: abs(z - cusp["axis_null_z_m"]))
        maximum = min(p2c["references"]["topology_dashboard_wall_abs_br_maxima_m"], key=lambda z: abs(z - cusp["z_c_m"]))
        if entry["nearest_pic_axis_null_plane_m"] != plane or entry["difference_axis_null_to_pic_plane_m"] != cusp["axis_null_z_m"] - plane:
            raise ValueError("P2 PIC-plane difference does not recompute")
        if entry["nearest_dashboard_wall_abs_br_maximum_m"] != maximum or entry["difference_to_dashboard_maximum_m"] != cusp["z_c_m"] - maximum:
            raise ValueError("P2 dashboard-maximum difference does not recompute")
    if p2c["references"] != protocol["design_sets"]["p2_divergent_exit"]["consistency_references"]:
        raise ValueError("P2 consistency references differ from the frozen protocol")

    # ---- lineage: the recorded v3 rejection and its audit, reproduced from sealed data ----
    lineage_campaign = lineage.load("artifacts/campaign-result.json")
    lineage_gates = lineage.load("artifacts/gates.json")
    lineage_terminal = lineage.load("terminal.json")
    lineage_lock = lineage.load("execution-lock.json")
    lineage_protocol = lineage.load("artifacts/protocol.json")
    lineage_dataset = lineage.load("artifacts/topology-dataset.json")
    if lineage_terminal["state"] != LINEAGE_TERMINAL_STATE or lineage_campaign["status"] != LINEAGE_CAMPAIGN_STATUS or lineage_campaign["gates_passed"] is not False:
        raise ValueError("the lineage bundle is not the recorded rejection")
    if lineage_lock["commit"] != LINEAGE_PREREGISTRATION_COMMIT_SHA or lineage_lock["attempt"] != 1 or lineage_lock["experiment_id"] != LINEAGE_EXPERIMENT_ID:
        raise ValueError("the lineage lock names a different preregistration or attempt")
    lineage_failing = {name: designs_ for name, designs_ in lineage_gates["failing_designs"].items() if designs_}
    if set(lineage_failing) != {LINEAGE_FAILING_GATE} or lineage_gates["campaign"][LINEAGE_FAILING_GATE] is not False:
        raise ValueError("the lineage bundle fails a gate other than the recorded held-out gate")
    if any(lineage_gates["campaign"][name] is not True for name in BINDING_GATE_NAMES if name != LINEAGE_FAILING_GATE):
        raise ValueError("a lineage binding gate other than the held-out gate is false")
    if lineage_campaign["headline"]["wall_cusp_count_histogram"] != headline["wall_cusp_count_histogram"] or lineage_campaign["headline"]["stable_design_count"] != headline["stable_design_count"]:
        raise ValueError("the lineage headline differs from the accepted campaign (the definition, sets and pipeline were unchanged)")
    disclosure = protocol["prior_campaign_disclosure"]["v3"]
    if disclosure["preregistration_commit"] != LINEAGE_PREREGISTRATION_COMMIT_SHA or disclosure["result_commit"] != LINEAGE_RESULTS_COMMIT_SHA:
        raise ValueError("the frozen protocol discloses different lineage commits")
    if disclosure["terminal_state"] != LINEAGE_TERMINAL_STATE or disclosure["status"] != LINEAGE_CAMPAIGN_STATUS or disclosure["failing_gate"] != LINEAGE_FAILING_GATE:
        raise ValueError("the frozen protocol discloses a different lineage outcome")
    if disclosure["failing_design_count"] != len(lineage_failing[LINEAGE_FAILING_GATE]) or disclosure["post_hoc_audit"].split(" ")[0] != LINEAGE_AUDIT.as_posix():
        raise ValueError("the frozen protocol discloses a different failing count or audit path")
    if lineage_protocol["definition_v3"]["numerical_parameters"] != definition["numerical_parameters"] or lineage_protocol["definition_v3"]["stability_tolerance_m"] != definition["stability_tolerance_m"] or lineage_protocol["definition_v3"]["held_out_tolerance_m"] != definition["held_out_tolerance_m"]:
        raise ValueError("the lineage definition parameters differ from the accepted campaign")
    if lineage_protocol["design_sets"] != protocol["design_sets"] or lineage_dataset["design_count"] != len(designs):
        raise ValueError("the lineage design sets differ from the accepted campaign")
    if lineage_dataset["dependency_source_sha256"] != dataset["dependency_source_sha256"] or lineage_dataset["field_pipeline_source_sha256"] != dataset["field_pipeline_source_sha256"]:
        raise ValueError("the lineage dependency or field-pipeline sources differ from the accepted campaign")
    audit = reproduce_lineage_audit(lineage, v1_dataset, definition["held_out_tolerance_m"])
    audit_text = _lf((repo / LINEAGE_AUDIT).read_bytes()).decode("utf-8")
    root_cause = AUDIT_ROOT_CAUSE_PATTERN.search(audit_text)
    corrected = AUDIT_CORRECTED_PATTERN.search(audit_text)
    if root_cause is None or corrected is None:
        raise ValueError("the lineage audit does not carry its documented numbers in the expected form")
    documented = {
        "dropped": int(root_cause.group(1)), "clusters": int(root_cause.group(2)), "dropped_in_channel": int(root_cause.group(3)),
        "failing_designs": int(root_cause.group(4)), "corrected_passed": int(corrected.group(1)), "corrected_total": int(corrected.group(2)),
        "corrected_max_difference_m": float(corrected.group(3)), "tolerance_m": float(corrected.group(4)),
    }
    if documented["dropped"] != audit["dropped_by_recorded_filter"] or documented["clusters"] != audit["sealed_axis_clusters"] or documented["dropped_in_channel"] != audit["dropped_in_channel"]:
        raise ValueError("the lineage audit's documented cluster counts do not reproduce")
    if documented["failing_designs"] != len(audit["recorded_failing_designs"]) or documented["corrected_passed"] != audit["corrected_filter_pass_count"] or documented["corrected_total"] != len(v1_dataset["cases"]):
        raise ValueError("the lineage audit's documented correspondence does not reproduce")
    if f"{audit['corrected_filter_max_difference_m']:.2e}" != f"{documented['corrected_max_difference_m']:.2e}" or documented["tolerance_m"] != definition["held_out_tolerance_m"]:
        raise ValueError("the lineage audit's documented maximum difference or tolerance does not reproduce")
    script_text = _lf((repo / LINEAGE_AUDIT_SCRIPT).read_bytes()).decode("utf-8")
    script_failing = AUDIT_SCRIPT_FAILING_PATTERN.search(script_text)
    script_gate = AUDIT_SCRIPT_GATE_PATTERN.search(script_text)
    if script_failing is None or int(script_failing.group(1)) != documented["failing_designs"] or script_gate is None or script_gate.group(1) != LINEAGE_FAILING_GATE:
        raise ValueError("the lineage audit script does not pin the recorded failing gate and count")
    if audit["corrected_filter_pass_count"] != headline["held_out"]["characterization_v1"]["passed_count"] or f"{audit['corrected_filter_max_difference_m']:.6e}" != f"{headline['held_out']['characterization_v1']['max_difference_m']:.6e}":
        raise ValueError("the corrected held-out correspondence of the audit differs from the accepted campaign's held-out record")

    # ---- the frozen-definition references: v1 in-channel roots and the v2 source policy ----
    v1_channel_roots = 0
    v1_channel_axis = 0
    v1_off_axis: list[dict[str, Any]] = []
    v1_stage_counts: set[int] = set()
    for case in v1_dataset["cases"]:
        v1_stage_counts.add(int(case["stage_count"]))
        for root in case["maps"]["primary"]["roots"]:
            if root["finite_box_boundary"] or root["geometry_association"]["zone"] not in V1_CHANNEL_ZONES:
                continue
            v1_channel_roots += 1
            if any(member["method"] in AXIS_METHODS for member in root["members"]):
                v1_channel_axis += 1
            else:
                v1_off_axis.append({"case_id": case["case_id"], "r_over_wall": root["r_m"] / case["chamber_radius_m"], "classification": root["local_topology"]["classification"], "exclusion_reason": root.get("exclusion_reason"), "eligible_cusp": root["eligible_cusp"]})
    if v1_channel_axis != headline["held_out"]["characterization_v1"]["reference_null_count"]:
        raise ValueError("the sealed v1 in-channel axis clusters differ from the held-out reference count")
    if v1_dataset["summary"]["stable_eligible_cusp_count"] != 0 or v1_dataset["summary"]["stable_eligible_cell_count"] != 0 or v1_dataset["summary"]["evaluated_count"] != len(v1_dataset["cases"]):
        raise ValueError("the sealed v1 summary differs from the admitted characterization null")
    if any(root["eligible_cusp"] for root in v1_off_axis) or any(root["classification"] != "X" for root in v1_off_axis):
        raise ValueError("a v1 off-axis in-channel root was eligible or not X-type")
    v1_exclusions = sorted({root["exclusion_reason"] for root in v1_off_axis})
    if v1_exclusions != ["no_cell_bounding_separatrix"]:
        raise ValueError("the v1 off-axis in-channel roots carry an unexpected exclusion reason")
    v2_ratios = [case["sampling"]["values"]["alternating_strength_ratio"] for case in v2_dataset["cases"]]
    if len(v2_ratios) != len([d for d in designs if d["set_id"] == "four_cell_v2"]) or v2_dataset["summary"]["stable_count"] != 0 or v2_dataset["summary"]["evaluated_count"] != len(v2_ratios):
        raise ValueError("the sealed v2 dataset differs from the admitted four-cell null or the v2 design set")
    if {case["candidate_id"] for case in v2_dataset["cases"]} != {d["design_id"] for d in designs if d["set_id"] == "four_cell_v2"}:
        raise ValueError("the v2 candidate ids differ between the sealed v2 dataset and the campaign")

    # ---- identity and lifecycle ----
    m.add("CtvClassification", "artifacts/campaign-result.json", "/classification", "ident", "screening classification string of the L1a design sets")
    m.add("CtvPTwoClassification", "artifacts/campaign-result.json", "/p2_row_classification", "ident", "classification string of the single P2 row")
    m.add("CtvTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("CtvCampaignStatus", "artifacts/campaign-result.json", "/status", "ident", "recorded campaign status")
    m.add_derived("CtvRecordedOutcome", RECORDED_OUTCOME, "ident", "recorded outcome admitted by the numerical-screening gate", "constant of the generator; the gate admits the study at exactly this outcome, which names campaign-result.json#/status", [{"artifact": "artifacts/campaign-result.json", "pointer": "/status"}])
    m.add_derived("CtvScreeningModel", SCREENING_MODEL, "text", "screening model label", "constant of the generator naming the definition of protocol.json#/definition_v3 in the field levels of protocol.json#/claim_boundary/field_level", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}, {"artifact": "artifacts/protocol.json", "pointer": "/definition_v3/wall_cusp_and_cell/cusp"}])
    m.add("CtvExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier")
    m.add_derived("CtvCampaignVersion", _version_token(protocol["experiment_id"]), "text", "version token of the campaign", "trailing version token of protocol.experiment_id", [{"artifact": "artifacts/protocol.json", "pointer": "/experiment_id"}])
    m.add("CtvAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("CtvLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutability flag")
    m.add("CtvPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    m.add_derived("CtvResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("CtvDashboardCommit", DASHBOARD_COMMIT_SHA, "sha_short", "dashboard commit prefix", "git commit of the results dashboard whose embedded extraction equals the bundle", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("CtvManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("CtvVerifiedFiles", len(bundle.hashes), "int_comma", "bundle files verified byte-for-byte", "count of manifest file entries whose sha256 and size equal the checkout", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("CtvArtifactCount", bundle.manifest["artifact_count"], "int_comma", "manifest entries (files and directories)", "manifest.artifact_count", [{"artifact": "manifest.json", "pointer": "/artifact_count"}])
    m.add_derived("CtvToleratedEolFiles", 0, "int", "manifest entries accepted through any end-of-line tolerance", "count of manifest file entries whose recorded byte_sha256 differs from sha256(bytes); the generator grants no tolerance and fails on any difference", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add("CtvProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("CtvExperimentCodeSha", "artifacts/authorities.json", "/experiment_code_sha256", "sha_short", "experiment code hash prefix")
    m.add("CtvDependencySourceSha", "artifacts/authorities.json", "/dependency_source_sha256", "sha_short", "dependency source hash prefix")
    m.add("CtvFieldPipelineSha", "artifacts/authorities.json", "/field_pipeline_source_sha256", "sha_short", "field pipeline source hash prefix")
    m.add_derived("CtvExperimentCodeFiles", len(source_binding["experiment_code_files"]), "int", "experiment code files hashed", "len(source-binding.experiment_code_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/experiment_code_files"}])
    m.add_derived("CtvDependencySourceFiles", len(source_binding["dependency_source_files"]), "int", "dependency source files hashed", "len(source-binding.dependency_source_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/dependency_source_files"}])
    m.add_derived("CtvFieldPipelineFiles", len(source_binding["field_pipeline_source_files"]), "int", "field pipeline source files hashed", "len(source-binding.field_pipeline_source_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/field_pipeline_source_files"}])
    m.add("CtvBackend", "artifacts/runtime.json", "/backend", "text", "solver and tracing backend")
    m.add("CtvCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "logical CPUs of the host")
    m.add("CtvWorkerPool", "artifacts/campaign-result.json", "/execution_mode/worker_pool_size", "int", "worker pool size")
    m.add("CtvDevice", "execution-lock.json", "/device", "ident", "device string recorded in the execution lock")
    m.add("CtvStageWallMin", "artifacts/campaign-result.json", "/execution_mode/stage_wall_s", "min1", "wall time of the design stage (min)")
    m.add("CtvAssessmentWallMin", "artifacts/campaign-result.json", "/execution_mode/assessment_wall_s", "min1", "wall time of the assessment (min)")
    m.add("CtvShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("CtvShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add("CtvShakedownOutcomesEnterEstimand", "artifacts/shakedown.json", "/outcomes_enter_estimand", "bool", "shakedown outcomes enter an estimand")
    m.add("CtvShakedownDesigns", "artifacts/shakedown.json", "/design_count", "int", "shakedown designs")
    m.add_derived("CtvShakedownDefectDesign", "characterization_v1:topology-s05-p0-r0-neg", "ident", "the shakedown design added for the corrected held-out extraction", "constant of the generator; required to be a shakedown design of the frozen protocol and a recorded failing design of the lineage campaign", [{"artifact": "artifacts/protocol.json", "pointer": "/shakedown/designs/characterization_v1"}])
    if "topology-s05-p0-r0-neg" not in protocol["shakedown"]["designs"]["characterization_v1"] or "characterization_v1:topology-s05-p0-r0-neg" not in audit["recorded_failing_designs"]:
        raise ValueError("the shakedown does not exercise the recorded defect design")
    m.add("CtvTimingWithinBudget", "artifacts/shakedown.json", "/timing_projection/within_budget", "bool", "evidentiary run projected within the wall-time budget")
    m.add("CtvTimingBudgetMin", "artifacts/shakedown.json", "/timing_projection/budget_wall_seconds", "min1", "wall-time budget (min)")
    m.add("CtvTimingProjectedMin", "artifacts/shakedown.json", "/timing_projection/projected_wall_seconds_at_pool", "min1", "projected wall time at the pool size (min)")
    m.add_derived("CtvBindingGateCount", len(BINDING_GATE_NAMES), "int", "binding integrity gates", "len(gates.campaign)", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("CtvBindingGatesTrue", sum(1 for name in BINDING_GATE_NAMES if gates["campaign"][name] is True), "int", "binding gates recorded true", "count(gates.campaign[*] == true)", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("CtvBindingGateNames", list(BINDING_GATE_NAMES), "list_ident_tt", "names of the binding gates", "sorted keys of gates.campaign", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("CtvReportedNotBindingCount", len(gates["definitions"]["reported_not_binding"]), "int", "quantities reported but not gated", "len(gates.definitions.reported_not_binding)", [{"artifact": "artifacts/gates.json", "pointer": "/definitions/reported_not_binding"}])
    m.add("CtvReportedNotBinding", "artifacts/gates.json", "/definitions/reported_not_binding", "list_clauses", "quantities the protocol reports without gating them")
    m.add_derived("CtvReplayDesigns", len(gates["replays"]), "int", "determinism replay designs", "len(gates.replays)", [{"artifact": "artifacts/gates.json", "pointer": "/replays"}])
    m.add_derived("CtvReplaysBitIdentical", replays_bit_identical, "int", "replays whose canonical topology bytes were bit-identical", "count(gates.replays[*].bit_identical == true)", [{"artifact": "artifacts/gates.json", "pointer": "/replays"}])
    m.add_derived("CtvFailedDesigns", len(failures["failed"]), "int", "designs that failed to resolve", "len(design-failures.failed)", [{"artifact": "artifacts/design-failures.json", "pointer": "/failed"}])
    m.add_derived("CtvRepresentativeCount", representative_count, "int", "representative designs with sampled separatrix paths", "count(designs[*].representative == true)", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/designs"}])

    # ---- definition ----
    field_level = protocol["classification"].split("_")[1].capitalize()
    if field_level != "L1a" or "L1A" not in CLASSIFICATION:
        raise ValueError("field model level differs from the screening classification")
    m.add_derived("CtvFieldModelLevel", field_level, "text", "field model level named by the screening classification", "protocol.classification.split('_')[1].capitalize()", [{"artifact": "artifacts/protocol.json", "pointer": "/classification"}])
    m.add_derived("CtvLiteratureSourceCount", len(definition["literature_basis"]), "int", "literature sources of the frozen definition", "len(protocol.definition_v3.literature_basis)", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3/literature_basis"}])
    m.add_derived("CtvLiteratureKeys", [s["key"] for s in definition["literature_basis"]], "list_ident_tt", "keys of the literature sources", "protocol.definition_v3.literature_basis[*].key", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3/literature_basis"}])
    m.add("CtvReviewDocument", "artifacts/protocol.json", "/definition_v3/review_document", "text", "literature review that fixed the definition")
    m.add_derived("CtvLiteratureCommit", LITERATURE_COMMIT_SHA, "sha_short", "commit of the bound literature review", "git commit at which the review named by protocol.definition_v3.review_document is bound", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3/review_document"}])
    m.add("CtvDefinitionSource", "artifacts/protocol.json", "/claim_boundary/definition_source", "text", "definition source statement")
    m.add("CtvFieldLevelStatement", "artifacts/protocol.json", "/claim_boundary/field_level", "text", "field level statement of the claim boundary")
    m.add("CtvIronSensitivity", "artifacts/protocol.json", "/claim_boundary/iron_sensitivity", "text", "iron sensitivity statement of the claim boundary")
    m.add("CtvUsableAs", "artifacts/protocol.json", "/claim_boundary/usable_as", "list_clauses", "permitted uses of the catalogue")
    m.add("CtvMirrorDescriptorsNotProbabilities", "artifacts/protocol.json", "/claim_boundary/mirror_ratios_are_field_descriptors_not_probabilities", "bool", "mirror ratios are field descriptors, not probabilities")
    m.add("CtvForbidMirrorProbability", "artifacts/protocol.json", "/claim_boundary/forbid_mirror_probability_publication", "bool", "mirror-probability publication forbidden")
    m.add("CtvForbidPlasmaPerformance", "artifacts/protocol.json", "/claim_boundary/forbid_plasma_performance_publication", "bool", "plasma or performance publication forbidden")
    m.add("CtvShakedownNotEvidence", "artifacts/protocol.json", "/claim_boundary/shakedown_outcomes_are_not_evidence", "bool", "shakedown outcomes are not evidence")
    m.add("CtvStabilityToleranceMm", "artifacts/protocol.json", "/definition_v3/stability_tolerance_m", "mm2", "refinement-stability tolerance (mm)")
    m.add("CtvHeldOutToleranceMm", "artifacts/protocol.json", "/definition_v3/held_out_tolerance_m", "mm2", "held-out correspondence tolerance (mm)")
    m.add("CtvBoundaryAmbiguityToleranceMm", "artifacts/protocol.json", "/definition_v3/numerical_parameters/boundary_ambiguity_tolerance_m", "mm2", "boundary-ambiguity tolerance (mm)")
    m.add("CtvAxisBracketTolerance", "artifacts/protocol.json", "/definition_v3/numerical_parameters/axis_root_bracket_tolerance_m", "sci1", "axis-root bisection bracket tolerance (m)")
    m.add("CtvAxisMaxBisections", "artifacts/protocol.json", "/definition_v3/numerical_parameters/axis_root_max_bisections", "int", "maximum bisections per axis root")
    m.add("CtvAxisSamplesPerInterval", "artifacts/protocol.json", "/definition_v3/numerical_parameters/axis_samples_per_interval", "int", "axis samples per grid interval")
    m.add("CtvTraceFluxRootToleranceUm", "artifacts/protocol.json", "/definition_v3/numerical_parameters/trace_flux_root_tolerance_m", "um0", "trace versus flux-root tolerance (um)")
    m.add("CtvTraceStepCellFraction", "artifacts/protocol.json", "/definition_v3/numerical_parameters/trace_step_cell_fraction", "g", "separatrix arc-length step as a fraction of the mesh cell")
    m.add("CtvWallEventMaxHalvings", "artifacts/protocol.json", "/definition_v3/numerical_parameters/wall_event_max_halvings", "int", "maximum wall-event step halvings")
    m.add("CtvWallToleranceM", "artifacts/protocol.json", "/definition_v3/numerical_parameters/wall_tolerance_m", "sci1", "wall snap tolerance (m)")
    m.add("CtvSeedRadiusCellFraction", "artifacts/protocol.json", "/definition_v3/numerical_parameters/seed_radius_cell_fraction", "g", "separatrix seed radius as a fraction of the radial mesh cell")
    m.add("CtvWallSamplesPerCell", "artifacts/protocol.json", "/definition_v3/numerical_parameters/wall_samples_per_cell", "int", "wall samples per cell for the mirror descriptors")
    m.add("CtvRefinementFactor", "artifacts/protocol.json", "/design_sets/sweep_v2/refinement", "int", "refinement factor of the stability map")
    if any(protocol["design_sets"][s]["refinement"] != protocol["design_sets"]["sweep_v2"]["refinement"] for s in ("four_cell_v2", "characterization_v1")):
        raise ValueError("refinement factor differs between the L1a design sets")
    m.add_derived("CtvFluxRootMaxDiff", flux_root_max, "sci1", "largest trace-versus-flux-root difference over every wall-reaching trace (m)", "max over accepted-map wall-reaching traces of abs(flux_root_difference_m)", [{"artifact": "artifacts/designs/p2_divergent_exit/divergent-exit-stack.json", "pointer": "/accepted/separatrix_traces"}])
    m.add_derived("CtvWallTraceCount", wall_trace_count, "int", "wall-reaching separatrix traces on the accepted maps", "count of accepted-map traces with reaches_wall == true (cusps plus outside intersections)", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/estimands/pooled_all/v4_bilinear_difference_m/count"}])
    m.add("CtvVFourBilinearMaxUm", "artifacts/topology-dataset.json", "/estimands/pooled_all/v4_bilinear_difference_m/max", "um1", "largest difference to the bilinear field-line step of the wall-loss campaign's coupling records (um)")
    m.add("CtvVFourBilinearMedianUm", "artifacts/topology-dataset.json", "/estimands/pooled_all/v4_bilinear_difference_m/median", "um1", "median difference to the bilinear field-line step (um)")
    if v4_bilinear_max != dataset["estimands"]["pooled_all"]["v4_bilinear_difference_m"]["max"]:
        raise ValueError("the bilinear-step difference maximum does not recompute from the traces")

    # ---- design sets ----
    m.add("CtvDesignCount", "artifacts/campaign-result.json", "/design_count", "int", "designs screened")
    m.add_derived("CtvDeclaredDesigns", len(plan["design_keys"]), "int", "designs declared by the frozen plan", "len(campaign-plan.design_keys)", [{"artifact": "artifacts/campaign-plan.json", "pointer": "/design_keys"}])
    m.add_derived("CtvSetCount", len(SET_IDS), "int", "design sets", "len(campaign.set_counts)", [{"artifact": "artifacts/campaign-result.json", "pointer": "/set_counts"}])
    for set_id in SET_IDS:
        token = SET_TOKENS[set_id]
        m.add(f"Ctv{token}Count", "artifacts/campaign-result.json", f"/set_counts/{set_id}", "int", f"designs of the {set_id} set")
    m.add_derived("CtvFourCellVersion", _set_version("four_cell_v2"), "text", "version token of the four-cell search whose candidates are screened", "trailing version token of the design-set key four_cell_v2", [{"artifact": "artifacts/protocol.json", "pointer": "/design_sets/four_cell_v2"}])
    m.add_derived("CtvCharacterizationVersion", _set_version("characterization_v1"), "text", "version token of the characterization whose cases are screened", "trailing version token of the design-set key characterization_v1", [{"artifact": "artifacts/protocol.json", "pointer": "/design_sets/characterization_v1"}])
    m.add_derived("CtvSweepVersion", _set_version("sweep_v2"), "text", "version token of the geometry sweep whose designs are screened", "trailing version token of the design-set key sweep_v2", [{"artifact": "artifacts/protocol.json", "pointer": "/design_sets/sweep_v2"}])
    m.add_derived("CtvLineageVersion", _version_token(LINEAGE_EXPERIMENT_ID), "text", "version token of the predecessor campaign", "trailing version token of prior_campaign_disclosure.v3.experiment", [{"artifact": "artifacts/protocol.json", "pointer": "/prior_campaign_disclosure/v3/experiment"}])
    m.add("CtvSweepPreregCommit", "artifacts/authorities.json", "/sealed_sources/sweep_v2/preregistration_commit", "sha_short", "preregistration commit of the sealed sweep")
    m.add("CtvSweepManifestSha", "artifacts/authorities.json", "/sealed_sources/sweep_v2/manifest_file_sha256", "sha_short", "sealed sweep manifest hash prefix")
    m.add("CtvFourCellPreregCommit", "artifacts/authorities.json", "/sealed_sources/four_cell_v2/preregistration_commit", "sha_short", "preregistration commit of the sealed four-cell search")
    m.add("CtvFourCellDatasetSha", "artifacts/authorities.json", "/sealed_sources/four_cell_v2/dataset_file_sha256", "sha_short", "sealed four-cell dataset hash prefix")
    m.add("CtvCharVPreregCommit", "artifacts/authorities.json", "/sealed_sources/characterization_v1/preregistration_commit", "sha_short", "preregistration commit of the sealed characterization")
    m.add("CtvCharVDatasetSha", "artifacts/authorities.json", "/sealed_sources/characterization_v1/dataset_file_sha256", "sha_short", "sealed characterization dataset hash prefix")
    m.add("CtvPTwoVFourProtocolSha", "artifacts/authorities.json", "/sealed_sources/p2_divergent_exit/v4_protocol_file_sha256", "sha_short", "wall-loss campaign protocol hash prefix that binds the P2 field")
    m.add("CtvPTwoPrimaryCheckpointSha", "artifacts/authorities.json", "/sealed_sources/p2_divergent_exit/maps/primary/checkpoint_file_sha256", "sha_short", "P2 level-one checkpoint hash prefix")
    m.add("CtvPTwoRefinedCheckpointSha", "artifacts/authorities.json", "/sealed_sources/p2_divergent_exit/maps/refined/checkpoint_file_sha256", "sha_short", "P2 level-two checkpoint hash prefix")
    m.add("CtvFourCellNote", "artifacts/protocol.json", "/design_sets/four_cell_v2/note", "text", "protocol note on the four-cell candidates' source policy")
    m.add_derived("CtvFourCellStrengthRatioMin", min(v2_ratios), "pct0", "smallest even-stage strength ratio of the four-cell candidates", "min over the sealed v2 dataset of sampling.values.alternating_strength_ratio", [{"artifact": f"reference:{V2_DATASET.as_posix()}", "pointer": "/cases"}])
    m.add_derived("CtvFourCellStrengthRatioMax", max(v2_ratios), "pct0", "largest even-stage strength ratio of the four-cell candidates", "max over the sealed v2 dataset of sampling.values.alternating_strength_ratio", [{"artifact": f"reference:{V2_DATASET.as_posix()}", "pointer": "/cases"}])
    m.add_derived("CtvFourCellReferenceStable", v2_dataset["summary"]["stable_count"], "int", "candidates stable under the frozen four-cell definition (sealed v2 summary)", "sealed v2 dataset summary.stable_count", [{"artifact": f"reference:{V2_DATASET.as_posix()}", "pointer": "/summary/stable_count"}])
    m.add_derived("CtvCharVReferenceEligibleCusps", v1_dataset["summary"]["stable_eligible_cusp_count"], "int", "stable eligible cusps under the frozen characterization definition (sealed v1 summary)", "sealed v1 dataset summary.stable_eligible_cusp_count", [{"artifact": f"reference:{V1_DATASET.as_posix()}", "pointer": "/summary/stable_eligible_cusp_count"}])
    m.add_derived("CtvCharVReferenceEligibleCells", v1_dataset["summary"]["stable_eligible_cell_count"], "int", "stable eligible cells under the frozen characterization definition (sealed v1 summary)", "sealed v1 dataset summary.stable_eligible_cell_count", [{"artifact": f"reference:{V1_DATASET.as_posix()}", "pointer": "/summary/stable_eligible_cell_count"}])
    m.add_derived("CtvCharVStages", sorted(v1_stage_counts), "list_int", "stage counts of the characterization cases", "sorted(set(sealed v1 cases[*].stage_count))", [{"artifact": f"reference:{V1_DATASET.as_posix()}", "pointer": "/cases"}])
    m.add_derived("CtvSweepStages", sorted({len(d["geometry"]["stage_centres_m"]) for d in designs if d["set_id"] == "sweep_v2"}), "list_int", "stage counts of the sweep designs", "sorted(set(len(geometry.stage_centres_m))) over sweep_v2 rows", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/designs"}])
    m.add("CtvPTwoGeometry", "artifacts/protocol.json", "/design_sets/p2_divergent_exit/geometry", "text", "P2 geometry statement of the frozen protocol")
    m.add("CtvPTwoStraightStartMm", "artifacts/topology-dataset.json", f"/designs/{designs.index(p2)}/geometry/straight_z_min_m", "mm1", "start of the regular P2 field domain (mm)")
    m.add("CtvPTwoStraightEndMm", "artifacts/topology-dataset.json", f"/designs/{designs.index(p2)}/geometry/straight_z_max_m", "mm1", "end of the P2 straight dielectric (mm)")
    m.add("CtvPTwoChamberLengthMm", "artifacts/topology-dataset.json", f"/designs/{designs.index(p2)}/geometry/chamber_length_m", "mm1", "P2 chamber length (mm)")
    m.add("CtvPTwoWallRadiusMm", "artifacts/topology-dataset.json", f"/designs/{designs.index(p2)}/geometry/wall_radius_m", "mm1", "P2 wall radius (mm)")
    m.add("CtvPTwoPitchMm", "artifacts/topology-dataset.json", f"/designs/{designs.index(p2)}/geometry/stage_pitch_m", "mm1", "P2 stage pitch (mm)")
    m.add_derived("CtvPTwoStages", len(p2["geometry"]["stage_centres_m"]), "int", "P2 magnet stages", "len(geometry.stage_centres_m) of the P2 row", [{"artifact": "artifacts/topology-dataset.json", "pointer": f"/designs/{designs.index(p2)}/geometry/stage_centres_m"}])
    m.add("CtvPTwoFieldLevel", p2["record_path"], "/evidence/field_level", "text", "field level statement of the P2 record")
    m.add("CtvPTwoRefinementNote", p2["record_path"], "/evidence/refinement_note", "text", "refinement note of the P2 record")
    m.add("CtvPTwoRadialCells", "artifacts/topology-dataset.json", f"/designs/{designs.index(p2)}/grid/radial_cells_across_bore", "g", "radial tracing cells across the P2 bore")
    sweep_rows = [d for d in designs if d["set_id"] == "sweep_v2"]
    m.add_derived("CtvSweepRadialCellsMin", min(d["grid"]["radial_cells_across_bore"] for d in sweep_rows), "fixed1", "fewest radial tracing cells across a sweep bore", "min over sweep_v2 rows of grid.radial_cells_across_bore", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/designs"}])
    m.add_derived("CtvSweepRadialCellsMax", max(d["grid"]["radial_cells_across_bore"] for d in sweep_rows), "fixed1", "most radial tracing cells across a sweep bore", "max over sweep_v2 rows of grid.radial_cells_across_bore", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/designs"}])

    # ---- held-out and stability ----
    design_inputs = [{"artifact": "artifacts/topology-dataset.json", "pointer": "/designs"}]
    m.add("CtvHeldOutCharPassed", "artifacts/topology-dataset.json", "/held_out/characterization_v1/passed_count", "int", "characterization designs whose sealed axis roots were reproduced")
    m.add("CtvHeldOutCharDesigns", "artifacts/topology-dataset.json", "/held_out/characterization_v1/design_count", "int", "characterization designs in the held-out check")
    m.add("CtvHeldOutCharNulls", "artifacts/topology-dataset.json", "/held_out/characterization_v1/observed_null_count", "int", "channel axis nulls matched against the sealed characterization roots")
    m.add("CtvHeldOutCharRefNulls", "artifacts/topology-dataset.json", "/held_out/characterization_v1/reference_null_count", "int", "sealed characterization axis roots in the channel")
    m.add("CtvHeldOutCharMaxUm", "artifacts/topology-dataset.json", "/held_out/characterization_v1/max_difference_m", "um1", "largest characterization held-out difference (um)")
    m.add("CtvHeldOutSweepPassed", "artifacts/topology-dataset.json", "/held_out/sweep_v2/passed_count", "int", "sweep designs whose sealed axis nulls were reproduced")
    m.add("CtvHeldOutSweepDesigns", "artifacts/topology-dataset.json", "/held_out/sweep_v2/design_count", "int", "sweep designs in the held-out check")
    m.add("CtvHeldOutSweepNulls", "artifacts/topology-dataset.json", "/held_out/sweep_v2/observed_null_count", "int", "axis nulls matched against the sealed sweep nulls")
    m.add("CtvHeldOutSweepRefNulls", "artifacts/topology-dataset.json", "/held_out/sweep_v2/reference_null_count", "int", "sealed sweep axis nulls in the window")
    m.add("CtvHeldOutSweepMaxUm", "artifacts/topology-dataset.json", "/held_out/sweep_v2/max_difference_m", "um1", "largest sweep held-out difference (um)")
    m.add("CtvHeldOutToleranceUm", "artifacts/protocol.json", "/definition_v3/held_out_tolerance_m", "um0", "held-out tolerance (um)")
    m.add("CtvStableDesigns", "terminal.json", "/payload/stable_design_count", "int", "designs stable under refinement")
    m.add("CtvMaxWallShiftUm", "artifacts/topology-dataset.json", "/headline/max_wall_intersection_shift_m", "um1", "largest wall-intersection shift under refinement (um)")
    m.add_derived("CtvMaxAxisShiftUm", axis_shift_max, "um1", "largest axis-null shift under refinement (um)", "max over designs of stability.max_axis_null_shift_m", design_inputs)
    m.add("CtvStabilityToleranceUm", "artifacts/protocol.json", "/definition_v3/stability_tolerance_m", "um0", "stability tolerance (um)")
    m.add("CtvMaxWallShiftMm", "artifacts/topology-dataset.json", "/headline/max_wall_intersection_shift_m", "mm3", "largest wall-intersection shift under refinement (mm)")

    # ---- results: histogram and legacy-target fractions ----
    for count in range(8):
        token = COUNT_TOKENS[count]
        m.add_derived(f"CtvHist{token}", headline["wall_cusp_count_histogram"].get(str(count), 0), "int", f"designs with {count} wall cusps (all sets)", f"headline.wall_cusp_count_histogram['{count}'] (0 when absent)", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/headline/wall_cusp_count_histogram"}])
    m.add("CtvHistogramText", "artifacts/topology-dataset.json", "/headline/wall_cusp_count_histogram", "histogram", "wall-cusp count histogram over every design")
    for set_id in SET_IDS:
        token = SET_TOKENS[set_id]
        hist = headline["wall_cusp_count_histogram_by_set"][set_id]
        for count in sorted(int(k) for k in hist):
            m.add_derived(f"Ctv{token}Hist{COUNT_TOKENS[count]}", hist[str(count)], "int", f"{set_id} designs with {count} wall cusps", f"headline.wall_cusp_count_histogram_by_set.{set_id}['{count}']", [{"artifact": "artifacts/topology-dataset.json", "pointer": f"/headline/wall_cusp_count_histogram_by_set/{set_id}"}])
        m.add(f"Ctv{token}FourWallCuspFraction", "artifacts/topology-dataset.json", f"/headline/four_wall_cusp_fraction_by_set/{set_id}", "fixed3", f"four-wall-cusp fraction of the {set_id} set")
        m.add(f"Ctv{token}FourCellFraction", "artifacts/topology-dataset.json", f"/headline/four_cell_fraction_by_set/{set_id}", "fixed3", f"four-cell fraction of the {set_id} set")
        m.add(f"Ctv{token}FourWallCusps", "artifacts/topology-dataset.json", f"/estimands/{set_id}/four_wall_cusp_count", "int", f"{set_id} designs with exactly four wall cusps")
        m.add(f"Ctv{token}FourCells", "artifacts/topology-dataset.json", f"/estimands/{set_id}/four_cell_count", "int", f"{set_id} designs with exactly four cells")
        m.add(f"Ctv{token}WithCusp", "artifacts/topology-dataset.json", f"/estimands/{set_id}/designs_with_at_least_one_cusp", "int", f"{set_id} designs with at least one wall cusp")
        m.add(f"Ctv{token}Stable", "artifacts/topology-dataset.json", f"/estimands/{set_id}/stable_design_count", "int", f"{set_id} designs stable under refinement")
        m.add(f"Ctv{token}Ambiguous", "artifacts/topology-dataset.json", f"/estimands/{set_id}/boundary_ambiguous_cusp_count", "int", f"{set_id} boundary-ambiguous cusps")
        m.add(f"Ctv{token}CuspCount", "artifacts/topology-dataset.json", f"/estimands/{set_id}/z_c_m/count", "int", f"{set_id} wall cusps in all")
        m.add(f"Ctv{token}MaxShiftUm", "artifacts/topology-dataset.json", f"/estimands/{set_id}/max_wall_intersection_shift_m/max", "um1", f"{set_id} largest wall-intersection shift (um)")
    m.add("CtvFourWallCuspsAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/four_wall_cusp_count", "int", "designs with exactly four wall cusps (all sets)")
    m.add("CtvFourCellsAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/four_cell_count", "int", "designs with exactly four cells (all sets)")
    m.add("CtvFourWallCuspFractionAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/four_wall_cusp_fraction", "fixed3", "four-wall-cusp fraction over every design")
    m.add("CtvFourCellFractionAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/four_cell_fraction", "fixed3", "four-cell fraction over every design")
    m.add("CtvWithCuspAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/designs_with_at_least_one_cusp", "int", "designs with at least one wall cusp")
    m.add_derived("CtvWithTwoCuspsAll", sum(1 for d in designs if d["wall_cusp_count"] >= 2), "int", "designs with at least two wall cusps (at least one interior cell)", "count(designs[*].wall_cusp_count >= 2)", design_inputs)
    m.add("CtvCuspCountAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/z_c_m/count", "int", "wall cusps over every design")
    m.add("CtvInteriorCellsAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/interior_cell_length_m/count", "int", "interior cells over every design")
    m.add("CtvInteriorWallMirrorMinAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/interior_wall_mirror_ratio/min", "fixed3", "smallest interior wall mirror ratio")
    m.add("CtvInteriorWallMirrorMaxAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/interior_wall_mirror_ratio/max", "fixed3", "largest interior wall mirror ratio")
    m.add("CtvInteriorAxisMirrorMinAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/interior_axis_mirror_ratio/min", "fixed2", "smallest interior axis mirror ratio")
    m.add("CtvInteriorAxisMirrorMaxAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/interior_axis_mirror_ratio/max", "fixed0", "largest interior axis mirror ratio")
    m.add("CtvAngleMedianAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/angle_to_wall_normal_deg/median", "deg1", "median separatrix angle to the wall normal (deg)")
    m.add("CtvAngleMaxAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/angle_to_wall_normal_deg/max", "deg1", "largest separatrix angle to the wall normal (deg)")
    m.add("CtvAmbiguousAll", "artifacts/topology-dataset.json", "/estimands/pooled_all/boundary_ambiguous_cusp_count", "int", "boundary-ambiguous cusps over every design")

    # ---- results: sweep structure ----
    sweep_est = dataset["estimands"]["sweep_v2"]
    stage_counts = [len(d["geometry"]["stage_centres_m"]) for d in sweep_rows]
    n_minus_one = sum(1 for d, n in zip(sweep_rows, stage_counts) if d["wall_cusp_count"] == n - 1)
    n_minus_two = sum(1 for d, n in zip(sweep_rows, stage_counts) if d["wall_cusp_count"] == n - 2)
    n_plus_one = [d for d, n in zip(sweep_rows, stage_counts) if d["wall_cusp_count"] == n + 1]
    if n_minus_one + n_minus_two + len(n_plus_one) != len(sweep_rows):
        raise ValueError("sweep cusp counts are not N-1, N-2 or N+1 for every design")
    m.add_derived("CtvSweepNMinusOne", n_minus_one, "int", "sweep designs with exactly N-1 wall cusps for N stages", "count over sweep_v2 rows of wall_cusp_count == len(geometry.stage_centres_m) - 1", design_inputs)
    m.add_derived("CtvSweepNMinusTwo", n_minus_two, "int", "sweep designs with N-2 wall cusps", "count over sweep_v2 rows of wall_cusp_count == len(geometry.stage_centres_m) - 2", design_inputs)
    m.add_derived("CtvSweepNPlusOne", len(n_plus_one), "int", "sweep designs with N+1 wall cusps", "count over sweep_v2 rows of wall_cusp_count == len(geometry.stage_centres_m) + 1", design_inputs)
    if len(n_plus_one) != 1:
        raise ValueError("exactly one sweep design was expected to carry N+1 cusps")
    extra = n_plus_one[0]
    extra_ambiguous = [c for c in extra["wall_cusps"] if c["boundary_ambiguous"]]
    m.add_derived("CtvSweepNPlusOneDesign", extra["design_id"], "ident", "the sweep design with N+1 wall cusps", "design_id of the sweep_v2 row with wall_cusp_count == stages + 1", design_inputs)
    m.add_derived("CtvSweepNPlusOneAmbiguous", len(extra_ambiguous), "int", "boundary-ambiguous cusps of that design", "count(wall_cusps[*].boundary_ambiguous) of that row", design_inputs)
    m.add_derived("CtvSweepNPlusOneChannelNulls", extra["channel_axis_null_count"], "int", "channel axis nulls of that design", "channel_axis_null_count of that row", design_inputs)
    m.add_derived("CtvSweepCuspsEqualChannelNulls", sum(1 for d in sweep_rows if d["wall_cusp_count"] == d["channel_axis_null_count"]), "int", "sweep designs whose wall-cusp count equals their channel axis-null count", "count over sweep_v2 rows of wall_cusp_count == channel_axis_null_count", design_inputs)
    m.add_derived("CtvSweepNMinusTwoEndNullOutside", sum(1 for d, n in zip(sweep_rows, stage_counts) if d["wall_cusp_count"] == n - 2 and d["channel_axis_null_count"] == n - 2), "int", "N-2 designs whose channel axis-null count is also N-2", "count over sweep_v2 rows with wall_cusp_count == N-2 of channel_axis_null_count == N-2", design_inputs)
    m.add("CtvSweepGapMedianMm", "artifacts/topology-dataset.json", "/estimands/sweep_v2/distance_to_nearest_stage_gap_m/median", "mm2", "median cusp distance to the nearest inter-magnet gap centre (mm)")
    m.add("CtvSweepGapMaxMm", "artifacts/topology-dataset.json", "/estimands/sweep_v2/distance_to_nearest_stage_gap_m/max", "mm2", "largest cusp distance to the nearest inter-magnet gap centre (mm)")
    m.add("CtvSweepCentreMinMm", "artifacts/topology-dataset.json", "/estimands/sweep_v2/distance_to_nearest_stage_centre_m/min", "mm2", "smallest cusp distance to a stage centre (mm)")
    m.add("CtvSweepCentreMedianMm", "artifacts/topology-dataset.json", "/estimands/sweep_v2/distance_to_nearest_stage_centre_m/median", "mm2", "median cusp distance to the nearest stage centre (mm)")
    m.add("CtvSweepInteriorCells", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_cell_length_m/count", "int", "interior cells of the sweep designs")
    m.add("CtvSweepInteriorLengthPitchMin", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_cell_length_over_pitch/min", "fixed2", "shortest interior cell in pitches")
    m.add("CtvSweepInteriorLengthPitchMax", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_cell_length_over_pitch/max", "fixed2", "longest interior cell in pitches")
    m.add("CtvSweepInteriorLengthPitchMedian", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_cell_length_over_pitch/median", "fixed2", "median interior cell length in pitches")
    m.add("CtvSweepInteriorWallMirrorMin", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_wall_mirror_ratio/min", "fixed3", "smallest interior wall mirror ratio of the sweep")
    m.add("CtvSweepInteriorWallMirrorMax", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_wall_mirror_ratio/max", "fixed3", "largest interior wall mirror ratio of the sweep")
    m.add("CtvSweepInteriorAxisMirrorMin", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_axis_mirror_ratio/min", "fixed2", "smallest interior axis mirror ratio of the sweep")
    m.add("CtvSweepInteriorAxisMirrorMax", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_axis_mirror_ratio/max", "fixed2", "largest interior axis mirror ratio of the sweep")
    m.add("CtvSweepInteriorAxisMirrorMedian", "artifacts/topology-dataset.json", "/estimands/sweep_v2/interior_axis_mirror_ratio/median", "fixed2", "median interior axis mirror ratio of the sweep")
    m.add("CtvSweepAllWallMirrorMax", "artifacts/topology-dataset.json", "/estimands/sweep_v2/all_cells_wall_mirror_ratio/max", "fixed2", "largest wall mirror ratio over every sweep cell incl. partials")
    m.add("CtvSweepAllAxisMirrorMax", "artifacts/topology-dataset.json", "/estimands/sweep_v2/all_cells_axis_mirror_ratio/max", "fixed2", "largest axis mirror ratio over every sweep cell incl. partials")
    m.add("CtvSweepAngleMedianDeg", "artifacts/topology-dataset.json", "/estimands/sweep_v2/angle_to_wall_normal_deg/median", "deg1", "median separatrix angle to the wall normal in the sweep (deg)")
    m.add("CtvSweepAngleMaxDeg", "artifacts/topology-dataset.json", "/estimands/sweep_v2/angle_to_wall_normal_deg/max", "deg1", "largest separatrix angle to the wall normal in the sweep (deg)")
    m.add_derived("CtvSweepAxisNullMin", min(d["axis_null_count"] for d in sweep_rows), "int", "fewest axis nulls in a sweep window", "min over sweep_v2 rows of axis_null_count", design_inputs)
    m.add_derived("CtvSweepAxisNullMax", max(d["axis_null_count"] for d in sweep_rows), "int", "most axis nulls in a sweep window", "max over sweep_v2 rows of axis_null_count", design_inputs)
    m.add_derived("CtvSweepChannelNullMin", min(d["channel_axis_null_count"] for d in sweep_rows), "int", "fewest channel axis nulls in a sweep design", "min over sweep_v2 rows of channel_axis_null_count", design_inputs)
    m.add_derived("CtvSweepChannelNullMax", max(d["channel_axis_null_count"] for d in sweep_rows), "int", "most channel axis nulls in a sweep design", "max over sweep_v2 rows of channel_axis_null_count", design_inputs)
    m.add("CtvSweepOutsideAnode", "artifacts/topology-dataset.json", "/estimands/sweep_v2/outside_intersection_zones/anode_side", "int", "sweep separatrix intersections on the anode side of the straight section")
    m.add("CtvSweepOutsideDivergent", "artifacts/topology-dataset.json", "/estimands/sweep_v2/outside_intersection_zones/divergent_exit", "int", "sweep separatrix intersections in a divergent exit")
    m.add("CtvSweepOutsideStray", "artifacts/topology-dataset.json", "/estimands/sweep_v2/outside_intersection_zones/downstream_stray_field", "int", "sweep separatrix intersections in the downstream stray field")
    m.add("CtvSweepZcOverLengthMin", "artifacts/topology-dataset.json", "/estimands/sweep_v2/z_c_over_length/min", "fixed3", "smallest sweep cusp position as a fraction of the chamber length")
    m.add("CtvSweepZcOverLengthMax", "artifacts/topology-dataset.json", "/estimands/sweep_v2/z_c_over_length/max", "fixed3", "largest sweep cusp position as a fraction of the chamber length")
    m.add("CtvSweepWallBMinT", "artifacts/topology-dataset.json", "/estimands/sweep_v2/wall_b_at_cusp_t/min", "fixed3", "smallest wall field at a sweep cusp (T)")
    m.add("CtvSweepWallBMaxT", "artifacts/topology-dataset.json", "/estimands/sweep_v2/wall_b_at_cusp_t/max", "fixed3", "largest wall field at a sweep cusp (T)")
    m.add_derived("CtvSweepAxisToWallMaxMm", max(abs(sweep_est["axis_to_wall_shift_m"]["min"]), abs(sweep_est["axis_to_wall_shift_m"]["max"])), "mm2", "largest axial offset between a sweep cusp and its generating axis null (mm)", "max(|min|, |max|) of estimands.sweep_v2.axis_to_wall_shift_m", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/estimands/sweep_v2/axis_to_wall_shift_m"}])
    m.add("CtvSweepMaxAxisShiftUm", "artifacts/topology-dataset.json", "/estimands/sweep_v2/max_axis_null_shift_m/max", "um1", "largest sweep axis-null shift under refinement (um)")
    stage_rows: list[str] = []
    for stages in sorted(set(stage_counts)):
        token = STAGE_TOKENS[stages]
        sub = [d for d, n in zip(sweep_rows, stage_counts) if n == stages]
        hist = _histogram([d["wall_cusp_count"] for d in sub])
        inter = [c for d in sub for c in d["cells"] if c["kind"] == "interior"]
        gaps = [c["distance_to_nearest_stage_gap_m"] for d in sub for c in d["wall_cusps"]]
        angles = [c["angle_to_wall_normal_deg"] for d in sub for c in d["wall_cusps"]]
        pitches = [d["geometry"]["stage_pitch_m"] for d in sub]
        prefix = f"CtvSweepStage{token}"
        m.add_derived(f"{prefix}Designs", len(sub), "int", f"sweep designs with {stages} stages", f"count over sweep_v2 rows of len(geometry.stage_centres_m) == {stages}", design_inputs)
        m.add_derived(f"{prefix}NMinusOne", sum(1 for d in sub if d["wall_cusp_count"] == stages - 1), "int", f"{stages}-stage designs with exactly {stages - 1} wall cusps", f"count of those rows with wall_cusp_count == {stages - 1}", design_inputs)
        for count in sorted(int(k) for k in hist):
            m.add_derived(f"{prefix}Hist{COUNT_TOKENS[count]}", hist[str(count)], "int", f"{stages}-stage designs with {count} wall cusps", f"histogram of wall_cusp_count over the {stages}-stage rows", design_inputs)
        m.add_derived(f"{prefix}FourCusps", sum(d["four_wall_cusps"] for d in sub), "int", f"{stages}-stage designs with exactly four wall cusps", f"count of four_wall_cusps over the {stages}-stage rows", design_inputs)
        m.add_derived(f"{prefix}FourCells", sum(d["four_cells"] for d in sub), "int", f"{stages}-stage designs with exactly four cells", f"count of four_cells over the {stages}-stage rows", design_inputs)
        m.add_derived(f"{prefix}InteriorCells", len(inter), "int", f"interior cells of the {stages}-stage designs", f"count of interior cells over the {stages}-stage rows", design_inputs)
        m.add_derived(f"{prefix}LengthPitchMin", min(c["length_over_pitch"] for c in inter), "fixed2", f"shortest interior cell of the {stages}-stage designs in pitches", "min of cells[kind == interior].length_over_pitch", design_inputs)
        m.add_derived(f"{prefix}LengthPitchMax", max(c["length_over_pitch"] for c in inter), "fixed2", f"longest interior cell of the {stages}-stage designs in pitches", "max of cells[kind == interior].length_over_pitch", design_inputs)
        m.add_derived(f"{prefix}WallMirrorMin", min(c["wall_mirror_ratio"] for c in inter), "fixed3", f"smallest interior wall mirror ratio of the {stages}-stage designs", "min of cells[kind == interior].wall_mirror_ratio", design_inputs)
        m.add_derived(f"{prefix}WallMirrorMax", max(c["wall_mirror_ratio"] for c in inter), "fixed3", f"largest interior wall mirror ratio of the {stages}-stage designs", "max of cells[kind == interior].wall_mirror_ratio", design_inputs)
        m.add_derived(f"{prefix}AxisMirrorMin", min(c["axis_mirror_ratio"] for c in inter), "fixed2", f"smallest interior axis mirror ratio of the {stages}-stage designs", "min of cells[kind == interior].axis_mirror_ratio", design_inputs)
        m.add_derived(f"{prefix}AxisMirrorMax", max(c["axis_mirror_ratio"] for c in inter), "fixed2", f"largest interior axis mirror ratio of the {stages}-stage designs", "max of cells[kind == interior].axis_mirror_ratio", design_inputs)
        m.add_derived(f"{prefix}GapMedianMm", statistics.median(gaps), "mm2", f"median cusp distance to the nearest gap centre of the {stages}-stage designs (mm)", "median of wall_cusps[*].distance_to_nearest_stage_gap_m", design_inputs)
        m.add_derived(f"{prefix}GapMaxMm", max(gaps), "mm2", f"largest cusp distance to the nearest gap centre of the {stages}-stage designs (mm)", "max of wall_cusps[*].distance_to_nearest_stage_gap_m", design_inputs)
        m.add_derived(f"{prefix}AngleMedianDeg", statistics.median(angles), "deg1", f"median separatrix angle of the {stages}-stage designs (deg)", "median of wall_cusps[*].angle_to_wall_normal_deg", design_inputs)
        m.add_derived(f"{prefix}PitchMinMm", min(pitches), "mm2", f"smallest stage pitch of the {stages}-stage designs (mm)", "min of geometry.stage_pitch_m", design_inputs)
        m.add_derived(f"{prefix}PitchMaxMm", max(pitches), "mm2", f"largest stage pitch of the {stages}-stage designs (mm)", "max of geometry.stage_pitch_m", design_inputs)
        stage_rows.append(
            f"{stages} & {len(sub)} & {_histogram_text(hist)} & {sum(1 for d in sub if d['wall_cusp_count'] == stages - 1)} & {sum(d['four_wall_cusps'] for d in sub)} & {sum(d['four_cells'] for d in sub)} & "
            f"{len(inter)} & {_range([c['length_over_pitch'] for c in inter], 'fixed2')} & {_range([c['wall_mirror_ratio'] for c in inter], 'fixed3')} & {_range([c['axis_mirror_ratio'] for c in inter], 'fixed2')} & "
            f"{format_value('mm2', statistics.median(gaps))} / {format_value('mm2', max(gaps))} & {format_value('deg1', statistics.median(angles))}\\\\"
        )
    sweep_interior = [c for d in sweep_rows for c in d["cells"] if c["kind"] == "interior"]
    stage_rows.append(
        f"\\midrule\nall & {len(sweep_rows)} & {_histogram_text(sweep_est['wall_cusp_count_histogram'])} & {n_minus_one} & {sweep_est['four_wall_cusp_count']} & {sweep_est['four_cell_count']} & "
        f"{len(sweep_interior)} & {_range([c['length_over_pitch'] for c in sweep_interior], 'fixed2')} & {_range([c['wall_mirror_ratio'] for c in sweep_interior], 'fixed3')} & {_range([c['axis_mirror_ratio'] for c in sweep_interior], 'fixed2')} & "
        f"{format_value('mm2', sweep_est['distance_to_nearest_stage_gap_m']['median'])} / {format_value('mm2', sweep_est['distance_to_nearest_stage_gap_m']['max'])} & {format_value('deg1', sweep_est['angle_to_wall_normal_deg']['median'])}\\\\"
    )

    # ---- results: four-cell candidates and characterization cases ----
    four_cell_rows = [d for d in designs if d["set_id"] == "four_cell_v2"]
    m.add_derived("CtvFourCellOneCusp", sum(1 for d in four_cell_rows if d["wall_cusp_count"] == 1), "int", "four-cell candidates with exactly one wall cusp", "count over four_cell_v2 rows of wall_cusp_count == 1", design_inputs)
    m.add_derived("CtvFourCellOneAxisNull", sum(1 for d in four_cell_rows if d["axis_null_count"] == 1), "int", "four-cell candidates with exactly one axis null in the window", "count over four_cell_v2 rows of axis_null_count == 1", design_inputs)
    m.add_derived("CtvFourCellChannelNullZero", sum(1 for d in four_cell_rows if d["channel_axis_null_count"] == 0), "int", "four-cell candidates whose single null lies just outside the channel", "count over four_cell_v2 rows of channel_axis_null_count == 0", design_inputs)
    m.add_derived("CtvFourCellChannelNullOne", sum(1 for d in four_cell_rows if d["channel_axis_null_count"] == 1), "int", "four-cell candidates whose single null lies inside the channel", "count over four_cell_v2 rows of channel_axis_null_count == 1", design_inputs)
    m.add("CtvFourCellZcOverLengthMin", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/z_c_over_length/min", "fixed2", "smallest four-cell cusp position as a fraction of the chamber length")
    m.add("CtvFourCellZcOverLengthMax", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/z_c_over_length/max", "fixed2", "largest four-cell cusp position as a fraction of the chamber length")
    m.add("CtvFourCellAngleMinDeg", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/angle_to_wall_normal_deg/min", "deg1", "smallest four-cell separatrix angle to the wall normal (deg)")
    m.add("CtvFourCellAngleMaxDeg", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/angle_to_wall_normal_deg/max", "deg1", "largest four-cell separatrix angle to the wall normal (deg)")
    m.add("CtvFourCellGapMedianMm", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/distance_to_nearest_stage_gap_m/median", "mm2", "median four-cell cusp distance to the nearest gap centre (mm)")
    m.add("CtvFourCellCentreMedianMm", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/distance_to_nearest_stage_centre_m/median", "mm2", "median four-cell cusp distance to the nearest stage centre (mm)")
    m.add("CtvFourCellAllWallMirrorMin", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/all_cells_wall_mirror_ratio/min", "fixed2", "smallest wall mirror ratio over the four-cell partial cells")
    m.add("CtvFourCellAllWallMirrorMax", "artifacts/topology-dataset.json", "/estimands/four_cell_v2/all_cells_wall_mirror_ratio/max", "fixed0", "largest wall mirror ratio over the four-cell partial cells")
    m.add_derived("CtvFourCellInteriorCells", dataset["estimands"]["four_cell_v2"]["interior_cell_length_m"]["count"], "int", "interior cells of the four-cell candidates", "estimands.four_cell_v2.interior_cell_length_m.count", [{"artifact": "artifacts/topology-dataset.json", "pointer": "/estimands/four_cell_v2/interior_cell_length_m"}])
    char_rows = [d for d in designs if d["set_id"] == "characterization_v1"]
    m.add_derived("CtvCharVNMinusOne", sum(1 for d in char_rows if d["wall_cusp_count"] == len(d["geometry"]["stage_centres_m"]) - 1), "int", "characterization cases with exactly N-1 wall cusps for N stages", "count over characterization_v1 rows of wall_cusp_count == len(geometry.stage_centres_m) - 1", design_inputs)
    m.add_derived("CtvCharVCuspsEqualChannelNulls", sum(1 for d in char_rows if d["wall_cusp_count"] == d["channel_axis_null_count"]), "int", "characterization cases whose wall-cusp count equals their channel axis-null count", "count over characterization_v1 rows of wall_cusp_count == channel_axis_null_count", design_inputs)
    m.add("CtvCharVInteriorCells", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_cell_length_m/count", "int", "interior cells of the characterization cases")
    m.add("CtvCharVInteriorWallMirrorMin", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_wall_mirror_ratio/min", "fixed3", "smallest interior wall mirror ratio of the characterization cases")
    m.add("CtvCharVInteriorWallMirrorMax", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_wall_mirror_ratio/max", "fixed3", "largest interior wall mirror ratio of the characterization cases")
    m.add("CtvCharVInteriorAxisMirrorMin", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_axis_mirror_ratio/min", "fixed1", "smallest interior axis mirror ratio of the characterization cases")
    m.add("CtvCharVInteriorAxisMirrorMax", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_axis_mirror_ratio/max", "fixed0", "largest interior axis mirror ratio of the characterization cases")
    m.add("CtvCharVGapMedianMm", "artifacts/topology-dataset.json", "/estimands/characterization_v1/distance_to_nearest_stage_gap_m/median", "mm2", "median characterization cusp distance to the nearest gap centre (mm)")
    m.add("CtvCharVGapMaxMm", "artifacts/topology-dataset.json", "/estimands/characterization_v1/distance_to_nearest_stage_gap_m/max", "mm2", "largest characterization cusp distance to the nearest gap centre (mm)")
    m.add("CtvCharVInteriorLengthPitchMin", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_cell_length_over_pitch/min", "fixed2", "shortest characterization interior cell in pitches")
    m.add("CtvCharVInteriorLengthPitchMax", "artifacts/topology-dataset.json", "/estimands/characterization_v1/interior_cell_length_over_pitch/max", "fixed2", "longest characterization interior cell in pitches")
    m.add("CtvCharVAngleMedianDeg", "artifacts/topology-dataset.json", "/estimands/characterization_v1/angle_to_wall_normal_deg/median", "deg1", "median characterization separatrix angle (deg)")
    m.add("CtvCharVOutsideStray", "artifacts/topology-dataset.json", "/estimands/characterization_v1/outside_intersection_zones/downstream_stray_field", "int", "characterization separatrix intersections in the downstream stray field")
    v1_inputs = [{"artifact": f"reference:{V1_DATASET.as_posix()}", "pointer": "/cases"}]
    m.add_derived("CtvVOneChannelRoots", v1_channel_roots, "int", "sealed characterization vector roots inside the plasma channel (primary maps, non-boundary)", "count over sealed v1 cases of primary roots with zone in the channel zones and finite_box_boundary == false", v1_inputs)
    m.add_derived("CtvVOneChannelAxisRoots", v1_channel_axis, "int", "of which clusters containing an axis-detected member", "count of those roots with a member of method axis_sign_change or axis_grid", v1_inputs)
    m.add_derived("CtvVOneChannelOffAxisRoots", len(v1_off_axis), "int", "of which off-axis bilinear roots", "count of those roots without an axis-detected member", v1_inputs)
    m.add_derived("CtvVOneOffAxisCases", len({r["case_id"] for r in v1_off_axis}), "int", "characterization cases carrying an off-axis in-channel root", "count of distinct case_id over the off-axis in-channel roots", v1_inputs)
    m.add_derived("CtvVOneOffAxisRadiusFractionMin", min(r["r_over_wall"] for r in v1_off_axis), "fixed2", "smallest radius of an off-axis in-channel root as a fraction of the wall radius", "min of root.r_m / case.chamber_radius_m over the off-axis in-channel roots", v1_inputs)
    m.add_derived("CtvVOneOffAxisRadiusFractionMax", max(r["r_over_wall"] for r in v1_off_axis), "fixed2", "largest radius of an off-axis in-channel root as a fraction of the wall radius", "max of root.r_m / case.chamber_radius_m over the off-axis in-channel roots", v1_inputs)
    m.add_derived("CtvVOneOffAxisExclusion", v1_exclusions[0], "ident", "exclusion reason recorded for every off-axis in-channel root", "the single distinct exclusion_reason over the off-axis in-channel roots", v1_inputs)
    m.add_derived("CtvVOneOffAxisEligible", sum(1 for r in v1_off_axis if r["eligible_cusp"]), "int", "off-axis in-channel roots that were eligible cusps", "count(eligible_cusp == true) over the off-axis in-channel roots", v1_inputs)

    # ---- results: P2 row ----
    p2_index = designs.index(p2)
    p2_pointer = f"/designs/{p2_index}"
    m.add("CtvPTwoCusps", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusp_count", "int", "wall cusps of the P2 row")
    m.add("CtvPTwoCells", "artifacts/topology-dataset.json", f"{p2_pointer}/cell_count", "int", "cells of the P2 row")
    m.add("CtvPTwoAxisNulls", "artifacts/topology-dataset.json", f"{p2_pointer}/axis_null_count", "int", "axis nulls of the P2 row")
    m.add("CtvPTwoDesignId", "artifacts/topology-dataset.json", f"{p2_pointer}/design_id", "ident", "design identifier of the P2 row")
    p2_table_rows: list[str] = []
    for index, (cusp, entry, token) in enumerate(zip(p2["wall_cusps"], p2c["cusps"], CUSP_TOKENS)):
        prefix = f"CtvPTwoCusp{token}"
        m.add(f"{prefix}Mm", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusps/{index}/z_c_m", "mm3", f"P2 wall cusp {index + 1} position (mm)")
        m.add(f"{prefix}NullMm", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusps/{index}/axis_null_z_m", "mm3", f"P2 axis null {index + 1} position (mm)")
        m.add(f"{prefix}PicPlaneMm", "artifacts/topology-dataset.json", f"/p2_consistency/cusps/{index}/nearest_pic_axis_null_plane_m", "mm2", f"kinetic-workstream axis-null plane nearest P2 cusp {index + 1} (mm)")
        m.add_derived(f"{prefix}NullToPicUm", abs(entry["difference_axis_null_to_pic_plane_m"]), "um1", f"|axis null - kinetic plane| for P2 cusp {index + 1} (um)", "abs(p2_consistency.cusps[i].difference_axis_null_to_pic_plane_m)", [{"artifact": "artifacts/topology-dataset.json", "pointer": f"/p2_consistency/cusps/{index}/difference_axis_null_to_pic_plane_m"}])
        m.add(f"{prefix}DashboardMm", "artifacts/topology-dataset.json", f"/p2_consistency/cusps/{index}/nearest_dashboard_wall_abs_br_maximum_m", "mm2", f"topology-dashboard wall |B_r| maximum nearest P2 cusp {index + 1} (mm)")
        m.add_derived(f"{prefix}ToDashboardMm", abs(entry["difference_to_dashboard_maximum_m"]), "mm2", f"|z_c - dashboard maximum| for P2 cusp {index + 1} (mm)", "abs(p2_consistency.cusps[i].difference_to_dashboard_maximum_m)", [{"artifact": "artifacts/topology-dataset.json", "pointer": f"/p2_consistency/cusps/{index}/difference_to_dashboard_maximum_m"}])
        m.add(f"{prefix}WallBT", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusps/{index}/wall_b_t", "fixed3", f"wall field at P2 cusp {index + 1} (T)")
        m.add(f"{prefix}AngleDeg", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusps/{index}/angle_to_wall_normal_deg", "deg2", f"separatrix angle to the wall normal at P2 cusp {index + 1} (deg)")
        m.add(f"{prefix}Ambiguous", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusps/{index}/boundary_ambiguous", "bool", f"boundary-ambiguity flag of P2 cusp {index + 1}")
        m.add(f"{prefix}GapUm", "artifacts/topology-dataset.json", f"{p2_pointer}/wall_cusps/{index}/distance_to_nearest_stage_gap_m", "um1", f"distance of P2 cusp {index + 1} to the nearest gap centre (um)")
        p2_table_rows.append(
            f"{index + 1} & {format_value('mm3', cusp['z_c_m'])} & {format_value('mm3', cusp['axis_null_z_m'])} & {format_value('mm2', entry['nearest_pic_axis_null_plane_m'])} & "
            f"{format_value('um1', abs(entry['difference_axis_null_to_pic_plane_m']))} & {format_value('mm2', entry['nearest_dashboard_wall_abs_br_maximum_m'])} & "
            f"{format_value('mm2', abs(entry['difference_to_dashboard_maximum_m']))} & {format_value('um1', cusp['distance_to_nearest_stage_gap_m'])} & {cusp['wall_b_t']:.3f} & "
            f"{cusp['angle_to_wall_normal_deg']:.2f} & {'yes' if cusp['boundary_ambiguous'] else 'no'}\\\\"
        )
    m.add_derived("CtvPTwoCuspPositionsMm", [c["z_c_m"] for c in p2["wall_cusps"]], "list_mm3", "P2 wall cusp positions (mm)", "wall_cusps[*].z_c_m of the P2 row", [{"artifact": "artifacts/topology-dataset.json", "pointer": f"{p2_pointer}/wall_cusps"}])
    m.add("CtvPTwoNullToPicMaxUm", "artifacts/topology-dataset.json", "/p2_consistency/max_abs_difference_axis_null_to_pic_plane_m", "um0", "largest |axis null - kinetic plane| over the P2 cusps (um)")
    m.add("CtvPTwoToDashboardMaxMm", "artifacts/topology-dataset.json", "/p2_consistency/max_abs_difference_to_dashboard_maximum_m", "mm2", "largest |z_c - dashboard maximum| over the P2 cusps (mm)")
    m.add("CtvPTwoCountEqualsReference", "artifacts/topology-dataset.json", "/p2_consistency/cusp_count_equals_reference_count", "bool", "P2 cusp count equals the count of both reference sets")
    m.add("CtvPTwoConsistencyRole", "artifacts/topology-dataset.json", "/p2_consistency/role", "text", "role of the P2 consistency check")
    m.add("CtvPTwoPicPlanesSource", "artifacts/topology-dataset.json", "/p2_consistency/references/pic_axis_null_planes_source", "text", "source of the kinetic-workstream axis-null planes")
    m.add("CtvPTwoDashboardSource", "artifacts/topology-dataset.json", "/p2_consistency/references/topology_dashboard_wall_abs_br_maxima_source", "text", "source of the topology-dashboard wall maxima")
    m.add("CtvPTwoPicPlanesMm", "artifacts/topology-dataset.json", "/p2_consistency/references/pic_axis_null_planes_m", "list_mm2", "kinetic-workstream axis-null planes (mm)")
    m.add("CtvPTwoDashboardMaximaMm", "artifacts/topology-dataset.json", "/p2_consistency/references/topology_dashboard_wall_abs_br_maxima_m", "list_mm2", "topology-dashboard wall maxima (mm)")
    m.add_derived("CtvPTwoThirdCuspInsideEndUm", p2["geometry"]["straight_z_max_m"] - p2["wall_cusps"][-1]["z_c_m"], "um1", "distance of the last P2 cusp inside the straight end (um)", "geometry.straight_z_max_m - wall_cusps[-1].z_c_m of the P2 row", [{"artifact": "artifacts/topology-dataset.json", "pointer": f"{p2_pointer}/wall_cusps"}])
    m.add("CtvPTwoAmbiguousCusps", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/boundary_ambiguous_cusp_count", "int", "boundary-ambiguous cusps of the P2 row")
    m.add("CtvPTwoInteriorWallMirrorMax", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/interior_wall_mirror_ratio/max", "fixed3", "largest interior wall mirror ratio of the P2 row")
    m.add("CtvPTwoInteriorAxisMirrorMin", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/interior_axis_mirror_ratio/min", "fixed3", "smallest interior axis mirror ratio of the P2 row")
    m.add("CtvPTwoInteriorAxisMirrorMax", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/interior_axis_mirror_ratio/max", "fixed3", "largest interior axis mirror ratio of the P2 row")
    m.add("CtvPTwoInteriorLengthPitchMin", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/interior_cell_length_over_pitch/min", "fixed3", "shortest P2 interior cell in pitches")
    m.add("CtvPTwoExitPartialLengthUm", "artifacts/topology-dataset.json", f"{p2_pointer}/cells/{len(p2['cells']) - 1}/length_m", "um1", "length of the P2 exit partial cell (um)")
    m.add("CtvPTwoExitPartialAxisMirror", "artifacts/topology-dataset.json", f"{p2_pointer}/cells/{len(p2['cells']) - 1}/axis_mirror_ratio", "fixed1", "axis mirror ratio of the P2 exit partial cell")
    m.add("CtvPTwoMaxShiftM", "artifacts/topology-dataset.json", f"{p2_pointer}/stability/max_wall_intersection_shift_m", "sci1", "largest P2 wall-intersection shift between the level-one and level-two maps (m)")
    m.add("CtvPTwoWallBMinT", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/wall_b_at_cusp_t/min", "fixed3", "smallest wall field at a P2 cusp (T)")
    m.add("CtvPTwoWallBMaxT", "artifacts/topology-dataset.json", "/estimands/p2_divergent_exit/wall_b_at_cusp_t/max", "fixed3", "largest wall field at a P2 cusp (T)")
    if p2["cells"][-1]["kind"] != "exit_partial":
        raise ValueError("the last P2 cell is not the exit partial cell")

    # ---- lineage: the recorded v3 rejection ----
    lineage_inputs = [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/gates.json').as_posix()}", "pointer": "/failing_designs"}]
    m.add("CtvLineageExperimentId", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/experiment", "ident", "predecessor experiment path as disclosed by the frozen protocol")
    m.add("CtvLineagePreregCommit", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/preregistration_commit", "sha_short", "predecessor preregistration commit prefix")
    m.add("CtvLineageResultsCommit", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/result_commit", "sha_short", "predecessor result commit prefix")
    m.add_derived("CtvLineageAuditCommit", LINEAGE_AUDIT_COMMIT_SHA, "sha_short", "predecessor post-hoc audit commit prefix", "git commit at which POSTHOC_AUDIT.md and audit_held_out.py are bound", [{"artifact": f"lineage:{LINEAGE_AUDIT.as_posix()}", "pointer": ""}])
    m.add("CtvLineageTerminalState", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/terminal_state", "ident", "predecessor terminal state")
    m.add("CtvLineageCampaignStatus", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/status", "ident", "predecessor campaign status")
    m.add("CtvLineageFailingGate", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/failing_gate", "ident", "predecessor failing gate")
    m.add("CtvLineageFailingDesigns", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/failing_design_count", "int", "predecessor failing designs")
    m.add("CtvLineageEveryOtherGate", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/every_other_gate", "text", "predecessor outcome of every other gate as disclosed")
    m.add("CtvLineageWhatChanged", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/what_v3_1_changes", "text", "what the corrected campaign changed")
    m.add("CtvLineageShakedownLesson", "artifacts/protocol.json", "/prior_campaign_disclosure/v3/shakedown_lesson", "text", "shakedown lesson recorded in the frozen protocol")
    m.add("CtvLineageRelation", "artifacts/protocol.json", "/relation_to_v3", "text", "relation to the predecessor as recorded in the frozen protocol")
    m.add("CtvRelationToPriorNulls", "artifacts/protocol.json", "/relation_to_prior_nulls", "text", "relation to the frozen-definition nulls as recorded in the frozen protocol")
    m.add("CtvPriorPaperDisclosure", "artifacts/protocol.json", "/prior_campaign_disclosure/paper", "text", "paper scope disclosed by the frozen protocol")
    m.add_derived("CtvLineageManifestSha", lineage.manifest_sha256, "sha_short", "predecessor results manifest SHA-256 prefix", "sha256(lineage results/manifest.json)", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'manifest.json').as_posix()}", "pointer": ""}])
    m.add_derived("CtvLineageVerifiedFiles", len(lineage.hashes), "int_comma", "predecessor bundle files verified byte-for-byte", "count of lineage manifest file entries whose sha256 and size equal the checkout", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'manifest.json').as_posix()}", "pointer": "/artifacts"}])
    m.add_derived("CtvLineageDesigns", lineage_campaign["design_count"], "int", "predecessor designs resolved", "lineage campaign-result.design_count", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/design_count"}])
    m.add_derived("CtvLineageStable", lineage_campaign["headline"]["stable_design_count"], "int", "predecessor designs stable under refinement", "lineage campaign-result.headline.stable_design_count", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/headline/stable_design_count"}])
    m.add_derived("CtvLineageGatesTrue", sum(1 for name in BINDING_GATE_NAMES if lineage_gates["campaign"][name] is True), "int", "predecessor binding gates recorded true", "count(lineage gates.campaign[*] == true)", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/gates.json').as_posix()}", "pointer": "/campaign"}])
    m.add_derived("CtvLineageHistogramEqual", lineage_campaign["headline"]["wall_cusp_count_histogram"] == headline["wall_cusp_count_histogram"], "bool", "predecessor wall-cusp histogram equals the accepted campaign's", "lineage headline.wall_cusp_count_histogram == headline.wall_cusp_count_histogram", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/headline/wall_cusp_count_histogram"}])
    m.add_derived("CtvLineageHeldOutCharPassed", lineage_campaign["headline"]["held_out"]["characterization_v1"]["passed_count"], "int", "predecessor characterization designs passing the held-out check", "lineage headline.held_out.characterization_v1.passed_count", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/headline/held_out/characterization_v1/passed_count"}])
    m.add_derived("CtvLineageHeldOutCharRefNulls", lineage_campaign["headline"]["held_out"]["characterization_v1"]["reference_null_count"], "int", "predecessor characterization reference nulls kept by the recorded filter", "lineage headline.held_out.characterization_v1.reference_null_count", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/headline/held_out/characterization_v1/reference_null_count"}])
    m.add_derived("CtvLineageHeldOutCharObsNulls", lineage_campaign["headline"]["held_out"]["characterization_v1"]["observed_null_count"], "int", "predecessor characterization channel nulls observed", "lineage headline.held_out.characterization_v1.observed_null_count", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/headline/held_out/characterization_v1/observed_null_count"}])
    m.add_derived("CtvLineageHeldOutSweepPassed", lineage_campaign["headline"]["held_out"]["sweep_v2"]["passed_count"], "int", "predecessor sweep designs passing the held-out check", "lineage headline.held_out.sweep_v2.passed_count", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/headline/held_out/sweep_v2/passed_count"}])
    m.add_derived("CtvLineageStageWallMin", lineage_campaign["execution_mode"]["stage_wall_s"], "min1", "predecessor design-stage wall time (min)", "lineage campaign-result.execution_mode.stage_wall_s", [{"artifact": f"lineage:{(LINEAGE_RESULTS / 'artifacts/campaign-result.json').as_posix()}", "pointer": "/execution_mode/stage_wall_s"}])
    m.add_derived("CtvLineageSealedAxisClusters", audit["sealed_axis_clusters"], "int", "sealed characterization axis clusters (member-method filter)", "count over sealed v1 cases of non-boundary primary roots with an axis-detected member", v1_inputs)
    m.add_derived("CtvLineageDroppedClusters", audit["dropped_by_recorded_filter"], "int", "of which dropped by the recorded centroid filter", "count of those clusters with centroid r_m != 0", v1_inputs)
    m.add_derived("CtvLineageDroppedInChannel", audit["dropped_in_channel"], "int", "of which inside the channel", "count of the dropped clusters with zone in the channel zones", v1_inputs)
    m.add_derived("CtvLineageMaxDroppedCentroidR", audit["max_dropped_centroid_r_m"], "sci1", "largest centroid radius of a dropped cluster (m)", "max abs(r_m) over the dropped clusters", v1_inputs)
    m.add_derived("CtvLineageCorrectedPassed", audit["corrected_filter_pass_count"], "int", "characterization designs in bijection under the intended filter", "count of sealed v1 cases whose channel axis clusters are in bijection with the lineage record's channel nulls within tolerance, all X", lineage_inputs + v1_inputs)
    m.add_derived("CtvLineageCorrectedMaxUm", audit["corrected_filter_max_difference_m"], "um1", "largest matched difference under the intended filter (um)", "max matched difference over the sealed v1 cases", lineage_inputs + v1_inputs)
    m.add_derived("CtvLineageFailuresExplained", audit["failures_explained_by_dropped_clusters"], "bool", "every recorded failure is exactly a design with a dropped in-channel cluster", "recorded held_out.passed == (no dropped in-channel cluster) and unmatched observed count == dropped in-channel count for every case", lineage_inputs + v1_inputs)
    m.add_derived("CtvLineageAuditDropped", documented["dropped"], "int", "dropped clusters as documented by the post-hoc audit", f"regex {AUDIT_ROOT_CAUSE_PATTERN.pattern!r} group 1 over POSTHOC_AUDIT.md", [{"artifact": f"lineage:{LINEAGE_AUDIT.as_posix()}", "pointer": ""}])
    m.add_derived("CtvLineageAuditClusters", documented["clusters"], "int", "sealed clusters as documented by the post-hoc audit", f"regex {AUDIT_ROOT_CAUSE_PATTERN.pattern!r} group 2 over POSTHOC_AUDIT.md", [{"artifact": f"lineage:{LINEAGE_AUDIT.as_posix()}", "pointer": ""}])
    m.add_derived("CtvLineageAuditCorrectedPassed", documented["corrected_passed"], "int", "designs in bijection under the intended filter as documented by the post-hoc audit", f"regex {AUDIT_CORRECTED_PATTERN.pattern!r} group 1 over POSTHOC_AUDIT.md", [{"artifact": f"lineage:{LINEAGE_AUDIT.as_posix()}", "pointer": ""}])
    m.add_derived("CtvLineageAuditCorrectedMaxUm", documented["corrected_max_difference_m"], "um1", "largest matched difference as documented by the post-hoc audit (um)", f"regex {AUDIT_CORRECTED_PATTERN.pattern!r} group 3 over POSTHOC_AUDIT.md", [{"artifact": f"lineage:{LINEAGE_AUDIT.as_posix()}", "pointer": ""}])

    # ---- claim-boundary flags ----
    m.add_derived("CtvPhysicsLevelOpened", False, "bool", "a physics level is opened", "constant of the admission: the numerical-screening gate declares opens_level null", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary"}])
    m.add_derived("CtvConfinementCellsDemonstrated", False, "bool", "plasma confinement cells are demonstrated", "constant of the admission: cells are geometric field descriptors (claim_boundary.mirror_ratios_are_field_descriptors_not_probabilities) and no plasma quantity is computed", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/mirror_ratios_are_field_descriptors_not_probabilities"}])
    m.add_derived("CtvFrozenDefinitionNullsRemainTrue", True, "bool", "the frozen-definition nulls of the earlier admissions remain true", "constant of the admission restating protocol.relation_to_prior_nulls; the sealed v1/v2 summaries bound here still record their nulls", [{"artifact": "artifacts/protocol.json", "pointer": "/relation_to_prior_nulls"}])
    m.add_derived("CtvIronSensitivityTested", False, "bool", "iron sensitivity tested", "constant of the admission restating protocol.claim_boundary.iron_sensitivity", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/iron_sensitivity"}])
    m.add_derived("CtvHardwareValidation", False, "bool", "hardware or experimental validation claimed", "constant of the admission: no measurement enters the campaign", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}])

    # ---- tables ----
    tex_lines = [
        "% Generated by paper/scripts/generate_cusp_topology_v3_1_evidence.py; do not hand edit.",
        f"% Evidence: {EXPERIMENT.as_posix()} at commit {RESULTS_COMMIT_SHA} (results manifest SHA-256 {bundle.manifest_sha256}).",
        f"% Every macro value traces to an artifact path and JSON pointer recorded in {EVIDENCE_PATH.as_posix()}.",
    ]
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    histogram_rows: list[str] = []
    for count in range(8):
        cells = [str(headline["wall_cusp_count_histogram_by_set"][s].get(str(count), 0)) for s in SET_IDS]
        histogram_rows.append(f"{count} & {' & '.join(cells)} & {headline['wall_cusp_count_histogram'].get(str(count), 0)}\\\\")
    histogram_rows.append("\\midrule\n" + "designs & " + " & ".join(str(headline["set_counts"][s]) for s in SET_IDS) + f" & {len(designs)}\\\\")
    histogram_rows.append("stable under refinement & " + " & ".join(str(dataset["estimands"][s]["stable_design_count"]) for s in SET_IDS) + f" & {headline['stable_design_count']}\\\\")
    histogram_rows.append("with at least one cusp & " + " & ".join(str(dataset["estimands"][s]["designs_with_at_least_one_cusp"]) for s in SET_IDS) + f" & {dataset['estimands']['pooled_all']['designs_with_at_least_one_cusp']}\\\\")
    histogram_rows.append("exactly four wall cusps & " + " & ".join(str(dataset["estimands"][s]["four_wall_cusp_count"]) for s in SET_IDS) + f" & {dataset['estimands']['pooled_all']['four_wall_cusp_count']}\\\\")
    histogram_rows.append("exactly four cells & " + " & ".join(str(dataset["estimands"][s]["four_cell_count"]) for s in SET_IDS) + f" & {dataset['estimands']['pooled_all']['four_cell_count']}\\\\")
    histogram_rows.append("boundary-ambiguous cusps & " + " & ".join(str(dataset["estimands"][s]["boundary_ambiguous_cusp_count"]) for s in SET_IDS) + f" & {dataset['estimands']['pooled_all']['boundary_ambiguous_cusp_count']}\\\\")
    tex_lines += _table(
        "CtvHistogramTable",
        "Wall-cusp count per design set under the literature definition, as sealed in "
        "\\texttt{topology-dataset.json} (\\texttt{headline} and \\texttt{estimands}); a wall cusp is the intersection "
        "of an axis null's separatrix with the straight dielectric, and every count is a property of the prescribed "
        "field map, not of a plasma. The geometry-sweep, four-cell-candidate and characterization sets are "
        "linear-vacuum screening fields; the P2 row is the qualified finite-element field.",
        "tab:cusp-topology-v3-1-histogram", f"{_p(3.1)}rrrrr",
        "wall cusps & sweep & four-cell cand. & charact. & P2 & all\\\\", histogram_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    tex_lines += _table(
        "CtvSweepStageTable",
        "Geometry-sweep designs by magnet-stage count $N$: cusp-count histogram, designs with exactly $N-1$ wall "
        "cusps, designs meeting the two legacy targets (four wall cusps; four cells), interior cells with their "
        "length in stage pitches and their wall and axis mirror ratios (field ratios, not probabilities), the "
        "cusp distance to the nearest inter-magnet gap centre (median / max) and the median separatrix angle "
        "to the wall normal. Every value is sealed in \\texttt{topology-dataset.json}.",
        "tab:cusp-topology-v3-1-sweep-stages",
        f"rr{_p(1.6)}rrrr{_p(1.35)}{_p(1.55)}{_p(1.35)}{_p(1.45)}r",
        "$N$ & designs & cusps & $N-1$ & \\shortstack{four\\\\cusps} & \\shortstack{four\\\\cells} & \\shortstack{interior\\\\cells} & \\shortstack{length\\\\(pitch)} & \\shortstack{wall\\\\mirror} & \\shortstack{axis\\\\mirror} & \\shortstack{gap dist.\\\\(mm)} & angle ($^\\circ$)\\\\",
        stage_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{2pt}",
    )
    tex_lines += _table(
        "CtvPTwoTable",
        "The three wall cusps of the P2 divergent-exit-stack row against two recorded development references "
        "carried by the frozen protocol as a reported, ungated consistency check: the axis-null planes of the "
        "kinetic workstream's steady-state dashboard (no kinetic result is admitted in this paper) and the wall "
        "$|B_r|$ maxima of the plasma-topology dashboard (a display diagnostic, not accepted cusp evidence). "
        "Columns: cusp position $z_c$, generating axis null $z_k$, nearest kinetic plane and $|z_k - $plane$|$, "
        "nearest dashboard maximum and $|z_c - $max$|$, distance to the nearest inter-magnet gap centre, wall "
        "field, separatrix angle to the wall normal, boundary-ambiguity flag.",
        "tab:cusp-topology-v3-1-p2", "rrrrrrrrrrl",
        "cusp & $z_c$ (mm) & $z_k$ (mm) & plane (mm) & $\\Delta$ ($\\mu$m) & max (mm) & $\\Delta$ (mm) & gap ($\\mu$m) & $|B|$ (T) & angle ($^\\circ$) & amb.\\\\",
        p2_table_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    lineage_rows = [
        f"experiment & \\texttt{{{_ident(LINEAGE_EXPERIMENT_ID)}}} & \\texttt{{{_ident(EXPERIMENT_ID)}}}\\\\",
        f"preregistration commit & \\texttt{{{LINEAGE_PREREGISTRATION_COMMIT_SHA[:12]}}} & \\texttt{{{PREREGISTRATION_COMMIT_SHA[:12]}}}\\\\",
        f"result commit & \\texttt{{{LINEAGE_RESULTS_COMMIT_SHA[:12]}}} & \\texttt{{{RESULTS_COMMIT_SHA[:12]}}}\\\\",
        f"terminal state & \\texttt{{{_ident(lineage_terminal['state'])}}} & \\texttt{{{_ident(terminal['state'])}}}\\\\",
        f"campaign status & \\texttt{{{_ident(lineage_campaign['status'])}}} & \\texttt{{{_ident(campaign['status'])}}}\\\\",
        f"designs resolved / stable & {lineage_campaign['design_count']} / {lineage_campaign['headline']['stable_design_count']} & {campaign['design_count']} / {headline['stable_design_count']}\\\\",
        f"binding gates true & {sum(1 for n in BINDING_GATE_NAMES if lineage_gates['campaign'][n] is True)} of {len(BINDING_GATE_NAMES)} & {sum(1 for n in BINDING_GATE_NAMES if gates['campaign'][n] is True)} of {len(BINDING_GATE_NAMES)}\\\\",
        f"failing gate (designs) & \\texttt{{{_ident(LINEAGE_FAILING_GATE)}}} ({len(lineage_failing[LINEAGE_FAILING_GATE])}) & none\\\\",
        f"held-out, characterization (passed / designs; reference nulls) & {lineage_campaign['headline']['held_out']['characterization_v1']['passed_count']} / {lineage_campaign['headline']['held_out']['characterization_v1']['design_count']}; {lineage_campaign['headline']['held_out']['characterization_v1']['reference_null_count']} & {headline['held_out']['characterization_v1']['passed_count']} / {headline['held_out']['characterization_v1']['design_count']}; {headline['held_out']['characterization_v1']['reference_null_count']}\\\\",
        f"held-out, sweep (passed / designs; reference nulls) & {lineage_campaign['headline']['held_out']['sweep_v2']['passed_count']} / {lineage_campaign['headline']['held_out']['sweep_v2']['design_count']}; {lineage_campaign['headline']['held_out']['sweep_v2']['reference_null_count']} & {headline['held_out']['sweep_v2']['passed_count']} / {headline['held_out']['sweep_v2']['design_count']}; {headline['held_out']['sweep_v2']['reference_null_count']}\\\\",
        f"wall-cusp histogram & {_histogram_text(lineage_campaign['headline']['wall_cusp_count_histogram'])} & {_histogram_text(headline['wall_cusp_count_histogram'])}\\\\",
        "sealed reference axis root & cluster centroid $r = 0$ exactly & cluster with an axis-detected member\\\\",
        f"sealed axis clusters kept & {audit['sealed_axis_clusters'] - audit['dropped_by_recorded_filter']} of {audit['sealed_axis_clusters']} ({audit['dropped_in_channel']} in-channel dropped) & {audit['sealed_axis_clusters']} of {audit['sealed_axis_clusters']}\\\\",
        f"design stage wall time (min) & {lineage_campaign['execution_mode']['stage_wall_s'] / 60.0:.1f} & {campaign['execution_mode']['stage_wall_s'] / 60.0:.1f}\\\\",
        f"post-hoc audit & \\texttt{{{LINEAGE_AUDIT_COMMIT_SHA[:12]}}} (read-only) & not needed\\\\",
        "cited for any number & no & yes\\\\",
    ]
    tex_lines += _table(
        "CtvLineageTable",
        "Lineage of the admitted campaign: the predecessor's single execution ended in a recorded "
        "\\texttt{assessment\\_rejection} because its held-out reference kept only sealed characterization axis "
        "clusters whose centroid radius was exactly zero (a recording-layer defect of the reference extraction, "
        "not a property of the topology), and the corrected re-preregistration changed that extraction and "
        "nothing else. Every predecessor value is read from its immutable bundle or its read-only post-hoc "
        "audit and is disclosed, not cited.",
        "tab:cusp-topology-v3-1-lineage", f"{_p(4.3)}{_p(4.6)}{_p(4.6)}",
        "quantity & predecessor (recorded rejection) & admitted campaign\\\\", lineage_rows,
        size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    tex = "\n".join(tex_lines) + "\n"

    lineage_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (lineage_audit_file, lineage_script_file, lineage_prereg_file)
    }
    if load_json_bytes((repo / LINEAGE_EXPERIMENT / "protocol.json").read_bytes(), "lineage protocol") != lineage_protocol:
        raise ValueError("the predecessor's frozen protocol differs from the sealed copy in its bundle")
    for relative, meta in sorted(lineage.used.items()):
        path = (LINEAGE_RESULTS / relative).as_posix()
        lineage_files[path] = {"sha256": meta["sha256"], "bytes": meta["bytes"], "revision": LINEAGE_RESULTS_COMMIT_SHA, "role": "lineage-rejected-campaign"}
    reference_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (v1_file, v2_file, sweep_file)
    }
    evidence = {
        "document_type": "paper-cusp-topology-v3-1-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "p2_row_classification": P2_CLASSIFICATION,
        "recorded_outcome": RECORDED_OUTCOME,
        "campaign_status": CAMPAIGN_STATUS,
        "screening_model": SCREENING_MODEL,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
        "definition_sources": {
            "revision": LITERATURE_COMMIT_SHA,
            "files": {literature_file["path"]: {k: literature_file[k] for k in ("sha256", "bytes", "revision", "role", "git_blob", "git_blob_sha256")}},
            "literature_keys": [s["key"] for s in definition["literature_basis"]],
            "rule": "the frozen definition names the literature review at its commit; every literature locator of the definition appears in the bound review, whose checkout equals the blob (LF-normalised)",
        },
        "lineage_artifacts": {
            "rule": "hash-bound lineage records of the recorded assessment_rejection and its read-only post-hoc audit; disclosed as lineage, never cited for a number; the generator byte-verifies the whole rejected bundle and reproduces the audit's counts from the sealed characterization dataset",
            "files": lineage_files,
            "audit": audit,
            "documented": documented,
        },
        "reference_artifacts": {
            "rule": "the sealed datasets and manifest the campaign compared against (held-out references and the frozen-definition results), bound at their own admitted revisions; each must hash to the sealed-source identity recorded in the bundle",
            "files": reference_files,
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
                "Every number in the section is a macro defined here; each macro is bound below to an artifact path, "
                "JSON pointer, formatter and SHA-256, or to a stated derivation over such inputs. Claim-bearing "
                "sentences are exact EvidenceClaim bodies registered in paper/evidence/claims.json; the "
                "numerical-screening gate in paper/evidence/result-gates.json names the typed manifest that admits "
                "the section at its recorded outcome (an accepted topology screening) without opening any physics "
                "level. Cusps and cells are geometric properties of prescribed field maps under the literature "
                "definition; mirror ratios are field ratios and never probabilities; nothing here is a plasma, "
                "confinement, wall-loss or performance claim."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size",
            "recomputation_rule": "the headline and every per-set estimand (except the bilinear-step comparison, whose maximum is recomputed from the traces) are re-derived from the per-design rows and must equal the sealed values exactly; every design record, field grid, catalogue entry and CSV row must agree with its row",
        },
        "lineage_bundle": {
            "manifest_path": (LINEAGE_RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": lineage.manifest_sha256,
            "artifact_count": lineage.manifest["artifact_count"],
            "verified_file_count": len(lineage.hashes),
            "state": LINEAGE_TERMINAL_STATE,
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": {
            "CtvHistogramTable": {"rows": len(histogram_rows), "source": "artifacts/topology-dataset.json#/headline, #/estimands"},
            "CtvSweepStageTable": {"rows": len(stage_rows), "source": "artifacts/topology-dataset.json#/designs (sweep_v2 rows), #/estimands/sweep_v2"},
            "CtvPTwoTable": {"rows": len(p2_table_rows), "source": "artifacts/topology-dataset.json#/designs (P2 row), #/p2_consistency"},
            "CtvLineageTable": {"rows": len(lineage_rows), "source": "lineage bundle campaign-result.json, gates.json, terminal.json; sealed characterization v1 dataset; POSTHOC_AUDIT.md"},
        },
        "generator": {
            "path": "paper/scripts/generate_cusp_topology_v3_1_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_cusp_topology_v3_1_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_cusp_topology_v3_1_evidence.py",
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
        "lineage_inputs": [
            {"path": path, "sha256": meta["sha256"], "bytes": meta["bytes"], "revision": meta["revision"]}
            for path, meta in evidence["lineage_artifacts"]["files"].items()
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
        print(f"cusp topology v3.1 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
