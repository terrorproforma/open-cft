"""Generate hash-bound paper evidence for the MDO L0 campaign v2.

Reads the sealed results bundle of ``modern/experiments/mdo_l0_campaign_v2``
(verified byte-for-byte against ``results/manifest.json``; no end-of-line
tolerance is needed or granted), binds it to the committed results revision,
re-verifies the sealed 96-design catalogue against the screening dataset it
was drawn from (bytes, Git blob at the screening record commit, manifest
entry) and against the Jeffreys and Wilson rules it declares, verifies the
prior campaign's bundle (``mdo_l0_campaign_v1``) for the v1-versus-v2
comparison, reads the prior campaign's post-hoc audit for the disclosures the
protocol says it closed, cross-checks the committed results dashboard against
both bundles, and writes:

* ``paper/evidence/mdo-l0-v2.json`` -- every macro value with the artifact
  path, JSON pointer, formatter and artifact SHA-256 it was read from (and the
  bundle it belongs to), or the derivation and inputs of a derived macro;
* ``paper/generated/mdo-l0-v2.tex`` -- ``\\newcommand`` macros (prefix
  ``Mdb``) and four generated tables (each wrapped in ``\\ArtifactClaim``) for
  the admitted results subsection ``paper/sections/mdo-l0-v2.tex``;
* ``paper/generated/mdo-l0-v2.provenance.json`` -- generator/input/output
  hashes in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The campaign is optimiser evidence about the L0
conservation model over a discrete catalogue of screened designs under the
declared closure CL-1, which identifies a collisionless test-particle wall-hit
probability with a per-cusp survival factor; it is not thruster performance,
and every number below is conditional on that declared identification.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
from typing import Any, Callable

import generate_mdo_l0_v1_evidence as mdo_l0_v1

EXPERIMENT = Path("modern/experiments/mdo_l0_campaign_v2")
RESULTS = EXPERIMENT / "results"
V1_EXPERIMENT = Path("modern/experiments/mdo_l0_campaign_v1")
V1_RESULTS = V1_EXPERIMENT / "results"
V1_AUDIT_PATH = V1_EXPERIMENT / "POSTHOC_AUDIT.md"
SCREENING_EXPERIMENT = Path("modern/experiments/orbit_wall_loss_geometry_screening_v1")
SCREENING_DATASET_PATH = SCREENING_EXPERIMENT / "results/artifacts/geometry-wall-loss-dataset.json"
SCREENING_MANIFEST_PATH = SCREENING_EXPERIMENT / "results/manifest.json"
EVIDENCE_PATH = Path("paper/evidence/mdo-l0-v2.json")
OUTPUT_PATH = Path("paper/generated/mdo-l0-v2.tex")
SIDECAR_PATH = Path("paper/generated/mdo-l0-v2.provenance.json")
SECTION_PATH = Path("paper/sections/mdo-l0-v2.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_mdo_l0_campaign_v2_dashboard.py")
DASHBOARD_HTML = Path("modern/visualization/mdo-l0-campaign-v2.html")

RESULTS_COMMIT_SHA = "a003f766c330d4e5648844ba49cdf1c3a3ce3bc1"
PREREGISTRATION_COMMIT_SHA = "99914dc2fdbe88d18ab11ca86acad634129b4e08"
DASHBOARD_COMMIT_SHA = "0ea33a7e25275e70271bc9deb18a9361b80b94e5"
EXPECTED_MANIFEST_SHA256 = "ca3b58ce21eedb8ef094a3d73894b508fe8c438183fa02620d05f759541f7b1f"
V1_RESULTS_COMMIT_SHA = mdo_l0_v1.RESULTS_COMMIT_SHA
V1_MANIFEST_SHA256 = "2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381"
V1_AUDIT_COMMIT_SHA = "e9f9af165a932a9f13438b297950efa951aff5c3"
SCREENING_RESULTS_COMMIT_SHA = "ab7c28977963822b2ad6eac451d2bafef5185e6c"

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "MDO-L0-V2-20260903-1440-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/mdo-l0-v2.json")
GATE_ID = "GATE-MDO-L0-V2"
ARTIFACT_ID = "TAB-MDO-L0-V2"
ARTIFACT_CLAIM_ID = "CLM-055"
PROSE_CLAIM_IDS = ("CLM-053", "CLM-054", "CLM-056", "CLM-057", "CLM-058", "CLM-059", "CLM-060")
SECTION_BINDING = "\\input{sections/mdo-l0-v2.tex}"
GENERATED_BINDING = "\\input{generated/mdo-l0-v2.tex}"
SECTION_HEADING = (
    "Catalogue optimisation over the screened sweep designs under the declared test-particle wall-loss closure"
)
TABLE_MACROS = ("MdbHvTable", "MdbCatalogueTable", "MdbClosureTable", "MdbComparisonTable")
MACRO_PREFIX = "Mdb"

EXPERIMENT_ID = "mdo-l0-campaign-v2"
V1_EXPERIMENT_ID = "mdo-l0-campaign-v1"
CLASSIFICATION = (
    "l0_model_optimisation_over_screened_design_catalogue_with_test_particle_wall_loss_closure_not_thruster_performance"
)
CLOSURE_ID = "CL-1-multiplicative-cusp-survival-per-cell-test-particle-wall-loss"
SENSITIVITY_CLOSURE_ID = "CL-2-pooled-test-particle-wall-loss-survival"
SCREENING_CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
AUDIT_DISCLOSURES = ("F9", "F10", "F22", "F26", "F27", "F28")
DISCLOSURE_TOKENS = {"F9": "Nine", "F10": "Ten", "F22": "TwentyTwo", "F26": "TwentySix", "F27": "TwentySeven", "F28": "TwentyEight"}
DENSE_NEGLIGIBLE_HYPERVOLUME = 1e-9
WILSON_Z = 1.959963984540054

STRATEGIES = mdo_l0_v1.STRATEGIES
STRATEGY_TOKENS = mdo_l0_v1.STRATEGY_TOKENS
STRATEGY_LABELS = mdo_l0_v1.STRATEGY_LABELS
SEED_TOKENS = mdo_l0_v1.SEED_TOKENS
OBJECTIVES = mdo_l0_v1.OBJECTIVES
WIDTH_TOKENS: dict[Any, str] = {0.25: "Quarter", 1.0: "Campaign", 4.0: "Four", "point": "Point"}
DESIGN_TOKENS = {46: "FortySix", 49: "FortyNine", 50: "Fifty", 73: "SeventyThree", 94: "NinetyFour"}
CELL_TOKENS = ("One", "Two", "Three", "Four")
RUN_ARTIFACTS = tuple(f"artifacts/runs/{s}-{seed}.json" for seed in (101, 202, 303) for s in STRATEGIES)
V1_COMPARISON_ARTIFACTS = (
    "artifacts/campaign-result.json",
    "artifacts/campaign-plan.json",
    "artifacts/metrics.json",
    "artifacts/gates.json",
    "artifacts/dense-reference-summary.json",
    "artifacts/pooled-fronts.json",
    "artifacts/protocol.json",
    "artifacts/sensitivity.json",
)
AUDIT_PATTERN = re.compile(r"Six disclosures are recorded below \((F\d+(?:, F\d+)*)\)")
TIE_PATTERN = re.compile(r"relative tolerance ([0-9.e+-]+)\)")

# --------------------------------------------------------------------------- #
# Formatting (shared with the v1 generator; extended for millimetres)
# --------------------------------------------------------------------------- #
FORMATTERS: dict[str, Callable[[Any], str]] = {
    **mdo_l0_v1.FORMATTERS,
    "fixed5": lambda v: f"{float(v):.5f}",
    "sci3": lambda v: mdo_l0_v1._sci(float(v), 3),
    "mm1": lambda v: f"{1e3 * float(v):.1f}",
    "mm2": lambda v: f"{1e3 * float(v):.2f}",
    "yesno": lambda v: "yes" if v is True else "no" if v is False else mdo_l0_v1._tex_escape(str(v)),
}
sha256_bytes = mdo_l0_v1.sha256_bytes
canonical_json = mdo_l0_v1.canonical_json
load_json_bytes = mdo_l0_v1.load_json_bytes
resolve_pointer = mdo_l0_v1.resolve_pointer
dashboard_payload = mdo_l0_v1.dashboard_payload
_git = mdo_l0_v1._git
_lf = mdo_l0_v1._lf
_seconds = mdo_l0_v1._seconds


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def wilson(successes: int, trials: int, z: float = WILSON_Z) -> tuple[float, float]:
    """The orbit_mc Wilson interval, re-implemented operation for operation."""

    if isinstance(successes, bool) or isinstance(trials, bool) or trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts are invalid")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def jeffreys_mean(successes: int, trials: int) -> float:
    return (successes + 0.5) / (trials + 1)


# --------------------------------------------------------------------------- #
# Bundle verification (either campaign)
# --------------------------------------------------------------------------- #
class Bundle:
    """A sealed results bundle, verified file by file against its own manifest."""

    def __init__(self, repo: Path, results: Path, experiment_id: str) -> None:
        self.repo = repo
        self.results = results
        self.root = repo / results
        manifest_raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.manifest = load_json_bytes(manifest_raw, f"{results.as_posix()}/manifest.json")
        if self.manifest.get("state") != "accepted_result":
            raise ValueError(f"{experiment_id}: results manifest state is not accepted_result")
        if self.manifest.get("experiment_id") != experiment_id:
            raise ValueError(f"{experiment_id}: results manifest experiment identity differs")
        self.hashes: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        entries = self.manifest["artifacts"]
        if len(entries) != self.manifest["artifact_count"]:
            raise ValueError(f"{experiment_id}: results manifest artifact count differs")
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
                raise ValueError(f"{experiment_id}: bundle file SHA-256 or size mismatch: {relative}")
            self.hashes[relative] = actual
            self.sizes[relative] = len(raw)
        if self.hashes["terminal.json"] != self.manifest["terminal_byte_sha256"]:
            raise ValueError(f"{experiment_id}: terminal.json hash differs from the manifest binding")
        if self.hashes["execution-lock.json"] != self.manifest["lock_byte_sha256"]:
            raise ValueError(f"{experiment_id}: execution-lock.json hash differs from the manifest binding")
        for relative in list(self.hashes):
            if relative.endswith(".sha256.json") or relative == "execution-lock.json":
                continue
            sidecar_rel = f"{relative}.sha256.json"
            if sidecar_rel not in self.hashes:
                raise ValueError(f"{experiment_id}: artifact without manifest-bound sidecar: {relative}")
            sidecar = load_json_bytes((self.root / sidecar_rel).read_bytes(), sidecar_rel)
            if sidecar["artifact"] != relative or sidecar["byte_sha256"] != self.hashes[relative]:
                raise ValueError(f"{experiment_id}: sidecar disagrees with the manifest: {sidecar_rel}")
            if sidecar["bytes"] != self.sizes[relative]:
                raise ValueError(f"{experiment_id}: sidecar size disagrees with the manifest: {sidecar_rel}")
        self.used: dict[str, dict[str, Any]] = {}
        self.docs: dict[str, Any] = {}

    def load(self, relative: str) -> Any:
        if relative not in self.hashes:
            raise ValueError(f"{relative} is not manifest-bound")
        if relative not in self.docs:
            raw = (self.root / relative).read_bytes()
            self.used[relative] = {"sha256": self.hashes[relative], "bytes": self.sizes[relative]}
            self.docs[relative] = load_json_bytes(raw, relative)
        return self.docs[relative]

    def committed_blob(self, commit: str) -> str:
        manifest_rel = (self.results / "manifest.json").as_posix()
        committed = _git(self.repo, "rev-parse", f"{commit}:{manifest_rel}")
        working = _git(self.repo, "hash-object", "--", manifest_rel)
        if committed != working:
            raise ValueError(f"working-tree results manifest differs from the blob committed at {commit[:8]}")
        return committed


def _is_ancestor(repo: Path, commit: str, head: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, head], cwd=repo, check=False, capture_output=True,
    ).returncode == 0


def _commit_files(repo: Path, commit: str) -> list[str]:
    out = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit)
    return [line for line in out.splitlines() if line]


def bind_committed(repo: Path, bundle: Bundle, v1: Bundle) -> dict[str, Any]:
    """Prove the working-tree bundles equal the committed revisions and record the commit facts."""

    head = _git(repo, "rev-parse", "HEAD")
    for commit, label in (
        (RESULTS_COMMIT_SHA, "results"),
        (PREREGISTRATION_COMMIT_SHA, "preregistration"),
        (DASHBOARD_COMMIT_SHA, "dashboard"),
        (V1_RESULTS_COMMIT_SHA, "v1 results"),
        (V1_AUDIT_COMMIT_SHA, "v1 audit"),
        (SCREENING_RESULTS_COMMIT_SHA, "screening results"),
    ):
        if not _is_ancestor(repo, commit, head):
            raise ValueError(f"{label} commit is not an ancestor of HEAD")
    if not _is_ancestor(repo, PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA) or PREREGISTRATION_COMMIT_SHA == RESULTS_COMMIT_SHA:
        raise ValueError("preregistration does not strictly precede the results commit")
    if not _is_ancestor(repo, RESULTS_COMMIT_SHA, DASHBOARD_COMMIT_SHA):
        raise ValueError("the dashboard commit does not descend from the results commit")
    if not _is_ancestor(repo, V1_AUDIT_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA):
        raise ValueError("the v1 audit does not precede the v2 preregistration")
    if not _is_ancestor(repo, SCREENING_RESULTS_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA):
        raise ValueError("the screening record does not precede the v2 preregistration")
    if bundle.manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("results manifest SHA-256 differs from the admitted identity")
    if v1.manifest_sha256 != V1_MANIFEST_SHA256:
        raise ValueError("v1 results manifest SHA-256 differs from the admitted identity")
    committed_blob = bundle.committed_blob(RESULTS_COMMIT_SHA)
    v1_blob = v1.committed_blob(V1_RESULTS_COMMIT_SHA)
    results_tree = _git(repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{RESULTS.as_posix()}")
    head_tree = _git(repo, "rev-parse", f"HEAD:{RESULTS.as_posix()}")
    if results_tree != head_tree:
        raise ValueError("results tree changed after the results commit")
    result_files = _commit_files(repo, RESULTS_COMMIT_SHA)
    outside = [path for path in result_files if not path.startswith(RESULTS.as_posix() + "/")]
    prereg_files = _commit_files(repo, PREREGISTRATION_COMMIT_SHA)
    prereg_outside = [path for path in prereg_files if not path.startswith(EXPERIMENT.as_posix() + "/")]
    prereg_results = [path for path in prereg_files if path.startswith(RESULTS.as_posix() + "/")]
    if prereg_outside or prereg_results or not prereg_files:
        raise ValueError("the preregistration commit is not experiment-path isolated or touches results")
    audit_blob = _git(repo, "rev-parse", f"{V1_AUDIT_COMMIT_SHA}:{V1_AUDIT_PATH.as_posix()}")
    audit_head = _git(repo, "rev-parse", f"HEAD:{V1_AUDIT_PATH.as_posix()}")
    audit_working = _git(repo, "hash-object", "--", V1_AUDIT_PATH.as_posix())
    if audit_blob != audit_head or audit_blob != audit_working:
        raise ValueError("the v1 post-hoc audit differs from the blob committed at the audit revision")
    return {
        "results_commit": RESULTS_COMMIT_SHA,
        "results_commit_subject": _git(repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA),
        "results_tree": results_tree,
        "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
        "dashboard_commit": DASHBOARD_COMMIT_SHA,
        "manifest_git_blob": committed_blob,
        "manifest_path": (RESULTS / "manifest.json").as_posix(),
        "result_commit_file_count": len(result_files),
        "result_commit_files_outside_results": outside,
        "preregistration_commit_file_count": len(prereg_files),
        "v1_results_commit": V1_RESULTS_COMMIT_SHA,
        "v1_manifest_git_blob": v1_blob,
        "v1_manifest_path": (V1_RESULTS / "manifest.json").as_posix(),
        "v1_audit_commit": V1_AUDIT_COMMIT_SHA,
        "v1_audit_path": V1_AUDIT_PATH.as_posix(),
        "v1_audit_git_blob": audit_blob,
        "screening_results_commit": SCREENING_RESULTS_COMMIT_SHA,
    }


def verify_catalogue_binding(repo: Path, binding: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """The screening dataset behind the catalogue must be the bytes and blob the protocol declares."""

    declared = protocol["catalogue_binding"]
    if binding["passed"] is not True or any(v is not True for v in binding["checks"].values()):
        raise ValueError("catalogue binding gate records a failed check")
    dataset_raw = (repo / SCREENING_DATASET_PATH).read_bytes()
    manifest_raw = (repo / SCREENING_MANIFEST_PATH).read_bytes()
    if b"\r" in dataset_raw:
        raise ValueError("screening dataset on disk carries a carriage return")
    for value, expected, label in (
        (sha256_bytes(dataset_raw), declared["dataset_file_sha256"], "dataset SHA-256"),
        (len(dataset_raw), declared["dataset_bytes"], "dataset size"),
        (sha256_bytes(manifest_raw), declared["manifest_file_sha256"], "screening manifest SHA-256"),
        (binding["dataset_file_sha256"], declared["dataset_file_sha256"], "gate dataset SHA-256"),
        (binding["manifest_file_sha256"], declared["manifest_file_sha256"], "gate manifest SHA-256"),
        (binding["screening_result_commit"], SCREENING_RESULTS_COMMIT_SHA, "screening result commit"),
        (declared["screening_result_commit"], SCREENING_RESULTS_COMMIT_SHA, "declared screening result commit"),
        (declared["dataset_path"], SCREENING_DATASET_PATH.as_posix(), "dataset path"),
        (declared["manifest_path"], SCREENING_MANIFEST_PATH.as_posix(), "screening manifest path"),
        (declared["classification"], SCREENING_CLASSIFICATION, "screening classification"),
    ):
        if value != expected:
            raise ValueError(f"catalogue binding: {label} differs")
    for path, blob_key in ((SCREENING_DATASET_PATH, "dataset_git_blob"), (SCREENING_MANIFEST_PATH, "manifest_git_blob")):
        at_record = _git(repo, "rev-parse", f"{SCREENING_RESULTS_COMMIT_SHA}:{path.as_posix()}")
        at_head = _git(repo, "rev-parse", f"HEAD:{path.as_posix()}")
        working = _git(repo, "hash-object", "--", path.as_posix())
        if not (at_record == at_head == working == declared[blob_key] == binding["git"][f"{blob_key.split('_git_')[0]}_blob_at_result_commit"]):
            raise ValueError(f"catalogue binding: {blob_key} differs between the record commit, HEAD and the checkout")
    screening_manifest = load_json_bytes(manifest_raw, "screening manifest")
    if screening_manifest["state"] != "accepted_result" or screening_manifest["experiment_id"] != "orbit-wall-loss-geometry-screening-v1":
        raise ValueError("screening manifest is not the accepted screening record")
    entry = next(e for e in screening_manifest["artifacts"] if e.get("path") == "artifacts/geometry-wall-loss-dataset.json")
    if entry["byte_sha256"] != declared["dataset_file_sha256"] or entry["bytes"] != declared["dataset_bytes"]:
        raise ValueError("screening manifest entry for the dataset disagrees with the declared bytes")
    return {
        "dataset_path": SCREENING_DATASET_PATH.as_posix(),
        "dataset_sha256": declared["dataset_file_sha256"],
        "dataset_bytes": declared["dataset_bytes"],
        "dataset_git_blob": declared["dataset_git_blob"],
        "screening_manifest_path": SCREENING_MANIFEST_PATH.as_posix(),
        "screening_manifest_sha256": declared["manifest_file_sha256"],
        "screening_manifest_git_blob": declared["manifest_git_blob"],
        "screening_results_commit": SCREENING_RESULTS_COMMIT_SHA,
        "screening_classification": SCREENING_CLASSIFICATION,
        "rule": (
            "the catalogue is the 96 accepted screening designs; the generator requires the dataset and screening "
            "manifest on disk to hash to the bytes the frozen protocol declares, their Git blobs to equal the blobs "
            "at the screening record commit and at HEAD, and the screening manifest entry to carry the same byte hash"
        ),
    }


def verify_catalogue(catalogue: dict[str, Any], protocol: dict[str, Any], authorities: dict[str, Any]) -> dict[str, Any]:
    """Recompute every probability, posterior mean, Wilson bound and nominal survival of the sealed catalogue."""

    identity = protocol["catalogue_binding_identity"]
    if catalogue["catalogue_sha256"] != identity["catalogue_sha256"] or catalogue["catalogue_sha256"] != authorities["catalogue_sha256"]:
        raise ValueError("sealed catalogue identity differs from the protocol and authorities")
    designs = catalogue["designs"]
    binding = protocol["catalogue_binding"]
    if len(designs) != protocol["design_space"]["catalogue"]["size"] or len(designs) != binding["design_count"]:
        raise ValueError("catalogue size differs from the protocol")
    saturated = 0
    zero_cells = 0
    for position, design in enumerate(designs):
        if design["catalogue_index"] != position or design["case_id"] != f"l1a-gs-v2-{position:03d}-{design['screening_design_id'][:10]}":
            raise ValueError(f"catalogue design {position} carries a foreign index or case id")
        cells = design["cells"]
        if [c["cell"] for c in cells] != binding["cells"]:
            raise ValueError(f"catalogue design {position} cells differ from the protocol")
        pooled = design["pooled"]
        if pooled["trials"] != binding["pooled_trials"] or pooled["wall_hits"] != sum(c["wall_hits"] for c in cells):
            raise ValueError(f"catalogue design {position} pooled counts do not sum from the cells")
        for estimate in (*cells, pooled):
            if estimate["trials"] not in (binding["cell_trials"], binding["pooled_trials"]):
                raise ValueError(f"catalogue design {position} carries a foreign trial count")
            if estimate["probability"] != estimate["wall_hits"] / estimate["trials"]:
                raise ValueError(f"catalogue design {position} probability is not hits over trials")
            if estimate["posterior_mean"] != jeffreys_mean(estimate["wall_hits"], estimate["trials"]):
                raise ValueError(f"catalogue design {position} posterior mean is not the Jeffreys rule")
            if tuple(estimate["wilson_95"]) != wilson(estimate["wall_hits"], estimate["trials"]):
                raise ValueError(f"catalogue design {position} Wilson interval does not recompute")
        survival = math.prod(1.0 - c["posterior_mean"] for c in cells)
        if abs(survival - design["nominal_survival_cl1"]) > 4e-16 * max(1.0, abs(survival)):
            raise ValueError(f"catalogue design {position} nominal CL-1 survival does not recompute")
        if abs((1.0 - pooled["posterior_mean"]) - design["nominal_survival_cl2"]) > 4e-16:
            raise ValueError(f"catalogue design {position} nominal CL-2 survival does not recompute")
        if design["geometry"]["stage_count"] != len(design["geometry"]["stage_centers_m"]):
            raise ValueError(f"catalogue design {position} stage count differs from its stage centres")
        if any(c["wall_hits"] == c["trials"] for c in cells):
            saturated += 1
        if any(c["wall_hits"] == 0 for c in cells):
            zero_cells += 1
    if saturated != identity["designs_with_a_saturated_cell_128_of_128"]:
        raise ValueError("saturated-cell design count differs from the protocol identity")
    pooled_values = [d["pooled"]["probability"] for d in designs]
    if [min(pooled_values), max(pooled_values)] != identity["pooled_wall_hit_probability_range"]:
        raise ValueError("pooled wall-hit probability range differs from the protocol identity")
    return {"saturated": saturated, "zero_cells": zero_cells}


def cross_check_dashboard(
    repo: Path, bundle: Bundle, v1: Bundle, campaign: dict[str, Any], metrics: dict[str, Any], gates: dict[str, Any],
) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of both bundles; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    for constant, value in (
        ("EXPECTED_MANIFEST_SHA256", bundle.manifest_sha256),
        ("RESULTS_COMMIT_SHA", RESULTS_COMMIT_SHA),
        ("PREREGISTRATION_COMMIT_SHA", PREREGISTRATION_COMMIT_SHA),
        ("V1_EXPECTED_MANIFEST_SHA256", v1.manifest_sha256),
        ("V1_RESULTS_COMMIT_SHA", V1_RESULTS_COMMIT_SHA),
    ):
        if f'"{value}"' not in generator_text or constant not in generator_text:
            raise ValueError(f"dashboard generator does not pin {constant} to the bundle identity")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_sha256"] != bundle.manifest_sha256:
        raise ValueError("dashboard payload names a different results manifest")
    if identity["results_commit"] != RESULTS_COMMIT_SHA or identity["preregistration_commit"] != PREREGISTRATION_COMMIT_SHA:
        raise ValueError("dashboard payload names different revisions")
    if identity["terminal_state"] != bundle.manifest["state"] or identity["verified_files"] != len(bundle.hashes):
        raise ValueError("dashboard payload identity differs from the bundle")
    if identity["lock_commit"] != PREREGISTRATION_COMMIT_SHA or identity["import_scope_matches"] is not True:
        raise ValueError("dashboard payload lock commit or import scope differs from the bundle")
    for key, value in payload["campaign_result"].items():
        if campaign[key] != value:
            raise ValueError(f"dashboard campaign_result.{key} differs from the sealed artifact")
    if payload["seed_variance"] != metrics["seed_variance"]:
        raise ValueError("dashboard seed variance differs from the sealed metrics")
    reported = gates["reported_not_binding"]
    for key in ("robust_vs_nominal", "closure_cl1_vs_cl2", "uncertainty_width_sensitivity", "bo_beats_random", "bo_beats_nsga3", "per_design_separability"):
        if payload["gates"][key] != reported[key]:
            raise ValueError(f"dashboard {key} differs from the sealed gates")
    if payload["gates"]["binding"] != {name: item["passed"] for name, item in gates["binding"].items()}:
        raise ValueError("dashboard binding-gate block differs from the sealed gates")
    for key, run in payload["runs"].items():
        if run["final_hypervolume"] != metrics["runs"][key]["final_hypervolume"] or run["pareto_catalogue_indices"] != metrics["runs"][key]["pareto_catalogue_indices"]:
            raise ValueError(f"dashboard final hypervolume or Pareto designs differ for {key}")
    for key in ("count", "designs", "robust_hypervolume", "robust_front_size", "robust_front_catalogue_indices", "nominal_hypervolume", "nominal_front_size"):
        if payload["dense_reference"][key] != metrics["dense_reference"][key]:
            raise ValueError(f"dashboard dense reference {key} differs from the sealed metrics")
    v1_block = payload["v1"]
    v1_metrics = v1.load("artifacts/metrics.json")
    v1_gates = v1.load("artifacts/gates.json")
    if v1_block["manifest_sha256"] != v1.manifest_sha256 or v1_block["results_commit"] != V1_RESULTS_COMMIT_SHA:
        raise ValueError("dashboard v1 block names a different v1 bundle")
    if v1_block["seed_variance"] != v1_metrics["seed_variance"] or v1_block["dense_reference"] != v1_metrics["dense_reference"]:
        raise ValueError("dashboard v1 block differs from the v1 metrics")
    for key in ("bo_beats_random", "bo_beats_nsga3"):
        if v1_block[key]["wins"] != v1_gates["reported_not_binding"][key]["wins"]:
            raise ValueError(f"dashboard v1 {key} differs from the v1 gates")
    for key, value in v1_block["robust_vs_nominal"].items():
        if v1_gates["reported_not_binding"]["robust_vs_nominal"][key] != value:
            raise ValueError(f"dashboard v1 robust_vs_nominal.{key} differs from the v1 gates")
    return {
        "generator_path": DASHBOARD_GENERATOR.as_posix(),
        "generator_sha256_lf": sha256_bytes(_lf(generator_raw)),
        "html_path": DASHBOARD_HTML.as_posix(),
        "html_sha256_lf": sha256_bytes(_lf(html_raw)),
        "html_schema": payload["schema"],
        "payload_manifest_sha256": identity["manifest_sha256"],
        "payload_v1_manifest_sha256": v1_block["manifest_sha256"],
        "rule": (
            "the committed dashboard pins both bundles' manifest SHA-256 values and revisions and embeds its own "
            "extraction of the campaign result and of the prior campaign; the generator requires that extraction to "
            "equal the sealed artifacts of both bundles before writing any macro"
        ),
    }


# --------------------------------------------------------------------------- #
# Macro construction (two bundles)
# --------------------------------------------------------------------------- #
class Macros:
    def __init__(self, bundle: Bundle, v1: Bundle) -> None:
        self.bundles = {"v2": bundle, "v1": v1}
        self.items: list[dict[str, Any]] = []
        self.names: set[str] = set()

    def doc(self, relative: str, bundle: str = "v2") -> Any:
        return self.bundles[bundle].load(relative)

    def add(self, name: str, artifact: str, pointer: str, fmt: str, description: str, *, bundle: str = "v2") -> str:
        if name in self.names or not name.isalpha() or not name.startswith(MACRO_PREFIX):
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        raw = resolve_pointer(self.doc(artifact, bundle), pointer)
        value = format_value(fmt, raw)
        source: dict[str, Any] = {"artifact": artifact, "pointer": pointer}
        if bundle != "v2":
            source["bundle"] = bundle
        self.items.append(
            {"name": name, "value": value, "raw": raw, "format": fmt, "derived": False, "source": source, "description": description}
        )
        self.names.add(name)
        return value

    def add_derived(
        self, name: str, raw: Any, fmt: str, description: str, derivation: str, inputs: list[dict[str, str]],
    ) -> str:
        if name in self.names or not name.isalpha() or not name.startswith(MACRO_PREFIX):
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        value = format_value(fmt, raw)
        self.items.append(
            {
                "name": name, "value": value, "raw": raw, "format": fmt, "derived": True,
                "derivation": derivation, "inputs": inputs, "description": description,
            }
        )
        self.names.add(name)
        return value


def _inp(artifact: str, pointer: str, bundle: str | None = None) -> dict[str, str]:
    entry = {"artifact": artifact, "pointer": pointer}
    if bundle:
        entry["bundle"] = bundle
    return entry


def build(repo: Path) -> tuple[dict[str, Any], str]:  # noqa: C901 - one linear verification pass
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo, RESULTS, EXPERIMENT_ID)
    v1 = Bundle(repo, V1_RESULTS, V1_EXPERIMENT_ID)
    binding = bind_committed(repo, bundle, v1)
    m = Macros(bundle, v1)
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    protocol = m.doc("artifacts/protocol.json")
    metrics = m.doc("artifacts/metrics.json")
    dense = m.doc("artifacts/dense-reference-summary.json")
    dense_full = m.doc("artifacts/dense-reference.json")
    dense_sep = m.doc("artifacts/dense-reference-separability.json")
    separability = m.doc("artifacts/separability.json")
    sensitivity = m.doc("artifacts/sensitivity.json")
    pooled = m.doc("artifacts/pooled-fronts.json")
    per_strategy = m.doc("artifacts/per-strategy-fronts.json")
    plan = m.doc("artifacts/campaign-plan.json")
    contract = m.doc("artifacts/code-contract.json")
    authorities = m.doc("artifacts/authorities.json")
    shakedown = m.doc("artifacts/shakedown.json")
    sample = m.doc("artifacts/uncertain-sample.json")
    catalogue = m.doc("artifacts/catalogue.json")
    catalogue_binding = m.doc("artifacts/catalogue-binding.json")
    import_scope = m.doc("artifacts/import-scope.json")
    consistency = m.doc("artifacts/protocol-consistency.json")
    probes = m.doc("artifacts/device-probes.json")
    runtime = m.doc("artifacts/runtime.json")
    curves = m.doc("artifacts/hypervolume-curves.json")
    pareto_sets = m.doc("artifacts/pareto-sets.json")
    runs = {rel: m.doc(rel) for rel in RUN_ARTIFACTS}
    for rel in V1_COMPARISON_ARTIFACTS:
        m.doc(rel, "v1")
    v1_campaign = m.doc("artifacts/campaign-result.json", "v1")
    v1_plan = m.doc("artifacts/campaign-plan.json", "v1")
    v1_metrics = m.doc("artifacts/metrics.json", "v1")
    v1_gates = m.doc("artifacts/gates.json", "v1")
    v1_dense = m.doc("artifacts/dense-reference-summary.json", "v1")
    v1_pooled = m.doc("artifacts/pooled-fronts.json", "v1")
    v1_protocol = m.doc("artifacts/protocol.json", "v1")
    v1_sensitivity = m.doc("artifacts/sensitivity.json", "v1")
    dashboard = cross_check_dashboard(repo, bundle, v1, campaign, metrics, gates)
    catalogue_facts = verify_catalogue_binding(repo, catalogue_binding, protocol)
    catalogue_counts = verify_catalogue(catalogue, protocol, authorities)
    reported = gates["reported_not_binding"]

    # Internal consistency of the sealed bundle (fail closed on any disagreement).
    if terminal["state"] != bundle.manifest["state"] or terminal["payload"]["all_binding_gates_passed"] is not True:
        raise ValueError("terminal record disagrees with the manifest or records a failed gate")
    if campaign["classification"] != CLASSIFICATION or campaign["closure"] != CLOSURE_ID or campaign["sensitivity_closure"] != SENSITIVITY_CLOSURE_ID:
        raise ValueError("campaign classification or closures differ from the admitted identity")
    if campaign["all_binding_gates_passed"] is not True or gates["all_binding_passed"] is not True:
        raise ValueError("binding gates are not all passed")
    if any(item["passed"] is not True for item in gates["binding"].values()):
        raise ValueError("a binding gate records a failure")
    if terminal["payload"]["binding_gate_results"] != {k: v["passed"] for k, v in gates["binding"].items()}:
        raise ValueError("terminal gate results differ from gates.json")
    if terminal["payload"]["total_evaluations"] != campaign["total_evaluations"] or terminal["payload"]["runs"] != campaign["runs"]:
        raise ValueError("terminal counts differ from the campaign result")
    if not (gates["semantics"] == campaign["gate_semantics"] == protocol["gates"]["semantics"]):
        raise ValueError("gate semantics differ between protocol, gates and campaign result")
    if "INTEGRITY" not in gates["semantics"] or "NOT that any optimiser is effective" not in gates["semantics"]:
        raise ValueError("gate semantics do not declare integrity-only acceptance")
    for key in ("hypervolume_table", "seed_variance", "dense_reference"):
        if campaign[key] != metrics[key]:
            raise ValueError(f"campaign-result and metrics disagree on {key}")
    for key in ("robust_vs_nominal", "closure_cl1_vs_cl2", "uncertainty_width_sensitivity"):
        if campaign[key] != reported[key]:
            raise ValueError(f"campaign-result and gates disagree on {key}")
    if campaign["claim_boundary"] != protocol["claim_boundary"]["statement"]:
        raise ValueError("campaign claim boundary differs from the protocol")
    if campaign["closure_identification_disclosure"] != protocol["closures"]["CL-1"]["identification_disclosure"]:
        raise ValueError("closure identification disclosure differs from the protocol")
    if lock["commit"] != PREREGISTRATION_COMMIT_SHA or lock["attempt"] != 1 or lock["immutable"] is not True:
        raise ValueError("execution lock does not record the single preregistered attempt")
    if protocol["classification"] != CLASSIFICATION or protocol["closures"]["CL-1"]["id"] != CLOSURE_ID or protocol["closures"]["CL-2"]["id"] != SENSITIVITY_CLOSURE_ID:
        raise ValueError("protocol classification or closures differ from the campaign")
    if protocol["closures"]["CL-1"]["role"] != "campaign" or protocol["closures"]["CL-2"]["role"] != "sensitivity":
        raise ValueError("closure roles differ from the admitted layout")
    if any(v is not True for v in consistency.values()):
        raise ValueError("protocol-consistency records a failed check")
    if plan != authorities["evidentiary_plan"] or plan["kind"] != "evidentiary":
        raise ValueError("campaign plan differs from the preregistered authorities")
    if contract["matches"] is not True or contract["source_sha256"] != authorities["source_sha256"]:
        raise ValueError("code contract does not match the preregistered authorities")
    if contract["source_line_endings"] != "LF" or contract["import_scope_at_prebundle"]["matches"] is not True:
        raise ValueError("code contract line endings or prebundle import scope are not as declared")
    scope = protocol["code_contract"]["source_hash_scope"]
    if import_scope["declared"] != scope or import_scope["imported"] != scope or import_scope["matches"] is not True:
        raise ValueError("import scope differs from the protocol hash scope")
    if import_scope["imported_not_in_scope"] or import_scope["in_scope_not_imported"]:
        raise ValueError("import scope records unbound imports or unused bindings")
    if gates["binding"]["code_hash_scope_matches_imports"]["imported_count"] != len(scope) or len(contract["source_files"]) != len(scope):
        raise ValueError("import-scope gate count differs from the hash scope")
    if [f["path"] for f in contract["source_files"]] != scope:
        raise ValueError("code contract source files differ from the hash scope")
    sample_spec = protocol["uncertain_inputs"]["sample"]
    if not (sample["unit_rows_sha256"] == sample_spec["unit_rows_sha256"] == authorities["unit_rows_sha256"]):
        raise ValueError("frozen unit-row hash differs between artifacts")
    if not (sample["catalogue_sample_sha256"] == sample_spec["catalogue_sample_sha256"] == authorities["catalogue_sample_sha256"]):
        raise ValueError("frozen catalogue-sample hash differs between artifacts")
    if len(sample["unit_rows"]) != sample_spec["count"] or len(sample["sample"]) != len(catalogue["designs"]) or any(len(rows) != sample_spec["count"] for rows in sample["sample"]):
        raise ValueError("frozen sample shape differs from the protocol")
    if len(sample["nominal"]) != len(catalogue["designs"]):
        raise ValueError("nominal points do not cover the catalogue")
    for design, nominal in zip(catalogue["designs"], sample["nominal"], strict=True):
        for k, cell in enumerate(design["cells"], start=1):
            if nominal[f"wall_loss_probability_cell_{k}"] != cell["posterior_mean"]:
                raise ValueError("nominal cell probability differs from the catalogue posterior mean")
        if nominal["wall_loss_probability_pooled"] != design["pooled"]["posterior_mean"]:
            raise ValueError("nominal pooled probability differs from the catalogue posterior mean")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if shakedown["disjointness"]["proven"] is not True or shakedown["import_scope"]["matches"] is not True:
        raise ValueError("shakedown disjointness or import scope is not proven")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"]:
        raise ValueError("shakedown artifact differs from the bound authority")
    for frozen, sealed in (("protocol.json", protocol), ("authorities.json", authorities), ("shakedown.json", shakedown)):
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != sealed:
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
    if list(plan["strategies"]) != list(STRATEGIES) or len(plan["seeds"]) != len(SEED_TOKENS):
        raise ValueError("plan strategies or seeds differ from the admitted layout")
    if len(metrics["runs"]) != len(plan["run_ids"]) or campaign["runs"] != len(plan["run_ids"]):
        raise ValueError("run count differs between plan, metrics and campaign result")
    if probes["cuda"]["device"] != "cuda:0" or probes["cpu"]["device"] != "cpu" or protocol["optimizers"]["qlognehvi"]["device"] != "cpu":
        raise ValueError("device probes or the declared BO device differ from the admitted layout")
    operating = protocol["design_space"]["operating_point"]
    if [d["name"] for d in operating] != ["discharge_voltage_v", "anode_current_a", "propellant_mass_flow_kg_per_s"]:
        raise ValueError("operating-point variables differ from the declared three")
    if protocol["design_space"]["catalogue"]["kind"] != "categorical" or protocol["design_space"]["catalogue"]["source_classification"] != SCREENING_CLASSIFICATION:
        raise ValueError("catalogue variable is not the categorical screened design index")
    if set(protocol["v1_audit_disclosures_closed"]) != set(AUDIT_DISCLOSURES):
        raise ValueError("the protocol's closed audit disclosures differ from the admitted list")
    if protocol["authority"]["v1_campaign"]["result_commit"] != V1_RESULTS_COMMIT_SHA[:8] or protocol["authority"]["v1_campaign"]["posthoc_audit_commit"] != V1_AUDIT_COMMIT_SHA[:8]:
        raise ValueError("protocol names a different prior campaign or audit")
    if len(protocol["authority"]["rejected_surrogates"]) != 2 or any(s["outcome"] != "rejected_surrogate" for s in protocol["authority"]["rejected_surrogates"]):
        raise ValueError("protocol does not record the two rejected surrogates")

    # The prior campaign's audit: the disclosures the protocol closes must be the audit's own list.
    audit_text = _lf((repo / V1_AUDIT_PATH).read_bytes()).decode("utf-8")
    audit_match = AUDIT_PATTERN.search(audit_text)
    if audit_match is None:
        raise ValueError("the v1 post-hoc audit does not carry its disclosure list in the fixed pattern")
    audit_ids = tuple(audit_match.group(1).split(", "))
    if audit_ids != AUDIT_DISCLOSURES or "ACCEPTED WITH DISCLOSURES" not in audit_text:
        raise ValueError("the v1 post-hoc audit disclosures differ from the ones the protocol closes")

    # Same reference frame as v1: reference point, scales, sample rows, robust formulation, operating domain.
    same_frame = (
        {k: v for k, v in protocol["reference_point"].items() if k != "normalization"}
        == {k: v for k, v in v1_protocol["reference_point"].items() if k != "normalization"}
        and [(o["name"], o["direction"], o["comparison_scale"]) for o in protocol["objectives"]]
        == [(o["name"], o["direction"], o["comparison_scale"]) for o in v1_protocol["objectives"]]
        and (protocol["robust_formulation"]["tail_count"], protocol["robust_formulation"]["tail_fraction"], protocol["robust_formulation"]["risk_measure"])
        == (v1_protocol["robust_formulation"]["tail_count"], v1_protocol["robust_formulation"]["tail_fraction"], v1_protocol["robust_formulation"]["risk_measure"])
        and (sample_spec["count"], sample_spec["bases"], sample_spec["seed"])
        == (v1_protocol["uncertain_inputs"]["sample"]["count"], v1_protocol["uncertain_inputs"]["sample"]["bases"], v1_protocol["uncertain_inputs"]["sample"]["seed"])
        and [(d["name"], d["lower"], d["upper"]) for d in operating]
        == [(d["name"], d["lower"], d["upper"]) for d in v1_protocol["design_variables"]]
        and protocol["closures"]["fixed"]["cathode_input_power_w"] == v1_protocol["closures"]["fixed"]["cathode_input_power_w"]
        and protocol["closures"]["fixed"]["ppu_efficiency_fraction"] == v1_protocol["closures"]["fixed"]["ppu_efficiency_fraction"]
        and protocol["constraints"][0]["name"] == v1_protocol["constraints"][0]["name"]
        and protocol["constraints"][0]["threshold"] == v1_protocol["constraints"][0]["threshold"]
    )
    if not same_frame:
        raise ValueError("the v2 protocol does not share v1's reference frame")
    if v1_campaign["classification"] != mdo_l0_v1.CLASSIFICATION or v1_campaign["closure"] != mdo_l0_v1.CLOSURE_ID:
        raise ValueError("the v1 bundle is not the admitted prior campaign")
    frame_inputs = [
        _inp("artifacts/protocol.json", "/reference_point"), _inp("artifacts/protocol.json", "/objectives"),
        _inp("artifacts/protocol.json", "/robust_formulation"), _inp("artifacts/protocol.json", "/uncertain_inputs/sample"),
        _inp("artifacts/protocol.json", "/design_space/operating_point"),
        _inp("artifacts/protocol.json", "/reference_point", "v1"), _inp("artifacts/protocol.json", "/objectives", "v1"),
        _inp("artifacts/protocol.json", "/robust_formulation", "v1"), _inp("artifacts/protocol.json", "/uncertain_inputs/sample", "v1"),
        _inp("artifacts/protocol.json", "/design_variables", "v1"),
    ]

    # Identity and lifecycle.
    m.add("MdbClassification", "artifacts/campaign-result.json", "/classification", "ident", "campaign classification string")
    m.add("MdbTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("MdbClosureId", "artifacts/campaign-result.json", "/closure", "ident", "declared campaign closure identifier")
    m.add("MdbSensitivityClosureId", "artifacts/campaign-result.json", "/sensitivity_closure", "ident", "declared sensitivity closure identifier")
    closure_keys = [k for k, v in protocol["closures"].items() if isinstance(v, dict) and v.get("id") == CLOSURE_ID]
    sensitivity_keys = [k for k, v in protocol["closures"].items() if isinstance(v, dict) and v.get("id") == SENSITIVITY_CLOSURE_ID]
    if closure_keys != ["CL-1"] or sensitivity_keys != ["CL-2"]:
        raise ValueError("closure short names differ from the admitted layout")
    m.add_derived("MdbClosureShort", closure_keys[0], "text", "short name of the campaign closure", "the key of protocol.closures whose id equals the campaign closure", [_inp("artifacts/protocol.json", "/closures")])
    m.add_derived("MdbSensitivityClosureShort", sensitivity_keys[0], "text", "short name of the sensitivity closure", "the key of protocol.closures whose id equals the sensitivity closure", [_inp("artifacts/protocol.json", "/closures")])
    m.add("MdbClosureStatus", "artifacts/protocol.json", "/closures/CL-1/status", "text", "campaign closure status")
    m.add("MdbSensitivityClosureStatus", "artifacts/protocol.json", "/closures/CL-2/status", "text", "sensitivity closure status")
    m.add("MdbFidelity", "artifacts/protocol.json", "/authority/l0_model/fidelity", "ident", "declared model fidelity label")
    m.add("MdbExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier")
    m.add("MdbPriorCampaignId", "artifacts/protocol.json", "/experiment_id", "ident", "prior campaign identifier", bundle="v1")
    m.add("MdbPriorClassification", "artifacts/campaign-result.json", "/classification", "ident", "prior campaign classification string", bundle="v1")
    m.add("MdbScreeningClassification", "artifacts/protocol.json", "/catalogue_binding/classification", "ident", "classification of the screening dataset behind the catalogue")
    m.add("MdbScreeningExperiment", "artifacts/protocol.json", "/catalogue_binding/experiment", "ident", "screening experiment path")
    m.add("MdbReportedCase", "artifacts/protocol.json", "/catalogue_binding/reported_case", "ident", "screening case whose counts the catalogue carries")
    m.add("MdbAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("MdbRuns", "artifacts/campaign-result.json", "/runs", "int", "optimiser runs (strategies times seeds)")
    m.add("MdbTotalEvaluations", "artifacts/campaign-result.json", "/total_evaluations", "int", "design evaluations in the campaign")
    m.add("MdbInfeasibleEvaluations", "artifacts/campaign-result.json", "/infeasible_evaluations", "int", "constraint-violating design evaluations")
    m.add("MdbEvaluationsPerRun", "artifacts/campaign-plan.json", "/evaluations_per_run", "int", "evaluation budget per run")
    m.add("MdbInitialDesign", "artifacts/campaign-plan.json", "/initial_design", "int", "shared initial design size per seed")
    m.add("MdbSeeds", "artifacts/campaign-plan.json", "/seeds", "list_int", "evidentiary seeds")
    m.add_derived("MdbSeedCount", len(plan["seeds"]), "int", "number of seeds", "len(plan.seeds)", [_inp("artifacts/campaign-plan.json", "/seeds")])
    m.add_derived("MdbStrategyCount", len(plan["strategies"]), "int", "number of optimisers", "len(plan.strategies)", [_inp("artifacts/campaign-plan.json", "/strategies")])
    m.add("MdbBoBatch", "artifacts/campaign-plan.json", "/qlognehvi_batch_size", "int", "BO batch size")
    m.add("MdbBoIterations", "artifacts/campaign-plan.json", "/qlognehvi_iterations", "int", "BO iterations")
    m.add("MdbNsgaPopulation", "artifacts/campaign-plan.json", "/nsga3_population_size", "int", "NSGA-III population size")
    m.add("MdbNsgaGenerations", "artifacts/campaign-plan.json", "/nsga3_generations", "int", "NSGA-III generations")
    m.add("MdbBoMcSamples", "artifacts/protocol.json", "/optimizers/qlognehvi/mc_samples", "int", "BO Monte Carlo samples")
    m.add("MdbBoCandidatesPerDesign", "artifacts/protocol.json", "/optimizers/qlognehvi/candidates_per_design", "int", "operating-point candidates per catalogue design in the discrete acquisition stage")
    m.add("MdbBoRefineMaxiter", "artifacts/protocol.json", "/optimizers/qlognehvi/refine_maxiter", "int", "L-BFGS-B iterations of the refinement stage")
    m.add_derived("MdbBoCandidateCount", protocol["design_space"]["catalogue"]["size"] * protocol["optimizers"]["qlognehvi"]["candidates_per_design"], "int", "discrete acquisition candidates per BO iteration", "catalogue size times candidates_per_design", [_inp("artifacts/protocol.json", "/design_space/catalogue/size"), _inp("artifacts/protocol.json", "/optimizers/qlognehvi/candidates_per_design")])
    m.add_derived("MdbSecondStage", plan["evaluations_per_run"] - plan["initial_design"], "int", "evaluations after the shared initial design per run", "plan.evaluations_per_run - plan.initial_design", [_inp("artifacts/campaign-plan.json", "/evaluations_per_run"), _inp("artifacts/campaign-plan.json", "/initial_design")])
    m.add("MdbBoDevice", "artifacts/protocol.json", "/optimizers/qlognehvi/device", "ident", "BO device")
    m.add("MdbBoDtype", "artifacts/protocol.json", "/optimizers/qlognehvi/dtype", "ident", "BO floating-point type")
    m.add("MdbBoThreads", "artifacts/protocol.json", "/optimizers/qlognehvi/torch_threads", "int", "torch threads declared for the BO runs")
    m.add("MdbBoLibrary", "artifacts/protocol.json", "/optimizers/qlognehvi/library", "text", "BO library")
    m.add("MdbNsgaLibrary", "artifacts/protocol.json", "/optimizers/nsga3/library", "text", "NSGA-III library")
    m.add("MdbNsgaReferenceDirectionCount", f"artifacts/runs/nsga3-{plan['seeds'][0]}.json", "/optimizer/reference_direction_count", "int", "NSGA-III energy reference directions")
    m.add("MdbNsgaReferenceDirectionSeed", "artifacts/protocol.json", "/optimizers/nsga3/reference_direction_seed", "int", "NSGA-III reference-direction seed")
    if any(runs[f"artifacts/runs/nsga3-{seed}.json"]["optimizer"]["reference_direction_count"] != runs[f"artifacts/runs/nsga3-{plan['seeds'][0]}.json"]["optimizer"]["reference_direction_count"] or runs[f"artifacts/runs/nsga3-{seed}.json"]["optimizer"]["reference_direction_seed"] != protocol["optimizers"]["nsga3"]["reference_direction_seed"] for seed in plan["seeds"]):
        raise ValueError("NSGA-III reference directions differ between runs or from the protocol")
    m.add("MdbBotorchVersion", "artifacts/code-contract.json", "/observed_package_versions/botorch", "text", "BoTorch version")
    m.add("MdbPymooVersion", "artifacts/code-contract.json", "/observed_package_versions/pymoo", "text", "pymoo version")
    m.add("MdbTorchVersion", "artifacts/code-contract.json", "/observed_package_versions/torch", "text", "torch version")
    m.add("MdbCodeContractMatches", "artifacts/code-contract.json", "/matches", "bool", "code contract matches the preregistered authorities")
    m.add("MdbSourceSha", "artifacts/code-contract.json", "/source_sha256", "sha_short", "hashed source prefix")
    m.add("MdbProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("MdbUnitRowsSha", "artifacts/uncertain-sample.json", "/unit_rows_sha256", "sha_short", "frozen unit-row hash prefix")
    m.add("MdbCatalogueSampleSha", "artifacts/uncertain-sample.json", "/catalogue_sample_sha256", "sha_short", "frozen per-design sample hash prefix")
    m.add("MdbCatalogueSha", "artifacts/catalogue.json", "/catalogue_sha256", "sha_short", "sealed catalogue identity prefix")
    m.add("MdbPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    m.add("MdbLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutability flag")
    m.add_derived("MdbResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [_inp("manifest.json", "")])
    m.add_derived("MdbDashboardCommit", DASHBOARD_COMMIT_SHA, "sha_short", "dashboard commit prefix", "git commit that added the results dashboard cross-checked by this generator", [_inp("manifest.json", "")])
    m.add_derived("MdbPriorResultsCommit", V1_RESULTS_COMMIT_SHA, "sha_short", "prior campaign results commit prefix", "git commit whose tree holds the v1 results manifest blob (verified with rev-parse against the working tree)", [_inp("manifest.json", "", "v1")])
    m.add_derived("MdbPriorAuditCommit", V1_AUDIT_COMMIT_SHA, "sha_short", "prior campaign post-hoc audit commit prefix", "git commit that added POSTHOC_AUDIT.md of the prior campaign; its blob must equal the checkout", [_inp("manifest.json", "", "v1")])
    m.add_derived("MdbScreeningResultsCommit", SCREENING_RESULTS_COMMIT_SHA, "sha_short", "screening record commit prefix", "git commit at which the screening dataset blob is bound (catalogue_binding.screening_result_commit)", [_inp("artifacts/catalogue-binding.json", "/screening_result_commit")])
    m.add("MdbScreeningDatasetBlob", "artifacts/catalogue-binding.json", "/git/dataset_blob_at_result_commit", "sha_short", "Git blob prefix of the screening dataset at the record commit")
    m.add("MdbScreeningDatasetSha", "artifacts/catalogue-binding.json", "/dataset_file_sha256", "sha_short", "screening dataset SHA-256 prefix")
    m.add("MdbCatalogueBindingPassed", "artifacts/catalogue-binding.json", "/passed", "bool", "catalogue binding gate outcome")
    m.add_derived("MdbCatalogueBindingChecks", len(catalogue_binding["checks"]), "int", "catalogue binding checks", "len(catalogue_binding.checks)", [_inp("artifacts/catalogue-binding.json", "/checks")])
    m.add_derived("MdbManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [_inp("manifest.json", "")])
    m.add_derived("MdbPriorManifestSha", v1.manifest_sha256, "sha_short", "prior campaign results manifest SHA-256 prefix", "sha256(v1 results/manifest.json)", [_inp("manifest.json", "", "v1")])
    m.add_derived("MdbVerifiedFiles", len(bundle.hashes), "int", "bundle files verified byte-for-byte", "count of manifest file entries whose sha256 and size equal the checkout", [_inp("manifest.json", "/artifacts")])
    m.add_derived("MdbPriorVerifiedFiles", len(v1.hashes), "int", "prior campaign bundle files verified byte-for-byte", "count of v1 manifest file entries whose sha256 and size equal the checkout", [_inp("manifest.json", "/artifacts", "v1")])
    m.add_derived("MdbArtifactCount", bundle.manifest["artifact_count"], "int", "manifest entries (files and directories)", "manifest.artifact_count", [_inp("manifest.json", "/artifact_count")])
    m.add_derived("MdbToleratedEolFiles", 0, "int", "manifest entries accepted through any end-of-line tolerance", "count of manifest file entries whose recorded byte_sha256 differs from sha256(bytes); the generator grants no tolerance and fails on any difference", [_inp("manifest.json", "/artifacts")])
    m.add_derived("MdbResultCommitFiles", binding["result_commit_file_count"], "int", "files added by the results commit", "len(git diff-tree --no-commit-id --name-only -r <results commit>)", [_inp("manifest.json", "")])
    m.add_derived("MdbResultCommitOutsideResults", len(binding["result_commit_files_outside_results"]), "int", "files of the results commit outside the results directory", "count of results-commit paths not under the experiment's results/ directory", [_inp("manifest.json", "")])
    m.add_derived("MdbPreregCommitFiles", binding["preregistration_commit_file_count"], "int", "files of the preregistration commit", "len(git diff-tree --no-commit-id --name-only -r <preregistration commit>); every path lies under the experiment directory and none under results/", [_inp("manifest.json", "")])

    # Gates.
    m.add_derived("MdbGateCount", len(gates["binding"]), "int", "binding gates", "len(gates.binding)", [_inp("artifacts/gates.json", "/binding")])
    m.add_derived("MdbGatesPassed", sum(1 for v in gates["binding"].values() if v["passed"] is True), "int", "binding gates passed", "count(gates.binding[*].passed == true)", [_inp("artifacts/gates.json", "/binding")])
    m.add("MdbReplayed", "artifacts/gates.json", "/binding/replay_bit_exact/replayed", "int", "evaluations replayed bit-exactly")
    m.add_derived("MdbReplayMismatches", len(gates["binding"]["replay_bit_exact"]["mismatches"]), "int", "replay mismatches", "len(gates.binding.replay_bit_exact.mismatches)", [_inp("artifacts/gates.json", "/binding/replay_bit_exact/mismatches")])
    if gates["binding"]["replay_bit_exact"]["replayed"] != campaign["total_evaluations"]:
        raise ValueError("replayed evaluation count differs from the total")
    m.add("MdbHvMonotoneLargestDecrease", "artifacts/gates.json", "/binding/hypervolume_monotone/largest_relative_decrease", "g", "largest recorded relative hypervolume decrease")
    m.add("MdbHvMonotoneTolerance", "artifacts/gates.json", "/binding/hypervolume_monotone/relative_tolerance", "sci1", "hypervolume monotonicity roundoff tolerance")
    m.add("MdbImportScopePassed", "artifacts/gates.json", "/binding/code_hash_scope_matches_imports/passed", "bool", "import-scope gate outcome")
    m.add("MdbImportedFiles", "artifacts/gates.json", "/binding/code_hash_scope_matches_imports/imported_count", "int", "repository files imported by the campaign and bound by the code contract")
    m.add_derived("MdbImportedNotInScope", len(import_scope["imported_not_in_scope"]), "int", "imported repository files outside the hash scope", "len(import_scope.imported_not_in_scope)", [_inp("artifacts/import-scope.json", "/imported_not_in_scope")])
    m.add_derived("MdbInScopeNotImported", len(import_scope["in_scope_not_imported"]), "int", "hash-bound files never imported", "len(import_scope.in_scope_not_imported)", [_inp("artifacts/import-scope.json", "/in_scope_not_imported")])
    m.add_derived("MdbSourceFileCount", len(contract["source_files"]), "int", "files in the code-contract hash scope", "len(code_contract.source_files)", [_inp("artifacts/code-contract.json", "/source_files")])
    m.add("MdbNsgaDuplicatesPassed", "artifacts/gates.json", "/binding/nsga3_duplicates_eliminated/passed", "bool", "NSGA-III duplicate-elimination gate outcome")
    duplicates = gates["binding"]["nsga3_duplicates_eliminated"]["runs"]
    if set(duplicates) != {f"nsga3:{seed}" for seed in plan["seeds"]}:
        raise ValueError("duplicate gate does not cover the NSGA-III runs")
    m.add_derived("MdbNsgaDuplicates", sum(v["duplicates"] for v in duplicates.values()), "int", "duplicate NSGA-III evaluations over the three seeds", "sum(gates.binding.nsga3_duplicates_eliminated.runs[*].duplicates)", [_inp("artifacts/gates.json", "/binding/nsga3_duplicates_eliminated/runs")])
    m.add("MdbLabelsPassed", "artifacts/gates.json", "/binding/labels_consistent/passed", "bool", "label-consistency gate outcome")
    label_checks = gates["binding"]["labels_consistent"]["checks"]
    if any(v is not True for v in label_checks.values()):
        raise ValueError("a label check failed")
    m.add_derived("MdbLabelChecks", len(label_checks), "int", "descriptive labels re-derived by the label gate", "len(gates.binding.labels_consistent.checks)", [_inp("artifacts/gates.json", "/binding/labels_consistent/checks")])
    m.add("MdbDenseReplayed", "artifacts/dense-reference-summary.json", "/replay/replayed", "int", "dense-reference designs replayed")
    m.add("MdbDenseReplayPassed", "artifacts/dense-reference-summary.json", "/replay/passed", "bool", "dense-reference replay passed")
    m.add("MdbSeparabilityPassed", "artifacts/gates.json", "/reported_not_binding/per_design_separability/passed", "bool", "per-design separability outcome")
    m.add("MdbSeparabilityDesigns", "artifacts/gates.json", "/reported_not_binding/per_design_separability/designs", "int", "designs covered by the separability check")
    if separability["passed"] is not True or dense_sep["passed"] is not True or len(separability["per_design"]) != len(catalogue["designs"]) or len(dense_sep["per_design"]) != len(catalogue["designs"]):
        raise ValueError("separability records disagree with the catalogue")
    spreads = [ratio["relative_spread"] for design in (*separability["per_design"], *dense_sep["per_design"]) for key, ratio in design.items() if key != "catalogue_index"]
    m.add_derived("MdbSeparabilitySpreadMax", max(spreads), "sci1", "largest within-design relative spread of the robust-to-nominal objective ratio (campaign and dense records)", "max over designs and objectives of relative_spread in separability.json and dense-reference-separability.json", [_inp("artifacts/separability.json", "/per_design"), _inp("artifacts/dense-reference-separability.json", "/per_design")])
    m.add("MdbSeparabilityTolerance", "artifacts/separability.json", "/tolerance_relative_spread", "sci1", "separability tolerance")

    # Design space, catalogue and uncertain inputs.
    m.add("MdbCatalogueSize", "artifacts/protocol.json", "/design_space/catalogue/size", "int", "catalogue designs")
    m.add_derived("MdbOperatingVariableCount", len(operating), "int", "continuous operating-point variables", "len(protocol.design_space.operating_point)", [_inp("artifacts/protocol.json", "/design_space/operating_point")])
    m.add("MdbUaLower", "artifacts/protocol.json", "/design_space/operating_point/0/lower", "g", "discharge voltage lower bound (V)")
    m.add("MdbUaUpper", "artifacts/protocol.json", "/design_space/operating_point/0/upper", "g", "discharge voltage upper bound (V)")
    m.add("MdbIaLower", "artifacts/protocol.json", "/design_space/operating_point/1/lower", "g", "anode current lower bound (A)")
    m.add("MdbIaUpper", "artifacts/protocol.json", "/design_space/operating_point/1/upper", "g", "anode current upper bound (A)")
    m.add("MdbMdotLower", "artifacts/protocol.json", "/design_space/operating_point/2/lower", "sci1", "mass flow lower bound (kg/s)")
    m.add("MdbMdotUpper", "artifacts/protocol.json", "/design_space/operating_point/2/upper", "sci1", "mass flow upper bound (kg/s)")
    m.add("MdbCellTrials", "artifacts/protocol.json", "/catalogue_binding/cell_trials", "int", "launches per cell in the screening counts")
    m.add("MdbPooledTrials", "artifacts/protocol.json", "/catalogue_binding/pooled_trials", "int", "pooled launches per design")
    m.add_derived("MdbCellCount", len(protocol["catalogue_binding"]["cells"]), "int", "launch cells per design", "len(protocol.catalogue_binding.cells)", [_inp("artifacts/protocol.json", "/catalogue_binding/cells")])
    per_design_inputs = protocol["uncertain_inputs"]["per_design_inputs"]
    shared_inputs = protocol["uncertain_inputs"]["shared_inputs"]
    if len(per_design_inputs) != len(protocol["catalogue_binding"]["cells"]) or any(u["distribution"] != "jeffreys-beta-posterior" for u in per_design_inputs):
        raise ValueError("per-design inputs are not four Jeffreys posteriors")
    if [u["name"] for u in shared_inputs] != ["ionized_number_fraction", "xe_double_plus_fraction_of_ions", "axial_momentum_fraction_of_ion_momentum"]:
        raise ValueError("shared uncertain inputs are not in the declared order")
    m.add_derived("MdbUncertainInputCount", len(per_design_inputs) + len(shared_inputs), "int", "uncertain inputs per design", "len(per_design_inputs) + len(shared_inputs)", [_inp("artifacts/protocol.json", "/uncertain_inputs/per_design_inputs"), _inp("artifacts/protocol.json", "/uncertain_inputs/shared_inputs")])
    m.add("MdbJeffreysPrior", "artifacts/protocol.json", "/uncertain_inputs/sample/jeffreys_prior", "g", "Jeffreys prior pseudo-count")
    m.add("MdbEtaLower", "artifacts/protocol.json", "/uncertain_inputs/shared_inputs/0/lower", "g", "ionised fraction lower bound")
    m.add("MdbEtaUpper", "artifacts/protocol.json", "/uncertain_inputs/shared_inputs/0/upper", "g", "ionised fraction upper bound")
    m.add("MdbZetaLower", "artifacts/protocol.json", "/uncertain_inputs/shared_inputs/1/lower", "g", "doubly charged share lower bound")
    m.add("MdbZetaUpper", "artifacts/protocol.json", "/uncertain_inputs/shared_inputs/1/upper", "g", "doubly charged share upper bound")
    m.add("MdbGammaLower", "artifacts/protocol.json", "/uncertain_inputs/shared_inputs/2/lower", "g", "divergence factor lower bound")
    m.add("MdbGammaUpper", "artifacts/protocol.json", "/uncertain_inputs/shared_inputs/2/upper", "g", "divergence factor upper bound")
    m.add("MdbSampleCount", "artifacts/protocol.json", "/uncertain_inputs/sample/count", "int", "frozen QMC sample rows per design")
    m.add("MdbSampleBases", "artifacts/protocol.json", "/uncertain_inputs/sample/bases", "list_int", "Halton bases")
    m.add("MdbSampleFrozen", "artifacts/protocol.json", "/uncertain_inputs/sample/frozen", "bool", "sample frozen flag")
    m.add("MdbTailCount", "artifacts/protocol.json", "/robust_formulation/tail_count", "int", "CVaR tail count")
    m.add("MdbTailFraction", "artifacts/protocol.json", "/robust_formulation/tail_fraction", "g", "CVaR tail fraction")
    m.add("MdbRiskMeasure", "artifacts/protocol.json", "/robust_formulation/risk_measure", "text", "risk measure")
    if protocol["robust_formulation"]["tail_count"] != round(protocol["robust_formulation"]["tail_fraction"] * sample_spec["count"]):
        raise ValueError("CVaR tail count is not tail_fraction times the sample count")
    m.add("MdbCathodePowerW", "artifacts/protocol.json", "/closures/fixed/cathode_input_power_w", "g", "fixed cathode input power (W)")
    m.add("MdbPpuEfficiency", "artifacts/protocol.json", "/closures/fixed/ppu_efficiency_fraction", "g", "fixed PPU efficiency")
    objectives = protocol["objectives"]
    if [o["name"] for o in objectives] != list(OBJECTIVES):
        raise ValueError("objectives differ from the declared four")
    m.add_derived("MdbObjectiveCount", len(objectives), "int", "objectives", "len(protocol.objectives)", [_inp("artifacts/protocol.json", "/objectives")])
    m.add("MdbRefAnodePowerW", "artifacts/protocol.json", "/reference_point/anode_input_power_w", "g", "hypervolume reference point, anode power (W)")
    m.add("MdbScaleThrust", "artifacts/protocol.json", "/objectives/0/comparison_scale", "g", "thrust comparison scale (N)")
    m.add("MdbScaleIsp", "artifacts/protocol.json", "/objectives/1/comparison_scale", "g", "specific impulse comparison scale (s)")
    m.add("MdbScalePower", "artifacts/protocol.json", "/objectives/3/comparison_scale", "g", "anode power comparison scale (W)")
    m.add("MdbConstraintName", "artifacts/protocol.json", "/constraints/0/name", "ident", "robust constraint name")
    m.add("MdbConstraintThreshold", "artifacts/protocol.json", "/constraints/0/threshold", "g", "robust constraint threshold (A)")
    m.add_derived("MdbSameReferenceFrame", same_frame, "bool", "whether the reference point, scales, frozen unit rows, robust formulation and operating domain equal the prior campaign's", "equality of protocol.reference_point, objectives (name, direction, scale), robust_formulation, sample (count, bases, seed), operating-point bounds, fixed closures and the robust constraint between the v2 and v1 protocols", frame_inputs)
    m.add_derived("MdbSaturatedDesigns", catalogue_counts["saturated"], "int", "catalogue designs with at least one cell that lost every launch", "count of catalogue designs with a cell whose wall_hits equal its trials (equal to protocol.catalogue_binding_identity.designs_with_a_saturated_cell_128_of_128)", [_inp("artifacts/catalogue.json", "/designs"), _inp("artifacts/protocol.json", "/catalogue_binding_identity/designs_with_a_saturated_cell_128_of_128")])
    m.add_derived("MdbZeroCellDesigns", catalogue_counts["zero_cells"], "int", "catalogue designs with a cell that lost no launch", "count of catalogue designs with a cell whose wall_hits are zero", [_inp("artifacts/catalogue.json", "/designs")])
    m.add("MdbPooledWallPMin", "artifacts/protocol.json", "/catalogue_binding_identity/pooled_wall_hit_probability_range/0", "fixed3", "smallest pooled wall-hit probability in the catalogue")
    m.add("MdbPooledWallPMax", "artifacts/protocol.json", "/catalogue_binding_identity/pooled_wall_hit_probability_range/1", "fixed3", "largest pooled wall-hit probability in the catalogue")
    designs = catalogue["designs"]
    nominal_survivals = [d["nominal_survival_cl1"] for d in designs]
    m.add_derived("MdbNominalSurvivalMax", max(nominal_survivals), "fixed3", "largest nominal CL-1 survival over the catalogue", "max(catalogue.designs[*].nominal_survival_cl1)", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbNominalSurvivalMin", min(nominal_survivals), "sci1", "smallest nominal CL-1 survival over the catalogue", "min(catalogue.designs[*].nominal_survival_cl1)", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbNominalSurvivalMedian", statistics.median(nominal_survivals), "sci1", "median nominal CL-1 survival over the catalogue", "median(catalogue.designs[*].nominal_survival_cl1)", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbDivergentDesigns", sum(1 for d in designs if d["geometry"]["has_divergent_exit"] is True), "int", "catalogue designs with a divergent exit", "count(catalogue.designs[*].geometry.has_divergent_exit == true)", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbStageCounts", sorted({d["geometry"]["stage_count"] for d in designs}), "list_int", "magnet stage counts present in the catalogue", "sorted set of catalogue.designs[*].geometry.stage_count", [_inp("artifacts/catalogue.json", "/designs")])
    pooled_rank = {d["catalogue_index"]: rank for rank, d in enumerate(sorted(designs, key=lambda d: (d["pooled"]["probability"], d["catalogue_index"])), start=1)}

    # Dense reference.
    m.add("MdbDenseCount", "artifacts/dense-reference-summary.json", "/count", "int_comma", "dense reference design evaluations")
    m.add("MdbDenseDesigns", "artifacts/dense-reference-summary.json", "/designs", "int", "dense reference catalogue designs")
    m.add("MdbDensePointsPerDesign", "artifacts/dense-reference-summary.json", "/points_per_design", "int_comma", "dense reference operating points per design")
    m.add("MdbDenseFeasible", "artifacts/dense-reference-summary.json", "/feasible", "int_comma", "dense reference feasible evaluations")
    m.add("MdbDenseInfeasible", "artifacts/dense-reference-summary.json", "/infeasible", "int", "dense reference infeasible evaluations")
    m.add("MdbDenseRobustHv", "artifacts/dense-reference-summary.json", "/robust_hypervolume", "sci3", "dense reference robust hypervolume")
    m.add("MdbDenseNominalHv", "artifacts/dense-reference-summary.json", "/nominal_hypervolume", "sci3", "dense reference nominal hypervolume")
    m.add("MdbDenseRobustFront", "artifacts/dense-reference-summary.json", "/robust_front_size", "int", "dense reference robust front size")
    m.add("MdbDenseNominalFront", "artifacts/dense-reference-summary.json", "/nominal_front_size", "int", "dense reference nominal front size")
    m.add("MdbDenseRobustFrontDesigns", "artifacts/dense-reference-summary.json", "/robust_front_catalogue_indices", "list_int", "catalogue designs on the dense-reference robust front")
    m.add("MdbDenseNominalFrontDesigns", "artifacts/dense-reference-summary.json", "/nominal_front_catalogue_indices", "list_int", "catalogue designs on the dense-reference nominal front")
    m.add_derived("MdbDenseRobustFrontDesignCount", len(dense["robust_front_catalogue_indices"]), "int", "catalogue designs on the dense-reference robust front", "len(dense.robust_front_catalogue_indices)", [_inp("artifacts/dense-reference-summary.json", "/robust_front_catalogue_indices")])
    m.add("MdbDenseSeed", "artifacts/protocol.json", "/dense_reference/seed", "int", "dense reference seed")
    m.add("MdbDenseWorkers", "artifacts/dense-reference-summary.json", "/workers", "int", "dense reference worker processes")
    m.add("MdbDenseEvaluationSeconds", "artifacts/dense-reference-summary.json", "/evaluation_seconds", "fixed1", "dense reference evaluation time (s)")
    if dense["count"] != dense["designs"] * dense["points_per_design"] or dense["designs"] != len(designs):
        raise ValueError("dense reference count is not designs times points")
    if dense["feasible"] + dense["infeasible"] != dense["count"]:
        raise ValueError("dense reference feasible and infeasible counts do not sum to the count")
    for key in ("count", "designs", "points_per_design", "robust_hypervolume", "nominal_hypervolume", "robust_front_size", "nominal_front_size", "robust_front_catalogue_indices"):
        if dense[key] != metrics["dense_reference"][key]:
            raise ValueError(f"dense reference {key} differs between summary and metrics")
    for key in ("count", "designs", "points_per_design", "feasible", "infeasible", "columns_sha256"):
        if dense_full[key] != dense[key]:
            raise ValueError(f"dense reference {key} differs between summary and full record")
    for front in ("robust", "nominal"):
        block = dense_full["fronts"][front]
        if block["front_size"] != dense[f"{front}_front_size"] or block["hypervolume"] != dense[f"{front}_hypervolume"] or block["catalogue_indices"] != dense[f"{front}_front_catalogue_indices"]:
            raise ValueError(f"dense {front} front differs between summary and full record")
        if len(block["records"]) != block["front_size"] or sorted({r["design"]["catalogue_index"] for r in block["records"]}) != block["catalogue_indices"]:
            raise ValueError(f"dense {front} front records disagree with the summary")
    if len(dense_full["per_design"]) != len(designs) or any(row["catalogue_index"] != i for i, row in enumerate(dense_full["per_design"])):
        raise ValueError("dense per-design rows do not cover the catalogue in order")
    if sum(row["feasible"] + row["infeasible"] for row in dense_full["per_design"]) != dense["count"]:
        raise ValueError("dense per-design counts do not sum to the count")
    per_design_hv = [row["robust_hypervolume"] for row in dense_full["per_design"]]
    negligible = [row["catalogue_index"] for row in dense_full["per_design"] if row["robust_hypervolume"] < DENSE_NEGLIGIBLE_HYPERVOLUME]
    saturated_ids = {d["catalogue_index"] for d in designs if any(c["wall_hits"] == c["trials"] for c in d["cells"])}
    if not saturated_ids <= set(negligible):
        raise ValueError("a design with a saturated cell has a non-negligible dense hypervolume")
    m.add_derived("MdbDenseNegligibleThreshold", DENSE_NEGLIGIBLE_HYPERVOLUME, "sci1", "threshold below which a design's dense robust hypervolume is called negligible", "declared constant of this generator", [_inp("artifacts/dense-reference.json", "/per_design")])
    m.add_derived("MdbDenseNegligibleDesigns", len(negligible), "int", "catalogue designs whose own dense robust hypervolume is below the threshold", "count(dense.per_design[*].robust_hypervolume < threshold)", [_inp("artifacts/dense-reference.json", "/per_design")])
    m.add_derived("MdbDenseNegligibleSaturated", len(saturated_ids & set(negligible)), "int", "negligible-hypervolume designs that have a saturated cell", "count of designs below the threshold whose catalogue record has a cell with wall_hits equal to trials", [_inp("artifacts/dense-reference.json", "/per_design"), _inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbDenseNegligibleUnsaturated", len(set(negligible) - saturated_ids), "int", "negligible-hypervolume designs without a saturated cell", "count of designs below the threshold whose catalogue record has no saturated cell", [_inp("artifacts/dense-reference.json", "/per_design"), _inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbDenseNonNegligibleDesigns", len(designs) - len(negligible), "int", "catalogue designs with a non-negligible own dense robust hypervolume", "catalogue size minus the negligible count", [_inp("artifacts/dense-reference.json", "/per_design")])
    ranked_hv = sorted(((hv, i) for i, hv in enumerate(per_design_hv)), reverse=True)
    m.add_derived("MdbDenseTopDesign", ranked_hv[0][1], "int", "catalogue design with the largest own dense robust hypervolume", "argmax(dense.per_design[*].robust_hypervolume)", [_inp("artifacts/dense-reference.json", "/per_design")])
    m.add_derived("MdbDenseSecondDesign", ranked_hv[1][1], "int", "catalogue design with the second-largest own dense robust hypervolume", "second argmax", [_inp("artifacts/dense-reference.json", "/per_design")])
    m.add_derived("MdbDenseThirdDesign", ranked_hv[2][1], "int", "catalogue design with the third-largest own dense robust hypervolume", "third argmax", [_inp("artifacts/dense-reference.json", "/per_design")])
    m.add_derived("MdbDenseToBudgetRatio", dense["count"] / plan["evaluations_per_run"], "fixed0", "dense reference evaluations per optimiser-run evaluation budget", "dense_reference.count / plan.evaluations_per_run", [_inp("artifacts/dense-reference-summary.json", "/count"), _inp("artifacts/campaign-plan.json", "/evaluations_per_run")])

    # Per-run estimands and the hypervolume table.
    seeds = [int(s) for s in plan["seeds"]]
    hv_rows: list[str] = []
    failed_runs = 0
    infeasible_total = 0
    bo_acq: list[float] = []
    bo_fit: list[float] = []
    bo_candidate: list[float] = []
    bo_refine: list[float] = []
    bo_accepted: list[int] = []
    bo_refinements: list[int] = []
    per_strategy_hv: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    per_strategy_pareto: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    per_strategy_infeasible: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    per_strategy_wall: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    per_strategy_attained: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    per_strategy_distinct: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    shared_initial_ok = True
    for seed in seeds:
        firsts = []
        for strategy in STRATEGIES:
            records = runs[f"artifacts/runs/{strategy}-{seed}.json"]["records"][: plan["initial_design"]]
            firsts.append([(r["design"]["catalogue_index"], tuple(r["design"]["values"])) for r in records])
        shared_initial_ok = shared_initial_ok and firsts[0] == firsts[1] == firsts[2] and len({f[0] for f in firsts[0]}) == plan["initial_design"]
    if not shared_initial_ok:
        raise ValueError("shared initial designs are not identical across strategies or not distinct")
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        for seed, seed_token in zip(seeds, SEED_TOKENS, strict=True):
            key = f"{strategy}:{seed}"
            summary = metrics["runs"][key]
            table = metrics["hypervolume_table"][key]
            run = runs[f"artifacts/runs/{strategy}-{seed}.json"]
            curve = curves[key]
            if summary["evaluations"] != plan["evaluations_per_run"] or summary["budget"] != plan["evaluations_per_run"]:
                failed_runs += 1
            if summary["final_hypervolume"] != table["final_hypervolume"] or run["summary"] != summary:
                raise ValueError(f"metrics disagree for {key}")
            for field in ("attained_fraction_of_dense_reference", "distinct_catalogue_designs", "pareto_catalogue_indices", "pareto_set_size", "infeasible_evaluations", "unique_designs", "wall_clock_seconds"):
                if field != "attained_fraction_of_dense_reference" and table[field] != summary[field]:
                    raise ValueError(f"hypervolume table disagrees with the run summary for {key} ({field})")
            if curve[-1]["hypervolume"] != summary["final_hypervolume"] or len(curve) != summary["evaluations"]:
                raise ValueError(f"hypervolume curve disagrees with the summary for {key}")
            if any(b["hypervolume"] < a["hypervolume"] for a, b in zip(curve, curve[1:], strict=False)):
                raise ValueError(f"hypervolume curve is not monotone for {key}")
            if summary["feasible_evaluations"] + summary["infeasible_evaluations"] != summary["evaluations"]:
                raise ValueError(f"feasible and infeasible counts do not sum to the evaluations for {key}")
            if len(run["records"]) != summary["evaluations"]:
                raise ValueError(f"record count differs from the evaluations for {key}")
            indices = [r["design"]["catalogue_index"] for r in run["records"]]
            if len(set(indices)) != summary["distinct_catalogue_designs"]:
                raise ValueError(f"distinct catalogue designs do not reproduce for {key}")
            if len({r["design"]["design_id"] for r in run["records"]}) != summary["unique_designs"]:
                raise ValueError(f"unique designs do not reproduce for {key}")
            pareto = pareto_sets[key]
            if pareto["size"] != summary["pareto_set_size"] or pareto["replay_passed"] is not True or pareto["nondominated_recomputed"] is not True:
                raise ValueError(f"pareto set disagrees with the summary for {key}")
            if sorted({d["catalogue_index"] for d in pareto["designs"]}) != summary["pareto_catalogue_indices"] or pareto["catalogue_indices"] != summary["pareto_catalogue_indices"]:
                raise ValueError(f"pareto catalogue designs do not reproduce for {key}")
            if [d["index"] for d in pareto["designs"]] != summary["pareto_record_indices"]:
                raise ValueError(f"pareto record indices do not reproduce for {key}")
            if abs(table["attained_fraction_of_dense_reference"] - summary["final_hypervolume"] / dense["robust_hypervolume"]) > 1e-12:
                raise ValueError(f"attained fraction does not reproduce for {key}")
            if reported["hypervolume_vs_dense_reference"][key] != table["attained_fraction_of_dense_reference"]:
                raise ValueError(f"reported attained fraction differs for {key}")
            infeasible_total += summary["infeasible_evaluations"]
            per_strategy_hv[strategy].append(summary["final_hypervolume"])
            per_strategy_pareto[strategy].append(summary["pareto_set_size"])
            per_strategy_infeasible[strategy].append(summary["infeasible_evaluations"])
            per_strategy_wall[strategy].append(summary["wall_clock_seconds"])
            per_strategy_attained[strategy].append(table["attained_fraction_of_dense_reference"])
            per_strategy_distinct[strategy].append(summary["distinct_catalogue_designs"])
            base = f"/runs/{key}"
            m.add(f"MdbHv{token}{seed_token}", "artifacts/metrics.json", f"{base}/final_hypervolume", "sci3", f"final robust hypervolume, {key}")
            m.add(f"MdbPareto{token}{seed_token}", "artifacts/metrics.json", f"{base}/pareto_set_size", "int", f"final Pareto-set size, {key}")
            m.add(f"MdbParetoDesigns{token}{seed_token}", "artifacts/metrics.json", f"{base}/pareto_catalogue_indices", "list_int", f"catalogue designs on the final Pareto set, {key}")
            m.add(f"MdbDistinct{token}{seed_token}", "artifacts/metrics.json", f"{base}/distinct_catalogue_designs", "int", f"distinct catalogue designs evaluated, {key}")
            m.add(f"MdbInfeasible{token}{seed_token}", "artifacts/metrics.json", f"{base}/infeasible_evaluations", "int", f"infeasible evaluations, {key}")
            m.add(f"MdbAttained{token}{seed_token}", "artifacts/metrics.json", f"/hypervolume_table/{key}/attained_fraction_of_dense_reference", "fixed2", f"fraction of the dense-reference robust hypervolume, {key}")
            m.add(f"MdbWall{token}{seed_token}", "artifacts/metrics.json", f"{base}/wall_clock_seconds", "fixed0", f"run wall time (s), {key}")
            timing = metrics["timing"][key]
            optimizer = run["optimizer"]
            if strategy == "qlognehvi":
                bo_acq.append(timing["bo_acquisition_seconds"])
                bo_fit.append(timing["bo_fit_seconds"])
                log = optimizer["iteration_log"]
                if len(log) != plan["qlognehvi_iterations"] or any(len(e["refinement"]) != plan["qlognehvi_batch_size"] for e in log):
                    raise ValueError(f"BO iteration log differs from the plan for {key}")
                bo_candidate.append(sum(e["candidate_stage_seconds"] for e in log))
                bo_refine.append(sum(e["refinement_seconds"] for e in log))
                bo_accepted.append(sum(1 for e in log for r in e["refinement"] if r["accepted"] is True))
                bo_refinements.append(sum(len(e["refinement"]) for e in log))
                arguments = optimizer["arguments"]
                if arguments["q"] != plan["qlognehvi_batch_size"] or arguments["mc_samples"] != protocol["optimizers"]["qlognehvi"]["mc_samples"] or arguments["device"] != "cpu" or arguments["torch_threads"] != protocol["optimizers"]["qlognehvi"]["torch_threads"]:
                    raise ValueError(f"BO arguments differ from the protocol for {key}")
                for needle in ("MixedSingleTaskGP", "CategoricalKernel", "MaternKernel", "Standardize"):
                    if needle not in optimizer["model"]:
                        raise ValueError(f"BO model label lacks {needle} for {key}")
                if "optimize_acqf_discrete" not in optimizer["acquisition"] or "refinement stage" not in optimizer["acquisition"]:
                    raise ValueError(f"BO acquisition label does not describe the two stages for {key}")
            elif strategy == "nsga3":
                if optimizer.get("iteration_log"):
                    raise ValueError(f"non-BO run carries a BO iteration log: {key}")
                if optimizer["eliminate_duplicates"] is not True or optimizer["declared_generations"] != plan["nsga3_generations"] or optimizer["pymoo_n_gen"] != plan["nsga3_generations"] + 1:
                    raise ValueError(f"NSGA-III labels differ from the plan for {key}")
                if optimizer["pymoo_reported_evaluations"] != plan["evaluations_per_run"] or optimizer["unique_designs"] != summary["unique_designs"] or summary["unique_designs"] != summary["evaluations"]:
                    raise ValueError(f"NSGA-III evaluations or unique designs differ for {key}")
                if duplicates[key]["duplicates"] != 0 or duplicates[key]["evaluations"] != summary["evaluations"] or duplicates[key]["unique_designs"] != summary["unique_designs"]:
                    raise ValueError(f"NSGA-III duplicate record differs for {key}")
            else:
                if optimizer["stages"] != [plan["initial_design"], plan["evaluations_per_run"] - plan["initial_design"]] or optimizer["points"] != plan["evaluations_per_run"]:
                    raise ValueError(f"LHS stages differ from the plan for {key}")
            hv_rows.append(
                f"{STRATEGY_LABELS[strategy]} & {seed} & {format_value('sci3', summary['final_hypervolume'])} & "
                f"{format_value('fixed2', table['attained_fraction_of_dense_reference'])} & {summary['pareto_set_size']} & "
                f"{format_value('list_int', summary['pareto_catalogue_indices'])} & {summary['distinct_catalogue_designs']} & "
                f"{summary['infeasible_evaluations']} & {format_value('fixed0', summary['wall_clock_seconds'])}\\\\"
            )
    if infeasible_total != campaign["infeasible_evaluations"]:
        raise ValueError("per-run infeasible counts do not sum to the campaign total")
    if sum(len(v) for v in per_strategy_hv.values()) * plan["evaluations_per_run"] != campaign["total_evaluations"]:
        raise ValueError("runs times budget differs from the total evaluations")
    m.add_derived("MdbFailedRuns", failed_runs, "int", "runs that did not record exactly the budget", "count of runs with evaluations != plan.evaluations_per_run", [_inp("artifacts/metrics.json", "/runs")])
    m.add_derived("MdbSharedInitialIdentical", shared_initial_ok, "bool", "whether the first initial-design records are identical across the three optimisers for every seed", "equality of (catalogue index, operating point) of the first initial_design records across strategies per seed, with initial_design distinct catalogue designs", [_inp(f"artifacts/runs/{s}-{seed}.json", "/records") for seed in seeds for s in STRATEGIES])
    summary_rows: list[str] = []
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        hv = per_strategy_hv[strategy]
        recorded = metrics["seed_variance"][strategy]
        if abs(statistics.mean(hv) - recorded["mean"]) > 1e-15 or abs(statistics.stdev(hv) - recorded["sample_std"]) > 1e-15:
            raise ValueError(f"seed variance does not reproduce for {strategy}")
        if min(hv) != recorded["minimum"] or max(hv) != recorded["maximum"]:
            raise ValueError(f"seed extrema do not reproduce for {strategy}")
        m.add(f"MdbHv{token}Mean", "artifacts/metrics.json", f"/seed_variance/{strategy}/mean", "sci3", f"mean final hypervolume, {strategy}")
        m.add(f"MdbHv{token}Std", "artifacts/metrics.json", f"/seed_variance/{strategy}/sample_std", "sci2", f"sample standard deviation of the final hypervolume, {strategy}")
        m.add(f"MdbHv{token}Min", "artifacts/metrics.json", f"/seed_variance/{strategy}/minimum", "sci3", f"minimum final hypervolume, {strategy}")
        m.add(f"MdbHv{token}Max", "artifacts/metrics.json", f"/seed_variance/{strategy}/maximum", "sci3", f"maximum final hypervolume, {strategy}")
        strategy_inputs = [_inp("artifacts/metrics.json", f"/runs/{strategy}:{seed}") for seed in seeds]
        attained_inputs = [_inp("artifacts/metrics.json", f"/hypervolume_table/{strategy}:{seed}/attained_fraction_of_dense_reference") for seed in seeds]
        m.add_derived(f"MdbAttained{token}Mean", statistics.mean(per_strategy_attained[strategy]), "fixed2", f"mean fraction of the dense-reference robust hypervolume, {strategy}", "mean over seeds of hypervolume_table[*].attained_fraction_of_dense_reference", attained_inputs)
        m.add_derived(f"MdbAttained{token}Min", min(per_strategy_attained[strategy]), "fixed2", f"minimum attained fraction, {strategy}", "min over seeds", attained_inputs)
        m.add_derived(f"MdbAttained{token}Max", max(per_strategy_attained[strategy]), "fixed2", f"maximum attained fraction, {strategy}", "max over seeds", attained_inputs)
        m.add_derived(f"MdbPareto{token}Min", min(per_strategy_pareto[strategy]), "int", f"minimum Pareto-set size, {strategy}", "min over seeds of runs[*].pareto_set_size", strategy_inputs)
        m.add_derived(f"MdbPareto{token}Max", max(per_strategy_pareto[strategy]), "int", f"maximum Pareto-set size, {strategy}", "max over seeds of runs[*].pareto_set_size", strategy_inputs)
        m.add_derived(f"MdbInfeasible{token}Min", min(per_strategy_infeasible[strategy]), "int", f"minimum infeasible evaluations, {strategy}", "min over seeds of runs[*].infeasible_evaluations", strategy_inputs)
        m.add_derived(f"MdbInfeasible{token}Max", max(per_strategy_infeasible[strategy]), "int", f"maximum infeasible evaluations, {strategy}", "max over seeds of runs[*].infeasible_evaluations", strategy_inputs)
        m.add_derived(f"MdbInfeasible{token}Total", sum(per_strategy_infeasible[strategy]), "int", f"infeasible evaluations over the seeds, {strategy}", "sum over seeds of runs[*].infeasible_evaluations", strategy_inputs)
        m.add_derived(f"MdbDistinct{token}Min", min(per_strategy_distinct[strategy]), "int", f"fewest distinct catalogue designs in a run, {strategy}", "min over seeds of runs[*].distinct_catalogue_designs", strategy_inputs)
        m.add_derived(f"MdbDistinct{token}Max", max(per_strategy_distinct[strategy]), "int", f"most distinct catalogue designs in a run, {strategy}", "max over seeds of runs[*].distinct_catalogue_designs", strategy_inputs)
        m.add_derived(f"MdbWall{token}Max", max(per_strategy_wall[strategy]), "fixed0", f"maximum run wall time (s), {strategy}", "max over seeds of runs[*].wall_clock_seconds", strategy_inputs)
        m.add_derived(f"MdbWall{token}Min", min(per_strategy_wall[strategy]), "fixed0", f"minimum run wall time (s), {strategy}", "min over seeds of runs[*].wall_clock_seconds", strategy_inputs)
        summary_rows.append(
            f"{STRATEGY_LABELS[strategy]} & mean $\\pm$ s & ${format_value('sci3', recorded['mean'])[1:-1]} \\pm {format_value('sci2', recorded['sample_std'])[1:-1]}$ & "
            f"{format_value('fixed2', statistics.mean(per_strategy_attained[strategy]))} & "
            f"{format_value('fixed1', statistics.mean(per_strategy_pareto[strategy]))} & --- & "
            f"{format_value('fixed1', statistics.mean(per_strategy_distinct[strategy]))} & "
            f"{format_value('fixed1', statistics.mean(per_strategy_infeasible[strategy]))} & "
            f"{format_value('fixed0', statistics.mean(per_strategy_wall[strategy]))}\\\\"
        )
    bo_inputs = [_inp("artifacts/metrics.json", f"/timing/qlognehvi:{seed}") for seed in seeds]
    log_inputs = [_inp(f"artifacts/runs/qlognehvi-{seed}.json", "/optimizer/iteration_log") for seed in seeds]
    m.add_derived("MdbBoAcqSecondsMin", min(bo_acq), "fixed0", "minimum BO acquisition seconds per seed", "min over BO seeds of timing[*].bo_acquisition_seconds", bo_inputs)
    m.add_derived("MdbBoAcqSecondsMax", max(bo_acq), "fixed0", "maximum BO acquisition seconds per seed", "max over BO seeds of timing[*].bo_acquisition_seconds", bo_inputs)
    m.add_derived("MdbBoFitSecondsMin", min(bo_fit), "fixed0", "minimum BO fit seconds per seed", "min over BO seeds of timing[*].bo_fit_seconds", bo_inputs)
    m.add_derived("MdbBoFitSecondsMax", max(bo_fit), "fixed0", "maximum BO fit seconds per seed", "max over BO seeds of timing[*].bo_fit_seconds", bo_inputs)
    m.add_derived("MdbBoCandidateSecondsMin", min(bo_candidate), "fixed0", "minimum discrete candidate-stage seconds per seed", "min over BO seeds of sum(iteration_log[*].candidate_stage_seconds)", log_inputs)
    m.add_derived("MdbBoCandidateSecondsMax", max(bo_candidate), "fixed0", "maximum discrete candidate-stage seconds per seed", "max over BO seeds of sum(iteration_log[*].candidate_stage_seconds)", log_inputs)
    m.add_derived("MdbBoRefineSecondsMin", min(bo_refine), "fixed0", "minimum refinement-stage seconds per seed", "min over BO seeds of sum(iteration_log[*].refinement_seconds)", log_inputs)
    m.add_derived("MdbBoRefineSecondsMax", max(bo_refine), "fixed0", "maximum refinement-stage seconds per seed", "max over BO seeds of sum(iteration_log[*].refinement_seconds)", log_inputs)
    m.add_derived("MdbBoRefinementsAccepted", sum(bo_accepted), "int", "refined operating points accepted over the three BO seeds", "count(iteration_log[*].refinement[*].accepted == true) summed over seeds", log_inputs)
    m.add_derived("MdbBoRefinements", sum(bo_refinements), "int", "refinement attempts over the three BO seeds", "count(iteration_log[*].refinement[*]) summed over seeds", log_inputs)
    m.add("MdbAssessmentSeconds", "artifacts/campaign-result.json", "/assessment_seconds", "fixed0", "assessment stage wall time (s)")
    first = m.doc("transitions/0001-lock-acquired.json")
    last = m.doc("transitions/0009-terminal.json")
    m.add_derived(
        "MdbLifecycleWallMin", _seconds(first["recorded_at_utc"]["value"], last["recorded_at_utc"]["value"]), "min1",
        "lock-acquired to terminal wall time (min)", "(terminal.recorded_at_utc - lock_acquired.recorded_at_utc) / 60",
        [_inp("transitions/0001-lock-acquired.json", "/recorded_at_utc/value"), _inp("transitions/0009-terminal.json", "/recorded_at_utc/value")],
    )
    m.add("MdbCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "host CPU count")
    m.add("MdbCudaDevice", "artifacts/device-probes.json", "/cuda/device_name", "text", "CUDA device probed and not used")

    # Seed 101: the optimiser converged on one design and never evaluated the design of the other two seeds' fronts.
    bo_a_indices = [r["design"]["catalogue_index"] for r in runs[f"artifacts/runs/qlognehvi-{seeds[0]}.json"]["records"]]
    bo_a_pareto = metrics["runs"][f"qlognehvi:{seeds[0]}"]["pareto_catalogue_indices"]
    if len(bo_a_pareto) != 1:
        raise ValueError("the first BO seed's Pareto set does not lie on a single catalogue design")
    stall_design = bo_a_pareto[0]
    other_fronts = set(metrics["runs"][f"qlognehvi:{seeds[1]}"]["pareto_catalogue_indices"]) & set(metrics["runs"][f"qlognehvi:{seeds[2]}"]["pareto_catalogue_indices"])
    if len(other_fronts) != 1:
        raise ValueError("the other two BO seeds do not share exactly one Pareto design")
    missed_design = next(iter(other_fronts))
    if missed_design in bo_a_indices:
        raise ValueError("the first BO seed did evaluate the design shared by the other seeds' fronts")
    bo_a_inputs = [_inp(f"artifacts/runs/qlognehvi-{seeds[0]}.json", "/records"), _inp("artifacts/metrics.json", "/runs")]
    m.add_derived("MdbBoAStallDesign", stall_design, "int", "the single catalogue design on the first BO seed's final Pareto set", "metrics.runs['qlognehvi:<first seed>'].pareto_catalogue_indices[0] (a one-element list)", bo_a_inputs)
    m.add_derived("MdbBoAStallEvaluations", bo_a_indices.count(stall_design), "int", "evaluations the first BO seed spent on that design", "count of records of the first BO seed whose catalogue index equals the stall design", bo_a_inputs)
    m.add_derived("MdbBoAMissedDesign", missed_design, "int", "the catalogue design shared by the other two BO seeds' Pareto sets and never evaluated by the first", "the single design in the intersection of the second and third BO seeds' pareto_catalogue_indices", bo_a_inputs)
    m.add_derived("MdbBoAMissedEvaluations", bo_a_indices.count(missed_design), "int", "evaluations the first BO seed spent on the missed design", "count of records of the first BO seed whose catalogue index equals the missed design", bo_a_inputs)
    m.add_derived("MdbBoAMissedInInitial", missed_design in bo_a_indices[: plan["initial_design"]], "bool", "whether the missed design was in the first seed's shared initial design", "membership of the missed design in the first initial_design records of the first BO seed", bo_a_inputs)
    m.add_derived("MdbBoAFirstSeed", seeds[0], "int", "the first evidentiary seed", "plan.seeds[0]", [_inp("artifacts/campaign-plan.json", "/seeds")])
    m.add_derived("MdbBoOtherSeeds", seeds[1:], "list_int", "the other evidentiary seeds", "plan.seeds[1:]", [_inp("artifacts/campaign-plan.json", "/seeds")])
    for seed, seed_token in zip(seeds[1:], SEED_TOKENS[1:], strict=True):
        indices = [r["design"]["catalogue_index"] for r in runs[f"artifacts/runs/qlognehvi-{seed}.json"]["records"]]
        m.add_derived(f"MdbBo{seed_token}MissedInInitial", missed_design in indices[: plan["initial_design"]], "bool", f"whether the design missed by the first seed was in seed {seed}'s shared initial design", "membership of the missed design in the first initial_design records", [_inp(f"artifacts/runs/qlognehvi-{seed}.json", "/records")])

    # Predeclared comparisons (reported, not binding).
    for key, token in (("bo_beats_random", "Random"), ("bo_beats_nsga3", "Nsga")):
        block = reported[key]
        if block["wins"] != sum(1 for p in block["pairs"] if p["left_wins"] is True) or block["seeds"] != len(block["pairs"]):
            raise ValueError(f"{key} wins do not reproduce from the pairs")
        right = block["right"]
        for pair in block["pairs"]:
            seed = pair["seed"]
            if pair["qlognehvi"] != metrics["runs"][f"qlognehvi:{seed}"]["final_hypervolume"] or pair[right] != metrics["runs"][f"{right}:{seed}"]["final_hypervolume"]:
                raise ValueError(f"{key} pair values differ from the run summaries")
            if pair["left_wins"] is not (pair["qlognehvi"] > pair[right]):
                raise ValueError(f"{key} left_wins flag is inconsistent")
        if block["passed"] is not (block["wins"] >= block["required_wins"]):
            raise ValueError(f"{key} pass flag is inconsistent")
        if campaign[f"{key}_wins"] != f"{block['wins']}/{block['seeds']}" or campaign[key] is not block["passed"]:
            raise ValueError(f"campaign-result {key} differs from gates.json")
        if "a count, not a significance statement" not in block["statement"]:
            raise ValueError(f"{key} statement does not disclaim significance")
        m.add(f"MdbBoBeats{token}Wins", "artifacts/gates.json", f"/reported_not_binding/{key}/wins", "int", f"{key}: seeds won by qLogNEHVI")
        m.add(f"MdbBoBeats{token}Seeds", "artifacts/gates.json", f"/reported_not_binding/{key}/seeds", "int", f"{key}: paired seeds")
        m.add(f"MdbBoBeats{token}Required", "artifacts/gates.json", f"/reported_not_binding/{key}/required_wins", "int", f"{key}: predeclared required wins")
        m.add(f"MdbBoBeats{token}Passed", "artifacts/gates.json", f"/reported_not_binding/{key}/passed", "bool", f"{key}: predeclared comparison outcome")

    # Robust versus nominal (pooled designs) and the catalogue designs on the fronts.
    rvn = reported["robust_vs_nominal"]
    if (
        rvn["robust_front_size"] != pooled["robust"]["front_size"]
        or rvn["nominal_front_size"] != pooled["nominal"]["front_size"]
        or rvn["shared_designs"] != len(pooled["shared_design_ids"])
        or rvn["jaccard"] != pooled["jaccard_robust_nominal"]
        or rvn["robust_hypervolume"] != pooled["robust"]["hypervolume"]
        or rvn["nominal_hypervolume"] != pooled["nominal"]["hypervolume"]
        or rvn["nominal_front_members_robust_feasible"] != pooled["nominal"]["robust_feasible_members"]
        or rvn["robust_front_catalogue_indices"] != pooled["robust"]["catalogue_indices"]
        or rvn["nominal_front_catalogue_indices"] != pooled["nominal"]["catalogue_indices"]
    ):
        raise ValueError("robust-versus-nominal summary differs from the pooled fronts")
    shared = set(pooled["robust"]["design_ids"]) & set(pooled["nominal"]["design_ids"])
    if shared != set(pooled["shared_design_ids"]) or len(pooled["robust"]["design_ids"]) != pooled["robust"]["front_size"] or len(pooled["nominal"]["design_ids"]) != pooled["nominal"]["front_size"]:
        raise ValueError("shared design ids do not reproduce from the fronts")
    union = pooled["robust"]["front_size"] + pooled["nominal"]["front_size"] - len(shared)
    if abs(len(shared) / union - pooled["jaccard_robust_nominal"]) > 1e-15:
        raise ValueError("Jaccard index does not reproduce")
    for front in ("robust", "nominal"):
        block = pooled[front]
        if len(block["designs"]) != block["front_size"] or sorted({d["catalogue_index"] for d in block["designs"]}) != block["catalogue_indices"]:
            raise ValueError(f"{front} front designs disagree with the front size or catalogue indices")
        if [d["catalogue_index"] for d in block["catalogue_membership"]] != block["catalogue_indices"] or sum(d["front_members"] for d in block["catalogue_membership"]) != block["front_size"]:
            raise ValueError(f"{front} front catalogue membership does not sum to the front")
        objective_key = f"{front}_objectives"
        for name in OBJECTIVES:
            values = [d[objective_key][name] for d in block["designs"]]
            rng = block["objective_ranges"][name]
            if min(values) != rng["minimum"] or max(values) != rng["maximum"]:
                raise ValueError(f"{front} objective range does not reproduce for {name}")
        for member in block["catalogue_membership"]:
            design = designs[member["catalogue_index"]]
            for field in ("case_id", "cells", "geometry", "geometry_sha256", "pooled", "nominal_survival_cl1", "nominal_survival_cl2", "screening_design_id", "design_values"):
                if member[field] != design[field]:
                    raise ValueError(f"{front} front catalogue member {member['catalogue_index']} differs from the sealed catalogue ({field})")
            if member["front_members"] != sum(1 for d in block["designs"] if d["catalogue_index"] == member["catalogue_index"]) or len(member["operating_points"]) != member["front_members"]:
                raise ValueError(f"{front} front membership count does not reproduce for design {member['catalogue_index']}")
    if any(d["constraints"]["robust_beam_current_margin_a"] < 0 for d in pooled["robust"]["designs"]):
        raise ValueError("a robust-front design violates the robust margin")
    if any(d["constraints"]["nominal_beam_current_margin_a"] < 0 for d in pooled["nominal"]["designs"]):
        raise ValueError("a nominal-front design violates the nominal margin")
    for entry in campaign["robust_front_catalogue_designs"]:
        design = designs[entry["catalogue_index"]]
        member = next(x for x in pooled["robust"]["catalogue_membership"] if x["catalogue_index"] == entry["catalogue_index"])
        if (
            entry["case_id"] != design["case_id"] or entry["geometry"] != design["geometry"] or entry["front_members"] != member["front_members"]
            or entry["cell_wall_hit_probabilities"] != [c["probability"] for c in design["cells"]] or entry["pooled_wall_hit_probability"] != design["pooled"]["probability"]
            or entry["pooled_wilson_95"] != design["pooled"]["wilson_95"] or entry["nominal_survival_cl1"] != design["nominal_survival_cl1"]
        ):
            raise ValueError(f"campaign-result robust-front design {entry['catalogue_index']} differs from the catalogue")
    if [e["catalogue_index"] for e in campaign["robust_front_catalogue_designs"]] != pooled["robust"]["catalogue_indices"]:
        raise ValueError("campaign-result robust-front designs differ from the pooled front")
    m.add("MdbUniqueDesigns", "artifacts/pooled-fronts.json", "/unique_designs", "int_comma", "unique evaluated designs pooled over all runs")
    m.add("MdbDistinctCatalogueDesigns", "artifacts/pooled-fronts.json", "/distinct_catalogue_designs", "int", "catalogue designs evaluated at least once over all runs")
    m.add("MdbRobustCandidates", "artifacts/pooled-fronts.json", "/robust/candidates", "int_comma", "robust-feasible pooled designs")
    m.add("MdbNominalCandidates", "artifacts/pooled-fronts.json", "/nominal/candidates", "int_comma", "nominally feasible pooled designs")
    m.add("MdbRobustFront", "artifacts/pooled-fronts.json", "/robust/front_size", "int", "pooled robust front size")
    m.add("MdbNominalFront", "artifacts/pooled-fronts.json", "/nominal/front_size", "int", "pooled nominal front size")
    m.add("MdbPooledRobustHv", "artifacts/pooled-fronts.json", "/robust/hypervolume", "sci3", "pooled robust hypervolume")
    m.add("MdbPooledNominalHv", "artifacts/pooled-fronts.json", "/nominal/hypervolume", "sci3", "pooled nominal hypervolume")
    m.add_derived("MdbSharedDesigns", len(pooled["shared_design_ids"]), "int", "designs on both pooled fronts", "len(pooled.shared_design_ids)", [_inp("artifacts/pooled-fronts.json", "/shared_design_ids")])
    m.add("MdbJaccard", "artifacts/pooled-fronts.json", "/jaccard_robust_nominal", "fixed2", "Jaccard index of the robust and nominal fronts")
    m.add("MdbNominalRobustFeasible", "artifacts/pooled-fronts.json", "/nominal/robust_feasible_members", "int", "nominal-front designs that are robust-feasible")
    m.add("MdbRobustFrontDesigns", "artifacts/pooled-fronts.json", "/robust/catalogue_indices", "list_int", "catalogue designs on the pooled robust front")
    m.add("MdbNominalFrontDesigns", "artifacts/pooled-fronts.json", "/nominal/catalogue_indices", "list_int", "catalogue designs on the pooled nominal front")
    m.add_derived("MdbRobustFrontDesignCount", len(pooled["robust"]["catalogue_indices"]), "int", "catalogue designs on the pooled robust front", "len(pooled.robust.catalogue_indices)", [_inp("artifacts/pooled-fronts.json", "/robust/catalogue_indices")])
    m.add_derived("MdbNominalFrontDesignCount", len(pooled["nominal"]["catalogue_indices"]), "int", "catalogue designs on the pooled nominal front", "len(pooled.nominal.catalogue_indices)", [_inp("artifacts/pooled-fronts.json", "/nominal/catalogue_indices")])
    nominal_only = sorted(set(pooled["nominal"]["catalogue_indices"]) - set(pooled["robust"]["catalogue_indices"]))
    if len(nominal_only) != 1:
        raise ValueError("the nominal front does not add exactly one catalogue design to the robust front's designs")
    m.add_derived("MdbNominalOnlyDesign", nominal_only[0], "int", "the catalogue design on the nominal front but not on the robust front", "set difference of the nominal and robust front catalogue indices (one design)", [_inp("artifacts/pooled-fronts.json", "/nominal/catalogue_indices"), _inp("artifacts/pooled-fronts.json", "/robust/catalogue_indices")])
    m.add_derived("MdbNominalOnlyDesignRank", pooled_rank[nominal_only[0]], "int", "ascending pooled wall-hit rank of that design", "rank of the design's pooled probability among the catalogue (ties by index)", [_inp("artifacts/catalogue.json", "/designs")])
    robust_ranks = [pooled_rank[i] for i in pooled["robust"]["catalogue_indices"]]
    if sorted(robust_ranks) != list(range(1, len(robust_ranks) + 1)):
        raise ValueError("the robust-front designs are not the lowest-ranked pooled wall-hit designs")
    m.add_derived("MdbRobustFrontLowestRanks", sorted(robust_ranks) == list(range(1, len(robust_ranks) + 1)), "bool", "whether the robust-front designs are exactly the lowest pooled wall-hit designs of the catalogue", "the ascending pooled wall-hit ranks of pooled.robust.catalogue_indices equal 1..k", [_inp("artifacts/pooled-fronts.json", "/robust/catalogue_indices"), _inp("artifacts/catalogue.json", "/designs")])
    range_fmt = {"axial_thrust_n": "sig3", "specific_impulse_s": "fixed0", "thruster_electrical_to_beam_efficiency": "fixed3", "anode_input_power_w": "fixed1"}
    range_token = {"axial_thrust_n": "Thrust", "specific_impulse_s": "Isp", "thruster_electrical_to_beam_efficiency": "Eff", "anode_input_power_w": "Power"}
    for front, ftoken in (("robust", "Robust"), ("nominal", "Nominal")):
        for name in OBJECTIVES:
            for bound, btoken in (("minimum", "Min"), ("maximum", "Max")):
                m.add(f"Mdb{ftoken}{range_token[name]}{btoken}", "artifacts/pooled-fronts.json", f"/{front}/objective_ranges/{name}/{bound}", range_fmt[name], f"{front} front {name} {bound}")
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        block = per_strategy[strategy]
        if len(block["robust"]["design_ids"]) != block["robust"]["front_size"] or metrics["per_strategy_pooled"][strategy]["robust_hypervolume"] != block["robust"]["hypervolume"]:
            raise ValueError(f"per-strategy robust front differs from its size or metrics for {strategy}")
        if metrics["per_strategy_pooled"][strategy]["robust_front_catalogue_indices"] != block["robust"]["catalogue_indices"]:
            raise ValueError(f"per-strategy robust front designs differ from metrics for {strategy}")
        m.add(f"MdbPooled{token}RobustFront", "artifacts/per-strategy-fronts.json", f"/{strategy}/robust/front_size", "int", f"pooled robust front size, {strategy}")
        m.add(f"MdbPooled{token}RobustHv", "artifacts/per-strategy-fronts.json", f"/{strategy}/robust/hypervolume", "sci3", f"pooled robust hypervolume, {strategy}")
        m.add(f"MdbPooled{token}RobustDesigns", "artifacts/per-strategy-fronts.json", f"/{strategy}/robust/catalogue_indices", "list_int", f"catalogue designs on the pooled robust front, {strategy}")
        m.add(f"MdbPooled{token}Distinct", "artifacts/per-strategy-fronts.json", f"/{strategy}/distinct_catalogue_designs", "int", f"catalogue designs evaluated, {strategy}")

    # Catalogue table: every design on the dense-reference robust front, with its pooled-front membership.
    membership = {x["catalogue_index"]: x["front_members"] for x in pooled["robust"]["catalogue_membership"]}
    table_designs = list(dense["robust_front_catalogue_indices"])
    if set(pooled["robust"]["catalogue_indices"]) - set(table_designs):
        raise ValueError("a pooled robust-front design is absent from the dense robust front")
    if set(table_designs) != set(DESIGN_TOKENS):
        raise ValueError("dense robust-front designs differ from the admitted token layout")
    catalogue_rows: list[str] = []
    for index in table_designs:
        token = DESIGN_TOKENS[index]
        design = designs[index]
        base = f"/designs/{index}"
        m.add(f"MdbDesign{token}Index", "artifacts/catalogue.json", f"{base}/catalogue_index", "int", f"catalogue index of design {index}")
        m.add(f"MdbDesign{token}CaseId", "artifacts/catalogue.json", f"{base}/case_id", "ident", f"screening case id of design {index}")
        m.add(f"MdbDesign{token}PooledP", "artifacts/catalogue.json", f"{base}/pooled/probability", "fixed3", f"pooled wall-hit probability of design {index}")
        m.add(f"MdbDesign{token}PooledLo", "artifacts/catalogue.json", f"{base}/pooled/wilson_95/0", "fixed3", f"pooled Wilson lower bound of design {index}")
        m.add(f"MdbDesign{token}PooledHi", "artifacts/catalogue.json", f"{base}/pooled/wilson_95/1", "fixed3", f"pooled Wilson upper bound of design {index}")
        m.add(f"MdbDesign{token}PooledHits", "artifacts/catalogue.json", f"{base}/pooled/wall_hits", "int", f"pooled wall hits of design {index}")
        m.add(f"MdbDesign{token}Reflected", "artifacts/catalogue.json", f"{base}/pooled/reflected", "int", f"reflections recorded for design {index}")
        for k, cell_token in enumerate(CELL_TOKENS):
            m.add(f"MdbDesign{token}Cell{cell_token}", "artifacts/catalogue.json", f"{base}/cells/{k}/probability", "fixed3", f"cell {k + 1} wall-hit probability of design {index}")
        m.add(f"MdbDesign{token}Survival", "artifacts/catalogue.json", f"{base}/nominal_survival_cl1", "fixed3", f"nominal CL-1 survival of design {index}")
        m.add(f"MdbDesign{token}SurvivalPooled", "artifacts/catalogue.json", f"{base}/nominal_survival_cl2", "fixed3", f"nominal CL-2 survival of design {index}")
        m.add(f"MdbDesign{token}LengthMm", "artifacts/catalogue.json", f"{base}/geometry/chamber_length_m", "mm1", f"chamber length of design {index} (mm)")
        m.add(f"MdbDesign{token}RadiusMm", "artifacts/catalogue.json", f"{base}/geometry/wall_radius_m", "mm2", f"wall radius of design {index} (mm)")
        m.add(f"MdbDesign{token}PitchMm", "artifacts/catalogue.json", f"{base}/geometry/stage_pitch_m", "mm1", f"stage pitch of design {index} (mm)")
        m.add(f"MdbDesign{token}Stages", "artifacts/catalogue.json", f"{base}/geometry/stage_count", "int", f"magnet stages of design {index}")
        m.add(f"MdbDesign{token}Divergent", "artifacts/catalogue.json", f"{base}/geometry/has_divergent_exit", "yesno", f"divergent exit of design {index}")
        m.add(f"MdbDesign{token}DenseHv", "artifacts/dense-reference.json", f"/per_design/{index}/robust_hypervolume", "sci3", f"own dense robust hypervolume of design {index}")
        m.add_derived(f"MdbDesign{token}Members", membership.get(index, 0), "int", f"pooled robust-front members on design {index}", "pooled.robust.catalogue_membership[*].front_members for the design, zero if absent", [_inp("artifacts/pooled-fronts.json", "/robust/catalogue_membership")])
        m.add_derived(f"MdbDesign{token}Rank", pooled_rank[index], "int", f"ascending pooled wall-hit rank of design {index}", "rank of the design's pooled probability among the catalogue (ties by index)", [_inp("artifacts/catalogue.json", "/designs")])
        geometry = design["geometry"]
        catalogue_rows.append(
            f"{index} & {pooled_rank[index]} & {membership.get(index, 0)} & "
            f"{format_value('fixed3', design['pooled']['probability'])} [{format_value('fixed3', design['pooled']['wilson_95'][0])}, {format_value('fixed3', design['pooled']['wilson_95'][1])}] & "
            + " / ".join(format_value("fixed3", c["probability"]) for c in design["cells"]) + " & "
            f"{format_value('fixed3', design['nominal_survival_cl1'])} & {format_value('mm1', geometry['chamber_length_m'])} & "
            f"{format_value('mm2', geometry['wall_radius_m'])} & {geometry['stage_count']} & {format_value('mm1', geometry['stage_pitch_m'])} & "
            f"{format_value('yesno', geometry['has_divergent_exit'])} & {format_value('sci3', per_design_hv[index])}\\\\"
        )
    front_design_ids = pooled["robust"]["catalogue_indices"]
    m.add_derived("MdbRobustFrontDesignsAllFiveStage", all(designs[i]["geometry"]["stage_count"] == 5 for i in front_design_ids), "bool", "whether every pooled robust-front design has five stages", "all(catalogue.designs[i].geometry.stage_count == 5 for i in pooled.robust.catalogue_indices)", [_inp("artifacts/catalogue.json", "/designs"), _inp("artifacts/pooled-fronts.json", "/robust/catalogue_indices")])
    m.add_derived("MdbRobustFrontDesignsAllDivergent", all(designs[i]["geometry"]["has_divergent_exit"] is True for i in front_design_ids), "bool", "whether every pooled robust-front design has a divergent exit", "all(catalogue.designs[i].geometry.has_divergent_exit for i in pooled.robust.catalogue_indices)", [_inp("artifacts/catalogue.json", "/designs"), _inp("artifacts/pooled-fronts.json", "/robust/catalogue_indices")])
    m.add_derived("MdbRobustFrontStages", designs[front_design_ids[0]]["geometry"]["stage_count"], "int", "stage count shared by the pooled robust-front designs", "catalogue.designs[i].geometry.stage_count, identical for every i in pooled.robust.catalogue_indices", [_inp("artifacts/catalogue.json", "/designs")])
    front_lengths = [designs[i]["geometry"]["chamber_length_m"] for i in front_design_ids]
    front_radii = [designs[i]["geometry"]["wall_radius_m"] for i in front_design_ids]
    m.add_derived("MdbRobustFrontLengthMinMm", min(front_lengths), "mm1", "shortest chamber among the robust-front designs (mm)", "min chamber_length_m over pooled.robust.catalogue_indices", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbRobustFrontLengthMaxMm", max(front_lengths), "mm1", "longest chamber among the robust-front designs (mm)", "max chamber_length_m over pooled.robust.catalogue_indices", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbRobustFrontRadiusMinMm", min(front_radii), "mm2", "smallest wall radius among the robust-front designs (mm)", "min wall_radius_m over pooled.robust.catalogue_indices", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbRobustFrontRadiusMaxMm", max(front_radii), "mm2", "largest wall radius among the robust-front designs (mm)", "max wall_radius_m over pooled.robust.catalogue_indices", [_inp("artifacts/catalogue.json", "/designs")])
    all_lengths = [d["geometry"]["chamber_length_m"] for d in designs]
    all_radii = [d["geometry"]["wall_radius_m"] for d in designs]
    m.add_derived("MdbCatalogueLengthMinMm", min(all_lengths), "mm1", "shortest chamber in the catalogue (mm)", "min catalogue.designs[*].geometry.chamber_length_m", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbCatalogueLengthMaxMm", max(all_lengths), "mm1", "longest chamber in the catalogue (mm)", "max catalogue.designs[*].geometry.chamber_length_m", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbCatalogueRadiusMinMm", min(all_radii), "mm2", "smallest wall radius in the catalogue (mm)", "min catalogue.designs[*].geometry.wall_radius_m", [_inp("artifacts/catalogue.json", "/designs")])
    m.add_derived("MdbCatalogueRadiusMaxMm", max(all_radii), "mm2", "largest wall radius in the catalogue (mm)", "max catalogue.designs[*].geometry.wall_radius_m", [_inp("artifacts/catalogue.json", "/designs")])

    # Closure dependence (CL-2) and uncertainty-width sensitivity.
    cl2 = sensitivity["closure_cl2"]
    widths = sensitivity["widths"]
    if sensitivity["unique_designs"] != pooled["unique_designs"]:
        raise ValueError("sensitivity unique designs differ from the pooled fronts")
    if cl2["closure"] != SENSITIVITY_CLOSURE_ID or cl2 != {**cl2, **{k: reported["closure_cl1_vs_cl2"][k] for k in reported["closure_cl1_vs_cl2"]}}:
        raise ValueError("CL-2 sensitivity differs from the reported block")
    if cl2["feasible"] + cl2["infeasible"] != sensitivity["unique_designs"] or len(cl2["front_design_ids"]) != cl2["front_size"] or len(cl2["front_members"]) != cl2["front_size"]:
        raise ValueError("CL-2 front counts do not reproduce")
    if sorted({x["catalogue_index"] for x in cl2["front_members"]}) != cl2["front_catalogue_indices"] or cl2["campaign_front_catalogue_indices"] != pooled["robust"]["catalogue_indices"] or cl2["campaign_hypervolume"] != pooled["robust"]["hypervolume"]:
        raise ValueError("CL-2 front designs or campaign references do not reproduce")
    cl2_shared = set(cl2["front_design_ids"]) & set(pooled["robust"]["design_ids"])
    if len(cl2_shared) != cl2["shared_with_campaign_front"] or abs(len(cl2_shared) / (cl2["front_size"] + pooled["robust"]["front_size"] - len(cl2_shared)) - cl2["jaccard_with_campaign_front"]) > 1e-15:
        raise ValueError("CL-2 shared designs or Jaccard do not reproduce")
    if set(cl2["front_catalogue_indices"]) & set(pooled["robust"]["catalogue_indices"]) != set(pooled["robust"]["catalogue_indices"]):
        raise ValueError("the CL-1 front designs are absent from the CL-2 front's catalogue designs")
    m.add("MdbClTwoFront", "artifacts/sensitivity.json", "/closure_cl2/front_size", "int", "pooled robust front size under the sensitivity closure")
    m.add("MdbClTwoFeasible", "artifacts/sensitivity.json", "/closure_cl2/feasible", "int_comma", "robust-feasible pooled designs under the sensitivity closure")
    m.add("MdbClTwoInfeasible", "artifacts/sensitivity.json", "/closure_cl2/infeasible", "int", "robust-infeasible pooled designs under the sensitivity closure")
    m.add("MdbClTwoHv", "artifacts/sensitivity.json", "/closure_cl2/hypervolume", "sci3", "pooled robust hypervolume under the sensitivity closure")
    m.add("MdbClTwoShared", "artifacts/sensitivity.json", "/closure_cl2/shared_with_campaign_front", "int", "designs shared by the sensitivity-closure front and the campaign front")
    m.add("MdbClTwoJaccard", "artifacts/sensitivity.json", "/closure_cl2/jaccard_with_campaign_front", "fixed1", "Jaccard index of the sensitivity-closure front with the campaign front")
    m.add("MdbClTwoCommonFeasible", "artifacts/sensitivity.json", "/closure_cl2/common_feasible_designs", "int", "designs feasible under both closures")
    m.add("MdbClTwoCommonSymmetricDifference", "artifacts/sensitivity.json", "/closure_cl2/common_front_symmetric_difference", "int", "symmetric difference of the two fronts restricted to the common feasible set")
    m.add("MdbClTwoIdenticalOnCommon", "artifacts/sensitivity.json", "/closure_cl2/identical_on_common_feasible_set_up_to_ties", "bool", "whether the two fronts agree on the common feasible set")
    m.add("MdbClTwoFrontDesigns", "artifacts/sensitivity.json", "/closure_cl2/front_catalogue_indices", "list_int", "catalogue designs on the sensitivity-closure front")
    m.add_derived("MdbClTwoFrontDesignCount", len(cl2["front_catalogue_indices"]), "int", "catalogue designs on the sensitivity-closure front", "len(sensitivity.closure_cl2.front_catalogue_indices)", [_inp("artifacts/sensitivity.json", "/closure_cl2/front_catalogue_indices")])
    m.add_derived("MdbClTwoHvRatio", cl2["hypervolume"] / pooled["robust"]["hypervolume"], "fixed1", "ratio of the sensitivity-closure hypervolume to the campaign hypervolume", "sensitivity.closure_cl2.hypervolume / pooled.robust.hypervolume", [_inp("artifacts/sensitivity.json", "/closure_cl2/hypervolume"), _inp("artifacts/pooled-fronts.json", "/robust/hypervolume")])
    cl2_pooled = [designs[i]["pooled"]["probability"] for i in cl2["front_catalogue_indices"]]
    m.add_derived("MdbClTwoFrontPooledPMin", min(cl2_pooled), "fixed3", "smallest pooled wall-hit probability among the sensitivity-closure front designs", "min catalogue pooled probability over closure_cl2.front_catalogue_indices", [_inp("artifacts/catalogue.json", "/designs"), _inp("artifacts/sensitivity.json", "/closure_cl2/front_catalogue_indices")])
    m.add_derived("MdbClTwoFrontPooledPMax", max(cl2_pooled), "fixed3", "largest pooled wall-hit probability among the sensitivity-closure front designs", "max catalogue pooled probability over closure_cl2.front_catalogue_indices", [_inp("artifacts/catalogue.json", "/designs"), _inp("artifacts/sensitivity.json", "/closure_cl2/front_catalogue_indices")])
    m.add_derived("MdbClTwoFrontSaturated", sum(1 for i in cl2["front_catalogue_indices"] if i in saturated_ids), "int", "sensitivity-closure front designs that have a saturated cell", "count of closure_cl2.front_catalogue_indices with a cell whose wall_hits equal its trials", [_inp("artifacts/catalogue.json", "/designs"), _inp("artifacts/sensitivity.json", "/closure_cl2/front_catalogue_indices")])
    declared_widths = protocol["uncertain_inputs"]["sensitivity_widths"]["width_scales"]
    if [w["width_scale"] for w in widths] != declared_widths or [w["width_scale"] for w in reported["uncertainty_width_sensitivity"]] != declared_widths:
        raise ValueError("width scales differ from the protocol declaration")
    campaign_width = next(w for w in widths if w["is_campaign_posterior"] is True)
    if campaign_width["width_scale"] != 1.0 or campaign_width["front_size"] != pooled["robust"]["front_size"] or campaign_width["hypervolume"] != pooled["robust"]["hypervolume"] or campaign_width["jaccard_with_campaign_front"] != 1.0:
        raise ValueError("the campaign-width row does not reproduce the pooled robust front")
    closure_rows: list[str] = []
    for width, reported_width in zip(widths, reported["uncertainty_width_sensitivity"], strict=True):
        for key in reported_width:
            if reported_width[key] != width[key]:
                raise ValueError(f"width sensitivity {width['width_scale']} differs between sensitivity.json and gates ({key})")
        if width["feasible"] + width["infeasible"] != sensitivity["unique_designs"] or len(width["front_design_ids"]) != width["front_size"] or len(width["front_members"]) != width["front_size"]:
            raise ValueError(f"width {width['width_scale']} counts do not reproduce")
        w_shared = set(width["front_design_ids"]) & set(pooled["robust"]["design_ids"])
        if len(w_shared) != width["shared_with_campaign_front"] or abs(len(w_shared) / (width["front_size"] + pooled["robust"]["front_size"] - len(w_shared)) - width["jaccard_with_campaign_front"]) > 1e-15:
            raise ValueError(f"width {width['width_scale']} shared designs or Jaccard do not reproduce")
        if sorted({x["catalogue_index"] for x in width["front_members"]}) != width["front_catalogue_indices"] or width["campaign_front_catalogue_indices"] != pooled["robust"]["catalogue_indices"]:
            raise ValueError(f"width {width['width_scale']} front designs do not reproduce")
        token = WIDTH_TOKENS[width["width_scale"]]
        index = widths.index(width)
        base = f"/widths/{index}"
        m.add(f"MdbWidth{token}Scale", "artifacts/sensitivity.json", f"{base}/width_scale", "g" if width["width_scale"] != "point" else "text", f"posterior width scale ({token})")
        m.add(f"MdbWidth{token}Feasible", "artifacts/sensitivity.json", f"{base}/feasible", "int_comma", f"robust-feasible designs at width {token}")
        m.add(f"MdbWidth{token}Infeasible", "artifacts/sensitivity.json", f"{base}/infeasible", "int", f"robust-infeasible designs at width {token}")
        m.add(f"MdbWidth{token}Front", "artifacts/sensitivity.json", f"{base}/front_size", "int", f"robust front size at width {token}")
        m.add(f"MdbWidth{token}Hv", "artifacts/sensitivity.json", f"{base}/hypervolume", "sci3", f"robust hypervolume at width {token}")
        m.add(f"MdbWidth{token}Shared", "artifacts/sensitivity.json", f"{base}/shared_with_campaign_front", "int", f"designs shared with the campaign front at width {token}")
        m.add(f"MdbWidth{token}Jaccard", "artifacts/sensitivity.json", f"{base}/jaccard_with_campaign_front", "fixed2", f"Jaccard index with the campaign front at width {token}")
        m.add(f"MdbWidth{token}CommonFeasible", "artifacts/sensitivity.json", f"{base}/common_feasible_designs", "int_comma", f"designs feasible at width {token} and under the campaign posterior")
        m.add(f"MdbWidth{token}Identical", "artifacts/sensitivity.json", f"{base}/identical_on_common_feasible_set_up_to_ties", "bool", f"front identical on the common feasible set at width {token}")
        m.add(f"MdbWidth{token}SymmetricDifference", "artifacts/sensitivity.json", f"{base}/common_front_symmetric_difference", "int", f"symmetric difference on the common feasible set at width {token}")
        m.add(f"MdbWidth{token}FrontDesigns", "artifacts/sensitivity.json", f"{base}/front_catalogue_indices", "list_int", f"catalogue designs on the front at width {token}")
        m.add(f"MdbWidth{token}SurvivalMin", "artifacts/sensitivity.json", f"{base}/survival_min", "sci1", f"smallest sampled CL-1 survival at width {token}")
        m.add(f"MdbWidth{token}SurvivalMax", "artifacts/sensitivity.json", f"{base}/survival_max", "fixed3", f"largest sampled CL-1 survival at width {token}")
        label = "point estimate" if width["width_scale"] == "point" else f"$w = {format_value('g', width['width_scale'])}$"
        closure_rows.append(
            f"{closure_keys[0]} & {label} & {format_value('int_comma', width['feasible'])} / {width['infeasible']} & {width['front_size']} & "
            f"{format_value('list_int', width['front_catalogue_indices'])} & {width['shared_with_campaign_front']} & "
            f"{format_value('fixed2', width['jaccard_with_campaign_front'])} & {format_value('sci3', width['hypervolume'])} & "
            f"{'identical' if width['identical_on_common_feasible_set_up_to_ties'] else 'differs'} ({width['common_front_symmetric_difference']})\\\\"
        )
    closure_rows.append(
        f"{sensitivity_keys[0]} & $w = {format_value('g', campaign_width['width_scale'])}$ & {format_value('int_comma', cl2['feasible'])} / {cl2['infeasible']} & {cl2['front_size']} & "
        f"{len(cl2['front_catalogue_indices'])} designs & {cl2['shared_with_campaign_front']} & "
        f"{format_value('fixed2', cl2['jaccard_with_campaign_front'])} & {format_value('sci3', cl2['hypervolume'])} & "
        f"{'identical' if cl2['identical_on_common_feasible_set_up_to_ties'] else 'differs'} ({cl2['common_front_symmetric_difference']})\\\\"
    )
    identical_widths = sum(1 for w in widths if w["identical_on_common_feasible_set_up_to_ties"] is True and w["is_campaign_posterior"] is not True)
    m.add_derived("MdbWidthCount", len(widths), "int", "posterior widths evaluated (including the campaign width)", "len(sensitivity.widths)", [_inp("artifacts/sensitivity.json", "/widths")])
    m.add_derived("MdbWidthAlternativeCount", len(widths) - 1, "int", "alternative widths evaluated", "len(sensitivity.widths) - 1", [_inp("artifacts/sensitivity.json", "/widths")])
    m.add_derived("MdbWidthIdenticalCount", identical_widths, "int", "alternative widths whose front agrees with the campaign front on the common feasible set", "count(widths[*].identical_on_common_feasible_set_up_to_ties) over the alternative widths", [_inp("artifacts/sensitivity.json", "/widths")])
    m.add_derived("MdbWidthFrontMin", min(w["front_size"] for w in widths), "int", "smallest front over the widths", "min(widths[*].front_size)", [_inp("artifacts/sensitivity.json", "/widths")])
    m.add_derived("MdbWidthFrontMax", max(w["front_size"] for w in widths), "int", "largest front over the widths", "max(widths[*].front_size)", [_inp("artifacts/sensitivity.json", "/widths")])
    m.add_derived("MdbWidthJaccardMin", min(w["jaccard_with_campaign_front"] for w in widths), "fixed2", "smallest Jaccard index of an alternative-width front with the campaign front", "min(widths[*].jaccard_with_campaign_front)", [_inp("artifacts/sensitivity.json", "/widths")])
    tie_match = TIE_PATTERN.search(protocol["gates"]["reported_not_binding"]["uncertainty_width_sensitivity"])
    if tie_match is None:
        raise ValueError("the width-sensitivity rule does not state its roundoff tolerance in the fixed pattern")
    m.add_derived("MdbTieTolerance", float(tie_match.group(1)), "sci1", "roundoff-aware dominance tolerance of the width sensitivity", "regex group of TIE_PATTERN over protocol.gates.reported_not_binding.uncertainty_width_sensitivity", [_inp("artifacts/protocol.json", "/gates/reported_not_binding/uncertainty_width_sensitivity")])
    m.add_derived("MdbCampaignSurvivalMax", campaign_width["survival_max"], "fixed3", "largest sampled CL-1 survival under the campaign posterior over the pooled designs", "widths[is_campaign_posterior].survival_max", [_inp("artifacts/sensitivity.json", "/widths")])
    m.add_derived("MdbCampaignSurvivalMin", campaign_width["survival_min"], "sci1", "smallest sampled CL-1 survival under the campaign posterior over the pooled designs", "widths[is_campaign_posterior].survival_min", [_inp("artifacts/sensitivity.json", "/widths")])

    # Shakedown disclosure.
    m.add("MdbShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("MdbShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add("MdbShakedownDisjoint", "artifacts/shakedown.json", "/disjointness/proven", "bool", "shakedown disjointness proven")
    m.add("MdbShakedownBudget", "artifacts/shakedown.json", "/shakedown_plan/evaluations_per_run", "int", "shakedown evaluations per run")
    m.add("MdbShakedownSeeds", "artifacts/shakedown.json", "/shakedown_plan/seeds", "list_int", "shakedown seeds")
    m.add("MdbShakedownDensePoints", "artifacts/shakedown.json", "/shakedown_plan/dense_reference_points_per_design", "int", "shakedown dense-reference points per design")
    m.add("MdbShakedownRuntimeS", "artifacts/shakedown.json", "/timing_s/runtime_total", "fixed0", "shakedown runtime (s)")
    m.add("MdbShakedownImportScopeMatches", "artifacts/shakedown.json", "/import_scope/matches", "bool", "shakedown import scope equals the hash scope")

    # Audit disclosures closed (the protocol's list equals the audit's own list).
    m.add_derived("MdbAuditDisclosuresClosed", len(AUDIT_DISCLOSURES), "int", "prior-campaign audit disclosures the protocol closes", "len(protocol.v1_audit_disclosures_closed); equal to the disclosure list parsed from POSTHOC_AUDIT.md at the audit revision with AUDIT_PATTERN", [_inp("artifacts/protocol.json", "/v1_audit_disclosures_closed")])
    m.add_derived("MdbAuditDisclosureIds", list(AUDIT_DISCLOSURES), "list_text", "identifiers of the closed disclosures", "keys of protocol.v1_audit_disclosures_closed in the audit's order; equal to AUDIT_PATTERN over POSTHOC_AUDIT.md", [_inp("artifacts/protocol.json", "/v1_audit_disclosures_closed")])
    for disclosure in AUDIT_DISCLOSURES:
        token = DISCLOSURE_TOKENS[disclosure]
        m.add(f"MdbAudit{token}Closure", "artifacts/protocol.json", f"/v1_audit_disclosures_closed/{disclosure}", "text", f"how the protocol closes disclosure {disclosure}")
        m.add_derived(f"MdbAudit{token}Id", disclosure, "text", f"disclosure identifier {disclosure}", "key of protocol.v1_audit_disclosures_closed", [_inp("artifacts/protocol.json", "/v1_audit_disclosures_closed")])
    m.add("MdbAuditVerdict", "artifacts/protocol.json", "/authority/v1_campaign/posthoc_audit_verdict", "text", "prior campaign audit verdict as recorded in the protocol")
    m.add("MdbSurrogateOneOutcome", "artifacts/protocol.json", "/authority/rejected_surrogates/0/outcome", "ident", "outcome of the first rejected surrogate")
    m.add("MdbSurrogateTwoOutcome", "artifacts/protocol.json", "/authority/rejected_surrogates/1/outcome", "ident", "outcome of the second rejected surrogate")
    m.add("MdbSurrogateOneCommit", "artifacts/protocol.json", "/authority/rejected_surrogates/0/result_commit", "text", "result commit prefix of the first rejected surrogate")
    m.add("MdbSurrogateTwoCommit", "artifacts/protocol.json", "/authority/rejected_surrogates/1/result_commit", "text", "result commit prefix of the second rejected surrogate")
    m.add_derived("MdbRejectedSurrogates", len(protocol["authority"]["rejected_surrogates"]), "int", "rejected surrogates recorded in the protocol", "len(protocol.authority.rejected_surrogates)", [_inp("artifacts/protocol.json", "/authority/rejected_surrogates")])
    m.add("MdbFourCellClosureCommit", "artifacts/protocol.json", "/authority/four_cell_closure_analysis/commit", "text", "four-cell closure analysis commit prefix recorded in the protocol")

    # The prior campaign (v1) for the comparison table.
    v1_seeds = [int(s) for s in v1_plan["seeds"]]
    if v1_seeds != seeds or list(v1_plan["strategies"]) != list(STRATEGIES):
        raise ValueError("the prior campaign's seeds or strategies differ from this campaign's")
    v1_prior = next(p for p in v1_sensitivity["priors"] if p["identical_to_campaign_front"] is True)
    if v1_prior["cusp_upper"] != v1_protocol["uncertain_inputs"]["inputs"][0]["upper"]:
        raise ValueError("the v1 campaign prior does not reproduce")
    m.add("MdbPriorEvaluationsPerRun", "artifacts/campaign-plan.json", "/evaluations_per_run", "int", "prior campaign evaluations per run", bundle="v1")
    m.add("MdbPriorInitialDesign", "artifacts/campaign-plan.json", "/initial_design", "int", "prior campaign shared initial design", bundle="v1")
    m.add("MdbPriorTotalEvaluations", "artifacts/campaign-result.json", "/total_evaluations", "int", "prior campaign evaluations", bundle="v1")
    m.add("MdbPriorInfeasibleEvaluations", "artifacts/campaign-result.json", "/infeasible_evaluations", "int", "prior campaign infeasible evaluations", bundle="v1")
    m.add_derived("MdbPriorGateCount", len(v1_gates["binding"]), "int", "prior campaign binding gates", "len(v1 gates.binding)", [_inp("artifacts/gates.json", "/binding", "v1")])
    m.add("MdbPriorCuspUpper", "artifacts/protocol.json", "/uncertain_inputs/inputs/0/upper", "g", "prior campaign uniform cusp prior upper bound", bundle="v1")
    m.add("MdbPriorCuspLower", "artifacts/protocol.json", "/uncertain_inputs/inputs/0/lower", "g", "prior campaign uniform cusp prior lower bound", bundle="v1")
    m.add_derived("MdbPriorDesignVariableCount", len(v1_protocol["design_variables"]), "int", "prior campaign design variables", "len(v1 protocol.design_variables)", [_inp("artifacts/protocol.json", "/design_variables", "v1")])
    m.add("MdbPriorDenseCount", "artifacts/dense-reference-summary.json", "/count", "int_comma", "prior campaign dense reference count", bundle="v1")
    m.add("MdbPriorDenseRobustHv", "artifacts/dense-reference-summary.json", "/robust_hypervolume", "sci3", "prior campaign dense reference robust hypervolume", bundle="v1")
    m.add("MdbPriorDenseRobustFront", "artifacts/dense-reference-summary.json", "/robust_front_size", "int", "prior campaign dense reference robust front size", bundle="v1")
    for strategy in STRATEGIES:
        token = STRATEGY_TOKENS[strategy]
        m.add(f"MdbPriorHv{token}Mean", "artifacts/metrics.json", f"/seed_variance/{strategy}/mean", "sci3", f"prior campaign mean final hypervolume, {strategy}", bundle="v1")
        m.add(f"MdbPriorHv{token}Std", "artifacts/metrics.json", f"/seed_variance/{strategy}/sample_std", "sci2", f"prior campaign sample standard deviation, {strategy}", bundle="v1")
        m.add(f"MdbPriorHv{token}Min", "artifacts/metrics.json", f"/seed_variance/{strategy}/minimum", "sci3", f"prior campaign minimum final hypervolume, {strategy}", bundle="v1")
        m.add(f"MdbPriorHv{token}Max", "artifacts/metrics.json", f"/seed_variance/{strategy}/maximum", "sci3", f"prior campaign maximum final hypervolume, {strategy}", bundle="v1")
        attained = [v1_metrics["hypervolume_table"][f"{strategy}:{seed}"]["attained_fraction_of_dense_reference"] for seed in seeds]
        attained_inputs = [_inp("artifacts/metrics.json", f"/hypervolume_table/{strategy}:{seed}/attained_fraction_of_dense_reference", "v1") for seed in seeds]
        m.add_derived(f"MdbPriorAttained{token}Min", min(attained), "fixed2", f"prior campaign minimum attained fraction, {strategy}", "min over seeds of v1 hypervolume_table[*].attained_fraction_of_dense_reference", attained_inputs)
        m.add_derived(f"MdbPriorAttained{token}Max", max(attained), "fixed2", f"prior campaign maximum attained fraction, {strategy}", "max over seeds", attained_inputs)
    for key, token in (("bo_beats_random", "Random"), ("bo_beats_nsga3", "Nsga")):
        m.add(f"MdbPriorBoBeats{token}Wins", "artifacts/gates.json", f"/reported_not_binding/{key}/wins", "int", f"prior campaign {key} wins", bundle="v1")
        m.add(f"MdbPriorBoBeats{token}Seeds", "artifacts/gates.json", f"/reported_not_binding/{key}/seeds", "int", f"prior campaign {key} seeds", bundle="v1")
    m.add("MdbPriorRobustFront", "artifacts/pooled-fronts.json", "/robust/front_size", "int", "prior campaign pooled robust front size", bundle="v1")
    m.add("MdbPriorNominalFront", "artifacts/pooled-fronts.json", "/nominal/front_size", "int", "prior campaign pooled nominal front size", bundle="v1")
    m.add_derived("MdbPriorSharedDesigns", len(v1_pooled["shared_design_ids"]), "int", "prior campaign designs on both pooled fronts", "len(v1 pooled.shared_design_ids)", [_inp("artifacts/pooled-fronts.json", "/shared_design_ids", "v1")])
    m.add("MdbPriorJaccard", "artifacts/pooled-fronts.json", "/jaccard_robust_nominal", "fixed2", "prior campaign Jaccard index of the robust and nominal fronts", bundle="v1")
    m.add("MdbPriorUniqueDesigns", "artifacts/pooled-fronts.json", "/unique_designs", "int", "prior campaign unique evaluated designs", bundle="v1")
    v1_prior_index = v1_sensitivity["priors"].index(v1_prior)
    m.add("MdbPriorSurvivalMin", "artifacts/sensitivity.json", f"/priors/{v1_prior_index}/survival_min", "fixed3", "prior campaign smallest sampled CL-1 survival", bundle="v1")
    m.add("MdbPriorSurvivalMax", "artifacts/sensitivity.json", f"/priors/{v1_prior_index}/survival_max", "fixed3", "prior campaign largest sampled CL-1 survival", bundle="v1")
    m.add("MdbPriorSurvivalMean", "artifacts/sensitivity.json", f"/priors/{v1_prior_index}/survival_mean", "fixed3", "prior campaign mean sampled CL-1 survival", bundle="v1")
    m.add_derived("MdbPriorToThisDenseHvRatio", v1_dense["robust_hypervolume"] / dense["robust_hypervolume"], "fixed1", "ratio of the prior campaign's dense robust hypervolume to this campaign's", "v1 dense.robust_hypervolume / dense.robust_hypervolume", [_inp("artifacts/dense-reference-summary.json", "/robust_hypervolume", "v1"), _inp("artifacts/dense-reference-summary.json", "/robust_hypervolume")])
    m.add_derived("MdbPriorToThisSurvivalRatio", v1_prior["survival_mean"] / designs[ranked_hv[0][1]]["nominal_survival_cl1"], "fixed1", "ratio of the prior campaign's mean sampled survival to the nominal survival of this campaign's largest-hypervolume design", "v1 survival_mean / catalogue.designs[argmax dense HV].nominal_survival_cl1", [_inp("artifacts/sensitivity.json", f"/priors/{v1_prior_index}/survival_mean", "v1"), _inp("artifacts/catalogue.json", "/designs")])

    # Generated TeX.
    lines = [
        "% Generated by paper/scripts/generate_mdo_l0_v2_evidence.py; do not hand edit.",
        f"% Evidence: {RESULTS.as_posix()} at commit {RESULTS_COMMIT_SHA} (manifest SHA-256 {bundle.manifest_sha256}).",
        f"% Prior campaign for the comparison: {V1_RESULTS.as_posix()} at commit {V1_RESULTS_COMMIT_SHA} (manifest SHA-256 {v1.manifest_sha256}).",
        "% Every macro value traces to an artifact path and JSON pointer recorded in paper/evidence/mdo-l0-v2.json.",
    ]
    for item in m.items:
        lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    artifact_open = f"\\ArtifactClaim{{{ARTIFACT_CLAIM_ID}}}{{{ARTIFACT_ID}}}{{%"
    lines.append("\\newcommand{\\MdbHvTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Final robust hypervolume after \\MdbEvaluationsPerRun{} evaluations per run "
        "(dimensionless all-maximise frame against the declared reference point, identical to the prior campaign's), "
        "fraction of the \\MdbDenseCount-evaluation dense-reference robust hypervolume \\MdbDenseRobustHv, final "
        "Pareto-set size, the catalogue designs on that set, distinct catalogue designs evaluated, "
        "constraint-violating evaluations and wall time per run; the last block gives the mean and sample standard "
        "deviation over the \\MdbSeedCount{} seeds. Wall times are diagnostic only.}"
    )
    lines.append("\\label{tab:mdo-l0-v2-hypervolume}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2.5pt}")
    lines.append("\\begin{tabular}{llrrr>{\\raggedright\\arraybackslash}p{1.7cm}rrr}")
    lines.append("\\toprule")
    lines.append(
        "optimiser & seed & final HV & \\shortstack[r]{fraction of\\\\dense ref.} & \\shortstack[r]{Pareto\\\\set} & "
        "\\shortstack[l]{Pareto-set\\\\designs} & \\shortstack[r]{distinct\\\\designs} & infeasible & wall (s)\\\\"
    )
    lines.append("\\midrule")
    lines.extend(hv_rows)
    lines.append("\\midrule")
    lines.extend(summary_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    lines.append("\\newcommand{\\MdbCatalogueTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{The \\MdbDenseRobustFrontDesignCount{} catalogue designs that appear on the dense-reference robust "
        "front under the campaign closure, with their ascending pooled wall-hit rank among the \\MdbCatalogueSize{} "
        "designs, the number of pooled robust-front members they carry (zero for designs the optimisers' pooled front "
        "does not reach), the screening pooled wall-hit probability with its Wilson interval, the four per-cell "
        "probabilities (anode side to exit side), the nominal survival under the campaign closure, the sealed geometry "
        "(chamber length, wall radius, stage count, stage pitch, divergent exit) and the design's own dense robust "
        "hypervolume. Probabilities are collisionless test-particle wall-hit fractions on linear-vacuum screening "
        "fields; none is a plasma quantity.}"
    )
    lines.append("\\label{tab:mdo-l0-v2-catalogue}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2.5pt}")
    lines.append("\\begin{tabular}{rrr>{\\raggedright\\arraybackslash}p{2.5cm}>{\\raggedright\\arraybackslash}p{2.9cm}rrrrrlr}")
    lines.append("\\toprule")
    lines.append(
        "design & rank & \\shortstack[r]{front\\\\members} & pooled $P(\\text{wall})$ [Wilson] & per-cell $P(\\text{wall})$ & "
        "$S$ & \\shortstack[r]{$L$\\\\(mm)} & \\shortstack[r]{$r_w$\\\\(mm)} & stages & \\shortstack[r]{pitch\\\\(mm)} & "
        "\\shortstack[l]{div.\\\\exit} & own HV\\\\"
    )
    lines.append("\\midrule")
    lines.extend(catalogue_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    lines.append("\\newcommand{\\MdbClosureTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Closure and uncertainty-width dependence of the pooled robust front of the \\MdbUniqueDesigns{} "
        "unique evaluated designs. Every row re-evaluates the same designs: under the campaign closure with the "
        "per-cell posterior widths rescaled by $w$ (the campaign posterior is $w = \\MdbWidthCampaignScale$; the point "
        "estimate replaces each posterior by its mean), and under the sensitivity closure at the campaign width. "
        "Columns: feasible / infeasible designs, front size, catalogue designs on the front, designs shared with the "
        "campaign front, Jaccard index with the campaign front, hypervolume, and whether the front restricted to the "
        "designs feasible in both settings equals the campaign set up to roundoff ties (symmetric difference in "
        "parentheses).}"
    )
    lines.append("\\label{tab:mdo-l0-v2-closure}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{llrr>{\\raggedright\\arraybackslash}p{2.3cm}rrrl}")
    lines.append("\\toprule")
    lines.append(
        "closure & width & \\shortstack[r]{feasible /\\\\infeasible} & front & catalogue designs on the front & shared & "
        "Jaccard & HV & \\shortstack[l]{common set\\\\(sym.\\ diff.)}\\\\"
    )
    lines.append("\\midrule")
    lines.extend(closure_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    comparison_rows = [
        "design space & operating point only (\\MdbPriorDesignVariableCount{} variables; geometry excluded) & catalogue index over \\MdbCatalogueSize{} screened designs $\\times$ operating point (\\MdbOperatingVariableCount{} variables)\\\\",
        "cusp probabilities & independent uniform $[\\MdbPriorCuspLower, \\MdbPriorCuspUpper]$ per cell & per-design Jeffreys posteriors of the screening counts (\\MdbCellTrials{} launches per cell)\\\\",
        "sampled survival $S$ & \\MdbPriorSurvivalMin--\\MdbPriorSurvivalMax{} (mean \\MdbPriorSurvivalMean) & \\MdbCampaignSurvivalMin--\\MdbCampaignSurvivalMax{} over the pooled designs\\\\",
        "evaluations per run / initial design & \\MdbPriorEvaluationsPerRun{} / \\MdbPriorInitialDesign & \\MdbEvaluationsPerRun{} / \\MdbInitialDesign\\\\",
        "evaluations (infeasible) & \\MdbPriorTotalEvaluations{} (\\MdbPriorInfeasibleEvaluations) & \\MdbTotalEvaluations{} (\\MdbInfeasibleEvaluations)\\\\",
        "binding integrity gates & \\MdbPriorGateCount & \\MdbGateCount\\\\",
        "dense reference (robust HV; front) & \\MdbPriorDenseCount{} (\\MdbPriorDenseRobustHv; \\MdbPriorDenseRobustFront) & \\MdbDenseCount{} (\\MdbDenseRobustHv; \\MdbDenseRobustFront)\\\\",
        "qLogNEHVI final HV, mean (min--max) & \\MdbPriorHvBoMean{} (\\MdbPriorHvBoMin--\\MdbPriorHvBoMax) & \\MdbHvBoMean{} (\\MdbHvBoMin--\\MdbHvBoMax)\\\\",
        "qLogNEHVI fraction of dense reference & \\MdbPriorAttainedBoMin--\\MdbPriorAttainedBoMax & \\MdbAttainedBoMin--\\MdbAttainedBoMax\\\\",
        "NSGA-III final HV, mean (min--max) & \\MdbPriorHvNsgaMean{} (\\MdbPriorHvNsgaMin--\\MdbPriorHvNsgaMax) & \\MdbHvNsgaMean{} (\\MdbHvNsgaMin--\\MdbHvNsgaMax)\\\\",
        "Latin-hypercube final HV, mean (min--max) & \\MdbPriorHvLhsMean{} (\\MdbPriorHvLhsMin--\\MdbPriorHvLhsMax) & \\MdbHvLhsMean{} (\\MdbHvLhsMin--\\MdbHvLhsMax)\\\\",
        "qLogNEHVI beats baseline / NSGA-III (seeds) & \\MdbPriorBoBeatsRandomWins/\\MdbPriorBoBeatsRandomSeeds{} / \\MdbPriorBoBeatsNsgaWins/\\MdbPriorBoBeatsNsgaSeeds & \\MdbBoBeatsRandomWins/\\MdbBoBeatsRandomSeeds{} / \\MdbBoBeatsNsgaWins/\\MdbBoBeatsNsgaSeeds\\\\",
        "pooled robust / nominal front (shared; Jaccard) & \\MdbPriorRobustFront{} / \\MdbPriorNominalFront{} (\\MdbPriorSharedDesigns; \\MdbPriorJaccard) & \\MdbRobustFront{} / \\MdbNominalFront{} (\\MdbSharedDesigns; \\MdbJaccard)\\\\",
        "catalogue designs on the robust front & --- & \\MdbRobustFrontDesigns\\\\",
    ]
    lines.append("\\newcommand{\\MdbComparisonTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{The prior operating-point campaign (\\texttt{\\MdbPriorCampaignId}, results commit "
        "\\texttt{\\MdbPriorResultsCommit}) beside this catalogue campaign (\\texttt{\\MdbExperimentId}, results commit "
        "\\texttt{\\MdbResultsCommit}). Both campaigns share the objectives, comparison scales, reference point, "
        "risk measure, frozen unit rows, operating-point bounds and constraint (same frame: \\MdbSameReferenceFrame), "
        "so the hypervolumes are comparable; they differ in the design space and in the source of the cusp "
        "probabilities. The prior campaign's numbers are read from its own sealed bundle, verified byte for byte.}"
    )
    lines.append("\\label{tab:mdo-l0-v2-comparison}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.1cm}>{\\raggedright\\arraybackslash}p{4.6cm}>{\\raggedright\\arraybackslash}p{5.3cm}}")
    lines.append("\\toprule")
    lines.append("quantity & prior campaign (operating point) & this campaign (catalogue $\\times$ operating point)\\\\")
    lines.append("\\midrule")
    lines.extend(comparison_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    tex = "\n".join(lines) + "\n"

    evidence = {
        "document_type": "paper-mdo-l0-v2-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "closure": CLOSURE_ID,
        "sensitivity_closure": SENSITIVITY_CLOSURE_ID,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
        "catalogue_binding": catalogue_facts,
        "audit": {
            "path": V1_AUDIT_PATH.as_posix(),
            "revision": V1_AUDIT_COMMIT_SHA,
            "git_blob": binding["v1_audit_git_blob"],
            "disclosures": list(AUDIT_DISCLOSURES),
            "rule": (
                "the disclosure list is parsed from the audit's verdict paragraph with a fixed pattern and must equal "
                "the keys of the frozen protocol's v1_audit_disclosures_closed; the audit blob must equal the one "
                "committed at the audit revision"
            ),
        },
        "manuscript_integration": {
            "status": "admitted",
            "section_path": SECTION_PATH.as_posix(),
            "section_heading": SECTION_HEADING,
            "section_binding": SECTION_BINDING,
            "generated_tex_path": OUTPUT_PATH.as_posix(),
            "generated_binding": GENERATED_BINDING,
            "manifest_id": MANIFEST_ID,
            "manifest_path": MANIFEST_PATH.as_posix(),
            "gate_id": GATE_ID,
            "gate_kind": "numerical-campaign",
            "artifact_id": ARTIFACT_ID,
            "artifact_claim_id": ARTIFACT_CLAIM_ID,
            "prose_claim_ids": list(PROSE_CLAIM_IDS),
            "rule": (
                "Every number in the section is a macro defined here; each macro is bound below to an "
                "artifact path, JSON pointer, formatter and SHA-256 of this campaign's bundle or, where marked, "
                "of the prior campaign's bundle, or to a stated derivation over such inputs. Claim-bearing "
                "sentences are exact EvidenceClaim bodies registered in paper/evidence/claims.json; the "
                "numerical-campaign gate in paper/evidence/result-gates.json names the typed manifest that admits "
                "the section. Every number is conditional on the declared closure, which identifies a "
                "collisionless test-particle wall-hit probability with a per-cusp survival factor, and none is "
                "thruster performance."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size",
        },
        "v1_bundle": {
            "experiment_id": v1.manifest["experiment_id"],
            "manifest_path": (V1_RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": v1.manifest_sha256,
            "manifest_git_blob": binding["v1_manifest_git_blob"],
            "results_commit": V1_RESULTS_COMMIT_SHA,
            "artifact_count": v1.manifest["artifact_count"],
            "verified_file_count": len(v1.hashes),
            "tolerated_eol_files": [],
            "rule": "the prior campaign's bundle is verified byte for byte against its own manifest and pinned to its admitted manifest SHA-256 and results commit before any comparison macro is read from it",
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "v1_artifacts": {path: v1.used[path] for path in sorted(v1.used)},
        "macros": m.items,
        "tables": {
            "MdbHvTable": {"rows": len(hv_rows) + len(summary_rows), "source": "artifacts/metrics.json#/runs, #/hypervolume_table, #/seed_variance"},
            "MdbCatalogueTable": {"rows": len(catalogue_rows), "source": "artifacts/catalogue.json#/designs, artifacts/pooled-fronts.json#/robust/catalogue_membership, artifacts/dense-reference.json#/per_design"},
            "MdbClosureTable": {"rows": len(closure_rows), "source": "artifacts/sensitivity.json#/widths, #/closure_cl2"},
            "MdbComparisonTable": {"rows": len(comparison_rows), "source": "both bundles' campaign-plan, campaign-result, metrics, gates, dense-reference-summary, pooled-fronts, protocol and sensitivity artifacts"},
        },
        "generator": {
            "path": "paper/scripts/generate_mdo_l0_v2_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_mdo_l0_v2_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_mdo_l0_v2_evidence.py",
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
            f"authorized by {ARTIFACT_CLAIM_ID} (quantitative-generated-table) in "
            f"paper/evidence/claims.json; admitted through {GATE_ID}"
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
        ] + [
            {"path": (V1_RESULTS / path).as_posix(), "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["v1_artifacts"].items()
        ],
        "bundle_manifest": {
            "path": evidence["bundle"]["manifest_path"],
            "sha256": evidence["bundle"]["manifest_sha256"],
            "git_blob": evidence["binding"]["manifest_git_blob"],
        },
        "v1_bundle_manifest": {
            "path": evidence["v1_bundle"]["manifest_path"],
            "sha256": evidence["v1_bundle"]["manifest_sha256"],
            "git_blob": evidence["v1_bundle"]["manifest_git_blob"],
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
        print(f"MDO L0 v2 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
