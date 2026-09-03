"""Generate hash-bound paper evidence for the orbit wall-loss geometry screening v1.

Reads the sealed results bundle of
``modern/experiments/orbit_wall_loss_geometry_screening_v1`` (every manifest file
verified byte-for-byte; no end-of-line tolerance is needed or granted), binds it
to the committed results revision, cross-checks the committed results dashboard
against the same bundle, recomputes every reported Wilson interval, and writes:

* ``paper/evidence/wall-loss-geometry-screening-v1.json`` — every macro value
  with the artifact path, JSON pointer, formatter and artifact SHA-256 it was
  read from, or the derivation and inputs of a derived macro;
* ``paper/generated/wall-loss-geometry-screening-v1.tex`` — ``\\newcommand``
  macros and four generated tables (each wrapped in ``\\ArtifactClaim``) for the
  admitted results subsection ``paper/sections/wall-loss-geometry-screening-v1.tex``;
* ``paper/generated/wall-loss-geometry-screening-v1.provenance.json`` —
  generator/input/output hashes in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The study is a screening dataset: collisionless
prescribed-field test-particle electron orbits integrated in the accepted
linear-vacuum L1a equivalent-current fields of the geometry sweep v2.  Those
fields are screening fields (not P2-qualified), so no number below is accepted
physical-orbit evidence and none is a plasma or performance claim; the dataset
is admitted at its recorded outcome, ``accepted_screening_dataset``, as
surrogate and optimisation input carrying its label.
"""

from __future__ import annotations

import gzip
import json
import math
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

EXPERIMENT = Path("modern/experiments/orbit_wall_loss_geometry_screening_v1")
RESULTS = EXPERIMENT / "results"
EVIDENCE_PATH = Path("paper/evidence/wall-loss-geometry-screening-v1.json")
OUTPUT_PATH = Path("paper/generated/wall-loss-geometry-screening-v1.tex")
SIDECAR_PATH = Path("paper/generated/wall-loss-geometry-screening-v1.provenance.json")
SECTION_PATH = Path("paper/sections/wall-loss-geometry-screening-v1.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_wall_loss_geometry_screening_dashboard.py")
DASHBOARD_TEMPLATE = Path("modern/visualization/wall-loss-geometry-screening-v1.template.html")
DASHBOARD_HTML = Path("modern/visualization/wall-loss-geometry-screening-v1.html")

# The results tree first exists at the record commit; the merge commit that carried it
# into feat/sota-foundation adds nothing under the experiment, so the record commit is
# the evidence revision.  The dashboard HTML was regenerated from the sealed bundle in
# the same commit, so the dashboard revision equals the results revision.
RESULTS_COMMIT_SHA = "ab7c28977963822b2ad6eac451d2bafef5185e6c"
PREREGISTRATION_COMMIT_SHA = "c86bfca37fdf285f4f2a53a01c2f32f14516d868"
DASHBOARD_COMMIT_SHA = RESULTS_COMMIT_SHA

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "WALL-LOSS-GEOMETRY-SCREENING-V1-20260903-96-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/wall-loss-geometry-screening-v1.json")
GATE_ID = "GATE-WALL-LOSS-GEOMETRY-SCREENING-V1"
GATE_KIND = "numerical-screening"
RECORDED_OUTCOME = "accepted-screening-dataset"
ARTIFACT_ID = "TAB-WALL-LOSS-GEOMETRY-SCREENING-V1"
ARTIFACT_CLAIM_ID = "CLM-047"
PROSE_CLAIM_IDS = ("CLM-045", "CLM-046", "CLM-048", "CLM-049", "CLM-050", "CLM-051", "CLM-052")
SECTION_BINDING = "\\input{sections/wall-loss-geometry-screening-v1.tex}"
GENERATED_BINDING = "\\input{generated/wall-loss-geometry-screening-v1.tex}"
SECTION_HEADING = "Collisionless test-particle wall loss across the accepted sweep designs"
TABLE_MACROS = ("WlgDatasetTable", "WlgExtremeTable", "WlgCellTable", "WlgTerminationTable")
REVISION_MACRO = "GeometryScreeningEvidenceRevision"
MACRO_PREFIX = "Wlg"

EXPERIMENT_ID = "orbit-wall-loss-geometry-screening-v1"
CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
CAMPAIGN_STATUS = "accepted_screening_dataset"
SCREENING_MODEL = (
    "collisionless prescribed-field relativistic-Boris test-particle electron orbits (orbit_mc) in "
    "linear-vacuum L1a equivalent-current axisymmetric screening fields (not P2-qualified; not a "
    "permanent-magnet or nonlinear-iron material model)"
)
FROZEN_FILES = ("protocol.json", "authorities.json", "shakedown.json", "design-authorities.json")
CELL_IDS = ("gs1-cell-1", "gs1-cell-2", "gs1-cell-3", "gs1-cell-4")
CELL_TOKENS = ("One", "Two", "Three", "Four")
RANK_TOKENS = ("One", "Two", "Three")
ESCAPE_SUBCLASSES = ("upstream_anode_plane", "exit_plane", "divergent_section_radial", "unclassified")
NUMERICAL_FAILURES = ("step_limit", "nonfinite_state", "extreme_relativity", "field_failure", "initial_state_invalid")
TIMEOUTS = ("path_timeout", "time_timeout")
WILSON_Z = 1.959963984540054


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _fixed_signed(value: float, digits: int) -> str:
    text = f"{float(value):+.{digits}f}"
    return text.replace("-", "$-$").replace("+", "$+$")


FORMATTERS: dict[str, Callable[[Any], str]] = {
    **_BASE_FORMATTERS,
    "mm1": lambda v: f"{1e3 * float(v):.1f}",
    "mm2": lambda v: f"{1e3 * float(v):.2f}",
    "pct0": lambda v: f"{100.0 * float(v):.0f}\\%",
    "pct2": lambda v: f"{100.0 * float(v):.2f}\\%",
    "signed2": lambda v: _fixed_signed(float(v), 2),
    "list_fixed3": lambda v: ", ".join(f"{float(x):.3f}" for x in v),
    "list_mm1": lambda v: ", ".join(f"{1e3 * float(x):.1f}" for x in v),
    "list_ident_tt": lambda v: ", ".join(f"\\texttt{{{_BASE_FORMATTERS['ident'](x)}}}" for x in v),
    "list_sentences": lambda v: " ".join(_tex_escape(str(x)) for x in v),
    "list_clauses": lambda v: "; ".join(_tex_escape(str(x)) for x in v),
    "sci3": lambda v: _sci(float(v), 3),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def wilson(successes: int, trials: int, z: float = WILSON_Z) -> tuple[float, float, float]:
    """The orbit_mc Wilson interval, re-implemented operation for operation."""

    if isinstance(successes, bool) or isinstance(trials, bool) or trials < 1 or not 0 <= successes <= trials:
        raise ValueError("binomial counts are invalid")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _check_estimate(estimate: dict[str, Any], label: str) -> None:
    if set(estimate) != {"lower", "method", "probability", "successes", "trials", "upper"} or estimate["method"] != "wilson-95":
        raise ValueError(f"{label}: estimate is not a closed Wilson-95 record")
    p, lower, upper = wilson(int(estimate["successes"]), int(estimate["trials"]))
    if estimate["probability"] != p or estimate["lower"] != lower or estimate["upper"] != upper:
        raise ValueError(f"{label}: Wilson interval does not recompute")


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties (pure Python)."""

    rx, ry = _rank(x), _rank(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator


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
        # sidecar whose byte hash must agree with the manifest.
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

    def bind_committed(self) -> dict[str, Any]:
        """Prove the working-tree bundle equals the committed results revision."""

        head = _git(self.repo, "rev-parse", "HEAD")
        for commit, label in (
            (RESULTS_COMMIT_SHA, "results"),
            (PREREGISTRATION_COMMIT_SHA, "preregistration"),
            (DASHBOARD_COMMIT_SHA, "dashboard"),
        ):
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, head],
                cwd=self.repo, check=False, capture_output=True,
            ).returncode == 0
            if not ancestor:
                raise ValueError(f"{label} commit is not an ancestor of HEAD")
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA],
            cwd=self.repo, check=False, capture_output=True,
        ).returncode != 0 or PREREGISTRATION_COMMIT_SHA == RESULTS_COMMIT_SHA:
            raise ValueError("preregistration does not strictly precede the results revision")
        manifest_rel = (RESULTS / "manifest.json").as_posix()
        committed_blob = _git(self.repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{manifest_rel}")
        working_blob = _git(self.repo, "hash-object", "--", manifest_rel)
        if committed_blob != working_blob:
            raise ValueError("working-tree results manifest differs from the committed blob")
        results_tree = _git(self.repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{RESULTS.as_posix()}")
        head_tree = _git(self.repo, "rev-parse", f"HEAD:{RESULTS.as_posix()}")
        if results_tree != head_tree:
            raise ValueError("results tree changed after the results revision")
        # The frozen preregistration files carry the same blob at the preregistration and
        # results revisions; the working tree must equal that blob.
        for name in FROZEN_FILES:
            relative = (EXPERIMENT / name).as_posix()
            frozen = _git(self.repo, "rev-parse", f"{PREREGISTRATION_COMMIT_SHA}:{relative}")
            recorded = _git(self.repo, "rev-parse", f"{RESULTS_COMMIT_SHA}:{relative}")
            working = _git(self.repo, "hash-object", "--", relative)
            if not frozen == recorded == working:
                raise ValueError(f"frozen {name} differs between preregistration, results and the working tree")
        subject = _git(self.repo, "show", "-s", "--format=%s", RESULTS_COMMIT_SHA)
        return {
            "results_commit": RESULTS_COMMIT_SHA,
            "results_commit_subject": subject,
            "results_tree": results_tree,
            "preregistration_commit": PREREGISTRATION_COMMIT_SHA,
            "dashboard_commit": DASHBOARD_COMMIT_SHA,
            "manifest_git_blob": committed_blob,
            "manifest_path": manifest_rel,
        }


def cross_check_dashboard(
    repo: Path, bundle: Bundle, dataset: dict[str, Any], campaign: dict[str, Any], gates: dict[str, Any], consumer: dict[str, Any]
) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of the same bundle; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    template_raw = (repo / DASHBOARD_TEMPLATE).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    if f'CLASSIFICATION = "{CLASSIFICATION}"' not in generator_text:
        raise ValueError("dashboard generator does not pin the screening classification")
    if 'if manifest.get("state") != "accepted_result"' not in generator_text:
        raise ValueError("dashboard generator does not verify the bundle state")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_file_sha256"] != bundle.manifest_sha256:
        raise ValueError("dashboard payload names a different results manifest")
    if identity["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA:
        raise ValueError("dashboard payload names a different preregistration commit")
    if identity["experiment_id"] != EXPERIMENT_ID or identity["verified_file_count"] != len(bundle.hashes):
        raise ValueError("dashboard payload identity differs from the bundle")
    if identity["artifact_count"] != bundle.manifest["artifact_count"]:
        raise ValueError("dashboard payload artifact count differs from the bundle")
    if identity["terminal_file_sha256"] != bundle.manifest["terminal_byte_sha256"] or identity["lock_file_sha256"] != bundle.manifest["lock_byte_sha256"]:
        raise ValueError("dashboard payload terminal/lock hashes differ from the bundle")
    if identity["protocol_semantic_sha256"] != dataset["protocol_semantic_sha256"]:
        raise ValueError("dashboard payload names a different protocol hash")
    if identity["generator_sha256"] != sha256_bytes(_lf(generator_raw)) or identity["template_sha256"] != sha256_bytes(_lf(template_raw)):
        raise ValueError("dashboard payload generator/template hashes differ from the checkout")
    if payload["classification"] != CLASSIFICATION or payload["campaign_status"] != campaign["status"]:
        raise ValueError("dashboard classification or campaign status differs from the bundle")
    for key, value in dataset["headline"].items():
        if payload["headline"][key] != value:
            raise ValueError(f"dashboard headline {key} differs from the sealed dataset")
    rows = {item["case_id"]: item for item in payload["designs"]}
    if set(rows) != {design["case_id"] for design in dataset["designs"]} or payload["design_count"] != dataset["design_count"]:
        raise ValueError("dashboard design rows differ from the sealed dataset")
    for design in dataset["designs"]:
        row = rows[design["case_id"]]
        for key, estimand in (("wall_2N", "wall_hit"), ("escape_2N", "domain_escape"), ("reflected_2N", "reflected"), ("timeout_2N", "timeout")):
            reported = design["reported"][estimand]
            shown = row["p"][key]
            if (shown["p"], shown["lo"], shown["hi"], shown["k"], shown["n"]) != (
                reported["probability"], reported["lower"], reported["upper"], reported["successes"], reported["trials"]
            ):
                raise ValueError(f"dashboard {design['case_id']} {key} differs from the sealed dataset")
        if row["p"]["wall_N"]["p"] != design["convergence"]["probabilities"]["accepted-N"]:
            raise ValueError(f"dashboard {design['case_id']} accepted-N probability differs from the sealed dataset")
        if row["convergence"]["change"] != design["convergence"]["successive_change"] or row["convergence"]["converged"] is not design["convergence"]["converged"]:
            raise ValueError(f"dashboard {design['case_id']} convergence differs from the sealed dataset")
        if row["reflections"]["2N"] != design["diagnostics"]["reflection_counts"]["accepted-2N"]:
            raise ValueError(f"dashboard {design['case_id']} reflections differ from the sealed dataset")
    if payload["headline"]["total_reflections_2N"] != sum(d["diagnostics"]["reflection_counts"]["accepted-2N"] for d in dataset["designs"]):
        raise ValueError("dashboard 2N reflection total does not reproduce")
    if payload["gates"]["validators"] != campaign["validators"] or payload["gates"]["validator_failures"] != gates["validator_failures"]:
        raise ValueError("dashboard validator counts differ from the sealed artifacts")
    if payload["execution"]["orbit_count"] != campaign["orbit_count"] or payload["execution"]["case_count"] != campaign["case_count"]:
        raise ValueError("dashboard execution counts differ from the campaign result")
    if payload["consumer"]["screening_consumed"] != sum(1 for c in consumer["screening_designs_consumed"] if c["consumption_status"] == "consumed_verified_handoff"):
        raise ValueError("dashboard consumer count differs from the consumer record")
    if payload["consumer"]["v4_reference"] != consumer["v4_reference"]["reference_row"] or payload["consumer"]["v4_design_in_screening_set"] is not False:
        raise ValueError("dashboard reference row differs from the consumer record")
    if payload["excluded_designs"] != dataset["excluded_designs"]:
        raise ValueError("dashboard exclusions differ from the sealed dataset")
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
            "the committed dashboard verifies the bundle against its manifest, embeds its own extraction of the "
            "dataset and pins the manifest SHA-256 and the preregistration commit; the generator requires that "
            "extraction (identity, headline, every per-design estimate and convergence flag, gate and consumer "
            "counts) to equal the sealed artifacts before writing any macro"
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


def _interval(estimate: dict[str, Any]) -> str:
    return f"{estimate['probability']:.3f} [{estimate['lower']:.3f}, {estimate['upper']:.3f}]"


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(repo: Path) -> tuple[dict[str, Any], str]:
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo)
    binding = bundle.bind_committed()
    m = Macros(bundle)
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    dataset = m.doc("artifacts/geometry-wall-loss-dataset.json")
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
    designs = dataset["designs"]
    dashboard = cross_check_dashboard(repo, bundle, dataset, campaign, gates, consumer)

    # ---- internal consistency of the sealed bundle (fail closed on any disagreement) ----
    if terminal["state"] != bundle.manifest["state"] or terminal["payload"] != campaign:
        raise ValueError("terminal record disagrees with the manifest or the campaign result")
    if terminal["counts"]["attempt_count"] != 1 or lock["attempt"] != 1 or lock["immutable"] is not True:
        raise ValueError("execution lock or terminal record does not record the single attempt")
    if lock["commit"] != PREREGISTRATION_COMMIT_SHA or lock["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("execution lock names a different preregistration commit or experiment")
    if campaign["status"] != CAMPAIGN_STATUS or campaign["evidentiary"] is not True or campaign["plan_kind"] != "evidentiary":
        raise ValueError("campaign result is not the accepted evidentiary screening dataset")
    if not (campaign["classification"] == dataset["classification"] == protocol["classification"] == authorities["classification"] == CLASSIFICATION):
        raise ValueError("classification differs between the sealed artifacts")
    if campaign["gates"] != gates or gates["passed"] is not True or gates["binding"] is not True or gates["structural_all_passed"] is not True:
        raise ValueError("gates.json disagrees with the campaign result or records a failure")
    if campaign["headline"] != dataset["headline"]:
        raise ValueError("campaign headline differs from the dataset headline")
    if not (len(designs) == dataset["design_count"] == campaign["design_count"] == gates["design_count"]):
        raise ValueError("design count differs between the dataset and the campaign result")
    if dataset["excluded_designs"] != [] or exclusions["excluded"] != [] or campaign["excluded_design_count"] != 0:
        raise ValueError("the bundle records an excluded design")
    if not (dataset["protocol_semantic_sha256"] == authorities["protocol_semantic_sha256"] == shakedown["protocol_semantic_sha256"] == design_authorities["protocol_semantic_sha256"]):
        raise ValueError("protocol semantic hash differs between the sealed artifacts")
    if not (dataset["orbit_mc_source_sha256"] == authorities["orbit_mc_source_sha256"] == contract["source_sha256"] == shakedown["orbit_mc_source_sha256"]):
        raise ValueError("orbit_mc source hash differs between the sealed artifacts")
    if not (dataset["field_pipeline_source_sha256"] == authorities["field_pipeline_source_sha256"] == field_binding["field_pipeline_source_sha256"] == shakedown["field_pipeline_source_sha256"]):
        raise ValueError("field pipeline source hash differs between the sealed artifacts")
    if contract["matches"] is not True or contract["expected"] != contract["observed"] or contract["expected"]["package_version"] != protocol["orbit_mc_contract"]["package_version"]:
        raise ValueError("orbit_mc code contract does not match the frozen protocol")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"]:
        raise ValueError("shakedown artifact differs from the bound authority")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if shakedown["disjointness"]["proven"] is not True or shakedown["timing_projection"]["within_budget"] is not True:
        raise ValueError("shakedown disjointness or timing projection is not recorded as passed")
    if authorities["shakedown_timing_projection"] != shakedown["timing_projection"] or protocol["designs"]["extension_batch_included"] is not True:
        raise ValueError("extension decision differs between shakedown, authorities and protocol")
    if manufactured["passed"] is not True or manufactured["checks"] != gates["manufactured"]:
        raise ValueError("manufactured gates differ from gates.json or record a failure")
    if field_binding["sweep_manifest_file_sha256"] != protocol["field_source"]["manifest_file_sha256"] or field_binding["sweep_raw_results_file_sha256"] != protocol["field_source"]["raw_results_file_sha256"]:
        raise ValueError("field pipeline binding names a different sweep record")
    if dataset["field_source"]["manifest_file_sha256"] != protocol["field_source"]["manifest_file_sha256"] or dataset["field_source"]["field_status"] != protocol["field_source"]["field_status"]:
        raise ValueError("dataset field source differs from the frozen protocol")
    primary = list(protocol["designs"]["primary_case_ids"])
    extension = list(protocol["designs"]["extension_case_ids"])
    representatives = list(protocol["designs"]["representative_case_ids"])
    declared = sorted(primary + extension)
    if len(declared) != len(set(declared)) or declared != sorted(d["case_id"] for d in designs) or plan["case_ids"] != [d["case_id"] for d in designs]:
        raise ValueError("declared designs differ from the dataset or the campaign plan")
    if plan["kind"] != "evidentiary" or plan["binding_gates"] is not True or plan["launches_per_case"] != protocol["launches"]["launches_per_case"]:
        raise ValueError("campaign plan differs from the frozen protocol")
    if len(plan["gyrophases_rad"]) != protocol["launches"]["gyrophase_count"] or plan["gyrophases_rad"][0] != protocol["launches"]["gyrophase_offset_rad"]:
        raise ValueError("campaign plan gyrophases differ from the frozen protocol")
    if design_authorities["design_count"] != len(designs) or design_authorities["case_count"] != campaign["case_count"] or design_authorities["total_launches"] != campaign["orbit_count"]:
        raise ValueError("design authorities differ from the campaign counts")
    if authorities["design_count"] != len(designs) or authorities["case_count"] != campaign["case_count"] or authorities["total_launches"] != campaign["orbit_count"]:
        raise ValueError("authorities differ from the campaign counts")
    # The frozen preregistration files (pretty-printed) must carry the payload the bundle sealed.
    for frozen in FROZEN_FILES:
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != m.doc(f"artifacts/{frozen}"):
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
    consumed = consumer["screening_designs_consumed"]
    if len(consumed) != len(designs) or consumer["classification"] != CLASSIFICATION or campaign["coupling"] != "consumer_record_published":
        raise ValueError("consumer record does not cover every design")
    reference = consumer["v4_reference"]
    if reference["passed"] is not True or reference["design_in_screening_set"] is not False or reference["consumed"]["passed"] is not True:
        raise ValueError("the v4 reference row was not consumed as recorded")
    if reference["consumed_export_file_sha256"] != protocol["coupling_consumer"]["v4_export_file_sha256"] or reference["v4_result_commit"] != protocol["coupling_consumer"]["v4_result_commit"]:
        raise ValueError("the v4 reference export differs from the frozen protocol")
    row = reference["reference_row"]
    derived = reference["consumed"]["derived"]
    if row["probability"] != derived["probability"] or row["trial_count"] != derived["trials"] or row["confidence_interval_95"] != [derived["wilson_lower"], derived["wilson_upper"]]:
        raise ValueError("the v4 reference row differs from the consumer's derivation")
    if wilson(int(derived["successes"]), int(derived["trials"]))[1:] != (derived["wilson_lower"], derived["wilson_upper"]):
        raise ValueError("the v4 reference Wilson interval does not recompute")
    if any(v is not True for v in reference["consumed"]["checks"].values()):
        raise ValueError("a consumer check of the v4 reference failed")

    # ---- per-design cross-checks against the sealed per-case artifacts ----
    consumed_by_id = {item["case_id"]: item for item in consumed}
    total_orbits = 0
    reflections_all = 0
    reflections_2n: list[int] = []
    wall_2n = 0
    escapes_2n = 0
    timeouts_all = 0
    failures_all = 0
    termination_2n = {key: 0 for key in ("wall_hit", "reflected", "domain_escape", *TIMEOUTS, *NUMERICAL_FAILURES)}
    termination_all = dict(termination_2n)
    subclasses_2n = {key: 0 for key in ESCAPE_SUBCLASSES}
    subclasses_all = dict(subclasses_2n)
    changes: list[float] = []
    refined_changes: list[float] = []
    cross_resolution: list[float] = []
    interpolation: list[float] = []
    stored_psi: list[float] = []
    stored_b: list[float] = []
    identity_checks = 0
    cell_probabilities: dict[str, list[float]] = {cell: [] for cell in CELL_IDS}
    cell_counts: dict[str, dict[str, int]] = {cell: {"wall_hit": 0, "reflected": 0, "domain_escape": 0, "timeout": 0, "trials": 0} for cell in CELL_IDS}
    mu_medians: list[float] = []
    mu_max: list[float] = []
    tolerance_close: list[float] = []
    energy_errors: list[float] = []
    case_count = 0
    for design in designs:
        case_id = design["case_id"]
        label = f"design {case_id}"
        if design["classification"] != CLASSIFICATION:
            raise ValueError(f"{label}: classification differs")
        expected_cases = {"accepted-N", "accepted-2N"} | ({"refined-N"} if case_id in representatives else set())
        if set(design["cases"]) != expected_cases or design["representative"] is not (case_id in representatives):
            raise ValueError(f"{label}: case set differs from the frozen protocol")
        if design["batch"] != ("primary" if case_id in primary else "extension"):
            raise ValueError(f"{label}: batch differs from the frozen protocol")
        reported = design["reported"]
        fine = design["cases"]["accepted-2N"]
        coarse = design["cases"]["accepted-N"]
        if reported["case"] != "accepted-2N":
            raise ValueError(f"{label}: reported case is not accepted-2N")
        for estimand in ("wall_hit", "domain_escape", "reflected", "timeout"):
            if reported[estimand] != fine[estimand]:
                raise ValueError(f"{label}: reported {estimand} differs from the accepted-2N case")
            _check_estimate(fine[estimand], f"{label} 2N {estimand}")
            _check_estimate(coarse[estimand], f"{label} N {estimand}")
        if reported["domain_escape_subclasses"] != fine["domain_escape_subclasses"]:
            raise ValueError(f"{label}: escape sub-classes differ")
        for key, case in design["cases"].items():
            case_key = f"{case_id}--{key}"
            summary = m.doc(f"artifacts/summaries/{case_key}.json")
            if summary["campaign_id"] != case["campaign_id"] or summary["sealed"] is not True or case["sealed"] is not True:
                raise ValueError(f"{case_key}: summary identity or seal differs from the dataset")
            if summary["summary"]["wall_hit"] != case["wall_hit"] or summary["summary"]["reflected"] != case["reflected"] or summary["summary"]["escaped"] != case["domain_escape"]:
                raise ValueError(f"{case_key}: summary estimates differ from the dataset")
            if summary["summary"]["termination_counts"] != case["termination_counts"] or summary["summary"]["trial_count"] != case["trial_count"]:
                raise ValueError(f"{case_key}: summary termination counts differ from the dataset")
            if summary["orbit_artifact_file_sha256"] != case["orbit_artifact_file_sha256"] or summary["endpoints_payload_sha256"] != case["endpoints_payload_sha256"]:
                raise ValueError(f"{case_key}: summary artifact hashes differ from the dataset")
            if summary["gate_facts"]["orbits_exceeding_energy_gate"] != 0 or summary["gate_facts"]["final_velocity_event_velocity_mismatches"] != 0:
                raise ValueError(f"{case_key}: summary records an energy or event-velocity defect")
            energy_errors.append(float(summary["diagnostics"]["maximum_relative_energy_error"]))
            sidecar_text = bundle.raw(f"artifacts/orbits/{case_key}.json.sha256").decode("ascii")
            if sidecar_text.split()[0] != case["orbit_artifact_file_sha256"]:
                raise ValueError(f"{case_key}: orbit artifact sidecar differs from the dataset")
            if bundle.hashes[f"artifacts/handoffs/{case_key}.json"] != case["handoff_sha256"]:
                raise ValueError(f"{case_key}: handoff hash differs from the dataset")
            if case_id in representatives:
                endpoints_raw = gzip.decompress(bundle.raw(f"artifacts/endpoints/{case_key}.json.gz"))
                if sha256_bytes(endpoints_raw) != case["endpoints_payload_sha256"]:
                    raise ValueError(f"{case_key}: endpoints payload hash differs from the dataset")
                endpoints = load_json_bytes(endpoints_raw, f"{case_key} endpoints")
                if len(endpoints["rows"]) != case["trial_count"] or endpoints["orbit_artifact_file_sha256"] != case["orbit_artifact_file_sha256"]:
                    raise ValueError(f"{case_key}: endpoints table differs from the dataset")
                if f"artifacts/orbits/{case_key}.json.gz" not in bundle.hashes:
                    raise ValueError(f"{case_key}: representative orbit artifact is not in the bundle")
            counts = case["termination_counts"]
            if set(counts) != set(termination_all) or sum(counts.values()) != case["trial_count"]:
                raise ValueError(f"{case_key}: termination counts do not partition the trials")
            if counts["reflected"] != case["reflected"]["successes"] or counts["wall_hit"] != case["wall_hit"]["successes"] or counts["domain_escape"] != case["domain_escape"]["successes"]:
                raise ValueError(f"{case_key}: termination counts differ from the estimates")
            if sum(counts[t] for t in TIMEOUTS) != case["timeout"]["successes"] or case["timeout_counts"] != {t: counts[t] for t in TIMEOUTS}:
                raise ValueError(f"{case_key}: timeout counts differ from the estimate")
            if sum(case["domain_escape_subclasses"].values()) != counts["domain_escape"] or not set(case["domain_escape_subclasses"]) <= set(ESCAPE_SUBCLASSES):
                raise ValueError(f"{case_key}: escape sub-classes do not partition the escapes")
            case_count += 1
            total_orbits += case["trial_count"]
            reflections_all += counts["reflected"]
            timeouts_all += sum(counts[t] for t in TIMEOUTS)
            failures_all += sum(counts[t] for t in NUMERICAL_FAILURES)
            for name, value in counts.items():
                termination_all[name] += value
            for name, value in case["domain_escape_subclasses"].items():
                subclasses_all[name] += value
            if key == "accepted-2N":
                for name, value in counts.items():
                    termination_2n[name] += value
                for name, value in case["domain_escape_subclasses"].items():
                    subclasses_2n[name] += value
                reflections_2n.append(counts["reflected"])
                wall_2n += counts["wall_hit"]
                escapes_2n += counts["domain_escape"]
        if design["diagnostics"]["reflection_counts"] != {k: c["termination_counts"]["reflected"] for k, c in design["cases"].items()}:
            raise ValueError(f"{label}: reflection diagnostics differ from the termination counts")
        convergence = design["convergence"]
        change = abs(fine["wall_hit"]["probability"] - coarse["wall_hit"]["probability"])
        if convergence["successive_change"] != change or convergence["probabilities"] != {"accepted-2N": fine["wall_hit"]["probability"], "accepted-N": coarse["wall_hit"]["probability"]}:
            raise ValueError(f"{label}: successive change does not recompute")
        overlap = coarse["wall_hit"]["lower"] <= fine["wall_hit"]["upper"] and fine["wall_hit"]["lower"] <= coarse["wall_hit"]["upper"]
        if convergence["adjacent_wilson_overlap"] is not overlap or convergence["maximum_allowed_change"] != protocol["gates"]["maximum_successive_probability_change"]:
            raise ValueError(f"{label}: Wilson overlap or convergence gate differs from the protocol")
        if convergence["converged"] is not (change <= convergence["maximum_allowed_change"] and overlap) or convergence["converged"] is not True or convergence["sealed"] is not True:
            raise ValueError(f"{label}: convergence flag does not recompute or the design is not converged and sealed")
        changes.append(change)
        if case_id in representatives:
            sensitivity = convergence["field_resolution_sensitivity"]
            refined = design["cases"]["refined-N"]["wall_hit"]["probability"]
            if sensitivity["probabilities"] != {"accepted-N": coarse["wall_hit"]["probability"], "refined-N": refined} or sensitivity["change"] != abs(refined - coarse["wall_hit"]["probability"]):
                raise ValueError(f"{label}: field-resolution sensitivity does not recompute")
            if sensitivity["binding"] is not False:
                raise ValueError(f"{label}: field-resolution sensitivity is recorded as binding")
            refined_changes.append(sensitivity["change"])
        per_design = gates["per_design"][case_id]
        if per_design != design["gates"] or per_design["passed"] is not True or per_design["sealed"] is not True or per_design["converged"] is not True:
            raise ValueError(f"{label}: per-design gates differ from gates.json or record a failure")
        if any(v is not True for v in per_design["checks"].values()) or per_design["timeout_count"] != 0:
            raise ValueError(f"{label}: a per-design check failed or a timeout was recorded")
        # Field evidence: the accepted L1a field was re-solved and its identity proven.
        evidence = m.doc(f"artifacts/field-evidence/{case_id}.json")
        if evidence["passed"] is not True or any(v is not True for v in evidence["checks"].values()):
            raise ValueError(f"{label}: field evidence records a failed check")
        if evidence["resolve"]["qoi_replay"]["passed"] is not True or evidence["resolve"]["converged"] is not True:
            raise ValueError(f"{label}: field re-solve or QoI replay did not pass")
        if evidence["case_sha256"] != design["identities"]["case_sha256"] or evidence["geometry_sha256"] != design["identities"]["geometry_sha256"]:
            raise ValueError(f"{label}: field identity differs from the dataset")
        if evidence["accepted_bore_field"]["interpolation_error_report"]["b_relative_rms"] != design["field"]["interpolation_b_relative_rms"]:
            raise ValueError(f"{label}: interpolation report differs from the dataset")
        identity_checks += 1
        interpolation.append(float(design["field"]["interpolation_b_relative_rms"]))
        if (evidence["cross_resolution"] is None) is not (design["field"]["cross_resolution_b_relative_rms"] is None) or (evidence["cross_resolution"] is None) is not (case_id not in representatives):
            raise ValueError(f"{label}: cross-resolution diagnostic presence differs from the representative set")
        if evidence["cross_resolution"] is not None:
            if evidence["cross_resolution"]["b_relative_rms"] != design["field"]["cross_resolution_b_relative_rms"]:
                raise ValueError(f"{label}: cross-resolution report differs from the dataset")
            cross_resolution.append(float(evidence["cross_resolution"]["b_relative_rms"]))
            stored = evidence["resolve"]["stored_representative"]
            if stored is None or stored["passed"] is not True:
                raise ValueError(f"{label}: stored representative map was not reproduced")
            stored_psi.append(float(stored["psi_max_abs_difference_wb"]))
            stored_b.append(float(stored["b_max_abs_difference_t"]))
        elif evidence["resolve"]["stored_representative"] is not None:
            raise ValueError(f"{label}: a non-representative design carries a stored map comparison")
        # Per-cell estimates and per-stratum counts at the reported timestep.
        cells = design["per_cell"]["accepted-2N"]
        if set(cells) != set(CELL_IDS):
            raise ValueError(f"{label}: cell set differs")
        cell_wall = 0
        for cell in CELL_IDS:
            record = cells[cell]
            for estimand in ("wall_hit", "domain_escape", "reflected", "timeout"):
                _check_estimate(record[estimand], f"{label} {cell} {estimand}")
                if record[estimand]["trials"] != record["trials"] or record[estimand]["successes"] != record["counts"][estimand]:
                    raise ValueError(f"{label} {cell}: estimate differs from the counts")
            if sum(record["counts"].values()) != record["trials"]:
                raise ValueError(f"{label} {cell}: counts do not partition the trials")
            cell_probabilities[cell].append(record["wall_hit"]["probability"])
            cell_counts[cell]["trials"] += record["trials"]
            for estimand in ("wall_hit", "domain_escape", "reflected", "timeout"):
                cell_counts[cell][estimand] += record["counts"][estimand]
            cell_wall += record["counts"]["wall_hit"]
        if cell_wall != reported["wall_hit"]["successes"]:
            raise ValueError(f"{label}: per-cell wall hits do not sum to the reported successes")
        strata = design["per_stratum"]["accepted-2N"]
        if len(strata) != protocol["launches"]["strata_per_case"] or sum(s["wall_hit"] for s in strata) != reported["wall_hit"]["successes"] or sum(s["trials"] for s in strata) != reported["wall_hit"]["trials"]:
            raise ValueError(f"{label}: strata do not sum to the reported case")
        mu = design["diagnostics"]["magnetic_moment_variation"]
        if mu["binding"] is not False or mu["role"] != "diagnostic_only":
            raise ValueError(f"{label}: magnetic-moment variation is recorded as a gate")
        mu_medians.append(float(mu["median"]))
        mu_max.append(float(mu["max"]))
        tolerance_close.append(float(design["diagnostics"]["tolerance_close_share"]))
        energy_errors.append(float(design["diagnostics"]["maximum_relative_energy_error"]))
        consumed_row = consumed_by_id[case_id]
        if consumed_row["consumption_status"] != "consumed_verified_handoff" or consumed_row["consumed"]["passed"] is not True or consumed_row["sealed"] is not True:
            raise ValueError(f"{label}: handoff was not consumed as a verified handoff")
        if consumed_row["handoff_sha256"] != fine["handoff_sha256"] or consumed_row["probability"] != reported["wall_hit"]["probability"] or consumed_row["trial_count"] != reported["wall_hit"]["trials"]:
            raise ValueError(f"{label}: consumer record differs from the reported case")
        if consumed_row["consumed"]["orbit_result_artifact_sha256"] != fine["orbit_artifact_file_sha256"] or consumed_row["label"] != CLASSIFICATION:
            raise ValueError(f"{label}: consumer binding differs from the sealed orbit artifact")
        if any(v is not True for v in consumed_row["consumed"]["checks"].values()):
            raise ValueError(f"{label}: a consumer check failed")
    if case_count != campaign["case_count"] or total_orbits != campaign["orbit_count"] or gates["sealed_case_count"] != case_count:
        raise ValueError("case or orbit totals differ from the campaign result")
    if reflections_all != dataset["headline"]["total_reflections"] or termination_all["reflected"] != reflections_all:
        raise ValueError("reflection total differs from the headline")
    if timeouts_all != 0 or failures_all != 0 or max(energy_errors) != 0.0:
        raise ValueError("the bundle records a timeout, a numerical failure or an energy drift")
    wall_probabilities = [d["reported"]["wall_hit"]["probability"] for d in designs]
    headline = dataset["headline"]
    if (min(wall_probabilities), max(wall_probabilities), statistics.median(wall_probabilities)) != (
        headline["wall_hit_probability_min"], headline["wall_hit_probability_max"], headline["wall_hit_probability_median"]
    ):
        raise ValueError("wall-hit probability range or median does not reproduce from the designs")
    ordered = sorted(designs, key=lambda d: (d["reported"]["wall_hit"]["probability"], d["case_id"]))
    if len({d["reported"]["wall_hit"]["probability"] for d in ordered[:3]}) != 3 or len({d["reported"]["wall_hit"]["probability"] for d in ordered[-3:]}) != 3:
        raise ValueError("the three least or most wall-loss designs are not uniquely ordered")
    least = ordered[:3]
    most = ordered[-3:][::-1]
    if [d["case_id"] for d in least] != headline["least_wall_loss_case_ids"] or [d["case_id"] for d in most] != headline["most_wall_loss_case_ids"]:
        raise ValueError("least/most wall-loss designs differ from the headline")
    if headline["designs_with_reflections"] != [d["case_id"] for d in designs if d["reported"]["reflected"]["successes"] > 0]:
        raise ValueError("designs with reflections differ from the headline")
    if headline["converged_design_count"] != len(designs) or headline["sealed_design_count"] != len(designs) or headline["timeout_free_design_count"] != len(designs):
        raise ValueError("headline design counts differ from the per-design records")
    if len(cross_resolution) != len(representatives) or len(refined_changes) != len(representatives):
        raise ValueError("refined diagnostics are not exactly the representative designs")

    # ---- identity and lifecycle ----
    m.add("WlgClassification", "artifacts/campaign-result.json", "/classification", "ident", "screening classification string")
    m.add("WlgTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("WlgCampaignStatus", "artifacts/campaign-result.json", "/status", "ident", "recorded campaign status")
    m.add_derived("WlgRecordedOutcome", RECORDED_OUTCOME, "ident", "recorded outcome admitted by the numerical-screening gate", "constant of the generator; the gate admits the study at exactly this outcome, which names campaign-result.json#/status", [{"artifact": "artifacts/campaign-result.json", "pointer": "/status"}])
    m.add_derived("WlgScreeningModel", SCREENING_MODEL, "text", "screening model label", "constant of the generator naming the orbit model of protocol.json#/claim_boundary/orbit_model in the field model of protocol.json#/claim_boundary/field_level", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/orbit_model"}, {"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}])
    m.add("WlgExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier")
    m.add("WlgAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("WlgLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutability flag")
    m.add("WlgPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    m.add_derived("WlgResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlgManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("WlgVerifiedFiles", len(bundle.hashes), "int_comma", "bundle files verified byte-for-byte", "count of manifest file entries whose sha256 and size equal the checkout", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("WlgArtifactCount", bundle.manifest["artifact_count"], "int_comma", "manifest entries (files and directories)", "manifest.artifact_count", [{"artifact": "manifest.json", "pointer": "/artifact_count"}])
    m.add_derived("WlgToleratedEolFiles", 0, "int", "manifest entries accepted through any end-of-line tolerance", "count of manifest file entries whose recorded byte_sha256 differs from sha256(bytes); the generator grants no tolerance and fails on any difference", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add("WlgOrbitMcVersion", "artifacts/orbit-mc-contract.json", "/observed/package_version", "text", "orbit_mc package version")
    m.add("WlgOrbitMcContractMatches", "artifacts/orbit-mc-contract.json", "/matches", "bool", "orbit_mc code contract matches the frozen protocol")
    m.add("WlgOrbitMcSourceSha", "artifacts/orbit-mc-contract.json", "/source_sha256", "sha_short", "orbit_mc source hash prefix")
    m.add("WlgFieldPipelineSha", "artifacts/field-pipeline-binding.json", "/field_pipeline_source_sha256", "sha_short", "field pipeline source hash prefix")
    m.add_derived("WlgFieldPipelineFiles", len(field_binding["field_pipeline_source_files"]), "int", "field pipeline source files hashed", "len(field-pipeline-binding.field_pipeline_source_files)", [{"artifact": "artifacts/field-pipeline-binding.json", "pointer": "/field_pipeline_source_files"}])
    m.add("WlgProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("WlgExperimentCodeSha", "artifacts/authorities.json", "/experiment_code_sha256", "sha_short", "experiment code hash prefix")
    m.add("WlgBackend", "artifacts/runtime.json", "/backend", "ident", "integration backend")
    m.add("WlgCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "logical CPUs of the host")
    m.add("WlgWorkerPool", "artifacts/campaign-result.json", "/execution_mode/worker_pool_size", "int", "worker pool size")
    m.add("WlgDevice", "execution-lock.json", "/device", "ident", "device string recorded in the execution lock")
    m.add("WlgCasesWallMin", "artifacts/campaign-result.json", "/execution_mode/cases_wall_s", "min1", "wall time of the case pool (min)")
    m.add("WlgAssessmentWallMin", "artifacts/campaign-result.json", "/execution_mode/assessment_wall_s", "min1", "wall time of the assessment (min)")
    m.add("WlgShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("WlgShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add_derived("WlgShakedownDesigns", len(protocol["shakedown"]["design_case_ids"]), "int", "shakedown designs", "len(protocol.shakedown.design_case_ids)", [{"artifact": "artifacts/protocol.json", "pointer": "/shakedown/design_case_ids"}])
    m.add("WlgShakedownLaunches", "artifacts/protocol.json", "/shakedown/launches_per_case", "int", "shakedown launches per case")
    m.add("WlgShakedownValidators", "artifacts/shakedown.json", "/validators/passed", "int", "shakedown validator calls passed")
    m.add("WlgShakedownDisjoint", "artifacts/shakedown.json", "/disjointness/proven", "bool", "shakedown launch design disjoint from the evidentiary design")
    m.add("WlgTimingWithinBudget", "artifacts/shakedown.json", "/timing_projection/within_budget", "bool", "extension batch within the wall-time budget")
    m.add("WlgTimingBudgetMin", "artifacts/shakedown.json", "/timing_projection/budget_wall_seconds", "min1", "wall-time budget for the extension decision (min)")
    m.add("WlgTimingProjectedMin", "artifacts/shakedown.json", "/timing_projection/projected_wall_seconds_at_pool", "min1", "projected wall time at the pool size (min)")

    # ---- design set and launch design ----
    m.add("WlgDesignCount", "artifacts/campaign-result.json", "/design_count", "int", "designs screened")
    m.add_derived("WlgDeclaredDesigns", len(declared), "int", "designs declared by the frozen protocol", "len(protocol.designs.primary_case_ids) + len(protocol.designs.extension_case_ids)", [{"artifact": "artifacts/protocol.json", "pointer": "/designs/primary_case_ids"}, {"artifact": "artifacts/protocol.json", "pointer": "/designs/extension_case_ids"}])
    m.add_derived("WlgPrimaryCount", len(primary), "int", "non-dominated (primary batch) designs", "len(protocol.designs.primary_case_ids)", [{"artifact": "artifacts/protocol.json", "pointer": "/designs/primary_case_ids"}])
    m.add_derived("WlgExtensionCount", len(extension), "int", "extension batch designs", "len(protocol.designs.extension_case_ids)", [{"artifact": "artifacts/protocol.json", "pointer": "/designs/extension_case_ids"}])
    m.add_derived("WlgRepresentativeCount", len(representatives), "int", "representative designs with a refined-N case", "len(protocol.designs.representative_case_ids)", [{"artifact": "artifacts/protocol.json", "pointer": "/designs/representative_case_ids"}])
    m.add("WlgRepresentativeIds", "artifacts/protocol.json", "/designs/representative_case_ids", "list_ident_tt", "representative case ids")
    m.add("WlgExcludedDesigns", "artifacts/campaign-result.json", "/excluded_design_count", "int", "designs excluded before integration")
    m.add("WlgCaseCount", "artifacts/campaign-result.json", "/case_count", "int", "orbit cases")
    m.add("WlgOrbitCount", "artifacts/campaign-result.json", "/orbit_count", "int_comma", "integrated electron orbits")
    m.add("WlgLaunchesPerCase", "artifacts/protocol.json", "/launches/launches_per_case", "int", "launches per case")
    m.add("WlgStrataPerCase", "artifacts/protocol.json", "/launches/strata_per_case", "int", "strata per case")
    m.add("WlgRepeatsPerStratum", "artifacts/protocol.json", "/launches/independent_repeats_per_stratum", "int", "independent position repeats per stratum")
    m.add("WlgGyrophaseCount", "artifacts/protocol.json", "/launches/gyrophase_count", "int", "gyrophases per launch position")
    m.add("WlgGyrophaseOffsetRule", "artifacts/protocol.json", "/launches/gyrophase_offset_rule", "text", "gyrophase offset rule")
    m.add("WlgEnergies", "artifacts/protocol.json", "/launches/energies_ev", "list_g", "launch kinetic energies (eV)")
    m.add("WlgPitches", "artifacts/protocol.json", "/launches/pitch_angles_deg", "list_g", "launch pitch angles (deg)")
    m.add("WlgDirections", "artifacts/protocol.json", "/launches/directions", "list_int", "parallel launch directions")
    m.add("WlgCellFractions", "artifacts/protocol.json", "/launches/cell_fractions_of_straight_span", "list_g", "cell centres as fractions of the straight span")
    m.add("WlgRadiusFractions", "artifacts/protocol.json", "/launches/radius_fractions_of_wall", "list_g", "launch radii as fractions of the wall radius")
    m.add_derived("WlgCellCount", len(CELL_IDS), "int", "launch cells per design", "len(protocol.launches.cell_fractions_of_straight_span)", [{"artifact": "artifacts/protocol.json", "pointer": "/launches/cell_fractions_of_straight_span"}])
    if len(protocol["launches"]["cell_fractions_of_straight_span"]) != len(CELL_IDS):
        raise ValueError("cell count differs from the frozen protocol")
    m.add("WlgEstimator", "artifacts/protocol.json", "/launches/estimator_policy", "ident", "estimator policy")
    m.add("WlgIntervalMethod", "artifacts/geometry-wall-loss-dataset.json", "/designs/0/reported/wall_hit/method", "ident", "interval method")
    m.add("WlgReportedCase", "artifacts/protocol.json", "/cases/reported_probability_case", "text", "reported probability case")
    m.add("WlgRotationN", "artifacts/protocol.json", "/orbit_geometry_rule/timestep_policies/N/max_rotation_rad", "g", "maximum gyro-rotation per step, policy N (rad)")
    m.add("WlgRotationTwoN", "artifacts/protocol.json", "/orbit_geometry_rule/timestep_policies/2N/max_rotation_rad", "g", "maximum gyro-rotation per step, policy 2N (rad)")
    m.add("WlgMaxPathLengths", "artifacts/protocol.json", "/orbit_geometry_rule/max_path_channel_lengths", "g", "path budget in channel lengths")
    m.add("WlgMaxTimeFactor", "artifacts/protocol.json", "/orbit_geometry_rule/max_time_transit_factor", "g", "time budget factor")
    m.add("WlgSlowestEnergy", "artifacts/protocol.json", "/orbit_geometry_rule/slowest_energy_ev", "g", "slowest launch energy for the time budget (eV)")
    m.add("WlgEventTolerance", "artifacts/protocol.json", "/orbit_geometry_rule/event_tolerance_m", "sci1", "event tolerance (m)")
    m.add("WlgMaxSteps", "artifacts/protocol.json", "/orbit_geometry_rule/max_steps", "int_comma", "step budget per orbit")
    m.add("WlgMaxGamma", "artifacts/protocol.json", "/orbit_geometry_rule/maximum_gamma", "g", "Lorentz-factor guard")

    # ---- field provenance ----
    m.add("WlgFieldStatus", "artifacts/field-pipeline-binding.json", "/field_status", "ident", "field status label")
    m.add("WlgSweepClassification", "artifacts/protocol.json", "/field_source/classification", "ident", "classification of the source sweep")
    m.add_derived("WlgFieldModelLevel", protocol["field_source"]["classification"].split("_")[0], "text", "field model level named by the sweep classification", "protocol.field_source.classification.split('_')[0]", [{"artifact": "artifacts/protocol.json", "pointer": "/field_source/classification"}])
    if protocol["field_source"]["classification"].split("_")[0] != "L1a" or "L1A" not in CLASSIFICATION:
        raise ValueError("field model level differs between the sweep classification and the screening classification")
    m.add("WlgFieldLevelStatement", "artifacts/protocol.json", "/claim_boundary/field_level", "text", "field level statement of the claim boundary")
    m.add("WlgOrbitModelStatement", "artifacts/protocol.json", "/claim_boundary/orbit_model", "text", "orbit model statement of the claim boundary")
    m.add("WlgSweepExperiment", "artifacts/protocol.json", "/field_source/experiment", "ident", "source sweep experiment path")
    m.add("WlgSweepPreregCommit", "artifacts/protocol.json", "/field_source/preregistration_commit", "sha_short", "source sweep preregistration commit prefix")
    m.add("WlgSweepManifestSha", "artifacts/field-pipeline-binding.json", "/sweep_manifest_file_sha256", "sha_short", "source sweep manifest hash prefix")
    m.add("WlgSweepRawSha", "artifacts/field-pipeline-binding.json", "/sweep_raw_results_file_sha256", "sha_short", "source sweep raw-results hash prefix")
    m.add("WlgResolveSolver", "artifacts/protocol.json", "/field_source/resolve/solver", "ident", "field re-solve function")
    m.add("WlgGridRadial", "artifacts/protocol.json", "/field_source/resolve/domain/radial_intervals", "int", "radial intervals of the re-solve")
    m.add("WlgGridAxial", "artifacts/protocol.json", "/field_source/resolve/domain/axial_intervals", "int", "axial intervals of the re-solve")
    m.add("WlgSolverRelTol", "artifacts/protocol.json", "/field_source/resolve/solver_config/relative_tolerance", "sci1", "re-solve relative tolerance")
    m.add("WlgStoredPsiTol", "artifacts/protocol.json", "/field_source/resolve/stored_map_node_tolerance/psi_max_abs_wb", "sci1", "stored-map node tolerance, flux (Wb)")
    m.add("WlgStoredBTol", "artifacts/protocol.json", "/field_source/resolve/stored_map_node_tolerance/b_max_abs_t", "sci1", "stored-map node tolerance, field (T)")
    m.add("WlgInterpolationGate", "artifacts/protocol.json", "/field_source/adapter_gates/maximum_b_relative_rms", "pct0", "interpolation gate on the relative rms field error")
    m.add("WlgCrossResolutionGate", "artifacts/protocol.json", "/field_source/adapter_gates/maximum_cross_resolution_b_relative_rms", "pct0", "cross-resolution gate on the relative rms field error")
    m.add("WlgRefinement", "artifacts/protocol.json", "/field_source/refined_diagnostic/refinement", "int", "refinement factor of the refined re-solve")
    m.add_derived("WlgIdentityProvenDesigns", identity_checks, "int", "designs whose rebuilt field identity and QoI replay passed", "count of field-evidence records with passed == true and every check true", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/checks"}])
    m.add_derived("WlgInterpolationRmsMax", max(interpolation), "pct2", "largest interpolation relative rms field error over the designs", "max over designs of field.interpolation_b_relative_rms", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/designs"}])
    m.add_derived("WlgCrossResolutionDesigns", len(cross_resolution), "int", "designs with a refined re-solve and cross-resolution diagnostic", "count of designs whose field.cross_resolution_b_relative_rms is not null (exactly the representative designs)", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/designs"}])
    m.add_derived("WlgCrossResolutionRmsMax", max(cross_resolution), "pct2", "largest cross-resolution relative rms field error over the representatives", "max over representative designs of field.cross_resolution_b_relative_rms", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/designs"}])
    m.add_derived("WlgStoredPsiMaxDiff", max(stored_psi), "sci1", "largest stored-map flux difference over the representatives (Wb)", "max over representative field-evidence records of resolve.stored_representative.psi_max_abs_difference_wb", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/resolve/stored_representative"}])
    m.add_derived("WlgStoredBMaxDiff", max(stored_b), "sci1", "largest stored-map field difference over the representatives (T)", "max over representative field-evidence records of resolve.stored_representative.b_max_abs_difference_t", [{"artifact": "artifacts/field-evidence/l1a-gs-v2-000-48d2ccedd5.json", "pointer": "/resolve/stored_representative"}])
    lengths = [d["geometry"]["chamber_length_m"] for d in designs]
    radii = [d["geometry"]["wall_radius_m"] for d in designs]
    pitches = [d["geometry"]["stage_pitch_m"] for d in designs]
    stage_counts = [d["geometry"]["stage_count"] for d in designs]
    design_inputs = [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/designs"}]
    m.add_derived("WlgLengthMinMm", min(lengths), "mm1", "shortest chamber length (mm)", "min over designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgLengthMaxMm", max(lengths), "mm1", "longest chamber length (mm)", "max over designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgRadiusMinMm", min(radii), "mm2", "smallest wall radius (mm)", "min over designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlgRadiusMaxMm", max(radii), "mm2", "largest wall radius (mm)", "max over designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlgStageCounts", sorted(set(stage_counts)), "list_int", "stage counts present", "sorted(set(geometry.stage_count))", design_inputs)
    m.add_derived("WlgDivergentDesigns", sum(1 for d in designs if d["geometry"]["has_divergent_exit"]), "int", "designs with a divergent exit section", "count(geometry.has_divergent_exit == true)", design_inputs)
    m.add_derived("WlgStraightDesigns", sum(1 for d in designs if not d["geometry"]["has_divergent_exit"]), "int", "designs with a full-length straight channel", "count(geometry.has_divergent_exit == false)", design_inputs)

    # ---- gates and verification ----
    m.add("WlgGatesPassed", "artifacts/gates.json", "/passed", "bool", "binding gates passed")
    m.add("WlgStructuralPassed", "artifacts/gates.json", "/structural_all_passed", "bool", "structural gates passed for every design")
    m.add("WlgValidatorsPassed", "artifacts/campaign-result.json", "/validators/passed", "int_comma", "validator calls passed")
    m.add("WlgValidatorsFailed", "artifacts/campaign-result.json", "/validators/failed", "int", "validator failures")
    m.add("WlgSealedCases", "artifacts/gates.json", "/sealed_case_count", "int", "sealed orbit cases")
    m.add("WlgReplayCount", "artifacts/gates.json", "/exact_authority_replay_count", "int", "exact authority replays")
    m.add("WlgConvergedDesigns", "artifacts/gates.json", "/converged_design_count", "int", "designs converged under timestep halving")
    m.add("WlgTimeoutFreeDesigns", "artifacts/gates.json", "/timeout_free_design_count", "int", "designs without a timeout")
    m.add_derived("WlgFailedCases", campaign["case_count"] - gates["sealed_case_count"], "int", "cases not sealed", "campaign.case_count - gates.sealed_case_count", [{"artifact": "artifacts/campaign-result.json", "pointer": "/case_count"}, {"artifact": "artifacts/gates.json", "pointer": "/sealed_case_count"}])
    m.add_derived("WlgFailedDesigns", sum(1 for v in gates["per_design"].values() if v["passed"] is not True), "int", "designs whose per-design gates failed", "count(gates.per_design[*].passed != true)", [{"artifact": "artifacts/gates.json", "pointer": "/per_design"}])
    m.add_derived("WlgPerDesignChecks", len(next(iter(gates["per_design"].values()))["checks"]), "int", "per-design structural checks", "len(gates.per_design[*].checks)", [{"artifact": "artifacts/gates.json", "pointer": "/per_design"}])
    m.add_derived("WlgManufacturedChecks", len(manufactured["checks"]), "int", "manufactured verification checks", "len(manufactured-gates.checks)", [{"artifact": "artifacts/manufactured-gates.json", "pointer": "/checks"}])
    m.add("WlgManufacturedPassed", "artifacts/manufactured-gates.json", "/passed", "bool", "manufactured checks passed")
    m.add("WlgCpuParityDiff", "artifacts/manufactured-gates.json", "/cpu_parity/maximum_relative_velocity_difference", "g", "numpy versus Warp CPU relative velocity difference")
    m.add("WlgCudaParityStatus", "artifacts/manufactured-gates.json", "/cuda_parity/status", "ident", "CUDA parity status")
    m.add("WlgEnergyGate", "artifacts/protocol.json", "/gates/maximum_relative_energy_error", "sci1", "relative energy drift gate")
    m.add_derived("WlgEnergyErrorMax", max(energy_errors), "g", "largest relative kinetic-energy drift over every case", "max over case summaries and designs of maximum_relative_energy_error", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/designs"}])
    m.add("WlgWallEndpointGate", "artifacts/protocol.json", "/gates/maximum_wall_endpoint_error_m", "sci1", "wall endpoint gate (m)")
    m.add("WlgMaxChangeGate", "artifacts/protocol.json", "/gates/maximum_successive_probability_change", "g", "screening convergence gate on the successive probability change")
    m.add_derived("WlgMaxSuccessiveChange", max(changes), "fixed4", "largest N to 2N wall-hit probability change over the designs", "max over designs of convergence.successive_change", design_inputs)
    m.add_derived("WlgMeanSuccessiveChange", statistics.fmean(changes), "sci2", "mean N to 2N wall-hit probability change over the designs", "mean over designs of convergence.successive_change", design_inputs)
    m.add_derived("WlgZeroChangeDesigns", sum(1 for c in changes if c == 0.0), "int", "designs whose N and 2N wall-hit probabilities are identical", "count(convergence.successive_change == 0)", design_inputs)
    m.add_derived("WlgRefinedSensitivityMax", max(refined_changes), "fixed4", "largest accepted-N to refined-N wall-hit probability change over the representatives", "max over representative designs of convergence.field_resolution_sensitivity.change", design_inputs)
    m.add_derived("WlgTimeouts", timeouts_all, "int", "timeouts over every case", "sum over cases of termination_counts.path_timeout + time_timeout", design_inputs)
    m.add_derived("WlgNumericalFailures", failures_all, "int", "numerical failures over every case", "sum over cases of the five numerical-failure termination counts", design_inputs)
    m.add_derived("WlgTotalOrbits", total_orbits, "int_comma", "orbits summed over every case", "sum over cases of trial_count", design_inputs)

    # ---- results: wall-hit probabilities ----
    m.add("WlgWallPMin", "artifacts/geometry-wall-loss-dataset.json", "/headline/wall_hit_probability_min", "fixed3", "smallest per-design wall-hit probability (accepted-2N)")
    m.add("WlgWallPMax", "artifacts/geometry-wall-loss-dataset.json", "/headline/wall_hit_probability_max", "fixed3", "largest per-design wall-hit probability (accepted-2N)")
    m.add("WlgWallPMedian", "artifacts/geometry-wall-loss-dataset.json", "/headline/wall_hit_probability_median", "fixed3", "median per-design wall-hit probability (accepted-2N)")
    m.add_derived("WlgWallPMean", statistics.fmean(wall_probabilities), "fixed3", "mean per-design wall-hit probability (accepted-2N)", "mean over designs of reported.wall_hit.probability", design_inputs)
    m.add_derived("WlgWallHitsTwoN", wall_2n, "int_comma", "wall hits at the reported timestep", "sum over designs of cases.accepted-2N.termination_counts.wall_hit", design_inputs)
    m.add_derived("WlgTrialsTwoN", len(designs) * protocol["launches"]["launches_per_case"], "int_comma", "orbits at the reported timestep", "design_count * launches_per_case", [{"artifact": "artifacts/campaign-result.json", "pointer": "/design_count"}, {"artifact": "artifacts/protocol.json", "pointer": "/launches/launches_per_case"}])
    if termination_2n["wall_hit"] + termination_2n["reflected"] + termination_2n["domain_escape"] != len(designs) * protocol["launches"]["launches_per_case"]:
        raise ValueError("2N terminations do not partition the reported orbits")
    m.add_derived("WlgWallShareTwoN", wall_2n / (len(designs) * protocol["launches"]["launches_per_case"]), "pct1", "share of reported orbits that hit the wall", "WlgWallHitsTwoN / WlgTrialsTwoN", design_inputs)

    # ---- results: reflections ----
    m.add("WlgReflectionsTotal", "artifacts/geometry-wall-loss-dataset.json", "/headline/total_reflections", "int_comma", "reflections over every case")
    m.add_derived("WlgReflectionShareAll", reflections_all / total_orbits, "pct1", "share of all orbits that reflected", "headline.total_reflections / orbit_count", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/headline/total_reflections"}, {"artifact": "artifacts/campaign-result.json", "pointer": "/orbit_count"}])
    m.add_derived("WlgReflectionsTwoN", sum(reflections_2n), "int_comma", "reflections at the reported timestep", "sum over designs of cases.accepted-2N.termination_counts.reflected", design_inputs)
    m.add_derived("WlgReflectionShareTwoN", sum(reflections_2n) / (len(designs) * protocol["launches"]["launches_per_case"]), "pct1", "share of reported orbits that reflected", "WlgReflectionsTwoN / WlgTrialsTwoN", design_inputs)
    m.add_derived("WlgReflectionsMin", min(reflections_2n), "int", "fewest reflections in a design (accepted-2N)", "min over designs of cases.accepted-2N.termination_counts.reflected", design_inputs)
    m.add_derived("WlgReflectionsMax", max(reflections_2n), "int", "most reflections in a design (accepted-2N)", "max over designs of cases.accepted-2N.termination_counts.reflected", design_inputs)
    m.add_derived("WlgDesignsWithReflections", len(headline["designs_with_reflections"]), "int", "designs with at least one reflection", "len(headline.designs_with_reflections)", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": "/headline/designs_with_reflections"}])
    reflected_probabilities = [d["reported"]["reflected"]["probability"] for d in designs]
    m.add_derived("WlgReflectedPMin", min(reflected_probabilities), "fixed3", "smallest per-design reflection probability", "min over designs of reported.reflected.probability", design_inputs)
    m.add_derived("WlgReflectedPMax", max(reflected_probabilities), "fixed3", "largest per-design reflection probability", "max over designs of reported.reflected.probability", design_inputs)
    m.add_derived("WlgReflectedPMedian", statistics.median(reflected_probabilities), "fixed3", "median per-design reflection probability", "median over designs of reported.reflected.probability", design_inputs)

    # ---- results: escapes and terminations ----
    escape_probabilities = [d["reported"]["domain_escape"]["probability"] for d in designs]
    m.add_derived("WlgEscapePMin", min(escape_probabilities), "fixed3", "smallest per-design escape probability", "min over designs of reported.domain_escape.probability", design_inputs)
    m.add_derived("WlgEscapePMax", max(escape_probabilities), "fixed3", "largest per-design escape probability", "max over designs of reported.domain_escape.probability", design_inputs)
    m.add_derived("WlgEscapePMedian", statistics.median(escape_probabilities), "fixed3", "median per-design escape probability", "median over designs of reported.domain_escape.probability", design_inputs)
    m.add_derived("WlgEscapesTwoN", escapes_2n, "int_comma", "domain escapes at the reported timestep", "sum over designs of cases.accepted-2N.termination_counts.domain_escape", design_inputs)
    m.add_derived("WlgEscapeAnode", subclasses_2n["upstream_anode_plane"], "int_comma", "escapes through the anode plane (accepted-2N)", "sum over designs of reported.domain_escape_subclasses.upstream_anode_plane", design_inputs)
    m.add_derived("WlgEscapeExit", subclasses_2n["exit_plane"], "int_comma", "escapes through the exit plane (accepted-2N)", "sum over designs of reported.domain_escape_subclasses.exit_plane", design_inputs)
    m.add_derived("WlgEscapeDivergent", subclasses_2n["divergent_section_radial"], "int", "radial escapes into the divergent section (accepted-2N)", "sum over designs of reported.domain_escape_subclasses.divergent_section_radial", design_inputs)
    m.add_derived("WlgEscapeUnclassified", subclasses_2n["unclassified"], "int", "unclassified escapes (accepted-2N)", "sum over designs of reported.domain_escape_subclasses.unclassified", design_inputs)
    m.add_derived("WlgZeroEscapeDesigns", sum(1 for p in escape_probabilities if p == 0.0), "int", "designs without any escape", "count(reported.domain_escape.probability == 0)", design_inputs)
    m.add_derived("WlgToleranceCloseShareMin", min(tolerance_close), "pct1", "smallest per-design share of tolerance-close terminations", "min over designs of diagnostics.tolerance_close_share", design_inputs)
    m.add_derived("WlgToleranceCloseShareMax", max(tolerance_close), "pct1", "largest per-design share of tolerance-close terminations", "max over designs of diagnostics.tolerance_close_share", design_inputs)

    # ---- results: per cell ----
    cell_means = {cell: statistics.fmean(values) for cell, values in cell_probabilities.items()}
    saturated_one = 0
    saturated_zero = 0
    cell_rows: list[str] = []
    fractions = protocol["launches"]["cell_fractions_of_straight_span"]
    for cell, token, fraction in zip(CELL_IDS, CELL_TOKENS, fractions):
        values = cell_probabilities[cell]
        ones = sum(1 for v in values if v == 1.0)
        zeros = sum(1 for v in values if v == 0.0)
        saturated_one += ones
        saturated_zero += zeros
        cell_input = [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": f"/designs/0/per_cell/accepted-2N/{cell}"}]
        m.add_derived(f"WlgCell{token}Mean", cell_means[cell], "fixed2", f"mean wall-hit probability of {cell} over the designs", f"mean over designs of per_cell.accepted-2N.{cell}.wall_hit.probability", cell_input)
        m.add_derived(f"WlgCell{token}Median", statistics.median(values), "fixed2", f"median wall-hit probability of {cell}", f"median over designs of per_cell.accepted-2N.{cell}.wall_hit.probability", cell_input)
        m.add_derived(f"WlgCell{token}Min", min(values), "fixed2", f"smallest wall-hit probability of {cell}", f"min over designs of per_cell.accepted-2N.{cell}.wall_hit.probability", cell_input)
        m.add_derived(f"WlgCell{token}Max", max(values), "fixed2", f"largest wall-hit probability of {cell}", f"max over designs of per_cell.accepted-2N.{cell}.wall_hit.probability", cell_input)
        m.add_derived(f"WlgCell{token}Saturated", ones, "int", f"designs whose {cell} lost every launch", f"count(per_cell.accepted-2N.{cell}.wall_hit.probability == 1)", cell_input)
        counts = cell_counts[cell]
        cell_rows.append(
            f"\\texttt{{{_ident(cell)}}} & {fraction:g} & {cell_means[cell]:.3f} & {statistics.median(values):.3f} & {min(values):.3f} & {max(values):.3f} & "
            f"{ones} & {zeros} & {counts['wall_hit']} & {counts['reflected']} & {counts['domain_escape']} & {counts['trials']}\\\\"
        )
    m.add_derived("WlgCellsSaturatedOne", saturated_one, "int", "design-cells whose every launch hit the wall", "count over designs and cells of per_cell.accepted-2N.*.wall_hit.probability == 1", design_inputs)
    m.add_derived("WlgCellsSaturatedZero", saturated_zero, "int", "design-cells with no wall hit", "count over designs and cells of per_cell.accepted-2N.*.wall_hit.probability == 0", design_inputs)
    m.add_derived("WlgDesignCells", len(designs) * len(CELL_IDS), "int", "design-cells", "design_count * cell_count", design_inputs)
    if saturated_zero != 0:
        raise ValueError("a design-cell with no wall hit contradicts the recorded headline")
    m.add_derived("WlgCellMeanMin", min(cell_means.values()), "fixed2", "smallest per-cell mean wall-hit probability", "min over cells of the per-cell mean", design_inputs)
    m.add_derived("WlgCellMeanMax", max(cell_means.values()), "fixed2", "largest per-cell mean wall-hit probability", "max over cells of the per-cell mean", design_inputs)
    m.add_derived("WlgCellTrials", cell_counts[CELL_IDS[0]]["trials"], "int_comma", "launches per cell over the designs (accepted-2N)", "sum over designs of per_cell.accepted-2N.<cell>.trials (equal for every cell)", design_inputs)
    if len({c["trials"] for c in cell_counts.values()}) != 1:
        raise ValueError("cells do not receive equal launch counts")

    # ---- results: least and most wall loss, geometry association ----
    extreme_rows: list[str] = []
    for group, token_group, items in (("least", "Least", least), ("most", "Most", most)):
        for token, design in zip(RANK_TOKENS, items):
            geometry = design["geometry"]
            reported = design["reported"]
            index = next(i for i, d in enumerate(designs) if d["case_id"] == design["case_id"])
            prefix = f"Wlg{token_group}{token}"
            m.add_derived(f"{prefix}Id", design["case_id"], "ident", f"{group} wall-loss design {token}", f"headline.{group}_wall_loss_case_ids (order verified against reported.wall_hit.probability)", [{"artifact": "artifacts/geometry-wall-loss-dataset.json", "pointer": f"/headline/{group}_wall_loss_case_ids"}])
            m.add(f"{prefix}P", "artifacts/geometry-wall-loss-dataset.json", f"/designs/{index}/reported/wall_hit/probability", "fixed3", f"wall-hit probability of the {group} wall-loss design {token}")
            m.add(f"{prefix}Lo", "artifacts/geometry-wall-loss-dataset.json", f"/designs/{index}/reported/wall_hit/lower", "fixed3", f"Wilson lower bound of the {group} wall-loss design {token}")
            m.add(f"{prefix}Hi", "artifacts/geometry-wall-loss-dataset.json", f"/designs/{index}/reported/wall_hit/upper", "fixed3", f"Wilson upper bound of the {group} wall-loss design {token}")
            m.add(f"{prefix}LengthMm", "artifacts/geometry-wall-loss-dataset.json", f"/designs/{index}/geometry/chamber_length_m", "mm1", f"chamber length of the {group} wall-loss design {token} (mm)")
            m.add(f"{prefix}RadiusMm", "artifacts/geometry-wall-loss-dataset.json", f"/designs/{index}/geometry/wall_radius_m", "mm2", f"wall radius of the {group} wall-loss design {token} (mm)")
            m.add(f"{prefix}Stages", "artifacts/geometry-wall-loss-dataset.json", f"/designs/{index}/geometry/stage_count", "int", f"stage count of the {group} wall-loss design {token}")
            extreme_rows.append(
                f"{group} & \\texttt{{{_ident(design['case_id'])}}} & {design['batch']} & {geometry['stage_count']} & {1e3 * geometry['chamber_length_m']:.1f} & "
                f"{1e3 * geometry['exit_start_m']:.1f} & {1e3 * geometry['wall_radius_m']:.2f} & {1e3 * geometry['stage_pitch_m']:.2f} & "
                f"{'yes' if geometry['has_divergent_exit'] else 'no'} & {_interval(reported['wall_hit'])} & {reported['reflected']['probability']:.3f} & {reported['domain_escape']['probability']:.3f}\\\\"
            )
    m.add_derived("WlgLeastLengthMinMm", min(d["geometry"]["chamber_length_m"] for d in least), "mm1", "shortest chamber among the three least wall-loss designs (mm)", "min over least designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgLeastLengthMaxMm", max(d["geometry"]["chamber_length_m"] for d in least), "mm1", "longest chamber among the three least wall-loss designs (mm)", "max over least designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgLeastRadiusMinMm", min(d["geometry"]["wall_radius_m"] for d in least), "mm2", "smallest wall radius among the three least wall-loss designs (mm)", "min over least designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlgLeastRadiusMaxMm", max(d["geometry"]["wall_radius_m"] for d in least), "mm2", "largest wall radius among the three least wall-loss designs (mm)", "max over least designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlgMostLengthMinMm", min(d["geometry"]["chamber_length_m"] for d in most), "mm1", "shortest chamber among the three most wall-loss designs (mm)", "min over most designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgMostLengthMaxMm", max(d["geometry"]["chamber_length_m"] for d in most), "mm1", "longest chamber among the three most wall-loss designs (mm)", "max over most designs of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgMostRadiusMinMm", min(d["geometry"]["wall_radius_m"] for d in most), "mm2", "smallest wall radius among the three most wall-loss designs (mm)", "min over most designs of geometry.wall_radius_m", design_inputs)
    m.add_derived("WlgMostRadiusMaxMm", max(d["geometry"]["wall_radius_m"] for d in most), "mm2", "largest wall radius among the three most wall-loss designs (mm)", "max over most designs of geometry.wall_radius_m", design_inputs)
    sorted_lengths = sorted(lengths)
    m.add_derived("WlgLeastLengthRankMin", min(sorted_lengths.index(d["geometry"]["chamber_length_m"]) + 1 for d in least), "int", "lowest chamber-length rank (ascending) among the three least wall-loss designs", "min over least designs of the ascending rank of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgLeastLengthRankMax", max(sorted_lengths.index(d["geometry"]["chamber_length_m"]) + 1 for d in least), "int", "highest chamber-length rank (ascending) among the three least wall-loss designs", "max over least designs of the ascending rank of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgMostLengthRankMin", min(sorted_lengths.index(d["geometry"]["chamber_length_m"]) + 1 for d in most), "int", "lowest chamber-length rank (ascending) among the three most wall-loss designs", "min over most designs of the ascending rank of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgMostLengthRankMax", max(sorted_lengths.index(d["geometry"]["chamber_length_m"]) + 1 for d in most), "int", "highest chamber-length rank (ascending) among the three most wall-loss designs", "max over most designs of the ascending rank of geometry.chamber_length_m", design_inputs)
    m.add_derived("WlgRhoLength", spearman(lengths, wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the chamber length", "spearman(geometry.chamber_length_m, reported.wall_hit.probability) over the designs", design_inputs)
    m.add_derived("WlgRhoRadius", spearman(radii, wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the wall radius", "spearman(geometry.wall_radius_m, reported.wall_hit.probability) over the designs", design_inputs)
    m.add_derived("WlgRhoPitch", spearman(pitches, wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the stage pitch", "spearman(geometry.stage_pitch_m, reported.wall_hit.probability) over the designs", design_inputs)
    m.add_derived("WlgRhoStages", spearman([float(s) for s in stage_counts], wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the stage count", "spearman(geometry.stage_count, reported.wall_hit.probability) over the designs", design_inputs)
    m.add_derived("WlgRhoMirror", spearman([d["field"]["sweep_qois"]["minimum_mirror_ratio"] for d in designs], wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the sweep's minimum mirror ratio", "spearman(field.sweep_qois.minimum_mirror_ratio, reported.wall_hit.probability) over the designs", design_inputs)
    m.add_derived("WlgRhoReflected", spearman(reflected_probabilities, wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the reflection probability", "spearman(reported.reflected.probability, reported.wall_hit.probability) over the designs", design_inputs)
    m.add_derived("WlgRhoMu", spearman(mu_medians, wall_probabilities), "signed2", "Spearman rank correlation of the wall-hit probability with the median magnetic-moment variation", "spearman(diagnostics.magnetic_moment_variation.median, reported.wall_hit.probability) over the designs", design_inputs)

    # ---- diagnostics ----
    m.add_derived("WlgMuMedianMin", min(mu_medians), "fixed2", "smallest per-design median magnetic-moment variation", "min over designs of diagnostics.magnetic_moment_variation.median", design_inputs)
    m.add_derived("WlgMuMedianMax", max(mu_medians), "fixed2", "largest per-design median magnetic-moment variation", "max over designs of diagnostics.magnetic_moment_variation.median", design_inputs)
    m.add_derived("WlgMuMaxMax", max(mu_max), "fixed1", "largest magnetic-moment variation of any orbit", "max over designs of diagnostics.magnetic_moment_variation.max", design_inputs)
    m.add("WlgMuRole", "artifacts/protocol.json", "/diagnostics/magnetic_moment_variation/role", "ident", "role of the magnetic-moment diagnostic")

    # ---- coupling consumer ----
    m.add("WlgConsumerId", "artifacts/coupling-consumer-record.json", "/consumer_id", "ident", "consumer identifier")
    m.add("WlgHandoffSchema", "artifacts/orbit-mc-contract.json", "/observed/handoff_schema_version", "ident", "coupling handoff schema version consumed")
    m.add_derived("WlgConsumedDesigns", len(consumed), "int", "screening handoffs consumed", "len(coupling-consumer-record.screening_designs_consumed)", [{"artifact": "artifacts/coupling-consumer-record.json", "pointer": "/screening_designs_consumed"}])
    m.add_derived("WlgConsumedVerified", sum(1 for c in consumed if c["consumption_status"] == "consumed_verified_handoff"), "int", "screening handoffs consumed as verified handoffs", "count(screening_designs_consumed[*].consumption_status == consumed_verified_handoff)", [{"artifact": "artifacts/coupling-consumer-record.json", "pointer": "/screening_designs_consumed"}])
    m.add_derived("WlgConsumerChecks", len(reference["consumed"]["checks"]), "int", "checks the consumer applies to every handoff", "len(v4_reference.consumed.checks)", [{"artifact": "artifacts/coupling-consumer-record.json", "pointer": "/v4_reference/consumed/checks"}])
    m.add("WlgConsumerCheckList", "artifacts/protocol.json", "/coupling_consumer/consumer_checks", "list_clauses", "consumer checks named by the frozen protocol")
    if len(protocol["coupling_consumer"]["consumer_checks"]) != len(reference["consumed"]["checks"]) - 1:
        raise ValueError("the frozen protocol's consumer checks do not match the recorded check set (its first clause names the closed-schema and constants checks together)")
    m.add("WlgVFourDesignId", "artifacts/coupling-consumer-record.json", "/v4_reference/design_id", "ident", "design of the reference export")
    m.add("WlgVFourInScreeningSet", "artifacts/coupling-consumer-record.json", "/v4_reference/design_in_screening_set", "bool", "reference design is in the screening set")
    m.add("WlgVFourConsumed", "artifacts/coupling-consumer-record.json", "/v4_reference/passed", "bool", "reference export consumed")
    m.add("WlgVFourEvidenceClass", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/evidence_class", "ident", "evidence class of the reference row")
    m.add("WlgVFourFieldQualification", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/field_qualification", "ident", "field qualification of the reference row")
    m.add("WlgVFourP", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/probability", "fixed3", "wall-loss probability of the reference row")
    m.add("WlgVFourLo", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/confidence_interval_95/0", "fixed3", "Wilson lower bound of the reference row")
    m.add("WlgVFourHi", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/confidence_interval_95/1", "fixed3", "Wilson upper bound of the reference row")
    m.add("WlgVFourTrials", "artifacts/coupling-consumer-record.json", "/v4_reference/reference_row/trial_count", "int", "trials of the reference row")
    m.add("WlgVFourResultCommit", "artifacts/coupling-consumer-record.json", "/v4_reference/v4_result_commit", "sha_short", "result commit of the reference campaign")
    m.add("WlgVFourExportSha", "artifacts/coupling-consumer-record.json", "/v4_reference/consumed_export_file_sha256", "sha_short", "file hash prefix of the consumed reference export")
    m.add("WlgVFourHeadline", "artifacts/protocol.json", "/prior_campaign_disclosure/v4/headline", "text", "headline of the prior campaign as disclosed by the frozen protocol")
    m.add("WlgPriorRelation", "artifacts/protocol.json", "/prior_campaign_disclosure/relation", "text", "relation to the prior campaign as disclosed by the frozen protocol")
    m.add("WlgHandoffIntegrationStatus", "artifacts/handoffs/l1a-gs-v2-000-48d2ccedd5--accepted-2N.json", "/integration_status", "ident", "integration status string carried by every handoff")
    for design in designs:
        handoff = m.doc(f"artifacts/handoffs/{design['case_id']}--accepted-2N.json")
        if handoff["integration_status"] != "export_only_pending_consumer_integration" or handoff["probability"] != design["reported"]["wall_hit"]["probability"]:
            raise ValueError(f"{design['case_id']}: handoff differs from the reported case")
        if handoff["schema_version"] != contract["observed"]["handoff_schema_version"] or handoff["orbit_result_artifact_sha256"] != design["cases"]["accepted-2N"]["orbit_artifact_file_sha256"]:
            raise ValueError(f"{design['case_id']}: handoff schema or artifact binding differs")

    # ---- claim boundary flags ----
    m.add("WlgNotAcceptedPhysicalOrbit", "artifacts/campaign-result.json", "/limitations/not_accepted_physical_orbit_evidence", "bool", "not accepted physical-orbit evidence")
    m.add("WlgNotPTwoQualified", "artifacts/campaign-result.json", "/limitations/not_p2_qualified", "bool", "fields not P2-qualified")
    m.add("WlgForbidPerformance", "artifacts/campaign-result.json", "/limitations/forbid_plasma_performance_publication", "bool", "plasma or performance publication forbidden")
    m.add("WlgForbidPic", "artifacts/campaign-result.json", "/limitations/forbid_pic_or_self_consistent_claim", "bool", "PIC or self-consistent claim forbidden")
    m.add("WlgForbidMirror", "artifacts/campaign-result.json", "/limitations/forbid_mirror_formula_publication", "bool", "mirror-formula publication forbidden")
    m.add("WlgHardwareValidation", "artifacts/campaign-result.json", "/limitations/hardware_or_experimental_validation", "bool", "hardware or experimental validation claimed")
    m.add("WlgUsableAs", "artifacts/campaign-result.json", "/limitations/usable_as", "list_clauses", "permitted uses of the dataset")
    m.add("WlgShakedownNotEvidence", "artifacts/campaign-result.json", "/limitations/shakedown_outcomes_are_not_evidence", "bool", "shakedown outcomes are not evidence")

    # ---- tables ----
    tex_lines = [
        f"% Generated by paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py; do not hand edit.",
        f"% Evidence: {EXPERIMENT.as_posix()} at commit {RESULTS_COMMIT_SHA} (results manifest SHA-256 {bundle.manifest_sha256}).",
        f"% Every macro value traces to an artifact path and JSON pointer recorded in {EVIDENCE_PATH.as_posix()}.",
    ]
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    dataset_rows = [
        f"designs screened (non-dominated + extension) & {len(designs)} ({len(primary)} + {len(extension)})\\\\",
        f"designs excluded before integration & {campaign['excluded_design_count']}\\\\",
        f"orbit cases (accepted-N, accepted-2N; refined-N for {len(representatives)} representatives) & {campaign['case_count']}\\\\",
        f"electron orbits integrated & {campaign['orbit_count']:,}\\\\".replace(",", "{,}"),
        f"validator calls passed / failed & {campaign['validators']['passed']:,} / {campaign['validators']['failed']}\\\\".replace(",", "{,}"),
        f"cases sealed and replayed by the exact authority & {gates['sealed_case_count']} / {campaign['case_count']}\\\\",
        f"handoffs consumed as verified handoffs & {sum(1 for c in consumed if c['consumption_status'] == 'consumed_verified_handoff')} / {len(consumed)}\\\\",
        f"designs converged (N to 2N change $\\le$ {protocol['gates']['maximum_successive_probability_change']:g}, intervals overlap) & {gates['converged_design_count']} / {len(designs)}\\\\",
        f"largest / mean N to 2N wall-hit probability change & {max(changes):.4f} / {statistics.fmean(changes):.5f}\\\\",
        f"designs with identical N and 2N wall-hit probability & {sum(1 for c in changes if c == 0.0)}\\\\",
        f"largest accepted-N to refined-N change ({len(representatives)} representatives; not binding) & {max(refined_changes):.4f}\\\\",
        f"largest relative kinetic-energy drift (gate {format_value('sci1', protocol['gates']['maximum_relative_energy_error'])}) & {max(energy_errors):g}\\\\",
        f"timeouts / numerical failures over every case & {timeouts_all} / {failures_all}\\\\",
        f"largest interpolation rms field error (gate {100 * protocol['field_source']['adapter_gates']['maximum_b_relative_rms']:.0f}\\%) & {100 * max(interpolation):.2f}\\%\\\\",
        f"largest cross-resolution rms field error ({len(cross_resolution)} representatives; gate {100 * protocol['field_source']['adapter_gates']['maximum_cross_resolution_b_relative_rms']:.0f}\\%) & {100 * max(cross_resolution):.2f}\\%\\\\",
        f"bundle files verified byte for byte / accepted through an end-of-line tolerance & {len(bundle.hashes):,} / 0\\\\".replace(",", "{,}"),
    ]
    tex_lines += _table(
        "WlgDatasetTable",
        "Dataset summary and convergence of the wall-loss geometry screening as sealed in "
        "\\texttt{campaign-result.json}, \\texttt{gates.json} and \\texttt{geometry-wall-loss-dataset.json}. "
        "Every count is a screening quantity within the collisionless test-particle model on "
        "linear-vacuum screening fields; none is a plasma or performance quantity.",
        "tab:wall-loss-geometry-screening-v1-dataset", f"{_p(11.2)}r",
        "quantity & value\\\\", dataset_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    tex_lines += _table(
        "WlgExtremeTable",
        "The three least and three most wall-loss designs at the reported timestep (accepted-2N; "
        "\\WlgLaunchesPerCase{} launches each) with their geometry as sealed in the dataset: batch, stage count, "
        "chamber length $L$, end of the straight dielectric $z_{\\mathrm{exit}}$, wall radius $r_w$, stage pitch, "
        "divergent exit, wall-hit probability with its Wilson interval, and reflection and escape probabilities. "
        "The ordering is an observation of the screening, not a design rule.",
        "tab:wall-loss-geometry-screening-v1-extremes",
        f"l{_p(2.7)}lrrrrrl{_p(2.45)}rr",
        "rank & case & batch & st. & $L$ (mm) & $z_{\\mathrm{exit}}$ (mm) & $r_w$ (mm) & pitch (mm) & div. & $P_{\\mathrm{wall}}$ [lo, hi] & $P_{\\mathrm{refl}}$ & $P_{\\mathrm{esc}}$\\\\",
        extreme_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{2pt}",
    )
    tex_lines += _table(
        "WlgCellTable",
        "Per-cell wall-hit probability over the \\WlgDesignCount{} designs at the reported timestep: cell centre as a "
        "fraction of each design's straight span, mean, median, minimum and maximum of the per-design cell probability, "
        "design-cells saturated at one and at zero, and pooled wall-hit, reflection and escape counts over the "
        "\\WlgCellTrials{} launches of each cell. The launch cells are protocol positions scaled to each design's "
        "straight channel, not demonstrated confinement cells.",
        "tab:wall-loss-geometry-screening-v1-cells", f"{_p(1.9)}rrrrrrrrrrr",
        "cell & frac. & mean & median & min & max & at $1$ & at $0$ & wall & refl. & esc. & launches\\\\",
        cell_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    termination_rows: list[str] = []
    n_2n = len(designs) * protocol["launches"]["launches_per_case"]
    for name in ("wall_hit", "reflected", "domain_escape"):
        termination_rows.append(
            f"\\texttt{{{_ident(name)}}} & {termination_2n[name]} & {100 * termination_2n[name] / n_2n:.1f}\\% & {termination_all[name]} & {100 * termination_all[name] / total_orbits:.1f}\\%\\\\"
        )
        if name == "domain_escape":
            for sub in ESCAPE_SUBCLASSES:
                termination_rows.append(
                    f"\\quad\\texttt{{{_ident(sub)}}} & {subclasses_2n[sub]} & {100 * subclasses_2n[sub] / n_2n:.1f}\\% & {subclasses_all[sub]} & {100 * subclasses_all[sub] / total_orbits:.1f}\\%\\\\"
                )
    for name in (*TIMEOUTS, *NUMERICAL_FAILURES):
        termination_rows.append(
            f"\\texttt{{{_ident(name)}}} & {termination_2n[name]} & {100 * termination_2n[name] / n_2n:.1f}\\% & {termination_all[name]} & {100 * termination_all[name] / total_orbits:.1f}\\%\\\\"
        )
    termination_rows.append(f"\\midrule\ntotal & {n_2n} & 100.0\\% & {total_orbits} & 100.0\\%\\\\")
    tex_lines += _table(
        "WlgTerminationTable",
        "Termination classes summed over the designs at the reported timestep (accepted-2N) and over every case "
        "(accepted-N, accepted-2N and the representatives' refined-N, which are separate orbit sets of the same "
        "launches), with the escape sub-classes recorded post hoc from the terminal position. Reflections are "
        "orbits that reversed their parallel velocity before reaching a boundary; timeouts are physical budget "
        "outcomes and numerical failures are integrator defects, and the bundle records none of either.",
        "tab:wall-loss-geometry-screening-v1-terminations", f"{_p(4.6)}rrrr",
        "termination & accepted-2N & share & all cases & share\\\\", termination_rows,
        extra="\\setlength{\\tabcolsep}{5pt}",
    )
    tex = "\n".join(tex_lines) + "\n"

    evidence = {
        "document_type": "paper-wall-loss-geometry-screening-v1-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "recorded_outcome": RECORDED_OUTCOME,
        "campaign_status": CAMPAIGN_STATUS,
        "screening_model": SCREENING_MODEL,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
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
                "level. Every number is a screening quantity on linear-vacuum fields and none is accepted "
                "physical-orbit, plasma or performance evidence."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size",
            "wilson_rule": "every reported, per-case and per-cell Wilson-95 interval is recomputed operation for operation and must equal the sealed value exactly",
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": {
            "WlgDatasetTable": {"rows": len(dataset_rows), "source": "artifacts/campaign-result.json, artifacts/gates.json, artifacts/geometry-wall-loss-dataset.json#/designs"},
            "WlgExtremeTable": {"rows": len(extreme_rows), "source": "artifacts/geometry-wall-loss-dataset.json#/headline/least_wall_loss_case_ids, #/headline/most_wall_loss_case_ids, #/designs"},
            "WlgCellTable": {"rows": len(cell_rows), "source": "artifacts/geometry-wall-loss-dataset.json#/designs/*/per_cell/accepted-2N"},
            "WlgTerminationTable": {"rows": len(termination_rows), "source": "artifacts/geometry-wall-loss-dataset.json#/designs/*/cases/*/termination_counts, #/domain_escape_subclasses"},
        },
        "generator": {
            "path": "paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_wall_loss_geometry_screening_v1_evidence.py",
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
        print(f"wall-loss geometry screening v1 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
