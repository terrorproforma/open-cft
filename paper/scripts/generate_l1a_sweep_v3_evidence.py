"""Generate hash-bound paper evidence for the L1a geometry sweep v3.

Reads the sealed results bundle of ``modern/experiments/l1a_geometry_sweep_v3``
(every manifest file verified byte-for-byte; no end-of-line tolerance is needed or
granted), binds it to the committed record revision, re-derives the headline and
every per-set estimand (including the preregistered hypothesis statistics) from the
224 per-design rows and their design records, cross-checks the committed results
dashboard against the same bundle, binds the sealed sweep-v2 manifest, the frozen
cusp-topology-v3.1 protocol whose wall-cusp definition the sweep imported, the
wall-loss campaign's frozen protocol (launch positions) and the P2 design record of
the topology screening (stage centres) as references, binds the TWT/PPM literature
review that fixed the hypothesis together with its read-only check script and its
committed output as the definition/hypothesis source, and writes:

* ``paper/evidence/l1a-sweep-v3.json`` -- every macro value with the artifact path,
  JSON pointer, formatter and artifact SHA-256 it was read from, or the derivation
  and inputs of a derived macro;
* ``paper/generated/l1a-sweep-v3.tex`` -- ``\\newcommand`` macros and four generated
  tables (each wrapped in ``\\ArtifactClaim``) for the admitted results subsection
  ``paper/sections/l1a-sweep-v3.tex``;
* ``paper/generated/l1a-sweep-v3.provenance.json`` -- generator/input/output hashes
  in the same shape as the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine path
enters any output.  The study is an L1a linear-vacuum equivalent-current screening
of a design space (the sweep-v2 builder rules on a wider box) with the literature
wall-cusp definition imported unchanged and the HEMP design ratio of Koch et al.
reported per cusp as a field ratio.  Nothing below is a plasma, mirror-probability,
wall-loss or performance claim; the declared iron pole pieces are vacuum in the
field, and the material-aware confirmation the protocol queues was not run.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

from generate_cusp_topology_v3_1_evidence import (
    FORMATTERS as _TOPOLOGY_FORMATTERS,
    Bundle,
    _bound_file,
    _distribution,
    _histogram,
    _histogram_text,
    _is_ancestor,
    _p,
)
from generate_mdo_l0_v1_evidence import (
    _git,
    _lf,
    _tex_escape,
    canonical_json,
    dashboard_payload,
    load_json_bytes,
    resolve_pointer,
    sha256_bytes,
)

EXPERIMENT = Path("modern/experiments/l1a_geometry_sweep_v3")
RESULTS = EXPERIMENT / "results"
SWEEP_V2_MANIFEST = Path("modern/experiments/l1a_geometry_sweep_v2/results/manifest.json")
TOPOLOGY_PROTOCOL = Path("modern/experiments/cusp_topology_search_v3_1/protocol.json")
TOPOLOGY_P2_RECORD = Path("modern/experiments/cusp_topology_search_v3_1/results/artifacts/designs/p2_divergent_exit/divergent-exit-stack.json")
WALL_LOSS_PROTOCOL = Path("modern/experiments/cft_orbit_wall_loss_v4/protocol.json")
LITERATURE_REVIEW = Path("modern/docs/literature/twt-ppm-physics-for-hemp.md")
PPM_CHECK_SCRIPT = Path("modern/docs/literature/scripts/ppm_axis_field_check.py")
PPM_CHECK_OUTPUT = Path("modern/docs/literature/scripts/ppm_axis_field_check.output.json")
EVIDENCE_PATH = Path("paper/evidence/l1a-sweep-v3.json")
OUTPUT_PATH = Path("paper/generated/l1a-sweep-v3.tex")
SIDECAR_PATH = Path("paper/generated/l1a-sweep-v3.provenance.json")
SECTION_PATH = Path("paper/sections/l1a-sweep-v3.tex")
DASHBOARD_GENERATOR = Path("modern/visualization/generate_l1a_geometry_sweep_v3_dashboard.py")
DASHBOARD_TEMPLATE = Path("modern/visualization/l1a-geometry-sweep-v3.template.html")
DASHBOARD_HTML = Path("modern/visualization/l1a-geometry-sweep-v3.html")

# Revisions.  The results tree first exists at the record commit; the dashboard was
# generated from the sealed bundle one commit later.  The literature review that fixed
# the hypothesis (and at whose commit the shakedown was run) is bound as the
# definition/hypothesis source together with its read-only check script and output.
# The sealed sweep-v2 manifest, the frozen topology-v3.1 protocol (definition import),
# the topology screening's P2 design record and the wall-loss campaign's frozen protocol
# are bound as references at their own admitted revisions.
RESULTS_COMMIT_SHA = "2cfe8223630fbef6bfe8099a5dcecaf4eb8c6b44"
PREREGISTRATION_COMMIT_SHA = "1923ef7601bcc07acafa28ce54db687f025922b6"
DASHBOARD_COMMIT_SHA = "44d0c63c2f3aaf73a6d4402e94522ceb9901c3fe"
LITERATURE_COMMIT_SHA = "beb4772c9afc04be5c2c04d3e8a4fc8c16bb771e"
SWEEP_V2_RESULTS_COMMIT_SHA = "f30cb42ec4a8633bf634a3d32ffa5b11f66be97a"
TOPOLOGY_RESULTS_COMMIT_SHA = "cec47f12f5909c5886424bf5d46ac20ce06f1ac5"
WALL_LOSS_PREREGISTRATION_COMMIT_SHA = "757e365f9f667620c7610663574294c3b71e1f51"

# Admission into the manuscript (paper/evidence/claims.json, result-gates.json,
# figure-table-contract.json).
MANIFEST_ID = "L1A-SWEEP-V3-20260903-128-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/l1a-sweep-v3.json")
GATE_ID = "GATE-L1A-SWEEP-V3"
GATE_KIND = "numerical-screening"
RECORDED_OUTCOME = "accepted-screening"
ARTIFACT_ID = "TAB-L1A-SWEEP-V3"
ARTIFACT_CLAIM_ID = "CLM-071"
# The section claims, the abstract sentence, the Discussion interpretation on the legacy
# design space, and the three Discussion claims re-scoped by the launch-position analysis
# bound with this manifest (the wall-loss campaign's zero reflections as a launch-position
# result: CLM-017, CLM-052 and the first finding of CLM-044).
PROSE_CLAIM_IDS = ("CLM-069", "CLM-070", "CLM-072", "CLM-073", "CLM-074", "CLM-075", "CLM-076", "CLM-017", "CLM-044", "CLM-052")
SECTION_BINDING = "\\input{sections/l1a-sweep-v3.tex}"
GENERATED_BINDING = "\\input{generated/l1a-sweep-v3.tex}"
SECTION_HEADING = "Geometry sweep into the HEMP-like wall-radius-to-pitch regime"
TABLE_MACROS = ("SwtDesignBoxTable", "SwtBandTable", "SwtHypothesisTable", "SwtHempLikeTable")
REVISION_MACRO = "SweepThreeEvidenceRevision"
MACRO_PREFIX = "Swt"

EXPERIMENT_ID = "l1a-geometry-sweep-v3"
CLASSIFICATION = "L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID"
TOPOLOGY_LABEL = "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
CAMPAIGN_STATUS = "accepted_l1a_sweep_v3"
SCREENING_MODEL = (
    "linear-vacuum L1a equivalent-current field-only screening of axisymmetric PPM-stack geometries "
    "(geometry schema v1.1, the sweep-v2 builder rules on a wider box reaching r_w / L = 1.24) with the "
    "cusp topology search v3.1 wall-cusp definition imported unchanged and the HEMP design ratio rho of "
    "Koch et al. reported per cusp as a field ratio; not P2-qualified, no permanent-magnet or "
    "nonlinear-iron material model (the declared soft-iron pole pieces and yoke are source-free vacuum "
    "in the field)"
)
FROZEN_FILES = ("protocol.json", "authorities.json", "shakedown.json", "design-authorities.json")
SET_IDS = ("sobol_v3", "sweep_v2")
SET_TOKENS = {"sobol_v3": "Sobol", "sweep_v2": "HeldOut"}
BINDING_GATE_NAMES = (
    "all_declared_designs_resolved",
    "determinism_replay",
    "every_null_converged",
    "every_trace_terminates_cleanly",
    "every_wall_trace_flux_consistent",
    "hash_bindings",
    "held_out_sweep_v2_reproduction",
    "identity_proven",
    "refinement_stability",
    "solver_converged_both_maps",
    "sweep_v2_gates_verbatim",
)
PER_DESIGN_GATES = tuple(sorted(set(BINDING_GATE_NAMES) - {"all_declared_designs_resolved", "determinism_replay", "hash_bindings"}))
V2_GATE_NAMES = ("boundary", "flux_identity", "manufacturability", "residual", "source_representation", "topology_confidence")
STAGE_TOKENS = {3: "Three", 4: "Four", 5: "Five"}
COUNT_TOKENS = {2: "Two", 3: "Three", 4: "Four"}
# Widened design variables (v3 box against the v2 box); the other seven variables carry
# identical bounds in both boxes (protocol.sampling.sweep_v2_box.other_variables).
WIDENED_VARIABLES = (
    ("stage_pitch_m", "Pitch", "stage pitch $L$"),
    ("chamber_outer_radius_m", "Radius", "wall radius $r_w$"),
    ("radial_clearance_m", "Clearance", "magnet radial clearance"),
    ("magnet_radial_thickness_m", "Thickness", "magnet radial thickness"),
)
# x_w bands of the band table: below the single-harmonic threshold x*, then three bands
# above it.  The upper edges are constants of this generator (the box maximum is 3.88).
BAND_EDGES = (2.5, 3.0, 3.9)
BAND_TOKENS = ("BelowStar", "One", "Two", "Three")
# The PPM review's launch-position classes (review section 3.3, Table G2): launch cells
# within NEAR_CENTRE_PITCH of a magnet centre against cells at least FAR_CENTRE_PITCH away.
NEAR_CENTRE_PITCH = 0.17
FAR_CENTRE_PITCH = 0.22
LITERATURE_KEYS = ("koch2007", "koch2011")
LITERATURE_TOKENS = {"koch2007": "IEPC-2007-110", "koch2011": "IEPC-2011-236"}
# Relative tolerance for float estimands the experiment computed with numpy pairwise sums
# and this generator recomputes with math.fsum; counts and medians must agree exactly.
FLOAT_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
FORMATTERS: dict[str, Callable[[Any], str]] = {
    **_TOPOLOGY_FORMATTERS,
    "fixed6": lambda v: f"{float(v):.6f}",
    "pct1": lambda v: f"{100.0 * float(v):.1f}\\%",
    "list_mm1": lambda v: ", ".join(f"{1e3 * float(x):.1f}" for x in v),
    "list_fixed2": lambda v: ", ".join(f"{float(x):.2f}" for x in v),
    "list_fixed3": lambda v: ", ".join(f"{float(x):.3f}" for x in v),
    # Mathematical symbols the macro-only section may not type because they carry a digit
    # (I_1, b_3/b_1, R^2, H1/H2); the raw value is emitted verbatim and must match a fixed
    # whitelist below so no arbitrary TeX can enter through this formatter.
    "symbol": lambda v: _symbol(str(v)),
}
SYMBOLS = frozenset({"I_1", "I_0", "b_3/b_1", "b_5/b_1", "R^2", "H1", "H2", "x^*", "L1a", "L1b", "v2", "v3", "v3.1", "v4"})


def _symbol(value: str) -> str:
    if value not in SYMBOLS:
        raise ValueError(f"symbol {value!r} is not whitelisted")
    return value


def _version_token(identifier: str) -> str:
    match = re.search(r"[-_](v\d+(?:\.\d+)?)$", identifier)
    if match is None:
        raise ValueError(f"identifier {identifier!r} carries no version token")
    return match.group(1)


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=FLOAT_TOLERANCE, abs_tol=0.0)


def _distribution_close(recomputed: dict[str, Any], recorded: dict[str, Any]) -> bool:
    if recomputed["count"] != recorded["count"]:
        return False
    for key in ("min", "median", "max"):
        a, b = recomputed[key], recorded[key]
        if (a is None) != (b is None):
            return False
        if a is not None and a != b:
            return False
    return True


# --------------------------------------------------------------------------- #
# Modified Bessel functions (the experiment's series, descriptors.py)
# --------------------------------------------------------------------------- #
def bessel_i(order: int, x: float) -> float:
    half = 0.5 * float(x)
    term = half**order / math.factorial(order)
    total = term
    for m in range(1, 400):
        term = term * half * half / (m * (m + order))
        total += term
        if abs(term) <= 1e-17 * abs(total):
            break
    return total


def i1_root(target: float, *, low: float = 1.0e-6, high: float = 12.0) -> float:
    if not bessel_i(1, low) < target < bessel_i(1, high):
        raise ValueError("target outside the bracket")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if bessel_i(1, middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# --------------------------------------------------------------------------- #
# Committed bindings
# --------------------------------------------------------------------------- #
def bind_committed(repo: Path, bundle: Bundle) -> dict[str, Any]:
    """Prove the working-tree bundle equals the committed record revision."""

    head = _git(repo, "rev-parse", "HEAD")
    for commit, label in (
        (RESULTS_COMMIT_SHA, "results"),
        (PREREGISTRATION_COMMIT_SHA, "preregistration"),
        (DASHBOARD_COMMIT_SHA, "dashboard"),
        (LITERATURE_COMMIT_SHA, "literature review"),
        (SWEEP_V2_RESULTS_COMMIT_SHA, "sweep v2 results"),
        (TOPOLOGY_RESULTS_COMMIT_SHA, "cusp topology v3.1 results"),
        (WALL_LOSS_PREREGISTRATION_COMMIT_SHA, "wall-loss v4 preregistration"),
    ):
        if not _is_ancestor(repo, commit, head):
            raise ValueError(f"{label} commit is not an ancestor of HEAD")
    for earlier, later, label in (
        (PREREGISTRATION_COMMIT_SHA, RESULTS_COMMIT_SHA, "preregistration -> results"),
        (RESULTS_COMMIT_SHA, DASHBOARD_COMMIT_SHA, "results -> dashboard"),
        (LITERATURE_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "literature review -> preregistration"),
        (TOPOLOGY_RESULTS_COMMIT_SHA, LITERATURE_COMMIT_SHA, "topology v3.1 results -> literature review"),
        (SWEEP_V2_RESULTS_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "sweep v2 results -> preregistration"),
        (WALL_LOSS_PREREGISTRATION_COMMIT_SHA, PREREGISTRATION_COMMIT_SHA, "wall-loss preregistration -> preregistration"),
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
        "literature_review_commit": LITERATURE_COMMIT_SHA,
        "reference_commits": {
            "sweep_v2": SWEEP_V2_RESULTS_COMMIT_SHA,
            "cusp_topology_v3_1": TOPOLOGY_RESULTS_COMMIT_SHA,
            "wall_loss_v4_preregistration": WALL_LOSS_PREREGISTRATION_COMMIT_SHA,
        },
    }


def cross_check_dashboard(repo: Path, bundle: Bundle, dataset: dict[str, Any], campaign: dict[str, Any], gates: dict[str, Any], catalogue: dict[str, Any]) -> dict[str, Any]:
    """The committed dashboard is an independent extraction of the same bundle; it must agree."""

    generator_raw = (repo / DASHBOARD_GENERATOR).read_bytes()
    template_raw = (repo / DASHBOARD_TEMPLATE).read_bytes()
    html_raw = (repo / DASHBOARD_HTML).read_bytes()
    generator_text = _lf(generator_raw).decode("utf-8")
    if f'CLASSIFICATION = "{CLASSIFICATION}"' not in generator_text or f'TOPOLOGY_LABEL = "{TOPOLOGY_LABEL}"' not in generator_text:
        raise ValueError("dashboard generator does not pin the sweep labels")
    if 'expected_state: str = "accepted_result"' not in generator_text or f'campaign["status"] != "{CAMPAIGN_STATUS}"' not in generator_text:
        raise ValueError("dashboard generator does not verify the bundle state and campaign status")
    payload = dashboard_payload(_lf(html_raw))
    identity = payload["identity"]
    if identity["manifest_file_sha256"] != bundle.manifest_sha256 or identity["state"] != "accepted_result":
        raise ValueError("dashboard payload names a different results manifest")
    if identity["preregistration_commit_sha"] != PREREGISTRATION_COMMIT_SHA or identity["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("dashboard payload names a different preregistration commit or experiment")
    if identity["verified_file_count"] != len(bundle.hashes) or identity["artifact_count"] != bundle.manifest["artifact_count"]:
        raise ValueError("dashboard payload file counts differ from the bundle")
    if identity["terminal_file_sha256"] != bundle.manifest["terminal_byte_sha256"] or identity["lock_file_sha256"] != bundle.manifest["lock_byte_sha256"]:
        raise ValueError("dashboard payload terminal/lock hashes differ from the bundle")
    for key in ("protocol_semantic_sha256", "experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
        if identity[key] != dataset[key]:
            raise ValueError(f"dashboard payload {key} differs from the sealed dataset")
    if identity["sealed_sources"] != dataset["sealed_sources"]:
        raise ValueError("dashboard payload sealed sources differ from the sealed dataset")
    if identity["generator_sha256"] != sha256_bytes(_lf(generator_raw)) or identity["template_sha256"] != sha256_bytes(_lf(template_raw)):
        raise ValueError("dashboard payload generator/template hashes differ from the checkout")
    if identity["protocol_file_sha256_lf"] != sha256_bytes(_lf((repo / EXPERIMENT / "protocol.json").read_bytes())):
        raise ValueError("dashboard payload protocol hash differs from the frozen protocol")
    if payload["classification"] != CLASSIFICATION or payload["topology_label"] != TOPOLOGY_LABEL:
        raise ValueError("dashboard classification differs from the bundle")
    if payload["headline"] != dataset["headline"] or payload["held_out"] != dataset["held_out"]:
        raise ValueError("dashboard headline or held-out block differs from the sealed dataset")
    if payload["claim_boundary"] != dataset["claim_boundary"] or payload["classification_statement"] != dataset["classification_statement"]:
        raise ValueError("dashboard claim boundary differs from the sealed dataset")
    if payload["hypothesis"] != dataset["hypothesis"] or payload["hypothesis_outcome"]["test"] != dataset["estimands"]["sobol_v3"]["hypothesis_test"]:
        raise ValueError("dashboard hypothesis block differs from the sealed dataset")
    if any(payload["estimands"][s] != dataset["estimands"][s] for s in ("sobol_v3", "sweep_v2", "pooled_all", "sweep_v2_region_pooled")):
        raise ValueError("dashboard estimands differ from the sealed dataset")
    if payload["gates"]["campaign"] != gates["campaign"] or payload["gates"]["replays"] != gates["replays"] or payload["gates"]["sweep_v2_gate_breakdown"] != gates["sweep_v2_gate_breakdown"]:
        raise ValueError("dashboard gates differ from the sealed gates")
    if payload["execution"] != campaign["execution_mode"] or payload["l1b_p2_queue"] != campaign["l1b_p2_confirmation_queue"]:
        raise ValueError("dashboard execution record or confirmation queue differs from the campaign result")
    rows = {(item["set"], item["id"]): item for item in payload["designs"]}
    if set(rows) != {(d["set_id"], d["design_id"]) for d in dataset["designs"]}:
        raise ValueError("dashboard design rows differ from the sealed dataset")
    for design in dataset["designs"]:
        row = rows[(design["set_id"], design["design_id"])]
        if row["cusps"] != design["wall_cusp_count"] or row["z_c_m"] != [c["z_c_m"] for c in design["wall_cusps"]] or row["rho"] != [r["rho_conservative"] for r in design["rho"]]:
            raise ValueError(f"dashboard row {design['key']} differs from the sealed dataset")
        if row["hemp"] is not design["hemp_like_all_cusps"] or row["stable"] is not design["stability"]["stable"] or row["x_w"] != design["x_w"] or row["in_v2_box"] is not design["inside_sweep_v2_box"]:
            raise ValueError(f"dashboard row {design['key']} flags or x_w differ")
    if payload["catalogue"]["design_count"] != catalogue["design_count"] or payload["catalogue"]["hemp_like_design_count"] != catalogue["hemp_like_design_count"]:
        raise ValueError("dashboard catalogue counts differ from the sealed catalogue")
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
            "the committed dashboard byte-verifies the accepted bundle, embeds its own extraction and pins the "
            "manifest SHA-256 and the preregistration commit; the generator requires that extraction (identity "
            "incl. the sealed sources, headline, held-out, hypothesis block, estimands, gates, execution, "
            "confirmation queue, every per-design row and the catalogue counts) to equal the sealed artifacts "
            "before writing any macro"
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

    def add_derived(self, name: str, raw: Any, fmt: str, description: str, derivation: str, inputs: list[dict[str, str]]) -> Any:
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


def _range_text(values: list[float], fmt: str) -> str:
    return f"{format_value(fmt, min(values))}--{format_value(fmt, max(values))}"


def _short_id(design_id: str) -> str:
    """``l1a-gs-v3-005-0e7f21e31d`` -> ``005`` (the ordinal token of the design identifier)."""

    match = re.fullmatch(r"l1a-gs-v[23]-(\d{3})-[0-9a-f]{10}", design_id)
    if match is None:
        raise ValueError(f"unexpected design identifier {design_id!r}")
    return match.group(1)


# --------------------------------------------------------------------------- #
# Recomputation of the sealed estimands from the rows
# --------------------------------------------------------------------------- #
def hypothesis_test(rows: list[dict[str, Any]], band: float, x_star: float) -> dict[str, Any]:
    """Re-derive experiment.hypothesis_test from the dataset rows (numpy sums -> fsum)."""

    pairs = [(d["ppm_prediction"]["i1_x_w"], r["rho_conservative"]) for d in rows for r in d["rho"] if r["rho_conservative"] is not None]
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    slope = math.fsum(a * b for a, b in pairs) / math.fsum(a * a for a in x)
    ss_res = math.fsum((b - slope * a) ** 2 for a, b in pairs)
    mean_y = math.fsum(y) / len(y)
    ss_tot = math.fsum((b - mean_y) ** 2 for b in y)
    ratios = [b / a for a, b in pairs]
    within = [abs(r - 1.0) <= band for r in ratios]
    design_level = [(bool(d["predicted_hemp_like_i1"]), bool(d["hemp_like_all_cusps"]), d["x_w"]) for d in rows if d["wall_cusp_count"] > 0]
    confusion = {
        "predicted_and_realised": sum(1 for p, r, _ in design_level if p and r),
        "predicted_not_realised": sum(1 for p, r, _ in design_level if p and not r),
        "not_predicted_but_realised": sum(1 for p, r, _ in design_level if not p and r),
        "neither": sum(1 for p, r, _ in design_level if not p and not r),
    }
    realised_x = sorted(xw for _, r, xw in design_level if r)
    not_realised_x = sorted(xw for _, r, xw in design_level if not r)
    threshold_from_slope = i1_root(1.5 / slope) if slope > 0.0 else None
    return {
        "cusp_count": len(pairs),
        "design_count_with_cusps": len(design_level),
        "slope_through_origin": slope,
        "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
        "rho_over_i1": _distribution(ratios),
        "fraction_within_band": sum(within) / len(within),
        "band": band,
        "confusion_predicted_i1_vs_realised": confusion,
        "prediction_accuracy": (confusion["predicted_and_realised"] + confusion["neither"]) / len(design_level),
        "smallest_x_w_realised_hemp_like": realised_x[0] if realised_x else None,
        "largest_x_w_not_hemp_like": not_realised_x[-1] if not_realised_x else None,
        "x_star_prediction": x_star,
        "x_star_from_fitted_slope": threshold_from_slope,
        "wall_radius_over_pitch_star_from_fitted_slope": (threshold_from_slope / math.pi) if threshold_from_slope else None,
    }


def _compare_hypothesis(recomputed: dict[str, Any], recorded: dict[str, Any], label: str) -> None:
    if set(recomputed) != set(recorded):
        raise ValueError(f"{label}: hypothesis-test keys differ from the sealed record")
    for key in ("cusp_count", "design_count_with_cusps", "band", "confusion_predicted_i1_vs_realised", "smallest_x_w_realised_hemp_like", "largest_x_w_not_hemp_like", "x_star_prediction"):
        if recomputed[key] != recorded[key]:
            raise ValueError(f"{label}: hypothesis-test {key} does not recompute from the rows")
    if not _distribution_close(recomputed["rho_over_i1"], recorded["rho_over_i1"]):
        raise ValueError(f"{label}: hypothesis-test rho_over_i1 distribution does not recompute from the rows")
    for key in ("slope_through_origin", "r_squared", "fraction_within_band", "prediction_accuracy", "x_star_from_fitted_slope", "wall_radius_over_pitch_star_from_fitted_slope"):
        a, b = recomputed[key], recorded[key]
        if (a is None) != (b is None) or (a is not None and not _close(a, b)):
            raise ValueError(f"{label}: hypothesis-test {key} does not recompute from the rows within {FLOAT_TOLERANCE:g}")


def set_estimands(rows: list[dict[str, Any]], band: float, x_star: float) -> dict[str, Any]:
    """Re-derive experiment.set_estimands from the dataset rows."""

    with_cusps = [d for d in rows if d["wall_cusp_count"] > 0]
    cusps = [r for d in rows for r in d["rho"]]
    by_stage: dict[str, dict[str, Any]] = {}
    for d in rows:
        stages = d["derived"]["stage_count"]
        bucket = by_stage.setdefault(str(stages), {"designs": 0, "hemp_like": 0, "n_minus_1_cusps": 0, "x_w": []})
        bucket["designs"] += 1
        bucket["hemp_like"] += int(d["hemp_like_all_cusps"])
        bucket["n_minus_1_cusps"] += int(d["wall_cusp_count"] == stages - 1)
        bucket["x_w"].append(d["x_w"])
    for bucket in by_stage.values():
        bucket["x_w"] = _distribution(bucket["x_w"])
    hemp = [d for d in rows if d["hemp_like_all_cusps"]]
    return {
        "design_count": len(rows),
        "stable_design_count": sum(d["stability"]["stable"] for d in rows),
        "v2_gates_passed_count": sum(d["v2_gates"]["passed"] for d in rows),
        "wall_cusp_count_histogram": _histogram([d["wall_cusp_count"] for d in rows]),
        "axis_null_count_histogram": _histogram([d["axis_null_count"] for d in rows]),
        "n_minus_1_cusp_fraction": sum(d["wall_cusp_count"] == d["derived"]["stage_count"] - 1 for d in rows) / len(rows),
        "hemp_like_count": len(hemp),
        "hemp_like_fraction": len(hemp) / len(rows),
        "hemp_like_fraction_among_designs_with_cusps": len(hemp) / len(with_cusps),
        "predicted_hemp_like_i1_count": sum(d["predicted_hemp_like_i1"] for d in rows),
        "five_stage_four_cusp_hemp_like_count": sum(d["five_stage_four_cusp_hemp_like"] for d in rows),
        "four_wall_cusp_count": sum(d["four_wall_cusps"] for d in rows),
        "by_stage_count": by_stage,
        "x_w": _distribution([d["x_w"] for d in rows]),
        "wall_radius_over_pitch": _distribution([d["wall_radius_over_pitch"] for d in rows]),
        "rho_conservative": _distribution([r["rho_conservative"] for r in cusps]),
        "rho_downstream": _distribution([r["rho_downstream"] for r in cusps]),
        "rho_wall": _distribution([r["rho_wall"] for r in cusps]),
        "cusp_is_wall_maximum_count": sum(r["cusp_is_wall_maximum"] for r in cusps),
        "cusp_count": len(cusps),
        "angle_to_wall_normal_deg": _distribution([c["angle_to_wall_normal_deg"] for d in rows for c in d["wall_cusps"]]),
        "wall_b3_over_b1": _distribution([d["wall_harmonics"]["b3_over_b1"] for d in rows if d["wall_harmonics"]["applies"]]),
        "wall_b5_over_b1": _distribution([d["wall_harmonics"]["b5_over_b1"] for d in rows if d["wall_harmonics"]["applies"]]),
        "rho_resolution_sensitivity_max": _distribution([d["resolution_sensitivity"]["max_relative_rho_difference"] for d in rows]),
        "max_wall_intersection_shift_m": _distribution([d["stability"]["max_wall_intersection_shift_m"] for d in rows]),
        "hemp_like_region": None if not hemp else {
            "x_w": _distribution([d["x_w"] for d in hemp]),
            "wall_radius_over_pitch": _distribution([d["wall_radius_over_pitch"] for d in hemp]),
            "x_m_inner": _distribution([d["x_m_inner"] for d in hemp]),
            "stage_counts": _histogram([d["derived"]["stage_count"] for d in hemp]),
            "wall_cusp_counts": _histogram([d["wall_cusp_count"] for d in hemp]),
            "min_rho_conservative": _distribution([d["min_rho_conservative"] for d in hemp]),
        },
        "hypothesis_test": hypothesis_test(rows, band, x_star),
    }


def _compare_estimands(recomputed: dict[str, Any], recorded: dict[str, Any], label: str) -> None:
    if set(recomputed) != set(recorded):
        raise ValueError(f"{label}: estimand keys differ from the sealed record")
    for key, value in recomputed.items():
        other = recorded[key]
        if key == "hypothesis_test":
            _compare_hypothesis(value, other, label)
        elif key in ("n_minus_1_cusp_fraction", "hemp_like_fraction", "hemp_like_fraction_among_designs_with_cusps"):
            if not _close(value, other):
                raise ValueError(f"{label}: estimand {key} does not recompute from the rows")
        elif key == "by_stage_count":
            if set(value) != set(other):
                raise ValueError(f"{label}: stage buckets differ from the sealed record")
            for stages, bucket in value.items():
                rec = other[stages]
                if any(bucket[k] != rec[k] for k in ("designs", "hemp_like", "n_minus_1_cusps")) or not _distribution_close(bucket["x_w"], rec["x_w"]):
                    raise ValueError(f"{label}: stage bucket {stages} does not recompute from the rows")
        elif key == "hemp_like_region":
            if (value is None) != (other is None):
                raise ValueError(f"{label}: HEMP-like region presence differs from the sealed record")
            if value is not None:
                for sub in ("x_w", "wall_radius_over_pitch", "x_m_inner", "min_rho_conservative"):
                    if not _distribution_close(value[sub], other[sub]):
                        raise ValueError(f"{label}: HEMP-like region {sub} does not recompute from the rows")
                if value["stage_counts"] != other["stage_counts"] or value["wall_cusp_counts"] != other["wall_cusp_counts"]:
                    raise ValueError(f"{label}: HEMP-like region histograms do not recompute from the rows")
        elif isinstance(value, dict) and "count" in value and set(value) == {"count", "min", "median", "max"}:
            if not _distribution_close(value, other):
                raise ValueError(f"{label}: estimand {key} does not recompute from the rows")
        elif value != other:
            raise ValueError(f"{label}: estimand {key} does not recompute from the rows")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(repo: Path) -> tuple[dict[str, Any], str]:
    """Return (evidence document, generated TeX)."""

    bundle = Bundle(repo, RESULTS, experiment_id=EXPERIMENT_ID, expected_state="accepted_result")
    binding = bind_committed(repo, bundle)
    m = Macros(bundle)
    terminal = m.doc("terminal.json")
    lock = m.doc("execution-lock.json")
    campaign = m.doc("artifacts/campaign-result.json")
    gates = m.doc("artifacts/gates.json")
    dataset = m.doc("artifacts/sweep-dataset.json")
    protocol = m.doc("artifacts/protocol.json")
    authorities = m.doc("artifacts/authorities.json")
    shakedown = m.doc("artifacts/shakedown.json")
    design_authorities = m.doc("artifacts/design-authorities.json")
    plan = m.doc("artifacts/campaign-plan.json")
    runtime = m.doc("artifacts/runtime.json")
    failures = m.doc("artifacts/design-failures.json")
    source_binding = m.doc("artifacts/source-binding.json")
    catalogue = m.doc("artifacts/cusp-cell-catalogue-v3.json")
    designs = dataset["designs"]
    headline = dataset["headline"]
    descriptors = protocol["descriptors_v3"]
    sampling = protocol["sampling"]

    # ---- reference files: the sealed sources and definition authorities ----
    sealed = dataset["sealed_sources"]
    sweep_v2_file = _bound_file(repo, SWEEP_V2_MANIFEST, SWEEP_V2_RESULTS_COMMIT_SHA, "reference-sweep-manifest", lf_equal=False)
    if sweep_v2_file["sha256"] != sealed["sweep_v2"]["manifest_file_sha256"]:
        raise ValueError("the sweep-v2 results manifest on disk differs from the sealed source the campaign bound")
    topology_protocol_file = _bound_file(repo, TOPOLOGY_PROTOCOL, TOPOLOGY_RESULTS_COMMIT_SHA, "reference-topology-protocol", lf_equal=False)
    topology_protocol = load_json_bytes((repo / TOPOLOGY_PROTOCOL).read_bytes(), "cusp topology v3.1 protocol")
    definition_import = protocol["definition_v3_import"]
    if definition_import["numerical_parameters"] != topology_protocol["definition_v3"]["numerical_parameters"]:
        raise ValueError("the imported definition parameters differ from the frozen cusp-topology-v3.1 protocol")
    for key in ("stability_tolerance_m", "held_out_tolerance_m"):
        if definition_import[key] != topology_protocol["definition_v3"][key]:
            raise ValueError(f"the imported {key} differs from the frozen cusp-topology-v3.1 protocol")
    if TOPOLOGY_RESULTS_COMMIT_SHA[:8] not in definition_import["source"] or TOPOLOGY_PROTOCOL.name not in definition_import["source"]:
        raise ValueError("the definition import does not name the cusp-topology-v3.1 protocol at its result commit")
    p2_record_file = _bound_file(repo, TOPOLOGY_P2_RECORD, TOPOLOGY_RESULTS_COMMIT_SHA, "reference-topology-p2-record", lf_equal=False)
    p2_record = load_json_bytes((repo / TOPOLOGY_P2_RECORD).read_bytes(), "cusp topology v3.1 P2 record")
    wall_loss_protocol_file = _bound_file(repo, WALL_LOSS_PROTOCOL, WALL_LOSS_PREREGISTRATION_COMMIT_SHA, "reference-wall-loss-protocol", lf_equal=False)
    wall_loss_protocol = load_json_bytes((repo / WALL_LOSS_PROTOCOL).read_bytes(), "wall-loss v4 protocol")
    review_file = _bound_file(repo, LITERATURE_REVIEW, LITERATURE_COMMIT_SHA, "definition-source-review", lf_equal=True)
    check_script_file = _bound_file(repo, PPM_CHECK_SCRIPT, LITERATURE_COMMIT_SHA, "definition-source-check-script", lf_equal=True)
    check_output_file = _bound_file(repo, PPM_CHECK_OUTPUT, LITERATURE_COMMIT_SHA, "definition-source-check-output", lf_equal=True)
    review_text = _lf((repo / LITERATURE_REVIEW).read_bytes()).decode("utf-8")
    check_output = load_json_bytes(_lf((repo / PPM_CHECK_OUTPUT).read_bytes()), "PPM check output")
    check_script_text = _lf((repo / PPM_CHECK_SCRIPT).read_bytes()).decode("utf-8")
    if LITERATURE_REVIEW.as_posix() not in protocol["purpose"] or LITERATURE_COMMIT_SHA[:8] not in protocol["purpose"]:
        raise ValueError("the frozen protocol does not name the bound literature review at its commit")
    for key in LITERATURE_KEYS:
        token = LITERATURE_TOKENS[key]
        if token not in descriptors["koch_rho"]["citation"] or token not in review_text:
            raise ValueError(f"literature source {key} ({token}) is absent from the protocol citation or the bound review")
    if "IEPC-2007-110" not in review_text or "Table G2" not in review_text:
        raise ValueError("the bound review does not carry the Koch design ratio and the launch-cell reflection table")
    if PPM_CHECK_OUTPUT.name not in check_script_text and "--json" not in check_script_text:
        raise ValueError("the check script does not write the committed output")
    if check_output["topology_catalogue"] != f"cusp_topology_search_v3_1 ({topology_protocol['experiment_id']}), recorded status accepted_topology_screening":
        raise ValueError("the check output does not read the accepted cusp-topology-v3.1 catalogue")
    dashboard = cross_check_dashboard(repo, bundle, dataset, campaign, gates, catalogue)

    # ---- internal consistency of the sealed bundle (fail closed on any disagreement) ----
    if terminal["state"] != bundle.manifest["state"] or terminal["counts"]["attempt_count"] != 1:
        raise ValueError("terminal record disagrees with the manifest or records more than one attempt")
    if terminal["payload"] != {"design_count": campaign["design_count"], "gates": campaign["campaign_gates"], "stable_design_count": headline["stable_design_count"], "status": campaign["status"]}:
        raise ValueError("terminal payload differs from the campaign result")
    if lock["attempt"] != 1 or lock["immutable"] is not True or lock["commit"] != PREREGISTRATION_COMMIT_SHA or lock["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("execution lock does not record the single immutable attempt at the preregistration commit")
    if campaign["status"] != CAMPAIGN_STATUS or campaign["evidentiary"] is not True or campaign["plan_kind"] != "evidentiary" or campaign["gates_passed"] is not True:
        raise ValueError("campaign result is not the accepted evidentiary sweep")
    if not (campaign["classification"] == dataset["classification"] == protocol["classification"] == authorities["classification"] == shakedown["classification"] == CLASSIFICATION):
        raise ValueError("classification differs between the sealed artifacts")
    if not (campaign["topology_label"] == dataset["topology_label"] == protocol["catalogue"]["label"] == TOPOLOGY_LABEL) or catalogue["labels"] != [TOPOLOGY_LABEL]:
        raise ValueError("topology label differs between the sealed artifacts")
    if campaign["campaign_gates"] != gates["campaign"] or gates["passed"] is not True or gates["binding"] is not True or set(gates["campaign"]) != set(BINDING_GATE_NAMES):
        raise ValueError("gates.json disagrees with the campaign result or names a different gate set")
    if any(gates["campaign"][name] is not True for name in BINDING_GATE_NAMES) or any(gates["failing_designs"].values()):
        raise ValueError("gates.json records a failed binding gate or a failing design")
    if set(gates["definitions"]["binding_integrity"]) != set(BINDING_GATE_NAMES) or gates["definitions"]["binding_integrity"] != protocol["gates"]["binding_integrity"]:
        raise ValueError("gate definitions differ from the frozen protocol")
    if gates["definitions"]["reported_not_binding"] != protocol["gates"]["reported_not_binding"] or set(gates["failing_designs"]) != set(PER_DESIGN_GATES):
        raise ValueError("reported-not-binding list or per-design gate set differs from the frozen protocol")
    if campaign["headline"] != headline or dataset["gates"] != {"campaign": gates["campaign"], "failing_designs": gates["failing_designs"], "passed": True, "sweep_v2_gate_breakdown": gates["sweep_v2_gate_breakdown"]}:
        raise ValueError("campaign headline or dataset gate block differs from the sealed gates")
    if not (len(designs) == dataset["design_count"] == campaign["design_count"] == gates["design_count"] == headline["design_count"] == catalogue["design_count"] == authorities["design_count"] == design_authorities["design_count"] == len(plan["design_keys"])):
        raise ValueError("design count differs between the sealed artifacts")
    if plan["kind"] != "evidentiary" or plan["binding_gates"] is not True or plan["design_keys"] != [d["key"] for d in designs]:
        raise ValueError("campaign plan differs from the dataset order")
    if failures["failed"] != [] or campaign["design_count"] != headline["stable_design_count"]:
        raise ValueError("the bundle records a failed or unstable design")
    if not (campaign["set_counts"] == headline["set_counts"] == authorities["set_counts"] == design_authorities["set_counts"] == {s: sum(1 for d in designs if d["set_id"] == s) for s in SET_IDS}):
        raise ValueError("set counts differ between the sealed artifacts and the rows")
    if headline["set_counts"]["sobol_v3"] != sampling["design_count"] or headline["set_counts"]["sweep_v2"] != protocol["design_sets"]["sweep_v2"]["design_count"]:
        raise ValueError("set counts differ from the frozen protocol")
    for key in ("protocol_semantic_sha256", "experiment_code_sha256", "dependency_source_sha256", "field_pipeline_source_sha256"):
        if not (dataset[key] == authorities[key] == shakedown[key] == source_binding[key]):
            raise ValueError(f"{key} differs between the sealed artifacts")
    if campaign["protocol_semantic_sha256"] != dataset["protocol_semantic_sha256"] or catalogue["protocol_semantic_sha256"] != dataset["protocol_semantic_sha256"]:
        raise ValueError("protocol semantic hash differs between campaign result, dataset and catalogue")
    if not (sealed == authorities["sealed_sources"] == source_binding["sealed_sources"] == shakedown["sealed_sources"]):
        raise ValueError("sealed-source identities differ between the sealed artifacts")
    if sealed["sweep_v2"]["preregistration_commit"] != "092f5fae692ee7d6711e0c7e1c94dac6a345f37c":
        raise ValueError("the sealed sweep-v2 source names a different preregistration")
    if authorities["shakedown_file_sha256"] != bundle.hashes["artifacts/shakedown.json"] or authorities["design_authorities_sha256"] != bundle.hashes["artifacts/design-authorities.json"]:
        raise ValueError("shakedown or design-authorities artifact differs from the bound authority")
    if shakedown["passed"] is not True or shakedown["evidentiary"] is not False or shakedown["outcomes_enter_estimand"] is not False:
        raise ValueError("shakedown is not a passing non-evidentiary record")
    if shakedown["timing_projection"] != authorities["shakedown_timing_projection"] or shakedown["timing_projection"]["within_budget"] is not True:
        raise ValueError("shakedown timing projection differs from the authorities or is out of budget")
    if authorities["shakedown_git_head"] != LITERATURE_COMMIT_SHA or shakedown["git"]["head"] != LITERATURE_COMMIT_SHA:
        raise ValueError("the shakedown was not run at the literature-review commit the protocol chain records")
    shakedown_designs = [f"{s}:{d}" for s in SET_IDS for d in protocol["shakedown"]["designs"][s]]
    if shakedown["shakedown_plan"]["design_keys"] != shakedown_designs or shakedown["design_count"] != len(shakedown_designs):
        raise ValueError("shakedown design keys differ from the frozen protocol")
    if authorities["sobol_predicted_hemp_like_count"] != headline["sobol_predicted_hemp_like_i1_count"]:
        raise ValueError("the frozen authorities predicted a different HEMP-like count than the dataset records")
    for frozen in FROZEN_FILES:
        if load_json_bytes((repo / EXPERIMENT / frozen).read_bytes(), frozen) != m.doc(f"artifacts/{frozen}"):
            raise ValueError(f"frozen {frozen} differs from the sealed copy in the bundle")
    if runtime["worker_pool_size"] != campaign["execution_mode"]["worker_pool_size"] or runtime["worker_pool_size"] != protocol["execution"]["max_design_workers"]:
        raise ValueError("worker pool differs between runtime, campaign result and protocol")
    if "gpu-not-used" not in lock["device"] or "GPU not used" not in runtime["backend"]:
        raise ValueError("the execution did not record a CPU-only run")
    replay_keys = [f"{s}:{d}" for s in SET_IDS for d in protocol["execution"]["replay_designs"][s]]
    if [r["key"] for r in gates["replays"]] != replay_keys or any(not (r["bit_identical"] and r["field_identity_equal"] and r["accepted_grid_equal"] and r["replay_topology_payload_sha256"] == r["worker_topology_payload_sha256"]) for r in gates["replays"]):
        raise ValueError("determinism replays differ from the frozen protocol or were not bit-identical")
    if dataset["claim_boundary"] != protocol["claim_boundary"] or dataset["hypothesis"] != descriptors["hypothesis"]:
        raise ValueError("dataset claim boundary or hypothesis differs from the frozen protocol")
    if campaign["l1b_p2_confirmation_queue"] != protocol["claim_boundary"]["l1b_p2_confirmation"] or campaign["l1b_p2_confirmation_queue"]["status"] != "queued_not_run":
        raise ValueError("the material-aware confirmation queue differs from the frozen protocol or was run")
    if catalogue["experiment_id"] != EXPERIMENT_ID or catalogue["mirror_descriptor_statement"] is not True or catalogue["hemp_like_rule"] != descriptors["hemp_like_rule"]:
        raise ValueError("catalogue identity, mirror statement or HEMP-like rule differ from the campaign")
    if catalogue["stable_design_count"] != headline["stable_design_count"] or len(catalogue["entries"]) != len(designs) or catalogue["hemp_like_design_count"] != headline["sobol_hemp_like_count"]:
        raise ValueError("catalogue counts differ from the dataset")
    if catalogue["schema_version"] != protocol["catalogue"]["schema_version"]:
        raise ValueError("catalogue schema differs from the frozen protocol")
    box_v2 = sampling["sweep_v2_box"]
    variables = {v["name"]: v for v in sampling["variables"]}
    for name, _token, _label in WIDENED_VARIABLES:
        lo, hi = box_v2[name]
        if not (variables[name]["lower"] <= lo and hi <= variables[name]["upper"]) or (variables[name]["lower"], variables[name]["upper"]) == (lo, hi):
            raise ValueError(f"the v3 interval of {name} does not strictly contain the v2 interval")
    coverage = sampling["regime_coverage"]
    box_ratio = (variables["chamber_outer_radius_m"]["lower"] / variables["stage_pitch_m"]["upper"], variables["chamber_outer_radius_m"]["upper"] / variables["stage_pitch_m"]["lower"])
    if not (_close(round(box_ratio[0], 4), coverage["wall_radius_over_pitch"][0]) and _close(round(box_ratio[1], 4), coverage["wall_radius_over_pitch"][1])):
        raise ValueError("the protocol's r_w / L coverage does not follow from the variable box")
    if not (_close(round(math.pi * box_ratio[0], 4), coverage["x_w"][0]) and _close(round(math.pi * box_ratio[1], 4), coverage["x_w"][1])):
        raise ValueError("the protocol's x_w coverage does not follow from the variable box")
    x_star = i1_root(1.5)
    if not _close(x_star, headline["sobol_hypothesis_test"]["x_star_prediction"]) or "1.937318" not in descriptors["ppm_prediction"]["i1_threshold"]:
        raise ValueError("the single-harmonic threshold x* does not recompute from the Bessel series")
    if not _close(bessel_i(1, x_star), 1.5):
        raise ValueError("I_1(x*) does not evaluate to the HEMP-like threshold")

    # ---- per-design cross-checks against the design records, catalogue, CSV and field grids ----
    by_key = {d["key"]: d for d in designs}
    catalogue_by_key = {f"{e['set_id']}:{e['design_id']}": e for e in catalogue["entries"]}
    if set(catalogue_by_key) != set(by_key):
        raise ValueError("catalogue entries do not cover exactly the dataset designs")
    csv_rows = list(csv.DictReader(io.StringIO(bundle.raw("artifacts/sweep-dataset.csv").decode("utf-8"))))
    if [f"{r['set_id']}:{r['design_id']}" for r in csv_rows] != [d["key"] for d in designs]:
        raise ValueError("dataset CSV rows differ from the dataset order")
    authority_by_key = {e["key"]: e for e in design_authorities["designs"]}
    if set(authority_by_key) != set(by_key):
        raise ValueError("the frozen design authorities do not declare exactly the dataset designs")
    representative_count = 0
    replays_bit_identical = sum(1 for r in gates["replays"] if r["bit_identical"])
    axis_shift_max = 0.0
    end_ratios: list[float] = []
    interior_ratios: list[float] = []
    predicted_only_end_failures = 0
    for index, design in enumerate(designs):
        key = design["key"]
        label = f"design {key}"
        set_id = design["set_id"]
        if set_id not in SET_IDS or key != f"{set_id}:{design['design_id']}" or design["ordinal"] != sum(1 for d in designs[:index] if d["set_id"] == set_id):
            raise ValueError(f"{label}: key, set or ordinal is inconsistent")
        if design["label"] != TOPOLOGY_LABEL or design["classification"] != CLASSIFICATION:
            raise ValueError(f"{label}: labels differ from the campaign")
        authority = authority_by_key[key]
        if authority["case_sha256"] != design["identity"]["case_sha256"] or authority["x_w"] != design["x_w"] or authority["representative"] is not design["representative"] or authority["inside_sweep_v2_box"] is not design["inside_sweep_v2_box"] or authority["stage_count"] != design["derived"]["stage_count"]:
            raise ValueError(f"{label}: the frozen design authority differs from the dataset row")
        checks = design["gate_checks"]
        if set(checks) != set(PER_DESIGN_GATES) or any(v is not True for v in checks.values()) or gates["per_design"][key] != checks:
            raise ValueError(f"{label}: a per-design gate check failed or differs from gates.json")
        cusps = design["wall_cusps"]
        rho = design["rho"]
        stages = design["derived"]["stage_count"]
        if design["wall_cusp_count"] != len(cusps) or len(rho) != len(cusps) or design["cell_count"] != len(design["cells"]) or design["cell_count"] != len(cusps) + 1:
            raise ValueError(f"{label}: cusp, rho and cell counts are inconsistent")
        if [c["cusp_id"] for c in cusps] != [r["cusp_id"] for r in rho] or [c["z_c_m"] for c in cusps] != [r["z_c_m"] for r in rho]:
            raise ValueError(f"{label}: rho readings are not aligned with the wall cusps")
        if [c["z_c_m"] for c in cusps] != sorted(c["z_c_m"] for c in cusps) or design["axis_null_count"] != len(design["axis_nulls"]) or any(n["classification"] != "X" for n in design["axis_nulls"]):
            raise ValueError(f"{label}: cusps are not sorted or the axis nulls are not all X-type")
        if stages != len(design["geometry"]["stage_centres_m"]) or stages != min(5, 3 + int(design["design_values"]["stage_count_selector"] * 3)):
            raise ValueError(f"{label}: stage count does not follow the frozen mapping")
        if len(cusps) not in (stages - 1, stages - 2, stages + 1) or stages not in STAGE_TOKENS:
            raise ValueError(f"{label}: unexpected wall-cusp count for the stage count")
        pitch = design["derived"]["represented_stage_pitch_m"]
        r_w = design["geometry"]["wall_radius_m"]
        if not _close(design["wall_radius_over_pitch"], r_w / pitch) or not _close(design["x_w"], math.pi * r_w / pitch):
            raise ValueError(f"{label}: x_w or r_w / L does not recompute from the geometry")
        prediction = design["ppm_prediction"]
        if not (_close(prediction["i1_x_w"], bessel_i(1, design["x_w"])) and _close(prediction["i0_x_w"], bessel_i(0, design["x_w"])) and _close(prediction["i1_over_i0_x_w"], prediction["i1_x_w"] / prediction["i0_x_w"])):
            raise ValueError(f"{label}: the single-harmonic prediction does not recompute from the Bessel series")
        if prediction["predicted_hemp_like"] is not (prediction["i1_x_w"] >= 1.5) or design["predicted_hemp_like_i1"] is not prediction["predicted_hemp_like"] or prediction["x_w"] != design["x_w"]:
            raise ValueError(f"{label}: the predicted HEMP-like flag does not recompute")
        for reading in rho:
            if not _close(reading["rho_conservative"], reading["wall_b_t"] / max(reading["upstream_axis_peak_t"], reading["downstream_axis_peak_t"])):
                raise ValueError(f"{label}: rho_conservative does not recompute from its inputs")
            if not _close(reading["rho_downstream"], reading["wall_b_t"] / reading["downstream_axis_peak_t"]) or not _close(reading["rho_upstream"], reading["wall_b_t"] / reading["upstream_axis_peak_t"]):
                raise ValueError(f"{label}: rho_downstream or rho_upstream does not recompute")
            if reading["hemp_like_conservative"] is not (reading["rho_conservative"] >= 1.5) or reading["cusp_is_wall_maximum"] is not (reading["rho_wall"] >= 1.0 - 1e-9):
                raise ValueError(f"{label}: per-cusp flags do not recompute")
        if design["hemp_like_all_cusps"] is not (bool(rho) and all(r["hemp_like_conservative"] for r in rho)):
            raise ValueError(f"{label}: the HEMP-like flag does not recompute from the cusps")
        if design["min_rho_conservative"] != min(r["rho_conservative"] for r in rho):
            raise ValueError(f"{label}: min_rho_conservative does not recompute")
        if design["five_stage_four_cusp_hemp_like"] is not (stages == 5 and len(cusps) == 4 and design["hemp_like_all_cusps"]) or design["four_wall_cusps"] is not (len(cusps) == 4):
            raise ValueError(f"{label}: legacy-target flags do not recompute")
        in_box = all(box_v2[name][0] <= design["design_values"][name] <= box_v2[name][1] for name, _t, _l in WIDENED_VARIABLES)
        if design["inside_sweep_v2_box"] is not (set_id == "sweep_v2" or in_box):
            raise ValueError(f"{label}: the sweep-v2 box flag does not recompute from the design values")
        for reading_index, reading in enumerate(rho):
            ratio = reading["rho_conservative"] / prediction["i1_x_w"]
            if set_id == "sobol_v3":
                (end_ratios if reading_index in (0, len(rho) - 1) else interior_ratios).append(ratio)
        if set_id == "sobol_v3" and design["predicted_hemp_like_i1"] and not design["hemp_like_all_cusps"]:
            failing = [i for i, r in enumerate(rho) if not r["hemp_like_conservative"]]
            if all(i in (0, len(rho) - 1) for i in failing):
                predicted_only_end_failures += 1
        stability = design["stability"]
        if stability["stable"] is not True or not (stability["axis_null_count_equal"] and stability["wall_cusp_count_equal"] and stability["wall_reaching_count_equal"]):
            raise ValueError(f"{label}: stability flags differ from the recorded acceptance")
        if stability["max_wall_intersection_shift_m"] > definition_import["stability_tolerance_m"]:
            raise ValueError(f"{label}: wall-intersection shift exceeds the stability tolerance")
        axis_shift_max = max(axis_shift_max, stability["max_axis_null_shift_m"])
        held_out = design["held_out"]
        if held_out["applies"] is not (set_id == "sweep_v2") or held_out["passed"] is not True:
            raise ValueError(f"{label}: held-out applicability or outcome differs from the protocol")
        if held_out["applies"] and (held_out["observed_count"] != held_out["reference_count"] or held_out["qoi_replay_passed"] is not True or held_out["max_difference_m"] > definition_import["held_out_tolerance_m"]):
            raise ValueError(f"{label}: held-out correspondence or QoI replay is not the recorded pass")
        if design["v2_gates"]["passed"] is not True or set(design["v2_gates"]["gates"]) != set(V2_GATE_NAMES) or any(v is not True for v in design["v2_gates"]["gates"].values()):
            raise ValueError(f"{label}: the six sweep-v2 metric gates were not all passed")
        sensitivity = design["resolution_sensitivity"]
        if sensitivity["comparable"] is not True or sensitivity["hemp_like_flag_agrees"] is not True:
            raise ValueError(f"{label}: the refined-map rho reading is not comparable or flips the HEMP-like flag")
        record = m.doc(design["record_path"])
        if record["key"] != key or record["status"] != "resolved" or record["gate_checks"] != checks or record["geometry"] != design["geometry"]:
            raise ValueError(f"{label}: design record identity, status, checks or geometry differ from the dataset")
        accepted = record["accepted"]
        if [c["z_c_m"] for c in accepted["topology"]["wall_cusps"]] != [c["z_c_m"] for c in cusps] or accepted["topology"]["cell_count"] != design["cell_count"]:
            raise ValueError(f"{label}: design record topology differs from the dataset row")
        if [n["z_m"] for n in accepted["axis_nulls"]["nulls"]] != [n["z_m"] for n in design["axis_nulls"]] or not (accepted["axis_nulls"]["all_converged"] and accepted["axis_nulls"]["all_x_type"] and accepted["axis_nulls"]["all_classifications_agree"]):
            raise ValueError(f"{label}: design record axis nulls differ from the dataset or were not all converged X nulls")
        if not (accepted["all_traces_terminate_cleanly"] and accepted["all_wall_traces_flux_consistent"]) or len(accepted["separatrix_traces"]) != len(design["axis_nulls"]):
            raise ValueError(f"{label}: separatrix traces are not one per null, clean and flux-consistent")
        record_descriptors = record["descriptors"]["accepted"]
        if [r["rho_conservative"] for r in record_descriptors["cusps"]] != [r["rho_conservative"] for r in rho] or record_descriptors["hemp_like_all_cusps"] is not design["hemp_like_all_cusps"]:
            raise ValueError(f"{label}: design record rho readings differ from the dataset row")
        if record_descriptors["hemp_like_threshold"]["rho"] != 1.5 or not _close(record_descriptors["hemp_like_threshold"]["x_star"], x_star) or record_descriptors["ppm_prediction"] != prediction or record_descriptors["x_w"] != design["x_w"]:
            raise ValueError(f"{label}: design record threshold or prediction differ from the dataset row")
        record_sensitivity = record["descriptors"]["resolution_sensitivity"]
        if {k: record_sensitivity[k] for k in sensitivity} != sensitivity or max(abs(r["relative_rho_difference"]) for r in record_sensitivity["rows"]) != sensitivity["max_relative_rho_difference"]:
            raise ValueError(f"{label}: design record resolution sensitivity differs from the dataset")
        if {k: record["stability"][k] for k in stability} != stability or record["stability"]["tolerance_m"] != definition_import["stability_tolerance_m"] or max(record["stability"]["axis_null_shifts_m"]) != stability["max_axis_null_shift_m"]:
            raise ValueError(f"{label}: design record stability differs from the dataset")
        if record["evidence"]["identity_proven"] is not True or record["identity"]["accepted_field_identity_sha256"] != design["identity"]["accepted_field_identity_sha256"] or record["identity"]["refined_field_identity_sha256"] != design["identity"]["refined_field_identity_sha256"]:
            raise ValueError(f"{label}: field identity is not proven or differs from the dataset")
        for solve in ("accepted_solve", "refined_solve"):
            if record["evidence"][solve]["converged"] is not True or record["evidence"][solve]["relative_residual_l2"] > protocol["field"]["solver"]["relative_tolerance"]:
                raise ValueError(f"{label}: {solve} did not converge within the solver tolerance")
        if record["v2_gates"]["passed"] is not design["v2_gates"]["passed"] or {k: g["passed"] for k, g in record["v2_gates"]["gates"].items()} != design["v2_gates"]["gates"]:
            raise ValueError(f"{label}: design record sweep-v2 gates differ from the dataset row")
        if design["representative"]:
            representative_count += 1
        grid_path = f"artifacts/fields/{set_id}/{design['design_id']}.json.gz"
        grid, payload_sha = bundle.load_gzip(grid_path)
        if any(grid["identity"][k] != design["identity"][k] for k in design["identity"]) or grid["identity"] != record["identity"] or grid["key"] != key:
            raise ValueError(f"{label}: accepted field grid identity differs from the design record")
        domain = protocol["field"]["domain"]
        if len(grid["z_m"]) != domain["axial_intervals"] + 1 or len(grid["r_m"]) != domain["radial_intervals"] + 1 or len(grid["psi_wb"]) != len(grid["r_m"]) or any(len(line) != len(grid["z_m"]) for line in grid["psi_wb"]):
            raise ValueError(f"{label}: accepted field grid shape differs from the frozen domain")
        if grid["z_m"][0] != domain["z_min_m"] or grid["z_m"][-1] != domain["z_max_m"] or grid["r_m"][0] != 0.0 or grid["r_m"][-1] != domain["radius_m"]:
            raise ValueError(f"{label}: accepted field grid extent differs from the frozen domain")
        if record["accepted_grid_path"] != grid_path or record["accepted_grid_payload_sha256"] != payload_sha:
            raise ValueError(f"{label}: accepted field grid path or payload hash differs from the design record")
        entry = catalogue_by_key[key]
        if entry["stable"] is not True or entry["label"] != design["label"] or entry["wall_cusp_count"] != len(cusps) or [c["z_c_m"] for c in entry["wall_cusps"]] != [c["z_c_m"] for c in cusps]:
            raise ValueError(f"{label}: catalogue entry differs from the dataset row")
        if [r["rho_conservative"] for r in entry["rho"]] != [r["rho_conservative"] for r in rho] or entry["hemp_like_all_cusps"] is not design["hemp_like_all_cusps"] or entry["x_w"] != design["x_w"] or entry["inside_sweep_v2_box"] is not design["inside_sweep_v2_box"]:
            raise ValueError(f"{label}: catalogue rho, flags or x_w differ from the dataset row")
        if entry["geometry"] != design["geometry"] or entry["record_path"] != design["record_path"] or entry["stage_count"] != stages:
            raise ValueError(f"{label}: catalogue geometry, record path or stage count differ from the dataset row")
        row = csv_rows[index]
        if int(row["wall_cusp_count"]) != len(cusps) or int(row["stage_count"]) != stages or row["hemp_like_all_cusps"] != str(design["hemp_like_all_cusps"]) or row["stable"] != "True" or row["label"] != design["label"]:
            raise ValueError(f"{label}: CSV row differs from the dataset row")
        if float(row["x_w"]) != design["x_w"] or float(row["min_rho_conservative"]) != design["min_rho_conservative"]:
            raise ValueError(f"{label}: CSV x_w or minimum rho differ from the dataset row")
    if representative_count != len(protocol["design_sets"]["sobol_v3"]["representative_indices"]) + headline["held_out"]["stored_representatives_checked"]:
        raise ValueError("representative count differs from the frozen protocol's list plus the stored sweep-v2 representatives")

    # ---- re-derive the headline and every per-set estimand from the rows ----
    sobol_rows = [d for d in designs if d["set_id"] == "sobol_v3"]
    held_out_rows = [d for d in designs if d["set_id"] == "sweep_v2"]
    region_rows = [d for d in designs if d["inside_sweep_v2_box"]]
    band = descriptors["hypothesis"]["agreement_band_relative"]
    estimand_sets = {"sobol_v3": sobol_rows, "sweep_v2": held_out_rows, "pooled_all": designs, "sweep_v2_region_pooled": region_rows}
    if set(dataset["estimands"]) != set(estimand_sets):
        raise ValueError("the sealed estimand sets differ from the four expected sets")
    for set_id, rows in estimand_sets.items():
        _compare_estimands(set_estimands(rows, band, x_star), dataset["estimands"][set_id], f"estimands {set_id}")
    sobol_est = dataset["estimands"]["sobol_v3"]
    held_out_est = dataset["estimands"]["sweep_v2"]
    region_est = dataset["estimands"]["sweep_v2_region_pooled"]
    pooled_est = dataset["estimands"]["pooled_all"]
    test = sobol_est["hypothesis_test"]
    recomputed_headline = {
        "design_count": len(designs),
        "held_out": {
            "applies": True,
            "axis_null_bijection_count": sum(1 for d in held_out_rows if d["held_out"]["observed_count"] == d["held_out"]["reference_count"]),
            "design_count": len(held_out_rows),
            "max_axis_null_difference_m": max(d["held_out"]["max_difference_m"] for d in held_out_rows),
            "observed_null_count": sum(d["held_out"]["observed_count"] for d in held_out_rows),
            "passed_count": sum(1 for d in held_out_rows if d["held_out"]["passed"]),
            "qoi_replay_passed_count": sum(1 for d in held_out_rows if d["held_out"]["qoi_replay_passed"]),
            "reference_null_count": sum(d["held_out"]["reference_count"] for d in held_out_rows),
            "stored_representatives_checked": sum(1 for d in held_out_rows if d["representative"]),
        },
        "max_wall_intersection_shift_m": max(d["stability"]["max_wall_intersection_shift_m"] for d in designs),
        "pooled_cusp_count": pooled_est["cusp_count"],
        "pooled_cusp_is_wall_maximum_count": pooled_est["cusp_is_wall_maximum_count"],
        "pooled_rho_conservative": pooled_est["rho_conservative"],
        "pooled_rho_wall": pooled_est["rho_wall"],
        "rho_resolution_sensitivity_max": pooled_est["rho_resolution_sensitivity_max"]["max"],
        "set_counts": {s: len(rows) for s, rows in (("sobol_v3", sobol_rows), ("sweep_v2", held_out_rows))},
        "sobol_five_stage_four_cusp_hemp_like_count": sobol_est["five_stage_four_cusp_hemp_like_count"],
        "sobol_hemp_like_count": sobol_est["hemp_like_count"],
        "sobol_hemp_like_fraction": sobol_est["hemp_like_fraction"],
        "sobol_hemp_like_region": sobol_est["hemp_like_region"],
        "sobol_hypothesis_test": {key: test.get(key) for key in ("cusp_count", "slope_through_origin", "r_squared", "fraction_within_band", "band", "confusion_predicted_i1_vs_realised", "prediction_accuracy", "smallest_x_w_realised_hemp_like", "largest_x_w_not_hemp_like", "x_star_prediction", "x_star_from_fitted_slope", "wall_radius_over_pitch_star_from_fitted_slope")},
        "sobol_predicted_hemp_like_i1_count": sobol_est["predicted_hemp_like_i1_count"],
        "sobol_rho_conservative": sobol_est["rho_conservative"],
        "sobol_wall_cusp_count_histogram": sobol_est["wall_cusp_count_histogram"],
        "stable_design_count": sum(1 for d in designs if d["stability"]["stable"]),
        "sweep_v2_region_hemp_like_count": region_est["hemp_like_count"],
        "sweep_v2_region_max_rho_conservative": region_est["rho_conservative"]["max"],
        "v2_gates_passed_count": pooled_est["v2_gates_passed_count"],
    }
    if recomputed_headline != headline or dataset["held_out"] != headline["held_out"]:
        raise ValueError("the sealed headline does not recompute from the per-design rows and estimands")
    if len(region_rows) != len(held_out_rows) + authorities["sobol_inside_sweep_v2_box_count"] or sum(1 for d in sobol_rows if d["inside_sweep_v2_box"]) != authorities["sobol_inside_sweep_v2_box_count"]:
        raise ValueError("the sweep-v2 region does not consist of the held-out set plus the Sobol designs inside the v2 box")
    if len(end_ratios) != 2 * len(sobol_rows) or len(end_ratios) + len(interior_ratios) != sobol_est["cusp_count"]:
        raise ValueError("end and interior cusp counts do not partition the Sobol cusps")

    # ---- x_w bands (Sobol designs) ----
    edges = (x_star, *BAND_EDGES)
    if max(d["x_w"] for d in sobol_rows) >= BAND_EDGES[-1] or BAND_EDGES[-1] <= coverage["x_w"][1]:
        raise ValueError("the band edges do not cover the realised and the box x_w range")
    bands: list[dict[str, Any]] = []
    for i, token in enumerate(BAND_TOKENS):
        lo = 0.0 if i == 0 else edges[i - 1]
        hi = edges[i]
        members = [d for d in sobol_rows if lo <= d["x_w"] < hi]
        rhos = [r["rho_conservative"] for d in members for r in d["rho"]]
        bands.append({
            "token": token, "low": lo, "high": hi, "designs": len(members),
            "hemp_like": sum(1 for d in members if d["hemp_like_all_cusps"]),
            "predicted": sum(1 for d in members if d["predicted_hemp_like_i1"]),
            "cusps": len(rhos), "rho": _distribution(rhos),
            "rho_over_i1": _distribution([r["rho_conservative"] / d["ppm_prediction"]["i1_x_w"] for d in members for r in d["rho"]]),
            "i1": _distribution([d["ppm_prediction"]["i1_x_w"] for d in members]),
        })
    if sum(b["designs"] for b in bands) != len(sobol_rows) or bands[0]["predicted"] != 0 or any(b["predicted"] != b["designs"] for b in bands[1:]):
        raise ValueError("the x_w bands do not partition the Sobol designs at the single-harmonic threshold")
    if sum(b["hemp_like"] for b in bands) != sobol_est["hemp_like_count"]:
        raise ValueError("the band HEMP-like counts do not sum to the sealed count")

    # ---- the PPM review's launch-position analysis (definition/hypothesis source) ----
    cells = [dict(cell, design_id=design_id) for design_id, block in sorted(check_output["reflections"].items()) for cell in block["cells"]]
    near = [c for c in cells if c["dist_to_centre_over_pitch"] <= NEAR_CENTRE_PITCH]
    far = [c for c in cells if c["dist_to_centre_over_pitch"] >= FAR_CENTRE_PITCH]
    if len(near) + len(far) != len(cells) or not near or not far:
        raise ValueError("the launch cells do not split into the review's near-centre and far-from-centre classes")
    if len({c["orbits"] for c in cells}) != 1:
        raise ValueError("the launch cells do not carry one common orbit count")
    if {d for d in check_output["reflections"]} != {d["design_id"] for d in held_out_rows if d["representative"]}:
        raise ValueError("the review's reflection analysis does not cover exactly the four stored sweep-v2 representatives")
    review_results = {r["design_id"]: r for r in check_output["results"]}
    alphas = [e["mendel_alpha"] for r in check_output["results"] for e in r["electrons"]]
    epsilons = [e["epsilon_wall_cusp"] for r in check_output["results"] for e in r["electrons"]]
    energies = sorted({e["energy_ev"] for r in check_output["results"] for e in r["electrons"]})
    if energies != wall_loss_protocol["launches"]["energies_ev"]:
        raise ValueError("the review's electron energies differ from the wall-loss campaign's launch energies")
    mu_by_design = {d: block["mu_median"] for d, block in check_output["reflections"].items()}
    eps_by_design = {d: next(e["epsilon_wall_cusp"] for e in review_results[d]["electrons"] if e["energy_ev"] == max(energies)) for d in mu_by_design}
    ordered_by_eps = sorted(mu_by_design, key=lambda d: eps_by_design[d])
    mu_ordered = all(mu_by_design[a] <= mu_by_design[b] for a, b in zip(ordered_by_eps, ordered_by_eps[1:]))
    # The wall-loss campaign launched 0.5 mm from the P2 magnet centres: read the launch
    # planes from its frozen protocol and the stage centres from the topology screening's
    # P2 record (the same finite-element field).
    launch_z = sorted({seed["position_m"][2] for seed in wall_loss_protocol["launches"]["position_seeds"]})
    p2_centres = p2_record["geometry"]["stage_centres_m"]
    p2_pitch = p2_record["geometry"]["stage_pitch_m"]
    if len(launch_z) != len(p2_centres) or len(launch_z) != len({seed["cell_id"] for seed in wall_loss_protocol["launches"]["position_seeds"]}):
        raise ValueError("the wall-loss launch planes do not pair one-to-one with the P2 magnet stages")
    offsets = [min(abs(z - c) for c in p2_centres) for z in launch_z]
    if max(offsets) - min(offsets) > 1e-12:
        raise ValueError("the wall-loss launch planes sit at different distances from their magnet centres")
    if p2_record["key"] != "p2_divergent_exit:divergent-exit-stack" or p2_record["status"] != "resolved":
        raise ValueError("the bound P2 record is not the resolved topology row")
    launch_offset_m = statistics.median(offsets)
    if not launch_offset_m / p2_pitch < NEAR_CENTRE_PITCH:
        raise ValueError("the wall-loss launch planes are not within the review's near-centre class")
    # The review traced the launch field lines from the magnet centres at the wall-loss
    # campaign's two launch radii (as fractions of the P2 wall radius) in every recorded field.
    field_lines = [line for r in check_output["results"] for line in r["field_lines"]]
    launch_radii = sorted({seed["position_m"][0] / p2_record["geometry"]["wall_radius_m"] for seed in wall_loss_protocol["launches"]["position_seeds"]})
    if sorted({line["launch_r_over_rw"] for line in field_lines}) != launch_radii:
        raise ValueError("the review's launch field lines are not at the wall-loss campaign's launch radii")
    if not all(line["reaches_wall"] for line in field_lines):
        raise ValueError("a launch field line of the review did not reach the wall before the cusp")

    # ================================================================== macros ====
    # ---- identity and lifecycle ----
    m.add("SwtClassification", "artifacts/campaign-result.json", "/classification", "ident", "screening classification string")
    m.add("SwtTopologyLabel", "artifacts/campaign-result.json", "/topology_label", "ident", "label carried by the wall-cusp catalogue")
    m.add("SwtTerminalState", "terminal.json", "/state", "ident", "runtime terminal state")
    m.add("SwtCampaignStatus", "artifacts/campaign-result.json", "/status", "ident", "recorded campaign status")
    m.add_derived("SwtRecordedOutcome", RECORDED_OUTCOME, "ident", "recorded outcome admitted by the numerical-screening gate", "constant of the generator; the gate admits the study at the field-only design-space screening outcome, which names campaign-result.json#/status", [{"artifact": "artifacts/campaign-result.json", "pointer": "/status"}])
    m.add_derived("SwtScreeningModel", SCREENING_MODEL, "text", "screening model label", "constant of the generator restating protocol.json#/classification_statement and #/claim_boundary/field_level", [{"artifact": "artifacts/protocol.json", "pointer": "/classification_statement"}, {"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}])
    m.add("SwtExperimentId", "artifacts/protocol.json", "/experiment_id", "ident", "experiment identifier")
    m.add_derived("SwtVersion", _version_token(protocol["experiment_id"]), "symbol", "version token of the campaign", "trailing version token of protocol.experiment_id", [{"artifact": "artifacts/protocol.json", "pointer": "/experiment_id"}])
    m.add_derived("SwtPriorVersion", _version_token("sweep_v2"), "symbol", "version token of the sweep whose designs form the held-out set", "trailing version token of the design-set key sweep_v2", [{"artifact": "artifacts/protocol.json", "pointer": "/design_sets/sweep_v2"}])
    m.add_derived("SwtTopologyVersion", _version_token(topology_protocol["experiment_id"]), "symbol", "version token of the cusp topology search whose definition is imported", "trailing version token of the bound topology protocol's experiment_id", [{"artifact": f"reference:{TOPOLOGY_PROTOCOL.as_posix()}", "pointer": "/experiment_id"}])
    m.add_derived("SwtWallLossVersion", _version_token(wall_loss_protocol["launches"]["campaign_id_prefix"]), "symbol", "version token of the wall-loss campaign whose launch design is read", "trailing version token of the bound wall-loss protocol's launches.campaign_id_prefix", [{"artifact": f"reference:{WALL_LOSS_PROTOCOL.as_posix()}", "pointer": "/launches/campaign_id_prefix"}])
    field_level = protocol["classification"].split("_")[0]
    if field_level != "L1a" or "L1a" not in protocol["claim_boundary"]["field_level"]:
        raise ValueError("field model level differs from the screening classification")
    m.add_derived("SwtFieldModelLevel", field_level, "symbol", "field model level named by the screening classification", "protocol.classification.split('_')[0]", [{"artifact": "artifacts/protocol.json", "pointer": "/classification"}])
    if "L1b" not in protocol["claim_boundary"]["l1b_p2_confirmation"]["statement"]:
        raise ValueError("the confirmation queue does not name the material-aware screening level")
    m.add_derived("SwtMaterialLevel", "L1b", "symbol", "material-aware screening field level named by the confirmation queue", "fixed token of protocol.claim_boundary.l1b_p2_confirmation.statement", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/l1b_p2_confirmation/statement"}])
    for name, symbol, description in (("SwtIOne", "I_1", "modified Bessel function of order one (the single-harmonic wall-cusp factor)"), ("SwtIZero", "I_0", "modified Bessel function of order zero"), ("SwtBThreeOverBOne", "b_3/b_1", "wall third-harmonic content"), ("SwtRSquaredSymbol", "R^2", "coefficient of determination"), ("SwtHOne", "H1", "first preregistered hypothesis"), ("SwtHTwo", "H2", "second preregistered hypothesis"), ("SwtXStarSymbol", "x^*", "the single-harmonic threshold symbol")):
        m.add_derived(name, symbol, "symbol", description, "typographic symbol whitelisted by the generator so the macro-only section types no digit", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/ppm_prediction/statement"}])
    m.add("SwtAttemptCount", "terminal.json", "/counts/attempt_count", "int", "execution attempts")
    m.add("SwtLockImmutable", "execution-lock.json", "/immutable", "bool", "execution lock immutability flag")
    m.add("SwtPreregCommit", "execution-lock.json", "/commit", "sha_short", "preregistration commit recorded in the execution lock")
    m.add_derived("SwtResultsCommit", RESULTS_COMMIT_SHA, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("SwtDashboardCommit", DASHBOARD_COMMIT_SHA, "sha_short", "dashboard commit prefix", "git commit of the results dashboard whose embedded extraction equals the bundle", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("SwtManifestSha", bundle.manifest_sha256, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "manifest.json", "pointer": ""}])
    m.add_derived("SwtVerifiedFiles", len(bundle.hashes), "int", "bundle files verified byte-for-byte", "count of manifest file entries whose sha256 and size equal the checkout", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add_derived("SwtArtifactCount", bundle.manifest["artifact_count"], "int", "manifest entries (files and directories)", "manifest.artifact_count", [{"artifact": "manifest.json", "pointer": "/artifact_count"}])
    m.add_derived("SwtToleratedEolFiles", 0, "int", "manifest entries accepted through any end-of-line tolerance", "count of manifest file entries whose recorded byte_sha256 differs from sha256(bytes); the generator grants no tolerance and fails on any difference", [{"artifact": "manifest.json", "pointer": "/artifacts"}])
    m.add("SwtProtocolSha", "artifacts/authorities.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("SwtExperimentCodeSha", "artifacts/authorities.json", "/experiment_code_sha256", "sha_short", "experiment code hash prefix")
    m.add("SwtDependencySourceSha", "artifacts/authorities.json", "/dependency_source_sha256", "sha_short", "dependency source hash prefix (includes the imported cusp-topology-v3.1 definition)")
    m.add("SwtFieldPipelineSha", "artifacts/authorities.json", "/field_pipeline_source_sha256", "sha_short", "field pipeline source hash prefix")
    m.add_derived("SwtExperimentCodeFiles", len(source_binding["experiment_code_files"]), "int", "experiment code files hashed", "len(source-binding.experiment_code_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/experiment_code_files"}])
    m.add_derived("SwtDependencySourceFiles", len(source_binding["dependency_source_files"]), "int", "dependency source files hashed", "len(source-binding.dependency_source_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/dependency_source_files"}])
    m.add_derived("SwtFieldPipelineFiles", len(source_binding["field_pipeline_source_files"]), "int", "field pipeline source files hashed", "len(source-binding.field_pipeline_source_files)", [{"artifact": "artifacts/source-binding.json", "pointer": "/field_pipeline_source_files"}])
    m.add("SwtCpuCount", "artifacts/runtime.json", "/cpu_count", "int", "logical CPUs of the host")
    m.add("SwtWorkerPool", "artifacts/campaign-result.json", "/execution_mode/worker_pool_size", "int", "worker pool size")
    m.add("SwtDevice", "execution-lock.json", "/device", "ident", "device string recorded in the execution lock")
    m.add("SwtStageWallMin", "artifacts/campaign-result.json", "/execution_mode/stage_wall_s", "min1", "wall time of the design stage (min)")
    m.add("SwtAssessmentWallMin", "artifacts/campaign-result.json", "/execution_mode/assessment_wall_s", "min1", "wall time of the assessment (min)")
    m.add_derived("SwtTotalWallMin", campaign["execution_mode"]["stage_wall_s"] + campaign["execution_mode"]["assessment_wall_s"], "min1", "design stage plus assessment wall time (min)", "execution_mode.stage_wall_s + execution_mode.assessment_wall_s", [{"artifact": "artifacts/campaign-result.json", "pointer": "/execution_mode"}])
    m.add("SwtShakedownPassed", "artifacts/shakedown.json", "/passed", "bool", "shakedown passed")
    m.add("SwtShakedownEvidentiary", "artifacts/shakedown.json", "/evidentiary", "bool", "shakedown evidentiary flag")
    m.add("SwtShakedownDesigns", "artifacts/shakedown.json", "/design_count", "int", "shakedown designs")
    m.add("SwtShakedownCommit", "artifacts/authorities.json", "/shakedown_git_head", "sha_short", "commit at which the shakedown was run (the literature-review commit)")
    m.add("SwtTimingWithinBudget", "artifacts/shakedown.json", "/timing_projection/within_budget", "bool", "evidentiary run projected within the wall-time budget")
    m.add("SwtTimingBudgetMin", "artifacts/shakedown.json", "/timing_projection/budget_wall_seconds", "min1", "wall-time budget (min)")
    m.add("SwtTimingProjectedMin", "artifacts/shakedown.json", "/timing_projection/projected_wall_seconds_at_pool", "min1", "projected wall time at the pool size (min)")
    m.add_derived("SwtBindingGateCount", len(BINDING_GATE_NAMES), "int", "binding integrity gates", "len(gates.campaign)", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("SwtBindingGatesTrue", sum(1 for name in BINDING_GATE_NAMES if gates["campaign"][name] is True), "int", "binding gates recorded true", "count(gates.campaign[*] == true)", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("SwtBindingGateNames", list(BINDING_GATE_NAMES), "list_ident_tt", "names of the binding gates", "sorted keys of gates.campaign", [{"artifact": "artifacts/gates.json", "pointer": "/campaign"}])
    m.add_derived("SwtReportedNotBindingCount", len(gates["definitions"]["reported_not_binding"]), "int", "quantities reported but not gated", "len(gates.definitions.reported_not_binding)", [{"artifact": "artifacts/gates.json", "pointer": "/definitions/reported_not_binding"}])
    m.add("SwtReportedNotBinding", "artifacts/gates.json", "/definitions/reported_not_binding", "list_clauses", "quantities the protocol reports without gating them")
    m.add_derived("SwtReplayDesigns", len(gates["replays"]), "int", "determinism replay designs", "len(gates.replays)", [{"artifact": "artifacts/gates.json", "pointer": "/replays"}])
    m.add_derived("SwtReplaysBitIdentical", replays_bit_identical, "int", "replays whose canonical payload bytes were bit-identical", "count(gates.replays[*].bit_identical == true)", [{"artifact": "artifacts/gates.json", "pointer": "/replays"}])
    m.add_derived("SwtFailedDesigns", len(failures["failed"]), "int", "designs that failed to resolve", "len(design-failures.failed)", [{"artifact": "artifacts/design-failures.json", "pointer": "/failed"}])
    m.add_derived("SwtRepresentativeCount", representative_count, "int", "representative designs (stored paths and sweep-v2 stored maps)", "count(designs[*].representative == true)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtVTwoGateCount", len(gates["sweep_v2_gate_breakdown"]), "int", "sweep-v2 metric gates applied verbatim", "len(gates.sweep_v2_gate_breakdown)", [{"artifact": "artifacts/gates.json", "pointer": "/sweep_v2_gate_breakdown"}])
    if set(gates["sweep_v2_gate_breakdown"]) != set(V2_GATE_NAMES) or any(not g["passed"] or g["failed_designs"] for g in gates["sweep_v2_gate_breakdown"].values()):
        raise ValueError("the sweep-v2 gate breakdown names a different gate set or records a failure")
    m.add("SwtVTwoGateBoundaryLimit", "artifacts/gates.json", "/sweep_v2_gate_breakdown/boundary/limit", "g", "sweep-v2 boundary-to-peak gate limit")
    m.add("SwtVTwoGateBoundaryObserved", "artifacts/gates.json", "/sweep_v2_gate_breakdown/boundary/observed_extreme", "fixed4", "largest boundary-to-peak ratio observed")
    m.add("SwtVTwoGateConfidenceLimit", "artifacts/gates.json", "/sweep_v2_gate_breakdown/topology_confidence/limit", "g", "sweep-v2 topology-confidence gate limit")
    m.add("SwtVTwoGateConfidenceObserved", "artifacts/gates.json", "/sweep_v2_gate_breakdown/topology_confidence/observed_extreme", "fixed3", "smallest topology confidence observed")
    m.add("SwtVTwoGateResidualLimit", "artifacts/gates.json", "/sweep_v2_gate_breakdown/residual/limit", "sci1", "sweep-v2 relative-residual gate limit")
    m.add("SwtVTwoGateResidualObserved", "artifacts/gates.json", "/sweep_v2_gate_breakdown/residual/observed_extreme", "sci2", "largest relative residual observed")
    m.add("SwtVTwoGateManufacturabilityObservedUm", "artifacts/gates.json", "/sweep_v2_gate_breakdown/manufacturability/observed_extreme", "um0", "smallest manufacturing margin observed (um)")
    m.add("SwtVTwoGateNotApplicable", "artifacts/gates.json", "/sweep_v2_gate_not_applicable/cpu_cuda_parity", "text", "why the seventh sweep-v2 gate does not apply")

    # ---- design space ----
    m.add("SwtDesignCount", "artifacts/campaign-result.json", "/design_count", "int", "designs screened (Sobol plus held-out)")
    m.add_derived("SwtDeclaredDesigns", len(plan["design_keys"]), "int", "designs declared by the frozen plan", "len(campaign-plan.design_keys)", [{"artifact": "artifacts/campaign-plan.json", "pointer": "/design_keys"}])
    m.add("SwtSobolDesigns", "artifacts/campaign-result.json", "/set_counts/sobol_v3", "int", "scrambled-Sobol designs of the v3 box")
    m.add("SwtHeldOutDesigns", "artifacts/campaign-result.json", "/set_counts/sweep_v2", "int", "accepted sweep-v2 designs re-solved as the held-out set")
    m.add_derived("SwtSetCount", len(SET_IDS), "int", "design sets", "len(campaign.set_counts)", [{"artifact": "artifacts/campaign-result.json", "pointer": "/set_counts"}])
    algorithm_clause = sampling["algorithm"].split(";")[0]
    if "scrambled Sobol" not in algorithm_clause or "Joe-Kuo" not in algorithm_clause:
        raise ValueError("the sampling algorithm statement does not name scrambled Sobol with Joe-Kuo direction numbers")
    m.add_derived("SwtSamplingAlgorithm", algorithm_clause, "text", "sampling algorithm statement of the frozen protocol (first clause)", "protocol.sampling.algorithm up to the first semicolon", [{"artifact": "artifacts/protocol.json", "pointer": "/sampling/algorithm"}])
    m.add("SwtSeed", "artifacts/protocol.json", "/sampling/seed", "int", "Sobol scramble seed")
    m.add_derived("SwtVariableCount", len(sampling["variables"]), "int", "design variables", "len(protocol.sampling.variables)", [{"artifact": "artifacts/protocol.json", "pointer": "/sampling/variables"}])
    m.add_derived("SwtWidenedVariableCount", len(WIDENED_VARIABLES), "int", "design variables whose v3 interval strictly contains the v2 interval", "count of protocol.sampling.sweep_v2_box entries that are intervals", [{"artifact": "artifacts/protocol.json", "pointer": "/sampling/sweep_v2_box"}])
    m.add("SwtStageMapping", "artifacts/protocol.json", "/geometry/stage_count_mapping", "text", "stage-count mapping of the selector")
    m.add("SwtBoxStatement", "artifacts/protocol.json", "/sampling/sweep_v2_box/statement", "text", "protocol statement on the v2 box as a subset of the v3 box")
    m.add("SwtBoundsProvenance", "artifacts/protocol.json", "/sampling/bounds_provenance", "text", "how the widened bounds were fixed")
    m.add("SwtPolarity", "artifacts/protocol.json", "/geometry/polarity", "text", "polarity policy of the stack")
    for name, token, _label in WIDENED_VARIABLES:
        index = next(i for i, v in enumerate(sampling["variables"]) if v["name"] == name)
        m.add(f"SwtBox{token}LoMm", "artifacts/protocol.json", f"/sampling/variables/{index}/lower", "mm2", f"v3 lower bound of {name} (mm)")
        m.add(f"SwtBox{token}HiMm", "artifacts/protocol.json", f"/sampling/variables/{index}/upper", "mm2", f"v3 upper bound of {name} (mm)")
        m.add(f"SwtVTwoBox{token}LoMm", "artifacts/protocol.json", f"/sampling/sweep_v2_box/{name}/0", "mm2", f"v2 lower bound of {name} (mm)")
        m.add(f"SwtVTwoBox{token}HiMm", "artifacts/protocol.json", f"/sampling/sweep_v2_box/{name}/1", "mm2", f"v2 upper bound of {name} (mm)")
    box_inputs = [{"artifact": "artifacts/protocol.json", "pointer": "/sampling/variables"}, {"artifact": "artifacts/protocol.json", "pointer": "/sampling/sweep_v2_box"}]
    m.add("SwtBoxRwOverLLo", "artifacts/protocol.json", "/sampling/regime_coverage/wall_radius_over_pitch/0", "fixed3", "smallest r_w / L of the v3 box")
    m.add("SwtBoxRwOverLHi", "artifacts/protocol.json", "/sampling/regime_coverage/wall_radius_over_pitch/1", "fixed3", "largest r_w / L of the v3 box")
    m.add("SwtBoxXwLo", "artifacts/protocol.json", "/sampling/regime_coverage/x_w/0", "fixed2", "smallest x_w of the v3 box")
    m.add("SwtBoxXwHi", "artifacts/protocol.json", "/sampling/regime_coverage/x_w/1", "fixed2", "largest x_w of the v3 box")
    m.add_derived("SwtVTwoBoxRwOverLLo", box_v2["chamber_outer_radius_m"][0] / box_v2["stage_pitch_m"][1], "fixed3", "smallest r_w / L of the v2 box", "sweep_v2_box.chamber_outer_radius_m[0] / sweep_v2_box.stage_pitch_m[1]", box_inputs)
    m.add_derived("SwtVTwoBoxRwOverLHi", box_v2["chamber_outer_radius_m"][1] / box_v2["stage_pitch_m"][0], "fixed3", "largest r_w / L of the v2 box", "sweep_v2_box.chamber_outer_radius_m[1] / sweep_v2_box.stage_pitch_m[0]", box_inputs)
    m.add_derived("SwtVTwoBoxXwLo", math.pi * box_v2["chamber_outer_radius_m"][0] / box_v2["stage_pitch_m"][1], "fixed2", "smallest x_w of the v2 box", "pi * sweep_v2_box.chamber_outer_radius_m[0] / sweep_v2_box.stage_pitch_m[1]", box_inputs)
    m.add_derived("SwtVTwoBoxXwHi", math.pi * box_v2["chamber_outer_radius_m"][1] / box_v2["stage_pitch_m"][0], "fixed2", "largest x_w of the v2 box", "pi * sweep_v2_box.chamber_outer_radius_m[1] / sweep_v2_box.stage_pitch_m[0]", box_inputs)
    m.add("SwtHeldOutXwMin", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/x_w/min", "fixed2", "smallest realised x_w of the held-out sweep-v2 designs")
    m.add("SwtHeldOutXwMax", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/x_w/max", "fixed2", "largest realised x_w of the held-out sweep-v2 designs")
    m.add("SwtHeldOutRwOverLMin", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/wall_radius_over_pitch/min", "fixed3", "smallest realised r_w / L of the held-out sweep-v2 designs")
    m.add("SwtHeldOutRwOverLMax", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/wall_radius_over_pitch/max", "fixed3", "largest realised r_w / L of the held-out sweep-v2 designs")
    m.add("SwtSobolXwMin", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/x_w/min", "fixed2", "smallest realised x_w of the Sobol designs")
    m.add("SwtSobolXwMax", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/x_w/max", "fixed2", "largest realised x_w of the Sobol designs")
    m.add("SwtSobolXwMedian", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/x_w/median", "fixed2", "median realised x_w of the Sobol designs")
    m.add("SwtSobolRwOverLMin", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/wall_radius_over_pitch/min", "fixed3", "smallest realised r_w / L of the Sobol designs")
    m.add("SwtSobolRwOverLMax", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/wall_radius_over_pitch/max", "fixed3", "largest realised r_w / L of the Sobol designs")
    m.add("SwtRegionXwMax", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/x_w/max", "fixed2", "largest realised x_w in the sweep-v2 region")
    m.add("SwtRegionRwOverLMax", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/wall_radius_over_pitch/max", "fixed3", "largest realised r_w / L in the sweep-v2 region")
    m.add("SwtSobolInsideVTwoBox", "artifacts/authorities.json", "/sobol_inside_sweep_v2_box_count", "int", "Sobol designs that fall inside the v2 box")
    m.add("SwtRegionDesigns", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/design_count", "int", "designs of the sweep-v2 region (held-out plus Sobol inside the v2 box)")
    for stages, token in STAGE_TOKENS.items():
        m.add(f"SwtSobolStage{token}Designs", "artifacts/sweep-dataset.json", f"/estimands/sobol_v3/by_stage_count/{stages}/designs", "int", f"Sobol designs with {stages} stages")
        m.add(f"SwtSobolStage{token}HempLike", "artifacts/sweep-dataset.json", f"/estimands/sobol_v3/by_stage_count/{stages}/hemp_like", "int", f"HEMP-like Sobol designs with {stages} stages")
        m.add(f"SwtSobolStage{token}NMinusOne", "artifacts/sweep-dataset.json", f"/estimands/sobol_v3/by_stage_count/{stages}/n_minus_1_cusps", "int", f"{stages}-stage Sobol designs with exactly {stages - 1} wall cusps")
    m.add("SwtGridRadial", "artifacts/protocol.json", "/field/domain/radial_intervals", "int", "radial intervals of the accepted map")
    m.add("SwtGridAxial", "artifacts/protocol.json", "/field/domain/axial_intervals", "int", "axial intervals of the accepted map")
    m.add("SwtRefinement", "artifacts/protocol.json", "/field/refinement", "int", "refinement factor of the stability map")
    m.add("SwtSolverRelTol", "artifacts/protocol.json", "/field/solver/relative_tolerance", "sci1", "solver relative tolerance")
    m.add("SwtDomainRadiusMm", "artifacts/protocol.json", "/field/domain/radius_m", "mm1", "domain radius (mm)")
    m.add("SwtStabilityToleranceUm", "artifacts/protocol.json", "/definition_v3_import/stability_tolerance_m", "um0", "refinement-stability tolerance (um)")
    m.add("SwtHeldOutToleranceUm", "artifacts/protocol.json", "/definition_v3_import/held_out_tolerance_m", "um0", "held-out correspondence tolerance (um)")
    m.add("SwtDefinitionSource", "artifacts/protocol.json", "/definition_v3_import/source", "text", "definition import statement of the frozen protocol")
    m.add_derived("SwtTopologyResultsCommit", TOPOLOGY_RESULTS_COMMIT_SHA, "sha_short", "record commit of the cusp topology search v3.1 whose protocol fixed the definition", "git commit at which the bound topology protocol and P2 record are read", [{"artifact": "artifacts/protocol.json", "pointer": "/definition_v3_import/source"}])
    m.add("SwtSweepPreregCommit", "artifacts/authorities.json", "/sealed_sources/sweep_v2/preregistration_commit", "sha_short", "preregistration commit of the sealed sweep-v2 source")
    m.add("SwtSweepManifestSha", "artifacts/authorities.json", "/sealed_sources/sweep_v2/manifest_file_sha256", "sha_short", "sealed sweep-v2 manifest hash prefix")
    m.add("SwtLengthQuantumPolicy", "artifacts/protocol.json", "/geometry/length_binary64_policy", "text", "binary64 length-quantisation policy")

    # ---- the Koch ratio and the single-harmonic prediction ----
    m.add("SwtKochCitation", "artifacts/protocol.json", "/descriptors_v3/koch_rho/citation", "text", "citation of the HEMP design ratio")
    m.add("SwtRhoConservativeDefinition", "artifacts/protocol.json", "/descriptors_v3/koch_rho/rho_conservative", "text", "definition of the binding HEMP-like classifier reading")
    m.add("SwtRhoWallDefinition", "artifacts/protocol.json", "/descriptors_v3/koch_rho/rho_wall", "text", "definition of the wall reading")
    m.add("SwtHempLikeRule", "artifacts/protocol.json", "/descriptors_v3/hemp_like_rule", "text", "HEMP-like classification rule")
    m.add("SwtRhoThreshold", designs[0]["record_path"], "/descriptors/accepted/hemp_like_threshold/rho", "fixed1", "HEMP-like threshold on rho (the rounded DM9-1 anode value of protocol.descriptors_v3.hemp_like_rule; verified equal in every design record)")
    m.add("SwtPpmStatement", "artifacts/protocol.json", "/descriptors_v3/ppm_prediction/statement", "text", "single-harmonic PPM prediction statement")
    m.add("SwtPpmWhyNotRealised", "artifacts/protocol.json", "/descriptors_v3/ppm_prediction/why_i1_not_the_realised_ratio", "text", "why I_1 is not the realised ratio (frozen protocol)")
    m.add("SwtXStar", "artifacts/sweep-dataset.json", "/headline/sobol_hypothesis_test/x_star_prediction", "fixed6", "single-harmonic threshold x* with I_1(x*) = 1.5")
    m.add_derived("SwtRwOverLStar", x_star / math.pi, "fixed6", "r_w / L at the single-harmonic threshold", "x* / pi", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sobol_hypothesis_test/x_star_prediction"}])
    m.add_derived("SwtXStarTwo", round(x_star, 2), "fixed2", "single-harmonic threshold x* (two decimals)", "round(x*, 2)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sobol_hypothesis_test/x_star_prediction"}])
    m.add_derived("SwtRwOverLStarThree", round(x_star / math.pi, 3), "fixed3", "r_w / L at the threshold (three decimals)", "round(x* / pi, 3)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sobol_hypothesis_test/x_star_prediction"}])
    m.add("SwtHypothesisStatement", "artifacts/protocol.json", "/descriptors_v3/hypothesis/statement", "text", "preregistered hypothesis statement")
    m.add("SwtHypothesisFalsifier", "artifacts/protocol.json", "/descriptors_v3/hypothesis/what_would_falsify", "text", "preregistered falsification conditions")
    m.add("SwtBand", "artifacts/protocol.json", "/descriptors_v3/hypothesis/agreement_band_relative", "pct0", "relative agreement band of H1")
    m.add_derived("SwtHOneSlopeLo", 0.80, "fixed2", "preregistered lower slope bound of H1", "fixed pattern '[0.80, 1.00]' of protocol.descriptors_v3.hypothesis.statement", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/hypothesis/statement"}])
    m.add_derived("SwtHOneSlopeHi", 1.00, "fixed2", "preregistered upper slope bound of H1", "fixed pattern '[0.80, 1.00]' of protocol.descriptors_v3.hypothesis.statement", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/hypothesis/statement"}])
    m.add_derived("SwtHOneBandFractionRequired", 0.80, "pct0", "preregistered band fraction of H1", "fixed pattern 'at least 80 %' of protocol.descriptors_v3.hypothesis.statement", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/hypothesis/statement"}])
    m.add_derived("SwtHTwoAccuracyRequired", 0.85, "fixed2", "preregistered prediction accuracy of H2", "fixed pattern 'accuracy >= 0.85' of protocol.descriptors_v3.hypothesis.statement", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/hypothesis/statement"}])
    statement = descriptors["hypothesis"]["statement"]
    if "[0.80, 1.00]" not in statement or "at least 80 %" not in statement or "accuracy >= 0.85" not in statement:
        raise ValueError("the preregistered hypothesis thresholds are not in the frozen statement")

    # ---- results: Sobol designs and the held-out set ----
    m.add("SwtSobolCuspCount", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/cusp_count", "int", "wall cusps of the Sobol designs")
    m.add("SwtSobolRhoMin", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/rho_conservative/min", "fixed2", "smallest Koch ratio (conservative reading) over the Sobol cusps")
    m.add("SwtSobolRhoMedian", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/rho_conservative/median", "fixed2", "median Koch ratio over the Sobol cusps")
    m.add("SwtSobolRhoMax", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/rho_conservative/max", "fixed1", "largest Koch ratio over the Sobol cusps")
    m.add("SwtSobolRhoDownstreamMax", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/rho_downstream/max", "fixed1", "largest downstream reading over the Sobol cusps")
    m.add("SwtHeldOutCuspCount", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/cusp_count", "int", "wall cusps of the held-out sweep-v2 designs")
    m.add("SwtHeldOutRhoMin", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/rho_conservative/min", "fixed2", "smallest Koch ratio over the held-out cusps")
    m.add("SwtHeldOutRhoMax", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/rho_conservative/max", "fixed2", "largest Koch ratio over the held-out cusps")
    m.add("SwtHeldOutRhoMedian", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/rho_conservative/median", "fixed2", "median Koch ratio over the held-out cusps")
    m.add("SwtHeldOutHempLike", "artifacts/sweep-dataset.json", "/estimands/sweep_v2/hemp_like_count", "int", "HEMP-like designs among the held-out set")
    for count, token in COUNT_TOKENS.items():
        m.add(f"SwtSobolHist{token}", "artifacts/sweep-dataset.json", f"/estimands/sobol_v3/wall_cusp_count_histogram/{count}", "int", f"Sobol designs with {count} wall cusps")
    m.add("SwtSobolHistogramText", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/wall_cusp_count_histogram", "histogram", "wall-cusp count histogram of the Sobol designs")
    m.add("SwtSobolNMinusOneFraction", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/n_minus_1_cusp_fraction", "pct0", "Sobol designs with N-1 wall cusps for N stages")
    m.add_derived("SwtSobolNMinusOne", sum(1 for d in sobol_rows if d["wall_cusp_count"] == d["derived"]["stage_count"] - 1), "int", "Sobol designs with exactly N-1 wall cusps", "count over sobol_v3 rows of wall_cusp_count == stage_count - 1", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add("SwtSobolFourWallCusps", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/four_wall_cusp_count", "int", "Sobol designs with exactly four wall cusps")
    m.add("SwtSobolAngleMedianDeg", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/angle_to_wall_normal_deg/median", "deg1", "median separatrix angle to the wall normal over the Sobol cusps (deg)")
    m.add("SwtSobolAngleMaxDeg", "artifacts/sweep-dataset.json", "/estimands/sobol_v3/angle_to_wall_normal_deg/max", "deg1", "largest separatrix angle over the Sobol cusps (deg)")
    m.add("SwtHempLikeCount", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_count", "int", "Sobol designs that are HEMP-like (rho >= 1.5 at every wall cusp)")
    m.add("SwtHempLikeFraction", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_fraction", "pct1", "HEMP-like fraction of the Sobol designs")
    m.add("SwtPredictedHempLike", "artifacts/sweep-dataset.json", "/headline/sobol_predicted_hemp_like_i1_count", "int", "Sobol designs predicted HEMP-like by I_1(x_w) >= 1.5")
    m.add("SwtFiveStageFourCuspHempLike", "artifacts/sweep-dataset.json", "/headline/sobol_five_stage_four_cusp_hemp_like_count", "int", "five-stage four-cusp HEMP-like Sobol designs")
    five_four = [d for d in sobol_rows if d["five_stage_four_cusp_hemp_like"]]
    m.add_derived("SwtFiveStageFourCuspIds", [_short_id(d["design_id"]) for d in five_four], "list_ident_tt", "ordinals of the five-stage four-cusp HEMP-like designs", "ordinal token of design_id over sobol_v3 rows with five_stage_four_cusp_hemp_like == true", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add("SwtHempRegionXwMin", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/x_w/min", "fixed2", "smallest x_w of a HEMP-like design")
    m.add("SwtHempRegionXwMax", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/x_w/max", "fixed2", "largest x_w of a HEMP-like design")
    m.add("SwtHempRegionXwMedian", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/x_w/median", "fixed2", "median x_w of the HEMP-like designs")
    m.add("SwtHempRegionRwOverLMin", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/wall_radius_over_pitch/min", "fixed3", "smallest r_w / L of a HEMP-like design")
    m.add("SwtHempRegionRwOverLMax", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/wall_radius_over_pitch/max", "fixed3", "largest r_w / L of a HEMP-like design")
    m.add("SwtHempRegionXmMin", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/x_m_inner/min", "fixed2", "smallest x_m (magnet inner radius) of a HEMP-like design")
    m.add("SwtHempRegionXmMax", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/x_m_inner/max", "fixed2", "largest x_m of a HEMP-like design")
    m.add("SwtHempRegionMinRhoMin", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/min_rho_conservative/min", "fixed2", "smallest per-design minimum rho among the HEMP-like designs")
    m.add("SwtHempRegionMinRhoMedian", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/min_rho_conservative/median", "fixed2", "median per-design minimum rho among the HEMP-like designs")
    m.add("SwtHempRegionMinRhoMax", "artifacts/sweep-dataset.json", "/headline/sobol_hemp_like_region/min_rho_conservative/max", "fixed2", "largest per-design minimum rho among the HEMP-like designs")
    for stages, token in STAGE_TOKENS.items():
        m.add_derived(f"SwtHempRegionStage{token}", headline["sobol_hemp_like_region"]["stage_counts"].get(str(stages), 0), "int", f"HEMP-like designs with {stages} stages", f"headline.sobol_hemp_like_region.stage_counts['{stages}'] (0 when absent)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sobol_hemp_like_region/stage_counts"}])
    for count, token in COUNT_TOKENS.items():
        m.add_derived(f"SwtHempRegionCusps{token}", headline["sobol_hemp_like_region"]["wall_cusp_counts"].get(str(count), 0), "int", f"HEMP-like designs with {count} wall cusps", f"headline.sobol_hemp_like_region.wall_cusp_counts['{count}'] (0 when absent)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sobol_hemp_like_region/wall_cusp_counts"}])
    m.add("SwtRegionHempLike", "artifacts/sweep-dataset.json", "/headline/sweep_v2_region_hemp_like_count", "int", "HEMP-like designs in the sweep-v2 region")
    m.add("SwtRegionMaxRho", "artifacts/sweep-dataset.json", "/headline/sweep_v2_region_max_rho_conservative", "fixed3", "largest Koch ratio over every cusp of the sweep-v2 region")
    m.add("SwtRegionCuspCount", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/cusp_count", "int", "wall cusps of the sweep-v2 region")
    m.add("SwtRegionRhoMedian", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/rho_conservative/median", "fixed2", "median Koch ratio over the sweep-v2 region cusps")
    m.add("SwtRegionRhoOverIOneMedian", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/hypothesis_test/rho_over_i1/median", "fixed2", "median rho / I_1 over the sweep-v2 region cusps")
    m.add("SwtRegionBandFraction", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/hypothesis_test/fraction_within_band", "pct0", "fraction of sweep-v2 region cusps within the agreement band")
    m.add("SwtRegionRSquared", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/hypothesis_test/r_squared", "fixed2", "R^2 of rho on I_1 over the sweep-v2 region cusps")
    m.add("SwtRegionIOneMax", "artifacts/sweep-dataset.json", "/estimands/sweep_v2_region_pooled/hypothesis_test/largest_x_w_not_hemp_like", "fixed2", "largest x_w of the sweep-v2 region (every design there is not HEMP-like)")
    m.add_derived("SwtRegionIOneOfMaxXw", bessel_i(1, region_est["x_w"]["max"]), "fixed2", "I_1 at the largest x_w of the sweep-v2 region", "I_1(estimands.sweep_v2_region_pooled.x_w.max) by the Bessel series", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/estimands/sweep_v2_region_pooled/x_w/max"}])
    m.add("SwtPooledCuspCount", "artifacts/sweep-dataset.json", "/headline/pooled_cusp_count", "int", "wall cusps over every design")
    m.add("SwtCuspIsWallMaximumCount", "artifacts/sweep-dataset.json", "/headline/pooled_cusp_is_wall_maximum_count", "int", "cusps that are the wall |B| maximum of their neighbourhood")
    m.add("SwtPooledRhoWallMax", "artifacts/sweep-dataset.json", "/headline/pooled_rho_wall/max", "fixed3", "largest wall reading over every cusp")
    m.add("SwtPooledRhoWallMedian", "artifacts/sweep-dataset.json", "/headline/pooled_rho_wall/median", "fixed2", "median wall reading over every cusp")
    m.add("SwtPooledRhoWallMin", "artifacts/sweep-dataset.json", "/headline/pooled_rho_wall/min", "fixed2", "smallest wall reading over every cusp")
    m.add("SwtWallBThreeMedian", "artifacts/sweep-dataset.json", "/estimands/pooled_all/wall_b3_over_b1/median", "fixed3", "median wall third-harmonic content |b_3 / b_1|")
    m.add("SwtWallBThreeMax", "artifacts/sweep-dataset.json", "/estimands/pooled_all/wall_b3_over_b1/max", "fixed2", "largest wall third-harmonic content")
    m.add("SwtWallBFiveMedian", "artifacts/sweep-dataset.json", "/estimands/pooled_all/wall_b5_over_b1/median", "fixed3", "median wall fifth-harmonic content")
    m.add("SwtStableDesigns", "terminal.json", "/payload/stable_design_count", "int", "designs stable under refinement")
    m.add("SwtMaxWallShiftUm", "artifacts/sweep-dataset.json", "/headline/max_wall_intersection_shift_m", "um1", "largest wall-intersection shift under refinement (um)")
    m.add_derived("SwtMaxAxisShiftUm", axis_shift_max, "um1", "largest axis-null shift under refinement (um)", "max over designs of stability.max_axis_null_shift_m", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtHempFlagStableDesigns", sum(1 for d in designs if d["resolution_sensitivity"]["hemp_like_flag_agrees"]), "int", "designs whose HEMP-like classification agrees between the accepted and the refined map", "count(designs[*].resolution_sensitivity.hemp_like_flag_agrees == true)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add("SwtRhoSensitivityMedian", "artifacts/sweep-dataset.json", "/estimands/pooled_all/rho_resolution_sensitivity_max/median", "pct1", "median per-design resolution sensitivity of rho")
    m.add("SwtRhoSensitivityMax", "artifacts/sweep-dataset.json", "/headline/rho_resolution_sensitivity_max", "pct1", "largest per-design resolution sensitivity of rho")
    m.add("SwtHeldOutPassed", "artifacts/sweep-dataset.json", "/held_out/passed_count", "int", "held-out designs reproduced")
    m.add("SwtHeldOutCount", "artifacts/sweep-dataset.json", "/held_out/design_count", "int", "held-out designs")
    m.add("SwtHeldOutQoiReplay", "artifacts/sweep-dataset.json", "/held_out/qoi_replay_passed_count", "int", "held-out designs whose sweep-v2 QoIs replayed within tolerance")
    m.add("SwtHeldOutBijections", "artifacts/sweep-dataset.json", "/held_out/axis_null_bijection_count", "int", "held-out designs whose sealed axis nulls are in bijection with the v3 nulls")
    m.add("SwtHeldOutNulls", "artifacts/sweep-dataset.json", "/held_out/observed_null_count", "int", "axis nulls matched against the sealed sweep-v2 nulls")
    m.add("SwtHeldOutRefNulls", "artifacts/sweep-dataset.json", "/held_out/reference_null_count", "int", "sealed sweep-v2 axis nulls in the window")
    m.add("SwtHeldOutMaxUm", "artifacts/sweep-dataset.json", "/held_out/max_axis_null_difference_m", "um1", "largest held-out axis-null difference (um)")
    m.add("SwtHeldOutStoredReps", "artifacts/sweep-dataset.json", "/held_out/stored_representatives_checked", "int", "stored sweep-v2 representative maps reproduced node-wise")
    m.add("SwtVTwoGatesPassed", "artifacts/sweep-dataset.json", "/headline/v2_gates_passed_count", "int", "designs passing the six sweep-v2 metric gates")

    # ---- hypothesis outcome ----
    test_pointer = "/estimands/sobol_v3/hypothesis_test"
    m.add("SwtSlope", "artifacts/sweep-dataset.json", f"{test_pointer}/slope_through_origin", "fixed3", "least-squares slope of rho on I_1(x_w) through the origin (Sobol cusps)")
    m.add("SwtRSquared", "artifacts/sweep-dataset.json", f"{test_pointer}/r_squared", "fixed2", "R^2 of rho against the fitted line (Sobol cusps)")
    m.add("SwtBandFraction", "artifacts/sweep-dataset.json", f"{test_pointer}/fraction_within_band", "pct0", "fraction of Sobol cusps with rho / I_1 within the agreement band")
    m.add("SwtRhoOverIOneMin", "artifacts/sweep-dataset.json", f"{test_pointer}/rho_over_i1/min", "fixed2", "smallest rho / I_1 over the Sobol cusps")
    m.add("SwtRhoOverIOneMedian", "artifacts/sweep-dataset.json", f"{test_pointer}/rho_over_i1/median", "fixed2", "median rho / I_1 over the Sobol cusps")
    m.add("SwtRhoOverIOneMax", "artifacts/sweep-dataset.json", f"{test_pointer}/rho_over_i1/max", "fixed2", "largest rho / I_1 over the Sobol cusps")
    m.add("SwtConfusionBoth", "artifacts/sweep-dataset.json", f"{test_pointer}/confusion_predicted_i1_vs_realised/predicted_and_realised", "int", "designs predicted and realised HEMP-like")
    m.add("SwtConfusionPredictedOnly", "artifacts/sweep-dataset.json", f"{test_pointer}/confusion_predicted_i1_vs_realised/predicted_not_realised", "int", "designs predicted but not realised HEMP-like")
    m.add("SwtConfusionRealisedOnly", "artifacts/sweep-dataset.json", f"{test_pointer}/confusion_predicted_i1_vs_realised/not_predicted_but_realised", "int", "designs realised but not predicted HEMP-like")
    m.add("SwtConfusionNeither", "artifacts/sweep-dataset.json", f"{test_pointer}/confusion_predicted_i1_vs_realised/neither", "int", "designs neither predicted nor realised HEMP-like")
    m.add("SwtAccuracy", "artifacts/sweep-dataset.json", f"{test_pointer}/prediction_accuracy", "fixed2", "prediction accuracy of the I_1 threshold over the Sobol designs")
    m.add("SwtSmallestXwHempLike", "artifacts/sweep-dataset.json", f"{test_pointer}/smallest_x_w_realised_hemp_like", "fixed2", "smallest x_w of a realised HEMP-like design")
    m.add("SwtLargestXwNotHempLike", "artifacts/sweep-dataset.json", f"{test_pointer}/largest_x_w_not_hemp_like", "fixed2", "largest x_w of a design that is not HEMP-like")
    m.add("SwtXStarFromSlope", "artifacts/sweep-dataset.json", f"{test_pointer}/x_star_from_fitted_slope", "fixed2", "x_w at which the fitted line reaches 1.5 (realised threshold)")
    m.add("SwtRwOverLStarFromSlope", "artifacts/sweep-dataset.json", f"{test_pointer}/wall_radius_over_pitch_star_from_fitted_slope", "fixed3", "r_w / L at which the fitted line reaches 1.5")
    m.add_derived("SwtRealisedThresholdShift", test["x_star_from_fitted_slope"] / test["x_star_prediction"] - 1.0, "pct0", "relative shift of the realised threshold above x*", "x_star_from_fitted_slope / x_star_prediction - 1", [{"artifact": "artifacts/sweep-dataset.json", "pointer": test_pointer}])
    m.add_derived("SwtHOneSlopeInRange", 0.80 <= test["slope_through_origin"] <= 1.00, "bool", "H1 slope condition met", "0.80 <= slope_through_origin <= 1.00", [{"artifact": "artifacts/sweep-dataset.json", "pointer": f"{test_pointer}/slope_through_origin"}])
    m.add_derived("SwtHOneBandMet", test["fraction_within_band"] >= 0.80, "bool", "H1 band-fraction condition met", "fraction_within_band >= 0.80", [{"artifact": "artifacts/sweep-dataset.json", "pointer": f"{test_pointer}/fraction_within_band"}])
    m.add_derived("SwtHTwoAccuracyMet", test["prediction_accuracy"] >= 0.85, "bool", "H2 accuracy condition met", "prediction_accuracy >= 0.85", [{"artifact": "artifacts/sweep-dataset.json", "pointer": f"{test_pointer}/prediction_accuracy"}])
    m.add_derived("SwtHTwoNoRegionHempLike", headline["sweep_v2_region_hemp_like_count"] == 0, "bool", "H2 condition that no design of the sweep-v2 region is HEMP-like", "sweep_v2_region_hemp_like_count == 0", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sweep_v2_region_hemp_like_count"}])
    m.add_derived("SwtEndCuspCount", len(end_ratios), "int", "end cusps of the Sobol designs (first and last cusp of each design)", "count of Sobol cusps at index 0 or n-1 of their design", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtEndCuspRhoOverIOneMedian", statistics.median(end_ratios), "fixed2", "median rho / I_1 at the end cusps", "median of rho_conservative / i1_x_w over the Sobol end cusps", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtInteriorCuspCount", len(interior_ratios), "int", "interior cusps of the Sobol designs", "count of Sobol cusps strictly between the first and the last of their design", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtInteriorCuspRhoOverIOneMedian", statistics.median(interior_ratios), "fixed2", "median rho / I_1 at the interior cusps", "median of rho_conservative / i1_x_w over the Sobol interior cusps", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtEndCuspRhoOverIOneMax", max(end_ratios), "fixed2", "largest rho / I_1 at an end cusp", "max of rho_conservative / i1_x_w over the Sobol end cusps", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtInteriorCuspRhoOverIOneMax", max(interior_ratios), "fixed2", "largest rho / I_1 at an interior cusp", "max of rho_conservative / i1_x_w over the Sobol interior cusps", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtPredictedOnlyEndCuspFailures", predicted_only_end_failures, "int", "predicted-but-not-realised designs whose only sub-threshold cusps are end cusps", "count of sobol_v3 rows with predicted_hemp_like_i1 and not hemp_like_all_cusps whose cusps with rho < 1.5 are all at index 0 or n-1", [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}])
    m.add_derived("SwtHOneAsPredicted", False, "bool", "H1 held as preregistered", "slope in [0.80, 1.00] and band fraction >= 0.80 (both required)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": test_pointer}])
    m.add_derived("SwtHTwoAsPredicted", False, "bool", "H2 held as preregistered", "accuracy >= 0.85 and no HEMP-like design in the sweep-v2 region (both required)", [{"artifact": "artifacts/sweep-dataset.json", "pointer": test_pointer}])
    if (0.80 <= test["slope_through_origin"] <= 1.00 and test["fraction_within_band"] >= 0.80) or (test["prediction_accuracy"] >= 0.85 and headline["sweep_v2_region_hemp_like_count"] == 0):
        raise ValueError("a preregistered hypothesis held; the admission text records both as not held")
    if statistics.median(end_ratios) >= statistics.median(interior_ratios):
        raise ValueError("the end-cusp median rho / I_1 is not below the interior-cusp median")
    band_inputs = [{"artifact": "artifacts/sweep-dataset.json", "pointer": "/designs"}, {"artifact": "artifacts/sweep-dataset.json", "pointer": "/headline/sobol_hypothesis_test/x_star_prediction"}]
    for i, b in enumerate(bands):
        prefix = f"SwtBand{b['token']}"
        if i > 0:
            m.add_derived(f"{prefix}Lo", b["low"], "fixed2", f"lower x_w edge of band {i}", "x* for band 1, otherwise a generator constant", band_inputs)
        m.add_derived(f"{prefix}Hi", b["high"], "fixed2", f"upper x_w edge of band {i}", "x* for the band below the threshold, otherwise a generator constant", band_inputs)
        m.add_derived(f"{prefix}Designs", b["designs"], "int", f"Sobol designs in band {i}", "count of sobol_v3 rows with low <= x_w < high", band_inputs)
        m.add_derived(f"{prefix}HempLike", b["hemp_like"], "int", f"HEMP-like Sobol designs in band {i}", "count of those rows with hemp_like_all_cusps", band_inputs)
        m.add_derived(f"{prefix}Cusps", b["cusps"], "int", f"wall cusps in band {i}", "count of cusps of those rows", band_inputs)
        m.add_derived(f"{prefix}RhoMedian", b["rho"]["median"], "fixed2", f"median rho in band {i}", "median rho_conservative over the cusps of those rows", band_inputs)
        m.add_derived(f"{prefix}RhoMin", b["rho"]["min"], "fixed2", f"smallest rho in band {i}", "min rho_conservative over the cusps of those rows", band_inputs)
        m.add_derived(f"{prefix}RhoMax", b["rho"]["max"], "fixed2", f"largest rho in band {i}", "max rho_conservative over the cusps of those rows", band_inputs)
        m.add_derived(f"{prefix}RhoOverIOneMedian", b["rho_over_i1"]["median"], "fixed2", f"median rho / I_1 in band {i}", "median of rho_conservative / i1_x_w over the cusps of those rows", band_inputs)
    m.add_derived("SwtBandCount", len(bands), "int", "x_w bands of the band table", "one band below x* and three above it", band_inputs)

    # ---- the PPM review: launch-position analysis, adiabaticity and the wall-loss launch design ----
    review_inputs = [{"artifact": f"definition-source:{PPM_CHECK_OUTPUT.as_posix()}", "pointer": "/reflections"}]
    m.add_derived("SwtLiteratureCommit", LITERATURE_COMMIT_SHA, "sha_short", "commit of the bound TWT/PPM review, its check script and its output", "git commit at which the review named by protocol.purpose is bound (also the shakedown commit)", [{"artifact": "artifacts/protocol.json", "pointer": "/purpose"}])
    m.add_derived("SwtReviewDocument", LITERATURE_REVIEW.as_posix(), "text", "path of the bound literature review", "constant of the generator; required to appear in protocol.purpose", [{"artifact": "artifacts/protocol.json", "pointer": "/purpose"}])
    m.add_derived("SwtLiteratureKeys", list(LITERATURE_KEYS), "list_ident_tt", "keys of the design-ratio sources", "constants of the generator whose IEPC numbers appear in protocol.descriptors_v3.koch_rho.citation and in the bound review", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/koch_rho/citation"}])
    m.add_derived("SwtLiteratureSourceCount", len(LITERATURE_KEYS), "int", "design-ratio sources named by the frozen protocol", "len(LITERATURE_KEYS)", [{"artifact": "artifacts/protocol.json", "pointer": "/descriptors_v3/koch_rho/citation"}])
    m.add_derived("SwtPpmReviewDesigns", len(check_output["reflections"]), "int", "sweep-v2 representatives whose recorded orbits the review re-read", "len(check output reflections)", review_inputs)
    m.add_derived("SwtPpmFieldCount", len(check_output["results"]), "int", "recorded fields the review fitted (four representatives plus P2)", "len(check output results)", [{"artifact": f"definition-source:{PPM_CHECK_OUTPUT.as_posix()}", "pointer": "/results"}])
    m.add_derived("SwtPpmLaunchCells", len(cells), "int", "launch cells of the geometry screening re-read by the review", "count of reflections[*].cells", review_inputs)
    m.add_derived("SwtPpmOrbitsPerCell", cells[0]["orbits"], "int", "orbits per launch cell", "the common reflections[*].cells[*].orbits", review_inputs)
    m.add_derived("SwtPpmNearCentrePitch", NEAR_CENTRE_PITCH, "fixed2", "near-centre class: launch within this many pitches of a magnet centre", "constant of the review (section 3.3) applied to cells[*].dist_to_centre_over_pitch", review_inputs)
    m.add_derived("SwtPpmFarCentrePitch", FAR_CENTRE_PITCH, "fixed2", "far class: launch at least this many pitches from a magnet centre", "constant of the review (section 3.3) applied to cells[*].dist_to_centre_over_pitch", review_inputs)
    m.add_derived("SwtPpmNearCells", len(near), "int", "launch cells within the near-centre class", "count of cells with dist_to_centre_over_pitch <= 0.17", review_inputs)
    m.add_derived("SwtPpmNearMaxPitch", max(c["dist_to_centre_over_pitch"] for c in near), "fixed3", "largest centre distance among the near cells (pitch)", "max dist_to_centre_over_pitch over the near cells", review_inputs)
    m.add_derived("SwtPpmNearReflectionsMin", min(c["reflected"] for c in near), "int", "fewest reflections in a near cell", "min reflected over the near cells", review_inputs)
    m.add_derived("SwtPpmNearReflectionsMax", max(c["reflected"] for c in near), "int", "most reflections in a near cell", "max reflected over the near cells", review_inputs)
    m.add_derived("SwtPpmFarCells", len(far), "int", "launch cells in the far class", "count of cells with dist_to_centre_over_pitch >= 0.22", review_inputs)
    m.add_derived("SwtPpmFarMinPitch", min(c["dist_to_centre_over_pitch"] for c in far), "fixed2", "smallest centre distance among the far cells (pitch)", "min dist_to_centre_over_pitch over the far cells", review_inputs)
    m.add_derived("SwtPpmFarMaxPitch", max(c["dist_to_centre_over_pitch"] for c in far), "fixed2", "largest centre distance among the far cells (pitch)", "max dist_to_centre_over_pitch over the far cells", review_inputs)
    m.add_derived("SwtPpmFarReflectionsMin", min(c["reflected"] for c in far), "int", "fewest reflections in a far cell", "min reflected over the far cells", review_inputs)
    m.add_derived("SwtPpmFarReflectionsMax", max(c["reflected"] for c in far), "int", "most reflections in a far cell", "max reflected over the far cells", review_inputs)
    m.add_derived("SwtPpmMirrorConditionMin", min(b["reflected_fraction_meeting_mirror_condition"] for b in check_output["reflections"].values()), "pct0", "smallest per-design fraction of reflections meeting the mirror condition", "min reflections[*].reflected_fraction_meeting_mirror_condition", review_inputs)
    m.add_derived("SwtPpmMirrorConditionMax", max(b["reflected_fraction_meeting_mirror_condition"] for b in check_output["reflections"].values()), "pct0", "largest per-design fraction of reflections meeting the mirror condition", "max reflections[*].reflected_fraction_meeting_mirror_condition", review_inputs)
    m.add_derived("SwtPpmBTurnMedianMin", min(b["reflected_median_b_turn_over_b_launch"] for b in check_output["reflections"].values()), "fixed2", "smallest per-design median |B|_turn / |B|_launch", "min reflections[*].reflected_median_b_turn_over_b_launch", review_inputs)
    m.add_derived("SwtPpmBTurnMedianMax", max(b["reflected_median_b_turn_over_b_launch"] for b in check_output["reflections"].values()), "fixed2", "largest per-design median |B|_turn / |B|_launch", "max reflections[*].reflected_median_b_turn_over_b_launch", review_inputs)
    m.add_derived("SwtPpmBTurnBelowLaunchMin", min(b["reflected_fraction_b_turn_below_launch"] for b in check_output["reflections"].values()), "pct0", "smallest per-design fraction of reflections turning at |B| below the launch value", "min reflections[*].reflected_fraction_b_turn_below_launch", review_inputs)
    m.add_derived("SwtPpmBTurnBelowLaunchMax", max(b["reflected_fraction_b_turn_below_launch"] for b in check_output["reflections"].values()), "pct0", "largest per-design fraction of reflections turning at |B| below the launch value", "max reflections[*].reflected_fraction_b_turn_below_launch", review_inputs)
    m.add_derived("SwtPpmWallHitNearNullMin", min(b["wall_hit_fraction_closer_to_null_than_centre"] for b in check_output["reflections"].values()), "pct0", "smallest per-design fraction of wall hits closer to a null than to a magnet centre", "min reflections[*].wall_hit_fraction_closer_to_null_than_centre", review_inputs)
    m.add_derived("SwtPpmWallHitNearNullMax", max(b["wall_hit_fraction_closer_to_null_than_centre"] for b in check_output["reflections"].values()), "pct0", "largest per-design fraction of wall hits closer to a null than to a magnet centre", "max reflections[*].wall_hit_fraction_closer_to_null_than_centre", review_inputs)
    electron_inputs = [{"artifact": f"definition-source:{PPM_CHECK_OUTPUT.as_posix()}", "pointer": "/results"}]
    m.add_derived("SwtPpmMendelAlphaMin", min(alphas), "sig3", "smallest Mendel stability parameter alpha over the launch energies and fields", "min results[*].electrons[*].mendel_alpha", electron_inputs)
    m.add_derived("SwtPpmMendelAlphaMax", max(alphas), "sig3", "largest Mendel stability parameter alpha", "max results[*].electrons[*].mendel_alpha", electron_inputs)
    m.add_derived("SwtPpmTwtAlphaLimit", 0.66, "fixed2", "first Mathieu stop band of the TWT PPM regime", "constant quoted by the review (section 3.2) from Mendel-Quate-Yocom / Carlsten et al.", [{"artifact": f"definition-source:{LITERATURE_REVIEW.as_posix()}", "pointer": ""}])
    if "alpha < 0.66" not in review_text:
        raise ValueError("the bound review does not quote the TWT stop-band limit")
    m.add_derived("SwtPpmEpsilonMin", min(epsilons), "fixed2", "smallest wall-cusp adiabaticity parameter epsilon", "min results[*].electrons[*].epsilon_wall_cusp", electron_inputs)
    m.add_derived("SwtPpmEpsilonMax", max(epsilons), "fixed2", "largest wall-cusp adiabaticity parameter epsilon", "max results[*].electrons[*].epsilon_wall_cusp", electron_inputs)
    m.add_derived("SwtPpmEnergiesEv", energies, "list_g", "electron launch energies of the campaigns (eV)", "sorted distinct results[*].electrons[*].energy_ev (equal to the wall-loss launch energies)", electron_inputs)
    m.add_derived("SwtPpmMuMedianMin", min(mu_by_design.values()), "fixed2", "smallest per-design median magnetic-moment variation", "min reflections[*].mu_median", review_inputs)
    m.add_derived("SwtPpmMuMedianMax", max(mu_by_design.values()), "fixed2", "largest per-design median magnetic-moment variation", "max reflections[*].mu_median", review_inputs)
    m.add_derived("SwtPpmMuOrderedByEpsilon", mu_ordered, "bool", "per-design mu-variation medians ordered by the wall-cusp epsilon", "reflections[*].mu_median is non-decreasing when the designs are sorted by their epsilon_wall_cusp at the higher launch energy", review_inputs + electron_inputs)
    m.add_derived("SwtPpmLeffelGrayEpsilon", 0.03, "fixed2", "Leffel-Gray adiabatic boundary in epsilon (upper end)", "constant quoted by the review (section 3.2): 2 pi epsilon ~ 0.11-0.19, epsilon ~ 0.02-0.03", [{"artifact": f"definition-source:{LITERATURE_REVIEW.as_posix()}", "pointer": ""}])
    if "epsilon ~ 0.02-0.03" not in review_text or min(epsilons) <= 0.03:
        raise ValueError("the review's Leffel-Gray boundary is absent or not exceeded by every launch class")
    launch_inputs = [{"artifact": f"reference:{WALL_LOSS_PROTOCOL.as_posix()}", "pointer": "/launches/position_seeds"}, {"artifact": f"reference:{TOPOLOGY_P2_RECORD.as_posix()}", "pointer": "/geometry/stage_centres_m"}]
    m.add_derived("SwtVFourLaunchCells", len(launch_z), "int", "launch cells of the wall-loss campaign", "distinct launches.position_seeds[*].cell_id of the frozen wall-loss protocol", launch_inputs)
    m.add_derived("SwtVFourLaunchZMm", launch_z, "list_mm1", "launch planes of the wall-loss campaign (mm)", "sorted distinct launches.position_seeds[*].position_m[2]", launch_inputs)
    m.add_derived("SwtPTwoStageCentresMm", p2_centres, "list_mm1", "magnet-stage centres of the P2 field (mm)", "geometry.stage_centres_m of the topology screening's P2 record", launch_inputs)
    m.add_derived("SwtPTwoPitchMm", p2_pitch, "mm1", "stage pitch of the P2 field (mm)", "geometry.stage_pitch_m of the topology screening's P2 record", launch_inputs)
    m.add_derived("SwtVFourLaunchOffsetMm", launch_offset_m, "mm1", "distance of every wall-loss launch plane from its nearest magnet centre (mm)", "min over stage centres of |launch z - centre|, identical for the four launch planes", launch_inputs)
    m.add_derived("SwtVFourLaunchOffsetPitch", launch_offset_m / p2_pitch, "fixed3", "the same distance in stage pitches", "launch offset / P2 stage pitch", launch_inputs)
    m.add_derived("SwtVFourLaunchInNearClass", launch_offset_m / p2_pitch <= NEAR_CENTRE_PITCH, "bool", "the wall-loss launch planes fall in the review's near-centre class", "launch offset in pitches <= 0.17", launch_inputs)
    line_inputs = [{"artifact": f"definition-source:{PPM_CHECK_OUTPUT.as_posix()}", "pointer": "/results"}, {"artifact": f"reference:{WALL_LOSS_PROTOCOL.as_posix()}", "pointer": "/launches/position_seeds"}]
    m.add_derived("SwtPpmLineCount", len(field_lines), "int", "launch field lines the review traced from the magnet centres (two launch radii in each recorded field)", "count of results[*].field_lines", line_inputs)
    m.add_derived("SwtVFourLaunchRadiiFraction", launch_radii, "list_fixed3", "wall-loss launch radii as fractions of the P2 wall radius", "sorted distinct launches.position_seeds[*].position_m[0] / P2 wall radius (equal to the review's launch_r_over_rw)", line_inputs)
    m.add_derived("SwtPpmLineAllReachWall", all(line["reaches_wall"] for line in field_lines), "bool", "every traced launch field line reaches the wall before the cusp plane", "all(results[*].field_lines[*].reaches_wall)", line_inputs)
    m.add_derived("SwtPpmLineMaxOverLaunchMax", max(line["max_along_line_over_launch"] for line in field_lines), "fixed2", "largest |B| along any traced launch field line over its launch value", "max results[*].field_lines[*].max_along_line_over_launch", line_inputs)
    m.add_derived("SwtPpmLineWallOverLaunchMin", min(line["ratio_wall_over_launch"] for line in field_lines), "fixed2", "smallest wall-hit |B| over launch |B| along the traced lines", "min results[*].field_lines[*].ratio_wall_over_launch", line_inputs)
    m.add_derived("SwtPpmLineWallOverLaunchMax", max(line["ratio_wall_over_launch"] for line in field_lines), "fixed2", "largest wall-hit |B| over launch |B| along the traced lines", "max results[*].field_lines[*].ratio_wall_over_launch", line_inputs)
    m.add_derived("SwtPpmLineWallFractionMin", min(line["wall_hit_fraction_to_cusp"] for line in field_lines), "pct0", "smallest fraction of the centre-to-cusp distance at which a traced line reaches the wall", "min results[*].field_lines[*].wall_hit_fraction_to_cusp", line_inputs)
    m.add_derived("SwtPpmLineWallFractionMax", max(line["wall_hit_fraction_to_cusp"] for line in field_lines), "pct0", "largest fraction of the centre-to-cusp distance at which a traced line reaches the wall", "max results[*].field_lines[*].wall_hit_fraction_to_cusp", line_inputs)
    if max(line["max_along_line_over_launch"] for line in field_lines) > 1.0 + 1e-9:
        raise ValueError("a traced launch field line carries a |B| maximum above its launch value")

    # ---- claim-boundary flags ----
    m.add("SwtFieldLevelStatement", "artifacts/protocol.json", "/claim_boundary/field_level", "text", "field level statement of the claim boundary")
    m.add("SwtIronSensitivity", "artifacts/protocol.json", "/claim_boundary/iron_sensitivity", "text", "iron sensitivity statement of the claim boundary")
    m.add("SwtConfirmationStatus", "artifacts/campaign-result.json", "/l1b_p2_confirmation_queue/status", "ident", "status of the material-aware confirmation")
    m.add("SwtConfirmationStatement", "artifacts/campaign-result.json", "/l1b_p2_confirmation_queue/statement", "text", "statement of the material-aware confirmation queue")
    m.add("SwtUsableAs", "artifacts/protocol.json", "/claim_boundary/usable_as", "list_clauses", "permitted uses of the catalogue")
    m.add("SwtMirrorDescriptorsNotProbabilities", "artifacts/protocol.json", "/claim_boundary/mirror_ratios_are_field_descriptors_not_probabilities", "bool", "mirror ratios are field descriptors, not probabilities")
    m.add("SwtForbidMirrorProbability", "artifacts/protocol.json", "/claim_boundary/forbid_mirror_probability_publication", "bool", "mirror-probability publication forbidden")
    m.add("SwtForbidPlasmaPerformance", "artifacts/protocol.json", "/claim_boundary/forbid_plasma_performance_publication", "bool", "plasma or performance publication forbidden")
    m.add("SwtShakedownNotEvidence", "artifacts/protocol.json", "/claim_boundary/shakedown_outcomes_are_not_evidence", "bool", "shakedown outcomes are not evidence")
    m.add("SwtCatalogueSchema", "artifacts/cusp-cell-catalogue-v3.json", "/schema_version", "ident", "catalogue schema version")
    m.add("SwtCatalogueHempLike", "artifacts/cusp-cell-catalogue-v3.json", "/hemp_like_design_count", "int", "HEMP-like designs recorded by the catalogue")
    m.add_derived("SwtPhysicsLevelOpened", False, "bool", "a physics level is opened", "constant of the admission: the numerical-screening gate declares opens_level null", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary"}])
    m.add_derived("SwtHardwareValidation", False, "bool", "hardware or experimental validation claimed", "constant of the admission: no measurement enters the campaign", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}])
    m.add_derived("SwtMaterialAwareConfirmationRun", False, "bool", "a material-aware (L1b or P2) confirmation of rho was run", "campaign-result.l1b_p2_confirmation_queue.status == queued_not_run", [{"artifact": "artifacts/campaign-result.json", "pointer": "/l1b_p2_confirmation_queue/status"}])
    m.add_derived("SwtIronInField", False, "bool", "the declared iron pole pieces carry a material response in the field", "constant of the admission restating protocol.claim_boundary.field_level (the pole pieces are source-free vacuum in the L1a field)", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/field_level"}])
    m.add_derived("SwtRhoIsProbability", False, "bool", "rho or any mirror descriptor is a probability", "constant of the admission restating protocol.claim_boundary.mirror_ratios_are_field_descriptors_not_probabilities", [{"artifact": "artifacts/protocol.json", "pointer": "/claim_boundary/mirror_ratios_are_field_descriptors_not_probabilities"}])

    # ================================================================== tables ====
    tex_lines = [
        "% Generated by paper/scripts/generate_l1a_sweep_v3_evidence.py; do not hand edit.",
        f"% Evidence: {EXPERIMENT.as_posix()} at commit {RESULTS_COMMIT_SHA} (results manifest SHA-256 {bundle.manifest_sha256}).",
        f"% Every macro value traces to an artifact path and JSON pointer recorded in {EVIDENCE_PATH.as_posix()}.",
    ]
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    # (a) design box v2 vs v3
    box_rows: list[str] = []
    for name, _token, label in WIDENED_VARIABLES:
        v3 = variables[name]
        box_rows.append(f"{label} (mm) & {format_value('mm2', box_v2[name][0])}--{format_value('mm2', box_v2[name][1])} & {format_value('mm2', v3['lower'])}--{format_value('mm2', v3['upper'])}\\\\")
    box_rows.append(f"other variables ({len(sampling['variables']) - len(WIDENED_VARIABLES)}) & \\multicolumn{{2}}{{c}}{{identical bounds}}\\\\")
    box_rows.append("\\midrule")
    box_rows.append(f"$r_w/L$ implied by the box & {format_value('fixed3', box_v2['chamber_outer_radius_m'][0] / box_v2['stage_pitch_m'][1])}--{format_value('fixed3', box_v2['chamber_outer_radius_m'][1] / box_v2['stage_pitch_m'][0])} & {format_value('fixed3', coverage['wall_radius_over_pitch'][0])}--{format_value('fixed3', coverage['wall_radius_over_pitch'][1])}\\\\")
    box_rows.append(f"$x_w = \\pi r_w/L$ implied by the box & {format_value('fixed2', math.pi * box_v2['chamber_outer_radius_m'][0] / box_v2['stage_pitch_m'][1])}--{format_value('fixed2', math.pi * box_v2['chamber_outer_radius_m'][1] / box_v2['stage_pitch_m'][0])} & {format_value('fixed2', coverage['x_w'][0])}--{format_value('fixed2', coverage['x_w'][1])}\\\\")
    box_rows.append(f"$r_w/L$ realised & {format_value('fixed3', held_out_est['wall_radius_over_pitch']['min'])}--{format_value('fixed3', held_out_est['wall_radius_over_pitch']['max'])} & {format_value('fixed3', sobol_est['wall_radius_over_pitch']['min'])}--{format_value('fixed3', sobol_est['wall_radius_over_pitch']['max'])}\\\\")
    box_rows.append(f"$x_w$ realised & {format_value('fixed2', held_out_est['x_w']['min'])}--{format_value('fixed2', held_out_est['x_w']['max'])} & {format_value('fixed2', sobol_est['x_w']['min'])}--{format_value('fixed2', sobol_est['x_w']['max'])}\\\\")
    box_rows.append(f"designs (stages $3/4/5$) & {held_out_est['design_count']} ({held_out_est['by_stage_count']['3']['designs']}/{held_out_est['by_stage_count']['4']['designs']}/{held_out_est['by_stage_count']['5']['designs']}) & {sobol_est['design_count']} ({sobol_est['by_stage_count']['3']['designs']}/{sobol_est['by_stage_count']['4']['designs']}/{sobol_est['by_stage_count']['5']['designs']})\\\\")
    box_rows.append(f"designs with $x_w \\ge x^*$ ($I_1 \\ge {format_value('fixed1', 1.5)}$) & {held_out_est['predicted_hemp_like_i1_count']} & {sobol_est['predicted_hemp_like_i1_count']}\\\\")
    box_rows.append(f"HEMP-like designs ($\\rho \\ge {format_value('fixed1', 1.5)}$ at every cusp) & {held_out_est['hemp_like_count']} & {sobol_est['hemp_like_count']}\\\\")
    box_rows.append(f"largest $\\rho$ over every cusp & {format_value('fixed3', held_out_est['rho_conservative']['max'])} & {format_value('fixed2', sobol_est['rho_conservative']['max'])}\\\\")
    tex_lines += _table(
        "SwtDesignBoxTable",
        "The sweep-v2 design box against the sweep-v3 box (frozen protocol, \\texttt{sampling.variables} and "
        "\\texttt{sampling.sweep\\_v2\\_box}): the four widened variables, the wall-radius-to-pitch ratio "
        "$r_w/L$ and $x_w = \\pi r_w/L$ implied by the box corners and realised by the designs, the design counts, "
        "the designs above the single-harmonic threshold $x^*$ and the HEMP-like designs. The v2 column reports "
        "the \\SwtHeldOutDesigns{} accepted sweep-v2 designs re-solved as the held-out set; every $\\rho$ is a "
        "field ratio of a linear-vacuum screening field.",
        "tab:l1a-sweep-v3-box", f"{_p(5.6)}rr",
        "quantity & sweep v2 (held-out set) & sweep v3 (Sobol set)\\\\", box_rows,
    )
    # (b) rho by x_w band
    band_rows: list[str] = []
    for i, b in enumerate(bands):
        # Rendered as one math interval so the row never starts with "[" (which booktabs'
        # \midrule would swallow as an optional argument).
        low = "0" if i == 0 else ("x^*" if i == 1 else format_value("fixed2", b["low"]))
        high = "x^*" if i == 0 else format_value("fixed2", b["high"])
        rho_med = format_value("fixed2", b["rho"]["median"]) if b["rho"]["median"] is not None else "--"
        rho_rng = f"{format_value('fixed2', b['rho']['min'])}--{format_value('fixed2', b['rho']['max'])}" if b["cusps"] else "--"
        ratio_med = format_value("fixed2", b["rho_over_i1"]["median"]) if b["rho_over_i1"]["median"] is not None else "--"
        i1_rng = f"{format_value('fixed2', b['i1']['min'])}--{format_value('fixed2', b['i1']['max'])}" if b["designs"] else "--"
        band_rows.append(f"$[{low}, {high})$ & {i1_rng} & {b['designs']} & {b['predicted']} & {b['hemp_like']} & {b['cusps']} & {rho_med} & {rho_rng} & {ratio_med}\\\\")
    band_rows.append("\\midrule")
    band_rows.append(f"all Sobol designs & {format_value('fixed2', min(d['ppm_prediction']['i1_x_w'] for d in sobol_rows))}--{format_value('fixed2', max(d['ppm_prediction']['i1_x_w'] for d in sobol_rows))} & {len(sobol_rows)} & {sobol_est['predicted_hemp_like_i1_count']} & {sobol_est['hemp_like_count']} & {sobol_est['cusp_count']} & {format_value('fixed2', sobol_est['rho_conservative']['median'])} & {format_value('fixed2', sobol_est['rho_conservative']['min'])}--{format_value('fixed2', sobol_est['rho_conservative']['max'])} & {format_value('fixed2', test['rho_over_i1']['median'])}\\\\")
    band_rows.append(f"sweep-v2 region ({len(region_rows)} designs) & {format_value('fixed2', min(d['ppm_prediction']['i1_x_w'] for d in region_rows))}--{format_value('fixed2', max(d['ppm_prediction']['i1_x_w'] for d in region_rows))} & {len(region_rows)} & {region_est['predicted_hemp_like_i1_count']} & {region_est['hemp_like_count']} & {region_est['cusp_count']} & {format_value('fixed2', region_est['rho_conservative']['median'])} & {format_value('fixed2', region_est['rho_conservative']['min'])}--{format_value('fixed3', region_est['rho_conservative']['max'])} & {format_value('fixed2', region_est['hypothesis_test']['rho_over_i1']['median'])}\\\\")
    tex_lines += _table(
        "SwtBandTable",
        "Koch design ratio $\\rho$ (conservative reading: wall $|B|$ at the cusp over the larger adjacent axis peak) "
        "by $x_w$ band over the \\SwtSobolDesigns{} Sobol designs, with the single-harmonic prediction "
        "$I_1(x_w)$, the designs predicted HEMP-like by $I_1 \\ge \\SwtRhoThreshold$, the designs realised "
        "HEMP-like ($\\rho \\ge \\SwtRhoThreshold$ at every wall cusp), the cusp count, the median and range of "
        "$\\rho$ and the median $\\rho / I_1$. The band below $x^* = \\SwtXStar$ is where the prediction says "
        "no design can be HEMP-like; the last row pools the held-out sweep-v2 designs with the Sobol designs "
        "inside the v2 box. Every value is a field ratio re-derived from the sealed per-design rows.",
        "tab:l1a-sweep-v3-bands", f"{_p(2.9)}{_p(1.5)}rrrr{_p(1.25)}{_p(1.6)}r",
        "$x_w$ band & $I_1(x_w)$ & designs & predicted & HEMP-like & cusps & $\\rho$ median & $\\rho$ range & $\\rho/I_1$ median\\\\", band_rows,
        size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    # (c) hypothesis test
    hyp_rows = [
        f"H1: slope of $\\rho$ on $I_1(x_w)$ through the origin & $[{format_value('fixed2', 0.80)}, {format_value('fixed2', 1.00)}]$ & {format_value('fixed3', test['slope_through_origin'])} & {'yes' if 0.80 <= test['slope_through_origin'] <= 1.00 else 'no'}\\\\",
        f"H1: cusps with $\\rho/I_1$ within $\\pm${format_value('pct0', band)} & $\\ge {format_value('pct0', 0.80)}$ & {format_value('pct0', test['fraction_within_band'])} ({sum(1 for d in sobol_rows for r in d['rho'] if abs(r['rho_conservative'] / d['ppm_prediction']['i1_x_w'] - 1.0) <= band)} of {test['cusp_count']}) & {'yes' if test['fraction_within_band'] >= 0.80 else 'no'}\\\\",
        f"$R^2$ against the fitted line & reported & {format_value('fixed2', test['r_squared'])} & --\\\\",
        f"$\\rho/I_1$: median (min--max), all cusps & reported & {format_value('fixed2', test['rho_over_i1']['median'])} ({format_value('fixed2', test['rho_over_i1']['min'])}--{format_value('fixed2', test['rho_over_i1']['max'])}) & --\\\\",
        f"$\\rho/I_1$ median, end cusps ($n = {len(end_ratios)}$) & reported & {format_value('fixed2', statistics.median(end_ratios))} & --\\\\",
        f"$\\rho/I_1$ median, interior cusps ($n = {len(interior_ratios)}$) & reported & {format_value('fixed2', statistics.median(interior_ratios))} & --\\\\",
        "\\midrule",
        f"H2: prediction accuracy over {test['design_count_with_cusps']} designs & $\\ge {format_value('fixed2', 0.85)}$ & {format_value('fixed2', test['prediction_accuracy'])} & {'yes' if test['prediction_accuracy'] >= 0.85 else 'no'}\\\\",
        f"H2: HEMP-like designs inside the sweep-v2 region & $0$ & {headline['sweep_v2_region_hemp_like_count']} of {len(region_rows)} & {'yes' if headline['sweep_v2_region_hemp_like_count'] == 0 else 'no'}\\\\",
        f"predicted and realised / predicted only / realised only / neither & reported & {test['confusion_predicted_i1_vs_realised']['predicted_and_realised']} / {test['confusion_predicted_i1_vs_realised']['predicted_not_realised']} / {test['confusion_predicted_i1_vs_realised']['not_predicted_but_realised']} / {test['confusion_predicted_i1_vs_realised']['neither']} & --\\\\",
        f"predicted-only designs failing at end cusps only & reported & {predicted_only_end_failures} of {test['confusion_predicted_i1_vs_realised']['predicted_not_realised']} & --\\\\",
        "\\midrule",
        f"predicted threshold $x^*$ ($I_1 = {format_value('fixed1', 1.5)}$); $r_w/L$ & preregistered & {format_value('fixed3', test['x_star_prediction'])}; {format_value('fixed3', test['x_star_prediction'] / math.pi)} & --\\\\",
        f"realised threshold from the fitted slope; $r_w/L$ & reported & {format_value('fixed3', test['x_star_from_fitted_slope'])}; {format_value('fixed3', test['wall_radius_over_pitch_star_from_fitted_slope'])} & --\\\\",
        f"smallest $x_w$ realised HEMP-like; largest $x_w$ not HEMP-like & reported & {format_value('fixed2', test['smallest_x_w_realised_hemp_like'])}; {format_value('fixed2', test['largest_x_w_not_hemp_like'])} & --\\\\",
    ]
    tex_lines += _table(
        "SwtHypothesisTable",
        "The preregistered hypothesis test (\\texttt{protocol.json\\#descriptors\\_v3.hypothesis}; reported, not "
        "gated) over the \\SwtSobolCuspCount{} wall cusps of the \\SwtSobolDesigns{} Sobol designs: H1 asks "
        "whether the realised Koch ratio tracks the single-harmonic prediction $I_1(x_w)$, H2 whether the "
        "threshold $x^*$ classifies the designs. The end-cusp and interior-cusp medians and the end-cusp "
        "failure count are derived by the evidence generator from the sealed per-design rows; every other value "
        "is read from the sealed hypothesis record and re-derived from the rows.",
        "tab:l1a-sweep-v3-hypothesis", f"{_p(6.2)}{_p(2.2)}{_p(3.9)}r",
        "statistic & preregistered & observed & met\\\\", hyp_rows,
        size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    # (d) the HEMP-like designs
    hemp_rows_sorted = sorted((d for d in sobol_rows if d["hemp_like_all_cusps"]), key=lambda d: d["x_w"])
    hemp_rows: list[str] = []
    for d in hemp_rows_sorted:
        rhos = [r["rho_conservative"] for r in d["rho"]]
        flag = "yes" if d["five_stage_four_cusp_hemp_like"] else "--"
        hemp_rows.append(
            f"\\texttt{{{_short_id(d['design_id'])}}} & {d['derived']['stage_count']} & {d['wall_cusp_count']} & {format_value('mm2', d['derived']['represented_stage_pitch_m'])} & "
            f"{format_value('mm2', d['geometry']['wall_radius_m'])} & {format_value('fixed3', d['wall_radius_over_pitch'])} & {format_value('fixed2', d['x_w'])} & {format_value('fixed2', d['ppm_prediction']['i1_x_w'])} & "
            f"{format_value('fixed2', min(rhos))}--{format_value('fixed2', max(rhos))} & {format_value('fixed2', min(rhos) / d['ppm_prediction']['i1_x_w'])} & {format_value('fixed3', max(r['rho_wall'] for r in d['rho']))} & {flag}\\\\"
        )
    tex_lines += _table(
        "SwtHempLikeTable",
        "The \\SwtHempLikeCount{} HEMP-like Sobol designs (every wall cusp at $\\rho \\ge \\SwtRhoThreshold$), "
        "ordered by $x_w$: design ordinal, magnet stages, wall cusps, stage pitch, wall radius, $r_w/L$, $x_w$, "
        "the single-harmonic prediction $I_1(x_w)$, the range of $\\rho$ over the design's cusps, the smallest "
        "$\\rho/I_1$, the largest wall reading $\\rho_{\\mathrm{wall}}$ (below one everywhere: no cusp is the wall "
        "$|B|$ maximum) and whether the design meets the legacy five-stage four-cusp target. Values are field "
        "ratios of linear-vacuum screening fields; the material-aware confirmation queued for these designs was "
        "not run.",
        "tab:l1a-sweep-v3-hemp-like", f"lrr{_p(0.95)}{_p(0.9)}rrr{_p(1.5)}rr{_p(1.2)}",
        "design & $N$ & cusps & \\shortstack{pitch\\\\(mm)} & \\shortstack{$r_w$\\\\(mm)} & $r_w/L$ & $x_w$ & $I_1$ & $\\rho$ range & \\shortstack{min\\\\$\\rho/I_1$} & \\shortstack{max\\\\$\\rho_{\\mathrm{wall}}$} & \\shortstack{5-stage\\\\4-cusp}\\\\",
        hemp_rows, size="\\scriptsize", extra="\\setlength{\\tabcolsep}{3pt}",
    )
    tex = "\n".join(tex_lines) + "\n"

    reference_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (sweep_v2_file, topology_protocol_file, p2_record_file, wall_loss_protocol_file)
    }
    definition_files = {
        f["path"]: {"sha256": f["sha256"], "bytes": f["bytes"], "revision": f["revision"], "role": f["role"], "git_blob": f["git_blob"], "git_blob_sha256": f["git_blob_sha256"]}
        for f in (review_file, check_script_file, check_output_file)
    }
    evidence = {
        "document_type": "paper-l1a-sweep-v3-evidence",
        "schema_version": "1.0",
        "experiment_id": bundle.manifest["experiment_id"],
        "classification": CLASSIFICATION,
        "topology_label": TOPOLOGY_LABEL,
        "recorded_outcome": RECORDED_OUTCOME,
        "campaign_status": CAMPAIGN_STATUS,
        "screening_model": SCREENING_MODEL,
        "evidence_revision": RESULTS_COMMIT_SHA,
        "binding": binding,
        "dashboard": dashboard,
        "definition_sources": {
            "revision": LITERATURE_COMMIT_SHA,
            "files": definition_files,
            "literature_keys": list(LITERATURE_KEYS),
            "rule": (
                "the frozen protocol names the TWT/PPM review at its commit as the source of the Koch design ratio and of the "
                "single-harmonic hypothesis; the review, its read-only check script and the committed output the script wrote "
                "are bound at that commit and their checkouts equal the blobs (LF-normalised); the launch-position, adiabaticity "
                "and magnetic-moment macros are derived from the committed output and are a recorded analysis of the geometry "
                "screening's sealed orbits, not a result of this campaign"
            ),
        },
        "reference_artifacts": {
            "rule": (
                "the sealed sweep-v2 manifest the held-out set was identity-proven against (must hash to the sealed-source identity "
                "the bundle recorded), the frozen cusp-topology-v3.1 protocol whose definition parameters the sweep imported "
                "unchanged, the topology screening's P2 design record (stage centres of the wall-loss field) and the wall-loss "
                "campaign's frozen protocol (launch planes), bound at their own admitted revisions"
            ),
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
                "the section at its recorded outcome (an accepted field-only design-space screening) without opening "
                "any physics level. The Koch ratio and every mirror descriptor are field ratios of linear-vacuum "
                "screening fields and never probabilities; the preregistered hypothesis is reported at its recorded "
                "outcome (not as predicted); nothing here is a plasma, wall-loss or performance claim, and the "
                "material-aware confirmation the protocol queues was not run."
            ),
        },
        "bundle": {
            "manifest_path": (RESULTS / "manifest.json").as_posix(),
            "manifest_sha256": bundle.manifest_sha256,
            "artifact_count": bundle.manifest["artifact_count"],
            "verified_file_count": len(bundle.hashes),
            "tolerated_eol_files": [],
            "tolerance_rule": "none: every manifest file entry must hash to its recorded byte_sha256 with its recorded size",
            "recomputation_rule": (
                "the headline and every per-set estimand, including the preregistered hypothesis statistics, are re-derived "
                "from the per-design rows and must equal the sealed values (counts, histograms and medians exactly; sums "
                f"recomputed with math.fsum within a relative tolerance of {FLOAT_TOLERANCE:g}); every design record, field "
                "grid, catalogue entry and CSV row must agree with its row; x_w, the Bessel prediction, every rho reading "
                "and every flag recompute from their inputs"
            ),
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": {
            "SwtDesignBoxTable": {"rows": len(box_rows), "source": "artifacts/protocol.json#/sampling, artifacts/sweep-dataset.json#/estimands"},
            "SwtBandTable": {"rows": len(band_rows), "source": "artifacts/sweep-dataset.json#/designs (sobol_v3 rows, x_w bands), #/estimands"},
            "SwtHypothesisTable": {"rows": len(hyp_rows), "source": "artifacts/sweep-dataset.json#/estimands/sobol_v3/hypothesis_test, #/designs"},
            "SwtHempLikeTable": {"rows": len(hemp_rows), "source": "artifacts/sweep-dataset.json#/designs (sobol_v3 rows with hemp_like_all_cusps)"},
        },
        "generator": {
            "path": "paper/scripts/generate_l1a_sweep_v3_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_l1a_sweep_v3_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_l1a_sweep_v3_evidence.py",
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
        "definition_source_inputs": [
            {"path": path, "sha256": meta["sha256"], "bytes": meta["bytes"], "revision": meta["revision"]}
            for path, meta in evidence["definition_sources"]["files"].items()
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
        print(f"L1a sweep v3 evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
