"""Generate hash-bound paper evidence for the preregistered L1a topology screening studies.

Three committed, single-execution experiments are read from their sealed
bundles, verified against their own manifests, bound to their committed results
revisions and rendered as macro files for the manuscript:

* ``modern/experiments/l1a_geometry_sweep_v2`` — accepted L1a field-only design
  screening (96 designs, seven terminal gates, axis-cusp statistics);
* ``modern/experiments/four_cell_topology_search_v2`` — preregistered null
  result of the four-cell wall-cusp topology search (0 of 128 candidates
  stable under the frozen cusp/cell definition);
* ``modern/experiments/cft_topology_characterization_v1`` — recorded
  developmental characterization (56 designs, 0 stable eligible cusps or
  cells).

For every experiment the script writes

* ``paper/evidence/<key>.json`` — every macro value with the artifact path,
  JSON pointer, formatter and artifact SHA-256 it was read from (or, for a
  derived macro, its derivation and inputs);
* ``paper/generated/<key>.tex`` — ``\\newcommand`` macros and generated tables
  (each wrapped in ``\\ArtifactClaim``) for the bound section
  ``paper/sections/<key>.tex``;
* ``paper/generated/<key>.provenance.json`` — generator/input/output hashes in
  the shape of the other paper sidecars.

Only the Python standard library is used.  No wall-clock value or machine
path enters any output.  Two disclosed recording-layer defects are tolerated
exactly as ``sha256(bytes.replace(LF, CRLF)) == recorded`` for exactly the
audited file of each experiment (``POSTHOC_AUDIT.md`` of the sweep and of the
four-cell search) and nothing else.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Callable

from generate_wall_loss_v4_evidence import (
    FORMATTERS as _BASE_FORMATTERS,
    _tex_escape,
    canonical_json,
    load_json_bytes,
    resolve_pointer,
    sha256_bytes,
)

SCREENING_MODEL = "linear-vacuum L1a equivalent-current axisymmetric field (not a permanent-magnet or nonlinear-iron material model)"
GATE_KIND = "numerical-screening"
SECTION_TITLE = "Preregistered topology screening: sweep acceptance and four-cell null result"

FORMATTERS: dict[str, Callable[[Any], str]] = {
    **_BASE_FORMATTERS,
    "list_mm1": lambda v: ", ".join(f"{1e3 * float(x):.1f}" for x in v),
    "list_mm2": lambda v: ", ".join(f"{1e3 * float(x):.2f}" for x in v),
    "list_fixed1": lambda v: ", ".join(f"{float(x):.1f}" for x in v),
    "list_int": lambda v: ", ".join(f"{int(x):d}" for x in v),
    "list_ident_tt": lambda v: ", ".join(f"\\texttt{{{_BASE_FORMATTERS['ident'](x)}}}" for x in v),
    "list_clauses": lambda v: "; ".join(_tex_escape(str(x)) for x in v),
    "list_sentences": lambda v: " ".join(_tex_escape(str(x)) for x in v),
    "sci1": lambda v: _sci(float(v), 1),
    "fixed0": lambda v: f"{float(v):.0f}",
}


def _sci(value: float, digits: int) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def canonical_hash(value: Any) -> str:
    """SHA-256 of ``json-sort-keys-compact-utf8-v1`` (the experiments' semantic identity)."""

    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


# --------------------------------------------------------------------------- #
# Experiment specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuditedEolFile:
    """One file whose recorded digest is the CRLF-era digest of the LF bytes Git stores."""

    lf_sha256: str
    recorded_sha256: str
    recorded_bytes: int
    audit_module: str  # repository-relative module whose constants the checker must match


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    experiment_id: str
    experiment_path: Path
    results_commit: str
    preregistration_commit: str
    classification: str
    recorded_outcome: str
    macro_prefix: str
    gate_id: str
    manifest_id: str
    artifact_id: str
    artifact_claim_id: str
    prose_claim_ids: tuple[str, ...]
    section_heading: str
    table_macros: tuple[str, ...]
    revision_macro: str
    audited_eol_files: dict[str, AuditedEolFile] = field(default_factory=dict)
    posthoc_audit_commit: str | None = None
    posthoc_audit_path: str | None = None
    lineage: tuple[str, ...] = ()

    @property
    def evidence_path(self) -> Path:
        return Path(f"paper/evidence/{self.key}.json")

    @property
    def output_path(self) -> Path:
        return Path(f"paper/generated/{self.key}.tex")

    @property
    def sidecar_path(self) -> Path:
        return Path(f"paper/generated/{self.key}.provenance.json")

    @property
    def section_path(self) -> Path:
        return Path(f"paper/sections/{self.key}.tex")

    @property
    def manifest_path(self) -> Path:
        return Path(f"paper/evidence/manifests/{self.key}.json")

    @property
    def section_binding(self) -> str:
        return f"\\input{{sections/{self.key}.tex}}"

    @property
    def generated_binding(self) -> str:
        return f"\\input{{generated/{self.key}.tex}}"

    @property
    def document_type(self) -> str:
        return f"paper-{self.key}-evidence"


SWEEP = ExperimentSpec(
    key="l1a-sweep-v2",
    experiment_id="l1a-geometry-sweep-v2",
    experiment_path=Path("modern/experiments/l1a_geometry_sweep_v2"),
    results_commit="f30cb42ec4a8633bf634a3d32ffa5b11f66be97a",
    preregistration_commit="092f5fae692ee7d6711e0c7e1c94dac6a345f37c",
    classification="L1a_FIELD_ONLY_SCREENING_NOT_HARDWARE_VALID",
    recorded_outcome="accepted-screening",
    macro_prefix="Swp",
    gate_id="GATE-L1A-SWEEP-V2",
    manifest_id="L1A-SWEEP-V2-20260902-96-V1",
    artifact_id="TAB-L1A-SWEEP-V2",
    artifact_claim_id="CLM-020",
    prose_claim_ids=("CLM-018", "CLM-019", "CLM-021"),
    section_heading="Accepted geometry sweep: field-only design-space screening",
    table_macros=("SwpGateTable", "SwpRepresentativeTable"),
    revision_macro="SweepEvidenceRevision",
    audited_eol_files={
        "protocol.json": AuditedEolFile(
            lf_sha256="2a5ba9e46c777225384539a4c453a43aa3298c956b32b022cc5ddeac72ba874c",
            recorded_sha256="64b2c58c3cecb2ea1836d2bf48e23ff83dffb114866bf21e7135b411beaa2b2c",
            recorded_bytes=7924,
            audit_module="modern/experiments/l1a_geometry_sweep_v2/protocol.py",
        )
    },
    posthoc_audit_commit="9e68df21dde5d5238f665ef685e6d457731136c9",
    posthoc_audit_path="modern/experiments/l1a_geometry_sweep_v2/POSTHOC_AUDIT.md",
)

FOUR_CELL = ExperimentSpec(
    key="four-cell-v2",
    experiment_id="four-cell-topology-search-v2",
    experiment_path=Path("modern/experiments/four_cell_topology_search_v2"),
    results_commit="7120e8edcb74c02c1df968c730d1f93b3758b4e1",
    preregistration_commit="d6317910703de91ca6dc25c4d4d855e36cc3b14d",
    classification="PREREGISTERED_PHYSICS_GATED_V2",
    recorded_outcome="preregistered-null",
    macro_prefix="Fcn",
    gate_id="GATE-FOUR-CELL-V2",
    manifest_id="FOUR-CELL-V2-20260902-128-V1",
    artifact_id="TAB-FOUR-CELL-V2",
    artifact_claim_id="CLM-023",
    prose_claim_ids=("CLM-018", "CLM-022", "CLM-024", "CLM-028"),
    section_heading="Preregistered four-cell topology search: null result",
    table_macros=("FcnFailureTable", "FcnCuspTable"),
    revision_macro="FourCellEvidenceRevision",
    audited_eol_files={
        "results/preregistered-protocol.json": AuditedEolFile(
            lf_sha256="5c195119c7a3c3c7e8b2c2d58e2e9836ac0ece6e000e52b0fd86c4718446c1b4",
            recorded_sha256="ec2e9a732b7d0e909ff742ebbbb0215e1102909c148b812306df6f0759f48e49",
            recorded_bytes=10811,
            audit_module="modern/experiments/four_cell_topology_search_v2/audit_sidecar_eol.py",
        )
    },
    posthoc_audit_commit="605be5ceffa407142a2acb11cc040b13eef89b0c",
    posthoc_audit_path="modern/experiments/four_cell_topology_search_v2/POSTHOC_AUDIT.md",
    lineage=(
        "modern/experiments/four_cell_topology_search",
        "modern/experiments/cft_wall_cusp_validation_v1",
        "modern/experiments/cft_wall_cusp_validation_v2",
    ),
)

CHARACTERIZATION = ExperimentSpec(
    key="topology-characterization-v1",
    experiment_id="cft-topology-characterization-v1",
    experiment_path=Path("modern/experiments/cft_topology_characterization_v1"),
    results_commit="3ce6c546194e1d3e943d0b3d0951d03e15e354d9",
    preregistration_commit="af88470b86fd95882ae7fddc48e2860cbfba1219",
    classification="developmental_topology_characterization",
    recorded_outcome="recorded-characterization",
    macro_prefix="Tch",
    gate_id="GATE-TOPOLOGY-CHAR-V1",
    manifest_id="TOPOLOGY-CHAR-V1-20260902-56-V1",
    artifact_id="TAB-TOPOLOGY-CHAR-V1",
    artifact_claim_id="CLM-026",
    prose_claim_ids=("CLM-018", "CLM-025", "CLM-027", "CLM-028"),
    section_heading="Recorded topology characterization: no stable eligible cusp or cell",
    table_macros=("TchNullClassTable", "TchStageTable"),
    revision_macro="CharacterizationEvidenceRevision",
)

EXPERIMENTS: dict[str, ExperimentSpec] = {spec.key: spec for spec in (SWEEP, FOUR_CELL, CHARACTERIZATION)}
BY_EXPERIMENT_ID: dict[str, ExperimentSpec] = {spec.experiment_id: spec for spec in EXPERIMENTS.values()}
COUPLING_V3_COMMIT = "f80a360fd740a30017cdac1874cedbfa2806874a"
COUPLING_V4_COMMIT = "f10d8213117fbafd8c2b69bdc103b6ef7b5d6d8c"
FOUR_CELL_V1_RESULTS_COMMIT = "4afcecfb024cd06e79d0ce8e063fc863ba3f79dc"
WCVAL_V1_RESULTS_COMMIT = "2504175ea845dca1b57fef159961f335a2b546ee"
WCVAL_V2_RESULTS_COMMIT = "7e1246b5b76830fe09afb10ffe076e953a3c2905"


# --------------------------------------------------------------------------- #
# Bundle access with the audited end-of-line rule
# --------------------------------------------------------------------------- #
class Bundle:
    """Read files of one experiment directory, recording the digest of every file used."""

    def __init__(self, repo: Path, spec: ExperimentSpec) -> None:
        self.repo = repo
        self.spec = spec
        self.root = repo / spec.experiment_path
        self.used: dict[str, dict[str, Any]] = {}
        self.tolerated: list[str] = []
        self.docs: dict[str, Any] = {}

    def _path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"path escapes the experiment: {relative}")
        return self.root / pure

    def read(self, relative: str) -> bytes:
        raw = self._path(relative).read_bytes()
        self.used[relative] = {"sha256": sha256_bytes(raw), "bytes": len(raw)}
        return raw

    def verify(self, relative: str, recorded_sha256: str, recorded_bytes: int | None = None) -> bytes:
        """Byte-exact check, or the audited EOL rule for exactly the audited files."""

        raw = self.read(relative)
        actual = sha256_bytes(raw)
        if actual == recorded_sha256:
            if recorded_bytes is not None and len(raw) != recorded_bytes:
                raise ValueError(f"{relative}: size differs from the recorded byte count")
            return raw
        audited = self.spec.audited_eol_files.get(relative)
        crlf = raw.replace(b"\n", b"\r\n")
        if (
            audited is not None
            and b"\r" not in raw
            and actual == audited.lf_sha256
            and sha256_bytes(crlf) == recorded_sha256 == audited.recorded_sha256
            and len(crlf) == audited.recorded_bytes
            and (recorded_bytes is None or recorded_bytes == audited.recorded_bytes)
        ):
            if relative not in self.tolerated:
                self.tolerated.append(relative)
            return raw
        raise ValueError(f"{relative}: SHA-256 differs from the recorded digest")

    def verify_sidecar(self, relative: str) -> str:
        """Check ``<relative>.sha256`` (``"{digest}  {name}\\n"``) and return the digest it attests."""

        raw = self.read(relative)
        sidecar = self.read(relative + ".sha256").decode("ascii")
        digest, name = sidecar.split()
        if name != PurePosixPath(relative).name or not sidecar.endswith("\n"):
            raise ValueError(f"{relative}: sidecar names a different file")
        self.verify(relative, digest)
        return digest

    def load(self, relative: str) -> Any:
        if relative not in self.docs:
            if relative not in self.used:
                self.read(relative)
            self.docs[relative] = load_json_bytes(self._path(relative).read_bytes(), relative)
        return self.docs[relative]

    def bind_committed(self) -> dict[str, Any]:
        head = _git(self.repo, "rev-parse", "HEAD")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", self.spec.results_commit, head],
            cwd=self.repo, check=False, capture_output=True,
        ).returncode == 0
        if not ancestor:
            raise ValueError(f"{self.spec.key}: results commit is not an ancestor of HEAD")
        manifest_rel = (self.spec.experiment_path / "results/manifest.json").as_posix()
        committed_blob = _git(self.repo, "rev-parse", f"{self.spec.results_commit}:{manifest_rel}")
        working_blob = _git(self.repo, "hash-object", "--", manifest_rel)
        if committed_blob != working_blob:
            raise ValueError(f"{self.spec.key}: working-tree results manifest differs from the committed blob")
        results_tree = _git(self.repo, "rev-parse", f"{self.spec.results_commit}:{(self.spec.experiment_path / 'results').as_posix()}")
        head_tree = _git(self.repo, "rev-parse", f"HEAD:{(self.spec.experiment_path / 'results').as_posix()}")
        if results_tree != head_tree:
            raise ValueError(f"{self.spec.key}: results tree changed after the results revision")
        subject = _git(self.repo, "show", "-s", "--format=%s", self.spec.results_commit)
        binding = {
            "results_commit": self.spec.results_commit,
            "results_commit_subject": subject,
            "preregistration_commit": self.spec.preregistration_commit,
            "manifest_git_blob": committed_blob,
            "manifest_path": manifest_rel,
            "results_tree": results_tree,
        }
        if self.spec.posthoc_audit_commit:
            binding["posthoc_audit_commit"] = self.spec.posthoc_audit_commit
            binding["posthoc_audit_path"] = self.spec.posthoc_audit_path
        return binding


def verify_sealed(value: Any, label: str, key: str = "integrity") -> str:
    integrity = value.get(key)
    if not isinstance(integrity, dict) or set(integrity) != {"algorithm", "canonicalization", "payload_sha256"}:
        raise ValueError(f"{label}: {key} declaration is not closed")
    if integrity["algorithm"] != "sha256" or integrity["canonicalization"] != "json-sort-keys-compact-utf8-v1":
        raise ValueError(f"{label}: unsupported {key} declaration")
    payload = {k: v for k, v in value.items() if k != key}
    if canonical_hash(payload) != integrity["payload_sha256"]:
        raise ValueError(f"{label}: canonical payload SHA-256 mismatch")
    return integrity["payload_sha256"]


class Macros:
    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle
        self.prefix = bundle.spec.macro_prefix
        self.items: list[dict[str, Any]] = []
        self.names: set[str] = set()

    def _check(self, name: str) -> None:
        if not name.startswith(self.prefix) or not name.isalpha() or name in self.names:
            raise ValueError(f"macro name {name!r} is invalid or duplicated")

    def add(self, name: str, artifact: str, pointer: str, fmt: str, description: str) -> Any:
        self._check(name)
        raw = resolve_pointer(self.bundle.load(artifact), pointer)
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


def _artifact_claim(spec: ExperimentSpec) -> str:
    return f"\\ArtifactClaim{{{spec.artifact_claim_id}}}{{{spec.artifact_id}}}{{%"


def _table(spec: ExperimentSpec, macro: str, caption: str, label: str, columns: str, header: str, rows: list[str], *, extra: str = "") -> list[str]:
    lines = [f"\\newcommand{{\\{macro}}}{{%", _artifact_claim(spec), "\\begin{table}[ht]", "\\centering"]
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\footnotesize")
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


def _ident(value: str) -> str:
    return format_value("ident", value)


def _p(width_cm: float) -> str:
    """Ragged-right paragraph column of the given width (array package syntax)."""

    return f">{{\\raggedright\\arraybackslash}}p{{{width_cm:g}cm}}"


def _num(value: Any) -> str:
    """Table number: scientific notation for small magnitudes, general format otherwise."""

    number = float(value)
    if number != 0.0 and abs(number) < 1e-3:
        return format_value("sci2", number)
    return f"{number:.4g}"


# --------------------------------------------------------------------------- #
# Experiment 1: L1a geometry sweep v2 (accepted screening)
# --------------------------------------------------------------------------- #
def build_sweep(repo: Path, spec: ExperimentSpec = SWEEP) -> tuple[dict[str, Any], str, dict[str, Any]]:
    b = Bundle(repo, spec)
    binding = b.bind_committed()
    m = Macros(b)
    manifest_sha = b.verify_sidecar("results/manifest.json")
    manifest = b.load("results/manifest.json")
    verify_sealed(manifest, "sweep manifest")
    if manifest["terminal_status"] != "ACCEPTED" or manifest["preregistration_commit_sha"] != spec.preregistration_commit:
        raise ValueError("sweep manifest terminal status or preregistration commit differs")
    if manifest["classification"] != spec.classification:
        raise ValueError("sweep manifest classification differs")
    # Every deterministic file byte-exact, sidecar attested, sealed payload recomputed.
    for entry in manifest["deterministic_files"]:
        digest = b.verify_sidecar(f"results/{entry['path']}")
        if digest != entry["file_sha256"]:
            raise ValueError(f"{entry['path']}: sidecar digest differs from the manifest listing")
        if entry["payload_sha256"] is not None:
            doc = b.load(f"results/{entry['path']}")
            if verify_sealed(doc, entry["path"]) != entry["payload_sha256"]:
                raise ValueError(f"{entry['path']}: payload binding mismatch")
    summary = b.load("results/summary.json")
    raw = b.load("results/raw-results.json")
    lock = b.load("results/execution-lock.json")
    if verify_sealed(summary, "summary") != manifest["summary_payload_sha256"]:
        raise ValueError("summary payload differs from the manifest binding")
    if verify_sealed(raw, "raw results") != manifest["raw_results_payload_sha256"]:
        raise ValueError("raw-results payload differs from the manifest binding")
    # Frozen protocol: LF bytes on disk, CRLF-era digest recorded (POSTHOC_AUDIT.md).
    audited = spec.audited_eol_files["protocol.json"]
    protocol_bytes = b.verify("protocol.json", manifest["protocol_file_sha256"])
    sidecar = b.read("protocol.json.sha256").decode("ascii")
    if sidecar != f"{audited.recorded_sha256}  protocol.json\n":
        raise ValueError("protocol.json.sha256 does not attest the audited recorded digest")
    protocol = b.load("protocol.json")
    if verify_sealed(protocol, "protocol") != manifest["protocol_payload_sha256"]:
        raise ValueError("protocol payload differs from the manifest binding")
    for name, doc in (("raw-results.json", raw), ("summary.json", summary), ("execution-lock.json", lock)):
        if doc["protocol_payload_sha256"] != manifest["protocol_payload_sha256"]:
            raise ValueError(f"{name} binds a different protocol payload")
        if name != "summary.json" and doc["protocol_file_sha256"] != audited.recorded_sha256:
            raise ValueError(f"{name} binds a different protocol file digest")
    if b.tolerated != ["protocol.json"]:
        raise ValueError("tolerated files differ from the audited protocol sidecar")
    del protocol_bytes
    cases = raw["cases"]
    gates = summary["terminal_gates"]
    if len(cases) != summary["evaluated_count"] or any(case["status"] != "success" for case in cases):
        raise ValueError("raw cases differ from the summary counts")
    if summary["failed_count"] != 0 or summary["requested_count"] != summary["evaluated_count"]:
        raise ValueError("sweep summary records failures or unevaluated designs")

    # Identity.
    m.add("SwpClassification", "results/summary.json", "/classification", "ident", "sweep classification string")
    m.add("SwpScreeningLevel", "results/summary.json", "/screening_level", "ident", "screening level label")
    if summary["classification"].split("_")[0] != summary["screening_level"].split("_")[0]:
        raise ValueError("classification and screening level name different model levels")
    m.add_derived("SwpModelLevel", summary["classification"].split("_")[0], "text", "field model level named by the classification", "summary.classification.split('_')[0]", [{"artifact": "results/summary.json", "pointer": "/classification"}])
    m.add("SwpTerminalStatus", "results/summary.json", "/terminal_status", "ident", "terminal acceptance status")
    m.add("SwpRequested", "results/summary.json", "/requested_count", "int", "designs requested")
    m.add("SwpEvaluated", "results/summary.json", "/evaluated_count", "int", "designs evaluated")
    m.add("SwpFailed", "results/summary.json", "/failed_count", "int", "designs failed")
    m.add("SwpNondominated", "results/summary.json", "/nondominated_count", "int", "non-dominated designs")
    m.add("SwpUniqueRepresentatives", "results/summary.json", "/unique_representative_count", "int", "unique representative designs")
    m.add_derived("SwpRoleCount", len(summary["representative_roles"]), "int", "representative roles", "len(summary.representative_roles)", [{"artifact": "results/summary.json", "pointer": "/representative_roles"}])
    m.add("SwpMaxExecutions", "protocol.json", "/execution/maximum_executions", "int", "maximum executions allowed by the protocol")
    m.add("SwpSeed", "protocol.json", "/sampling/seed", "int", "sampling seed")
    m.add("SwpSamplingAlgorithm", "protocol.json", "/sampling/algorithm", "text", "sampling algorithm")
    m.add_derived("SwpDesignVariableCount", len(protocol["sampling"]["variables"]), "int", "design variables", "len(protocol.sampling.variables)", [{"artifact": "protocol.json", "pointer": "/sampling/variables"}])
    m.add_derived("SwpObjectiveCount", len(protocol["objectives"]), "int", "preregistered objectives", "len(protocol.objectives)", [{"artifact": "protocol.json", "pointer": "/objectives"}])
    m.add_derived("SwpParityCaseCount", len(protocol["execution"]["parity_case_indices"]), "int", "CPU parity cases", "len(protocol.execution.parity_case_indices)", [{"artifact": "protocol.json", "pointer": "/execution/parity_case_indices"}])
    m.add("SwpGridRadial", "protocol.json", "/field/domain/radial_intervals", "int", "radial intervals")
    m.add("SwpGridAxial", "protocol.json", "/field/domain/axial_intervals", "int", "axial intervals")
    m.add("SwpDomainRadiusMm", "protocol.json", "/field/domain/radius_m", "mm1", "domain radius (mm)")
    m.add("SwpDomainZMinMm", "protocol.json", "/field/domain/z_min_m", "mm1", "domain z minimum (mm)")
    m.add("SwpDomainZMaxMm", "protocol.json", "/field/domain/z_max_m", "mm1", "domain z maximum (mm)")
    m.add("SwpSolverRelTol", "protocol.json", "/field/solver/relative_tolerance", "sci1", "solver relative tolerance")
    m.add("SwpPreviewAuthoritative", "protocol.json", "/field/preview/authoritative", "bool", "equivalent-current preview authoritative flag")
    m.add("SwpPreviewDescription", "protocol.json", "/field/preview/description", "text", "field source description")
    m.add("SwpStageMapping", "protocol.json", "/geometry/stage_count_mapping", "text", "stage count mapping")
    m.add("SwpMirrorRatioDefinition", "protocol.json", "/qoi_policy/mirror_ratio", "text", "mirror ratio QoI definition")
    m.add("SwpTopologyClaim", "protocol.json", "/qoi_policy/topology_claim", "text", "topology claim limit of the QoI policy")
    m.add("SwpClaimLimits", "protocol.json", "/claim_limits", "list_sentences", "protocol claim limits")
    m.add("SwpTimingPolicy", "protocol.json", "/execution/timing_policy", "text", "timing policy")
    stage_counts = Counter(case["derived_geometry"]["stage_count"] for case in cases)
    m.add_derived("SwpStageMin", min(stage_counts), "int", "minimum stage count", "min over cases of derived_geometry.stage_count", [{"artifact": "results/raw-results.json", "pointer": "/cases"}])
    m.add_derived("SwpStageMax", max(stage_counts), "int", "maximum stage count", "max over cases of derived_geometry.stage_count", [{"artifact": "results/raw-results.json", "pointer": "/cases"}])

    # Gates.
    m.add_derived("SwpGateCount", len(gates), "int", "terminal gates", "len(summary.terminal_gates)", [{"artifact": "results/summary.json", "pointer": "/terminal_gates"}])
    m.add_derived("SwpGatesPassed", sum(1 for g in gates if g["passed"] is True), "int", "terminal gates passed", "count(summary.terminal_gates.passed == true)", [{"artifact": "results/summary.json", "pointer": "/terminal_gates"}])
    m.add_derived("SwpGateFailures", sum(g["failure_count"] for g in gates), "int", "gate failures summed", "sum(summary.terminal_gates.failure_count)", [{"artifact": "results/summary.json", "pointer": "/terminal_gates"}])
    gate_rows: list[str] = []
    expected_gate_ids = ("boundary", "residual", "cpu_cuda_parity", "flux_identity", "source_representation", "topology_confidence", "manufacturability")
    if tuple(g["gate_id"] for g in gates) != expected_gate_ids:
        raise ValueError("terminal gate order differs from the preregistered protocol")
    if [g["definition"] for g in gates] != protocol["terminal_acceptance"]["gates"]:
        raise ValueError("terminal gate definitions differ from the frozen protocol")
    for gate in gates:
        definition = gate["definition"]
        if "limit" in definition:
            limit = _num(definition["limit"])
            observed = _num(gate["observed"])
        else:
            limit = ", ".join(f"{k} {_num(v)}" for k, v in sorted(definition["limits"].items()))
            observed = ", ".join(f"{k} {_num(v)}" for k, v in sorted(gate["observed"].items()))
        comparator = definition["comparator"].replace("<=", "$\\le$").replace(">=", "$\\ge$")
        gate_rows.append(
            f"\\texttt{{{_ident(gate['gate_id'])}}} & \\texttt{{{_ident(definition['metric'])}}} & "
            f"{definition['aggregation']} {comparator} {limit} & {observed} & {gate['failure_count']} & "
            f"{'PASS' if gate['passed'] is True else 'FAIL'}\\\\"
        )
    m.add("SwpBoundaryLimit", "protocol.json", "/terminal_acceptance/gates/0/limit", "g", "boundary gate limit")
    m.add("SwpBoundaryObserved", "results/summary.json", "/terminal_gates/0/observed", "sci2", "boundary gate observed maximum")
    m.add("SwpResidualLimit", "protocol.json", "/terminal_acceptance/gates/1/limit", "sci1", "residual gate limit")
    m.add("SwpResidualObserved", "results/summary.json", "/terminal_gates/1/observed", "sci2", "residual gate observed maximum")
    m.add("SwpParityPsiObserved", "results/summary.json", "/terminal_gates/2/observed/psi", "sci2", "CPU/CUDA parity observed maximum (psi)")
    m.add("SwpParityBrObserved", "results/summary.json", "/terminal_gates/2/observed/br", "sci2", "CPU/CUDA parity observed maximum (B_r)")
    m.add("SwpParityBzObserved", "results/summary.json", "/terminal_gates/2/observed/bz", "sci2", "CPU/CUDA parity observed maximum (B_z)")
    m.add("SwpParityPsiLimit", "protocol.json", "/terminal_acceptance/gates/2/limits/psi", "sci1", "CPU/CUDA parity limit (psi)")
    m.add("SwpParityBLimit", "protocol.json", "/terminal_acceptance/gates/2/limits/br", "sci1", "CPU/CUDA parity limit (B components)")
    m.add("SwpFluxIdentityObserved", "results/summary.json", "/terminal_gates/3/observed", "sci2", "flux identity observed maximum (T/m)")
    m.add("SwpSourceErrorObserved", "results/summary.json", "/terminal_gates/4/observed", "sci2", "source representation observed maximum")
    m.add("SwpTopologyConfidenceLimit", "protocol.json", "/terminal_acceptance/gates/5/limit", "g", "topology confidence gate limit")
    m.add("SwpTopologyConfidenceObserved", "results/summary.json", "/terminal_gates/5/observed", "fixed3", "topology confidence observed minimum")
    m.add("SwpManufacturabilityObservedMm", "results/summary.json", "/terminal_gates/6/observed", "mm2", "manufacturability margin observed minimum (mm)")
    m.add("SwpZeroFailuresRequired", "protocol.json", "/terminal_acceptance/requires_zero_case_failures", "bool", "zero case failures required")

    # QoI ranges.
    ranges = summary["qoi_ranges"]
    m.add("SwpAxisCuspMin", "results/summary.json", "/qoi_ranges/axis_cusp_count/0", "g", "minimum axis cusp count")
    m.add("SwpAxisCuspMax", "results/summary.json", "/qoi_ranges/axis_cusp_count/1", "g", "maximum axis cusp count")
    m.add("SwpAxisNullMin", "results/summary.json", "/qoi_ranges/axis_null_count/0", "g", "minimum axis null count")
    m.add("SwpAxisNullMax", "results/summary.json", "/qoi_ranges/axis_null_count/1", "g", "maximum axis null count")
    m.add("SwpFieldPeakMin", "results/summary.json", "/qoi_ranges/field_peak_t/0", "fixed3", "minimum field peak (T)")
    m.add("SwpFieldPeakMax", "results/summary.json", "/qoi_ranges/field_peak_t/1", "fixed3", "maximum field peak (T)")
    m.add("SwpMinMirrorMin", "results/summary.json", "/qoi_ranges/minimum_mirror_ratio/0", "fixed2", "smallest per-design minimum mirror ratio")
    m.add("SwpMinMirrorMax", "results/summary.json", "/qoi_ranges/minimum_mirror_ratio/1", "fixed2", "largest per-design minimum mirror ratio")
    m.add("SwpMaxMirrorMin", "results/summary.json", "/qoi_ranges/maximum_mirror_ratio/0", "fixed2", "smallest per-design maximum mirror ratio")
    m.add("SwpMaxMirrorMax", "results/summary.json", "/qoi_ranges/maximum_mirror_ratio/1", "fixed0", "largest per-design maximum mirror ratio")
    m.add("SwpTopologyConfidenceMin", "results/summary.json", "/qoi_ranges/topology_confidence/0", "fixed3", "minimum topology confidence")
    m.add("SwpTopologyConfidenceMax", "results/summary.json", "/qoi_ranges/topology_confidence/1", "fixed3", "maximum topology confidence")
    m.add("SwpFieldEnergyMin", "results/summary.json", "/qoi_ranges/field_energy_j/0", "fixed4", "minimum field energy (J)")
    m.add("SwpFieldEnergyMax", "results/summary.json", "/qoi_ranges/field_energy_j/1", "fixed3", "maximum field energy (J)")
    del ranges
    cusp_hist = Counter(int(case["qois"]["axis_cusp_count"]) for case in cases)
    null_hist = Counter(int(case["qois"]["axis_null_count"]) for case in cases)
    if set(cusp_hist) != {3, 4, 5} or any(null_hist[k + 1] != v for k, v in cusp_hist.items()):
        raise ValueError("axis cusp/null histograms differ from the accepted result")
    if any(len(case["qois"]["mirror_ratios"]) != int(case["qois"]["axis_cusp_count"]) - 1 for case in cases):
        raise ValueError("mirror ratio count is not cusp count minus one")
    if any(stage_counts[k] != v for k, v in cusp_hist.items()):
        raise ValueError("axis cusp count does not equal the stage count on every design")
    hist_inputs = [{"artifact": "results/raw-results.json", "pointer": "/cases"}]
    m.add_derived("SwpCuspThreeDesigns", cusp_hist[3], "int", "designs with three axis cusps", "count(cases.qois.axis_cusp_count == 3)", hist_inputs)
    m.add_derived("SwpCuspFourDesigns", cusp_hist[4], "int", "designs with four axis cusps", "count(cases.qois.axis_cusp_count == 4)", hist_inputs)
    m.add_derived("SwpCuspFiveDesigns", cusp_hist[5], "int", "designs with five axis cusps", "count(cases.qois.axis_cusp_count == 5)", hist_inputs)
    m.add_derived("SwpCuspLow", 3, "int", "lowest axis cusp count observed", "min(cases.qois.axis_cusp_count)", hist_inputs)
    m.add_derived("SwpCuspMid", 4, "int", "middle axis cusp count observed", "the one value strictly between min and max of cases.qois.axis_cusp_count", hist_inputs)
    m.add_derived("SwpCuspHigh", 5, "int", "highest axis cusp count observed", "max(cases.qois.axis_cusp_count)", hist_inputs)
    positions = [z for case in cases for z in case["qois"]["axis_cusp_positions_m"]]
    if len(positions) != sum(int(case["qois"]["axis_cusp_count"]) for case in cases):
        raise ValueError("axis cusp positions do not match the counts")
    m.add_derived("SwpAxisCuspTotal", len(positions), "int", "axis cusps recorded over all designs", "sum(cases.qois.axis_cusp_count)", hist_inputs)
    m.add_derived("SwpAxisCuspZMinMm", min(positions), "mm2", "smallest axis cusp position (mm)", "min over cases of qois.axis_cusp_positions_m", hist_inputs)
    m.add_derived("SwpAxisCuspZMaxMm", max(positions), "mm2", "largest axis cusp position (mm)", "max over cases of qois.axis_cusp_positions_m", hist_inputs)
    statuses = Counter(case["qois"]["topology_status"] for case in cases)
    if set(statuses) != {"resolved_axis_nulls"}:
        raise ValueError("topology status is not resolved on every design")
    m.add_derived("SwpResolvedDesigns", statuses["resolved_axis_nulls"], "int", "designs with resolved axis nulls", "count(cases.qois.topology_status == resolved_axis_nulls)", hist_inputs)
    m.add_derived("SwpTopologyStatus", "resolved_axis_nulls", "ident", "the single topology status observed", "unique(cases.qois.topology_status)", hist_inputs)

    # Representatives.
    roles: dict[str, list[str]] = {}
    for item in summary["representative_roles"]:
        roles.setdefault(item["case_id"], []).append(item["role"])
    by_id = {case["case_id"]: case for case in cases}
    rep_ids = sorted(roles)
    if len(rep_ids) != summary["unique_representative_count"] or set(rep_ids) != set(summary["nondominated_case_ids"]) & set(rep_ids):
        raise ValueError("representatives are not unique members of the non-dominated set")
    artifacts_by_id = {item["case_id"]: item for item in manifest["representative_artifacts"]}
    if set(artifacts_by_id) != set(rep_ids):
        raise ValueError("representative artifacts differ from the representative roles")
    rep_rows: list[str] = []
    for case_id in rep_ids:
        case = by_id[case_id]
        q = case["qois"]
        geometry = b.load(f"results/{artifacts_by_id[case_id]['geometry']['path']}")
        if geometry.get("schema_version") != protocol["geometry"]["schema_version"]:
            raise ValueError(f"{case_id}: geometry schema differs from the protocol")
        rep_rows.append(
            f"{_tex_escape(', '.join(sorted(roles[case_id])))} & \\texttt{{{_ident(case_id)}}} & {case['derived_geometry']['stage_count']} & "
            f"{int(q['axis_cusp_count'])} & {format_value('list_mm2', q['axis_cusp_positions_m'])} & "
            f"{format_value('list_fixed1', q['mirror_ratios'])} & {format_value('fixed3', q['field_peak_t'])}\\\\"
        )
        short = case_id.split("-")[3]
        word = _digits_to_words(short)
        m.add_derived(f"SwpRep{word}Id", case_id, "ident", f"representative {short} case id", "summary.representative_roles[*].case_id", [{"artifact": "results/summary.json", "pointer": "/representative_roles"}])
        index = next(i for i, c in enumerate(cases) if c["case_id"] == case_id)
        m.add(f"SwpRep{word}Cusps", "results/raw-results.json", f"/cases/{index}/qois/axis_cusp_count", "g", f"representative {short} axis cusp count")
        m.add(f"SwpRep{word}CuspZMm", "results/raw-results.json", f"/cases/{index}/qois/axis_cusp_positions_m", "list_mm2", f"representative {short} axis cusp positions (mm)")
        m.add(f"SwpRep{word}Mirror", "results/raw-results.json", f"/cases/{index}/qois/mirror_ratios", "list_fixed1", f"representative {short} per-cell mirror ratios")
        artifact_index = [item["case_id"] for item in manifest["representative_artifacts"]].index(case_id)
        if sorted(manifest["representative_artifacts"][artifact_index]["roles"]) != sorted(roles[case_id]):
            raise ValueError(f"{case_id}: manifest roles differ from the summary roles")
        m.add(f"SwpRep{word}Roles", "results/manifest.json", f"/representative_artifacts/{artifact_index}/roles", "list_text", f"representative {short} roles")
    m.add("SwpStrongestMirrorId", "results/summary.json", "/representative_roles/1/case_id", "ident", "strongest-mirror representative")
    if summary["representative_roles"][1]["role"] != "strongest-mirror":
        raise ValueError("representative role order differs from the accepted summary")

    # Environment and binding.
    m.add("SwpGpuName", "results/summary.json", "/environment/gpu/nvidia_smi_name", "text", "GPU name")
    m.add("SwpWarpVersion", "results/summary.json", "/environment/warp/version", "text", "Warp version")
    m.add("SwpPythonVersion", "results/summary.json", "/environment/python", "text", "Python version")
    m.add("SwpFloatingReplayPolicy", "results/summary.json", "/environment/floating_replay_policy", "text", "floating replay policy")
    m.add("SwpPreregCommit", "results/summary.json", "/preregistration_commit_sha", "sha_short", "preregistration commit prefix")
    m.add("SwpProtocolPayloadSha", "results/summary.json", "/protocol_payload_sha256", "sha_short", "protocol payload hash prefix")
    m.add_derived("SwpResultsCommit", spec.results_commit, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "results/manifest.json", "pointer": ""}])
    m.add_derived("SwpManifestSha", manifest_sha, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "results/manifest.json", "pointer": ""}])
    m.add_derived("SwpAuditCommit", spec.posthoc_audit_commit, "sha_short", "post-hoc audit commit prefix", "commit that added POSTHOC_AUDIT.md (bound in the typed manifest)", [{"artifact": "protocol.json.sha256", "pointer": ""}])
    m.add_derived("SwpToleratedEolFiles", len(b.tolerated), "int", "files whose recorded digest differs from the checkout by end-of-line bytes only", "count of verified files accepted through the audited EOL rule", [{"artifact": "protocol.json.sha256", "pointer": ""}])
    m.add_derived("SwpProtocolRecordedSha", audited.recorded_sha256, "sha_short", "recorded (CRLF-era) protocol file digest prefix", "protocol.json.sha256 as frozen at the preregistration commit", [{"artifact": "protocol.json.sha256", "pointer": ""}])
    m.add_derived("SwpProtocolLfSha", audited.lf_sha256, "sha_short", "LF protocol file digest prefix", "sha256(protocol.json) on an eol=lf checkout", [{"artifact": "protocol.json", "pointer": ""}])

    tex_lines = _header(spec, manifest_sha)
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    tex_lines += _table(
        spec, "SwpGateTable",
        "Seven preregistered terminal gates of the L1a geometry sweep v2 as sealed in \\texttt{summary.json}: "
        "aggregate observed over all \\SwpEvaluated{} designs, per-design failures and outcome.",
        "tab:l1a-sweep-v2-gates", f"{_p(3.0)}{_p(2.8)}{_p(2.8)}{_p(3.0)}rl",
        "gate & metric & rule & observed & fail. & result\\\\", gate_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    tex_lines += _table(
        spec, "SwpRepresentativeTable",
        "Representative designs selected by the preregistered roles from the non-dominated set: stage count, "
        "axis cusp count and positions, per-cell centreline mirror ratios (field-only screening QoIs, not confinement claims) "
        "and peak $|B|$.",
        "tab:l1a-sweep-v2-representatives", f"{_p(2.7)}{_p(2.6)}rr{_p(3.1)}{_p(2.5)}r",
        "roles & case & stages & cusps & axis cusp $z$ (mm) & mirror ratios & $|B|_{\\max}$ (T)\\\\", rep_rows,
        extra="\\setlength{\\tabcolsep}{3pt}",
    )
    tex = "\n".join(tex_lines) + "\n"
    tables = {
        "SwpGateTable": {"rows": len(gate_rows), "source": "results/summary.json#/terminal_gates"},
        "SwpRepresentativeTable": {"rows": len(rep_rows), "source": "results/raw-results.json#/cases (representative ids from summary.json)"},
    }
    return _evidence(repo, spec, b, binding, m, tex, tables, manifest_sha), tex, {}


def _digits_to_words(short: str) -> str:
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(words[ch] for ch in short)


def _camel(identifier: str) -> str:
    return "".join(part.capitalize() for part in identifier.replace("-", "_").split("_"))


# --------------------------------------------------------------------------- #
# Experiment 2: four-cell topology search v2 (preregistered null)
# --------------------------------------------------------------------------- #
def build_four_cell(repo: Path, spec: ExperimentSpec = FOUR_CELL) -> tuple[dict[str, Any], str, dict[str, Any]]:
    b = Bundle(repo, spec)
    binding = b.bind_committed()
    m = Macros(b)
    manifest_sha = b.verify_sidecar("results/manifest.json")
    manifest = b.load("results/manifest.json")
    verify_sealed(manifest, "four-cell manifest")
    if (
        manifest["experiment_id"] != spec.experiment_id
        or manifest["preregistration_commit_sha"] != spec.preregistration_commit
        or manifest["accepted_coupling_v3_commit_sha"] != COUPLING_V3_COMMIT
        or manifest["single_execution"] is not True
    ):
        raise ValueError("four-cell manifest identity differs")
    audited = spec.audited_eol_files["results/preregistered-protocol.json"]
    if manifest["protocol_sha256"] != audited.recorded_sha256:
        raise ValueError("four-cell manifest binds a different protocol digest")
    for entry in manifest["artifacts"]:
        relative = f"results/{entry['path']}"
        b.verify(relative, entry["sha256"], entry["bytes"])
        sidecar = b.read(relative + ".sha256").decode("ascii")
        if sidecar != f"{entry['sha256']}  {PurePosixPath(entry['path']).name}\n":
            raise ValueError(f"{relative}: sidecar differs from the manifest listing")
    if b.tolerated != ["results/preregistered-protocol.json"]:
        raise ValueError("tolerated files differ from the audited protocol copy")
    protocol_copy = b.read("results/preregistered-protocol.json")
    if b.read("protocol.json") != protocol_copy:
        raise ValueError("frozen protocol differs from the results copy")
    protocol = b.load("protocol.json")
    dataset = b.load("results/dataset.json")
    lock = b.load("results/execution-lock.json")
    runtime = b.load("results/runtime.json")
    verify_sealed(dataset, "four-cell dataset")
    verify_sealed(runtime, "four-cell runtime")
    if dataset["protocol_sha256"] != audited.recorded_sha256 or lock["protocol_sha256"] != audited.recorded_sha256:
        raise ValueError("dataset or lock binds a different protocol digest")
    if dataset["protocol_payload_sha256"] != canonical_hash(protocol):
        raise ValueError("protocol canonical payload differs from the dataset binding")
    if dataset["summary"] != manifest["summary"] or dataset["classification"] != spec.classification:
        raise ValueError("dataset summary or classification differs from the manifest")
    if dataset["preregistration_commit_sha"] != spec.preregistration_commit or lock["preregistration_commit_sha"] != spec.preregistration_commit:
        raise ValueError("dataset or lock preregistration commit differs")
    if dataset["claim_boundary"] != protocol["claim_boundary"]:
        raise ValueError("dataset claim boundary differs from the protocol")
    cases = dataset["cases"]
    summary = dataset["summary"]
    if len(cases) != summary["evaluated_count"] or summary["stable_count"] != 0:
        raise ValueError("four-cell cases differ from the recorded null summary")
    if any(case["stable"] or case["topology"]["exact_count"] or case["coupled"] or case["adiabatic"] for case in cases):
        raise ValueError("a candidate passed a gate the summary records as failed")
    failure_counter = Counter(code for case in cases for code in case["failures"])
    if {k: v for k, v in summary["failure_counts"].items() if v} != dict(failure_counter):
        raise ValueError("failure taxonomy counts differ from the per-candidate failures")
    if set(summary["failure_counts"]) != set(protocol["failure_taxonomy"]):
        raise ValueError("failure taxonomy differs from the protocol")

    # Identity and protocol.
    m.add("FcnClassification", "results/dataset.json", "/classification", "ident", "search classification string")
    m.add("FcnFieldModelLevel", "protocol.json", "/accepted_dependency_baseline/field_model_level", "text", "field model level of the accepted baseline")
    m.add("FcnSingleExecution", "results/manifest.json", "/single_execution", "bool", "single execution flag")
    m.add("FcnDeclared", "results/dataset.json", "/summary/declared_candidate_count", "int", "declared candidates")
    m.add("FcnEvaluated", "results/dataset.json", "/summary/evaluated_count", "int", "candidates evaluated")
    m.add("FcnThreeMapAccepted", "results/dataset.json", "/summary/three_map_accepted_count", "int", "candidates whose three maps passed every field gate")
    m.add("FcnStable", "results/dataset.json", "/summary/stable_count", "int", "candidates with a stable four-cusp topology")
    m.add("FcnAdiabatic", "results/dataset.json", "/summary/adiabatic_count", "int", "candidates passing adiabaticity")
    m.add("FcnCoupled", "results/dataset.json", "/summary/coupled_count", "int", "candidates coupled into the plasma network")
    m.add("FcnUniqueStates", "results/dataset.json", "/summary/unique_state_count", "int", "unique plasma states published")
    m.add("FcnPerformancePublications", "results/dataset.json", "/summary/power_or_performance_publication_count", "int", "power or performance publications")
    m.add("FcnResidualRootScenarios", "results/dataset.json", "/summary/plasma_residual_root_scenario_count", "int", "plasma residual-root scenarios")
    m.add("FcnTopologyCountFailures", "results/dataset.json", "/summary/failure_counts/TOPOLOGY_COUNT", "int", "candidates failing TOPOLOGY_COUNT")
    m.add("FcnTopologyUnstableFailures", "results/dataset.json", "/summary/failure_counts/TOPOLOGY_UNSTABLE", "int", "candidates failing TOPOLOGY_UNSTABLE")
    m.add("FcnFieldFailures", "results/dataset.json", "/summary/failure_counts/FIELD_PRIMARY_INVALID", "int", "candidates failing FIELD_PRIMARY_INVALID")
    evaluation_codes = {"GEOMETRY_INVALID", "FIELD_PRIMARY_INVALID", "FIELD_DOWNSAMPLED_INVALID", "FIELD_ENLARGED_INVALID", "FIELD_DOMAIN_UNSTABLE"}
    if not evaluation_codes <= set(summary["failure_counts"]):
        raise ValueError("evaluation failure codes are missing from the taxonomy")
    m.add_derived(
        "FcnEvaluationFailures", sum(1 for case in cases if evaluation_codes & set(case["failures"])), "int",
        "candidates whose geometry or any field map failed to evaluate",
        "count(cases with any of GEOMETRY_INVALID, FIELD_PRIMARY_INVALID, FIELD_DOWNSAMPLED_INVALID, FIELD_ENLARGED_INVALID, FIELD_DOMAIN_UNSTABLE in failures)",
        [{"artifact": "results/dataset.json", "pointer": "/cases"}],
    )
    m.add_derived("FcnFailureCodeCount", len(summary["failure_counts"]), "int", "failure taxonomy codes", "len(summary.failure_counts)", [{"artifact": "results/dataset.json", "pointer": "/summary/failure_counts"}])
    m.add_derived("FcnNonzeroFailureCodes", sum(1 for v in summary["failure_counts"].values() if v), "int", "failure codes with a non-zero count", "count(summary.failure_counts > 0)", [{"artifact": "results/dataset.json", "pointer": "/summary/failure_counts"}])
    m.add("FcnRequiredCells", "protocol.json", "/topology/required_stable_cell_count", "int", "required stable cell count")
    m.add("FcnRequiredRegisteredCusps", "protocol.json", "/topology/required_geometry_registered_cusp_count", "int", "required geometry-registered cusps")
    m.add("FcnMaxCuspShiftMm", "protocol.json", "/topology/maximum_cross_map_cusp_shift_m", "mm1", "maximum cross-map cusp shift (mm)")
    m.add("FcnSlotErrorFraction", "protocol.json", "/topology/maximum_geometry_slot_error_pitch_fraction", "g", "maximum geometry slot error (pitch fraction)")
    m.add("FcnEndpointExclusionCells", "protocol.json", "/topology/endpoint_exclusion_cells", "int", "endpoint exclusion cells")
    m.add("FcnFluxQuantiles", "protocol.json", "/topology/flux_quantiles", "list_g", "flux quantiles")
    m.add("FcnSaddlePolicy", "protocol.json", "/topology/saddle_tie_policy", "ident", "saddle tie policy")
    m.add("FcnCuspSlots", "protocol.json", "/topology/geometry_cusp_slots", "text", "geometry cusp slot rule")
    m.add("FcnBoundaryNullPolicy", "protocol.json", "/topology/boundary_null_policy", "text", "boundary null policy")
    m.add("FcnCountFailureDefinition", "protocol.json", "/failure_taxonomy/TOPOLOGY_COUNT", "text", "TOPOLOGY_COUNT definition")
    m.add("FcnUnstableFailureDefinition", "protocol.json", "/failure_taxonomy/TOPOLOGY_UNSTABLE", "text", "TOPOLOGY_UNSTABLE definition")
    m.add("FcnZeroPassPolicy", "protocol.json", "/execution/zero_pass_policy", "text", "zero-pass policy")
    m.add("FcnMapCount", "protocol.json", "/maps/independent_solves_per_candidate", "int", "independent maps per candidate")
    m.add("FcnPrimaryRadial", "protocol.json", "/maps/roles/primary/radial_intervals", "int", "primary radial intervals")
    m.add("FcnPrimaryAxial", "protocol.json", "/maps/roles/primary/axial_intervals", "int", "primary axial intervals")
    m.add("FcnDownsampledRadial", "protocol.json", "/maps/roles/downsampled/radial_intervals", "int", "downsampled radial intervals")
    m.add("FcnDownsampledAxial", "protocol.json", "/maps/roles/downsampled/axial_intervals", "int", "downsampled axial intervals")
    m.add("FcnEnlargedRadial", "protocol.json", "/maps/roles/enlarged_domain/radial_intervals", "int", "enlarged-domain radial intervals")
    m.add("FcnEnlargedAxial", "protocol.json", "/maps/roles/enlarged_domain/axial_intervals", "int", "enlarged-domain axial intervals")
    m.add("FcnEnlargedScale", "protocol.json", "/maps/roles/enlarged_domain/domain_scale", "g", "enlarged-domain scale factor")
    m.add("FcnSamplingAlgorithm", "protocol.json", "/sampling/algorithm", "text", "sampling algorithm")
    m.add("FcnCandidateCount", "protocol.json", "/sampling/candidate_count", "int", "preregistered candidate count")
    m.add("FcnStageCount", "protocol.json", "/sampling/stage_count", "int", "magnet stages per candidate")
    m.add_derived("FcnDesignVariableCount", len(protocol["sampling"]["variables"]), "int", "design variables", "len(protocol.sampling.variables)", [{"artifact": "protocol.json", "pointer": "/sampling/variables"}])
    m.add("FcnElectronEnergyEv", "protocol.json", "/coupling_v3/electron_distribution/kinetic_energy_ev", "g", "screening electron energy (eV)")
    m.add("FcnGyroLimit", "protocol.json", "/coupling_v3/electron_distribution/maximum_gyroradius_to_scale_length", "g", "adiabaticity limit rho_e/L_B")
    m.add("FcnDefaultPublication", "protocol.json", "/claim_boundary/default_publication", "text", "default publication policy")
    m.add("FcnSameZProxyAllowed", "protocol.json", "/claim_boundary/same_z_proxy_allowed", "bool", "same-z proxy allowed")
    m.add("FcnCouplingCommit", "results/dataset.json", "/accepted_coupling_v3_commit_sha", "sha_short", "accepted coupling v3 commit prefix")
    m.add("FcnPreregCommit", "results/dataset.json", "/preregistration_commit_sha", "sha_short", "preregistration commit prefix")
    m.add("FcnProtocolPayloadSha", "results/dataset.json", "/protocol_payload_sha256", "sha_short", "protocol canonical payload hash prefix")
    m.add("FcnGpuName", "results/runtime.json", "/gpu_name", "text", "GPU name")
    m.add("FcnWarpVersion", "results/runtime.json", "/warp_version", "text", "Warp version")

    # Interior cusp statistics by map role.
    role_order = ("primary", "downsampled", "enlarged_domain")
    cusp_rows: list[str] = []
    grid = {"primary": ("FcnPrimaryRadial", "FcnPrimaryAxial"), "downsampled": ("FcnDownsampledRadial", "FcnDownsampledAxial"), "enlarged_domain": ("FcnEnlargedRadial", "FcnEnlargedAxial")}
    required = protocol["topology"]["required_stable_cell_count"]
    case_inputs = [{"artifact": "results/dataset.json", "pointer": "/cases"}]
    for role in role_order:
        counts = [len(case["maps"][role]["interior_cusp_z_m"]) for case in cases]
        if any(case["topology"]["count_by_role"][role] != len(case["maps"][role]["interior_cusp_z_m"]) for case in cases):
            raise ValueError(f"{role}: count_by_role differs from the recorded cusp positions")
        exact = sum(1 for c in counts if c == required)
        camel = _camel(role)
        m.add_derived(f"Fcn{camel}CuspMin", min(counts), "int", f"minimum interior cusps per candidate ({role})", f"min over cases of len(maps.{role}.interior_cusp_z_m)", case_inputs)
        m.add_derived(f"Fcn{camel}CuspMax", max(counts), "int", f"maximum interior cusps per candidate ({role})", f"max over cases of len(maps.{role}.interior_cusp_z_m)", case_inputs)
        m.add_derived(f"Fcn{camel}CuspTotal", sum(counts), "int", f"interior cusps recorded over all candidates ({role})", f"sum over cases of len(maps.{role}.interior_cusp_z_m)", case_inputs)
        m.add_derived(f"Fcn{camel}ExactCount", exact, "int", f"candidates with exactly the required cusp count ({role})", f"count over cases of len(maps.{role}.interior_cusp_z_m) == protocol.topology.required_stable_cell_count", case_inputs)
        radial, axial = grid[role]
        cusp_rows.append(
            f"{_tex_escape(role)} & \\{radial}$\\times$\\{axial} & {min(counts)} & {max(counts)} & {sum(counts)} & {exact}\\\\"
        )
    m.add_derived("FcnAnyExactCount", sum(1 for case in cases if case["topology"]["exact_count"]), "int", "candidates with exactly four cusps on all three maps", "count(cases.topology.exact_count == true)", case_inputs)
    m.add_derived("FcnGeometryRegistered", sum(1 for case in cases if case["topology"]["geometry_registered"]), "int", "candidates whose cusps registered to the geometry slots", "count(cases.topology.geometry_registered == true)", case_inputs)
    m.add_derived("FcnAllFieldGates", sum(1 for case in cases if case["topology"]["all_field_gates"]), "int", "candidates whose three maps passed every field gate", "count(cases.topology.all_field_gates == true)", case_inputs)
    if m.items[-1]["raw"] != summary["three_map_accepted_count"]:
        raise ValueError("field-gate count differs from the summary")
    m.add_derived("FcnEndpointExclusionPassed", sum(1 for case in cases if case["topology"]["endpoint_exclusion"]), "int", "candidates with no cusp in the endpoint exclusion region", "count(cases.topology.endpoint_exclusion == true)", case_inputs)
    m.add_derived("FcnDomainStable", sum(1 for case in cases if case["topology"]["enlarged_domain_boundary_comparison"]), "int", "candidates passing the enlarged-domain boundary comparison", "count(cases.topology.enlarged_domain_boundary_comparison == true)", case_inputs)
    boundary_nulls = [case["maps"]["primary"]["boundary_null_count"] for case in cases]
    m.add_derived("FcnBoundaryNullMin", min(boundary_nulls), "int", "minimum finite-box boundary nulls (primary)", "min(cases.maps.primary.boundary_null_count)", case_inputs)
    m.add_derived("FcnBoundaryNullMax", max(boundary_nulls), "int", "maximum finite-box boundary nulls (primary)", "max(cases.maps.primary.boundary_null_count)", case_inputs)
    peaks = [case["maps"]["primary"]["quality"]["field_peak_t"] for case in cases]
    m.add_derived("FcnFieldPeakMin", min(peaks), "fixed3", "minimum primary field peak (T)", "min(cases.maps.primary.quality.field_peak_t)", case_inputs)
    m.add_derived("FcnFieldPeakMax", max(peaks), "fixed3", "maximum primary field peak (T)", "max(cases.maps.primary.quality.field_peak_t)", case_inputs)

    # Failure taxonomy table.
    codes = sorted(summary["failure_counts"])
    failure_rows: list[str] = []
    half = (len(codes) + 1) // 2
    for left, right in zip(codes[:half], codes[half:] + [None] * (2 * half - len(codes))):
        left_cell = f"\\texttt{{{_ident(left)}}} & {summary['failure_counts'][left]}"
        right_cell = f"\\texttt{{{_ident(right)}}} & {summary['failure_counts'][right]}" if right else " & "
        failure_rows.append(f"{left_cell} & {right_cell}\\\\")

    # GPU replay as recorded.
    replay = dataset["gpu_replay"]
    if len(replay) != len(protocol["replay"]["gpu_replay_candidate_ids"]) or sum(1 for r in replay if r["passed"]) != summary["gpu_replay_pass_count"]:
        raise ValueError("GPU replay records differ from the summary")
    m.add("FcnGpuReplayRequired", "results/dataset.json", "/summary/gpu_replay_required_count", "int", "GPU replay candidates required")
    m.add("FcnGpuReplayPassed", "results/dataset.json", "/summary/gpu_replay_pass_count", "int", "GPU replay candidates passing all replay tolerances")
    m.add("FcnGpuReplayDiagnosticLimit", "protocol.json", "/replay/maximum_diagnostic_relative_difference", "sci1", "replay diagnostic relative-difference limit")
    m.add("FcnGpuReplayFieldLimit", "protocol.json", "/replay/maximum_b_component_absolute_difference_t", "sci1", "replay field component limit (T)")
    m.add("FcnGpuReplayPsiLimit", "protocol.json", "/replay/maximum_psi_absolute_difference_wb", "sci1", "replay psi limit (Wb)")
    replay_inputs = [{"artifact": "results/dataset.json", "pointer": "/gpu_replay"}]
    m.add_derived("FcnGpuReplayMaxFieldDiff", max(max(r["differences"]["br_max_abs_t"], r["differences"]["bz_max_abs_t"]) for r in replay), "sci1", "largest replay field component difference (T)", "max over gpu_replay of max(differences.br_max_abs_t, differences.bz_max_abs_t)", replay_inputs)
    m.add_derived("FcnGpuReplayMaxPsiDiff", max(r["differences"]["psi_max_abs_wb"] for r in replay), "sci1", "largest replay psi difference (Wb)", "max over gpu_replay of differences.psi_max_abs_wb", replay_inputs)
    m.add_derived("FcnGpuReplayMaxDiagnostic", max(r["diagnostic_relative_difference"] for r in replay), "sci1", "largest replay diagnostic relative difference", "max over gpu_replay of diagnostic_relative_difference", replay_inputs)
    m.add_derived("FcnGpuReplayFailedIds", [r["candidate_id"] for r in replay if not r["passed"]], "list_ident_tt", "replay candidates exceeding the diagnostic limit", "gpu_replay[passed == false].candidate_id", replay_inputs)
    m.add_derived("FcnGpuReplayFailed", sum(1 for r in replay if not r["passed"]), "int", "replay candidates exceeding the diagnostic limit", "count(gpu_replay.passed == false)", replay_inputs)
    field_limit = protocol["replay"]["maximum_b_component_absolute_difference_t"]
    psi_limit = protocol["replay"]["maximum_psi_absolute_difference_wb"]
    if any(max(r["differences"]["br_max_abs_t"], r["differences"]["bz_max_abs_t"]) > field_limit or r["differences"]["psi_max_abs_wb"] > psi_limit for r in replay):
        raise ValueError("a replay exceeded a field or psi tolerance")
    if any((r["diagnostic_relative_difference"] > protocol["replay"]["maximum_diagnostic_relative_difference"]) == r["passed"] for r in replay):
        raise ValueError("replay pass flags do not follow the diagnostic limit")

    # Representatives.
    rep_ids = sorted({item["candidate_id"] for item in manifest["representatives"]})
    ranking = [item["candidate_id"] for item in dataset["ranking"]]
    if ranking[: len(rep_ids)] != rep_ids or len(ranking) != len(cases):
        raise ValueError("representatives are not the top-ranked candidates")
    m.add_derived("FcnRepresentativeIds", rep_ids, "list_ident_tt", "representative candidates archived with full artifacts", "sorted unique manifest.representatives.candidate_id", [{"artifact": "results/manifest.json", "pointer": "/representatives"}])
    m.add("FcnTopRankedId", "results/dataset.json", "/ranking/0/candidate_id", "ident", "top-ranked candidate")
    m.add("FcnRankingPolicy", "protocol.json", "/ranking/missing_stage_policy", "text", "ranking policy for missing stages")

    # Lineage (non-evidence): superseded v1 proxy search and the failed coupling-v4 validations.
    lineage = _four_cell_lineage(repo, m)

    # Binding.
    m.add_derived("FcnResultsCommit", spec.results_commit, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "results/manifest.json", "pointer": ""}])
    m.add_derived("FcnManifestSha", manifest_sha, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "results/manifest.json", "pointer": ""}])
    m.add_derived("FcnAuditCommit", spec.posthoc_audit_commit, "sha_short", "post-hoc audit commit prefix", "commit that added POSTHOC_AUDIT.md (bound in the typed manifest)", [{"artifact": "results/preregistered-protocol.json.sha256", "pointer": ""}])
    m.add_derived("FcnToleratedEolFiles", len(b.tolerated), "int", "files whose recorded digest differs from the checkout by end-of-line bytes only", "count of verified files accepted through the audited EOL rule", [{"artifact": "results/preregistered-protocol.json.sha256", "pointer": ""}])
    m.add_derived("FcnProtocolRecordedSha", audited.recorded_sha256, "sha_short", "recorded (CRLF-era) protocol digest prefix", "manifest.protocol_sha256 as sealed at the results commit", [{"artifact": "results/manifest.json", "pointer": "/protocol_sha256"}])
    m.add_derived("FcnProtocolLfSha", audited.lf_sha256, "sha_short", "LF protocol digest prefix", "sha256(protocol.json) on an eol=lf checkout", [{"artifact": "protocol.json", "pointer": ""}])

    tex_lines = _header(spec, manifest_sha)
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    tex_lines += _table(
        spec, "FcnFailureTable",
        "Preregistered failure taxonomy of the four-cell topology search v2 with the number of the \\FcnEvaluated{} candidates "
        "that recorded each code; every candidate failed exactly the two topology gates and no field, contour, coupling or plasma gate was reached.",
        "tab:four-cell-v2-failures", "lr@{\\hspace{2.5em}}lr",
        "failure code & candidates & failure code & candidates\\\\", failure_rows,
    )
    tex_lines += _table(
        spec, "FcnCuspTable",
        "Interior cusps (vector nulls of the accepted $\\psi$ map after finite-box boundary exclusion) recorded per candidate on the "
        "three independent maps against the preregistered target of exactly \\FcnRequiredCells{} per map.",
        "tab:four-cell-v2-cusps", "llrrrr",
        "map & grid & min & max & total & exactly \\FcnRequiredCells\\\\", cusp_rows,
    )
    tex = "\n".join(tex_lines) + "\n"
    tables = {
        "FcnFailureTable": {"rows": len(failure_rows), "source": "results/dataset.json#/summary/failure_counts"},
        "FcnCuspTable": {"rows": len(cusp_rows), "source": "results/dataset.json#/cases/*/maps/<role>/interior_cusp_z_m"},
    }
    return _evidence(repo, spec, b, binding, m, tex, tables, manifest_sha, lineage=lineage), tex, lineage


def _four_cell_lineage(repo: Path, m: Macros) -> dict[str, Any]:
    """Hash-bound lineage records: superseded v1 proxy search and failed coupling-v4 validations (not evidence)."""

    lineage: dict[str, Any] = {}

    def record(relative: str, raw: bytes, identity: str, commit: str) -> None:
        lineage[relative] = {"sha256": sha256_bytes(raw), "bytes": len(raw), "identity": identity, "results_commit": commit}

    # Superseded four-cell v1 (coupling v2 proxy; not preregistered).
    v1 = repo / "modern/experiments/four_cell_topology_search/results"
    v1_manifest_raw = (v1 / "manifest.json").read_bytes()
    v1_manifest = load_json_bytes(v1_manifest_raw, "four-cell v1 manifest")
    verify_sealed(v1_manifest, "four-cell v1 manifest")
    if (v1 / "manifest.json.sha256").read_text(encoding="ascii") != f"{sha256_bytes(v1_manifest_raw)}  manifest.json\n":
        raise ValueError("four-cell v1 manifest sidecar mismatch")
    v1_dataset_raw = (v1 / "dataset.json").read_bytes()
    v1_dataset = load_json_bytes(v1_dataset_raw, "four-cell v1 dataset")
    if verify_sealed(v1_dataset, "four-cell v1 dataset") != v1_manifest["dataset_payload_sha256"]:
        raise ValueError("four-cell v1 dataset payload differs from its manifest")
    if (v1 / "dataset.json.sha256").read_text(encoding="ascii") != f"{sha256_bytes(v1_dataset_raw)}  dataset.json\n":
        raise ValueError("four-cell v1 dataset sidecar mismatch")
    status = v1_dataset["protocol_status"]
    if status["preregistered"] is not False or status["valid_for_physical_mirror_claims"] is not False or status["status"] != "development_evidence_only":
        raise ValueError("four-cell v1 protocol status differs from the archived semantics")
    record("modern/experiments/four_cell_topology_search/results/manifest.json", v1_manifest_raw, "sidecar-sha256+canonical-payload", FOUR_CELL_V1_RESULTS_COMMIT)
    record("modern/experiments/four_cell_topology_search/results/dataset.json", v1_dataset_raw, "sidecar-sha256+canonical-payload", FOUR_CELL_V1_RESULTS_COMMIT)
    compatible = sum(1 for case in v1_dataset["cases"] if case["topology"]["compatible"])
    inputs = [{"artifact": "lineage:modern/experiments/four_cell_topology_search/results/dataset.json", "pointer": "/cases"}]
    m.add_derived("FcnLineageVOneEvaluated", len(v1_dataset["cases"]), "int", "v1 proxy search candidates (lineage, not evidence)", "len(v1 dataset.cases)", inputs)
    m.add_derived("FcnLineageVOneCompatible", compatible, "int", "v1 candidates screened compatible by the deprecated proxy (lineage, not evidence)", "count(v1 dataset.cases.topology.compatible == true)", inputs)
    m.add_derived("FcnLineageVOneStatus", status["status"], "ident", "v1 protocol status", "v1 dataset.protocol_status.status", [{"artifact": "lineage:modern/experiments/four_cell_topology_search/results/dataset.json", "pointer": "/protocol_status/status"}])
    m.add_derived("FcnLineageVOnePreregistered", status["preregistered"], "bool", "v1 preregistered flag", "v1 dataset.protocol_status.preregistered", [{"artifact": "lineage:modern/experiments/four_cell_topology_search/results/dataset.json", "pointer": "/protocol_status/preregistered"}])
    m.add_derived("FcnLineageVOneMirrorValid", status["valid_for_physical_mirror_claims"], "bool", "v1 validity for physical mirror claims", "v1 dataset.protocol_status.valid_for_physical_mirror_claims", [{"artifact": "lineage:modern/experiments/four_cell_topology_search/results/dataset.json", "pointer": "/protocol_status/valid_for_physical_mirror_claims"}])
    m.add_derived("FcnLineageVOneClassification", v1_manifest["classification"], "ident", "v1 classification", "v1 manifest.classification", [{"artifact": "lineage:modern/experiments/four_cell_topology_search/results/manifest.json", "pointer": "/classification"}])

    # Failed coupling-v4 wall-cusp validations v1 and v2 (semantic identity; manifests recorded CRLF byte digests).
    for version, commit in (("v1", WCVAL_V1_RESULTS_COMMIT), ("v2", WCVAL_V2_RESULTS_COMMIT)):
        root = repo / f"modern/experiments/cft_wall_cusp_validation_{version}/results"
        manifest_raw = (root / "manifest.json").read_bytes()
        manifest = load_json_bytes(manifest_raw, f"wcval {version} manifest")
        verify_sealed(manifest, f"wcval {version} manifest", "semantic_integrity")
        if manifest["accepted_coupling_commit_sha"] != COUPLING_V4_COMMIT or manifest["summary"]["criterion_numerically_promoted"] is not False:
            raise ValueError(f"wcval {version}: manifest identity or promotion differs")
        listed = {item["path"]: item for item in manifest["artifacts"]}
        failure_raw = (root / "failure.json").read_bytes()
        failure = load_json_bytes(failure_raw, f"wcval {version} failure")
        entry = listed["failure.json"]
        if entry["identity_method"] != "byte-and-canonical-json-sha256" or canonical_hash(failure) != entry["semantic_sha256"]:
            raise ValueError(f"wcval {version}: failure.json semantic identity mismatch")
        record(f"modern/experiments/cft_wall_cusp_validation_{version}/results/manifest.json", manifest_raw, "canonical-semantic-payload", commit)
        record(f"modern/experiments/cft_wall_cusp_validation_{version}/results/failure.json", failure_raw, "manifest canonical-json-sha256 (recorded byte digest is CRLF-era; not relied on)", commit)
        camel = "One" if version == "v1" else "Two"
        art = f"lineage:modern/experiments/cft_wall_cusp_validation_{version}/results/failure.json"
        m.add_derived(f"FcnLineageWcval{camel}Phase", failure["failure"]["phase"], "ident", f"coupling-v4 validation {version} failure phase (lineage)", "failure.failure.phase", [{"artifact": art, "pointer": "/failure/phase"}])
        m.add_derived(f"FcnLineageWcval{camel}Exception", failure["failure"]["exception_type"], "text", f"coupling-v4 validation {version} exception type (lineage)", "failure.failure.exception_type", [{"artifact": art, "pointer": "/failure/exception_type"}])
        m.add_derived(f"FcnLineageWcval{camel}Declared", manifest["summary"]["declared_case_count"], "int", f"coupling-v4 validation {version} declared cases (lineage)", "manifest.summary.declared_case_count", [{"artifact": art.replace("failure.json", "manifest.json"), "pointer": "/summary/declared_case_count"}])
        m.add_derived(f"FcnLineageWcval{camel}Attempted", manifest["summary"]["attempted_case_count"], "int", f"coupling-v4 validation {version} attempted cases (lineage)", "manifest.summary.attempted_case_count", [{"artifact": art.replace("failure.json", "manifest.json"), "pointer": "/summary/attempted_case_count"}])
        m.add_derived(f"FcnLineageWcval{camel}Promoted", manifest["summary"]["criterion_numerically_promoted"], "bool", f"coupling-v4 validation {version} criterion promoted (lineage)", "manifest.summary.criterion_numerically_promoted", [{"artifact": art.replace("failure.json", "manifest.json"), "pointer": "/summary/criterion_numerically_promoted"}])
    return lineage


# --------------------------------------------------------------------------- #
# Experiment 3: CFT topology characterization v1 (recorded characterization)
# --------------------------------------------------------------------------- #
def build_characterization(repo: Path, spec: ExperimentSpec = CHARACTERIZATION) -> tuple[dict[str, Any], str, dict[str, Any]]:
    b = Bundle(repo, spec)
    binding = b.bind_committed()
    m = Macros(b)
    manifest_raw = b.read("results/manifest.json")
    manifest_sha = sha256_bytes(manifest_raw)
    manifest = b.load("results/manifest.json")
    verify_sealed(manifest, "characterization manifest", "semantic_integrity")
    if (
        manifest["experiment_id"] != spec.experiment_id
        or manifest["preregistration_commit_sha"] != spec.preregistration_commit
        or manifest["accepted_coupling_v3_commit_sha"] != COUPLING_V3_COMMIT
        or manifest["single_execution"] is not True
    ):
        raise ValueError("characterization manifest identity differs")
    for entry in manifest["artifacts"]:
        relative = f"results/{entry['path']}"
        raw = b.read(relative)
        if entry["identity_method"] == "canonical-json-sha256":
            if canonical_hash(load_json_bytes(raw, relative)) != entry["semantic_sha256"]:
                raise ValueError(f"{relative}: canonical semantic SHA-256 mismatch")
        elif entry["identity_method"] == "normalized-lf-text-sha256":
            if sha256_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")) != entry["semantic_sha256"]:
                raise ValueError(f"{relative}: normalized text SHA-256 mismatch")
        else:
            raise ValueError(f"{relative}: unsupported identity method")
    protocol = b.load("protocol.json")
    protocol_copy = b.load("results/preregistered-protocol.json")
    if canonical_hash(protocol) != manifest["protocol_semantic_sha256"] or canonical_hash(protocol_copy) != manifest["protocol_semantic_sha256"]:
        raise ValueError("protocol semantic identity differs from the manifest binding")
    dataset = b.load("results/dataset.json")
    verify_sealed(dataset, "characterization dataset", "semantic_integrity")
    if dataset["summary"] != manifest["summary"] or dataset["classification"] != spec.classification:
        raise ValueError("dataset summary or classification differs from the manifest")
    if dataset["preregistration_commit_sha"] != spec.preregistration_commit or dataset["protocol_semantic_sha256"] != manifest["protocol_semantic_sha256"]:
        raise ValueError("dataset preregistration binding differs")
    if b.tolerated:
        raise ValueError("characterization bundle needs no EOL tolerance")
    cases = dataset["cases"]
    summary = dataset["summary"]
    if len(cases) != summary["evaluated_count"] or summary["stable_eligible_cusp_count"] != 0 or summary["stable_eligible_cell_count"] != 0:
        raise ValueError("characterization cases differ from the recorded summary")
    if any(root["eligible_cusp"] or root["eligible_cell"] for case in cases for role in case["maps"] for root in case["maps"][role]["roots"]):
        raise ValueError("an eligible root exists that the summary does not record")
    if sum(case["cross_map"]["stable_eligible_cusp_count"] + case["cross_map"]["stable_eligible_cell_count"] for case in cases) != 0:
        raise ValueError("stable eligible counts differ from the summary")
    failure_counter = Counter(code for case in cases for code in case["failures"])
    if {k: v for k, v in summary["failure_counts"].items() if v} != dict(failure_counter):
        raise ValueError("failure counts differ from the per-case failures")

    m.add("TchClassification", "results/dataset.json", "/classification", "ident", "characterization classification string")
    m.add("TchPurpose", "results/dataset.json", "/purpose", "text", "declared purpose")
    m.add("TchFieldModel", "results/preregistered-protocol.json", "/accepted_baseline/field_model", "text", "accepted field model of the baseline")
    m.add("TchSingleExecution", "results/manifest.json", "/single_execution", "bool", "single execution flag")
    m.add("TchDeclared", "results/dataset.json", "/summary/declared_case_count", "int", "declared cases")
    m.add("TchEvaluated", "results/dataset.json", "/summary/evaluated_count", "int", "cases evaluated")
    m.add("TchThreeMapAccepted", "results/dataset.json", "/summary/three_map_accepted_count", "int", "cases whose three maps passed every field gate")
    m.add("TchStableEligibleCusps", "results/dataset.json", "/summary/stable_eligible_cusp_count", "int", "stable eligible cusps")
    m.add("TchStableEligibleCells", "results/dataset.json", "/summary/stable_eligible_cell_count", "int", "stable eligible cells")
    m.add("TchMirrorProbabilityCount", "results/dataset.json", "/summary/mirror_probability_count", "int", "mirror probabilities computed")
    m.add("TchPlasmaPublicationCount", "results/dataset.json", "/summary/plasma_publication_count", "int", "plasma publications")
    m.add("TchGpuReplayPassed", "results/dataset.json", "/summary/gpu_replay_pass_count", "int", "GPU replay passes")
    m.add("TchGpuReplayRequired", "results/dataset.json", "/summary/gpu_replay_required_count", "int", "GPU replays required")
    m.add("TchFailSeparatrix", "results/dataset.json", "/summary/failure_counts/SEPARATRIX_UNRESOLVED", "int", "cases with SEPARATRIX_UNRESOLVED")
    m.add("TchFailUnmatched", "results/dataset.json", "/summary/failure_counts/CROSS_MAP_UNMATCHED", "int", "cases with CROSS_MAP_UNMATCHED")
    m.add("TchFailClassChanged", "results/dataset.json", "/summary/failure_counts/CROSS_MAP_CLASS_CHANGED", "int", "cases with CROSS_MAP_CLASS_CHANGED")
    m.add("TchFailHardware", "results/dataset.json", "/summary/failure_counts/ROOT_EXCLUDED_HARDWARE", "int", "cases with ROOT_EXCLUDED_HARDWARE")
    m.add("TchFailOutsideChannel", "results/dataset.json", "/summary/failure_counts/ROOT_OUTSIDE_CHANNEL", "int", "cases with ROOT_OUTSIDE_CHANNEL")
    m.add("TchFailFieldPrimary", "results/dataset.json", "/summary/failure_counts/FIELD_PRIMARY_INVALID", "int", "cases with FIELD_PRIMARY_INVALID")
    evaluation_codes = {"GEOMETRY_INVALID", "FIELD_PRIMARY_INVALID", "FIELD_REFINED_INVALID", "FIELD_ENLARGED_INVALID"}
    if not evaluation_codes <= set(summary["failure_counts"]):
        raise ValueError("evaluation failure codes are missing from the taxonomy")
    m.add_derived(
        "TchEvaluationFailures", sum(1 for case in cases if evaluation_codes & set(case["failures"])), "int",
        "cases whose geometry or any field map failed to evaluate",
        "count(cases with any of GEOMETRY_INVALID, FIELD_PRIMARY_INVALID, FIELD_REFINED_INVALID, FIELD_ENLARGED_INVALID in failures)",
        [{"artifact": "results/dataset.json", "pointer": "/cases"}],
    )
    m.add_derived("TchMapCount", len(protocol["maps"]["roles"]), "int", "independent maps per case", "len(protocol.maps.roles)", [{"artifact": "results/preregistered-protocol.json", "pointer": "/maps/roles"}])
    m.add("TchNotOptimization", "results/preregistered-protocol.json", "/not_a_design_optimization", "bool", "not a design optimization")
    m.add("TchNotValidation", "results/preregistered-protocol.json", "/not_a_blind_validation", "bool", "not a blind validation")
    m.add("TchPublicationAllowed", "results/dataset.json", "/publication/allowed", "text", "allowed publication scope")
    m.add("TchMirrorPublished", "results/dataset.json", "/publication/mirror_probability", "bool", "mirror probability publication flag")
    m.add("TchPlasmaPublished", "results/dataset.json", "/publication/plasma_state_power_or_performance", "bool", "plasma publication flag")
    m.add("TchStageCounts", "results/preregistered-protocol.json", "/families/stage_counts", "list_int", "stage counts of the family")
    m.add("TchPitches", "results/preregistered-protocol.json", "/families/pitch_m", "list_mm1", "pitches (mm)")
    m.add("TchRadii", "results/preregistered-protocol.json", "/families/chamber_outer_radius_m", "list_mm1", "chamber radii (mm)")
    m.add("TchPolarities", "results/preregistered-protocol.json", "/families/first_polarity", "list_int", "first polarities")
    m.add("TchFamilyCaseCount", "results/preregistered-protocol.json", "/families/case_count", "int", "declared family case count")
    m.add_derived("TchStageMin", min(protocol["families"]["stage_counts"]), "int", "minimum stage count", "min(protocol.families.stage_counts)", [{"artifact": "results/preregistered-protocol.json", "pointer": "/families/stage_counts"}])
    m.add_derived("TchStageMax", max(protocol["families"]["stage_counts"]), "int", "maximum stage count", "max(protocol.families.stage_counts)", [{"artifact": "results/preregistered-protocol.json", "pointer": "/families/stage_counts"}])
    m.add_derived("TchStageLevels", len(protocol["families"]["stage_counts"]), "int", "stage count levels", "len(protocol.families.stage_counts)", [{"artifact": "results/preregistered-protocol.json", "pointer": "/families/stage_counts"}])
    m.add("TchPrimaryRadial", "results/preregistered-protocol.json", "/maps/roles/primary/radial_intervals", "int", "primary radial intervals")
    m.add("TchPrimaryAxial", "results/preregistered-protocol.json", "/maps/roles/primary/axial_intervals", "int", "primary axial intervals")
    m.add("TchRefinedRadial", "results/preregistered-protocol.json", "/maps/roles/refined/radial_intervals", "int", "refined radial intervals")
    m.add("TchRefinedAxial", "results/preregistered-protocol.json", "/maps/roles/refined/axial_intervals", "int", "refined axial intervals")
    m.add("TchEnlargedRadial", "results/preregistered-protocol.json", "/maps/roles/enlarged_domain/radial_intervals", "int", "enlarged-domain radial intervals")
    m.add("TchEnlargedAxial", "results/preregistered-protocol.json", "/maps/roles/enlarged_domain/axial_intervals", "int", "enlarged-domain axial intervals")
    m.add("TchEnlargedScale", "results/preregistered-protocol.json", "/maps/roles/enlarged_domain/domain_scale", "g", "enlarged-domain scale")
    m.add("TchRootDetector", "results/preregistered-protocol.json", "/root_detection/accepted_detector", "ident", "root detector")
    m.add("TchClusterFactor", "results/preregistered-protocol.json", "/root_detection/clustering/tolerance_mesh_factor", "g", "clustering tolerance (mesh factor)")
    m.add("TchCuspEligibility", "results/preregistered-protocol.json", "/eligibility/eligible_cusp", "list_clauses", "eligible cusp conditions")
    m.add("TchCellEligibility", "results/preregistered-protocol.json", "/eligibility/eligible_cell", "list_clauses", "eligible cell conditions")
    m.add("TchCellBoundingRequirement", "results/preregistered-protocol.json", "/separatrix/cell_bounding_requirement", "text", "cell-bounding requirement")
    m.add("TchStableRootDefinition", "results/preregistered-protocol.json", "/cross_map_correspondence/stable_root", "text", "stable root definition")
    m.add("TchCorrespondenceAlgorithm", "results/preregistered-protocol.json", "/cross_map_correspondence/algorithm", "text", "correspondence algorithm")
    m.add("TchRecommendationClass", "results/dataset.json", "/analyses/search_v3_recommendation/classification", "ident", "search-v3 recommendation class")
    m.add("TchRecommendedStages", "results/dataset.json", "/analyses/search_v3_recommendation/recommended_stage_counts", "list_int", "recommended stage counts (descriptive)")
    m.add("TchRecommendationNotValidated", "results/dataset.json", "/analyses/search_v3_recommendation/not_validated_or_optimal", "bool", "recommendation not validated or optimal")
    m.add("TchCouplingCommit", "results/dataset.json", "/accepted_coupling_v3_commit_sha", "sha_short", "accepted coupling v3 commit prefix")
    m.add("TchPreregCommit", "results/dataset.json", "/preregistration_commit_sha", "sha_short", "preregistration commit prefix")
    m.add("TchProtocolSemanticSha", "results/dataset.json", "/protocol_semantic_sha256", "sha_short", "protocol semantic hash prefix")
    m.add("TchGpuName", "results/dataset.json", "/runtime_identity/gpu_name", "text", "GPU name")
    m.add("TchWarpVersion", "results/dataset.json", "/runtime_identity/warp_version", "text", "Warp version")

    # Root classes, zones and exclusions over the primary maps.
    case_inputs = [{"artifact": "results/dataset.json", "pointer": "/cases"}]
    classes: Counter[str] = Counter()
    zone_rows: dict[str, dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    channel_classes: Counter[str] = Counter()
    channel_exclusions: Counter[str] = Counter()
    clustered = raw_detections = 0
    for case in cases:
        primary = case["maps"]["primary"]
        clustered += primary["clustered_root_count"]
        raw_detections += primary["raw_detection_count"]
        if len(primary["roots"]) != primary["clustered_root_count"]:
            raise ValueError(f"{case['case_id']}: clustered root count differs from the roots recorded")
        for root in primary["roots"]:
            cls = root["local_topology"]["classification"]
            zone = root["geometry_association"]["zone"]
            classes[cls] += 1
            exclusions[root["exclusion_reason"]] += 1
            row = zone_rows.setdefault(zone, {"total": 0, "X": 0, "O": 0, "degenerate": 0, "eligible": 0, "reasons": Counter()})
            row["total"] += 1
            row[cls] += 1
            row["eligible"] += int(bool(root["eligible_cusp"] or root["eligible_cell"]))
            row["reasons"][root["exclusion_reason"]] += 1
            if zone == "plasma_channel":
                channel_classes[cls] += 1
                channel_exclusions[root["exclusion_reason"]] += 1
    if sum(classes.values()) != clustered:
        raise ValueError("root class tally differs from the clustered root count")
    m.add_derived("TchClusteredRoots", clustered, "int", "clustered primary-map vector nulls", "sum(cases.maps.primary.clustered_root_count)", case_inputs)
    m.add_derived("TchRawDetections", raw_detections, "int", "raw primary-map detections before clustering", "sum(cases.maps.primary.raw_detection_count)", case_inputs)
    m.add_derived("TchXRoots", classes["X"], "int", "primary-map roots classified X", "count(roots.local_topology.classification == X)", case_inputs)
    m.add_derived("TchORoots", classes["O"], "int", "primary-map roots classified O", "count(roots.local_topology.classification == O)", case_inputs)
    m.add_derived("TchDegenerateRoots", classes["degenerate"], "int", "primary-map roots classified degenerate", "count(roots.local_topology.classification == degenerate)", case_inputs)
    m.add_derived("TchChannelRoots", sum(channel_classes.values()), "int", "primary-map roots inside the plasma channel", "count(roots.geometry_association.zone == plasma_channel)", case_inputs)
    m.add_derived("TchChannelXRoots", channel_classes["X"], "int", "plasma-channel roots classified X", "count(channel roots with classification X)", case_inputs)
    m.add_derived("TchChannelORoots", channel_classes["O"], "int", "plasma-channel roots classified O", "count(channel roots with classification O)", case_inputs)
    m.add_derived("TchChannelUnresolved", channel_exclusions["no_cell_bounding_separatrix"], "int", "plasma-channel roots excluded for an unresolved cell-bounding separatrix", "count(channel roots with exclusion_reason no_cell_bounding_separatrix)", case_inputs)
    if set(channel_exclusions) != {"no_cell_bounding_separatrix"} or set(channel_classes) != {"X"}:
        raise ValueError("plasma-channel roots are not all X with an unresolved separatrix")
    m.add_derived("TchExclMagnet", exclusions["exterior_magnet_or_current_sheet"], "int", "roots excluded as magnet or current sheet", "count(roots.exclusion_reason == exterior_magnet_or_current_sheet)", case_inputs)
    m.add_derived("TchExclFiniteBox", exclusions["finite_box_boundary"], "int", "roots excluded on the finite-box boundary", "count(roots.exclusion_reason == finite_box_boundary)", case_inputs)
    m.add_derived("TchExclNoSeparatrix", exclusions["no_cell_bounding_separatrix"], "int", "roots excluded for no cell-bounding separatrix", "count(roots.exclusion_reason == no_cell_bounding_separatrix)", case_inputs)
    m.add_derived("TchExclYoke", exclusions["yoke_or_material"], "int", "roots excluded as yoke or material", "count(roots.exclusion_reason == yoke_or_material)", case_inputs)
    m.add_derived("TchExclOutsideAxial", exclusions["outside_channel_axial"], "int", "roots excluded outside the channel axially", "count(roots.exclusion_reason == outside_channel_axial)", case_inputs)
    m.add_derived("TchExclusionReasonCount", len(exclusions), "int", "distinct exclusion reasons observed", "len(set(roots.exclusion_reason))", case_inputs)
    m.add_derived("TchZoneCount", len(zone_rows), "int", "distinct geometric zones observed", "len(set(roots.geometry_association.zone))", case_inputs)
    stable_roots = sum(case["cross_map"]["stable_root_count"] for case in cases)
    m.add_derived("TchStableRoots", stable_roots, "int", "roots matched in all three maps with unchanged class and eligibility", "sum(cases.cross_map.stable_root_count)", case_inputs)
    for role in ("refined", "enlarged_domain"):
        m.add_derived(f"Tch{_camel(role)}Roots", sum(len(case["maps"][role]["roots"]) for case in cases), "int", f"clustered roots on the {role} maps", f"sum(len(cases.maps.{role}.roots))", case_inputs)
    m.add_derived("TchEligibleCuspsAnyMap", sum(case["maps"][role]["eligible_cusp_count"] for case in cases for role in case["maps"]), "int", "eligible cusps on any map before cross-map stability", "sum(cases.maps.*.eligible_cusp_count)", case_inputs)
    m.add_derived("TchEligibleCellsAnyMap", sum(case["maps"][role]["eligible_cell_count"] for case in cases for role in case["maps"]), "int", "eligible cells on any map before cross-map stability", "sum(cases.maps.*.eligible_cell_count)", case_inputs)
    complete = sum(1 for case in cases if case["cross_map"]["complete_primary_correspondence"])
    m.add_derived("TchCompleteCorrespondence", complete, "int", "cases with complete primary correspondence", "count(cases.cross_map.complete_primary_correspondence == true)", case_inputs)
    zone_order = sorted(zone_rows, key=lambda z: -zone_rows[z]["total"])
    class_rows: list[str] = []
    for zone in zone_order:
        row = zone_rows[zone]
        reasons = ", ".join(f"\\texttt{{{_ident(r)}}}" for r in sorted(row["reasons"]))
        class_rows.append(
            f"\\texttt{{{_ident(zone)}}} & {row['total']} & {row['X']} & {row['O']} & {row['degenerate']} & {reasons} & {row['eligible']}\\\\"
        )
    class_rows.append(
        f"\\midrule\ntotal & {clustered} & {classes['X']} & {classes['O']} & {classes['degenerate']} & --- & \\TchStableEligibleCusps{{}}/\\TchStableEligibleCells\\\\"
    )

    # Stage relation table.
    stage_rows: list[str] = []
    relation = dataset["analyses"]["stage_relation"]
    if [item["stage_count"] for item in relation] != protocol["families"]["stage_counts"]:
        raise ValueError("stage relation rows differ from the family stage counts")
    for item in relation:
        if sum(item["stable_eligible_cusp_counts"]) or sum(item["stable_eligible_cell_counts"]) or item["case_count"] != len(item["stable_eligible_cusp_counts"]):
            raise ValueError("stage relation records a stable eligible root the summary does not")
        stage_rows.append(
            f"{item['stage_count']} & {item['case_count']} & {item['modal_stable_eligible_cusp_count']} & {item['modal_stable_eligible_cell_count']} & "
            f"{format_value('fixed3', item['complete_correspondence_fraction'])} & {format_value('mm2', item['median_maximum_shift_m'])}\\\\"
        )
    m.add_derived("TchCasesPerStage", relation[0]["case_count"], "int", "cases per stage count", "analyses.stage_relation[*].case_count (all equal)", [{"artifact": "results/dataset.json", "pointer": "/analyses/stage_relation"}])
    if len({item["case_count"] for item in relation}) != 1:
        raise ValueError("stage relation case counts are not uniform")
    m.add_derived("TchMedianShiftMaxMm", max(item["median_maximum_shift_m"] for item in relation), "mm2", "largest median maximum root shift across stage counts (mm)", "max(analyses.stage_relation.median_maximum_shift_m)", [{"artifact": "results/dataset.json", "pointer": "/analyses/stage_relation"}])

    m.add_derived("TchResultsCommit", spec.results_commit, "sha_short", "results commit prefix", "git commit whose tree holds the results manifest blob (verified with rev-parse against the working tree)", [{"artifact": "results/manifest.json", "pointer": ""}])
    m.add_derived("TchManifestSha", manifest_sha, "sha_short", "results manifest SHA-256 prefix", "sha256(results/manifest.json)", [{"artifact": "results/manifest.json", "pointer": ""}])
    m.add_derived("TchToleratedEolFiles", len(b.tolerated), "int", "files whose recorded digest differs from the checkout by end-of-line bytes only", "count of verified files accepted through the audited EOL rule (none: this bundle binds canonical semantic identities)", [{"artifact": "results/manifest.json", "pointer": "/artifacts"}])

    tex_lines = _header(spec, manifest_sha)
    for item in m.items:
        tex_lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    tex_lines += _table(
        spec, "TchNullClassTable",
        "Clustered vector nulls of the \\TchEvaluated{} primary maps by geometric zone, local classification and exclusion reason. "
        "Every root was excluded under the preregistered eligibility rules; the \\TchChannelRoots{} roots inside the plasma channel are all X-type "
        "and none established a cell-bounding separatrix. The last column counts eligible cusps/cells.",
        "tab:topology-char-v1-classes", f"lrrrr{_p(4.2)}r",
        "zone & roots & X & O & degen. & exclusion reason & eligible\\\\", class_rows,
        extra="\\setlength{\\tabcolsep}{4pt}",
    )
    tex_lines += _table(
        spec, "TchStageTable",
        "Empirical stage relation: per stage count, cases, modal stable eligible cusp and cell counts, fraction of cases with complete "
        "primary-to-refined-to-enlarged root correspondence and median maximum root shift.",
        "tab:topology-char-v1-stages", "rrrrrr",
        "stages & cases & modal cusps & modal cells & complete corr. & median shift (mm)\\\\", stage_rows,
    )
    tex = "\n".join(tex_lines) + "\n"
    tables = {
        "TchNullClassTable": {"rows": len(class_rows), "source": "results/dataset.json#/cases/*/maps/primary/roots"},
        "TchStageTable": {"rows": len(stage_rows), "source": "results/dataset.json#/analyses/stage_relation"},
    }
    return _evidence(repo, spec, b, binding, m, tex, tables, manifest_sha), tex, {}


# --------------------------------------------------------------------------- #
# Shared output assembly
# --------------------------------------------------------------------------- #
def _header(spec: ExperimentSpec, manifest_sha: str) -> list[str]:
    return [
        f"% Generated by paper/scripts/generate_topology_screening_evidence.py ({spec.key}); do not hand edit.",
        f"% Evidence: {spec.experiment_path.as_posix()} at commit {spec.results_commit} (results manifest SHA-256 {manifest_sha}).",
        f"% Every macro value traces to an artifact path and JSON pointer recorded in {spec.evidence_path.as_posix()}.",
    ]


def _evidence(
    repo: Path, spec: ExperimentSpec, bundle: Bundle, binding: dict[str, Any], m: Macros, tex: str,
    tables: dict[str, Any], manifest_sha: str, *, lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len({item["name"] for item in m.items}) != len(m.items):
        raise ValueError("duplicate macro names")
    evidence: dict[str, Any] = {
        "document_type": spec.document_type,
        "schema_version": "1.0",
        "experiment_id": spec.experiment_id,
        "classification": spec.classification,
        "recorded_outcome": spec.recorded_outcome,
        "screening_model": SCREENING_MODEL,
        "evidence_revision": spec.results_commit,
        "binding": binding,
        "manuscript_integration": {
            "status": "admitted",
            "gate_kind": GATE_KIND,
            "section_path": spec.section_path.as_posix(),
            "section_binding": spec.section_binding,
            "section_heading": spec.section_heading,
            "generated_tex_path": spec.output_path.as_posix(),
            "generated_binding": spec.generated_binding,
            "manifest_id": spec.manifest_id,
            "manifest_path": spec.manifest_path.as_posix(),
            "gate_id": spec.gate_id,
            "artifact_id": spec.artifact_id,
            "artifact_claim_id": spec.artifact_claim_id,
            "prose_claim_ids": list(spec.prose_claim_ids),
            "table_macros": list(spec.table_macros),
            "rule": (
                "Every number in the section is a macro defined here; each macro is bound below to an artifact path, "
                "JSON pointer, formatter and SHA-256. Claim-bearing sentences are exact EvidenceClaim bodies registered in "
                "paper/evidence/claims.json; the numerical-screening gate in paper/evidence/result-gates.json names the typed "
                "manifest that admits the section and records the outcome as accepted screening, preregistered null or "
                "recorded characterization without opening any physics level."
            ),
        },
        "bundle": {
            "manifest_path": (spec.experiment_path / "results/manifest.json").as_posix(),
            "manifest_sha256": manifest_sha,
            "verified_file_count": len(bundle.used),
            "tolerated_eol_files": list(bundle.tolerated),
            "tolerance_rule": (
                "for exactly the audited files: bytes contain no CR, sha256(bytes) == audited LF digest and "
                "sha256(bytes.replace(LF, CRLF)) == recorded digest with the recorded byte count; every other mismatch fails"
            ),
            "audited_eol_files": {
                path: {"lf_sha256": item.lf_sha256, "recorded_sha256": item.recorded_sha256, "recorded_bytes": item.recorded_bytes, "audit_module": item.audit_module}
                for path, item in sorted(spec.audited_eol_files.items())
            },
        },
        "artifacts": {path: bundle.used[path] for path in sorted(bundle.used)},
        "macros": m.items,
        "tables": tables,
        "generator": {
            "path": "paper/scripts/generate_topology_screening_evidence.py",
            "sha256": sha256_bytes((repo / "paper/scripts/generate_topology_screening_evidence.py").read_bytes().replace(b"\r\n", b"\n")),
            "command": "python paper/scripts/generate_topology_screening_evidence.py",
        },
        "output": {"path": spec.output_path.as_posix(), "sha256": sha256_bytes(tex.encode("utf-8"))},
    }
    if lineage:
        evidence["lineage_artifacts"] = {
            "rule": "hash-bound lineage records quoted as non-claims only; they are not evidence and authorize no result",
            "files": {path: lineage[path] for path in sorted(lineage)},
        }
    return evidence


BUILDERS: dict[str, Callable[[Path, ExperimentSpec], tuple[dict[str, Any], str, dict[str, Any]]]] = {
    SWEEP.key: build_sweep,
    FOUR_CELL.key: build_four_cell,
    CHARACTERIZATION.key: build_characterization,
}


def build(repo: Path, spec: ExperimentSpec) -> tuple[dict[str, Any], str]:
    evidence, tex, _lineage = BUILDERS[spec.key](repo, spec)
    return evidence, tex


def render(repo: Path, spec: ExperimentSpec) -> tuple[bytes, bytes, bytes]:
    evidence, tex = build(repo, spec)
    tex_bytes = tex.encode("utf-8")
    build_config = json.loads((repo / "paper/build-config.json").read_text("utf-8"))
    sidecar = {
        "document_type": "paper-generated-artifact-provenance",
        "schema_version": "1.0",
        "artifact_id": spec.artifact_id,
        "claim_ids": [spec.artifact_claim_id],
        "claim_status": (
            f"authorized by {spec.artifact_claim_id} (quantitative-generated-table) in paper/evidence/claims.json; "
            f"admitted through {spec.gate_id} ({GATE_KIND}, recorded outcome {spec.recorded_outcome})"
        ),
        "evidence_revision": spec.results_commit,
        "source_date_epoch": build_config["source_date_epoch"],
        "generator": evidence["generator"],
        "manifest": {
            "path": spec.evidence_path.as_posix(),
            "sha256": sha256_bytes(canonical_json(evidence)),
            "manifest_id": spec.manifest_id,
            "gate_manifest_path": spec.manifest_path.as_posix(),
        },
        "inputs": [
            {"path": (spec.experiment_path / path).as_posix(), "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["artifacts"].items()
        ],
        "bundle_manifest": {
            "path": evidence["bundle"]["manifest_path"],
            "sha256": evidence["bundle"]["manifest_sha256"],
            "git_blob": evidence["binding"]["manifest_git_blob"],
        },
        "output": {"path": spec.output_path.as_posix(), "sha256": sha256_bytes(tex_bytes)},
    }
    return canonical_json(evidence), tex_bytes, canonical_json(sidecar)


def write_generated(repo: Path, spec: ExperimentSpec) -> None:
    evidence, tex, sidecar = render(repo, spec)
    (repo / spec.evidence_path).parent.mkdir(parents=True, exist_ok=True)
    (repo / spec.output_path).parent.mkdir(parents=True, exist_ok=True)
    (repo / spec.evidence_path).write_bytes(evidence)
    (repo / spec.output_path).write_bytes(tex)
    (repo / spec.sidecar_path).write_bytes(sidecar)


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    keys = argv if argv else list(EXPERIMENTS)
    for key in keys:
        spec = EXPERIMENTS.get(key)
        if spec is None:
            print(f"unknown experiment key {key!r}; choose from {', '.join(EXPERIMENTS)}")
            return 2
        try:
            write_generated(repo, spec)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"{key}: evidence generation failed: {exc}")
            return 1
        print(f"Generated {spec.evidence_path}, {spec.output_path} and {spec.sidecar_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
