"""Fail-closed evidence, claim, artifact, citation, and submission checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import generate_tables
import generate_topology_screening_evidence as topology_screening
import generate_wall_loss_v4_evidence as wall_loss_v4


REQUIRED_SECTIONS = (
    "Introduction",
    "Literature lineage and scope",
    "Methods architecture",
    "Legacy audit",
    "Verification, validation, and uncertainty protocol",
    "Accepted L0 result",
    "Accepted numerical campaign: collisionless electron wall loss",
    "Preregistered topology screening: sweep acceptance and four-cell null result",
    "Planned L1 result: field-resolved reduction",
    "Planned L2 result: coupled hybrid model",
    "Planned L3 result: PIC and experimental comparison",
    "Discussion",
    "Limitations",
    "Reproducibility and data availability",
)

# Gate kinds.  Physics-level gates (L1--L3) open a paper level and stay closed
# until their manifest is accepted; numerical-campaign gates admit one accepted,
# preregistered numerical campaign that opens no paper level; numerical-screening
# gates admit one preregistered, single-execution L1a field-only screening study
# at its recorded outcome (accepted screening, preregistered null or recorded
# characterization) and open no paper level either.
PHYSICS_GATE_KIND = "physics-level"
CAMPAIGN_GATE_KIND = "numerical-campaign"
SCREENING_GATE_KIND = topology_screening.GATE_KIND
SCREENING_OUTCOMES = frozenset({"accepted-screening", "preregistered-null", "recorded-characterization"})
KNOWN_GATE_KINDS = frozenset({PHYSICS_GATE_KIND, CAMPAIGN_GATE_KIND, SCREENING_GATE_KIND})
PHYSICS_GATE_IDS = frozenset({"GATE-L1", "GATE-L2", "GATE-L3"})

# Screening manifest metric -> evidence macro whose raw artifact value it must equal,
# keyed by the experiment key of generate_topology_screening_evidence.
SCREENING_METRIC_MACROS: dict[str, dict[str, str]] = {
    "l1a-sweep-v2": {
        "classification": "SwpClassification",
        "maximum_executions": "SwpMaxExecutions",
        "declared_count": "SwpRequested",
        "evaluated_count": "SwpEvaluated",
        "failed_cases_count": "SwpFailed",
        "terminal_status": "SwpTerminalStatus",
        "gate_count": "SwpGateCount",
        "gates_passed": "SwpGatesPassed",
        "gate_failures": "SwpGateFailures",
        "zero_failures_required": "SwpZeroFailuresRequired",
        "nondominated_count": "SwpNondominated",
        "unique_representative_count": "SwpUniqueRepresentatives",
        "representative_role_count": "SwpRoleCount",
        "objective_count": "SwpObjectiveCount",
        "design_variable_count": "SwpDesignVariableCount",
        "parity_case_count": "SwpParityCaseCount",
        "axis_cusp_count_minimum": "SwpAxisCuspMin",
        "axis_cusp_count_maximum": "SwpAxisCuspMax",
        "designs_with_three_axis_cusps": "SwpCuspThreeDesigns",
        "designs_with_four_axis_cusps": "SwpCuspFourDesigns",
        "designs_with_five_axis_cusps": "SwpCuspFiveDesigns",
        "axis_cusp_total": "SwpAxisCuspTotal",
        "axis_cusp_z_minimum_m": "SwpAxisCuspZMinMm",
        "axis_cusp_z_maximum_m": "SwpAxisCuspZMaxMm",
        "resolved_axis_null_designs": "SwpResolvedDesigns",
        "preview_source_authoritative": "SwpPreviewAuthoritative",
        "tolerated_eol_file_count": "SwpToleratedEolFiles",
    },
    "four-cell-v2": {
        "classification": "FcnClassification",
        "single_execution": "FcnSingleExecution",
        "declared_count": "FcnDeclared",
        "evaluated_count": "FcnEvaluated",
        "failed_cases_count": "FcnEvaluationFailures",
        "field_maps_per_design": "FcnMapCount",
        "three_map_accepted_count": "FcnThreeMapAccepted",
        "stable_count": "FcnStable",
        "required_stable_cell_count": "FcnRequiredCells",
        "topology_count_failures": "FcnTopologyCountFailures",
        "topology_unstable_failures": "FcnTopologyUnstableFailures",
        "failure_code_count": "FcnFailureCodeCount",
        "nonzero_failure_codes": "FcnNonzeroFailureCodes",
        "primary_interior_cusp_minimum": "FcnPrimaryCuspMin",
        "primary_interior_cusp_maximum": "FcnPrimaryCuspMax",
        "primary_interior_cusp_total": "FcnPrimaryCuspTotal",
        "exact_count_candidates": "FcnAnyExactCount",
        "geometry_registered_candidates": "FcnGeometryRegistered",
        "adiabatic_count": "FcnAdiabatic",
        "coupled_count": "FcnCoupled",
        "unique_state_count": "FcnUniqueStates",
        "power_or_performance_publication_count": "FcnPerformancePublications",
        "gpu_replay_required_count": "FcnGpuReplayRequired",
        "gpu_replay_pass_count": "FcnGpuReplayPassed",
        "gpu_replay_failed_count": "FcnGpuReplayFailed",
        "same_z_proxy_allowed": "FcnSameZProxyAllowed",
        "lineage_v1_evaluated": "FcnLineageVOneEvaluated",
        "lineage_v1_compatible": "FcnLineageVOneCompatible",
        "lineage_v1_preregistered": "FcnLineageVOnePreregistered",
        "lineage_validation_v1_attempted": "FcnLineageWcvalOneAttempted",
        "lineage_validation_v2_attempted": "FcnLineageWcvalTwoAttempted",
        "lineage_validation_v1_promoted": "FcnLineageWcvalOnePromoted",
        "lineage_validation_v2_promoted": "FcnLineageWcvalTwoPromoted",
        "tolerated_eol_file_count": "FcnToleratedEolFiles",
    },
    "topology-characterization-v1": {
        "classification": "TchClassification",
        "single_execution": "TchSingleExecution",
        "declared_count": "TchDeclared",
        "evaluated_count": "TchEvaluated",
        "failed_cases_count": "TchEvaluationFailures",
        "field_maps_per_design": "TchMapCount",
        "three_map_accepted_count": "TchThreeMapAccepted",
        "stable_eligible_cusp_count": "TchStableEligibleCusps",
        "stable_eligible_cell_count": "TchStableEligibleCells",
        "eligible_cusps_any_map": "TchEligibleCuspsAnyMap",
        "eligible_cells_any_map": "TchEligibleCellsAnyMap",
        "clustered_root_count": "TchClusteredRoots",
        "raw_detection_count": "TchRawDetections",
        "x_root_count": "TchXRoots",
        "o_root_count": "TchORoots",
        "degenerate_root_count": "TchDegenerateRoots",
        "plasma_channel_root_count": "TchChannelRoots",
        "plasma_channel_x_root_count": "TchChannelXRoots",
        "plasma_channel_unresolved_separatrix_count": "TchChannelUnresolved",
        "complete_correspondence_count": "TchCompleteCorrespondence",
        "stable_root_count": "TchStableRoots",
        "gpu_replay_required_count": "TchGpuReplayRequired",
        "gpu_replay_pass_count": "TchGpuReplayPassed",
        "mirror_probability_count": "TchMirrorProbabilityCount",
        "plasma_publication_count": "TchPlasmaPublicationCount",
        "not_a_design_optimization": "TchNotOptimization",
        "not_a_blind_validation": "TchNotValidation",
        "recommendation_not_validated_or_optimal": "TchRecommendationNotValidated",
        "tolerated_eol_file_count": "TchToleratedEolFiles",
    },
}
# Policy metrics every screening manifest must carry with exactly these values.
SCREENING_POLICY_METRICS = {
    "preregistered_one_shot": True,
    "hardware_or_experimental_validation": False,
    "permanent_magnet_material_model": False,
    "plasma_or_performance_claim_forbidden": True,
    "stable_multicell_wall_cusp_topology_demonstrated": False,
}

# Campaign manifest metric -> evidence macro whose raw artifact value it must equal.
WALL_LOSS_METRIC_MACROS = {
    "attempt_count": "WlfAttemptCount",
    "binding_gate_count": "WlfGateCount",
    "binding_gates_true": "WlfGatesTrue",
    "case_count": "WlfCaseCount",
    "classification": "WlfClassification",
    "coupling_integration_status": "WlfCouplingStatus",
    "cross_map_convergence_passed": "WlfCrossMapPassed",
    "exact_authority_replays": "WlfReplayCount",
    "hardware_or_experimental_validation": "WlfHardwareValidation",
    "incomplete_or_numerical_failure_count": "WlfPooledIncomplete",
    "interval_method": "WlfIntervalMethod",
    "launches_per_case": "WlfLaunchesPerCase",
    "maximum_relative_energy_error": "WlfEnergyErrorMax",
    "maximum_successive_probability_change": "WlfMaxSuccessiveChange",
    "mirror_formula_publication_forbidden": "WlfForbidMirror",
    "orbit_count": "WlfOrbitCount",
    "orbits_exceeding_energy_gate": "WlfEnergyExceed",
    "pic_or_self_consistent_claim_forbidden": "WlfForbidPic",
    "plasma_performance_publication_forbidden": "WlfForbidPerformance",
    "pooled_domain_escape_count": "WlfPooledEscape",
    "pooled_wall_hit_count": "WlfPooledWall",
    "reflected_count": "WlfPooledReflected",
    "relative_energy_gate": "WlfEnergyGate",
    "successive_probability_change_gate": "WlfGateThreshold",
    "terminal_state": "WlfTerminalState",
    "timestep_convergence_passed": "WlfTimestepPassed",
    "tolerated_crlf_sidecar_count": "WlfToleratedSidecars",
    "validator_calls_passed": "WlfValidatorsPassed",
    "validator_failures": "WlfValidatorsFailed",
    "wall_hit_probability_maximum": "WlfWallPMax",
    "wall_hit_probability_minimum": "WlfWallPMin",
}
WALL_LOSS_CELL_MACROS = {
    "exit_cell_escapes": "WlfCellFourEscape",
    "exit_cell_launches": "WlfCellFourTrials",
    "injector_cell_minus_direction_wall_fraction": "WlfCellOneMinusWallP",
    "injector_cell_plus_direction_launches": "WlfCellOnePlusTrials",
    "injector_cell_plus_direction_wall_hits": "WlfCellOnePlusWall",
    "interior_cells_launches": "WlfCellTwoThreeTrials",
    "interior_cells_wall_hits": "WlfCellTwoThreeWall",
}

EXPECTED_MANIFEST_TYPES = {
    "paper-L0-run-evidence-manifest": {
        "supported_versions": ["1.0"],
        "level": "L0",
        "required_file_roles": [
            "sweep-config",
            "first-results-report",
            "equation-ledger",
            "model-code",
            "reference-code",
            "cuda-code",
            "workflow-code",
            "cli-code",
            "dashboard-generator",
            "gallery-generator",
            "gallery-data",
            "accepted-html",
        ],
        "required_metrics": [
            "sample_count",
            "published_numeric_fields",
            "parity_mismatch_count",
            "failed_or_rejected_points",
            "raw_ranges",
            "maximum_cuda_absolute_residuals",
            "maximum_cuda_relative_residuals",
            "timing_controlled",
        ],
    },
    "paper-L1-result-manifest": {
        "supported_versions": ["1.0"],
        "level": "L1",
        "required_file_roles": [
            "equation-ledger",
            "closure-provenance",
            "geometry",
            "materials",
            "boundary-conditions",
            "solver-config",
            "result-data",
            "verification-report",
        ],
        "required_metrics": [
            "manufactured_solution_passed",
            "mesh_levels",
            "domain_levels",
            "convergence_reported",
            "numerical_uncertainty_reported",
            "l0_mapping_present",
            "failed_cases_count",
        ],
    },
    "paper-L2-result-manifest": {
        "supported_versions": ["1.0"],
        "level": "L2",
        "required_file_roles": [
            "equation-ledger",
            "closure-provenance",
            "coupling-contract",
            "solver-config",
            "result-data",
            "verification-report",
            "uncertainty-report",
        ],
        "required_metrics": [
            "interface_conservation_passed",
            "spatial_levels",
            "temporal_levels",
            "code_comparison_passed",
            "numerical_uncertainty_reported",
            "failed_cases_count",
            "uncertainty_components",
        ],
    },
    "paper-L3-result-manifest": {
        "supported_versions": ["1.0"],
        "level": "L3",
        "required_file_roles": [
            "pic-model-config",
            "collision-data",
            "boundary-data",
            "result-data",
            "verification-report",
            "experimental-protocol",
            "measurement-data",
            "facility-metadata",
            "uncertainty-report",
        ],
        "required_metrics": [
            "pic_convergence_passed",
            "preregistered_case_count",
            "withheld_validation_case_count",
            "measurement_uncertainty_reported",
            "facility_metadata_present",
            "applicability_domain_defined",
            "failed_cases_count",
        ],
    },
    "paper-test-particle-campaign-manifest": {
        "supported_versions": ["1.0"],
        "level": "numerical-campaign",
        "required_file_roles": [
            "authorities",
            "binding-gates",
            "campaign-result",
            "case-summary",
            "coupling-export",
            "execution-lock",
            "field-evidence",
            "field-map-convergence",
            "manufactured-gates",
            "p2-input-authority",
            "preregistered-authorities",
            "preregistered-protocol",
            "preregistered-shakedown",
            "probability-convergence",
            "protocol",
            "results-manifest",
            "shakedown",
            "terminal-record",
            "transition",
        ],
        "required_metrics": sorted(
            [
                *WALL_LOSS_METRIC_MACROS,
                "failed_cases_count",
                "per_cell_bimodality",
                "pooled_wall_hit_fraction_is_design_average",
                "preregistered_one_shot",
            ]
        ),
    },
    "paper-l1a-screening-manifest": {
        "supported_versions": ["1.0"],
        "level": "numerical-screening",
        "required_file_roles": [
            "execution-lock",
            "preregistered-protocol",
            "primary-dataset",
            "report",
            "representative-artifact",
            "results-manifest",
        ],
        "required_metrics": sorted(
            [
                *SCREENING_POLICY_METRICS,
                "classification",
                "declared_count",
                "evaluated_count",
                "failed_cases_count",
                "recorded_outcome",
                "screening_model",
                "tolerated_eol_file_count",
            ]
        ),
    },
}

PLACEHOLDERS = {
    "generic task marker": re.compile(r"\b(?:TODO|TBD|TK|FIXME|XXX)\b"),
    "insert marker": re.compile(r"\[(?:insert|add|replace|fill)[^\]]*\]", re.IGNORECASE),
    "angle-bracket placeholder": re.compile(
        r"<(?:author|affiliation|title|date|value|citation|insert)[^>]*>",
        re.IGNORECASE,
    ),
    "dummy prose": re.compile(r"\blorem ipsum\b|\byour name here\b", re.IGNORECASE),
}

FORBIDDEN_MODEL_WORDING = {
    "L0 presented as one-dimensional": re.compile(
        r"\bL0\s+(?:is|was|provides|constitutes)\s+(?:an?\s+)?"
        r"(?:one[- ]dimensional|1D)\b",
        re.IGNORECASE,
    ),
    "L0 presented as geometrically predictive": re.compile(
        r"\bL0\s+(?:is|was)\s+(?:geometrically predictive|geometry-resolving)\b"
        r"|\bL0\s+predicts?\s+(?:the\s+)?geometry\b",
        re.IGNORECASE,
    ),
    "L0 presented as physically calibrated": re.compile(
        r"\bL0\s+(?:is|was|has been)\s+physically calibrated\b",
        re.IGNORECASE,
    ),
    "implementations presented as independent": re.compile(
        r"\bindependent\s+(?:Python\s*(?:/|and)\s*(?:CUDA|Warp)|"
        r"(?:Python|CUDA|Warp)\s+implementations?)\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Macro:
    name: str
    arguments: tuple[str, ...]
    start: int
    end: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def _git_bytes(repo: Path, revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip())
    return completed.stdout


def _resolves_to_commit(repo: Path, revision: object) -> bool:
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return False
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == revision


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: invalid or unreadable JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path}: top-level JSON value must be an object")
        return {}
    return value


def _parse_group(text: str, position: int) -> tuple[str, int]:
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        raise ValueError("expected braced macro argument")
    depth = 1
    start = position + 1
    position += 1
    while position < len(text):
        if text[position] == "{" and (position == 0 or text[position - 1] != "\\"):
            depth += 1
        elif text[position] == "}" and (position == 0 or text[position - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start:position], position + 1
        position += 1
    raise ValueError("unterminated braced macro argument")


def extract_macros(text: str, name: str, argument_count: int) -> list[Macro]:
    token = f"\\{name}"
    macros: list[Macro] = []
    position = 0
    while True:
        start = text.find(token, position)
        if start < 0:
            return macros
        after = start + len(token)
        if after < len(text) and text[after].isalpha():
            position = after
            continue
        arguments: list[str] = []
        cursor = after
        try:
            for _ in range(argument_count):
                argument, cursor = _parse_group(text, cursor)
                arguments.append(argument)
        except ValueError:
            position = after
            continue
        macros.append(Macro(name, tuple(arguments), start, cursor))
        position = cursor


def _mask_spans(text: str, macros: list[Macro]) -> str:
    characters = list(text)
    for macro in macros:
        for index in range(macro.start, macro.end):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def _normalize_tex(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _heading_at(manuscript: str, position: int) -> str:
    abstract_start = manuscript.find("\\begin{abstract}")
    abstract_end = manuscript.find("\\end{abstract}")
    if abstract_start <= position <= abstract_end:
        return "Abstract"
    matches = list(
        re.finditer(r"\\(?:sub)*section\{([^{}]+)\}", manuscript[:position])
    )
    return matches[-1].group(1) if matches else "Preamble"


SECTION_INPUT = re.compile(r"\\input\{(sections/[^}]+)\}")


def flatten_sections(repo: Path, manuscript: str, errors: list[str]) -> str:
    """Inline every ``\\input{sections/...}`` so section prose faces the same checks."""

    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        path = repo / "paper" / relative
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"manuscript.tex: cannot read input section {relative!r}: {exc}")
            return ""

    return SECTION_INPUT.sub(replace, manuscript)


def section_literal_digits(section: str, macro_prefix: str) -> list[str]:
    """Return every digit typed into a macro-only section outside structural syntax."""

    body = re.sub(r"(?m)(?<!\\)%.*$", "", section)
    body = re.sub(r"\\(?:label|ref)\{[^}]*\}", "", body)
    body = re.sub(r"\\EvidenceClaim\{CLM-\d+\}", "", body)
    body = re.sub(rf"\\{re.escape(macro_prefix)}[A-Za-z]+", "", body)
    body = re.sub(r"\bP2\b", "", body)  # the P2 field-reference level is a name, not a value
    body = re.sub(r"\\begin\{minipage\}\{[^}]*\}", "", body)  # layout width, not a result
    return re.findall(r"\d", body)


def tex_unescape(value: str) -> str:
    """Undo the generator's identifier escaping (``\\_\\allowbreak{}`` and friends)."""

    return (
        value.replace("\\allowbreak{}", "")
        .replace("\\textbackslash{}", "\\")
        .replace("\\_", "_")
        .replace("\\%", "%")
        .replace("\\&", "&")
        .replace("\\#", "#")
        .replace("\\{", "{")
        .replace("\\}", "}")
    )


def find_unregistered_claims(text: str) -> list[str]:
    """Return risk labels for claim-bearing prose outside structured macros."""

    protected = extract_macros(text, "EvidenceClaim", 2)
    protected += extract_macros(text, "ArtifactClaim", 3)
    protected += extract_macros(text, "EvidenceGate", 2)
    exposed = _mask_spans(text, protected)
    exposed = re.sub(r"(?m)%.*$", "", exposed)
    findings: list[str] = []

    quantitative = re.compile(
        r"(?i)(?<![A-Za-z])\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?"
        r"(?:\s*(?:~|\\,|-)\s*|\s+)(?:points?(?:/s)?|records?|fields?|files?|"
        r"seconds?|milliseconds?|ms|percent|%|[WVASN](?![A-Za-z])|[x×](?![A-Za-z]))"
    )
    if quantitative.search(exposed):
        findings.append("unregistered quantitative claim")

    normalized = re.sub(r"[^a-z0-9×%]+", " ", exposed.casefold())
    experimental_patterns = (
        r"\b(?:experimental\w*|measur\w*)\b(?:\s+\w+){0,8}\s+"
        r"(?:accur\w*|validat(?:ed|es)|agreement|predict\w*)\b",
        r"\b(?:accur\w*|validat(?:ed|es)|agreement|predict\w*)\b"
        r"(?:\s+\w+){0,8}\s+(?:experimental\w*|measur\w*)\b",
    )
    if any(re.search(pattern, normalized) for pattern in experimental_patterns):
        findings.append("unregistered experimental accuracy or validation claim")

    accelerator_patterns = (
        r"\b(?:cuda|gpu)\b(?:\s+\w+){0,8}\s+"
        r"(?:speedup\b|faster\b|slower\b|accelerat\w*\b|throughput\b|"
        r"[0-9]+\s*[x×](?=\s|$))",
        r"(?:\b[0-9]+\s*[x×](?=\s|$)|\bten\s+fold\b|\bspeedup\b|"
        r"\bfaster\b|\bslower\b|\baccelerat\w*\b)"
        r"(?:\s+\w+){0,8}\s+(?:cuda|gpu)\b",
    )
    if any(re.search(pattern, normalized) for pattern in accelerator_patterns):
        findings.append("unregistered GPU/CUDA performance claim")

    if re.search(
        r"\b(?:python\s*(?:/|and)\s*(?:cuda|warp)|(?:cuda|warp)\s+and\s+python)"
        r"[^.\n]{0,80}\b(?:parity|agreement)\b",
        exposed,
        re.IGNORECASE,
    ):
        findings.append("unregistered cross-backend parity claim")
    return findings


def _check_text_policy(repo: Path, errors: list[str]) -> None:
    paths = sorted((repo / "paper").rglob("*.tex"))
    paths += sorted((repo / "paper").rglob("*.md"))
    paths += sorted((repo / "modern/docs/workstreams").glob("paper-*.md"))
    for path in paths:
        if "build" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(repo)
        for label, pattern in FORBIDDEN_MODEL_WORDING.items():
            if match := pattern.search(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: forbidden wording: {label}")
        for label, pattern in PLACEHOLDERS.items():
            if match := pattern.search(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: unreplaced placeholder: {label}")


def _check_bibliography(repo: Path, manuscript: str, errors: list[str]) -> set[str]:
    bib_text = (repo / "paper/references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))
    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", manuscript):
        citation_keys.update(key.strip() for key in group.split(",") if key.strip())
    for key in sorted(citation_keys - bib_keys):
        errors.append(f"manuscript.tex: missing bibliography key {key!r}")
    for key in sorted(bib_keys - citation_keys):
        errors.append(f"references.bib: uncited entry {key!r}")

    starts = list(re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(bib_text)
        entry = bib_text[match.start():end]
        key = match.group(1)
        doi_match = re.search(r"\bdoi\s*=\s*\{([^}]+)\}", entry, re.IGNORECASE)
        if doi_match:
            doi = doi_match.group(1)
            if not re.fullmatch(r"10\.\d{4,9}/\S+", doi, re.IGNORECASE):
                errors.append(f"references.bib: malformed DOI for {key!r}")
            if f"https://doi.org/{doi}".casefold() not in entry.casefold():
                errors.append(f"references.bib: DOI resolver URL missing for {key!r}")
        elif not re.search(r"\bno DOI\b", entry, re.IGNORECASE):
            errors.append(f"references.bib: DOI status missing for {key!r}")
    return citation_keys


def _check_revision_chain(
    repo: Path, base_revision: str, manifest_revision: object, errors: list[str], label: str
) -> None:
    if not _resolves_to_commit(repo, manifest_revision):
        errors.append(f"{label}: evidence_revision must be a resolvable 40-hex commit")
        return
    head = _run_git(repo, "rev-parse", "HEAD")
    revision = str(manifest_revision)
    if not _is_ancestor(repo, base_revision, revision):
        errors.append(f"{label}: base evidence revision is not an ancestor")
    if not _is_ancestor(repo, revision, head):
        errors.append(f"{label}: evidence revision is not an ancestor of HEAD")


def _validate_source_files(
    repo: Path,
    revision: str,
    sources: object,
    required_roles: set[str],
    errors: list[str],
    label: str,
) -> None:
    if not isinstance(sources, list) or not sources:
        errors.append(f"{label}: source_files must be a non-empty array")
        errors.append(
            f"{label}: missing required file roles: {', '.join(sorted(required_roles))}"
        )
        return
    roles: set[str] = set()
    paths: set[str] = set()
    for index, source in enumerate(sources):
        source_label = f"{label}: source_files[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{source_label} must be an object")
            continue
        role = source.get("role")
        path = source.get("path")
        blob = source.get("git_blob")
        digest = source.get("git_blob_sha256")
        if not all(isinstance(value, str) for value in (role, path, blob, digest)):
            errors.append(f"{source_label} lacks string role/path/hash fields")
            continue
        if path in paths:
            errors.append(f"{source_label}: duplicate path {path!r}")
            continue
        paths.add(path)
        roles.add(role)
        if Path(path).is_absolute() or ".." in Path(path).parts:
            errors.append(f"{source_label}: path must be repository-relative")
            continue
        if not re.fullmatch(r"[0-9a-f]{40}", blob):
            errors.append(f"{source_label}: git_blob must be 40-hex")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{source_label}: git_blob_sha256 must be 64-hex")
        try:
            actual_blob = _run_git(repo, "rev-parse", f"{revision}:{path}")
            content = _git_bytes(repo, revision, path)
        except RuntimeError as exc:
            errors.append(f"{source_label}: cannot resolve committed source: {exc}")
            continue
        if actual_blob != blob:
            errors.append(f"{source_label}: Git blob mismatch")
        if sha256_bytes(content) != digest:
            errors.append(f"{source_label}: SHA-256 mismatch")
    missing = sorted(required_roles - roles)
    if missing:
        errors.append(f"{label}: missing required file roles: {', '.join(missing)}")


def _check_metric_constraints(
    metrics: object, constraints: object, errors: list[str], label: str
) -> None:
    if not isinstance(metrics, dict):
        errors.append(f"{label}: metrics must be an object")
        return
    if not isinstance(constraints, dict):
        errors.append(f"{label}: metric_constraints must be an object")
        return
    for name, rule in constraints.items():
        if name not in metrics:
            errors.append(f"{label}: required metric {name!r} is missing")
            continue
        value = metrics[name]
        if "equals" in rule and value != rule["equals"]:
            errors.append(f"{label}: metric {name!r} does not equal required value")
        if "integer_minimum" in rule:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{label}: metric {name!r} must be an integer")
            elif value < rule["integer_minimum"]:
                errors.append(f"{label}: metric {name!r} is below its minimum")
        if "contains_all" in rule:
            if not isinstance(value, list) or not set(rule["contains_all"]) <= set(value):
                errors.append(f"{label}: metric {name!r} lacks required components")


def _validate_manifest_payload(
    repo: Path,
    base_revision: str,
    gate: dict[str, Any],
    payload: object,
    manifest_path: Path,
    errors: list[str],
    *,
    require_committed: bool,
) -> None:
    """Validate one accepted gate manifest; exposed for adversarial tests."""

    label = f"{gate.get('id', 'unknown gate')} manifest"
    normalized = manifest_path.as_posix()
    if not normalized.startswith("paper/evidence/manifests/") or manifest_path.suffix != ".json":
        errors.append(f"{label}: path must be a JSON file under paper/evidence/manifests")
    if not isinstance(payload, dict):
        errors.append(f"{label}: payload must be a JSON object")
        return
    expected_type = gate.get("required_manifest_document_type")
    expected_version = gate.get("required_manifest_schema_version")
    if payload.get("document_type") != expected_type:
        errors.append(f"{label}: unrecognized document_type")
    if payload.get("schema_version") != expected_version:
        errors.append(f"{label}: unsupported schema_version")
    schema = EXPECTED_MANIFEST_TYPES.get(str(expected_type))
    if schema is None or expected_version not in schema["supported_versions"]:
        errors.append(f"{label}: type/version is absent from the compiled schema registry")
        return
    if payload.get("level") != schema["level"]:
        errors.append(f"{label}: level does not match recognized manifest type")
    if payload.get("status") != "accepted":
        errors.append(f"{label}: status must be accepted")
    if not isinstance(payload.get("manifest_id"), str) or not payload["manifest_id"]:
        errors.append(f"{label}: manifest_id is required")
    revision = payload.get("evidence_revision")
    _check_revision_chain(repo, base_revision, revision, errors, label)
    if isinstance(revision, str) and _resolves_to_commit(repo, revision):
        _validate_source_files(
            repo,
            revision,
            payload.get("source_files"),
            set(schema["required_file_roles"]),
            errors,
            label,
        )
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for required in schema["required_metrics"]:
            if required not in metrics:
                errors.append(f"{label}: required metric {required!r} is missing")
    _check_metric_constraints(metrics, gate.get("metric_constraints"), errors, label)

    if require_committed:
        relative = manifest_path.as_posix()
        try:
            committed_blob = _run_git(repo, "rev-parse", f"HEAD:{relative}")
            working_blob = _run_git(repo, "hash-object", "--", relative)
        except RuntimeError as exc:
            errors.append(f"{label}: accepted manifest is not committed at HEAD: {exc}")
        else:
            if committed_blob != working_blob:
                errors.append(f"{label}: working manifest differs from committed blob")


def _check_schema_registry(repo: Path, errors: list[str]) -> None:
    registry = _load_json(repo / "paper/evidence/manifest-schemas.json", errors)
    if registry.get("document_type") != "paper-evidence-manifest-schema-registry":
        errors.append("manifest-schemas.json: wrong document_type")
    if registry.get("schema_version") != "1.0":
        errors.append("manifest-schemas.json: unsupported schema_version")
    if registry.get("manifest_types") != EXPECTED_MANIFEST_TYPES:
        errors.append("manifest-schemas.json: registry differs from compiled strict schema")


def _check_l0_manifest(repo: Path, errors: list[str]) -> dict[str, Any]:
    path = repo / "paper/evidence/l0-run-manifest.json"
    manifest = _load_json(path, errors)
    if not manifest:
        return {}
    schema = EXPECTED_MANIFEST_TYPES["paper-L0-run-evidence-manifest"]
    if manifest.get("document_type") != "paper-L0-run-evidence-manifest":
        errors.append("l0-run-manifest.json: wrong document_type")
    if manifest.get("schema_version") != "1.0" or manifest.get("level") != "L0":
        errors.append("l0-run-manifest.json: unsupported schema or level")
    revision = manifest.get("evidence_revision")
    base = "41bf909127dc021abe8078fd77a98aa3a6e4cf33"
    _check_revision_chain(repo, base, revision, errors, "L0 manifest")
    if isinstance(revision, str) and _resolves_to_commit(repo, revision):
        _validate_source_files(
            repo,
            revision,
            manifest.get("source_files"),
            set(schema["required_file_roles"]),
            errors,
            "L0 manifest",
        )
    if manifest.get("run_revision", {}).get("value", object()) is not None:
        errors.append("L0 manifest: unrecorded run revision must remain null")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("L0 manifest: metrics must be an object")
        return manifest
    for required in schema["required_metrics"]:
        if required not in metrics:
            errors.append(f"L0 manifest: required metric {required!r} is missing")
    if metrics.get("sample_count") != 8192:
        errors.append("L0 manifest: sample_count must match accepted evidence")
    if metrics.get("published_numeric_fields") != 26:
        errors.append("L0 manifest: published_numeric_fields mismatch")
    if metrics.get("parity_mismatch_count") != 0:
        errors.append("L0 manifest: parity_mismatch_count mismatch")
    if metrics.get("failed_or_rejected_points") != 0:
        errors.append("L0 manifest: failed/rejected count mismatch")
    if metrics.get("timing_controlled") is not False:
        errors.append("L0 manifest: timing must remain explicitly uncontrolled")
    caveat = str(metrics.get("timing_caveat", "")).casefold()
    if "neither gpu speedup nor slowdown" not in caveat:
        errors.append("L0 manifest: timing caveat must prohibit speedup and slowdown")

    try:
        html_source = next(
            source
            for source in manifest["source_files"]
            if source["role"] == "accepted-html"
        )
        html = _git_bytes(repo, str(revision), html_source["path"])
        payload = generate_tables._embedded_payload(html)
        contract = manifest["accepted_html"]
        raw = contract["raw_per_point_output"]
        columns = payload["columns"]
        if payload["documentType"] != contract["embedded_document_type"]:
            errors.append("L0 manifest: accepted HTML document type mismatch")
        if payload["schemaVersion"] != contract["embedded_schema_version"]:
            errors.append("L0 manifest: accepted HTML schema mismatch")
        if payload["sampleCount"] != raw["sample_count"]:
            errors.append("L0 manifest: accepted HTML sample count mismatch")
        if len(columns) != raw["column_count"]:
            errors.append("L0 manifest: accepted HTML column count mismatch")
        if {len(values) for values in columns.values()} != {raw["all_column_lengths"]}:
            errors.append("L0 manifest: accepted HTML column lengths mismatch")
        dataset_sha = (
            payload["operatingConceptGallery"]["source"]["dataset_identity"]["sha256"]
        )
        if dataset_sha != raw["dataset_sha256"]:
            errors.append("L0 manifest: accepted HTML dataset SHA-256 mismatch")
        if payload["firstRunParity"]["comparedCount"] != metrics["sample_count"]:
            errors.append("L0 manifest: HTML parity count mismatch")
        if payload["firstRunParity"]["publishedNumericFields"] != metrics[
            "published_numeric_fields"
        ]:
            errors.append("L0 manifest: HTML numeric-field count mismatch")
        if payload["firstRunParity"]["mismatchCount"] != metrics[
            "parity_mismatch_count"
        ]:
            errors.append("L0 manifest: HTML parity mismatch count differs")
        range_map = {
            "axial_thrust_n": "thrust",
            "specific_impulse_s": "isp",
            "beam_current_a": "beamCurrent",
            "anode_input_w": "anodePower",
            "beam_kinetic_power_w": "beamPower",
            "ppu_input_to_beam_efficiency": "ppuEfficiency",
        }
        for manifest_key, html_key in range_map.items():
            registered = metrics["raw_ranges"][manifest_key]
            embedded = payload["ranges"][html_key]
            if registered["minimum"] != embedded["minimum"]:
                errors.append(f"L0 manifest: {manifest_key} minimum mismatch")
            if registered["maximum"] != embedded["maximum"]:
                errors.append(f"L0 manifest: {manifest_key} maximum mismatch")
        if payload["provenance"]["timing_controlled"] is not False:
            errors.append("L0 manifest: accepted HTML timing is not diagnostic")
    except (KeyError, StopIteration, TypeError, RuntimeError, ValueError) as exc:
        errors.append(f"L0 manifest: accepted HTML validation failed: {exc}")
    return manifest


def _check_wall_loss_campaign(
    repo: Path,
    gate: dict[str, Any],
    payload: dict[str, Any],
    manuscript: str,
    flattened: str,
    matrix: dict[str, Any],
    errors: list[str],
) -> None:
    """Verify the admitted collisionless test-particle wall-loss campaign end to end.

    The typed manifest has already passed ``_validate_manifest_payload``.  This
    check binds it to the sealed results bundle, the hash-bound evidence file,
    the generated macros, the section prose and the claim matrix:

    * the evidence file, generated TeX and sidecar regenerate byte-identically
      from the bundle, and every artifact hash recorded in the evidence file
      matches the bundle on disk;
    * every manifest metric equals the raw artifact value behind its macro;
    * preregistration, results and post-hoc audit revisions resolve, chain, and
      bind the frozen files;
    * the section is bound exactly once, uses only generated macros, types no
      literal digit, names the classification string, and carries the
      registered non-claims of its claim records;
    * the manuscript's full-revision macro equals the manifest revision.
    """

    gate_id = str(gate.get("id"))
    label = f"{gate_id} campaign"
    try:
        evidence_bytes, tex_bytes, sidecar_bytes = wall_loss_v4.render(repo)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: evidence regeneration from the sealed bundle failed: {exc}")
        return
    evidence = json.loads(evidence_bytes)
    evidence_meta = payload.get("paper_evidence_file")
    if not isinstance(evidence_meta, dict) or not isinstance(evidence_meta.get("path"), str):
        errors.append(f"{label}: manifest lacks paper_evidence_file.path")
        return
    evidence_path = repo / evidence_meta["path"]
    generated_path = repo / wall_loss_v4.OUTPUT_PATH
    sidecar_path = repo / wall_loss_v4.SIDECAR_PATH
    for path, expected, name in (
        (evidence_path, evidence_bytes, "evidence file"),
        (generated_path, tex_bytes, "generated TeX"),
        (sidecar_path, sidecar_bytes, "provenance sidecar"),
    ):
        if not path.is_file() or path.read_bytes() != expected:
            errors.append(f"{label}: committed {name} differs from regeneration")
    if evidence_meta.get("document_type") != evidence.get("document_type"):
        errors.append(f"{label}: evidence document_type differs from the manifest")
    if evidence_meta.get("macro_count") != len(evidence.get("macros", [])):
        errors.append(f"{label}: evidence macro count differs from the manifest")

    # Evidence-file artifact hashes against the bundle on disk (independent of the generator).
    results_root = repo / wall_loss_v4.RESULTS
    for relative, meta in evidence.get("artifacts", {}).items():
        artifact = results_root / relative
        if not artifact.is_file():
            errors.append(f"{label}: evidence artifact missing on disk: {relative}")
            continue
        raw = artifact.read_bytes()
        if sha256_bytes(raw) != meta.get("sha256") or len(raw) != meta.get("bytes"):
            errors.append(f"{label}: evidence artifact hash mismatch: {relative}")
    bundle = payload.get("results_bundle", {})
    bundle_manifest = results_root / "manifest.json"
    if not bundle_manifest.is_file():
        errors.append(f"{label}: results manifest is missing on disk")
    else:
        digest = sha256_bytes(bundle_manifest.read_bytes())
        if digest != evidence["bundle"]["manifest_sha256"] or digest != bundle.get("manifest_sha256"):
            errors.append(f"{label}: results manifest SHA-256 differs from the evidence bindings")
    if evidence["bundle"]["tolerated_crlf_sidecars"] != bundle.get("tolerated_crlf_sidecars"):
        errors.append(f"{label}: tolerated sidecar list differs from the evidence file")

    # Revisions.
    revision = str(payload.get("evidence_revision"))
    if evidence.get("evidence_revision") != revision:
        errors.append(f"{label}: evidence file revision differs from the manifest")
    try:
        committed_blob = _run_git(repo, "rev-parse", f"{revision}:{evidence['bundle']['manifest_path']}")
    except RuntimeError as exc:
        errors.append(f"{label}: results manifest is not committed at the evidence revision: {exc}")
        committed_blob = None
    if committed_blob is not None and (
        committed_blob != evidence["binding"]["manifest_git_blob"]
        or committed_blob != bundle.get("manifest_git_blob")
    ):
        errors.append(f"{label}: results manifest Git blob differs from the evidence bindings")
    try:
        results_tree = _run_git(repo, "rev-parse", f"{revision}:{wall_loss_v4.RESULTS.as_posix()}")
        head_tree = _run_git(repo, "rev-parse", f"HEAD:{wall_loss_v4.RESULTS.as_posix()}")
    except RuntimeError as exc:
        errors.append(f"{label}: results tree cannot be resolved: {exc}")
    else:
        if results_tree != bundle.get("results_tree"):
            errors.append(f"{label}: results tree differs from the manifest binding")
        if head_tree != results_tree:
            errors.append(f"{label}: results tree changed after the evidence revision")
    prereg = payload.get("preregistration_revision")
    if not _resolves_to_commit(repo, prereg):
        errors.append(f"{label}: preregistration_revision is not resolvable")
    else:
        prereg = str(prereg)
        if evidence["binding"].get("preregistration_commit") != prereg:
            errors.append(f"{label}: evidence preregistration commit differs from the manifest")
        if not _is_ancestor(repo, prereg, revision) or prereg == revision:
            errors.append(f"{label}: preregistration must strictly precede the results revision")
        for source in payload.get("source_files", []):
            if isinstance(source, dict) and str(source.get("role", "")).startswith("preregistered-"):
                try:
                    frozen = _run_git(repo, "rev-parse", f"{prereg}:{source['path']}")
                except RuntimeError as exc:
                    errors.append(f"{label}: frozen file missing at preregistration: {exc}")
                    continue
                if frozen != source.get("git_blob"):
                    errors.append(f"{label}: {source['path']} changed after preregistration")
    audit = payload.get("posthoc_audit")
    if not isinstance(audit, dict) or not _resolves_to_commit(repo, audit.get("revision")):
        errors.append(f"{label}: posthoc_audit must bind a resolvable revision")
    else:
        audit_revision = str(audit["revision"])
        head = _run_git(repo, "rev-parse", "HEAD")
        if not _is_ancestor(repo, revision, audit_revision) or not _is_ancestor(repo, audit_revision, head):
            errors.append(f"{label}: posthoc audit revision does not chain results -> audit -> HEAD")
        _validate_source_files(
            repo, audit_revision, [audit], {"posthoc-audit"}, errors, f"{label} posthoc audit"
        )

    # Metrics against the raw artifact values behind the macros.
    raw = {item["name"]: item["raw"] for item in evidence.get("macros", [])}
    values = {item["name"]: item["value"] for item in evidence.get("macros", [])}
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{label}: metrics must be an object")
        return
    for metric, macro in WALL_LOSS_METRIC_MACROS.items():
        if macro not in raw:
            errors.append(f"{label}: evidence lacks macro {macro}")
        elif metrics.get(metric) != raw[macro]:
            errors.append(f"{label}: metric {metric!r} differs from artifact value")
    cells = metrics.get("per_cell_bimodality")
    if not isinstance(cells, dict):
        errors.append(f"{label}: per_cell_bimodality must be an object")
    else:
        for metric, macro in WALL_LOSS_CELL_MACROS.items():
            if cells.get(metric) != raw.get(macro):
                errors.append(f"{label}: per-cell metric {metric!r} differs from artifact value")
    if metrics.get("preregistered_one_shot") is not True or raw.get("WlfAttemptCount") != 1:
        errors.append(f"{label}: campaign must be a single preregistered attempt")
    if metrics.get("pooled_wall_hit_fraction_is_design_average") is not True:
        errors.append(f"{label}: pooled fraction must be declared a design average")
    try:
        campaign = json.loads((results_root / "artifacts/campaign-result.json").read_bytes())
        failed = sum(
            1
            for case in wall_loss_v4.CASES
            if campaign["campaigns"][case]["incomplete"]["successes"] != 0
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: cannot derive failed_cases_count: {exc}")
    else:
        if metrics.get("failed_cases_count") != failed:
            errors.append(f"{label}: failed_cases_count differs from the campaign result")
    classification = payload.get("classification")
    expected = gate.get("metric_constraints", {}).get("classification", {}).get("equals")
    if classification != wall_loss_v4.CLASSIFICATION or expected != classification:
        errors.append(f"{label}: classification differs between gate, manifest and generator")
    if evidence.get("classification") != classification:
        errors.append(f"{label}: evidence classification differs from the manifest")
    if tex_unescape(values.get("WlfClassification", "")) != classification:
        errors.append(f"{label}: WlfClassification macro does not render the classification string")

    # Manuscript bindings.
    binding = gate.get("accepted_manuscript_binding")
    if not isinstance(binding, str) or manuscript.count(binding) != 1:
        errors.append(f"{label}: section binding must occur exactly once in manuscript.tex")
    generated_binding = wall_loss_v4.GENERATED_BINDING
    document_start = manuscript.find("\\begin{document}")
    if manuscript.count(generated_binding) != 1 or manuscript.find(generated_binding) > document_start:
        errors.append(f"{label}: generated macro file must be input exactly once in the preamble")
    macro_name = gate.get("manuscript_revision_macro")
    if not isinstance(macro_name, str):
        errors.append(f"{label}: gate lacks manuscript_revision_macro")
    else:
        definitions = [
            macro
            for macro in extract_macros(manuscript, "newcommand", 2)
            if macro.arguments[0] == f"\\{macro_name}"
        ]
        rendered = ""
        if len(definitions) == 1:
            body = re.sub(r"(?m)(?<!\\)%.*$", "", definitions[0].arguments[1])
            rendered = re.sub(r"\\texttt\{|\}|\s", "", tex_unescape(body))
        if rendered != revision:
            errors.append(f"{label}: \\{macro_name} does not spell the manifest revision")

    # Section content.
    section_path = repo / wall_loss_v4.SECTION_PATH
    try:
        section = section_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{label}: section unreadable: {exc}")
        return
    heading = gate.get("section_heading")
    if not isinstance(heading, str) or f"\\subsection{{{heading}}}" not in section:
        errors.append(f"{label}: section heading differs from the gate registration")
    prefix = str(evidence_meta.get("macro_prefix", "Wlf"))
    defined = set(re.findall(rf"\\newcommand\{{\\({re.escape(prefix)}[A-Za-z]+)\}}", tex_bytes.decode("utf-8")))
    used = set(re.findall(rf"\\({re.escape(prefix)}[A-Za-z]+)", section))
    if not used:
        errors.append(f"{label}: section uses no evidence macro")
    for name in sorted(used - defined):
        errors.append(f"{label}: section uses undefined macro \\{name}")
    for required in ("WlfCaseTable", "WlfCellTable", "WlfClassification"):
        if required not in used:
            errors.append(f"{label}: section must use \\{required}")
    digits = section_literal_digits(section, prefix)
    if digits:
        errors.append(f"{label}: section types {len(digits)} literal digit(s); every number must be a macro")
    if "\\input{" in re.sub(r"(?m)(?<!\\)%.*$", "", section):
        errors.append(f"{label}: section must not input further files")
    for finding in find_unregistered_claims(section):
        errors.append(f"{label}: {finding}")

    # Claim-matrix cross-references.
    integration = evidence.get("manuscript_integration", {})
    if integration.get("status") != "admitted":
        errors.append(f"{label}: evidence file does not record admission")
    if integration.get("gate_id") != gate_id or integration.get("manifest_id") != payload.get("manifest_id"):
        errors.append(f"{label}: evidence file names a different gate or manifest")
    if integration.get("manifest_path") != gate.get("manifest_path"):
        errors.append(f"{label}: evidence file names a different manifest path")
    if integration.get("section_binding") != binding:
        errors.append(f"{label}: evidence file names a different section binding")
    records = {
        claim.get("id"): claim
        for claim in matrix.get("claims", [])
        if isinstance(claim, dict) and isinstance(claim.get("id"), str)
    }
    manifest_id = payload.get("manifest_id")
    section_claims = set(re.findall(r"\\EvidenceClaim\{(CLM-\d+)\}", section))
    prose_ids = integration.get("prose_claim_ids", [])
    if not section_claims or not section_claims <= set(prose_ids):
        errors.append(f"{label}: section claims are not all registered as campaign prose claims")
    normalized_section = _normalize_tex(section)
    for claim_id in prose_ids:
        record = records.get(claim_id)
        if record is None or record.get("status") != "verified":
            errors.append(f"{label}: prose claim {claim_id} is not a verified claim record")
            continue
        if manifest_id not in record.get("manifest_ids", []):
            errors.append(f"{label}: claim {claim_id} is not bound to manifest {manifest_id}")
        if not isinstance(record.get("authorized_tex"), str):
            errors.append(f"{label}: claim {claim_id} must be a prose claim")
        if "classification" in record and record["classification"] != classification:
            errors.append(f"{label}: claim {claim_id} names a different classification")
        for phrase in record.get("non_claims", []):
            if _normalize_tex(str(phrase)) not in normalized_section:
                errors.append(f"{label}: non-claim of {claim_id} is absent from the section: {phrase!r}")
        if claim_id in section_claims and heading not in record.get("allowed_locations", []):
            errors.append(f"{label}: claim {claim_id} does not allow the section heading")
    if not any(records.get(claim_id, {}).get("non_claims") for claim_id in prose_ids):
        errors.append(f"{label}: no campaign claim registers non_claims")
    artifact_claim = integration.get("artifact_claim_id")
    record = records.get(artifact_claim, {})
    if integration.get("artifact_id") not in record.get("authorized_artifact_ids", []):
        errors.append(f"{label}: artifact claim {artifact_claim} does not authorize the generated tables")
    if manifest_id not in record.get("manifest_ids", []):
        errors.append(f"{label}: artifact claim {artifact_claim} is not bound to manifest {manifest_id}")
    if flattened.count(f"\\subsection{{{heading}}}") != 1:
        errors.append(f"{label}: section heading must appear exactly once in the flattened manuscript")


def _eol_audited_file_errors(repo: Path, spec: Any, label: str) -> list[str]:
    """Recompute the audited end-of-line rule for exactly the audited files of one experiment.

    The rule is the one written into each experiment's ``POSTHOC_AUDIT.md`` and
    verification module: the checkout bytes contain no CR, hash to the audited LF
    digest, and their LF->CRLF transform hashes to the recorded digest with the
    recorded byte count.  The digests must also appear verbatim in the audited
    module (``protocol.py::EOL_AUDITED_SIDECARS`` for the sweep,
    ``audit_sidecar_eol.py`` for the four-cell search) so the paper's tolerance
    can never drift from the experiment's own.
    """

    errors: list[str] = []
    for relative, audited in sorted(spec.audited_eol_files.items()):
        path = repo / spec.experiment_path / relative
        if not path.is_file():
            errors.append(f"{label}: audited file missing on disk: {relative}")
            continue
        raw = path.read_bytes()
        crlf = raw.replace(b"\n", b"\r\n")
        if b"\r" in raw or sha256_bytes(raw) != audited.lf_sha256:
            errors.append(f"{label}: audited file is not the LF bytes of the audit: {relative}")
        if sha256_bytes(crlf) != audited.recorded_sha256 or len(crlf) != audited.recorded_bytes:
            errors.append(f"{label}: audited file does not reproduce the recorded CRLF digest: {relative}")
        module = repo / audited.audit_module
        try:
            text = module.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{label}: audit module unreadable: {exc}")
            continue
        if f'"{audited.lf_sha256}"' not in text or f'"{audited.recorded_sha256}"' not in text:
            errors.append(f"{label}: audit module does not bind the audited digests: {audited.audit_module}")
    return errors


def _check_topology_screening(
    repo: Path,
    gate: dict[str, Any],
    payload: dict[str, Any],
    manuscript: str,
    flattened: str,
    matrix: dict[str, Any],
    errors: list[str],
) -> None:
    """Verify one admitted L1a topology-screening study end to end.

    Mirrors ``_check_wall_loss_campaign`` for the ``numerical-screening`` gate
    kind: byte-identical regeneration of evidence/TeX/sidecar from the sealed
    bundle, artifact hashes on disk with the audited end-of-line rule for exactly
    the audited files, metric == raw macro value, results tree unchanged,
    preregistration/results/audit revision chains, frozen files unchanged, the
    macro-only section with no literal digit, the classification macro, the
    registered non-claims, bindings exactly once, the revision macro, and the
    claim-matrix cross-references.  In addition the recorded outcome (accepted
    screening, preregistered null or recorded characterization) must agree
    between gate, manifest, evidence file and generator.
    """

    gate_id = str(gate.get("id"))
    label = f"{gate_id} screening"
    spec = topology_screening.BY_EXPERIMENT_ID.get(str(payload.get("experiment_id")))
    if spec is None or spec.gate_id != gate_id:
        errors.append(f"{label}: manifest experiment_id is not a registered screening study of this gate")
        return
    try:
        evidence_bytes, tex_bytes, sidecar_bytes = topology_screening.render(repo, spec)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"{label}: evidence regeneration from the sealed bundle failed: {exc}")
        return
    evidence = json.loads(evidence_bytes)
    evidence_meta = payload.get("paper_evidence_file")
    if not isinstance(evidence_meta, dict) or evidence_meta.get("path") != spec.evidence_path.as_posix():
        errors.append(f"{label}: manifest paper_evidence_file.path differs from the registered evidence file")
        return
    for path, expected, name in (
        (repo / spec.evidence_path, evidence_bytes, "evidence file"),
        (repo / spec.output_path, tex_bytes, "generated TeX"),
        (repo / spec.sidecar_path, sidecar_bytes, "provenance sidecar"),
    ):
        if not path.is_file() or path.read_bytes() != expected:
            errors.append(f"{label}: committed {name} differs from regeneration")
    if evidence_meta.get("document_type") != evidence.get("document_type"):
        errors.append(f"{label}: evidence document_type differs from the manifest")
    if evidence_meta.get("macro_count") != len(evidence.get("macros", [])):
        errors.append(f"{label}: evidence macro count differs from the manifest")
    if evidence_meta.get("macro_prefix") != spec.macro_prefix:
        errors.append(f"{label}: evidence macro prefix differs from the manifest")

    # Artifact hashes on disk (independent of the generator), including lineage records.
    experiment_root = repo / spec.experiment_path
    for relative, meta in evidence.get("artifacts", {}).items():
        artifact = experiment_root / relative
        if not artifact.is_file():
            errors.append(f"{label}: evidence artifact missing on disk: {relative}")
            continue
        raw = artifact.read_bytes()
        if sha256_bytes(raw) != meta.get("sha256") or len(raw) != meta.get("bytes"):
            errors.append(f"{label}: evidence artifact hash mismatch: {relative}")
    for relative, meta in evidence.get("lineage_artifacts", {}).get("files", {}).items():
        artifact = repo / relative
        if not artifact.is_file():
            errors.append(f"{label}: lineage artifact missing on disk: {relative}")
            continue
        raw = artifact.read_bytes()
        if sha256_bytes(raw) != meta.get("sha256") or len(raw) != meta.get("bytes"):
            errors.append(f"{label}: lineage artifact hash mismatch: {relative}")
    bundle = payload.get("results_bundle", {})
    bundle_manifest = experiment_root / "results/manifest.json"
    if not bundle_manifest.is_file():
        errors.append(f"{label}: results manifest is missing on disk")
    else:
        digest = sha256_bytes(bundle_manifest.read_bytes())
        if digest != evidence["bundle"]["manifest_sha256"] or digest != bundle.get("manifest_sha256"):
            errors.append(f"{label}: results manifest SHA-256 differs from the evidence bindings")
    tolerated = evidence["bundle"].get("tolerated_eol_files")
    if tolerated != bundle.get("tolerated_eol_files") or tolerated != sorted(spec.audited_eol_files):
        errors.append(f"{label}: tolerated end-of-line file list differs between evidence, manifest and generator")
    if evidence["bundle"].get("audited_eol_files") != bundle.get("audited_eol_files"):
        errors.append(f"{label}: audited end-of-line digests differ between evidence and manifest")
    errors.extend(_eol_audited_file_errors(repo, spec, label))

    # Revisions.
    revision = str(payload.get("evidence_revision"))
    if revision != spec.results_commit or evidence.get("evidence_revision") != revision:
        errors.append(f"{label}: evidence revision differs between manifest, evidence file and generator")
    try:
        committed_blob = _run_git(repo, "rev-parse", f"{revision}:{evidence['bundle']['manifest_path']}")
    except RuntimeError as exc:
        errors.append(f"{label}: results manifest is not committed at the evidence revision: {exc}")
        committed_blob = None
    if committed_blob is not None and (
        committed_blob != evidence["binding"]["manifest_git_blob"]
        or committed_blob != bundle.get("manifest_git_blob")
    ):
        errors.append(f"{label}: results manifest Git blob differs from the evidence bindings")
    results_rel = (spec.experiment_path / "results").as_posix()
    try:
        results_tree = _run_git(repo, "rev-parse", f"{revision}:{results_rel}")
        head_tree = _run_git(repo, "rev-parse", f"HEAD:{results_rel}")
    except RuntimeError as exc:
        errors.append(f"{label}: results tree cannot be resolved: {exc}")
    else:
        if results_tree != bundle.get("results_tree") or results_tree != evidence["binding"].get("results_tree"):
            errors.append(f"{label}: results tree differs from the manifest binding")
        if head_tree != results_tree:
            errors.append(f"{label}: results tree changed after the evidence revision")
    prereg = payload.get("preregistration_revision")
    if not _resolves_to_commit(repo, prereg) or prereg != spec.preregistration_commit:
        errors.append(f"{label}: preregistration_revision is not the registered resolvable commit")
    else:
        prereg = str(prereg)
        if evidence["binding"].get("preregistration_commit") != prereg:
            errors.append(f"{label}: evidence preregistration commit differs from the manifest")
        if not _is_ancestor(repo, prereg, revision) or prereg == revision:
            errors.append(f"{label}: preregistration must strictly precede the results revision")
        for source in payload.get("source_files", []):
            if isinstance(source, dict) and str(source.get("role", "")).startswith("preregistered-"):
                try:
                    frozen = _run_git(repo, "rev-parse", f"{prereg}:{source['path']}")
                except RuntimeError as exc:
                    errors.append(f"{label}: frozen file missing at preregistration: {exc}")
                    continue
                if frozen != source.get("git_blob"):
                    errors.append(f"{label}: {source['path']} changed after preregistration")
    head = _run_git(repo, "rev-parse", "HEAD")
    audit = payload.get("posthoc_audit")
    if spec.posthoc_audit_commit is None:
        if audit is not None:
            errors.append(f"{label}: manifest binds a post-hoc audit the generator does not register")
    elif not isinstance(audit, dict) or audit.get("revision") != spec.posthoc_audit_commit or not _resolves_to_commit(repo, audit.get("revision")):
        errors.append(f"{label}: posthoc_audit must bind the registered resolvable revision")
    else:
        audit_revision = str(audit["revision"])
        if not _is_ancestor(repo, revision, audit_revision) or not _is_ancestor(repo, audit_revision, head):
            errors.append(f"{label}: posthoc audit revision does not chain results -> audit -> HEAD")
        if audit.get("path") != spec.posthoc_audit_path or gate.get("posthoc_audit_path") != spec.posthoc_audit_path:
            errors.append(f"{label}: posthoc audit path differs between gate, manifest and generator")
        _validate_source_files(repo, audit_revision, [audit], {"posthoc-audit"}, errors, f"{label} posthoc audit")
    lineage_files = payload.get("lineage_files", [])
    lineage_expected = set(evidence.get("lineage_artifacts", {}).get("files", {}))
    if {entry.get("path") for entry in lineage_files if isinstance(entry, dict)} != lineage_expected:
        errors.append(f"{label}: lineage files differ between manifest and evidence file")
    for entry in lineage_files:
        if not isinstance(entry, dict) or not _resolves_to_commit(repo, entry.get("revision")):
            errors.append(f"{label}: lineage entry lacks a resolvable revision")
            continue
        if not _is_ancestor(repo, str(entry["revision"]), head):
            errors.append(f"{label}: lineage revision is not an ancestor of HEAD: {entry.get('path')}")
        if not str(entry.get("role", "")).startswith("lineage-"):
            errors.append(f"{label}: lineage entry must carry a lineage- role: {entry.get('path')}")
        _validate_source_files(repo, str(entry["revision"]), [entry], {str(entry.get("role"))}, errors, f"{label} lineage")

    # Metrics against the raw artifact values behind the macros.
    raw = {item["name"]: item["raw"] for item in evidence.get("macros", [])}
    values = {item["name"]: item["value"] for item in evidence.get("macros", [])}
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(f"{label}: metrics must be an object")
        return
    metric_macros = SCREENING_METRIC_MACROS.get(spec.key, {})
    if not metric_macros:
        errors.append(f"{label}: checker has no metric map for {spec.key}")
    for metric, macro in metric_macros.items():
        if macro not in raw:
            errors.append(f"{label}: evidence lacks macro {macro}")
        elif metric not in metrics:
            errors.append(f"{label}: manifest lacks metric {metric!r}")
        elif metrics[metric] != raw[macro] or type(metrics[metric]) is not type(raw[macro]):
            errors.append(f"{label}: metric {metric!r} differs from artifact value")
    for metric, expected in SCREENING_POLICY_METRICS.items():
        if metrics.get(metric) is not expected:
            errors.append(f"{label}: policy metric {metric!r} must be {expected!r}")
    outcome = gate.get("recorded_outcome")
    if outcome not in SCREENING_OUTCOMES:
        errors.append(f"{label}: gate recorded_outcome is not a recognized screening outcome")
    if not (outcome == payload.get("recorded_outcome") == metrics.get("recorded_outcome") == evidence.get("recorded_outcome") == spec.recorded_outcome):
        errors.append(f"{label}: recorded_outcome differs between gate, manifest, evidence file and generator")
    if not (topology_screening.SCREENING_MODEL == payload.get("screening_model") == metrics.get("screening_model") == evidence.get("screening_model")):
        errors.append(f"{label}: screening_model differs between manifest, evidence file and generator")
    classification = payload.get("classification")
    expected = gate.get("metric_constraints", {}).get("classification", {}).get("equals")
    if not (classification == spec.classification == expected == evidence.get("classification") == metrics.get("classification")):
        errors.append(f"{label}: classification differs between gate, manifest, evidence and generator")
    classification_macro = f"{spec.macro_prefix}Classification"
    if tex_unescape(values.get(classification_macro, "")) != classification:
        errors.append(f"{label}: \\{classification_macro} macro does not render the classification string")
    if gate.get("opens_level") is not None or payload.get("evidence_level", {}).get("opens_gate") is not None:
        errors.append(f"{label}: a screening study cannot open a physics level")

    # Manuscript bindings.
    binding = gate.get("accepted_manuscript_binding")
    if binding != spec.section_binding or manuscript.count(binding) != 1:
        errors.append(f"{label}: section binding must be the registered \\input and occur exactly once in manuscript.tex")
    generated_binding = spec.generated_binding
    document_start = manuscript.find("\\begin{document}")
    if manuscript.count(generated_binding) != 1 or manuscript.find(generated_binding) > document_start:
        errors.append(f"{label}: generated macro file must be input exactly once in the preamble")
    macro_name = gate.get("manuscript_revision_macro")
    if macro_name != spec.revision_macro:
        errors.append(f"{label}: gate manuscript_revision_macro differs from the generator registration")
    else:
        definitions = [
            macro
            for macro in extract_macros(manuscript, "newcommand", 2)
            if macro.arguments[0] == f"\\{macro_name}"
        ]
        rendered = ""
        if len(definitions) == 1:
            body = re.sub(r"(?m)(?<!\\)%.*$", "", definitions[0].arguments[1])
            rendered = re.sub(r"\\texttt\{|\}|\s", "", tex_unescape(body))
        if rendered != revision:
            errors.append(f"{label}: \\{macro_name} does not spell the manifest revision")

    # Section content.
    try:
        section = (repo / spec.section_path).read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{label}: section unreadable: {exc}")
        return
    heading = gate.get("section_heading")
    if heading != spec.section_heading or payload.get("section_heading") != heading or f"\\subsection{{{heading}}}" not in section:
        errors.append(f"{label}: section heading differs between gate, manifest, generator and section")
    prefix = spec.macro_prefix
    defined = set(re.findall(rf"\\newcommand\{{\\({re.escape(prefix)}[A-Za-z]+)\}}", tex_bytes.decode("utf-8")))
    used = set(re.findall(rf"\\({re.escape(prefix)}[A-Za-z]+)", section))
    if not used:
        errors.append(f"{label}: section uses no evidence macro")
    for name in sorted(used - defined):
        errors.append(f"{label}: section uses undefined macro \\{name}")
    for required in (*spec.table_macros, classification_macro):
        if required not in used:
            errors.append(f"{label}: section must use \\{required}")
    digits = section_literal_digits(section, prefix)
    if digits:
        errors.append(f"{label}: section types {len(digits)} literal digit(s); every number must be a macro")
    if "\\input{" in re.sub(r"(?m)(?<!\\)%.*$", "", section):
        errors.append(f"{label}: section must not input further files")
    for finding in find_unregistered_claims(section):
        errors.append(f"{label}: {finding}")
    artifact_macros = extract_macros(tex_bytes.decode("utf-8"), "ArtifactClaim", 3)
    if len(artifact_macros) != len(spec.table_macros) or any(
        macro.arguments[:2] != (spec.artifact_claim_id, spec.artifact_id) for macro in artifact_macros
    ):
        errors.append(f"{label}: generated tables are not each wrapped in the registered ArtifactClaim")

    # Claim-matrix cross-references.
    integration = evidence.get("manuscript_integration", {})
    if integration.get("status") != "admitted":
        errors.append(f"{label}: evidence file does not record admission")
    if integration.get("gate_id") != gate_id or not (
        integration.get("manifest_id") == payload.get("manifest_id") == spec.manifest_id
    ):
        errors.append(f"{label}: evidence file names a different gate or manifest")
    if integration.get("manifest_path") != gate.get("manifest_path") or integration.get("manifest_path") != spec.manifest_path.as_posix():
        errors.append(f"{label}: evidence file names a different manifest path")
    if integration.get("section_binding") != binding or integration.get("gate_kind") != SCREENING_GATE_KIND:
        errors.append(f"{label}: evidence file names a different section binding or gate kind")
    records = {
        claim.get("id"): claim
        for claim in matrix.get("claims", [])
        if isinstance(claim, dict) and isinstance(claim.get("id"), str)
    }
    manifest_id = payload.get("manifest_id")
    section_claims = set(re.findall(r"\\EvidenceClaim\{(CLM-\d+)\}", section))
    prose_ids = integration.get("prose_claim_ids", [])
    if not section_claims or not section_claims <= set(prose_ids):
        errors.append(f"{label}: section claims are not all registered as screening prose claims")
    normalized_section = _normalize_tex(section)
    for claim_id in prose_ids:
        record = records.get(claim_id)
        if record is None or record.get("status") != "verified":
            errors.append(f"{label}: prose claim {claim_id} is not a verified claim record")
            continue
        if manifest_id not in record.get("manifest_ids", []):
            errors.append(f"{label}: claim {claim_id} is not bound to manifest {manifest_id}")
        if not isinstance(record.get("authorized_tex"), str):
            errors.append(f"{label}: claim {claim_id} must be a prose claim")
        if "classification" in record and record["classification"] != classification:
            errors.append(f"{label}: claim {claim_id} names a different classification")
        if "recorded_outcome" in record and record["recorded_outcome"] != outcome:
            errors.append(f"{label}: claim {claim_id} names a different recorded outcome")
        for phrase in record.get("non_claims", []):
            if _normalize_tex(str(phrase)) not in normalized_section:
                errors.append(f"{label}: non-claim of {claim_id} is absent from the section: {phrase!r}")
        if claim_id in section_claims and heading not in record.get("allowed_locations", []):
            errors.append(f"{label}: claim {claim_id} does not allow the section heading")
    if not any(records.get(claim_id, {}).get("non_claims") for claim_id in prose_ids):
        errors.append(f"{label}: no screening claim registers non_claims")
    artifact_claim = integration.get("artifact_claim_id")
    record = records.get(artifact_claim, {})
    if artifact_claim != spec.artifact_claim_id or integration.get("artifact_id") not in record.get("authorized_artifact_ids", []):
        errors.append(f"{label}: artifact claim {artifact_claim} does not authorize the generated tables")
    if manifest_id not in record.get("manifest_ids", []):
        errors.append(f"{label}: artifact claim {artifact_claim} is not bound to manifest {manifest_id}")
    if flattened.count(f"\\subsection{{{heading}}}") != 1:
        errors.append(f"{label}: section heading must appear exactly once in the flattened manuscript")


CAMPAIGN_CHECKERS = {
    "paper-test-particle-campaign-manifest": _check_wall_loss_campaign,
    "paper-l1a-screening-manifest": _check_topology_screening,
}


def _check_gates(repo: Path, manuscript: str, flattened: str, errors: list[str]) -> None:
    registry = _load_json(repo / "paper/evidence/result-gates.json", errors)
    matrix = _load_json(repo / "paper/evidence/claims.json", errors)
    if not registry or not matrix:
        return
    if registry.get("schema_version") != "2.0":
        errors.append("result-gates.json: unsupported schema_version")
    base_revision = registry.get("evidence_revision")
    if not _resolves_to_commit(repo, base_revision):
        errors.append("result-gates.json: evidence_revision is not resolvable")
        return
    gate_list = registry.get("gates")
    if not isinstance(gate_list, list):
        errors.append("result-gates.json: gates must be an array")
        return
    gates = {
        gate.get("id"): gate
        for gate in gate_list
        if isinstance(gate, dict) and isinstance(gate.get("id"), str)
    }
    required_ids = set(PHYSICS_GATE_IDS)
    if not required_ids <= set(gates):
        errors.append("result-gates.json: gate IDs must include the L1/L2/L3 physics-level gates")
    declared_kinds = registry.get("acceptance_policy", {}).get("gate_kinds")
    if not isinstance(declared_kinds, dict) or set(declared_kinds) != KNOWN_GATE_KINDS:
        errors.append("result-gates.json: acceptance_policy.gate_kinds must define exactly the recognized gate kinds")
    campaign_ids: list[str] = []
    for gate_id, gate in gates.items():
        kind = gate.get("kind")
        if gate_id in required_ids:
            if kind != PHYSICS_GATE_KIND:
                errors.append(f"{gate_id}: physics-level gate must declare kind {PHYSICS_GATE_KIND!r}")
        elif kind == CAMPAIGN_GATE_KIND:
            campaign_ids.append(gate_id)
        elif kind == SCREENING_GATE_KIND:
            campaign_ids.append(gate_id)
            if gate.get("recorded_outcome") not in SCREENING_OUTCOMES:
                errors.append(f"{gate_id}: a numerical-screening gate must declare a recognized recorded_outcome")
        else:
            errors.append(f"{gate_id}: unrecognized gate kind {kind!r}")
    visible = {macro.arguments[0] for macro in extract_macros(flattened, "EvidenceGate", 2)}
    claim_gate_ids = {
        claim.get("id")
        for claim in matrix.get("claims", [])
        if isinstance(claim, dict) and claim.get("status") == "evidence-gate"
    }
    for gate_id in sorted(required_ids):
        gate = gates.get(gate_id, {})
        status = gate.get("status")
        path = gate.get("manifest_path")
        if status == "closed":
            if path is not None:
                errors.append(f"{gate_id}: closed gate must have null manifest_path")
            if gate_id not in visible:
                errors.append(f"{gate_id}: closed gate lacks visible manuscript block")
            if gate_id not in claim_gate_ids:
                errors.append(f"{gate_id}: closed gate lacks claim-matrix record")
        elif status == "accepted":
            dependencies = gate.get("dependencies", [])
            for dependency in dependencies:
                if gates.get(dependency, {}).get("status") != "accepted":
                    errors.append(f"{gate_id}: dependency {dependency} is not accepted")
            if not isinstance(path, str):
                errors.append(f"{gate_id}: accepted gate lacks manifest_path")
                continue
            absolute = repo / path
            payload = _load_json(absolute, errors)
            _validate_manifest_payload(
                repo,
                str(base_revision),
                gate,
                payload,
                Path(path),
                errors,
                require_committed=True,
            )
            if gate_id in visible:
                errors.append(f"{gate_id}: accepted gate still has a closed block")
        else:
            errors.append(f"{gate_id}: invalid status {status!r}")

    for gate_id in sorted(campaign_ids):
        gate = gates[gate_id]
        kind = str(gate.get("kind"))
        if gate.get("status") != "accepted":
            errors.append(f"{gate_id}: a {kind} gate must be accepted or absent")
            continue
        if gate.get("opens_level") is not None:
            errors.append(f"{gate_id}: a {kind} gate cannot open a physics level")
        if gate_id in visible:
            errors.append(f"{gate_id}: accepted gate still has a closed block")
        path = gate.get("manifest_path")
        if not isinstance(path, str):
            errors.append(f"{gate_id}: accepted gate lacks manifest_path")
            continue
        payload = _load_json(repo / path, errors)
        _validate_manifest_payload(
            repo,
            str(base_revision),
            gate,
            payload,
            Path(path),
            errors,
            require_committed=True,
        )
        if payload.get("gate_id") != gate_id:
            errors.append(f"{gate_id}: manifest names a different gate")
        if kind == SCREENING_GATE_KIND and (
            payload.get("gate_kind") != kind or payload.get("recorded_outcome") != gate.get("recorded_outcome")
        ):
            errors.append(f"{gate_id}: manifest gate_kind or recorded_outcome differs from the gate")
        checker = CAMPAIGN_CHECKERS.get(str(gate.get("required_manifest_document_type")))
        if checker is None:
            errors.append(f"{gate_id}: no campaign checker for its manifest type")
        elif payload:
            checker(repo, gate, payload, manuscript, flattened, matrix, errors)


def _check_claims(
    repo: Path, manuscript: str, citation_keys: set[str], errors: list[str]
) -> None:
    matrix = _load_json(repo / "paper/evidence/claims.json", errors)
    if not matrix:
        return
    if matrix.get("schema_version") != "2.0":
        errors.append("claims.json: unsupported schema_version")
    revision = matrix.get("evidence_revision")
    if not _resolves_to_commit(repo, revision):
        errors.append("claims.json: evidence_revision is not resolvable")
        return
    head = _run_git(repo, "rev-parse", "HEAD")
    if not _is_ancestor(repo, str(revision), head):
        errors.append("claims.json: evidence_revision is not an ancestor of HEAD")

    sources = matrix.get("sources")
    if not isinstance(sources, dict):
        errors.append("claims.json: sources must be an object")
        sources = {}
    for source_id, source in sources.items():
        if not isinstance(source, dict):
            errors.append(f"claims.json: source {source_id} must be an object")
            continue
        source_revision = source.get("revision")
        if not _resolves_to_commit(repo, source_revision):
            errors.append(f"claims.json: source {source_id} revision is invalid")
            continue
        _validate_source_files(
            repo,
            str(source_revision),
            [
                {
                    "role": source_id,
                    "path": source.get("path"),
                    "git_blob": source.get("git_blob"),
                    "git_blob_sha256": source.get("git_blob_sha256"),
                }
            ],
            {source_id},
            errors,
            f"claims.json source {source_id}",
        )

    records = {
        claim.get("id"): claim
        for claim in matrix.get("claims", [])
        if isinstance(claim, dict) and isinstance(claim.get("id"), str)
    }
    verified = {
        claim_id: claim
        for claim_id, claim in records.items()
        if claim.get("status") == "verified"
    }
    manifests = matrix.get("manifests", {})
    if not isinstance(manifests, dict):
        errors.append("claims.json: manifests must be an object")
        manifests = {}
    for manifest_id, entry in manifests.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append(f"claims.json: manifest {manifest_id} lacks a path")
            continue
        payload = _load_json(repo / entry["path"], errors)
        if payload.get("document_type") != entry.get("document_type"):
            errors.append(f"claims.json: manifest {manifest_id} document_type differs from its file")
        if payload.get("schema_version") != entry.get("schema_version"):
            errors.append(f"claims.json: manifest {manifest_id} schema_version differs from its file")
        if payload.get("manifest_id") != manifest_id:
            errors.append(f"claims.json: manifest {manifest_id} file carries a different manifest_id")
    manifest_ids = set(manifests)
    macros = extract_macros(manuscript, "EvidenceClaim", 2)
    counts: dict[str, int] = {}
    for macro in macros:
        claim_id, body = macro.arguments
        counts[claim_id] = counts.get(claim_id, 0) + 1
        record = verified.get(claim_id)
        if record is None:
            errors.append(f"manuscript.tex: unregistered EvidenceClaim {claim_id!r}")
            continue
        authorized = record.get("authorized_tex")
        if not isinstance(authorized, str):
            errors.append(f"claims.json: claim {claim_id} is not a prose claim")
        elif _normalize_tex(body) != _normalize_tex(authorized):
            errors.append(f"manuscript.tex: claim {claim_id} body is not authorized")
        location = _heading_at(manuscript, macro.start)
        if location not in record.get("allowed_locations", []):
            errors.append(
                f"manuscript.tex: claim {claim_id} is not allowed in {location!r}"
            )

    for claim_id, record in verified.items():
        has_text = isinstance(record.get("authorized_tex"), str)
        has_artifact = isinstance(record.get("authorized_artifact_ids"), list)
        if has_text == has_artifact:
            errors.append(
                f"claims.json: claim {claim_id} must authorize exactly text or artifacts"
            )
        if has_text and counts.get(claim_id, 0) != 1:
            errors.append(
                f"claims.json: text claim {claim_id} must occur exactly once"
            )
        if not record.get("permitted_scope") or not record.get("prohibited_inferences"):
            errors.append(f"claims.json: claim {claim_id} lacks scope boundaries")
        for source_id in record.get("evidence", []):
            if source_id not in sources:
                errors.append(f"claims.json: claim {claim_id} has unknown source")
        for manifest_id in record.get("manifest_ids", []):
            if manifest_id not in manifest_ids:
                errors.append(f"claims.json: claim {claim_id} has unknown manifest")
        for key in record.get("bibliography", []):
            if key not in citation_keys:
                errors.append(f"claims.json: claim {claim_id} cites unused key {key}")

    masked = _mask_spans(
        manuscript,
        macros
        + extract_macros(manuscript, "ArtifactClaim", 3)
        + extract_macros(manuscript, "EvidenceGate", 2),
    )
    if re.search(r"\\Claim\{", manuscript):
        errors.append("manuscript.tex: detached Claim macro is prohibited")
    if re.search(r"\bCLM-\d+\b", masked):
        errors.append("manuscript.tex: claim ID appears outside a structured claim")
    for finding in find_unregistered_claims(manuscript):
        errors.append(f"manuscript.tex: {finding}")


def _render_l0_table(repo: Path, item: dict[str, Any]) -> tuple[bytes, bytes]:
    output, sidecar = generate_tables.render(repo)
    return output, generate_tables.canonical_json(sidecar)


def _render_wall_loss_tables(repo: Path, item: dict[str, Any]) -> tuple[bytes, bytes]:
    _evidence, output, sidecar = wall_loss_v4.render(repo)
    return output, sidecar


def _render_topology_screening_tables(repo: Path, item: dict[str, Any]) -> tuple[bytes, bytes]:
    spec = next(
        (spec for spec in topology_screening.EXPERIMENTS.values() if spec.artifact_id == item.get("id")),
        None,
    )
    if spec is None:
        raise ValueError(f"{item.get('id')}: no screening study registers this artifact")
    if item.get("evidence_file") != spec.evidence_path.as_posix() or item.get("required_gate") != spec.gate_id:
        raise ValueError(f"{item.get('id')}: contract evidence file or gate differs from the generator registration")
    _evidence, output, sidecar = topology_screening.render(repo, spec)
    return output, sidecar


# Contract ``generator_module`` -> renderer(repo, item) returning (output bytes, canonical sidecar bytes).
ARTIFACT_RENDERERS = {
    "generate_tables": _render_l0_table,
    "generate_wall_loss_v4_evidence": _render_wall_loss_tables,
    "generate_topology_screening_evidence": _render_topology_screening_tables,
}


def _check_artifacts(repo: Path, manuscript: str, errors: list[str]) -> None:
    contract = _load_json(
        repo / "paper/evidence/figure-table-contract.json", errors
    )
    matrix = _load_json(repo / "paper/evidence/claims.json", errors)
    registry = _load_json(repo / "paper/evidence/result-gates.json", errors)
    if contract.get("schema_version") != "2.0":
        errors.append("figure-table-contract.json: unsupported schema_version")
    claims = {
        claim.get("id"): claim
        for claim in matrix.get("claims", [])
        if isinstance(claim, dict)
    }
    gates = {
        gate.get("id"): gate
        for gate in registry.get("gates", [])
        if isinstance(gate, dict)
    }
    manifest_ids = set(matrix.get("manifests", {}))
    for item in contract.get("items", []):
        if not isinstance(item, dict) or item.get("status") != "verified":
            continue
        artifact_id = item.get("id")
        binding = item.get("manuscript_binding")
        if not isinstance(binding, str) or manuscript.count(binding) != 1:
            errors.append(f"{artifact_id}: manuscript binding must occur exactly once")
        if item.get("manifest_id") not in manifest_ids:
            errors.append(f"{artifact_id}: manifest_id is not registered in claims.json")
        required_gate = item.get("required_gate")
        if required_gate is not None and gates.get(required_gate, {}).get("status") != "accepted":
            errors.append(f"{artifact_id}: required gate {required_gate!r} is not accepted")
        renderer = ARTIFACT_RENDERERS.get(str(item.get("generator_module", "generate_tables")))
        if renderer is None:
            errors.append(f"{artifact_id}: unrecognized generator_module")
            continue
        try:
            expected_output, expected_sidecar_bytes = renderer(repo, item)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{artifact_id}: generator validation failed: {exc}")
            continue
        output_path = repo / item["output_path"]
        sidecar_path = repo / item["sidecar_path"]
        if not output_path.is_file() or output_path.read_bytes() != expected_output:
            errors.append(f"{artifact_id}: generated output is missing or stale")
            continue
        if not sidecar_path.is_file() or sidecar_path.read_bytes() != expected_sidecar_bytes:
            errors.append(f"{artifact_id}: provenance sidecar is missing or stale")
        sidecar = _load_json(sidecar_path, errors)
        if sidecar.get("artifact_id") != artifact_id:
            errors.append(f"{artifact_id}: sidecar artifact ID mismatch")
        if sidecar.get("output", {}).get("sha256") != sha256_bytes(expected_output):
            errors.append(f"{artifact_id}: sidecar output hash mismatch")
        if sorted(sidecar.get("claim_ids", [])) != sorted(item.get("claim_ids", [])):
            errors.append(f"{artifact_id}: sidecar claim IDs differ from the contract")
        artifact_macros = extract_macros(
            output_path.read_text(encoding="utf-8"), "ArtifactClaim", 3
        )
        expected_count = item.get("artifact_claim_count", 1)
        if len(artifact_macros) != expected_count:
            errors.append(
                f"{artifact_id}: output requires exactly {expected_count} ArtifactClaim macro(s)"
            )
        for macro in artifact_macros:
            claim_id, macro_artifact, _ = macro.arguments
            if macro_artifact != artifact_id or claim_id not in item.get("claim_ids", []):
                errors.append(f"{artifact_id}: output claim binding mismatch")
        for claim_id in item.get("claim_ids", []):
            authorized = claims.get(claim_id, {}).get("authorized_artifact_ids", [])
            if artifact_id not in authorized:
                errors.append(f"{artifact_id}: claim {claim_id} does not authorize artifact")
        for finding in find_unregistered_claims(
            output_path.read_text(encoding="utf-8")
        ):
            errors.append(f"{artifact_id}: {finding}")


def _check_submission_and_build_config(repo: Path, manuscript: str, errors: list[str]) -> None:
    gates = _load_json(repo / "paper/evidence/submission-gates.json", errors)
    records = {
        gate.get("id"): gate
        for gate in gates.get("gates", [])
        if isinstance(gate, dict)
    }
    if records.get("AUTHOR-IDENTITY", {}).get("value") != "Angus Muffatti":
        errors.append("submission-gates.json: author identity must be Angus Muffatti")
    required_human = {
        "COAUTHOR-APPROVAL",
        "CONTRIBUTION-STATEMENT-APPROVAL",
        "AFFILIATION-APPROVAL",
        "CORRESPONDING-AUTHOR-APPROVAL",
    }
    for gate_id in required_human:
        if records.get(gate_id, {}).get("status") != "human-approval-required":
            errors.append(f"submission-gates.json: {gate_id} must remain a human gate")
    if "\\author{Angus Muffatti}" not in manuscript:
        errors.append("manuscript.tex: author must be Angus Muffatti")

    config = _load_json(repo / "paper/build-config.json", errors)
    revision = config.get("evidence_revision")
    if not _resolves_to_commit(repo, revision):
        errors.append("build-config.json: evidence_revision is not resolvable")
        return
    expected_epoch = int(_run_git(repo, "show", "-s", "--format=%ct", str(revision)))
    if config.get("source_date_epoch") != expected_epoch:
        errors.append("build-config.json: SOURCE_DATE_EPOCH differs from commit time")
    if config.get("pdf_metadata", {}).get("author") != "Angus Muffatti":
        errors.append("build-config.json: deterministic PDF author mismatch")

    ignore = (repo / "paper/.gitignore").read_text(encoding="utf-8").splitlines()
    if "build/" not in ignore or "__pycache__/" not in ignore:
        errors.append("paper/.gitignore: local build/cache exclusions are missing")
    for trackable in (
        "paper/evidence/l0-run-manifest.json",
        "paper/generated/l0-ranges.tex",
        "paper/generated/l0-ranges.provenance.json",
        "paper/evidence/wall-loss-v4.json",
        "paper/evidence/manifests/wall-loss-v4.json",
        "paper/generated/wall-loss-v4.tex",
        "paper/generated/wall-loss-v4.provenance.json",
        "paper/sections/wall-loss-v4.tex",
        *(
            path.as_posix()
            for spec in topology_screening.EXPERIMENTS.values()
            for path in (spec.evidence_path, spec.manifest_path, spec.output_path, spec.sidecar_path, spec.section_path)
        ),
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", trackable],
            cwd=repo,
            check=False,
        ).returncode == 0
        if ignored:
            errors.append(f"paper/.gitignore: source/evidence is ignored: {trackable}")


def collect_errors(repo: Path) -> list[str]:
    errors: list[str] = []
    manuscript_path = repo / "paper/manuscript.tex"
    if not manuscript_path.is_file():
        return ["paper/manuscript.tex is missing"]
    manuscript = manuscript_path.read_text(encoding="utf-8")
    # Section files bound through \input{sections/...} face every prose check.
    flattened = flatten_sections(repo, manuscript, errors)
    for section in REQUIRED_SECTIONS:
        if f"\\section{{{section}}}" not in manuscript:
            errors.append(f"manuscript.tex: required section is missing: {section}")

    _check_text_policy(repo, errors)
    citation_keys = _check_bibliography(repo, flattened, errors)
    _check_schema_registry(repo, errors)
    _check_l0_manifest(repo, errors)
    _check_claims(repo, flattened, citation_keys, errors)
    _check_gates(repo, manuscript, flattened, errors)
    _check_artifacts(repo, flattened, errors)
    _check_submission_and_build_config(repo, manuscript, errors)
    for path in sorted((repo / "paper/evidence").glob("*.json")):
        _load_json(path, errors)
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    errors = collect_errors(repo)
    if errors:
        print("Paper checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        "Paper checks passed: typed manifests, exact claims, generated artifacts, "
        "citations, submission gates, and deterministic-build policy."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
