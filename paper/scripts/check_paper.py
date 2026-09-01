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


REQUIRED_SECTIONS = (
    "Introduction",
    "Literature lineage and scope",
    "Methods architecture",
    "Legacy audit",
    "Verification, validation, and uncertainty protocol",
    "Accepted L0 result",
    "Planned L1 result: field-resolved reduction",
    "Planned L2 result: coupled hybrid model",
    "Planned L3 result: PIC and experimental comparison",
    "Limitations",
    "Reproducibility and data availability",
)

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


def _check_gates(repo: Path, manuscript: str, errors: list[str]) -> None:
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
    required_ids = {"GATE-L1", "GATE-L2", "GATE-L3"}
    if set(gates) != required_ids:
        errors.append("result-gates.json: gate IDs must be exactly L1/L2/L3")
    visible = {macro.arguments[0] for macro in extract_macros(manuscript, "EvidenceGate", 2)}
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
    manifest_ids = set(matrix.get("manifests", {}))
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


def _check_artifacts(repo: Path, manuscript: str, errors: list[str]) -> None:
    contract = _load_json(
        repo / "paper/evidence/figure-table-contract.json", errors
    )
    matrix = _load_json(repo / "paper/evidence/claims.json", errors)
    if contract.get("schema_version") != "2.0":
        errors.append("figure-table-contract.json: unsupported schema_version")
    claims = {
        claim.get("id"): claim
        for claim in matrix.get("claims", [])
        if isinstance(claim, dict)
    }
    for item in contract.get("items", []):
        if not isinstance(item, dict) or item.get("status") != "verified":
            continue
        artifact_id = item.get("id")
        binding = item.get("manuscript_binding")
        if not isinstance(binding, str) or manuscript.count(binding) != 1:
            errors.append(f"{artifact_id}: manuscript binding must occur exactly once")
        try:
            expected_output, expected_sidecar = generate_tables.render(repo)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{artifact_id}: generator validation failed: {exc}")
            continue
        output_path = repo / item["output_path"]
        sidecar_path = repo / item["sidecar_path"]
        if not output_path.is_file() or output_path.read_bytes() != expected_output:
            errors.append(f"{artifact_id}: generated output is missing or stale")
            continue
        expected_sidecar_bytes = generate_tables.canonical_json(expected_sidecar)
        if not sidecar_path.is_file() or sidecar_path.read_bytes() != expected_sidecar_bytes:
            errors.append(f"{artifact_id}: provenance sidecar is missing or stale")
        sidecar = _load_json(sidecar_path, errors)
        if sidecar.get("artifact_id") != artifact_id:
            errors.append(f"{artifact_id}: sidecar artifact ID mismatch")
        if sidecar.get("output", {}).get("sha256") != sha256_bytes(expected_output):
            errors.append(f"{artifact_id}: sidecar output hash mismatch")
        artifact_macros = extract_macros(
            output_path.read_text(encoding="utf-8"), "ArtifactClaim", 3
        )
        if len(artifact_macros) != 1:
            errors.append(f"{artifact_id}: output requires exactly one ArtifactClaim")
        else:
            claim_id, macro_artifact, _ = artifact_macros[0].arguments
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
    for section in REQUIRED_SECTIONS:
        if f"\\section{{{section}}}" not in manuscript:
            errors.append(f"manuscript.tex: required section is missing: {section}")

    _check_text_policy(repo, errors)
    citation_keys = _check_bibliography(repo, manuscript, errors)
    _check_schema_registry(repo, errors)
    _check_l0_manifest(repo, errors)
    _check_claims(repo, manuscript, citation_keys, errors)
    _check_gates(repo, manuscript, errors)
    _check_artifacts(repo, manuscript, errors)
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
