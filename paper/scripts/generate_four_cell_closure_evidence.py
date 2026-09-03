"""Generate hash-bound paper evidence for the four-cell power-balance closure analysis.

The admitted object is an *analytic consistency result*: substituting rows
R00-R26 of the corrected four-cell discharge ledger into the global power row
R27 leaves the closed form

    R27 = 2 (j_e3 (1 - p4) + I4) (phi_4 - Ua) + EI (p1 j_e0 + p2 j_e1 + p3 j_e2),

whose two terms are non-negative on the admissible region, so the equation set
has no admissible root for any positive interior cusp probability.  The
derivation lives in ``modern/docs/workstreams/global-plasma-closure-analysis.md``
and in ``modern/spec/plasma/equation-ledger.json#global_row_consistency`` at
commit 266d8a99; the diagnostics ``potential_parametrized_state`` and
``global_row_closed_form`` in ``cft_revival.plasma`` implement it and committed
tests pin it.

This generator does three things, all from the repository checkout:

* it binds every consumed document, ledger entry, source file, test file, the
  frozen MDO protocol disclosure and the legacy ``FYP/Power_B_EQs.m`` blob by
  Git blob and SHA-256 at the analysis revision, and requires the executed
  ``cft_revival.plasma`` package in the checkout to equal the bound blobs;
* it RECOMPUTES the numerical verification with that package: the closed form
  against the full residual over a fixed seeded sample, the continuation ladder
  ``p = eps (1,1,1,1)`` through the production solver, the anode-only closures
  ``p = (0,0,0,eps)``, the published-state misfit, one relaxed-constraint root,
  the Jacobian rank at the floor points and the anode-fall coefficient; it
  refuses to write anything if a recomputed number departs from the analysis
  document beyond the declared tolerance;
* it reads the ``13/80`` probe pattern from the frozen MDO protocol disclosure
  (it is *not* recomputed here: eighty multistart solves take minutes) and
  requires the analysis document's reproduction to agree.

Outputs (Python standard library plus the repository's own ``cft_revival.plasma``):

* ``paper/evidence/four-cell-closure.json`` -- every macro value with the
  document/ledger/protocol path, pointer or regular expression and SHA-256 it
  was read from, or the recomputation it came from with the inputs it used;
* ``paper/generated/four-cell-closure.tex`` -- ``\\newcommand`` macros and two
  generated tables wrapped in ``\\ArtifactClaim``;
* ``paper/generated/four-cell-closure.provenance.json`` -- generator/input/
  output hashes in the shape of the other paper sidecars.

No wall-clock value or machine path enters any output.  Recomputed values are
recorded to a declared number of significant digits because the least-squares
floors are solver-stall values: they are checked against the document at the
declared tolerance, not for bitwise reproduction across platforms.  Nothing
here is a statement about the physical thruster; the analysis is about the
equation set.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Callable

from generate_mdo_l0_v1_evidence import PROBE_PATTERN as MDO_PROBE_PATTERN

# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
ANALYSIS_COMMIT_SHA = "266d8a99ce75fe35b4870d5d046c9069d7b26c0b"
VERIFIED_TREE_COMMIT_SHA = "ba6875f604746e8fbeaf2aee2bdf06b8f06bdc04"
MDO_PREREGISTRATION_COMMIT_SHA = "4898d0fd3decddc5f308072e724d1936660c00e9"

GATE_KIND = "analytic-consistency"
GATE_ID = "GATE-FOUR-CELL-CLOSURE-V1"
MANIFEST_ID = "FOUR-CELL-CLOSURE-20260903-R27-V1"
MANIFEST_PATH = Path("paper/evidence/manifests/four-cell-closure.json")
MANIFEST_DOCUMENT_TYPE = "paper-analytic-consistency-manifest"
EVIDENCE_PATH = Path("paper/evidence/four-cell-closure.json")
OUTPUT_PATH = Path("paper/generated/four-cell-closure.tex")
SIDECAR_PATH = Path("paper/generated/four-cell-closure.provenance.json")
SECTION_PATH = Path("paper/sections/four-cell-closure.tex")
SECTION_BINDING = "\\input{sections/four-cell-closure.tex}"
GENERATED_BINDING = "\\input{generated/four-cell-closure.tex}"
SECTION_HEADING = "Closed form of the global power row and its admissible roots"
REVISION_MACRO = "ClosureEvidenceRevision"
ARTIFACT_ID = "TAB-FOUR-CELL-CLOSURE-V1"
ARTIFACT_CLAIM_ID = "CLM-039"
PROSE_CLAIM_IDS = (
    "CLM-036", "CLM-037", "CLM-038", "CLM-040", "CLM-041", "CLM-042", "CLM-043", "CLM-044",
)
TABLE_MACROS = ("FccContinuationTable", "FccGlobalSearchTable")
MACRO_PREFIX = "Fcc"
CLASSIFICATION = "analytic_consistency_of_the_corrected_four_cell_power_balance_not_thruster_physics"
CORRECTION_STATUS = "PROPOSED_NOT_ACCEPTED"
PROBE_SOURCE = "mdo-protocol-disclosure"

DOCUMENT = Path("modern/docs/workstreams/global-plasma-closure-analysis.md")
LEDGER = Path("modern/spec/plasma/equation-ledger.json")
PROTOCOL = Path("modern/experiments/mdo_l0_campaign_v1/protocol.json")
LEGACY = Path("FYP/Power_B_EQs.m")
AUDIT = Path("modern/docs/AUDIT.md")
REFERENCES = Path("modern/docs/REFERENCES.md")
PACKAGE_DIR = Path("modern/src/cft_revival/plasma")
PACKAGE_FILES = ("__init__.py", "models.py", "reference.py", "residuals.py", "solver.py")
TEST_FILES = (
    Path("modern/tests/plasma/test_closure_p_nonzero.py"),
    Path("modern/tests/plasma_network/test_plasma_network_closure_p_nonzero.py"),
    Path("modern/tests/plasma/test_solver.py"),
)
# Repository-relative path -> manifest role.  Every entry is bound by blob + SHA-256 at
# ANALYSIS_COMMIT_SHA; the package files must additionally equal the checkout.
SOURCE_ROLES: dict[str, str] = {
    DOCUMENT.as_posix(): "analysis-document",
    LEDGER.as_posix(): "equation-ledger",
    (PACKAGE_DIR / "residuals.py").as_posix(): "diagnostics-source",
    (PACKAGE_DIR / "solver.py").as_posix(): "solver-source",
    (PACKAGE_DIR / "models.py").as_posix(): "model-source",
    (PACKAGE_DIR / "reference.py").as_posix(): "reference-source",
    (PACKAGE_DIR / "__init__.py").as_posix(): "package-init",
    TEST_FILES[0].as_posix(): "closure-tests",
    TEST_FILES[1].as_posix(): "network-closure-tests",
    TEST_FILES[2].as_posix(): "solver-tests",
    PROTOCOL.as_posix(): "mdo-protocol-disclosure",
    LEGACY.as_posix(): "legacy-lineage",
    AUDIT.as_posix(): "legacy-audit",
    REFERENCES.as_posix(): "references",
}

# --------------------------------------------------------------------------- #
# Declared recomputation protocol and tolerances (fixed; part of the admission)
# --------------------------------------------------------------------------- #
SAMPLE_SEED = 20260903
SAMPLE_COUNT = 400
SAMPLE_VOLTAGES = (150.0, 300.0, 500.0, 1000.0)
SAMPLE_CURRENTS = (0.1, 0.5, 1.0, 3.0)
SAMPLE_PROBABILITY_UPPER = 0.7
CONTINUATION_VOLTAGE_V = 300.0
CONTINUATION_CURRENT_A = 1.0
CONTINUATION_EPSILONS = (1.0e-4, 1.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1)
CONTINUATION_START_COUNT = 1
ANODE_ONLY_START_COUNT = 5
CONTINUATION_MAX_ITERATIONS = 600
RESIDUAL_TOLERANCE = 1.0e-8
DM92_PROBABILITIES = (0.060, 0.119, 0.160, 0.254)
DM92_VOLTAGE_V = 1000.0
DM92_CURRENT_A = 1.0
DM92_POTENTIALS_V = (14.1, 1000.0, 1000.0, 1000.0)
RELAXED_VOLTAGE_V = 300.0
RELAXED_CURRENT_A = 1.0
RELAXED_INTERIOR_POTENTIALS_V = (4.23, 270.0, 285.0)
RELAXED_BISECTIONS = 200
COEFFICIENT_DELTA_V = 1.0

TOLERANCES: dict[str, float] = {
    # recomputed closed form vs full residual: max relative difference must stay below
    "closed_form_relative_difference_upper_bound": 1.0e-12,
    # recomputed R00-R26 residual on the manifold must stay below the document's bound
    "manifold_normalized_residual_upper_bound": 1.0e-11,
    # each recomputed continuation floor within this relative distance of the document
    "continuation_floor_relative": 0.25,
    # floor/eps must stay within this factor over the ladder (no branch, linear in eps)
    "continuation_slope_spread_maximum": 2.0,
    # anode-only closures must converge (max|r| at or below the production residual tolerance)
    "anode_only_residual_upper_bound": RESIDUAL_TOLERANCE,
    # recomputed published-state misfit within this relative distance of the document
    "dm92_misfit_relative": 0.02,
    # recomputed relaxed root: every normalized residual below this
    "relaxed_root_residual_upper_bound": 1.0e-10,
    # recomputed relaxed-root depth within this relative distance of the document's 300 V value
    "relaxed_root_depth_relative": 0.02,
    # recomputed anode-fall coefficient within this absolute distance of the ledger's 2
    "anode_fall_coefficient_absolute": 1.0e-9,
}
ROUNDING: dict[str, int] = {
    "closed_form_relative_difference": 2,
    "manifold_normalized_residual": 2,
    "continuation_floor": 3,
    "continuation_slope": 3,
    "anode_only_residual": 1,
    "dm92_misfit": 3,
    "relaxed_root_depth_v": 3,
    "relaxed_root_residual": 1,
    "jacobian_condition": 2,
    "departure": 2,
}


# --------------------------------------------------------------------------- #
# Formatting (shared with the tests through this module)
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


def _sig(value: float, digits: int) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e5:
        return _sci(value, digits - 1)
    decimals = digits - 1 - math.floor(math.log10(magnitude))
    if decimals <= 0:
        return f"{round(value, decimals):.0f}"
    return f"{value:.{decimals}f}"


def _expr(text: str) -> str:
    """Escape a ledger expression and allow stretchable line breaks after operators.

    Typewriter expressions carry no inter-word glue, so a bare ``\\allowbreak``
    leaves TeX unable to justify the line; a zero-width stretchable space after
    every operator gives both a break point and the stretch to fill the line.
    """

    escaped = _tex_escape(text)
    for operator in ("+", "*", "-", ")", "]"):
        escaped = escaped.replace(operator, operator + "\\hspace{0pt plus 1.5pt}")
    return escaped


FORMATTERS: dict[str, Callable[[Any], str]] = {
    "int": lambda v: f"{int(v):d}",
    "int_comma": lambda v: f"{int(v):,d}".replace(",", "{,}"),
    "fixed1": lambda v: f"{float(v):.1f}",
    "fixed2": lambda v: f"{float(v):.2f}",
    "pct0": lambda v: f"{100.0 * float(v):.0f}\\%",
    "sci0": lambda v: _sci(float(v), 0),
    "sci1": lambda v: _sci(float(v), 1),
    "sci2": lambda v: _sci(float(v), 2),
    "sig2": lambda v: _sig(float(v), 2),
    "sig3": lambda v: _sig(float(v), 3),
    "g": lambda v: f"{float(v):g}",
    "text": lambda v: _tex_escape(str(v)),
    "ident": lambda v: _tex_escape(str(v)).replace("\\_", "\\_\\allowbreak{}").replace("-", "-\\allowbreak{}"),
    "expr": lambda v: _expr(str(v)),
    "bool": lambda v: "true" if v is True else "false" if v is False else _tex_escape(str(v)),
    "list_g": lambda v: ", ".join(f"{float(x):g}" for x in v),
    "sha_short": lambda v: _tex_escape(str(v)[:12]),
}


def format_value(fmt: str, value: Any) -> str:
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown formatter {fmt!r}")
    return FORMATTERS[fmt](value)


def round_sig(value: float, digits: int) -> float:
    """Round to ``digits`` significant digits (the declared recording precision)."""

    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return value
    return float(f"{value:.{digits - 1}e}")


# --------------------------------------------------------------------------- #
# Strict JSON, hashing, git
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


def _git_bytes(repo: Path, revision: str, path: str) -> bytes:
    completed = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=repo, check=False, capture_output=True)
    if completed.returncode:
        raise ValueError(f"git show {revision}:{path} failed: {completed.stderr.decode(errors='replace').strip()}")
    return completed.stdout


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo, check=False, capture_output=True,
    ).returncode == 0


def _lf(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def _tree_blobs(repo: Path, revision: str) -> dict[str, str]:
    """Map every path of a commit's tree to its blob id with one ``git ls-tree`` call."""

    completed = subprocess.run(
        ["git", "ls-tree", "-r", "-z", revision], cwd=repo, check=False, capture_output=True,
    )
    if completed.returncode:
        raise ValueError(f"git ls-tree {revision} failed: {completed.stderr.decode(errors='replace').strip()}")
    blobs: dict[str, str] = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        _mode, kind, blob = meta.split(b" ")
        if kind == b"blob":
            blobs[path.decode("utf-8")] = blob.decode("ascii")
    return blobs


def _blob_contents(repo: Path, blobs: list[str]) -> dict[str, bytes]:
    """Read several blobs with one ``git cat-file --batch`` call."""

    completed = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=repo, check=False, capture_output=True,
        input=("\n".join(blobs) + "\n").encode("ascii"),
    )
    if completed.returncode:
        raise ValueError(f"git cat-file --batch failed: {completed.stderr.decode(errors='replace').strip()}")
    contents: dict[str, bytes] = {}
    data = completed.stdout
    position = 0
    while position < len(data):
        newline = data.index(b"\n", position)
        header = data[position:newline].decode("ascii").split(" ")
        if len(header) != 3 or header[1] != "blob":
            raise ValueError(f"git cat-file returned an unexpected header: {header}")
        size = int(header[2])
        start = newline + 1
        contents[header[0]] = data[start:start + size]
        position = start + size + 1
    if set(contents) != set(blobs):
        raise ValueError("git cat-file did not return every requested blob")
    return contents


# --------------------------------------------------------------------------- #
# Document patterns: every documented number is read from the bound blob with a
# fixed regular expression whose name is recorded as the macro's pointer.
# --------------------------------------------------------------------------- #
_NUM = r"[0-9][0-9.]*(?:e[+-]?[0-9]+)?"
DOCUMENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "closed_form_verification": re.compile(
        rf"agrees with the evaluated R27 to\s+`(?P<rel>{_NUM})` relative over (?P<n>\d+) random cases"
    ),
    "manifold_verification": re.compile(
        rf"\(Verified: (?P<n>\d+) random `\(Ua, Ia, p, phi\)` give `max\|R00\.\.R26\| < (?P<bound>{_NUM})`\.\)"
    ),
    "continuation_setup": re.compile(
        r"### 3\.2 Continuation in `p` \((?P<v>\d+) V, (?P<i>\d+) A, (?P<starts>\d+) starts, (?P<iterations>\d+) iterations\)"
    ),
    "continuation_header": re.compile(
        rf"\| pattern \| `eps = (?P<a>{_NUM})` \| `(?P<b>{_NUM})` \| `(?P<c>{_NUM})` \| `(?P<d>{_NUM})` \| `(?P<e>{_NUM})` \| `(?P<f>{_NUM})` \|"
    ),
    "continuation_all_cells": re.compile(
        rf"\| `p = eps\*\(1,1,1,1\)` floor \| (?P<a>{_NUM}) \| (?P<b>{_NUM}) \| (?P<c>{_NUM}) \| (?P<d>{_NUM}) \| (?P<e>{_NUM}) \| (?P<f>{_NUM}) \|"
    ),
    "continuation_interior_cells": re.compile(
        rf"\| `p = eps\*\(1,1,1,0\)` floor \| (?P<a>{_NUM}) \| (?P<b>{_NUM}) \| (?P<c>{_NUM}) \| (?P<d>{_NUM}) \| (?P<e>{_NUM}) \| (?P<f>{_NUM}) \|"
    ),
    "continuation_anode_only": re.compile(
        r"\| `p = \(0,0,0,eps\)` \| closes \| closes \| closes \| closes \| closes \| closes \|"
    ),
    "anode_only_residual": re.compile(
        rf"Loss at the anode cusp alone\s+\(`p_4`\) closes to `(?P<r>{_NUM})` at every `eps`"
    ),
    "differential_evolution": re.compile(
        rf"Differential evolution over the full (?P<dim>\d+)-value box \(DM9\.2 `p`, (?P<v>\d+) V, (?P<i>\d+) A,\s+(?P<n>[\d ]+) evaluations\): best `max\|r\| = (?P<best>{_NUM})`"
    ),
    "random_starts": re.compile(
        rf"(?P<n>\d+) random feasible starts through the production LM: (?P<closed>\d+)/(?P<total>\d+) closed; floors\s+`(?P<min>{_NUM})` \(min\) / `(?P<median>{_NUM})` \(median\) / `(?P<max>{_NUM})` \(max\)"
    ),
    "relaxed_roots": re.compile(
        rf"Dropping only `phi_4 >= Ua` admits exact roots \(`max\|r\| ~ (?P<r>{_NUM})`\) with\s+`phi_4` below the anode potential by (?P<lo>{_NUM})-(?P<hi>{_NUM}) V \(DM9\.2 `p`: (?P<a>{_NUM}) V at\s+(?P<va>\d+) V/(?P<ia>\d+) A, (?P<b>{_NUM}) V at (?P<vb>\d+) V/(?P<ib>\d+) A\)"
    ),
    "jacobian_rank": re.compile(r"rank (?P<rank>\d+) of (?P<n>\d+) at every floor point and at every closed point"),
    "jacobian_condition": re.compile(r"condition of the independent subspace (?P<lo>\d+)-(?P<hi>\d+)"),
    "probe_reproduction": re.compile(
        rf"Result: \*\*(?P<closed>\d+) of (?P<total>\d+) converged, (?P<seconds>\d+) s total \((?P<failing>{_NUM}) s per failing\s+multistart solve, (?P<lo>{_NUM})-(?P<hi>{_NUM}) s per closing one\)\*\*, floors `(?P<fmin>{_NUM}) \.\. (?P<fmax>{_NUM})`"
    ),
    "dm92_misfit": re.compile(
        rf"For DM9\.2 `p` at (?P<v>\d+) V/(?P<i>\d+) A with the published potentials it is `(?P<misfit>{_NUM})`,\s+which is the R27 misfit `(?P<ledger>{_NUM})` the ledger already records"
    ),
    "legacy_line": re.compile(r"legacy `FYP/Power_B_EQs\.m` line (?P<line>\d+)\s+carries the same terms"),
    "kornfeld_assumption": re.compile(r"Assumption (?P<n>\d+): \"Ionization and excitation losses are considered as frozen"),
    "kornfeld_id": re.compile(r"Kornfeld, Koch and Harmann, (?P<id>IEPC-2007-108)"),
    "corrected_rank": re.compile(
        r"structural rank drops\s+from (?P<before>\d+) to (?P<after>\d+) \(nullity (?P<nullity>\d+): all four potentials free\)"
    ),
    "zero_cusp_grid": re.compile(r"Zero-cusp probe grid: (?P<after>\d+)/(?P<total>\d+) close \(was (?P<before>\d+)/(?P<total_before>\d+)\)"),
    "dm92_probabilities": re.compile(r"p_B = \((?P<p>[0-9., ]+)\)\s+# Kornfeld (?P<label>DM9\.2)"),
    "audit_corrections_cancel": re.compile(r"cancel exactly in the substitution and do not\s+appear in \(\*\)\. They did not introduce the inconsistency"),
    "classification_a": re.compile(r"\*\*\(a\) Genuine inconsistency of the corrected equation set for interior\s+`p != 0`\.\*\*"),
    "classification_d": re.compile(r"\*\*\(d\) Sub-region with solutions:\*\* exactly `p1 = p2 = p3 = 0`, any `p4`,\s+`phi_4 = Ua`"),
}
LEGACY_CUSP_LINE = re.compile(r"^\s*-x\(4\)\+x\(18\)\*p1\*\(-phi0\+x\(6\)\+IE\)\+x\(19\)\*p2\*\(-x\(6\)\+x\(7\)\+IE\+x\(10\)\)\+x\(20\)\*p3\*\(-x\(7\)\+x\(8\)\+IE\+x\(11\)\);")
LEGACY_ANODE_LINE = re.compile(r"^\s*-x\(5\)\+x\(21\)\*p4\*\(Ua-x\(8\)\+x\(12\)\)\+\(x\(17\)\+x\(21\)\*\(1-p4\)\)\*\(x\(9\)-Ua\+x\(13\)\)-x\(27\)\*\(x\(9\)-Ua\);")
LEDGER_CLOSED_FORM = re.compile(r"R27 = (?P<coefficient>\d+)\*\(j_e3\*\(1-p4\)\+I4\)\*\(phi_4-Ua\) \+ EI\*\(p1\*j_e0\+p2\*j_e1\+p3\*j_e2\)")
AUDIT_ACCEPTANCE = re.compile(
    r"### 7\. Solver acceptance ignores residual quality\s+`HEMP_solver\.m:(?P<line>\d+)` discards residual norm and residual vector\. In\s+`Performance_est\.m:(?P<lo>\d+)-(?P<hi>\d+)`, exit flags (?P<flags>1-3) are accepted solely by status and\s+flag (?P<rejected>\d) is rejected\."
)
AUDIT_TOLFUN = re.compile(r"With `TolFun=(?P<tolfun>1e-50)` \(`HEMP_solver\.m:(?P<line>\d+)`\)")


def document_value(text: str, name: str, group: str) -> str:
    pattern = DOCUMENT_PATTERNS[name]
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"analysis document does not match the fixed pattern {name!r}")
    return match.group(group)


# --------------------------------------------------------------------------- #
# Source binding and executed-package verification
# --------------------------------------------------------------------------- #
_BINDING_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}


def bind_sources(repo: Path) -> dict[str, dict[str, Any]]:
    """Bind every consumed file by blob and SHA-256 at the analysis revision.

    The bound commits are immutable, so the binding is cached per process and
    HEAD (the ancestry checks depend on HEAD only).
    """

    head = _git(repo, "rev-parse", "HEAD")
    key = (str(repo.resolve()), head)
    cached = _BINDING_CACHE.get(key)
    if cached is not None:
        return cached
    for commit, label in (
        (ANALYSIS_COMMIT_SHA, "analysis"),
        (VERIFIED_TREE_COMMIT_SHA, "verified-tree"),
        (MDO_PREREGISTRATION_COMMIT_SHA, "MDO preregistration"),
    ):
        if not _is_ancestor(repo, commit, head):
            raise ValueError(f"{label} commit is not an ancestor of HEAD")
    if not _is_ancestor(repo, ANALYSIS_COMMIT_SHA, VERIFIED_TREE_COMMIT_SHA):
        raise ValueError("the analysis commit must precede the verified-tree commit")
    analysis_tree = _tree_blobs(repo, ANALYSIS_COMMIT_SHA)
    verified_tree = _tree_blobs(repo, VERIFIED_TREE_COMMIT_SHA)
    prereg_tree = _tree_blobs(repo, MDO_PREREGISTRATION_COMMIT_SHA)
    missing = [path for path in SOURCE_ROLES if path not in analysis_tree]
    if missing:
        raise ValueError(f"source files absent at the analysis revision: {missing}")
    for path in SOURCE_ROLES:
        if verified_tree.get(path) != analysis_tree[path]:
            raise ValueError(f"{path} differs between the analysis and verified-tree revisions")
    contents = _blob_contents(repo, sorted({analysis_tree[path] for path in SOURCE_ROLES}))
    bound: dict[str, dict[str, Any]] = {}
    for path, role in SOURCE_ROLES.items():
        blob = analysis_tree[path]
        raw = contents[blob]
        bound[path] = {
            "role": role,
            "git_blob": blob,
            "git_blob_sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "content": raw,
        }
    if prereg_tree.get(PROTOCOL.as_posix()) != bound[PROTOCOL.as_posix()]["git_blob"]:
        raise ValueError("the MDO protocol blob differs from the frozen preregistration blob")
    _BINDING_CACHE[key] = bound
    return bound


def executed_package_digests(repo: Path) -> dict[str, str]:
    """LF-normalised SHA-256 of every ``cft_revival.plasma`` file in the checkout."""

    digests: dict[str, str] = {}
    for name in PACKAGE_FILES:
        path = repo / PACKAGE_DIR / name
        digests[(PACKAGE_DIR / name).as_posix()] = sha256_bytes(_lf(path.read_bytes()))
    return digests


def compare_package(executed: dict[str, str], bound: dict[str, dict[str, Any]]) -> None:
    """Refuse to recompute with a package that differs from the admitted blobs."""

    for path, digest in executed.items():
        expected = bound.get(path, {}).get("git_blob_sha256")
        if expected is None:
            raise ValueError(f"executed package file is not bound: {path}")
        if digest != expected:
            raise ValueError(
                f"the checkout's {path} differs from the blob admitted at {ANALYSIS_COMMIT_SHA[:8]}; "
                "the recomputation would not exercise the admitted diagnostics (re-admit at the new revision)"
            )
    missing = {path for path, meta in bound.items() if path.startswith(PACKAGE_DIR.as_posix() + "/")} - set(executed)
    if missing:
        raise ValueError(f"bound package files were not executed: {sorted(missing)}")


def _plasma(repo: Path) -> Any:
    """Import ``cft_revival.plasma`` from this checkout and nowhere else."""

    source_root = (repo / "modern/src").resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module = importlib.import_module("cft_revival.plasma")
    location = Path(module.__file__).resolve()
    try:
        location.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"cft_revival.plasma was imported from outside the checkout: {location}") from exc
    return module


# --------------------------------------------------------------------------- #
# Recomputation (cached per process; deterministic; pure Python)
# --------------------------------------------------------------------------- #
_RECOMPUTATION_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}


def _random_case(rng: random.Random, plasma: Any) -> tuple[Any, tuple[float, ...]]:
    voltage = rng.choice(SAMPLE_VOLTAGES)
    current = rng.choice(SAMPLE_CURRENTS)
    probability = tuple(rng.uniform(0.0, SAMPLE_PROBABILITY_UPPER) for _ in range(4))
    interior = sorted(rng.uniform(0.01 * voltage, voltage) for _ in range(3))
    potentials = (*interior, rng.uniform(voltage, 1.5 * voltage))
    return plasma.XenonGlobalInputs(voltage, current, probability), potentials


def recompute(repo: Path, digests: dict[str, str]) -> dict[str, Any]:
    """Run the declared verification protocol with the checkout's package."""

    key = (str(repo.resolve()), tuple(sorted(digests.items())))
    cached = _RECOMPUTATION_CACHE.get(key)
    if cached is not None:
        return cached
    plasma = _plasma(repo)
    global_row = 27

    # 1. Closed form against the full residual; R00-R26 on the manifold.
    rng = random.Random(SAMPLE_SEED)
    closed_form_relative = 0.0
    manifold_residual = 0.0
    for _ in range(SAMPLE_COUNT):
        inputs, potentials = _random_case(rng, plasma)
        state = plasma.potential_parametrized_state(inputs, potentials)
        evaluation = plasma.evaluate_plasma_residual_cpu(state, inputs)
        manifold_residual = max(manifold_residual, max(abs(value) for value in evaluation.normalized[:global_row]))
        predicted = plasma.global_row_closed_form(state, inputs)
        closed_form_relative = max(closed_form_relative, abs(evaluation.raw[global_row] - predicted) / abs(predicted))

    # 2. Anode-fall coefficient: p = 0 kills the recombination term, so on the manifold
    #    R27 / ((j_e3 + I4) (phi_4 - Ua)) is the coefficient of the anode-fall term.
    zero = plasma.XenonGlobalInputs(CONTINUATION_VOLTAGE_V, CONTINUATION_CURRENT_A, (0.0, 0.0, 0.0, 0.0))
    shifted = plasma.potential_parametrized_state(zero, (21.0, 120.0, 260.0, CONTINUATION_VOLTAGE_V + COEFFICIENT_DELTA_V))
    transported = shifted.electron_current_a[3] + shifted.ionization_source_current_a[3]
    coefficient = plasma.evaluate_plasma_residual_cpu(shifted, zero).raw[global_row] / (transported * COEFFICIENT_DELTA_V)

    # 3. Continuation ladder p = eps (1,1,1,1) and anode-only closures p = (0,0,0,eps).
    options = plasma.SolverOptions(max_iterations=CONTINUATION_MAX_ITERATIONS, residual_tolerance=RESIDUAL_TOLERANCE)
    ladder: list[dict[str, Any]] = []
    for epsilon in CONTINUATION_EPSILONS:
        interior = plasma.XenonGlobalInputs(CONTINUATION_VOLTAGE_V, CONTINUATION_CURRENT_A, (epsilon,) * 4)
        result = plasma.solve_global_discharge_multistart(interior, start_count=CONTINUATION_START_COUNT, options=options)
        diagnostics = result.best.diagnostics
        rows = diagnostics.normalized_residuals
        anode = plasma.XenonGlobalInputs(CONTINUATION_VOLTAGE_V, CONTINUATION_CURRENT_A, (0.0, 0.0, 0.0, epsilon))
        anode_result = plasma.solve_global_discharge_multistart(anode, start_count=ANODE_ONLY_START_COUNT, options=options)
        anode_diagnostics = anode_result.best.diagnostics
        ladder.append(
            {
                "epsilon": epsilon,
                "floor": result.residual_floor,
                "converged": diagnostics.converged,
                "reason": diagnostics.reason,
                "dominant_row": max(range(len(rows)), key=lambda index: abs(rows[index])),
                "jacobian_rank": diagnostics.jacobian_rank,
                "jacobian_condition": diagnostics.jacobian_condition_estimate,
                "anode_only_converged": anode_diagnostics.converged,
                "anode_only_residual": anode_diagnostics.residual_inf_norm,
                "anode_only_phi4_minus_ua": (
                    anode_result.best.state.plasma_potential_v[3] - CONTINUATION_VOLTAGE_V
                    if anode_result.best.state is not None else None
                ),
            }
        )

    # 4. Published-state misfit of the rounded DM9.2 table on the exact manifold.
    dm92 = plasma.XenonGlobalInputs(DM92_VOLTAGE_V, DM92_CURRENT_A, DM92_PROBABILITIES)
    dm92_state = plasma.potential_parametrized_state(dm92, DM92_POTENTIALS_V)
    dm92_normalized = plasma.evaluate_plasma_residual_cpu(dm92_state, dm92).normalized
    dm92_manifold = max(abs(value) for value in dm92_normalized[:global_row])
    dm92_misfit = dm92_normalized[global_row]

    # 5. One relaxed-constraint root: bisection in phi_4 below Ua with fixed interior potentials.
    relaxed = plasma.XenonGlobalInputs(RELAXED_VOLTAGE_V, RELAXED_CURRENT_A, DM92_PROBABILITIES)

    def relaxed_global_row(phi_4: float) -> float:
        state = plasma.potential_parametrized_state(relaxed, (*RELAXED_INTERIOR_POTENTIALS_V, phi_4))
        return plasma.evaluate_plasma_residual_cpu(state, relaxed).raw[global_row]

    low, high = RELAXED_INTERIOR_POTENTIALS_V[2], RELAXED_VOLTAGE_V
    if not (relaxed_global_row(low) < 0.0 < relaxed_global_row(high)):
        raise ValueError("relaxed-root bracket does not change sign")
    for _ in range(RELAXED_BISECTIONS):
        middle = 0.5 * (low + high)
        if relaxed_global_row(middle) > 0.0:
            high = middle
        else:
            low = middle
    root = 0.5 * (low + high)
    root_state = plasma.potential_parametrized_state(relaxed, (*RELAXED_INTERIOR_POTENTIALS_V, root))
    root_evaluation = plasma.evaluate_plasma_residual_cpu(root_state, relaxed)
    root_margins = plasma.constraint_margins(root_state, relaxed)

    recomputed = {
        "closed_form_relative_difference": closed_form_relative,
        "manifold_normalized_residual": manifold_residual,
        "anode_fall_coefficient": coefficient,
        "ladder": ladder,
        "dm92_manifold_residual": dm92_manifold,
        "dm92_misfit": dm92_misfit,
        "relaxed_root_depth_v": RELAXED_VOLTAGE_V - root,
        "relaxed_root_residual": max(abs(value) for value in root_evaluation.normalized),
        "relaxed_root_feasible": plasma.is_feasible(root_state, relaxed),
        "relaxed_root_anode_margin_v": root_margins[4],
        "state_dimension": len(root_state.to_vector()),
        "row_count": len(root_evaluation.normalized),
    }
    _RECOMPUTATION_CACHE[key] = recomputed
    return recomputed


# --------------------------------------------------------------------------- #
# Departure checks against the analysis document
# --------------------------------------------------------------------------- #
def check_against_document(recomputed: dict[str, Any], documented: dict[str, Any], tolerances: dict[str, float]) -> dict[str, Any]:
    """Refuse if any recomputed number departs from the document beyond its tolerance.

    Returns the departure summary that the evidence file records.  Exposed for
    the adversarial tests, which feed it synthetic inputs.
    """

    if recomputed["closed_form_relative_difference"] > tolerances["closed_form_relative_difference_upper_bound"]:
        raise ValueError("recomputed closed-form relative difference exceeds the declared bound")
    if recomputed["manifold_normalized_residual"] > tolerances["manifold_normalized_residual_upper_bound"]:
        raise ValueError("recomputed R00-R26 residual on the manifold exceeds the documented bound")
    if abs(recomputed["anode_fall_coefficient"] - documented["anode_fall_coefficient"]) > tolerances["anode_fall_coefficient_absolute"]:
        raise ValueError("recomputed anode-fall coefficient differs from the ledger's closed form")
    ladder = recomputed["ladder"]
    floors = documented["continuation_floors"]
    epsilons = documented["continuation_epsilons"]
    if len(ladder) != len(floors) or len(ladder) != len(epsilons):
        raise ValueError("continuation ladder length differs from the document")
    departures: list[float] = []
    slopes: list[float] = []
    for step, floor, epsilon in zip(ladder, floors, epsilons, strict=True):
        if step["epsilon"] != epsilon:
            raise ValueError("continuation epsilon differs from the document")
        if step["converged"]:
            raise ValueError(f"the interior continuation closed at eps = {epsilon}; the document records no branch")
        if step["floor"] <= RESIDUAL_TOLERANCE:
            raise ValueError("a continuation floor lies below the residual tolerance")
        departure = abs(step["floor"] - floor) / floor
        if departure > tolerances["continuation_floor_relative"]:
            raise ValueError(
                f"recomputed continuation floor at eps = {epsilon} departs from the document by {departure:.3f} "
                f"(tolerance {tolerances['continuation_floor_relative']})"
            )
        departures.append(departure)
        slopes.append(step["floor"] / epsilon)
        if not step["anode_only_converged"] or step["anode_only_residual"] > tolerances["anode_only_residual_upper_bound"]:
            raise ValueError(f"the anode-only closure at eps = {epsilon} did not close to the declared bound")
    spread = max(slopes) / min(slopes)
    if spread > tolerances["continuation_slope_spread_maximum"]:
        raise ValueError("floor/eps is not within the declared spread over the ladder (a branch or a non-linear floor)")
    if abs(recomputed["dm92_misfit"] - documented["dm92_misfit"]) / documented["dm92_misfit"] > tolerances["dm92_misfit_relative"]:
        raise ValueError("recomputed published-state misfit departs from the document")
    if recomputed["relaxed_root_residual"] > tolerances["relaxed_root_residual_upper_bound"]:
        raise ValueError("the relaxed-constraint root does not satisfy every row to the declared bound")
    if recomputed["relaxed_root_feasible"] or recomputed["relaxed_root_anode_margin_v"] >= 0.0:
        raise ValueError("the relaxed-constraint root was not rejected by the admissible region")
    if recomputed["relaxed_root_depth_v"] <= 0.0:
        raise ValueError("the relaxed-constraint root does not lie below the anode potential")
    depth = documented["relaxed_root_depth_v"]
    if abs(recomputed["relaxed_root_depth_v"] - depth) / depth > tolerances["relaxed_root_depth_relative"]:
        raise ValueError("the recomputed relaxed-root depth departs from the document's value at the same operating point")
    ranks = {step["jacobian_rank"] for step in ladder}
    if ranks != {documented["jacobian_rank"]}:
        raise ValueError(f"Jacobian rank at the floor points {sorted(ranks)} differs from the documented rank")
    if max(step["jacobian_condition"] for step in ladder) > documented["jacobian_condition_max"]:
        raise ValueError("Jacobian condition at a floor point exceeds the documented range")
    return {
        "continuation_max_relative_departure": max(departures),
        "continuation_slope_spread": spread,
        "continuation_slopes": slopes,
    }


# --------------------------------------------------------------------------- #
# Macro construction
# --------------------------------------------------------------------------- #
class Macros:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.names: set[str] = set()

    def _register(self, item: dict[str, Any]) -> str:
        name = item["name"]
        if name in self.names or not name.isalpha() or not name.startswith(MACRO_PREFIX):
            raise ValueError(f"macro name {name!r} is invalid or duplicated")
        self.items.append(item)
        self.names.add(name)
        return item["value"]

    def add(self, name: str, artifact: str, pointer: str, raw: Any, fmt: str, description: str) -> str:
        """A value read from a bound file: JSON pointer, or ``regex:<pattern>[<group>]``."""

        return self._register(
            {
                "name": name,
                "value": format_value(fmt, raw),
                "raw": raw,
                "format": fmt,
                "derived": False,
                "recomputed": False,
                "source": {"artifact": artifact, "pointer": pointer},
                "description": description,
            }
        )

    def add_derived(
        self, name: str, raw: Any, fmt: str, description: str, derivation: str,
        inputs: list[dict[str, str]], *, recomputed: bool = False, rounding: int | None = None,
    ) -> str:
        item = {
            "name": name,
            "value": format_value(fmt, raw),
            "raw": raw,
            "format": fmt,
            "derived": True,
            "recomputed": recomputed,
            "derivation": derivation,
            "inputs": inputs,
            "description": description,
        }
        if rounding is not None:
            item["recorded_significant_digits"] = rounding
        return self._register(item)


def _doc_number(text: str) -> float:
    return float(text.replace(" ", ""))


def build(repo: Path) -> tuple[dict[str, Any], str]:
    """Return (evidence document, generated TeX)."""

    bound = bind_sources(repo)
    executed = executed_package_digests(repo)
    compare_package(executed, bound)
    recomputed = recompute(repo, executed)

    document_text = bound[DOCUMENT.as_posix()]["content"].decode("utf-8")
    ledger = load_json_bytes(bound[LEDGER.as_posix()]["content"], "equation ledger")
    protocol = load_json_bytes(bound[PROTOCOL.as_posix()]["content"], "MDO protocol")
    legacy_text = bound[LEGACY.as_posix()]["content"].decode("utf-8", errors="replace")
    audit_text = bound[AUDIT.as_posix()]["content"].decode("utf-8")
    references_text = bound[REFERENCES.as_posix()]["content"].decode("utf-8", errors="replace")
    doc_path = DOCUMENT.as_posix()
    ledger_path = LEDGER.as_posix()

    # --- Ledger facts -------------------------------------------------------- #
    consistency = ledger["global_row_consistency"]
    if consistency["status"] != CORRECTION_STATUS:
        raise ValueError("the ledger's global_row_consistency status differs from the admitted status")
    rows = ledger["residual_rows"]
    if [row["id"] for row in rows] != [f"R{index:02d}" for index in range(28)]:
        raise ValueError("the ledger does not carry rows R00..R27 in order")
    if len(ledger["power_expressions"]) != 7:
        raise ValueError("the ledger does not carry the seven power expressions")
    closed_form_match = LEDGER_CLOSED_FORM.search(consistency["closed_form_on_manifold"])
    if closed_form_match is None:
        raise ValueError("the ledger's closed form does not match the fixed pattern")
    ledger_coefficient = int(closed_form_match.group("coefficient"))
    proposals = {item["id"]: item for item in consistency["proposed_corrections"]}
    if set(proposals) != {"Pcusp", "Panode_e"} or any(item["status"] != CORRECTION_STATUS for item in proposals.values()):
        raise ValueError("the ledger's proposed corrections differ from the admitted record")
    if "+EI" not in proposals["Pcusp"]["current_expression"].replace(" ", "") or "+EI" in proposals["Pcusp"]["proposed_expression"].replace(" ", ""):
        raise ValueError("the Pcusp proposal does not drop the ionisation energy")
    if "Ua-phi_4+T4" not in proposals["Panode_e"]["proposed_expression"].replace(" ", ""):
        raise ValueError("the Panode_e proposal does not restore the electron sign")
    cusp_expression = next(item for item in ledger["power_expressions"] if item["id"] == "Pcusp")
    if "+EI" not in cusp_expression["expression"].replace(" ", ""):
        raise ValueError("the executable Pcusp no longer carries +EI; the admitted equation set changed")
    if consistency["document"] != "docs/workstreams/global-plasma-closure-analysis.md":
        raise ValueError("the ledger names a different analysis document")
    if consistency["analysis_date"] != "2026-09-03":
        raise ValueError("the ledger analysis date differs from the admitted record")

    # --- Documented values (fixed patterns over the bound blob) ---------------- #
    documented = {
        "anode_fall_coefficient": float(ledger_coefficient),
        "continuation_epsilons": tuple(
            _doc_number(document_value(document_text, "continuation_header", group)) for group in "abcdef"
        ),
        "continuation_floors": tuple(
            _doc_number(document_value(document_text, "continuation_all_cells", group)) for group in "abcdef"
        ),
        "continuation_interior_floors": tuple(
            _doc_number(document_value(document_text, "continuation_interior_cells", group)) for group in "abcdef"
        ),
        "dm92_misfit": _doc_number(document_value(document_text, "dm92_misfit", "misfit")),
        "relaxed_root_depth_v": _doc_number(document_value(document_text, "relaxed_roots", "a")),
        "jacobian_rank": int(document_value(document_text, "jacobian_rank", "rank")),
        "jacobian_condition_max": float(document_value(document_text, "jacobian_condition", "hi")),
    }
    if documented["continuation_epsilons"] != CONTINUATION_EPSILONS:
        raise ValueError("the declared continuation ladder differs from the document's")
    if int(document_value(document_text, "continuation_setup", "v")) != CONTINUATION_VOLTAGE_V or int(document_value(document_text, "continuation_setup", "i")) != CONTINUATION_CURRENT_A:
        raise ValueError("the declared continuation operating point differs from the document's")
    if int(document_value(document_text, "relaxed_roots", "va")) != RELAXED_VOLTAGE_V or int(document_value(document_text, "relaxed_roots", "ia")) != RELAXED_CURRENT_A:
        raise ValueError("the declared relaxed-root operating point differs from the document's")
    if DOCUMENT_PATTERNS["continuation_anode_only"].search(document_text) is None:
        raise ValueError("the document's anode-only row no longer reads 'closes' at every eps")
    for name in ("audit_corrections_cancel", "classification_a", "classification_d", "kornfeld_id"):
        if DOCUMENT_PATTERNS[name].search(document_text) is None:
            raise ValueError(f"analysis document does not match the fixed pattern {name!r}")
    departures = check_against_document(recomputed, documented, TOLERANCES)

    # --- Protocol disclosure (13/80 probe): read, not recomputed --------------- #
    probe_text = protocol["prior_model_disclosure"]["corrected_four_cell_solver_probe"]
    probe = MDO_PROBE_PATTERN.search(probe_text)
    if probe is None:
        raise ValueError("the MDO protocol's four-cell probe disclosure does not match the fixed pattern")
    probe_closed = int(probe.group("closed"))
    probe_total = int(probe.group("total"))
    doc_probe_closed = int(document_value(document_text, "probe_reproduction", "closed"))
    doc_probe_total = int(document_value(document_text, "probe_reproduction", "total"))
    if (probe_closed, probe_total) != (doc_probe_closed, doc_probe_total):
        raise ValueError("the analysis document's probe reproduction differs from the protocol disclosure")

    # --- Legacy lineage: the same two power-row terms in FYP/Power_B_EQs.m ------ #
    legacy_lines = legacy_text.splitlines()
    cusp_line = next((index for index, line in enumerate(legacy_lines, start=1) if LEGACY_CUSP_LINE.match(line)), None)
    anode_line = next((index for index, line in enumerate(legacy_lines, start=1) if LEGACY_ANODE_LINE.match(line)), None)
    if cusp_line is None or anode_line is None:
        raise ValueError("the legacy Power_B_EQs.m blob does not carry the expected cusp and anode power terms")
    # The document names one line ("line 137 carries the same terms"); in the bound blob the
    # +IE cusp terms sit on the line before the anode-loss line, so the documented line must
    # fall inside the two-line span that carries both terms.
    documented_line = int(document_value(document_text, "legacy_line", "line"))
    if anode_line != cusp_line + 1 or not (cusp_line <= documented_line <= anode_line):
        raise ValueError("the legacy cusp-loss and anode-loss lines do not bracket the line the document names")
    legacy_ie_terms = legacy_lines[cusp_line - 1].count("+IE")
    if legacy_ie_terms != 3:
        raise ValueError("the legacy cusp-loss line does not carry three +IE terms")

    # --- Audit: acceptance by exit-flag status alone ------------------------------ #
    audit = AUDIT_ACCEPTANCE.search(audit_text)
    tolfun = AUDIT_TOLFUN.search(audit_text)
    if audit is None or tolfun is None:
        raise ValueError("AUDIT.md section 7 does not match the fixed acceptance pattern")
    if "IEPC 2007" not in references_text or "Kornfeld" not in references_text:
        raise ValueError("REFERENCES.md does not cite Kornfeld IEPC 2007")

    # --- Macros ----------------------------------------------------------------- #
    m = Macros()
    doc_in = [{"artifact": doc_path, "pointer": "regex:*"}]
    exec_in = [{"artifact": (PACKAGE_DIR / "residuals.py").as_posix(), "pointer": "potential_parametrized_state, global_row_closed_form, evaluate_residual"}]
    solver_in = exec_in + [{"artifact": (PACKAGE_DIR / "solver.py").as_posix(), "pointer": "solve_global_discharge_multistart"}]

    # Identity.
    m.add_derived("FccClassification", CLASSIFICATION, "ident", "classification string of the admitted result", "declared in the generator and the manifest", doc_in)
    m.add("FccCorrectionStatus", ledger_path, "/global_row_consistency/status", consistency["status"], "ident", "status of the proposed ledger correction")
    m.add("FccAnalysisDate", ledger_path, "/global_row_consistency/analysis_date", consistency["analysis_date"], "text", "date of the analysis recorded in the ledger")
    m.add_derived("FccAnalysisCommit", ANALYSIS_COMMIT_SHA, "sha_short", "analysis commit prefix", "commit that added the analysis document, ledger entry, diagnostics and tests", doc_in)
    m.add_derived("FccVerifiedTreeCommit", VERIFIED_TREE_COMMIT_SHA, "sha_short", "later commit at which every bound blob was verified unchanged", "git rev-parse at both commits", doc_in)
    m.add_derived("FccMdoPreregCommit", MDO_PREREGISTRATION_COMMIT_SHA, "sha_short", "MDO preregistration commit whose frozen protocol carries the probe disclosure", "git rev-parse", [{"artifact": PROTOCOL.as_posix(), "pointer": ""}])
    m.add_derived("FccBoundFileCount", len(bound), "int", "files bound by blob and SHA-256", "len(SOURCE_ROLES)", doc_in)
    m.add_derived("FccPackageFileCount", len(executed), "int", "package files executed and required to equal their bound blobs", "len(PACKAGE_FILES)", exec_in)
    m.add_derived("FccPackageMatches", True, "bool", "executed package equals the bound blobs", "LF-normalised SHA-256 of every checkout file equals the blob SHA-256 at the analysis commit", exec_in)
    m.add_derived("FccProbeSource", PROBE_SOURCE, "ident", "where the probe pattern was read from", "read from the frozen MDO protocol disclosure; not recomputed", [{"artifact": PROTOCOL.as_posix(), "pointer": "/prior_model_disclosure/corrected_four_cell_solver_probe"}])
    m.add("FccKornfeldId", doc_path, "regex:kornfeld_id[id]", document_value(document_text, "kornfeld_id", "id"), "ident", "IEPC identifier of the source paper as cited in the analysis document")
    m.add("FccKornfeldAssumption", doc_path, "regex:kornfeld_assumption[n]", int(document_value(document_text, "kornfeld_assumption", "n")), "int", "number of the source assumption that books recombination losses at boundaries")
    m.add("FccDmLabel", doc_path, "regex:dm92_probabilities[label]", document_value(document_text, "dm92_probabilities", "label"), "text", "label of the source's published operating point")
    m.add("FccDmProbabilities", doc_path, "regex:dm92_probabilities[p]", [float(x) for x in document_value(document_text, "dm92_probabilities", "p").split(",")], "list_g", "published cusp probabilities of that operating point")

    # Ledger structure.
    m.add_derived("FccRowCount", len(rows), "int", "residual rows of the ledger", "len(residual_rows)", [{"artifact": ledger_path, "pointer": "/residual_rows"}])
    m.add_derived("FccGlobalRowIndex", len(rows) - 1, "int", "index of the global power row", "len(residual_rows) - 1 (row id R27)", [{"artifact": ledger_path, "pointer": "/residual_rows/27/id"}])
    m.add_derived("FccLastCellRowIndex", len(rows) - 2, "int", "index of the last cell row", "len(residual_rows) - 2 (row id R26)", [{"artifact": ledger_path, "pointer": "/residual_rows/26/id"}])
    m.add_derived("FccPowerExpressionCount", len(ledger["power_expressions"]), "int", "power expressions of the ledger", "len(power_expressions)", [{"artifact": ledger_path, "pointer": "/power_expressions"}])
    m.add_derived("FccCellCount", len(DM92_PROBABILITIES), "int", "cells of the four-cell model", "len(cusp_arrival_probabilities)", [{"artifact": ledger_path, "pointer": "/state_layout"}])
    m.add_derived("FccStateDimension", recomputed["state_dimension"], "int", "state variables", "len(PlasmaState.to_vector())", exec_in, recomputed=True)
    m.add_derived("FccRowCountRecomputed", recomputed["row_count"], "int", "rows evaluated by the residual", "len(evaluate_residual(...).normalized)", exec_in, recomputed=True)
    if recomputed["row_count"] != len(rows) or recomputed["state_dimension"] != 25:
        raise ValueError("the executed residual does not have the ledger's 28-by-25 shape")
    m.add("FccLedgerCoefficient", ledger_path, "regex:LEDGER_CLOSED_FORM[coefficient]", ledger_coefficient, "int", "anode-fall coefficient in the ledger's closed form")
    m.add_derived("FccAnodeFallCoefficient", round_sig(recomputed["anode_fall_coefficient"], 12), "g", "recomputed anode-fall coefficient", "R27 / ((j_e3 + I4)(phi_4 - Ua)) on the manifold at p = 0", exec_in, recomputed=True, rounding=12)
    m.add("FccPcuspCurrent", ledger_path, "/global_row_consistency/proposed_corrections/0/current_expression", proposals["Pcusp"]["current_expression"], "expr", "executable cusp-loss expression")
    m.add("FccPcuspProposed", ledger_path, "/global_row_consistency/proposed_corrections/0/proposed_expression", proposals["Pcusp"]["proposed_expression"], "expr", "proposed cusp-loss expression (not accepted)")
    m.add("FccPanodeCurrent", ledger_path, "/global_row_consistency/proposed_corrections/1/current_expression", proposals["Panode_e"]["current_expression"], "expr", "executable anode electron expression")
    m.add("FccPanodeProposed", ledger_path, "/global_row_consistency/proposed_corrections/1/proposed_expression", proposals["Panode_e"]["proposed_expression"], "expr", "proposed anode electron expression (not accepted)")
    m.add("FccLedgerDmMisfit", ledger_path, "/evidence_comparisons/dm92_published_max_normalized_residual", ledger["evidence_comparisons"]["dm92_published_max_normalized_residual"], "sci2", "largest normalized residual of the rounded published table recorded by the ledger")
    m.add("FccLedgerDmRow", ledger_path, "/evidence_comparisons/dm92_published_max_row", ledger["evidence_comparisons"]["dm92_published_max_row"], "int", "row of that residual")
    m.add("FccLedgerDmRank", ledger_path, "/evidence_comparisons/jacobian_rank_at_dm92", ledger["evidence_comparisons"]["jacobian_rank_at_dm92"], "int", "Jacobian rank at the published state recorded by the ledger")
    m.add("FccLedgerSingleStartFloor", ledger_path, "/evidence_comparisons/single_start_observed_residual_floor_500_iterations", ledger["evidence_comparisons"]["single_start_observed_residual_floor_500_iterations"], "sci2", "single-start residual floor recorded by the ledger")
    if ledger["evidence_comparisons"]["dm92_published_max_row"] != len(rows) - 1:
        raise ValueError("the ledger's published-state misfit is not on the global row")

    # Closed form: documented and recomputed.
    m.add("FccDocClosedFormRelDiff", doc_path, "regex:closed_form_verification[rel]", _doc_number(document_value(document_text, "closed_form_verification", "rel")), "sci1", "documented relative agreement of the closed form with the evaluated global row")
    m.add("FccDocClosedFormSamples", doc_path, "regex:closed_form_verification[n]", int(document_value(document_text, "closed_form_verification", "n")), "int", "documented random cases behind that agreement")
    m.add("FccDocManifoldSamples", doc_path, "regex:manifold_verification[n]", int(document_value(document_text, "manifold_verification", "n")), "int", "documented random cases behind the manifold check")
    m.add("FccDocManifoldBound", doc_path, "regex:manifold_verification[bound]", _doc_number(document_value(document_text, "manifold_verification", "bound")), "sci0", "documented bound on the R00-R26 residual on the manifold")
    m.add_derived("FccClosedFormRelDiff", round_sig(recomputed["closed_form_relative_difference"], ROUNDING["closed_form_relative_difference"]), "sci1", "recomputed maximum relative difference between the closed form and the evaluated global row", "max over the seeded sample of |raw R27 - global_row_closed_form| / |global_row_closed_form|", exec_in, recomputed=True, rounding=ROUNDING["closed_form_relative_difference"])
    m.add_derived("FccClosedFormSamples", SAMPLE_COUNT, "int", "recomputed sample size", "SAMPLE_COUNT", exec_in, recomputed=True)
    m.add_derived("FccClosedFormSeed", SAMPLE_SEED, "int", "seed of the recomputed sample", "SAMPLE_SEED (random.Random)", exec_in, recomputed=True)
    m.add_derived("FccClosedFormBound", TOLERANCES["closed_form_relative_difference_upper_bound"], "sci0", "declared upper bound on the recomputed relative difference", "TOLERANCES", exec_in)
    m.add_derived("FccManifoldMaxResidual", round_sig(recomputed["manifold_normalized_residual"], ROUNDING["manifold_normalized_residual"]), "sci1", "recomputed maximum normalized residual of rows R00-R26 on the manifold", "max over the seeded sample of max|normalized R00..R26|", exec_in, recomputed=True, rounding=ROUNDING["manifold_normalized_residual"])
    m.add_derived("FccManifoldBound", TOLERANCES["manifold_normalized_residual_upper_bound"], "sci0", "declared upper bound on the manifold residual", "TOLERANCES", exec_in)
    m.add_derived("FccSampleVoltages", list(SAMPLE_VOLTAGES), "list_g", "anode voltages of the seeded sample (V)", "SAMPLE_VOLTAGES", exec_in)
    m.add_derived("FccSampleCurrents", list(SAMPLE_CURRENTS), "list_g", "anode currents of the seeded sample (A)", "SAMPLE_CURRENTS", exec_in)
    m.add_derived("FccSampleProbabilityUpper", SAMPLE_PROBABILITY_UPPER, "g", "upper bound of the uniform cusp probabilities in the seeded sample", "SAMPLE_PROBABILITY_UPPER", exec_in)

    # Continuation: documented and recomputed.
    m.add("FccDocContinuationVoltage", doc_path, "regex:continuation_setup[v]", int(document_value(document_text, "continuation_setup", "v")), "int", "documented continuation anode voltage (V)")
    m.add("FccDocContinuationCurrent", doc_path, "regex:continuation_setup[i]", int(document_value(document_text, "continuation_setup", "i")), "int", "documented continuation anode current (A)")
    m.add("FccDocContinuationStarts", doc_path, "regex:continuation_setup[starts]", int(document_value(document_text, "continuation_setup", "starts")), "int", "documented multistart count")
    m.add("FccDocContinuationIterations", doc_path, "regex:continuation_setup[iterations]", int(document_value(document_text, "continuation_setup", "iterations")), "int", "documented iteration limit")
    m.add_derived("FccContinuationStarts", CONTINUATION_START_COUNT, "int", "recomputation start count", "CONTINUATION_START_COUNT", solver_in, recomputed=True)
    m.add_derived("FccAnodeOnlyStarts", ANODE_ONLY_START_COUNT, "int", "recomputation start count of the anode-only closures", "ANODE_ONLY_START_COUNT", solver_in, recomputed=True)
    m.add_derived("FccContinuationIterations", CONTINUATION_MAX_ITERATIONS, "int", "recomputation iteration limit", "CONTINUATION_MAX_ITERATIONS", solver_in, recomputed=True)
    m.add_derived("FccResidualTolerance", RESIDUAL_TOLERANCE, "sci0", "residual tolerance of the recomputation solves", "RESIDUAL_TOLERANCE", solver_in, recomputed=True)
    m.add_derived("FccLadderCount", len(CONTINUATION_EPSILONS), "int", "rungs of the continuation ladder", "len(CONTINUATION_EPSILONS)", doc_in)
    m.add("FccDocAnodeOnlyResidual", doc_path, "regex:anode_only_residual[r]", _doc_number(document_value(document_text, "anode_only_residual", "r")), "sci0", "documented residual of the anode-only closures")
    tokens = ("A", "B", "C", "D", "E", "F")
    continuation_rows: list[str] = []
    anode_only_max = 0.0
    condition_max = 0.0
    for token, group, step, floor, interior_floor in zip(tokens, "abcdef", recomputed["ladder"], documented["continuation_floors"], documented["continuation_interior_floors"], strict=True):
        m.add(f"FccDocEps{token}", doc_path, f"regex:continuation_header[{group}]", step["epsilon"], "g", f"continuation epsilon {token}")
        m.add(f"FccDocFloor{token}", doc_path, f"regex:continuation_all_cells[{group}]", floor, "sci2", f"documented floor for p = eps (1,1,1,1), rung {token}")
        m.add(f"FccDocInteriorFloor{token}", doc_path, f"regex:continuation_interior_cells[{group}]", interior_floor, "sci2", f"documented floor for p = eps (1,1,1,0), rung {token}")
        recomputed_floor = round_sig(step["floor"], ROUNDING["continuation_floor"])
        m.add_derived(f"FccFloor{token}", recomputed_floor, "sci2", f"recomputed floor for p = eps (1,1,1,1), rung {token}", "residual_floor of solve_global_discharge_multistart (max|normalized r| at the least-squares stall)", solver_in, recomputed=True, rounding=ROUNDING["continuation_floor"])
        slope = round_sig(step["floor"] / step["epsilon"], ROUNDING["continuation_slope"])
        m.add_derived(f"FccSlope{token}", slope, "sig3", f"recomputed floor divided by eps, rung {token}", "floor / eps", solver_in, recomputed=True, rounding=ROUNDING["continuation_slope"])
        anode_residual = round_sig(step["anode_only_residual"], ROUNDING["anode_only_residual"])
        m.add_derived(f"FccAnodeOnlyResidual{token}", anode_residual, "sci0", f"recomputed max|normalized r| of the anode-only closure, rung {token}", "residual_inf_norm of the converged anode-only solve", solver_in, recomputed=True, rounding=ROUNDING["anode_only_residual"])
        anode_only_max = max(anode_only_max, step["anode_only_residual"])
        condition_max = max(condition_max, step["jacobian_condition"])
        continuation_rows.append(
            f"{format_value('g', step['epsilon'])} & {format_value('sci2', floor)} & {format_value('sci2', interior_floor)} & "
            f"{format_value('sci2', recomputed_floor)} & {format_value('sig3', slope)} & {step['jacobian_rank']} & "
            f"closes & {format_value('sci0', anode_residual)}\\\\"
        )
    floors = [step["floor"] for step in recomputed["ladder"]]
    m.add_derived("FccFloorMin", round_sig(min(floors), ROUNDING["continuation_floor"]), "sci2", "smallest recomputed floor", "min over the ladder", solver_in, recomputed=True, rounding=ROUNDING["continuation_floor"])
    m.add_derived("FccFloorMax", round_sig(max(floors), ROUNDING["continuation_floor"]), "sci2", "largest recomputed floor", "max over the ladder", solver_in, recomputed=True, rounding=ROUNDING["continuation_floor"])
    m.add_derived("FccDocFloorMin", min(documented["continuation_floors"]), "sci2", "smallest documented floor", "min over the documented ladder", doc_in)
    m.add_derived("FccDocFloorMax", max(documented["continuation_floors"]), "sci2", "largest documented floor", "max over the documented ladder", doc_in)
    m.add_derived("FccFloorDepartureMax", round_sig(departures["continuation_max_relative_departure"], ROUNDING["departure"]), "pct0", "largest relative departure of a recomputed floor from the document", "max over the ladder of |recomputed - documented| / documented", solver_in + doc_in, recomputed=True, rounding=ROUNDING["departure"])
    m.add_derived("FccFloorTolerance", TOLERANCES["continuation_floor_relative"], "pct0", "declared relative tolerance on each recomputed floor", "TOLERANCES", doc_in)
    m.add_derived("FccSlopeSpread", round_sig(departures["continuation_slope_spread"], ROUNDING["departure"]), "fixed2", "ratio of the largest to the smallest recomputed floor/eps over the ladder", "max(floor/eps) / min(floor/eps)", solver_in, recomputed=True, rounding=ROUNDING["departure"])
    m.add_derived("FccSlopeSpreadMax", TOLERANCES["continuation_slope_spread_maximum"], "g", "declared maximum of that ratio (no branch)", "TOLERANCES", doc_in)
    m.add_derived("FccBranchFound", any(step["converged"] for step in recomputed["ladder"]), "bool", "whether any interior rung of the ladder closed", "any(converged)", solver_in, recomputed=True)
    m.add_derived("FccDominantRowIsGlobal", all(step["dominant_row"] == len(rows) - 1 for step in recomputed["ladder"]), "bool", "whether the global row dominates every recomputed floor", "all(argmax |normalized r| == 27)", solver_in, recomputed=True)
    m.add_derived("FccAnodeOnlyClosed", sum(1 for step in recomputed["ladder"] if step["anode_only_converged"]), "int", "anode-only rungs that closed", "count(anode_only_converged)", solver_in, recomputed=True)
    m.add_derived("FccAnodeOnlyMaxResidual", round_sig(anode_only_max, ROUNDING["anode_only_residual"]), "sci0", "largest max|normalized r| over the anode-only closures", "max over the ladder", solver_in, recomputed=True, rounding=ROUNDING["anode_only_residual"])
    m.add_derived("FccAnodeOnlyBound", TOLERANCES["anode_only_residual_upper_bound"], "sci0", "declared bound on the anode-only residual", "TOLERANCES", doc_in)
    anode_phi = [step["anode_only_phi4_minus_ua"] for step in recomputed["ladder"]]
    if any(value is None for value in anode_phi):
        raise ValueError("an anode-only closure returned no state")
    m.add_derived("FccAnodeOnlyPhiFourGap", round_sig(max(abs(value) for value in anode_phi), 1), "g", "largest |phi_4 - Ua| over the anode-only closures (V)", "max |state.plasma_potential_v[3] - Ua|", solver_in, recomputed=True, rounding=1)

    # Jacobian.
    m.add("FccDocJacobianRank", doc_path, "regex:jacobian_rank[rank]", documented["jacobian_rank"], "int", "documented Jacobian rank at every floor and closed point")
    m.add("FccDocJacobianColumns", doc_path, "regex:jacobian_rank[n]", int(document_value(document_text, "jacobian_rank", "n")), "int", "documented state dimension")
    m.add("FccDocConditionMin", doc_path, "regex:jacobian_condition[lo]", int(document_value(document_text, "jacobian_condition", "lo")), "int", "documented smallest condition of the independent subspace")
    m.add("FccDocConditionMax", doc_path, "regex:jacobian_condition[hi]", int(document_value(document_text, "jacobian_condition", "hi")), "int", "documented largest condition of the independent subspace")
    ranks = {step["jacobian_rank"] for step in recomputed["ladder"]}
    m.add_derived("FccJacobianRank", ranks.pop() if len(ranks) == 1 else -1, "int", "recomputed Jacobian rank at every floor point of the ladder", "SolverDiagnostics.jacobian_rank (column-pivoted QR)", solver_in, recomputed=True)
    m.add_derived("FccJacobianNullity", recomputed["state_dimension"] - documented["jacobian_rank"], "int", "dimension of the potential null space", "state_dimension - rank", solver_in, recomputed=True)
    m.add_derived("FccConditionMax", round_sig(condition_max, ROUNDING["jacobian_condition"]), "sig2", "largest recomputed condition estimate of the independent subspace over the ladder", "max SolverDiagnostics.jacobian_condition_estimate", solver_in, recomputed=True, rounding=ROUNDING["jacobian_condition"])

    # Global search: documented only (scipy-free, minutes of solver time; not recomputed).
    m.add("FccDocDeDimension", doc_path, "regex:differential_evolution[dim]", int(document_value(document_text, "differential_evolution", "dim")), "int", "documented box dimension of the differential-evolution search")
    m.add("FccDocDeEvaluations", doc_path, "regex:differential_evolution[n]", int(_doc_number(document_value(document_text, "differential_evolution", "n"))), "int_comma", "documented differential-evolution evaluations")
    m.add("FccDocDeBest", doc_path, "regex:differential_evolution[best]", _doc_number(document_value(document_text, "differential_evolution", "best")), "sci2", "documented best max|r| of the differential-evolution search")
    m.add("FccDocDeVoltage", doc_path, "regex:differential_evolution[v]", int(document_value(document_text, "differential_evolution", "v")), "int", "documented anode voltage of the global search (V)")
    m.add("FccDocDeCurrent", doc_path, "regex:differential_evolution[i]", int(document_value(document_text, "differential_evolution", "i")), "int", "documented anode current of the global search (A)")
    m.add("FccDocLmStarts", doc_path, "regex:random_starts[n]", int(document_value(document_text, "random_starts", "n")), "int", "documented random feasible starts")
    m.add("FccDocLmClosed", doc_path, "regex:random_starts[closed]", int(document_value(document_text, "random_starts", "closed")), "int", "documented closures among the random starts")
    m.add("FccDocLmFloorMin", doc_path, "regex:random_starts[min]", _doc_number(document_value(document_text, "random_starts", "min")), "sci2", "documented smallest floor of the random starts")
    m.add("FccDocLmFloorMedian", doc_path, "regex:random_starts[median]", _doc_number(document_value(document_text, "random_starts", "median")), "sci2", "documented median floor of the random starts")
    m.add("FccDocLmFloorMax", doc_path, "regex:random_starts[max]", _doc_number(document_value(document_text, "random_starts", "max")), "sci2", "documented largest floor of the random starts")
    if int(document_value(document_text, "random_starts", "total")) != int(document_value(document_text, "random_starts", "n")):
        raise ValueError("the document's random-start total differs from its start count")

    # Relaxed roots: documented and one recomputed example.
    m.add("FccDocRelaxedResidual", doc_path, "regex:relaxed_roots[r]", _doc_number(document_value(document_text, "relaxed_roots", "r")), "sci0", "documented residual of the relaxed-constraint roots")
    m.add("FccDocRelaxedDepthMin", doc_path, "regex:relaxed_roots[lo]", _doc_number(document_value(document_text, "relaxed_roots", "lo")), "fixed1", "documented smallest depth of a relaxed root below the anode (V)")
    m.add("FccDocRelaxedDepthMax", doc_path, "regex:relaxed_roots[hi]", _doc_number(document_value(document_text, "relaxed_roots", "hi")), "fixed1", "documented largest depth of a relaxed root below the anode (V)")
    m.add("FccDocRelaxedDepthThreeHundred", doc_path, "regex:relaxed_roots[a]", _doc_number(document_value(document_text, "relaxed_roots", "a")), "fixed2", "documented relaxed-root depth at the published probabilities and 300 V (V)")
    m.add("FccDocRelaxedDepthThousand", doc_path, "regex:relaxed_roots[b]", _doc_number(document_value(document_text, "relaxed_roots", "b")), "fixed2", "documented relaxed-root depth at the published probabilities and 1000 V (V)")
    m.add_derived("FccRelaxedDepth", round_sig(recomputed["relaxed_root_depth_v"], ROUNDING["relaxed_root_depth_v"]), "fixed2", "recomputed depth of one relaxed-constraint root below the anode (V)", "Ua - phi_4 at the bisection root of raw R27 with fixed interior potentials", exec_in, recomputed=True, rounding=ROUNDING["relaxed_root_depth_v"])
    m.add_derived("FccRelaxedResidual", round_sig(recomputed["relaxed_root_residual"], ROUNDING["relaxed_root_residual"]), "sci0", "recomputed max|normalized r| at that root", "max|normalized residual|", exec_in, recomputed=True, rounding=ROUNDING["relaxed_root_residual"])
    m.add_derived("FccRelaxedFeasible", recomputed["relaxed_root_feasible"], "bool", "whether that root is admissible", "is_feasible", exec_in, recomputed=True)
    m.add_derived("FccRelaxedInterior", list(RELAXED_INTERIOR_POTENTIALS_V), "list_g", "fixed interior potentials of the relaxed-root bisection (V)", "RELAXED_INTERIOR_POTENTIALS_V", exec_in)
    m.add_derived("FccRelaxedVoltage", RELAXED_VOLTAGE_V, "g", "anode voltage of the relaxed-root bisection (V)", "RELAXED_VOLTAGE_V", exec_in)
    m.add_derived("FccRelaxedCurrent", RELAXED_CURRENT_A, "g", "anode current of the relaxed-root bisection (A)", "RELAXED_CURRENT_A", exec_in)

    # Published-state misfit.
    m.add("FccDocDmMisfit", doc_path, "regex:dm92_misfit[misfit]", documented["dm92_misfit"], "sci2", "documented global-row misfit of the published state on the exact manifold")
    m.add("FccDocDmLedgerMisfit", doc_path, "regex:dm92_misfit[ledger]", _doc_number(document_value(document_text, "dm92_misfit", "ledger")), "sci2", "documented ledger misfit of the rounded published table")
    m.add("FccDocDmVoltage", doc_path, "regex:dm92_misfit[v]", int(document_value(document_text, "dm92_misfit", "v")), "int", "documented published-state anode voltage (V)")
    m.add("FccDocDmCurrent", doc_path, "regex:dm92_misfit[i]", int(document_value(document_text, "dm92_misfit", "i")), "int", "documented published-state anode current (A)")
    m.add_derived("FccDmMisfit", round_sig(recomputed["dm92_misfit"], ROUNDING["dm92_misfit"]), "sci2", "recomputed normalized global-row residual of the published state on the exact manifold", "normalized R27 of potential_parametrized_state at the published potentials", exec_in, recomputed=True, rounding=ROUNDING["dm92_misfit"])
    m.add_derived("FccDmManifoldResidual", round_sig(recomputed["dm92_manifold_residual"], 1), "sci0", "recomputed max|normalized R00..R26| at that state", "max|normalized R00..R26|", exec_in, recomputed=True, rounding=1)
    m.add_derived("FccDmPotentials", list(DM92_POTENTIALS_V), "list_g", "published cell potentials used for the misfit (V)", "DM92_POTENTIALS_V", exec_in)

    # Probe (read from the frozen protocol; reproduction documented).
    probe_pointer = "/prior_model_disclosure/corrected_four_cell_solver_probe"
    m.add("FccProbeClosed", PROTOCOL.as_posix(), f"regex:MDO_PROBE_PATTERN[closed]@{probe_pointer}", probe_closed, "int", "probe cases that closed (frozen MDO protocol)")
    m.add("FccProbeTotal", PROTOCOL.as_posix(), f"regex:MDO_PROBE_PATTERN[total]@{probe_pointer}", probe_total, "int", "probe cases (frozen MDO protocol)")
    m.add("FccProbeFloorMin", PROTOCOL.as_posix(), f"regex:MDO_PROBE_PATTERN[rmin]@{probe_pointer}", float(probe.group("rmin")), "sci1", "smallest probe residual floor (frozen MDO protocol)")
    m.add("FccProbeFloorMax", PROTOCOL.as_posix(), f"regex:MDO_PROBE_PATTERN[rmax]@{probe_pointer}", float(probe.group("rmax")), "sig3", "largest probe residual floor (frozen MDO protocol)")
    m.add("FccDocProbeClosed", doc_path, "regex:probe_reproduction[closed]", doc_probe_closed, "int", "probe cases that closed in the document's exact reproduction")
    m.add("FccDocProbeTotal", doc_path, "regex:probe_reproduction[total]", doc_probe_total, "int", "probe cases in the document's exact reproduction")
    m.add("FccDocProbeSeconds", doc_path, "regex:probe_reproduction[seconds]", int(document_value(document_text, "probe_reproduction", "seconds")), "int", "documented wall time of the probe reproduction (s)")
    m.add("FccDocProbeFloorMin", doc_path, "regex:probe_reproduction[fmin]", _doc_number(document_value(document_text, "probe_reproduction", "fmin")), "sci2", "documented smallest floor of the probe reproduction")
    m.add("FccDocProbeFloorMax", doc_path, "regex:probe_reproduction[fmax]", _doc_number(document_value(document_text, "probe_reproduction", "fmax")), "sci2", "documented largest floor of the probe reproduction")
    m.add("FccDocZeroCuspAfter", doc_path, "regex:zero_cusp_grid[after]", int(document_value(document_text, "zero_cusp_grid", "after")), "int", "zero-cusp grid cases closing after the projection fix")
    m.add("FccDocZeroCuspBefore", doc_path, "regex:zero_cusp_grid[before]", int(document_value(document_text, "zero_cusp_grid", "before")), "int", "zero-cusp grid cases closing before the projection fix")
    m.add("FccDocZeroCuspTotal", doc_path, "regex:zero_cusp_grid[total]", int(document_value(document_text, "zero_cusp_grid", "total")), "int", "zero-cusp grid cases")

    # Attribution and correction.
    m.add("FccDocLegacyLine", doc_path, "regex:legacy_line[line]", int(document_value(document_text, "legacy_line", "line")), "int", "documented line of the legacy cusp-loss term")
    m.add_derived("FccLegacyCuspLine", cusp_line, "int", "line of the legacy cusp-loss term found in the bound blob", "first line matching LEGACY_CUSP_LINE", [{"artifact": LEGACY.as_posix(), "pointer": "regex:LEGACY_CUSP_LINE"}], recomputed=True)
    m.add_derived("FccLegacyAnodeLine", anode_line, "int", "line of the legacy anode-loss term found in the bound blob", "first line matching LEGACY_ANODE_LINE", [{"artifact": LEGACY.as_posix(), "pointer": "regex:LEGACY_ANODE_LINE"}], recomputed=True)
    m.add_derived("FccLegacyIeTerms", legacy_ie_terms, "int", "+IE terms on the legacy cusp-loss line", "count('+IE') on that line", [{"artifact": LEGACY.as_posix(), "pointer": "regex:LEGACY_CUSP_LINE"}], recomputed=True)
    m.add_derived("FccLegacyBlob", bound[LEGACY.as_posix()]["git_blob"], "sha_short", "Git blob prefix of the legacy file", "git rev-parse", [{"artifact": LEGACY.as_posix(), "pointer": ""}])
    m.add("FccDocCorrectedRankBefore", doc_path, "regex:corrected_rank[before]", int(document_value(document_text, "corrected_rank", "before")), "int", "structural rank before the proposed correction")
    m.add("FccDocCorrectedRankAfter", doc_path, "regex:corrected_rank[after]", int(document_value(document_text, "corrected_rank", "after")), "int", "structural rank if the proposed correction were accepted")
    m.add("FccDocCorrectedNullity", doc_path, "regex:corrected_rank[nullity]", int(document_value(document_text, "corrected_rank", "nullity")), "int", "nullity if the proposed correction were accepted")
    m.add_derived("FccAuditAcceptedFlags", audit.group("flags"), "text", "legacy exit flags accepted by status alone (AUDIT.md section 7)", "regex group 'flags' of AUDIT_ACCEPTANCE", [{"artifact": AUDIT.as_posix(), "pointer": "regex:AUDIT_ACCEPTANCE"}])
    m.add_derived("FccAuditRejectedFlag", int(audit.group("rejected")), "int", "legacy exit flag rejected (AUDIT.md section 7)", "regex group 'rejected' of AUDIT_ACCEPTANCE", [{"artifact": AUDIT.as_posix(), "pointer": "regex:AUDIT_ACCEPTANCE"}])
    m.add_derived("FccAuditTolFun", tolfun.group("tolfun"), "text", "legacy lsqnonlin TolFun (AUDIT.md section 7)", "regex group 'tolfun' of AUDIT_TOLFUN", [{"artifact": AUDIT.as_posix(), "pointer": "regex:AUDIT_TOLFUN"}])
    m.add_derived("FccAuditSolverLine", int(audit.group("line")), "int", "legacy line that discards the residual norm (AUDIT.md section 7)", "regex group 'line' of AUDIT_ACCEPTANCE", [{"artifact": AUDIT.as_posix(), "pointer": "regex:AUDIT_ACCEPTANCE"}])

    # Consistency between documented and protocol probe counts is asserted above.
    if documented["jacobian_rank"] != ledger["evidence_comparisons"]["jacobian_rank_at_dm92"]:
        raise ValueError("the documented rank differs from the ledger's rank at the published state")
    if int(document_value(document_text, "corrected_rank", "before")) != documented["jacobian_rank"]:
        raise ValueError("the document's rank before correction differs from its floor-point rank")

    # --- Generated TeX --------------------------------------------------------- #
    lines = [
        "% Generated by paper/scripts/generate_four_cell_closure_evidence.py; do not hand edit.",
        f"% Evidence: {DOCUMENT.as_posix()} and {LEDGER.as_posix()} at commit {ANALYSIS_COMMIT_SHA};",
        "% recomputed values come from the checkout's cft_revival.plasma, which must equal the bound blobs.",
        "% Every macro value traces to a path and pointer/pattern recorded in paper/evidence/four-cell-closure.json.",
    ]
    for item in m.items:
        lines.append(f"\\newcommand{{\\{item['name']}}}{{{item['value']}}}")
    artifact_open = f"\\ArtifactClaim{{{ARTIFACT_CLAIM_ID}}}{{{ARTIFACT_ID}}}{{%"
    lines.append("\\newcommand{\\FccContinuationTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Continuation in the cusp probabilities at \\FccDocContinuationVoltage{}~V and "
        "\\FccDocContinuationCurrent{}~A. Documented floors (analysis document; \\FccDocContinuationStarts{} starts, "
        "\\FccDocContinuationIterations{} iterations) are the least-squares stall values $\\max_i|r_i|$ of the "
        "production solver for $\\mathbf p=\\varepsilon(1,1,1,1)$ and $\\varepsilon(1,1,1,0)$; recomputed floors "
        "(this checker; \\FccContinuationStarts{} start, \\FccContinuationIterations{} iterations) are recorded to "
        "three significant digits with the floor divided by $\\varepsilon$ and the Jacobian rank at the stall. The "
        "last two columns give the anode-only pattern $\\mathbf p=(0,0,0,\\varepsilon)$: documented outcome and "
        "recomputed $\\max_i|r_i|$ of the converged state (\\FccAnodeOnlyStarts{} starts; convergence means "
        "$\\max_i|r_i|$ at or below \\FccResidualTolerance).}"
    )
    lines.append("\\label{tab:four-cell-closure-continuation}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{lrrrrrlr}")
    lines.append("\\toprule")
    lines.append(
        "$\\varepsilon$ & \\shortstack[r]{documented\\\\$(1,1,1,1)$} & \\shortstack[r]{documented\\\\$(1,1,1,0)$} & "
        "\\shortstack[r]{recomputed\\\\$(1,1,1,1)$} & floor$/\\varepsilon$ & rank & "
        "\\shortstack[l]{anode-only\\\\documented} & \\shortstack[r]{anode-only\\\\recomputed}\\\\"
    )
    lines.append("\\midrule")
    lines.extend(continuation_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    search_rows = [
        f"differential evolution over the full \\FccDocDeDimension-value box at \\FccDocDeVoltage{{}}~V, \\FccDocDeCurrent{{}}~A ({format_value('int_comma', int(_doc_number(document_value(document_text, 'differential_evolution', 'n'))))} evaluations) & best $\\max_i|r_i|$ & {format_value('sci2', _doc_number(document_value(document_text, 'differential_evolution', 'best')))} & documented\\\\",
        f"random feasible starts through the production solver ({format_value('int', int(document_value(document_text, 'random_starts', 'n')))} starts) & closed / floors (min, median, max) & {format_value('int', int(document_value(document_text, 'random_starts', 'closed')))}; {format_value('sci2', _doc_number(document_value(document_text, 'random_starts', 'min')))}, {format_value('sci2', _doc_number(document_value(document_text, 'random_starts', 'median')))}, {format_value('sci2', _doc_number(document_value(document_text, 'random_starts', 'max')))} & documented\\\\",
        f"relaxing $\\varphi_N\\ge U_a$ (exact roots, $\\max_i|r_i|\\sim{format_value('sci0', _doc_number(document_value(document_text, 'relaxed_roots', 'r')))[1:-1]}$) & depth of $\\varphi_N$ below $U_a$ (V) & {format_value('fixed1', _doc_number(document_value(document_text, 'relaxed_roots', 'lo')))}--{format_value('fixed1', _doc_number(document_value(document_text, 'relaxed_roots', 'hi')))} & documented\\\\",
        f"one relaxed root by bisection (fixed interior potentials, published probabilities) & depth (V); $\\max_i|r_i|$; admissible & {format_value('fixed2', round_sig(recomputed['relaxed_root_depth_v'], ROUNDING['relaxed_root_depth_v']))}; {format_value('sci0', round_sig(recomputed['relaxed_root_residual'], ROUNDING['relaxed_root_residual']))}; {format_value('bool', recomputed['relaxed_root_feasible'])} & recomputed\\\\",
        f"Jacobian at every floor and closed point & rank of {format_value('int', int(document_value(document_text, 'jacobian_rank', 'n')))}; condition & {format_value('int', documented['jacobian_rank'])}; {format_value('int', int(document_value(document_text, 'jacobian_condition', 'lo')))}--{format_value('int', int(document_value(document_text, 'jacobian_condition', 'hi')))} & documented\\\\",
        f"Jacobian at the recomputed ladder floors & rank; largest condition & {format_value('int', documented['jacobian_rank'])}; {format_value('sig2', round_sig(condition_max, ROUNDING['jacobian_condition']))} & recomputed\\\\",
        f"published-state misfit (\\FccDmLabel{{}} potentials on the exact manifold) & normalized global row & {format_value('sci2', documented['dm92_misfit'])} (document); {format_value('sci2', ledger['evidence_comparisons']['dm92_published_max_normalized_residual'])} (ledger, rounded table) & documented\\\\",
        f"published-state misfit, recomputed & normalized global row; $\\max|R_{{00..{len(rows) - 2}}}|$ & {format_value('sci2', round_sig(recomputed['dm92_misfit'], ROUNDING['dm92_misfit']))}; {format_value('sci0', round_sig(recomputed['dm92_manifold_residual'], 1))} & recomputed\\\\",
        f"solver probe of the optimisation protocol & closed / cases & {probe_closed} / {probe_total} & frozen protocol\\\\",
        f"exact reproduction of that probe & closed / cases; floors & {doc_probe_closed} / {doc_probe_total}; {format_value('sci2', _doc_number(document_value(document_text, 'probe_reproduction', 'fmin')))}--{format_value('sci2', _doc_number(document_value(document_text, 'probe_reproduction', 'fmax')))} & documented\\\\",
    ]
    lines.append("\\newcommand{\\FccGlobalSearchTable}{%")
    lines.append(artifact_open)
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Global search for an admissible root and related checks at the published cusp probabilities. "
        "Rows marked documented are read from the analysis document at the analysis revision and were not "
        "recomputed by the checker (the differential-evolution and random-start searches need a SciPy optimiser "
        "and minutes of solver time); rows marked recomputed are evaluated by the checker from the bound "
        "package at every run; the probe row is read from the frozen protocol of the optimisation campaign.}"
    )
    lines.append("\\label{tab:four-cell-closure-search}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{5.4cm}>{\\raggedright\\arraybackslash}p{3.2cm}"
        ">{\\raggedright\\arraybackslash}p{4.6cm}l}"
    )
    lines.append("\\toprule")
    lines.append("check & quantity & value & status\\\\")
    lines.append("\\midrule")
    lines.extend(search_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}%")
    lines.append("}}")
    tex = "\n".join(lines) + "\n"

    sources = [
        {"path": path, "role": meta["role"], "git_blob": meta["git_blob"], "git_blob_sha256": meta["git_blob_sha256"], "bytes": meta["bytes"]}
        for path, meta in bound.items()
    ]
    evidence = {
        "document_type": "paper-four-cell-closure-evidence",
        "schema_version": "1.0",
        "classification": CLASSIFICATION,
        "correction_status": CORRECTION_STATUS,
        "evidence_revision": ANALYSIS_COMMIT_SHA,
        "binding": {
            "analysis_commit": ANALYSIS_COMMIT_SHA,
            "analysis_commit_subject": _git(repo, "show", "-s", "--format=%s", ANALYSIS_COMMIT_SHA),
            "verified_tree_commit": VERIFIED_TREE_COMMIT_SHA,
            "mdo_preregistration_commit": MDO_PREREGISTRATION_COMMIT_SHA,
            "rule": (
                "every source file is bound by Git blob and SHA-256 at the analysis commit and verified unchanged at the "
                "verified-tree commit; the MDO protocol blob equals the frozen preregistration blob; the executed "
                "cft_revival.plasma package in the checkout must equal the bound blobs (LF-normalised SHA-256) or the "
                "generator refuses to recompute"
            ),
        },
        "sources": sources,
        "executed_package": {
            "files": [{"path": path, "sha256_lf": digest, "git_blob_sha256": bound[path]["git_blob_sha256"]} for path, digest in sorted(executed.items())],
            "import_root": "modern/src",
            "matches_bound_blobs": True,
        },
        "recomputation_protocol": {
            "closed_form_sample": {"seed": SAMPLE_SEED, "count": SAMPLE_COUNT, "voltages_v": list(SAMPLE_VOLTAGES), "currents_a": list(SAMPLE_CURRENTS), "probability_upper": SAMPLE_PROBABILITY_UPPER, "interior_potentials": "three uniform draws on [0.01 Ua, Ua] sorted; anode-cell potential uniform on [Ua, 1.5 Ua]"},
            "continuation": {"voltage_v": CONTINUATION_VOLTAGE_V, "current_a": CONTINUATION_CURRENT_A, "epsilons": list(CONTINUATION_EPSILONS), "patterns": ["eps*(1,1,1,1)", "(0,0,0,eps)"], "start_count_interior": CONTINUATION_START_COUNT, "start_count_anode_only": ANODE_ONLY_START_COUNT, "max_iterations": CONTINUATION_MAX_ITERATIONS, "residual_tolerance": RESIDUAL_TOLERANCE, "solver": "cft_revival.plasma.solve_global_discharge_multistart"},
            "published_state": {"voltage_v": DM92_VOLTAGE_V, "current_a": DM92_CURRENT_A, "probabilities": list(DM92_PROBABILITIES), "potentials_v": list(DM92_POTENTIALS_V)},
            "relaxed_root": {"voltage_v": RELAXED_VOLTAGE_V, "current_a": RELAXED_CURRENT_A, "probabilities": list(DM92_PROBABILITIES), "interior_potentials_v": list(RELAXED_INTERIOR_POTENTIALS_V), "bisections": RELAXED_BISECTIONS},
            "anode_fall_coefficient": {"voltage_v": CONTINUATION_VOLTAGE_V, "current_a": CONTINUATION_CURRENT_A, "probabilities": [0.0, 0.0, 0.0, 0.0], "delta_v": COEFFICIENT_DELTA_V},
            "not_recomputed": ["differential evolution (205 312 evaluations; SciPy)", "200 random feasible LM starts", "the 80-case solver probe (read from the frozen MDO protocol; its reproduction is documented)"],
            "tolerances": TOLERANCES,
            "recorded_significant_digits": ROUNDING,
        },
        "recomputed_summary": {
            "closed_form_relative_difference": round_sig(recomputed["closed_form_relative_difference"], ROUNDING["closed_form_relative_difference"]),
            "manifold_normalized_residual": round_sig(recomputed["manifold_normalized_residual"], ROUNDING["manifold_normalized_residual"]),
            "anode_fall_coefficient": round_sig(recomputed["anode_fall_coefficient"], 12),
            "continuation_floors": [round_sig(step["floor"], ROUNDING["continuation_floor"]) for step in recomputed["ladder"]],
            "continuation_reasons": [step["reason"] for step in recomputed["ladder"]],
            "continuation_dominant_rows": [step["dominant_row"] for step in recomputed["ladder"]],
            "anode_only_residuals": [round_sig(step["anode_only_residual"], ROUNDING["anode_only_residual"]) for step in recomputed["ladder"]],
            "jacobian_ranks": [step["jacobian_rank"] for step in recomputed["ladder"]],
            "jacobian_condition_max": round_sig(condition_max, ROUNDING["jacobian_condition"]),
            "continuation_max_relative_departure": round_sig(departures["continuation_max_relative_departure"], ROUNDING["departure"]),
            "continuation_slope_spread": round_sig(departures["continuation_slope_spread"], ROUNDING["departure"]),
            "dm92_misfit": round_sig(recomputed["dm92_misfit"], ROUNDING["dm92_misfit"]),
            "relaxed_root_depth_v": round_sig(recomputed["relaxed_root_depth_v"], ROUNDING["relaxed_root_depth_v"]),
            "relaxed_root_feasible": recomputed["relaxed_root_feasible"],
        },
        "documented_summary": {
            "continuation_floors": list(documented["continuation_floors"]),
            "continuation_interior_floors": list(documented["continuation_interior_floors"]),
            "dm92_misfit": documented["dm92_misfit"],
            "jacobian_rank": documented["jacobian_rank"],
            "probe": {"closed": doc_probe_closed, "total": doc_probe_total, "source": PROBE_SOURCE, "protocol_closed": probe_closed, "protocol_total": probe_total},
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
            "gate_kind": GATE_KIND,
            "artifact_id": ARTIFACT_ID,
            "artifact_claim_id": ARTIFACT_CLAIM_ID,
            "prose_claim_ids": list(PROSE_CLAIM_IDS),
            "revision_macro": REVISION_MACRO,
            "rule": (
                "Every number in the section is a macro defined here; each macro is bound below to a document, "
                "ledger, protocol or legacy path with a JSON pointer or fixed regular expression and the file's "
                "SHA-256, or to a recomputation whose inputs and protocol are recorded. Claim-bearing sentences are "
                "exact EvidenceClaim bodies registered in paper/evidence/claims.json; the analytic-consistency gate in "
                "paper/evidence/result-gates.json names the typed manifest that admits the section. The result is "
                "about the corrected equation set; it is not a statement about the physical thruster, it accepts no "
                "correction, and it opens no physics level."
            ),
        },
        "artifacts": {
            path: {"revision": ANALYSIS_COMMIT_SHA, "git_blob": meta["git_blob"], "sha256": meta["git_blob_sha256"], "bytes": meta["bytes"]}
            for path, meta in sorted(bound.items())
        },
        "macros": m.items,
        "tables": {
            "FccContinuationTable": {"rows": len(continuation_rows), "source": "analysis document section 3.2 (documented) and the recomputed ladder"},
            "FccGlobalSearchTable": {"rows": len(search_rows), "source": "analysis document sections 2-4 and the ledger (documented), recomputed checks, frozen MDO protocol"},
        },
        "generator": {
            "path": "paper/scripts/generate_four_cell_closure_evidence.py",
            "sha256": sha256_bytes(_lf((repo / "paper/scripts/generate_four_cell_closure_evidence.py").read_bytes())),
            "command": "python paper/scripts/generate_four_cell_closure_evidence.py",
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
        "evidence_revision": ANALYSIS_COMMIT_SHA,
        "source_date_epoch": build_config["source_date_epoch"],
        "generator": evidence["generator"],
        "manifest": {
            "path": EVIDENCE_PATH.as_posix(),
            "sha256": sha256_bytes(canonical_json(evidence)),
            "manifest_id": MANIFEST_ID,
            "gate_manifest_path": MANIFEST_PATH.as_posix(),
        },
        "inputs": [
            {"path": path, "revision": meta["revision"], "git_blob": meta["git_blob"], "sha256": meta["sha256"], "bytes": meta["bytes"]}
            for path, meta in evidence["artifacts"].items()
        ],
        "executed_package": evidence["executed_package"],
        "recomputation_protocol": evidence["recomputation_protocol"],
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
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ImportError) as exc:
        print(f"four-cell closure evidence generation failed: {exc}")
        return 1
    print(f"Generated {EVIDENCE_PATH}, {OUTPUT_PATH} and {SIDECAR_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
