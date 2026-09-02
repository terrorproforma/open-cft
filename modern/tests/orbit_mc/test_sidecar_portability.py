"""v1.7 byte-portability of hash sidecars and a fail-closed newline lint.

Background: ``write_artifact`` wrote ``<name>.json.sha256`` through the text
layer without ``newline="\\n"``; on Windows the sidecar was CRLF while Git
(root ``.gitattributes`` ``eol=lf``) stores LF. The v4 wall-loss bundle
therefore recorded CRLF byte hashes for nine sidecars that a fresh checkout
cannot reproduce. These tests pin the LF contract and grep the packages that
write hash-bound text for the same defect.
"""

from __future__ import annotations

import ast
import hashlib
from math import pi
from pathlib import Path

import numpy as np
import pytest

import cft_revival.orbit_mc as orbit_mc
from cft_revival.orbit_mc import (
    AnalyticField,
    EstimatorPolicy,
    OrbitConfig,
    build_launch_ensemble,
    frozen_batch_manifest,
    load_and_verify_artifact,
    load_artifact,
    result_artifact,
    run_ensemble,
    write_artifact,
)

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "cft_revival"
POLICY_SHA256 = "d" * 64
CERTIFICATE_FLOOR = 0.001
ESTIMATOR = EstimatorPolicy.UNWEIGHTED_BINOMIAL

# Packages whose text writes feed byte-hashed evidence (manifests, sidecars,
# canonical JSON). Every text-mode write in them must pin newline="\n".
LINTED_PACKAGES = (
    "orbit_mc",
    "experiment_runtime",
    "fem_reference",
    "coupling",
    "fields",
)

# Allowlist for ``.open(...)`` calls the fail-closed lint cannot classify from
# a string-literal mode but which are provably not text writes. Keyed by
# (relative posix path, stripped source line); every entry names why it is
# binary or not a file stream. Text writes are never allowlisted: fix them.
NEWLINE_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # zipfile.ZipFile.open(member) yields a binary member stream (read).
        ("fem_reference/artifacts.py", "with archive.open(member) as source:"),
        # PinnedDirectory.open is a classmethod returning a directory handle,
        # not a file stream; there is no text layer.
        ("experiment_runtime/filesystem.py", "return PinnedDirectory.open(path)"),
        ("experiment_runtime/platformfs.py", "other = PinnedDirectory.open(self.path)"),
    }
)


def _campaign():
    field = AnalyticField(lambda _x: np.array([0.0, 0.0, 0.1]), None, 0.1)
    launches = build_launch_ensemble(
        ensemble_id="portability",
        energies_ev=(10.0, 30.0),
        pitch_angles_rad=(0.2, 0.8),
        positions=(("core", (0.0, 0.0, 0.0)),),
        directions=(-1, 1),
        gyrophase_count=4,
    )
    gyroperiod = 2 * pi * 9.1093837139e-31 / (1.602176634e-19 * 0.1)
    config = OrbitConfig(
        0.1, -0.2, 0.2, 0.2, -0.3, 0.3, 1.5 * gyroperiod, 1.0,
        max_steps=1000, max_rotation_rad=0.12,
    )
    return launches, field, config


def _write_verified_artifact(path: Path):
    launches, field, config = _campaign()
    results, summary = run_ensemble("portability", launches, field, config)
    batches = frozen_batch_manifest(launches, batch_size=16)
    artifact = result_artifact(
        campaign_id="portability",
        field_identity_sha256="a" * 64,
        config_identity_sha256="b" * 64,
        policy_identity_sha256=POLICY_SHA256,
        minimum_certificate_tightness_ratio_authority=CERTIFICATE_FLOOR,
        estimator_policy=ESTIMATOR,
        launches=launches,
        results=results,
        batch_manifest=batches,
        summary=summary,
        interpolation_evidence={
            "certified_max_b_t": 0.1,
            "reference_max_b_t": None,
            "runtime_max_seen_t": 0.1,
            "dense_diagnostic_max_b_t": 0.1,
            "certificate_tightness_ratio": 1.0,
            "minimum_certificate_tightness_ratio": CERTIFICATE_FLOOR,
            "certificate_preflight_passed": True,
            "material_map_sha256": "c" * 64,
            "field_error_report": {
                "sample_count": 1,
                "psi_node_max_abs_wb": 0.0,
                "br_max_abs_t": 0.0,
                "bz_max_abs_t": 0.0,
                "b_rms_t": 0.0,
                "b_relative_rms": 0.0,
            },
            "passed": True,
        },
        convergence_evidence={
            "timestep_passed": True,
            "cross_map_passed": True,
            "backend_parity_passed": True,
        },
        preregistration={
            "protocol_id": "test-protocol",
            "frozen_before_outcomes": True,
            "held_out_geometry_status": "pending",
        },
    )
    replay = {
        "field": field,
        "config": config,
        "expected_field_sha256": "a" * 64,
        "expected_config_sha256": "b" * 64,
        "expected_launches_sha256": artifact["identities"]["launches_sha256"],
        "expected_batch_manifest_sha256": artifact["identities"]["batch_manifest_sha256"],
        "expected_policy_sha256": POLICY_SHA256,
        "expected_estimator_policy": ESTIMATOR,
        "expected_minimum_certificate_tightness_ratio": CERTIFICATE_FLOOR,
    }
    verified = write_artifact(path, artifact, **replay)
    return verified, replay


def test_package_version_is_1_7_0_with_unchanged_on_disk_contract() -> None:
    # The fix changes sidecar EOL bytes only; the artifact JSON bytes and the
    # sidecar text content are identical to v1.6, so schemas do not move.
    assert orbit_mc.__version__ == "1.7.0"
    assert orbit_mc.SCHEMA_VERSION == "cft-revival-orbit-mc-result/1.6.0"
    assert orbit_mc.CHECKPOINT_VERSION == "cft-revival-orbit-mc-checkpoint/1.6.0"
    assert orbit_mc.HANDOFF_VERSION == "cft-revival-orbit-mc-coupling-v4.2/1.3.0"


def test_sidecar_bytes_contain_no_carriage_return(tmp_path: Path) -> None:
    path = tmp_path / "portability-orbit.json"
    verified, _ = _write_verified_artifact(path)
    sidecar = path.with_name(path.name + ".sha256").read_bytes()
    assert b"\r" not in sidecar
    assert sidecar == f"{verified.file_sha256}  {path.name}\n".encode("ascii")
    assert sidecar.endswith(b"\n") and sidecar.count(b"\n") == 1
    # 64 hex + two spaces + name + one LF: the exact byte length Git stores.
    assert len(sidecar) == 64 + 2 + len(path.name) + 1
    # The artifact itself is canonical compact JSON: no EOL bytes at all.
    assert b"\r" not in path.read_bytes() and b"\n" not in path.read_bytes()


def test_artifact_and_sidecar_revalidate_byte_exactly_after_lf_normalisation(
    tmp_path: Path,
) -> None:
    """Simulate Git's ``text=auto eol=lf`` normalisation (CRLF -> LF) on commit."""

    path = tmp_path / "portability-orbit.json"
    verified, replay = _write_verified_artifact(path)
    sidecar_path = path.with_name(path.name + ".sha256")
    written_artifact = path.read_bytes()
    written_sidecar = sidecar_path.read_bytes()
    recorded = {
        "artifact": hashlib.sha256(written_artifact).hexdigest(),
        "sidecar": hashlib.sha256(written_sidecar).hexdigest(),
    }
    # What the index would store after normalisation.
    normalised_artifact = written_artifact.replace(b"\r\n", b"\n")
    normalised_sidecar = written_sidecar.replace(b"\r\n", b"\n")
    assert normalised_artifact == written_artifact
    assert normalised_sidecar == written_sidecar
    assert hashlib.sha256(normalised_sidecar).hexdigest() == recorded["sidecar"]
    assert hashlib.sha256(normalised_artifact).hexdigest() == recorded["artifact"]
    assert recorded["artifact"] == verified.file_sha256
    # Rewrite from the normalised bytes (a fresh LF checkout) and re-validate.
    path.write_bytes(normalised_artifact)
    sidecar_path.write_bytes(normalised_sidecar)
    reloaded = load_artifact(
        path,
        expected_file_sha256=verified.file_sha256,
        expected_policy_sha256=POLICY_SHA256,
        expected_estimator_policy=ESTIMATOR,
        expected_minimum_certificate_tightness_ratio=CERTIFICATE_FLOOR,
    )
    assert reloaded.file_sha256 == verified.file_sha256
    replayed = load_and_verify_artifact(
        path, expected_file_sha256=verified.file_sha256, **replay
    )
    assert replayed.file_sha256 == verified.file_sha256
    # The pre-fix defect for the record: a CRLF sidecar has a different byte
    # hash and is one byte longer per line, which is exactly what the v4
    # bundle recorded on Windows.
    crlf_sidecar = normalised_sidecar.replace(b"\n", b"\r\n")
    assert len(crlf_sidecar) == len(normalised_sidecar) + 1
    assert hashlib.sha256(crlf_sidecar).hexdigest() != recorded["sidecar"]


# --------------------------------------------------------------------------
# Fail-closed newline lint over the hash-bound packages
# --------------------------------------------------------------------------

_TEXT_OPENERS = {"open", "fdopen", "TextIOWrapper"}
_TEMPFILE_OPENERS = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}


def _call_name(node: ast.Call) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _mode_literal(node: ast.Call, positional_index: int) -> str | None | object:
    """Return the mode string, ``None`` when defaulted, or a sentinel if dynamic."""

    value = _keyword(node, "mode")
    if value is None and len(node.args) > positional_index:
        value = node.args[positional_index]
    if value is None:
        return None
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return _DYNAMIC


_DYNAMIC = object()


def _is_text_write_mode(mode: str) -> bool:
    return "b" not in mode and any(flag in mode for flag in "wax+")


def _text_write_violations(path: Path) -> list[tuple[int, str, str]]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    violations: list[tuple[int, str, str]] = []

    def report(node: ast.AST, reason: str) -> None:
        line = lines[node.lineno - 1].strip()
        violations.append((node.lineno, line, reason))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "write_text":
            if _keyword(node, "newline") is None:
                report(node, "write_text without newline=")
            continue
        if name in _TEXT_OPENERS:
            if name == "TextIOWrapper":
                if _keyword(node, "newline") is None:
                    report(node, "TextIOWrapper without newline=")
                continue
            receiver = (
                node.func.value.id
                if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                else None
            )
            if name == "open" and receiver == "os":
                continue  # os.open returns a raw descriptor: never a text layer
            # builtins.open / io.open / os.fdopen take the mode as the second
            # positional argument; <object>.open(...) (Path.open) as the first.
            module_level = isinstance(node.func, ast.Name) or receiver in {
                "io", "builtins", "codecs", "os"
            }
            positional_index = 1 if module_level else 0
            mode = _mode_literal(node, positional_index)
            if mode is None:
                continue  # default mode "r": a read
            if mode is _DYNAMIC:
                report(node, "dynamic open() mode; declare a literal mode and newline=")
                continue
            if _is_text_write_mode(mode) and _keyword(node, "newline") is None:
                report(node, f"text write mode {mode!r} without newline=")
            continue
        if name in _TEMPFILE_OPENERS:
            mode = _mode_literal(node, 0)
            if mode is None:
                continue  # default "w+b": binary
            if mode is _DYNAMIC:
                report(node, "dynamic tempfile mode; declare a literal mode and newline=")
                continue
            if _is_text_write_mode(mode) and _keyword(node, "newline") is None:
                report(node, f"text tempfile mode {mode!r} without newline=")
    return violations


def _linted_sources() -> list[Path]:
    files: list[Path] = []
    for package in LINTED_PACKAGES:
        root = SRC_ROOT / package
        assert root.is_dir(), root
        files.extend(sorted(root.rglob("*.py")))
    return files


def test_lint_scope_is_nonempty_and_allowlist_entries_exist() -> None:
    files = _linted_sources()
    assert len(files) >= 30
    relative = {path.relative_to(SRC_ROOT).as_posix() for path in files}
    for entry_path, _line in NEWLINE_ALLOWLIST:
        assert entry_path in relative, f"stale allowlist path {entry_path}"


def test_every_text_mode_write_in_hash_bound_packages_pins_newline() -> None:
    offenders: list[str] = []
    for path in _linted_sources():
        relative = path.relative_to(SRC_ROOT).as_posix()
        for lineno, line, reason in _text_write_violations(path):
            if (relative, line) in NEWLINE_ALLOWLIST:
                continue
            offenders.append(f"{relative}:{lineno}: {reason}: {line}")
    assert offenders == [], "\n".join(offenders)


@pytest.mark.parametrize(
    ("snippet", "expected_count"),
    [
        ('p.write_text("x", encoding="ascii")', 1),
        ('p.write_text("x", encoding="ascii", newline="\\n")', 0),
        ('open(p, "w", encoding="utf-8")', 1),
        ('open(p, "w", encoding="utf-8", newline="\\n")', 0),
        ('open(p, mode="a")', 1),
        ('open(p, "rb")', 0),
        ('open(p, "wb")', 0),
        ('open(p)', 0),
        ('p.open("w")', 1),
        ('p.open(encoding="utf-8")', 0),
        ('p.open("rb")', 0),
        ('os.fdopen(fd, "w", encoding="ascii")', 1),
        ('os.fdopen(fd, "w", encoding="ascii", newline="\\n")', 0),
        ('open(p, mode)', 1),
        ('io.open(p, "w", encoding="utf-8")', 1),
        ('io.open(p, "w", encoding="utf-8", newline="\\n")', 0),
        ('os.open(p, 0)', 0),
        ('handle.open(member)', 1),
        ('io.TextIOWrapper(raw, encoding="utf-8")', 1),
        ('tempfile.NamedTemporaryFile()', 0),
        ('tempfile.NamedTemporaryFile("w")', 1),
        ('tempfile.NamedTemporaryFile("w", newline="\\n")', 0),
    ],
)
def test_newline_lint_detects_text_writes(
    tmp_path: Path, snippet: str, expected_count: int
) -> None:
    module = tmp_path / "sample.py"
    preamble = "import io, os, tempfile\np = None\nfd = 0\nraw = None\nmode = 'w'\n"
    module.write_bytes(f"{preamble}{snippet}\n".encode("utf-8"))
    assert len(_text_write_violations(module)) == expected_count, snippet
