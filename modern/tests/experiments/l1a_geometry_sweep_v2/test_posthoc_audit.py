"""Integrity tests for the sweep-v2 posthoc audit overlay (protocol sidecar EOL).

Everything asserted here is re-derived from the frozen ``protocol.json``, the
immutable ``results/`` bundle or Git; the overlay must never change a byte of
either.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.l1a_geometry_sweep_v2 import audit_sidecar_eol as audit_module
from experiments.l1a_geometry_sweep_v2.audit_sidecar_eol import (
    EXPECTED_EOL_ONLY_PATHS,
    PROTOCOL_LF_SHA256,
    PROTOCOL_PAYLOAD_SHA256,
    PROTOCOL_RECORDED_SHA256,
    audit,
    format_table,
)
from experiments.l1a_geometry_sweep_v2.protocol import (
    EOL_AUDITED_SIDECARS,
    PROTOCOL_PATH,
    eol_equivalent_digest,
    load_protocol,
    verify_sidecar,
)

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT = REPO / "modern/experiments/l1a_geometry_sweep_v2"
RESULTS = EXPERIMENT / "results"
AUDIT_MD = EXPERIMENT / "POSTHOC_AUDIT.md"
GENERATOR_PATH = EXPERIMENT / "visualization" / "generate_dashboard.py"
EXPERIMENT_REL = "modern/experiments/l1a_geometry_sweep_v2"
RESULTS_REL = f"{EXPERIMENT_REL}/results"

PREREGISTRATION_COMMIT = "092f5fae692ee7d6711e0c7e1c94dac6a345f37c"
RESULT_COMMIT = "f30cb42ec4a8633bf634a3d32ffa5b11f66be97a"
RESULTS_TREE = "de85a158a01aa4113154ef256c9d11032bdf6538"
PROTOCOL_BLOB = "37d455a952306d9a6fe36456a1c0a3c6fd4c747a"
SIDECAR_BLOB = "270da0c4c0939ed727b0ceb1c4ad9cc9cfb762c1"
MANIFEST_SHA256 = "768b345e946a45e623f83aaa18e01f8ec5bc7f823e81858a0a8c3a3e2e448754"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _load_generator():
    spec = importlib.util.spec_from_file_location("l1a_sweep_v2_dashboard_audit", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report() -> dict:
    before = _tree_digest(EXPERIMENT)
    value = audit()
    assert _tree_digest(EXPERIMENT) == before, "audit must be read-only on the experiment"
    return value


# --------------------------------------------------------------------------
# the finding
# --------------------------------------------------------------------------


def test_audit_passes_with_exactly_the_protocol_as_eol_only(report: dict) -> None:
    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["counts"] == {"byte_exact": 33, "eol_only": 1, "mismatch": 0}
    assert report["file_entries"] == 34
    assert report["mismatch"] == []
    assert tuple(row["path"] for row in report["eol_only"]) == EXPECTED_EOL_ONLY_PATHS == (
        "protocol.json",
    )
    assert report["eol_only_paths_are_exactly_expected"] is True
    assert report["results"]["files_containing_cr"] == []
    assert report["results"]["manifest_sha256"] == MANIFEST_SHA256
    assert report["results"]["manifest_terminal_status"] == "ACCEPTED"
    assert report["results"]["preregistration_commit_sha"] == PREREGISTRATION_COMMIT
    protocol = report["protocol"]
    data = PROTOCOL_PATH.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert protocol["checkout_bytes"] == len(data) == 7790
    assert protocol["lf_count"] == data.count(b"\n") == 134
    assert protocol["crlf_bytes"] == 7924 == protocol["checkout_bytes"] + protocol["lf_count"]
    assert protocol["checkout_sha256"] == hashlib.sha256(data).hexdigest() == PROTOCOL_LF_SHA256
    assert (
        protocol["crlf_recomputed_sha256"]
        == hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest()
        == PROTOCOL_RECORDED_SHA256
    )
    assert protocol["recorded_sha256"] == PROTOCOL_RECORDED_SHA256
    assert protocol["checkout_sha256"] != protocol["recorded_sha256"]
    sidecar = (EXPERIMENT / "protocol.json.sha256").read_bytes()
    assert sidecar == f"{PROTOCOL_RECORDED_SHA256}  protocol.json\n".encode("ascii")


def test_protocol_content_is_untouched_and_bound_by_every_bundle_file(report: dict) -> None:
    protocol = report["protocol"]
    assert protocol["payload_recomputes"] is True
    assert protocol["recomputed_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
    assert protocol["recorded_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
    assert protocol["schema_version"] == "cft-revival.experiment.l1a-geometry-sweep-v2.protocol/1.0.0"
    bindings = report["results"]["protocol_bindings"]
    assert set(bindings) == {"manifest.json", "raw-results.json", "summary.json", "execution-lock.json"}
    for name, bound in bindings.items():
        assert bound["protocol_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256, name
        if name != "summary.json":
            assert bound["protocol_file_sha256"] == PROTOCOL_RECORDED_SHA256, name
    assert report["bundle_binds_recorded_protocol_file_digest"] is True
    assert report["bundle_binds_protocol_payload_digest"] is True
    loaded = load_protocol()
    assert loaded["integrity"]["payload_sha256"] == PROTOCOL_PAYLOAD_SHA256


def test_every_results_file_is_byte_exact(report: dict) -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_bytes())
    assert len(manifest["deterministic_files"]) == 16
    for entry in manifest["deterministic_files"]:
        path = RESULTS / entry["path"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["file_sha256"], entry["path"]
        assert b"\r" not in data
        assert verify_sidecar(path) == entry["file_sha256"]
    sidecars = sorted(RESULTS.rglob("*.sha256"))
    assert len(sidecars) == 17
    for sidecar in sidecars:
        target = sidecar.with_name(sidecar.name[: -len(".sha256")])
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        assert sidecar.read_bytes() == f"{digest}  {target.name}\n".encode("ascii")
    tracked = _git("ls-files", "--", RESULTS_REL).splitlines()
    assert len(tracked) == 35  # 34 result files + the preregistered README.md


# --------------------------------------------------------------------------
# the tolerance is bound to exactly the audited file
# --------------------------------------------------------------------------


def test_verify_sidecar_returns_recorded_digest_only_through_the_audited_rule() -> None:
    assert set(EOL_AUDITED_SIDECARS) == {PROTOCOL_PATH}
    audited = EOL_AUDITED_SIDECARS[PROTOCOL_PATH]
    assert audited.lf_sha256 == PROTOCOL_LF_SHA256
    assert audited.recorded_sha256 == PROTOCOL_RECORDED_SHA256
    data = PROTOCOL_PATH.read_bytes()
    assert eol_equivalent_digest(PROTOCOL_PATH, data) == PROTOCOL_RECORDED_SHA256
    assert verify_sidecar(PROTOCOL_PATH) == PROTOCOL_RECORDED_SHA256
    # the rule is refused for any other byte change of the audited file
    assert eol_equivalent_digest(PROTOCOL_PATH, data + b"\n") is None
    assert eol_equivalent_digest(PROTOCOL_PATH, data.replace(b"\n", b"\r\n")) is None
    assert eol_equivalent_digest(PROTOCOL_PATH, data.replace(b"96", b"97", 1)) is None
    # and for any other path with identical bytes
    assert eol_equivalent_digest(RESULTS / "protocol.json", data) is None
    assert eol_equivalent_digest(EXPERIMENT / "other.json", data) is None


def test_tolerance_does_not_apply_outside_the_audited_path(tmp_path: Path) -> None:
    copied = tmp_path / "protocol.json"
    copied.write_bytes(PROTOCOL_PATH.read_bytes())
    copied.with_name("protocol.json.sha256").write_bytes(
        (EXPERIMENT / "protocol.json.sha256").read_bytes()
    )
    with pytest.raises(ValueError, match="invalid SHA-256 sidecar"):
        verify_sidecar(copied)


def test_crlf_restoration_in_scratch_copy_verifies_through_the_ordinary_path(
    tmp_path: Path,
) -> None:
    """The ONLY difference is EOL: restore CRLF and the byte-exact rule passes."""

    copied = tmp_path / "protocol.json"
    copied.write_bytes(PROTOCOL_PATH.read_bytes().replace(b"\n", b"\r\n"))
    copied.with_name("protocol.json.sha256").write_bytes(
        (EXPERIMENT / "protocol.json.sha256").read_bytes()
    )
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == PROTOCOL_RECORDED_SHA256
    assert verify_sidecar(copied) == PROTOCOL_RECORDED_SHA256


def test_tampered_protocol_content_is_rejected_even_with_matching_lf_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the audited path at a scratch file: same rule, altered bytes, refused."""

    scratch = tmp_path / "protocol.json"
    tampered = PROTOCOL_PATH.read_bytes().replace(b'"case_count": 96', b'"case_count": 97', 1)
    assert tampered != PROTOCOL_PATH.read_bytes()
    scratch.write_bytes(tampered)
    scratch.with_name("protocol.json.sha256").write_bytes(
        (EXPERIMENT / "protocol.json.sha256").read_bytes()
    )
    from experiments.l1a_geometry_sweep_v2 import protocol as protocol_module

    monkeypatch.setattr(
        protocol_module,
        "EOL_AUDITED_SIDECARS",
        {scratch.resolve(): EOL_AUDITED_SIDECARS[PROTOCOL_PATH]},
    )
    with pytest.raises(ValueError, match="invalid SHA-256 sidecar"):
        protocol_module.verify_sidecar(scratch)


def test_dashboard_generator_applies_the_same_bound_rule(tmp_path: Path) -> None:
    generator = _load_generator()
    assert generator.EXPECTED_PROTOCOL_FILE_SHA256 == PROTOCOL_RECORDED_SHA256
    assert generator.AUDITED_PROTOCOL_LF_SHA256 == PROTOCOL_LF_SHA256
    assert (
        generator._verify_file(PROTOCOL_PATH, "protocol", generator.EXPECTED_PROTOCOL_FILE_SHA256)
        == PROTOCOL_RECORDED_SHA256
    )
    # the LF digest is not what the frozen sidecar attests (the bundle binds CRLF)
    with pytest.raises(ValueError, match="sidecar is invalid"):
        generator._verify_file(PROTOCOL_PATH, "protocol", PROTOCOL_LF_SHA256)
    # a copy elsewhere is not the audited file
    copied = tmp_path / "protocol.json"
    copied.write_bytes(PROTOCOL_PATH.read_bytes())
    copied.with_name("protocol.json.sha256").write_bytes(
        (EXPERIMENT / "protocol.json.sha256").read_bytes()
    )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        generator._verify_file(copied, "protocol", generator.EXPECTED_PROTOCOL_FILE_SHA256)
    # bundle files are byte-exact and take the ordinary path
    manifest_digest = generator._verify_file(
        RESULTS / "manifest.json", "manifest", generator.EXPECTED_MANIFEST_FILE_SHA256
    )
    assert manifest_digest == MANIFEST_SHA256


# --------------------------------------------------------------------------
# document and immutability
# --------------------------------------------------------------------------


def test_audit_document_table_matches_live_recomputation(report: dict) -> None:
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "\r" not in text
    for line in format_table(report).splitlines():
        assert line in text, line
    for needle in (
        "ACCEPTED",
        "L1a_field_only_design_space_screening",
        PREREGISTRATION_COMMIT,
        RESULT_COMMIT,
        RESULTS_TREE,
        PROTOCOL_BLOB,
        SIDECAR_BLOB,
        MANIFEST_SHA256,
        PROTOCOL_LF_SHA256,
        PROTOCOL_RECORDED_SHA256,
        PROTOCOL_PAYLOAD_SHA256,
        "EOL_AUDITED_SIDECARS",
        "AUDITED_PROTOCOL_LF_SHA256",
        "l1a_field_surrogate_v1",
        "l1a_field_surrogate_v2",
        "Results/",
        "audit_sidecar_eol",
        "byte_exact: 33, eol_only: 1, mismatch: 0",
    ):
        assert needle in text, needle


def test_frozen_inputs_and_results_tree_are_unchanged_since_their_commits() -> None:
    protocol_rel = f"{EXPERIMENT_REL}/protocol.json"
    assert _git("rev-parse", f"{PREREGISTRATION_COMMIT}:{protocol_rel}") == PROTOCOL_BLOB
    assert _git("rev-parse", f"HEAD:{protocol_rel}") == PROTOCOL_BLOB
    assert _git("rev-parse", f"{PREREGISTRATION_COMMIT}:{protocol_rel}.sha256") == SIDECAR_BLOB
    assert _git("rev-parse", f"HEAD:{protocol_rel}.sha256") == SIDECAR_BLOB
    assert _git("rev-parse", f"{RESULT_COMMIT}:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-parse", f"HEAD:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-list", "--parents", "-n", "1", RESULT_COMMIT) == (
        f"{RESULT_COMMIT} {PREREGISTRATION_COMMIT}"
    )
    assert _git("status", "--porcelain", "--", RESULTS_REL, protocol_rel, f"{protocol_rel}.sha256") == ""
    eol = _git("ls-files", "--eol", "--", EXPERIMENT_REL)
    assert "w/crlf" not in eol and "w/mixed" not in eol
    # Git stores LF: the blob hashes to the LF digest, never to the recorded one.
    blob = subprocess.run(
        ["git", "cat-file", "blob", PROTOCOL_BLOB], cwd=REPO, check=True, capture_output=True
    ).stdout
    assert hashlib.sha256(blob).hexdigest() == PROTOCOL_LF_SHA256


def test_script_refuses_to_write_inside_the_experiment(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        audit_module.main(["--json", str(RESULTS / "posthoc.json")])
    assert "must not point inside the experiment directory" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        audit_module.main(["--json", str(EXPERIMENT / "posthoc.json")])
    capsys.readouterr()
    target = tmp_path / "report.json"
    assert audit_module.main(["--json", str(target)]) == 0
    written = json.loads(target.read_bytes())
    assert written["passed"] is True and written["counts"]["eol_only"] == 1
    assert not (RESULTS / "posthoc.json").exists()
    assert not (EXPERIMENT / "posthoc.json").exists()
    out = capsys.readouterr().out
    assert "protocol.json" in out


def test_table_flag_prints_only_the_markdown_table(capsys) -> None:
    assert audit_module.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| path | checkout bytes |")
    assert PROTOCOL_LF_SHA256 in out and PROTOCOL_RECORDED_SHA256 in out
    assert out.count("\n") == 3
