"""Integrity tests for the four-cell v2 posthoc audit overlay (protocol copy EOL).

Everything asserted here is re-derived from the frozen ``protocol.json``, the
immutable ``results/`` bundle or Git; the overlay must never change a byte of
either.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.four_cell_topology_search_v2 import audit_sidecar_eol as audit_module
from experiments.four_cell_topology_search_v2.audit_sidecar_eol import (
    EXPECTED_EOL_ONLY_PATHS,
    PROTOCOL_COPY_PATH,
    PROTOCOL_LF_BYTES,
    PROTOCOL_LF_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_PAYLOAD_SHA256,
    PROTOCOL_RECORDED_BYTES,
    PROTOCOL_RECORDED_SHA256,
    audit,
    eol_equivalent_digest,
    format_table,
)

REPO = Path(__file__).resolve().parents[4]
EXPERIMENT = REPO / "modern/experiments/four_cell_topology_search_v2"
RESULTS = EXPERIMENT / "results"
AUDIT_MD = EXPERIMENT / "POSTHOC_AUDIT.md"
EXPERIMENT_REL = "modern/experiments/four_cell_topology_search_v2"
RESULTS_REL = f"{EXPERIMENT_REL}/results"

PREREGISTRATION_COMMIT = "d6317910703de91ca6dc25c4d4d855e36cc3b14d"
RESULT_COMMIT = "7120e8edcb74c02c1df968c730d1f93b3758b4e1"
RESULTS_TREE = "56b41d451d94e0fde1f86bd4d3a40b7fbc2470b2"
PROTOCOL_BLOB = "9fda58faa70e49da8e17a94a478329fd6d408f3c"
SIDECAR_BLOB = "dc73a9d384b7ad17bd39f4b1def43e189b4e8529"
MANIFEST_SHA256 = "f5e26373d72bd13aa5631516009797567e0f66812941d993d5fad534be07240a"


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


@pytest.fixture(scope="module")
def report() -> dict:
    before = _tree_digest(EXPERIMENT)
    value = audit()
    assert _tree_digest(EXPERIMENT) == before, "audit must be read-only on the experiment"
    return value


# --------------------------------------------------------------------------
# the finding
# --------------------------------------------------------------------------


def test_audit_passes_with_exactly_the_protocol_copy_as_eol_only(report: dict) -> None:
    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["counts"] == {"byte_exact": 25, "eol_only": 2, "mismatch": 0}
    assert report["file_entries"] == 27
    assert report["mismatch"] == []
    assert {row["path"] for row in report["eol_only"]} == set(EXPECTED_EOL_ONLY_PATHS) == {
        "preregistered-protocol.json",
    }
    assert report["eol_only_paths_are_exactly_expected"] is True
    assert report["results"]["files_containing_cr"] == []
    assert report["results"]["manifest_sha256"] == MANIFEST_SHA256
    assert report["results"]["manifest_payload_recomputes"] is True
    assert report["results"]["manifest_single_execution"] is True
    assert report["results"]["preregistration_commit_sha"] == PREREGISTRATION_COMMIT
    protocol = report["protocol"]
    data = PROTOCOL_COPY_PATH.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert data == PROTOCOL_PATH.read_bytes()
    assert protocol["copy_equals_frozen_protocol"] is True
    assert protocol["checkout_bytes"] == len(data) == PROTOCOL_LF_BYTES == 10580
    assert protocol["lf_count"] == data.count(b"\n") == 231
    assert protocol["crlf_bytes"] == PROTOCOL_RECORDED_BYTES == 10811
    assert protocol["crlf_bytes"] == protocol["checkout_bytes"] + protocol["lf_count"]
    assert protocol["checkout_sha256"] == hashlib.sha256(data).hexdigest() == PROTOCOL_LF_SHA256
    assert (
        protocol["crlf_recomputed_sha256"]
        == hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest()
        == PROTOCOL_RECORDED_SHA256
    )
    assert protocol["recorded_sha256"] == PROTOCOL_RECORDED_SHA256
    assert protocol["checkout_sha256"] != protocol["recorded_sha256"]
    sidecar = (RESULTS / "preregistered-protocol.json.sha256").read_bytes()
    assert sidecar == f"{PROTOCOL_RECORDED_SHA256}  preregistered-protocol.json\n".encode("ascii")


def test_protocol_content_is_untouched_and_bound_by_the_bundle(report: dict) -> None:
    protocol = report["protocol"]
    assert protocol["payload_recomputes"] is True
    assert protocol["recomputed_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
    assert protocol["schema_version"] == "cft-revival.four-cell-topology-search-v2.protocol/1.0.0"
    assert protocol["protocol_status"] == "preregistered_pending_single_execution"
    assert protocol["candidate_count"] == 128
    bindings = report["results"]["protocol_bindings"]
    assert set(bindings) == {"manifest.json", "dataset.json", "execution-lock.json"}
    for name, bound in bindings.items():
        assert bound["protocol_sha256"] == PROTOCOL_RECORDED_SHA256, name
    assert bindings["dataset.json"]["protocol_payload_sha256"] == PROTOCOL_PAYLOAD_SHA256
    assert report["bundle_binds_recorded_protocol_file_digest"] is True
    assert report["bundle_binds_protocol_payload_digest"] is True
    manifest = json.loads((RESULTS / "manifest.json").read_bytes())
    entry = next(item for item in manifest["artifacts"] if item["path"] == "preregistered-protocol.json")
    assert entry["sha256"] == PROTOCOL_RECORDED_SHA256
    assert entry["bytes"] == PROTOCOL_RECORDED_BYTES
    # the manifest is sealed over the recorded digest, so its payload recomputes as is
    payload = {k: v for k, v in manifest.items() if k != "integrity"}
    assert audit_module.canonical_sha256(payload) == manifest["integrity"]["payload_sha256"]


def test_every_other_results_file_is_byte_exact(report: dict) -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_bytes())
    assert len(manifest["artifacts"]) == 13
    for entry in manifest["artifacts"]:
        path = RESULTS / entry["path"]
        data = path.read_bytes()
        assert b"\r" not in data
        if entry["path"] == "preregistered-protocol.json":
            continue
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
        assert len(data) == entry["bytes"], entry["path"]
    sidecars = sorted(RESULTS.rglob("*.sha256"))
    assert len(sidecars) == 14
    for sidecar in sidecars:
        target = sidecar.with_name(sidecar.name[: -len(".sha256")])
        if target.name == "preregistered-protocol.json":
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        assert sidecar.read_bytes() == f"{digest}  {target.name}\n".encode("ascii")
    tracked = _git("ls-files", "--", RESULTS_REL).splitlines()
    assert len(tracked) == 28
    summary = report["results"]["summary"]
    assert summary["evaluated_count"] == 128
    assert summary["stable_count"] == 0
    assert summary["failure_counts"]["TOPOLOGY_COUNT"] == 128
    assert summary["failure_counts"]["TOPOLOGY_UNSTABLE"] == 128
    assert summary["gpu_replay_pass_count"] == 2 and summary["gpu_replay_required_count"] == 4


# --------------------------------------------------------------------------
# the tolerance is bound to exactly the audited file
# --------------------------------------------------------------------------


def test_eol_rule_returns_recorded_digest_only_for_the_audited_copy(tmp_path: Path) -> None:
    data = PROTOCOL_COPY_PATH.read_bytes()
    assert eol_equivalent_digest(PROTOCOL_COPY_PATH, data) == PROTOCOL_RECORDED_SHA256
    # the rule is refused for any other byte change of the audited file
    assert eol_equivalent_digest(PROTOCOL_COPY_PATH, data + b"\n") is None
    assert eol_equivalent_digest(PROTOCOL_COPY_PATH, data.replace(b"\n", b"\r\n")) is None
    assert eol_equivalent_digest(PROTOCOL_COPY_PATH, data.replace(b"128", b"127", 1)) is None
    # and for any other path with identical bytes, including the frozen original
    assert eol_equivalent_digest(PROTOCOL_PATH, data) is None
    assert eol_equivalent_digest(RESULTS / "dataset.json", data) is None
    copied = tmp_path / "preregistered-protocol.json"
    copied.write_bytes(data)
    assert eol_equivalent_digest(copied, data) is None


def test_crlf_restoration_in_scratch_copy_hashes_to_the_recorded_digest(tmp_path: Path) -> None:
    """The ONLY difference is EOL: restore CRLF and the recorded digest and length reappear."""

    copied = tmp_path / "preregistered-protocol.json"
    copied.write_bytes(PROTOCOL_COPY_PATH.read_bytes().replace(b"\n", b"\r\n"))
    restored = copied.read_bytes()
    assert hashlib.sha256(restored).hexdigest() == PROTOCOL_RECORDED_SHA256
    assert len(restored) == PROTOCOL_RECORDED_BYTES
    row = audit_module.classify_file("preregistered-protocol.json", restored, PROTOCOL_RECORDED_SHA256)
    assert row["status"] == "byte_exact"


def test_classify_file_reports_mismatch_for_content_changes() -> None:
    data = PROTOCOL_COPY_PATH.read_bytes()
    tampered = data.replace(b'"candidate_count": 128', b'"candidate_count": 127', 1)
    assert tampered != data
    assert audit_module.classify_file("x", tampered, PROTOCOL_RECORDED_SHA256)["status"] == "mismatch"
    assert audit_module.classify_file("x", data, PROTOCOL_RECORDED_SHA256)["status"] == "eol_only"
    assert audit_module.classify_file("x", data, PROTOCOL_LF_SHA256)["status"] == "byte_exact"


# --------------------------------------------------------------------------
# document and immutability
# --------------------------------------------------------------------------


def test_audit_document_table_matches_live_recomputation(report: dict) -> None:
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "\r" not in text
    for line in format_table(report).splitlines():
        assert line in text, line
    for needle in (
        "preregistered null",
        PREREGISTRATION_COMMIT,
        RESULT_COMMIT,
        RESULTS_TREE,
        PROTOCOL_BLOB,
        SIDECAR_BLOB,
        MANIFEST_SHA256,
        PROTOCOL_LF_SHA256,
        PROTOCOL_RECORDED_SHA256,
        PROTOCOL_PAYLOAD_SHA256,
        "eol_equivalent_digest",
        "validate_results",
        "gpu_replay_pass_count: 2",
        "audit_sidecar_eol",
        "byte_exact: 25, eol_only: 2, mismatch: 0",
    ):
        assert needle in text, needle


def test_frozen_inputs_and_results_tree_are_unchanged_since_their_commits() -> None:
    protocol_rel = f"{EXPERIMENT_REL}/protocol.json"
    copy_rel = f"{RESULTS_REL}/preregistered-protocol.json"
    assert _git("rev-parse", f"{PREREGISTRATION_COMMIT}:{protocol_rel}") == PROTOCOL_BLOB
    assert _git("rev-parse", f"HEAD:{protocol_rel}") == PROTOCOL_BLOB
    assert _git("rev-parse", f"{RESULT_COMMIT}:{copy_rel}") == PROTOCOL_BLOB
    assert _git("rev-parse", f"HEAD:{copy_rel}") == PROTOCOL_BLOB
    assert _git("rev-parse", f"{RESULT_COMMIT}:{copy_rel}.sha256") == SIDECAR_BLOB
    assert _git("rev-parse", f"HEAD:{copy_rel}.sha256") == SIDECAR_BLOB
    assert _git("rev-parse", f"{RESULT_COMMIT}:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-parse", f"HEAD:{RESULTS_REL}") == RESULTS_TREE
    assert _git("rev-list", "--parents", "-n", "1", RESULT_COMMIT) == (
        f"{RESULT_COMMIT} {PREREGISTRATION_COMMIT}"
    )
    assert _git("status", "--porcelain", "--", RESULTS_REL, protocol_rel) == ""
    eol = _git("ls-files", "--eol", "--", EXPERIMENT_REL)
    assert "w/crlf" not in eol and "w/mixed" not in eol
    # Git stores LF: the blob hashes to the LF digest, never to the recorded one.
    blob = subprocess.run(
        ["git", "cat-file", "blob", PROTOCOL_BLOB], cwd=REPO, check=True, capture_output=True
    ).stdout
    assert hashlib.sha256(blob).hexdigest() == PROTOCOL_LF_SHA256
    assert hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest() == PROTOCOL_RECORDED_SHA256


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
    assert written["passed"] is True and written["counts"]["eol_only"] == 2
    assert not (RESULTS / "posthoc.json").exists()
    assert not (EXPERIMENT / "posthoc.json").exists()
    out = capsys.readouterr().out
    assert "preregistered-protocol.json" in out


def test_table_flag_prints_only_the_markdown_table(capsys) -> None:
    assert audit_module.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| path | checkout bytes |")
    assert PROTOCOL_LF_SHA256 in out and PROTOCOL_RECORDED_SHA256 in out
    assert out.count("\n") == 3
