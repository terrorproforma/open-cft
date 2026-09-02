"""Integrity tests for the material-fields implementation-digest EOL audit.

Everything asserted here is re-derived from the artifacts on disk, the live
source and Git; the audit script must never write anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from cft_revival.material_fields import validate_artifact_bundle
from cft_revival.material_fields.acceptance import _evidence_implementation_sha256
from cft_revival.material_fields.numerics import _implementation_sha256

MODERN = Path(__file__).resolve().parents[2]
REPO = MODERN.parent
EXAMPLE = MODERN / "examples" / "material_fields"
ARTIFACTS = EXAMPLE / "artifacts"
AUDIT_MD = EXAMPLE / "POSTHOC_AUDIT.md"
SCRIPT = EXAMPLE / "audit_implementation_eol.py"
SOURCE_REL = "modern/src/cft_revival/material_fields"

CRLF_ERA_COMMIT = "8603a905f8b19873e9a91c1afd237864e8b31aff"
CRLF_ERA_DIGESTS = {
    "evidence": "d229f62d7ba6289646291d925f404785ab879b91f59185a91a90c327e92966b8",
    "warp": "dc988f4b01648e825ac7a1934b8ddca88ad53d1fa5859c8471e1dfcec745cd0b",
    "python": "734cff6aabe3964690ee6ccfa3bc5c3f9f88f2bc7184ffc9390a06b5b903e6b5",
}
LF_DIGESTS = {
    "evidence": "ef17d1618a934bd1038e24ead341a519dcf365f8f54f1d360afbfedf4bf908db",
    "warp": "6ced73daca60f883440d9f1a4287549ecd2cb8335c138e0fb121b319c0038d2f",
    "python": "2ce98ebd46cab554fc38e81a099935eafcf4a93cf1059a3d86c64fb498fcdd61",
}
ERA_PAYLOADS = {
    "compact-high-gradient-stack.material-field.json": (
        "7579f1602c75cdf6773b24279dd7621dd9c290dd9c2205b875da0962e5c7ed67"
    ),
    "divergent-exit-stack.material-field.json": (
        "da7ef3f3660f1b2e6f6ea3ac9840bed23b1f177efc9168f75d3c90f5c5f12966"
    ),
    "historical-envelope-baseline.material-field.json": (
        "d91f4dd8b86251ec4294948b7df4e8a700ec362e45dc920752fca4592039a860"
    ),
    "manifest.json": "32ce64983a03fc7278be1ceaa7bf20fb73e9b04926786530a1801148893ba134",
}
REBOUND_PAYLOADS = {
    "compact-high-gradient-stack.material-field.json": (
        "a6b69d03bde3626f31858b2b914ef8b6d2d9bdc26a3925371b7914a300f60da0"
    ),
    "divergent-exit-stack.material-field.json": (
        "cf703753108f36a0d175e0540262b367d034080141634a062c8820c582194a06"
    ),
    "historical-envelope-baseline.material-field.json": (
        "dc1ab5ed462fd34271db64866fb45097767ce64c0f0129bcae69591a68244dcf"
    ),
    "manifest.json": "eba362d8b18f46f8e5254eceecb4092c0e12b35673a011014a3751c43113ae7c",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("material_fields_eol_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def script():
    return _load_script()


@pytest.fixture(scope="module")
def report(script) -> dict:
    before = _tree_digest(EXAMPLE), _tree_digest(MODERN / "src" / "cft_revival" / "material_fields")
    value = script.audit()
    after = _tree_digest(EXAMPLE), _tree_digest(MODERN / "src" / "cft_revival" / "material_fields")
    assert after == before, "audit must be read-only"
    return value


def test_audit_passes_in_the_rebound_lf_state(report: dict) -> None:
    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["source_is_lf"] is True
    assert report["live_state"] == "rebound_lf"
    assert report["artifacts"]["counts"] == {"byte_exact": 3, "eol_only": 0, "mismatch": 0}
    assert report["artifacts"]["all_files_byte_exact"] is True
    recorded = {item["role"]: item for item in report["artifacts"]["recorded_implementation_digests"]}
    assert set(recorded) == {"evidence", "warp", "python"}
    for role, item in recorded.items():
        assert item["status"] == "byte_exact"
        assert item["digest"] == LF_DIGESTS[role]
    assert recorded["evidence"]["occurrences"] == 36
    assert recorded["warp"]["occurrences"] == 63
    assert recorded["python"]["occurrences"] == 6
    assert recorded["evidence"]["keys"] == ["evidence_implementation_sha256"]
    assert recorded["warp"]["keys"] == recorded["python"]["keys"] == ["implementation_sha256"]


def test_history_anchors_the_crlf_era_digests_to_the_generating_commit(report: dict, script) -> None:
    history = report["history"]
    assert history["commit"] == CRLF_ERA_COMMIT == script.CRLF_ERA_COMMIT
    assert history["all_crlf_reproduce_recorded"] is True
    assert history["era_artifacts_recorded_exactly_the_three"] is True
    assert history["artifact_digests_recorded_at_era"] == {
        CRLF_ERA_DIGESTS["evidence"]: 36,
        CRLF_ERA_DIGESTS["warp"]: 63,
        CRLF_ERA_DIGESTS["python"]: 6,
    }
    for role, item in history["digests"].items():
        assert item["recorded_at_era"] == CRLF_ERA_DIGESTS[role]
        assert item["blob_crlf_sha256"] == CRLF_ERA_DIGESTS[role]
        assert item["blob_lf_sha256"] == LF_DIGESTS[role]
        assert item["blob_lf_sha256"] != item["blob_crlf_sha256"]
    # the era artifacts sealed the payload digests the devlog recorded
    for name, payload in ERA_PAYLOADS.items():
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"{CRLF_ERA_COMMIT}:modern/examples/material_fields/artifacts/{name}"],
            cwd=REPO, check=True, capture_output=True,
        ).stdout
        assert json.loads(blob)["integrity"]["payload_sha256"] == payload


def test_live_source_is_the_generating_source_and_its_lf_digests_are_recorded(script) -> None:
    assert _git("diff", "--quiet", CRLF_ERA_COMMIT, "HEAD", "--", SOURCE_REL) == ""
    assert _git("status", "--porcelain", "--", SOURCE_REL) == ""
    for role, names in script.FILE_SETS.items():
        assert _implementation_sha256(*names) == LF_DIGESTS[role]
        assert (
            script.implementation_digest(
                lambda f: (MODERN / "src/cft_revival/material_fields" / f).read_bytes(), names, True
            )
            == CRLF_ERA_DIGESTS[role]
        )
    assert _evidence_implementation_sha256() == LF_DIGESTS["evidence"]
    eol = _git("ls-files", "--eol", "--", SOURCE_REL, "modern/examples/material_fields")
    assert "w/crlf" not in eol and "w/mixed" not in eol


def test_rebound_artifacts_validate_and_seal_the_documented_digests() -> None:
    validate_artifact_bundle(ARTIFACTS)
    for name, payload in REBOUND_PAYLOADS.items():
        value = json.loads((ARTIFACTS / name).read_bytes())
        assert value["integrity"]["payload_sha256"] == payload, name
    for path in sorted(ARTIFACTS.glob("*.json")):
        data = path.read_bytes()
        sidecar = path.with_name(path.name + ".sha256").read_bytes()
        assert b"\r" not in data and b"\r" not in sidecar
        assert sidecar == f"{hashlib.sha256(data).hexdigest()}  {path.name}\n".encode("ascii")


def test_audit_document_table_matches_live_recomputation(report: dict, script) -> None:
    text = AUDIT_MD.read_text(encoding="utf-8")
    assert "\r" not in text
    for line in script.format_table(report).splitlines():
        assert line in text, line
    for needle in (
        CRLF_ERA_COMMIT,
        *CRLF_ERA_DIGESTS.values(),
        *LF_DIGESTS.values(),
        *ERA_PAYLOADS.values(),
        *REBOUND_PAYLOADS.values(),
        "refresh_artifact_metadata.py",
        "SCREENING_NOT_ACCEPTED",
        "structurally impossible",
        'newline="\\n"',
        "audit_implementation_eol",
        "rebound_lf",
    ):
        assert needle in text, needle


def test_script_refuses_to_write_inside_the_example_directory(script, tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        script.main(["--json", str(ARTIFACTS / "posthoc.json")])
    assert "must not point inside examples/material_fields" in capsys.readouterr().err
    target = tmp_path / "report.json"
    assert script.main(["--json", str(target)]) == 0
    written = json.loads(target.read_bytes())
    assert written["passed"] is True and written["live_state"] == "rebound_lf"
    assert not (ARTIFACTS / "posthoc.json").exists()
    capsys.readouterr()
    assert script.main(["--table"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| digest role |") and out.count("\n") == 5
