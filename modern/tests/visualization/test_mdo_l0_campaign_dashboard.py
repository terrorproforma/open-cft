"""MDO L0 campaign v1 dashboard: bundle-bound, deterministic, offline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
VISUALIZATION = MODERN / "visualization"
GENERATOR = VISUALIZATION / "generate_mdo_l0_campaign_v1_dashboard.py"
OUTPUT = VISUALIZATION / "mdo-l0-campaign-v1.html"
RESULTS = MODERN / "experiments" / "mdo_l0_campaign_v1" / "results"


def _module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mdo_dashboard", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle_available() -> bool:
    return (RESULTS / "manifest.json").is_file()


def test_generator_is_offline_and_has_an_error_sink() -> None:
    module = _module()
    template = module.TEMPLATE
    assert 'id="jserrors"' in template
    assert "__PAYLOAD__" in template
    assert not re.search(r'(src|href)="https?://', template)
    assert "<link" not in template
    assert module.MAX_HTML_BYTES <= 2_500_000


def test_render_html_escapes_script_terminators() -> None:
    module = _module()
    html = module.render_html({"schema": module.SCHEMA, "text": "</script><b>x"})
    assert "</script><b>x" not in html.split('id="payload"')[1].split("</script>")[0]
    assert "__PAYLOAD__" not in html


def test_bundle_verification_fails_closed_on_tampering(tmp_path: Path) -> None:
    module = _module()
    if not _bundle_available():
        pytest.skip("campaign bundle not recorded yet")
    import shutil

    copy = tmp_path / "results"
    shutil.copytree(RESULTS, copy)
    target = copy / "artifacts" / "metrics.json"
    target.write_bytes(target.read_bytes().replace(b"final_hypervolume", b"final_hyperVolume", 1))
    with pytest.raises(ValueError, match="byte mismatch"):
        module.Bundle(copy, expected_manifest_sha256=None)
    with pytest.raises(ValueError, match="pinned"):
        module.Bundle(RESULTS, expected_manifest_sha256="0" * 64)


def test_committed_dashboard_is_the_deterministic_render_of_the_pinned_bundle(tmp_path: Path) -> None:
    module = _module()
    if not _bundle_available():
        pytest.skip("campaign bundle not recorded yet")
    assert module.EXPECTED_MANIFEST_SHA256 is not None
    assert (
        hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest()
        == module.EXPECTED_MANIFEST_SHA256
    )
    first = tmp_path / "a.html"
    second = tmp_path / "b.html"
    report = module.generate(RESULTS, first)
    module.generate(RESULTS, second)
    assert first.read_bytes() == second.read_bytes()
    assert OUTPUT.is_file(), "committed dashboard missing"
    assert OUTPUT.read_bytes() == first.read_bytes(), "committed dashboard is stale"
    assert report["bytes"] <= module.MAX_HTML_BYTES
    assert b"\r" not in OUTPUT.read_bytes()


def _payload() -> dict:
    module = _module()
    html = OUTPUT.read_text(encoding="utf-8")
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</")), module


def test_payload_numbers_are_the_bundle_numbers() -> None:
    if not _bundle_available() or not OUTPUT.is_file():
        pytest.skip("campaign bundle or dashboard not present")
    payload, module = _payload()
    metrics = json.loads((RESULTS / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    gates = json.loads((RESULTS / "artifacts" / "gates.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert payload["identity"]["terminal_state"] == manifest["state"]
    assert payload["identity"]["manifest_sha256"] == module.EXPECTED_MANIFEST_SHA256
    for key, run in payload["runs"].items():
        assert run["final_hypervolume"] == metrics["runs"][key]["final_hypervolume"]
        assert run["pareto_set_size"] == metrics["runs"][key]["pareto_set_size"]
        assert run["evaluations"] == payload["plan"]["evaluations_per_run"]
        curve = [point[1] for point in run["curve"]]
        assert all(b >= a for a, b in zip(curve, curve[1:], strict=False))
    assert payload["gates"]["all_binding_passed"] == gates["all_binding_passed"]
    assert payload["gates"]["bo_beats_random"]["passed"] == gates["reported_not_binding"]["bo_beats_random"]["passed"]
    assert "no thruster-performance claim" in payload["protocol"]["claim_boundary"]["statement"]
    assert payload["identity"]["preregistration_commit"] == module.PREREGISTRATION_COMMIT_SHA
    assert len(payload["pooled"]["robust"]["designs"]) == payload["pooled"]["robust"]["front_size"]


def test_dashboard_has_claim_boundary_panel_and_all_sections() -> None:
    if not OUTPUT.is_file():
        pytest.skip("dashboard not present")
    html = OUTPUT.read_text(encoding="utf-8")
    for section in ("claim", "hv", "hvtable", "gates", "fronts", "parallel", "sensitivity", "timing", "protocol", "provenance"):
        assert f'id="{section}"' in html
    assert "Claim boundary" in html
    assert not re.search(r'(src|href)="https?://', html)
