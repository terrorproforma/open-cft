"""MDO L0 campaign v2 dashboard: bundle-bound (v2 and v1), deterministic, offline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
VISUALIZATION = MODERN / "visualization"
GENERATOR = VISUALIZATION / "generate_mdo_l0_campaign_v2_dashboard.py"
OUTPUT = VISUALIZATION / "mdo-l0-campaign-v2.html"
RESULTS = MODERN / "experiments" / "mdo_l0_campaign_v2" / "results"
V1_RESULTS = MODERN / "experiments" / "mdo_l0_campaign_v1" / "results"


def _module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mdo_v2_dashboard", GENERATOR)
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
    assert module.V1_EXPECTED_MANIFEST_SHA256 == "2a326f3cf2e286a0ba8f9c91871d78b935a11e3e2c56bd9164acdc6166485381"


def test_render_html_escapes_script_terminators() -> None:
    module = _module()
    html = module.render_html({"schema": module.SCHEMA, "text": "</script><b>x"})
    assert "</script><b>x" not in html.split('id="payload"')[1].split("</script>")[0]
    assert "__PAYLOAD__" not in html


def test_v1_bundle_is_verified_and_pinned() -> None:
    module = _module()
    if not (V1_RESULTS / "manifest.json").is_file():
        pytest.skip("v1 bundle not present")
    v1 = module.load_v1_bundle()
    assert v1.manifest_sha256 == module.V1_EXPECTED_MANIFEST_SHA256
    summary = module._v1_summary(v1)
    assert summary["total_evaluations"] == 864 and summary["evaluations_per_run"] == 96
    assert summary["bo_beats_random"] == {"wins": 3, "seeds": 3}


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
        module.load_v2_bundle(copy, None)
    with pytest.raises(ValueError, match="pinned"):
        module.load_v2_bundle(RESULTS, "0" * 64)


def test_committed_dashboard_is_the_deterministic_render_of_the_pinned_bundles(tmp_path: Path) -> None:
    module = _module()
    if not _bundle_available():
        pytest.skip("campaign bundle not recorded yet")
    assert module.EXPECTED_MANIFEST_SHA256 is not None
    assert hashlib.sha256((RESULTS / "manifest.json").read_bytes()).hexdigest() == module.EXPECTED_MANIFEST_SHA256
    first = tmp_path / "a.html"
    second = tmp_path / "b.html"
    report = module.generate(RESULTS, first)
    module.generate(RESULTS, second)
    assert first.read_bytes() == second.read_bytes()
    assert OUTPUT.is_file(), "committed dashboard missing"
    assert OUTPUT.read_bytes() == first.read_bytes(), "committed dashboard is stale"
    assert report["bytes"] <= module.MAX_HTML_BYTES
    assert report["v1_manifest_sha256"] == module.V1_EXPECTED_MANIFEST_SHA256
    assert b"\r" not in OUTPUT.read_bytes()


def _payload() -> tuple[dict, object]:
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
    pooled = json.loads((RESULTS / "artifacts" / "pooled-fronts.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((RESULTS / "artifacts" / "sensitivity.json").read_text(encoding="utf-8"))
    assert payload["identity"]["terminal_state"] == manifest["state"]
    assert payload["identity"]["manifest_sha256"] == module.EXPECTED_MANIFEST_SHA256
    assert payload["identity"]["preregistration_commit"] == module.PREREGISTRATION_COMMIT_SHA
    for key, run in payload["runs"].items():
        assert run["final_hypervolume"] == metrics["runs"][key]["final_hypervolume"]
        assert run["pareto_set_size"] == metrics["runs"][key]["pareto_set_size"]
        assert run["pareto_catalogue_indices"] == metrics["runs"][key]["pareto_catalogue_indices"]
        assert run["evaluations"] == payload["plan"]["evaluations_per_run"]
    assert payload["gates"]["all_binding_passed"] == gates["all_binding_passed"]
    assert set(payload["gates"]["binding"]) == set(gates["binding"])
    assert payload["gates"]["bo_beats_random"]["wins"] == gates["reported_not_binding"]["bo_beats_random"]["wins"]
    assert payload["gates"]["closure_cl1_vs_cl2"]["front_size"] == sensitivity["closure_cl2"]["front_size"]
    assert payload["pooled"]["robust"]["catalogue_indices"] == pooled["robust"]["catalogue_indices"]
    assert len(payload["pooled"]["robust"]["designs"]) == pooled["robust"]["front_size"]
    assert [m["catalogue_index"] for m in payload["pooled"]["robust"]["catalogue_membership"]] == pooled["robust"]["catalogue_indices"]
    assert len(payload["catalogue"]) == 96
    assert "no thruster-performance claim" in payload["protocol"]["claim_boundary"]["statement"]
    assert payload["protocol"]["classification"].endswith("not_thruster_performance")
    if (V1_RESULTS / "manifest.json").is_file():
        v1_metrics = json.loads((V1_RESULTS / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
        for key, row in payload["v1"]["hypervolume_table"].items():
            assert row["final_hypervolume"] == v1_metrics["hypervolume_table"][key]["final_hypervolume"]
        assert payload["v1"]["dense_reference"]["robust_hypervolume"] == v1_metrics["dense_reference"]["robust_hypervolume"]


def test_dashboard_has_claim_boundary_v1v2_and_all_sections() -> None:
    if not OUTPUT.is_file():
        pytest.skip("dashboard not present")
    html = OUTPUT.read_text(encoding="utf-8")
    for section in ("claim", "v1v2", "hv", "hvtable", "gates", "catalogue", "fronts", "closures", "widths", "timing", "protocol", "provenance"):
        assert f'id="{section}"' in html
    assert "Claim boundary" in html
    assert not re.search(r'(src|href)="https?://', html)
