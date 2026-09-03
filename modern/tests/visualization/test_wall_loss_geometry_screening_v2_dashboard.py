"""Wall-loss geometry screening v2 dashboard: bundle-bound payload, deterministic render, committed HTML."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from statistics import median

import pytest

from visualization import generate_wall_loss_geometry_screening_v2_dashboard as G

RESULTS = G.RESULTS
COMMITTED = G.DEFAULT_OUTPUT

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="the v2 screening has not executed yet")


@pytest.fixture(scope="module")
def payload() -> dict:
    return G.build_payload()


def test_bundle_verification_is_byte_exact_and_refuses_tampering(tmp_path: Path) -> None:
    identity = G.verify_bundle(RESULTS)
    assert identity["experiment_id"] == "orbit-wall-loss-geometry-screening-v2"
    manifest_entries = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
    assert identity["verified_file_count"] == sum(entry["type"] == "file" for entry in manifest_entries) > 1000
    assert identity["artifact_count"] == len(manifest_entries)
    assert re.fullmatch(r"[0-9a-f]{40}", identity["preregistration_commit_sha"])
    clone = tmp_path / "results"
    clone.mkdir()
    for name in ("manifest.json", "terminal.json", "execution-lock.json"):
        (clone / name).write_bytes((RESULTS / name).read_bytes())
    manifest = json.loads((clone / "manifest.json").read_text(encoding="utf-8"))
    first = next(entry for entry in manifest["artifacts"] if entry["type"] == "file")
    target = clone / first["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError)):
        G.verify_bundle(clone)


def test_payload_recomputes_headline_and_carries_labels(payload: dict) -> None:
    assert payload["classification"] == "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
    assert payload["evidentiary"] is True and payload["campaign_status"] == "accepted_screening_dataset"
    assert payload["design_count"] == len(payload["designs"]) == 97
    assert payload["cell_count"] == len(payload["cells"]) == 377
    assert payload["block_count"] == 4
    p2 = [item for item in payload["designs"] if item["set_id"] == "p2_divergent_exit"]
    assert len(p2) == 1 and p2[0]["label"] == G.LABEL_P2 and p2[0]["v1"] is None
    assert all(item["v1"] is not None for item in payload["designs"] if item["set_id"] == "sweep_v2")
    cells = payload["cells"]
    headline = payload["headline"]
    assert headline["cells_topped_up"] == sum(cell["topped"] for cell in cells)
    assert headline["cells_surrogate_ready"] == sum(cell["ready"] for cell in cells)
    assert headline["jeffreys_floor_median"] == median(cell["jfloor"] for cell in cells)
    assert all(cell["n"] in (128, 512) for cell in cells)
    assert all(cell["control"]["n"] == cell["n"] // 8 for cell in cells)
    assert payload["gates"]["passed"] is True and payload["gates"]["control_gate"]["passed"] is True
    assert abs(payload["control_gate"]["estimated_bias_2N_minus_N"]) <= 0.02
    assert len(payload["axis_profiles"]) == sum(item["representative"] for item in payload["designs"]) == 5
    assert "http" not in json.dumps(payload)


def test_render_is_deterministic_and_matches_the_committed_dashboard(payload: dict) -> None:
    html_a = G.render_html(payload)
    html_b = G.render_html(G.build_payload())
    assert html_a == html_b
    assert html_a.count("__PAYLOAD_JSON__") == 0
    assert "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS" in html_a and "P2_QUALIFIED_FIELD_SCREENING_LAUNCH_DESIGN" in html_a
    assert len(html_a.encode("utf-8")) <= G.MAX_HTML_BYTES
    if COMMITTED.is_file():
        committed = COMMITTED.read_bytes().replace(b"\r\n", b"\n")
        assert sha256(committed).hexdigest() == sha256(html_a.encode("utf-8")).hexdigest()


def test_payload_validation_rejects_malformed_rows(payload: dict) -> None:
    bad = json.loads(json.dumps(payload))
    bad["cells"][0]["wall"]["lo"] = 1.5
    with pytest.raises(ValueError, match="interval is malformed"):
        G.validate_payload(bad)
    bad = json.loads(json.dumps(payload))
    bad["cells"][0]["n"] = 200
    with pytest.raises(ValueError, match="partition|blocks"):
        G.validate_payload(bad)
    bad = json.loads(json.dumps(payload))
    cell = bad["cells"][0]
    cell["n1"] = cell["n"]  # n equals one block of the wrong size -> the block rule fires
    cell["topped"] = True
    with pytest.raises(ValueError, match="blocks"):
        G.validate_payload(bad)
    bad = json.loads(json.dumps(payload))
    bad["claim_boundary"]["not_p2_qualified"] = False
    with pytest.raises(ValueError, match="not P2-qualified"):
        G.validate_payload(bad)
