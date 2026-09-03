"""The recorded v1 bundle is a development_rejection (2/15 level-0 meshes below the 10 deg gate).

The one execution at b9449ee5 resolved 13 designs and recorded two mesh-angle failures before
any solve; per protocol (single_execution, no_patch_or_rerun) the bundle is committed as is and
the campaign continues as l1b_hemp_confirmation_v1_1. These tests bind that record: the bundle
is byte-exact, its terminal state is the rejection, exactly the two declared designs failed for
the declared reason, and the 13 resolved records reproduce their own comparison payloads.
"""

from __future__ import annotations

import hashlib

import pytest

from cft_revival.experiment_runtime.canonical import canonical_bytes, strict_json_file

from experiments.cusp_topology_search_v3_1 import topology as T
from experiments.l1b_hemp_confirmation_v1 import experiment as E

RESULTS = E.RESULTS_ROOT
ARTIFACTS = RESULTS / "artifacts"
REJECTED = ("l1a-gs-v3-028-f012c0bf33", "l1a-gs-v3-048-aabacb3a59")

pytestmark = pytest.mark.skipif(not (RESULTS / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return strict_json_file(RESULTS / "manifest.json")


def test_bundle_is_a_recorded_development_rejection_and_byte_exact(manifest: dict) -> None:
    assert manifest["state"] == "development_rejection"
    terminal = strict_json_file(RESULTS / "terminal.json")
    assert terminal["state"] == "development_rejection" and terminal["primary_error"] is None
    assert terminal["payload"] == {"failed_design_count": 2, "resolved_design_count": 13, "stage_wall_s": terminal["payload"]["stage_wall_s"]}
    lock = strict_json_file(RESULTS / "execution-lock.json")
    assert lock["command"].endswith("run execute") and "gpu-not-used" in lock["device"] and lock["commit"].startswith("b9449ee5")
    for entry in manifest["artifacts"]:
        if entry["type"] != "file":
            continue
        raw = (RESULTS / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["byte_sha256"], entry["path"]
        assert len(raw) == entry["bytes"], entry["path"]
    assert not (ARTIFACTS / "gates.json").exists() and not (ARTIFACTS / "campaign-result.json").exists()


def test_exactly_the_two_declared_designs_failed_the_angle_gate() -> None:
    failures = strict_json_file(ARTIFACTS / "design-failures.json")["failed"]
    assert sorted(item["key"].split(":")[1] for item in failures) == sorted(REJECTED)
    for item in failures:
        assert item["stage"] == "resolve" and item["resource_blocked"] is False
        assert "minimum-angle rejection gate" in item["reason"]
    protocol = strict_json_file(ARTIFACTS / "protocol.json")
    assert protocol["p2"]["mesh"]["reject_below_angle_deg"] == 10.0 and protocol == E.protocol()


def test_frozen_preregistration_files_match_the_bundle() -> None:
    assert (ARTIFACTS / "shakedown.json").read_bytes() == E.SHAKEDOWN_PATH.read_bytes()
    assert strict_json_file(ARTIFACTS / "authorities.json") == strict_json_file(E.AUTHORITIES_PATH)
    assert strict_json_file(ARTIFACTS / "design-authorities.json") == strict_json_file(E.DESIGN_AUTHORITIES_PATH)


def test_the_thirteen_resolved_records_are_self_consistent() -> None:
    value = E.protocol()
    tolerance = value["definition_v3_import"]["stability_tolerance_m"]
    records = sorted(path for path in (ARTIFACTS / "designs" / "hemp_like_v3").glob("*.json") if not path.name.endswith(".sha256.json"))
    assert len(records) == 13 and not any(path.stem in REJECTED for path in records)
    for path in records:
        record = strict_json_file(path)
        assert record["status"] == "resolved" and all(record["gate_checks"].values()), path.name
        assert record["gate_checks"] == E.design_gate_checks(record)
        assert T.compare_resolutions(record["accepted"], record["refined"], tolerance) == record["sampling_stability"]
        payload = {key: record[key] for key in ("axis_window_m", "axis_window_reproduced", *E.MAP_ROLES, "descriptors", "sampling_stability", "p2_discretisation", "comparison")}
        assert hashlib.sha256(canonical_bytes(payload)).hexdigest() == record["comparison_payload_sha256"]
        assert record["evidence"]["p2"]["all_levels_converged"] and record["evidence"]["p2"]["level_count"] == 2
