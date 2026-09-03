"""The post-hoc audit re-derives the recorded held-out failure from the sealed data (read-only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.cusp_topology_search_v3 import audit_held_out as A
from experiments.cusp_topology_search_v3 import experiment as E

pytestmark = pytest.mark.skipif(not (E.RESULTS_ROOT / "manifest.json").is_file(), reason="not executed yet")


@pytest.fixture(scope="module")
def report() -> dict:
    return A.audit()


def test_audit_binds_the_recorded_outcome(report: dict) -> None:
    bundle = report["bundle"]
    assert bundle["manifest_state"] == bundle["terminal_state"] == A.RECORDED_TERMINAL_STATE
    assert bundle["recorded_failing_gate"] == A.RECORDED_FAILING_GATE
    assert len(bundle["recorded_failing_designs"]) == A.RECORDED_FAILING_DESIGN_COUNT
    assert bundle["other_campaign_gates_all_true"] is True
    assert bundle["design_count"] == bundle["stable_design_count"] == 281


def test_every_recorded_failure_is_a_dropped_axis_cluster_and_the_intended_filter_passes(report: dict) -> None:
    assert report["recorded_failures_explained_by_dropped_clusters"] is True
    assert report["sealed_axis_clusters_total"] == 206
    assert report["clusters_dropped_by_recorded_filter_total"] == 26
    assert report["max_dropped_centroid_r_m"] < 1.0e-7
    assert report["corrected_filter_pass_count"] == 56
    assert report["corrected_filter_max_difference_m"] <= report["held_out_tolerance_m"]
    failing = {row["case_id"] for row in report["designs"] if not row["recorded_gate_passed"]}
    assert len(failing) == A.RECORDED_FAILING_DESIGN_COUNT
    for row in report["designs"]:
        assert row["corrected_filter_passed"] is True
        if row["case_id"] in failing:
            assert row["clusters_dropped_in_channel"] == len(row["recorded_unmatched_observed_z_m"]) > 0
            assert row["stage_count"] >= 5
        else:
            assert row["clusters_dropped_in_channel"] == 0


def test_audit_refuses_to_write_inside_results(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="immutable results"):
        A.main(["--json", str(E.RESULTS_ROOT / "audit.json")])
    target = tmp_path / "audit.json"
    assert A.main(["--json", str(target)]) == 0 and target.is_file()


def test_audit_document_states_the_root_cause() -> None:
    text = (E.EXPERIMENT / "POSTHOC_AUDIT.md").read_text(encoding="utf-8")
    assert "r_m == 0.0" in text and "assessment_rejection" in text
    assert "56/56" in text and "cusp_topology_search_v3_1" in text
    assert "\r" not in text
