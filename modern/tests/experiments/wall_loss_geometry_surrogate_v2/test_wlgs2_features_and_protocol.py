"""Derived features (deterministic, provenance-complete), protocol consistency, v1 partition inheritance and bindings."""

from __future__ import annotations

import copy
import hashlib
import math

import pytest

from cft_revival.experiment_runtime import semantic_sha256
from cft_revival.experiment_runtime.canonical import strict_json_file

from experiments.wall_loss_geometry_surrogate_v1 import data as v1d
from experiments.wall_loss_geometry_surrogate_v1.experiment import PROTOCOL_PATH as V1_PROTOCOL_PATH
from experiments.wall_loss_geometry_surrogate_v2 import data as d, experiment, features as f
from experiments.wall_loss_geometry_surrogate_v2.experiment import PROTOCOL_PATH, REPOSITORY, protocol, protocol_consistency


@pytest.fixture(scope="module")
def value() -> dict:
    return protocol()


@pytest.fixture(scope="module")
def rows(value):
    return experiment.load_rows(value)


# ---- features -------------------------------------------------------------------


def test_feature_table_is_complete_and_declared_derived(value) -> None:
    manifest = f.feature_manifest()
    assert manifest["derived_not_fitted"] is True
    assert manifest["names"] == list(value["inputs"]["names"]) == list(f.FEATURE_NAMES)
    assert len(f.FEATURE_NAMES) == 31 == int(value["inputs"]["count"])
    assert set(manifest["provenance"]) == set(f.FEATURE_NAMES)
    assert all(provenance for provenance in manifest["provenance"].values())
    assert set(f.DISCRETE_FEATURES) == {"stage_count", "stage_count_is_3", "stage_count_is_4", "stage_count_is_5", "first_polarity"}
    assert value["inputs"]["derived_not_fitted"] is True
    shared = {name for name in strict_json_file(V1_PROTOCOL_PATH)["inputs"]["names"] if name in f.FEATURE_NAMES}
    assert shared == {"stage_pitch_m", "magnet_radial_thickness_m"}, "only realised lengths that the sweep takes verbatim from the design may coincide with a v1 raw input (no selector)"


def test_features_are_deterministic_functions_of_the_dataset_row(value, rows) -> None:
    dataset = strict_json_file(REPOSITORY / value["dataset"]["dataset_path"])
    record = next(item for item in dataset["designs"] if item["case_id"] == rows[0].case_id)
    cells = tuple(value["outputs"]["cells"])
    once = f.derive_features(record, cells)
    twice = f.derive_features(copy.deepcopy(record), cells)
    assert once == twice
    assert f.feature_vector(once) == rows[0].inputs
    geometry = record["geometry"]
    assert once["straight_length_m"] == pytest.approx(geometry["exit_start_m"] - geometry["injector_length_m"])
    assert once["exit_length_fraction_realised"] == pytest.approx(geometry["exit_length_m"] / geometry["chamber_length_m"])
    assert once["stage_count_is_3"] + once["stage_count_is_4"] + once["stage_count_is_5"] == 1.0
    assert once[f"stage_count_is_{geometry['stage_count']}"] == 1.0
    assert once["log10_maximum_mirror_ratio"] == pytest.approx(math.log10(record["field"]["sweep_qois"]["maximum_mirror_ratio"]))
    cell = record["launch_design"]["cell_to_field"][2]
    assert once["cell3_cusp_distance_pitches"] == pytest.approx(cell["nearest_axis_bz_peak_distance_m"] / geometry["stage_pitch_m"])
    # Declared exclusions are exact identities on every design.
    for item in dataset["designs"]:
        g = item["geometry"]
        assert math.isclose(g["chamber_length_m"], g["stage_count"] * g["stage_pitch_m"], abs_tol=1e-12)
        assert len(item["field"]["sweep_qois"]["axis_cusp_positions_m"]) == g["stage_count"]
        assert len(item["field"]["sweep_qois"]["axis_null_positions_m"]) == g["stage_count"] + 1


def test_feature_derivation_fails_closed_on_inconsistent_records(value, rows) -> None:
    dataset = strict_json_file(REPOSITORY / value["dataset"]["dataset_path"])
    record = copy.deepcopy(dataset["designs"][0])
    cells = tuple(value["outputs"]["cells"])
    broken = copy.deepcopy(record)
    broken["geometry"]["chamber_length_m"] *= 1.01
    with pytest.raises(ValueError, match="stage_count \\* pitch"):
        f.derive_features(broken, cells)
    broken = copy.deepcopy(record)
    broken["geometry"]["stage_count"] = 6
    with pytest.raises(ValueError, match="outside"):
        f.derive_features(broken, cells)
    broken = copy.deepcopy(record)
    broken["field"]["sweep_qois"]["axis_cusp_positions_m"] = broken["field"]["sweep_qois"]["axis_cusp_positions_m"][1:]
    with pytest.raises(ValueError, match="cusp/null"):
        f.derive_features(broken, cells)


def test_feature_degeneracy_check_passes_and_detects_constants(rows) -> None:
    report = f.feature_degeneracy_report(rows)
    assert report["passed"] is True and report["failures"] == []
    assert report["distinct_feature_tuples"] == 96
    assert report["discrete_level_counts"]["stage_count"] == {"3.0": 26, "4.0": 45, "5.0": 25}
    from dataclasses import replace

    column = f.FEATURE_NAMES.index("wall_radius_m")
    constant = [replace(row, inputs=tuple(0.001 if i == column else v for i, v in enumerate(row.inputs))) for row in rows]
    assert f.feature_degeneracy_report(constant)["failures"] == ["wall_radius_m"]


# ---- protocol -------------------------------------------------------------------


def test_protocol_is_consistent_with_the_modules_and_v1(value) -> None:
    checks = protocol_consistency(value)
    assert checks and all(checks.values()), {k: v for k, v in checks.items() if not v}
    assert b"\r" not in PROTOCOL_PATH.read_bytes()
    v1 = strict_json_file(V1_PROTOCOL_PATH)
    for gate in ("interpolation_rmse_pooled", "interpolation_rmse_cells"):
        assert value["gates"]["binding"][gate]["threshold"] == v1["gates"]["binding"][gate]["threshold"]
    assert value["gates"]["binding"]["coverage_90"]["interval"] == v1["gates"]["binding"]["coverage_90"]["interval"]
    assert value["gates"]["binding"]["beats_best_baseline_2x"]["ratio_minimum"] == v1["gates"]["binding"]["beats_best_baseline_2x"]["ratio_minimum"]
    assert value["partition"]["counts"] == v1["partition"]["counts"]
    assert value["partition"]["seed"] == v1["partition"]["seed"] and value["partition"]["seed_namespace"] == v1["partition"]["seed_namespace"]
    assert value["dataset"]["dataset_file_sha256"] == v1["dataset"]["dataset_file_sha256"]


def test_protocol_mismatches_are_detected(value) -> None:
    edited = copy.deepcopy(value)
    edited["candidates"]["order"] = ["botorch-stgp-logit"]
    assert protocol_consistency(edited)["candidate_order"] is False
    edited = copy.deepcopy(value)
    edited["gates"]["binding"]["interpolation_rmse_pooled"]["threshold"] = 0.06
    checks = protocol_consistency(edited)
    assert checks["gate_thresholds"] is False and checks["gates_verbatim_v1"] is False
    edited = copy.deepcopy(value)
    edited["partition"]["seed"] = 1
    assert protocol_consistency(edited)["partition_inherited_from_v1"] is False
    edited = copy.deepcopy(value)
    edited["inputs"]["names"] = list(value["inputs"]["names"])[:-1]
    assert protocol_consistency(edited)["inputs"] is False
    with pytest.raises(ValueError, match="protocol/module mismatch"):
        experiment.require_protocol_consistency(edited)


# ---- partition inheritance and bindings -------------------------------------------


def test_evidentiary_partition_equals_v1_by_hash_and_shakedown_differs(value, rows) -> None:
    partition = experiment.plan_partition(value, experiment.evidentiary_plan(value), rows)
    assert semantic_sha256(partition) == value["v1_binding"]["partitions_semantic_sha256"]
    v1_partition = d.v1_partition(value["v1_binding"])
    assert partition["roles"] == v1_partition["roles"]
    assert hashlib.sha256((REPOSITORY / value["v1_binding"]["files"]["partitions"]["path"]).read_bytes()).hexdigest() == value["v1_binding"]["files"]["partitions"]["sha256"]
    fit = d.labels_for_role(rows, partition, "fit")
    assert {k: sum(row.stage_count == k for row in fit) for k in (3, 4, 5)} == {int(k): v for k, v in value["partition"]["fit_role_stage_counts"].items()}
    disjointness = experiment.shakedown_disjointness(value, rows)
    assert disjointness["proven"] is True and disjointness["evidentiary_partition_equals_v1"] is True
    assert disjointness["per_role_overlap"]["extrapolation"]["shared"] == 10
    assert disjointness["shakedown_design_sha256"] != disjointness["evidentiary_design_sha256"]


def test_v1_binding_report_passes_and_fails_closed(value, rows) -> None:
    report = d.v1_binding_report(value["v1_binding"], rows)
    assert report["passed"] is True, {k: v for k, v in report["checks"].items() if not v}
    assert report["v1_status"] == "rejected_surrogate"
    edited = copy.deepcopy(value["v1_binding"])
    edited["files"]["assessment"]["sha256"] = "0" * 64
    assert d.v1_binding_report(edited, rows, use_git=False)["checks"]["assessment_file_sha256"] is False
    with pytest.raises(d.V1BindingError, match="assessment_file_sha256"):
        d.require_v1_binding(edited, rows, use_git=False)
    edited = copy.deepcopy(value["v1_binding"])
    edited["recorded_status"] = "accepted_surrogate"
    assert d.v1_binding_report(edited, rows, use_git=False)["checks"]["v1_status_recorded"] is False


def test_v1_assessment_loader_reproduces_v1_recorded_metrics(value) -> None:
    v1 = d.load_v1_assessment(value["v1_binding"])
    assert v1["selected_candidate"] == value["v1_binding"]["recorded_selected_candidate"]
    scope = v1["scopes"]["interpolation"]
    assert len(scope["case_ids"]) == 16 and len(v1["scopes"]["extrapolation"]["case_ids"]) == 10
    v1_partition = d.v1_partition(value["v1_binding"])
    assert sorted(scope["case_ids"]) == sorted(v1_partition["roles"]["assessment"])
    errors = [scope["designs"][c]["p_wall_pooled"]["error"] for c in scope["case_ids"]]
    assert math.sqrt(sum(e * e for e in errors) / len(errors)) == pytest.approx(scope["per_output_rmse"]["p_wall_pooled"], abs=1e-12)


def test_dataset_binding_is_v1s(value) -> None:
    report = v1d.dataset_binding_report(value["dataset"])
    assert report["passed"] is True
