"""Mixture candidate routing and replay, the tree baseline, the dispatching predictor, the learning curve and no-tautology."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.wall_loss_geometry_surrogate_v2 import data as d, experiment, models as m
from experiments.wall_loss_geometry_surrogate_v2.experiment import ALL_OUTPUTS, GATED_OUTPUTS, protocol
from experiments.wall_loss_geometry_surrogate_v2.predictor import (
    CLASSIFICATION,
    NOT_USABLE_LABEL,
    PREDICTOR_SCHEMA,
    SOURCE_CLASSIFICATION,
    USABLE_LABEL,
    Predictor,
    PredictorContractError,
    route_blocks,
)


@pytest.fixture(scope="module")
def context():
    value = protocol()
    rows = experiment.load_rows(value)
    partition = experiment.plan_partition(value, experiment.evidentiary_plan(value), rows)
    fit_rows = d.labels_for_role(rows, partition, "fit")
    trials = {name: int(value["outputs"]["trials"][name]) for name in ALL_OUTPUTS}
    table = m.TrainingTable.build(fit_rows, value["inputs"]["names"], ALL_OUTPUTS, trials)
    return {"value": value, "rows": rows, "partition": partition, "fit_rows": fit_rows, "table": table, "trials": trials}


def _require_ml() -> None:
    for name in ("torch", "botorch", "gpytorch"):
        pytest.importorskip(name)


# ---- toy dispatching contract -----------------------------------------------------


def _toy_block(shift: float) -> dict:
    return {
        "kind": "gp-matern52-ard-fixed-noise-task-covariance",
        "family": "toy",
        "outputs": ["y"],
        "lengthscales": [0.5, 0.7],
        "outputscale": 1.2,
        "task_covariance": [[1.0]],
        "mean_constants": [0.1],
        "standardize": {"mean": 0.6, "scale": 0.2},
        "train": {"x": [[0.0, 0.0], [1.0, 0.5], [0.3, 0.9]], "task": [0, 0, 0], "y_working": [0.5 + shift, 0.8 + shift, 0.7 + shift], "noise_working": [0.001, 0.001, 0.002], "jitter": 0.0},
    }


def _toy_contract() -> dict:
    value = protocol()
    return {
        "schema_version": PREDICTOR_SCHEMA,
        "classification": CLASSIFICATION,
        "source_dataset_classification": SOURCE_CLASSIFICATION,
        "claim_boundary": value["claim_boundary"],
        "mdo_v2_input_status": NOT_USABLE_LABEL,
        "inputs": {"names": ["stage_count", "b"], "units": ["1", "1"], "derived_not_fitted": True, "normaliser": {"minimum": [3.0, 0.0], "span": [2.0, 2.0]}},
        "interpolation_scope": {"chamber_length_max_m": 0.03},
        "outputs": [
            {"name": "y", "model": "all", "task": 0, "transform": "direct", "trials": 128, "dispatch": {"feature": "stage_count", "models": {"3": "sc3", "5": "sc5"}, "default": "all"}}
        ],
        "models": {"all": _toy_block(0.0), "sc3": _toy_block(-0.3), "sc5": _toy_block(0.1)},
        "calibration": {"variance_scale": 1.0, "nominal_probability": 0.9},
    }


def test_dispatching_predictor_routes_on_the_stage_count_feature() -> None:
    contract = _toy_contract()
    predictor = Predictor(contract)
    physical = np.asarray([[3.0, 0.4], [4.0, 0.4], [5.0, 0.4]])
    assert route_blocks(contract["outputs"][0], physical, predictor.input_names) == ["sc3", "all", "sc5"]
    predictions = predictor.predict(physical.tolist())
    assert [p["outputs"]["y"]["model"] for p in predictions] == ["sc3", "all", "sc5"]
    assert predictions[0]["outputs"]["y"]["probability"] < predictions[1]["outputs"]["y"]["probability"] < predictions[2]["outputs"]["y"]["probability"]
    assert all(p["mdo_v2_input_status"] == NOT_USABLE_LABEL for p in predictions)
    plain = json.loads(json.dumps(contract))
    del plain["outputs"][0]["dispatch"]
    fixed = Predictor(plain).predict(physical.tolist())
    assert fixed[1]["outputs"]["y"]["probability"] == pytest.approx(predictions[1]["outputs"]["y"]["probability"])
    assert fixed[0]["outputs"]["y"]["probability"] != pytest.approx(predictions[0]["outputs"]["y"]["probability"])
    bad = json.loads(json.dumps(contract))
    bad["outputs"][0]["dispatch"]["models"]["3"] = "missing"
    with pytest.raises(PredictorContractError, match="dispatch target"):
        Predictor(bad)
    bad = json.loads(json.dumps(contract))
    bad["inputs"]["derived_not_fitted"] = False
    with pytest.raises(PredictorContractError, match="derived"):
        Predictor(bad)
    bad = json.loads(json.dumps(contract))
    bad["mdo_v2_input_status"] = "usable"
    with pytest.raises(PredictorContractError, match="mdo_v2_input_status"):
        Predictor(bad)
    assert Predictor({**contract, "mdo_v2_input_status": USABLE_LABEL}).predict([[4.0, 0.1]])[0]["mdo_v2_input_status"] == USABLE_LABEL


# ---- candidates on the real fit role --------------------------------------------


def test_mixture_candidate_fits_one_gp_per_stage_count_and_replays(context) -> None:
    _require_ml()
    table = context["table"]
    candidate = m.fit_candidate("stage-mixture-stgp-logit", table, threads=8, seed=0)
    assert isinstance(candidate, m.MixtureCandidate)
    assert candidate.diagnostics["served_stage_counts"] == [3, 4, 5]
    assert candidate.diagnostics["fit_designs_per_stage_count"] == {"3": 13, "4": 29, "5": 8}
    spec = candidate.output_spec("p_wall_pooled")
    assert spec["dispatch"]["feature"] == "stage_count" and set(spec["dispatch"]["models"]) == {"3", "4", "5"}
    assert spec["model"] == spec["dispatch"]["default"] == "stage-mixture-stgp-logit:all:p_wall_pooled"
    physical = m.physical_matrix(context["rows"][:12])
    normalized = table.normaliser.transform(physical)
    routes = candidate.route("p_wall_pooled", normalized)
    assert routes == [f"stage-mixture-stgp-logit:sc{row.stage_count}:p_wall_pooled" for row in context["rows"][:12]]
    for name in ALL_OUTPUTS:
        mean, variance = candidate.latent(normalized, name)
        native_mean, native_variance = m.native_latent(candidate, table, physical, name)
        assert np.max(np.abs(mean - native_mean)) < 1e-9, name
        assert np.max(np.abs(variance - native_variance)) < 1e-9, name
    # A design's prediction equals the prediction of its stage count's part alone.
    part = candidate.parts[f"sc{context['rows'][0].stage_count}"]
    alone, _ = part.latent(normalized[:1], "p_wall_pooled")
    assert candidate.latent(normalized[:1], "p_wall_pooled")[0][0] == pytest.approx(alone[0], abs=1e-12)
    again = m.fit_candidate("stage-mixture-stgp-logit", table, threads=8, seed=0)
    assert np.max(np.abs(candidate.hyperparameter_vector() - again.hyperparameter_vector())) < 1e-9


def test_mixture_falls_back_to_the_all_count_gp_below_the_minimum(context) -> None:
    _require_ml()
    table = context["table"]
    candidate = m.fit_stage_mixture(table, "logit", "stage-mixture-stgp-logit", threads=8, seed=0, minimum_per_count=10)
    assert candidate.diagnostics["served_stage_counts"] == [3, 4] and candidate.diagnostics["fallback_stage_counts"] == [5]
    five = [row for row in context["rows"] if row.stage_count == 5][:3]
    routes = candidate.route("p_wall_cell1", table.normaliser.transform(m.physical_matrix(five)))
    assert routes == ["stage-mixture-stgp-logit:all:p_wall_cell1"] * 3


def test_stgp_candidate_on_derived_features_replays_and_exports_31_lengthscales(context) -> None:
    _require_ml()
    table = context["table"]
    candidate = m.fit_candidate("botorch-stgp-direct", table, threads=8, seed=0)
    assert set(candidate.blocks) == {f"botorch-stgp-direct:{name}" for name in ALL_OUTPUTS}
    assert len(candidate.blocks["botorch-stgp-direct:p_wall_pooled"]["lengthscales"]) == 31
    physical = m.physical_matrix(context["rows"][:8])
    normalized = table.normaliser.transform(physical)
    for name in ("p_wall_pooled", "p_wall_cell4"):
        mean, variance = candidate.latent(normalized, name)
        native_mean, native_variance = m.native_latent(candidate, table, physical, name)
        assert np.max(np.abs(mean - native_mean)) < 1e-9 and np.max(np.abs(variance - native_variance)) < 1e-9
    ard = m.ard_length_scales(candidate, table.input_names)
    assert set(ard) == set(candidate.blocks)


# ---- baselines ------------------------------------------------------------------


def test_gbt_baseline_is_deterministic_clipped_and_reports_importances(context) -> None:
    pytest.importorskip("sklearn")
    table = context["table"]
    first = m.fit_gbt(table, m.GBT_GRID[2])
    second = m.fit_gbt(table, m.GBT_GRID[2])
    x = table.normaliser.transform(m.physical_matrix(context["rows"][:20]))
    for name in ALL_OUTPUTS:
        assert np.array_equal(first.predict(x, name), second.predict(x, name))
        assert np.all((first.predict(x, name) >= 0.0) & (first.predict(x, name) <= 1.0))
    importance = first.parameters["feature_importance_mean_over_outputs"]
    assert set(importance) == set(table.input_names)
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-6)
    assert sorted(first.parameters["feature_ranking_mean_over_outputs"]) == sorted(table.input_names)
    assert m.BASELINE_ORDER == ("global-mean", "knn-3", "ridge", "gbt") and len(m.GBT_GRID) == 6


def test_subset_table_keeps_the_parent_unit_box(context) -> None:
    table = context["table"]
    sub = m.subset_table(table, table.rows[:5])
    assert sub.normaliser is table.normaliser
    assert np.array_equal(sub.normalized, table.normalized[:5])


# ---- learning curve -------------------------------------------------------------


def test_power_law_extrapolation_reports_designs_needed_or_a_flat_curve() -> None:
    falling = [{"size": 20, "pooled_rmse_mean": 0.10}, {"size": 40, "pooled_rmse_mean": 0.05 * 2 ** 0.5}, {"size": 80, "pooled_rmse_mean": 0.05}]
    report = m.power_law_extrapolation(falling, target=0.05)
    assert report["fitted"] is True and report["b"] == pytest.approx(-0.5, abs=1e-6)
    assert report["designs_needed_for_target"] == pytest.approx(80.0, rel=1e-6)
    flat = [{"size": 20, "pooled_rmse_mean": 0.06}, {"size": 40, "pooled_rmse_mean": 0.061}]
    report = m.power_law_extrapolation(flat, target=0.05)
    assert report["fitted"] is True and report["designs_needed_for_target"] is None


def test_learning_curve_is_nested_and_scored_on_the_evaluation_role(context) -> None:
    _require_ml()
    value = context["value"]
    rows = context["rows"]
    selection = d.labels_for_role(rows, context["partition"], "method-selection")
    curve = m.learning_curve(
        "botorch-stgp-logit", context["fit_rows"], selection, value["inputs"]["names"], GATED_OUTPUTS, context["trials"],
        sizes=[20, 50], seeds=[1], threads=8, torch_seed=0, namespace="test",
    )
    run = curve["runs"][0]["curve"]
    assert run[0]["size"] == 20 and len(run[0]["case_ids"]) == 20
    assert set(run[0]["case_ids"]) <= set(run[1]["case_ids"]) and len(run[1]["case_ids"]) == 50
    assert curve["summary"][1]["pooled_rmse_mean"] > 0.0 and curve["gated"] is False


# ---- no tautology ---------------------------------------------------------------


def test_no_tautology_uses_leave_one_out_ridge_and_passes(context) -> None:
    report = experiment.no_tautology_report(context["rows"], context["value"]["inputs"]["names"], context["fit_rows"])
    assert report["passed"] is True
    assert report["max_single_input_affine_r2"] < 0.99
    assert all(report["ridge_leave_one_out_rmse_fit_role"][name] > report["binomial_floor_fit_role"][name] for name in GATED_OUTPUTS)
