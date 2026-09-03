"""Transforms, baselines, candidate fits, the predictor contract and the no-tautology checks."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from experiments.wall_loss_geometry_surrogate_v1 import data as d
from experiments.wall_loss_geometry_surrogate_v1 import experiment, models as m
from experiments.wall_loss_geometry_surrogate_v1.experiment import ALL_OUTPUTS, GATED_OUTPUTS, protocol
from experiments.wall_loss_geometry_surrogate_v1.predictor import (
    CLASSIFICATION,
    PREDICTOR_SCHEMA,
    SOURCE_CLASSIFICATION,
    CompiledModel,
    Predictor,
    PredictorContractError,
    matern52,
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


# ---- transforms -----------------------------------------------------------------


def test_logit_transform_is_haldane_anscombe_and_inverts() -> None:
    value, variance = d.to_working(64, 128, "logit")
    assert value == 0.0
    assert variance == pytest.approx(2.0 / 64.5)
    value_full, variance_full = d.to_working(128, 128, "logit")
    assert math.isfinite(value_full) and variance_full == pytest.approx(1.0 / 128.5 + 1.0 / 0.5)
    assert d.working_to_probability(value_full, "logit") == pytest.approx(128.5 / 129.0)
    assert d.working_to_probability(0.0, "logit") == 0.5
    assert d.sigmoid(-800.0) == 0.0 and d.sigmoid(800.0) == 1.0


def test_direct_transform_uses_laplace_smoothed_variance_and_clips() -> None:
    value, variance = d.to_working(128, 128, "direct")
    assert value == 1.0 and variance == pytest.approx((129 / 130) * (1 / 130) / 128)
    assert d.working_to_probability(1.3, "direct") == 1.0 and d.working_to_probability(-0.2, "direct") == 0.0
    with pytest.raises(ValueError):
        d.to_working(1, 2, "probit")


def test_prediction_time_noise_is_bounded_and_matches_the_binomial_law() -> None:
    assert d.observation_noise_at(0.7, 512, "direct") == pytest.approx(0.21 / 512)
    assert d.observation_noise_at(5.0, 128, "direct") == pytest.approx((129 / 130) * (1 / 130) / 128)
    p = d.sigmoid(0.4)
    assert d.observation_noise_at(0.4, 128, "logit") == pytest.approx(1.0 / (128 * p * (1 - p)))
    assert math.isfinite(d.observation_noise_at(60.0, 128, "logit"))


def test_binomial_floor_and_floor_correction() -> None:
    assert d.binomial_floor([(64, 128), (64, 128)]) == pytest.approx(math.sqrt(0.25 / 128))
    assert d.floor_corrected(0.05, 0.03) == pytest.approx(0.04)
    assert d.floor_corrected(0.02, 0.03) == 0.0


# ---- baselines ------------------------------------------------------------------


def test_baselines_behave_on_the_fit_table(context) -> None:
    table = context["table"]
    mean = m.fit_global_mean(table)
    assert mean.predict(table.normalized[:3], "p_wall_pooled").tolist() == [pytest.approx(table.probabilities("p_wall_pooled").mean())] * 3
    knn = m.fit_knn(table, 3)
    prediction = knn.predict(table.normalized[:1], "p_wall_pooled")[0]
    distances = np.sqrt(((table.normalized - table.normalized[0]) ** 2).sum(axis=1))
    nearest = np.argsort(distances, kind="stable")[:3]
    assert prediction == pytest.approx(table.probabilities("p_wall_pooled")[nearest].mean())
    ridge = m.fit_ridge(table, 1e-12)
    design = np.hstack([np.ones((len(table.rows), 1)), table.normalized])
    ols, *_ = np.linalg.lstsq(design, table.probabilities("p_wall_pooled"), rcond=None)
    assert np.allclose(ridge.predict(table.normalized, "p_wall_pooled"), design @ ols, atol=1e-6)
    strong = m.fit_ridge(table, 1e6)
    assert np.allclose(strong.predict(table.normalized, "p_wall_pooled"), table.probabilities("p_wall_pooled").mean(), atol=1e-3)


# ---- predictor contract ---------------------------------------------------------


def _toy_contract(block: dict) -> dict:
    value = protocol()
    return {
        "schema_version": PREDICTOR_SCHEMA,
        "classification": CLASSIFICATION,
        "source_dataset_classification": SOURCE_CLASSIFICATION,
        "claim_boundary": value["claim_boundary"],
        "inputs": {"names": ["a", "b"], "units": ["1", "1"], "normaliser": {"minimum": [0.0, 0.0], "span": [1.0, 2.0]}},
        "interpolation_scope": {"chamber_length_max_m": 0.03},
        "outputs": [{"name": "y", "model": "toy", "task": 0, "transform": "direct", "trials": 128}],
        "models": {"toy": block},
        "calibration": {"variance_scale": 1.0, "nominal_probability": 0.9},
    }


def _toy_block() -> dict:
    return {
        "kind": "gp-matern52-ard-fixed-noise-task-covariance",
        "family": "toy",
        "outputs": ["y"],
        "lengthscales": [0.5, 0.7],
        "outputscale": 1.2,
        "task_covariance": [[1.0]],
        "mean_constants": [0.1],
        "standardize": {"mean": 0.6, "scale": 0.2},
        "train": {"x": [[0.0, 0.0], [1.0, 0.5], [0.3, 0.9]], "task": [0, 0, 0], "y_working": [0.5, 0.8, 0.7], "noise_working": [0.001, 0.001, 0.002], "jitter": 0.0},
    }


def test_compiled_model_reproduces_the_closed_form_gp_posterior() -> None:
    block = _toy_block()
    model = CompiledModel("toy", block)
    x = np.asarray(block["train"]["x"])
    y = (np.asarray(block["train"]["y_working"]) - 0.6) / 0.2
    noise = np.asarray(block["train"]["noise_working"]) / 0.04
    k = 1.2 * matern52(x, x, np.asarray(block["lengthscales"])) + np.diag(noise)
    query = np.asarray([[0.2, 0.4]])
    ks = 1.2 * matern52(query, x, np.asarray(block["lengthscales"]))
    alpha = np.linalg.solve(k, y - 0.1)
    mean_std = ks @ alpha + 0.1
    var_std = 1.2 - ks @ np.linalg.solve(k, ks.T)
    mean, variance = model.latent(query, 0)
    assert mean[0] == pytest.approx(mean_std[0] * 0.2 + 0.6)
    assert variance[0] == pytest.approx(var_std[0, 0] * 0.04)
    # Training points are reproduced within the noise.
    mean_train, _ = model.latent(x, 0)
    assert np.all(np.abs(mean_train - np.asarray(block["train"]["y_working"])) < 0.05)


def test_predictor_rejects_malformed_contracts_and_labels_every_prediction() -> None:
    contract = _toy_contract(_toy_block())
    predictor = Predictor(contract)
    with pytest.raises(PredictorContractError):
        Predictor({**contract, "schema_version": "other"})
    with pytest.raises(PredictorContractError):
        Predictor({**contract, "claim_boundary": {**contract["claim_boundary"], "not_performance_model": False}})
    bad = json.loads(json.dumps(contract))
    bad["models"]["toy"]["task_covariance"] = [[1.0, 0.0]]
    with pytest.raises(PredictorContractError):
        Predictor(bad)
    bad = json.loads(json.dumps(contract))
    bad["models"]["toy"]["alpha"] = [0.0, 0.0, 0.0]
    with pytest.raises(PredictorContractError, match="alpha"):
        Predictor(bad)
    with pytest.raises(PredictorContractError):
        predictor.predict([[0.1]])
    prediction = predictor.predict([[0.2, 0.8]])[0]
    assert prediction["classification"] == CLASSIFICATION
    assert prediction["source_dataset_classification"] == SOURCE_CLASSIFICATION
    item = prediction["outputs"]["y"]
    assert item["latent_interval"][0] <= item["probability"] <= item["latent_interval"][1]
    assert item["observation_interval"][0] <= item["latent_interval"][0]
    assert item["observation_interval"][1] >= item["latent_interval"][1]
    assert 0.0 <= item["observation_interval"][0] and item["observation_interval"][1] <= 1.0


def test_calibration_scale_widens_intervals_monotonically() -> None:
    contract = _toy_contract(_toy_block())
    narrow = Predictor(contract).predict([[0.5, 0.5]])[0]["outputs"]["y"]
    wide = Predictor({**contract, "calibration": {"variance_scale": 4.0, "nominal_probability": 0.9}}).predict([[0.5, 0.5]])[0]["outputs"]["y"]
    assert wide["observation_interval"][1] - wide["observation_interval"][0] > narrow["observation_interval"][1] - narrow["observation_interval"][0]
    assert wide["probability"] == narrow["probability"]


# ---- candidates on the real fit role --------------------------------------------


def test_package_exactgp_candidate_exports_a_replaying_contract_block(context) -> None:
    table = context["table"]
    candidate = m.fit_candidate("pkg-exactgp-logit", table, threads=8, seed=0)
    assert set(candidate.blocks) == {f"pkg-exactgp-logit:{name}" for name in ALL_OUTPUTS}
    physical = m.physical_matrix(context["rows"][:12])
    normalized = table.normaliser.transform(physical)
    for name in ("p_wall_pooled", "p_wall_cell2"):
        mean, variance = candidate.latent(normalized, name)
        native_mean, native_variance = m.native_latent(candidate, table, physical, name)
        assert np.max(np.abs(mean - native_mean)) < 1e-9
        assert np.max(np.abs(variance - native_variance)) < 1e-9
        assert np.all(variance >= 0.0)
    probabilities = m.candidate_probabilities(candidate, normalized, "p_wall_pooled")
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    again = m.fit_candidate("pkg-exactgp-logit", table, threads=8, seed=0)
    assert np.array_equal(candidate.hyperparameter_vector(), again.hyperparameter_vector())


@pytest.mark.parametrize("candidate_id", ["botorch-stgp-direct", "botorch-icm-logit"])
def test_botorch_candidates_replay_natively_and_deterministically(context, candidate_id) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    table = context["table"]
    candidate = m.fit_candidate(candidate_id, table, threads=8, seed=0)
    physical = m.physical_matrix(context["rows"][:10])
    normalized = table.normaliser.transform(physical)
    for name in ALL_OUTPUTS:
        mean, variance = candidate.latent(normalized, name)
        native_mean, native_variance = m.native_latent(candidate, table, physical, name)
        assert np.max(np.abs(mean - native_mean)) < 1e-9, name
        assert np.max(np.abs(variance - native_variance)) < 1e-9, name
    again = m.fit_candidate(candidate_id, table, threads=8, seed=0)
    assert np.max(np.abs(candidate.hyperparameter_vector() - again.hyperparameter_vector())) < 1e-9
    if candidate_id == "botorch-icm-logit":
        block = candidate.blocks["botorch-icm-logit:cells"]
        assert len(block["task_covariance"]) == 4 and len(block["mean_constants"]) == 4
        assert block["outputs"] == list(m.CELL_OUTPUTS)
        assert len(block["train"]["x"]) == 4 * len(table.rows)


def test_no_tautology_checks_pass_on_the_screening_dataset(context) -> None:
    report = experiment.no_tautology_report(context["rows"], context["value"]["inputs"]["names"], context["fit_rows"])
    assert report["passed"] is True
    assert report["stored_probabilities_equal_count_ratios"] is True
    assert report["max_single_input_affine_r2"] < 0.99
    assert all(report["ridge_in_sample_rmse_fit_role"][name] > report["binomial_floor_fit_role"][name] for name in GATED_OUTPUTS)


def test_permutation_importance_and_ard_report_shapes(context) -> None:
    table = context["table"]
    candidate = m.fit_candidate("pkg-exactgp-direct", table, threads=8, seed=0)
    importance = m.permutation_importance(candidate, table, GATED_OUTPUTS, repeats=2, namespace="test")
    assert set(importance["increase_by_input"]) == set(table.input_names)
    assert sorted(importance["ranking"]) == sorted(table.input_names)
    ard = m.ard_length_scales(candidate, table.input_names)
    assert set(ard) == set(candidate.blocks)
    for block in ard.values():
        assert len(block["most_sensitive_inputs"]) == 4
