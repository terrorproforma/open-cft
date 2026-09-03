"""Dependency-light evaluator of ``predictor.json`` (numpy only; no torch).

This is the consumer contract an MDO v2 can use: physical design inputs in,
per-output probability, calibrated 90 % latent and observation intervals out,
always carrying the screening label.  The contract encodes every fitted model
as a Gaussian process with a Matérn-5/2 ARD kernel times a task covariance
(single-task models have a 1x1 task covariance), fixed per-row observation
noise, per-task constant means and one affine output standardisation, so the
same arithmetic reproduces the package ``ExactGP``, BoTorch ``SingleTaskGP``
(fixed noise) and BoTorch ``MultiTaskGP`` (ICM) posteriors.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from .data import observation_noise_at, working_to_probability

PREDICTOR_SCHEMA = "cft-revival.wall-loss-geometry-surrogate-v1.predictor/1.0.0"
CLASSIFICATION = "SURROGATE_OF_SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
SOURCE_CLASSIFICATION = "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS"
MODEL_KIND = "gp-matern52-ard-fixed-noise-task-covariance"


class PredictorContractError(ValueError):
    """The predictor artifact is malformed or internally inconsistent."""


def matern52(left: np.ndarray, right: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    """Matérn-5/2 correlation between rows of ``left`` (m x d) and ``right`` (n x d)."""

    scaled = (left[:, None, :] - right[None, :, :]) / lengthscales
    radius = np.sqrt(np.einsum("ijk,ijk->ij", scaled, scaled))
    s = math.sqrt(5.0) * radius
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictorContractError(message)


class CompiledModel:
    """One fitted GP block of the contract with its Cholesky factor."""

    def __init__(self, model_id: str, block: Mapping[str, Any]) -> None:
        _require(block.get("kind") == MODEL_KIND, f"{model_id}: unsupported model kind")
        self.model_id = model_id
        self.family = str(block["family"])
        self.lengthscales = np.asarray(block["lengthscales"], dtype=float)
        self.outputscale = float(block["outputscale"])
        self.task_covariance = np.asarray(block["task_covariance"], dtype=float)
        self.mean_constants = np.asarray(block["mean_constants"], dtype=float)
        self.standardize_mean = float(block["standardize"]["mean"])
        self.standardize_scale = float(block["standardize"]["scale"])
        train = block["train"]
        self.train_x = np.asarray(train["x"], dtype=float)
        self.train_task = np.asarray(train["task"], dtype=int)
        self.train_y = np.asarray(train["y_working"], dtype=float)
        self.train_noise = np.asarray(train["noise_working"], dtype=float)
        self.jitter = float(train["jitter"])
        tasks = self.task_covariance.shape[0]
        _require(self.task_covariance.shape == (tasks, tasks), f"{model_id}: task covariance is not square")
        _require(self.mean_constants.shape == (tasks,), f"{model_id}: one mean constant per task is required")
        _require(self.train_x.ndim == 2 and self.train_x.shape[1] == self.lengthscales.shape[0], f"{model_id}: train_x width differs from the length-scales")
        rows = self.train_x.shape[0]
        _require(self.train_task.shape == (rows,) and self.train_y.shape == (rows,) and self.train_noise.shape == (rows,), f"{model_id}: training columns have different lengths")
        _require(bool(np.all((self.train_task >= 0) & (self.train_task < tasks))), f"{model_id}: task index out of range")
        _require(self.outputscale > 0.0 and self.standardize_scale > 0.0 and bool(np.all(self.lengthscales > 0.0)), f"{model_id}: non-positive scale")
        _require(bool(np.all(np.isfinite(self.train_x))) and bool(np.all(np.isfinite(self.train_y))) and bool(np.all(self.train_noise >= 0.0)), f"{model_id}: non-finite training data")
        y_std = (self.train_y - self.standardize_mean) / self.standardize_scale
        noise_std = self.train_noise / (self.standardize_scale * self.standardize_scale)
        covariance = self.outputscale * matern52(self.train_x, self.train_x, self.lengthscales)
        covariance *= self.task_covariance[self.train_task][:, self.train_task]
        covariance[np.diag_indices(rows)] += noise_std + self.jitter
        try:
            self.cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise PredictorContractError(f"{model_id}: training covariance is not positive definite") from error
        residual = y_std - self.mean_constants[self.train_task]
        self.alpha = np.linalg.solve(self.cholesky.T, np.linalg.solve(self.cholesky, residual))
        stored_alpha = block.get("alpha")
        if stored_alpha is not None:
            _require(
                bool(np.allclose(np.asarray(stored_alpha, dtype=float), self.alpha, rtol=0.0, atol=1e-8)),
                f"{model_id}: stored alpha does not reproduce from the training data",
            )

    def latent(self, normalized_x: np.ndarray, task: int) -> tuple[np.ndarray, np.ndarray]:
        """Working-space latent mean and variance at normalised inputs for one task."""

        cross = self.outputscale * matern52(normalized_x, self.train_x, self.lengthscales)
        cross *= self.task_covariance[task][self.train_task][None, :]
        mean_std = cross @ self.alpha + self.mean_constants[task]
        projection = np.linalg.solve(self.cholesky, cross.T)
        variance_std = self.outputscale * self.task_covariance[task, task] - np.einsum("ij,ij->j", projection, projection)
        variance_std = np.maximum(variance_std, 0.0)
        mean = mean_std * self.standardize_scale + self.standardize_mean
        variance = variance_std * self.standardize_scale * self.standardize_scale
        return mean, variance


class Predictor:
    """Compiled ``predictor.json``: physical inputs -> probabilities with intervals."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        _require(contract.get("schema_version") == PREDICTOR_SCHEMA, "unsupported predictor schema")
        _require(contract.get("classification") == CLASSIFICATION, "predictor classification is not the surrogate label")
        _require(contract.get("source_dataset_classification") == SOURCE_CLASSIFICATION, "source dataset label missing")
        boundary = contract.get("claim_boundary")
        _require(isinstance(boundary, Mapping) and boundary.get("surrogate_of_screening_dataset") is True and boundary.get("not_physical_orbit_evidence") is True and boundary.get("not_performance_model") is True, "claim boundary flags missing")
        self.contract = contract
        inputs = contract["inputs"]
        self.input_names = tuple(inputs["names"])
        self.minimum = np.asarray(inputs["normaliser"]["minimum"], dtype=float)
        self.span = np.asarray(inputs["normaliser"]["span"], dtype=float)
        _require(self.minimum.shape == self.span.shape == (len(self.input_names),), "input normaliser shape")
        _require(bool(np.all(self.span > 0.0)), "input spans must be positive")
        self.models = {model_id: CompiledModel(model_id, block) for model_id, block in contract["models"].items()}
        self.outputs = tuple(contract["outputs"])
        for output in self.outputs:
            _require(output["model"] in self.models, f"{output['name']}: unknown model {output['model']}")
            model = self.models[output["model"]]
            _require(0 <= int(output["task"]) < model.task_covariance.shape[0], f"{output['name']}: task out of range")
            _require(output["transform"] in {"logit", "direct"}, f"{output['name']}: unknown transform")
            _require(int(output["trials"]) > 0, f"{output['name']}: trials must be positive")
        calibration = contract["calibration"]
        self.variance_scale = float(calibration["variance_scale"])
        self.nominal_probability = float(calibration["nominal_probability"])
        _require(self.variance_scale > 0.0 and 0.5 < self.nominal_probability < 1.0, "calibration values are invalid")
        self.z = NormalDist().inv_cdf(0.5 + self.nominal_probability / 2.0)
        self.interpolation_scope = dict(contract["interpolation_scope"])

    def normalize(self, physical: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(physical, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        _require(x.ndim == 2 and x.shape[1] == len(self.input_names), "input width differs from the contract")
        _require(bool(np.all(np.isfinite(x))), "inputs must be finite")
        return (x - self.minimum) / self.span

    def scope_flags(self, physical: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
        """Interpolation-scope diagnostics per input row (unit-box excess and chamber length)."""

        normalized = self.normalize(physical)
        flags = []
        names = self.input_names
        length_max = float(self.interpolation_scope["chamber_length_max_m"])
        has_length_inputs = "stage_count_selector" in names and "stage_pitch_m" in names
        for row_physical, row in zip(np.asarray(physical, dtype=float).reshape(normalized.shape[0], -1), normalized, strict=True):
            excess = float(np.sqrt(np.sum(np.where(row < 0.0, row, np.where(row > 1.0, row - 1.0, 0.0)) ** 2)))
            values = dict(zip(names, row_physical.tolist(), strict=True))
            chamber_length: float | None = None
            within_length: bool | None = None
            if has_length_inputs:
                # The sweep's geometry v1.1 rule: stage_count = min(5, 3 + int(3 s)), length = stage_count * pitch.
                stage_count = min(5, 3 + int(3.0 * values["stage_count_selector"]))
                chamber_length = stage_count * values["stage_pitch_m"]
                within_length = chamber_length < length_max
            flags.append(
                {
                    "unit_box_excess": excess,
                    "inside_fit_box": excess == 0.0,
                    "chamber_length_m_estimate": chamber_length,
                    "within_interpolation_length": within_length,
                    "in_interpolation_scope": excess == 0.0 and within_length is not False,
                }
            )
        return flags

    def predict_working(self, physical: Sequence[Sequence[float]]) -> dict[str, dict[str, np.ndarray]]:
        """Uncalibrated working-space latent mean/variance per output (for replay checks)."""

        normalized = self.normalize(physical)
        result = {}
        for output in self.outputs:
            model = self.models[output["model"]]
            mean, variance = model.latent(normalized, int(output["task"]))
            result[output["name"]] = {"mean": mean, "variance": variance}
        return result

    def predict(self, physical: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
        """Per input row: per output probability, calibrated latent and observation intervals."""

        working = self.predict_working(physical)
        rows = self.normalize(physical).shape[0]
        flags = self.scope_flags(physical)
        predictions = []
        for index in range(rows):
            outputs = {}
            for output in self.outputs:
                name = output["name"]
                transform = output["transform"]
                trials = int(output["trials"])
                mean = float(working[name]["mean"][index])
                latent_variance = float(working[name]["variance"][index]) * self.variance_scale
                noise = observation_noise_at(mean, trials, transform)
                total_variance = (float(working[name]["variance"][index]) + noise) * self.variance_scale
                latent_radius = self.z * math.sqrt(latent_variance)
                total_radius = self.z * math.sqrt(total_variance)
                outputs[name] = {
                    "probability": working_to_probability(mean, transform),
                    "latent_interval": [
                        working_to_probability(mean - latent_radius, transform),
                        working_to_probability(mean + latent_radius, transform),
                    ],
                    "observation_interval": [
                        working_to_probability(mean - total_radius, transform),
                        working_to_probability(mean + total_radius, transform),
                    ],
                    "working_mean": mean,
                    "working_latent_variance_calibrated": latent_variance,
                    "working_total_variance_calibrated": total_variance,
                    "transform": transform,
                    "trials": trials,
                }
            predictions.append(
                {
                    "classification": CLASSIFICATION,
                    "source_dataset_classification": SOURCE_CLASSIFICATION,
                    "nominal_probability": self.nominal_probability,
                    "scope": flags[index],
                    "outputs": outputs,
                }
            )
        return predictions


def load_predictor(contract: Mapping[str, Any]) -> Predictor:
    return Predictor(contract)
