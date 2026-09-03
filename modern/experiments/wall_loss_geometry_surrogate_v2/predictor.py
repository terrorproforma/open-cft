"""Dependency-light evaluator of the v2 ``predictor.json`` (numpy only; no torch, no sklearn).

Same GP block arithmetic as v1 (:class:`..wall_loss_geometry_surrogate_v1.predictor.CompiledModel`,
re-used verbatim) with two v2 extensions: the inputs are DERIVED geometry/field
features (the contract carries the feature manifest; the consumer computes the
features with ``features.derive_features`` on a design's committed sweep record),
and an output may carry a ``dispatch`` rule that routes a design to the GP block
of its stage count (the per-stage-count mixture) with a declared default block.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from ..wall_loss_geometry_surrogate_v1.data import observation_noise_at, working_to_probability
from ..wall_loss_geometry_surrogate_v1.predictor import (
    CLASSIFICATION,
    MODEL_KIND,
    SOURCE_CLASSIFICATION,
    CompiledModel,
    PredictorContractError,
    matern52,
)

PREDICTOR_SCHEMA = "cft-revival.wall-loss-geometry-surrogate-v2.predictor/1.0.0"
USABLE_LABEL = "usable_as_mdo_v2_input_with_screening_label"
NOT_USABLE_LABEL = "not_usable_as_mdo_v2_input_rejected_surrogate"

__all__ = [
    "CLASSIFICATION",
    "MODEL_KIND",
    "NOT_USABLE_LABEL",
    "PREDICTOR_SCHEMA",
    "SOURCE_CLASSIFICATION",
    "USABLE_LABEL",
    "CompiledModel",
    "Predictor",
    "PredictorContractError",
    "matern52",
    "route_blocks",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PredictorContractError(message)


def route_blocks(output: Mapping[str, Any], physical: np.ndarray, input_names: Sequence[str]) -> list[str]:
    """Per input row: the model id serving this output (dispatch on a physical feature, else the fixed model)."""

    dispatch = output.get("dispatch")
    if not dispatch:
        return [str(output["model"])] * physical.shape[0]
    column = list(input_names).index(str(dispatch["feature"]))
    routes = {str(key): str(model_id) for key, model_id in dispatch["models"].items()}
    default = str(dispatch["default"])
    result = []
    for value in physical[:, column]:
        key = str(int(round(float(value))))
        result.append(routes.get(key, default))
    return result


class Predictor:
    """Compiled v2 ``predictor.json``: derived-feature inputs -> probabilities with intervals."""

    def __init__(self, contract: Mapping[str, Any]) -> None:
        _require(contract.get("schema_version") == PREDICTOR_SCHEMA, "unsupported predictor schema")
        _require(contract.get("classification") == CLASSIFICATION, "predictor classification is not the surrogate label")
        _require(contract.get("source_dataset_classification") == SOURCE_CLASSIFICATION, "source dataset label missing")
        boundary = contract.get("claim_boundary")
        _require(
            isinstance(boundary, Mapping)
            and boundary.get("surrogate_of_screening_dataset") is True
            and boundary.get("not_physical_orbit_evidence") is True
            and boundary.get("not_performance_model") is True,
            "claim boundary flags missing",
        )
        _require(contract.get("mdo_v2_input_status") in {USABLE_LABEL, NOT_USABLE_LABEL}, "mdo_v2_input_status missing")
        self.contract = contract
        inputs = contract["inputs"]
        self.input_names = tuple(inputs["names"])
        _require(inputs.get("derived_not_fitted") is True, "inputs must be declared derived, not fitted")
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
            dispatch = output.get("dispatch")
            if dispatch:
                _require(dispatch["feature"] in self.input_names, f"{output['name']}: dispatch feature unknown")
                _require(dispatch["default"] in self.models, f"{output['name']}: dispatch default unknown")
                for model_id in dispatch["models"].values():
                    _require(model_id in self.models, f"{output['name']}: dispatch target {model_id} unknown")
                    _require(0 <= int(output["task"]) < self.models[model_id].task_covariance.shape[0], f"{output['name']}: dispatch task out of range")
        calibration = contract["calibration"]
        self.variance_scale = float(calibration["variance_scale"])
        self.nominal_probability = float(calibration["nominal_probability"])
        _require(self.variance_scale > 0.0 and 0.5 < self.nominal_probability < 1.0, "calibration values are invalid")
        self.z = NormalDist().inv_cdf(0.5 + self.nominal_probability / 2.0)
        self.interpolation_scope = dict(contract["interpolation_scope"])

    def _physical(self, physical: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(physical, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        _require(x.ndim == 2 and x.shape[1] == len(self.input_names), "input width differs from the contract")
        _require(bool(np.all(np.isfinite(x))), "inputs must be finite")
        return x

    def normalize(self, physical: Sequence[Sequence[float]]) -> np.ndarray:
        return (self._physical(physical) - self.minimum) / self.span

    def scope_flags(self, physical: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
        """Interpolation-scope diagnostics per input row (unit-box excess and realised chamber length)."""

        x = self._physical(physical)
        normalized = (x - self.minimum) / self.span
        names = self.input_names
        length_max = float(self.interpolation_scope["chamber_length_max_m"])
        has_length_inputs = "stage_count" in names and "stage_pitch_m" in names
        flags = []
        for row_physical, row in zip(x, normalized, strict=True):
            excess = float(np.sqrt(np.sum(np.where(row < 0.0, row, np.where(row > 1.0, row - 1.0, 0.0)) ** 2)))
            values = dict(zip(names, row_physical.tolist(), strict=True))
            chamber_length: float | None = None
            within_length: bool | None = None
            if has_length_inputs:
                chamber_length = float(round(values["stage_count"])) * values["stage_pitch_m"]
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

        x = self._physical(physical)
        normalized = (x - self.minimum) / self.span
        result = {}
        for output in self.outputs:
            routes = route_blocks(output, x, self.input_names)
            mean = np.empty(x.shape[0])
            variance = np.empty(x.shape[0])
            for model_id in sorted(set(routes)):
                index = np.asarray([i for i, route in enumerate(routes) if route == model_id], dtype=int)
                block_mean, block_variance = self.models[model_id].latent(normalized[index], int(output["task"]))
                mean[index] = block_mean
                variance[index] = block_variance
            result[output["name"]] = {"mean": mean, "variance": variance, "routes": routes}
        return result

    def predict(self, physical: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
        """Per input row: per output probability, calibrated latent and observation intervals."""

        working = self.predict_working(physical)
        rows = self._physical(physical).shape[0]
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
                    "model": working[name]["routes"][index],
                }
            predictions.append(
                {
                    "classification": CLASSIFICATION,
                    "source_dataset_classification": SOURCE_CLASSIFICATION,
                    "mdo_v2_input_status": self.contract["mdo_v2_input_status"],
                    "nominal_probability": self.nominal_probability,
                    "scope": flags[index],
                    "outputs": outputs,
                }
            )
        return predictions


def load_predictor(contract: Mapping[str, Any]) -> Predictor:
    return Predictor(contract)
