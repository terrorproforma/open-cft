"""Tamper-evident autoregressive two-fidelity discrepancy models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Mapping, Sequence

from .gp import ExactGP, Prediction, SurrogateSchema
from .identity import canonical_hash, require_exact_keys, strict_json_loads
from .normalization import SurrogateValidationError, finite_matrix, finite_vector

AR1_SCHEMA_VERSION = "cft-surrogate-ar1/1.0.0"


@dataclass(frozen=True, slots=True)
class AR1Diagnostics:
    rho: float
    rho_bounds: tuple[float, float]
    paired_high_rows: int
    variance_assumption: str = "independent-low-and-discrepancy-posteriors"


class TwoFidelityAR1:
    """``high(x) = rho*low(x) + delta(x)`` with a separate exact GP."""

    def __init__(
        self,
        low_model: ExactGP,
        discrepancy_model: ExactGP,
        rho: float,
        rho_bounds: tuple[float, float],
    ) -> None:
        if (
            not isfinite(rho)
            or len(rho_bounds) != 2
            or not all(isfinite(value) for value in rho_bounds)
            or rho_bounds[0] >= rho_bounds[1]
            or not rho_bounds[0] <= rho <= rho_bounds[1]
        ):
            raise SurrogateValidationError("AR1 rho policy is invalid")
        if (
            low_model.schema.input_names
            != discrepancy_model.schema.input_names
        ):
            raise SurrogateValidationError(
                "AR1 component input schemas do not match"
            )
        self.low_model = low_model
        self.discrepancy_model = discrepancy_model
        self.rho = rho
        self.rho_bounds = rho_bounds
        self.diagnostics = AR1Diagnostics(
            rho,
            rho_bounds,
            len(discrepancy_model.train_x),
        )

    @classmethod
    def fit(
        cls,
        low_x: Sequence[Sequence[float]],
        low_y: Sequence[float],
        high_x: Sequence[Sequence[float]],
        high_y: Sequence[float],
        *,
        low_observation_variance: Sequence[float] | None = None,
        high_observation_variance: Sequence[float] | None = None,
        schema: SurrogateSchema | None = None,
        rho_bounds: tuple[float, float] = (-3.0, 3.0),
        length_scale_mode: str = "ard",
        nominal_probability: float = 0.95,
    ) -> TwoFidelityAR1:
        high_matrix = finite_matrix(high_x, "high_x")
        high_values = finite_vector(high_y, "high_y", length=len(high_matrix))
        if len(high_matrix) < 2:
            raise SurrogateValidationError(
                "AR1 discrepancy requires two high-fidelity rows"
            )
        try:
            lower, upper = (float(value) for value in rho_bounds)
        except (TypeError, ValueError, OverflowError) as error:
            raise SurrogateValidationError("rho bounds must be numeric") from error
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise SurrogateValidationError(
                "rho bounds must be finite and increasing"
            )
        model_schema = schema or SurrogateSchema(
            tuple(f"x{index}" for index in range(len(high_matrix[0]))),
            ("y",),
        )
        low_model = ExactGP.fit(
            low_x,
            low_y,
            observation_variance=low_observation_variance,
            schema=model_schema,
            length_scale_mode=length_scale_mode,
            nominal_probability=nominal_probability,
        )
        low_at_high = tuple(
            prediction.mean for prediction in low_model.predict(high_matrix)
        )
        try:
            denominator = fsum(value * value for value in low_at_high)
            numerator = fsum(
                low * high
                for low, high in zip(
                    low_at_high, high_values, strict=True
                )
            )
            raw_rho = numerator / denominator if denominator > 1e-30 else 1.0
        except (ArithmeticError, OverflowError) as error:
            raise SurrogateValidationError("AR1 rho fit overflowed") from error
        if not isfinite(raw_rho):
            raise SurrogateValidationError("AR1 rho fit is nonfinite")
        rho = min(upper, max(lower, raw_rho))
        residual = []
        for high, low in zip(high_values, low_at_high, strict=True):
            value = high - rho * low
            if not isfinite(value):
                raise SurrogateValidationError("AR1 discrepancy target overflowed")
            residual.append(value)
        discrepancy_schema = SurrogateSchema(
            model_schema.input_names,
            (f"{model_schema.output_names[0]}__discrepancy",),
            model_schema.input_units,
            model_schema.output_units,
        )
        discrepancy = ExactGP.fit(
            high_matrix,
            residual,
            observation_variance=high_observation_variance,
            schema=discrepancy_schema,
            length_scale_mode=length_scale_mode,
            nominal_probability=nominal_probability,
        )
        return cls(low_model, discrepancy, rho, (lower, upper))

    def predict(
        self,
        points: Sequence[Sequence[float]],
        *,
        fidelity: str = "high",
    ) -> tuple[Prediction, ...]:
        low = self.low_model.predict(points)
        if fidelity == "low":
            return low
        if fidelity != "high":
            raise SurrogateValidationError("fidelity must be 'low' or 'high'")
        discrepancy = self.discrepancy_model.predict(points)
        predictions = []
        for low_prediction, delta in zip(low, discrepancy, strict=True):
            try:
                mean = self.rho * low_prediction.mean + delta.mean
                variance = (
                    self.rho * self.rho * low_prediction.variance
                    + delta.variance
                )
            except (ArithmeticError, OverflowError) as error:
                raise SurrogateValidationError(
                    "AR1 prediction overflowed"
                ) from error
            if not isfinite(mean) or not isfinite(variance):
                raise SurrogateValidationError("AR1 prediction is nonfinite")
            predictions.append(
                Prediction(
                    mean,
                    variance,
                    low_prediction.nominal_probability,
                    "ar1-independent-low-plus-discrepancy",
                )
            )
        return tuple(predictions)

    def _without_hash(self) -> dict[str, object]:
        return {
            "artifact_schema_version": AR1_SCHEMA_VERSION,
            "model_type": "TwoFidelityAR1",
            "low_model": self.low_model.to_dict(),
            "discrepancy_model": self.discrepancy_model.to_dict(),
            "rho": self.rho,
            "rho_bounds": list(self.rho_bounds),
            "policy": {
                "equation": "high=rho*low+discrepancy",
                "rho_fit": "bounded-origin-least-squares-v1",
                "variance": "independent-low-and-discrepancy-posteriors",
                "output_semantics": "high-fidelity-emulator-not-physical-truth",
            },
        }

    @property
    def model_hash(self) -> str:
        return canonical_hash(self._without_hash())

    def to_dict(self) -> dict[str, object]:
        artifact = self._without_hash()
        artifact["model_hash"] = self.model_hash
        return artifact

    def dumps(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TwoFidelityAR1:
        expected = {
            "artifact_schema_version",
            "model_type",
            "low_model",
            "discrepancy_model",
            "rho",
            "rho_bounds",
            "policy",
            "model_hash",
        }
        require_exact_keys(payload, expected, "AR1 artifact")
        without_hash = {
            key: value for key, value in payload.items() if key != "model_hash"
        }
        if payload["model_hash"] != canonical_hash(without_hash):
            raise SurrogateValidationError("AR1 model hash mismatch")
        expected_policy = {
            "equation": "high=rho*low+discrepancy",
            "rho_fit": "bounded-origin-least-squares-v1",
            "variance": "independent-low-and-discrepancy-posteriors",
            "output_semantics": "high-fidelity-emulator-not-physical-truth",
        }
        if (
            payload["artifact_schema_version"] != AR1_SCHEMA_VERSION
            or payload["model_type"] != "TwoFidelityAR1"
            or payload["policy"] != expected_policy
            or not isinstance(payload["low_model"], Mapping)
            or not isinstance(payload["discrepancy_model"], Mapping)
            or not isinstance(payload["rho_bounds"], list)
        ):
            raise SurrogateValidationError("unsupported AR1 artifact policy")
        try:
            model = cls(
                ExactGP.from_dict(payload["low_model"]),
                ExactGP.from_dict(payload["discrepancy_model"]),
                float(payload["rho"]),
                tuple(float(value) for value in payload["rho_bounds"]),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise SurrogateValidationError("invalid AR1 artifact values") from error
        if model._without_hash() != without_hash:
            raise SurrogateValidationError(
                "AR1 artifact differs from deterministic reconstruction"
            )
        return model

    @classmethod
    def loads(cls, payload: str) -> TwoFidelityAR1:
        decoded = strict_json_loads(payload)
        if not isinstance(decoded, Mapping):
            raise SurrogateValidationError("AR1 artifact must be an object")
        return cls.from_dict(decoded)


class IndependentMultiOutputAR1:
    """Independent AR1 discrepancy models for matrix-valued outputs."""

    def __init__(self, models: Sequence[TwoFidelityAR1]) -> None:
        self.models = tuple(models)
        if not self.models:
            raise SurrogateValidationError(
                "at least one AR1 output model is required"
            )

    @classmethod
    def fit(
        cls,
        low_x: Sequence[Sequence[float]],
        low_y: Sequence[Sequence[float]],
        high_x: Sequence[Sequence[float]],
        high_y: Sequence[Sequence[float]],
        *,
        schema: SurrogateSchema | None = None,
        length_scale_mode: str = "ard",
        nominal_probability: float = 0.95,
    ) -> IndependentMultiOutputAR1:
        low_outputs = finite_matrix(low_y, "low_y")
        high_outputs = finite_matrix(
            high_y, "high_y", width=len(low_outputs[0])
        )
        if len(low_outputs) != len(low_x) or len(high_outputs) != len(high_x):
            raise SurrogateValidationError(
                "input and output row counts must match"
            )
        model_schema = schema or SurrogateSchema(
            tuple(f"x{index}" for index in range(len(low_x[0]))),
            tuple(f"y{index}" for index in range(len(low_outputs[0]))),
        )
        models = tuple(
            TwoFidelityAR1.fit(
                low_x,
                tuple(row[output] for row in low_outputs),
                high_x,
                tuple(row[output] for row in high_outputs),
                schema=SurrogateSchema(
                    model_schema.input_names,
                    (model_schema.output_names[output],),
                    model_schema.input_units,
                    (
                        (model_schema.output_units[output],)
                        if model_schema.output_units
                        else ()
                    ),
                ),
                length_scale_mode=length_scale_mode,
                nominal_probability=nominal_probability,
            )
            for output in range(len(low_outputs[0]))
        )
        return cls(models)

    def predict(
        self,
        points: Sequence[Sequence[float]],
        *,
        fidelity: str = "high",
    ) -> tuple[tuple[Prediction, ...], ...]:
        by_output = tuple(
            model.predict(points, fidelity=fidelity) for model in self.models
        )
        return tuple(
            tuple(column[row] for column in by_output)
            for row in range(len(by_output[0]))
        )
