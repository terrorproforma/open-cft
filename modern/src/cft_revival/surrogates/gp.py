"""Tamper-evident dependency-light exact Gaussian-process regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp, fsum, hypot, isfinite, log, pi, sqrt
from pathlib import Path
from statistics import NormalDist, median
from typing import TYPE_CHECKING, Mapping, Sequence

from ._linalg import cholesky, dot, solve_cholesky, solve_lower
from .identity import (
    canonical_hash,
    require_exact_keys,
    strict_json_loads,
)
from .normalization import (
    InputNormalizer,
    OutputNormalizer,
    SurrogateValidationError,
    finite_matrix,
    finite_vector,
)

if TYPE_CHECKING:
    from .validation import OODDetector

MODEL_SCHEMA_VERSION = "cft-surrogate-exact-gp/2.0.0"
KERNEL_FAMILY = "Matern"
KERNEL_VERSION = "matern-5/2-v1"
DEFAULT_JITTER_POLICY = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
LENGTH_SCALE_BOUNDS = (0.03, 4.0)
LENGTH_CANDIDATE_FACTORS = (0.35, 0.7, 1.4)
SIGNAL_VARIANCE_CANDIDATES = (0.5, 1.0, 2.0)
OUTPUT_SEMANTICS = {
    "mean": "posterior latent mean in physical output units",
    "variance": "calibrated posterior latent variance in squared physical output units",
    "observation_noise": (
        "optional prediction-time addition of mean fitted observation variance"
    ),
}
OOD_POLICY_VERSION = "unit-box-nearest-euclidean-v1"


def _probability(value: float, name: str) -> float:
    converted = float(value)
    if not isfinite(converted) or not 0.5 < converted < 1.0:
        raise SurrogateValidationError(f"{name} must be finite and lie in (0.5, 1)")
    return converted


@dataclass(frozen=True, slots=True)
class SurrogateSchema:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_units: tuple[str, ...] = ()
    output_units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (*self.input_names, *self.output_names):
            if not isinstance(name, str) or not name:
                raise SurrogateValidationError("schema names must be non-empty strings")
        if len(set(self.input_names)) != len(self.input_names):
            raise SurrogateValidationError("input schema names must be unique")
        if len(set(self.output_names)) != len(self.output_names):
            raise SurrogateValidationError("output schema names must be unique")
        if self.input_units and len(self.input_units) != len(self.input_names):
            raise SurrogateValidationError("input units do not match input names")
        if self.output_units and len(self.output_units) != len(self.output_names):
            raise SurrogateValidationError("output units do not match output names")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "input_units": list(self.input_units),
            "output_units": list(self.output_units),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SurrogateSchema:
        require_exact_keys(
            value,
            {"input_names", "output_names", "input_units", "output_units"},
            "surrogate schema",
        )
        fields = []
        for name in ("input_names", "output_names", "input_units", "output_units"):
            raw = value[name]
            if (
                not isinstance(raw, list)
                or any(not isinstance(item, str) for item in raw)
            ):
                raise SurrogateValidationError(
                    f"surrogate schema {name} must be a string list"
                )
            fields.append(tuple(raw))
        return cls(*fields)


@dataclass(frozen=True, slots=True)
class Prediction:
    mean: float
    variance: float
    nominal_probability: float = 0.95
    uncertainty_semantics: str = "posterior-latent"

    def __post_init__(self) -> None:
        if not isfinite(self.mean):
            raise SurrogateValidationError("prediction mean must be finite")
        if not isfinite(self.variance) or self.variance < 0.0:
            raise SurrogateValidationError(
                "prediction variance must be finite and non-negative"
            )
        _probability(self.nominal_probability, "prediction nominal_probability")
        if not isinstance(self.uncertainty_semantics, str) or not self.uncertainty_semantics:
            raise SurrogateValidationError(
                "prediction uncertainty semantics must be non-empty"
            )

    @property
    def standard_deviation(self) -> float:
        return sqrt(self.variance)

    def interval(
        self, probability: float | None = None
    ) -> tuple[float, float]:
        selected = self.nominal_probability if probability is None else _probability(
            probability, "interval probability"
        )
        z_score = NormalDist().inv_cdf(0.5 + selected / 2.0)
        try:
            radius = z_score * self.standard_deviation
            lower = self.mean - radius
            upper = self.mean + radius
        except (ArithmeticError, OverflowError) as error:
            raise SurrogateValidationError("predictive interval overflowed") from error
        if not isfinite(radius) or not isfinite(lower) or not isfinite(upper):
            raise SurrogateValidationError("predictive interval is nonfinite")
        if lower > upper:
            raise SurrogateValidationError("predictive interval is reversed")
        return lower, upper


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    length_scales: tuple[float, ...]
    length_scale_mode: str
    signal_variance: float
    log_marginal_likelihood: float
    jitter: float
    heteroskedastic_noise: bool
    training_rows: int

    @property
    def length_scale(self) -> float:
        """Compatibility scalar for explicitly isotropic fits."""
        if self.length_scale_mode != "isotropic":
            raise SurrogateValidationError("ARD diagnostics have no scalar length scale")
        return self.length_scales[0]


def _matern52(
    left: Sequence[float],
    right: Sequence[float],
    length_scales: Sequence[float],
) -> float:
    scaled_differences = []
    for a, b, length_scale in zip(left, right, length_scales, strict=True):
        difference = (a - b) / length_scale
        if not isfinite(difference):
            return 0.0
        scaled_differences.append(difference)
    radius = hypot(*scaled_differences)
    if not isfinite(radius):
        return 0.0
    scaled = sqrt(5.0) * radius
    if scaled > 750.0:
        return 0.0
    result = (1.0 + scaled + scaled * scaled / 3.0) * exp(-scaled)
    if not isfinite(result):
        raise SurrogateValidationError("kernel evaluation produced a nonfinite value")
    return result


class ExactGP:
    """Scalar Matérn-5/2 GP with deterministic isotropic or ARD fitting."""

    def __init__(
        self,
        *,
        train_x: tuple[tuple[float, ...], ...],
        train_y: tuple[float, ...],
        observation_variance: tuple[float, ...],
        schema: SurrogateSchema,
        input_normalizer: InputNormalizer,
        output_normalizer: OutputNormalizer,
        normalized_x: tuple[tuple[float, ...], ...],
        normalized_y: tuple[float, ...],
        normalized_noise: tuple[float, ...],
        length_scales: tuple[float, ...],
        length_scale_mode: str,
        signal_variance: float,
        lower: tuple[tuple[float, ...], ...],
        alpha: tuple[float, ...],
        diagnostics: FitDiagnostics,
        calibration_scale: float,
        nominal_probability: float,
        calibration_source: str,
        ood_threshold_multiplier: float,
        ood_quantile: float,
    ) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.observation_variance = observation_variance
        self.schema = schema
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self._normalized_x = normalized_x
        self._normalized_y = normalized_y
        self._normalized_noise = normalized_noise
        self._length_scales = length_scales
        self._length_scale_mode = length_scale_mode
        self._signal_variance = signal_variance
        self._lower = lower
        self._alpha = alpha
        self.diagnostics = diagnostics
        self._calibration_scale = calibration_scale
        self._nominal_probability = nominal_probability
        self._calibration_source = calibration_source
        self._ood_threshold_multiplier = ood_threshold_multiplier
        self._ood_quantile = ood_quantile

    @property
    def calibration_scale(self) -> float:
        return self._calibration_scale

    @property
    def nominal_probability(self) -> float:
        return self._nominal_probability

    @classmethod
    def fit(
        cls,
        train_x: Sequence[Sequence[float]],
        train_y: Sequence[float],
        *,
        observation_variance: Sequence[float] | None = None,
        schema: SurrogateSchema | None = None,
        length_scale_mode: str = "ard",
        calibration_scale: float = 1.0,
        nominal_probability: float = 0.95,
        calibration_source: str = "uncalibrated",
        ood_threshold_multiplier: float = 1.5,
        ood_quantile: float = 0.95,
    ) -> ExactGP:
        x = finite_matrix(train_x, "train_x")
        y = finite_vector(train_y, "train_y", length=len(x))
        if len(x) < 2:
            raise SurrogateValidationError("an exact GP requires at least two rows")
        if length_scale_mode not in {"ard", "isotropic"}:
            raise SurrogateValidationError(
                "length_scale_mode must be 'ard' or 'isotropic'"
            )
        if not isfinite(calibration_scale) or calibration_scale <= 0.0:
            raise SurrogateValidationError(
                "calibration_scale must be finite and positive"
            )
        nominal_probability = _probability(
            nominal_probability, "nominal_probability"
        )
        if not isinstance(calibration_source, str) or not calibration_source:
            raise SurrogateValidationError("calibration_source must be non-empty")
        if (
            not isfinite(ood_threshold_multiplier)
            or ood_threshold_multiplier <= 0.0
            or not isfinite(ood_quantile)
            or not 0.0 < ood_quantile < 1.0
        ):
            raise SurrogateValidationError("OOD policy values are invalid")
        if observation_variance is None:
            noise = (0.0,) * len(x)
            heteroskedastic = False
        else:
            noise = finite_vector(
                observation_variance, "observation_variance", length=len(x)
            )
            if any(value < 0.0 for value in noise):
                raise SurrogateValidationError(
                    "observation variances must be non-negative"
                )
            heteroskedastic = len(set(noise)) > 1
        model_schema = schema or SurrogateSchema(
            tuple(f"x{index}" for index in range(len(x[0]))), ("y",)
        )
        if (
            len(model_schema.input_names) != len(x[0])
            or len(model_schema.output_names) != 1
        ):
            raise SurrogateValidationError(
                "schema dimensions do not match scalar GP data"
            )
        x_normalizer = InputNormalizer.fit(x)
        y_normalizer = OutputNormalizer.fit(y)
        normalized_x = x_normalizer.transform(x)
        normalized_y = y_normalizer.transform(y)
        normalized_noise = tuple(
            value / (y_normalizer.scale * y_normalizer.scale) for value in noise
        )
        if any(not isfinite(value) for value in normalized_noise):
            raise SurrogateValidationError("normalized observation noise is nonfinite")
        best = cls._fit_hyperparameters(
            normalized_x,
            normalized_y,
            normalized_noise,
            length_scale_mode,
        )
        likelihood, length_scales, signal, lower, alpha, jitter = best
        diagnostics = FitDiagnostics(
            length_scales,
            length_scale_mode,
            signal,
            likelihood,
            jitter,
            heteroskedastic,
            len(x),
        )
        return cls(
            train_x=x,
            train_y=y,
            observation_variance=noise,
            schema=model_schema,
            input_normalizer=x_normalizer,
            output_normalizer=y_normalizer,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
            normalized_noise=normalized_noise,
            length_scales=length_scales,
            length_scale_mode=length_scale_mode,
            signal_variance=signal,
            lower=lower,
            alpha=alpha,
            diagnostics=diagnostics,
            calibration_scale=calibration_scale,
            nominal_probability=nominal_probability,
            calibration_source=calibration_source,
            ood_threshold_multiplier=ood_threshold_multiplier,
            ood_quantile=ood_quantile,
        )

    @classmethod
    def _fit_hyperparameters(
        cls,
        x: tuple[tuple[float, ...], ...],
        y: tuple[float, ...],
        noise: tuple[float, ...],
        mode: str,
    ) -> tuple[
        float,
        tuple[float, ...],
        float,
        tuple[tuple[float, ...], ...],
        tuple[float, ...],
        float,
    ]:
        dimensions = len(x[0])
        dimension_bases = []
        for dimension in range(dimensions):
            differences = [
                abs(left[dimension] - right[dimension])
                for index, left in enumerate(x)
                for right in x[index + 1 :]
                if left[dimension] != right[dimension]
            ]
            dimension_bases.append(median(differences) if differences else 1.0)
        if mode == "isotropic":
            distances = [
                hypot(*(a - b for a, b in zip(left, right, strict=True)))
                for index, left in enumerate(x)
                for right in x[index + 1 :]
                if left != right
            ]
            base = median(distances) if distances else 1.0
            candidates = [
                (min(4.0, max(0.03, base * factor)),) * dimensions
                for factor in LENGTH_CANDIDATE_FACTORS
            ]
            return cls._best_candidate(x, y, noise, candidates)

        current = tuple(
            min(4.0, max(0.03, base * 0.7)) for base in dimension_bases
        )
        best = cls._best_candidate(x, y, noise, [current])
        for dimension, base in enumerate(dimension_bases):
            candidates = []
            for factor in LENGTH_CANDIDATE_FACTORS:
                candidate = list(best[1])
                candidate[dimension] = min(4.0, max(0.03, base * factor))
                candidates.append(tuple(candidate))
            candidate_best = cls._best_candidate(x, y, noise, candidates)
            if candidate_best[0] > best[0]:
                best = candidate_best
        return best

    @classmethod
    def _best_candidate(
        cls,
        x: tuple[tuple[float, ...], ...],
        y: tuple[float, ...],
        noise: tuple[float, ...],
        length_candidates: Sequence[tuple[float, ...]],
    ) -> tuple[
        float,
        tuple[float, ...],
        float,
        tuple[tuple[float, ...], ...],
        tuple[float, ...],
        float,
    ]:
        best = None
        for length_scales in length_candidates:
            if (
                len(length_scales) != len(x[0])
                or any(
                    not isfinite(value)
                    or not LENGTH_SCALE_BOUNDS[0] <= value <= LENGTH_SCALE_BOUNDS[1]
                    for value in length_scales
                )
            ):
                raise SurrogateValidationError("ARD length-scale vector is invalid")
            for signal in SIGNAL_VARIANCE_CANDIDATES:
                lower, alpha, jitter, likelihood = cls._factor(
                    x, y, noise, length_scales, signal
                )
                candidate = (
                    likelihood,
                    tuple(length_scales),
                    signal,
                    lower,
                    alpha,
                    jitter,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            raise SurrogateValidationError("no GP hyperparameter candidate succeeded")
        return best

    @staticmethod
    def _factor(
        x: tuple[tuple[float, ...], ...],
        y: tuple[float, ...],
        noise: tuple[float, ...],
        length_scales: tuple[float, ...],
        signal_variance: float,
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], float, float]:
        base = [
            [
                signal_variance * _matern52(left, right, length_scales)
                for right in x
            ]
            for left in x
        ]
        for jitter in DEFAULT_JITTER_POLICY:
            matrix = [row[:] for row in base]
            for index in range(len(matrix)):
                matrix[index][index] += noise[index] + jitter
            try:
                lower_list = cholesky(matrix)
                alpha_list = solve_cholesky(lower_list, y)
                likelihood = (
                    -0.5 * dot(y, alpha_list)
                    - fsum(
                        log(lower_list[index][index])
                        for index in range(len(y))
                    )
                    - 0.5 * len(y) * log(2.0 * pi)
                )
            except (ArithmeticError, OverflowError, ValueError):
                continue
            if not isfinite(likelihood) or any(
                not isfinite(value) for value in alpha_list
            ):
                continue
            return (
                tuple(tuple(row) for row in lower_list),
                tuple(alpha_list),
                jitter,
                likelihood,
            )
        raise SurrogateValidationError(
            "GP covariance remained invalid after the bounded jitter policy"
        )

    def predict(
        self,
        points: Sequence[Sequence[float]],
        *,
        include_observation_noise: bool = False,
    ) -> tuple[Prediction, ...]:
        normalized = self.input_normalizer.transform(points)
        try:
            average_noise = (
                fsum(self._normalized_noise) / len(self._normalized_noise)
                if include_observation_noise
                else 0.0
            )
        except OverflowError as error:
            raise SurrogateValidationError(
                "observation-noise prediction overflowed"
            ) from error
        predictions = []
        for point in normalized:
            covariance = tuple(
                self._signal_variance
                * _matern52(point, training, self._length_scales)
                for training in self._normalized_x
            )
            try:
                normalized_mean = dot(covariance, self._alpha)
                projection = solve_lower(self._lower, covariance)
                reduction = dot(projection, projection)
                normalized_variance = max(
                    self._signal_variance - reduction + average_noise,
                    0.0,
                )
                mean = self.output_normalizer.inverse_mean(normalized_mean)
                variance = (
                    self.output_normalizer.inverse_variance(normalized_variance)
                    * self._calibration_scale
                )
            except (ArithmeticError, OverflowError, ValueError) as error:
                raise SurrogateValidationError(
                    "prediction arithmetic was not finite"
                ) from error
            if not isfinite(variance):
                raise SurrogateValidationError("predicted variance overflowed")
            semantics = (
                "posterior-latent-plus-average-observation-noise"
                if include_observation_noise
                else "posterior-latent"
            )
            predictions.append(
                Prediction(
                    mean,
                    variance,
                    self._nominal_probability,
                    semantics,
                )
            )
        return tuple(predictions)

    def ood_detector(self) -> OODDetector:
        """Build the detector governed by this model's hashed OOD policy."""
        from .validation import OODDetector

        return OODDetector.fit(
            self.train_x,
            threshold_multiplier=self._ood_threshold_multiplier,
            threshold_quantile=self._ood_quantile,
        )

    @property
    def schema_hash(self) -> str:
        return canonical_hash(self.schema.to_dict())

    @property
    def training_data_hash(self) -> str:
        return canonical_hash(
            {
                "raw": {
                    "train_x": [list(row) for row in self.train_x],
                    "train_y": list(self.train_y),
                    "observation_variance": list(self.observation_variance),
                },
                "normalized": {
                    "train_x": [list(row) for row in self._normalized_x],
                    "train_y": list(self._normalized_y),
                    "observation_variance": list(self._normalized_noise),
                },
                "input_normalizer": self.input_normalizer.to_dict(),
                "output_normalizer": self.output_normalizer.to_dict(),
            }
        )

    def _artifact_without_hash(self) -> dict[str, object]:
        return {
            "artifact_schema_version": MODEL_SCHEMA_VERSION,
            "model_type": "ExactGP",
            "schema": self.schema.to_dict(),
            "schema_hash": self.schema_hash,
            "training_data": {
                "raw": {
                    "train_x": [list(row) for row in self.train_x],
                    "train_y": list(self.train_y),
                    "observation_variance": list(self.observation_variance),
                },
                "normalized": {
                    "train_x": [list(row) for row in self._normalized_x],
                    "train_y": list(self._normalized_y),
                    "observation_variance": list(self._normalized_noise),
                },
            },
            "normalization": {
                "policy_version": "affine-unit-box-and-population-standardize-v1",
                "input": self.input_normalizer.to_dict(),
                "output": self.output_normalizer.to_dict(),
            },
            "training_data_hash": self.training_data_hash,
            "executable_policy": {
                "kernel": {
                    "family": KERNEL_FAMILY,
                    "version": KERNEL_VERSION,
                    "nu": 2.5,
                },
                "hyperparameters": {
                    "length_scale_mode": self._length_scale_mode,
                    "length_scale_bounds": list(LENGTH_SCALE_BOUNDS),
                    "length_candidate_factors": list(LENGTH_CANDIDATE_FACTORS),
                    "signal_variance_candidates": list(
                        SIGNAL_VARIANCE_CANDIDATES
                    ),
                },
                "jitter_policy": list(DEFAULT_JITTER_POLICY),
                "calibration": {
                    "variance_scale": self._calibration_scale,
                    "nominal_probability": self._nominal_probability,
                    "source": self._calibration_source,
                },
                "ood": {
                    "policy_version": OOD_POLICY_VERSION,
                    "distance": "euclidean-in-fitted-unit-box",
                    "threshold_quantile": self._ood_quantile,
                    "threshold_multiplier": self._ood_threshold_multiplier,
                    "coordinate_tolerance": 0.0,
                },
                "output_semantics": OUTPUT_SEMANTICS,
            },
            "fitted_parameters": {
                "length_scales": list(self._length_scales),
                "signal_variance": self._signal_variance,
                "jitter": self.diagnostics.jitter,
                "log_marginal_likelihood": (
                    self.diagnostics.log_marginal_likelihood
                ),
            },
        }

    @property
    def model_hash(self) -> str:
        return canonical_hash(self._artifact_without_hash())

    def to_dict(self) -> dict[str, object]:
        artifact = self._artifact_without_hash()
        artifact["model_hash"] = self.model_hash
        return artifact

    def dumps(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExactGP:
        top_keys = {
            "artifact_schema_version",
            "model_type",
            "schema",
            "schema_hash",
            "training_data",
            "normalization",
            "training_data_hash",
            "executable_policy",
            "fitted_parameters",
            "model_hash",
        }
        require_exact_keys(payload, top_keys, "ExactGP artifact")
        if payload["artifact_schema_version"] != MODEL_SCHEMA_VERSION:
            raise SurrogateValidationError("unsupported ExactGP artifact version")
        if payload["model_type"] != "ExactGP":
            raise SurrogateValidationError("artifact model_type is not ExactGP")
        without_hash = {key: value for key, value in payload.items() if key != "model_hash"}
        if payload["model_hash"] != canonical_hash(without_hash):
            raise SurrogateValidationError("ExactGP model hash mismatch")

        schema_raw = payload["schema"]
        training_raw = payload["training_data"]
        policy_raw = payload["executable_policy"]
        if (
            not isinstance(schema_raw, Mapping)
            or not isinstance(training_raw, Mapping)
            or not isinstance(policy_raw, Mapping)
        ):
            raise SurrogateValidationError("ExactGP artifact sections must be objects")
        schema = SurrogateSchema.from_dict(schema_raw)
        if payload["schema_hash"] != canonical_hash(schema.to_dict()):
            raise SurrogateValidationError("ExactGP schema hash mismatch")
        require_exact_keys(training_raw, {"raw", "normalized"}, "training_data")
        raw = training_raw["raw"]
        if not isinstance(raw, Mapping):
            raise SurrogateValidationError("raw training data must be an object")
        require_exact_keys(
            raw,
            {"train_x", "train_y", "observation_variance"},
            "raw training data",
        )
        require_exact_keys(
            policy_raw,
            {
                "kernel",
                "hyperparameters",
                "jitter_policy",
                "calibration",
                "ood",
                "output_semantics",
            },
            "executable_policy",
        )
        kernel = policy_raw["kernel"]
        hyperparameters = policy_raw["hyperparameters"]
        calibration = policy_raw["calibration"]
        ood = policy_raw["ood"]
        if not all(
            isinstance(value, Mapping)
            for value in (kernel, hyperparameters, calibration, ood)
        ):
            raise SurrogateValidationError("ExactGP policy sections must be objects")
        require_exact_keys(kernel, {"family", "version", "nu"}, "kernel policy")
        require_exact_keys(
            hyperparameters,
            {
                "length_scale_mode",
                "length_scale_bounds",
                "length_candidate_factors",
                "signal_variance_candidates",
            },
            "hyperparameter policy",
        )
        require_exact_keys(
            calibration,
            {"variance_scale", "nominal_probability", "source"},
            "calibration policy",
        )
        require_exact_keys(
            ood,
            {
                "policy_version",
                "distance",
                "threshold_quantile",
                "threshold_multiplier",
                "coordinate_tolerance",
            },
            "OOD policy",
        )
        expected_constants = {
            "kernel": {
                "family": KERNEL_FAMILY,
                "version": KERNEL_VERSION,
                "nu": 2.5,
            },
            "jitter": list(DEFAULT_JITTER_POLICY),
            "bounds": list(LENGTH_SCALE_BOUNDS),
            "factors": list(LENGTH_CANDIDATE_FACTORS),
            "signals": list(SIGNAL_VARIANCE_CANDIDATES),
            "output_semantics": OUTPUT_SEMANTICS,
            "ood_version": OOD_POLICY_VERSION,
        }
        if (
            kernel != expected_constants["kernel"]
            or policy_raw["jitter_policy"] != expected_constants["jitter"]
            or hyperparameters["length_scale_bounds"] != expected_constants["bounds"]
            or hyperparameters["length_candidate_factors"] != expected_constants["factors"]
            or hyperparameters["signal_variance_candidates"] != expected_constants["signals"]
            or policy_raw["output_semantics"] != expected_constants["output_semantics"]
            or ood["policy_version"] != expected_constants["ood_version"]
            or ood["distance"] != "euclidean-in-fitted-unit-box"
            or ood["coordinate_tolerance"] != 0.0
        ):
            raise SurrogateValidationError(
                "serialized kernel or executable policy is unsupported"
            )
        try:
            model = cls.fit(
                raw["train_x"],  # type: ignore[arg-type]
                raw["train_y"],  # type: ignore[arg-type]
                observation_variance=raw["observation_variance"],  # type: ignore[arg-type]
                schema=schema,
                length_scale_mode=str(hyperparameters["length_scale_mode"]),
                calibration_scale=float(calibration["variance_scale"]),
                nominal_probability=float(calibration["nominal_probability"]),
                calibration_source=str(calibration["source"]),
                ood_threshold_multiplier=float(ood["threshold_multiplier"]),
                ood_quantile=float(ood["threshold_quantile"]),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise SurrogateValidationError("invalid ExactGP artifact values") from error
        if model._artifact_without_hash() != without_hash:
            raise SurrogateValidationError(
                "ExactGP artifact differs from deterministic reconstruction"
            )
        return model

    @classmethod
    def loads(cls, payload: str) -> ExactGP:
        decoded = strict_json_loads(payload)
        if not isinstance(decoded, Mapping):
            raise SurrogateValidationError("ExactGP artifact must be an object")
        return cls.from_dict(decoded)

    @classmethod
    def load(cls, path: str | Path) -> ExactGP:
        return cls.loads(Path(path).read_text(encoding="utf-8"))


class IndependentMultiOutputGP:
    """One exact GP per output with matrix-shaped probability-bearing results."""

    def __init__(self, models: Sequence[ExactGP]) -> None:
        self.models = tuple(models)
        if not self.models:
            raise SurrogateValidationError("at least one output model is required")
        if len({model.schema.input_names for model in self.models}) != 1:
            raise SurrogateValidationError("output models must share an input schema")

    @classmethod
    def fit(
        cls,
        train_x: Sequence[Sequence[float]],
        train_y: Sequence[Sequence[float]],
        *,
        observation_variance: Sequence[Sequence[float]] | None = None,
        schema: SurrogateSchema | None = None,
        length_scale_mode: str = "ard",
        nominal_probability: float = 0.95,
    ) -> IndependentMultiOutputGP:
        x = finite_matrix(train_x, "train_x")
        y = finite_matrix(train_y, "train_y")
        if len(y) != len(x):
            raise SurrogateValidationError("train_x and train_y row counts must match")
        output_count = len(y[0])
        model_schema = schema or SurrogateSchema(
            tuple(f"x{index}" for index in range(len(x[0]))),
            tuple(f"y{index}" for index in range(output_count)),
        )
        if len(model_schema.output_names) != output_count:
            raise SurrogateValidationError("schema output width does not match train_y")
        noise = (
            None
            if observation_variance is None
            else finite_matrix(
                observation_variance, "observation_variance", width=output_count
            )
        )
        if noise is not None and len(noise) != len(x):
            raise SurrogateValidationError(
                "observation noise row count must match train_x"
            )
        models = []
        for output in range(output_count):
            scalar_schema = SurrogateSchema(
                model_schema.input_names,
                (model_schema.output_names[output],),
                model_schema.input_units,
                (
                    (model_schema.output_units[output],)
                    if model_schema.output_units
                    else ()
                ),
            )
            models.append(
                ExactGP.fit(
                    x,
                    tuple(row[output] for row in y),
                    observation_variance=(
                        None
                        if noise is None
                        else tuple(row[output] for row in noise)
                    ),
                    schema=scalar_schema,
                    length_scale_mode=length_scale_mode,
                    nominal_probability=nominal_probability,
                )
            )
        return cls(models)

    def predict(
        self, points: Sequence[Sequence[float]]
    ) -> tuple[tuple[Prediction, ...], ...]:
        by_output = tuple(model.predict(points) for model in self.models)
        return tuple(
            tuple(column[row] for column in by_output)
            for row in range(len(by_output[0]))
        )
