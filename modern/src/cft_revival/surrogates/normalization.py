"""Finite-data validation and reversible surrogate normalizations."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Sequence


class SurrogateError(Exception):
    """Base class for all typed surrogate-runtime failures."""


class SurrogateValidationError(SurrogateError, ValueError):
    """Training or prediction data violates a surrogate invariant."""


def finite_matrix(
    values: Sequence[Sequence[float]], name: str, *, width: int | None = None
) -> tuple[tuple[float, ...], ...]:
    try:
        matrix = tuple(
            tuple(0.0 if float(item) == 0.0 else float(item) for item in row)
            for row in values
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SurrogateValidationError(f"{name} must be numeric") from error
    if not matrix:
        raise SurrogateValidationError(f"{name} must contain at least one row")
    expected = len(matrix[0]) if width is None else width
    if expected < 1 or any(len(row) != expected for row in matrix):
        raise SurrogateValidationError(f"{name} must be a non-ragged matrix")
    if any(not isfinite(item) for row in matrix for item in row):
        raise SurrogateValidationError(f"{name} must contain only finite values")
    return matrix


def finite_vector(
    values: Sequence[float], name: str, *, length: int | None = None
) -> tuple[float, ...]:
    try:
        vector = tuple(
            0.0 if float(item) == 0.0 else float(item) for item in values
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SurrogateValidationError(f"{name} must be numeric") from error
    if not vector or (length is not None and len(vector) != length):
        raise SurrogateValidationError(f"{name} has an invalid length")
    if any(not isfinite(item) for item in vector):
        raise SurrogateValidationError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True, slots=True)
class InputNormalizer:
    minimum: tuple[float, ...]
    span: tuple[float, ...]
    constant: tuple[bool, ...]

    @classmethod
    def fit(cls, values: Sequence[Sequence[float]]) -> InputNormalizer:
        matrix = finite_matrix(values, "inputs")
        columns = tuple(zip(*matrix, strict=True))
        minimum = tuple(min(column) for column in columns)
        maximum = tuple(max(column) for column in columns)
        constant = tuple(high == low for low, high in zip(minimum, maximum, strict=True))
        span_values = []
        for low, high, fixed in zip(minimum, maximum, constant, strict=True):
            difference = 1.0 if fixed else high - low
            if not isfinite(difference) or difference <= 0.0:
                raise SurrogateValidationError(
                    "input range is not representable as a finite affine span"
                )
            span_values.append(difference)
        span = tuple(span_values)
        return cls(minimum, span, constant)

    @property
    def dimensions(self) -> int:
        return len(self.minimum)

    def transform(
        self, values: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        matrix = finite_matrix(values, "inputs", width=self.dimensions)
        transformed = []
        for row in matrix:
            normalized_row = []
            for value, low, span in zip(
                row, self.minimum, self.span, strict=True
            ):
                try:
                    normalized = (value - low) / span
                except (ArithmeticError, OverflowError) as error:
                    raise SurrogateValidationError(
                        "input normalization overflowed"
                    ) from error
                if not isfinite(normalized):
                    raise SurrogateValidationError(
                        "input normalization produced a nonfinite value"
                    )
                normalized_row.append(0.0 if normalized == 0.0 else normalized)
            transformed.append(tuple(normalized_row))
        return tuple(transformed)

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum": list(self.minimum),
            "span": list(self.span),
            "constant": list(self.constant),
        }


@dataclass(frozen=True, slots=True)
class OutputNormalizer:
    mean: float
    scale: float
    constant: bool

    @classmethod
    def fit(cls, values: Sequence[float]) -> OutputNormalizer:
        vector = finite_vector(values, "outputs")
        try:
            mean = fsum(value / len(vector) for value in vector)
        except OverflowError as error:
            raise SurrogateValidationError("output mean overflowed") from error
        deviations = tuple(value - mean for value in vector)
        maximum = max(abs(value) for value in deviations)
        try:
            scale = (
                0.0
                if maximum == 0.0
                else maximum
                * sqrt(
                    fsum((value / maximum) ** 2 for value in deviations)
                    / len(vector)
                )
            )
        except (ArithmeticError, OverflowError) as error:
            raise SurrogateValidationError("output scaling overflowed") from error
        if not isfinite(mean) or not isfinite(scale):
            raise SurrogateValidationError("output normalization is nonfinite")
        constant = scale <= max(abs(mean), 1.0) * 1e-14
        return cls(mean, 1.0 if constant else scale, constant)

    def transform(self, values: Sequence[float]) -> tuple[float, ...]:
        vector = finite_vector(values, "outputs")
        return tuple((value - self.mean) / self.scale for value in vector)

    def inverse_mean(self, value: float) -> float:
        result = self.mean + self.scale * value
        if not isfinite(result):
            raise SurrogateValidationError("predicted mean overflowed")
        return result

    def inverse_variance(self, value: float) -> float:
        if not isfinite(value) or value < 0.0:
            raise SurrogateValidationError("normalized predictive variance is invalid")
        try:
            result = (self.scale * sqrt(value)) ** 2
        except (ArithmeticError, OverflowError) as error:
            raise SurrogateValidationError("predicted variance overflowed") from error
        if not isfinite(result):
            raise SurrogateValidationError("predicted variance overflowed")
        return result

    def to_dict(self) -> dict[str, object]:
        return {"mean": self.mean, "scale": self.scale, "constant": self.constant}
