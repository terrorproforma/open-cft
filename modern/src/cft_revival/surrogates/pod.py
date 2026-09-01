"""Deterministic POD field reduction for future fixed-mesh surrogates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Mapping, Sequence

from ._linalg import dot, snapshot_pod
from .gp import IndependentMultiOutputGP
from .identity import canonical_hash, require_exact_keys, strict_json_loads
from .normalization import SurrogateValidationError, finite_matrix

POD_SCHEMA_VERSION = "cft-surrogate-pod-basis/1.0.0"


def fixed_mesh_hash(coordinates: Sequence[Sequence[float]]) -> str:
    """Hash finite coordinates after canonicalizing both signs of zero."""
    mesh = finite_matrix(coordinates, "mesh_coordinates")
    return canonical_hash({"coordinates": [list(row) for row in mesh]})


def _validate_mesh_hash(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SurrogateValidationError(
            "mesh_hash must be a SHA-256 hexadecimal digest"
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise SurrogateValidationError(
            "mesh_hash must be a SHA-256 hexadecimal digest"
        ) from error
    return value


@dataclass(frozen=True, slots=True)
class PODBasis:
    mean_field: tuple[float, ...]
    modes: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    mesh_hash: str
    retained_energy_fraction: float
    requested_rank: int
    representation: str

    def __post_init__(self) -> None:
        _validate_mesh_hash(self.mesh_hash)
        if not self.mean_field or any(
            not isfinite(value) for value in self.mean_field
        ):
            raise SurrogateValidationError("POD mean field is invalid")
        if (
            isinstance(self.requested_rank, bool)
            or self.requested_rank < 0
            or len(self.modes) != len(self.singular_values)
            or len(self.modes) > self.requested_rank
            or not isfinite(self.retained_energy_fraction)
            or not 0.0 <= self.retained_energy_fraction <= 1.0
        ):
            raise SurrogateValidationError("POD rank or energy metadata is invalid")
        if any(
            len(mode) != len(self.mean_field)
            or any(not isfinite(value) for value in mode)
            for mode in self.modes
        ) or any(
            not isfinite(value) or value <= 0.0
            for value in self.singular_values
        ):
            raise SurrogateValidationError("POD modes or singular values are invalid")
        for index, mode in enumerate(self.modes):
            if abs(dot(mode, mode) - 1.0) > 1e-8:
                raise SurrogateValidationError("POD mode is not normalized")
            if any(
                abs(dot(mode, other)) > 1e-8
                for other in self.modes[:index]
            ):
                raise SurrogateValidationError("POD modes are not orthogonal")
        if self.representation == "mean-only-rank-0":
            if self.modes or self.singular_values:
                raise SurrogateValidationError(
                    "mean-only POD cannot contain modal data"
                )
        elif self.representation == "modal":
            if not self.modes:
                raise SurrogateValidationError("modal POD requires at least one mode")
        else:
            raise SurrogateValidationError("unknown POD representation")

    @classmethod
    def fit(
        cls,
        fields: Sequence[Sequence[float]],
        *,
        rank: int,
        mesh_hash: str,
    ) -> PODBasis:
        matrix = finite_matrix(fields, "fields")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank <= min(len(matrix), len(matrix[0]))
        ):
            raise SurrogateValidationError(
                "POD rank is outside the snapshot dimensions"
            )
        _validate_mesh_hash(mesh_hash)
        mean = tuple(
            fsum(row[column] / len(matrix) for row in matrix)
            for column in range(len(matrix[0]))
        )
        if any(not isfinite(value) for value in mean):
            raise SurrogateValidationError("POD mean field is nonfinite")
        centered = tuple(
            tuple(value - average for value, average in zip(row, mean, strict=True))
            for row in matrix
        )
        total = fsum(value * value for row in centered for value in row)
        if not isfinite(total):
            raise SurrogateValidationError("POD snapshot energy overflowed")
        if rank == 0 or total == 0.0:
            return cls(
                mean,
                (),
                (),
                mesh_hash,
                1.0,
                rank,
                "mean-only-rank-0",
            )
        modes_raw, singular_raw = snapshot_pod(centered, rank)
        threshold = total**0.5 * 1e-14
        retained_pairs = [
            (tuple(mode), float(value))
            for mode, value in zip(modes_raw, singular_raw, strict=True)
            if isfinite(value) and value > threshold
        ]
        if not retained_pairs:
            return cls(
                mean,
                (),
                (),
                mesh_hash,
                1.0,
                rank,
                "mean-only-rank-0",
            )
        modes = tuple(pair[0] for pair in retained_pairs)
        singular_values = tuple(pair[1] for pair in retained_pairs)
        retained = fsum(value * value for value in singular_values)
        energy = min(1.0, retained / total)
        return cls(
            mean,
            modes,
            singular_values,
            mesh_hash,
            energy,
            rank,
            "modal",
        )

    @property
    def effective_rank(self) -> int:
        return len(self.modes)

    def project(
        self, fields: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        matrix = finite_matrix(fields, "fields", width=len(self.mean_field))
        return tuple(
            tuple(
                dot(
                    tuple(
                        value - average
                        for value, average in zip(
                            row, self.mean_field, strict=True
                        )
                    ),
                    mode,
                )
                for mode in self.modes
            )
            for row in matrix
        )

    def reconstruct(self, coefficients: Sequence[float]) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in coefficients)
        except (TypeError, ValueError, OverflowError) as error:
            raise SurrogateValidationError("POD coefficients must be numeric") from error
        if len(values) != len(self.modes) or any(
            not isfinite(value) for value in values
        ):
            raise SurrogateValidationError("POD coefficient width is invalid")
        reconstructed = tuple(
            average
            + fsum(
                coefficient * mode[index]
                for coefficient, mode in zip(values, self.modes, strict=True)
            )
            for index, average in enumerate(self.mean_field)
        )
        if any(not isfinite(value) for value in reconstructed):
            raise SurrogateValidationError("POD reconstruction is nonfinite")
        return reconstructed

    def _without_hash(self) -> dict[str, object]:
        return {
            "artifact_schema_version": POD_SCHEMA_VERSION,
            "model_type": "PODBasis",
            "mean_field": list(self.mean_field),
            "modes": [list(mode) for mode in self.modes],
            "singular_values": list(self.singular_values),
            "mesh_hash": self.mesh_hash,
            "retained_energy_fraction": self.retained_energy_fraction,
            "requested_rank": self.requested_rank,
            "effective_rank": self.effective_rank,
            "representation": self.representation,
            "policy": {
                "method": "snapshot-pod-v1",
                "zero_energy": "mean-only-rank-0",
                "mode_threshold_relative_to_snapshot_norm": 1e-14,
                "mesh_coordinate_tolerance": 0.0,
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
    def from_dict(cls, payload: Mapping[str, object]) -> PODBasis:
        expected = {
            "artifact_schema_version",
            "model_type",
            "mean_field",
            "modes",
            "singular_values",
            "mesh_hash",
            "retained_energy_fraction",
            "requested_rank",
            "effective_rank",
            "representation",
            "policy",
            "model_hash",
        }
        require_exact_keys(payload, expected, "POD basis artifact")
        without_hash = {
            key: value for key, value in payload.items() if key != "model_hash"
        }
        if payload["model_hash"] != canonical_hash(without_hash):
            raise SurrogateValidationError("POD basis model hash mismatch")
        if (
            payload["artifact_schema_version"] != POD_SCHEMA_VERSION
            or payload["model_type"] != "PODBasis"
            or payload["policy"]
            != {
                "method": "snapshot-pod-v1",
                "zero_energy": "mean-only-rank-0",
                "mode_threshold_relative_to_snapshot_norm": 1e-14,
                "mesh_coordinate_tolerance": 0.0,
            }
        ):
            raise SurrogateValidationError("unsupported POD basis policy")
        try:
            basis = cls(
                tuple(float(value) for value in payload["mean_field"]),  # type: ignore[arg-type]
                tuple(
                    tuple(float(value) for value in mode)
                    for mode in payload["modes"]  # type: ignore[union-attr]
                ),
                tuple(
                    float(value)
                    for value in payload["singular_values"]  # type: ignore[union-attr]
                ),
                _validate_mesh_hash(str(payload["mesh_hash"])),
                float(payload["retained_energy_fraction"]),
                int(payload["requested_rank"]),
                str(payload["representation"]),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise SurrogateValidationError("invalid POD basis artifact") from error
        if (
            basis.effective_rank != payload["effective_rank"]
            or basis._without_hash() != without_hash
            or any(
                not isfinite(value)
                for value in (
                    *basis.mean_field,
                    *basis.singular_values,
                    basis.retained_energy_fraction,
                )
            )
        ):
            raise SurrogateValidationError(
                "POD basis differs from deterministic artifact"
            )
        return basis

    @classmethod
    def loads(cls, payload: str) -> PODBasis:
        decoded = strict_json_loads(payload)
        if not isinstance(decoded, Mapping):
            raise SurrogateValidationError("POD basis artifact must be an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class FieldPrediction:
    mean_field: tuple[float, ...]
    pointwise_variance: tuple[float, ...]
    mesh_hash: str
    nominal_probability: float
    uncertainty_semantics: str

    def __post_init__(self) -> None:
        if (
            not self.mean_field
            or len(self.mean_field) != len(self.pointwise_variance)
            or any(not isfinite(value) for value in self.mean_field)
            or any(
                not isfinite(value) or value < 0.0
                for value in self.pointwise_variance
            )
            or not isfinite(self.nominal_probability)
            or not 0.5 < self.nominal_probability < 1.0
            or not self.uncertainty_semantics
        ):
            raise SurrogateValidationError("field prediction is invalid")


class PODFieldSurrogate:
    """POD coefficients modelled by exact GPs on one fixed mesh."""

    def __init__(
        self,
        basis: PODBasis,
        coefficient_model: IndependentMultiOutputGP | None,
        input_dimensions: int,
        nominal_probability: float,
    ) -> None:
        self.basis = basis
        self.coefficient_model = coefficient_model
        self.input_dimensions = input_dimensions
        self.nominal_probability = nominal_probability

    @classmethod
    def fit(
        cls,
        train_x: Sequence[Sequence[float]],
        fields: Sequence[Sequence[float]],
        *,
        rank: int,
        mesh_hash: str,
        nominal_probability: float = 0.95,
    ) -> PODFieldSurrogate:
        inputs = finite_matrix(train_x, "train_x")
        field_matrix = finite_matrix(fields, "fields")
        if len(inputs) != len(field_matrix):
            raise SurrogateValidationError(
                "POD input and field row counts must match"
            )
        basis = PODBasis.fit(field_matrix, rank=rank, mesh_hash=mesh_hash)
        if basis.effective_rank == 0:
            return cls(basis, None, len(inputs[0]), nominal_probability)
        coefficients = basis.project(field_matrix)
        model = IndependentMultiOutputGP.fit(
            inputs,
            coefficients,
            nominal_probability=nominal_probability,
        )
        return cls(basis, model, len(inputs[0]), nominal_probability)

    def predict(
        self,
        points: Sequence[Sequence[float]],
        *,
        mesh_hash: str,
    ) -> tuple[FieldPrediction, ...]:
        if mesh_hash != self.basis.mesh_hash:
            raise SurrogateValidationError(
                "field prediction mesh hash differs from the fitted fixed mesh"
            )
        inputs = finite_matrix(points, "points", width=self.input_dimensions)
        if self.coefficient_model is None:
            return tuple(
                FieldPrediction(
                    self.basis.mean_field,
                    (0.0,) * len(self.basis.mean_field),
                    self.basis.mesh_hash,
                    self.nominal_probability,
                    "mean-only-zero-energy",
                )
                for _ in inputs
            )
        coefficient_predictions = self.coefficient_model.predict(inputs)
        results = []
        for row in coefficient_predictions:
            means = tuple(prediction.mean for prediction in row)
            try:
                field_variance = tuple(
                    fsum(
                        prediction.variance * mode[index] ** 2
                        for prediction, mode in zip(
                            row, self.basis.modes, strict=True
                        )
                    )
                    for index in range(len(self.basis.mean_field))
                )
            except (ArithmeticError, OverflowError) as error:
                raise SurrogateValidationError(
                    "POD pointwise variance overflowed"
                ) from error
            if any(
                not isfinite(value) or value < 0.0 for value in field_variance
            ):
                raise SurrogateValidationError(
                    "POD pointwise variance is invalid"
                )
            results.append(
                FieldPrediction(
                    self.basis.reconstruct(means),
                    field_variance,
                    self.basis.mesh_hash,
                    self.nominal_probability,
                    "independent-modal-posterior",
                )
            )
        return tuple(results)
