"""Shared validation and deterministic serialization for magnetics contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite, sqrt
from typing import Protocol


class MagneticsError(Exception):
    """Base error for the independent magnetics workstream."""


class MagneticsValidationError(MagneticsError, ValueError):
    """A magnetic model or contract violates its documented domain."""


def finite_float(name: str, value: float) -> float:
    """Return ``value`` as a finite float or raise a typed validation error."""

    if isinstance(value, bool):
        raise MagneticsValidationError(f"{name} must be a finite real number")
    converted = float(value)
    if not isfinite(converted):
        raise MagneticsValidationError(f"{name} must be finite")
    return 0.0 if converted == 0.0 else converted


def nonempty_identifier(name: str, value: str) -> str:
    """Validate stable human/machine identifiers."""

    if not isinstance(value, str) or not value.strip():
        raise MagneticsValidationError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class VectorRZ:
    """Axisymmetric meridional vector with explicit component units in context."""

    radial: float
    axial: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "radial", finite_float("radial", self.radial))
        object.__setattr__(self, "axial", finite_float("axial", self.axial))

    @property
    def magnitude(self) -> float:
        scale = max(abs(self.radial), abs(self.axial))
        if scale == 0.0:
            return 0.0
        normalized_radial = self.radial / scale
        normalized_axial = self.axial / scale
        return finite_float(
            "vector magnitude",
            scale * sqrt(
                normalized_radial * normalized_radial
                + normalized_axial * normalized_axial
            ),
        )

    def dot(self, other: VectorRZ) -> float:
        return finite_float(
            "vector dot product",
            self.radial * other.radial + self.axial * other.axial,
        )

    def normalized(self) -> VectorRZ:
        scale = max(abs(self.radial), abs(self.axial))
        if scale == 0.0:
            raise MagneticsValidationError("a zero vector has no direction")
        radial = self.radial / scale
        axial = self.axial / scale
        local_magnitude = sqrt(radial * radial + axial * axial)
        return VectorRZ(radial / local_magnitude, axial / local_magnitude)

    def scaled(self, scale: float) -> VectorRZ:
        finite_scale = finite_float("scale", scale)
        return VectorRZ(self.radial * finite_scale, self.axial * finite_scale)

    def to_dict(self) -> dict[str, object]:
        return {"radial": self.radial, "axial": self.axial}


class Serializable(Protocol):
    """Structural protocol for canonical JSON publication."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping."""


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise MagneticsValidationError("canonical JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return finite_float("serialized float", value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict"):
        method = getattr(value, "to_dict")
        return _json_value(method())
    raise MagneticsValidationError(
        f"value of type {type(value).__name__} is not deterministically serializable"
    )


def canonical_json(value: Serializable | dict[str, object]) -> str:
    """Serialize with sorted keys, stable separators, UTF-8 characters and no NaN."""

    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_sha256(value: Serializable | dict[str, object]) -> str:
    """Return a cross-process SHA-256 digest of canonical UTF-8 content."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()
