"""Versioned checkpoint and provenance contracts for reproducible hybrid runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Sequence

from .models import (
    HybridValidationError,
    Particle,
    VelocityTimeLevel,
    XenonSpecies,
    finite_scalar,
    validated_particle_batch,
)
from .rng import RNG_ALGORITHM

CHECKPOINT_SCHEMA_VERSION = "hybrid-checkpoint-v1"
TIME_INTEGRATION_CONTRACT = "x^n,v^(n-1/2);E^n,B^n"
_TIME_LEVELS = {
    "position": "n",
    "velocity": "n_minus_one_half",
    "fields": "n",
}


class _DuplicateJSONKeyError(HybridValidationError):
    """A JSON object repeated a member name."""


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(
                f"duplicate JSON object key {key!r} is forbidden"
            )
        result[key] = value
    return result


def _reject_nonfinite_json_constant(token: str) -> None:
    raise HybridValidationError(
        f"nonfinite JSON numeric literal {token!r} is forbidden"
    )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Inputs needed to distinguish verification fixtures from sourced data."""

    model_scope: str
    created_utc: str
    code_revision: str | None
    cross_section_provenance: str
    notes: tuple[str, ...] = ()
    time_integration_contract: str = TIME_INTEGRATION_CONTRACT

    def __post_init__(self) -> None:
        for name in (
            "model_scope",
            "created_utc",
            "cross_section_provenance",
            "time_integration_contract",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise HybridValidationError(f"{name} must be a non-empty string")
        if (
            self.code_revision is not None
            and (
                not isinstance(self.code_revision, str)
                or not self.code_revision.strip()
            )
        ):
            raise HybridValidationError("code_revision must be null or non-empty")
        if not isinstance(self.notes, tuple):
            raise HybridValidationError("provenance notes must be a tuple")
        if any(not isinstance(note, str) or not note.strip() for note in self.notes):
            raise HybridValidationError("provenance notes must be non-empty strings")
        if self.time_integration_contract != TIME_INTEGRATION_CONTRACT:
            raise HybridValidationError(
                f"time_integration_contract must be {TIME_INTEGRATION_CONTRACT}"
            )
        if (
            "T" not in self.created_utc
            or not self.created_utc.endswith("Z")
        ):
            raise HybridValidationError(
                "created_utc must be an ISO-8601 UTC timestamp ending in Z"
            )
        try:
            parsed = datetime.fromisoformat(
                self.created_utc[:-1] + "+00:00"
            )
        except ValueError as error:
            raise HybridValidationError(
                "created_utc must be a valid ISO-8601 timestamp"
            ) from error
        if (
            parsed.tzinfo is None
            or parsed.utcoffset() != timedelta(0)
            or parsed.tzinfo != timezone.utc
        ):
            raise HybridValidationError("created_utc must represent UTC")


@dataclass(frozen=True, slots=True)
class HybridCheckpoint:
    step: int
    time_s: float
    dt_s: float
    rng_seed: int
    particles: tuple[Particle, ...]
    provenance: ProvenanceRecord
    rng_algorithm: str = RNG_ALGORITHM
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    velocity_staggering: VelocityTimeLevel = (
        VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
    )

    def __post_init__(self) -> None:
        if (
            type(self.step) is not int
            or self.step < 0
        ):
            raise HybridValidationError("step must be a non-negative integer")
        time = finite_scalar("time_s", self.time_s)
        dt = finite_scalar("dt_s", self.dt_s)
        if time < 0.0 or dt < 0.0:
            raise HybridValidationError("time_s and dt_s must be non-negative")
        if (
            type(self.rng_seed) is not int
            or not 0 <= self.rng_seed < 1 << 64
        ):
            raise HybridValidationError("rng_seed must be an unsigned 64-bit integer")
        if self.rng_algorithm != RNG_ALGORITHM:
            raise HybridValidationError(f"rng_algorithm must be {RNG_ALGORITHM}")
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise HybridValidationError(
                f"schema_version must be {CHECKPOINT_SCHEMA_VERSION}"
            )
        particles = validated_particle_batch(self.particles)
        if not isinstance(self.provenance, ProvenanceRecord):
            raise HybridValidationError("provenance must be a ProvenanceRecord")
        if (
            self.velocity_staggering
            is not VelocityTimeLevel.LEAPFROG_N_MINUS_HALF
        ):
            raise HybridValidationError(
                "checkpoint velocity_staggering must be leapfrog_n_minus_one_half"
            )
        if any(
            particle.velocity_time_level is not self.velocity_staggering
            for particle in particles
        ):
            raise HybridValidationError(
                "all particle velocity time levels must match the checkpoint"
            )
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "dt_s", dt)
        object.__setattr__(self, "particles", particles)


def checkpoint_payload(checkpoint: HybridCheckpoint) -> dict[str, Any]:
    return {
        "schema_version": checkpoint.schema_version,
        "step": checkpoint.step,
        "time_s": checkpoint.time_s,
        "dt_s": checkpoint.dt_s,
        "rng": {"algorithm": checkpoint.rng_algorithm, "seed": checkpoint.rng_seed},
        "time_levels": dict(_TIME_LEVELS),
        "particles": [
            {
                "particle_id": particle.particle_id,
                "species": {
                    "identifier": particle.species.identifier,
                    "symbol": particle.species.symbol,
                    "charge_state": particle.species.charge_state,
                    "mass_kg": particle.species.mass_kg,
                    "charge_c": particle.species.charge_c,
                },
                "position_m": list(particle.position_m),
                "velocity_m_per_s": list(particle.velocity_m_per_s),
                "velocity_time_level": particle.velocity_time_level.value,
                "weight": particle.weight,
                "alive": particle.alive,
            }
            for particle in checkpoint.particles
        ],
        "provenance": asdict(checkpoint.provenance),
    }


def canonical_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise HybridValidationError(
            "checkpoint content must be finite and JSON-serializable"
        ) from error


def checkpoint_digest(checkpoint: HybridCheckpoint) -> str:
    return sha256(canonical_json(checkpoint_payload(checkpoint)).encode("utf-8")).hexdigest()


def save_checkpoint(checkpoint: HybridCheckpoint, path: str | Path) -> None:
    if not isinstance(checkpoint, HybridCheckpoint):
        raise HybridValidationError("checkpoint must be a HybridCheckpoint")
    payload = checkpoint_payload(checkpoint)
    envelope = {
        "sha256": sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        "payload": payload,
    }
    try:
        Path(path).write_text(
            json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as error:
        raise HybridValidationError("checkpoint could not be written") from error


def _require_exact_keys(
    mapping: Any,
    expected: set[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise HybridValidationError(
            f"{context} must contain exactly {sorted(expected)}"
        )
    return mapping


def _require_integer(name: str, value: Any, *, maximum: int | None = None) -> int:
    if (
        type(value) is not int
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise HybridValidationError(f"{name} must be a non-negative integer")
    return value


def _require_json_number(name: str, value: Any) -> float:
    """Accept only JSON parser numeric primitives, never coercible objects."""

    if type(value) not in {int, float}:
        raise HybridValidationError(
            f"{name} must be an actual JSON finite number"
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise HybridValidationError(f"{name} must be finite") from error
    if not isfinite(converted):
        raise HybridValidationError(f"{name} must be finite")
    return converted


def load_checkpoint(path: str | Path) -> HybridCheckpoint:
    try:
        envelope = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except _DuplicateJSONKeyError as error:
        raise HybridValidationError(str(error)) from error
    except HybridValidationError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HybridValidationError(
            "checkpoint is not valid duplicate-free JSON"
        ) from error
    try:
        envelope = _require_exact_keys(
            envelope, {"payload", "sha256"}, "checkpoint envelope"
        )
        payload = envelope["payload"]
        expected_digest = envelope["sha256"]
        if (
            not isinstance(payload, dict)
            or not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            raise HybridValidationError("checkpoint envelope fields are malformed")
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        HybridValidationError,
    ) as error:
        raise HybridValidationError("checkpoint is not a valid v1 envelope") from error
    actual_digest = sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    if actual_digest != expected_digest:
        raise HybridValidationError("checkpoint SHA-256 does not match its payload")

    try:
        payload = _require_exact_keys(
            payload,
            {
                "schema_version",
                "step",
                "time_s",
                "dt_s",
                "rng",
                "time_levels",
                "particles",
                "provenance",
            },
            "checkpoint payload",
        )
        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise HybridValidationError("unsupported checkpoint schema_version")
        rng = _require_exact_keys(payload["rng"], {"algorithm", "seed"}, "rng")
        time_levels = _require_exact_keys(
            payload["time_levels"],
            {"position", "velocity", "fields"},
            "time_levels",
        )
        if time_levels != _TIME_LEVELS:
            raise HybridValidationError("checkpoint time_levels are unsupported")
        particle_data = payload["particles"]
        if not isinstance(particle_data, list):
            raise HybridValidationError("particles must be an array")

        def restore_particle(raw_entry: Any) -> Particle:
            entry = _require_exact_keys(
                raw_entry,
                {
                    "particle_id",
                    "species",
                    "position_m",
                    "velocity_m_per_s",
                    "velocity_time_level",
                    "weight",
                    "alive",
                },
                "particle",
            )
            species_data = _require_exact_keys(
                entry["species"],
                {
                    "identifier",
                    "symbol",
                    "charge_state",
                    "mass_kg",
                    "charge_c",
                },
                "species",
            )
            if not isinstance(entry["alive"], bool):
                raise HybridValidationError("particle alive must be boolean")
            position = entry["position_m"]
            velocity = entry["velocity_m_per_s"]
            if (
                not isinstance(position, list)
                or len(position) != 3
                or not isinstance(velocity, list)
                or len(velocity) != 3
            ):
                raise HybridValidationError(
                    "particle position and velocity must be length-three arrays"
                )
            species = XenonSpecies(
                symbol=species_data["symbol"],
                charge_state=_require_integer(
                    "species charge_state",
                    species_data["charge_state"],
                    maximum=2,
                ),
                mass_kg=_require_json_number(
                    "species mass_kg", species_data["mass_kg"]
                ),
                identifier=species_data["identifier"],
                charge_c_override=_require_json_number(
                    "species charge_c", species_data["charge_c"]
                ),
            )
            try:
                time_level = VelocityTimeLevel(entry["velocity_time_level"])
            except (TypeError, ValueError) as error:
                raise HybridValidationError(
                    "particle velocity_time_level is unsupported"
                ) from error
            return Particle(
                particle_id=_require_integer(
                    "particle_id", entry["particle_id"], maximum=(1 << 64) - 1
                ),
                species=species,
                position_m=tuple(
                    _require_json_number(
                        f"position_m[{index}]", value
                    )
                    for index, value in enumerate(position)
                ),
                velocity_m_per_s=tuple(
                    _require_json_number(
                        f"velocity_m_per_s[{index}]", value
                    )
                    for index, value in enumerate(velocity)
                ),
                weight=_require_json_number("weight", entry["weight"]),
                alive=entry["alive"],
                velocity_time_level=time_level,
            )

        particles = tuple(
            restore_particle(entry) for entry in particle_data
        )
        provenance_data = _require_exact_keys(
            payload["provenance"],
            {
                "model_scope",
                "created_utc",
                "code_revision",
                "cross_section_provenance",
                "notes",
                "time_integration_contract",
            },
            "provenance",
        )
        if not isinstance(provenance_data["notes"], list):
            raise HybridValidationError("provenance notes must be an array")
        provenance = ProvenanceRecord(
            model_scope=provenance_data["model_scope"],
            created_utc=provenance_data["created_utc"],
            code_revision=provenance_data["code_revision"],
            cross_section_provenance=provenance_data["cross_section_provenance"],
            notes=tuple(provenance_data["notes"]),
            time_integration_contract=provenance_data[
                "time_integration_contract"
            ],
        )
        return HybridCheckpoint(
            step=_require_integer("step", payload["step"]),
            time_s=_require_json_number("time_s", payload["time_s"]),
            dt_s=_require_json_number("dt_s", payload["dt_s"]),
            rng_seed=_require_integer(
                "rng seed", rng["seed"], maximum=(1 << 64) - 1
            ),
            rng_algorithm=rng["algorithm"],
            particles=particles,
            provenance=provenance,
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError, HybridValidationError) as error:
        if isinstance(error, HybridValidationError):
            raise
        raise HybridValidationError("checkpoint payload violates the v1 contract") from error


def particles_tuple(particles: Sequence[Particle]) -> tuple[Particle, ...]:
    """Validate a caller sequence before checkpoint construction."""

    return validated_particle_batch(particles)
