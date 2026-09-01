"""Stateless counter-based random contract shared by deterministic operators."""

from __future__ import annotations

from .models import HybridValidationError

RNG_ALGORITHM = "splitmix64-counter-v1"
_MASK_64 = (1 << 64) - 1


def _uint64(name: str, value: int) -> int:
    if type(value) is not int or not 0 <= value <= _MASK_64:
        raise HybridValidationError(f"{name} must be an unsigned 64-bit integer")
    return value


def random_u64(
    seed: int,
    particle_id: int,
    step: int,
    *,
    stream: int = 0,
    draw: int = 0,
) -> int:
    """Return one value keyed only by immutable counters, never call order."""

    seed = _uint64("seed", seed)
    particle_id = _uint64("particle_id", particle_id)
    step = _uint64("step", step)
    stream = _uint64("stream", stream)
    draw = _uint64("draw", draw)
    value = (
        seed
        ^ ((particle_id * 0xD2B74407B1CE6E93) & _MASK_64)
        ^ ((step * 0xCA5A826395121157) & _MASK_64)
        ^ ((stream * 0x9E3779B97F4A7C15) & _MASK_64)
        ^ ((draw * 0x94D049BB133111EB) & _MASK_64)
    )
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def random_uniform(
    seed: int,
    particle_id: int,
    step: int,
    *,
    stream: int = 0,
    draw: int = 0,
) -> float:
    """Return a reproducible binary64 value in the half-open interval [0, 1)."""

    return (random_u64(seed, particle_id, step, stream=stream, draw=draw) >> 11) * (
        1.0 / (1 << 53)
    )
