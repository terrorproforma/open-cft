"""Deterministic shifted-Halton initial designs and leakage-safe grouped splits.

This is deliberately not called Sobol: the implementation is a prime-base
Halton sequence with a deterministic Cranley-Patterson shift.
"""

from __future__ import annotations

from hashlib import sha256
from random import Random
from typing import Callable, Sequence, TypeVar

from .domain import Design, Variable

T = TypeVar("T")

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def shifted_halton(
    count: int,
    dimensions: int,
    *,
    seed: int = 0,
    skip: int = 16,
) -> tuple[tuple[float, ...], ...]:
    if count < 0 or dimensions < 1 or dimensions > len(_PRIMES) or skip < 0:
        raise ValueError("invalid Halton shape or skip")
    random = Random(seed)
    shifts = tuple(random.random() for _ in range(dimensions))
    return tuple(
        tuple(
            (_radical_inverse(index, _PRIMES[dimension]) + shifts[dimension]) % 1.0
            for dimension in range(dimensions)
        )
        for index in range(skip + 1, skip + count + 1)
    )


def boundary_challenge_points(dimensions: int) -> tuple[tuple[float, ...], ...]:
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    points: list[tuple[float, ...]] = [
        (0.0,) * dimensions,
        (1.0,) * dimensions,
        (0.5,) * dimensions,
        tuple(float(index % 2) for index in range(dimensions)),
        tuple(float((index + 1) % 2) for index in range(dimensions)),
    ]
    for dimension in range(dimensions):
        low = [0.5] * dimensions
        high = [0.5] * dimensions
        low[dimension] = 0.0
        high[dimension] = 1.0
        points.extend((tuple(low), tuple(high)))
    return tuple(dict.fromkeys(points))


def initial_designs(
    variables: Sequence[Variable],
    count: int,
    *,
    seed: int = 0,
    include_boundary_challenges: bool = True,
) -> tuple[Design, ...]:
    """Generate exactly ``count`` bounded, unique designs."""
    if count < 0:
        raise ValueError("count cannot be negative")
    if count == 0:
        return ()
    variables_tuple = tuple(variables)
    normalized: list[tuple[tuple[float, ...], str]] = []
    if include_boundary_challenges:
        normalized.extend(
            (point, f"boundary-challenge:index={index}")
            for index, point in enumerate(
                boundary_challenge_points(len(variables_tuple))
            )
        )
    normalized.extend(
        (point, f"shifted-halton:seed={seed}:index={index}")
        for index, point in enumerate(
            shifted_halton(
                count + len(normalized),
                len(variables_tuple),
                seed=seed,
            )
        )
    )
    designs: list[Design] = []
    seen: set[str] = set()
    for point, provenance in normalized:
        values = tuple(
            variable.lower + coordinate * (variable.upper - variable.lower)
            for coordinate, variable in zip(point, variables_tuple, strict=True)
        )
        design = Design(values, variables_tuple, provenance=provenance)
        if design.design_id not in seen:
            seen.add(design.design_id)
            designs.append(design)
        if len(designs) == count:
            break
    if len(designs) != count:
        raise RuntimeError("unable to generate requested number of unique designs")
    return tuple(designs)


def grouped_train_validation_split(
    items: Sequence[T],
    design_id: Callable[[T], str],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    """Split by design ID so fidelities and stochastic seeds never leak."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    groups: dict[str, list[T]] = {}
    for item in items:
        groups.setdefault(design_id(item), []).append(item)
    if len(groups) < 2:
        raise ValueError("at least two design groups are required")
    ordered = sorted(
        groups,
        key=lambda key: sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
    validation_ids = set(ordered[:validation_count])
    train = tuple(item for item in items if design_id(item) not in validation_ids)
    validation = tuple(item for item in items if design_id(item) in validation_ids)
    return train, validation
