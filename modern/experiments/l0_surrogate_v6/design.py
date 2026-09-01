"""Global, input-only v6 design and role partition."""

from __future__ import annotations

from hashlib import sha256
from typing import Mapping, Sequence

from experiments.l0_surrogate_v5.design import (
    group_key,
    operating_points,
    scrambled_coordinate,
    stratum,
    surrogate_inputs,
)


def normalized_design(declaration: Mapping[str, object]) -> tuple[tuple[float, ...], ...]:
    policy = declaration["design"]
    bases = tuple(int(value) for value in policy["bases"])  # type: ignore[index]
    return tuple(
        tuple(
            scrambled_coordinate(
                int(policy["skip"]) + row + 1,  # type: ignore[index]
                base=base,
                dimension=dimension,
                seed=int(policy["scramble_seed"]),  # type: ignore[index]
                digits=int(policy["digits_per_coordinate"]),  # type: ignore[index]
            )
            for dimension, base in enumerate(bases)
        )
        for row in range(int(policy["rows"]))  # type: ignore[index]
    )


def _ordered(groups: Sequence[str], seed: int, role: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                sha256(f"{seed}:{role}:{group}".encode()).hexdigest(),
                group,
            ),
        )
    )


def _take(
    groups: Sequence[str],
    rows: Mapping[str, tuple[int, ...]],
    *,
    minimum_rows: int,
    minimum_groups: int,
    seed: int,
    role: str,
) -> tuple[str, ...]:
    chosen = []
    count = 0
    for group in _ordered(groups, seed, role):
        chosen.append(group)
        count += len(rows[group])
        if len(chosen) >= minimum_groups and count >= minimum_rows:
            return tuple(chosen)
    raise ValueError(f"insufficient input groups for {role}")


def global_partition(
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    policy = declaration["partition"]
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(inputs):
        grouped.setdefault(group_key(row, policy), []).append(index)  # type: ignore[arg-type]
    rows = {key: tuple(value) for key, value in grouped.items()}
    strata = {group: stratum(group, policy) for group in rows}  # type: ignore[arg-type]
    seed = int(policy["seed"])  # type: ignore[index]
    reserved: set[str] = set()
    result: dict[str, object] = {}
    for role in ("method-selection", "final-calibration", "assessment"):
        splits = {}
        for label in ("interpolation", "boundary", "ood"):
            available = tuple(
                group
                for group, assigned in strata.items()
                if assigned == label and group not in reserved
            )
            selected = _take(
                available,
                rows,
                minimum_rows=int(policy["minimum_rows"][role]),  # type: ignore[index]
                minimum_groups=int(policy["minimum_groups_per_stage_stratum"]),  # type: ignore[index]
                seed=seed,
                role=f"{role}:{label}",
            )
            reserved.update(selected)
            splits[label] = {
                "groups": list(selected),
                "indices": [index for group in selected for index in rows[group]],
            }
        result[role] = splits
    result["candidate_indices"] = [
        index
        for index, row in enumerate(inputs)
        if group_key(row, policy) not in reserved  # type: ignore[arg-type]
    ]
    return result
