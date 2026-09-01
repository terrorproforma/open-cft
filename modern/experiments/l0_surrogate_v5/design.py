"""Blind digit-scrambled continuous design and input-only role partitions."""

from __future__ import annotations

from hashlib import sha256
from math import fsum
from typing import Mapping, Sequence

from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    XENON_ATOM_MASS_KG,
    BeamDivergenceFactors,
    ChargeStateFractions,
    MassUtilization,
    PowerBoundaryInputs,
    PropellantMassFlow,
    XenonOperatingPoint,
)


def _digits(index: int, base: int, count: int) -> tuple[int, ...]:
    values = []
    for _ in range(count):
        index, digit = divmod(index, base)
        values.append(digit)
    return tuple(values)


def scrambled_coordinate(
    index: int,
    *,
    base: int,
    dimension: int,
    seed: int,
    digits: int,
) -> float:
    raw = _digits(index, base, digits)
    result = 0.0
    factor = 1.0 / base
    for position, digit in enumerate(raw):
        ordered = sorted(
            range(base),
            key=lambda value: sha256(
                f"{seed}:perm:{dimension}:{position}:{value}".encode()
            ).digest(),
        )
        shift = int.from_bytes(
            sha256(f"{seed}:shift:{dimension}:{position}".encode()).digest()[:8],
            "big",
        ) % base
        transformed = (ordered[digit] + shift) % base
        result += transformed * factor
        factor /= base
    return result


def normalized_design(declaration: Mapping[str, object]) -> tuple[tuple[float, ...], ...]:
    policy = declaration["design"]
    if not isinstance(policy, Mapping):
        raise ValueError("design policy must be an object")
    bases = tuple(int(value) for value in policy["bases"])  # type: ignore[arg-type]
    rows = int(policy["rows"])
    skip = int(policy["skip"])
    seed = int(policy["scramble_seed"])
    digits = int(policy["digits_per_coordinate"])
    result = tuple(
        tuple(
            scrambled_coordinate(
                skip + row + 1,
                base=base,
                dimension=dimension,
                seed=seed,
                digits=digits,
            )
            for dimension, base in enumerate(bases)
        )
        for row in range(rows)
    )
    if len(result) != len(set(result)):
        raise ValueError("scrambled design contains duplicate coordinates")
    return result


def operating_points(
    normalized: Sequence[Sequence[float]],
    source_ranges: Mapping[str, object],
) -> tuple[XenonOperatingPoint, ...]:
    names = (
        "discharge_voltage_v",
        "propellant_mass_flow_kg_per_s",
        "ionized_number_fraction",
        "xe_double_plus_fraction_of_ions",
        "beam_current_fraction_of_anode_current",
        "axial_momentum_fraction_of_ion_momentum",
        "cathode_input_power_w",
        "ppu_efficiency_fraction",
    )
    bounds = []
    for name in names:
        raw = source_ranges[name]
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"source range {name} is invalid")
        bounds.append((float(raw[0]), float(raw[1])))
    points = []
    for row in normalized:
        values = tuple(
            low + float(coordinate) * (high - low)
            for coordinate, (low, high) in zip(row, bounds, strict=True)
        )
        (
            voltage,
            mass_flow,
            ionized,
            double_share,
            beam_fraction,
            axial_fraction,
            cathode_power,
            ppu_efficiency,
        ) = values
        neutral = 1.0 - ionized
        double_plus = ionized * double_share
        plus = 1.0 - neutral - double_plus
        fractions = ChargeStateFractions(neutral, plus, double_plus)
        beam_current = (
            mass_flow
            * ELEMENTARY_CHARGE_C
            / XENON_ATOM_MASS_KG
            * fractions.charge_weighted_ion_fraction
        )
        anode_power = voltage * beam_current / beam_fraction
        thruster_power = fsum((anode_power, cathode_power))
        points.append(
            XenonOperatingPoint(
                discharge_voltage_v=voltage,
                propellant_mass_flow=PropellantMassFlow(mass_flow),
                charge_state_fractions=fractions,
                mass_utilization=MassUtilization.from_charge_states(fractions),
                beam_divergence_factors=BeamDivergenceFactors(
                    beam_fraction,
                    axial_fraction,
                ),
                power_boundaries=PowerBoundaryInputs(
                    cathode_power,
                    thruster_power / ppu_efficiency,
                ),
            )
        )
    return tuple(points)


def surrogate_inputs(
    normalized: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        (row[0], row[1], row[2], row[3], row[5])
        for row in normalized
    )


def group_key(row: Sequence[float], policy: Mapping[str, object]) -> str:
    dimensions = tuple(int(value) for value in policy["grouping_dimensions"])  # type: ignore[arg-type]
    bins = int(policy["bins_per_dimension"])
    values = tuple(min(bins - 1, int(row[index] * bins)) for index in dimensions)
    return ":".join(f"{dimension}={value}" for dimension, value in zip(dimensions, values))


def _group_bins(group: str) -> dict[int, int]:
    return {
        int(field.split("=")[0]): int(field.split("=")[1])
        for field in group.split(":")
    }


def stratum(group: str, policy: Mapping[str, object]) -> str | None:
    values = _group_bins(group)
    strata = policy["strata"]
    if not isinstance(strata, Mapping):
        raise ValueError("strata policy must be an object")
    ood = strata["ood"]["dimension_bins"]  # type: ignore[index]
    if all(values[int(dimension)] in set(bins) for dimension, bins in ood.items()):
        return "ood"
    boundary = set(strata["boundary"]["any_dimension_in"])  # type: ignore[index]
    if any(value in boundary for value in values.values()):
        return "boundary"
    low, high = strata["interpolation"]["all_dimensions_between"]  # type: ignore[index]
    if all(int(low) <= value <= int(high) for value in values.values()):
        return "interpolation"
    return None


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
    raise ValueError(f"insufficient groups for {role}")


def partitions(
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> list[dict[str, object]]:
    policy = declaration["partition"]
    if not isinstance(policy, Mapping):
        raise ValueError("partition policy must be an object")
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(inputs):
        grouped.setdefault(group_key(row, policy), []).append(index)
    rows = {key: tuple(value) for key, value in grouped.items()}
    strata = {group: stratum(group, policy) for group in rows}
    minimum_rows = int(policy["minimum_rows_per_stage_stratum"])
    minimum_groups = int(policy["minimum_groups_per_stage_stratum"])
    result = []
    for raw_seed in policy["replicate_seeds"]:  # type: ignore[union-attr]
        seed = int(raw_seed)
        reserved: set[str] = set()
        roles: dict[str, object] = {}
        for role in ("method-selection", "final-calibration", "assessment"):
            role_record = {}
            for label in ("interpolation", "boundary", "ood"):
                available = tuple(
                    group
                    for group, assigned in strata.items()
                    if assigned == label and group not in reserved
                )
                selected = _take(
                    available,
                    rows,
                    minimum_rows=minimum_rows,
                    minimum_groups=minimum_groups,
                    seed=seed,
                    role=f"{role}:{label}",
                )
                reserved.update(selected)
                role_record[label] = {
                    "groups": list(selected),
                    "indices": [index for group in selected for index in rows[group]],
                }
            roles[role] = role_record
        candidate = [
            index
            for index, row in enumerate(inputs)
            if group_key(row, policy) not in reserved
        ]
        record: dict[str, object] = {
            "replicate_id": f"split-{seed}",
            "seed": seed,
            "candidate_indices": candidate,
            **roles,
        }
        from cft_revival.surrogates.identity import canonical_hash

        record["replicate_partition_hash"] = canonical_hash(record)
        result.append(record)
    return result
