"""Fresh continuous design, physics points, groups and global roles for v7."""

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


def scrambled_coordinate(index: int, base: int, dimension: int, seed: int, digits: int) -> float:
    raw = []
    for _ in range(digits):
        index, digit = divmod(index, base)
        raw.append(digit)
    result = 0.0
    factor = 1.0 / base
    for position, digit in enumerate(raw):
        permutation = sorted(
            range(base),
            key=lambda value: sha256(
                f"{seed}:perm:{dimension}:{position}:{value}".encode()
            ).digest(),
        )
        shift = int.from_bytes(
            sha256(f"{seed}:shift:{dimension}:{position}".encode()).digest()[:8],
            "big",
        ) % base
        result += ((permutation[digit] + shift) % base) * factor
        factor /= base
    return result


def normalized_design(declaration: Mapping[str, object]) -> tuple[tuple[float, ...], ...]:
    policy = declaration["design"]
    bases = tuple(int(value) for value in policy["bases"])  # type: ignore[index]
    return tuple(
        tuple(
            scrambled_coordinate(
                int(policy["skip"]) + row + 1,  # type: ignore[index]
                base,
                dimension,
                int(policy["scramble_seed"]),  # type: ignore[index]
                int(policy["digits_per_coordinate"]),  # type: ignore[index]
            )
            for dimension, base in enumerate(bases)
        )
        for row in range(int(policy["rows"]))  # type: ignore[index]
    )


def surrogate_inputs(normalized: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple((row[0], row[1], row[2], row[3], row[5]) for row in normalized)


def operating_points(
    normalized: Sequence[Sequence[float]],
    ranges: Mapping[str, object],
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
    bounds = tuple(
        (float(ranges[name][0]), float(ranges[name][1]))  # type: ignore[index]
        for name in names
    )
    points = []
    for row in normalized:
        values = tuple(
            low + coordinate * (high - low)
            for coordinate, (low, high) in zip(row, bounds, strict=True)
        )
        voltage, flow, ionized, double_share, beam, axial, cathode, efficiency = values
        neutral = 1.0 - ionized
        double = ionized * double_share
        fractions = ChargeStateFractions(neutral, 1.0 - neutral - double, double)
        beam_current = (
            flow
            * ELEMENTARY_CHARGE_C
            / XENON_ATOM_MASS_KG
            * fractions.charge_weighted_ion_fraction
        )
        anode_power = voltage * beam_current / beam
        thruster_power = fsum((anode_power, cathode))
        points.append(
            XenonOperatingPoint(
                discharge_voltage_v=voltage,
                propellant_mass_flow=PropellantMassFlow(flow),
                charge_state_fractions=fractions,
                mass_utilization=MassUtilization.from_charge_states(fractions),
                beam_divergence_factors=BeamDivergenceFactors(beam, axial),
                power_boundaries=PowerBoundaryInputs(
                    cathode, thruster_power / efficiency
                ),
            )
        )
    return tuple(points)


def group_key(row: Sequence[float], policy: Mapping[str, object]) -> str:
    dimensions = tuple(int(value) for value in policy["grouping_dimensions"])  # type: ignore[index]
    bins = int(policy["bins_per_dimension"])
    values = tuple(min(bins - 1, int(row[index] * bins)) for index in dimensions)
    return ":".join(f"{dimension}={value}" for dimension, value in zip(dimensions, values))


def stratum(group: str, policy: Mapping[str, object]) -> str | None:
    values = {
        int(field.split("=")[0]): int(field.split("=")[1])
        for field in group.split(":")
    }
    strata = policy["strata"]
    ood = strata["ood"]["dimension_bins"]  # type: ignore[index]
    if all(values[int(dimension)] in set(bins) for dimension, bins in ood.items()):
        return "ood"
    if any(
        value in set(strata["boundary"]["any_dimension_in"])  # type: ignore[index]
        for value in values.values()
    ):
        return "boundary"
    low, high = strata["interpolation"]["all_dimensions_between"]  # type: ignore[index]
    if all(int(low) <= value <= int(high) for value in values.values()):
        return "interpolation"
    return None


def global_partition(
    inputs: Sequence[Sequence[float]],
    declaration: Mapping[str, object],
) -> dict[str, object]:
    policy = declaration["partition"]
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(inputs):
        grouped.setdefault(group_key(row, policy), []).append(index)  # type: ignore[arg-type]
    rows = {group: tuple(indices) for group, indices in grouped.items()}
    strata = {group: stratum(group, policy) for group in rows}  # type: ignore[arg-type]
    reserved: set[str] = set()
    result: dict[str, object] = {}
    seed = int(policy["seed"])  # type: ignore[index]
    for role in ("method-selection", "final-calibration", "assessment"):
        splits = {}
        for label in ("interpolation", "boundary", "ood"):
            available = [
                group
                for group, assigned in strata.items()
                if assigned == label and group not in reserved
            ]
            ordered = sorted(
                available,
                key=lambda group: (
                    sha256(f"{seed}:{role}:{label}:{group}".encode()).hexdigest(),
                    group,
                ),
            )
            selected = []
            row_count = 0
            for group in ordered:
                selected.append(group)
                row_count += len(rows[group])
                if (
                    len(selected) >= int(policy["minimum_groups_per_stage_stratum"])  # type: ignore[index]
                    and row_count >= int(policy["minimum_rows_per_stage_stratum"])  # type: ignore[index]
                ):
                    break
            else:
                raise ValueError(f"insufficient groups for {role}:{label}")
            reserved.update(selected)
            splits[label] = {
                "groups": selected,
                "indices": [index for group in selected for index in rows[group]],
            }
        result[role] = splits
    result["candidate_indices"] = [
        index
        for index, row in enumerate(inputs)
        if group_key(row, policy) not in reserved  # type: ignore[arg-type]
    ]
    return result
