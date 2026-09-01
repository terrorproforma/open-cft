"""Manufactured exact network states for conservation and solver verification."""

from __future__ import annotations

from dataclasses import dataclass

from .models import NetworkInputs, NetworkState, NetworkValidationError
from .topology import make_chain_topology, provenance_hash


@dataclass(frozen=True, slots=True)
class ManufacturedCase:
    inputs: NetworkInputs
    state: NetworkState


def manufactured_zero_cusp_case(
    cell_count: int,
    *,
    anode_voltage_v: float = 1000.0,
    first_potential_fraction: float = 0.01,
) -> ManufacturedCase:
    """Construct an exact zero-cusp solution from the balance recurrences."""

    if cell_count < 1:
        raise NetworkValidationError("cell_count must be >= 1")
    topology = make_chain_topology(
        cell_count,
        (0.0,) * (cell_count - 1),
        provenance_seed=f"manufactured-zero-cusp-n{cell_count}",
    )
    if cell_count == 1:
        phi = (anode_voltage_v,)
    else:
        first = first_potential_fraction * anode_voltage_v
        phi = tuple(
            first
            + (anode_voltage_v - first) * index / (cell_count - 1)
            for index in range(cell_count)
        )
    electron = [0.002 * phi[0] ** 1.5]
    source: list[float] = []
    temperature: list[float] = []
    for cell in range(cell_count):
        gain = (
            phi[cell]
            - (0.0 if cell == 0 else phi[cell - 1])
            + (0.0 if cell == 0 else temperature[cell - 1])
        )
        ionization = electron[cell] * 0.07 * gain / 12.1
        source.append(ionization)
        transported = electron[cell] + ionization
        temperature.append(0.68 * electron[cell] * gain / transported)
        if cell < cell_count - 1:
            electron.append(transported)
    electron.append(electron[-1] + source[-1])
    current = electron[-1]
    inputs = NetworkInputs(
        topology=topology,
        anode_voltage_v=anode_voltage_v,
        anode_current_a=current,
        anode_arrival_probability=0.0,
        anode_arrival_standard_uncertainty=0.0,
        anode_arrival_provenance_sha256=provenance_hash(
            f"manufactured-zero-cusp-n{cell_count}:anode-loss"
        ),
    )
    ion = tuple(current - value for value in electron)
    state = NetworkState(
        plasma_potential_v=phi,
        electron_temperature_ev=tuple(temperature),
        ionization_source_current_a=tuple(source),
        electron_current_a=tuple(electron),
        ion_current_a=ion,
        cusp_ion_current_a=(0.0,) * (cell_count - 1),
    )
    return ManufacturedCase(inputs, state)
