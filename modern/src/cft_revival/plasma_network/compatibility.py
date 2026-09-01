"""Auditable N=4 adapter and row-parity report for the accepted implementation."""

from __future__ import annotations

from dataclasses import dataclass

from cft_revival.plasma import (
    PlasmaState,
    XenonGlobalInputs,
    evaluate_plasma_residual_cpu,
)

from .models import NetworkInputs, NetworkState, NetworkValidationError
from .residuals import evaluate_residual
from .topology import make_chain_topology, provenance_hash


@dataclass(frozen=True, slots=True)
class CompatibilityRow:
    row_id: str
    accepted_raw: float
    network_raw: float
    accepted_normalized: float
    network_normalized: float
    exact: bool


@dataclass(frozen=True, slots=True)
class FourCellCompatibilityReport:
    rows: tuple[CompatibilityRow, ...]

    @property
    def compatible(self) -> bool:
        return len(self.rows) == 28 and all(row.exact for row in self.rows)


def from_accepted_four_cell(
    inputs: XenonGlobalInputs,
    state: PlasmaState,
) -> tuple[NetworkInputs, NetworkState]:
    """Map equivalent branch inputs/state without changing accepted code."""

    topology = make_chain_topology(
        4,
        inputs.cusp_arrival_probabilities[:3],
        provenance_seed="accepted-corrected-four-cell-compatibility",
    )
    network_inputs = NetworkInputs(
        topology=topology,
        anode_voltage_v=inputs.anode_voltage_v,
        anode_current_a=inputs.anode_current_a,
        anode_arrival_probability=inputs.cusp_arrival_probabilities[3],
        anode_arrival_standard_uncertainty=0.0,
        anode_arrival_provenance_sha256=provenance_hash(
            "accepted-corrected-four-cell:anode-arrival"
        ),
        cathode_potential_v=inputs.cathode_potential_v,
        cathode_electron_temperature_ev=inputs.cathode_electron_temperature_ev,
        cathode_perveance_a_per_v_3_2=inputs.cathode_perveance_a_per_v_3_2,
        xenon_ionization_energy_ev=inputs.xenon_ionization_energy_ev,
        excitation_fraction=inputs.excitation_fraction,
        ionization_fraction=inputs.ionization_fraction,
        thermalization_fraction=inputs.thermalization_fraction,
        anode_ion_energy_sign=inputs.anode_ion_energy_sign,
    )
    return network_inputs, NetworkState(
        plasma_potential_v=state.plasma_potential_v,
        electron_temperature_ev=state.electron_temperature_ev,
        ionization_source_current_a=state.ionization_source_current_a,
        electron_current_a=state.electron_current_a,
        ion_current_a=state.ion_current_a,
        cusp_ion_current_a=state.cusp_ion_current_a,
    )


def prove_four_cell_compatibility(
    inputs: XenonGlobalInputs,
    state: PlasmaState,
) -> FourCellCompatibilityReport:
    """Compare all 28 raw and normalized rows by identity, not aggregate norm."""

    if len(state.plasma_potential_v) != 4:
        raise NetworkValidationError("compatibility proof requires a four-cell state")
    network_inputs, network_state = from_accepted_four_cell(inputs, state)
    accepted = evaluate_plasma_residual_cpu(state, inputs)
    network = evaluate_residual(network_state, network_inputs)
    rows = tuple(
        CompatibilityRow(
            row_id=network.equation_ids[index],
            accepted_raw=accepted.raw[index],
            network_raw=network.raw[index],
            accepted_normalized=accepted.normalized[index],
            network_normalized=network.normalized[index],
            exact=(
                accepted.raw[index] == network.raw[index]
                and accepted.normalized[index] == network.normalized[index]
            ),
        )
        for index in range(28)
    )
    return FourCellCompatibilityReport(rows)
