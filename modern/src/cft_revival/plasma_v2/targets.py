"""Published four-cell states used as REPRODUCTION TARGETS (not truth).

Two sources print states of the Kornfeld four-cell system:

* Kornfeld, Koch, Harmann, IEPC-2007-108 (2007), Table 3.1: DM9.2 and DM10,
  4-stage columns, ``Ua = 1 kV``, ``Ja = 1 A``.  The paper states the values
  come from MathCAD's minimum-error solver ("minfehl"), "power accuracy ...
  within 0.5 %"; the printed cusp potentials cancel from every printed
  equation, so they are a property of the solver start, not of the model.
* Puca, Panelli, Battista, Aerotecnica Missili & Spazio 103(4), 321-338
  (2024), Table 1: a genetic-algorithm minimum of a weighted sum of squares
  of their 33-equation variant (p_1..p_4 as unknowns, cathode current as an
  input) for the same two thrusters; the DM9.2*/DM10* columns of that table
  reproduce Kornfeld's numbers and are not repeated here.

Every number below is transcribed from the published table; nothing is
fitted.  ``je0`` is not printed by Puca (their cathode current is an input)
and is derived from their own row R01 with a flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from cft_revival.plasma import PlasmaState, XenonGlobalInputs, evaluate_plasma_residual_cpu

from .constants import MASS_FLUX_RATIO
from .models import (
    AnodeRow,
    CuspLossClosure,
    CuspSheathSpec,
    FourthPotentialRow,
    PotentialClosure,
    SheathClosureInputs,
    SheathClosureState,
    SheathRegime,
)
from .residuals import evaluate_residual


@dataclass(frozen=True, slots=True)
class PublishedFourCellState:
    identifier: str
    source: str
    anode_voltage_v: float
    anode_current_a: float
    plasma_potential_v: tuple[float, float, float, float]
    cusp_wall_potential_v: tuple[float, float, float]
    electron_temperature_ev: tuple[float, float, float, float]
    ionization_source_current_a: tuple[float, float, float, float]
    electron_current_a: tuple[float | None, float, float, float, float]
    ion_current_a: tuple[float, float, float, float, float]
    cusp_ion_current_a: tuple[float, float, float]
    cusp_probabilities: tuple[float, float, float, float]
    published_powers_w: dict[str, float] | None
    notes: tuple[str, ...]

    @property
    def sheath_drops_v(self) -> tuple[float, float, float]:
        """``phi_k - phi_ck`` as printed (informational; see module docstring)."""

        return tuple(  # type: ignore[return-value]
            self.plasma_potential_v[k] - self.cusp_wall_potential_v[k] for k in range(3)
        )

    def cathode_emission_a(self) -> tuple[float, bool]:
        """Return ``je0`` and whether it was derived (Puca does not print it)."""

        printed = self.electron_current_a[0]
        if printed is not None:
            return printed, False
        p1 = self.cusp_probabilities[0]
        derived = (self.electron_current_a[1] - self.ionization_source_current_a[0]) / (1.0 - p1)
        return derived, True

    def core_state(self) -> PlasmaState:
        je0, _ = self.cathode_emission_a()
        return PlasmaState(
            plasma_potential_v=self.plasma_potential_v,
            electron_temperature_ev=self.electron_temperature_ev,
            ionization_source_current_a=self.ionization_source_current_a,
            electron_current_a=(je0, *self.electron_current_a[1:]),  # type: ignore[arg-type]
            ion_current_a=self.ion_current_a,
            cusp_ion_current_a=self.cusp_ion_current_a,
        )

    def v2_state(self) -> SheathClosureState:
        return SheathClosureState(
            self.core_state(), self.sheath_drops_v, self.cusp_probabilities[:3]  # type: ignore[arg-type]
        )

    def declared_potential_closure(self, *, anode_row: AnodeRow = AnodeRow.DECLARED_FALL) -> PotentialClosure:
        """Potential closure that reproduces the published potentials exactly (mode C) or leaves phi_1 solved."""

        phi = self.plasma_potential_v
        fall = phi[3] - self.anode_voltage_v
        if anode_row is AnodeRow.DECLARED_FALL:
            return PotentialClosure(
                interior_step_3_v=max(phi[2] - phi[1], 0.0),
                interior_step_4_v=max(phi[3] - phi[2], 0.0),
                anode_row=AnodeRow.DECLARED_FALL,
                fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED,
                anode_fall_v=max(fall, 0.0),
                cathode_coupling_v=phi[0],
            )
        return PotentialClosure(
            interior_step_3_v=max(phi[2] - phi[1], 0.0),
            interior_step_4_v=max(phi[3] - phi[2], 0.0),
            anode_row=AnodeRow.SHEATH,
            fourth_row=FourthPotentialRow.ANODE_FALL_DECLARED,
            anode_fall_v=max(fall, 0.0),
        )

    def v2_inputs(
        self,
        *,
        regime: SheathRegime = SheathRegime.SPACE_CHARGE_LIMITED,
        anode_row: AnodeRow = AnodeRow.DECLARED_FALL,
    ) -> SheathClosureInputs:
        """CL-1 inputs with the published p and the published potential structure."""

        cusps = tuple(CuspSheathSpec(regime=regime) for _ in range(3))
        return SheathClosureInputs(
            anode_voltage_v=self.anode_voltage_v,
            anode_current_a=self.anode_current_a,
            cusps=cusps,  # type: ignore[arg-type]
            anode_cusp_probability=self.cusp_probabilities[3],
            cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
            declared_cusp_probabilities=self.cusp_probabilities[:3],
            potentials=self.declared_potential_closure(anode_row=anode_row),
        )

    def implied_anode_fall_v(self) -> float | None:
        """Anode fall the anode-sheath row would assign to the printed anode currents."""

        anode_ion = -self.ion_current_a[4]
        electron = self.electron_current_a[4]
        if anode_ion <= 0.0 or electron <= 0.0:
            return None
        return self.electron_temperature_ev[3] * log(MASS_FLUX_RATIO * anode_ion / electron)


KORNFELD_2007_SOURCE = (
    "Kornfeld, Koch, Harmann, IEPC-2007-108 (2007), Table 3.1, 4-stage column; "
    "Ua = 1 kV, Ja = 1 A; MathCAD minimum-error solution"
)
PUCA_2024_SOURCE = (
    "Puca, Panelli, Battista, Aerotecnica Missili & Spazio 103(4), 321-338 (2024), "
    "DOI 10.1007/s42496-024-00203-x, Table 1 (GA minimum of the 33-equation variant)"
)

KORNFELD_DM92 = PublishedFourCellState(
    identifier="kornfeld-2007-table-3.1-dm9.2-4-stage",
    source=KORNFELD_2007_SOURCE,
    anode_voltage_v=1000.0,
    anode_current_a=1.0,
    plasma_potential_v=(14.1, 1000.0, 1000.0, 1000.0),
    cusp_wall_potential_v=(8.1, 960.0, 965.0),
    electron_temperature_ev=(8.9, 100.1, 43.1, 23.5),
    ionization_source_current_a=(0.008, 0.543, 0.310, 0.157),
    electron_current_a=(0.106, 0.107, 0.637, 0.845, 1.002),
    ion_current_a=(0.894, 0.893, 0.363, 0.155, -0.002),
    cusp_ion_current_a=(0.007, 0.013, 0.102),
    cusp_probabilities=(0.060, 0.119, 0.160, 0.254),
    published_powers_w={
        "beam": 891.6,
        "ionization": 12.3,
        "anode": 27.7,
        "cusp": 22.9,
        "excitation": 51.43,
    },
    notes=(
        "printed component powers sum to 1005.9 W against Ua*Ja = 1000 W",
        "cusp potentials cancel from every printed equation",
    ),
)

KORNFELD_DM10 = PublishedFourCellState(
    identifier="kornfeld-2007-table-3.1-dm10-4-stage",
    source=KORNFELD_2007_SOURCE,
    anode_voltage_v=1000.0,
    anode_current_a=1.0,
    plasma_potential_v=(12.3, 979.0, 999.0, 1000.0),
    cusp_wall_potential_v=(12.2, 979.0, 979.0),
    electron_temperature_ev=(7.8, 99.8, 48.1, 26.2),
    ionization_source_current_a=(0.006, 0.473, 0.361, 0.229),
    electron_current_a=(0.086, 0.090, 0.557, 0.882, 1.111),
    ion_current_a=(0.914, 0.910, 0.442, 0.118, -0.111),
    cusp_ion_current_a=(0.002, 0.006, 0.037),
    cusp_probabilities=(0.024, 0.064, 0.066, 0.092),
    published_powers_w={
        "beam": 899.4,
        "ionization": 12.1,
        "anode": 31.0,
        "cusp": 10.0,
        "excitation": 51.43,
    },
    notes=("printed component powers sum to 1003.9 W against Ua*Ja = 1000 W",),
)

PUCA_DM92 = PublishedFourCellState(
    identifier="puca-2024-table-1-dm9.2-ga",
    source=PUCA_2024_SOURCE,
    anode_voltage_v=1000.0,
    anode_current_a=1.0,
    plasma_potential_v=(14.02, 979.15, 998.90, 1000.0),
    cusp_wall_potential_v=(12.2, 978.99, 978.99),
    electron_temperature_ev=(7.88, 99.52, 47.99, 26.0),
    ionization_source_current_a=(-0.0644, 0.5235, 0.3180, 0.2212),
    electron_current_a=(None, 0.2528, 0.6165, 0.7788, 0.9432),
    ion_current_a=(0.8160, 0.8994, 0.3835, 0.2212, 0.0),
    cusp_ion_current_a=(0.0484, 0.0835, 0.1556),
    cusp_probabilities=(0.49, 0.63, 0.25, 6.1e-13),
    published_powers_w=None,
    notes=(
        "I_1 is negative (a negative ionization source current)",
        "j_i4 = 0: no ion current to the anode, so the anode sheath row is undefined",
        "j_e0 is not printed (cathode current is an input of the 33-equation variant); derived from R01",
    ),
)

PUCA_DM10 = PublishedFourCellState(
    identifier="puca-2024-table-1-dm10-ga",
    source=PUCA_2024_SOURCE,
    anode_voltage_v=1000.0,
    anode_current_a=1.0,
    plasma_potential_v=(12.27, 979.26, 998.88, 1000.0),
    cusp_wall_potential_v=(12.2, 978.99, 978.99),
    electron_temperature_ev=(7.89, 99.84, 48.04, 26.02),
    ionization_source_current_a=(-0.0294, 0.5066, 0.3182, 0.2214),
    electron_current_a=(None, 0.2267, 0.5964, 0.7786, 1.0),
    ion_current_a=(0.8931, 0.9050, 0.4036, 0.2214, 0.0),
    cusp_ion_current_a=(0.0159, 0.0709, 0.1360),
    cusp_probabilities=(0.57, 0.60, 0.23, 7.6e-14),
    published_powers_w=None,
    notes=(
        "I_1 is negative",
        "j_i4 = 0: anode sheath row undefined",
        "j_e0 derived from R01",
    ),
)

REPRODUCTION_TARGETS: tuple[PublishedFourCellState, ...] = (
    KORNFELD_DM92,
    KORNFELD_DM10,
    PUCA_DM92,
    PUCA_DM10,
)


def v1_power_components(target: PublishedFourCellState) -> dict[str, float]:
    """Kornfeld-convention power components (v1 rows, +EI in Pcusp, printed anode sign)."""

    state = target.core_state()
    inputs = XenonGlobalInputs(
        anode_voltage_v=target.anode_voltage_v,
        anode_current_a=target.anode_current_a,
        cusp_arrival_probabilities=target.cusp_probabilities,
    )
    powers = evaluate_plasma_residual_cpu(state, inputs).powers
    return {
        "beam": powers.beam_power_w,
        "ionization": powers.ionization_loss_w,
        "excitation": powers.excitation_loss_w,
        "cusp": powers.cusp_loss_w,
        "anode": powers.anode_net_power_w,
        "closure": powers.closure_w,
    }


def v2_power_components(target: PublishedFourCellState, regime: SheathRegime) -> dict[str, float]:
    state = target.v2_state()
    inputs = target.v2_inputs(regime=regime, anode_row=AnodeRow.DECLARED_FALL)
    powers = evaluate_residual(state, inputs).powers
    return {
        "beam": powers.beam_power_w,
        "ionization": powers.ionization_loss_w,
        "excitation": powers.excitation_loss_w,
        "cusp": powers.cusp_loss_w,
        "cusp_electron_wall": powers.cusp_electron_wall_w,
        "cusp_ion_wall": powers.cusp_ion_wall_w,
        "anode": powers.anode_electron_loss_w + powers.anode_ion_loss_w,
        "closure": powers.closure_w,
    }
