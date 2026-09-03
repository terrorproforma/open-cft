"""Typed contracts for the sheath-closed four-cell power balance (v2).

The v2 model is the corrected Kornfeld ledger (rows R00-R26 of
``cft_revival.plasma`` unchanged, the two ``PROPOSED_NOT_ACCEPTED``
corrections applied to R27) plus per-cusp floating-dielectric sheath rows
R28-R30, an anode row R31, three declared potential-closure rows R32-R34 and
three cusp-loss rows R35-R37.  It is a DEVELOPMENT model: nothing in this
package is accepted evidence, and no thruster claim follows from it.

The v1 package is imported read-only (its five files are hash-bound by the
paper's ``analytic-consistency`` gate); this module never modifies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log
from typing import Sequence

from cft_revival.plasma import (
    PlasmaNumericsError,
    PlasmaState,
    PlasmaValidationError,
    SolverDiagnostics,
    XenonGlobalInputs,
)

from .constants import (
    CRITICAL_EMISSION_YIELD,
    MASS_FLUX_RATIO,
    SPACE_CHARGE_LIMITED_COEFFICIENT,
)

CELL_COUNT = 4
DIELECTRIC_CUSP_COUNT = 3
CORE_STATE_SIZE = 25
STATE_SIZE = 31
RESIDUAL_SIZE = 38

ROW_IDS: tuple[str, ...] = tuple(f"R{index:02d}" for index in range(RESIDUAL_SIZE))
CURRENT_ROWS: tuple[int, ...] = tuple(range(0, 12)) + tuple(range(15, 23))
POWER_ROWS: tuple[int, ...] = tuple(range(12, 15)) + tuple(range(23, 28))
VOLTAGE_ROWS: tuple[int, ...] = tuple(range(28, 35))
CUSP_PROBABILITY_ROWS: tuple[int, ...] = tuple(range(35, 38))


class SheathRegime(str, Enum):
    """Which floating-sheath relation closes ``phi_k - phi_ck`` at cusp k."""

    FLOATING_NO_EMISSION = "floating_no_emission"
    FLOATING_WITH_EMISSION = "floating_with_emission"
    SPACE_CHARGE_LIMITED = "space_charge_limited"


class CuspLossClosure(str, Enum):
    """How the interior cusp loss probabilities p_1..p_3 are closed."""

    CL1_DECLARED = "CL-1-declared"
    CL3_SHEATH_LIMITED = "CL-3-sheath-limited"
    CL4_HYBRID_AREA = "CL-4-hybrid-area"


class AnodeRow(str, Enum):
    """Row R31."""

    SHEATH = "anode_sheath"
    DECLARED_FALL = "declared_anode_fall"


class FourthPotentialRow(str, Enum):
    """Row R34 (the fourth potential-closure relation)."""

    ANODE_FALL_DECLARED = "anode_fall_declared"
    CATHODE_COUPLING_DECLARED = "cathode_coupling_declared"


def _finite(name: str, value: float) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise PlasmaValidationError(f"{name} must be finite")
    return converted


def _positive(name: str, value: float) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise PlasmaValidationError(f"{name} must be greater than zero")
    return converted


def _tuple_n(name: str, values: Sequence[float], count: int) -> tuple[float, ...]:
    if len(values) != count:
        raise PlasmaValidationError(f"{name} must contain exactly {count} values")
    return tuple(_finite(f"{name}[{index}]", value) for index, value in enumerate(values))


@dataclass(frozen=True, slots=True)
class CuspSheathSpec:
    """Declared inputs of one dielectric cusp (k = 1..3).

    ``area_ratio`` is ``rho_k = A_e,k / A_i,k``, the electron leak area over the
    ion collection area in the ambipolar balance; 1 is the pointwise floating
    dielectric (Lieberman & Lichtenberg 6.2.17).  ``access_fraction`` is the
    collisionless geometric access fraction ``A_k`` used by CL-3 (a screening
    quantity, never a cusp probability).  ``electron_density_per_m3`` and
    ``wall_field_t`` are the declared cusp density and wall field used by
    CL-4 (they must come with their own provenance; the package supplies no
    default).
    """

    regime: SheathRegime = SheathRegime.FLOATING_NO_EMISSION
    emission_yield: float = 0.0
    area_ratio: float = 1.0
    access_fraction: float = 0.0
    electron_density_per_m3: float | None = None
    wall_field_t: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.regime, SheathRegime):
            raise PlasmaValidationError("regime must be a SheathRegime")
        gamma = _finite("emission_yield", self.emission_yield)
        if gamma < 0.0 or gamma >= 1.0:
            raise PlasmaValidationError("emission_yield must be in [0, 1)")
        if self.regime is SheathRegime.FLOATING_NO_EMISSION and gamma != 0.0:
            raise PlasmaValidationError(
                "emission_yield must be zero under floating_no_emission"
            )
        ratio = _positive("area_ratio", self.area_ratio)
        access = _finite("access_fraction", self.access_fraction)
        if access < 0.0 or access > 1.0:
            raise PlasmaValidationError("access_fraction must be in [0, 1]")
        density = self.electron_density_per_m3
        field = self.wall_field_t
        if density is not None:
            density = _positive("electron_density_per_m3", density)
        if field is not None:
            field = _positive("wall_field_t", field)
        object.__setattr__(self, "emission_yield", gamma)
        object.__setattr__(self, "area_ratio", ratio)
        object.__setattr__(self, "access_fraction", access)
        object.__setattr__(self, "electron_density_per_m3", density)
        object.__setattr__(self, "wall_field_t", field)

    def sheath_coefficient(self) -> float:
        """Return ``c_s,k`` with ``Delta phi_s,k = c_s,k T_k`` (rows R28-R30).

        * no emission: ``ln(K0 rho_k)`` (Lieberman & Lichtenberg 6.2.17);
        * with emission: ``max(ln((1-gamma_k) K0 rho_k), 1.02)`` (Hobbs &
          Wesson 1967: the floating value until the space-charge limit);
        * space-charge limited: ``1.02`` (Hobbs & Wesson 1967).
        """

        if self.regime is SheathRegime.SPACE_CHARGE_LIMITED:
            return SPACE_CHARGE_LIMITED_COEFFICIENT
        floating = log((1.0 - self.emission_yield) * MASS_FLUX_RATIO * self.area_ratio)
        if self.regime is SheathRegime.FLOATING_NO_EMISSION:
            return floating
        return max(floating, SPACE_CHARGE_LIMITED_COEFFICIENT)

    @property
    def emission_is_space_charge_limited(self) -> bool:
        return self.emission_yield >= CRITICAL_EMISSION_YIELD


@dataclass(frozen=True, slots=True)
class PotentialClosure:
    """Declared potential closure (rows R31-R34).

    ``CL-3-potentials`` (Koch 2011 finding ii; Brandt 2016): a flat interior
    with declared steps ``phi_3 - phi_2 = interior_step_3_v`` and
    ``phi_4 - phi_3 = interior_step_4_v``.  Row R31 is either the anode
    electron-collecting sheath (identifies ``phi_4 - Ua`` from the anode ion
    fraction) or a declared anode fall.  Row R34 declares either the anode
    fall (then ``phi_1`` is SOLVED through R31) or the cathode coupling
    ``phi_1 - phi_0`` (then the anode fall is SOLVED through R31).
    """

    interior_step_3_v: float = 0.0
    interior_step_4_v: float = 0.0
    anode_row: AnodeRow = AnodeRow.SHEATH
    fourth_row: FourthPotentialRow = FourthPotentialRow.ANODE_FALL_DECLARED
    anode_fall_v: float = 0.0
    cathode_coupling_v: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.anode_row, AnodeRow):
            raise PlasmaValidationError("anode_row must be an AnodeRow")
        if not isinstance(self.fourth_row, FourthPotentialRow):
            raise PlasmaValidationError("fourth_row must be a FourthPotentialRow")
        step_3 = _finite("interior_step_3_v", self.interior_step_3_v)
        step_4 = _finite("interior_step_4_v", self.interior_step_4_v)
        if step_3 < 0.0 or step_4 < 0.0:
            raise PlasmaValidationError("interior potential steps must be non-negative")
        fall = _finite("anode_fall_v", self.anode_fall_v)
        if fall < 0.0:
            raise PlasmaValidationError("anode_fall_v must be non-negative (phi_4 >= Ua)")
        if (
            self.anode_row is AnodeRow.DECLARED_FALL
            and self.fourth_row is FourthPotentialRow.ANODE_FALL_DECLARED
        ):
            raise PlasmaValidationError(
                "declared_anode_fall together with anode_fall_declared would duplicate "
                "a row; declare the cathode coupling instead"
            )
        coupling = self.cathode_coupling_v
        if self.fourth_row is FourthPotentialRow.CATHODE_COUPLING_DECLARED:
            if coupling is None:
                raise PlasmaValidationError("cathode_coupling_v is required for that row")
            coupling = _positive("cathode_coupling_v", coupling)
        elif coupling is not None:
            coupling = _positive("cathode_coupling_v", coupling)
        object.__setattr__(self, "interior_step_3_v", step_3)
        object.__setattr__(self, "interior_step_4_v", step_4)
        object.__setattr__(self, "anode_fall_v", fall)
        object.__setattr__(self, "cathode_coupling_v", coupling)

    @property
    def solved_potential(self) -> str:
        """Which potential quantity row R31 solves (or ``none`` if all declared)."""

        if self.anode_row is AnodeRow.DECLARED_FALL:
            return "none"
        if self.fourth_row is FourthPotentialRow.ANODE_FALL_DECLARED:
            return "phi_1"
        return "phi_4 - Ua"


@dataclass(frozen=True, slots=True)
class SheathClosureInputs:
    """Boundary conditions and declared closures for one v2 solve."""

    anode_voltage_v: float
    anode_current_a: float
    cusps: tuple[CuspSheathSpec, CuspSheathSpec, CuspSheathSpec]
    anode_cusp_probability: float
    cusp_loss_closure: CuspLossClosure = CuspLossClosure.CL3_SHEATH_LIMITED
    declared_cusp_probabilities: tuple[float, float, float] = (0.0, 0.0, 0.0)
    potentials: PotentialClosure = PotentialClosure()
    leak_width_prefactor: float = 1.0
    wall_radius_m: float | None = None
    cathode_potential_v: float = 0.0
    cathode_electron_temperature_ev: float = 0.0
    cathode_perveance_a_per_v_3_2: float = 0.002
    xenon_ionization_energy_ev: float = 12.1
    excitation_fraction: float = 0.25
    ionization_fraction: float = 0.07
    thermalization_fraction: float = 0.68

    def __post_init__(self) -> None:
        if len(self.cusps) != DIELECTRIC_CUSP_COUNT or any(
            not isinstance(cusp, CuspSheathSpec) for cusp in self.cusps
        ):
            raise PlasmaValidationError("cusps must contain exactly three CuspSheathSpec")
        if not isinstance(self.cusp_loss_closure, CuspLossClosure):
            raise PlasmaValidationError("cusp_loss_closure must be a CuspLossClosure")
        if not isinstance(self.potentials, PotentialClosure):
            raise PlasmaValidationError("potentials must be a PotentialClosure")
        declared = _tuple_n("declared_cusp_probabilities", self.declared_cusp_probabilities, 3)
        for index, value in enumerate(declared):
            if not 0.0 <= value < 1.0:
                raise PlasmaValidationError(
                    f"declared_cusp_probabilities[{index}] must be in [0, 1)"
                )
        anode_probability = _finite("anode_cusp_probability", self.anode_cusp_probability)
        if not 0.0 <= anode_probability < 1.0:
            raise PlasmaValidationError("anode_cusp_probability must be in [0, 1)")
        prefactor = _positive("leak_width_prefactor", self.leak_width_prefactor)
        radius = self.wall_radius_m
        if radius is not None:
            radius = _positive("wall_radius_m", radius)
        if self.cusp_loss_closure is CuspLossClosure.CL4_HYBRID_AREA:
            if radius is None:
                raise PlasmaValidationError("CL-4 requires wall_radius_m")
            for index, cusp in enumerate(self.cusps):
                if cusp.electron_density_per_m3 is None or cusp.wall_field_t is None:
                    raise PlasmaValidationError(
                        f"CL-4 requires electron_density_per_m3 and wall_field_t at cusp {index + 1}"
                    )
        # Validate the shared v1 scalar contract by constructing a v1 input.
        reference = self.v1_inputs(declared + (anode_probability,))
        object.__setattr__(self, "anode_voltage_v", reference.anode_voltage_v)
        object.__setattr__(self, "anode_current_a", reference.anode_current_a)
        object.__setattr__(self, "declared_cusp_probabilities", declared)
        object.__setattr__(self, "anode_cusp_probability", anode_probability)
        object.__setattr__(self, "leak_width_prefactor", prefactor)
        object.__setattr__(self, "wall_radius_m", radius)
        object.__setattr__(self, "cathode_potential_v", reference.cathode_potential_v)
        object.__setattr__(
            self, "cathode_electron_temperature_ev", reference.cathode_electron_temperature_ev
        )
        object.__setattr__(
            self, "cathode_perveance_a_per_v_3_2", reference.cathode_perveance_a_per_v_3_2
        )
        object.__setattr__(
            self, "xenon_ionization_energy_ev", reference.xenon_ionization_energy_ev
        )
        object.__setattr__(self, "excitation_fraction", reference.excitation_fraction)
        object.__setattr__(self, "ionization_fraction", reference.ionization_fraction)
        object.__setattr__(self, "thermalization_fraction", reference.thermalization_fraction)
        coupling = self.potentials.cathode_coupling_v
        if coupling is not None and self.cathode_potential_v + coupling >= self.anode_voltage_v:
            raise PlasmaValidationError("cathode coupling must keep phi_1 below the anode voltage")

    def v1_inputs(self, probabilities: Sequence[float]) -> XenonGlobalInputs:
        """Return the read-only v1 input object for a given probability vector.

        The v1 rows R00-R26 are reused through this object (manifold
        parametrization and parity tests); the v1 power row R27 is NOT used.
        """

        probability = _tuple_n("probabilities", probabilities, 4)
        return XenonGlobalInputs(
            anode_voltage_v=self.anode_voltage_v,
            anode_current_a=self.anode_current_a,
            cusp_arrival_probabilities=probability,  # type: ignore[arg-type]
            cathode_potential_v=self.cathode_potential_v,
            cathode_electron_temperature_ev=self.cathode_electron_temperature_ev,
            cathode_perveance_a_per_v_3_2=self.cathode_perveance_a_per_v_3_2,
            xenon_ionization_energy_ev=self.xenon_ionization_energy_ev,
            excitation_fraction=self.excitation_fraction,
            ionization_fraction=self.ionization_fraction,
            thermalization_fraction=self.thermalization_fraction,
        )

    def sheath_coefficients(self) -> tuple[float, float, float]:
        return tuple(cusp.sheath_coefficient() for cusp in self.cusps)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SheathClosureState:
    """31-variable v2 state: the v1 core plus sheath drops and cusp probabilities."""

    core: PlasmaState
    sheath_drop_v: tuple[float, float, float]
    cusp_probability: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.core, PlasmaState):
            raise PlasmaValidationError("core must be a v1 PlasmaState")
        object.__setattr__(self, "sheath_drop_v", _tuple_n("sheath_drop_v", self.sheath_drop_v, 3))
        object.__setattr__(
            self, "cusp_probability", _tuple_n("cusp_probability", self.cusp_probability, 3)
        )

    def to_vector(self) -> tuple[float, ...]:
        return (*self.core.to_vector(), *self.sheath_drop_v, *self.cusp_probability)

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> SheathClosureState:
        if len(values) != STATE_SIZE:
            raise PlasmaValidationError(f"state vector must contain exactly {STATE_SIZE} values")
        vector = tuple(_finite(f"state[{index}]", value) for index, value in enumerate(values))
        return cls(
            core=PlasmaState.from_vector(vector[0:CORE_STATE_SIZE]),
            sheath_drop_v=vector[25:28],  # type: ignore[arg-type]
            cusp_probability=vector[28:31],  # type: ignore[arg-type]
        )

    @property
    def cusp_wall_potential_v(self) -> tuple[float, float, float]:
        """The re-identified Kornfeld cusp potentials ``phi_ck = phi_k - Delta phi_s,k``."""

        phi = self.core.plasma_potential_v
        return tuple(phi[k] - self.sheath_drop_v[k] for k in range(3))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CuspEnergySplit:
    """Energy bookkeeping at one dielectric cusp (all in W).

    ``total_w = L_k dE_k`` is what the corrected R27 books (the cusp potential
    cancels, as in Kornfeld).  The sheath rows identify the split into the
    electron kinetic energy at the wall ``L_k (dE_k - Delta phi_s,k)`` and the
    ion fall ``L_k Delta phi_s,k``.  The Maxwellian estimates
    (``2 T_k + Delta phi`` per electron, ``Delta phi + T_k/2`` per ion; Goebel
    & Katz Ch. 4, Lieberman & Lichtenberg) are reported as DIAGNOSTICS: they
    are not part of the balance because the cascade rows R23-R26 carry no
    presheath term and the lost electrons are the monoenergetic entering beam.
    """

    lost_electron_current_a: float
    entering_energy_ev: float
    sheath_drop_v: float
    total_w: float
    electron_wall_w: float
    ion_wall_w: float
    electron_wall_energy_margin_ev: float
    maxwellian_electron_estimate_w: float
    maxwellian_ion_estimate_w: float


@dataclass(frozen=True, slots=True)
class PowerBalanceV2:
    beam_power_w: float
    ionization_loss_w: float
    excitation_loss_w: float
    cusp_loss_w: float
    cusp_electron_wall_w: float
    cusp_ion_wall_w: float
    anode_electron_loss_w: float
    anode_ion_loss_w: float
    input_power_w: float
    closure_w: float
    cusps: tuple[CuspEnergySplit, CuspEnergySplit, CuspEnergySplit]


@dataclass(frozen=True, slots=True)
class ResidualEvaluationV2:
    raw: tuple[float, ...]
    normalized: tuple[float, ...]
    powers: PowerBalanceV2
    margins: tuple[float, ...]
    margin_names: tuple[str, ...]
    cusp_energy_margins_ev: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SolverPolicy:
    """Fail-closed publication policy of the v2 solver."""

    enforce_cusp_energy_margin: bool = True
    seed_from_manifold: bool = True


@dataclass(frozen=True, slots=True)
class SheathSolveResult:
    state: SheathClosureState | None
    evaluation: ResidualEvaluationV2 | None
    diagnostics: SolverDiagnostics
    seeded_from_manifold: bool


@dataclass(frozen=True, slots=True)
class SheathMultiStartResult:
    best: SheathSolveResult
    attempts: tuple[SheathSolveResult, ...]
    selected_start_index: int
    residual_floor: float


@dataclass(frozen=True, slots=True)
class RankReport:
    """Structural rank of the v2 Jacobian at a state, block by block."""

    rows: int
    unknowns: int
    rank_full: int
    rank_corrected_core: int
    rank_with_sheath_and_anode: int
    nullity_before_potential_closure: int
    solved_potential: str
    declared_relations: tuple[str, ...]
    condition_estimate: float


def require_numerics(name: str, value: float) -> float:
    if not isfinite(value):
        raise PlasmaNumericsError(f"{name} produced a non-finite value")
    return float(value)
