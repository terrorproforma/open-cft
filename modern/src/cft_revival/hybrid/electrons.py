"""Fluid-electron closure interfaces without an invented transport model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import (
    BOLTZMANN_CONSTANT_J_PER_K,
    ELEMENTARY_CHARGE_C,
    DepositedMoments,
    ElectronClosureResult,
    ElectronFluidState,
    HybridValidationError,
    SourceExchange,
    Vec3,
    finite_scalar,
    finite_vec3,
)


@runtime_checkable
class FluidElectronClosure(Protocol):
    """Interface for a future electron density/energy/transport closure."""

    def close(
        self,
        ion_moments: DepositedMoments,
        source_exchange: SourceExchange | None = None,
    ) -> ElectronClosureResult:
        """Return an auditable electron state and coupling exchange."""


def conservative_electron_exchange(
    ion_momentum_delta_kg_m_per_s: Vec3,
    ion_energy_delta_j: float,
) -> SourceExchange:
    """Construct exact opposite electron/background source terms."""

    ion_momentum = finite_vec3(
        "ion_momentum_delta_kg_m_per_s", ion_momentum_delta_kg_m_per_s
    )
    ion_energy = finite_scalar("ion_energy_delta_j", ion_energy_delta_j)
    return SourceExchange(
        ion_momentum_delta_kg_m_per_s=ion_momentum,
        background_momentum_delta_kg_m_per_s=tuple(  # type: ignore[arg-type]
            -component for component in ion_momentum
        ),
        ion_energy_delta_j=ion_energy,
        background_energy_delta_j=-ion_energy,
    )


@dataclass(frozen=True, slots=True)
class IsothermalQuasineutralClosure:
    """Verification closure: n_e=rho_i/e and prescribed constant T_e.

    This closure deliberately returns no electric field and no anomalous
    mobility. It is an interface fixture, not an electron-energy or transport
    solution.
    """

    electron_temperature_k: float

    def __post_init__(self) -> None:
        temperature = finite_scalar("electron_temperature_k", self.electron_temperature_k)
        if temperature <= 0.0:
            raise HybridValidationError("electron_temperature_k must be positive")
        object.__setattr__(self, "electron_temperature_k", temperature)

    def close(
        self,
        ion_moments: DepositedMoments,
        source_exchange: SourceExchange | None = None,
    ) -> ElectronClosureResult:
        if not isinstance(ion_moments, DepositedMoments):
            raise HybridValidationError("ion_moments must be DepositedMoments")
        if any(charge_density < 0.0 for charge_density in ion_moments.charge_c_per_m3):
            raise HybridValidationError("ion charge density must be non-negative")
        exchange = SourceExchange() if source_exchange is None else source_exchange
        if not isinstance(exchange, SourceExchange):
            raise HybridValidationError("source_exchange must be SourceExchange")
        if exchange.momentum_residual_kg_m_per_s != (0.0, 0.0, 0.0):
            raise HybridValidationError("electron momentum exchange must be conservative")
        if exchange.energy_residual_j != 0.0:
            raise HybridValidationError("electron energy exchange must be conservative")

        density = tuple(
            charge_density / ELEMENTARY_CHARGE_C
            for charge_density in ion_moments.charge_c_per_m3
        )
        temperatures = tuple(self.electron_temperature_k for _ in density)
        pressure = tuple(
            number_density * BOLTZMANN_CONSTANT_J_PER_K * self.electron_temperature_k
            for number_density in density
        )
        return ElectronClosureResult(
            state=ElectronFluidState(
                number_density_per_m3=density,
                temperature_k=temperatures,
                pressure_pa=pressure,
                anomalous_mobility_m2_per_v_s=None,
            ),
            source_exchange=exchange,
            electric_field_v_per_m=None,
            closure_name="isothermal-quasineutral-verification-only",
        )
