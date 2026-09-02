"""Simulation configuration, CPU reference backend, and the time-stepping driver.

Step ``n`` (positions ``x^n``, velocities ``v^(n-1/2)``):

1. deposit node charges from ``x^n`` (fixed-point bilinear), add wall surface charge;
2. solve Poisson for ``phi^n`` (warm-started from ``phi^(n-1)``), form nodal ``E^n``;
3. gather ``E^n``, ``B`` at ``x^n``; Boris push to ``v^(n+1/2)``; advance to ``x^(n+1)``;
4. classify boundaries: anode/exit absorption (counted currents), dielectric
   wall absorption with surface-charge deposition, Courant violations fail closed;
5. null-collision MCC on electrons; ionisation products appended;
6. inject exit-plane electrons; ``t <- t + dt``.

Diagnostics are accumulated at ``x^n`` inside the configured averaging window.
The CPU backend is the numerical reference; ``warp_backend.WarpBackend`` must
reproduce it (bit-identical deposition, roundoff-level push, distributional MCC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd, isfinite, pi, sqrt
from typing import Any, Literal, Mapping

import numpy as np

from . import kernels
from .fields import MagneticFieldMap
from .mcc import MCCConfig, NullCollisionMCC, XenonCrossSections, maxwellian_velocity
from .mesh import MeshMasks, build_mesh_masks
from .neutrals import NeutralInventory, NeutralInventoryConfig, NeutralState
from .models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EV_J,
    BoundaryPotentials,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
    ParticleArrays,
    PoissonConfig2D,
    Species2D,
    StabilityLimits,
    StabilityReport2D,
    electron_species,
    require_stable,
    stability_report,
    xenon_ion_species,
)
from .poisson import Poisson2D, electric_field_nodes, field_energy_j, induced_electrode_charge_c

BackendName = Literal["cpu", "warp-cpu", "warp-cuda"]


@dataclass(frozen=True, slots=True)
class InjectionConfig:
    electron_current_a: float
    electron_temperature_ev: float

    def __post_init__(self) -> None:
        if not isfinite(self.electron_current_a) or self.electron_current_a < 0.0:
            raise PIC2DValidationError("injection current must be finite and non-negative")
        if not isfinite(self.electron_temperature_ev) or self.electron_temperature_ev <= 0.0:
            raise PIC2DValidationError("injection temperature must be positive")

    def to_dict(self) -> dict[str, float]:
        return {"electron_current_a": self.electron_current_a, "electron_temperature_ev": self.electron_temperature_ev}


@dataclass(frozen=True, slots=True)
class SeedPlasmaConfig:
    density_per_m3: float
    electron_temperature_ev: float
    ion_temperature_ev: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.density_per_m3) or self.density_per_m3 < 0.0:
            raise PIC2DValidationError("seed density must be finite and non-negative")
        if not isfinite(self.electron_temperature_ev) or self.electron_temperature_ev <= 0.0:
            raise PIC2DValidationError("seed electron temperature must be positive")
        if not isfinite(self.ion_temperature_ev) or self.ion_temperature_ev < 0.0:
            raise PIC2DValidationError("seed ion temperature must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "density_per_m3": self.density_per_m3,
            "electron_temperature_ev": self.electron_temperature_ev,
            "ion_temperature_ev": self.ion_temperature_ev,
        }


@dataclass(frozen=True, slots=True)
class PIC2DConfig:
    grid: Grid2D
    potentials: BoundaryPotentials
    dt_s: float
    macro_weight: float
    seed: int = 0
    injection: InjectionConfig | None = None
    seed_plasma: SeedPlasmaConfig | None = None
    mcc: MCCConfig | None = None
    poisson: PoissonConfig2D = PoissonConfig2D()
    limits: StabilityLimits = StabilityLimits()
    reference_density_per_m3: float = 1.0e17
    reference_electron_temperature_ev: float = 10.0
    max_electron_energy_ev: float = 400.0
    fixed_point_deposition: bool = True
    series_interval_steps: int = 100
    runtime_stability_check_steps: int = 100
    # v1.1: ions are pushed every ``ion_subcycle`` steps with ``ion_subcycle * dt``
    # (positions and charge frozen in between, births added incrementally).
    ion_subcycle: int = 1
    # v1.1: the device backend reads its ledger/gate statistics back every
    # ``device_sync_steps`` steps (default: gcd of the series and gate cadences);
    # series and gate cadences must be multiples of it.
    device_sync_steps: int | None = None
    # v1.3: quasi-steady 0-D neutral inventory (feed, ionisation, effusion, artificial
    # relaxation) updated at every series interval; requires ``mcc`` whose density is
    # the initial value and the null-collision ceiling.
    neutral_inventory: NeutralInventoryConfig | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise PIC2DValidationError("dt_s must be positive")
        if self.neutral_inventory is not None and (self.mcc is None or self.mcc.neutral_density_per_m3 <= 0.0):
            raise PIC2DValidationError("neutral_inventory requires an MCC configuration with a positive neutral density")
        if not isfinite(self.macro_weight) or self.macro_weight <= 0.0:
            raise PIC2DValidationError("macro_weight must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise PIC2DValidationError("seed must be a non-negative integer")
        for name in ("series_interval_steps", "runtime_stability_check_steps", "ion_subcycle"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PIC2DValidationError(f"{name} must be a positive integer")
        if self.device_sync_steps is not None:
            value = self.device_sync_steps
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PIC2DValidationError("device_sync_steps must be a positive integer")
            if self.series_interval_steps % value != 0:
                raise PIC2DValidationError("series_interval_steps must be a multiple of device_sync_steps")
            if self.runtime_stability_check_steps % value != 0:
                raise PIC2DValidationError("runtime_stability_check_steps must be a multiple of device_sync_steps")
        for name in ("reference_density_per_m3", "reference_electron_temperature_ev", "max_electron_energy_ev"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise PIC2DValidationError(f"{name} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "potentials": self.potentials.to_dict(),
            "dt_s": self.dt_s,
            "macro_weight": self.macro_weight,
            "seed": self.seed,
            "injection": None if self.injection is None else self.injection.to_dict(),
            "seed_plasma": None if self.seed_plasma is None else self.seed_plasma.to_dict(),
            "mcc": None if self.mcc is None else self.mcc.to_dict(),
            "poisson": self.poisson.to_dict(),
            "limits": self.limits.to_dict(),
            "reference_density_per_m3": self.reference_density_per_m3,
            "reference_electron_temperature_ev": self.reference_electron_temperature_ev,
            "max_electron_energy_ev": self.max_electron_energy_ev,
            "fixed_point_deposition": self.fixed_point_deposition,
            "series_interval_steps": self.series_interval_steps,
            "runtime_stability_check_steps": self.runtime_stability_check_steps,
            "ion_subcycle": self.ion_subcycle,
            "device_sync_steps": self.sync_steps,
        } | ({} if self.neutral_inventory is None else {"neutral_inventory": self.neutral_inventory.to_dict()})
        # (the key is present only when the inventory is on, so v1.0-v1.2 config identities are unchanged)

    @property
    def sync_steps(self) -> int:
        if self.device_sync_steps is not None:
            return self.device_sync_steps
        return gcd(self.series_interval_steps, self.runtime_stability_check_steps)


@dataclass(slots=True)
class SimulationState:
    """Complete dynamical state (numpy) for checkpoints and backend exchange."""

    step: int
    time_s: float
    electrons: ParticleArrays
    ions: ParticleArrays
    surface_charge_c: np.ndarray
    phi_v: np.ndarray
    injection_carry: float
    cumulative: dict[str, float]
    # v1.3: the neutral inventory (None when the background is static)
    neutral: NeutralState | None = None

    def copy(self) -> "SimulationState":
        return SimulationState(
            self.step, self.time_s, self.electrons.copy(), self.ions.copy(),
            self.surface_charge_c.copy(), self.phi_v.copy(), self.injection_carry, dict(self.cumulative),
            None if self.neutral is None else self.neutral.copy(),
        )


CUMULATIVE_KEYS = (
    "anode_electrons", "anode_ions", "exit_electrons", "exit_ions", "wall_electrons", "wall_ions",
    "injected_electrons", "ionizations", "excitations", "elastic",
    "ke_injected_j", "ke_absorbed_anode_j", "ke_absorbed_exit_j", "ke_absorbed_wall_j",
    "inelastic_loss_j", "ke_born_ions_j", "field_work_j",
)


def empty_cumulative() -> dict[str, float]:
    return {key: 0.0 for key in CUMULATIVE_KEYS}


@dataclass(slots=True)
class StepTally:
    poisson_iterations: int
    max_omega_pe_dt: float
    max_electron_speed_m_per_s: float
    electron_count: int
    ion_count: int


class DiagnosticAccumulator:
    """Time-window sums of node maps and boundary fluxes (CPU numpy)."""

    def __init__(self, masks: MeshMasks) -> None:
        self.masks = masks
        shape = masks.grid.node_shape
        nz = masks.grid.axial_cells
        nr = masks.grid.radial_cells
        self.steps = 0
        self.n_e = np.zeros(shape)
        self.n_i = np.zeros(shape)
        self.phi = np.zeros(shape)
        self.e_weight = np.zeros(shape)
        self.e_vr = np.zeros(shape)
        self.e_vt = np.zeros(shape)
        self.e_vz = np.zeros(shape)
        self.e_v2 = np.zeros(shape)
        self.ionization = np.zeros(shape)
        self.wall_electrons = np.zeros(nz)
        self.wall_ions = np.zeros(nz)
        self.wall_electron_energy_j = np.zeros(nz)
        self.wall_ion_energy_j = np.zeros(nz)
        self.exit_ions = np.zeros(nr)
        self.exit_electrons = np.zeros(nr)

    def reset(self) -> None:
        self.__init__(self.masks)

    def to_arrays(self, electron_weight: float, dt_s: float) -> dict[str, np.ndarray]:
        steps = max(self.steps, 1)
        masks = self.masks
        grid = masks.grid
        window_s = steps * dt_s
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_v2 = np.where(self.e_weight > 0.0, self.e_v2 / self.e_weight, 0.0)
            mean_vr = np.where(self.e_weight > 0.0, self.e_vr / self.e_weight, 0.0)
            mean_vt = np.where(self.e_weight > 0.0, self.e_vt / self.e_weight, 0.0)
            mean_vz = np.where(self.e_weight > 0.0, self.e_vz / self.e_weight, 0.0)
        drift2 = mean_vr**2 + mean_vt**2 + mean_vz**2
        t_e_ev = ELECTRON_MASS_KG * np.maximum(mean_v2 - drift2, 0.0) / (3.0 * EV_J)
        r = grid.r_m
        dz = grid.dz_m
        wall_radius = grid.geometry.wall_radius_m(grid.z_m[:-1] + 0.5 * dz)
        wall_area = 2.0 * pi * wall_radius * dz
        exit_area = pi * (r[1:] ** 2 - r[:-1] ** 2)
        volume = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
        return {
            "n_e_per_m3": self.n_e / steps,
            "n_i_per_m3": self.n_i / steps,
            "phi_v": self.phi / steps,
            "t_e_ev": t_e_ev,
            "ionization_rate_per_m3_s": self.ionization * electron_weight / (volume * window_s),
            "wall_electron_flux_per_m2_s": self.wall_electrons * electron_weight / (wall_area * window_s),
            "wall_ion_flux_per_m2_s": self.wall_ions * electron_weight / (wall_area * window_s),
            "wall_electron_mean_energy_ev": np.where(
                self.wall_electrons > 0,
                self.wall_electron_energy_j / np.maximum(self.wall_electrons, 1) / (electron_weight * EV_J), 0.0,
            ),
            "wall_ion_mean_energy_ev": np.where(
                self.wall_ions > 0,
                self.wall_ion_energy_j / np.maximum(self.wall_ions, 1) / (electron_weight * EV_J), 0.0,
            ),
            "exit_ion_current_density_a_per_m2": self.exit_ions * electron_weight * ELEMENTARY_CHARGE_C / (exit_area * window_s),
            "exit_electron_current_density_a_per_m2": self.exit_electrons * electron_weight * ELEMENTARY_CHARGE_C / (exit_area * window_s),
            "window_steps": np.array([self.steps]),
        }


def instantaneous_maps(config: PIC2DConfig, masks: MeshMasks, state: SimulationState) -> dict[str, np.ndarray]:
    """Single-sample node maps (n_e, n_i, phi, T_e) from a state, in the window-map layout.

    Used to finalize a run from its checkpoint when the device-side window
    accumulators are gone; the flux/ionisation maps have no history and are zero,
    ``window_steps`` is 1.
    """

    diag = DiagnosticAccumulator(masks)
    electron = electron_species(config.macro_weight)
    ion = xenon_ion_species(config.macro_weight)
    q_e = kernels.deposit_node_charge(masks, electron, state.electrons, fixed_point=config.fixed_point_deposition)
    q_i = kernels.deposit_node_charge(masks, ion, state.ions, fixed_point=config.fixed_point_deposition)
    with np.errstate(invalid="ignore", divide="ignore"):
        volume = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
        diag.n_e += np.abs(q_e) / (ELEMENTARY_CHARGE_C * volume)
        diag.n_i += np.abs(q_i) / (ELEMENTARY_CHARGE_C * volume)
    diag.phi += state.phi_v
    electrons = state.electrons
    if electrons.count:
        diag.e_weight += kernels.deposit_node_moment(masks, electrons, np.ones(electrons.count))
        diag.e_vr += kernels.deposit_node_moment(masks, electrons, electrons.vr_m_per_s)
        diag.e_vt += kernels.deposit_node_moment(masks, electrons, electrons.vt_m_per_s)
        diag.e_vz += kernels.deposit_node_moment(masks, electrons, electrons.vz_m_per_s)
        diag.e_v2 += kernels.deposit_node_moment(masks, electrons, electrons.speed_squared())
    diag.steps = 1
    return diag.to_arrays(config.macro_weight, config.dt_s)


class CPUBackend:
    """Numpy reference implementation of one PIC-MCC cycle."""

    name = "cpu-numpy-reference"

    def __init__(
        self,
        config: PIC2DConfig,
        masks: MeshMasks,
        field: MagneticFieldMap,
        cross_sections: XenonCrossSections | None,
    ) -> None:
        self.config = config
        self.masks = masks
        self.field = field
        self.electron = electron_species(config.macro_weight)
        self.ion = xenon_ion_species(config.macro_weight)
        self.poisson = Poisson2D(masks, config.poisson)
        self.mcc = None
        if config.mcc is not None:
            if cross_sections is None:
                raise PIC2DValidationError("MCC requires cross sections")
            self.mcc = NullCollisionMCC(cross_sections, config.mcc, self.ion)
        self.state: SimulationState | None = None
        self.diagnostics = DiagnosticAccumulator(masks)
        self.quantum_c = ELEMENTARY_CHARGE_C * config.macro_weight
        self.last_tally: StepTally | None = None

    # -- state exchange -------------------------------------------------
    def load_state(self, state: SimulationState) -> None:
        self.state = state.copy()

    def export_state(self) -> SimulationState:
        assert self.state is not None
        return self.state.copy()

    def set_neutral_scale(self, scale: float) -> None:
        """v1.3: real-collision frequency factor ``n_g / n_g0`` (null ceiling fixed at ``n_g0``)."""

        if self.mcc is None:
            raise PIC2DValidationError("neutral scale requires MCC")
        self.mcc.set_neutral_scale(scale)

    @property
    def step_index(self) -> int:
        assert self.state is not None
        return self.state.step

    def flush(self) -> StepTally | None:
        return self.last_tally

    def series_sample(self) -> dict[str, Any]:
        assert self.state is not None
        state = self.state
        return {
            "step": state.step, "time_s": state.time_s,
            "electrons": state.electrons.count, "ions": state.ions.count,
            "kinetic_electron_j": kernels.kinetic_energy_j(self.electron, state.electrons),
            "kinetic_ion_j": kernels.kinetic_energy_j(self.ion, state.ions),
            "surface_charge_c": float(state.surface_charge_c.sum()), "phi_v": state.phi_v.copy(),
            "cumulative": dict(state.cumulative),
        }

    # -- one cycle --------------------------------------------------------
    def step(self, accumulate: bool) -> StepTally:
        assert self.state is not None
        state = self.state
        config = self.config
        masks = self.masks
        grid = masks.grid
        dt = config.dt_s
        electrons, ions = state.electrons, state.ions
        fixed = config.fixed_point_deposition
        ion_step = (state.step + 1) % config.ion_subcycle == 0

        q_e = kernels.deposit_node_charge(masks, self.electron, electrons, fixed_point=fixed)
        q_i = kernels.deposit_node_charge(masks, self.ion, ions, fixed_point=fixed)
        volume_charge = q_e + q_i
        source = volume_charge * masks.charge_to_source + state.surface_charge_c
        result = self.poisson.solve(source, config.potentials, initial_phi_v=state.phi_v)
        phi = result.phi_v
        e_r, e_z = electric_field_nodes(masks, phi)

        if accumulate:
            self._accumulate_maps(q_e, q_i, phi, electrons)

        max_speed = 0.0
        field_work = 0.0
        for species, particles, is_electron in ((self.electron, electrons, True), (self.ion, ions, False)):
            if particles.count == 0 or (not is_electron and not ion_step):
                continue
            species_dt = dt if is_electron else dt * config.ion_subcycle
            er = kernels.gather_nodes(grid, e_r, particles.r_m, particles.z_m)
            ez = kernels.gather_nodes(grid, e_z, particles.r_m, particles.z_m)
            br = kernels.gather_nodes(grid, self.field.b_r_t, particles.r_m, particles.z_m)
            bz = kernels.gather_nodes(grid, self.field.b_z_t, particles.r_m, particles.z_m)
            k_before = kernels.kinetic_energy_j(species, particles)
            vx, vy, vz = kernels.boris_push(
                particles.vr_m_per_s, particles.vt_m_per_s, particles.vz_m_per_s,
                er, ez, br, bz, species.charge_c, species.mass_kg, species_dt,
            )
            r_new, z_new, vr_new, vt_new, _, _ = kernels.advance_positions(
                particles.r_m, particles.z_m, vx, vy, vz, species_dt
            )
            moved = ParticleArrays(r_new, z_new, vr_new, vt_new, vz)
            field_work += kernels.kinetic_energy_j(species, moved) - k_before
            if is_electron:
                max_speed = float(np.sqrt(np.max(moved.speed_squared())))
            codes = kernels.classify_boundary(masks, moved.r_m, moved.z_m)
            if np.any(codes == kernels.BOUNDARY_INVALID):
                raise PIC2DStabilityError("a particle crossed more than one cell in a step (Courant violation)")
            self._absorb(species, moved, codes, is_electron, accumulate)
            keep = codes == kernels.BOUNDARY_INSIDE
            if is_electron:
                electrons = moved.select(keep)
            else:
                ions = moved.select(keep)
        state.cumulative["field_work_j"] += field_work

        if self.mcc is not None and electrons.count:
            rng = np.random.default_rng([config.seed, state.step, 1])
            mcc_result = self.mcc.apply(electrons, dt, rng)
            electrons = mcc_result.electrons
            tally = mcc_result.tally
            if mcc_result.new_electrons.count:
                electrons = electrons.append(mcc_result.new_electrons)
                ions = ions.append(mcc_result.new_ions)
                state.cumulative["ke_born_ions_j"] += kernels.kinetic_energy_j(self.ion, mcc_result.new_ions)
                if accumulate:
                    self.diagnostics.ionization += kernels.deposit_node_moment(
                        masks, mcc_result.new_ions, np.ones(mcc_result.new_ions.count)
                    )
            state.cumulative["ionizations"] += tally.ionization
            state.cumulative["excitations"] += tally.excitation
            state.cumulative["elastic"] += tally.elastic
            state.cumulative["inelastic_loss_j"] += tally.inelastic_energy_loss_j

        if config.injection is not None and config.injection.electron_current_a > 0.0:
            injected, state.injection_carry = self._inject(state.step, state.injection_carry)
            if injected.count:
                electrons = electrons.append(injected)
                state.cumulative["injected_electrons"] += injected.count
                state.cumulative["ke_injected_j"] += kernels.kinetic_energy_j(self.electron, injected)

        state.electrons = electrons
        state.ions = ions
        state.phi_v = phi
        state.step += 1
        state.time_s = state.step * dt
        if accumulate:
            self.diagnostics.steps += 1
        peak_density = float(np.max(np.abs(q_e[masks.plasma_node]) / (ELEMENTARY_CHARGE_C * masks.shape_volume_m3[masks.plasma_node])))
        omega_pe = sqrt(peak_density * ELEMENTARY_CHARGE_C**2 / (8.8541878128e-12 * ELECTRON_MASS_KG))
        self.last_tally = StepTally(result.diagnostics.iterations, omega_pe * dt, max_speed, electrons.count, ions.count)
        return self.last_tally

    def _accumulate_maps(self, q_e: np.ndarray, q_i: np.ndarray, phi: np.ndarray, electrons: ParticleArrays) -> None:
        masks = self.masks
        diag = self.diagnostics
        with np.errstate(invalid="ignore", divide="ignore"):
            volume = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
            diag.n_e += np.abs(q_e) / (ELEMENTARY_CHARGE_C * volume)
            diag.n_i += np.abs(q_i) / (ELEMENTARY_CHARGE_C * volume)
        diag.phi += phi
        if electrons.count:
            ones = np.ones(electrons.count)
            diag.e_weight += kernels.deposit_node_moment(masks, electrons, ones)
            diag.e_vr += kernels.deposit_node_moment(masks, electrons, electrons.vr_m_per_s)
            diag.e_vt += kernels.deposit_node_moment(masks, electrons, electrons.vt_m_per_s)
            diag.e_vz += kernels.deposit_node_moment(masks, electrons, electrons.vz_m_per_s)
            diag.e_v2 += kernels.deposit_node_moment(masks, electrons, electrons.speed_squared())

    def _absorb(self, species: Species2D, moved: ParticleArrays, codes: np.ndarray, is_electron: bool, accumulate: bool) -> None:
        assert self.state is not None
        state = self.state
        grid = self.masks.grid
        label = "electrons" if is_electron else "ions"
        c2 = 299792458.0**2
        speed2 = moved.speed_squared()
        ke = (speed2 / c2 / (1.0 + np.sqrt(1.0 - speed2 / c2))) * species.mass_kg * c2 * species.macro_weight
        for code, name, energy_key in (
            (kernels.BOUNDARY_ANODE, "anode", "ke_absorbed_anode_j"),
            (kernels.BOUNDARY_EXIT, "exit", "ke_absorbed_exit_j"),
            (kernels.BOUNDARY_WALL, "wall", "ke_absorbed_wall_j"),
        ):
            mask = codes == code
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            state.cumulative[f"{name}_{label}"] += count
            state.cumulative[energy_key] += float(ke[mask].sum())
            if code == kernels.BOUNDARY_WALL:
                charge = np.full(count, species.charge_c * species.macro_weight)
                state.surface_charge_c += kernels.wall_surface_deposit(
                    self.masks, moved.r_m[mask], moved.z_m[mask], charge,
                    fixed_point=self.config.fixed_point_deposition, quantum_c=self.quantum_c,
                )
                if accumulate:
                    j = np.clip(((moved.z_m[mask] - grid.geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, grid.axial_cells - 1)
                    target = self.diagnostics.wall_electrons if is_electron else self.diagnostics.wall_ions
                    energy_target = self.diagnostics.wall_electron_energy_j if is_electron else self.diagnostics.wall_ion_energy_j
                    np.add.at(target, j, 1.0)
                    np.add.at(energy_target, j, ke[mask])
            elif code == kernels.BOUNDARY_EXIT and accumulate:
                i = np.clip((moved.r_m[mask] / grid.dr_m).astype(np.int64), 0, grid.radial_cells - 1)
                target = self.diagnostics.exit_electrons if is_electron else self.diagnostics.exit_ions
                np.add.at(target, i, 1.0)

    def _inject(self, step: int, carry: float) -> tuple[ParticleArrays, float]:
        config = self.config
        assert config.injection is not None
        grid = self.masks.grid
        expected = config.injection.electron_current_a * config.dt_s / (ELEMENTARY_CHARGE_C * config.macro_weight) + carry
        count = int(np.floor(expected))
        carry = expected - count
        if count == 0:
            return ParticleArrays.empty(), carry
        rng = np.random.default_rng([config.seed, step, 2])
        u = rng.random((7, count))
        return injection_sample(config, self.masks, u), carry

    def diagnostic_arrays(self) -> dict[str, np.ndarray]:
        return self.diagnostics.to_arrays(self.config.macro_weight, self.config.dt_s)

    def reset_diagnostics(self) -> None:
        self.diagnostics.reset()


def injection_sample(config: PIC2DConfig, masks: MeshMasks, u: np.ndarray) -> ParticleArrays:
    """Map uniforms ``u`` (shape (7, N)) to exit-plane injected electrons.

    Radius uniform in area over the exit aperture, position inside the last
    half cell, transverse velocities Maxwellian, axial velocity a flux-weighted
    half-Maxwellian directed into the channel (``v_z < 0``).  Shared by the CPU
    and Warp backends so both sample the same distribution.
    """

    assert config.injection is not None
    grid = masks.grid
    nz = grid.axial_cells
    r_max = grid.r_m[masks.top_plasma_cell[nz - 1] + 1]
    r = r_max * np.sqrt(u[0]) * (1.0 - 1e-9)
    z = grid.geometry.z_max_m - 0.5 * grid.dz_m * u[1] - 1e-9 * grid.dz_m
    temperature_k = config.injection.electron_temperature_ev * EV_J / 1.380649e-23
    thermal = sqrt(EV_J * config.injection.electron_temperature_ev / ELECTRON_MASS_KG)
    vr, vt, _ = maxwellian_velocity(ELECTRON_MASS_KG, temperature_k, u[2:6])
    vz = -thermal * np.sqrt(-2.0 * np.log(np.maximum(u[6], 1e-300)))
    return ParticleArrays(r, z, vr, vt, vz)


def seed_plasma_state(config: PIC2DConfig, masks: MeshMasks) -> SimulationState:
    """Uniform quasi-neutral seed plasma (or empty) as the initial state."""

    grid = masks.grid
    electrons = ParticleArrays.empty()
    ions = ParticleArrays.empty()
    if config.seed_plasma is not None and config.seed_plasma.density_per_m3 > 0.0:
        count = int(round(config.seed_plasma.density_per_m3 * masks.plasma_volume_m3 / config.macro_weight))
        rng = np.random.default_rng([config.seed, 0, 3])
        r_list: list[np.ndarray] = []
        z_list: list[np.ndarray] = []
        accepted = 0
        r_box = grid.geometry.max_radius_m
        while accepted < count:
            batch = max(1024, 2 * (count - accepted))
            r = r_box * np.sqrt(rng.random(batch))
            z = grid.geometry.z_min_m + grid.geometry.length_m * rng.random(batch)
            codes = kernels.classify_boundary(masks, r, z)
            keep = codes == kernels.BOUNDARY_INSIDE
            r_list.append(r[keep])
            z_list.append(z[keep])
            accepted += int(np.count_nonzero(keep))
        r = np.concatenate(r_list)[:count]
        z = np.concatenate(z_list)[:count]
        te = config.seed_plasma.electron_temperature_ev * EV_J / 1.380649e-23
        ve = maxwellian_velocity(ELECTRON_MASS_KG, te, rng.random((4, count)))
        electrons = ParticleArrays(r, z, *ve)
        if config.seed_plasma.ion_temperature_ev > 0.0:
            ti = config.seed_plasma.ion_temperature_ev * EV_J / 1.380649e-23
            vi = maxwellian_velocity(xenon_ion_species(1.0).mass_kg, ti, rng.random((4, count)))
        else:
            vi = (np.zeros(count), np.zeros(count), np.zeros(count))
        ions = ParticleArrays(r.copy(), z.copy(), *vi)
    return SimulationState(
        0, 0.0, electrons, ions, np.zeros(grid.node_shape), np.zeros(grid.node_shape), 0.0, empty_cumulative()
    )


@dataclass(slots=True)
class SeriesRecord:
    step: int
    time_s: float
    electrons: int
    ions: int
    phi_mean_v: float
    phi_min_v: float
    phi_max_v: float
    kinetic_electron_j: float
    kinetic_ion_j: float
    field_energy_j: float
    surface_charge_c: float
    peak_omega_pe_dt: float
    poisson_iterations: int
    currents_a: dict[str, float]
    ledger: dict[str, float]
    # v1.3: neutral inventory sample (None for a static background)
    neutral: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "time_s": self.time_s, "electrons": self.electrons, "ions": self.ions,
            "phi_mean_v": self.phi_mean_v, "phi_min_v": self.phi_min_v, "phi_max_v": self.phi_max_v,
            "kinetic_electron_j": self.kinetic_electron_j, "kinetic_ion_j": self.kinetic_ion_j,
            "field_energy_j": self.field_energy_j, "surface_charge_c": self.surface_charge_c,
            "peak_omega_pe_dt": self.peak_omega_pe_dt, "poisson_iterations": self.poisson_iterations,
            "currents_a": dict(self.currents_a), "ledger": dict(self.ledger),
            "neutral": None if self.neutral is None else dict(self.neutral),
        }


class Simulation:
    """Driver: stability gate, backend stepping, time series, diagnostics."""

    def __init__(
        self,
        config: PIC2DConfig,
        field: MagneticFieldMap,
        *,
        cross_sections: XenonCrossSections | None = None,
        backend: BackendName = "cpu",
        device: str = "cuda:0",
    ) -> None:
        if field.grid != config.grid:
            raise PIC2DValidationError("field map grid must equal the configuration grid")
        self.config = config
        self.field = field
        self.masks = build_mesh_masks(config.grid)
        self.cross_sections = cross_sections
        probability = 0.0
        if config.mcc is not None:
            if cross_sections is None:
                raise PIC2DValidationError("MCC configuration requires cross sections")
            probability = NullCollisionMCC(cross_sections, config.mcc, xenon_ion_species(config.macro_weight)).collision_probability(config.dt_s)
        self.stability = require_stable(
            stability_report(
                config.grid, config.dt_s,
                reference_density_per_m3=config.reference_density_per_m3,
                reference_electron_temperature_ev=config.reference_electron_temperature_ev,
                max_b_t=field.max_b_t,
                max_electron_energy_ev=config.max_electron_energy_ev,
                max_collision_probability=probability,
                limits=config.limits,
            )
        )
        self.backend_name = backend
        if backend == "cpu":
            self.backend: Any = CPUBackend(config, self.masks, field, cross_sections)
        else:
            from .warp_backend import WarpBackend

            self.backend = WarpBackend(config, self.masks, field, cross_sections, device=("cpu" if backend == "warp-cpu" else device))
        self.backend.load_state(seed_plasma_state(config, self.masks))
        self.series: list[SeriesRecord] = []
        self._last_cumulative = empty_cumulative()
        self._last_energy: float | None = None
        self._last_electrode: tuple[float, float] | None = None
        self._series_base_step = 0
        self.neutrals: NeutralInventory | None = None
        self.neutral_state: NeutralState | None = None
        if config.neutral_inventory is not None:
            assert config.mcc is not None
            geometry = config.grid.geometry
            self.neutrals = NeutralInventory(
                config.neutral_inventory,
                ceiling_density_per_m3=config.mcc.neutral_density_per_m3,
                exit_area_m2=pi * geometry.exit_radius_m**2,
                temperature_k=config.mcc.neutral_temperature_k,
                volume_m3=float(self.masks.to_dict()["plasma_volume_m3"]),
            )
            self.neutral_state = NeutralState.initial(config.mcc.neutral_density_per_m3)
            self.backend.set_neutral_scale(1.0)

    @property
    def state(self) -> SimulationState:
        state = self.backend.export_state()
        state.neutral = None if self.neutral_state is None else self.neutral_state.copy()
        return state

    def load_state(self, state: SimulationState) -> None:
        """Load a (checkpoint) state and re-base the interval bookkeeping on it.

        The dynamical state resumes bitwise; the series intervals restart here, so
        the first record after a resume reports currents over the interval since
        the checkpoint (not since step 0) and a zero interval residual / electrode
        work (no previous energy sample to difference against).
        """

        self.backend.load_state(state)
        self._last_cumulative = {key: float(state.cumulative.get(key, 0.0)) for key in CUMULATIVE_KEYS}
        self._last_energy = None
        self._last_electrode = None
        self._series_base_step = int(state.step)
        if self.neutrals is not None:
            if state.neutral is None:
                raise PIC2DValidationError("state has no neutral inventory but the configuration enables one")
            self.neutral_state = state.neutral.copy()
            self.backend.set_neutral_scale(self.neutrals.scale(self.neutral_state))
        elif state.neutral is not None:
            raise PIC2DValidationError("state carries a neutral inventory but the configuration is a static background")

    def run(self, steps: int, *, accumulate_from_step: int | None = None, progress: Any = None) -> SimulationState:
        """Advance ``steps`` cycles; accumulate diagnostics from ``accumulate_from_step`` on."""

        config = self.config
        start = self.backend.step_index
        target = start + steps
        window_start = target if accumulate_from_step is None else accumulate_from_step
        tally: StepTally | None = None
        for step in range(start, target):
            accumulate = step >= window_start
            fresh = self.backend.step(accumulate)
            if fresh is not None:
                tally = fresh
            record = (step + 1) % config.series_interval_steps == 0 or step + 1 == target
            if record:
                tally = self.backend.flush() or tally
            if (
                tally is not None and (step + 1) % config.runtime_stability_check_steps == 0
                and tally.max_omega_pe_dt > config.limits.max_omega_pe_dt
            ):
                raise PIC2DStabilityError(
                    f"observed peak omega_pe*dt = {tally.max_omega_pe_dt:.3g} exceeds {config.limits.max_omega_pe_dt}"
                )
            if record:
                assert tally is not None
                self._record(tally)
                if progress is not None:
                    progress(self.series[-1])
        return self.backend.export_state()

    def _record(self, tally: StepTally) -> None:
        sample = self.backend.series_sample()
        masks = self.masks
        config = self.config
        k_e = float(sample["kinetic_electron_j"])
        k_i = float(sample["kinetic_ion_j"])
        phi = sample["phi_v"]
        cumulative = sample["cumulative"]
        u_e = field_energy_j(masks, phi)
        step = int(sample["step"])
        interval_steps = step - (self.series[-1].step if self.series else self._series_base_step)
        interval = max(interval_steps, 1) * config.dt_s
        current_unit = ELEMENTARY_CHARGE_C * config.macro_weight / interval
        delta = {key: cumulative[key] - self._last_cumulative[key] for key in CUMULATIVE_KEYS}
        # Electrode work: the source holding electrode k at V_k supplies
        # dQ_src = dQ_induced - q_absorbed; its energy V_k dQ_src closes the ledger
        # (Gauss: the induced charge is the conductor charge, A phi on Dirichlet nodes).
        q_anode, q_exit = induced_electrode_charge_c(masks, phi)
        quantum = ELEMENTARY_CHARGE_C * config.macro_weight
        absorbed_anode_c = quantum * (delta["anode_ions"] - delta["anode_electrons"])
        absorbed_exit_c = quantum * (delta["exit_ions"] - delta["exit_electrons"])
        if self._last_electrode is None:
            electrode_work = 0.0
        else:
            electrode_work = (
                config.potentials.anode_v * ((q_anode - self._last_electrode[0]) - absorbed_anode_c)
                + config.potentials.exit_v * ((q_exit - self._last_electrode[1]) - absorbed_exit_c)
            )
        self._last_electrode = (q_anode, q_exit)
        currents = {
            "anode_electron_a": delta["anode_electrons"] * current_unit,
            "anode_ion_a": delta["anode_ions"] * current_unit,
            "discharge_a": (delta["anode_electrons"] - delta["anode_ions"]) * current_unit,
            "exit_electron_a": delta["exit_electrons"] * current_unit,
            "exit_ion_beam_a": delta["exit_ions"] * current_unit,
            "wall_electron_a": delta["wall_electrons"] * current_unit,
            "wall_ion_a": delta["wall_ions"] * current_unit,
            "injected_electron_a": delta["injected_electrons"] * current_unit,
            "ionization_rate_per_s": delta["ionizations"] * config.macro_weight / interval,
        }
        total = k_e + k_i + u_e
        sources = (
            delta["ke_injected_j"] - delta["ke_absorbed_anode_j"] - delta["ke_absorbed_exit_j"]
            - delta["ke_absorbed_wall_j"] - delta["inelastic_loss_j"] + delta["ke_born_ions_j"] + electrode_work
        )
        residual = 0.0 if self._last_energy is None else (total - self._last_energy) - sources
        ledger = {
            "total_energy_j": total,
            "interval_sources_j": sources,
            "interval_residual_j": residual,
            "interval_field_work_j": delta["field_work_j"],
            "interval_electrode_work_j": electrode_work,
            "anode_induced_charge_c": q_anode,
            "exit_induced_charge_c": q_exit,
            "cumulative": dict(cumulative),
        }
        plasma_phi = phi[masks.plasma_node]
        neutral: dict[str, Any] | None = None
        if self.neutrals is not None and self.neutral_state is not None:
            # v1.3: advance the inventory with the ionisation measured over this interval,
            # then hand the new n_g / n_g0 to the MCC for the next interval (fails closed
            # on exhaustion or on exceeding the null-collision ceiling).
            advance = self.neutrals.advance(self.neutral_state, currents["ionization_rate_per_s"], interval)
            self.neutral_state = advance.state
            self.backend.set_neutral_scale(self.neutrals.scale(self.neutral_state))
            neutral = {
                "density_per_m3": advance.state.density_per_m3,
                "fixed_point_per_m3": advance.fixed_point_per_m3,
                "scale": self.neutrals.scale(advance.state),
                "ionization_rate_per_s": advance.ionization_rate_per_s,
                "effusion_rate_per_s": advance.effusion_rate_per_s,
                "artificial_rate_per_s": advance.artificial_rate_per_s,
                "feed_atoms_per_s": self.neutrals.config.feed_atoms_per_s,
                "interval_ledger_residual_atoms": advance.ledger_residual_atoms,
                "ledger": dict(advance.state.ledger),
            }
        self.series.append(
            SeriesRecord(
                step, float(sample["time_s"]), int(sample["electrons"]), int(sample["ions"]),
                float(plasma_phi.mean()), float(plasma_phi.min()), float(plasma_phi.max()),
                k_e, k_i, u_e, float(sample["surface_charge_c"]), tally.max_omega_pe_dt,
                tally.poisson_iterations, currents, ledger, neutral,
            )
        )
        self._last_cumulative = dict(cumulative)
        self._last_energy = total

    def diagnostic_arrays(self) -> dict[str, np.ndarray]:
        return self.backend.diagnostic_arrays()

    def to_provenance(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "config": self.config.to_dict(),
            "mesh": self.masks.to_dict(),
            "field": self.field.to_dict(),
            "stability_gate": self.stability.to_dict(),
            "backend": getattr(self.backend, "name", self.backend_name),
        }
        if self.cross_sections is not None:
            record["cross_sections"] = self.cross_sections.to_dict()
        if getattr(self.backend, "mcc", None) is not None:
            record["mcc"] = self.backend.mcc.to_dict()
        if self.neutrals is not None:
            record["neutral_inventory"] = self.neutrals.to_dict()
        return record


__all__ = [
    "CPUBackend",
    "CUMULATIVE_KEYS",
    "DiagnosticAccumulator",
    "InjectionConfig",
    "PIC2DConfig",
    "SeedPlasmaConfig",
    "SeriesRecord",
    "Simulation",
    "SimulationState",
    "StepTally",
    "empty_cumulative",
    "instantaneous_maps",
    "seed_plasma_state",
]
