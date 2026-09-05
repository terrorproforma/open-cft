"""L2 v2: per-cell hybrid (kinetic Xe+, per-cell Boltzmann electron fluid, self-consistent field).

One step ``n -> n+1`` (``dt`` of order 1 ns; the electron time scales are not resolved):

1. deposit the kinetic ions (``x^n``) -> node charge (Gauss source form);
2. per-cell Poisson-Boltzmann field step (``pb_solver``): ``phi^n``, ``n_e^n``, the implicit
   electron wall deposit ``sigma^(n+1/2)`` and the Boltzmann electron wall fluxes;
3. ``E^n = -grad phi^n`` on the nodes; Boris/leapfrog ion push ``v^(n-1/2) -> v^(n+1/2)``,
   ``x^n -> x^(n+1)``; absorbed ions deposit their charge on the wall (``sigma^(n+1)``) or count
   as anode / beam current;
4. electron fluid update per cell: cusp currents from the cusp-conductance closure
   ``e F_k = G_k [ (phi_k - phi_k+1) - (p_k - p_k+1) / (e n_k,k+1) ]`` (generalised Ohm's law
   across the cusp with the declared conductance), Boltzmann thermal fluxes to the exit
   plane and the anode, the wall fluxes of step 2, ionisation ``S_k = n_g k_iz(T_k) N_k`` and
   excitation with the Maxwellian rates of the PIC's cross sections; counts and thermal
   energies advanced explicitly (Euler);
5. ion births sampled from the node-resolved ionisation source at the neutral temperature;
6. the 0-D neutral inventory (PIC model v1.3 closure, ``pic2d.neutrals``) advanced with the
   step's ionisation rate;
7. ledgers: charge (plasma + wall + induced = 0 every step), atoms (inventory ledger),
   energy (electrode work from induced + absorbed electrode charge against the kinetic /
   thermal / field energy change and every boundary and inelastic loss; the residual is
   reported and gated one-sided like the PIC's v2.0.3 windowed residual).

Everything is CPU numpy.  The model is a development/screening model: see
``modern/docs/hybrid-l2-v2.md`` for the claim boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, sqrt
from typing import Any

import numpy as np

from ..pic2d.fields import MagneticFieldMap
from ..pic2d.mcc import XenonCrossSections
from ..pic2d.mesh import MeshMasks, build_mesh_masks
from ..pic2d.models import (
    ELEMENTARY_CHARGE_C,
    EV_J,
    BoundaryPotentials,
    Grid2D,
    PoissonConfig2D,
    xenon_ion_species,
)
from ..pic2d.neutrals import NeutralInventory, NeutralInventoryConfig, NeutralState
from ..pic2d.poisson import Poisson2D, electric_field_nodes, field_energy_j
from .cells import CellPartition
from .ions import IonPopulation, births_this_step, sample_births, uniform_seed_ions
from .models import HybridValidationError
from .pb_solver import (
    HybridConvergenceError,
    PBConfig,
    PoissonBoltzmannSolver,
    electrode_face_areas,
    electron_mean_speed_m_per_s,
    wall_effective_areas,
)
from .rates import RateTable, build_rate_table

MODEL_VERSION = "hybrid-l2-v2.0.0"
CONVECTED_ENERGY_PER_ELECTRON_OVER_T = 2.0   # mean kinetic energy carried per electron across an interface / wall

CUMULATIVE_KEYS = (
    "injected_electrons", "exit_electrons", "anode_electrons", "wall_electrons", "ionizations", "excitations",
    "anode_ions", "exit_ions", "wall_ions", "born_ions",
    "ke_absorbed_anode_j", "ke_absorbed_exit_j", "ke_absorbed_wall_j", "ke_born_ions_j", "field_work_ions_j",
    "electron_energy_to_anode_j", "electron_energy_to_exit_j", "electron_energy_to_wall_j", "electron_energy_injected_j",
    "electron_joule_j", "electron_redistribution_j", "inelastic_loss_j", "electrode_work_j", "anode_absorbed_charge_c",
    "exit_absorbed_charge_c", "energy_residual_j",
)


@dataclass(frozen=True, slots=True)
class PlateauRule:
    min_transit_times: float = 3.0
    ion_transit_time_s: float = 2.4e-6
    threshold: float = 0.05
    window_fraction: float = 0.2

    def to_dict(self) -> dict[str, float]:
        return {"min_transit_times": self.min_transit_times, "ion_transit_time_s": self.ion_transit_time_s,
                "threshold": self.threshold, "window_fraction": self.window_fraction}


@dataclass(frozen=True, slots=True)
class HybridL2Config:
    grid: Grid2D
    potentials: BoundaryPotentials
    dt_s: float
    macro_weight: float
    seed: int
    injection_current_a: float
    injection_temperature_ev: float
    seed_density_per_m3: float
    seed_electron_temperature_ev: float
    neutral_ceiling_per_m3: float
    neutral_temperature_k: float
    neutral_inventory: NeutralInventoryConfig
    cusp_conductance_s: tuple[float, ...]
    leak_half_width_m: tuple[float, ...] = ()
    access_floor: float = 0.0
    pressure_term: bool = True
    pb: PBConfig = field(default_factory=PBConfig)
    series_interval_steps: int = 10
    averaging_window_steps: int = 500
    checkpoint_every_steps: int = 1000
    plateau: PlateauRule = PlateauRule()
    max_steps: int = 20000
    residual_window_steps: int = 500

    def __post_init__(self) -> None:
        for name in ("dt_s", "macro_weight", "injection_current_a", "injection_temperature_ev", "seed_density_per_m3",
                     "seed_electron_temperature_ev", "neutral_ceiling_per_m3", "neutral_temperature_k"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise HybridValidationError(f"{name} must be finite and positive")
        if any(not isfinite(g) or g < 0.0 for g in self.cusp_conductance_s):
            raise HybridValidationError("cusp conductances must be finite and non-negative")
        if self.leak_half_width_m and (len(self.leak_half_width_m) != len(self.cusp_conductance_s)
                                       or any(not isfinite(w) or w <= 0.0 for w in self.leak_half_width_m)):
            raise HybridValidationError("leak_half_width_m needs one positive entry per cusp (or none: every flux tube populated)")
        if not 0.0 <= self.access_floor <= 1.0:
            raise HybridValidationError("access_floor must lie in [0, 1]")
        if self.series_interval_steps < 1 or self.averaging_window_steps % self.series_interval_steps != 0:
            raise HybridValidationError("averaging_window_steps must be a positive multiple of series_interval_steps")
        if self.checkpoint_every_steps % self.series_interval_steps != 0:
            raise HybridValidationError("checkpoint_every_steps must be a multiple of series_interval_steps")
        if self.residual_window_steps % self.series_interval_steps != 0:
            raise HybridValidationError("residual_window_steps must be a multiple of series_interval_steps")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "grid": self.grid.to_dict(), "potentials": self.potentials.to_dict(), "dt_s": self.dt_s,
            "macro_weight": self.macro_weight, "seed": self.seed,
            "injection": {"electron_current_a": self.injection_current_a, "electron_temperature_ev": self.injection_temperature_ev},
            "seed_plasma": {"density_per_m3": self.seed_density_per_m3, "electron_temperature_ev": self.seed_electron_temperature_ev},
            "neutrals": {"ceiling_per_m3": self.neutral_ceiling_per_m3, "temperature_k": self.neutral_temperature_k,
                         "inventory": self.neutral_inventory.to_dict()},
            "closure": {"cusp_conductance_s": list(self.cusp_conductance_s), "leak_half_width_m": list(self.leak_half_width_m),
                        "access_floor": self.access_floor, "pressure_term": self.pressure_term,
                        "convected_energy_per_electron_over_t": CONVECTED_ENERGY_PER_ELECTRON_OVER_T},
            "poisson_boltzmann": self.pb.to_dict(),
            "series_interval_steps": self.series_interval_steps, "averaging_window_steps": self.averaging_window_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps, "residual_window_steps": self.residual_window_steps,
            "plateau": self.plateau.to_dict(), "max_steps": self.max_steps,
        }


@dataclass(slots=True)
class HybridL2State:
    step: int
    time_s: float
    ions: IonPopulation
    surface_charge_c: np.ndarray
    phi_v: np.ndarray
    log_reference: np.ndarray
    electron_count: np.ndarray
    electron_energy_j: np.ndarray
    neutral: NeutralState
    birth_carry: float
    cumulative: dict[str, float]
    rng: np.random.Generator

    @property
    def electron_temperature_ev(self) -> np.ndarray:
        return self.electron_energy_j / (1.5 * EV_J * self.electron_count)


# -- plateau rule (the PIC's) ------------------------------------------------------------------------------------------

def trailing_time_drift(time_s: np.ndarray, values: np.ndarray, fraction: float) -> float | None:
    """Relative drift ``slope * window / |mean|`` of a linear fit over the trailing ``fraction`` of the elapsed time."""

    if time_s.size < 8:
        return None
    t_end = float(time_s[-1])
    mask = time_s >= t_end - fraction * t_end
    if int(mask.sum()) < 8:
        return None
    x = time_s[mask].astype(np.float64)
    y = values[mask].astype(np.float64)
    mean = float(np.mean(y))
    if not np.isfinite(mean) or abs(mean) < 1e-300:
        return None
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    return slope * float(x[-1] - x[0]) / abs(mean)


def evaluate_plateau(time_s: np.ndarray, discharge_a: np.ndarray, electrons: np.ndarray, neutral_density: np.ndarray,
                     rule: PlateauRule) -> dict[str, Any]:
    elapsed = float(time_s[-1]) if time_s.size else 0.0
    transits = elapsed / rule.ion_transit_time_s
    drifts = {
        "discharge_current_drift": trailing_time_drift(time_s, discharge_a, rule.window_fraction),
        "electron_count_drift": trailing_time_drift(time_s, electrons, rule.window_fraction),
        "neutral_density_drift": trailing_time_drift(time_s, neutral_density, rule.window_fraction),
    }
    ok = all(v is not None and abs(v) < rule.threshold for v in drifts.values())
    return {"reached": bool(ok and transits >= rule.min_transit_times), "drifts_within_threshold": bool(ok),
            "transit_times_elapsed": transits, "min_transit_times": rule.min_transit_times, **drifts,
            "threshold": rule.threshold, "window_fraction": rule.window_fraction, "tracked": sorted(drifts)}


# -- window accumulators (maps) ------------------------------------------------------------------------------------------

@dataclass(slots=True)
class WindowAccumulator:
    start_step: int
    steps: int
    n_e: np.ndarray
    n_i: np.ndarray
    phi: np.ndarray
    t_e: np.ndarray
    ionization: np.ndarray
    wall_ion_hits: np.ndarray
    wall_ion_energy_j: np.ndarray
    wall_electron: np.ndarray            # electrons (count) per axial cell
    wall_electron_energy_j: np.ndarray
    exit_ion_hits: np.ndarray
    exit_electrons: np.ndarray           # electrons (count) per radial cell

    @classmethod
    def empty(cls, grid: Grid2D, start_step: int) -> WindowAccumulator:
        nr, nz = grid.cell_shape
        node = grid.node_shape
        return cls(start_step, 0, np.zeros(node), np.zeros(node), np.zeros(node), np.zeros(node), np.zeros(node),
                   np.zeros(nz), np.zeros(nz), np.zeros(nz), np.zeros(nz), np.zeros(nr), np.zeros(nr))


def flux_function_from_field(grid: Grid2D, b_z_t: np.ndarray) -> np.ndarray:
    """``psi(r, z) = int_0^r B_z r' dr'`` (Wb / 2 pi) by cumulative trapezoid on the node grid; ``psi = 0`` on the axis."""

    r = grid.r_m
    integrand = b_z_t * r[:, None]
    psi = np.zeros(grid.node_shape, dtype=np.float64)
    psi[1:, :] = np.cumsum(0.5 * (integrand[:-1, :] + integrand[1:, :]) * grid.dr_m, axis=0)
    return psi


def populated_flux_tubes(
    grid: Grid2D, masks: MeshMasks, psi: np.ndarray, node_cell: np.ndarray, partition: CellPartition, leak_half_width_m: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Node mask of the electron-populated flux tubes and the per-cell |psi| thresholds.

    Without leak widths every plasma node is populated (the pure per-cell Boltzmann model).  With them, the wall
    flux ``|psi_w(z)|`` of the straight bore is read at ``z_c +- w`` for every cusp; a cell's threshold is the
    largest such value among its cusps, and a node is populated when ``|psi| <= threshold`` of its cell.
    """

    plasma = masks.plasma_node
    k_cells = partition.cell_count
    thresholds = np.full(k_cells, np.inf)
    if leak_half_width_m:
        bore_index = round(grid.geometry.bore_radius_m / grid.dr_m)
        wall_psi = np.abs(psi[bore_index, :])
        z = grid.z_m
        per_cusp = []
        for z_c, w in zip(partition.cusp_z_m, leak_half_width_m, strict=True):
            edges = [float(np.interp(z_c - w, z, wall_psi)), float(np.interp(z_c + w, z, wall_psi))]
            per_cusp.append(max(edges))
        for k in range(k_cells):
            candidates = [per_cusp[c] for c in (k - 1, k) if 0 <= c < len(per_cusp)]
            thresholds[k] = max(candidates) if candidates else np.inf
    populated = plasma.copy()
    for k in range(k_cells):
        in_cell = plasma & (node_cell == k)
        populated[in_cell] = np.abs(psi[in_cell]) <= thresholds[k]
    if not populated.any():
        raise HybridValidationError("no populated flux tube: leak windows too narrow for this grid")
    return populated, thresholds


def wall_cell_area_m2(grid: Grid2D) -> np.ndarray:
    """``2 pi r_w(z) dz x slant`` per axial cell of the channel (the extraction's convention)."""

    geometry = grid.geometry
    nz = grid.axial_cells
    z_cells = geometry.z_min_m + (np.arange(nz) + 0.5) * grid.dz_m
    radius = np.asarray(geometry.wall_radius_m(z_cells))
    if geometry.cone_start_z_m < geometry.z_max_m:
        slope = (geometry.exit_radius_m - geometry.bore_radius_m) / (geometry.z_max_m - geometry.cone_start_z_m)
        slant = np.where(z_cells > geometry.cone_start_z_m, sqrt(1.0 + slope * slope), 1.0)
    else:
        slant = np.ones(nz)
    return 2.0 * pi * radius * grid.dz_m * slant


# -- the simulation --------------------------------------------------------------------------------------------------------

class HybridL2Simulation:
    def __init__(
        self,
        config: HybridL2Config,
        field: MagneticFieldMap,
        cross_sections: XenonCrossSections,
        partition: CellPartition,
        *,
        rate_table: RateTable | None = None,
    ) -> None:
        if field.grid != config.grid:
            raise HybridValidationError("the field map must be sampled on the configuration grid")
        if len(config.cusp_conductance_s) != partition.cell_count - 1:
            raise HybridValidationError("one cusp conductance per cell interface is required")
        if abs(partition.z_min_m - config.grid.geometry.z_min_m) > 1e-12 or abs(partition.z_max_m - config.grid.geometry.domain_z_max_m) > 1e-12:
            raise HybridValidationError("the cell partition must span the simulated domain")
        self.config = config
        self.field = field
        self.cross_sections = cross_sections
        self.partition = partition
        self.masks: MeshMasks = build_mesh_masks(config.grid)
        self.node_cell = partition.node_cells(config.grid)
        self.rates = build_rate_table(cross_sections) if rate_table is None else rate_table
        if self.rates.cross_section_payload_sha256 != cross_sections.payload_sha256:
            raise HybridValidationError("the rate table is not bound to the given cross sections")
        area_r, area_z, effective, access_r = wall_effective_areas(self.masks, field.b_r_t, field.b_z_t, access_floor=config.access_floor)
        self.wall_area_r, self.wall_area_z, self.wall_effective_area, self.wall_access_r = area_r, area_z, effective, access_r
        self.wall_node = (area_r + area_z) > 0.0
        anode_area, exit_area = electrode_face_areas(self.masks)
        magnitude = np.hypot(field.b_r_t, field.b_z_t)
        with np.errstate(invalid="ignore", divide="ignore"):
            access_z = np.where(magnitude > 0.0, np.abs(field.b_z_t) / magnitude, 1.0)
        access_z = np.maximum(access_z, config.access_floor)
        self.anode_effective_area = anode_area * access_z
        self.exit_effective_area = exit_area * access_z
        # flux-tube population: electrons live on the flux tubes whose dielectric footprint lies inside a cusp leak
        # window (|psi| <= |psi_wall(z_c +- w)|); the other tubes (wall-to-wall arcs between cusps) are depleted
        self.flux_function_wb = flux_function_from_field(config.grid, field.b_z_t)
        self.populated_node, self.population_threshold_wb = populated_flux_tubes(
            config.grid, self.masks, self.flux_function_wb, self.node_cell, partition, config.leak_half_width_m,
        )
        self.solver = PoissonBoltzmannSolver(
            self.masks, self.node_cell, wall_effective_area_m2=effective,
            electrode_effective_area_m2=self.anode_effective_area + self.exit_effective_area,
            populated_node=self.populated_node, config=config.pb,
        )
        self.anode_effective_area = np.where(self.populated_node, self.anode_effective_area, 0.0)
        self.exit_effective_area = np.where(self.populated_node, self.exit_effective_area, 0.0)
        self.cell_masks = self.solver.cell_masks
        self.cell_volume_m3 = self.solver.cell_volume_m3
        self.species = xenon_ion_species(config.macro_weight)
        geometry = config.grid.geometry
        self.neutrals = NeutralInventory(
            config.neutral_inventory, ceiling_density_per_m3=config.neutral_ceiling_per_m3,
            exit_area_m2=pi * geometry.exit_radius_m**2, temperature_k=config.neutral_temperature_k,
            volume_m3=self.masks.plasma_volume_m3, mass_kg=self.species.mass_kg,
        )
        self.wall_cell_area = wall_cell_area_m2(config.grid)
        # wall nodes -> axial cell index (for the per-axial-cell wall electron flux map)
        nz = config.grid.axial_cells
        self.node_axial_cell = np.minimum(np.broadcast_to(np.arange(nz + 1)[None, :], config.grid.node_shape), nz - 1)
        nr = config.grid.radial_cells
        self.node_radial_cell = np.minimum(np.broadcast_to(np.arange(nr + 1)[:, None], config.grid.node_shape), nr - 1)
        self.thresholds_ev = self.rates.thresholds_ev
        self.state = self._initial_state()
        self.series: list[dict[str, Any]] = []
        self.window = WindowAccumulator.empty(config.grid, 0)
        self.completed_window: WindowAccumulator | None = None
        self._interval = self._zero_interval()
        self._pending: dict[str, float] | None = None
        self._previous_density: np.ndarray | None = None
        self._previous_cell_phi: np.ndarray | None = None
        self.last_field = None
        self.stop_reason: str | None = None
        self.field_energy_j = 0.0

    # -- initial state ---------------------------------------------------------------------------------------------------

    def _initial_state(self) -> HybridL2State:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        ions = IonPopulation(self.species, uniform_seed_ions(
            self.masks, cfg.seed_density_per_m3, cfg.macro_weight, rng, mass_kg=self.species.mass_kg, temperature_k=cfg.neutral_temperature_k,
            accept_node=self.populated_node, accept_volume_m3=float(self.cell_volume_m3.sum()),
        ))
        count = cfg.seed_density_per_m3 * self.cell_volume_m3
        energy = 1.5 * count * cfg.seed_electron_temperature_ev * EV_J
        neutral = NeutralState.initial(self.neutrals.initial_density)
        cumulative = {key: 0.0 for key in CUMULATIVE_KEYS}
        # warm start: the vacuum (Laplace) potential between the electrodes
        phi = Poisson2D(self.masks, PoissonConfig2D(method="direct")).solve(np.zeros(cfg.grid.node_shape), cfg.potentials).phi_v
        return HybridL2State(0, 0.0, ions, np.zeros(cfg.grid.node_shape), phi, np.zeros(self.partition.cell_count),
                             count, energy, neutral, 0.0, cumulative, rng)

    def _zero_interval(self) -> dict[str, float]:
        keys = ("anode_electrons", "anode_ions", "exit_electrons", "exit_ions", "injected_electrons", "wall_electrons", "wall_ions",
                "ionizations", "excitations", "electrode_work_j", "energy_residual_j", "field_work_ions_j", "electron_joule_j",
                "sources_j", "newton_iterations", "factorisations", "steps")
        return {key: 0.0 for key in keys}

    # -- one step --------------------------------------------------------------------------------------------------------

    def _cusp_fluxes(self, cell_phi: np.ndarray, count: np.ndarray, temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Electron flux (electrons/s) from cell k+1 into cell k through cusp k and its driving voltage."""

        cfg = self.config
        k_cells = self.partition.cell_count
        flux = np.zeros(max(k_cells - 1, 0))
        drive = np.zeros(max(k_cells - 1, 0))
        density = count / self.cell_volume_m3
        for k in range(k_cells - 1):
            d = cell_phi[k] - cell_phi[k + 1]
            if cfg.pressure_term:
                mean_density = 0.5 * (density[k] + density[k + 1])
                d -= (density[k] * temperature[k] - density[k + 1] * temperature[k + 1]) / mean_density
            drive[k] = d
            flux[k] = cfg.cusp_conductance_s[k] * d / ELEMENTARY_CHARGE_C
        return flux, drive

    def step(self) -> dict[str, Any]:
        cfg = self.config
        st = self.state
        masks = self.masks
        dt = cfg.dt_s
        e = ELEMENTARY_CHARGE_C
        k_cells = self.partition.cell_count
        anode_cell, exit_cell = 0, k_cells - 1
        temperature = st.electron_temperature_ev
        if not np.isfinite(temperature).all() or np.any(temperature <= 0.0):
            raise HybridConvergenceError("electron temperature is nonfinite or non-positive")
        n_g = st.neutral.density_per_m3
        k_iz = self.rates.rate("ionization", temperature)
        k_exc = self.rates.rate("excitation", temperature)
        # 1. explicit electron sources: ionisation, cusp currents (previous solve's cell potentials), injection
        ionisation_cell = n_g * k_iz * st.electron_count
        excitation_cell = n_g * k_exc * st.electron_count
        ionisation_total = float(ionisation_cell.sum())
        injected = cfg.injection_current_a / e
        if self._previous_cell_phi is None:
            flux = np.zeros(max(k_cells - 1, 0))
            drive = np.zeros(max(k_cells - 1, 0))
        else:
            flux, drive = self._cusp_fluxes(self._previous_cell_phi, st.electron_count, temperature)
        target = st.electron_count + dt * ionisation_cell
        for k in range(k_cells - 1):
            target[k] += dt * flux[k]
            target[k + 1] -= dt * flux[k]
        target[exit_cell] += dt * injected
        if np.any(target <= 0.0) or not np.isfinite(target).all():
            raise HybridConvergenceError("electron count target of a cell is not positive (fail closed)")
        # 2. ion deposit and the field step (implicit electron losses to the wall and the electrodes)
        ion_charge = st.ions.deposit_charge_c(masks)
        ion_source = ion_charge * masks.charge_to_source
        result = self.solver.solve(
            ion_source_c=ion_source, surface_charge_c=st.surface_charge_c, temperature_ev=temperature, count=target,
            potentials=cfg.potentials, dt_s=dt, initial_phi_v=st.phi_v,
        )
        phi = result.phi_v
        n_e = result.electron_density_per_m3
        self.last_field = result
        self.field_energy_j = field_energy_j(masks, phi)
        # close the PREVIOUS step's energy ledger now that its final field (phi^n) is known: the electrode work is
        # phi_e (dQ_induced,e - dQ_absorbed,e) between the two solves, exact for the discrete Gauss law.
        energy_now = st.ions.kinetic_energy_j() + float(st.electron_energy_j.sum()) + self.field_energy_j
        pending = self._pending
        if pending is not None:
            electrode_work = (cfg.potentials.anode_v * (result.anode_induced_charge_c - pending["anode_induced_c"] - pending["anode_absorbed_c"])
                              + cfg.potentials.exit_v * (result.exit_induced_charge_c - pending["exit_induced_c"] - pending["exit_absorbed_c"]))
            sources = pending["sources_j"] + electrode_work
            residual = (energy_now - pending["energy_before_j"]) - sources
            st.cumulative["electrode_work_j"] += electrode_work
            st.cumulative["energy_residual_j"] += residual
            self._interval["electrode_work_j"] += electrode_work
            self._interval["energy_residual_j"] += residual
            self._interval["sources_j"] += sources
        energy_before = energy_now
        # 3. ions
        e_r, e_z = electric_field_nodes(masks, phi)
        tally = st.ions.push(masks, e_r_nodes=e_r, e_z_nodes=e_z, b_r_nodes=self.field.b_r_t, b_z_nodes=self.field.b_z_t, dt_s=dt,
                             partition=self.partition)
        surface_after = result.surface_charge_c + tally.surface_deposit_c
        # 4. electron fluid energies (explicit; counts came out of the implicit solve)
        volume = self.solver.volume
        weight_node = n_e * volume
        cell_phi = np.array([float(np.sum(phi[m] * weight_node[m]) / max(float(weight_node[m].sum()), 1e-300)) for m in self.cell_masks])
        new_count = result.remaining_count
        wall_flux_node = result.wall_electron_flux_per_s
        wall_loss = np.array([float(wall_flux_node[m].sum()) for m in self.cell_masks])
        anode_flux = float(result.anode_flux_per_s.sum())
        exit_flux = float(result.exit_flux_per_s.sum())
        # field work of the intra-cell Boltzmann redistribution since the previous solve (exact bookkeeping form
        # e sum (phi_n - phi_k) dn_e,n V_n; zero for a pure rescaling of the cell's density, so births and fluxes
        # do not enter); one-step lag, credited to the cell's thermal energy in this step
        redistribution = np.zeros(k_cells)
        if self._previous_density is not None and self._previous_cell_phi is not None:
            # midpoint potential of the interval and the SAME cell reference the interval's transfers used
            d_density = (n_e - self._previous_density) * volume
            phi_mid = 0.5 * (phi + st.phi_v)
            for k, m in enumerate(self.cell_masks):
                redistribution[k] = e * float(np.sum((phi_mid[m] - self._previous_cell_phi[k]) * d_density[m]))
        self._previous_density = n_e
        d_energy = np.zeros(k_cells)
        joule = 0.0
        for k in range(k_cells - 1):
            f = flux[k]
            src, dst = (k + 1, k) if f >= 0.0 else (k, k + 1)
            carried = abs(f) * CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[src] * EV_J
            work = abs(f) * (cell_phi[dst] - cell_phi[src]) * e
            d_energy[src] -= carried
            d_energy[dst] += carried + work
            joule += work
        injected_energy = injected * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * cfg.injection_temperature_ev + (cell_phi[exit_cell] - cfg.potentials.exit_v)) * EV_J
        d_energy[exit_cell] += injected_energy
        joule += injected * (cell_phi[exit_cell] - cfg.potentials.exit_v) * e
        # boundary losses: the electron gas loses the escaping electrons' thermal energy (2 T) plus the sheath it
        # climbs (repelling boundary); the BOUNDARY receives 2 T plus the fall through an attracting boundary
        # (that energy comes from the field) - the two differ in sign of the potential step
        exit_step = cell_phi[exit_cell] - cfg.potentials.exit_v
        anode_step = cell_phi[anode_cell] - cfg.potentials.anode_v
        exit_loss_energy = exit_flux * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[exit_cell] + max(exit_step, 0.0)) * EV_J
        anode_loss_energy = anode_flux * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[anode_cell] + max(anode_step, 0.0)) * EV_J
        exit_sink_energy = exit_flux * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[exit_cell] + max(-exit_step, 0.0)) * EV_J
        anode_sink_energy = anode_flux * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[anode_cell] + max(-anode_step, 0.0)) * EV_J
        d_energy[exit_cell] -= exit_loss_energy
        d_energy[anode_cell] -= anode_loss_energy
        joule -= (exit_flux * max(exit_step, 0.0) + anode_flux * max(anode_step, 0.0)) * e
        wall_energy = np.zeros(k_cells)
        wall_sink = np.zeros(k_cells)
        for k, m in enumerate(self.cell_masks):
            step_v = cell_phi[k] - phi[m]
            climb = np.maximum(step_v, 0.0)
            fall = np.maximum(-step_v, 0.0)
            wall_energy[k] = float(np.sum(wall_flux_node[m] * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[k] + climb))) * EV_J
            wall_sink[k] = float(np.sum(wall_flux_node[m] * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[k] + fall))) * EV_J
            joule -= float(np.sum(wall_flux_node[m] * climb)) * e
        d_energy -= wall_energy
        inelastic = (ionisation_cell * self.thresholds_ev[2] + excitation_cell * self.thresholds_ev[1]) * EV_J
        d_energy -= inelastic
        new_energy = st.electron_energy_j + dt * d_energy + redistribution
        if np.any(new_energy <= 0.0) or not np.isfinite(new_energy).all():
            raise HybridConvergenceError("electron thermal energy of a cell was exhausted (fail closed)")
        # 5. births (positions follow the cell's Boltzmann density; totals equal the explicit ionisation rates)
        expected = ionisation_total * dt / cfg.macro_weight
        n_births, carry = births_this_step(expected, st.birth_carry)
        source_node = np.zeros(cfg.grid.node_shape)
        for k, m in enumerate(self.cell_masks):
            total_weight = float(weight_node[m].sum())
            if total_weight > 0.0:
                source_node[m] = ionisation_cell[k] * weight_node[m] / total_weight      # ionisations per second per node
        born_ke = 0.0
        if n_births > 0:
            newborn = sample_births(masks, source_node, n_births, st.rng, mass_kg=self.species.mass_kg, temperature_k=cfg.neutral_temperature_k)
            born_ke = 0.5 * self.species.mass_kg * float(newborn.speed_squared().sum()) * cfg.macro_weight
            st.ions.add(newborn)
        # 6. neutrals
        advance = self.neutrals.advance(st.neutral, ionisation_total, dt)
        # 7. ledgers
        w = cfg.macro_weight
        absorbed_anode_c = e * w * tally.anode - e * anode_flux * dt
        absorbed_exit_c = e * w * tally.exit - e * (exit_flux - injected) * dt
        cum = st.cumulative
        cum["injected_electrons"] += injected * dt
        cum["exit_electrons"] += exit_flux * dt
        cum["anode_electrons"] += anode_flux * dt
        cum["wall_electrons"] += float(wall_loss.sum()) * dt
        cum["ionizations"] += ionisation_total * dt
        cum["excitations"] += float(excitation_cell.sum()) * dt
        cum["anode_ions"] += w * tally.anode
        cum["exit_ions"] += w * tally.exit
        cum["wall_ions"] += w * tally.wall
        cum["born_ions"] += w * n_births
        cum["ke_absorbed_anode_j"] += tally.ke_anode_j
        cum["ke_absorbed_exit_j"] += tally.ke_exit_j
        cum["ke_absorbed_wall_j"] += tally.ke_wall_j
        cum["ke_born_ions_j"] += born_ke
        cum["field_work_ions_j"] += tally.field_work_j
        cum["electron_energy_to_anode_j"] += anode_sink_energy * dt
        cum["electron_energy_to_exit_j"] += exit_sink_energy * dt
        cum["electron_energy_to_wall_j"] += float(wall_sink.sum()) * dt
        cum["electron_energy_injected_j"] += injected * CONVECTED_ENERGY_PER_ELECTRON_OVER_T * cfg.injection_temperature_ev * EV_J * dt
        cum["electron_joule_j"] += joule * dt + float(redistribution.sum())
        cum["electron_redistribution_j"] += float(redistribution.sum())
        cum["inelastic_loss_j"] += float(inelastic.sum()) * dt
        cum["anode_absorbed_charge_c"] += absorbed_anode_c
        cum["exit_absorbed_charge_c"] += absorbed_exit_c
        # commit state
        st.surface_charge_c = surface_after
        st.phi_v = phi
        st.log_reference = result.log_boltzmann_reference
        st.electron_count = new_count
        st.electron_energy_j = new_energy
        st.neutral = advance.state
        st.birth_carry = carry
        st.step += 1
        st.time_s = st.step * dt
        self._previous_cell_phi = cell_phi
        # the energy sources / sinks of THIS step other than the electrode work (closed at the next solve)
        sources_no_electrode = (injected * CONVECTED_ENERGY_PER_ELECTRON_OVER_T * cfg.injection_temperature_ev * EV_J * dt + born_ke
                                - tally.ke_anode_j - tally.ke_exit_j - tally.ke_wall_j
                                - (anode_sink_energy + exit_sink_energy + float(wall_sink.sum())) * dt - float(inelastic.sum()) * dt)
        self._pending = {
            "energy_before_j": energy_before, "sources_j": sources_no_electrode, "anode_absorbed_c": absorbed_anode_c,
            "exit_absorbed_c": absorbed_exit_c, "anode_induced_c": result.anode_induced_charge_c, "exit_induced_c": result.exit_induced_charge_c,
        }
        interval = self._interval
        interval["anode_electrons"] += anode_flux * dt
        interval["anode_ions"] += w * tally.anode
        interval["exit_electrons"] += exit_flux * dt
        interval["exit_ions"] += w * tally.exit
        interval["injected_electrons"] += injected * dt
        interval["wall_electrons"] += float(wall_loss.sum()) * dt
        interval["wall_ions"] += w * tally.wall
        interval["ionizations"] += ionisation_total * dt
        interval["excitations"] += float(excitation_cell.sum()) * dt
        interval["field_work_ions_j"] += tally.field_work_j
        interval["electron_joule_j"] += joule * dt + float(redistribution.sum())
        interval["newton_iterations"] += result.newton_iterations
        interval["factorisations"] += result.factorisations
        interval["steps"] += 1
        # window accumulation (maps)
        win = self.window
        win.steps += 1
        win.n_e += n_e
        win.n_i += ion_charge / (e * np.where(masks.plasma_node, masks.shape_volume_m3, np.inf))
        win.phi += phi
        for k, m in enumerate(self.solver.cell_masks_all_plasma):
            win.t_e[m] += temperature[k]
        win.ionization += source_node / np.where(volume > 0.0, volume, np.inf)
        win.wall_ion_hits += tally.wall_hits_per_axial_cell
        win.wall_ion_energy_j += tally.wall_energy_per_axial_cell_j
        wall_e_cells = np.zeros(cfg.grid.axial_cells)
        np.add.at(wall_e_cells, self.node_axial_cell[self.wall_node], wall_flux_node[self.wall_node] * dt)
        win.wall_electron += wall_e_cells
        wall_e_energy = np.zeros(cfg.grid.axial_cells)
        for k, m in enumerate(self.cell_masks):
            wm = m & self.wall_node
            np.add.at(wall_e_energy, self.node_axial_cell[wm], wall_flux_node[wm] * dt * (CONVECTED_ENERGY_PER_ELECTRON_OVER_T * temperature[k]))
        win.wall_electron_energy_j += wall_e_energy * EV_J
        win.exit_ion_hits += tally.exit_hits_per_radial_cell
        exit_e = np.zeros(cfg.grid.radial_cells)
        exit_flux_node = 0.25 * electron_mean_speed_m_per_s(temperature[exit_cell]) * self.exit_effective_area * n_e
        np.add.at(exit_e, self.node_radial_cell[masks.exit_node], exit_flux_node[masks.exit_node] * dt)
        win.exit_electrons += exit_e
        if win.steps >= cfg.averaging_window_steps:
            self.completed_window = win
            self.window = WindowAccumulator.empty(cfg.grid, st.step)
        record: dict[str, Any] = {}
        if st.step % cfg.series_interval_steps == 0:
            record = self._record_series(result, cell_phi, new_count / self.cell_volume_m3, flux, drive, ionisation_cell, wall_loss, tally, n_g, advance)
        return record

    # -- series -------------------------------------------------------------------------------------------------------------

    def _record_series(self, result, cell_phi, cell_density, flux, drive, ionisation_cell, wall_loss, tally, n_g, advance) -> dict[str, Any]:
        cfg = self.config
        st = self.state
        e = ELEMENTARY_CHARGE_C
        interval = self._interval
        span = interval["steps"] * cfg.dt_s
        anode_e = e * interval["anode_electrons"] / span
        anode_i = e * interval["anode_ions"] / span
        record = {
            "step": st.step, "time_s": st.time_s, "ions": st.ions.count, "electrons": float(st.electron_count.sum()),
            "current_anode_electron_a": anode_e, "current_anode_ion_a": anode_i, "current_discharge_a": anode_e + anode_i,
            "current_exit_electron_a": e * interval["exit_electrons"] / span, "current_exit_ion_beam_a": e * interval["exit_ions"] / span,
            "current_injected_electron_a": e * interval["injected_electrons"] / span,
            "current_wall_electron_a": e * interval["wall_electrons"] / span, "current_wall_ion_a": e * interval["wall_ions"] / span,
            "current_ionization_rate_per_s": interval["ionizations"] / span, "excitation_rate_per_s": interval["excitations"] / span,
            "neutral_density_per_m3": st.neutral.density_per_m3, "neutral_fixed_point_per_m3": advance.fixed_point_per_m3,
            "neutral_ionization_rate_per_s": advance.ionization_rate_per_s, "neutral_effusion_rate_per_s": advance.effusion_rate_per_s,
            "neutral_artificial_rate_per_s": advance.artificial_rate_per_s, "neutral_interval_ledger_residual_atoms": advance.ledger_residual_atoms,
            "neutral_scale": self.neutrals.scale(st.neutral),
            "phi_max_v": float(st.phi_v.max()), "phi_min_v": float(st.phi_v[self.masks.plasma_node].min()), "phi_mean_v": float(st.phi_v[self.masks.plasma_node].mean()),
            "kinetic_ion_j": st.ions.kinetic_energy_j(), "thermal_electron_j": float(st.electron_energy_j.sum()), "field_energy_j": self.field_energy_j,
            "total_energy_j": st.ions.kinetic_energy_j() + float(st.electron_energy_j.sum()) + self.field_energy_j,
            "interval_electrode_work_j": interval["electrode_work_j"], "interval_residual_j": interval["energy_residual_j"],
            "interval_sources_j": interval["sources_j"], "interval_field_work_ions_j": interval["field_work_ions_j"],
            "interval_electron_joule_j": interval["electron_joule_j"],
            "surface_charge_c": float(st.surface_charge_c.sum()), "anode_induced_charge_c": result.anode_induced_charge_c,
            "exit_induced_charge_c": result.exit_induced_charge_c,
            "newton_iterations_mean": interval["newton_iterations"] / interval["steps"], "factorisations_mean": interval["factorisations"] / interval["steps"],
            "gauss_relative_residual": result.gauss_residual_c / max(result.gauss_source_norm_c, 1e-300),
            "constraint_residual_max": result.constraint_residual_max, "total_charge_identity_c": result.total_charge_identity_c,
            "peak_electron_density_per_m3": float(result.electron_density_per_m3.max()),
        }
        for k in range(self.partition.cell_count):
            record[f"cell{k}_electron_count"] = float(st.electron_count[k])
            record[f"cell{k}_temperature_ev"] = float(st.electron_temperature_ev[k])
            record[f"cell{k}_potential_v"] = float(cell_phi[k])
            record[f"cell{k}_mean_density_per_m3"] = float(cell_density[k])
            record[f"cell{k}_ionization_rate_per_s"] = float(ionisation_cell[k])
            record[f"cell{k}_wall_electron_loss_per_s"] = float(wall_loss[k])
            record[f"cell{k}_wall_ion_hits"] = float(tally.wall_hits_per_cell[k])
        for k in range(self.partition.cell_count - 1):
            record[f"cusp{k}_electron_current_a"] = float(e * flux[k])
            record[f"cusp{k}_drive_v"] = float(drive[k])
        self.series.append(record)
        self._interval = self._zero_interval()
        return record

    # -- driver ------------------------------------------------------------------------------------------------------------

    def series_arrays(self) -> dict[str, np.ndarray]:
        if not self.series:
            return {}
        keys = sorted(self.series[0])
        return {key: np.array([row[key] for row in self.series], dtype=np.float64) for key in keys}

    def plateau(self) -> dict[str, Any] | None:
        arrays = self.series_arrays()
        if not arrays:
            return None
        return evaluate_plateau(arrays["time_s"], arrays["current_discharge_a"], arrays["electrons"], arrays["neutral_density_per_m3"], self.config.plateau)

    def windowed_residual_over_electrode_work(self) -> dict[str, Any]:
        """Trailing ``residual_window_steps`` energy residual over the electrode work (one-sided heating gate, as PIC v2.0.3)."""

        arrays = self.series_arrays()
        if not arrays:
            return {"complete": False, "ratio": None}
        records = self.config.residual_window_steps // self.config.series_interval_steps
        if arrays["step"].size < records + 1:
            return {"complete": False, "ratio": None}
        residual = float(arrays["interval_residual_j"][-records:].sum())
        work = float(arrays["interval_electrode_work_j"][-records:].sum())
        return {"complete": True, "ratio": residual / work if work != 0.0 else None, "residual_j": residual, "electrode_work_j": work}

    def maps(self) -> tuple[dict[str, np.ndarray], str] | None:
        """Window-averaged maps in the PIC's ``maps.npz`` layout (last complete window, or the current one if >= half full)."""

        win = self.completed_window
        kind = "window_average"
        if self.window.steps >= self.config.averaging_window_steps // 2 and (win is None or self.window.start_step > win.start_step):
            win = self.window
            kind = "window_average_partial"
        if win is None or win.steps == 0:
            return None
        cfg = self.config
        steps = float(win.steps)
        span = steps * cfg.dt_s
        w = cfg.macro_weight
        r = cfg.grid.r_m
        exit_area = pi * (r[1:] ** 2 - r[:-1] ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            wall_ion_energy = np.where(win.wall_ion_hits > 0, win.wall_ion_energy_j / (w * EV_J * np.maximum(win.wall_ion_hits, 1e-300)), 0.0)
            wall_e_energy = np.where(win.wall_electron > 0, win.wall_electron_energy_j / (EV_J * np.maximum(win.wall_electron, 1e-300)), 0.0)
        maps = {
            "n_e_per_m3": win.n_e / steps, "n_i_per_m3": win.n_i / steps, "phi_v": win.phi / steps, "t_e_ev": win.t_e / steps,
            "ionization_rate_per_m3_s": win.ionization / steps,
            "wall_ion_flux_per_m2_s": win.wall_ion_hits * w / (self.wall_cell_area * span),
            "wall_ion_mean_energy_ev": wall_ion_energy,
            "wall_electron_flux_per_m2_s": win.wall_electron / (self.wall_cell_area * span),
            "wall_electron_mean_energy_ev": wall_e_energy,
            "exit_ion_current_density_a_per_m2": win.exit_ion_hits * w * ELEMENTARY_CHARGE_C / (exit_area * span),
            "exit_electron_current_density_a_per_m2": win.exit_electrons * ELEMENTARY_CHARGE_C / (exit_area * span),
            "window_steps": np.array([win.steps], dtype=np.int64),
        }
        return maps, kind

    def run(self, steps: int, *, on_record=None, stop_on_plateau: bool = True) -> str:
        """Advance up to ``steps`` steps; returns the stop reason."""

        for _ in range(steps):
            if self.state.step >= self.config.max_steps:
                self.stop_reason = "max_steps_reached"
                return self.stop_reason
            record = self.step()
            if record:
                if on_record is not None:
                    on_record(record)
                if stop_on_plateau:
                    plateau = self.plateau()
                    if plateau is not None and plateau["reached"]:
                        self.stop_reason = "plateau_reached_after_min_transit_times"
                        return self.stop_reason
        self.stop_reason = "requested_steps_completed"
        return self.stop_reason

    # -- provenance ----------------------------------------------------------------------------------------------------------

    def to_provenance(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "config": self.config.to_dict(),
            "partition": self.partition.to_dict(),
            "field": self.field.to_dict(),
            "cross_sections": self.cross_sections.to_dict(),
            "rates": self.rates.to_dict(),
            "mesh": self.masks.to_dict(),
            "neutral_inventory": self.neutrals.to_dict(),
            "wall": {"geometric_area_m2": float(self.wall_area_r.sum() + self.wall_area_z.sum()),
                     "effective_area_m2": float(self.wall_effective_area.sum()),
                     "populated_effective_area_m2": float(self.wall_effective_area[self.populated_node].sum()),
                     "access_floor": self.config.access_floor,
                     "access_rule": "|B.n|/|B| per wall face (radial faces: |B_r|/|B|; stair-step risers: |B_z|/|B|), floored"},
            "flux_tubes": {"populated_nodes": int(self.populated_node.sum()), "plasma_nodes": int(self.masks.plasma_node.sum()),
                           "population_threshold_wb": [None if not np.isfinite(t) else float(t) for t in self.population_threshold_wb],
                           "rule": "electron-populated tubes: |psi| <= |psi_wall(z_c +- leak_half_width)| of the cell's cusps; psi = int B_z r dr"},
            "electrodes": {"anode_effective_area_m2": float(self.anode_effective_area.sum()), "exit_effective_area_m2": float(self.exit_effective_area.sum())},
            "cell_volume_m3": [float(v) for v in self.cell_volume_m3],
        }


__all__ = [
    "CONVECTED_ENERGY_PER_ELECTRON_OVER_T",
    "CUMULATIVE_KEYS",
    "MODEL_VERSION",
    "HybridL2Config",
    "HybridL2Simulation",
    "HybridL2State",
    "PlateauRule",
    "WindowAccumulator",
    "evaluate_plateau",
    "trailing_time_drift",
    "wall_cell_area_m2",
]
